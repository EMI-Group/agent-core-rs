//! `SKILL.md` front-matter parsing, owned skill-root mounts, and resource validation.
//!
//! Ports `openjiuwen/harness/skills/mount.py`: the front-matter split is the same naive
//! `splitn(3, "---")` Python performs, a root mount only releases a root it introduced
//! itself, and the validator re-checks a skill when the backend or the file metadata
//! changes.
//!
//! The front-matter splitter is tolerant by design: [`split_skill_front_matter`] never
//! fails (a malformed skill simply yields no metadata), while
//! [`parse_skill_front_matter`] reports malformed YAML as an error so callers can decide
//! whether to warn or reject.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_sysop::fs::types::{ReadFileParams, ReadMode};
use openjiuwen_sysop::operation::FsOperation;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Name of the file that marks a directory as a skill.
pub const SKILL_FILENAME: &str = "SKILL.md";

/// Number of leading lines read when probing a `SKILL.md` front matter.
///
/// Python reads the first 64 lines (`read_file(..., head=64)`); a fence beyond that
/// boundary is recovered by `read_front_matter_text` through a whole-file re-read.
pub const FRONT_MATTER_PROBE_LINES: usize = 64;

/// Front-matter metadata of a `SKILL.md` file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SkillMetadata {
    /// The `name` field of the front matter.
    pub name: String,
    /// The `description` field of the front matter.
    pub description: String,
}

/// Splits `SKILL.md` text into its front-matter map and body.
///
/// This is the tolerant convenience wrapper around [`parse_skill_front_matter`]: when the
/// text has no complete, object-shaped front matter the result is an empty map and the
/// original text, so a caller can always render something.
///
/// # Examples
///
/// ```
/// use openjiuwen_harness::skills::split_skill_front_matter;
///
/// let (metadata, body) = split_skill_front_matter("---\nname: alpha\ndescription: A\n---\n# alpha\n");
/// assert_eq!(metadata["name"], "alpha");
/// assert_eq!(body, "# alpha\n");
/// ```
#[must_use]
pub fn split_skill_front_matter(text: &str) -> (BTreeMap<String, Value>, String) {
    match parse_skill_front_matter(text) {
        Ok(Some((metadata, body))) => (metadata, body),
        _ => (BTreeMap::new(), text.to_string()),
    }
}

/// Parses the native `SKILL.md` front matter without interpreting its body.
///
/// The text must start with `---` exactly (no BOM or leading-whitespace tolerance).
/// Splitting is the naive first-two-`---` split Python performs, so a `---` inside the
/// body is part of the body and never re-parsed. The returned body has leading
/// whitespace removed (Python `body.lstrip()`).
///
/// Returns `Ok(None)` when the text has no front matter, when the closing fence is
/// missing, or when the YAML block is not an object. An empty or whitespace-only block
/// and a YAML `null` block yield an empty map.
///
/// # Errors
///
/// Returns [`Error::validation`] when the YAML block is malformed.
pub fn parse_skill_front_matter(text: &str) -> Result<Option<(BTreeMap<String, Value>, String)>> {
    if !text.starts_with("---") {
        return Ok(None);
    }
    let mut parts = text.splitn(3, "---");
    let _ = parts.next();
    let Some(yaml_block) = parts.next() else {
        return Ok(None);
    };
    let Some(body) = parts.next() else {
        return Ok(None);
    };
    let metadata = if yaml_block.trim().is_empty() {
        BTreeMap::new()
    } else {
        let value: Value = serde_yaml_ng::from_str(yaml_block)
            .map_err(|error| Error::validation(format!("invalid SKILL.md frontmatter: {error}")))?;
        match value {
            Value::Null => BTreeMap::new(),
            Value::Object(map) => map.into_iter().collect(),
            _ => return Ok(None),
        }
    };
    Ok(Some((metadata, body.trim_start().to_string())))
}

/// An ownership handle for a skill discovery root.
///
/// A mount releases its root only when it introduced that root itself; roots that were
/// already present (host configuration, another mount) remain mounted after
/// [`unmount`](Self::unmount). This mirrors Python's `SkillRootMount`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillRootMount {
    root: PathBuf,
    owned: bool,
}

impl SkillRootMount {
    /// Appends `root` to `roots` when it is not already present and returns its handle.
    ///
    /// Paths are compared after absolute/lexical normalization, so `./a` and `a` are the
    /// same root. The handle is owned only when this call appended the root.
    #[must_use]
    pub fn mount(roots: &mut Vec<PathBuf>, root: impl Into<PathBuf>) -> Self {
        let root = normalize_path(root.into());
        let owned = !contains_root(roots, &root);
        if owned {
            roots.push(root.clone());
        }
        Self { root, owned }
    }

    /// Returns the normalized root path held by this handle.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns `true` when this handle introduced the root and may release it.
    #[must_use]
    pub fn owned(&self) -> bool {
        self.owned
    }

    /// Returns `true` when the root is present in `roots`.
    #[must_use]
    pub fn is_mounted(&self, roots: &[PathBuf]) -> bool {
        contains_root(roots, &self.root)
    }

    /// Re-adds the root when it is absent and returns the resulting ownership.
    ///
    /// Restoring a detached required root keeps prior ownership: a mount that owned the
    /// root before it was detached still owns it afterwards.
    #[must_use]
    pub fn ensure_mounted(&mut self, roots: &mut Vec<PathBuf>) -> bool {
        if !self.is_mounted(roots) {
            roots.push(self.root.clone());
            self.owned = true;
        }
        self.owned
    }

    /// Releases the root when this handle owns it and returns whether one was removed.
    ///
    /// The first matching root is removed (Python removes every match, but a normalized
    /// root list contains at most one). The call is idempotent: a borrowed mount and a
    /// second call after a release do nothing.
    #[must_use]
    pub fn unmount(&mut self, roots: &mut Vec<PathBuf>) -> bool {
        if !self.owned {
            return false;
        }
        self.owned = false;
        let Some(index) = roots
            .iter()
            .position(|candidate| same_path(candidate, &self.root))
        else {
            return false;
        };
        roots.remove(index);
        true
    }
}

/// The cached result of a successful [`SkillResourceValidator`] check.
#[derive(Debug, Clone)]
struct VerifiedSkill {
    signature: ValidationSignature,
    metadata: SkillMetadata,
}

/// Identity of a validated `SKILL.md`: backend, modification time, and size.
///
/// Python uses `(id(operation), st_mtime_ns, st_size)`; the Rust port derives the backend
/// identity from the data pointer of the `&dyn FsOperation` trait object and uses `0` for
/// the local filesystem backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ValidationSignature {
    backend: usize,
    modified_nanos: u128,
    size: u64,
}

impl ValidationSignature {
    /// Builds a signature for the given backend identity and file metadata.
    fn new(backend: usize, metadata: &std::fs::Metadata) -> Self {
        Self {
            backend,
            modified_nanos: metadata
                .modified()
                .ok()
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |duration| duration.as_nanos()),
            size: metadata.len(),
        }
    }
}

/// Validates a required skill through the host's filesystem policy.
///
/// The validator requires `<directory>/SKILL.md` to exist, its front matter `name` to
/// match the validator's name, and its `description` to be a non-empty string. Successful
/// checks are cached and re-run only when the backend identity or the file
/// `(mtime_ns, size)` changes; failures are never cached. Python's `__init__` resolves the
/// directory with `Path.resolve`; the Rust port uses [`std::path::absolute`].
#[derive(Debug, Clone)]
pub struct SkillResourceValidator {
    name: String,
    directory: PathBuf,
    path: PathBuf,
    verified: Option<VerifiedSkill>,
}

impl SkillResourceValidator {
    /// Creates a validator for the skill named `name` located in `directory`.
    #[must_use]
    pub fn new(name: impl Into<String>, directory: impl Into<PathBuf>) -> Self {
        let directory = normalize_path(directory.into());
        let path = directory.join(SKILL_FILENAME);
        Self {
            name: name.into(),
            directory,
            path,
            verified: None,
        }
    }

    /// Returns the required skill name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the normalized skill directory.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the path of the required `SKILL.md` file.
    #[must_use]
    pub fn skill_path(&self) -> &Path {
        &self.path
    }

    /// Validates the skill through `fs`, re-reading only when its signature changed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with Python's text
    /// `is not readable: <path>: <detail>` when the file cannot be stat'ed or read, when
    /// the backend reports a non-zero code, or when the front matter is missing, invalid,
    /// or names a different skill.
    pub async fn validate(&mut self, fs: &dyn FsOperation) -> Result<SkillMetadata> {
        let file = tokio::fs::metadata(&self.path)
            .await
            .map_err(|error| unreadable(&self.path, error))?;
        let backend = std::ptr::from_ref(fs).cast::<()>() as usize;
        let signature = ValidationSignature::new(backend, &file);
        if let Some(metadata) = self.cached(signature) {
            return Ok(metadata);
        }
        let params = ReadFileParams {
            path: self.path.display().to_string(),
            mode: ReadMode::Text,
            head: Some(FRONT_MATTER_PROBE_LINES),
            encoding: "utf-8".to_string(),
            only_read: true,
            ..ReadFileParams::default()
        };
        let result = fs.read_file(params).await;
        if result.code != 0 {
            return Err(unreadable(&self.path, result.message));
        }
        let content = result
            .data
            .as_ref()
            .and_then(|data| data.content.as_text())
            .ok_or_else(|| invalid_front_matter(&self.path))?;
        let metadata = parse_validated_metadata(&self.name, content, &self.path)?;
        self.cache(signature, &metadata);
        Ok(metadata)
    }

    /// Validates the skill directly on the local filesystem.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with Python's text
    /// `is not readable: <path>: <detail>` when the file is missing or unreadable, or when
    /// its front matter is missing, invalid, or names a different skill.
    pub async fn validate_local(&mut self) -> Result<SkillMetadata> {
        let file = tokio::fs::metadata(&self.path)
            .await
            .map_err(|error| unreadable(&self.path, error))?;
        let signature = ValidationSignature::new(0, &file);
        if let Some(metadata) = self.cached(signature) {
            return Ok(metadata);
        }
        let content = read_head_lines(&self.path, FRONT_MATTER_PROBE_LINES)
            .await
            .map_err(|error| unreadable(&self.path, error))?;
        let metadata = parse_validated_metadata(&self.name, &content, &self.path)?;
        self.cache(signature, &metadata);
        Ok(metadata)
    }

    /// Forgets the cached validation, forcing the next call to re-read the file.
    pub fn clear(&mut self) {
        self.verified = None;
    }

    /// Returns the cached metadata when it was verified with `signature`.
    fn cached(&self, signature: ValidationSignature) -> Option<SkillMetadata> {
        self.verified
            .as_ref()
            .filter(|verified| verified.signature == signature)
            .map(|verified| verified.metadata.clone())
    }

    /// Stores a successful validation result.
    fn cache(&mut self, signature: ValidationSignature, metadata: &SkillMetadata) {
        self.verified = Some(VerifiedSkill {
            signature,
            metadata: metadata.clone(),
        });
    }
}

/// Reads at most `lines` leading lines from `path`, joined with `\n`.
///
/// Line terminators are not preserved, which is fine for front-matter parsing. A missing
/// final newline is not invented.
pub(crate) async fn read_head_lines(path: &Path, lines: usize) -> io::Result<String> {
    let file = tokio::fs::File::open(path).await?;
    let mut reader = BufReader::new(file).lines();
    let mut head = String::new();
    let mut count = 0_usize;
    while count < lines {
        let Some(line) = reader.next_line().await? else {
            break;
        };
        if count > 0 {
            head.push('\n');
        }
        head.push_str(&line);
        count += 1;
    }
    Ok(head)
}

/// Reads the `SKILL.md` head lines, falling back to the whole file for a late fence.
///
/// The first [`FRONT_MATTER_PROBE_LINES`] lines are read. When they contain a complete
/// front matter, they are returned as-is. When the head starts with `---` but the closing
/// fence lies beyond the probe window, the whole file is re-read so the front matter can
/// still be parsed; any other unparseable head is returned unchanged.
///
/// # Errors
///
/// Returns the IO error when the file cannot be opened or read.
pub(crate) async fn read_front_matter_text(path: &Path) -> io::Result<String> {
    let head = read_head_lines(path, FRONT_MATTER_PROBE_LINES).await?;
    if !head.starts_with("---") {
        return Ok(head);
    }
    if head.splitn(3, "---").count() >= 3 {
        return Ok(head);
    }
    tokio::fs::read_to_string(path).await
}

/// Parses `content` into metadata, enforcing the required name and description.
fn parse_validated_metadata(name: &str, content: &str, path: &Path) -> Result<SkillMetadata> {
    let Ok(Some((metadata, _))) = parse_skill_front_matter(content) else {
        return Err(invalid_front_matter(path));
    };
    if metadata.get("name").and_then(Value::as_str) != Some(name) {
        return Err(invalid_front_matter(path));
    }
    let Some(description) = metadata.get("description").and_then(Value::as_str) else {
        return Err(invalid_front_matter(path));
    };
    if description.trim().is_empty() {
        return Err(invalid_front_matter(path));
    }
    Ok(SkillMetadata {
        name: name.to_string(),
        description: description.to_string(),
    })
}

/// Normalizes a path to an absolute, lexically cleaned form when possible.
pub(crate) fn normalize_path(path: PathBuf) -> PathBuf {
    std::path::absolute(&path).unwrap_or(path)
}

/// Compares two paths after normalization.
fn same_path(left: &Path, right: &Path) -> bool {
    normalize_path(left.to_path_buf()) == normalize_path(right.to_path_buf())
}

/// Returns `true` when `roots` already contains `root` after normalization.
fn contains_root(roots: &[PathBuf], root: &Path) -> bool {
    roots.iter().any(|candidate| same_path(candidate, root))
}

/// Builds Python's `is not readable: <path>: <detail>` validation error.
fn unreadable(path: &Path, detail: impl std::fmt::Display) -> Error {
    Error::validation(format!("is not readable: {}: {detail}", path.display()))
}

/// Builds Python's invalid-front-matter validation error.
fn invalid_front_matter(path: &Path) -> Error {
    unreadable(path, "invalid SKILL.md frontmatter")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::fmt::Write as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use openjiuwen_sysop::fs::types::{
        DownloadFileData, DownloadFileParams, FileContent, ListDirsParams, ListFilesData,
        ListFilesParams, ReadChunkData, ReadFileData, SearchFilesData, SearchFilesParams,
        TransferChunkData, UploadFileData, UploadFileParams, WriteFileData, WriteFileParams,
    };
    use openjiuwen_sysop::operation::OpStream;
    use openjiuwen_sysop::result::OpResult;
    use openjiuwen_sysop::status::SysOpStatus;

    use super::*;

    /// A filesystem backend that counts reads and returns canned content.
    struct CountingFs {
        content: String,
        failure: Option<String>,
        reads: AtomicUsize,
    }

    impl CountingFs {
        fn new(content: &str) -> Self {
            Self {
                content: content.to_string(),
                failure: None,
                reads: AtomicUsize::new(0),
            }
        }

        fn failing(message: &str) -> Self {
            Self {
                content: String::new(),
                failure: Some(message.to_string()),
                reads: AtomicUsize::new(0),
            }
        }

        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }
    }

    fn unsupported<T>() -> OpResult<T> {
        OpResult::failure(SysOpStatus::Error, "unsupported")
    }

    #[async_trait::async_trait]
    impl FsOperation for CountingFs {
        async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if let Some(message) = &self.failure {
                return OpResult::failure(SysOpStatus::Error, message.clone());
            }
            OpResult::success(ReadFileData {
                path: params.path,
                content: FileContent::text(self.content.clone()),
                mode: ReadMode::Text,
                encoding: None,
                truncated: false,
                total_bytes: 0,
            })
        }

        fn read_file_stream(self: Arc<Self>, _params: ReadFileParams) -> OpStream<ReadChunkData> {
            Box::pin(futures::stream::empty())
        }

        async fn write_file(&self, _params: WriteFileParams) -> OpResult<WriteFileData> {
            unsupported()
        }

        async fn upload_file(&self, _params: UploadFileParams) -> OpResult<UploadFileData> {
            unsupported()
        }

        fn upload_file_stream(
            self: Arc<Self>,
            _params: UploadFileParams,
        ) -> OpStream<TransferChunkData> {
            Box::pin(futures::stream::empty())
        }

        async fn download_file(&self, _params: DownloadFileParams) -> OpResult<DownloadFileData> {
            unsupported()
        }

        fn download_file_stream(
            self: Arc<Self>,
            _params: DownloadFileParams,
        ) -> OpStream<TransferChunkData> {
            Box::pin(futures::stream::empty())
        }

        async fn list_files(&self, _params: ListFilesParams) -> OpResult<ListFilesData> {
            unsupported()
        }

        async fn list_directories(&self, _params: ListDirsParams) -> OpResult<ListFilesData> {
            unsupported()
        }

        async fn search_files(&self, _params: SearchFilesParams) -> OpResult<SearchFilesData> {
            unsupported()
        }
    }

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn write_file(directory: &Path, content: &str) {
        std::fs::create_dir_all(directory).unwrap();
        std::fs::write(directory.join(SKILL_FILENAME), content).unwrap();
    }

    #[test]
    fn parses_front_matter_and_trims_the_body_start() {
        let (metadata, body) =
            parse_skill_front_matter("---\nname: alpha\ndescription: A skill\n---\n\n# alpha\n")
                .unwrap()
                .unwrap();
        assert_eq!(metadata.get("name").and_then(Value::as_str), Some("alpha"));
        assert_eq!(
            metadata.get("description").and_then(Value::as_str),
            Some("A skill")
        );
        assert_eq!(body, "# alpha\n");
    }

    #[test]
    fn text_without_front_matter_is_not_parsed() {
        assert_eq!(parse_skill_front_matter("# alpha\n").unwrap(), None);
        let (metadata, body) = split_skill_front_matter("# alpha\n");
        assert!(metadata.is_empty());
        assert_eq!(body, "# alpha\n");
    }

    #[test]
    fn unterminated_front_matter_is_not_parsed() {
        assert_eq!(parse_skill_front_matter("---\nname: x").unwrap(), None);
        let (metadata, body) = split_skill_front_matter("---\nname: x");
        assert!(metadata.is_empty());
        assert_eq!(body, "---\nname: x");
    }

    #[test]
    fn malformed_yaml_is_an_error() {
        let error = parse_skill_front_matter("---\nname: [unclosed\n---\n").unwrap_err();
        assert!(error.message().contains("invalid SKILL.md frontmatter"));
    }

    #[test]
    fn non_object_yaml_is_not_parsed() {
        assert_eq!(
            parse_skill_front_matter("---\n- alpha\n- beta\n---\n").unwrap(),
            None
        );
        assert_eq!(parse_skill_front_matter("---\nalpha\n---\n").unwrap(), None);
    }

    #[test]
    fn fence_inside_the_body_is_not_reparsed() {
        let (metadata, body) = parse_skill_front_matter(
            "---\nname: alpha\ndescription: A skill\n---\nbody\n---\nend\n",
        )
        .unwrap()
        .unwrap();
        assert_eq!(metadata.get("name").and_then(Value::as_str), Some("alpha"));
        assert_eq!(body, "body\n---\nend\n");
    }

    #[test]
    fn empty_front_matter_yields_an_empty_map() {
        let (metadata, body) = parse_skill_front_matter("---\n\n---\nbody")
            .unwrap()
            .unwrap();
        assert!(metadata.is_empty());
        assert_eq!(body, "body");
    }

    #[test]
    fn mount_ownership_tracks_host_and_candidate_roots() {
        let temp = temp_dir();
        let host = temp.path().join("host-skills");
        let candidate = temp.path().join("candidate");
        let mut roots = vec![host.clone()];

        let host_mount = SkillRootMount::mount(&mut roots, host.clone());
        assert!(!host_mount.owned());
        assert!(host_mount.is_mounted(&roots));
        assert_eq!(roots, vec![host.clone()]);

        let mut candidate_mount = SkillRootMount::mount(&mut roots, &candidate);
        assert!(candidate_mount.owned());
        let normalized_candidate = std::path::absolute(&candidate).unwrap();
        assert_eq!(roots, vec![host.clone(), normalized_candidate.clone()]);

        let mut borrowed_mount = SkillRootMount::mount(&mut roots, &candidate);
        assert!(!borrowed_mount.owned());
        assert_eq!(roots.len(), 2);

        assert!(!host_mount.clone().unmount(&mut roots));
        assert!(!borrowed_mount.unmount(&mut roots));
        assert_eq!(roots, vec![host.clone(), normalized_candidate.clone()]);

        assert!(candidate_mount.unmount(&mut roots));
        assert_eq!(roots, vec![host.clone()]);
        assert!(!candidate_mount.unmount(&mut roots));
        assert_eq!(roots, vec![host.clone()]);

        assert!(candidate_mount.ensure_mounted(&mut roots));
        assert_eq!(roots, vec![host.clone(), normalized_candidate.clone()]);
        assert!(candidate_mount.unmount(&mut roots));
        assert_eq!(roots, vec![host.clone()]);
    }

    #[test]
    fn ensure_mounted_does_not_duplicate_a_present_root() {
        let temp = temp_dir();
        let candidate = temp.path().join("candidate");
        let mut roots = Vec::new();
        let mut mount = SkillRootMount::mount(&mut roots, &candidate);
        assert!(mount.ensure_mounted(&mut roots));
        assert_eq!(roots.len(), 1);
        assert_eq!(mount.root(), std::path::absolute(&candidate).unwrap());
    }

    #[tokio::test]
    async fn validate_local_returns_metadata_for_a_valid_skill() {
        let temp = temp_dir();
        let directory = temp.path().join("alpha");
        write_file(&directory, "---\nname: alpha\ndescription: A skill\n---\n");
        let mut validator = SkillResourceValidator::new("alpha", &directory);
        assert_eq!(validator.name(), "alpha");
        assert_eq!(validator.directory(), directory.as_path());
        assert_eq!(
            validator.skill_path(),
            directory.join(SKILL_FILENAME).as_path()
        );
        assert_eq!(
            validator.validate_local().await.unwrap(),
            SkillMetadata {
                name: "alpha".to_string(),
                description: "A skill".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn validate_local_accepts_a_directory_that_is_a_skill_root() {
        let temp = temp_dir();
        write_file(
            temp.path(),
            "---\nname: alpha\ndescription: Root skill\n---\n",
        );
        let mut validator = SkillResourceValidator::new("alpha", temp.path());
        assert_eq!(
            validator.validate_local().await.unwrap(),
            SkillMetadata {
                name: "alpha".to_string(),
                description: "Root skill".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn validate_local_rejects_name_mismatch_and_empty_description() {
        let temp = temp_dir();
        let directory = temp.path().join("alpha");
        write_file(&directory, "---\nname: beta\ndescription: A skill\n---\n");
        let mut validator = SkillResourceValidator::new("alpha", &directory);
        let error = validator.validate_local().await.unwrap_err();
        assert!(error.message().contains("invalid SKILL.md frontmatter"));

        write_file(&directory, "---\nname: alpha\ndescription:   \n---\n");
        let mut validator = SkillResourceValidator::new("alpha", &directory);
        let error = validator.validate_local().await.unwrap_err();
        assert!(error.message().contains("invalid SKILL.md frontmatter"));

        write_file(&directory, "---\nname: alpha\n---\n");
        let mut validator = SkillResourceValidator::new("alpha", &directory);
        let error = validator.validate_local().await.unwrap_err();
        assert!(error.message().contains("invalid SKILL.md frontmatter"));
    }

    #[tokio::test]
    async fn validate_local_reports_a_missing_skill_file() {
        let temp = temp_dir();
        let directory = temp.path().join("missing");
        std::fs::create_dir_all(&directory).unwrap();
        let mut validator = SkillResourceValidator::new("alpha", &directory);
        let error = validator.validate_local().await.unwrap_err();
        assert!(error.message().contains("is not readable"));
        assert!(error.message().contains("SKILL.md"));
    }

    #[tokio::test]
    async fn validate_caches_successes_and_retries_failures() {
        let temp = temp_dir();
        let directory = temp.path().join("alpha");
        let content = "---\nname: alpha\ndescription: A skill\n---\n";
        write_file(&directory, content);
        let fs = Arc::new(CountingFs::new(content));
        let mut validator = SkillResourceValidator::new("alpha", &directory);

        let first = validator.validate(fs.as_ref()).await.unwrap();
        let second = validator.validate(fs.as_ref()).await.unwrap();
        assert_eq!(first, second);
        assert_eq!(fs.reads(), 1);

        let denied = Arc::new(CountingFs::failing("access denied"));
        let error = validator.validate(denied.as_ref()).await.unwrap_err();
        assert!(error.message().contains("access denied"));
        let error = validator.validate(denied.as_ref()).await.unwrap_err();
        assert!(error.message().contains("access denied"));
        assert_eq!(denied.reads(), 2);

        validator.clear();
        let _ = validator.validate(fs.as_ref()).await.unwrap();
        assert_eq!(fs.reads(), 2);
    }

    #[tokio::test]
    async fn validate_rejects_a_backend_reporting_an_invalid_skill() {
        let temp = temp_dir();
        let directory = temp.path().join("alpha");
        write_file(&directory, "---\nname: alpha\ndescription: A skill\n---\n");
        let fs = CountingFs::new("---\nname: other\ndescription: A skill\n---\n");
        let mut validator = SkillResourceValidator::new("alpha", &directory);
        let error = validator.validate(&fs).await.unwrap_err();
        assert!(error.message().contains("invalid SKILL.md frontmatter"));
    }

    #[tokio::test]
    async fn read_front_matter_text_stops_after_the_probe_window() {
        let temp = temp_dir();
        let path = temp.path().join(SKILL_FILENAME);
        let mut content = String::from("---\nname: alpha\ndescription: A skill\n---\n");
        for index in 0..200 {
            writeln!(content, "line {index}").unwrap();
        }
        std::fs::write(&path, content).unwrap();

        let head = read_front_matter_text(&path).await.unwrap();
        assert!(head.starts_with("---\nname: alpha"));
        assert!(head.lines().count() <= FRONT_MATTER_PROBE_LINES);
    }

    #[tokio::test]
    async fn read_front_matter_text_rereads_the_file_for_a_late_fence() {
        let temp = temp_dir();
        let path = temp.path().join(SKILL_FILENAME);
        let mut content = String::from("---\n");
        for index in 0..100 {
            writeln!(content, "key{index}: value").unwrap();
        }
        content.push_str("---\nbody\n");
        std::fs::write(&path, content).unwrap();

        let text = read_front_matter_text(&path).await.unwrap();
        assert!(text.contains("key99: value"));
        assert!(text.ends_with("---\nbody\n"));
    }
}
