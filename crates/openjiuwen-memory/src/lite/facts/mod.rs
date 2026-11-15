//! Per-fact Markdown files and the `MEMORY.md`/`USER.md` rendered index split.
//!
//! Ports the deferred file layout of Python `core/memory/lite/`: instead of keeping every
//! fact inline in one document, each fact owns a Markdown file under `facts/`, and the
//! two top-level documents are rendered views over those files.
//!
//! # Layout
//!
//! ```text
//! <root>/
//!   facts/<slug>-<hash>.md   one fact per file: front matter plus the Markdown body
//!   MEMORY.md                rendered index of the non-user facts
//!   USER.md                  rendered index of the user-level facts
//! ```
//!
//! A fact file carries the interpreted front-matter fields (`name`, `description`,
//! `type`, `tags`, `path`, `updated_at`) in a fixed order, so a fact survives a round
//! trip through either representation; see [`file`] for the document format. The file
//! name is `file_name_for(name)` — a readable slug plus a hash of the exact name — so two
//! names that slug the same still get distinct files, and the name stored in the front
//! matter is authoritative rather than the file name.
//!
//! # Index split
//!
//! `user`-level facts (durable preferences and identity facts about the person) are
//! rendered into `USER.md`; every other fact type (`feedback`, `project`, `reference`) is
//! rendered into `MEMORY.md` (see the `split` submodule). Both indexes are prompt-facing
//! summaries in the existing [`MarkdownMemory`](crate::lite::markdown::MarkdownMemory)
//! document shape, so an index can be read by the local store's parser as well as by hand.
//!
//! # Hand edits
//!
//! A rendered index may be edited by hand and read back with `split::parse_level`: it
//! returns the facts a document holds, and `FactFiles::write` writes each of them back to
//! its own fact file. Rendering the re-imported facts reproduces the edited document byte
//! for byte, because a rendered index and a fact file carry the same field set. Diffing an index against the fact files
//! (to decide which entry changed or was deleted) is left to the caller; this module owns
//! the file and document formats only.
//!
//! # Constraints
//!
//! `facts/` is the source of truth and the rendered indexes are derived views of it, so
//! any index can be regenerated at any time. Renders are deterministic (fixed fact-type
//! order, name-ascending inside a type) and end with exactly one newline; writes are
//! atomic (temporary file plus rename in the same directory) and create `facts/` on
//! demand. A directory holding non-fact Markdown is tolerated: files that do not decode as
//! a coding fact are skipped and reported, never fatal.
use crate::lite::fact::CodingFact;
use crate::record::MemoryScope;
use openjiuwen_common::error::{Error, Result};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;
pub mod file;
pub mod slug;
pub mod split;
pub use file::FactDocument;
pub use slug::{file_name_for, slug};
pub use split::{level_of, parse_level, render_level, IndexLevel};
/// Directory, relative to the memory root, holding one Markdown file per fact.
pub const FACTS_DIR: &str = "facts";
/// File name of the rendered index of the non-user facts.
pub const MEMORY_FILE: &str = "MEMORY.md";
/// File name of the rendered index of the user-level facts.
pub const USER_FILE: &str = "USER.md";
/// The tracing target of lite-memory diagnostics.
const TARGET: &str = "openjiuwen_memory::lite::facts";
/// The file extension a fact file must carry to be read back.
const MARKDOWN_EXTENSION: &str = "md";
/// A cheap change stamp of the `facts/` directory.
///
/// Comparing two stamps tells a caller whether the directory changed without reading
/// every fact file: the count, the summed lengths, and the newest modification time
/// (milliseconds since the Unix epoch) all have to match for the directory to be
/// considered unchanged. Only `*.md` files are counted, so a temporary file of an
/// in-flight write never perturbs the stamp.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirStamp {
    /// Number of Markdown files in the directory.
    pub files: usize,
    /// Total length of those files, in bytes.
    pub total_len: u64,
    /// Modification time of the newest of them, in milliseconds since the Unix epoch.
    pub newest_ms: u128,
}
/// The per-fact file store of one memory root and scope.
///
/// The store is a thin, cloneable handle over a root directory: it holds no open file
/// and no cache, so several stores may point at the same root (and a store may be moved
/// across tasks). Every fact written through it is rebound to
/// [`FactFiles::scope`] on read, because a fact file does not record its scope — one
/// root holds one scope.
#[derive(Debug, Clone)]
pub struct FactFiles {
    /// The memory root holding `facts/`, `MEMORY.md`, and `USER.md`.
    root: PathBuf,
    /// The scope every fact read through this store is bound to.
    scope: MemoryScope,
}
impl FactFiles {
    /// Builds a store over `root` whose facts belong to `scope`.
    ///
    /// Nothing is created or read until a method is called.
    #[must_use]
    pub fn new(root: impl AsRef<Path>, scope: MemoryScope) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            scope,
        }
    }
    /// The memory root this store writes under.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// The scope facts read through this store are bound to.
    #[must_use]
    pub fn scope(&self) -> &MemoryScope {
        &self.scope
    }
    /// The fact directory of this store: `<root>/facts`.
    #[must_use]
    pub fn dir(&self) -> PathBuf {
        self.root.join(FACTS_DIR)
    }
    /// The file the fact called `name` is stored in: `<root>/facts/<slug>-<hash>.md`.
    #[must_use]
    pub fn path_for(&self, name: &str) -> PathBuf {
        self.dir().join(file_name_for(name))
    }
    /// Writes `fact` to its own file and returns the path.
    ///
    /// A missing `facts/` directory is created. The file is written to a temporary name
    /// in the same directory and renamed over the target, so a reader either sees the
    /// previous file or the complete new one, and a failed write leaves no partial fact
    /// file behind. An existing file of the same fact name is replaced.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the fact name is blank (a fact without a name
    /// cannot be read back) and [`Error::External`] with the failing path as context when
    /// the directory or the file cannot be written.
    pub async fn write(&self, fact: &CodingFact) -> Result<PathBuf> {
        if fact.name.trim().is_empty() {
            return Err(Error::validation(
                "coding fact `name` must not be empty when writing a fact file",
            ));
        }
        let dir = self.dir();
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|error| io_error(error, "creating the fact directory", &dir))?;
        let path = dir.join(file_name_for(&fact.name));
        let rendered = FactDocument::from_fact(fact).render();
        let temporary = temporary_path(&path);
        tokio::fs::write(&temporary, rendered.as_bytes())
            .await
            .map_err(|error| io_error(error, "writing the temporary fact file", &temporary))?;
        if let Err(error) = tokio::fs::rename(&temporary, &path).await {
            discard(&temporary).await;
            return Err(io_error(error, "replacing the fact file", &path));
        }
        Ok(path)
    }
    /// Reads the fact called `name`, or `None` when it has no file.
    ///
    /// The returned fact is bound to [`FactFiles::scope`], not to the scope that was
    /// written, and a blank name can never name a stored fact, so it reads as absent.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error with the failing path as context when the file cannot
    /// be read, and the [`FactDocument::parse`]/[`FactDocument::to_fact`] error when the
    /// file is not decodable (an unterminated front-matter block, a blank name, or an
    /// unknown fact type).
    pub async fn read(&self, name: &str) -> Result<Option<CodingFact>> {
        if name.trim().is_empty() {
            return Ok(None);
        }
        let path = self.path_for(name);
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error(error, "reading the fact file", &path)),
        };
        let fact = FactDocument::parse(&text)?.to_fact(&self.scope)?;
        Ok(Some(fact))
    }
    /// Reads every fact of the directory, sorted by name ascending.
    ///
    /// Only regular `*.md` files are read: directories, symlinks (which might point out of
    /// the directory), and other files are ignored. A Markdown file that does not decode
    /// as a coding fact — hand-written prose, a missing or unknown `type`, an unterminated
    /// front matter, a blank `name` — is skipped with a warning naming the path, so one
    /// stray file never makes the store unreadable. An absent directory holds no facts.
    ///
    /// The skipped files are reported by [`FactFiles::read_all_report`], which a caller
    /// that deletes or rewrites fact files must use: the returned list is the facts that
    /// decoded, not necessarily every fact file on disk.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error with the failing path as context when the directory
    /// cannot be listed or one of its fact files cannot be read.
    pub async fn read_all(&self) -> Result<Vec<CodingFact>> {
        Ok(self.read_all_report().await?.0)
    }

    /// Reads every fact of the directory and reports the files that did not decode.
    ///
    /// The first element is exactly [`FactFiles::read_all`]: the facts that decoded,
    /// sorted by name ascending. The second element names every Markdown file that was
    /// skipped, with the parse error that skipped it, so a caller can surface how much of
    /// the directory it could not read. A skipped file is *not* evidence that its fact was
    /// deleted: a caller that removes stale fact files must leave these paths alone,
    /// because a file it cannot decode is a file it cannot judge.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error with the failing path as context when the directory
    /// cannot be listed or a file cannot be read as UTF-8 text.
    pub async fn read_all_report(&self) -> Result<(Vec<CodingFact>, Vec<(PathBuf, Error)>)> {
        let mut facts = Vec::new();
        let mut skipped: Vec<(PathBuf, Error)> = Vec::new();
        for path in self.markdown_files().await? {
            let text = tokio::fs::read_to_string(&path)
                .await
                .map_err(|error| io_error(error, "reading the fact file", &path))?;
            match FactDocument::parse(&text).and_then(|document| document.to_fact(&self.scope)) {
                Ok(fact) => facts.push(fact),
                Err(error) => {
                    tracing::warn!(
                        target: TARGET,
                        path = %path.display(),
                        error = %error,
                        "skipping a Markdown file that is not a coding fact"
                    );
                    skipped.push((path, error));
                }
            }
        }
        facts.sort_by(|left, right| left.name.cmp(&right.name));
        Ok((facts, skipped))
    }
    /// Deletes the file of the fact called `name`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error with the failing path as context when the file exists
    /// but cannot be removed.
    pub async fn remove(&self, name: &str) -> Result<bool> {
        if name.trim().is_empty() {
            return Ok(false);
        }
        let path = self.path_for(name);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
            Err(error) => Err(io_error(error, "removing the fact file", &path)),
        }
    }
    /// Whether the fact called `name` has a file.
    ///
    /// A symlink is not a fact file, so it reports absent even when it resolves to one.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error with the failing path as context when the path cannot
    /// be inspected (a permission error, for example).
    pub async fn contains(&self, name: &str) -> Result<bool> {
        if name.trim().is_empty() {
            return Ok(false);
        }
        let path = self.path_for(name);
        match tokio::fs::symlink_metadata(&path).await {
            Ok(metadata) => Ok(metadata.file_type().is_file()),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
            Err(error) => Err(io_error(error, "inspecting the fact file", &path)),
        }
    }
    /// The change stamp of the directory.
    ///
    /// An absent directory stamps as [`DirStamp::default`], so a caller can poll it before
    /// anything has been written.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error with the failing path as context when the directory
    /// cannot be listed or a fact file's metadata cannot be read.
    pub async fn stamp(&self) -> Result<DirStamp> {
        let dir = self.dir();
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(DirStamp::default()),
            Err(error) => return Err(io_error(error, "listing the fact directory", &dir)),
        };
        let mut stamp = DirStamp::default();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| io_error(error, "listing the fact directory", &dir))?
        {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .await
                .map_err(|error| io_error(error, "inspecting the fact file", &path))?;
            if !file_type.is_file() || !is_markdown(&path) {
                continue;
            }
            let metadata = entry
                .metadata()
                .await
                .map_err(|error| io_error(error, "inspecting the fact file", &path))?;
            stamp.files += 1;
            stamp.total_len = stamp.total_len.saturating_add(metadata.len());
            if let Ok(modified) = metadata.modified() {
                if let Ok(elapsed) = modified.duration_since(UNIX_EPOCH) {
                    stamp.newest_ms = stamp.newest_ms.max(elapsed.as_millis());
                }
            }
        }
        Ok(stamp)
    }
    /// The paths of the directory's Markdown files, sorted.
    ///
    /// Only regular files are returned: a symlink or a subdirectory is skipped, so the
    /// store never reads or writes outside its `facts/` directory.
    async fn markdown_files(&self) -> Result<Vec<PathBuf>> {
        let dir = self.dir();
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(io_error(error, "listing the fact directory", &dir)),
        };
        let mut paths = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| io_error(error, "listing the fact directory", &dir))?
        {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .await
                .map_err(|error| io_error(error, "inspecting the fact file", &path))?;
            if file_type.is_file() && is_markdown(&path) {
                paths.push(path);
            }
        }
        paths.sort();
        Ok(paths)
    }
}
/// Whether `path` carries the Markdown extension (ASCII case-insensitive).
fn is_markdown(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(MARKDOWN_EXTENSION))
}
/// The temporary sibling path an atomic write goes through.
///
/// The name is hidden and does not carry the Markdown extension, so a concurrent
/// [`FactFiles::read_all`]/[`FactFiles::stamp`] ignores it; the per-process counter keeps
/// two concurrent writes of the same fact from sharing one temporary file.
fn temporary_path(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("fact.md");
    let mut temporary = String::with_capacity(file_name.len() + 16);
    temporary.push('.');
    temporary.push_str(file_name);
    temporary.push('.');
    temporary.push_str(&sequence.to_string());
    temporary.push_str(".tmp");
    match path.parent() {
        Some(parent) => parent.join(temporary),
        None => PathBuf::from(temporary),
    }
}
/// Best-effort removal of a temporary file after a failed write.
async fn discard(path: &Path) {
    if let Err(error) = tokio::fs::remove_file(path).await {
        if error.kind() != ErrorKind::NotFound {
            tracing::debug!(
                target: TARGET,
                path = %path.display(),
                error = %error,
                "could not remove the temporary fact file"
            );
        }
    }
}
/// Wraps a filesystem error as [`Error::External`] naming the path.
fn io_error(error: std::io::Error, action: &str, path: &Path) -> Error {
    Error::external(error).with_context(format!("{action} {}", path.display()))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite::fact::CodingFactType;
    use chrono::DateTime;
    use tempfile::tempdir;
    /// The scope every test store is bound to.
    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "workspace")
    }
    /// The facts bound to `scope` (a rendered index does not record the scope).
    fn rebind(facts: Vec<CodingFact>, scope: MemoryScope) -> Vec<CodingFact> {
        facts
            .into_iter()
            .map(|mut fact| {
                fact.scope = scope.clone();
                fact
            })
            .collect()
    }
    /// The canonical sort rank of a fact type (its position in [`CodingFactType::ALL`]).
    fn rank(fact_type: CodingFactType) -> usize {
        CodingFactType::ALL
            .iter()
            .position(|candidate| *candidate == fact_type)
            .unwrap_or(CodingFactType::ALL.len())
    }
    /// A fact with a pinned, second-precision timestamp.
    fn fact(name: &str, fact_type: CodingFactType, index: i64) -> CodingFact {
        let mut fact = CodingFact::new(&scope(), name, format!("body of {name}"))
            .with_type(fact_type)
            .with_description(format!("about {name}"));
        fact.updated_at = DateTime::from_timestamp(1_700_000_000 + index, 0).unwrap_or_default();
        fact
    }
    #[test]
    fn paths_live_under_the_facts_directory() {
        let root = Path::new("/tmp/memory-root");
        let files = FactFiles::new(root, scope());
        assert_eq!(files.root(), root);
        assert_eq!(files.dir(), root.join("facts"));
        let path = files.path_for("Build Command");
        assert_eq!(path.parent(), Some(root.join("facts").as_path()));
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        assert!(file_name.starts_with("build-command-"), "got {file_name}");
        assert_eq!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("md")
        );
    }
    #[tokio::test]
    async fn write_then_read_round_trips_the_fact() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        let mut original = fact("build", CodingFactType::Project, 7)
            .with_tags(vec!["build".to_owned(), "ci".to_owned()])
            .with_path("docs/BUILD.md");
        original.content = "# Build\n\ncargo build\ncargo test".to_owned();
        let path = files.write(&original).await.expect("the write succeeds");
        assert!(path.is_file());
        assert!(files.contains("build").await.expect("contains works"));
        let read = files
            .read("build")
            .await
            .expect("the read succeeds")
            .expect("the fact exists");
        assert_eq!(read, original);
    }
    #[tokio::test]
    async fn write_creates_the_directory_and_leaves_no_temporary_file() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        assert!(!files.dir().exists());
        files
            .write(&fact("build", CodingFactType::Project, 1))
            .await
            .expect("the write succeeds");
        assert!(files.dir().is_dir());
        let mut entries = tokio::fs::read_dir(files.dir()).await.expect("readable");
        let mut names = Vec::new();
        while let Some(entry) = entries.next_entry().await.expect("entry") {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        assert_eq!(names.len(), 1, "leftover files: {names:?}");
        assert_eq!(
            std::path::Path::new(&names[0])
                .extension()
                .and_then(|extension| extension.to_str()),
            Some("md"),
            "unexpected file: {names:?}"
        );
    }
    #[tokio::test]
    async fn writing_the_same_name_replaces_the_file() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        let first = fact("build", CodingFactType::Project, 1);
        let (first_path, second_path) = (
            files.write(&first).await.expect("the first write"),
            files
                .write(&first.clone().with_description("rewritten"))
                .await
                .expect("the second write"),
        );
        assert_eq!(first_path, second_path);
        let read = files.read("build").await.expect("read").expect("exists");
        assert_eq!(read.description, "rewritten");
        assert_eq!(files.stamp().await.expect("stamp").files, 1);
    }
    #[tokio::test]
    async fn a_missing_fact_reads_as_absent() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        assert_eq!(files.read("build").await.expect("read"), None);
        assert!(!files.contains("build").await.expect("contains"));
        assert!(!files.remove("build").await.expect("remove"));
        assert_eq!(
            files.read_all().await.expect("read_all"),
            Vec::<CodingFact>::new()
        );
        assert_eq!(files.stamp().await.expect("stamp"), DirStamp::default());
    }
    #[tokio::test]
    async fn remove_deletes_only_the_named_fact() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        files
            .write(&fact("build", CodingFactType::Project, 1))
            .await
            .expect("write build");
        files
            .write(&fact("test", CodingFactType::Project, 2))
            .await
            .expect("write test");
        assert!(files.remove("build").await.expect("remove"));
        assert!(!files.contains("build").await.expect("contains"));
        assert!(files.contains("test").await.expect("contains"));
        let names: Vec<String> = files
            .read_all()
            .await
            .expect("read_all")
            .into_iter()
            .map(|fact| fact.name)
            .collect();
        assert_eq!(names, vec!["test"]);
    }
    #[tokio::test]
    async fn a_blank_name_is_rejected_on_write_and_absent_on_read() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        let error = files
            .write(&fact("   ", CodingFactType::Project, 1))
            .await
            .expect_err("a blank name is invalid");
        assert!(matches!(error, Error::Validation(_)), "got {error}");
        assert_eq!(files.read("   ").await.expect("read"), None);
        assert!(!files.contains(" ").await.expect("contains"));
    }
    #[tokio::test]
    async fn read_rebinds_the_scope() {
        let directory = tempdir().expect("temp dir");
        let writer = FactFiles::new(directory.path(), MemoryScope::new("alice", "workspace"));
        writer
            .write(&fact("build", CodingFactType::Project, 1))
            .await
            .expect("write");
        let reader = FactFiles::new(directory.path(), MemoryScope::new("bob", "other"));
        let read = reader.read("build").await.expect("read").expect("exists");
        assert_eq!(read.scope, MemoryScope::new("bob", "other"));
        let all = reader.read_all().await.expect("read_all");
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].scope, MemoryScope::new("bob", "other"));
    }
    #[tokio::test]
    async fn read_all_skips_non_fact_markdown_and_sorts_by_name() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        for (index, name) in ["zeta", "alpha", "middle"].iter().enumerate() {
            files
                .write(&fact(
                    name,
                    CodingFactType::Project,
                    i64::try_from(index).unwrap_or_default(),
                ))
                .await
                .expect("write");
        }
        tokio::fs::write(files.dir().join("notes.md"), "just some notes\n")
            .await
            .expect("write prose");
        tokio::fs::write(
            files.dir().join("broken.md"),
            "---\nname: broken\nupdated_at: 2023-11-14T22:13:20Z\n---\n\nbody\n",
        )
        .await
        .expect("write a fact without a type");
        tokio::fs::write(files.dir().join("unterminated.md"), "---\nname: x\n")
            .await
            .expect("write an unterminated block");
        tokio::fs::write(files.dir().join("ignored.txt"), "not markdown at all")
            .await
            .expect("write a text file");
        let names: Vec<String> = files
            .read_all()
            .await
            .expect("read_all tolerates the directory")
            .into_iter()
            .map(|fact| fact.name)
            .collect();
        assert_eq!(names, vec!["alpha", "middle", "zeta"]);
    }
    #[tokio::test]
    async fn read_all_does_not_follow_a_symlink_out_of_the_directory() {
        let directory = tempdir().expect("temp dir");
        let outside = tempdir().expect("outside dir");
        let outside_file = outside.path().join("outside.md");
        tokio::fs::write(
            &outside_file,
            FactDocument::from_fact(&fact("outside", CodingFactType::Project, 3)).render(),
        )
        .await
        .expect("write the outside fact");
        let files = FactFiles::new(directory.path(), scope());
        files
            .write(&fact("inside", CodingFactType::Project, 1))
            .await
            .expect("write");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside_file, files.dir().join("linked.md"))
            .expect("create the symlink");
        let names: Vec<String> = files
            .read_all()
            .await
            .expect("read_all")
            .into_iter()
            .map(|fact| fact.name)
            .collect();
        assert_eq!(names, vec!["inside"]);
        assert!(!files.contains("outside").await.expect("contains"));
    }
    #[tokio::test]
    async fn the_stamp_tracks_the_directory() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        assert_eq!(files.stamp().await.expect("stamp"), DirStamp::default());
        files
            .write(&fact("one", CodingFactType::Project, 1))
            .await
            .expect("write one");
        let first = files.stamp().await.expect("stamp");
        assert_eq!(first.files, 1);
        assert!(first.total_len > 0);
        assert!(first.newest_ms > 0);
        files
            .write(
                &fact("two", CodingFactType::User, 2)
                    .with_description("a much longer description than before"),
            )
            .await
            .expect("write two");
        let second = files.stamp().await.expect("stamp");
        assert_eq!(second.files, 2);
        assert!(second.total_len > first.total_len);
        assert!(second.newest_ms >= first.newest_ms);
        files.remove("one").await.expect("remove");
        let third = files.stamp().await.expect("stamp");
        assert_eq!(third.files, 1);
    }
    #[tokio::test]
    async fn facts_survive_a_write_read_all_round_trip() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        let facts = vec![
            fact("build", CodingFactType::Project, 3)
                .with_tags(vec!["build".to_owned()])
                .with_path("docs/BUILD.md"),
            fact("theme", CodingFactType::User, 2),
            fact("never-again", CodingFactType::Feedback, 1),
        ];
        for fact in &facts {
            files.write(fact).await.expect("write");
        }
        let mut expected = facts.clone();
        expected.sort_by(|left, right| left.name.cmp(&right.name));
        assert_eq!(files.read_all().await.expect("read_all"), expected);
    }
    #[tokio::test]
    async fn read_all_report_names_the_files_it_could_not_decode() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        files
            .write(&fact("build", CodingFactType::Project, 1))
            .await
            .expect("write");
        let prose = files.dir().join("hand-written.md");
        std::fs::write(&prose, "just prose, not a coding fact\n").expect("write the prose");

        let (facts, skipped) = files.read_all_report().await.expect("read_all_report");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].name, "build");
        assert_eq!(skipped.len(), 1, "the undecodable file is reported");
        assert_eq!(skipped[0].0, prose);
        assert!(
            !skipped[0].1.to_string().is_empty(),
            "the skip reason is reported"
        );
        // `read_all` keeps its documented meaning: only the decodable facts.
        assert_eq!(files.read_all().await.expect("read_all"), facts);
    }

    #[tokio::test]
    async fn the_level_round_trip_holds_through_files() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        let facts = vec![
            fact("build", CodingFactType::Project, 3),
            fact("theme", CodingFactType::User, 2),
            fact("never-again", CodingFactType::Feedback, 1),
        ];
        for fact in &facts {
            files.write(fact).await.expect("write");
        }
        let stored = files.read_all().await.expect("read_all");
        for level in IndexLevel::ALL {
            let rendered = render_level(&stored, level);
            let parsed = parse_level(&rendered).expect("the rendered index parses");
            let expected: Vec<CodingFact> = {
                let mut selected: Vec<CodingFact> = facts
                    .iter()
                    .filter(|fact| level.contains(fact.fact_type))
                    .cloned()
                    .collect();
                selected.sort_by(|left, right| {
                    rank(left.fact_type)
                        .cmp(&rank(right.fact_type))
                        .then_with(|| left.name.cmp(&right.name))
                });
                selected
            };
            // A rendered index does not record the scope, so both sides are compared in
            // the default scope the parser binds them to.
            assert_eq!(
                rebind(parsed, MemoryScope::default()),
                rebind(expected, MemoryScope::default())
            );
        }
    }
    #[tokio::test]
    async fn a_hand_edited_index_re_imports_byte_for_byte() {
        let directory = tempdir().expect("temp dir");
        let files = FactFiles::new(directory.path(), scope());
        let facts = vec![
            fact("build", CodingFactType::Project, 3),
            fact("gamma", CodingFactType::Project, 2),
            fact("theme", CodingFactType::User, 1),
        ];
        for fact in &facts {
            files.write(fact).await.expect("write");
        }
        let rendered = render_level(&facts, IndexLevel::Memory);
        let edited = rendered
            .replace(
                "- description: about build",
                "- description: about build (edited)",
            )
            .replace(
                "body of build\n",
                "body of build\nplus a hand-written line\n",
            );
        assert_ne!(edited, rendered, "the edit must change the document");
        let edited_facts = parse_level(&edited).expect("the edited index parses");
        for fact in &edited_facts {
            files.write(fact).await.expect("re-import");
        }
        let stored = files.read_all().await.expect("read_all");
        assert_eq!(render_level(&stored, IndexLevel::Memory), edited);
        let imported = stored
            .iter()
            .find(|fact| fact.name == "build")
            .expect("build survives");
        assert_eq!(imported.description, "about build (edited)");
        assert_eq!(imported.content, "body of build\nplus a hand-written line");
    }
}
