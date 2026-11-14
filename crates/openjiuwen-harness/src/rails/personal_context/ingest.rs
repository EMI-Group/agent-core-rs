//! Local-directory ingestion for the offline `PersonalContext` subset.
//!
//! This is the deterministic local-file half of the Python
//! `openjiuwen.harness.personal_context` fetch pipeline
//! (`fetch/local_files.py`, the local scan/read portion of
//! `context_pipeline.py`). It does **not** port the vendor fetchers, the
//! Markdown compilation pass, PDF/DOCX parsing or the atomic publication
//! step; those stay outside this subtree.
//!
//! # Scan contract
//!
//! [`ingest_local_directory`] walks one configured directory and turns each
//! regular, plain-text file into an [`IngestedDocument`]:
//!
//! - The directory is canonicalised once, then walked without following
//!   symlinks. Symbolic links and non-regular files are skipped, and so is
//!   any path with a `.git`, `.hg`, `.svn`, `node_modules` or `target`
//!   component (those directories are pruned whole). Hidden files and
//!   directories are otherwise *not* filtered by name.
//! - Only extensions in [`TEXT_EXTENSIONS`] are read; matching is
//!   case-insensitive and a file without an extension is skipped.
//! - A file becomes a document only if it is strictly valid UTF-8 and no
//!   larger than [`LocalDirectoryConfig::max_file_bytes`]. At most
//!   [`LocalDirectoryConfig::max_files`] documents are collected; every
//!   later eligible file is counted as skipped.
//! - The indexed text is a trimmed snippet of at most
//!   [`LocalDirectoryConfig::max_text_chars`] characters, truncated on a
//!   character boundary. The title comes from the page's first level-1
//!   heading (falling back to the file stem), and both `locator` and
//!   `source_id` are derived from the canonical absolute path.
//!
//! Read failures, walk errors and per-file skip reasons are reported as
//! bounded [`IngestReport::warnings`]; a broken subtree never aborts the
//! scan. [`IngestReport::ingested`] is simply the number of returned
//! documents: classification against a persisted index (`unchanged` versus
//! `updated`) needs a dedupe step and belongs to `store.rs`, so this module
//! leaves those two counters at zero.
//!
//! Nothing here panics on user input and no path opens a network
//! connection: the only IO is a canonicalising `stat` and per-file reads
//! under the configured root.
use std::cell::Cell;
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use openjiuwen_common::error::{Error, Result};

use super::search::page_heading_fields;
use super::source::{display_title, normalize_source_locator, source_id_for_locator};
use openjiuwen_sysop::util::sha256_hex;

/// Default per-file byte cap: files larger than 2 MiB are skipped.
pub const DEFAULT_MAX_FILE_BYTES: usize = 2 * 1024 * 1024;

/// Default document cap for one scan.
pub const DEFAULT_MAX_FILES: usize = 10_000;

/// Default number of characters kept per document and indexed for search.
pub const DEFAULT_SNIPPET_CHARS: usize = 8_000;

/// Extensions read by the local-files scan, without the leading dot.
///
/// The allowlist deliberately covers plain text, source code and
/// configuration formats whose bytes can be decoded as UTF-8 directly.
/// Formats that need a parser (PDF, DOCX, spreadsheets, images, archives)
/// belong to the deferred fetch pipeline and are skipped here. Matching is
/// case-insensitive on the file's final extension.
pub const TEXT_EXTENSIONS: [&str; 42] = [
    "md", "markdown", "txt", "text", "json", "jsonl", "toml", "yaml", "yml", "rs", "py", "js",
    "jsx", "ts", "tsx", "html", "htm", "css", "scss", "csv", "tsv", "xml", "sh", "bash", "zsh",
    "go", "java", "kt", "c", "h", "cpp", "hpp", "cs", "rb", "php", "sql", "ini", "cfg", "conf",
    "log", "rst", "tex",
];

/// Path components whose directories are pruned before descent.
///
/// These are version-control metadata, dependency caches and build output:
/// scanning them would produce noise, duplicate content and potentially
/// millions of entries. The check is exact and case-sensitive on Unix,
/// mirroring Python's `_SKIPPED_DIRECTORIES` discipline.
const SKIPPED_COMPONENTS: [&str; 5] = [".git", ".hg", ".svn", "node_modules", "target"];

/// Maximum number of distinct warning messages kept in a report.
///
/// Beyond this cap the report carries one trailing summary entry instead of
/// ever-growing diagnostics.
const MAX_WARNINGS: usize = 32;

/// Provider id stamped on every document produced here.
const LOCAL_FILES_PROVIDER: &str = "local_files";

/// Configuration for one local-directory scan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LocalDirectoryConfig {
    /// Non-empty service id used to namespace the ingested sources.
    pub service: String,
    /// Directory to scan; it is canonicalised before the walk.
    pub directory: PathBuf,
    /// Per-file byte cap; a larger file is skipped with a warning.
    pub max_file_bytes: usize,
    /// Maximum number of documents produced by one scan.
    pub max_files: usize,
    /// Maximum characters of text kept per document.
    pub max_text_chars: usize,
}

impl LocalDirectoryConfig {
    /// Builds a configuration with the default caps.
    #[must_use]
    pub fn new(service: impl Into<String>, directory: impl Into<PathBuf>) -> Self {
        Self {
            service: service.into(),
            directory: directory.into(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_files: DEFAULT_MAX_FILES,
            max_text_chars: DEFAULT_SNIPPET_CHARS,
        }
    }

    /// Overrides the per-file byte cap.
    #[must_use]
    pub fn with_max_file_bytes(mut self, max_file_bytes: usize) -> Self {
        self.max_file_bytes = max_file_bytes;
        self
    }

    /// Overrides the document cap.
    #[must_use]
    pub fn with_max_files(mut self, max_files: usize) -> Self {
        self.max_files = max_files;
        self
    }

    /// Overrides the per-document character cap.
    #[must_use]
    pub fn with_max_text_chars(mut self, max_text_chars: usize) -> Self {
        self.max_text_chars = max_text_chars;
        self
    }
}

impl Default for LocalDirectoryConfig {
    fn default() -> Self {
        Self {
            service: String::new(),
            directory: PathBuf::new(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_files: DEFAULT_MAX_FILES,
            max_text_chars: DEFAULT_SNIPPET_CHARS,
        }
    }
}

/// One plain-text file materialised by a local-directory scan.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct IngestedDocument {
    /// Stable `src_<32 hex>` identifier derived from the normalised locator.
    pub source_id: String,
    /// Service id from the scan configuration.
    pub service: String,
    /// Provider id; always `"local_files"` for documents from this module.
    pub provider: String,
    /// Display title: the first level-1 heading, or the file stem.
    pub title: String,
    /// Canonical absolute path of the file as a display string.
    pub locator: String,
    /// Path relative to the scanned directory, with `/` separators.
    pub relative_path: String,
    /// SHA-256 hex digest (64 lowercase characters) of the raw bytes.
    pub content_hash: String,
    /// Size of the source file in bytes.
    pub size_bytes: u64,
    /// Trimmed text snippet of at most the configured character cap.
    pub text: String,
}

/// Outcome of one local-directory scan.
///
/// `scanned` counts every walk item examined: yielded entries, walk errors and
/// pruned excluded directories. `ingested` is the number of returned
/// documents; `skipped` covers excluded, oversized, non-UTF-8, capped and
/// non-regular files; `failed` covers IO and walk errors. `unchanged` and
/// `updated` stay zero here: only a store that dedupes against its persisted
/// index can decide whether an existing source is stale or changed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct IngestReport {
    /// Number of walk items examined.
    pub scanned: usize,
    /// Number of items deliberately skipped.
    pub skipped: usize,
    /// Number of documents already present in a caller's index (always 0 here).
    pub unchanged: usize,
    /// Number of documents whose text changed (always 0 here).
    pub updated: usize,
    /// Number of documents returned.
    pub ingested: usize,
    /// Number of items that failed to read or stat.
    pub failed: usize,
    /// Documents collected, sorted by `relative_path`.
    pub documents: Vec<IngestedDocument>,
    /// Bounded human-readable diagnostics, oldest first.
    pub warnings: Vec<String>,
}

impl IngestReport {
    /// Returns whether the scan produced no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

/// Scans one local directory and materialises its plain-text files.
///
/// The directory is canonicalised, walked without following symlinks, and
/// every eligible regular file is decoded as strict UTF-8 and bounded to the
/// configured snippet size. The returned documents are sorted by
/// `relative_path`, so the output is deterministic for a given tree.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// when the configured `service` is blank, when `directory` cannot be
/// canonicalised, or when the canonical path is not a directory. Individual
/// file failures never fail the scan; they are counted in
/// [`IngestReport::failed`] or [`IngestReport::skipped`] and described in
/// [`IngestReport::warnings`].
///
/// # Panics
///
/// Never panics on user input.
pub fn ingest_local_directory(config: &LocalDirectoryConfig) -> Result<IngestReport> {
    let service = config.service.trim();
    if service.is_empty() {
        return Err(Error::validation("local_files service is invalid"));
    }
    let root = std::fs::canonicalize(&config.directory).map_err(|_| {
        Error::validation(format!(
            "local_files directory {} is unavailable",
            config.directory.display()
        ))
    })?;
    if !root.is_dir() {
        return Err(Error::validation(format!(
            "local_files directory {} is not a directory",
            root.display()
        )));
    }

    // Pruned directories are never yielded by the walker, so their counts are
    // accumulated here and folded into the report after the scan.
    let pruned_scanned = Cell::new(0usize);
    let pruned_skipped = Cell::new(0usize);
    let walker = WalkDir::new(&root)
        .min_depth(1)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            let pruned = entry.file_type().is_dir() && is_skipped_component(entry.file_name());
            if pruned {
                pruned_scanned.set(pruned_scanned.get() + 1);
                pruned_skipped.set(pruned_skipped.get() + 1);
            }
            !pruned
        });

    let mut report = IngestReport::default();
    let mut warnings = WarningBuffer::default();
    let mut documents: Vec<IngestedDocument> = Vec::new();
    let mut limit_warned = false;

    for item in walker {
        report.scanned += 1;
        let entry = match item {
            Ok(entry) => entry,
            Err(error) => {
                report.failed += 1;
                warnings.push(format!("local_files entry error: {error}"));
                continue;
            }
        };
        let capped = documents.len() >= config.max_files;
        match attempt_ingest(
            &root,
            &entry,
            config,
            service,
            capped,
            &mut limit_warned,
            &mut warnings,
        ) {
            Attempt::Document(document) => {
                documents.push(document);
                report.ingested += 1;
            }
            Attempt::Skipped => report.skipped += 1,
            Attempt::Failed => report.failed += 1,
        }
    }

    report.scanned += pruned_scanned.get();
    report.skipped += pruned_skipped.get();
    documents.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    report.documents = documents;
    report.warnings = warnings.finish();
    Ok(report)
}

/// What one walk entry produced.
enum Attempt {
    /// A materialised document.
    Document(IngestedDocument),
    /// A deliberately excluded entry.
    Skipped,
    /// An entry that could not be read.
    Failed,
}

/// Classifies and materialises one walk entry.
///
/// `capped` means the document cap has already been reached; `limit_warned`
/// ensures the cap diagnostic is emitted only once.
fn attempt_ingest(
    root: &Path,
    entry: &walkdir::DirEntry,
    config: &LocalDirectoryConfig,
    service: &str,
    capped: bool,
    limit_warned: &mut bool,
    warnings: &mut WarningBuffer,
) -> Attempt {
    let Ok(relative) = entry.path().strip_prefix(root) else {
        warnings.push(format!(
            "local_files skipped {}: path is outside the configured directory",
            entry.path().display()
        ));
        return Attempt::Skipped;
    };
    if relative
        .components()
        .any(|component| matches!(component, Component::Normal(name) if is_skipped_component(name)))
    {
        return Attempt::Skipped;
    }
    let file_type = entry.file_type();
    if file_type.is_symlink() || !file_type.is_file() {
        return Attempt::Skipped;
    }
    if !extension_is_allowed(entry.path()) {
        return Attempt::Skipped;
    }
    if capped {
        if !*limit_warned {
            warnings.push(format!(
                "local_files file limit {} reached; remaining files are skipped",
                config.max_files
            ));
            *limit_warned = true;
        }
        return Attempt::Skipped;
    }

    let relative_path = relative_path_string(relative);
    let bytes = match std::fs::read(entry.path()) {
        Ok(bytes) => bytes,
        Err(error) => {
            warnings.push(format!(
                "local_files could not read {relative_path}: {error}"
            ));
            return Attempt::Failed;
        }
    };
    if bytes.len() > config.max_file_bytes {
        warnings.push(format!(
            "local_files skipped {relative_path}: {} bytes exceeds the {} byte limit",
            bytes.len(),
            config.max_file_bytes
        ));
        return Attempt::Skipped;
    }
    let Ok(text) = std::str::from_utf8(&bytes) else {
        warnings.push(format!(
            "local_files skipped {relative_path}: content is not valid UTF-8"
        ));
        return Attempt::Skipped;
    };

    // The walk root is already canonical, so joining the relative path yields
    // the canonical absolute path. Symlinked ancestors are never followed.
    let locator = entry.path().to_string_lossy().into_owned();
    let Ok(normalized) = normalize_source_locator(&locator) else {
        warnings.push(format!(
            "local_files skipped {relative_path}: locator is unusable"
        ));
        return Attempt::Skipped;
    };
    let Ok(source_id) = source_id_for_locator(&normalized) else {
        warnings.push(format!(
            "local_files skipped {relative_path}: source id is unavailable"
        ));
        return Attempt::Skipped;
    };

    let title = display_title(&page_heading_fields(&relative_path, text).0);
    let size_bytes = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    Attempt::Document(IngestedDocument {
        source_id,
        service: service.to_string(),
        provider: LOCAL_FILES_PROVIDER.to_string(),
        title,
        locator,
        relative_path,
        content_hash: sha256_hex(&bytes),
        size_bytes,
        text: truncate_chars(text.trim(), config.max_text_chars),
    })
}

/// Returns whether `name` is a path component whose directory is pruned.
fn is_skipped_component(name: &OsStr) -> bool {
    name.to_str()
        .is_some_and(|name| SKIPPED_COMPONENTS.contains(&name))
}

/// Returns whether `path` has an allowlisted extension, ignoring ASCII case.
fn extension_is_allowed(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| {
            TEXT_EXTENSIONS
                .iter()
                .any(|allowed| extension.eq_ignore_ascii_case(allowed))
        })
}

/// Renders a relative path with POSIX separators.
///
/// Components that cannot appear below a stripped walk root (`Prefix`,
/// `RootDir`, `CurDir`, `ParentDir`) are dropped defensively; non-UTF-8
/// names are rendered lossily rather than aborting the scan.
fn relative_path_string(relative: &Path) -> String {
    relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy()),
            Component::Prefix(_)
            | Component::RootDir
            | Component::CurDir
            | Component::ParentDir => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// Keeps at most `max_chars` characters, truncating on a character boundary.
fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.char_indices().nth(max_chars).map_or_else(
        || value.to_string(),
        |(index, _)| value[..index].to_string(),
    )
}

/// Bounded warning log: the first [`MAX_WARNINGS`] messages plus one summary.
#[derive(Debug, Default)]
struct WarningBuffer {
    messages: Vec<String>,
    suppressed: usize,
}

impl WarningBuffer {
    /// Records one warning, or counts it when the buffer is full.
    fn push(&mut self, message: String) {
        if self.messages.len() < MAX_WARNINGS {
            self.messages.push(message);
        } else {
            self.suppressed += 1;
        }
    }

    /// Returns the bounded messages and appends the suppression summary.
    fn finish(mut self) -> Vec<String> {
        if self.suppressed > 0 {
            self.messages.push(format!(
                "additional warnings suppressed: {} more",
                self.suppressed
            ));
        }
        self.messages
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::path::Path;

    use openjiuwen_common::error::ErrorKind;
    use tempfile::tempdir;

    use super::*;

    /// Writes `contents` to `relative` below `root`, creating parents.
    fn write(root: &Path, relative: &str, contents: impl AsRef<[u8]>) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    /// Builds a scan configuration for `directory`.
    fn config(directory: &Path) -> LocalDirectoryConfig {
        LocalDirectoryConfig::new("notes", directory)
    }

    #[test]
    fn ingests_plain_text_and_skips_excluded_entries() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        write(root, "note.md", "# Monday\n\nA body paragraph.");
        write(root, "plain.txt", "plain body");
        write(root, "image.png", [0x89, b'P', b'N', b'G']);
        write(root, ".git/config.md", "# hidden");
        write(root, "invalid.txt", [0xff, 0xfe, 0x00]);

        let report = ingest_local_directory(&config(root)).unwrap();

        let relative: Vec<&str> = report
            .documents
            .iter()
            .map(|document| document.relative_path.as_str())
            .collect();
        assert_eq!(relative, vec!["note.md", "plain.txt"]);
        assert_eq!(report.documents[0].title, "Monday");
        assert_eq!(report.documents[1].title, "plain");
        assert_eq!(report.documents[0].provider, "local_files");
        assert_eq!(report.documents[0].service, "notes");
        assert_eq!(report.ingested, 2);
        assert_eq!(report.unchanged, 0);
        assert_eq!(report.updated, 0);
        assert_eq!(report.skipped, 3);
        assert_eq!(report.failed, 0);
        // Two documents, three skipped entries, one pruned `.git` directory.
        assert_eq!(
            report.scanned,
            report.ingested + report.skipped + report.failed
        );
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("not valid UTF-8")));
        assert!(!report.is_empty());
    }

    #[test]
    fn relative_paths_use_posix_separators() {
        let temp = tempdir().unwrap();
        write(temp.path(), "a/b/deep.md", "# Deep\n");

        let report = ingest_local_directory(&config(temp.path())).unwrap();

        assert_eq!(report.documents.len(), 1);
        assert_eq!(report.documents[0].relative_path, "a/b/deep.md");
        assert!(report.documents[0].locator.ends_with("a/b/deep.md"));
        assert_eq!(report.documents[0].title, "Deep");
    }

    #[test]
    fn per_file_cap_skips_oversized_files() {
        let temp = tempdir().unwrap();
        write(temp.path(), "long.txt", "0123456789");
        write(temp.path(), "ok.txt", "short");
        let config = config(temp.path()).with_max_file_bytes(8);

        let report = ingest_local_directory(&config).unwrap();

        assert_eq!(report.documents.len(), 1);
        assert_eq!(report.documents[0].relative_path, "ok.txt");
        assert_eq!(report.skipped, 1);
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("byte limit")));
    }

    #[test]
    fn total_cap_limits_documents_and_warns_once() {
        let temp = tempdir().unwrap();
        write(temp.path(), "a.txt", "a");
        write(temp.path(), "b.txt", "b");
        write(temp.path(), "c.txt", "c");
        let config = config(temp.path()).with_max_files(1);

        let report = ingest_local_directory(&config).unwrap();

        assert_eq!(report.documents.len(), 1);
        assert_eq!(report.ingested, 1);
        assert_eq!(report.skipped, 2);
        assert_eq!(report.scanned, 3);
        assert_eq!(
            report
                .warnings
                .iter()
                .filter(|warning| warning.contains("file limit"))
                .count(),
            1
        );
    }

    #[test]
    fn source_id_is_stable_for_a_path_and_changes_with_it() {
        let first = tempdir().unwrap();
        let second = tempdir().unwrap();
        write(first.path(), "note.md", "# Note\n");
        write(second.path(), "note.md", "# Note\n");

        let one = ingest_local_directory(&config(first.path())).unwrap();
        let again = ingest_local_directory(&config(first.path())).unwrap();
        let other = ingest_local_directory(&config(second.path())).unwrap();

        let source_id = &one.documents[0].source_id;
        assert!(source_id.starts_with("src_"));
        assert_eq!(source_id.len(), "src_".len() + 32);
        assert!(source_id["src_".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
        assert_eq!(source_id, &again.documents[0].source_id);
        assert_ne!(source_id, &other.documents[0].source_id);
        assert_eq!(one.documents[0].locator, again.documents[0].locator);
    }

    #[test]
    fn content_hash_is_the_sha256_of_the_raw_bytes() {
        let temp = tempdir().unwrap();
        write(temp.path(), "abc.txt", "abc");

        let report = ingest_local_directory(&config(temp.path())).unwrap();

        assert_eq!(
            report.documents[0].content_hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(report.documents[0].content_hash, sha256_hex(b"abc"));
        assert_eq!(report.documents[0].size_bytes, 3);
    }

    #[test]
    fn snippet_truncates_on_a_char_boundary() {
        let temp = tempdir().unwrap();
        write(temp.path(), "cjk.txt", "你好世界");
        let config = config(temp.path()).with_max_text_chars(3);

        let report = ingest_local_directory(&config).unwrap();

        assert_eq!(report.documents[0].text, "你好世");
        assert_eq!(report.documents[0].text.len(), "你好世".len());
        assert_eq!(report.documents[0].title, "cjk");
    }

    #[test]
    fn zero_text_cap_yields_an_empty_snippet() {
        let temp = tempdir().unwrap();
        write(temp.path(), "note.md", "# Title\n\nbody");
        let config = config(temp.path()).with_max_text_chars(0);

        let report = ingest_local_directory(&config).unwrap();

        assert_eq!(report.documents[0].text, "");
        assert_eq!(report.documents[0].title, "Title");
    }

    #[test]
    fn documents_are_sorted_by_relative_path() {
        let temp = tempdir().unwrap();
        write(temp.path(), "z.txt", "z");
        write(temp.path(), "a.txt", "a");
        write(temp.path(), "m/n.txt", "n");

        let report = ingest_local_directory(&config(temp.path())).unwrap();

        let relative: Vec<&str> = report
            .documents
            .iter()
            .map(|document| document.relative_path.as_str())
            .collect();
        assert_eq!(relative, vec!["a.txt", "m/n.txt", "z.txt"]);
    }

    #[test]
    fn missing_directory_is_a_validation_error() {
        let temp = tempdir().unwrap();
        let config = config(&temp.path().join("missing"));

        let error = ingest_local_directory(&config).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("unavailable"));
    }

    #[test]
    fn non_directory_root_is_a_validation_error() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("file.txt");
        std::fs::write(&path, "not a directory").unwrap();
        let config = config(&path);

        let error = ingest_local_directory(&config).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("not a directory"));
    }

    #[test]
    fn blank_service_is_a_validation_error() {
        let temp = tempdir().unwrap();
        let config = LocalDirectoryConfig::new("   ", temp.path());

        let error = ingest_local_directory(&config).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("service is invalid"));
    }

    #[test]
    fn empty_directory_reports_an_empty_scan() {
        let temp = tempdir().unwrap();

        let report = ingest_local_directory(&config(temp.path())).unwrap();

        assert!(report.is_empty());
        assert_eq!(report.documents.len(), 0);
        assert_eq!(report.scanned, 0);
        assert_eq!(report.skipped, 0);
        assert_eq!(report.failed, 0);
        assert_eq!(report.warnings, Vec::<String>::new());
    }

    #[test]
    fn warnings_are_bounded() {
        let temp = tempdir().unwrap();
        for index in 0..(MAX_WARNINGS + 3) {
            write(temp.path(), &format!("bad-{index}.txt"), [0xff, 0xfe, 0x00]);
        }

        let report = ingest_local_directory(&config(temp.path())).unwrap();

        assert_eq!(report.documents.len(), 0);
        assert_eq!(report.skipped, MAX_WARNINGS + 3);
        assert_eq!(report.warnings.len(), MAX_WARNINGS + 1);
        assert!(report
            .warnings
            .last()
            .unwrap()
            .contains("additional warnings suppressed"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_files_are_skipped() {
        let temp = tempdir().unwrap();
        write(temp.path(), "target.md", "# Target\n");
        std::os::unix::fs::symlink(temp.path().join("target.md"), temp.path().join("link.md"))
            .unwrap();

        let report = ingest_local_directory(&config(temp.path())).unwrap();

        assert_eq!(report.documents.len(), 1);
        assert_eq!(report.documents[0].relative_path, "target.md");
        assert!(report.skipped >= 1);
    }

    #[test]
    fn default_configuration_keeps_the_documented_caps() {
        let config = LocalDirectoryConfig::default();

        assert_eq!(config.service, String::new());
        assert!(config.directory.as_os_str().is_empty());
        assert_eq!(config.max_file_bytes, DEFAULT_MAX_FILE_BYTES);
        assert_eq!(config.max_files, DEFAULT_MAX_FILES);
        assert_eq!(config.max_text_chars, DEFAULT_SNIPPET_CHARS);
    }

    #[test]
    fn configuration_round_trips_through_json() {
        let config = LocalDirectoryConfig::new("notes", "/tmp/notes")
            .with_max_file_bytes(64)
            .with_max_files(3)
            .with_max_text_chars(12);

        let json = serde_json::to_string(&config).unwrap();
        let decoded: LocalDirectoryConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, config);
        assert!(json.contains("\"max_file_bytes\":64"));
        assert!(json.contains("\"max_files\":3"));
        assert!(json.contains("\"max_text_chars\":12"));
    }
}
