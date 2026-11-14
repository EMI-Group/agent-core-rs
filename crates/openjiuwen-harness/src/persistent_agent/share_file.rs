//! Shareable state bundles: validated export and verified import on disk.
//!
//! Ports the self-contained subset of `openjiuwen/harness/persistent_agent/execution/share_file.py`
//! (v0.1.20): the listing rules Python applies to a file that an ending hands off, plus a bundle
//! format that carries those files next to the session binding they were produced for.
//!
//! A bundle is a directory — `<root>/<percent-encoded bundle id>/bundle.json` plus a `files/`
//! subdirectory holding one `<index>.bin` payload per listed file. Export validates the listing
//! (logical relative paths, captions, at most [`MAX_SHARED_FILES`] entries, no duplicate
//! `(path, share_mode)` pair), reads each source through the size limit, records the digest and
//! size of what it actually read, and writes `bundle.json` atomically last, so a partially written
//! bundle is never importable.
//!
//! Import is the verification side: it validates the record, resolves every payload, and rejects a
//! bundle whose payload length or SHA-256 no longer matches what the record claims. The module
//! never reads outside the caller-provided root, never touches the network and takes every
//! timestamp from an injected [`Clock`], so it is deterministic and offline in tests.
// Size note: ~480 lines of implementation plus an inline test module; the tests stay here so the export/import contract and its verification live together.
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_sysop::util::sha256_hex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::identity::SessionBinding;
use super::util::{
    atomic_write, is_valid_identifier, percent_encode_identifier, read_json, write_json_atomic,
};
use super::{Clock, SystemClock};

/// Schema version of a share bundle written by this module.
pub const SHARE_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Largest payload, in bytes, one shared file may have by default (32 MiB).
pub const SHARE_FILE_MAX_BYTES: u64 = 32 * 1024 * 1024;

/// Largest number of files one bundle may list, counted after duplicate listings are merged.
pub const MAX_SHARED_FILES: usize = 64;

/// Maximum length, in Unicode scalar values, of a shared file's logical path.
pub const FILE_PATH_MAX_CHARS: usize = 1024;

/// Maximum length, in Unicode scalar values, of a shared file's public caption.
pub const CAPTION_MAX_CHARS: usize = 4096;

/// Name of the JSON record inside a bundle directory.
pub const BUNDLE_FILE: &str = "bundle.json";

/// Every share mode, in the order Python's `SHARE_MODES` declares them.
pub const SHARE_MODES: [ShareMode; 2] = [ShareMode::Upload, ShareMode::Attachment];

/// Subdirectory of a bundle directory holding the copied payloads.
const PAYLOAD_DIR: &str = "files";

/// Number of hex characters in a SHA-256 digest.
const SHA256_HEX_CHARS: usize = 64;

/// How a shared file is delivered when the ending reaches the reply target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShareMode {
    /// Upload the file into the shared group space.
    Upload,
    /// Send the file as an attachment of the reply.
    Attachment,
}

impl ShareMode {
    /// Returns the persisted snake-case spelling of the mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Attachment => "attachment",
        }
    }
}

/// One file listed for sharing: its logical path, how it is shared and its public caption.
///
/// A caller builds an entry with [`sha256`](Self::sha256) and [`size`](Self::size) unset; export
/// fills both from the bytes it reads, so a listing can never claim a digest of content the bundle
/// does not carry. Unknown JSON keys are ignored on load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedFileEntry {
    /// Path of the file relative to the workspace that produced it.
    pub path: String,
    /// How the file is delivered.
    pub share_mode: ShareMode,
    /// Optional text published together with the file.
    #[serde(default)]
    pub caption: Option<String>,
    /// Lowercase hex SHA-256 of the payload; filled by export, required once persisted.
    #[serde(default)]
    pub sha256: Option<String>,
    /// Payload length in bytes; filled by export, required once persisted.
    #[serde(default)]
    pub size: Option<u64>,
}

impl SharedFileEntry {
    /// Validates the caller-supplied fields of a listing.
    ///
    /// [`sha256`](Self::sha256) and [`size`](Self::size) are not inspected: a caller lists a file
    /// by path, and [`ShareFile::export`] overwrites both from the bytes it reads.
    ///
    /// # Errors
    /// `Error::Validation` when [`path`](Self::path) is not a logical relative path (see
    /// [`is_logical_path`]) or [`caption`](Self::caption) is present but not a valid caption (see
    /// [`valid_caption`]).
    pub fn validate_input(&self) -> Result<()> {
        if !is_logical_path(&self.path) {
            return Err(Error::validation(format!(
                "share file path {:?} is not a logical relative path",
                self.path
            )));
        }
        if let Some(caption) = &self.caption {
            if !valid_caption(caption) {
                return Err(Error::validation(format!(
                    "share file caption for {} must be non-blank and at most {CAPTION_MAX_CHARS} characters",
                    self.path
                )));
            }
        }
        Ok(())
    }

    /// Validates a persisted listing, including the recorded digest and size.
    ///
    /// # Errors
    /// `Error::Validation` when [`validate_input`](Self::validate_input) fails, when
    /// [`sha256`](Self::sha256) is absent or not 64 lowercase hex characters, or when
    /// [`size`](Self::size) is absent.
    pub fn validate_persisted(&self) -> Result<()> {
        self.validate_input()?;
        let digest_ok = self.sha256.as_deref().is_some_and(is_sha256_hex);
        if !digest_ok {
            return Err(Error::validation(format!(
                "share file {} requires a {SHA256_HEX_CHARS}-character lowercase hex sha256 digest",
                self.path
            )));
        }
        if self.size.is_none() {
            return Err(Error::validation(format!(
                "share file {} requires a recorded size",
                self.path
            )));
        }
        Ok(())
    }
}

/// One exported bundle of shared files, as persisted in `bundle.json`.
///
/// Field names are the wire contract; [`extra`](Self::extra) keeps every key this version does not
/// know, so a record written by a newer writer survives an import/save round-trip untouched.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShareBundle {
    /// Schema version of the bundle; always [`SHARE_BUNDLE_SCHEMA_VERSION`] for a valid bundle.
    pub schema_version: u32,
    /// Identifier of the bundle; also the name of its directory.
    pub bundle_id: String,
    /// RFC 3339 timestamp the bundle was exported at, read from the injected clock.
    pub exported_at: String,
    /// Session the listed files were produced by.
    pub binding: SessionBinding,
    /// The listed files, in listing order; the index is the payload file name.
    pub files: Vec<SharedFileEntry>,
    /// Keys this version does not know, preserved verbatim across a load/save round-trip.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ShareBundle {
    /// Validates the whole record.
    ///
    /// # Errors
    /// `Error::Validation` when the schema version is unknown, the bundle id is not an identifier,
    /// the bundle lists more than [`MAX_SHARED_FILES`] files, one `(path, share_mode)` pair appears
    /// twice, or an entry fails [`SharedFileEntry::validate_persisted`].
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SHARE_BUNDLE_SCHEMA_VERSION {
            return Err(Error::validation(format!(
                "unsupported share bundle schema version {}; expected {SHARE_BUNDLE_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if !is_valid_identifier(&self.bundle_id) {
            return Err(Error::validation("share bundle id must be an identifier"));
        }
        check_file_list(&self.files)?;
        for entry in &self.files {
            entry.validate_persisted()?;
        }
        Ok(())
    }
}

/// A verified bundle plus the on-disk path of every payload it lists.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedShareBundle {
    /// The validated record, including any unknown keys it carried.
    pub bundle: ShareBundle,
    /// `(entry, payload path)` pairs, in listing order.
    pub files: Vec<(SharedFileEntry, PathBuf)>,
}

/// Returns whether `value` can be carried as a shared file's logical path.
///
/// A logical path is 1..=[`FILE_PATH_MAX_CHARS`] Unicode scalar values, relative (no leading `/`),
/// contains no `\` and no control characters, and has no empty, `.` or `..` segment.
#[must_use]
pub fn is_logical_path(value: &str) -> bool {
    let count = value.chars().count();
    if count == 0 || count > FILE_PATH_MAX_CHARS || value.starts_with('/') || value.contains('\\') {
        return false;
    }
    if value.chars().any(char::is_control) {
        return false;
    }
    value
        .split('/')
        .all(|segment| !matches!(segment, "" | "." | ".."))
}

/// Returns whether `value` is a usable public caption.
///
/// A caption holds at least one non-whitespace character and at most [`CAPTION_MAX_CHARS`] Unicode
/// scalar values.
#[must_use]
pub fn valid_caption(value: &str) -> bool {
    !value.trim().is_empty() && value.chars().count() <= CAPTION_MAX_CHARS
}

/// File-backed export and verified import of share bundles.
///
/// The service owns one root directory: `<root>/<percent-encoded bundle id>/` holds `bundle.json`
/// and its `files/` payloads. All timestamps come from the injected [`Clock`] (the wall clock by
/// default) and every payload is capped at [`SHARE_FILE_MAX_BYTES`] unless
/// [`with_max_bytes`](Self::with_max_bytes) says otherwise.
#[derive(Debug, Clone)]
pub struct ShareFile {
    /// Directory holding one subdirectory per bundle.
    root: PathBuf,
    /// Source of the `exported_at` timestamp.
    clock: Arc<dyn Clock>,
    /// Largest payload this service will export.
    max_bytes: u64,
}

impl ShareFile {
    /// Builds a service rooted at `root`, using the wall clock and the default size limit.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            clock: Arc::new(SystemClock::new()),
            max_bytes: SHARE_FILE_MAX_BYTES,
        }
    }

    /// Replaces the clock the `exported_at` timestamp is read from.
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Replaces the largest payload size this service accepts.
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Returns the root directory of this service.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the directory of the bundle with id `bundle_id`.
    ///
    /// The id is percent-encoded, so the result is always one safe path component.
    #[must_use]
    pub fn bundle_dir(&self, bundle_id: &str) -> PathBuf {
        self.root.join(percent_encode_identifier(bundle_id))
    }

    /// Returns the path of the record of the bundle with id `bundle_id`.
    #[must_use]
    pub fn bundle_path(&self, bundle_id: &str) -> PathBuf {
        self.bundle_dir(bundle_id).join(BUNDLE_FILE)
    }

    /// Exports `files` from `sources` as the bundle `bundle_id` and returns the written record.
    ///
    /// Entries keep their listing order: file `index` is copied to `<bundle dir>/files/<index>.bin`
    /// and its digest and size are recorded from the bytes actually read. Any `sha256`/`size` the
    /// caller supplied is overwritten. `bundle.json` is written last, so a failed export never
    /// leaves an importable bundle behind.
    ///
    /// # Errors
    /// `Error::Validation` when the bundle id is not an identifier, a listing fails
    /// [`SharedFileEntry::validate_input`], more than [`MAX_SHARED_FILES`] files are listed, a
    /// `(path, share_mode)` pair repeats, a path has no entry in `sources`, a source is missing,
    /// not a regular file or larger than the configured `max_bytes` limit; `Error::Session` when a payload or
    /// the record cannot be written.
    pub async fn export(
        &self,
        bundle_id: &str,
        binding: &SessionBinding,
        files: &[SharedFileEntry],
        sources: &BTreeMap<String, PathBuf>,
    ) -> Result<ShareBundle> {
        if !is_valid_identifier(bundle_id) {
            return Err(Error::validation("share bundle id must be an identifier"));
        }
        check_file_list(files)?;
        for entry in files {
            entry.validate_input()?;
        }
        let payload_dir = self.bundle_dir(bundle_id).join(PAYLOAD_DIR);
        let mut persisted = Vec::with_capacity(files.len());
        for (index, entry) in files.iter().enumerate() {
            let source = sources.get(&entry.path).ok_or_else(|| {
                Error::validation(format!("share source is missing for path {}", entry.path))
            })?;
            let bytes = self.read_source(&entry.path, source).await?;
            atomic_write(&payload_dir.join(format!("{index}.bin")), &bytes).await?;
            persisted.push(SharedFileEntry {
                path: entry.path.clone(),
                share_mode: entry.share_mode,
                caption: entry.caption.clone(),
                sha256: Some(sha256_hex(&bytes)),
                size: Some(bytes.len() as u64),
            });
        }
        let bundle = ShareBundle {
            schema_version: SHARE_BUNDLE_SCHEMA_VERSION,
            bundle_id: bundle_id.to_string(),
            exported_at: self.clock.now_rfc3339(),
            binding: binding.clone(),
            files: persisted,
            extra: BTreeMap::new(),
        };
        bundle.validate()?;
        write_json_atomic(&self.bundle_path(bundle_id), &bundle).await?;
        Ok(bundle)
    }

    /// Reads and verifies the bundle at `bundle_path`.
    ///
    /// The payload directory is derived from the record's own bundle id, so a record copied to
    /// another path still resolves against its bundle directory. Every payload must exist, have
    /// the recorded length and hash to the recorded SHA-256.
    ///
    /// # Errors
    /// `Error::NotFound` when the record is missing; `Error::Validation` when the record is not
    /// valid JSON, fails [`ShareBundle::validate`], a payload is missing, or a payload does not
    /// match its recorded size and digest; `Error::Session` when a file cannot be read.
    pub async fn import(&self, bundle_path: &Path) -> Result<ImportedShareBundle> {
        let bundle = read_json::<ShareBundle>(bundle_path)
            .await?
            .ok_or_else(|| {
                Error::not_found(format!(
                    "share bundle {} does not exist",
                    bundle_path.display()
                ))
            })?;
        bundle.validate()?;
        let payload_dir = self.bundle_dir(&bundle.bundle_id).join(PAYLOAD_DIR);
        let mut files = Vec::with_capacity(bundle.files.len());
        for (index, entry) in bundle.files.iter().enumerate() {
            let payload = payload_dir.join(format!("{index}.bin"));
            let bytes = match tokio::fs::read(&payload).await {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    return Err(Error::validation(format!(
                        "share bundle file {index} is missing at {}",
                        payload.display()
                    )));
                }
                Err(error) => {
                    return Err(Error::session(format!(
                        "failed to read {}: {error}",
                        payload.display()
                    )));
                }
            };
            let expected_sha = entry.sha256.as_deref().ok_or_else(|| {
                Error::validation(format!("share bundle file {index} has no recorded digest"))
            })?;
            let expected_size = entry.size.ok_or_else(|| {
                Error::validation(format!("share bundle file {index} has no recorded size"))
            })?;
            let actual_size = bytes.len() as u64;
            if actual_size != expected_size || sha256_hex(&bytes) != expected_sha {
                return Err(Error::validation(format!(
                    "share bundle file {index} does not match its recorded digest"
                )));
            }
            files.push((entry.clone(), payload));
        }
        Ok(ImportedShareBundle { bundle, files })
    }

    /// Imports the bundle with id `bundle_id` from this service's root.
    ///
    /// # Errors
    /// The errors of [`Self::import`].
    pub async fn import_bundle(&self, bundle_id: &str) -> Result<ImportedShareBundle> {
        self.import(&self.bundle_path(bundle_id)).await
    }

    /// Reads one source file after checking that it is a regular file within the size limit.
    ///
    /// The size is checked twice: once on the metadata and once on the bytes actually read, so a
    /// source that grows between the two checks is still refused.
    async fn read_source(&self, label: &str, source: &Path) -> Result<Vec<u8>> {
        let metadata = tokio::fs::metadata(source).await.map_err(|error| {
            Error::validation(format!("share source {label} is not readable: {error}"))
        })?;
        if !metadata.is_file() {
            return Err(Error::validation(format!(
                "share source {label} is not a regular file"
            )));
        }
        if metadata.len() > self.max_bytes {
            return Err(self.too_large(label, metadata.len()));
        }
        let bytes = tokio::fs::read(source).await.map_err(|error| {
            Error::validation(format!("share source {label} is not readable: {error}"))
        })?;
        if bytes.len() as u64 > self.max_bytes {
            return Err(self.too_large(label, bytes.len() as u64));
        }
        Ok(bytes)
    }

    /// Builds the size-limit rejection for `label`.
    fn too_large(&self, label: &str, size: u64) -> Error {
        Error::validation(format!(
            "share source {label} is {size} bytes; the limit is {} bytes",
            self.max_bytes
        ))
    }
}

/// Checks the file-count and uniqueness rules of a listing.
fn check_file_list(files: &[SharedFileEntry]) -> Result<()> {
    if files.len() > MAX_SHARED_FILES {
        return Err(Error::validation(format!(
            "share bundle lists {} files; the maximum is {MAX_SHARED_FILES}",
            files.len()
        )));
    }
    let mut seen: BTreeSet<(&str, ShareMode)> = BTreeSet::new();
    for entry in files {
        if !seen.insert((entry.path.as_str(), entry.share_mode)) {
            return Err(Error::validation(format!(
                "share bundle lists duplicate file {} with mode {}",
                entry.path,
                entry.share_mode.as_str()
            )));
        }
    }
    Ok(())
}

/// Returns whether `value` is a SHA-256 digest in lowercase hex.
fn is_sha256_hex(value: &str) -> bool {
    value.len() == SHA256_HEX_CHARS
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    use openjiuwen_common::error::Error;
    use openjiuwen_sysop::util::sha256_hex;
    use serde_json::json;

    use super::{
        is_logical_path, valid_caption, ShareBundle, ShareFile, ShareMode, SharedFileEntry,
        BUNDLE_FILE, CAPTION_MAX_CHARS, FILE_PATH_MAX_CHARS, MAX_SHARED_FILES,
        SHARE_BUNDLE_SCHEMA_VERSION, SHARE_MODES,
    };
    use crate::persistent_agent::identity::SessionBinding;
    use crate::persistent_agent::util::sha256_hex_str;
    use crate::persistent_agent::ManualClock;

    const STAMP: i64 = 1_700_000_000_000;
    const STAMP_TEXT: &str = "2023-11-14T22:13:20.000Z";

    fn service(dir: &tempfile::TempDir) -> ShareFile {
        ShareFile::new(dir.path().join("bundles")).with_clock(Arc::new(ManualClock::new(STAMP)))
    }

    fn binding() -> SessionBinding {
        SessionBinding::new("agent", "s/1").unwrap()
    }

    fn listing(path: &str, share_mode: ShareMode) -> SharedFileEntry {
        SharedFileEntry {
            path: path.to_string(),
            share_mode,
            caption: None,
            sha256: None,
            size: None,
        }
    }

    /// Writes `bytes` to a fresh source file under `<dir>/sources` and returns its path.
    fn source(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join("sources").join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn payload(service: &ShareFile, bundle_id: &str, index: usize) -> PathBuf {
        service
            .bundle_dir(bundle_id)
            .join("files")
            .join(format!("{index}.bin"))
    }

    #[tokio::test]
    async fn export_writes_the_record_and_payloads_and_import_returns_them() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let notes = source(&dir, "notes.md", b"# notes\n");
        let chart = source(&dir, "chart.png", b"\x89PNG payload");
        let sources = BTreeMap::from([
            ("docs/notes.md".to_string(), notes),
            ("assets/chart.png".to_string(), chart),
        ]);
        let files = vec![
            SharedFileEntry {
                caption: Some("release notes".to_string()),
                // Export overwrites a caller-supplied digest and size.
                sha256: Some("not-a-digest".to_string()),
                size: Some(1),
                ..listing("docs/notes.md", ShareMode::Upload)
            },
            listing("assets/chart.png", ShareMode::Attachment),
        ];

        let bundle = service
            .export("bundle-1", &binding(), &files, &sources)
            .await
            .unwrap();
        assert_eq!(bundle.schema_version, SHARE_BUNDLE_SCHEMA_VERSION);
        assert_eq!(bundle.bundle_id, "bundle-1");
        assert_eq!(bundle.binding, binding());
        assert_eq!(bundle.exported_at, STAMP_TEXT);
        assert!(bundle.extra.is_empty());
        assert_eq!(bundle.files.len(), 2);
        assert_eq!(bundle.files[0].caption.as_deref(), Some("release notes"));
        assert_eq!(
            bundle.files[0].sha256.as_deref(),
            Some(sha256_hex_str("# notes\n").as_str())
        );
        assert_eq!(bundle.files[0].size, Some(8));
        assert_eq!(bundle.files[1].share_mode, ShareMode::Attachment);
        assert_eq!(
            bundle.files[1].sha256.as_deref(),
            Some(sha256_hex(b"\x89PNG payload").as_str())
        );

        assert!(service.bundle_path("bundle-1").exists());
        let recorded: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(service.bundle_path("bundle-1"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(recorded["bundle_id"], json!("bundle-1"));
        assert_eq!(recorded["exported_at"], json!(STAMP_TEXT));
        assert_eq!(recorded["files"][0]["size"], json!(8));
        assert_eq!(
            tokio::fs::read(payload(&service, "bundle-1", 0))
                .await
                .unwrap(),
            b"# notes\n"
        );
        assert_eq!(
            tokio::fs::read(payload(&service, "bundle-1", 1))
                .await
                .unwrap(),
            b"\x89PNG payload"
        );

        let imported = service.import_bundle("bundle-1").await.unwrap();
        assert_eq!(imported.bundle, bundle);
        assert_eq!(imported.files.len(), 2);
        assert_eq!(imported.files[0].1, payload(&service, "bundle-1", 0));
        assert_eq!(imported.files[1].1, payload(&service, "bundle-1", 1));
        assert_eq!(
            imported.files[0].0.caption.as_deref(),
            Some("release notes")
        );
        assert_eq!(
            tokio::fs::read(&imported.files[1].1).await.unwrap(),
            b"\x89PNG payload"
        );
    }

    #[tokio::test]
    async fn exported_at_follows_the_injected_clock() {
        let dir = tempfile::TempDir::new().unwrap();
        let clock = Arc::new(ManualClock::new(STAMP));
        let service =
            ShareFile::new(dir.path()).with_clock(Arc::clone(&clock) as Arc<dyn super::Clock>);
        let bytes = source(&dir, "a.txt", b"a");
        let sources = BTreeMap::from([("a.txt".to_string(), bytes)]);
        let files = vec![listing("a.txt", ShareMode::Upload)];

        let first = service
            .export("one", &binding(), &files, &sources)
            .await
            .unwrap();
        assert_eq!(first.exported_at, STAMP_TEXT);

        clock.advance(1_000);
        let second = service
            .export("two", &binding(), &files, &sources)
            .await
            .unwrap();
        assert_eq!(second.exported_at, "2023-11-14T22:13:21.000Z");
    }

    #[tokio::test]
    async fn tampered_payload_is_rejected_on_import() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let sources = BTreeMap::from([(
            "docs/notes.md".to_string(),
            source(&dir, "notes.md", b"# notes\n"),
        )]);
        let files = vec![listing("docs/notes.md", ShareMode::Upload)];
        service
            .export("bundle-1", &binding(), &files, &sources)
            .await
            .unwrap();

        let path = payload(&service, "bundle-1", 0);
        let mut bytes = tokio::fs::read(&path).await.unwrap();
        bytes[0] ^= 0xFF;
        tokio::fs::write(&path, &bytes).await.unwrap();

        let error = service.import_bundle("bundle-1").await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error
            .to_string()
            .contains("share bundle file 0 does not match its recorded digest"));
    }

    #[tokio::test]
    async fn truncated_and_removed_payloads_are_rejected_on_import() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let sources = BTreeMap::from([(
            "docs/notes.md".to_string(),
            source(&dir, "notes.md", b"# notes\n"),
        )]);
        let files = vec![listing("docs/notes.md", ShareMode::Upload)];
        service
            .export("bundle-1", &binding(), &files, &sources)
            .await
            .unwrap();

        let path = payload(&service, "bundle-1", 0);
        let bytes = tokio::fs::read(&path).await.unwrap();
        tokio::fs::write(&path, &bytes[..bytes.len() - 1])
            .await
            .unwrap();
        let truncated = service.import_bundle("bundle-1").await.unwrap_err();
        assert!(matches!(truncated, Error::Validation(_)));
        assert!(truncated
            .to_string()
            .contains("does not match its recorded digest"));

        tokio::fs::remove_file(&path).await.unwrap();
        let missing = service.import_bundle("bundle-1").await.unwrap_err();
        assert!(matches!(missing, Error::Validation(_)));
        assert!(missing
            .to_string()
            .contains("share bundle file 0 is missing at"));
    }

    #[tokio::test]
    async fn export_rejects_a_missing_source() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let files = vec![listing("docs/notes.md", ShareMode::Upload)];
        let error = service
            .export("bundle-1", &binding(), &files, &BTreeMap::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error
            .to_string()
            .contains("share source is missing for path docs/notes.md"));
        assert!(!service.bundle_path("bundle-1").exists());
    }

    #[tokio::test]
    async fn export_rejects_a_source_above_the_size_limit_but_accepts_the_boundary() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir).with_max_bytes(4);
        let sources =
            BTreeMap::from([("small.bin".to_string(), source(&dir, "small.bin", b"1234"))]);
        let files = vec![listing("small.bin", ShareMode::Upload)];
        service
            .export("boundary", &binding(), &files, &sources)
            .await
            .unwrap();

        let large =
            BTreeMap::from([("large.bin".to_string(), source(&dir, "large.bin", b"12345"))]);
        let error = service
            .export(
                "too-large",
                &binding(),
                &[listing("large.bin", ShareMode::Upload)],
                &large,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error
            .to_string()
            .contains("is 5 bytes; the limit is 4 bytes"));
        assert!(!service.bundle_path("too-large").exists());
    }

    #[tokio::test]
    async fn export_rejects_a_non_file_source() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let directory = dir.path().join("sources").join("folder");
        std::fs::create_dir_all(&directory).unwrap();
        let sources = BTreeMap::from([("folder".to_string(), directory)]);
        let error = service
            .export(
                "bundle-1",
                &binding(),
                &[listing("folder", ShareMode::Upload)],
                &sources,
            )
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("share source folder is not a regular file"));
    }

    #[tokio::test]
    async fn export_enforces_the_file_count_limit() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let (files, sources) = numbered_listing(&dir, MAX_SHARED_FILES);
        let bundle = service
            .export("full", &binding(), &files, &sources)
            .await
            .unwrap();
        assert_eq!(bundle.files.len(), MAX_SHARED_FILES);

        let (over, over_sources) = numbered_listing(&dir, MAX_SHARED_FILES + 1);
        let error = service
            .export("over", &binding(), &over, &over_sources)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error
            .to_string()
            .contains("lists 65 files; the maximum is 64"));
        assert!(!service.bundle_path("over").exists());
    }

    /// Returns `count` distinct listings and their sources.
    fn numbered_listing(
        dir: &tempfile::TempDir,
        count: usize,
    ) -> (Vec<SharedFileEntry>, BTreeMap<String, PathBuf>) {
        let mut files = Vec::with_capacity(count);
        let mut sources = BTreeMap::new();
        for index in 0..count {
            let name = format!("f{index}.txt");
            sources.insert(
                name.clone(),
                source(dir, &name, format!("file {index}").as_bytes()),
            );
            files.push(listing(&name, ShareMode::Upload));
        }
        (files, sources)
    }

    #[test]
    fn logical_paths_reject_escapes_and_malformed_segments() {
        assert!(is_logical_path("a.txt"));
        assert!(is_logical_path("docs/notes.md"));
        assert!(is_logical_path("dir with space/a.txt"));
        assert!(is_logical_path(&"a".repeat(FILE_PATH_MAX_CHARS)));
        assert!(!is_logical_path(""));
        assert!(!is_logical_path("../x"));
        assert!(!is_logical_path("a/../x"));
        assert!(!is_logical_path("./a"));
        assert!(!is_logical_path("."));
        assert!(!is_logical_path("/abs"));
        assert!(!is_logical_path("a\\b"));
        assert!(!is_logical_path("a//b"));
        assert!(!is_logical_path("a/"));
        assert!(!is_logical_path("a\nb"));
        assert!(!is_logical_path("a\u{7f}b"));
        assert!(!is_logical_path(&"a".repeat(FILE_PATH_MAX_CHARS + 1)));
    }

    #[tokio::test]
    async fn export_rejects_an_invalid_logical_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let sources = BTreeMap::from([("a.txt".to_string(), source(&dir, "a.txt", b"a"))]);
        for path in ["../x", "/abs", "a\\b", "a//b", ""] {
            let error = service
                .export(
                    "bundle-1",
                    &binding(),
                    &[listing(path, ShareMode::Upload)],
                    &sources,
                )
                .await
                .unwrap_err();
            assert!(matches!(error, Error::Validation(_)), "path {path:?}");
            assert!(
                error.to_string().contains("is not a logical relative path"),
                "path {path:?}"
            );
        }
        assert!(!service.bundle_path("bundle-1").exists());
    }

    #[test]
    fn captions_must_be_non_blank_and_within_the_limit() {
        assert!(valid_caption("release notes"));
        assert!(valid_caption(&"a".repeat(CAPTION_MAX_CHARS)));
        assert!(!valid_caption(""));
        assert!(!valid_caption("   \n\t"));
        assert!(!valid_caption(&"a".repeat(CAPTION_MAX_CHARS + 1)));
    }

    #[tokio::test]
    async fn export_rejects_an_invalid_caption() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let sources = BTreeMap::from([("a.txt".to_string(), source(&dir, "a.txt", b"a"))]);
        for caption in [
            String::new(),
            "   ".to_string(),
            "a".repeat(CAPTION_MAX_CHARS + 1),
        ] {
            let files = vec![SharedFileEntry {
                caption: Some(caption),
                ..listing("a.txt", ShareMode::Upload)
            }];
            let error = service
                .export("bundle-1", &binding(), &files, &sources)
                .await
                .unwrap_err();
            assert!(matches!(error, Error::Validation(_)));
            assert!(error.to_string().contains("caption for a.txt"));
        }
    }

    #[tokio::test]
    async fn duplicate_path_and_mode_is_rejected_but_two_modes_are_allowed() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let sources = BTreeMap::from([("a.txt".to_string(), source(&dir, "a.txt", b"a"))]);
        let duplicate = vec![
            listing("a.txt", ShareMode::Upload),
            listing("a.txt", ShareMode::Upload),
        ];
        let error = service
            .export("duplicate", &binding(), &duplicate, &sources)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error
            .to_string()
            .contains("duplicate file a.txt with mode upload"));
        assert!(!service.bundle_path("duplicate").exists());

        let both_modes = vec![
            listing("a.txt", ShareMode::Upload),
            listing("a.txt", ShareMode::Attachment),
        ];
        let bundle = service
            .export("both-modes", &binding(), &both_modes, &sources)
            .await
            .unwrap();
        assert_eq!(bundle.files.len(), 2);
        assert_eq!(
            tokio::fs::read(payload(&service, "both-modes", 1))
                .await
                .unwrap(),
            b"a"
        );
    }

    #[tokio::test]
    async fn unknown_bundle_keys_survive_import_and_re_encoding() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let sources = BTreeMap::from([("a.txt".to_string(), source(&dir, "a.txt", b"a"))]);
        service
            .export(
                "bundle-1",
                &binding(),
                &[listing("a.txt", ShareMode::Upload)],
                &sources,
            )
            .await
            .unwrap();

        let path = service.bundle_path("bundle-1");
        let mut value: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        value["future_field"] = json!({"keep": true});
        value["schema_version"] = json!(SHARE_BUNDLE_SCHEMA_VERSION);
        tokio::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap())
            .await
            .unwrap();

        let imported = service.import_bundle("bundle-1").await.unwrap();
        assert_eq!(
            imported.bundle.extra.get("future_field"),
            Some(&json!({"keep": true}))
        );
        assert_eq!(imported.files.len(), 1);

        let reencoded = serde_json::to_value(&imported.bundle).unwrap();
        assert_eq!(reencoded.get("future_field"), Some(&json!({"keep": true})));
    }

    #[tokio::test]
    async fn a_missing_bundle_is_not_found() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let error = service.import_bundle("absent").await.unwrap_err();
        assert!(matches!(error, Error::NotFound(_)));
        assert!(error.to_string().contains("does not exist"));
    }

    #[test]
    fn bundle_dir_percent_encodes_the_id() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        assert_eq!(
            service.bundle_dir("b/1"),
            dir.path().join("bundles").join("b%2F1")
        );
        assert_eq!(
            service.bundle_path("b/1"),
            dir.path().join("bundles").join("b%2F1").join(BUNDLE_FILE)
        );
        assert_eq!(service.root(), dir.path().join("bundles"));
    }

    #[tokio::test]
    async fn export_rejects_a_non_identifier_bundle_id() {
        let dir = tempfile::TempDir::new().unwrap();
        let service = service(&dir);
        let error = service
            .export("has space", &binding(), &[], &BTreeMap::new())
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("share bundle id must be an identifier"));
    }

    #[test]
    fn bundle_validation_rejects_bad_versions_ids_digests_and_sizes() {
        let entry = SharedFileEntry {
            sha256: Some("ab".repeat(32)),
            size: Some(3),
            ..listing("a.txt", ShareMode::Upload)
        };
        let bundle = ShareBundle {
            schema_version: SHARE_BUNDLE_SCHEMA_VERSION,
            bundle_id: "bundle-1".to_string(),
            exported_at: STAMP_TEXT.to_string(),
            binding: binding(),
            files: vec![entry.clone()],
            extra: BTreeMap::new(),
        };
        bundle.validate().unwrap();

        let mut wrong_version = bundle.clone();
        wrong_version.schema_version = SHARE_BUNDLE_SCHEMA_VERSION + 1;
        assert!(wrong_version
            .validate()
            .unwrap_err()
            .to_string()
            .contains("unsupported share bundle schema version 2"));

        let mut bad_id = bundle.clone();
        bad_id.bundle_id = "has space".to_string();
        assert!(bad_id.validate().is_err());

        let mut uppercase = bundle.clone();
        uppercase.files[0].sha256 = Some("AB".repeat(32));
        assert!(uppercase
            .validate()
            .unwrap_err()
            .to_string()
            .contains("lowercase hex sha256"));

        let mut short = bundle.clone();
        short.files[0].sha256 = Some("abc".to_string());
        assert!(short.validate().is_err());

        let mut no_digest = bundle.clone();
        no_digest.files[0].sha256 = None;
        assert!(no_digest.validate().is_err());

        let mut no_size = bundle.clone();
        no_size.files[0].size = None;
        assert!(no_size.validate().is_err());

        let mut too_many = bundle.clone();
        too_many.files = vec![entry; MAX_SHARED_FILES + 1];
        assert!(too_many
            .validate()
            .unwrap_err()
            .to_string()
            .contains("lists 65 files; the maximum is 64"));
    }

    #[test]
    fn share_modes_use_snake_case_and_keep_their_order() {
        assert_eq!(
            serde_json::to_string(&ShareMode::Upload).unwrap(),
            "\"upload\""
        );
        assert_eq!(
            serde_json::to_string(&ShareMode::Attachment).unwrap(),
            "\"attachment\""
        );
        assert_eq!(SHARE_MODES, [ShareMode::Upload, ShareMode::Attachment]);
        assert_eq!(ShareMode::Upload.as_str(), "upload");
        assert!(ShareMode::Upload < ShareMode::Attachment);
    }

    #[test]
    fn entries_round_trip_and_tolerate_unknown_fields() {
        let entry = SharedFileEntry {
            sha256: Some("ab".repeat(32)),
            size: Some(3),
            caption: None,
            ..listing("a.txt", ShareMode::Upload)
        };
        let encoded = serde_json::to_value(&entry).unwrap();
        assert_eq!(encoded["share_mode"], json!("upload"));
        assert_eq!(encoded.get("caption"), Some(&json!(null)));
        let decoded: SharedFileEntry = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, entry);

        let tolerant: SharedFileEntry = serde_json::from_value(json!({
            "path": "a.txt",
            "share_mode": "attachment",
            "future_field": 1,
        }))
        .unwrap();
        assert_eq!(tolerant.share_mode, ShareMode::Attachment);
        assert!(tolerant.caption.is_none());
        assert!(tolerant.sha256.is_none());
        assert!(tolerant.size.is_none());
    }
}
