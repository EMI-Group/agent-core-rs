//! Source metadata for `PersonalContext` sources.
//!
//! The portable subset of Python's `personal_context/source_metadata.py`:
//! locator normalisation, the stable `src_<32 hex>` id, the ten-field record and
//! a store persisting one record per source.
//!
//! **Deviation from Python.** Python renders a program-owned Markdown document
//! per source (`<source_id>.md`); this port keeps the same fields and names in
//! one atomic JSON document (`<source_id>.json`), so ids stay wire-compatible
//! while the container follows the crate-wide `<path>.tmp` + rename form. The
//! Markdown heading cross-check has no JSON analogue; the identity, locator and
//! digest invariants do.
//!
//! Locators use the hand-rolled parser below instead of the `url` crate (which
//! this crate does not depend on): scheme, optional userinfo, bracketed IPv6,
//! port and path only, with `[redacted]` for an unusable authority. Timestamps
//! are RFC3339 UTC strings, so the persisted form is timezone-explicit.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use chrono::SecondsFormat;
use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::rails::sha256::sha256_prefix;

use super::atomic::{read_json_opt, write_json_atomic};

/// Number of hex characters kept from the SHA-256 digest of a locator.
pub const SOURCE_ID_HEX_CHARS: usize = 32;

/// Largest metadata document accepted from disk, in bytes.
pub const MAX_METADATA_BYTES: u64 = 64 * 1024;

/// Longest title preserved after whitespace folding, in characters.
pub const MAX_TITLE_CHARS: usize = 512;

/// Title stored when the folded candidate title is empty.
pub const DEFAULT_SOURCE_TITLE: &str = "Atomic source";

/// Prefix every source id carries.
const SOURCE_ID_PREFIX: &str = "src_";

/// Extension of a persisted metadata document.
const METADATA_EXTENSION: &str = "json";

/// The document label used in every store error message.
const METADATA_LABEL: &str = "source metadata";

/// One `PersonalContext` source, with exactly the Python field set and names.
///
/// Every field is a `String` because Python persists these values as text, and
/// unknown fields in a stored document are ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SourceMetadata {
    /// Stable id of the source, `src_` plus 32 lowercase hex characters.
    pub source_id: String,
    /// Coarse kind of the source (Python's `resource` when set, else provider).
    pub source_type: String,
    /// Human-readable title, already folded by [`display_title`].
    pub title: String,
    /// Normalised locator the id was derived from.
    pub locator: String,
    /// Provider that observed the source.
    pub provider: String,
    /// Service within the provider (Python's `service_id`).
    pub service: String,
    /// RFC3339 UTC timestamp of the first observation.
    pub first_seen: String,
    /// RFC3339 UTC timestamp of the latest observation.
    pub last_seen: String,
    /// Identity of the latest revision reported by the provider.
    pub latest_revision: String,
    /// SHA-256 of the latest content, 64 lowercase hex characters.
    pub latest_hash: String,
}

impl SourceMetadata {
    /// Builds an empty-except-identity record; other fields keep their defaults.
    ///
    /// [`SourceMetadataStore::upsert`] validates and normalises them before
    /// anything is written.
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        locator: impl Into<String>,
        provider: impl Into<String>,
        service: impl Into<String>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            locator: locator.into(),
            provider: provider.into(),
            service: service.into(),
            ..Self::default()
        }
    }
}

/// Returns the compiled source-id pattern, or `None` if it failed to compile.
fn source_id_pattern() -> Option<&'static Regex> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    PATTERN
        .get_or_init(|| Regex::new(r"^src_[0-9a-f]{32}$").ok())
        .as_ref()
}

/// Returns whether `value` is a source id: `src_` plus 32 lowercase hex chars.
#[must_use]
pub fn is_source_id(value: &str) -> bool {
    match source_id_pattern() {
        Some(pattern) => pattern.is_match(value),
        None => is_source_id_manually(value),
    }
}

/// Panic-free fallback for [`is_source_id`] when the pattern is unavailable.
fn is_source_id_manually(value: &str) -> bool {
    value.len() == SOURCE_ID_PREFIX.len() + SOURCE_ID_HEX_CHARS
        && value.starts_with(SOURCE_ID_PREFIX)
        && value[SOURCE_ID_PREFIX.len()..]
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// Returns whether `value` is 64 lowercase hex characters.
fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// Builds the validation error for a blank metadata field.
fn field_error(field: &str) -> Error {
    Error::validation(format!("atomic source metadata {field} is invalid"))
}

/// Builds the config error for an IO failure on the store directory.
fn config_error(action: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::config(format!(
        "failed to {action} {METADATA_LABEL} {}: {error}",
        path.display()
    ))
}

/// Normalises a source locator so it can be hashed into a stable id.
///
/// HTTP and HTTPS locators keep only the scheme, the credential-free authority
/// and the path: scheme and host are lowercased, userinfo is dropped, a valid
/// numeric port is kept, IPv6 hosts are bracketed, and query and fragment are
/// discarded. An unusable authority (no host, bad port, unclosed IPv6 bracket,
/// bare multiple-colon host) becomes `<scheme>://[redacted]`; any other input is
/// returned trimmed.
///
/// # Errors
///
/// `Validation` for an empty or whitespace-only locator.
pub fn normalize_source_locator(original: &str) -> Result<String> {
    let locator = original.trim();
    if locator.is_empty() {
        return Err(Error::validation(
            "atomic source metadata locator is invalid",
        ));
    }
    let Some((scheme, remainder)) = locator.split_once("://") else {
        return Ok(locator.to_string());
    };
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return Ok(locator.to_string());
    }
    let (authority, path) = split_authority(remainder);
    match normalize_authority(authority) {
        Some(authority) => Ok(format!("{scheme}://{authority}{path}")),
        None => Ok(format!("{scheme}://[redacted]")),
    }
}

/// Splits the text after `://` into its authority and its query-free path.
fn split_authority(remainder: &str) -> (&str, &str) {
    let authority_end = remainder.find(['/', '?', '#']).unwrap_or(remainder.len());
    let (authority, rest) = remainder.split_at(authority_end);
    let path_end = rest.find(['?', '#']).unwrap_or(rest.len());
    (authority, &rest[..path_end])
}

/// Reduces an authority to `host[:port]`, or `None` when it is unusable.
fn normalize_authority(authority: &str) -> Option<String> {
    let origin = authority
        .rsplit_once('@')
        .map_or(authority, |(_userinfo, origin)| origin);
    if origin.is_empty() {
        return None;
    }
    if let Some(bracketed) = origin.strip_prefix('[') {
        let (host, after) = bracketed.split_once(']')?;
        if host.is_empty() {
            return None;
        }
        let port = match after {
            "" => None,
            _ => Some(parse_port(after.strip_prefix(':')?)?),
        };
        let host = host.to_ascii_lowercase();
        return Some(match port {
            Some(port) => format!("[{host}]:{port}"),
            None => format!("[{host}]"),
        });
    }
    if origin.matches(':').count() > 1 {
        // A bare IPv6 host: ambiguous without brackets, so unusable.
        return None;
    }
    let (host, port) = match origin.split_once(':') {
        Some((host, port)) => (host, Some(port)),
        None => (origin, None),
    };
    if host.is_empty() {
        return None;
    }
    let port = match port {
        None | Some("") => None,
        Some(port) => Some(parse_port(port)?),
    };
    let host = host.to_ascii_lowercase();
    Some(match port {
        Some(port) => format!("{host}:{port}"),
        None => host,
    })
}

/// Parses a decimal port, rejecting non-digits and values above `65535`.
fn parse_port(value: &str) -> Option<u16> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse::<u16>().ok()
}

/// Returns the stable `src_<32 hex>` id of a locator: the first
/// [`SOURCE_ID_HEX_CHARS`] characters of the SHA-256 digest of the *normalised*
/// locator's UTF-8 bytes, so two spellings of one URL share one id.
///
/// # Errors
///
/// Propagates the error of [`normalize_source_locator`].
pub fn source_id_for_locator(locator: &str) -> Result<String> {
    let normalized = normalize_source_locator(locator)?;
    Ok(format!(
        "{SOURCE_ID_PREFIX}{}",
        sha256_prefix(normalized.as_bytes(), SOURCE_ID_HEX_CHARS)
    ))
}

/// Renders an observation timestamp as an RFC3339 UTC string with a `Z` offset;
/// fractional seconds appear only when present. A value outside chrono's range
/// falls back to the Unix epoch instead of panicking.
#[must_use]
pub fn observed_at_string(observed_at_ms: i64) -> String {
    let moment = chrono::DateTime::from_timestamp_millis(observed_at_ms).unwrap_or_default();
    moment.to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

/// Folds a candidate title: whitespace runs collapse to one space, the result
/// is trimmed and truncated to [`MAX_TITLE_CHARS`] characters on a character
/// boundary, and an empty result becomes [`DEFAULT_SOURCE_TITLE`].
#[must_use]
pub fn display_title(value: &str) -> String {
    let folded = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if folded.is_empty() {
        return DEFAULT_SOURCE_TITLE.to_string();
    }
    folded.chars().take(MAX_TITLE_CHARS).collect()
}

/// Rejects a path whose chain contains a symlink.
///
/// Every ancestor is inspected (a handful of `symlink_metadata` calls), so a
/// redirecting parent cannot smuggle a metadata read or write elsewhere;
/// missing components are skipped because they cannot be symlinks.
///
/// # Errors
///
/// `Security` when any existing component of the chain is a symlink.
pub(crate) fn assert_no_symlink_chain(path: &Path) -> Result<()> {
    let mut current = Some(path);
    while let Some(component) = current {
        let is_symlink = std::fs::symlink_metadata(component)
            .is_ok_and(|metadata| metadata.file_type().is_symlink());
        if is_symlink {
            return Err(Error::security(
                "atomic source metadata path must not traverse a symlink",
            ));
        }
        current = component.parent();
    }
    Ok(())
}

/// Per-source metadata documents under one root directory.
///
/// The store owns only the root path, so several handles can point at the same
/// directory; every read re-checks the symlink chain and document identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceMetadataStore {
    root: PathBuf,
}

impl SourceMetadataStore {
    /// Creates a store over `root` without touching the filesystem.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Returns the root directory metadata documents live in.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the document path of `source_id`, without checking existence.
    ///
    /// # Errors
    ///
    /// `Validation` when `source_id` is not a well-formed source id.
    pub fn path_for(&self, source_id: &str) -> Result<PathBuf> {
        if !is_source_id(source_id) {
            return Err(Error::validation("atomic source ID is invalid"));
        }
        Ok(self.root.join(format!("{source_id}.{METADATA_EXTENSION}")))
    }

    /// Reads one record, or `None` when the document does not exist.
    ///
    /// # Errors
    ///
    /// `Validation` for a bad id, a malformed or oversized document, an
    /// identity mismatch or a bad `latest_hash`; `Security` for a symlink;
    /// `Config` for other IO failures.
    pub fn read(&self, source_id: &str) -> Result<Option<SourceMetadata>> {
        let path = self.path_for(source_id)?;
        assert_no_symlink_chain(&path)?;
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(config_error("read", &path, &error)),
        };
        if !metadata.is_file() || metadata.len() > MAX_METADATA_BYTES {
            return Err(Error::validation(
                "atomic source metadata file is missing or too large",
            ));
        }
        let Some(record) = read_json_opt::<SourceMetadata>(&path, METADATA_LABEL)? else {
            return Ok(None);
        };
        if record.source_id != source_id || source_id_for_locator(&record.locator)? != source_id {
            return Err(Error::validation(
                "atomic source metadata identity does not match its filename",
            ));
        }
        if !is_sha256_hex(&record.latest_hash) {
            return Err(Error::validation(
                "atomic source metadata latest_hash is invalid",
            ));
        }
        Ok(Some(record))
    }

    /// Lists every readable record, sorted by `source_id`.
    ///
    /// A missing root is an empty list; anything that is not a
    /// `src_<32 hex>.json` entry is ignored, and a matching entry is read
    /// strictly, so one corrupt document fails the whole listing.
    ///
    /// # Errors
    ///
    /// `Security` for a symlinked root, ancestor or entry; `Validation` for a
    /// malformed or inconsistent document; `Config` for other IO failures.
    pub fn list(&self) -> Result<Vec<SourceMetadata>> {
        assert_no_symlink_chain(&self.root)?;
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(config_error("list", &self.root, &error)),
        };
        let suffix = format!(".{METADATA_EXTENSION}");
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| config_error("list", &self.root, &error))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(stem) = name.strip_suffix(&suffix) else {
                continue;
            };
            if is_source_id(stem) {
                ids.push(stem.to_string());
            }
        }
        ids.sort();
        let mut records = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(record) = self.read(&id)? {
                records.push(record);
            }
        }
        Ok(records)
    }

    /// Creates or updates the document of `candidate` and returns what was
    /// stored.
    ///
    /// The locator is normalised, the id is recomputed from it and must match
    /// the candidate's, `source_type`/`provider`/`service` must be non-empty
    /// after trimming, and the title goes through [`display_title`]. On update
    /// the existing `first_seen` is preserved while `last_seen`,
    /// `latest_revision`, `latest_hash`, `title`, `provider`, `service` and
    /// `source_type` come from the candidate.
    ///
    /// `latest_hash` is written as given; [`SourceMetadataStore::read`] enforces
    /// its 64-lowercase-hex shape, so the caller must supply a SHA-256 digest.
    ///
    /// # Errors
    ///
    /// `Validation` for an invalid locator, a blank field or an id/locator
    /// mismatch; `Security` for a symlinked root or ancestor; `Config` when the
    /// root, the existing document or the write cannot be handled.
    pub fn upsert(&self, candidate: SourceMetadata) -> Result<SourceMetadata> {
        for (name, value) in [
            ("source_type", &candidate.source_type),
            ("provider", &candidate.provider),
            ("service", &candidate.service),
        ] {
            if value.trim().is_empty() {
                return Err(field_error(name));
            }
        }
        let locator = normalize_source_locator(&candidate.locator)?;
        let source_id = source_id_for_locator(&locator)?;
        if candidate.source_id != source_id {
            return Err(Error::validation(
                "atomic source metadata identity does not match its locator",
            ));
        }
        std::fs::create_dir_all(&self.root)
            .map_err(|error| config_error("create", &self.root, &error))?;
        assert_no_symlink_chain(&self.root)?;
        let existing = self.read(&source_id)?;
        let first_seen = existing
            .as_ref()
            .filter(|record| !record.first_seen.is_empty())
            .map_or_else(
                || candidate.first_seen.trim().to_string(),
                |record| record.first_seen.clone(),
            );
        let stored = SourceMetadata {
            source_id: source_id.clone(),
            source_type: candidate.source_type.trim().to_string(),
            title: display_title(&candidate.title),
            locator,
            provider: candidate.provider.trim().to_string(),
            service: candidate.service.trim().to_string(),
            first_seen,
            last_seen: candidate.last_seen.trim().to_string(),
            latest_revision: candidate.latest_revision,
            latest_hash: candidate.latest_hash,
        };
        write_json_atomic(&self.path_for(&source_id)?, &stored, METADATA_LABEL)?;
        Ok(stored)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    fn norm(raw: &str) -> String {
        normalize_source_locator(raw).unwrap()
    }

    fn temp_store() -> (tempfile::TempDir, SourceMetadataStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = SourceMetadataStore::new(dir.path());
        (dir, store)
    }

    fn metadata(locator: &str, title: &str, first: &str, last: &str) -> SourceMetadata {
        SourceMetadata {
            source_id: source_id_for_locator(locator).unwrap(),
            source_type: "web".to_string(),
            title: title.to_string(),
            locator: locator.to_string(),
            provider: "browser".to_string(),
            service: "chrome".to_string(),
            first_seen: first.to_string(),
            last_seen: last.to_string(),
            latest_revision: "rev-1".to_string(),
            latest_hash: "a".repeat(64),
        }
    }

    #[test]
    fn locators_are_normalised() {
        assert_eq!(
            norm("https://user:pw@Example.COM/a?x=1#f"),
            "https://example.com/a"
        );
        assert_eq!(norm("HTTP://Example.COM"), "http://example.com");
        assert_eq!(
            norm("https://example.com:8443/Path/"),
            "https://example.com:8443/Path/"
        );
        assert_eq!(norm("  /tmp/notes.txt  "), "/tmp/notes.txt");
        assert_eq!(norm("git://Host/Repo?x=1"), "git://Host/Repo?x=1");
        assert_eq!(norm("plain text"), "plain text");
        assert_eq!(
            norm("http://[2001:DB8::1]:8080/x"),
            "http://[2001:db8::1]:8080/x"
        );
        assert_eq!(norm("https://[::1]/"), "https://[::1]/");
        for raw in [
            "http://example.com:notaport/x",
            "https://example.com:70000/x",
            "http://[::1/x",
            "http://2001:db8::1/x",
            "http://user:pw@/x",
            "https://",
        ] {
            let redacted = norm(raw);
            assert!(redacted.ends_with("://[redacted]"), "{raw} => {redacted}");
        }
        for blank in ["", "   ", "\n\t"] {
            let error = normalize_source_locator(blank).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("locator is invalid"));
        }
    }

    #[test]
    fn source_ids_are_stable_and_strictly_shaped() {
        let id = source_id_for_locator("https://user:pw@Example.COM/a?x=1#f").unwrap();
        assert_eq!(id, source_id_for_locator("https://example.com/a").unwrap());
        assert_eq!(
            id,
            source_id_for_locator("  https://example.com/a  ").unwrap()
        );
        assert!(id.starts_with(SOURCE_ID_PREFIX));
        assert_eq!(id.len(), SOURCE_ID_PREFIX.len() + SOURCE_ID_HEX_CHARS);
        assert!(is_source_id(&id));
        assert!(is_source_id_manually(&id));
        assert_eq!(
            source_id_for_locator("").unwrap_err().kind(),
            ErrorKind::Validation
        );
        for bad in [
            "src_0123456789ABCDEF0123456789ABCDEF",
            "src_0123456789abcdef0123456789abcde",
            "src_0123456789abcdef0123456789abcdef0",
            "xyz_0123456789abcdef0123456789abcdef",
            "src_0123456789abcdef0123456789abcdeg",
            "",
        ] {
            assert!(!is_source_id(bad), "{bad}");
        }
        assert!(!is_source_id_manually("src_0123456789abcdef"));
    }

    #[test]
    fn titles_fold_whitespace_and_fall_back() {
        assert_eq!(display_title("  a \n\t b   c "), "a b c");
        assert_eq!(display_title("   "), DEFAULT_SOURCE_TITLE);
        assert_eq!(display_title(""), DEFAULT_SOURCE_TITLE);
        let long = format!("{}  tail", "x".repeat(MAX_TITLE_CHARS));
        assert_eq!(display_title(&long).chars().count(), MAX_TITLE_CHARS);
        let unicode = "é".repeat(MAX_TITLE_CHARS + 10);
        assert_eq!(display_title(&unicode).chars().count(), MAX_TITLE_CHARS);
    }

    #[test]
    fn timestamps_render_as_rfc3339_utc() {
        assert_eq!(observed_at_string(0), "1970-01-01T00:00:00Z");
        assert_eq!(
            observed_at_string(1_700_000_000_000),
            "2023-11-14T22:13:20Z"
        );
        assert_eq!(
            observed_at_string(1_700_000_000_123),
            "2023-11-14T22:13:20.123Z"
        );
        assert_eq!(observed_at_string(i64::MAX), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn store_handles_missing_documents_and_checks_ids() {
        let (dir, store) = temp_store();
        let id = source_id_for_locator("https://example.com/a").unwrap();
        assert_eq!(
            store.path_for(&id).unwrap(),
            dir.path().join(format!("{id}.json"))
        );
        assert_eq!(
            store.path_for("../escape").unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(store.read(&id).unwrap(), None);
        assert!(store.list().unwrap().is_empty());
        assert!(SourceMetadataStore::new(dir.path().join("absent"))
            .list()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn store_round_trips_and_lists_sorted() {
        let (dir, store) = temp_store();
        let record = metadata("https://example.com/a", "Example A", "first", "last");
        assert_eq!(store.upsert(record.clone()).unwrap(), record);
        assert_eq!(store.read(&record.source_id).unwrap(), Some(record.clone()));
        let other = metadata("https://example.com/b", "Example B", "first", "last");
        store.upsert(other.clone()).unwrap();
        let mut expected = vec![record, other];
        expected.sort_by(|left, right| left.source_id.cmp(&right.source_id));
        assert_eq!(store.list().unwrap(), expected);
        assert_eq!(store.root(), dir.path());
    }

    #[test]
    fn upsert_preserves_first_seen_and_updates_the_rest() {
        let (_dir, store) = temp_store();
        let first = metadata(
            "https://example.com/a",
            "First title",
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
        );
        store.upsert(first.clone()).unwrap();
        let mut update = first;
        update.title = "  Second \n title ".to_string();
        update.first_seen = "1999-01-01T00:00:00Z".to_string();
        update.last_seen = "2024-03-01T00:00:00Z".to_string();
        update.latest_revision = "rev-2".to_string();
        update.latest_hash = "b".repeat(64);
        update.provider = "other-provider".to_string();
        update.service = "other-service".to_string();
        update.source_type = "feed".to_string();
        let stored = store.upsert(update).unwrap();
        assert_eq!(stored.first_seen, "2024-01-01T00:00:00Z");
        assert_eq!(stored.last_seen, "2024-03-01T00:00:00Z");
        assert_eq!(stored.latest_revision, "rev-2");
        assert_eq!(stored.latest_hash, "b".repeat(64));
        assert_eq!(stored.title, "Second title");
        assert_eq!(stored.provider, "other-provider");
        assert_eq!(stored.service, "other-service");
        assert_eq!(stored.source_type, "feed");
        assert_eq!(store.read(&stored.source_id).unwrap(), Some(stored));
    }

    #[test]
    fn upsert_rejects_invalid_candidates() {
        let (_dir, store) = temp_store();
        for field in ["source_type", "provider", "service"] {
            let mut candidate = metadata("https://example.com/a", "A", "now", "now");
            match field {
                "provider" => candidate.provider.clear(),
                "service" => candidate.service.clear(),
                _ => candidate.source_type.clear(),
            }
            let error = store.upsert(candidate).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(
                error
                    .to_string()
                    .contains(&format!("atomic source metadata {field} is invalid")),
                "{error}"
            );
        }
        let mut mismatched = metadata("https://example.com/a", "A", "now", "now");
        mismatched.source_id = source_id_for_locator("https://example.com/b").unwrap();
        let error = store.upsert(mismatched).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("identity does not match its locator"));
    }

    #[test]
    fn store_rejects_inconsistent_or_oversized_documents() {
        let (dir, store) = temp_store();
        let record = metadata("https://example.com/a", "A", "now", "now");
        let path = store.path_for(&record.source_id).unwrap();

        std::fs::write(&path, "{\"source_id\":").unwrap();
        let error = store.read(&record.source_id).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("malformed source metadata file"));

        let mut swapped = record.clone();
        swapped.locator = "https://example.com/other".to_string();
        write_json_atomic(&path, &swapped, METADATA_LABEL).unwrap();
        let error = store.read(&record.source_id).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("identity does not match its filename"));

        let mut bad_hash = record.clone();
        bad_hash.latest_hash = "not-a-digest".to_string();
        write_json_atomic(&path, &bad_hash, METADATA_LABEL).unwrap();
        let error = store.read(&record.source_id).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("latest_hash is invalid"));

        let mut json = serde_json::to_value(&record).unwrap();
        json["future_field"] = serde_json::json!(true);
        write_json_atomic(&path, &json, METADATA_LABEL).unwrap();
        assert_eq!(store.read(&record.source_id).unwrap(), Some(record.clone()));

        let oversized = "x".repeat(usize::try_from(MAX_METADATA_BYTES).unwrap() + 1);
        std::fs::write(
            dir.path().join(format!("{}.json", record.source_id)),
            oversized,
        )
        .unwrap();
        let error = store.read(&record.source_id).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("file is missing or too large"));
    }

    #[cfg(unix)]
    #[test]
    fn store_rejects_symlink_chains() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        symlink(&real, &link).unwrap();
        let linked = SourceMetadataStore::new(&link);
        let id = source_id_for_locator("https://example.com/a").unwrap();
        assert_eq!(linked.read(&id).unwrap_err().kind(), ErrorKind::Security);
        assert_eq!(linked.list().unwrap_err().kind(), ErrorKind::Security);
        let error = linked
            .upsert(metadata("https://example.com/a", "A", "now", "now"))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert!(error.to_string().contains("must not traverse a symlink"));

        let store = SourceMetadataStore::new(dir.path());
        let record = metadata("https://example.com/b", "B", "now", "now");
        let elsewhere = dir.path().join("elsewhere.json");
        write_json_atomic(&elsewhere, &record, METADATA_LABEL).unwrap();
        symlink(&elsewhere, store.path_for(&record.source_id).unwrap()).unwrap();
        assert_eq!(
            store.read(&record.source_id).unwrap_err().kind(),
            ErrorKind::Security
        );
    }
}
