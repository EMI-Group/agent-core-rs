//! Portable-path safety checks for `PersonalContext`-managed Context trees.
//!
//! Port of `openjiuwen/harness/personal_context/path_safety.py`: the portable
//! segment rules (forbidden characters, Windows device names, length limits,
//! trailing space or dot), the stricter user-visible semantic-name rules, the
//! service-id validation/storage encoding, and the resolver that keeps every
//! managed path a plain POSIX path below its root.
//!
//! # Documented deviations
//!
//! Python additionally rejects a segment that is not `NFC`-normalised; without a
//! Unicode normalisation crate the Rust port cannot detect that and does not
//! check it.
//! Python rejects the `Cc` and `Cf` Unicode categories; the Rust port rejects
//! only [`char::is_control`] (which covers `Cc`), so formatting characters
//! (`Cf`, e.g. a zero-width space) are accepted.
//! Python also detects Windows reparse points; the Rust port targets Unix and
//! detects links through `std::fs::symlink_metadata(...).file_type().is_symlink()`
//! only, so it sets no Windows reparse-point checks.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use openjiuwen_common::error::{Error, Result};
use regex::Regex;

use openjiuwen_sysop::util::sha256_hex;

/// Characters that may never appear in a portable Context path segment.
///
/// Python parity: `PORTABLE_FORBIDDEN = frozenset('<>:"/\\|?*')`.
pub const PORTABLE_FORBIDDEN: [char; 9] = ['<', '>', ':', '"', '/', '\\', '|', '?', '*'];

/// Reserved Windows device names that may not name a segment.
///
/// Python parity: `WINDOWS_DEVICE_NAMES`, i.e. `con`, `prn`, `aux`, `nul`,
/// `com1`–`com9` and `lpt1`–`lpt9`.
pub const WINDOWS_DEVICE_NAMES: [&str; 22] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Maximum length of one portable segment, in Unicode scalar values.
pub const MAX_SEGMENT_CHARS: usize = 80;

/// Maximum length of one portable segment, in `UTF-8` bytes.
pub const MAX_SEGMENT_UTF8_BYTES: usize = 240;

/// Maximum length of a user-visible semantic name, in Unicode scalar values.
pub const SEMANTIC_NAME_MAX_CHARS: usize = 20;

/// Maximum service-id length, in `UTF-16` code units (500 characters).
const MAX_SERVICE_ID_UTF16_UNITS: usize = 500;

/// Matches an identifier that is a plain, portable storage segment as-is.
const LEGACY_SERVICE_SEGMENT_PATTERN: &str = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$";

/// Matches the hashed storage segment produced for every other service id.
const HASHED_SERVICE_SEGMENT_PATTERN: &str = r"^\.service-[0-9a-f]{64}$";

/// Returns whether one Context path segment is portable and canonical.
///
/// A segment is portable when it is non-empty, is neither `.` nor `..`, fits
/// [`MAX_SEGMENT_CHARS`] characters and [`MAX_SEGMENT_UTF8_BYTES`] `UTF-8`
/// bytes, does not end with a space or a dot, contains no
/// [`PORTABLE_FORBIDDEN`] character and no control character, and its first
/// dot-separated component is not a [`WINDOWS_DEVICE_NAMES`] entry.
#[must_use]
pub fn portable_context_segment_is_safe(segment: &str) -> bool {
    if segment.is_empty() || segment == "." || segment == ".." {
        return false;
    }
    if segment.chars().count() > MAX_SEGMENT_CHARS || segment.len() > MAX_SEGMENT_UTF8_BYTES {
        return false;
    }
    if segment.ends_with(' ') || segment.ends_with('.') {
        return false;
    }
    if segment
        .chars()
        .any(|character| PORTABLE_FORBIDDEN.contains(&character) || character.is_control())
    {
        return false;
    }
    !is_windows_device_name(first_dot_segment(segment))
}

/// Returns whether one new user-visible semantic name fits the product limit.
///
/// The portable check applies first. When `markdown_file` is set, the segment
/// must end with `.md` (ASCII case-insensitively), must not be
/// `description.md`, and only its stem is held to [`SEMANTIC_NAME_MAX_CHARS`];
/// otherwise the whole segment is.
#[must_use]
pub fn semantic_context_segment_is_safe(segment: &str, markdown_file: bool) -> bool {
    if !portable_context_segment_is_safe(segment) {
        return false;
    }
    if !markdown_file {
        return segment.chars().count() <= SEMANTIC_NAME_MAX_CHARS;
    }
    let bytes = segment.as_bytes();
    if bytes.len() < 3 || !bytes[bytes.len() - 3..].eq_ignore_ascii_case(b".md") {
        return false;
    }
    if segment.eq_ignore_ascii_case("description.md") {
        return false;
    }
    let Some(stem) = segment.get(..segment.len() - 3) else {
        return false;
    };
    !stem.is_empty() && stem.chars().count() <= SEMANTIC_NAME_MAX_CHARS
}

/// Validates a `PersonalContext` service id and returns the trimmed text.
///
/// Mirrors the frontend's `UTF-16` length limit: the trimmed value must be
/// non-empty and at most 500 `UTF-16` code units. The length is computed as
/// `value.encode_utf16().count() * 2`, which equals Python's
/// `len(text.encode("utf-16-le"))` for every value a Rust `&str` can hold,
/// because a Rust string cannot contain a lone surrogate (the only case where
/// Python's surrogate-passing encoding would differ).
///
/// # Errors
///
/// Returns [`Error::Validation`] with `service_id must be a non-empty string`
/// when the trimmed value is empty, and `service_id must be at most 500
/// characters` when it exceeds the 1000-byte `UTF-16`-LE limit.
pub fn validate_service_id(value: &str) -> Result<String> {
    let text = value.trim();
    if text.is_empty() {
        return Err(Error::validation("service_id must be a non-empty string"));
    }
    if text.encode_utf16().count() > MAX_SERVICE_ID_UTF16_UNITS {
        return Err(Error::validation(
            "service_id must be at most 500 characters",
        ));
    }
    Ok(text.to_owned())
}

/// Returns the on-disk storage segment for `service_id`.
///
/// A validated id that is already a portable legacy segment (fully matching
/// `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`, not ending with `.`, and whose first
/// dot-segment is not a Windows device name) is kept unchanged, so existing
/// trees stay readable. Every other id is encoded as
/// `.service-<sha256 hex>`, which cannot collide with the legacy namespace.
///
/// # Errors
///
/// Propagates the validation errors of [`validate_service_id`].
pub fn service_storage_segment(service_id: &str) -> Result<String> {
    let text = validate_service_id(service_id)?;
    if is_legacy_service_segment(&text)
        && !text.ends_with('.')
        && !is_windows_device_name(first_dot_segment(&text))
    {
        return Ok(text);
    }
    Ok(format!(".service-{}", sha256_hex(text.as_bytes())))
}

/// Returns whether `segment` names a storage directory written by this module.
///
/// Accepts both the legacy portable identifier and the hashed
/// `.service-<sha256 hex>` form, so cleanup can recognise disk components
/// without encoding them again.
#[must_use]
pub fn service_storage_segment_is_safe(segment: &str) -> bool {
    is_legacy_service_segment(segment) || is_hashed_service_segment(segment)
}

/// Resolves one canonical POSIX path below a plain managed Context root.
///
/// The value must be a relative POSIX path (no empty, `.`, or `..` segment and
/// no backslash), every segment must pass [`portable_context_segment_is_safe`],
/// and no existing component between the root and the candidate may be a
/// symlink. The root is canonicalised first, so a missing root is a security
/// failure rather than a path under an unverified directory.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a malformed value (`Context path must be a
/// POSIX relative path`, `Context path is unsafe`, `Context path contains a
/// non-portable segment`, or, with `must_exist`, `Context path is
/// unavailable`), and [`Error::Security`] when the root is unavailable, a link
/// is traversed, or the resolved path leaves the root.
pub fn resolve_context_relative_path(
    root: &Path,
    value: &str,
    must_exist: bool,
) -> Result<PathBuf> {
    let parts = split_posix_relative(value)?;
    let root_canonical = std::fs::canonicalize(root)
        .map_err(|_| Error::security("managed Context root is unavailable"))?;
    let mut candidate = root_canonical.clone();
    let mut deepest_existing = root_canonical.clone();
    for part in &parts {
        candidate.push(part);
        if let Ok(metadata) = std::fs::symlink_metadata(&candidate) {
            if metadata.file_type().is_symlink() {
                return Err(Error::security(
                    "managed Context path traverses a link or reparse point",
                ));
            }
            deepest_existing.clone_from(&candidate);
        }
    }
    ensure_within_root(&root_canonical, &deepest_existing)?;
    if must_exist && !candidate.exists() {
        return Err(Error::validation("Context path is unavailable"));
    }
    Ok(candidate)
}

/// Splits and validates a POSIX relative path into its segments.
///
/// The error texts match the Python resolver: a non-string, empty, or
/// backslash-bearing value is not a POSIX relative path; an absolute path, a
/// `.`/`..` value, or any empty/dot/dot-dot segment is unsafe; and a segment
/// that is not portable is reported as such.
fn split_posix_relative(value: &str) -> Result<Vec<&str>> {
    if value.is_empty() || value.contains('\\') {
        return Err(Error::validation(
            "Context path must be a POSIX relative path",
        ));
    }
    if value == "." || value == ".." || value.starts_with('/') {
        return Err(Error::validation("Context path is unsafe"));
    }
    let parts: Vec<&str> = value.split('/').collect();
    if parts
        .iter()
        .any(|part| part.is_empty() || *part == "." || *part == "..")
    {
        return Err(Error::validation("Context path is unsafe"));
    }
    if parts
        .iter()
        .any(|part| !portable_context_segment_is_safe(part))
    {
        return Err(Error::validation(
            "Context path contains a non-portable segment",
        ));
    }
    Ok(parts)
}

/// Verifies that `path` canonicalises to a location inside `root_canonical`.
///
/// This is the defensive half of the resolver: the per-segment link scan
/// already rejects every way a candidate could leave the root today, so the
/// escape branch exists to keep the invariant explicit (and to guard the root
/// against a swap between the scan and this check).
fn ensure_within_root(root_canonical: &Path, path: &Path) -> Result<()> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|_| Error::security("managed Context root is unavailable"))?;
    if canonical.starts_with(root_canonical) {
        Ok(())
    } else {
        Err(Error::security("managed Context path escaped its root"))
    }
}

/// Returns the part of `segment` before its first dot.
fn first_dot_segment(segment: &str) -> &str {
    match segment.split_once('.') {
        Some((head, _)) => head,
        None => segment,
    }
}

/// Returns whether `name` is a reserved Windows device name (ASCII case-insensitive).
fn is_windows_device_name(name: &str) -> bool {
    WINDOWS_DEVICE_NAMES
        .iter()
        .any(|device| name.eq_ignore_ascii_case(device))
}

/// Returns the compiled legacy-service-segment pattern.
///
/// The pattern is a compile-time constant; the accessor exists so it is
/// compiled once per process. A `None` would mean the constant stopped being a
/// valid regex, which the unit tests below rule out.
fn legacy_service_segment_regex() -> Option<&'static Regex> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    PATTERN
        .get_or_init(|| Regex::new(LEGACY_SERVICE_SEGMENT_PATTERN).ok())
        .as_ref()
}

/// Returns the compiled hashed-service-segment pattern.
fn hashed_service_segment_regex() -> Option<&'static Regex> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    PATTERN
        .get_or_init(|| Regex::new(HASHED_SERVICE_SEGMENT_PATTERN).ok())
        .as_ref()
}

/// Returns whether `text` fully matches the legacy service-segment grammar.
fn is_legacy_service_segment(text: &str) -> bool {
    legacy_service_segment_regex().is_some_and(|pattern| pattern.is_match(text))
}

/// Returns whether `text` fully matches the hashed service-segment grammar.
fn is_hashed_service_segment(text: &str) -> bool {
    hashed_service_segment_regex().is_some_and(|pattern| pattern.is_match(text))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    #[test]
    fn portable_segments_accept_plain_names() {
        for segment in [
            "notes",
            "2026-01-01",
            ".hidden",
            "a.md",
            "console",
            "con-artist",
        ] {
            assert!(
                portable_context_segment_is_safe(segment),
                "expected {segment:?} to be portable"
            );
        }
    }

    #[test]
    fn portable_segments_reject_dot_entries_and_empty() {
        for segment in ["", ".", ".."] {
            assert!(!portable_context_segment_is_safe(segment));
        }
    }

    #[test]
    fn portable_segments_reject_oversized_names() {
        assert!(!portable_context_segment_is_safe(&"a".repeat(81)));
        assert!(portable_context_segment_is_safe(&"a".repeat(80)));
        // 61 four-byte scalar values: 61 characters but 244 UTF-8 bytes.
        let wide = "\u{1d11e}".repeat(61);
        assert_eq!(wide.chars().count(), 61);
        assert!(wide.len() > MAX_SEGMENT_UTF8_BYTES);
        assert!(!portable_context_segment_is_safe(&wide));
        // 80 three-byte scalar values sit exactly on the byte limit.
        let at_limit = "\u{4f60}".repeat(80);
        assert_eq!(at_limit.len(), MAX_SEGMENT_UTF8_BYTES);
        assert!(portable_context_segment_is_safe(&at_limit));
    }

    #[test]
    fn portable_segments_reject_trailing_space_or_dot() {
        assert!(!portable_context_segment_is_safe("notes "));
        assert!(!portable_context_segment_is_safe("notes."));
    }

    #[test]
    fn portable_segments_reject_forbidden_and_control_characters() {
        for segment in ["a:b", "a/b", "a\\b", "a<b", "a?b", "a*b", "a|b", "a\"b"] {
            assert!(
                !portable_context_segment_is_safe(segment),
                "expected {segment:?} to be rejected"
            );
        }
        assert!(!portable_context_segment_is_safe("a\u{7}b"));
        assert!(!portable_context_segment_is_safe("\u{0}"));
    }

    #[test]
    fn portable_segments_reject_windows_device_names() {
        for segment in ["CON", "con", "nul.txt", "Com1", "lpt9.md", "aux.tar.gz"] {
            assert!(
                !portable_context_segment_is_safe(segment),
                "expected {segment:?} to be rejected"
            );
        }
    }

    #[test]
    fn semantic_names_enforce_the_twenty_character_limit() {
        assert!(semantic_context_segment_is_safe("notes", false));
        assert!(semantic_context_segment_is_safe(
            "12345678901234567890",
            false
        ));
        assert!(!semantic_context_segment_is_safe(
            "123456789012345678901",
            false
        ));
        assert!(!semantic_context_segment_is_safe("", false));
    }

    #[test]
    fn semantic_markdown_names_measure_the_stem() {
        assert!(semantic_context_segment_is_safe("notes.md", true));
        assert!(semantic_context_segment_is_safe("NOTES.MD", true));
        assert!(semantic_context_segment_is_safe(
            "12345678901234567890.md",
            true
        ));
        assert!(!semantic_context_segment_is_safe(
            "123456789012345678901.md",
            true
        ));
        assert!(!semantic_context_segment_is_safe("notes.txt", true));
        assert!(!semantic_context_segment_is_safe("description.md", true));
        assert!(!semantic_context_segment_is_safe("DESCRIPTION.MD", true));
        assert!(!semantic_context_segment_is_safe(".md", true));
        // The portable check still applies first.
        assert!(!semantic_context_segment_is_safe("a:b.md", true));
    }

    #[test]
    fn validate_service_id_trims_and_enforces_the_utf16_limit() {
        assert_eq!(validate_service_id("  alpha  ").unwrap(), "alpha");

        let empty = validate_service_id("   ").unwrap_err();
        assert_eq!(empty.kind(), ErrorKind::Validation);
        assert_eq!(
            empty.to_string(),
            "validation: service_id must be a non-empty string"
        );

        // 500 UTF-16 code units is the limit (1000 UTF-16-LE bytes).
        assert!(validate_service_id(&"a".repeat(500)).is_ok());
        let too_long = validate_service_id(&"a".repeat(501)).unwrap_err();
        assert_eq!(too_long.kind(), ErrorKind::Validation);
        assert_eq!(
            too_long.to_string(),
            "validation: service_id must be at most 500 characters"
        );

        // Astral scalar values count as two UTF-16 code units each.
        assert!(validate_service_id(&"\u{1f600}".repeat(250)).is_ok());
        assert!(validate_service_id(&"\u{1f600}".repeat(251)).is_err());
    }

    #[test]
    fn service_storage_segment_keeps_portable_legacy_ids() {
        for service in ["alpha", "alpha.beta-1_x", &"a".repeat(128)] {
            assert_eq!(service_storage_segment(service).unwrap(), service);
        }
    }

    #[test]
    fn service_storage_segment_hashes_every_other_id() {
        assert_eq!(
            service_storage_segment("hashed name").unwrap(),
            ".service-e8b31fc8a87558b573b58f6bc1736411d3e76f8fc70e650bbb2b795647a7864d"
        );
        assert_eq!(
            service_storage_segment("CON").unwrap(),
            ".service-a3dbc4b644a9a2c51e74509ded5c46ccf31d84c65ca417fe0126af1555c66ceb"
        );
        assert_eq!(
            service_storage_segment("alpha.").unwrap(),
            ".service-1d8efbb0763a9f80cc7a1de816210b19d4fd22da0eeb4696adce0ba42bdab600"
        );
        // A trailing space is trimmed before the legacy check.
        assert_eq!(service_storage_segment(" alpha ").unwrap(), "alpha");
        assert!(service_storage_segment("a b")
            .unwrap()
            .starts_with(".service-"));
        assert!(service_storage_segment(&"a".repeat(129))
            .unwrap()
            .starts_with(".service-"));
    }

    #[test]
    fn service_storage_segment_rejects_invalid_ids() {
        let error = service_storage_segment("").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("service_id must be a non-empty string"));
    }

    #[test]
    fn service_storage_segment_is_safe_recognises_both_forms() {
        assert!(service_storage_segment_is_safe("alpha"));
        assert!(service_storage_segment_is_safe("alpha.beta-1"));
        assert!(service_storage_segment_is_safe("alpha."));
        assert!(service_storage_segment_is_safe(&format!(
            ".service-{}",
            "0".repeat(64)
        )));
        assert!(!service_storage_segment_is_safe(&format!(
            ".service-{}",
            "0".repeat(63)
        )));
        assert!(!service_storage_segment_is_safe(".service-XYZ"));
        assert!(!service_storage_segment_is_safe("-alpha"));
        assert!(!service_storage_segment_is_safe(""));
        assert!(!service_storage_segment_is_safe("a/b"));
    }

    #[test]
    fn resolve_rejects_malformed_values() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path();

        for value in ["", "a\\b"] {
            let error = resolve_context_relative_path(path, value, false).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: Context path must be a POSIX relative path"
            );
        }
        for value in [
            "/etc/passwd",
            ".",
            "..",
            "a//b",
            "a/",
            "./a",
            "a/./b",
            "a/../b",
        ] {
            let error = resolve_context_relative_path(path, value, false).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(error.to_string(), "validation: Context path is unsafe");
        }
        for value in ["a:b", "CON", "notes."] {
            let error = resolve_context_relative_path(path, value, false).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: Context path contains a non-portable segment"
            );
        }
    }

    #[test]
    fn resolve_accepts_relative_paths_below_the_root() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("memories").join("2026");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("note.md"), "hello").unwrap();

        let resolved =
            resolve_context_relative_path(root.path(), "memories/2026/note.md", true).unwrap();
        let expected = root
            .path()
            .canonicalize()
            .unwrap()
            .join("memories")
            .join("2026")
            .join("note.md");
        assert_eq!(resolved, expected);

        let pending = resolve_context_relative_path(root.path(), "memories/new.md", false).unwrap();
        assert_eq!(
            pending,
            root.path()
                .canonicalize()
                .unwrap()
                .join("memories")
                .join("new.md")
        );
    }

    #[test]
    fn resolve_requires_existence_when_asked() {
        let root = tempfile::tempdir().unwrap();
        let error = resolve_context_relative_path(root.path(), "missing.md", true).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.to_string(), "validation: Context path is unavailable");
    }

    #[test]
    fn resolve_requires_an_existing_root() {
        let root = tempfile::tempdir().unwrap();
        let absent = root.path().join("absent");
        let error = resolve_context_relative_path(&absent, "note.md", false).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert_eq!(
            error.to_string(),
            "security: managed Context root is unavailable"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_rejects_symlink_components() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("real")).unwrap();
        symlink(outside.path(), root.path().join("link")).unwrap();

        let error = resolve_context_relative_path(root.path(), "link/note.md", false).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert_eq!(
            error.to_string(),
            "security: managed Context path traverses a link or reparse point"
        );

        // A link at the leaf is rejected too.
        symlink(outside.path().join("note.md"), root.path().join("note.md")).unwrap();
        let error = resolve_context_relative_path(root.path(), "note.md", false).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert!(error
            .to_string()
            .contains("traverses a link or reparse point"));
    }

    #[test]
    fn ensure_within_root_rejects_an_outside_path() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let root_canonical = root.path().canonicalize().unwrap();

        ensure_within_root(&root_canonical, &root_canonical).unwrap();
        let error = ensure_within_root(&root_canonical, &outside.path().canonicalize().unwrap())
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert_eq!(
            error.to_string(),
            "security: managed Context path escaped its root"
        );
    }

    #[test]
    fn regex_patterns_compile_and_anchor() {
        assert!(legacy_service_segment_regex().is_some());
        assert!(hashed_service_segment_regex().is_some());
        assert!(is_legacy_service_segment("a"));
        assert!(!is_legacy_service_segment("a\nb"));
        assert!(!is_legacy_service_segment(".a"));
        assert!(is_hashed_service_segment(&format!(
            ".service-{}",
            "a".repeat(64)
        )));
    }
}
