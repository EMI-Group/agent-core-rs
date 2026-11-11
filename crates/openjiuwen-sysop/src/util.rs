//! Small shared helpers: timestamps, permissions, hashing, and byte truncation.

use std::{
    fs::Metadata,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use chrono::{Local, TimeZone};
use sha2::{Digest, Sha256};

/// Formats a timestamp the way Python's `str(datetime.fromtimestamp(ts))` does.
///
/// Python renders local time with microsecond precision and no timezone suffix, for
/// example `"2024-01-02 03:04:05.123456"`. That string is what
/// `FileSystemItem.modified_time` carries, and it is also the sort key of
/// `sort_by="modified_time"`.
#[must_use]
pub fn format_modified_time(time: SystemTime) -> String {
    let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = i64::try_from(duration.as_secs()).unwrap_or(i64::MAX);
    let micros = i64::from(duration.subsec_micros());
    Local
        .timestamp_opt(secs, u32::try_from(micros * 1000).unwrap_or(0))
        .single()
        .map(|stamp| stamp.format("%Y-%m-%d %H:%M:%S%.6f").to_string())
        .unwrap_or_default()
}

/// Formats a path as a lossy UTF-8 string (diagnostics and tool output).
#[must_use]
pub fn format_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Returns the Unix permission bits of `metadata` as an octal string, e.g. `"644"`.
///
/// Returns `None` on platforms without Unix permission bits.
#[must_use]
pub fn octal_permissions(metadata: &Metadata) -> Option<String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        Some(format!("{:03o}", metadata.permissions().mode() & 0o777))
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        None
    }
}

/// Returns the lowercase hex SHA-256 digest of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Cuts `bytes` to at most `limit` bytes on a UTF-8 character boundary.
///
/// Returns the retained slice and whether anything was dropped.
#[must_use]
pub fn truncate_utf8_bytes(bytes: &[u8], limit: u64) -> (&[u8], bool) {
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    if bytes.len() <= limit {
        return (bytes, false);
    }
    let mut end = limit;
    while end > 0 && !is_char_boundary(bytes, end) {
        end -= 1;
    }
    (&bytes[..end], true)
}

/// Returns `true` when `index` is a UTF-8 character boundary (or past the end).
fn is_char_boundary(bytes: &[u8], index: usize) -> bool {
    index >= bytes.len() || (bytes[index] & 0xC0) != 0x80
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn modified_time_matches_python_datetime_shape() {
        // 2024-01-02 03:04:05 UTC, rendered in local time by both implementations.
        let time = UNIX_EPOCH + Duration::from_micros(1_704_164_645_123_456);
        let formatted = format_modified_time(time);
        assert_eq!(formatted.len(), 26, "unexpected format: {formatted}");
        assert!(formatted.contains(':'));
        assert!(formatted.contains('.'));
        let parts: Vec<&str> = formatted.split(' ').collect();
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn epoch_formats_without_panic() {
        assert_ne!(format_modified_time(UNIX_EPOCH), "");
    }

    #[test]
    fn sha256_matches_reference_digest() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        let text = "héllo";
        let (kept, truncated) = truncate_utf8_bytes(text.as_bytes(), 2);
        assert!(truncated);
        assert_eq!(kept.len(), 1, "must not split a two-byte character");
        assert_eq!(std::str::from_utf8(kept).unwrap(), "h");

        let (kept, truncated) = truncate_utf8_bytes(text.as_bytes(), 100);
        assert!(!truncated);
        assert_eq!(kept, text.as_bytes());
    }

    #[test]
    fn octal_permissions_on_unix() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let file = dir.path().join("f");
            std::fs::write(&file, b"x").unwrap();
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o640)).unwrap();
            let metadata = std::fs::metadata(&file).unwrap();
            assert_eq!(octal_permissions(&metadata).as_deref(), Some("640"));
        }
    }
}
