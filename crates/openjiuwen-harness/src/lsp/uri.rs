//! `file://` URI conversion for LSP messages.
//!
//! Ports `openjiuwen/harness/lsp/core/utils/file_uri.py`: paths become absolute `file://`
//! URIs with percent-encoded special characters, and URIs become native paths again.
//! Non-`file` URIs (for example `untitled:`) pass through unchanged, matching Python.
use std::path::{Path, PathBuf};

/// Characters left unescaped in a URI path, mirroring Python's `quote(..., safe="/:")`.
const SAFE: &[u8] = b"/:";

/// Converts a filesystem `path` into an absolute `file://` URI.
///
/// Relative paths are made absolute against the current directory. Spaces and other
/// characters outside the RFC 3986 unreserved set (plus `/` and `:`) are percent-encoded.
/// Non-UTF-8 path bytes are replaced lossily, which is the closest cross-platform analogue
/// of Python's `Path.as_posix()`.
#[must_use]
pub fn path_to_uri(path: &Path) -> String {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let text = absolute.to_string_lossy();
    let mut encoded = String::with_capacity(text.len() + 8);
    for byte in text.as_bytes() {
        let escaped = byte.is_ascii_alphanumeric()
            || matches!(*byte, b'-' | b'.' | b'_' | b'~')
            || SAFE.contains(byte);
        if escaped {
            encoded.push(char::from(*byte));
        } else {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(*byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(*byte & 0x0F)]));
        }
    }
    format!("file://{encoded}")
}

/// Converts a `file://` URI back into a native filesystem path.
///
/// Percent escapes (including `%3A` for a Windows drive colon) are decoded. A URI that does
/// not start with `file://` is returned as a path unchanged, matching Python's fallback.
#[must_use]
pub fn uri_to_path(uri: &str) -> PathBuf {
    let Some(rest) = uri.strip_prefix("file://") else {
        return PathBuf::from(uri);
    };
    let mut decoded = percent_decode(rest);
    // Windows URIs look like `/C:/foo`; strip the leading slash before the drive letter.
    if cfg!(windows)
        && decoded.len() >= 3
        && decoded.starts_with('/')
        && decoded.as_bytes().get(2) == Some(&b':')
    {
        decoded.remove(0);
        decoded = decoded.replace('/', "\\");
    }
    PathBuf::from(decoded)
}

/// Decodes `%XX` escapes, keeping malformed sequences verbatim.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let high = hex_value(bytes[index + 1]);
            let low = hex_value(bytes[index + 2]);
            if let (Some(high), Some(low)) = (high, low) {
                out.push((high << 4) | low);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The numeric value of one hexadecimal digit.
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
