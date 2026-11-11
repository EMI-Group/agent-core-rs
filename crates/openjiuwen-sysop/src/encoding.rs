//! Text encoding detection, BOM handling, and binary sniffing.
//!
//! Python parity: `local/_text_encoding.py` (`detect_text_encoding`) plus the
//! `errors="replace"` decoding every read uses. Rust additions: an explicit
//! [`BinaryPolicy`] so a caller can refuse binary content instead of decoding it.
//!
//! Supported encodings: UTF-8 (with or without BOM), UTF-16 LE/BE (BOM or heuristic),
//! US-ASCII, and Latin-1. Legacy CJK encodings (GBK, Big5, Shift-JIS) are reported as
//! [`openjiuwen_common::error::Error::Unsupported`] instead of being decoded incorrectly —
//! see `CONTEXT.md` for the rationale.
//!
//! ```
//! use openjiuwen_sysop::encoding::{TextEncoding, detect_text_encoding};
//!
//! assert_eq!(detect_text_encoding(b"hello"), TextEncoding::Utf8);
//! assert_eq!(detect_text_encoding(&[0xFF, 0xFE, b'h', 0]), TextEncoding::Utf16Le);
//! ```

use openjiuwen_common::error::{Error, Result};

use crate::config::BinaryPolicy;
use crate::constants::AUTO_ENCODING;

/// A text encoding this crate can decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextEncoding {
    /// UTF-8, optionally with a byte-order mark.
    Utf8,
    /// UTF-16 little endian.
    Utf16Le,
    /// UTF-16 big endian.
    Utf16Be,
    /// ISO-8859-1 / Latin-1 (every byte maps to one character).
    Latin1,
}

impl TextEncoding {
    /// The canonical name passed back to callers.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Utf8 => "utf-8",
            Self::Utf16Le => "utf-16le",
            Self::Utf16Be => "utf-16be",
            Self::Latin1 => "latin-1",
        }
    }

    /// Parses a case-insensitive encoding name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        let normalized = name.trim().to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "utf-8" | "utf8" | "us-ascii" | "ascii" => Some(Self::Utf8),
            "utf-16" | "utf-16le" | "utf16" | "utf16le" => Some(Self::Utf16Le),
            "utf-16be" | "utf16be" => Some(Self::Utf16Be),
            "latin-1" | "latin1" | "iso-8859-1" | "l1" => Some(Self::Latin1),
            _ => None,
        }
    }
}

impl std::fmt::Display for TextEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The UTF-8 byte-order mark.
pub const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// The UTF-16 little-endian byte-order mark.
pub const UTF16LE_BOM: &[u8] = &[0xFF, 0xFE];

/// The UTF-16 big-endian byte-order mark.
pub const UTF16BE_BOM: &[u8] = &[0xFE, 0xFF];

/// Bytes inspected when deciding whether content is binary.
const SNIFF_LIMIT: usize = 8192;

/// Numerator of the control-byte ratio above which content is treated as binary
/// (`3/10`, i.e. 30%).
const BINARY_CONTROL_NUMERATOR: usize = 3;

/// Denominator of the control-byte ratio (30%).
const BINARY_CONTROL_DENOMINATOR: usize = 10;

/// Returns the encoding declared by a byte-order mark, if any.
#[must_use]
pub fn bom_encoding(bytes: &[u8]) -> Option<TextEncoding> {
    if bytes.starts_with(UTF8_BOM) {
        return Some(TextEncoding::Utf8);
    }
    if bytes.starts_with(UTF16LE_BOM) {
        return Some(TextEncoding::Utf16Le);
    }
    if bytes.starts_with(UTF16BE_BOM) {
        return Some(TextEncoding::Utf16Be);
    }
    None
}

/// Strips a byte-order mark, returning the remaining bytes and the declared encoding.
#[must_use]
pub fn strip_bom(bytes: &[u8]) -> (&[u8], Option<TextEncoding>) {
    match bom_encoding(bytes) {
        Some(TextEncoding::Utf8) => (&bytes[UTF8_BOM.len()..], Some(TextEncoding::Utf8)),
        Some(TextEncoding::Utf16Le) => (&bytes[UTF16LE_BOM.len()..], Some(TextEncoding::Utf16Le)),
        Some(TextEncoding::Utf16Be) => (&bytes[UTF16BE_BOM.len()..], Some(TextEncoding::Utf16Be)),
        _ => (bytes, None),
    }
}

/// Detects the encoding of `bytes` (Python `detect_text_encoding`).
///
/// The decision order is: BOM, valid UTF-8, UTF-16 heuristic (interleaved NUL bytes),
/// Latin-1.
#[must_use]
pub fn detect_text_encoding(bytes: &[u8]) -> TextEncoding {
    if let Some(encoding) = bom_encoding(bytes) {
        return encoding;
    }
    if bytes.is_empty() {
        return TextEncoding::Utf8;
    }
    // UTF-16 ASCII text is also valid UTF-8 (NUL is a legal UTF-8 byte), so the UTF-16
    // heuristic has to run first whenever NUL bytes are present.
    if bytes.contains(&0) {
        if let Some(encoding) = detect_utf16_without_bom(bytes) {
            return encoding;
        }
    }
    if std::str::from_utf8(bytes).is_ok() {
        return TextEncoding::Utf8;
    }
    if let Some(encoding) = detect_utf16_without_bom(bytes) {
        return encoding;
    }
    TextEncoding::Latin1
}

/// Detects BOM-less UTF-16 by looking at NUL-byte parity in the first few units.
fn detect_utf16_without_bom(bytes: &[u8]) -> Option<TextEncoding> {
    let sample = &bytes[..bytes.len().min(512)];
    if sample.len() < 4 {
        return None;
    }
    let pairs = sample.len() / 2;
    let mut even_nuls = 0_usize;
    let mut odd_nuls = 0_usize;
    for pair in sample.chunks_exact(2) {
        if pair[0] == 0 {
            even_nuls += 1;
        }
        if pair[1] == 0 {
            odd_nuls += 1;
        }
    }
    // ASCII-heavy UTF-16LE has NUL in every high byte (odd index); BE has them even.
    if odd_nuls * 10 >= pairs * 6 {
        return Some(TextEncoding::Utf16Le);
    }
    if even_nuls * 10 >= pairs * 6 {
        return Some(TextEncoding::Utf16Be);
    }
    None
}

/// Returns `true` when `bytes` look like binary data.
///
/// Heuristic: a NUL byte, an invalid UTF-8 byte sequence that is not valid UTF-16, or
/// more than 30% control bytes in the sniffed prefix. Only the first 8 KiB are inspected,
/// so detection is cheap and deterministic.
#[must_use]
pub fn is_binary(bytes: &[u8]) -> bool {
    let sample = &bytes[..bytes.len().min(SNIFF_LIMIT)];
    if sample.is_empty() {
        return false;
    }
    if detect_utf16_without_bom(sample).is_some() || bom_encoding(sample).is_some() {
        return false;
    }
    if sample.contains(&0) {
        return true;
    }
    if std::str::from_utf8(sample).is_ok() {
        // Valid UTF-8: only unusual control characters signal binary content.
        let control = sample
            .iter()
            .filter(|byte| is_control(**byte) && !is_whitespace(**byte))
            .count();
        return ratio_exceeds(control, sample.len());
    }
    let control = sample.iter().filter(|byte| is_control(**byte)).count();
    ratio_exceeds(control, sample.len())
}

/// Decodes `bytes` into text.
///
/// # Errors
///
/// Returns `ErrorKind::Validation` when [`BinaryPolicy::Reject`] is set and the content
/// looks binary, and `ErrorKind::Unsupported` for an encoding this crate cannot decode.
pub fn decode_text(bytes: &[u8], encoding: &str, policy: BinaryPolicy) -> Result<String> {
    let (payload, bom) = strip_bom(bytes);
    let encoding = if encoding.trim().eq_ignore_ascii_case(AUTO_ENCODING) {
        bom.unwrap_or_else(|| detect_text_encoding(payload))
    } else {
        TextEncoding::parse(encoding).ok_or_else(|| {
            Error::unsupported(format!(
                "encoding '{encoding}' is not supported; supported encodings are utf-8, utf-16le, utf-16be, latin-1, and 'auto'"
            ))
        })?
    };
    if matches!(policy, BinaryPolicy::Reject) && is_binary(payload) {
        return Err(Error::validation(
            "content looks binary; use mode=\"bytes\" to read it",
        ));
    }
    Ok(match encoding {
        TextEncoding::Utf8 => String::from_utf8_lossy(payload).into_owned(),
        TextEncoding::Latin1 => payload.iter().map(|byte| char::from(*byte)).collect(),
        TextEncoding::Utf16Le => decode_utf16(payload, true),
        TextEncoding::Utf16Be => decode_utf16(payload, false),
    })
}

/// The encoding detected for `bytes` under the `auto` policy, for reporting purposes.
#[must_use]
pub fn detected_encoding_name(bytes: &[u8]) -> String {
    strip_bom(bytes)
        .1
        .unwrap_or_else(|| detect_text_encoding(bytes))
        .as_str()
        .to_string()
}

/// Encodes text as UTF-8, the only encoding this crate writes.
///
/// Writing in a legacy encoding is not supported (Python lets the caller pick one); the
/// resulting bytes are always UTF-8, which keeps the written file round-trippable.
///
/// # Errors
///
/// Returns `ErrorKind::Unsupported` when `encoding` is neither UTF-8 nor ASCII.
pub fn encode_text(text: &str, encoding: &str) -> Result<Vec<u8>> {
    match TextEncoding::parse(encoding) {
        Some(TextEncoding::Utf8) => Ok(text.as_bytes().to_vec()),
        Some(other) => Ok(match other {
            TextEncoding::Utf16Le => encode_utf16(text, true),
            TextEncoding::Utf16Be => encode_utf16(text, false),
            _ => text.as_bytes().to_vec(),
        }),
        None => Err(Error::unsupported(format!(
            "encoding '{encoding}' is not supported for writing; use utf-8"
        ))),
    }
}

/// Decodes a UTF-16 stream, honoring surrogate pairs and replacing invalid units.
fn decode_utf16(bytes: &[u8], little_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| {
            if little_endian {
                u16::from_le_bytes([pair[0], pair[1]])
            } else {
                u16::from_be_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    String::from_utf16_lossy(&units)
}

/// Encodes text as UTF-16.
fn encode_utf16(text: &str, little_endian: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() * 2);
    for unit in text.encode_utf16() {
        let bytes = if little_endian {
            unit.to_le_bytes()
        } else {
            unit.to_be_bytes()
        };
        out.extend_from_slice(&bytes);
    }
    out
}

/// Returns `true` for ASCII control bytes.
fn is_control(byte: u8) -> bool {
    byte < 0x20 || byte == 0x7F
}

/// Returns `true` for whitespace control bytes, which never indicate binary content.
fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

/// Returns `true` when `part / total` exceeds the binary threshold.
///
/// Compares `part * 10 > total * 3` to avoid floating-point conversion.
fn ratio_exceeds(part: usize, total: usize) -> bool {
    if total == 0 {
        return false;
    }
    part.saturating_mul(BINARY_CONTROL_DENOMINATOR) > total.saturating_mul(BINARY_CONTROL_NUMERATOR)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn detects_utf8_and_bom_variants() {
        assert_eq!(detect_text_encoding(b"plain"), TextEncoding::Utf8);
        assert_eq!(detect_text_encoding(b""), TextEncoding::Utf8);
        assert_eq!(
            detect_text_encoding("\u{FEFF}hello".as_bytes()),
            TextEncoding::Utf8
        );
        // The BOM is stripped before decoding.
        let text = decode_text("\u{FEFF}hello".as_bytes(), "auto", BinaryPolicy::Lossy).unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn detects_utf16_le_and_be() {
        let le = encode_utf16("hello", true);
        assert_eq!(detect_text_encoding(&le), TextEncoding::Utf16Le);
        assert_eq!(
            decode_text(&le, "auto", BinaryPolicy::Lossy).unwrap(),
            "hello"
        );

        let be = encode_utf16("hello", false);
        assert_eq!(detect_text_encoding(&be), TextEncoding::Utf16Be);
        assert_eq!(
            decode_text(&be, "utf-16be", BinaryPolicy::Lossy).unwrap(),
            "hello"
        );
    }

    #[test]
    fn latin1_round_trip() {
        let bytes = [0xE9_u8, 0xE8];
        assert_eq!(detect_text_encoding(&bytes), TextEncoding::Latin1);
        assert_eq!(
            decode_text(&bytes, "latin-1", BinaryPolicy::Lossy).unwrap(),
            "éè"
        );
    }

    #[test]
    fn invalid_utf8_is_replaced_not_rejected() {
        let text = decode_text(b"ok\xFFbad", "utf-8", BinaryPolicy::Lossy).unwrap();
        assert!(text.starts_with("ok"));
        assert!(text.contains('\u{FFFD}'));
    }

    #[test]
    fn unknown_encoding_is_unsupported() {
        let error = decode_text(b"x", "gbk", BinaryPolicy::Lossy).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.message().contains("gbk"));
    }

    #[test]
    fn binary_sniffing() {
        assert!(!is_binary(b"hello\nworld\n"));
        assert!(!is_binary("héllo".as_bytes()));
        assert!(is_binary(&[0x00, 0x01, 0x02, 0x03]));
        assert!(is_binary(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]));
        assert!(!is_binary(&[]));
    }

    #[test]
    fn reject_policy_blocks_binary_content() {
        let error = decode_text(&[0x00, 0x01], "utf-8", BinaryPolicy::Reject).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(decode_text(&[0x00, 0x01], "utf-8", BinaryPolicy::Lossy).is_ok());
    }

    #[test]
    fn bom_helpers_report_and_strip() {
        assert_eq!(
            bom_encoding(&[0xFF, 0xFE, 1, 0]),
            Some(TextEncoding::Utf16Le)
        );
        let (payload, bom) = strip_bom(UTF8_BOM);
        assert_eq!(payload.len(), 0, "the BOM must be stripped");
        assert_eq!(bom, Some(TextEncoding::Utf8));
        assert_eq!(strip_bom(b"x"), (b"x".as_slice(), None));
    }

    #[test]
    fn encoding_names_parse_leniently() {
        assert_eq!(TextEncoding::parse("UTF_8"), Some(TextEncoding::Utf8));
        assert_eq!(TextEncoding::parse("ascii"), Some(TextEncoding::Utf8));
        assert_eq!(
            TextEncoding::parse("iso-8859-1"),
            Some(TextEncoding::Latin1)
        );
        assert_eq!(TextEncoding::parse("gbk"), None);
        assert_eq!(TextEncoding::Latin1.to_string(), "latin-1");
    }

    #[test]
    fn detected_encoding_name_reports_bom() {
        assert_eq!(detected_encoding_name(&[0xFF, 0xFE, b'h', 0]), "utf-16le");
        assert_eq!(detected_encoding_name(b"plain"), "utf-8");
    }
}
