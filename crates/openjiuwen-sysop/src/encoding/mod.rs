//! Text encoding detection, BOM handling, binary sniffing, and decoding.
//!
//! Python parity: `local/_text_encoding.py` (`detect_text_encoding`) plus the
//! `errors="replace"` decoding every read uses. Rust additions: an explicit
//! [`BinaryPolicy`] so a caller can refuse binary content instead of decoding it.
//!
//! Supported encodings: UTF-8 (with or without BOM), UTF-16 LE/BE (BOM or heuristic),
//! US-ASCII, Latin-1, and the legacy CJK encodings GB18030/GBK, Big5 and Shift-JIS.
//! The CJK decoders are table driven ([`cjk`]); their tables are generated offline from
//! Python's codecs by `scripts/gen_cjk_tables.py`, so decoding matches the Python
//! reference byte for byte, including `U+FFFD` replacement of invalid sequences.
//! Writes stay UTF-8: [`encode_text`] still rejects the legacy names.
//!
//! `auto` detection follows Python: BOM, valid UTF-8, a UTF-16 NUL-parity heuristic, then a
//! quality-scored choice between strict GB18030 and strict Big5, and Latin-1 as the last
//! resort. Shift-JIS is **never** auto-detected (decoding it explicitly is supported):
//! Shift-JIS bytes are usually also valid GB18030 or Big5, and Python's detector does not
//! consider the encoding either.
//!
//! ```
//! use openjiuwen_sysop::encoding::{TextEncoding, detect_text_encoding};
//!
//! assert_eq!(detect_text_encoding(b"hello"), TextEncoding::Utf8);
//! assert_eq!(detect_text_encoding(&[0xFF, 0xFE, b'h', 0]), TextEncoding::Utf16Le);
//! assert_eq!(detect_text_encoding(&[0xD6, 0xD0, 0xCE, 0xC4]), TextEncoding::Gb18030);
//! ```

use openjiuwen_common::error::{Error, Result};

use crate::config::BinaryPolicy;
use crate::constants::AUTO_ENCODING;

pub mod cjk;
mod tables;

pub use cjk::{CjkDecoder, CjkEncoding};

/// A text encoding this crate can decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextEncoding {
    /// UTF-8, optionally with a byte-order mark.
    Utf8,
    /// UTF-16 little endian.
    Utf16Le,
    /// UTF-16 big endian.
    Utf16Be,
    /// ISO-8859-1 / Latin-1 (every byte maps to one character).
    Latin1,
    /// GB18030, the superset of GBK/GB2312 (lead `0x81..=0xFE`).
    ///
    /// Name aliases `gbk`, `gb2312`, `cp936` and `gb18030` all select this variant, which
    /// reports itself as `gb18030`; GBK text therefore decodes correctly under either name.
    Gb18030,
    /// Big5 (Python's `big5` codec, lead `0xA1..=0xF9`); Big5-HKSCS extensions are not
    /// included and decode as `U+FFFD`.
    Big5,
    /// Shift-JIS / JIS X 0208 (Python's `shift_jis`, not the CP932 vendor extension).
    ShiftJis,
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
            Self::Gb18030 => "gb18030",
            Self::Big5 => "big5",
            Self::ShiftJis => "shift-jis",
        }
    }

    /// Parses a case-insensitive encoding name.
    ///
    /// Legacy aliases collapse onto the canonical variant (`gbk`/`gb2312`/`cp936` →
    /// [`TextEncoding::Gb18030`], `big5-hkscs`/`cp950` → [`TextEncoding::Big5`],
    /// `sjis`/`cp932` → [`TextEncoding::ShiftJis`]) because the decoder accepts the whole
    /// covering code space.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        let normalized = name.trim().to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "utf-8" | "utf8" | "us-ascii" | "ascii" => Some(Self::Utf8),
            "utf-16" | "utf-16le" | "utf16" | "utf16le" => Some(Self::Utf16Le),
            "utf-16be" | "utf16be" => Some(Self::Utf16Be),
            "latin-1" | "latin1" | "iso-8859-1" | "l1" => Some(Self::Latin1),
            "gb18030" | "gbk" | "gb2312" | "cp936" => Some(Self::Gb18030),
            "big5" | "big5-hkscs" | "cp950" => Some(Self::Big5),
            "shift-jis" | "shiftjis" | "sjis" | "cp932" => Some(Self::ShiftJis),
            _ => None,
        }
    }

    /// Returns `true` for the legacy CJK encodings decoded through [`cjk`].
    #[must_use]
    pub const fn is_cjk(self) -> bool {
        matches!(self, Self::Gb18030 | Self::Big5 | Self::ShiftJis)
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

/// Characters inspected when scoring a CJK candidate (Python `_QUALITY_SAMPLE_LIMIT`).
const QUALITY_SAMPLE_CHARS: usize = 2 * 1024 * 1024;

/// The encodings `auto` detection considers after UTF-8 fails, in Python's order.
const CJK_CANDIDATES: [CjkEncoding; 2] = [CjkEncoding::Gb18030, CjkEncoding::Big5];

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
/// The decision order is: BOM, UTF-16 heuristic (interleaved NUL bytes), valid UTF-8,
/// quality-scored strict GB18030/Big5, Latin-1.
///
/// The CJK step decodes the whole input strictly with GB18030 and then Big5
/// and scores the result like Python's `_text_quality`: CJK characters add 2, private-use
/// characters subtract 20 and control characters subtract 10. Only a positive score is
/// trusted, and GB18030 wins ties, so ambiguous Simplified Chinese stays GB18030 while
/// Big5-only text (which decodes to private-use characters under GB18030) is reported as
/// Big5. Shift-JIS is never auto-detected, matching Python.
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
    if let Some(encoding) = detect_cjk(bytes) {
        return encoding;
    }
    TextEncoding::Latin1
}

/// Chooses between strict GB18030 and strict Big5 decodes, preferring GB18030 on a tie.
fn detect_cjk(bytes: &[u8]) -> Option<TextEncoding> {
    let mut best: Option<(TextEncoding, i64)> = None;
    for encoding in CJK_CANDIDATES {
        let Some(text) = cjk::decode_strict(bytes, encoding) else {
            continue;
        };
        let score = text_quality(&text);
        // Strictly greater, so the first candidate in Python's order wins a tie.
        if best.is_none_or(|(_, current)| score > current) {
            best = Some((encoding.text_encoding(), score));
        }
    }
    best.filter(|(_, score)| *score > 0).map(|(kind, _)| kind)
}

/// Scores decoded text by CJK content and suspicious characters (Python `_text_quality`).
///
/// At most 2 Mi characters are inspected (the first and last half), mirroring Python.
fn text_quality(text: &str) -> i64 {
    let total = text.chars().count();
    if total <= QUALITY_SAMPLE_CHARS {
        return text.chars().map(character_weight).sum();
    }
    let half = QUALITY_SAMPLE_CHARS / 2;
    text.chars()
        .enumerate()
        .filter(|(index, _)| *index < half || *index >= total - half)
        .map(|(_, character)| character_weight(character))
        .sum()
}

/// Weight of one decoded character: CJK counts for text, private-use and control subtract.
const fn character_weight(character: char) -> i64 {
    match character as u32 {
        0x3000..=0x303F | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => 2,
        0xE000..=0xF8FF => -20,
        0x00..=0x08 | 0x0B..=0x0C | 0x0E..=0x1F | 0x7F..=0x9F => -10,
        _ => 0,
    }
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
/// Heuristic: a NUL byte (outside recognized UTF-16), more than 30% control bytes in the
/// sniffed prefix, or valid UTF-8 with unusual control characters. Only the first 8 KiB are
/// inspected, so detection is cheap and deterministic.
///
/// Bytes `0x80..=0xFF` are not control characters here, so legacy CJK text (whose lead and
/// trail bytes all live above `0x7F`) is never classified as binary.
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

/// Decodes `bytes` into text, replacing undecodable sequences with `U+FFFD`.
///
/// `encoding` accepts `"auto"` plus every name [`TextEncoding::parse`] understands.
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
                "encoding '{encoding}' is not supported; supported encodings are utf-8, \
                 utf-16le, utf-16be, latin-1, gb18030 (gbk), big5, shift-jis, and 'auto'"
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
        TextEncoding::Gb18030 => cjk::decode(payload, CjkEncoding::Gb18030),
        TextEncoding::Big5 => cjk::decode(payload, CjkEncoding::Big5),
        TextEncoding::ShiftJis => cjk::decode(payload, CjkEncoding::ShiftJis),
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

/// Encodes text for writing.
///
/// UTF-8 is the only encoding this crate writes; UTF-16 LE/BE are also available because
/// they round-trip exactly, and Latin-1 (like every other single-byte request) falls back
/// to UTF-8 bytes.
///
/// # Errors
///
/// Returns `ErrorKind::Unsupported` for the legacy CJK names: decoding them is supported,
/// but writing GB18030/Big5/Shift-JIS bytes is not (Python lets the caller pick a codec).
pub fn encode_text(text: &str, encoding: &str) -> Result<Vec<u8>> {
    match TextEncoding::parse(encoding) {
        Some(TextEncoding::Utf16Le) => Ok(encode_utf16(text, true)),
        Some(TextEncoding::Utf16Be) => Ok(encode_utf16(text, false)),
        Some(TextEncoding::Gb18030 | TextEncoding::Big5 | TextEncoding::ShiftJis) => {
            Err(Error::unsupported(format!(
                "encoding '{encoding}' is not supported for writing; use utf-8"
            )))
        }
        Some(TextEncoding::Utf8 | TextEncoding::Latin1) => Ok(text.as_bytes().to_vec()),
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
/// Compares `part * 10 > total * 3` on integers to avoid floating-point conversion.
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
        // Python's detector reports GB18030 for byte pairs that decode as CJK text; the
        // explicit name still decodes as Latin-1.
        assert_eq!(detect_text_encoding(&bytes), TextEncoding::Gb18030);
        assert_eq!(
            decode_text(&bytes, "latin-1", BinaryPolicy::Lossy).unwrap(),
            "éè"
        );
        // Bytes that no CJK codec accepts fall back to Latin-1, as before.
        assert_eq!(detect_text_encoding(&[0x80]), TextEncoding::Latin1);
        assert_eq!(
            decode_text(&[0x80], "auto", BinaryPolicy::Lossy).unwrap(),
            "\u{80}"
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
        let error = decode_text(b"x", "koi8-r", BinaryPolicy::Lossy).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.message().contains("koi8-r"));
    }

    #[test]
    fn legacy_cjk_names_decode() {
        assert_eq!(
            decode_text(&[0xD6, 0xD0, 0xCE, 0xC4], "gbk", BinaryPolicy::Lossy).unwrap(),
            "中文"
        );
        assert_eq!(
            decode_text(&[0xD6, 0xD0, 0xCE, 0xC4], "gb18030", BinaryPolicy::Lossy).unwrap(),
            "中文"
        );
        assert_eq!(
            decode_text(
                &[0x93, 0xFA, 0x96, 0x7B, 0x8C, 0xEA],
                "sjis",
                BinaryPolicy::Lossy
            )
            .unwrap(),
            "日本語"
        );
        // A caller asking for `gbk` gets the canonical `gb18030` name back.
        assert_eq!(detected_encoding_name(&[0xD6, 0xD0, 0xCE, 0xC4]), "gb18030");
    }

    #[test]
    fn legacy_cjk_names_are_rejected_for_writing() {
        let error = encode_text("中文", "gbk").unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.message().contains("not supported for writing"));
        assert_eq!(encode_text("hello", "ascii").unwrap(), b"hello");
    }

    #[test]
    fn auto_detection_prefers_big5_only_for_big5_text() {
        // Ambiguous Simplified Chinese: both codecs decode, so GB18030 wins the tie.
        assert_eq!(
            detect_text_encoding(&[0xD6, 0xD0, 0xCE, 0xC4]),
            TextEncoding::Gb18030
        );
        // Traditional characters decode to private-use code points under GB18030.
        assert_eq!(
            detect_text_encoding(&[0xC0, 0x73, 0xAA, 0xF9]),
            TextEncoding::Big5
        );
        assert_eq!(detected_encoding_name(&[0xC0, 0x73, 0xAA, 0xF9]), "big5");
        // Shift-JIS is deliberately not auto-detected.
        assert_eq!(
            detect_text_encoding(&[0x93, 0xFA, 0x96, 0x7B, 0x8C, 0xEA]),
            TextEncoding::Gb18030
        );
    }

    #[test]
    fn binary_sniffing() {
        assert!(!is_binary(b"hello\nworld\n"));
        assert!(!is_binary("héllo".as_bytes()));
        assert!(is_binary(&[0x00, 0x01, 0x02, 0x03]));
        assert!(is_binary(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]));
        assert!(!is_binary(&[]));
        // Legacy CJK bytes are high bytes, never control bytes.
        assert!(!is_binary(&[0xD6, 0xD0, 0xCE, 0xC4]));
        assert!(!is_binary(&[0xA4, 0xA4, 0xA4, 0xE5]));
    }

    #[test]
    fn reject_policy_blocks_binary_content() {
        let error = decode_text(&[0x00, 0x01], "utf-8", BinaryPolicy::Reject).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(decode_text(&[0x00, 0x01], "utf-8", BinaryPolicy::Lossy).is_ok());
        // CJK content is text, so the reject policy accepts it.
        assert!(decode_text(&[0xD6, 0xD0], "gbk", BinaryPolicy::Reject).is_ok());
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
        for name in ["gbk", "GB2312", "cp936", "GB18030"] {
            assert_eq!(TextEncoding::parse(name), Some(TextEncoding::Gb18030));
        }
        for name in ["big5", "big5_hkscs", "cp950"] {
            assert_eq!(TextEncoding::parse(name), Some(TextEncoding::Big5));
        }
        for name in ["shift_jis", "shift-jis", "sjis", "cp932"] {
            assert_eq!(TextEncoding::parse(name), Some(TextEncoding::ShiftJis));
        }
        assert_eq!(TextEncoding::parse("koi8-r"), None);
        assert_eq!(TextEncoding::Latin1.to_string(), "latin-1");
        assert_eq!(TextEncoding::Gb18030.to_string(), "gb18030");
        assert_eq!(TextEncoding::ShiftJis.as_str(), "shift-jis");
        assert!(TextEncoding::Big5.is_cjk());
        assert!(!TextEncoding::Utf8.is_cjk());
    }

    #[test]
    fn detected_encoding_name_reports_bom() {
        assert_eq!(detected_encoding_name(&[0xFF, 0xFE, b'h', 0]), "utf-16le");
        assert_eq!(detected_encoding_name(b"plain"), "utf-8");
    }

    #[test]
    fn quality_scores_cjk_private_use_and_controls() {
        assert_eq!(text_quality("中文"), 4);
        assert_eq!(text_quality("abc"), 0);
        assert_eq!(text_quality("\u{E000}"), -20);
        assert_eq!(text_quality("\u{1}"), -10);
    }
}
