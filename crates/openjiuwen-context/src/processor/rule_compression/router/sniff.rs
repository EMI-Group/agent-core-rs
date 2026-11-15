//! Content sniffing: byte-order marks, UTF-16 payloads and binary bodies.
//!
//! A tool result reaches the router as a `String`, but it may still be a
//! mis-decoded payload: a file read as bytes, a UTF-16 log, or a binary blob.
//! [`sniff_content`] classifies the body once, before any content-type
//! detection, and the router only ever classifies the text it returns.
//!
//! The sniff order is fixed:
//!
//! 1. **UTF-16**: a `FF FE` (`UTF-16LE`) or `FE FF` (`UTF-16BE`) byte-order mark
//!    re-decodes the body as UTF-16 and reports the conversion, so compression
//!    runs on UTF-8 text.
//! 2. **UTF-8 BOM**: a leading `U+FEFF` is stripped; the body is already UTF-8.
//! 3. **Binary**: a NUL byte, or more than [`MAX_CONTROL_RATIO_PERCENT`] percent
//!    of the first [`BINARY_SNIFF_BYTES`] bytes being control characters other
//!    than tab/newline/carriage-return/form-feed/vertical-tab, marks the body as
//!    binary; it is passed through untouched rather than compressed.
//! 4. **Replacement-heavy text**: at least [`REPLACEMENT_CHAR_MIN`] `U+FFFD` and
//!    at least [`REPLACEMENT_CHAR_RATIO_PERCENT`] percent of the body means an
//!    earlier lossy decode already destroyed the content, so it is passed
//!    through untouched as well.
//!
//! A `String` that a caller already decoded from UTF-16LE bytes (rather than a
//! byte slice handed to [`sniff_bytes`]) carries the BOM as the two characters
//! `U+00FF U+00FE`; [`sniff_content`] recognises that spelling and reconstructs
//! the bytes before re-decoding, so a latin-1 projection of a UTF-16 body is
//! recovered too.
use serde::{Deserialize, Serialize};

/// How much of a body the binary scan looks at.
pub const BINARY_SNIFF_BYTES: usize = 8 * 1024;
/// Control-character percentage above which a body counts as binary.
pub const MAX_CONTROL_RATIO_PERCENT: usize = 5;
/// Replacement characters a body needs before it counts as replacement-heavy.
pub const REPLACEMENT_CHAR_MIN: usize = 8;
/// Replacement-character percentage above which a body counts as mangled text.
pub const REPLACEMENT_CHAR_RATIO_PERCENT: usize = 5;
/// The `U+FFFD` replacement character.
const REPLACEMENT_CHARACTER: char = '\u{fffd}';

/// What [`sniff_content`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SniffKind {
    /// Plain UTF-8 with no byte-order mark.
    Utf8,
    /// UTF-8 behind a byte-order mark, which was stripped.
    Utf8Bom,
    /// `UTF-16LE` behind a byte-order mark, re-encoded to UTF-8.
    Utf16Le,
    /// `UTF-16BE` behind a byte-order mark, re-encoded to UTF-8.
    Utf16Be,
    /// A binary payload: NUL bytes or too many control characters.
    Binary,
    /// Text an earlier lossy decode already replaced heavily.
    ReplacementHeavy,
}
impl SniffKind {
    /// The short label this kind is reported under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Utf8 => "utf8",
            Self::Utf8Bom => "utf8_bom",
            Self::Utf16Le => "utf16_le",
            Self::Utf16Be => "utf16_be",
            Self::Binary => "binary",
            Self::ReplacementHeavy => "replacement_heavy",
        }
    }
    /// The encoding a re-decoded body came from, when it was re-decoded.
    #[must_use]
    pub fn source_encoding(self) -> Option<&'static str> {
        match self {
            Self::Utf16Le => Some("utf-16le"),
            Self::Utf16Be => Some("utf-16be"),
            Self::Utf8Bom | Self::Utf8 | Self::Binary | Self::ReplacementHeavy => None,
        }
    }
    /// Whether the body must be passed through untouched.
    #[must_use]
    pub fn is_passthrough(self) -> bool {
        matches!(self, Self::Binary | Self::ReplacementHeavy)
    }
}
impl std::fmt::Display for SniffKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A sniffed body: the text to classify plus what the sniff saw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SniffedContent {
    /// The body as UTF-8, after any byte-order mark or UTF-16 re-encoding.
    pub text: String,
    /// What the sniff detected.
    pub kind: SniffKind,
    /// Whether [`SniffedContent::text`] differs from the input.
    pub converted: bool,
    /// NUL bytes found in the scanned prefix.
    pub nul_bytes: usize,
    /// Control characters found in the scanned prefix, as a percentage.
    pub control_ratio: f64,
    /// `U+FFFD` characters in the text.
    pub replacement_characters: usize,
}
impl SniffedContent {
    /// The sniffed text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
    /// Whether the body must be passed through instead of compressed.
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.kind.is_passthrough()
    }
    /// `utf-16le`/`utf-16be` when the body was re-encoded, else `None`.
    #[must_use]
    pub fn source_encoding(&self) -> Option<&'static str> {
        self.kind.source_encoding()
    }
    /// The `details` fragment a compressor may merge into its own report.
    #[must_use]
    pub fn details(&self) -> Option<serde_json::Value> {
        if self.kind == SniffKind::Utf8 {
            return None;
        }
        Some(serde_json::json!({
            "content_kind": self.kind.as_str(),
            "source_encoding": self.source_encoding(),
            "recoded": self.converted,
            "nul_bytes": self.nul_bytes,
            "control_ratio": self.control_ratio,
            "replacement_characters": self.replacement_characters,
        }))
    }
}

/// Sniffs a `&str` body, recovering a latin-1 projection of a UTF-16 payload.
#[must_use]
pub fn sniff_content(content: &str) -> SniffedContent {
    match content.chars().take(2).collect::<Vec<_>>().as_slice() {
        // A UTF-16 BOM that an earlier decode turned into `ÿþ` or `þÿ`.
        ['\u{ff}', '\u{fe}'] | ['\u{fe}', '\u{ff}'] => sniff_bytes(&latin1_bytes(content)),
        _ => sniff_bytes(content.as_bytes()),
    }
}

/// Sniffs a raw byte body.
#[must_use]
pub fn sniff_bytes(bytes: &[u8]) -> SniffedContent {
    // UTF-16 wins over everything else: its BOM cannot occur at the start of a
    // UTF-8 document.
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        let text = decode_utf16(rest, true);
        return finish(text, SniffKind::Utf16Le, true);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        let text = decode_utf16(rest, false);
        return finish(text, SniffKind::Utf16Be, true);
    }
    let (text, kind) = match bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        Some(rest) => (
            String::from_utf8_lossy(rest).into_owned(),
            SniffKind::Utf8Bom,
        ),
        None => (String::from_utf8_lossy(bytes).into_owned(), SniffKind::Utf8),
    };
    finish(text, kind, false)
}

/// Fills in the binary, replacement and conversion facts of a sniffed body.
fn finish(text: String, kind: SniffKind, converted: bool) -> SniffedContent {
    let scan = scan_bytes(text.as_bytes());
    let replacement_characters = text
        .chars()
        .filter(|value| *value == REPLACEMENT_CHARACTER)
        .count();
    let characters = text.chars().count().max(1);
    let class = if scan.nul_bytes > 0 || scan.control_ratio > MAX_CONTROL_RATIO_PERCENT as f64 {
        SniffKind::Binary
    } else if replacement_characters >= REPLACEMENT_CHAR_MIN
        && replacement_characters * 100 >= characters * REPLACEMENT_CHAR_RATIO_PERCENT
    {
        SniffKind::ReplacementHeavy
    } else {
        kind
    };
    SniffedContent {
        text,
        kind: class,
        converted,
        nul_bytes: scan.nul_bytes,
        control_ratio: scan.control_ratio,
        replacement_characters,
    }
}

/// The NUL and control-character counts of the scanned prefix.
struct ByteScan {
    nul_bytes: usize,
    control_ratio: f64,
}

/// Counts NULs and control characters in the first [`BINARY_SNIFF_BYTES`] bytes.
fn scan_bytes(bytes: &[u8]) -> ByteScan {
    let scanned = bytes.get(..BINARY_SNIFF_BYTES).unwrap_or(bytes);
    if scanned.is_empty() {
        return ByteScan {
            nul_bytes: 0,
            control_ratio: 0.0,
        };
    }
    let mut nul_bytes = 0_usize;
    let mut controls = 0_usize;
    for byte in scanned {
        match *byte {
            0 => nul_bytes += 1,
            // C0 controls other than tab/newline/carriage-return/form-feed/
            // vertical-tab, plus DEL: printable text never carries these.
            0x01..=0x08 | 0x0E..=0x1F | 0x7F => controls += 1,
            _ => {}
        }
    }
    ByteScan {
        nul_bytes,
        control_ratio: controls as f64 * 100.0 / scanned.len() as f64,
    }
}

/// Decodes UTF-16 code units into text, dropping a redundant leading BOM.
///
/// A trailing odd byte is dropped, and an unpaired surrogate decodes to `U+FFFD`
/// through `String::from_utf16_lossy`.
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
    let decoded = String::from_utf16_lossy(&units);
    match decoded.strip_prefix('\u{feff}') {
        Some(rest) => rest.to_owned(),
        None => decoded,
    }
}

/// The latin-1 projection of `text`, one byte per character below `U+0100`.
fn latin1_bytes(text: &str) -> Vec<u8> {
    text.chars()
        .map(|value| u8::try_from(u32::from(value)).unwrap_or(b'?'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_utf8_is_untouched() {
        let sniffed = sniff_content("[{\"a\": 1}]");
        assert_eq!(sniffed.kind, SniffKind::Utf8);
        assert_eq!(sniffed.text, "[{\"a\": 1}]");
        assert!(!sniffed.converted);
        assert!(sniffed.details().is_none());
    }

    #[test]
    fn a_utf8_bom_is_stripped() {
        let sniffed = sniff_content("\u{feff}hello");
        assert_eq!(sniffed.kind, SniffKind::Utf8Bom);
        assert_eq!(sniffed.text, "hello");
        assert!(!sniffed.converted);
        assert!(sniffed.details().is_some());
    }

    #[test]
    fn utf16_payloads_are_recoded() {
        let le = sniff_bytes(b"\xFF\xFEh\x00i\x00");
        assert_eq!(le.kind, SniffKind::Utf16Le);
        assert_eq!(le.text, "hi");
        assert!(le.converted);
        assert_eq!(le.source_encoding(), Some("utf-16le"));

        let be = sniff_bytes(b"\xFE\xFF\x00h\x00i");
        assert_eq!(be.kind, SniffKind::Utf16Be);
        assert_eq!(be.text, "hi");
        assert_eq!(be.source_encoding(), Some("utf-16be"));

        // The latin-1 projection of the same payload, as a decoded `String`.
        let projected = sniff_content("\u{ff}\u{fe}h\u{0}i\u{0}");
        assert_eq!(projected.kind, SniffKind::Utf16Le);
        assert_eq!(projected.text, "hi");
        assert!(projected.converted);
    }

    #[test]
    fn nul_bytes_and_control_characters_are_binary() {
        let nul = sniff_content("a\u{0}b");
        assert_eq!(nul.kind, SniffKind::Binary);
        assert_eq!(nul.nul_bytes, 1);
        assert!(nul.is_passthrough());

        let controls = "\u{1}".repeat(20);
        let binary = sniff_content(&controls);
        assert_eq!(binary.kind, SniffKind::Binary);
        assert!(binary.control_ratio > MAX_CONTROL_RATIO_PERCENT as f64);

        // Tabs and newlines are text.
        assert_eq!(sniff_content("a\tb\nc\r\n").kind, SniffKind::Utf8);
    }

    #[test]
    fn replacement_heavy_text_is_reported() {
        let mangled = REPLACEMENT_CHARACTER.to_string().repeat(10);
        let sniffed = sniff_content(&mangled);
        assert_eq!(sniffed.kind, SniffKind::ReplacementHeavy);
        assert_eq!(sniffed.replacement_characters, 10);
        assert!(sniffed.is_passthrough());
    }

    #[test]
    fn kinds_report_their_labels() {
        assert_eq!(SniffKind::Utf16Le.as_str(), "utf16_le");
        assert_eq!(SniffKind::Binary.to_string(), "binary");
        assert!(SniffKind::Binary.is_passthrough());
        assert!(!SniffKind::Utf8Bom.is_passthrough());
    }
}
