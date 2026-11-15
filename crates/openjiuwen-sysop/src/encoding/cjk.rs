//! Bounded decoders for the legacy CJK encodings (GB18030/GBK, Big5, Shift-JIS).
//!
//! Python parity: the local filesystem backend decodes with `errors="replace"`, so an
//! invalid or truncated sequence becomes `U+FFFD` and decoding resynchronises instead of
//! failing the read. This module reproduces that behavior byte for byte (the generator in
//! `scripts/gen_cjk_tables.py` fuzzes the model against Python's `gb18030`, `big5` and
//! `shift_jis` codecs in replace mode).
//!
//! Decoding is table driven; writing uses the reverse geometry at the bottom of this module
//! (the crate-internal `two_byte_bytes`, `four_byte_index` and `four_byte_bytes`) through the
//! private `encode` module, which verifies every candidate code by decoding it back first.
//!
//! ```
//! use openjiuwen_sysop::encoding::cjk::{CjkDecoder, CjkEncoding};
//!
//! assert_eq!(
//!     CjkDecoder::new(CjkEncoding::Gb18030).push(&[0xD6, 0xD0, 0xCE, 0xC4]),
//!     "中文"
//! );
//! ```
//!
//! Work is bounded per input byte: a two-byte pair is one table lookup, a four-byte
//! sequence is one binary search over the 207 range-compressed runs, and the streaming
//! state holds at most three buffered bytes. Nothing scans backwards, so decoding is
//! linear in the input.

use std::fmt;

use super::tables;
use super::TextEncoding;

/// The replacement character Python's `errors="replace"` emits for invalid input.
const REPLACEMENT: char = '\u{FFFD}';

/// Longest multi-byte sequence this module decodes (GB18030 four-byte form).
const MAX_SEQUENCE: usize = 4;

/// First byte of the Shift-JIS halfwidth katakana block.
const HALFWIDTH_KATAKANA_START: u8 = 0xA1;

/// Last byte of the Shift-JIS halfwidth katakana block.
const HALFWIDTH_KATAKANA_END: u8 = 0xDF;

/// First code point of the Shift-JIS halfwidth katakana block.
const HALFWIDTH_KATAKANA_UNICODE: u32 = 0xFF61;

/// A legacy CJK encoding this crate can decode.
///
/// The variants are also exposed as [`TextEncoding::Gb18030`], [`TextEncoding::Big5`] and
/// [`TextEncoding::ShiftJis`]; both spellings share the same decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CjkEncoding {
    /// GB18030, the superset of GBK/GB2312 (the decoder accepts every GBK spelling).
    Gb18030,
    /// Big5 (as implemented by Python's `big5` codec; Big5-HKSCS is not included).
    Big5,
    /// Shift-JIS (JIS X 0208; the CP932 vendor extension is not included).
    ShiftJis,
}

impl CjkEncoding {
    /// The canonical name reported back to callers.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gb18030 => "gb18030",
            Self::Big5 => "big5",
            Self::ShiftJis => "shift-jis",
        }
    }

    /// The [`TextEncoding`] variant that selects this decoder.
    #[must_use]
    pub const fn text_encoding(self) -> TextEncoding {
        match self {
            Self::Gb18030 => TextEncoding::Gb18030,
            Self::Big5 => TextEncoding::Big5,
            Self::ShiftJis => TextEncoding::ShiftJis,
        }
    }

    /// Returns the decoder for a legacy [`TextEncoding`], or `None` for UTF-8, UTF-16 and
    /// Latin-1, which are not table driven.
    #[must_use]
    pub const fn from_text_encoding(encoding: TextEncoding) -> Option<Self> {
        match encoding {
            TextEncoding::Gb18030 => Some(Self::Gb18030),
            TextEncoding::Big5 => Some(Self::Big5),
            TextEncoding::ShiftJis => Some(Self::ShiftJis),
            TextEncoding::Utf8
            | TextEncoding::Utf16Le
            | TextEncoding::Utf16Be
            | TextEncoding::Latin1 => None,
        }
    }
}

impl fmt::Display for CjkEncoding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Decodes `bytes`, replacing every invalid or truncated sequence with `U+FFFD`.
///
/// Invalid sequences consume one byte so a following ASCII byte is still decoded
/// (Python's replace handler reports the same one-byte span); a truncated sequence at the
/// end of the input contributes a single `U+FFFD`, matching Python's
/// `incomplete multibyte sequence` handling.
///
/// ```
/// use openjiuwen_sysop::encoding::cjk::{decode, CjkEncoding};
///
/// assert_eq!(decode(&[0x93, 0xFA, 0x96, 0x7B, 0x8C, 0xEA], CjkEncoding::ShiftJis), "日本語");
/// assert_eq!(decode(b"ok\xD6", CjkEncoding::Gb18030), "ok\u{FFFD}");
/// ```
#[must_use]
pub fn decode(bytes: &[u8], encoding: CjkEncoding) -> String {
    let mut decoder = CjkDecoder::new(encoding);
    let mut text = decoder.push(bytes);
    text.push_str(&decoder.finish());
    text
}

/// Strictly decodes `bytes`, returning `None` when any sequence is invalid or truncated.
///
/// Used by [`detect_text_encoding`](super::detect_text_encoding) as Python's
/// `codecs.getincrementaldecoder(...)(errors="strict")` check.
#[must_use]
pub fn decode_strict(bytes: &[u8], encoding: CjkEncoding) -> Option<String> {
    let mut decoder = CjkDecoder::new(encoding);
    let mut text = decoder.push(bytes);
    text.push_str(&decoder.finish());
    (!decoder.invalid_input()).then_some(text)
}

/// An incremental decoder for one legacy CJK encoding.
///
/// Feed chunks with [`CjkDecoder::push`] and call [`CjkDecoder::finish`] once the stream
/// ends; a multi-byte sequence split across chunks is buffered (at most three bytes) and
/// decoded on the next push. The output is identical to decoding the concatenation in one
/// call.
///
/// ```
/// use openjiuwen_sysop::encoding::cjk::{CjkDecoder, CjkEncoding};
///
/// let mut decoder = CjkDecoder::new(CjkEncoding::Big5);
/// let mut text = decoder.push(&[0xA4]);
/// text.push_str(&decoder.push(&[0xA4, 0xA4, 0xE5]));
/// text.push_str(&decoder.finish());
/// assert_eq!(text, "中文");
/// ```
#[derive(Debug, Clone)]
pub struct CjkDecoder {
    encoding: CjkEncoding,
    pending: [u8; MAX_SEQUENCE - 1],
    pending_len: usize,
    invalid: bool,
}

impl CjkDecoder {
    /// Creates a decoder for `encoding`.
    #[must_use]
    pub const fn new(encoding: CjkEncoding) -> Self {
        Self {
            encoding,
            pending: [0; MAX_SEQUENCE - 1],
            pending_len: 0,
            invalid: false,
        }
    }

    /// The encoding this decoder was built for.
    #[must_use]
    pub const fn encoding(&self) -> CjkEncoding {
        self.encoding
    }

    /// Returns `true` once any invalid or truncated sequence was replaced with `U+FFFD`.
    #[must_use]
    pub const fn invalid_input(&self) -> bool {
        self.invalid
    }

    /// Decodes `chunk`, appending to the text returned by previous calls.
    ///
    /// A trailing partial sequence is buffered, so the returned string may be empty even
    /// though `chunk` was not.
    pub fn push(&mut self, chunk: &[u8]) -> String {
        if self.pending_len == 0 {
            return self.run(chunk);
        }
        let mut joined = Vec::with_capacity(self.pending_len + chunk.len());
        joined.extend_from_slice(&self.pending[..self.pending_len]);
        joined.extend_from_slice(chunk);
        self.pending_len = 0;
        self.run(&joined)
    }

    /// Flushes the decoder, replacing a trailing truncated sequence with `U+FFFD`.
    pub fn finish(&mut self) -> String {
        if self.pending_len == 0 {
            return String::new();
        }
        self.pending_len = 0;
        self.invalid = true;
        REPLACEMENT.to_string()
    }

    /// Decodes a complete buffer, buffering a trailing partial sequence.
    fn run(&mut self, data: &[u8]) -> String {
        let mut out = String::with_capacity(data.len());
        let mut index = 0;
        while index < data.len() {
            let byte = data[index];
            if byte < 0x80 {
                out.push(char::from(byte));
                index += 1;
                continue;
            }
            if self.encoding == CjkEncoding::ShiftJis
                && (HALFWIDTH_KATAKANA_START..=HALFWIDTH_KATAKANA_END).contains(&byte)
            {
                out.push(halfwidth_katakana(byte));
                index += 1;
                continue;
            }
            if self.encoding == CjkEncoding::Gb18030
                && index + 1 < data.len()
                && (0x30..=0x39).contains(&data[index + 1])
            {
                // A digit second byte starts a four-byte candidate whatever the lead is:
                // Python's codec also accepts `0x80`/`0xFF` as a lead and then rejects the
                // sequence, so the shapes stay identical.
                if index + 3 < data.len() {
                    if let Some(character) =
                        gb18030_four_byte(byte, data[index + 1], data[index + 2], data[index + 3])
                    {
                        out.push(character);
                        index += MAX_SEQUENCE;
                        continue;
                    }
                    self.invalid = true;
                    out.push(REPLACEMENT);
                    index += 1;
                    continue;
                }
                self.buffer(&data[index..]);
                break;
            }
            if index + 1 >= data.len() {
                self.buffer(&data[index..]);
                break;
            }
            if let Some(character) = two_byte_char(self.encoding, byte, data[index + 1]) {
                out.push(character);
                index += 2;
            } else {
                self.invalid = true;
                out.push(REPLACEMENT);
                index += 1;
            }
        }
        out
    }

    /// Copies the tail of a chunk into the pending buffer.
    fn buffer(&mut self, tail: &[u8]) {
        let length = tail.len().min(self.pending.len());
        self.pending[..length].copy_from_slice(&tail[..length]);
        self.pending_len = length;
    }
}

/// Decodes one two-byte pair, returning `None` when the pair is unmapped.
fn two_byte_char(encoding: CjkEncoding, lead: u8, trail: u8) -> Option<char> {
    match encoding {
        CjkEncoding::Gb18030 => {
            if !(0x81..=0xFE).contains(&lead) {
                return None;
            }
            let column = gb18030_column(trail)?;
            let row = usize::from(lead - 0x81);
            let value = tables::gb18030::TWO_BYTE
                .get(row * tables::gb18030::TWO_BYTE_COLUMNS + column)
                .copied()?;
            mapped(value)
        }
        CjkEncoding::Big5 => {
            if !(0xA1..=0xF9).contains(&lead) {
                return None;
            }
            let column = big5_column(trail)?;
            let row = usize::from(lead - 0xA1);
            let value = tables::big5::TWO_BYTE
                .get(row * tables::big5::TWO_BYTE_COLUMNS + column)
                .copied()?;
            mapped(value)
        }
        CjkEncoding::ShiftJis => {
            let row = shift_jis_row(lead)?;
            let column = shift_jis_column(trail)?;
            let value = tables::shift_jis::TWO_BYTE
                .get(row * tables::shift_jis::TWO_BYTE_COLUMNS + column)
                .copied()?;
            mapped(value)
        }
    }
}

/// GB18030 two-byte trail column (`0x40..=0x7E` then `0x80..=0xFE`).
fn gb18030_column(trail: u8) -> Option<usize> {
    match trail {
        0x40..=0x7E => Some(usize::from(trail - 0x40)),
        0x80..=0xFE => Some(usize::from(trail - 0x41)),
        _ => None,
    }
}

/// Big5 trail column (`0x40..=0x7E` then `0xA1..=0xFE`).
fn big5_column(trail: u8) -> Option<usize> {
    match trail {
        0x40..=0x7E => Some(usize::from(trail - 0x40)),
        0xA1..=0xFE => Some(usize::from(trail - 0x62)),
        _ => None,
    }
}

/// Shift-JIS trail column (`0x40..=0x7E` then `0x80..=0xFC`).
fn shift_jis_column(trail: u8) -> Option<usize> {
    match trail {
        0x40..=0x7E => Some(usize::from(trail - 0x40)),
        0x80..=0xFC => Some(usize::from(trail - 0x80) + 63),
        _ => None,
    }
}

/// Shift-JIS lead row: rows `0..=30` are leads `0x81..=0x9F`, rows `31..=59` are
/// `0xE0..=0xFC`; every other high byte is invalid and is replaced.
fn shift_jis_row(lead: u8) -> Option<usize> {
    match lead {
        0x81..=0x9F => Some(usize::from(lead - 0x81)),
        0xE0..=0xFC => Some(usize::from(lead - 0xE0) + 31),
        _ => None,
    }
}

/// Decodes one GB18030 four-byte sequence from its four bytes.
fn gb18030_four_byte(lead: u8, second: u8, third: u8, fourth: u8) -> Option<char> {
    if !(0x81..=0xFE).contains(&lead)
        || !(0x81..=0xFE).contains(&third)
        || !(0x30..=0x39).contains(&second)
        || !(0x30..=0x39).contains(&fourth)
    {
        return None;
    }
    let index = (u32::from(lead - 0x81) * 10 + u32::from(second - 0x30)) * 1260
        + u32::from(third - 0x81) * 10
        + u32::from(fourth - 0x30);
    four_byte_char(index)
}

/// Looks up a GB18030 four-byte linear index in the range-compressed table.
fn four_byte_char(index: u32) -> Option<char> {
    if index > tables::gb18030::FOUR_BYTE_MAX_INDEX {
        return None;
    }
    let runs = &tables::gb18030::FOUR_BYTE_RUNS;
    let position = runs.partition_point(|run| run.0 <= index);
    let (start, first, length) = runs.get(position.checked_sub(1)?).copied()?;
    if index >= start + length {
        return None;
    }
    char::from_u32(first + (index - start))
}

/// Maps a table value to a character; `0` is the unmapped sentinel.
fn mapped(value: u16) -> Option<char> {
    if value == 0 {
        None
    } else {
        char::from_u32(u32::from(value))
    }
}

/// Maps a Shift-JIS single byte `0xA1..=0xDF` to halfwidth katakana.
fn halfwidth_katakana(byte: u8) -> char {
    char::from_u32(HALFWIDTH_KATAKANA_UNICODE + u32::from(byte - HALFWIDTH_KATAKANA_START))
        .unwrap_or(REPLACEMENT)
}

// ---------------------------------------------------------------------------------------
// Reverse geometry for the encoders.
//
// The tables above are the decode direction: a position in the dense two-byte array plus the
// lead/trail geometry yields a character. [`super::encode`] builds its reverse lookup from
// exactly these functions, so a position it emits is always a position the decoder above
// accepts, and it verifies every entry by decoding it back before trusting it.
// ---------------------------------------------------------------------------------------

/// Row and column count of an encoding's dense two-byte table.
///
/// The counts match the generated tables (`126 × 190` for GB18030, `89 × 157` for Big5,
/// `60 × 188` for Shift-JIS).
#[must_use]
pub(crate) const fn two_byte_shape(encoding: CjkEncoding) -> (usize, usize) {
    match encoding {
        CjkEncoding::Gb18030 => (126, tables::gb18030::TWO_BYTE_COLUMNS),
        CjkEncoding::Big5 => (89, tables::big5::TWO_BYTE_COLUMNS),
        CjkEncoding::ShiftJis => (60, tables::shift_jis::TWO_BYTE_COLUMNS),
    }
}

/// Returns the two bytes of a two-byte table position, or `None` for a position outside it.
///
/// This is the inverse of [`two_byte_char`]'s row/column geometry.
pub(crate) fn two_byte_bytes(encoding: CjkEncoding, row: usize, column: usize) -> Option<[u8; 2]> {
    let (rows, columns) = two_byte_shape(encoding);
    if row >= rows || column >= columns {
        return None;
    }
    let row = u32::try_from(row).ok()?;
    let column = u32::try_from(column).ok()?;
    let lead = match encoding {
        CjkEncoding::Gb18030 => 0x81_u8.checked_add(to_u8(row)?)?,
        CjkEncoding::Big5 => 0xA1_u8.checked_add(to_u8(row)?)?,
        CjkEncoding::ShiftJis if row <= 30 => 0x81_u8.checked_add(to_u8(row)?)?,
        CjkEncoding::ShiftJis => 0xE0_u8.checked_add(to_u8(row.checked_sub(31)?)?)?,
    };
    // Trails start at `0x40` and skip the `0x7F` hole; Big5 resumes at `0xA1`, the other
    // two codecs at `0x80`.
    let trail = if column <= 62 {
        0x40_u8.checked_add(to_u8(column)?)?
    } else {
        let offset = to_u8(column.checked_sub(63)?)?;
        match encoding {
            CjkEncoding::Big5 => 0xA1_u8.checked_add(offset)?,
            CjkEncoding::Gb18030 | CjkEncoding::ShiftJis => 0x80_u8.checked_add(offset)?,
        }
    };
    Some([lead, trail])
}

/// Decodes one two-byte pair, verifying a candidate code before it is used for writing.
pub(crate) fn decode_two_byte(encoding: CjkEncoding, lead: u8, trail: u8) -> Option<char> {
    two_byte_char(encoding, lead, trail)
}

/// Decodes one GB18030 four-byte sequence, verifying a candidate code before it is written.
pub(crate) fn decode_four_byte(bytes: [u8; 4]) -> Option<char> {
    gb18030_four_byte(bytes[0], bytes[1], bytes[2], bytes[3])
}

/// Returns the GB18030 four-byte linear index that maps to `character`.
///
/// `FOUR_BYTE_RUNS` is sorted by both index and code point, so this is one binary search
/// over the 207 runs and allocates nothing. Every code point in a run maps back to itself
/// through [`four_byte_bytes`] → [`decode_four_byte`].
pub(crate) fn four_byte_index(character: char) -> Option<u32> {
    let code = u32::from(character);
    let runs = &tables::gb18030::FOUR_BYTE_RUNS;
    let position = runs.partition_point(|run| run.1 <= code);
    let (start, first, length) = runs.get(position.checked_sub(1)?).copied()?;
    if code >= first + length {
        return None;
    }
    Some(start + (code - first))
}

/// Returns the four bytes of a GB18030 four-byte linear index.
///
/// This inverts `index = ((b1 - 0x81) * 10 + (b2 - 0x30)) * 1260 + (b3 - 0x81) * 10 +
/// (b4 - 0x30)`.
pub(crate) fn four_byte_bytes(index: u32) -> Option<[u8; 4]> {
    if index > tables::gb18030::FOUR_BYTE_MAX_INDEX {
        return None;
    }
    let units = index % 1260;
    let high = index / 1260;
    let lead = 0x81_u8.checked_add(to_u8(high / 10)?)?;
    let second = 0x30_u8.checked_add(to_u8(high % 10)?)?;
    let third = 0x81_u8.checked_add(to_u8(units / 10)?)?;
    let fourth = 0x30_u8.checked_add(to_u8(units % 10)?)?;
    Some([lead, second, third, fourth])
}

/// Returns the Shift-JIS halfwidth katakana byte for `character`, if it has one.
///
/// The block is positional (`U+FF61..=U+FF9F` ↔ `0xA1..=0xDF`), so no table is needed.
pub(crate) fn halfwidth_katakana_byte(character: char) -> Option<u8> {
    let code = u32::from(character);
    let last = HALFWIDTH_KATAKANA_UNICODE + u32::from(HALFWIDTH_KATAKANA_END)
        - u32::from(HALFWIDTH_KATAKANA_START);
    if (HALFWIDTH_KATAKANA_UNICODE..=last).contains(&code) {
        to_u8(code - HALFWIDTH_KATAKANA_UNICODE).map(|offset| HALFWIDTH_KATAKANA_START + offset)
    } else {
        None
    }
}

/// The byte range of the Shift-JIS halfwidth katakana block (`0xA1..=0xDF`).
pub(crate) const HALFWIDTH_KATAKANA_BYTES: std::ops::RangeInclusive<u8> =
    HALFWIDTH_KATAKANA_START..=HALFWIDTH_KATAKANA_END;

/// Decodes one Shift-JIS halfwidth katakana byte, or `None` outside the block.
pub(crate) fn decode_halfwidth_katakana(byte: u8) -> Option<char> {
    HALFWIDTH_KATAKANA_BYTES
        .contains(&byte)
        .then(|| halfwidth_katakana(byte))
}

/// Narrows a table position or offset to a byte.
fn to_u8(value: u32) -> Option<u8> {
    u8::try_from(value).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn gb18030_decodes_gbk_text_and_ascii() {
        assert_eq!(
            decode(&[0xD6, 0xD0, 0xCE, 0xC4], CjkEncoding::Gb18030),
            "中文"
        );
        assert_eq!(
            decode(b"GBK: \xD6\xD0\xCE\xC4 and ASCII", CjkEncoding::Gb18030),
            "GBK: 中文 and ASCII"
        );
        assert_eq!(decode(b"", CjkEncoding::Gb18030), "");
    }

    #[test]
    fn gb18030_decodes_four_byte_sequences() {
        // Index 0 maps to U+0080, index 189000 to the first supplementary code point.
        assert_eq!(
            decode(&[0x81, 0x30, 0x81, 0x30], CjkEncoding::Gb18030),
            "\u{80}"
        );
        assert_eq!(
            decode(&[0x90, 0x30, 0x81, 0x30], CjkEncoding::Gb18030),
            "\u{10000}"
        );
        assert_eq!(
            decode(&[0x95, 0x32, 0x82, 0x36], CjkEncoding::Gb18030),
            "\u{20000}"
        );
    }

    #[test]
    fn gb18030_replaces_invalid_and_truncated_sequences() {
        assert_eq!(decode(&[0xD6], CjkEncoding::Gb18030), "\u{FFFD}");
        assert_eq!(decode(b"abc\xD6", CjkEncoding::Gb18030), "abc\u{FFFD}");
        // The lead byte is replaced and the following ASCII byte survives.
        assert_eq!(decode(&[0xD6, b' '], CjkEncoding::Gb18030), "\u{FFFD} ");
        assert_eq!(
            decode(b"\xD6\xD0\xD6 x", CjkEncoding::Gb18030),
            "中\u{FFFD} x"
        );
        // A truncated four-byte sequence is one replacement, never a panic.
        assert_eq!(
            decode(&[0x81, 0x30, 0x81], CjkEncoding::Gb18030),
            "\u{FFFD}"
        );
        assert_eq!(decode(&[0x80, 0x30], CjkEncoding::Gb18030), "\u{FFFD}");
    }

    #[test]
    fn big5_decodes_simplified_and_traditional_text() {
        assert_eq!(decode(&[0xA4, 0xA4, 0xA4, 0xE5], CjkEncoding::Big5), "中文");
        assert_eq!(decode(&[0xC0, 0x73, 0xAA, 0xF9], CjkEncoding::Big5), "龍門");
        assert_eq!(decode(b"Big5: \xA4\xA4", CjkEncoding::Big5), "Big5: 中");
        // Unmapped pair: one replacement, then the trail byte resynchronises as ASCII.
        assert_eq!(decode(&[0xA1, 0x7F], CjkEncoding::Big5), "\u{FFFD}\u{7F}");
    }

    #[test]
    fn shift_jis_decodes_two_byte_and_halfwidth_forms() {
        assert_eq!(
            decode(&[0x93, 0xFA, 0x96, 0x7B, 0x8C, 0xEA], CjkEncoding::ShiftJis),
            "日本語"
        );
        // Halfwidth katakana is single byte, 0xA1 -> U+FF61.
        assert_eq!(decode(&[0xA1], CjkEncoding::ShiftJis), "\u{FF61}");
        assert_eq!(
            decode(&[0xB1, 0xB2, 0xB3], CjkEncoding::ShiftJis),
            "\u{FF71}\u{FF72}\u{FF73}"
        );
        assert_eq!(decode(&[0x80, b'A'], CjkEncoding::ShiftJis), "\u{FFFD}A");
    }

    #[test]
    fn strict_decoding_reports_invalid_input() {
        assert_eq!(
            decode_strict(&[0xD6, 0xD0], CjkEncoding::Gb18030).as_deref(),
            Some("中")
        );
        assert!(decode_strict(&[0xD6], CjkEncoding::Gb18030).is_none());
        assert!(decode_strict(b"plain", CjkEncoding::Big5).is_some());
        assert!(decode_strict(&[0xFF, 0xFF], CjkEncoding::ShiftJis).is_none());
    }

    #[test]
    fn incremental_decoding_matches_a_single_pass() {
        let bytes = b"a\xD6\xD0\xCE\xC4 b\x93\xFA".to_vec();
        let expected = decode(&bytes, CjkEncoding::Gb18030);
        for split in 0..=bytes.len() {
            let mut decoder = CjkDecoder::new(CjkEncoding::Gb18030);
            let mut text = decoder.push(&bytes[..split]);
            text.push_str(&decoder.push(&bytes[split..]));
            text.push_str(&decoder.finish());
            assert_eq!(text, expected, "split at {split}");
        }
    }

    #[test]
    fn incremental_decoding_buffers_partial_sequences() {
        let mut decoder = CjkDecoder::new(CjkEncoding::Big5);
        assert_eq!(decoder.push(&[0xA4]), "");
        assert_eq!(decoder.push(&[0xA4]), "中");
        // A dangling lead byte becomes one replacement on finish.
        assert_eq!(decoder.push(&[0xA4]), "");
        assert_eq!(decoder.finish(), "\u{FFFD}");
        assert!(decoder.invalid_input());
        assert_eq!(decoder.encoding(), CjkEncoding::Big5);
    }

    #[test]
    fn encodings_report_canonical_names() {
        assert_eq!(CjkEncoding::Gb18030.to_string(), "gb18030");
        assert_eq!(CjkEncoding::Big5.as_str(), "big5");
        assert_eq!(
            CjkEncoding::from_text_encoding(TextEncoding::ShiftJis),
            Some(CjkEncoding::ShiftJis)
        );
        assert_eq!(CjkEncoding::from_text_encoding(TextEncoding::Utf8), None);
        assert_eq!(CjkEncoding::Big5.text_encoding(), TextEncoding::Big5);
    }

    #[test]
    fn four_byte_index_bounds_are_enforced() {
        // U+FFFF is the last BMP mapping; one index later belongs to the unmapped gap.
        assert_eq!(
            decode(&[0x84, 0x31, 0xA4, 0x39], CjkEncoding::Gb18030),
            "\u{FFFF}"
        );
        assert_eq!(
            decode(&[0x84, 0x31, 0xA5, 0x30], CjkEncoding::Gb18030),
            "\u{FFFD}1\u{FFFD}"
        );
        // The last supplementary mapping is U+10FFFF.
        assert_eq!(
            decode(&[0xE3, 0x32, 0x9A, 0x35], CjkEncoding::Gb18030),
            "\u{10FFFF}"
        );
    }
}
