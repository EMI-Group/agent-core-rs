//! Reverse encoding for the write path: Latin-1 and the legacy CJK codecs.
//!
//! Decoding owns the generated tables in [`super::tables`]; this module builds the reverse
//! direction from them instead of shipping a second table (the generator is authoring-time
//! only, and a reverse table would roughly double the committed data). The mechanism:
//!
//! - **Two-byte areas** — one dense decode table per encoding (GB18030 `126 × 190`,
//!   Big5 `89 × 157`, Shift-JIS `60 × 188`). A reverse map is built on first use and cached
//!   in a `OnceLock` for the process lifetime: 23940 entries for GB18030 and ~14k/~7k for
//!   Big5/Shift-JIS, stored as a sorted `Vec<(char, Code)>` (12 bytes per entry, no hash
//!   table), so a lookup is a binary search (~15 comparisons) and startup stays cheap.
//!   Resident cost is one-time and bounded: ~280 KB for GB18030, ~160 KB for Big5 and
//!   ~80 KB for Shift-JIS, and only for encodings a process actually writes with.
//!   Every position is verified by decoding it back before it is inserted, and positions
//!   that decode to the same character are resolved to the shortest, then smallest, code so
//!   encoding is deterministic and `decode(encode(s)) == s` always holds.
//! - **GB18030 four-byte form** — no map at all: the 207 range runs are sorted by both
//!   index and code point, so the inverse is one binary search ([`cjk::four_byte_index`])
//!   plus the documented index algebra ([`cjk::four_byte_bytes`]), and the candidate bytes
//!   are decoded back before they are returned. That covers the BMP region and the linear
//!   supplementary region `189000..=1237575 ↔ U+10000..=U+10FFFF` without a 1.1M-entry map.
//! - **Shift-JIS halfwidth katakana** — positional (`U+FF61..=U+FF9F` ↔ `0xA1..=0xDF`), so
//!   it is arithmetic ([`cjk::halfwidth_katakana_byte`]), not table data.
//!
//! Memory is therefore bounded by one lazy map per CJK encoding that is actually written
//! with, and nothing is allocated for ASCII-only text (the ASCII fast path never touches the
//! map's build). Latin-1 is pure arithmetic.
//!
//! ```
//! use openjiuwen_sysop::encoding::{decode_text, encode_text};
//! use openjiuwen_sysop::BinaryPolicy;
//!
//! let bytes = encode_text("中文", "gb18030").unwrap();
//! assert_eq!(bytes, vec![0xD6, 0xD0, 0xCE, 0xC4]);
//! assert_eq!(decode_text(&bytes, "gb18030", BinaryPolicy::Lossy).unwrap(), "中文");
//! ```

use std::sync::OnceLock;

use openjiuwen_common::error::{Error, Result};

use super::cjk::{self, CjkEncoding};
use super::TextEncoding;

/// Longest code this encoder emits (the GB18030 four-byte form).
const MAX_CODE: usize = 4;

/// The bytes of one encoded character plus how many of them are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Code {
    /// Zero-padded code bytes; only the first `len` entries are meaningful.
    bytes: [u8; MAX_CODE],
    /// Number of meaningful bytes (1, 2, or 4).
    len: u8,
}

impl Code {
    /// The meaningful bytes of this code.
    fn as_slice(&self) -> &[u8] {
        self.bytes.get(..usize::from(self.len)).unwrap_or(&[])
    }

    /// A one-byte code (ASCII or a Shift-JIS halfwidth katakana).
    const fn single(byte: u8) -> Self {
        Self {
            bytes: [byte, 0, 0, 0],
            len: 1,
        }
    }

    /// A two-byte code.
    const fn pair(lead: u8, trail: u8) -> Self {
        Self {
            bytes: [lead, trail, 0, 0],
            len: 2,
        }
    }

    /// A four-byte code.
    const fn quad(bytes: [u8; MAX_CODE]) -> Self {
        Self { bytes, len: 4 }
    }
}

/// Process-lifetime reverse map for GB18030/GBK.
static GB18030_MAP: OnceLock<Vec<(char, Code)>> = OnceLock::new();

/// Process-lifetime reverse map for Big5.
static BIG5_MAP: OnceLock<Vec<(char, Code)>> = OnceLock::new();

/// Process-lifetime reverse map for Shift-JIS.
static SHIFT_JIS_MAP: OnceLock<Vec<(char, Code)>> = OnceLock::new();

/// The cached reverse map for `encoding`, built on first use.
fn reverse_map(encoding: CjkEncoding) -> &'static [(char, Code)] {
    match encoding {
        CjkEncoding::Gb18030 => GB18030_MAP.get_or_init(|| build_map(CjkEncoding::Gb18030)),
        CjkEncoding::Big5 => BIG5_MAP.get_or_init(|| build_map(CjkEncoding::Big5)),
        CjkEncoding::ShiftJis => SHIFT_JIS_MAP.get_or_init(|| build_map(CjkEncoding::ShiftJis)),
    }
}

/// Builds the reverse map of `encoding` from its dense decode table.
///
/// The result is sorted by character and holds exactly one code per character: the
/// shortest code, ties broken by the smallest byte sequence. A position whose bytes do not
/// decode back to the same character is dropped, so the encoder can never emit bytes that
/// decode to something else.
fn build_map(encoding: CjkEncoding) -> Vec<(char, Code)> {
    let (rows, columns) = cjk::two_byte_shape(encoding);
    let mut entries: Vec<(char, Code)> = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            let Some([lead, trail]) = cjk::two_byte_bytes(encoding, row, column) else {
                continue;
            };
            let Some(character) = cjk::decode_two_byte(encoding, lead, trail) else {
                continue;
            };
            entries.push((character, Code::pair(lead, trail)));
        }
    }
    if encoding == CjkEncoding::ShiftJis {
        for byte in cjk::HALFWIDTH_KATAKANA_BYTES {
            if let Some(character) = cjk::decode_halfwidth_katakana(byte) {
                entries.push((character, Code::single(byte)));
            }
        }
    }
    entries.sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.len.cmp(&right.1.len))
            .then_with(|| left.1.bytes.cmp(&right.1.bytes))
    });
    entries.dedup_by_key(|entry| entry.0);
    entries
}

/// Encodes `text` as ISO-8859-1/Latin-1: every `U+0000..=U+00FF` character is one byte.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) naming
/// the first character outside Latin-1.
pub(crate) fn encode_latin1(text: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for character in text.chars() {
        match u8::try_from(u32::from(character)) {
            Ok(byte) => out.push(byte),
            Err(_) => return Err(unrepresentable(character, TextEncoding::Latin1)),
        }
    }
    Ok(out)
}

/// Encodes `text` with a legacy CJK encoding.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) naming
/// the encoding and the first character it cannot represent.
pub(crate) fn encode_cjk(text: &str, encoding: CjkEncoding) -> Result<Vec<u8>> {
    let map = reverse_map(encoding);
    let mut out = Vec::with_capacity(text.len());
    for character in text.chars() {
        match code_for(character, encoding, map) {
            Some(code) => out.extend_from_slice(code.as_slice()),
            None => return Err(unrepresentable(character, encoding.text_encoding())),
        }
    }
    Ok(out)
}

/// Encodes one character, returning `None` when the encoding cannot represent it.
fn code_for(character: char, encoding: CjkEncoding, map: &[(char, Code)]) -> Option<Code> {
    if character.is_ascii() {
        // ASCII always maps to the same single byte in all three codecs.
        return u8::try_from(u32::from(character)).ok().map(Code::single);
    }
    if encoding == CjkEncoding::ShiftJis {
        if let Some(byte) = cjk::halfwidth_katakana_byte(character) {
            return Some(Code::single(byte));
        }
    }
    if let Ok(position) = map.binary_search_by(|(candidate, _)| candidate.cmp(&character)) {
        return Some(map.get(position)?.1);
    }
    if encoding == CjkEncoding::Gb18030 {
        return four_byte_code(character);
    }
    None
}

/// Encodes a character in the GB18030 four-byte form, verifying the bytes decode back.
fn four_byte_code(character: char) -> Option<Code> {
    let index = cjk::four_byte_index(character)?;
    let bytes = cjk::four_byte_bytes(index)?;
    if cjk::decode_four_byte(bytes) != Some(character) {
        return None;
    }
    Some(Code::quad(bytes))
}

/// The shared "content cannot be written in this encoding" failure.
///
/// The kind is `Validation`, not `Unsupported`: the encoding exists and writing with it
/// works, the *content* holds a character it cannot represent, and the caller can fix the
/// content (unlike an absent codec).
fn unrepresentable(character: char, encoding: TextEncoding) -> Error {
    Error::validation(format!(
        "character U+{:04X} ('{}') cannot be represented in encoding '{}'",
        u32::from(character),
        character.escape_debug(),
        encoding.as_str()
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// Looks a character up in a freshly built reverse map.
    fn lookup(encoding: CjkEncoding, character: char) -> Option<Code> {
        let map = build_map(encoding);
        let position = map
            .binary_search_by(|(candidate, _)| candidate.cmp(&character))
            .ok()?;
        map.get(position).map(|entry| entry.1)
    }

    #[test]
    fn reverse_maps_are_sorted_and_unique() {
        for encoding in [
            CjkEncoding::Gb18030,
            CjkEncoding::Big5,
            CjkEncoding::ShiftJis,
        ] {
            let map = build_map(encoding);
            assert!(!map.is_empty(), "{encoding}");
            let unique = map.windows(2).all(|pair| {
                pair.first()
                    .is_some_and(|left| pair.get(1).is_some_and(|right| left.0 < right.0))
            });
            assert!(unique, "{encoding} entries must be sorted and unique");
        }
    }

    #[test]
    fn every_reverse_entry_decodes_back_to_its_character() {
        for encoding in [
            CjkEncoding::Gb18030,
            CjkEncoding::Big5,
            CjkEncoding::ShiftJis,
        ] {
            for (character, code) in build_map(encoding) {
                let bytes = code.as_slice();
                let decoded = match bytes {
                    [byte] => cjk::decode(&[*byte], encoding),
                    [lead, trail] => cjk::decode(&[*lead, *trail], encoding),
                    _ => unreachable!("the reverse map only holds one- and two-byte codes"),
                };
                assert_eq!(decoded, character.to_string(), "{encoding}");
            }
        }
    }

    #[test]
    fn duplicate_codes_resolve_to_the_smallest_one() {
        // Big5 maps `十` twice (`0xA2CC` and `0xA451`); the smallest code wins, and both
        // positions decode to the same character.
        let map = build_map(CjkEncoding::Big5);
        let position = map
            .binary_search_by(|(candidate, _)| candidate.cmp(&'十'))
            .unwrap();
        assert_eq!(map[position].1.as_slice(), &[0xA2, 0xCC]);
        assert_eq!(cjk::decode(&[0xA2, 0xCC], CjkEncoding::Big5), "十");
        assert_eq!(cjk::decode(&[0xA4, 0x51], CjkEncoding::Big5), "十");
    }

    #[test]
    fn ascii_and_katakana_use_single_bytes() {
        assert_eq!(
            lookup(CjkEncoding::Gb18030, 'A'),
            None,
            "ASCII is not mapped"
        );
        assert_eq!(
            code_for('A', CjkEncoding::Gb18030, &[]),
            Some(Code::single(0x41))
        );
        assert_eq!(
            code_for('\u{FF61}', CjkEncoding::ShiftJis, &[]),
            Some(Code::single(0xA1))
        );
        assert_eq!(
            code_for('\u{FF9F}', CjkEncoding::ShiftJis, &[]),
            Some(Code::single(0xDF))
        );
        assert_eq!(code_for('\u{FF61}', CjkEncoding::Big5, &[]), None);
    }

    #[test]
    fn gb18030_four_byte_form_covers_bmp_and_supplementary_planes() {
        // U+FFFD lives in the BMP four-byte region; U+1F600 in the linear supplementary one.
        assert_eq!(
            four_byte_code('\u{FFFD}').unwrap().as_slice(),
            &[0x84, 0x31, 0xA4, 0x37]
        );
        assert_eq!(
            four_byte_code('\u{1F600}').unwrap().as_slice(),
            &[0x94, 0x39, 0xFC, 0x36]
        );
        assert_eq!(
            four_byte_code('\u{10FFFF}').unwrap().as_slice(),
            &[0xE3, 0x32, 0x9A, 0x35]
        );
    }

    #[test]
    fn unrepresentable_characters_are_validation_errors() {
        let error = encode_latin1("é中").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("U+4E2D"), "{}", error.message());
        assert!(error.message().contains("latin-1"), "{}", error.message());

        let error = encode_cjk("中\u{1F600}", CjkEncoding::Big5).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(!error.message().contains('中'));
        assert!(error.message().contains("U+1F600"), "{}", error.message());
        assert!(error.message().contains("big5"), "{}", error.message());
    }

    #[test]
    fn cjk_encoders_round_trip_their_own_output() {
        for (text, encoding) in [
            ("中文", CjkEncoding::Gb18030),
            ("中文", CjkEncoding::Big5),
            ("日本語", CjkEncoding::ShiftJis),
            ("mixed ascii 中文", CjkEncoding::Gb18030),
            ("\u{FF61}\u{FF71}", CjkEncoding::ShiftJis),
            ("\u{1F600}", CjkEncoding::Gb18030),
        ] {
            let bytes = encode_cjk(text, encoding).unwrap();
            assert_eq!(cjk::decode(&bytes, encoding), text, "{encoding}");
        }
    }

    #[test]
    fn latin1_writes_single_bytes() {
        assert_eq!(encode_latin1("é").unwrap(), vec![0xE9]);
        assert_eq!(encode_latin1("café").unwrap(), b"caf\xE9".to_vec());
        assert_eq!(encode_latin1("").unwrap(), Vec::<u8>::new());
        assert_eq!(encode_latin1("\u{FF}").unwrap(), vec![0xFF]);
    }

    #[test]
    fn lazy_maps_are_reused_across_threads() {
        // The same map instance must come back on every thread (one build per process).
        let handles: Vec<_> = (0..4)
            .map(|_| std::thread::spawn(|| reverse_map(CjkEncoding::Gb18030).as_ptr() as usize))
            .collect();
        let pointers: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let reused = pointers.iter().all(|pointer| *pointer == pointers[0]);
        assert!(reused, "the reverse map must be built once per process");
        assert_eq!(reverse_map(CjkEncoding::Gb18030).len(), 23940);
    }
}
