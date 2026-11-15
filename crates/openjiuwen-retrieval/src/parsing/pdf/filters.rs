//! `PDF` stream filters that can be decoded without a compression crate.
//!
//! A stream dictionary names one filter or a chain of them (`/Filter [/ASCII85Decode
//! /FlateDecode]`, decoded left to right). This module owns the filter identity
//! ([`Filter`]) and the four decoders the crate can honour — `ASCIIHexDecode`,
//! `ASCII85Decode`, `RunLengthDecode`, and `FlateDecode` (the last one delegating
//! to [`super::super::inflate::zlib_decompress`]).
//!
//! The decoders follow the filter definitions in the `PDF` specification:
//! whitespace is ignored, the terminators (`>`, `~>`, end-of-data byte `128`) are
//! honoured, and every buffer is bounded by the caller's `max_output` so a crafted
//! stream cannot exhaust memory. LZW (`/LZWDecode`) and encryption (`/Crypt`) are
//! reported as [`Error::Unsupported`] — the caller treats them as fatal — while
//! image payload filters (CCITT fax, `DCT`, `JPX`, `JBIG2`) and unknown names are
//! also typed errors here, so nothing is ever decoded incorrectly.
//!
//! - [`Error::Validation`] — malformed filter data (a bad digit, an illegal ASCII85
//!   group, a truncated run, or an output buffer over `max_output`).
//! - [`Error::Unsupported`] — a filter this crate cannot decode.

use openjiuwen_common::error::{Error, Result};

use super::super::inflate::{zlib_decompress, DEFAULT_MAX_OUTPUT_BYTES};
use super::{hex_value, is_whitespace};

/// One stream filter, resolved from its dictionary name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Filter {
    /// `/ASCIIHexDecode` (`/AHx`): two hex digits per byte.
    AsciiHex,
    /// `/ASCII85Decode` (`/A85`): base-85 groups and the `z` shortcut.
    Ascii85,
    /// `/LZWDecode` (`/LZW`): deferred, text extraction cannot continue past it.
    Lzw,
    /// `/FlateDecode` (`/Fl`): `zlib`-wrapped `DEFLATE`.
    Flate,
    /// `/RunLengthDecode` (`/RL`): literal and repeat runs.
    RunLength,
    /// `/CCITTFaxDecode` (`/CCF`): bilevel image data.
    CcittFax,
    /// `/DCTDecode` (`/DCT`): JPEG image data.
    Dct,
    /// `/JPXDecode`: JPEG 2000 image data.
    Jpx,
    /// `/JBIG2Decode`: JBIG2 image data.
    Jbig2,
    /// `/Crypt`: encrypted payload.
    Crypt,
    /// A name this crate does not know.
    Unknown(String),
}

impl Filter {
    /// Resolves a filter name, accepting both the long and the abbreviation form.
    pub(crate) fn from_name(name: &str) -> Self {
        match name {
            "ASCIIHexDecode" | "AHx" => Self::AsciiHex,
            "ASCII85Decode" | "A85" => Self::Ascii85,
            "LZWDecode" | "LZW" => Self::Lzw,
            "FlateDecode" | "Fl" => Self::Flate,
            "RunLengthDecode" | "RL" => Self::RunLength,
            "CCITTFaxDecode" | "CCF" => Self::CcittFax,
            "DCTDecode" | "DCT" => Self::Dct,
            "JPXDecode" => Self::Jpx,
            "JBIG2Decode" => Self::Jbig2,
            "Crypt" => Self::Crypt,
            other => Self::Unknown(other.to_owned()),
        }
    }

    /// Returns the display name of the filter (`"ASCIIHexDecode"`, …).
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::AsciiHex => "ASCIIHexDecode",
            Self::Ascii85 => "ASCII85Decode",
            Self::Lzw => "LZWDecode",
            Self::Flate => "FlateDecode",
            Self::RunLength => "RunLengthDecode",
            Self::CcittFax => "CCITTFaxDecode",
            Self::Dct => "DCTDecode",
            Self::Jpx => "JPXDecode",
            Self::Jbig2 => "JBIG2Decode",
            Self::Crypt => "Crypt",
            Self::Unknown(name) => name.as_str(),
        }
    }

    /// Returns `true` for filters that only carry image payloads.
    ///
    /// Such a stream never holds the text operators a content stream has, so the
    /// caller skips it instead of feeding it to the lexer.
    pub(crate) fn is_image_payload(&self) -> bool {
        matches!(self, Self::CcittFax | Self::Dct | Self::Jpx | Self::Jbig2)
    }

    /// Returns `true` for filters that make the whole document unreadable.
    pub(crate) fn aborts_extraction(&self) -> bool {
        matches!(self, Self::Lzw | Self::Crypt)
    }
}

/// Decodes `data` with `filter`, bounded by `max_output` bytes.
///
/// # Errors
///
/// [`Error::Validation`] for malformed filter data and [`Error::Unsupported`] for
/// a filter this crate cannot decode.
pub(crate) fn decode(filter: &Filter, data: &[u8], max_output: usize) -> Result<Vec<u8>> {
    match filter {
        Filter::AsciiHex => ascii_hex_decode(data, max_output),
        Filter::Ascii85 => ascii85_decode(data, max_output),
        Filter::RunLength => run_length_decode(data, max_output),
        Filter::Flate => flate_decode(data, max_output),
        Filter::Unknown(name) => Err(Error::Unsupported(format!(
            "unknown PDF stream filter {name}"
        ))),
        other => Err(Error::Unsupported(format!(
            "PDF stream filter {} is not decoded by the text extractor",
            other.name()
        ))),
    }
}

/// Decodes an `/ASCIIHexDecode` payload (hex pairs, `>` terminator).
///
/// # Errors
///
/// [`Error::Validation`] for a non-hex byte or an output buffer over `max_output`.
pub(crate) fn ascii_hex_decode(data: &[u8], max_output: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut high: Option<u8> = None;
    for &byte in data {
        if byte == b'>' {
            break;
        }
        if is_whitespace(byte) {
            continue;
        }
        let Some(value) = hex_value(byte) else {
            return Err(Error::Validation(format!(
                "invalid ASCIIHex byte 0x{byte:02X}"
            )));
        };
        match high.take() {
            None => high = Some(value),
            Some(upper) => {
                check_capacity(out.len(), 1, max_output)?;
                out.push((upper << 4) | value);
            }
        }
    }
    // An odd trailing nibble is completed with a low zero nibble.
    if let Some(upper) = high {
        check_capacity(out.len(), 1, max_output)?;
        out.push(upper << 4);
    }
    Ok(out)
}

/// Decodes an `/ASCII85Decode` payload (`!`..`u` digits, `z`, `~>` terminator).
///
/// # Errors
///
/// [`Error::Validation`] for an illegal byte, an overflowing group, a `z` inside a
/// group, a one-character final group, or an output buffer over `max_output`.
pub(crate) fn ascii85_decode(data: &[u8], max_output: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut group = [0u8; 5];
    let mut count = 0usize;
    for &byte in data {
        match byte {
            // The terminator ends the payload; anything after it is padding.
            b'~' => break,
            b'z' if count == 0 => {
                check_capacity(out.len(), 4, max_output)?;
                out.extend_from_slice(&[0, 0, 0, 0]);
            }
            b'z' => {
                return Err(Error::Validation(
                    "the ASCII85 `z` shortcut is only legal between groups".to_owned(),
                ));
            }
            b'!'..=b'u' => {
                group[count] = byte - b'!';
                count += 1;
                if count == 5 {
                    let bytes = ascii85_group(&group)?;
                    check_capacity(out.len(), 4, max_output)?;
                    out.extend_from_slice(&bytes);
                    count = 0;
                }
            }
            other if is_whitespace(other) => {}
            other => {
                return Err(Error::Validation(format!(
                    "invalid ASCII85 byte 0x{other:02X}"
                )));
            }
        }
    }
    match count {
        0 => {}
        1 => {
            return Err(Error::Validation(
                "an ASCII85 final group needs at least two characters".to_owned(),
            ));
        }
        partial => {
            for slot in group.iter_mut().skip(partial) {
                *slot = 84;
            }
            let bytes = ascii85_group(&group)?;
            let kept = partial - 1;
            check_capacity(out.len(), kept, max_output)?;
            out.extend_from_slice(&bytes[..kept]);
        }
    }
    Ok(out)
}

/// Packs one five-digit ASCII85 group into four big-endian bytes.
fn ascii85_group(group: &[u8; 5]) -> Result<[u8; 4]> {
    let mut value: u32 = 0;
    for &digit in group {
        value = value
            .checked_mul(85)
            .and_then(|value| value.checked_add(u32::from(digit)))
            .ok_or_else(|| Error::Validation("ASCII85 group overflows a 32-bit word".to_owned()))?;
    }
    Ok(value.to_be_bytes())
}

/// Decodes a `/RunLengthDecode` payload (literal runs, repeat runs, `128`).
///
/// # Errors
///
/// [`Error::Validation`] for a truncated run or an output buffer over `max_output`.
pub(crate) fn run_length_decode(data: &[u8], max_output: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < data.len() {
        let length = data[cursor];
        cursor += 1;
        match length {
            // End-of-data marker; a payload without one is tolerated.
            128 => return Ok(out),
            0..=127 => {
                let count = usize::from(length) + 1;
                let Some(chunk) = data.get(cursor..cursor + count) else {
                    return Err(Error::Validation(
                        "truncated RunLength literal run".to_owned(),
                    ));
                };
                check_capacity(out.len(), count, max_output)?;
                out.extend_from_slice(chunk);
                cursor += count;
            }
            _ => {
                let count = 257 - usize::from(length);
                let Some(&byte) = data.get(cursor) else {
                    return Err(Error::Validation(
                        "truncated RunLength repeat run".to_owned(),
                    ));
                };
                check_capacity(out.len(), count, max_output)?;
                out.extend(std::iter::repeat_n(byte, count));
                cursor += 1;
            }
        }
    }
    Ok(out)
}

/// Decodes a `/FlateDecode` payload through the crate's `zlib` reader.
///
/// # Errors
///
/// Every [`zlib_decompress`] error: [`Error::Retrieval`] for a truncated or
/// corrupt stream, [`Error::Unsupported`] for a preset dictionary.
pub(crate) fn flate_decode(data: &[u8], max_output: usize) -> Result<Vec<u8>> {
    zlib_decompress(data, max_output.min(DEFAULT_MAX_OUTPUT_BYTES))
}

/// Rejects an output buffer that would grow past `max_output`.
fn check_capacity(current: usize, additional: usize, max_output: usize) -> Result<()> {
    if current.saturating_add(additional) > max_output {
        return Err(Error::Validation(format!(
            "decoded PDF stream exceeds the {max_output}-byte output limit"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    #[test]
    fn names_resolve_both_forms_and_report_their_display_name() {
        for (name, filter, display) in [
            ("ASCIIHexDecode", Filter::AsciiHex, "ASCIIHexDecode"),
            ("AHx", Filter::AsciiHex, "ASCIIHexDecode"),
            ("ASCII85Decode", Filter::Ascii85, "ASCII85Decode"),
            ("A85", Filter::Ascii85, "ASCII85Decode"),
            ("LZWDecode", Filter::Lzw, "LZWDecode"),
            ("LZW", Filter::Lzw, "LZWDecode"),
            ("FlateDecode", Filter::Flate, "FlateDecode"),
            ("Fl", Filter::Flate, "FlateDecode"),
            ("RunLengthDecode", Filter::RunLength, "RunLengthDecode"),
            ("RL", Filter::RunLength, "RunLengthDecode"),
            ("CCITTFaxDecode", Filter::CcittFax, "CCITTFaxDecode"),
            ("CCF", Filter::CcittFax, "CCITTFaxDecode"),
            ("DCTDecode", Filter::Dct, "DCTDecode"),
            ("DCT", Filter::Dct, "DCTDecode"),
            ("JPXDecode", Filter::Jpx, "JPXDecode"),
            ("JBIG2Decode", Filter::Jbig2, "JBIG2Decode"),
            ("Crypt", Filter::Crypt, "Crypt"),
        ] {
            assert_eq!(Filter::from_name(name), filter, "{name}");
            assert_eq!(filter.name(), display, "{name}");
        }
        assert_eq!(
            Filter::from_name("FooDecode"),
            Filter::Unknown("FooDecode".to_owned())
        );
        assert_eq!(Filter::from_name("FooDecode").name(), "FooDecode");
    }

    #[test]
    fn image_payload_and_aborting_filters_are_flagged() {
        for filter in [Filter::CcittFax, Filter::Dct, Filter::Jpx, Filter::Jbig2] {
            assert!(filter.is_image_payload(), "{}", filter.name());
            assert!(!filter.aborts_extraction(), "{}", filter.name());
        }
        for filter in [Filter::Lzw, Filter::Crypt] {
            assert!(filter.aborts_extraction(), "{}", filter.name());
            assert!(!filter.is_image_payload(), "{}", filter.name());
        }
        for filter in [
            Filter::AsciiHex,
            Filter::Ascii85,
            Filter::Flate,
            Filter::RunLength,
        ] {
            assert!(!filter.is_image_payload());
            assert!(!filter.aborts_extraction());
        }
    }

    #[test]
    fn ascii_hex_ignores_whitespace_and_pads_an_odd_nibble() {
        assert_eq!(ascii_hex_decode(b"48656C6C6F>", 64).unwrap(), b"Hello");
        assert_eq!(
            ascii_hex_decode(b"48 65\n6C\t6C 6F>", 64).unwrap(),
            b"Hello"
        );
        // An odd final nibble is completed with a low zero nibble.
        assert_eq!(ascii_hex_decode(b"41424>", 64).unwrap(), b"AB@");
        // The terminator is optional at the end of the payload.
        assert_eq!(ascii_hex_decode(b"4142", 64).unwrap(), b"AB");
        assert_eq!(ascii_hex_decode(b"", 64).unwrap(), Vec::<u8>::new());
        assert_eq!(ascii_hex_decode(b">", 64).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn ascii_hex_rejects_bad_digits_and_oversized_output() {
        let error = ascii_hex_decode(b"41ZZ", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("0x5A"), "{error}");
        let error = ascii_hex_decode(b"414243", 2).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn ascii85_decodes_full_partial_and_zero_groups() {
        assert_eq!(ascii85_decode(b"87cURD]o~>", 64).unwrap(), b"Hello!");
        assert_eq!(ascii85_decode(b"88/~>", 64).unwrap(), b"Hi");
        // `z` writes four zero bytes.
        assert_eq!(ascii85_decode(b"z~>", 64).unwrap(), vec![0, 0, 0, 0]);
        assert_eq!(
            ascii85_decode(b"@:E_WzAS(~>", 64).unwrap(),
            b"abcd\x00\x00\x00\x00ef"
        );
        // A final group of two characters yields one byte.
        assert_eq!(ascii85_decode(b"!!~>", 64).unwrap(), vec![0]);
        // The terminator is optional at the end of the payload.
        assert_eq!(ascii85_decode(b"87cURD]o", 64).unwrap(), b"Hello!");
        assert_eq!(ascii85_decode(b"", 64).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn ascii85_rejects_illegal_groups_and_bytes() {
        let error = ascii85_decode(b"z!z~>", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("between groups"), "{error}");

        let error = ascii85_decode(b"!~>", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("at least two"), "{error}");

        let error = ascii85_decode(b"uuuuu~>", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("overflows"), "{error}");

        let error = ascii85_decode(b"87cv~>", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("0x76"), "{error}");

        let error = ascii85_decode(b"zzzz~>", 8).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn run_length_decodes_literals_repeats_and_the_end_marker() {
        // A three-byte literal run, then two copies of `!`, then end of data.
        assert_eq!(
            run_length_decode(&[2, b'a', b'b', b'c', 255, b'!', 128], 64).unwrap(),
            b"abc!!"
        );
        // 129 repeats the next byte 128 times.
        let mut payload = vec![129u8, b'x'];
        payload.push(128);
        assert_eq!(run_length_decode(&payload, 200).unwrap(), vec![b'x'; 128]);
        // A missing end marker is tolerated.
        assert_eq!(run_length_decode(&[0, b'a'], 64).unwrap(), b"a");
        assert_eq!(run_length_decode(&[], 64).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn run_length_rejects_truncated_runs_and_oversized_output() {
        let error = run_length_decode(&[3, b'a'], 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("literal run"), "{error}");

        let error = run_length_decode(&[254], 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("repeat run"), "{error}");

        let error = run_length_decode(&[255, b'x', 128], 1).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn flate_decodes_a_stored_zlib_stream() {
        // `0x78 0x01` header, one final stored block holding "Hi", Adler-32.
        let mut stream = vec![0x78, 0x01, 0x01, 0x02, 0x00, 0xFD, 0xFF, b'H', b'i'];
        stream.extend_from_slice(&super::super::super::inflate::adler32(b"Hi").to_be_bytes());
        assert_eq!(flate_decode(&stream, 64).unwrap(), b"Hi");
        // A non-zlib payload is rejected by the header check.
        let error = flate_decode(b"not a zlib stream", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = flate_decode(&stream[..stream.len() - 2], 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Retrieval);
    }

    #[test]
    fn decode_dispatches_and_rejects_undecodable_filters() {
        assert_eq!(decode(&Filter::AsciiHex, b"4869>", 64).unwrap(), b"Hi");
        assert_eq!(
            decode(&Filter::RunLength, &[1, b'h', b'i', 128], 64).unwrap(),
            b"hi"
        );
        let error = decode(&Filter::Lzw, b"", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("LZWDecode"), "{error}");
        let error = decode(&Filter::Dct, b"", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("DCTDecode"), "{error}");
        let error = decode(&Filter::Unknown("FooDecode".to_owned()), b"", 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("FooDecode"), "{error}");
    }
}
