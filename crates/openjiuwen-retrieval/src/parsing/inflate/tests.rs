//! Unit tests for the `DEFLATE`/`zlib` decoder.
//!
//! Includes a small in-test `DEFLATE` writer (stored, fixed-Huffman, and
//! dynamic-Huffman blocks) so round trips and error paths are checked against
//! streams this crate builds itself, plus canned streams from a reference
//! encoder.

use super::*;

/// `zlib` stream for "hello" produced by a reference encoder.
const ZLIB_HELLO: [u8; 13] = [
    0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00, 0x06, 0x2c, 0x02, 0x15,
];

/// Least-significant-bit-first writer used to build test streams.
#[derive(Default)]
struct BitWriter {
    /// Completed (and partially filled) output bytes.
    bytes: Vec<u8>,
    /// Bits already used in the last byte.
    bit_count: u32,
}

impl BitWriter {
    /// Starts an empty stream.
    fn new() -> Self {
        Self::default()
    }

    /// Appends a single bit, starting a new byte when needed.
    fn write_bit(&mut self, bit: u32) {
        if self.bit_count == 0 {
            self.bytes.push(0);
        }
        if bit & 1 == 1 {
            let last = self.bytes.len() - 1;
            self.bytes[last] |= 1 << self.bit_count;
        }
        self.bit_count = (self.bit_count + 1) % 8;
    }

    /// Appends `count` bits of `value`, least significant bit first.
    fn write_bits(&mut self, value: u32, count: u32) {
        for position in 0..count {
            self.write_bit((value >> position) & 1);
        }
    }

    /// Appends a Huffman code, most significant bit first (RFC 1951 §3.1.1).
    fn write_code(&mut self, code: u32, length: u8) {
        for position in (0..length).rev() {
            self.write_bit((code >> position) & 1);
        }
    }

    /// Pads the current byte with zeros.
    fn align_to_byte(&mut self) {
        self.bit_count = 0;
    }

    /// Appends a complete stored block.
    fn write_stored_block(&mut self, payload: &[u8], final_block: bool) {
        self.write_bit(u32::from(final_block));
        self.write_bits(0, 2);
        self.align_to_byte();
        let len = payload.len() as u32;
        self.write_bits(len, 16);
        self.write_bits(!len & 0xFFFF, 16);
        for &byte in payload {
            self.write_bits(u32::from(byte), 8);
        }
    }
}

/// Returns the fixed-Huffman code (value and bit length) for `symbol`.
fn fixed_code(symbol: u32) -> (u32, u8) {
    match symbol {
        0..=143 => (0x30 + symbol, 8),
        144..=255 => (0x190 + (symbol - 144), 9),
        256..=279 => (symbol - 256, 7),
        _ => (0xC0 + (symbol - 280), 8),
    }
}

/// Builds a literal-only fixed-Huffman stream (no back-references).
fn fixed_literal_stream(payload: &[u8]) -> Vec<u8> {
    let mut writer = BitWriter::new();
    writer.write_bit(1);
    writer.write_bits(1, 2);
    for &byte in payload {
        let (code, length) = fixed_code(u32::from(byte));
        writer.write_code(code, length);
    }
    let (end_code, end_length) = fixed_code(256);
    writer.write_code(end_code, end_length);
    writer.bytes
}

/// Assigns a complete canonical Huffman code to `count` equally likely symbols.
fn equal_lengths(count: usize) -> Vec<u8> {
    assert!(count >= 1, "at least one symbol is required");
    if count == 1 {
        return vec![1];
    }
    let depth = (usize::BITS - (count - 1).leading_zeros()) as usize;
    let total = 1usize << depth;
    let deep = 2 * count - total;
    let shallow = count - deep;
    let mut lengths = vec![(depth - 1) as u8; shallow];
    lengths.extend(std::iter::repeat_n(depth as u8, deep));
    lengths
}

/// Derives canonical codes (RFC 1951 §3.2.2) from per-symbol code lengths.
fn canonical_codes(lengths: &[u8]) -> Vec<(u32, u8)> {
    let mut codes = vec![(0u32, 0u8); lengths.len()];
    let mut counts = [0u16; 16];
    for &length in lengths {
        counts[usize::from(length)] += 1;
    }
    counts[0] = 0;
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..=15 {
        code = (code + u32::from(counts[bits - 1])) << 1;
        next_code[bits] = code;
    }
    for (symbol, &length) in lengths.iter().enumerate() {
        if length != 0 {
            codes[symbol] = (next_code[usize::from(length)], length);
            next_code[usize::from(length)] += 1;
        }
    }
    codes
}

/// One code-length alphabet operation: symbol plus its extra bits.
type LengthOp = (u16, u32, u32);

/// Run-length encodes a code-length array with the 16/17/18 repeat codes.
fn encode_lengths(lengths: &[u8], use_repeats: bool) -> Vec<LengthOp> {
    let mut ops = Vec::new();
    let mut index = 0;
    while index < lengths.len() {
        let value = lengths[index];
        let mut run = 1;
        while index + run < lengths.len() && lengths[index + run] == value {
            run += 1;
        }
        let mut remaining = run;
        if value == 0 && use_repeats {
            while remaining >= 11 {
                let take = remaining.min(138);
                ops.push((18, (take - 11) as u32, 7));
                remaining -= take;
            }
            while remaining >= 3 {
                let take = remaining.min(10);
                ops.push((17, (take - 3) as u32, 3));
                remaining -= take;
            }
        } else if value != 0 && use_repeats && index > 0 {
            ops.push((u16::from(value), 0, 0));
            remaining -= 1;
            while remaining >= 3 {
                let take = remaining.min(6);
                ops.push((16, (take - 3) as u32, 2));
                remaining -= take;
            }
        }
        for _ in 0..remaining {
            ops.push((u16::from(value), 0, 0));
        }
        index += run;
    }
    ops
}

/// Builds a dynamic-Huffman block for `payload`.
///
/// The literal alphabet spans all 286 symbols (HLIT=286) and the distance
/// alphabet holds a single unused code of length 1 (`HDIST=1`), which RFC
/// 1951 explicitly allows.
fn dynamic_stream(payload: &[u8], use_repeats: bool) -> Vec<u8> {
    let mut used: Vec<u16> = payload.iter().map(|&byte| u16::from(byte)).collect();
    used.push(256);
    used.sort_unstable();
    used.dedup();

    let mut literal_lengths = vec![0u8; 286];
    for (&symbol, &length) in used.iter().zip(equal_lengths(used.len()).iter()) {
        literal_lengths[usize::from(symbol)] = length;
    }
    let mut all_lengths = literal_lengths.clone();
    all_lengths.push(1);

    let ops = encode_lengths(&all_lengths, use_repeats);
    let mut used_length_codes: Vec<u16> = ops.iter().map(|op| op.0).collect();
    used_length_codes.sort_unstable();
    used_length_codes.dedup();
    let mut code_length_lengths = [0u8; 19];
    for (&symbol, &length) in used_length_codes
        .iter()
        .zip(equal_lengths(used_length_codes.len()).iter())
    {
        code_length_lengths[usize::from(symbol)] = length;
    }
    let mut code_length_count = 4;
    for (position, &symbol) in CODE_LENGTH_ORDER.iter().enumerate() {
        if code_length_lengths[symbol] != 0 {
            code_length_count = code_length_count.max(position + 1);
        }
    }

    let mut writer = BitWriter::new();
    writer.write_bit(1);
    writer.write_bits(2, 2);
    writer.write_bits(286 - 257, 5);
    writer.write_bits(0, 5);
    writer.write_bits(code_length_count as u32 - 4, 4);
    for &symbol in CODE_LENGTH_ORDER.iter().take(code_length_count) {
        writer.write_bits(u32::from(code_length_lengths[symbol]), 3);
    }
    let code_length_codes = canonical_codes(&code_length_lengths);
    for (symbol, extra, extra_bits) in &ops {
        let (code, length) = code_length_codes[usize::from(*symbol)];
        writer.write_code(code, length);
        writer.write_bits(*extra, *extra_bits);
    }
    let literal_codes = canonical_codes(&literal_lengths);
    for &byte in payload {
        let (code, length) = literal_codes[usize::from(byte)];
        writer.write_code(code, length);
    }
    let (end_code, end_length) = literal_codes[256];
    writer.write_code(end_code, end_length);
    writer.bytes
}

#[test]
fn crc32_matches_known_vectors() {
    assert_eq!(crc32(b""), 0);
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b"hello"), 0x3610_A686);
    assert_eq!(
        crc32(b"The quick brown fox jumps over the lazy dog"),
        0x414F_A339
    );
    // The ZIP container uses the same CRC-32 as the PNG/zip format.
    assert_eq!(crc32(&[0u8; 32]), 0x190A_55AD);
}

#[test]
fn adler32_matches_known_vectors() {
    assert_eq!(adler32(b""), 1);
    assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    assert_eq!(adler32(b"hello"), 0x062C_0215);
    // Long inputs span several accumulation blocks; compare with a naive reference.
    let long = vec![b'a'; 20_000];
    let mut low = 1u32;
    let mut high = 0u32;
    for &byte in &long {
        low = (low + u32::from(byte)) % 65_521;
        high = (high + low) % 65_521;
    }
    assert_eq!(adler32(&long), (high << 16) | low);
}

#[test]
fn decodes_canned_zlib_stream_for_hello() {
    assert_eq!(
        zlib_decompress(&ZLIB_HELLO, DEFAULT_MAX_OUTPUT_BYTES).unwrap(),
        b"hello"
    );
    assert_eq!(adler32(b"hello").to_be_bytes().as_slice(), &ZLIB_HELLO[9..]);
}

#[test]
fn decodes_empty_stored_block() {
    // BFINAL=1, BTYPE=00, LEN=0, NLEN=0xffff.
    let stream = [0x01, 0x00, 0x00, 0xFF, 0xFF];
    assert_eq!(inflate(&stream, 16).unwrap(), Vec::<u8>::new());
}

#[test]
fn decodes_multiple_stored_blocks() {
    let mut writer = BitWriter::new();
    writer.write_stored_block(b"first", false);
    writer.write_stored_block(b"-second", true);
    assert_eq!(inflate(&writer.bytes, 64).unwrap(), b"first-second");
}

#[test]
fn decodes_fixed_huffman_literals() {
    // BFINAL=1, BTYPE=01, then the 8-bit fixed code for 'A' (0x30 + 65 = 0x71,
    // transmitted 0-1-1-1-0-0-0-1), then the 7-bit end-of-block code (all zeros).
    // Packed LSB-first that is 0x73, then the remaining bits 0-0-1-0-0-0-0-0 = 0x04,
    // then a zero pad byte.
    let stream = [0x73, 0x04, 0x00];
    assert_eq!(inflate(&stream, 16).unwrap(), b"A");
}

#[test]
fn round_trips_fixed_huffman_streams() {
    let payloads: [Vec<u8>; 4] = [
        Vec::new(),
        b"A".to_vec(),
        (0..=255u8).collect(),
        b"the quick brown fox jumps over the lazy dog".repeat(4),
    ];
    for payload in payloads {
        let stream = fixed_literal_stream(&payload);
        assert_eq!(
            inflate(&stream, DEFAULT_MAX_OUTPUT_BYTES).unwrap(),
            payload,
            "payload length {}",
            payload.len()
        );
    }
}

#[test]
fn round_trips_dynamic_huffman_streams() {
    let payloads: [Vec<u8>; 5] = [
        Vec::new(),
        b"hello".to_vec(),
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec(),
        (0..=255u8).collect(),
        b"dynamic huffman blocks repeat code lengths ".repeat(8),
    ];
    for payload in payloads {
        for use_repeats in [false, true] {
            let stream = dynamic_stream(&payload, use_repeats);
            assert_eq!(
                inflate(&stream, DEFAULT_MAX_OUTPUT_BYTES).unwrap(),
                payload,
                "payload length {} (repeats: {use_repeats})",
                payload.len()
            );
        }
    }
}

#[test]
fn inflates_back_references_produced_by_a_real_encoder() {
    // Raw DEFLATE payload of the canned "hello" zlib stream.
    assert_eq!(inflate(&ZLIB_HELLO[2..9], 16).unwrap(), b"hello");
}

#[test]
fn decodes_stored_payloads_through_zlib_framing() {
    let payload = b"stored block payload".repeat(3);
    let mut writer = BitWriter::new();
    // 0x78 0x01 is the fastest-level zlib header (FCHECK valid for 0x7801).
    writer.write_bits(0x78, 8);
    writer.write_bits(0x01, 8);
    writer.write_stored_block(&payload, true);
    let checksum = adler32(&payload).to_be_bytes();
    for &byte in &checksum {
        writer.write_bits(u32::from(byte), 8);
    }
    assert_eq!(
        zlib_decompress(&writer.bytes, DEFAULT_MAX_OUTPUT_BYTES).unwrap(),
        payload
    );
}

#[test]
fn rejects_truncated_streams() {
    // A final stored block header with no LEN/NLEN.
    let error = inflate(&[0x01], 16).unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Retrieval);
    assert!(error.to_string().contains("truncated"), "{error}");

    // A fixed-Huffman block that ends before the end-of-block symbol.
    let error = inflate(&[0x03], 16).unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Retrieval);

    // A stored block that announces more bytes than it carries.
    let error = inflate(&[0x01, 0x04, 0x00, 0xFB, 0xFF, b'h', b'i'], 16).unwrap_err();
    assert!(error.to_string().contains("truncated"), "{error}");
}

#[test]
fn rejects_invalid_block_type() {
    // BFINAL=1, BTYPE=11.
    let error = inflate(&[0x07], 16).unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(
        error.to_string().contains("invalid DEFLATE block type 3"),
        "{error}"
    );
}

#[test]
fn rejects_inconsistent_stored_block_lengths() {
    let stream = [0x01, 0x03, 0x00, 0x00, 0x00, b'a', b'b', b'c'];
    let error = inflate(&stream, 16).unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("length mismatch"), "{error}");
}

#[test]
fn rejects_output_beyond_the_limit() {
    let mut writer = BitWriter::new();
    writer.write_stored_block(b"0123456789", true);
    let error = inflate(&writer.bytes, 5).unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("limit"), "{error}");

    let stream = fixed_literal_stream(b"abcd");
    let error = inflate(&stream, 3).unwrap_err();
    assert!(error.to_string().contains("limit"), "{error}");
}

#[test]
fn rejects_invalid_back_reference_distances() {
    // A length code (symbol 257, 3 bytes) with no history to copy from.
    let mut writer = BitWriter::new();
    writer.write_bit(1);
    writer.write_bits(1, 2);
    let (code, length) = fixed_code(257);
    writer.write_code(code, length);
    writer.write_code(0, 5); // distance symbol 0 => distance 1
    let error = inflate(&writer.bytes, 16).unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("distance 1"), "{error}");

    // Distance symbol 30 is not a legal distance code.
    let mut writer = BitWriter::new();
    writer.write_bit(1);
    writer.write_bits(1, 2);
    let (code, length) = fixed_code(257);
    writer.write_code(code, length);
    writer.write_code(30, 5);
    let error = inflate(&writer.bytes, 16).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("invalid DEFLATE distance symbol 30"),
        "{error}"
    );
}

#[test]
fn rejects_dynamic_blocks_without_an_end_of_block_code() {
    // Hand-built dynamic block whose literal alphabet has one code ('A') and no
    // end-of-block symbol. The code-length alphabet is {0, 1}, one bit each.
    let mut literal_lengths = vec![0u8; 257];
    literal_lengths[65] = 1;
    let mut all_lengths = literal_lengths;
    all_lengths.push(1); // single distance code, unused

    let mut writer = BitWriter::new();
    writer.write_bit(1);
    writer.write_bits(2, 2);
    writer.write_bits(257 - 257, 5);
    writer.write_bits(1 - 1, 5);
    writer.write_bits(18 - 4, 4); // symbols 0 and 1 sit at positions 3 and 17
    let mut code_length_lengths = [0u8; 19];
    code_length_lengths[0] = 1;
    code_length_lengths[1] = 1;
    for &symbol in CODE_LENGTH_ORDER.iter().take(18) {
        writer.write_bits(u32::from(code_length_lengths[symbol]), 3);
    }
    for &length in &all_lengths {
        writer.write_bit(u32::from(length));
    }
    let error = inflate(&writer.bytes, 16).unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(
        error.to_string().contains("no end-of-block code"),
        "{error}"
    );
}

#[test]
fn rejects_over_subscribed_and_out_of_range_code_lengths() {
    let error = Huffman::new(&[1, 1, 1]).unwrap_err();
    assert!(error.to_string().contains("over-subscribed"), "{error}");
    let error = Huffman::new(&[16]).unwrap_err();
    assert!(
        error.to_string().contains("invalid Huffman code length 16"),
        "{error}"
    );
    // An incomplete code is accepted; it only fails when a symbol is decoded.
    assert!(Huffman::new(&[1]).is_ok());
    assert!(Huffman::new(&[]).is_ok());
    let mut reader = BitReader::new(&[0x00, 0x00]);
    let error = Huffman::new(&[]).unwrap().decode(&mut reader).unwrap_err();
    assert!(
        error.to_string().contains("invalid Huffman code"),
        "{error}"
    );
}

#[test]
fn rejects_bad_zlib_headers() {
    // CM = 7 (not DEFLATE); FCHECK adjusted so only the method can be wrong.
    let error = zlib_decompress(&[0x77, 0x9C], 16).unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(
        error.to_string().contains("compression method 7"),
        "{error}"
    );

    // CINFO = 8 (window larger than 32 KiB).
    let error = zlib_decompress(&[0x88, 0x98], 16).unwrap_err();
    assert!(error.to_string().contains("CINFO=8"), "{error}");

    // Corrupted FCHECK bits.
    let error = zlib_decompress(&[0x78, 0x9D], 16).unwrap_err();
    assert!(error.to_string().contains("check bits"), "{error}");

    // Short header.
    let error = zlib_decompress(&[0x78], 16).unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Retrieval);
}

#[test]
fn rejects_preset_dictionaries() {
    // FDICT set: FLEVEL=2, FDICT=1, FCHECK chosen so CMF*256+FLG is a multiple of 31.
    let error = zlib_decompress(&[0x78, 0xBB], 16).unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Unsupported
    );
    assert!(error.to_string().contains("preset dictionaries"), "{error}");
}

#[test]
fn rejects_adler32_mismatches() {
    let mut stream = ZLIB_HELLO;
    let last = stream.len() - 1;
    stream[last] ^= 0xFF;
    let error = zlib_decompress(&stream, DEFAULT_MAX_OUTPUT_BYTES).unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Retrieval);
    assert!(error.to_string().contains("Adler-32 mismatch"), "{error}");
}

#[test]
fn rejects_zlib_trailers_that_are_too_short() {
    let error = zlib_decompress(&[0x78, 0x9C, 0x03], 16).unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Retrieval);
}

#[test]
fn rejects_zlib_inputs_that_cannot_hold_a_trailer_without_panicking() {
    // A 4- or 5-byte stream has a well-formed header, but `len - 4` lands before
    // the end of the header, so the payload slice `2..payload_end` used to be
    // built backwards and panic. Every short input must be a typed error instead.
    // 0x78 0x9C is a valid DEFLATE/32 KiB header; the trailing bytes are too few
    // to hold the 4-byte Adler-32.
    for stream in [
        vec![0x78, 0x9C, 0x03, 0x00],
        vec![0x78, 0x9C, 0x03, 0x00, 0x00],
        vec![0x78, 0x9C, 0x03, 0x00, 0x00, 0x00],
    ] {
        let error = zlib_decompress(&stream, DEFAULT_MAX_OUTPUT_BYTES).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Retrieval,
            "{stream:02x?}"
        );
        assert!(error.to_string().contains("truncated"), "{error}");
    }
}

/// `zlib` stream (level 9) produced by a reference encoder for
/// [`real_encoder_text`]; it holds a dynamic-Huffman block with
/// back-references, the shape real DOCX/XLSX producers emit.
const REAL_ENCODER_STREAM: [u8; 167] = [
    0x78, 0xda, 0x95, 0xd2, 0x4d, 0x0e, 0x82, 0x30, 0x10, 0x40, 0xe1, 0xab, 0x4c, 0x70, 0xaf, 0x6d,
    0x69, 0x4b, 0x6b, 0x90, 0xbd, 0xc7, 0x50, 0xa9, 0x8a, 0x82, 0x45, 0x04, 0xff, 0x4e, 0x6f, 0x22,
    0xa3, 0x19, 0x96, 0xb3, 0x7e, 0xc9, 0xb7, 0x7a, 0x79, 0x0b, 0x55, 0xb9, 0x4a, 0x44, 0x52, 0xac,
    0xfb, 0xd0, 0x80, 0x58, 0x42, 0x7f, 0x0c, 0x70, 0x1d, 0xaa, 0xdd, 0x19, 0xb6, 0x5d, 0x7c, 0x5c,
    0x60, 0x1f, 0x9f, 0x70, 0x1a, 0x9a, 0xf6, 0x06, 0xf1, 0x1e, 0xba, 0x6f, 0xae, 0x37, 0xef, 0x17,
    0x94, 0xf1, 0x00, 0x33, 0x31, 0xcf, 0x17, 0x6d, 0x91, 0x8f, 0x88, 0x44, 0x44, 0x32, 0x11, 0x49,
    0x11, 0x85, 0x88, 0x62, 0x22, 0x8a, 0x22, 0x29, 0x22, 0x29, 0x13, 0x49, 0x29, 0xa2, 0x11, 0xd1,
    0x4c, 0x44, 0x53, 0xc4, 0x20, 0x62, 0x98, 0x88, 0xa1, 0x88, 0x45, 0xc4, 0x32, 0x11, 0x4b, 0x91,
    0x0c, 0x91, 0x8c, 0x89, 0x64, 0x14, 0x71, 0x88, 0x38, 0x26, 0xe2, 0x28, 0xe2, 0x11, 0xf1, 0x4c,
    0xc4, 0x4f, 0x66, 0xfb, 0x2d, 0x2b, 0xb9, 0xcf, 0xca, 0xe9, 0xb4, 0xff, 0x6b, 0xd9, 0xdb, 0x8e,
    0xdf, 0x7e, 0x00, 0xce, 0x43, 0x0f, 0x30,
];

/// Plaintext encoded by [`REAL_ENCODER_STREAM`] (834 bytes).
fn real_encoder_text() -> String {
    use std::fmt::Write as _;

    let mut text = String::new();
    for index in 0..12 {
        write!(
            text,
            "<p id=\"{index}\">Item {index}: the quick brown fox \
             jumps over the lazy dog #{index}.</p>"
        )
        .unwrap();
    }
    text
}

#[test]
fn inflates_a_dynamic_block_with_back_references_from_a_real_encoder() {
    // 0x95 = BFINAL=1, BTYPE=10 (dynamic Huffman).
    assert_eq!((REAL_ENCODER_STREAM[2] >> 1) & 0b11, 2);
    let decoded = zlib_decompress(&REAL_ENCODER_STREAM, DEFAULT_MAX_OUTPUT_BYTES).unwrap();
    assert_eq!(decoded.len(), 834);
    assert_eq!(decoded, real_encoder_text().as_bytes());
    // The declared limit is inclusive: exactly 834 bytes still succeeds.
    assert_eq!(zlib_decompress(&REAL_ENCODER_STREAM, 834).unwrap(), decoded);
    let error = zlib_decompress(&REAL_ENCODER_STREAM, 833).unwrap_err();
    assert!(error.to_string().contains("limit"), "{error}");
}
