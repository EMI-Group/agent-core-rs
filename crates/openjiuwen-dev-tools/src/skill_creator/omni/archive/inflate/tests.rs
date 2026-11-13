//! Unit tests for the RFC 1951 decoder.
//!
//! Every stream is assembled bit by bit from the RFC tables, so the tests are
//! deterministic and need no compressor: stored blocks, the fixed alphabet, and
//! dynamic blocks with repeats and extra bits are all covered, together with the
//! malformed-input and output-cap matrices.

use super::*;
use openjiuwen_common::error::ErrorKind;

/// The canonical 7-byte fixed-Huffman encoding of `"hello"`.
const HELLO_DEFLATE: [u8; 7] = [0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00];

/// Writes bits in DEFLATE order (least significant bit first within a byte).
#[derive(Debug, Default)]
struct BitWriter {
    /// Bytes written so far, including a partially filled last byte.
    data: Vec<u8>,
    /// Number of bits written.
    bit_len: usize,
}

impl BitWriter {
    /// An empty writer.
    fn new() -> Self {
        Self::default()
    }

    /// Appends the low `count` bits of `value`, least significant bit first.
    fn write_bits(&mut self, value: u32, count: u32) {
        for offset in 0..count {
            let bit = (value >> offset) & 1;
            if self.bit_len % 8 == 0 {
                self.data.push(0);
            }
            if bit == 1 {
                let index = self.bit_len / 8;
                self.data[index] |= 1 << (self.bit_len % 8);
            }
            self.bit_len += 1;
        }
    }

    /// Appends a Huffman code, most significant bit first as RFC 1951 requires.
    fn write_code(&mut self, code: u32, length: u8) {
        for shift in (0..length).rev() {
            self.write_bits((code >> shift) & 1, 1);
        }
    }

    /// Pads with zero bits up to the next byte boundary.
    fn align(&mut self) {
        while self.bit_len % 8 != 0 {
            self.write_bits(0, 1);
        }
    }

    /// Appends a stored block (RFC 1951 §3.2.4).
    fn write_stored(&mut self, bytes: &[u8], final_block: bool) {
        self.write_bits(u32::from(final_block), 1);
        self.write_bits(0, 2);
        self.align();
        let len = u16::try_from(bytes.len()).expect("a stored payload fits a u16");
        self.write_bits(u32::from(len), 16);
        self.write_bits(u32::from(!len), 16);
        self.data.extend_from_slice(bytes);
        self.bit_len += bytes.len() * 8;
    }

    /// The finished stream, with the final partial byte zero-padded.
    fn finish(self) -> Vec<u8> {
        self.data
    }
}

/// The fixed Huffman code for `symbol` (RFC 1951 §3.2.6).
fn fixed_code(symbol: u16) -> (u32, u8) {
    match symbol {
        0..=143 => (0x30 + u32::from(symbol), 8),
        144..=255 => (0x190 + u32::from(symbol - 144), 9),
        256..=279 => (u32::from(symbol - 256), 7),
        _ => (0xC0 + u32::from(symbol - 280), 8),
    }
}

/// Writes one literal/length symbol with the fixed alphabet.
fn write_fixed_symbol(writer: &mut BitWriter, symbol: u16) {
    let (code, length) = fixed_code(symbol);
    writer.write_code(code, length);
}

/// Writes one distance symbol with the fixed five-bit alphabet.
fn write_fixed_distance(writer: &mut BitWriter, symbol: u16) {
    writer.write_code(u32::from(symbol), 5);
}

/// Writes the `BFINAL`/`BTYPE` header of a fixed-Huffman block.
fn write_fixed_header(writer: &mut BitWriter, final_block: bool) {
    writer.write_bits(u32::from(final_block), 1);
    writer.write_bits(1, 2);
}

/// Assigns canonical codes to `(symbol, length)` pairs (RFC 1951 §3.2.2).
fn canonical_codes(alphabet: &[(u16, u8)]) -> Vec<(u16, u32, u8)> {
    let max_length = alphabet
        .iter()
        .map(|&(_, length)| length)
        .max()
        .unwrap_or(0);
    let mut sorted = alphabet.to_vec();
    sorted.sort_by_key(|&(symbol, _)| symbol);
    let mut table = Vec::new();
    let mut code = 0u32;
    for length in 1..=max_length {
        for &(symbol, symbol_length) in &sorted {
            if symbol_length == length {
                table.push((symbol, code, length));
                code += 1;
            }
        }
        code <<= 1;
    }
    table
}

/// The code and its bit length for `symbol` in `table`.
fn code_for(table: &[(u16, u32, u8)], symbol: u16) -> (u32, u8) {
    let &(_, code, length) = table
        .iter()
        .find(|&&(value, _, _)| value == symbol)
        .expect("the symbol is part of the table");
    (code, length)
}

/// Writes one symbol of a hand-built alphabet.
fn write_symbol(writer: &mut BitWriter, table: &[(u16, u32, u8)], symbol: u16) {
    let (code, length) = code_for(table, symbol);
    writer.write_code(code, length);
}

/// Encodes `lengths` with the code-length alphabet, using the RFC repeat codes
/// for runs of equal values.
fn write_length_list(writer: &mut BitWriter, codes: &[(u16, u32, u8)], lengths: &[u8]) {
    let mut index = 0;
    while index < lengths.len() {
        let value = lengths[index];
        let mut run = 1;
        while index + run < lengths.len() && lengths[index + run] == value {
            run += 1;
        }
        if value == 0 {
            let mut remaining = run;
            while remaining >= 11 {
                let chunk = remaining.min(138);
                write_symbol(writer, codes, 18);
                writer.write_bits(u32::try_from(chunk - 11).expect("repeat fits 7 bits"), 7);
                remaining -= chunk;
            }
            while remaining >= 3 {
                let chunk = remaining.min(10);
                write_symbol(writer, codes, 17);
                writer.write_bits(u32::try_from(chunk - 3).expect("repeat fits 3 bits"), 3);
                remaining -= chunk;
            }
            for _ in 0..remaining {
                write_symbol(writer, codes, 0);
            }
        } else {
            write_symbol(writer, codes, u16::from(value));
            let mut remaining = run - 1;
            while remaining >= 3 {
                let chunk = remaining.min(6);
                write_symbol(writer, codes, 16);
                writer.write_bits(u32::try_from(chunk - 3).expect("repeat fits 2 bits"), 2);
                remaining -= chunk;
            }
            for _ in 0..remaining {
                write_symbol(writer, codes, u16::from(value));
            }
        }
        index += run;
    }
}

/// Writes a complete dynamic-block header and returns the literal/length table.
///
/// The code-length alphabet covers every length value used plus the repeat codes
/// `16`, `17`, and `18`, so the header exercises the repeat paths of the decoder.
fn write_dynamic_header(
    writer: &mut BitWriter,
    final_block: bool,
    literal_lengths: &[u8],
    distance_lengths: &[u8],
) -> Vec<(u16, u32, u8)> {
    writer.write_bits(u32::from(final_block), 1);
    writer.write_bits(0b10, 2);
    let hlit = literal_lengths.len() - 257;
    let hdist = distance_lengths.len() - 1;
    writer.write_bits(u32::try_from(hlit).expect("257..=286 literal codes"), 5);
    writer.write_bits(u32::try_from(hdist).expect("1..=30 distance codes"), 5);

    let mut symbols: Vec<u16> = literal_lengths
        .iter()
        .chain(distance_lengths.iter())
        .map(|&length| u16::from(length))
        .collect();
    symbols.extend([16, 17, 18]);
    symbols.sort_unstable();
    symbols.dedup();
    let mut bits = 1u8;
    while (1usize << bits) < symbols.len() {
        bits += 1;
    }
    let mut code_length_lengths = [0u8; 19];
    for &symbol in &symbols {
        code_length_lengths[usize::from(symbol)] = bits;
    }

    let needed = symbols
        .iter()
        .map(|&symbol| {
            CODE_LENGTH_ORDER
                .iter()
                .position(|&entry| entry == usize::from(symbol))
                .expect("every code-length symbol is in the RFC order")
                + 1
        })
        .max()
        .unwrap_or(4)
        .max(4);
    writer.write_bits(u32::try_from(needed - 4).expect("HCLEN fits four bits"), 4);
    for &symbol in CODE_LENGTH_ORDER.iter().take(needed) {
        writer.write_bits(u32::from(code_length_lengths[symbol]), 3);
    }

    let code_length_codes = canonical_codes(
        &symbols
            .iter()
            .map(|&symbol| (symbol, code_length_lengths[usize::from(symbol)]))
            .collect::<Vec<_>>(),
    );
    let mut all_lengths = Vec::with_capacity(literal_lengths.len() + distance_lengths.len());
    all_lengths.extend_from_slice(literal_lengths);
    all_lengths.extend_from_slice(distance_lengths);
    write_length_list(writer, &code_length_codes, &all_lengths);

    canonical_codes(
        &literal_lengths
            .iter()
            .enumerate()
            .map(|(symbol, &length)| (u16::try_from(symbol).expect("symbol fits a u16"), length))
            .collect::<Vec<_>>(),
    )
}

/// Returns the validation error for a stream that must be rejected with a
/// message containing `needle`.
fn validation_error(stream: &[u8], max_output: usize, needle: &str) -> Error {
    let error = inflate_stream(stream, max_output).expect_err("the stream should be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
    assert!(error.to_string().contains(needle), "{error}");
    error
}

/// Asserts that `stream` fails validation with a message containing `needle`.
fn assert_validation(stream: &[u8], max_output: usize, needle: &str) {
    let _ = validation_error(stream, max_output, needle);
}

/// Asserts that `stream` is reported as truncated at a named byte offset.
fn assert_truncated(stream: &[u8]) {
    let error = validation_error(stream, 1024, "truncated DEFLATE stream");
    assert!(error.to_string().contains("byte"), "{error}");
}

#[test]
fn inflate_stored_block_round_trips() {
    let mut writer = BitWriter::new();
    writer.write_stored(b"stored payload", true);
    let stream = writer.finish();
    assert_eq!(stream[0], 0x01, "BFINAL=1 with BTYPE=00");
    assert_eq!(
        inflate(&stream, 1024).expect("stored block"),
        b"stored payload"
    );
    assert_eq!(
        inflate_stream(&stream, 1024)
            .expect("stored block")
            .consumed,
        stream.len()
    );
}

#[test]
fn inflate_multi_block_stream_combines_stored_and_fixed_blocks() {
    let mut writer = BitWriter::new();
    writer.write_stored(b"abc", false);
    write_fixed_header(&mut writer, true);
    for letter in b"hello" {
        write_fixed_symbol(&mut writer, u16::from(*letter));
    }
    write_fixed_symbol(&mut writer, 256);
    let stream = writer.finish();
    assert_eq!(inflate(&stream, 1024).expect("mixed blocks"), b"abchello");
}

#[test]
fn inflate_fixed_huffman_matches_the_canonical_hello_stream() {
    assert_eq!(inflate(&HELLO_DEFLATE, 1024).expect("hello"), b"hello");

    let mut writer = BitWriter::new();
    write_fixed_header(&mut writer, true);
    for letter in b"hello" {
        write_fixed_symbol(&mut writer, u16::from(*letter));
    }
    write_fixed_symbol(&mut writer, 256);
    assert_eq!(writer.finish(), HELLO_DEFLATE);
}

#[test]
fn inflate_dynamic_literal_only_block() {
    let mut literal_lengths = vec![0u8; 257];
    literal_lengths[97..=103].fill(3);
    literal_lengths[256] = 3;
    let distance_lengths = vec![0u8; 1];

    let mut writer = BitWriter::new();
    let table = write_dynamic_header(&mut writer, true, &literal_lengths, &distance_lengths);
    for letter in b"abcdefg" {
        write_symbol(&mut writer, &table, u16::from(*letter));
    }
    write_symbol(&mut writer, &table, 256);
    let stream = writer.finish();
    assert_eq!(inflate(&stream, 64).expect("dynamic literals"), b"abcdefg");
}

#[test]
fn inflate_dynamic_block_with_length_distance_and_extra_bits() {
    // Complete eight-symbol literal alphabet: a, b, c, d, a filler, EOB, and the
    // two length symbols the payload uses.
    let mut literal_lengths = vec![0u8; 266];
    for symbol in [97usize, 98, 99, 100, 101, 256, 257, 265] {
        literal_lengths[symbol] = 3;
    }
    // One distance code (symbol 3 = distance 4) in a one-bit alphabet.
    let mut distance_lengths = vec![0u8; 4];
    distance_lengths[3] = 1;

    let mut writer = BitWriter::new();
    let literals = write_dynamic_header(&mut writer, true, &literal_lengths, &distance_lengths);
    for letter in b"abcd" {
        write_symbol(&mut writer, &literals, u16::from(*letter));
    }
    // Length symbol 265 (base 11, one extra bit) with extra bit 1 -> length 12.
    write_symbol(&mut writer, &literals, 265);
    writer.write_bits(1, 1);
    // Distance symbol 3 (base 4, no extra bits) -> distance 4.
    let distances = canonical_codes(&[(3, 1)]);
    write_symbol(&mut writer, &distances, 3);
    write_symbol(&mut writer, &literals, 256);
    let stream = writer.finish();
    assert_eq!(
        inflate(&stream, 64).expect("length/distance pair"),
        b"abcdabcdabcdabcd"
    );
}

#[test]
fn inflate_overlapping_back_reference_repeats_a_run() {
    let mut writer = BitWriter::new();
    write_fixed_header(&mut writer, true);
    for letter in b"abc" {
        write_fixed_symbol(&mut writer, u16::from(*letter));
    }
    // Length symbol 265 (base 11, one extra bit) with extra bit 1 -> length 12.
    write_fixed_symbol(&mut writer, 265);
    writer.write_bits(1, 1);
    // Distance symbol 2 (base 3, no extra bits) -> distance 3, shorter than the
    // length, so the copy overlaps the bytes it is writing.
    write_fixed_distance(&mut writer, 2);
    write_fixed_symbol(&mut writer, 256);
    let stream = writer.finish();
    assert_eq!(
        inflate(&stream, 64).expect("overlapping copy"),
        b"abcabcabcabcabc"
    );
}

#[test]
fn inflate_reports_truncated_input() {
    // No bits at all: the block header cannot be read.
    assert_truncated(&[]);
    // The header finishes inside this byte, but LEN/NLEN are missing.
    assert_truncated(&[0x01]);
    // A stored block with only one of its two LEN bytes.
    assert_truncated(&[0x01, 0x05]);
    // A fixed block whose first literal code is cut short.
    let mut writer = BitWriter::new();
    write_fixed_header(&mut writer, true);
    writer.write_bits(0, 3);
    assert_truncated(&writer.finish());
}

#[test]
fn inflate_rejects_reserved_block_type() {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1);
    writer.write_bits(3, 2);
    let error = validation_error(&writer.finish(), 64, "block type 3");
    assert!(error.to_string().contains("at bit 1 (byte 0)"), "{error}");
}

#[test]
fn inflate_rejects_stored_length_mismatch() {
    let stream = [0x01, 0x05, 0x00, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
    assert_validation(&stream, 64, "LEN=0x0005 NLEN=0x0000");
}

#[test]
fn inflate_rejects_repeat_without_previous_length() {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1);
    writer.write_bits(0b10, 2);
    writer.write_bits(0, 5); // HLIT: 257 literal/length codes
    writer.write_bits(0, 5); // HDIST: one distance code
    writer.write_bits(0, 4); // HCLEN: four code-length lengths

    // Symbols 16, 17, 18, 0; symbols 16 and 0 get a one-bit code each.
    for length in [1u32, 0, 0, 1] {
        writer.write_bits(length, 3);
    }
    // Code-length symbol 16 (code 1) before any length has been read.
    writer.write_code(1, 1);
    assert_validation(&writer.finish(), 64, "no previous length");
}

#[test]
fn inflate_rejects_code_length_overshoot() {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1);
    writer.write_bits(0b10, 2);
    writer.write_bits(0, 5); // HLIT: 257 literal/length codes
    writer.write_bits(0, 5); // HDIST: one distance code
    writer.write_bits(0, 4); // HCLEN: four code-length lengths

    // Only symbol 18 (repeat zero 11..=138) is used, with a one-bit code.
    for length in [0u32, 0, 1, 0] {
        writer.write_bits(length, 3);
    }
    for _ in 0..2 {
        writer.write_code(0, 1);
        writer.write_bits(127, 7); // repeat 138 zeros
    }
    assert_validation(&writer.finish(), 64, "runs past the declared code count");
}

#[test]
fn inflate_rejects_over_subscribed_code_lengths() {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1);
    writer.write_bits(0b10, 2);
    writer.write_bits(0, 5);
    writer.write_bits(0, 5);
    writer.write_bits(14, 4); // HCLEN = 18, covering symbols 0, 1, and 2
    let mut lengths = [0u32; 19];
    lengths[0] = 1;
    lengths[1] = 1;
    lengths[2] = 1;
    for &symbol in CODE_LENGTH_ORDER.iter().take(18) {
        writer.write_bits(lengths[symbol], 3);
    }
    assert_validation(&writer.finish(), 64, "over-subscribed");
}

#[test]
fn inflate_rejects_missing_end_of_block_code() {
    let mut literal_lengths = vec![0u8; 257];
    literal_lengths[97] = 1;
    let mut writer = BitWriter::new();
    let _table = write_dynamic_header(&mut writer, true, &literal_lengths, &[0]);
    assert_validation(&writer.finish(), 64, "end-of-block");
}

#[test]
fn inflate_rejects_distance_beyond_history() {
    let mut writer = BitWriter::new();
    write_fixed_header(&mut writer, true);
    // Length symbol 257 (length 3, no extra bits) before any literal.
    write_fixed_symbol(&mut writer, 257);
    // Distance symbol 0 -> distance 1, which reaches past the empty output.
    write_fixed_distance(&mut writer, 0);
    assert_validation(&writer.finish(), 64, "distance 1 with only 0 bytes");
}

#[test]
fn inflate_rejects_invalid_fixed_symbol_286() {
    let mut writer = BitWriter::new();
    write_fixed_header(&mut writer, true);
    write_fixed_symbol(&mut writer, 286);
    assert_validation(&writer.finish(), 64, "length symbol 286");
}

#[test]
fn inflate_stored_block_respects_the_output_cap() {
    let mut writer = BitWriter::new();
    writer.write_stored(b"abcde", true);
    let error = validation_error(&writer.finish(), 4, "output limit");
    assert!(error.to_string().contains("4-byte output limit"), "{error}");
}

#[test]
fn inflate_match_respects_the_output_cap() {
    let mut writer = BitWriter::new();
    write_fixed_header(&mut writer, true);
    write_fixed_symbol(&mut writer, u16::from(b'a'));
    write_fixed_symbol(&mut writer, 257); // length 3
    write_fixed_distance(&mut writer, 0); // distance 1
    let error = validation_error(&writer.finish(), 2, "output limit");
    assert!(error.to_string().contains("2-byte output limit"), "{error}");
}

#[test]
fn inflate_stream_reports_consumed_bytes_and_ignores_trailing_data() {
    let mut stream = HELLO_DEFLATE.to_vec();
    stream.extend_from_slice(b"trailing garbage that is not part of the stream");
    let outcome = inflate_stream(&stream, 1024).expect("hello with trailing data");
    assert_eq!(outcome.data, b"hello");
    assert_eq!(outcome.consumed, HELLO_DEFLATE.len());
    assert_eq!(
        inflate(&stream, 1024).expect("wrapper ignores trailing data"),
        b"hello"
    );
}

#[test]
fn inflate_decodes_real_compressor_output() {
    // Raw DEFLATE (RFC 1951, no zlib wrapper) of
    // "the quick brown fox jumps over the lazy dog. " repeated four times,
    // produced by a production compressor at level 9.
    let sentence = [
        0x2b, 0xc9, 0x48, 0x55, 0x28, 0x2c, 0xcd, 0x4c, 0xce, 0x56, 0x48, 0x2a, 0xca, 0x2f, 0xcf,
        0x53, 0x48, 0xcb, 0xaf, 0x50, 0xc8, 0x2a, 0xcd, 0x2d, 0x28, 0x56, 0xc8, 0x2f, 0x4b, 0x2d,
        0x52, 0x28, 0x01, 0x4a, 0xe7, 0x24, 0x56, 0x55, 0x2a, 0xa4, 0xe4, 0xa7, 0xeb, 0x81, 0x79,
        0x83, 0x40, 0x31, 0x00,
    ];
    let expected = "the quick brown fox jumps over the lazy dog. ".repeat(4);
    let outcome = inflate_stream(&sentence, 4096).expect("compressor output");
    assert_eq!(outcome.data, expected.as_bytes());
    assert_eq!(outcome.consumed, sentence.len());

    // A long run of one byte, which a compressor encodes as an overlapping match.
    let run = [
        0x4b, 0x4c, 0x1c, 0x05, 0xa3, 0x60, 0x14, 0x0c, 0x77, 0x00, 0x00,
    ];
    assert_eq!(inflate(&run, 4096).expect("run"), vec![b'a'; 1000]);
}
