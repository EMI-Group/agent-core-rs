//! Raw `DEFLATE` (RFC 1951), `zlib` framing (RFC 1950), and the `Adler-32` /
//! `CRC-32` checksums needed by the ZIP container.
//!
//! The crate has no compression dependency, so container formats such as DOCX,
//! XLSX, and PDF streams are decoded with this in-crate implementation.
//! The decoder is deliberately strict: it validates every code, never emits
//! partial output, and refuses to grow a buffer beyond the caller's
//! `max_output` limit ([`DEFAULT_MAX_OUTPUT_BYTES`] guards against
//! decompression bombs).
//!
//! # Error classification
//!
//! - [`Error::Retrieval`] — the byte stream ended early, or a trailing checksum
//!   did not match the decoded bytes (the payload is intact-by-contract but not
//!   what it claims to be).
//! - [`Error::Validation`] — the stream is structurally invalid: unknown block
//!   type, bad Huffman code, invalid distance, output limit exceeded, or a bad
//!   `zlib` header field.
//! - [`Error::Unsupported`] — `zlib` preset dictionaries, which this decoder
//!   deliberately does not implement.
//!
//! # Example
//!
//! ```
//! use openjiuwen_retrieval::parsing::{zlib_decompress, DEFAULT_MAX_OUTPUT_BYTES};
//!
//! // `zlib` stream for "hello" (a fixed-Huffman block plus its Adler-32).
//! let stream = [0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00, 0x06, 0x2c, 0x02, 0x15];
//! assert_eq!(
//!     zlib_decompress(&stream, DEFAULT_MAX_OUTPUT_BYTES).unwrap(),
//!     b"hello"
//! );
//! ```

use openjiuwen_common::error::{Error, Result};

/// Upper bound applied by [`inflate`] and [`zlib_decompress`] callers that have
/// no better estimate.
///
/// A few kilobytes of `DEFLATE` can expand to gigabytes; the ZIP reader passes
/// this limit so a malicious archive cannot exhaust memory.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;

/// Base lengths for the `DEFLATE` length codes 257..=285.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra-bit counts for the `DEFLATE` length codes 257..=285.
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Base distances for the `DEFLATE` distance codes 0..=29.
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Extra-bit counts for the `DEFLATE` distance codes 0..=29.
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Transmission order of the code-length alphabet (RFC 1951 §3.2.7).
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Builds the `CRC-32` lookup table (reflected polynomial `0xEDB8_8320`).
const CRC32_TABLE: [u32; 256] = build_crc32_table();

/// Computes the `CRC-32` table at compile time.
const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 0 {
                value >> 1
            } else {
                (value >> 1) ^ 0xEDB8_8320
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

/// Returns the message for a stream that ended in the middle of a structure.
fn truncated(what: &str) -> Error {
    Error::Retrieval(format!(
        "truncated DEFLATE stream: unexpected end of data while {what}"
    ))
}

/// Returns the error raised when decoded output would pass `max_output`.
fn output_limit(max_output: usize) -> Error {
    Error::Validation(format!(
        "inflate output exceeds the {max_output}-byte limit"
    ))
}

/// Returns the standard `CRC-32` (IEEE 802.3) of `data`.
///
/// ```
/// # use openjiuwen_retrieval::parsing::crc32;
/// assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
/// ```
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Returns the `Adler-32` checksum used by the `zlib` wrapper (RFC 1950).
///
/// ```
/// # use openjiuwen_retrieval::parsing::adler32;
/// assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
/// ```
pub fn adler32(data: &[u8]) -> u32 {
    const MODULUS: u32 = 65_521;
    // 5552 is the largest block that cannot overflow a 32-bit accumulator.
    const MAX_BLOCK: usize = 5552;

    let mut low: u32 = 1;
    let mut high: u32 = 0;
    for block in data.chunks(MAX_BLOCK) {
        for &byte in block {
            low += u32::from(byte);
            high += low;
        }
        low %= MODULUS;
        high %= MODULUS;
    }
    (high << 16) | low
}

/// Reads `count` bits in `DEFLATE` order (least significant bit first).
struct BitReader<'a> {
    /// The whole input; the reader never consumes past its end.
    data: &'a [u8],
    /// Index of the next byte to load into `current`.
    byte_pos: usize,
    /// The byte being consumed, shifted right as bits are taken.
    current: u32,
    /// Number of unread bits left in `current`.
    bits_left: u32,
}

impl<'a> BitReader<'a> {
    /// Starts a reader at the beginning of `data`.
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            current: 0,
            bits_left: 0,
        }
    }

    /// Reads one bit.
    fn read_bit(&mut self) -> Result<u32> {
        if self.bits_left == 0 {
            let byte = *self
                .data
                .get(self.byte_pos)
                .ok_or_else(|| truncated("reading a bit"))?;
            self.byte_pos += 1;
            self.current = u32::from(byte);
            self.bits_left = 8;
        }
        let bit = self.current & 1;
        self.current >>= 1;
        self.bits_left -= 1;
        Ok(bit)
    }

    /// Reads `count` (0..=16) bits as an unsigned integer, least significant bit first.
    fn read_bits(&mut self, count: u32) -> Result<u32> {
        let mut value = 0u32;
        for position in 0..count {
            value |= self.read_bit()? << position;
        }
        Ok(value)
    }

    /// Discards the rest of the current byte so the next read starts byte-aligned.
    fn align_to_byte(&mut self) {
        self.current = 0;
        self.bits_left = 0;
    }

    /// Reads one byte; the caller must have aligned the reader first.
    fn read_aligned_byte(&mut self) -> Result<u8> {
        debug_assert_eq!(self.bits_left, 0, "reader must be byte-aligned");
        let byte = *self
            .data
            .get(self.byte_pos)
            .ok_or_else(|| truncated("reading a stored block byte"))?;
        self.byte_pos += 1;
        Ok(byte)
    }

    /// Reads a 16-bit little-endian value; the caller must have aligned the reader first.
    fn read_aligned_u16(&mut self) -> Result<u16> {
        let low = u16::from(self.read_aligned_byte()?);
        let high = u16::from(self.read_aligned_byte()?);
        Ok(low | (high << 8))
    }

    /// Number of bytes that have not been consumed yet.
    fn remaining_bytes(&self) -> usize {
        self.data.len().saturating_sub(self.byte_pos)
    }
}

/// A canonical Huffman decoding table (the "puff" representation).
#[derive(Debug)]
struct Huffman {
    /// Number of codes per bit length, indexed 1..=15.
    count: [u16; 16],
    /// Symbols ordered by code length, then by symbol value.
    symbol: Vec<u16>,
}

impl Huffman {
    /// Builds a decoding table from per-symbol code lengths.
    ///
    /// Incomplete codes are accepted because RFC 1951 allows them for the
    /// distance alphabet (a single used code is stored with length 1);
    /// over-subscribed codes are always rejected.
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut count = [0u16; 16];
        for &length in lengths {
            if length > 15 {
                return Err(Error::Validation(format!(
                    "invalid Huffman code length {length} (maximum is 15)"
                )));
            }
            count[usize::from(length)] += 1;
        }
        let mut left: i32 = 1;
        for &available in count.iter().skip(1) {
            left <<= 1;
            left -= i32::from(available);
            if left < 0 {
                return Err(Error::Validation(
                    "over-subscribed Huffman code in DEFLATE stream".to_owned(),
                ));
            }
        }
        count[0] = 0;
        let mut offsets = [0u16; 16];
        for length in 1..15 {
            offsets[length + 1] = offsets[length] + count[length];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (value, &length) in lengths.iter().enumerate() {
            if length != 0 {
                let slot = usize::from(offsets[usize::from(length)]);
                symbol[slot] = value as u16;
                offsets[usize::from(length)] += 1;
            }
        }
        Ok(Self { count, symbol })
    }

    /// Decodes one symbol, reading bits most-significant-bit first as RFC 1951 requires.
    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;
        for length in 1..=15 {
            code |= reader.read_bit()? as i32;
            let count = i32::from(self.count[length]);
            if code - first < count {
                return Ok(self.symbol[(index + code - first) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(Error::Validation(
            "invalid Huffman code in DEFLATE stream".to_owned(),
        ))
    }
}

/// Builds the two fixed Huffman tables of RFC 1951 §3.2.6.
fn fixed_tables() -> Result<(Huffman, Huffman)> {
    let mut literal_lengths = [0u8; 288];
    for (symbol, length) in literal_lengths.iter_mut().enumerate() {
        *length = match symbol {
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    let distance_lengths = [5u8; 32];
    Ok((
        Huffman::new(&literal_lengths)?,
        Huffman::new(&distance_lengths)?,
    ))
}

/// Decompresses a raw `DEFLATE` stream (RFC 1951) with no framing header and no
/// trailing checksum.
///
/// `max_output` caps the number of bytes that may be produced; exceeding it
/// fails the call instead of growing the buffer further.
///
/// # Errors
///
/// - [`Error::Validation`] for an invalid block type, a bad Huffman code, an
///   invalid back-reference distance, an inconsistent stored-block length, or
///   output past `max_output`.
/// - [`Error::Retrieval`] when the input ends in the middle of a structure.
///
/// ```
/// # use openjiuwen_retrieval::parsing::inflate;
/// // One final stored block (BTYPE=00) holding "hi".
/// let stream = [0x01, 0x02, 0x00, 0xfd, 0xff, b'h', b'i'];
/// assert_eq!(inflate(&stream, 1024).unwrap(), b"hi");
/// ```
pub fn inflate(data: &[u8], max_output: usize) -> Result<Vec<u8>> {
    let mut reader = BitReader::new(data);
    let mut output = Vec::new();
    loop {
        let final_block = reader.read_bit()? == 1;
        match reader.read_bits(2)? {
            0 => stored_block(&mut reader, &mut output, max_output)?,
            1 => {
                let (literals, distances) = fixed_tables()?;
                compressed_block(&mut reader, &mut output, max_output, &literals, &distances)?;
            }
            2 => dynamic_block(&mut reader, &mut output, max_output)?,
            other => {
                return Err(Error::Validation(format!(
                    "invalid DEFLATE block type {other}"
                )));
            }
        }
        if final_block {
            return Ok(output);
        }
    }
}

/// Copies an uncompressed ("stored") block, RFC 1951 §3.2.4.
fn stored_block(reader: &mut BitReader<'_>, output: &mut Vec<u8>, max_output: usize) -> Result<()> {
    reader.align_to_byte();
    let len = reader.read_aligned_u16()?;
    let nlen = reader.read_aligned_u16()?;
    if len != !nlen {
        return Err(Error::Validation(format!(
            "stored block length mismatch: LEN={len:#06x} NLEN={nlen:#06x}"
        )));
    }
    let len = usize::from(len);
    if reader.remaining_bytes() < len {
        return Err(truncated("copying a stored block"));
    }
    if output.len() + len > max_output {
        return Err(output_limit(max_output));
    }
    let start = reader.byte_pos;
    output.extend_from_slice(&reader.data[start..start + len]);
    reader.byte_pos += len;
    Ok(())
}

/// Reads a dynamic-Huffman block header and decodes its payload, RFC 1951 §3.2.7.
fn dynamic_block(
    reader: &mut BitReader<'_>,
    output: &mut Vec<u8>,
    max_output: usize,
) -> Result<()> {
    let literal_count = reader.read_bits(5)? as usize + 257;
    let distance_count = reader.read_bits(5)? as usize + 1;
    let code_length_count = reader.read_bits(4)? as usize + 4;
    if literal_count > 286 {
        return Err(Error::Validation(format!(
            "dynamic block declares {literal_count} literal/length codes (maximum is 286)"
        )));
    }
    if distance_count > 30 {
        return Err(Error::Validation(format!(
            "dynamic block declares {distance_count} distance codes (maximum is 30)"
        )));
    }

    let mut code_length_lengths = [0u8; 19];
    for &symbol in CODE_LENGTH_ORDER.iter().take(code_length_count) {
        code_length_lengths[symbol] = reader.read_bits(3)? as u8;
    }
    let code_length_tree = Huffman::new(&code_length_lengths)?;

    let total = literal_count + distance_count;
    let mut lengths = vec![0u8; total];
    let mut index = 0;
    while index < total {
        match code_length_tree.decode(reader)? {
            value @ 0..=15 => {
                lengths[index] = value as u8;
                index += 1;
            }
            16 => {
                let previous = index
                    .checked_sub(1)
                    .map(|slot| lengths[slot])
                    .ok_or_else(|| {
                        Error::Validation(
                            "dynamic block repeats a code length with no previous length"
                                .to_owned(),
                        )
                    })?;
                let repeat = 3 + reader.read_bits(2)? as usize;
                fill_lengths(&mut lengths, index, repeat, previous)?;
                index += repeat;
            }
            17 => {
                let repeat = 3 + reader.read_bits(3)? as usize;
                fill_lengths(&mut lengths, index, repeat, 0)?;
                index += repeat;
            }
            other => {
                debug_assert_eq!(other, 18);
                let repeat = 11 + reader.read_bits(7)? as usize;
                fill_lengths(&mut lengths, index, repeat, 0)?;
                index += repeat;
            }
        }
    }

    if lengths[256] == 0 {
        return Err(Error::Validation(
            "dynamic block has no end-of-block code".to_owned(),
        ));
    }
    let literals = Huffman::new(&lengths[..literal_count])?;
    let distances = Huffman::new(&lengths[literal_count..])?;
    compressed_block(reader, output, max_output, &literals, &distances)
}

/// Writes `repeat` copies of `value`, rejecting runs past the declared code count.
fn fill_lengths(lengths: &mut [u8], start: usize, repeat: usize, value: u8) -> Result<()> {
    let end = start
        .checked_add(repeat)
        .filter(|&end| end <= lengths.len())
        .ok_or_else(|| {
            Error::Validation(format!(
                "dynamic block code-length repeat of {repeat} exceeds the declared code count"
            ))
        })?;
    lengths[start..end].fill(value);
    Ok(())
}

/// Decodes the symbols of a Huffman-coded block into `output`.
fn compressed_block(
    reader: &mut BitReader<'_>,
    output: &mut Vec<u8>,
    max_output: usize,
    literals: &Huffman,
    distances: &Huffman,
) -> Result<()> {
    loop {
        match literals.decode(reader)? {
            byte @ 0..=255 => {
                if output.len() >= max_output {
                    return Err(output_limit(max_output));
                }
                output.push(byte as u8);
            }
            256 => return Ok(()),
            symbol => copy_back_reference(reader, output, max_output, symbol, distances)?,
        }
    }
}

/// Decodes a length/distance pair and copies the referenced bytes, RFC 1951 §3.2.3.
fn copy_back_reference(
    reader: &mut BitReader<'_>,
    output: &mut Vec<u8>,
    max_output: usize,
    symbol: u16,
    distances: &Huffman,
) -> Result<()> {
    let index = usize::from(symbol) - 257;
    let (Some(&length_base), Some(&length_extra)) =
        (LENGTH_BASE.get(index), LENGTH_EXTRA.get(index))
    else {
        return Err(Error::Validation(format!(
            "invalid DEFLATE length symbol {symbol}"
        )));
    };
    let length = usize::from(length_base) + reader.read_bits(u32::from(length_extra))? as usize;

    let distance_symbol = distances.decode(reader)?;
    let (Some(&distance_base), Some(&distance_extra)) = (
        DISTANCE_BASE.get(usize::from(distance_symbol)),
        DISTANCE_EXTRA.get(usize::from(distance_symbol)),
    ) else {
        return Err(Error::Validation(format!(
            "invalid DEFLATE distance symbol {distance_symbol}"
        )));
    };
    let distance =
        usize::from(distance_base) + reader.read_bits(u32::from(distance_extra))? as usize;
    if distance > output.len() {
        return Err(Error::Validation(format!(
            "invalid DEFLATE back-reference distance {distance} (only {} bytes of history)",
            output.len()
        )));
    }
    if output.len() + length > max_output {
        return Err(output_limit(max_output));
    }
    // Byte-at-a-time so the copy also works when it overlaps itself
    // (distance < length), which RFC 1951 explicitly allows.
    let mut position = output.len() - distance;
    let mut remaining = length;
    while remaining > 0 {
        let byte = output[position];
        output.push(byte);
        position += 1;
        remaining -= 1;
    }
    Ok(())
}

/// Decompresses a `zlib` stream (RFC 1950): 2-byte header, the `DEFLATE`
/// payload, and the trailing `Adler-32` checksum.
///
/// # Errors
///
/// - [`Error::Validation`] for a header whose method is not `DEFLATE`, a window
///   exponent above 7, or a failed header check; plus every [`inflate`] error.
/// - [`Error::Unsupported`] when the header flags a preset dictionary.
/// - [`Error::Retrieval`] for a short payload or a mismatched `Adler-32`.
///
/// ```
/// # use openjiuwen_retrieval::parsing::zlib_decompress;
/// let stream = [0x78, 0x01, 0x01, 0x01, 0x00, 0xfe, 0xff, b'A', 0x00, 0x42, 0x00, 0x42];
/// assert_eq!(zlib_decompress(&stream, 16).unwrap(), b"A");
/// ```
pub fn zlib_decompress(data: &[u8], max_output: usize) -> Result<Vec<u8>> {
    if data.len() < 2 {
        return Err(truncated("reading the zlib header"));
    }
    let method = data[0];
    let flags = data[1];
    let compression_method = method & 0x0F;
    let window_exponent = method >> 4;
    if compression_method != 8 {
        return Err(Error::Validation(format!(
            "unsupported zlib compression method {compression_method} (only DEFLATE/8 is supported)"
        )));
    }
    if window_exponent > 7 {
        return Err(Error::Validation(format!(
            "invalid zlib window exponent CINFO={window_exponent} (maximum is 7)"
        )));
    }
    if (u16::from(method) * 256 + u16::from(flags)) % 31 != 0 {
        return Err(Error::Validation(
            "invalid zlib header check bits".to_owned(),
        ));
    }
    if flags & 0x20 != 0 {
        return Err(Error::Unsupported(
            "zlib preset dictionaries (FDICT) are not supported".to_owned(),
        ));
    }
    let Some(payload_end) = data.len().checked_sub(4) else {
        return Err(truncated("reading the zlib trailer"));
    };
    let output = inflate(&data[2..payload_end], max_output)?;
    let expected = u32::from_be_bytes([
        data[payload_end],
        data[payload_end + 1],
        data[payload_end + 2],
        data[payload_end + 3],
    ]);
    let actual = adler32(&output);
    if actual != expected {
        return Err(Error::Retrieval(format!(
            "zlib Adler-32 mismatch: trailer declares {expected:#010x}, decoded data yields {actual:#010x}"
        )));
    }
    Ok(output)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
