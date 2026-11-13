//! RFC 1951 DEFLATE decoding (shared by `zip` and `gzip`).
//!
//! The decoder is bounded by an explicit output cap: a stored block, literal
//! run, or back-reference that would exceed the cap fails with a validation
//! error naming the cap instead of allocating. Every read is bounds-checked and
//! every structural failure names the bit and byte position of the offending
//! input, so a caller can report where a corrupt stream went wrong.
//!
//! [`inflate_stream`] also reports how many whole input bytes the DEFLATE
//! stream occupied: `gzip` needs that offset to read its trailer and to detect
//! concatenated members, and `zip` uses it to walk past a deflated entry.
//!
//! Nothing here is conditional on a compression dependency: the bit reader, the
//! canonical Huffman tables, and the LZ77 window are implemented in this module.

use openjiuwen_common::error::{Error, Result};
use std::fmt;

/// Base lengths for the length symbols 257..=285 (RFC 1951 §3.2.5).
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra-bit counts for the length symbols 257..=285 (RFC 1951 §3.2.5).
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Base distances for the distance symbols 0..=29 (RFC 1951 §3.2.5).
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Extra-bit counts for the distance symbols 0..=29 (RFC 1951 §3.2.5).
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Transmission order of the code-length alphabet (RFC 1951 §3.2.7).
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Longest Huffman code DEFLATE defines, in bits.
const MAX_CODE_BITS: usize = 15;

/// Initial output capacity, kept small so a hostile stream cannot make the
/// decoder allocate `max_output` up front.
const INITIAL_OUTPUT_CAPACITY: usize = 1024;

/// One decoded DEFLATE stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InflateOutcome {
    /// The decompressed bytes.
    pub data: Vec<u8>,
    /// Number of whole input bytes the DEFLATE stream occupied.
    ///
    /// This is the bit reader's position after the final block, rounded up to a
    /// byte boundary. It is at most `input.len()`; any trailing bytes belong to
    /// the caller (a `gzip` trailer, a concatenated member, or the next `zip`
    /// entry).
    pub consumed: usize,
}

/// Decodes one raw DEFLATE stream starting at the first bit of `input`.
///
/// # Errors
///
/// Returns [`Error::validation`] for truncated or malformed input, and for
/// output that would exceed `max_output`.
pub(crate) fn inflate_stream(input: &[u8], max_output: usize) -> Result<InflateOutcome> {
    let mut reader = BitReader::new(input);
    let mut output = Vec::with_capacity(max_output.min(INITIAL_OUTPUT_CAPACITY));
    loop {
        let block_type_bit = reader.bit_pos.saturating_add(1);
        let final_block = reader.read_bits(1)? == 1;
        match reader.read_bits(2)? {
            0 => stored_block(&mut reader, &mut output, max_output)?,
            1 => {
                let (literals, distances) = fixed_tables(block_type_bit)?;
                compressed_block(&mut reader, &mut output, max_output, &literals, &distances)?;
            }
            2 => dynamic_block(&mut reader, &mut output, max_output)?,
            other => {
                return Err(invalid(
                    format!("block type {other} (11 is reserved)"),
                    block_type_bit,
                ));
            }
        }
        if final_block {
            return Ok(InflateOutcome {
                data: output,
                consumed: reader.consumed(),
            });
        }
    }
}

/// Decompresses one raw DEFLATE stream, returning only the decoded bytes.
///
/// This is the convenience wrapper around [`inflate_stream`] for callers that do
/// not need the consumed-byte count.
///
/// # Errors
///
/// Returns [`Error::validation`] for truncated or malformed input, and for
/// output that would exceed `max_output`.
pub(crate) fn inflate(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    inflate_stream(input, max_output).map(|outcome| outcome.data)
}

/// Builds the validation error for a malformed structure at `bit_pos`.
fn invalid(what: impl fmt::Display, bit_pos: usize) -> Error {
    Error::validation(format!(
        "invalid DEFLATE {what} at bit {bit_pos} (byte {})",
        bit_pos / 8
    ))
}

/// Builds the validation error for output that would exceed `max_output`.
fn output_limit(max_output: usize, bit_pos: usize) -> Error {
    Error::validation(format!(
        "inflate output limit exceeded: decoding would pass the {max_output}-byte output limit at \
         bit {bit_pos} (byte {})",
        bit_pos / 8
    ))
}

/// Reads bits least-significant-first within each byte (RFC 1951 §3.1.1).
#[derive(Debug, Clone, Copy)]
struct BitReader<'a> {
    /// The stream being decoded.
    input: &'a [u8],
    /// Number of bits consumed so far.
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Starts a reader at the first bit of `input`.
    const fn new(input: &'a [u8]) -> Self {
        Self { input, bit_pos: 0 }
    }

    /// Index of the byte holding the next unread bit.
    const fn byte_pos(self) -> usize {
        self.bit_pos / 8
    }

    /// Number of whole bytes consumed, rounding the final partial byte up.
    const fn consumed(self) -> usize {
        self.bit_pos.div_ceil(8)
    }

    /// Reads `count` bits as an unsigned value, first bit least significant.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] naming the byte offset when the input ends
    /// before `count` bits are available.
    fn read_bits(&mut self, count: u32) -> Result<u32> {
        let start = self.bit_pos;
        let available = self.input.len().saturating_mul(8).saturating_sub(start);
        let needed = count as usize;
        if available < needed {
            return Err(self.truncated(count));
        }
        let mut value = 0u32;
        for offset in 0..needed {
            let index = self.bit_pos / 8;
            let bit = (self.input[index] >> (self.bit_pos % 8)) & 1;
            value |= u32::from(bit) << offset;
            self.bit_pos += 1;
        }
        Ok(value)
    }

    /// Discards the remaining bits of the current byte.
    fn align_to_byte(&mut self) {
        self.bit_pos = self.bit_pos.next_multiple_of(8);
    }

    /// Reads `len` whole bytes; the caller must have aligned the reader first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] naming the byte offset when the input is
    /// shorter than `len` bytes.
    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let start = self.byte_pos();
        let end = start.checked_add(len).ok_or_else(|| {
            Error::validation(format!(
                "truncated DEFLATE stream: a {len}-byte block at byte {start} overflows the input \
                 length"
            ))
        })?;
        let bytes = self.input.get(start..end).ok_or_else(|| {
            Error::validation(format!(
                "truncated DEFLATE stream: need {len} bytes at byte {start} but the input is {} \
                 bytes",
                self.input.len()
            ))
        })?;
        self.bit_pos = end.saturating_mul(8);
        Ok(bytes)
    }

    /// The error for a read that runs past the end of the input.
    fn truncated(self, bits: u32) -> Error {
        let unit = if bits == 1 { "bit" } else { "bits" };
        Error::validation(format!(
            "truncated DEFLATE stream: need {bits} {unit} at byte {} but the input is {} bytes",
            self.byte_pos(),
            self.input.len()
        ))
    }
}

/// A canonical Huffman decoding table built from per-symbol code lengths.
///
/// The representation follows the classic "puff" layout: the number of codes per
/// bit length plus the symbols ordered by (length, symbol value). Incomplete
/// tables are accepted because RFC 1951 allows a one-bit distance alphabet with
/// an unused code, and because a literal-only block may declare a single
/// zero-length distance code; over-subscribed tables are rejected.
#[derive(Debug, Clone)]
struct Huffman {
    /// Number of codes for each bit length, indexed by length (`0` is unused).
    count: [u16; MAX_CODE_BITS + 1],
    /// Symbols ordered by code length, then by symbol value.
    symbol: Vec<u16>,
}

impl Huffman {
    /// Builds a decoding table from per-symbol code lengths.
    ///
    /// `bit_pos` locates the declaration in the input and is only used to word
    /// error messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when a code length exceeds
    /// [`MAX_CODE_BITS`] or the lengths describe an over-subscribed code.
    fn build(lengths: &[u8], bit_pos: usize) -> Result<Self> {
        let mut count = [0u16; MAX_CODE_BITS + 1];
        for (symbol, &length) in lengths.iter().enumerate() {
            if usize::from(length) > MAX_CODE_BITS {
                return Err(invalid(
                    format!("code length {length} for symbol {symbol}"),
                    bit_pos,
                ));
            }
            count[usize::from(length)] += 1;
        }
        count[0] = 0;
        let mut left: i32 = 1;
        for &available in count.iter().skip(1) {
            left <<= 1;
            left -= i32::from(available);
            if left < 0 {
                return Err(invalid("over-subscribed Huffman code lengths", bit_pos));
            }
        }

        let mut offsets = [0u16; MAX_CODE_BITS + 2];
        for length in 1..=MAX_CODE_BITS {
            offsets[length + 1] = offsets[length] + count[length];
        }
        let total = usize::from(offsets[MAX_CODE_BITS + 1]);
        let mut symbol = vec![0u16; total];
        for (value, &length) in lengths.iter().enumerate() {
            if length != 0 {
                let slot = usize::from(offsets[usize::from(length)]);
                let value = u16::try_from(value).map_err(|_| {
                    invalid(
                        format!("symbol index {value} which does not fit a symbol"),
                        bit_pos,
                    )
                })?;
                symbol[slot] = value;
                offsets[usize::from(length)] += 1;
            }
        }
        Ok(Self { count, symbol })
    }

    /// Decodes one symbol, reading bits most-significant-bit first as RFC 1951
    /// requires for Huffman codes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] naming the starting bit when the next bits
    /// are not a code of this table, or when the input ends mid-code.
    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        let start = reader.bit_pos;
        if self.symbol.is_empty() {
            return Err(invalid("reference to an empty Huffman table", start));
        }
        let mut code = 0usize;
        let mut first = 0usize;
        let mut index = 0usize;
        for length in 1..=MAX_CODE_BITS {
            code |= reader.read_bits(1)? as usize;
            let count = usize::from(self.count[length]);
            if code >= first && code - first < count {
                let slot = index + (code - first);
                return self
                    .symbol
                    .get(slot)
                    .copied()
                    .ok_or_else(|| invalid("Huffman symbol index", start));
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(invalid("Huffman code", start))
    }
}

/// Builds the two fixed Huffman tables of RFC 1951 §3.2.6.
///
/// # Errors
///
/// Propagates [`Huffman::build`] failures, which cannot happen for the fixed
/// lengths.
fn fixed_tables(bit_pos: usize) -> Result<(Huffman, Huffman)> {
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
        Huffman::build(&literal_lengths, bit_pos)?,
        Huffman::build(&distance_lengths, bit_pos)?,
    ))
}

/// Copies an uncompressed ("stored") block, RFC 1951 §3.2.4.
///
/// # Errors
///
/// Returns [`Error::validation`] for a truncated header or payload, a `LEN` that
/// is not the complement of `NLEN`, or a block that would exceed `max_output`.
fn stored_block(reader: &mut BitReader<'_>, output: &mut Vec<u8>, max_output: usize) -> Result<()> {
    reader.align_to_byte();
    let header = reader.bit_pos;
    let len = reader.read_bits(16)?;
    let nlen = reader.read_bits(16)?;
    if len ^ nlen != 0xFFFF {
        return Err(invalid(
            format!("stored block length mismatch: LEN={len:#06x} NLEN={nlen:#06x}"),
            header,
        ));
    }
    let len = len as usize;
    if output.len().saturating_add(len) > max_output {
        return Err(output_limit(max_output, header));
    }
    let bytes = reader.read_bytes(len)?;
    output.extend_from_slice(bytes);
    Ok(())
}

/// Reads a dynamic-Huffman block header and decodes its payload, RFC 1951 §3.2.7.
///
/// # Errors
///
/// Returns [`Error::validation`] for an out-of-range code count, a malformed
/// code-length alphabet, a bad repeat, a missing end-of-block code, or any error
/// from the payload decoder.
fn dynamic_block(
    reader: &mut BitReader<'_>,
    output: &mut Vec<u8>,
    max_output: usize,
) -> Result<()> {
    let header = reader.bit_pos;
    let literal_count = reader.read_bits(5)? as usize + 257;
    let distance_count = reader.read_bits(5)? as usize + 1;
    let code_length_count = reader.read_bits(4)? as usize + 4;
    if literal_count > 286 {
        return Err(invalid(
            format!("literal/length code count {literal_count} (the maximum is 286)"),
            header,
        ));
    }
    if distance_count > 30 {
        return Err(invalid(
            format!("distance code count {distance_count} (the maximum is 30)"),
            header,
        ));
    }

    let mut code_length_lengths = [0u8; 19];
    for &symbol in CODE_LENGTH_ORDER.iter().take(code_length_count) {
        code_length_lengths[symbol] = reader.read_bits(3)? as u8;
    }
    let code_lengths = Huffman::build(&code_length_lengths, reader.bit_pos)?;

    let total = literal_count + distance_count;
    let mut lengths = vec![0u8; total];
    let mut index = 0usize;
    while index < total {
        let symbol_bit = reader.bit_pos;
        match code_lengths.decode(reader)? {
            value @ 0..=15 => {
                lengths[index] = value as u8;
                index += 1;
            }
            16 => {
                let repeat = 3 + reader.read_bits(2)? as usize;
                let previous = index
                    .checked_sub(1)
                    .map(|slot| lengths[slot])
                    .ok_or_else(|| {
                        invalid("code-length repeat 16 with no previous length", symbol_bit)
                    })?;
                fill_lengths(&mut lengths, index, repeat, previous, symbol_bit)?;
                index += repeat;
            }
            17 => {
                let repeat = 3 + reader.read_bits(3)? as usize;
                fill_lengths(&mut lengths, index, repeat, 0, symbol_bit)?;
                index += repeat;
            }
            _ => {
                let repeat = 11 + reader.read_bits(7)? as usize;
                fill_lengths(&mut lengths, index, repeat, 0, symbol_bit)?;
                index += repeat;
            }
        }
    }

    if lengths[256] == 0 {
        return Err(invalid(
            "dynamic block without an end-of-block (256) code",
            reader.bit_pos,
        ));
    }
    let (literal_lengths, distance_lengths) = lengths.split_at(literal_count);
    let literals = Huffman::build(literal_lengths, reader.bit_pos)?;
    let distances = Huffman::build(distance_lengths, reader.bit_pos)?;
    compressed_block(reader, output, max_output, &literals, &distances)
}

/// Writes `repeat` copies of `value`, rejecting runs past the declared count.
///
/// # Errors
///
/// Returns [`Error::validation`] naming `bit_pos` when `start + repeat` would
/// pass the end of `lengths`.
fn fill_lengths(
    lengths: &mut [u8],
    start: usize,
    repeat: usize,
    value: u8,
    bit_pos: usize,
) -> Result<()> {
    let end = start
        .checked_add(repeat)
        .filter(|&end| end <= lengths.len())
        .ok_or_else(|| {
            invalid(
                format!("code-length repeat of {repeat} runs past the declared code count"),
                bit_pos,
            )
        })?;
    lengths[start..end].fill(value);
    Ok(())
}

/// Decodes the symbols of a Huffman-coded block into `output`.
///
/// # Errors
///
/// Returns [`Error::validation`] for an invalid symbol, an invalid
/// back-reference, or output past `max_output`.
fn compressed_block(
    reader: &mut BitReader<'_>,
    output: &mut Vec<u8>,
    max_output: usize,
    literals: &Huffman,
    distances: &Huffman,
) -> Result<()> {
    loop {
        let symbol_bit = reader.bit_pos;
        match literals.decode(reader)? {
            symbol @ 0..=255 => {
                if output.len() >= max_output {
                    return Err(output_limit(max_output, symbol_bit));
                }
                output.push(symbol as u8);
            }
            256 => return Ok(()),
            symbol @ 257..=285 => {
                copy_back_reference(reader, output, max_output, symbol, distances, symbol_bit)?;
            }
            symbol => {
                return Err(invalid(
                    format!("length symbol {symbol} (the maximum is 285)"),
                    symbol_bit,
                ));
            }
        }
    }
}

/// Decodes a length/distance pair and copies the referenced bytes, RFC 1951
/// §3.2.3.
///
/// The copy is byte-at-a-time so an overlapping reference (distance shorter than
/// the length) repeats the earlier run, as RFC 1951 explicitly allows.
///
/// # Errors
///
/// Returns [`Error::validation`] when the distance reaches past the bytes
/// already produced or the match would exceed `max_output`.
fn copy_back_reference(
    reader: &mut BitReader<'_>,
    output: &mut Vec<u8>,
    max_output: usize,
    symbol: u16,
    distances: &Huffman,
    symbol_bit: usize,
) -> Result<()> {
    let index = usize::from(symbol - 257);
    let (Some(&base), Some(&extra)) = (LENGTH_BASE.get(index), LENGTH_EXTRA.get(index)) else {
        return Err(invalid(format!("length symbol {symbol}"), symbol_bit));
    };
    let length = usize::from(base) + reader.read_bits(u32::from(extra))? as usize;
    let distance_bit = reader.bit_pos;
    let distance_symbol = distances.decode(reader)?;
    if distance_symbol > 29 {
        return Err(invalid(
            format!("distance symbol {distance_symbol} (the maximum is 29)"),
            distance_bit,
        ));
    }
    let distance_index = usize::from(distance_symbol);
    let base = DISTANCE_BASE[distance_index];
    let extra = DISTANCE_EXTRA[distance_index];
    let distance = usize::from(base) + reader.read_bits(u32::from(extra))? as usize;
    if distance > output.len() {
        return Err(invalid(
            format!(
                "back-reference distance {distance} with only {} bytes of history",
                output.len()
            ),
            distance_bit,
        ));
    }
    if length > max_output.saturating_sub(output.len()) {
        return Err(output_limit(max_output, distance_bit));
    }
    let start = output.len() - distance;
    output.reserve(length);
    for offset in 0..length {
        let byte = output[start + offset];
        output.push(byte);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
