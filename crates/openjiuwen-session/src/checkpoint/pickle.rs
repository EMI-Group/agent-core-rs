//! A dependency-free Python-pickle codec for checkpoint interop.
//!
//! Python's checkpointer (`openjiuwen/core/session/checkpointer/persistence.py`)
//! serializes checkpoint payloads with `pickle`, so a checkpoint written by the
//! Python SDK cannot be read by the JSON-only backends in this crate.
//! JSON stays the crate's native checkpoint format; this module is the bridge that
//! reads and writes the Python streams, with [`serde_json::Value`] at the boundary.
//!
//! # Decoding ([`loads`])
//!
//! [`loads`] accepts the opcode subset Python emits for plain-data payloads,
//! protocols 0 through 5: `PROTO`, `FRAME`, `STOP`, marks and memo opcodes,
//! integers, floats, strings, bytes, and the list/dict/tuple container opcodes.
//! Anything else (globals, `REDUCE`, `BUILD`, buffers, …) is a typed
//! [`openjiuwen_common::error::ErrorKind::Validation`] naming the opcode and its
//! offset — never a panic and never an infinite loop.
//!
//! # Type mapping and its documented lossiness
//!
//! - Pickle dicts become JSON objects; dict keys must be strings, because Python
//!   checkpoint payloads are `str`-keyed and silently stringifying a key would
//!   corrupt data. A non-string key is an error naming the key type.
//! - Pickle lists *and tuples* become JSON arrays; JSON has no tuple type, so
//!   [`dumps`] writes every array back as a list.
//! - Pickle `bytes` become the reserved object shape
//!   `{"__bytes__": "<lowercase hex>"}`. An object with exactly one `__bytes__`
//!   key is reserved for this purpose: [`dumps`] treats that shape as bytes (and
//!   rejects a non-hex value), so real data must not use it as a plain dict.
//!   [`loads`] and [`dumps`] therefore round-trip `bytes` losslessly.
//! - Integers must fit in `i64`; longer pickle longs and out-of-range platform
//!   lengths are errors. Floats must be finite, since JSON cannot carry NaN or
//!   infinity.
//! - Nesting deeper than `MAX_DEPTH` (512) levels and length prefixes above
//!   `MAX_LENGTH` (512 MiB) are rejected.
//!
//! # Encoding ([`dumps`])
//!
//! [`dumps`] emits protocol 4 (`PROTO 4` + one `FRAME` covering the body and the
//! `STOP`, exactly as `CPython` frames it) so the result is readable by `CPython`
//! 3.4 and later. Strings and containers are memoized, so a repeated string is
//! written once and referenced with `BINGET`.
//!
//! # Checkpoints
//!
//! [`checkpoint_from_pickle`] and [`checkpoint_to_pickle`] map the payload dict
//! onto the crate's [`Checkpoint`] model; the dict is the checkpoint's
//! `snake_case` JSON shape.
//!
//! # Example
//!
//! ```
//! use openjiuwen_session::checkpoint::pickle;
//! use serde_json::json;
//!
//! let value = json!({"session_id": "s1", "n": 42});
//! let bytes = pickle::dumps(&value).expect("encodes");
//! assert_eq!(pickle::loads(&bytes).expect("decodes"), value);
//! ```
#![allow(clippy::module_name_repetitions)]

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;
use serde_json::{Map, Number, Value};

use super::{validate_id, validate_session_id, Checkpoint};

/// The reserved object key that carries `bytes` payloads as lowercase hex.
pub const BYTES_KEY: &str = "__bytes__";

/// The deepest container nesting [`loads`] accepts.
const MAX_DEPTH: usize = 512;

/// The largest length prefix [`loads`] accepts (512 MiB, mirroring `resp`).
const MAX_LENGTH: usize = 512 * 1024 * 1024;

/// The newest pickle protocol [`loads`] understands.
const PROTOCOL_MAX: u8 = 5;

/// Decodes one Python-pickle stream into a JSON value.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for empty or
/// truncated input, an unsupported opcode or protocol, a non-string dict key, a
/// memo reference that was never stored, an integer or float JSON cannot carry,
/// and input nested deeper than `MAX_DEPTH` (512) levels.
pub fn loads(bytes: &[u8]) -> Result<Value> {
    if bytes.is_empty() {
        return Err(Error::validation("pickle input is empty"));
    }
    Unpickler::new(bytes).run()
}

/// Encodes one JSON value as a protocol-4 Python-pickle stream.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when an object has
/// the reserved single-[`BYTES_KEY`] shape but does not hold a hex string, when an
/// integer does not fit in `i64`, when a float is not finite, or when a number is
/// otherwise not representable.
pub fn dumps(value: &Value) -> Result<Vec<u8>> {
    let mut pickler = Pickler::new();
    pickler.emit(value)?;
    frame(&pickler.output)
}

/// Decodes a Python-written checkpoint payload into a [`Checkpoint`].
///
/// The payload dict uses the checkpoint's `snake_case` JSON field names; missing
/// optional fields take their [`Checkpoint`] defaults and a missing
/// `checkpoint_id` is generated. `session_id` is required.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the payload is
/// not a dict, when `session_id` is missing, empty, or not a string, when
/// `checkpoint_id` is not a string or is an invalid identifier, and for any field
/// that does not fit the [`Checkpoint`] model.
pub fn checkpoint_from_pickle(bytes: &[u8]) -> Result<Checkpoint> {
    let value = loads(bytes)?;
    let Value::Object(map) = &value else {
        return Err(Error::validation(format!(
            "checkpoint pickle payload must be a dict, found {}",
            value_type(&value)
        )));
    };
    match map.get("session_id") {
        Some(Value::String(session_id)) => validate_session_id(session_id)?,
        Some(other) => {
            return Err(Error::validation(format!(
                "checkpoint field `session_id` must be a string, found {}",
                value_type(other)
            )));
        }
        None => {
            return Err(Error::validation(
                "checkpoint pickle payload is missing the required `session_id` field",
            ));
        }
    }
    match map.get("checkpoint_id") {
        Some(Value::String(checkpoint_id)) => {
            if !checkpoint_id.is_empty() {
                validate_id("checkpoint", checkpoint_id)?;
            }
        }
        Some(other) => {
            return Err(Error::validation(format!(
                "checkpoint field `checkpoint_id` must be a string, found {}",
                value_type(other)
            )));
        }
        None => {}
    }
    let mut checkpoint: Checkpoint = serde_json::from_value(value).map_err(|error| {
        Error::validation(format!("invalid checkpoint pickle payload: {error}"))
    })?;
    if checkpoint.checkpoint_id.is_empty() {
        checkpoint.checkpoint_id = uuid_hex();
    }
    Ok(checkpoint)
}

/// Encodes a [`Checkpoint`] as a Python-pickle payload.
///
/// # Errors
///
/// Propagates [`dumps`] failures.
pub fn checkpoint_to_pickle(checkpoint: &Checkpoint) -> Result<Vec<u8>> {
    let value = serde_json::to_value(checkpoint)
        .map_err(|error| Error::validation(format!("checkpoint cannot be encoded: {error}")))?;
    dumps(&value)
}

/// Frames an encoded body as `PROTO 4` + `FRAME` + body + `STOP`.
///
/// `CPython` counts the trailing `STOP` inside the last frame, so the length prefix
/// covers it; that keeps the pinned encoding byte-identical to `CPython`'s.
fn frame(body: &[u8]) -> Result<Vec<u8>> {
    let length = u64::try_from(body.len().saturating_add(1))
        .map_err(|_| Error::validation("pickle payload is too large to frame on this platform"))?;
    let mut output = Vec::with_capacity(body.len() + 12);
    output.extend_from_slice(&[0x80, 0x04, 0x95]);
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(body);
    output.push(b'.');
    Ok(output)
}

/// Answers whether a JSON number looks like an integer.
fn is_integer(number: &Number) -> bool {
    number.is_i64() || number.is_u64()
}

/// Names a JSON value's type for error messages.
fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if is_integer(number) => "integer",
        Value::Number(_) => "float",
        Value::String(_) => "string",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Names the pickle opcode at `opcode` when it is a real opcode this crate
/// deliberately does not decode.
fn opcode_label(opcode: u8) -> &'static str {
    match opcode {
        b'c' => "GLOBAL",
        b'R' => "REDUCE",
        b'b' => "BUILD",
        b'i' => "INST",
        b'o' => "OBJ",
        b'P' => "PERSID",
        b'Q' => "BINPERSID",
        0x82 => "EXT1",
        0x83 => "EXT2",
        0x84 => "EXT4",
        0x8f => "EMPTY_SET",
        0x90 => "ADDITEMS",
        0x91 => "FROZENSET",
        0x92 => "NEWOBJ",
        0x93 => "STACK_GLOBAL",
        0x96 => "BYTEARRAY8",
        0x97 => "NEXT_BUFFER",
        0x98 => "READONLY_BUFFER",
        _ => "unrecognised opcode",
    }
}

/// The current depth of one heap value: 1 for a scalar, one more than its deepest
/// child for a container.
fn value_depth(value: &Value) -> usize {
    match value {
        Value::Array(items) => 1 + items.iter().map(value_depth).max().unwrap_or(0),
        Value::Object(map) => 1 + map.values().map(value_depth).max().unwrap_or(0),
        _ => 1,
    }
}

/// Decodes a latin-1 (byte-per-codepoint) pickle string.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

/// Renders bytes as lowercase hex.
fn hex_encode(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    text
}

/// Decodes one hex digit.
const fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Decodes a lowercase-or-uppercase hex string with an even length.
fn hex_decode(text: &str) -> Option<Vec<u8>> {
    let digits = text.as_bytes();
    if digits.len() % 2 != 0 {
        return None;
    }
    let mut decoded = Vec::with_capacity(digits.len() / 2);
    for pair in digits.chunks_exact(2) {
        let high = hex_digit(pair[0])?;
        let low = hex_digit(pair[1])?;
        decoded.push((high << 4) | low);
    }
    Some(decoded)
}

/// Builds the reserved [`BYTES_KEY`] shape for a `bytes` payload.
fn bytes_value(bytes: &[u8]) -> Value {
    Value::Object(Map::from_iter([(
        BYTES_KEY.to_string(),
        Value::String(hex_encode(bytes)),
    )]))
}

/// Recognizes the reserved [`BYTES_KEY`] shape on the write side.
///
/// A single-key `__bytes__` object is reserved unconditionally, so a non-string or
/// non-hex value is an error rather than a silently mis-encoded dict.
fn bytes_shape(map: &Map<String, Value>) -> Result<Option<Vec<u8>>> {
    if map.len() != 1 {
        return Ok(None);
    }
    let Some(value) = map.get(BYTES_KEY) else {
        return Ok(None);
    };
    let Value::String(text) = value else {
        return Err(Error::validation(format!(
            "reserved `{BYTES_KEY}` shape must hold a hex string, found {}",
            value_type(value)
        )));
    };
    let bytes = hex_decode(text).ok_or_else(|| {
        Error::validation(format!(
            "reserved `{BYTES_KEY}` shape holds invalid hex: {text:?}"
        ))
    })?;
    Ok(Some(bytes))
}

/// Parses a protocol-0 text integer (`I`/`L` opcodes, optional Python 2 `L` suffix).
fn parse_text_int(line: &[u8], offset: usize) -> Result<i64> {
    let text = std::str::from_utf8(line)
        .map_err(|error| Unpickler::error(offset, format!("text integer is not UTF-8: {error}")))?;
    let trimmed = text.strip_suffix('L').unwrap_or(text);
    trimmed.parse::<i64>().map_err(|_| {
        Unpickler::error(
            offset,
            format!("integer {trimmed:?} is not representable as i64"),
        )
    })
}

/// Parses a protocol-0 text float (`F` opcode).
fn parse_text_float(line: &[u8], offset: usize) -> Result<f64> {
    let text = std::str::from_utf8(line)
        .map_err(|error| Unpickler::error(offset, format!("text float is not UTF-8: {error}")))?;
    text.trim()
        .parse::<f64>()
        .map_err(|_| Unpickler::error(offset, format!("malformed text float {text:?}")))
}

/// Parses a protocol-0 text memo index (`PUT`/`GET` opcodes).
fn parse_text_u32(line: &[u8], offset: usize) -> Result<u32> {
    let text = std::str::from_utf8(line)
        .map_err(|error| Unpickler::error(offset, format!("memo index is not UTF-8: {error}")))?;
    text.trim()
        .parse::<u32>()
        .map_err(|_| Unpickler::error(offset, format!("malformed memo index {text:?}")))
}

/// Decodes a little-endian two's-complement pickle long into `i64`.
fn decode_long(bytes: &[u8], offset: usize) -> Result<i64> {
    let mut significant = bytes;
    while significant.len() > 1 {
        let last = significant[significant.len() - 1];
        let previous = significant[significant.len() - 2];
        let redundant =
            (last == 0x00 && previous & 0x80 == 0) || (last == 0xff && previous & 0x80 != 0);
        if !redundant {
            break;
        }
        significant = &significant[..significant.len() - 1];
    }
    if significant.is_empty() {
        return Ok(0);
    }
    if significant.len() > 8 {
        return Err(Unpickler::error(
            offset,
            format!("long integer of {} bytes does not fit in i64", bytes.len()),
        ));
    }
    let negative = significant[significant.len() - 1] & 0x80 != 0;
    let mut magnitude = 0_i128;
    for (index, byte) in significant.iter().enumerate() {
        magnitude |= i128::from(*byte) << (8 * index);
    }
    if negative {
        magnitude -= 1_i128 << (8 * significant.len());
    }
    i64::try_from(magnitude)
        .map_err(|_| Unpickler::error(offset, "long integer does not fit in i64"))
}

/// Reads `digits` hex digits from `line` starting at `index`.
fn read_hex(text: &[u8], index: &mut usize, digits: usize) -> std::result::Result<u32, String> {
    let mut value = 0_u32;
    for _ in 0..digits {
        let Some(&byte) = text.get(*index) else {
            return Err(format!(
                "truncated escape in text string at byte {}",
                *index
            ));
        };
        let Some(digit) = hex_digit(byte) else {
            return Err(format!(
                "invalid hex digit {:?} in text string",
                char::from(byte)
            ));
        };
        value = value * 16 + u32::from(digit);
        *index += 1;
    }
    Ok(value)
}

/// Decodes the body of a protocol-0 `STRING` opcode: a Python `repr` literal of a
/// byte string, whose bytes map to latin-1 code points.
fn decode_repr(line: &[u8], offset: usize) -> Result<String> {
    let fail = |message: String| Unpickler::error(offset, message);
    let Some((&quote, rest)) = line.split_first() else {
        return Err(fail("empty STRING literal".to_string()));
    };
    if quote != b'\'' && quote != b'"' {
        return Err(fail(format!(
            "STRING literal must start with a quote, found {:?}",
            char::from(quote)
        )));
    }
    let mut decoded = Vec::with_capacity(rest.len());
    let mut index = 0;
    while let Some(&byte) = rest.get(index) {
        index += 1;
        if byte == quote {
            if index != rest.len() {
                return Err(fail("trailing bytes after the STRING literal".to_string()));
            }
            return Ok(latin1(&decoded));
        }
        if byte != b'\\' {
            decoded.push(byte);
            continue;
        }
        let Some(&escaped) = rest.get(index) else {
            return Err(fail("STRING literal ends with a backslash".to_string()));
        };
        index += 1;
        match escaped {
            b'n' => decoded.push(b'\n'),
            b'r' => decoded.push(b'\r'),
            b't' => decoded.push(b'\t'),
            b'\\' => decoded.push(b'\\'),
            b'\'' => decoded.push(b'\''),
            b'"' => decoded.push(b'"'),
            b'x' => {
                let value = read_hex(rest, &mut index, 2).map_err(fail)?;
                decoded.push(u8::try_from(value).map_err(|_| fail("octet out of range".into()))?);
            }
            b'0'..=b'7' => {
                let mut value = u32::from(escaped - b'0');
                let mut extra = 0;
                while extra < 2 {
                    match rest.get(index) {
                        Some(&digit @ b'0'..=b'7') => {
                            value = value * 8 + u32::from(digit - b'0');
                            index += 1;
                            extra += 1;
                        }
                        _ => break,
                    }
                }
                decoded.push(
                    u8::try_from(value).map_err(|_| fail("octal escape out of range".into()))?,
                );
            }
            other => decoded.push(other),
        }
    }
    Err(fail("unterminated STRING literal".to_string()))
}

/// Decodes a protocol-0 `UNICODE` line (raw-unicode-escape).
fn decode_raw_unicode_escape(line: &[u8], offset: usize) -> Result<String> {
    let fail = |message: String| Unpickler::error(offset, message);
    let mut text = String::with_capacity(line.len());
    let mut index = 0;
    while let Some(&byte) = line.get(index) {
        index += 1;
        if byte != b'\\' {
            text.push(char::from(byte));
            continue;
        }
        let Some(&escape) = line.get(index) else {
            return Err(fail(
                "raw-unicode-escape line ends with a backslash".to_string(),
            ));
        };
        index += 1;
        match escape {
            b'u' => {
                let value = read_hex(line, &mut index, 4).map_err(fail)?;
                text.push(
                    char::from_u32(value)
                        .ok_or_else(|| fail(format!("invalid code point U+{value:04X}")))?,
                );
            }
            b'U' => {
                let value = read_hex(line, &mut index, 8).map_err(fail)?;
                text.push(
                    char::from_u32(value)
                        .ok_or_else(|| fail(format!("invalid code point U+{value:08X}")))?,
                );
            }
            b'x' => {
                let value = read_hex(line, &mut index, 2).map_err(fail)?;
                text.push(char::from_u32(value).unwrap_or('\u{fffd}'));
            }
            b'n' => text.push('\n'),
            b'r' => text.push('\r'),
            b't' => text.push('\t'),
            b'\\' => text.push('\\'),
            other => text.push(char::from(other)),
        }
    }
    Ok(text)
}

/// One entry of the unpickler's value stack.
#[derive(Clone, Copy)]
enum Entry {
    /// A `MARK` pushed by the `(` opcode.
    Mark,
    /// An index into the unpickler's value heap.
    Value(usize),
}

/// The unpickling state machine.
///
/// Values live in a heap (an arena) rather than directly on the stack so that a
/// memo entry can alias the object it memoized: a container that is filled after
/// being memoized is still seen populated through `BINGET`, matching `CPython`.
struct Unpickler<'a> {
    input: &'a [u8],
    position: usize,
    stack: Vec<Entry>,
    heap: Vec<Value>,
    depths: Vec<usize>,
    memo: BTreeMap<u32, usize>,
}

impl<'a> Unpickler<'a> {
    /// Creates an unpickler over `input`.
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            position: 0,
            stack: Vec::new(),
            heap: Vec::new(),
            depths: Vec::new(),
            memo: BTreeMap::new(),
        }
    }

    /// Runs the opcode loop until `STOP`.
    fn run(mut self) -> Result<Value> {
        loop {
            let offset = self.position;
            let opcode = self.read_u8(offset)?;
            if opcode == b'.' {
                return self.finish(offset);
            }
            self.execute(opcode, offset)?;
        }
    }

    /// Builds one validation error that names the offending offset.
    fn error(offset: usize, message: impl fmt::Display) -> Error {
        Error::validation(format!("pickle: {message} at offset {offset}"))
    }

    /// Executes one opcode other than `STOP`.
    // One flat dispatch table is easier to audit against CPython's `pickle.py`
    // than a split across helper functions.
    #[allow(clippy::too_many_lines)]
    fn execute(&mut self, opcode: u8, offset: usize) -> Result<()> {
        match opcode {
            0x80 => self.read_protocol(offset),
            0x95 => self.read_frame(offset),
            b'(' => {
                self.stack.push(Entry::Mark);
                Ok(())
            }
            b'N' => self.push(Value::Null),
            0x88 => self.push(Value::Bool(true)),
            0x89 => self.push(Value::Bool(false)),
            b'I' | b'L' => {
                let line = self.read_line(offset)?;
                let value = parse_text_int(line, offset)?;
                self.push(Value::from(value))
            }
            b'F' => {
                let line = self.read_line(offset)?;
                let value = parse_text_float(line, offset)?;
                self.push_number(value, offset)
            }
            b'K' => {
                let value = self.read_u8(offset)?;
                self.push(Value::from(value))
            }
            b'M' => {
                let value = self.read_u16_le(offset)?;
                self.push(Value::from(value))
            }
            b'J' => {
                let value = self.read_i32_le(offset)?;
                self.push(Value::from(value))
            }
            0x8a => {
                let length = usize::from(self.read_u8(offset)?);
                let bytes = self.read_exact(length, offset)?;
                let value = decode_long(bytes, offset)?;
                self.push(Value::from(value))
            }
            0x8b => {
                let length = self.read_length_u32(offset)?;
                let bytes = self.read_exact(length, offset)?;
                let value = decode_long(bytes, offset)?;
                self.push(Value::from(value))
            }
            b'G' => {
                let value = self.read_f64_be(offset)?;
                self.push_number(value, offset)
            }
            0x8c => {
                let length = usize::from(self.read_u8(offset)?);
                let text = self.read_text(length, offset)?;
                self.push(Value::String(text))
            }
            b'X' => {
                let length = self.read_length_u32(offset)?;
                let text = self.read_text(length, offset)?;
                self.push(Value::String(text))
            }
            0x8d => {
                let length = self.read_length_u64(offset)?;
                let text = self.read_text(length, offset)?;
                self.push(Value::String(text))
            }
            b'V' => {
                let line = self.read_line(offset)?;
                let text = decode_raw_unicode_escape(line, offset)?;
                self.push(Value::String(text))
            }
            b'U' => {
                let length = usize::from(self.read_u8(offset)?);
                let bytes = self.read_exact(length, offset)?;
                self.push(Value::String(latin1(bytes)))
            }
            b'T' => {
                let length = self.read_length_u32(offset)?;
                let bytes = self.read_exact(length, offset)?;
                self.push(Value::String(latin1(bytes)))
            }
            b'S' => {
                let line = self.read_line(offset)?;
                let text = decode_repr(line, offset)?;
                self.push(Value::String(text))
            }
            b'C' => {
                let length = usize::from(self.read_u8(offset)?);
                let bytes = self.read_exact(length, offset)?;
                self.push(bytes_value(bytes))
            }
            b'B' => {
                let length = self.read_length_u32(offset)?;
                let bytes = self.read_exact(length, offset)?;
                self.push(bytes_value(bytes))
            }
            0x8e => {
                let length = self.read_length_u64(offset)?;
                let bytes = self.read_exact(length, offset)?;
                self.push(bytes_value(bytes))
            }
            b'}' => self.push(Value::Object(Map::new())),
            b']' | b')' => self.push(Value::Array(Vec::new())),
            b'd' => self.build_dict(offset),
            b'l' | b't' => self.build_list(offset),
            0x85 => self.build_tuple(1, offset),
            0x86 => self.build_tuple(2, offset),
            0x87 => self.build_tuple(3, offset),
            b's' => self.set_item(offset),
            b'u' => self.set_items(offset),
            b'a' => self.append(offset),
            b'e' => self.append_items(offset),
            b'q' => {
                let index = u32::from(self.read_u8(offset)?);
                self.memo_put(index, offset)
            }
            b'r' => {
                let index = self.read_u32_le(offset)?;
                self.memo_put(index, offset)
            }
            b'h' => {
                let index = u32::from(self.read_u8(offset)?);
                self.memo_get(index, offset)
            }
            b'j' => {
                let index = self.read_u32_le(offset)?;
                self.memo_get(index, offset)
            }
            0x94 => self.memoize(offset),
            b'p' => {
                let line = self.read_line(offset)?;
                let index = parse_text_u32(line, offset)?;
                self.memo_put(index, offset)
            }
            b'g' => {
                let line = self.read_line(offset)?;
                let index = parse_text_u32(line, offset)?;
                self.memo_get(index, offset)
            }
            _ => Err(Self::error(
                offset,
                format!(
                    "unsupported opcode 0x{opcode:02x} ({})",
                    opcode_label(opcode)
                ),
            )),
        }
    }

    /// Consumes the single value left on the stack when `STOP` is reached.
    fn finish(&mut self, offset: usize) -> Result<Value> {
        let mut values = Vec::new();
        for entry in self.stack.drain(..) {
            match entry {
                Entry::Value(index) => values.push(index),
                Entry::Mark => {
                    return Err(Self::error(offset, "STOP left a MARK on the pickle stack"));
                }
            }
        }
        match values.as_slice() {
            [index] => Ok(self.heap[*index].clone()),
            _ => Err(Self::error(
                offset,
                format!("STOP expects exactly one item, found {}", values.len()),
            )),
        }
    }

    /// Reads one byte, or reports the end of the input.
    fn read_u8(&mut self, offset: usize) -> Result<u8> {
        let Some(&byte) = self.input.get(self.position) else {
            return Err(Self::error(offset, "unexpected end of input"));
        };
        self.position += 1;
        Ok(byte)
    }

    /// Reads exactly `length` bytes, enforcing [`MAX_LENGTH`].
    fn read_exact(&mut self, length: usize, offset: usize) -> Result<&'a [u8]> {
        if length > MAX_LENGTH {
            return Err(Self::error(
                offset,
                format!("length {length} exceeds the {MAX_LENGTH}-byte limit"),
            ));
        }
        let end = self.position.saturating_add(length);
        let Some(bytes) = self.input.get(self.position..end) else {
            return Err(Self::error(
                offset,
                format!(
                    "truncated input: needed {length} bytes, {} remain",
                    self.input.len().saturating_sub(self.position)
                ),
            ));
        };
        self.position = end;
        Ok(bytes)
    }

    /// Reads one `\n`-terminated line without the terminator.
    fn read_line(&mut self, offset: usize) -> Result<&'a [u8]> {
        let rest = self.input.get(self.position..).unwrap_or_default();
        let Some(end) = rest.iter().position(|byte| *byte == b'\n') else {
            return Err(Self::error(offset, "unterminated text line"));
        };
        let line = rest.get(..end).unwrap_or_default();
        self.position += end + 1;
        Ok(line)
    }

    /// Reads a little-endian `u16`.
    fn read_u16_le(&mut self, offset: usize) -> Result<u16> {
        let bytes = self.read_exact(2, offset)?;
        let array: [u8; 2] = bytes
            .try_into()
            .map_err(|_| Self::error(offset, "expected two bytes"))?;
        Ok(u16::from_le_bytes(array))
    }

    /// Reads a little-endian `i32`.
    fn read_i32_le(&mut self, offset: usize) -> Result<i32> {
        let bytes = self.read_exact(4, offset)?;
        let array: [u8; 4] = bytes
            .try_into()
            .map_err(|_| Self::error(offset, "expected four bytes"))?;
        Ok(i32::from_le_bytes(array))
    }

    /// Reads a little-endian `u32`.
    fn read_u32_le(&mut self, offset: usize) -> Result<u32> {
        let bytes = self.read_exact(4, offset)?;
        let array: [u8; 4] = bytes
            .try_into()
            .map_err(|_| Self::error(offset, "expected four bytes"))?;
        Ok(u32::from_le_bytes(array))
    }

    /// Reads a little-endian `u64`.
    fn read_u64_le(&mut self, offset: usize) -> Result<u64> {
        let bytes = self.read_exact(8, offset)?;
        let array: [u8; 8] = bytes
            .try_into()
            .map_err(|_| Self::error(offset, "expected eight bytes"))?;
        Ok(u64::from_le_bytes(array))
    }

    /// Reads a big-endian `f64` (`BINFLOAT`).
    fn read_f64_be(&mut self, offset: usize) -> Result<f64> {
        let bytes = self.read_exact(8, offset)?;
        let array: [u8; 8] = bytes
            .try_into()
            .map_err(|_| Self::error(offset, "expected eight bytes"))?;
        Ok(f64::from_be_bytes(array))
    }

    /// Reads a `u32` length prefix and converts it to `usize`.
    fn read_length_u32(&mut self, offset: usize) -> Result<usize> {
        let length = self.read_u32_le(offset)?;
        usize::try_from(length)
            .map_err(|_| Self::error(offset, format!("length {length} does not fit the platform")))
    }

    /// Reads a `u64` length prefix and converts it to `usize`.
    fn read_length_u64(&mut self, offset: usize) -> Result<usize> {
        let length = self.read_u64_le(offset)?;
        let length = usize::try_from(length).map_err(|_| {
            Self::error(offset, format!("length {length} does not fit the platform"))
        })?;
        if length > MAX_LENGTH {
            return Err(Self::error(
                offset,
                format!("length {length} exceeds the {MAX_LENGTH}-byte limit"),
            ));
        }
        Ok(length)
    }

    /// Reads `length` bytes as a UTF-8 string.
    fn read_text(&mut self, length: usize, offset: usize) -> Result<String> {
        let bytes = self.read_exact(length, offset)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|error| Self::error(offset, format!("unicode string is not UTF-8: {error}")))
    }

    /// Reads a text `PROTO` header.
    fn read_protocol(&mut self, offset: usize) -> Result<()> {
        let protocol = self.read_u8(offset)?;
        if protocol > PROTOCOL_MAX {
            return Err(Self::error(
                offset,
                format!("unsupported protocol {protocol}"),
            ));
        }
        Ok(())
    }

    /// Reads a `FRAME` header and checks it against the remaining input.
    fn read_frame(&mut self, offset: usize) -> Result<()> {
        let length = self.read_u64_le(offset)?;
        let remaining = self.input.len().saturating_sub(self.position);
        if u64::try_from(remaining).unwrap_or(u64::MAX) < length {
            return Err(Self::error(
                offset,
                format!("frame length {length} exceeds the {remaining} remaining bytes"),
            ));
        }
        Ok(())
    }

    /// Pushes one value, enforcing [`MAX_DEPTH`].
    fn push(&mut self, value: Value) -> Result<()> {
        let depth = value_depth(&value);
        if depth > MAX_DEPTH {
            return Err(Error::validation(format!(
                "pickle: nesting depth {depth} exceeds the {MAX_DEPTH}-level limit"
            )));
        }
        let index = self.heap.len();
        self.heap.push(value);
        self.depths.push(depth);
        self.stack.push(Entry::Value(index));
        Ok(())
    }

    /// Pushes one finite number.
    fn push_number(&mut self, value: f64, offset: usize) -> Result<()> {
        let Some(number) = Number::from_f64(value) else {
            return Err(Self::error(
                offset,
                format!("float {value} cannot be represented in JSON"),
            ));
        };
        self.push(Value::Number(number))
    }

    /// Pops one value index.
    fn pop(&mut self, offset: usize) -> Result<usize> {
        match self.stack.pop() {
            Some(Entry::Value(index)) => Ok(index),
            Some(Entry::Mark) => Err(Self::error(
                offset,
                "opcode found a MARK where a value was required",
            )),
            None => Err(Self::error(offset, "value stack underflow")),
        }
    }

    /// Pops every value above the innermost `MARK`.
    fn pop_to_mark(&mut self, offset: usize) -> Result<Vec<usize>> {
        let mut items = Vec::new();
        loop {
            match self.stack.pop() {
                Some(Entry::Value(index)) => items.push(index),
                Some(Entry::Mark) => {
                    items.reverse();
                    return Ok(items);
                }
                None => return Err(Self::error(offset, "MARK not found")),
            }
        }
    }

    /// Deepens a container's recorded depth for one inserted child.
    fn grow_depth(&mut self, target: usize, child_depth: usize, offset: usize) -> Result<()> {
        let depth = self.depths[target].max(child_depth.saturating_add(1));
        if depth > MAX_DEPTH {
            return Err(Self::error(
                offset,
                format!("nesting depth {depth} exceeds the {MAX_DEPTH}-level limit"),
            ));
        }
        self.depths[target] = depth;
        Ok(())
    }

    /// Inserts one key/value pair into the dict at `target`.
    fn insert_pair(
        &mut self,
        target: usize,
        key_index: usize,
        value_index: usize,
        offset: usize,
    ) -> Result<()> {
        let key = match self.heap.get(key_index) {
            Some(Value::String(key)) => key.clone(),
            Some(other) => {
                return Err(Self::error(
                    offset,
                    format!("dict key must be a string, found {}", value_type(other)),
                ));
            }
            None => return Err(Self::error(offset, "dict key is missing")),
        };
        let child = self.heap.get(value_index).cloned().unwrap_or(Value::Null);
        let child_depth = self.depths.get(value_index).copied().unwrap_or(1);
        match self.heap.get_mut(target) {
            Some(Value::Object(map)) => {
                map.insert(key, child);
            }
            Some(other) => {
                return Err(Self::error(
                    offset,
                    format!("opcode needs a dict, found {}", value_type(other)),
                ));
            }
            None => return Err(Self::error(offset, "dict target is missing")),
        }
        self.grow_depth(target, child_depth, offset)
    }

    /// Appends one value to the list at the top of the stack.
    fn append(&mut self, offset: usize) -> Result<()> {
        let value_index = self.pop(offset)?;
        let target = self.pop(offset)?;
        let child = self.heap.get(value_index).cloned().unwrap_or(Value::Null);
        let child_depth = self.depths.get(value_index).copied().unwrap_or(1);
        match self.heap.get_mut(target) {
            Some(Value::Array(items)) => items.push(child),
            Some(other) => {
                return Err(Self::error(
                    offset,
                    format!("APPEND needs a list, found {}", value_type(other)),
                ));
            }
            None => return Err(Self::error(offset, "list target is missing")),
        }
        self.grow_depth(target, child_depth, offset)?;
        self.stack.push(Entry::Value(target));
        Ok(())
    }

    /// Appends every value above the innermost `MARK` to the list below it.
    fn append_items(&mut self, offset: usize) -> Result<()> {
        let items = self.pop_to_mark(offset)?;
        let target = self.pop(offset)?;
        if !matches!(self.heap.get(target), Some(Value::Array(_))) {
            let found = self.heap.get(target).map_or("nothing", value_type);
            return Err(Self::error(
                offset,
                format!("APPENDS needs a list, found {found}"),
            ));
        }
        for index in items {
            let child = self.heap.get(index).cloned().unwrap_or(Value::Null);
            let child_depth = self.depths.get(index).copied().unwrap_or(1);
            if let Some(Value::Array(array)) = self.heap.get_mut(target) {
                array.push(child);
            }
            self.grow_depth(target, child_depth, offset)?;
        }
        self.stack.push(Entry::Value(target));
        Ok(())
    }

    /// Inserts the top key/value pair into the dict below it (`SETITEM`).
    fn set_item(&mut self, offset: usize) -> Result<()> {
        let value_index = self.pop(offset)?;
        let key_index = self.pop(offset)?;
        let target = self.pop(offset)?;
        self.insert_pair(target, key_index, value_index, offset)?;
        self.stack.push(Entry::Value(target));
        Ok(())
    }

    /// Inserts every key/value pair above the innermost `MARK` (`SETITEMS`).
    fn set_items(&mut self, offset: usize) -> Result<()> {
        let items = self.pop_to_mark(offset)?;
        let target = self.pop(offset)?;
        if items.len() % 2 != 0 {
            return Err(Self::error(
                offset,
                format!(
                    "SETITEMS needs key/value pairs, found {} items",
                    items.len()
                ),
            ));
        }
        for pair in items.chunks_exact(2) {
            self.insert_pair(target, pair[0], pair[1], offset)?;
        }
        self.stack.push(Entry::Value(target));
        Ok(())
    }

    /// Builds a dict from every key/value pair above the innermost `MARK` (`DICT`).
    fn build_dict(&mut self, offset: usize) -> Result<()> {
        let items = self.pop_to_mark(offset)?;
        let target = self.pop(offset)?;
        if !matches!(self.heap.get(target), Some(Value::Object(_))) {
            return Err(Self::error(offset, "DICT needs a dict below its MARK"));
        }
        if items.len() % 2 != 0 {
            return Err(Self::error(
                offset,
                format!("DICT needs key/value pairs, found {} items", items.len()),
            ));
        }
        for pair in items.chunks_exact(2) {
            self.insert_pair(target, pair[0], pair[1], offset)?;
        }
        self.stack.push(Entry::Value(target));
        Ok(())
    }

    /// Builds a list from every value above the innermost `MARK` (`LIST`/`TUPLE`).
    fn build_list(&mut self, offset: usize) -> Result<()> {
        let items = self.pop_to_mark(offset)?;
        let values = items
            .iter()
            .map(|index| self.heap.get(*index).cloned().unwrap_or(Value::Null))
            .collect();
        self.push(Value::Array(values))
    }

    /// Builds a fixed-size tuple from the top `size` values (`TUPLE1`/`2`/`3`).
    fn build_tuple(&mut self, size: usize, offset: usize) -> Result<()> {
        let mut values = Vec::with_capacity(size);
        for _ in 0..size {
            let index = self.pop(offset)?;
            values.push(self.heap.get(index).cloned().unwrap_or(Value::Null));
        }
        values.reverse();
        self.push(Value::Array(values))
    }

    /// Memoizes the value at the top of the stack without popping it.
    fn memoize(&mut self, offset: usize) -> Result<()> {
        let Some(Entry::Value(index)) = self.stack.last().copied() else {
            return Err(Self::error(
                offset,
                "MEMOIZE needs a value on top of the stack",
            ));
        };
        let next = u32::try_from(self.memo.len()).unwrap_or(u32::MAX);
        self.memo.insert(next, index);
        Ok(())
    }

    /// Stores the value at the top of the stack under memo index `index`.
    fn memo_put(&mut self, index: u32, offset: usize) -> Result<()> {
        let Some(Entry::Value(value_index)) = self.stack.last().copied() else {
            return Err(Self::error(
                offset,
                "memo PUT needs a value on top of the stack",
            ));
        };
        self.memo.insert(index, value_index);
        Ok(())
    }

    /// Pushes the value memoized under `index`.
    fn memo_get(&mut self, index: u32, offset: usize) -> Result<()> {
        let Some(&value_index) = self.memo.get(&index) else {
            return Err(Self::error(
                offset,
                format!("memo reference {index} was never stored"),
            ));
        };
        self.stack.push(Entry::Value(value_index));
        Ok(())
    }
}

/// The pickler: one output buffer plus the string memo.
struct Pickler {
    output: Vec<u8>,
    next_memo: u32,
    strings: HashMap<String, u32>,
}

impl Pickler {
    /// Creates an empty pickler.
    fn new() -> Self {
        Self {
            output: Vec::new(),
            next_memo: 0,
            strings: HashMap::new(),
        }
    }

    /// Appends raw opcode bytes.
    fn write(&mut self, bytes: &[u8]) {
        self.output.extend_from_slice(bytes);
    }

    /// Encodes one JSON value.
    fn emit(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Null => {
                self.write(b"N");
                Ok(())
            }
            Value::Bool(true) => {
                self.write(&[0x88]);
                Ok(())
            }
            Value::Bool(false) => {
                self.write(&[0x89]);
                Ok(())
            }
            Value::Number(number) => self.emit_number(number),
            Value::String(text) => {
                self.emit_string(text);
                Ok(())
            }
            Value::Array(items) => {
                self.write(b"]");
                self.memoize();
                for item in items {
                    self.emit(item)?;
                    self.write(b"a");
                }
                Ok(())
            }
            Value::Object(map) => {
                if let Some(bytes) = bytes_shape(map)? {
                    self.emit_bytes(&bytes);
                    return Ok(());
                }
                self.write(b"}");
                self.memoize();
                for (key, item) in map {
                    self.emit_string(key);
                    self.emit(item)?;
                    self.write(b"s");
                }
                Ok(())
            }
        }
    }

    /// Encodes one number with the shortest integer opcode and `BINFLOAT` otherwise.
    fn emit_number(&mut self, number: &Number) -> Result<()> {
        if let Some(value) = number.as_i64() {
            self.emit_int(value);
            return Ok(());
        }
        if let Some(value) = number.as_u64() {
            let signed = i64::try_from(value).map_err(|_| {
                Error::validation(format!("pickle integers must fit in i64, found {value}"))
            })?;
            self.emit_int(signed);
            return Ok(());
        }
        let Some(value) = number.as_f64() else {
            return Err(Error::validation(format!(
                "pickle cannot represent the JSON number {number}"
            )));
        };
        if !value.is_finite() {
            return Err(Error::validation(
                "pickle cannot represent a non-finite float",
            ));
        }
        self.write(b"G");
        self.write(&value.to_be_bytes());
        Ok(())
    }

    /// Encodes a signed integer.
    fn emit_int(&mut self, value: i64) {
        if let Ok(byte) = u8::try_from(value) {
            self.write(&[b'K', byte]);
            return;
        }
        if let Ok(short) = u16::try_from(value) {
            self.write(b"M");
            self.write(&short.to_le_bytes());
            return;
        }
        if let Ok(int) = i32::try_from(value) {
            self.write(b"J");
            self.write(&int.to_le_bytes());
            return;
        }
        self.emit_long(&encode_long_i64(value));
    }

    /// Emits one `LONG1` value (the encodings here never exceed 255 bytes).
    fn emit_long(&mut self, bytes: &[u8]) {
        let length = u8::try_from(bytes.len()).unwrap_or(u8::MAX);
        self.write(&[0x8a, length]);
        self.write(bytes);
    }

    /// Encodes one UTF-8 string, reusing the memo for repeated content.
    fn emit_string(&mut self, text: &str) {
        if let Some(index) = self.strings.get(text).copied() {
            write_binget(&mut self.output, index);
            return;
        }
        let bytes = text.as_bytes();
        if let Ok(length) = u8::try_from(bytes.len()) {
            self.write(&[0x8c, length]);
        } else if let Ok(length) = u32::try_from(bytes.len()) {
            self.write(b"X");
            self.write(&length.to_le_bytes());
        } else {
            self.write(&[0x8d]);
            let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
            self.write(&length.to_le_bytes());
        }
        self.write(bytes);
        self.write(&[0x94]);
        let index = self.next_memo;
        self.next_memo = self.next_memo.saturating_add(1);
        self.strings.entry(text.to_owned()).or_insert(index);
    }

    /// Encodes one `bytes` payload.
    fn emit_bytes(&mut self, bytes: &[u8]) {
        if let Ok(length) = u8::try_from(bytes.len()) {
            self.write(&[b'C', length]);
        } else if let Ok(length) = u32::try_from(bytes.len()) {
            self.write(b"B");
            self.write(&length.to_le_bytes());
        } else {
            self.write(&[0x8e]);
            let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
            self.write(&length.to_le_bytes());
        }
        self.write(bytes);
    }

    /// Records one container in the memo.
    fn memoize(&mut self) {
        self.write(&[0x94]);
        self.next_memo = self.next_memo.saturating_add(1);
    }
}

/// Writes a `BINGET`/`LONG_BINGET` reference to memo `index`.
fn write_binget(output: &mut Vec<u8>, index: u32) {
    if let Ok(short) = u8::try_from(index) {
        output.extend_from_slice(&[b'h', short]);
    } else {
        output.push(b'j');
        output.extend_from_slice(&index.to_le_bytes());
    }
}

/// Encodes an `i64` as a minimal little-endian two's-complement long.
fn encode_long_i64(value: i64) -> Vec<u8> {
    let mut bytes = value.to_le_bytes().to_vec();
    while bytes.len() > 1 {
        let last = bytes[bytes.len() - 1];
        let previous = bytes[bytes.len() - 2];
        let redundant =
            (last == 0x00 && previous & 0x80 == 0) || (last == 0xff && previous & 0x80 != 0);
        if !redundant {
            break;
        }
        bytes.pop();
    }
    bytes
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use serde_json::json;

    #[test]
    fn hex_round_trips() {
        assert_eq!(hex_encode(&[0x00, 0x7f, 0xff]), "007fff");
        assert_eq!(hex_decode("007FFF"), Some(vec![0x00, 0x7f, 0xff]));
        assert_eq!(hex_decode("abc"), None);
        assert_eq!(hex_decode("zz"), None);
        assert_eq!(hex_decode(""), Some(Vec::new()));
    }

    #[test]
    fn latin1_maps_bytes_to_code_points() {
        assert_eq!(latin1(&[b'a', 0xe9, 0xff]), "a\u{e9}\u{ff}");
    }

    #[test]
    fn long_decoding_sign_extends() {
        assert_eq!(decode_long(&[], 0).unwrap(), 0);
        assert_eq!(decode_long(&[0x2a], 0).unwrap(), 42);
        assert_eq!(decode_long(&[0xff], 0).unwrap(), -1);
        assert_eq!(decode_long(&[0x00, 0x00, 0x01], 0).unwrap(), 65536);
        assert_eq!(
            decode_long(&[0x00, 0x00, 0x00, 0x00, 0x01], 0).unwrap(),
            4_294_967_296
        );
        assert_eq!(
            decode_long(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80], 0).unwrap(),
            i64::MIN
        );
        assert!(decode_long(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01], 0).is_err());
    }

    #[test]
    fn text_parsers_accept_python_forms() {
        assert_eq!(parse_text_int(b"42", 0).unwrap(), 42);
        assert_eq!(parse_text_int(b"-7", 0).unwrap(), -7);
        assert_eq!(parse_text_int(b"010", 0).unwrap(), 10);
        assert_eq!(parse_text_int(b"42L", 0).unwrap(), 42);
        assert!(parse_text_int(b"4294967296", 0).is_ok());
        assert!(parse_text_int(b"9223372036854775808", 0).is_err());
        assert_eq!(parse_text_float(b"1.5", 0).unwrap(), 1.5);
        assert!(parse_text_float(b"nan", 0).unwrap().is_nan());
        assert_eq!(parse_text_u32(b"3", 0).unwrap(), 3);
    }

    #[test]
    fn repr_decoding_handles_escapes() {
        assert_eq!(decode_repr(b"'ab'", 0).unwrap(), "ab");
        assert_eq!(decode_repr(b"'a\\x00'", 0).unwrap(), "a\u{0}");
        assert_eq!(decode_repr(b"'a\\n'", 0).unwrap(), "a\n");
        assert_eq!(decode_repr(b"'\\101'", 0).unwrap(), "A");
        assert!(decode_repr(b"'ab", 0).is_err());
        assert!(decode_repr(b"ab'", 0).is_err());
    }

    #[test]
    fn raw_unicode_escape_decodes() {
        assert_eq!(
            decode_raw_unicode_escape(b"h\xe9llo", 0).unwrap(),
            "h\u{e9}llo"
        );
        assert_eq!(decode_raw_unicode_escape(b"a\\u000ab", 0).unwrap(), "a\nb");
        assert_eq!(
            decode_raw_unicode_escape(b"\\U0001F600", 0).unwrap(),
            "\u{1f600}"
        );
    }

    #[test]
    fn round_trips_json_shapes() {
        let values = [
            Value::Null,
            json!(true),
            json!(false),
            json!(0),
            json!(-1),
            json!(255),
            json!(65536),
            json!(-4_294_967_296_i64),
            json!(i64::MIN),
            json!(1.5),
            json!(""),
            json!("héllo"),
            json!(["a", "a", [1, 2], {}]),
            json!({"k": {"n": null}}),
            json!({"__bytes__": "00ff"}),
        ];
        for value in values {
            let bytes = dumps(&value).expect("encodes");
            let decoded = loads(&bytes).unwrap_or_else(|error| panic!("{value} failed: {error}"));
            assert_eq!(decoded, value, "{value}");
        }
    }

    #[test]
    fn rejects_reserved_bytes_shape_with_bad_hex() {
        let error = dumps(&json!({"__bytes__": "nope"})).expect_err("invalid hex");
        assert!(error.to_string().contains(BYTES_KEY));
    }

    #[test]
    fn rejects_integers_above_i64() {
        let error = dumps(&json!(u64::MAX)).expect_err("does not fit");
        assert!(error.to_string().contains("i64"));
    }

    #[test]
    fn rejects_nesting_beyond_the_cap() {
        let mut value = json!(1);
        for _ in 0..MAX_DEPTH {
            value = Value::Array(vec![value]);
        }
        let bytes = dumps(&value).expect("encodes");
        let error = loads(&bytes).expect_err("too deep");
        assert!(error.to_string().contains("nesting depth"));
    }
}
