//! Redis serialization protocol (`RESP2`/`RESP3`) encoding and incremental decoding.
//!
//! [`encode_command`] writes a command as a `RESP` array of bulk strings, which is
//! byte-for-byte safe for binary arguments (embedded NULs, `\r`, `\n`, non-UTF-8).
//! [`parse_reply`] is a total, incremental parser: it reports `Ok(None)` whenever
//! the input does not yet hold a complete reply, so a caller can accumulate bytes
//! from a socket and retry without copying partial state.
//!
//! Deliberate limits keep the parser total on hostile input: nesting deeper than
//! `MAX_DEPTH`, element counts above `MAX_ELEMENTS`, and bulk payloads above
//! `MAX_BULK_LEN` are rejected with [`openjiuwen_common::error::ErrorKind::Transport`]
//! instead of allocating or recursing without bound.
//!
//! `RESP3` attributes (`|N`) are metadata: the parser consumes the `N` key/value
//! pairs and returns the reply that follows them, so a request/response client
//! sees the value and never the attribute bag.
//! `RESP3` streamed strings (`;N`) are decoded by concatenating their chunks into
//! one binary-safe [`Reply::Bulk`] up to the `;0` terminator.
//!
//! Server error replies (`-...`) are returned as [`Reply::Error`], including
//! `-MOVED`/`-ASK` redirections; mapping those to typed errors is the
//! [`crate::redis::client`] layer's job.
use openjiuwen_common::error::{Error, Result};
/// Maximum accepted nesting depth for aggregated replies.
const MAX_DEPTH: usize = 128;
/// Maximum accepted element count for a single array, set, map, or push reply.
const MAX_ELEMENTS: i64 = 16 * 1024 * 1024;
/// Maximum accepted payload length for a bulk or verbatim string, in bytes.
const MAX_BULK_LEN: usize = 512 * 1024 * 1024;
/// A decoded `RESP2`/`RESP3` reply value.
///
/// `RESP2` produces [`Reply::Simple`], [`Reply::Error`], [`Reply::Integer`],
/// [`Reply::Bulk`], [`Reply::Array`], and [`Reply::Null`] (for a null bulk string
/// or null array). `RESP3` additionally produces [`Reply::Boolean`],
/// [`Reply::Double`], [`Reply::Map`], [`Reply::Set`], and [`Reply::Push`]; a
/// `RESP3` verbatim string is decoded to [`Reply::Bulk`] without its three-byte
/// encoding prefix, and a big number that fits in an `i64` becomes
/// [`Reply::Integer`] while a larger one keeps its literal digits as
/// [`Reply::Bulk`].
#[derive(Debug, Clone, PartialEq)]
pub enum Reply {
    /// `+simple string`
    Simple(String),
    /// `-error message`; `MOVED`/`ASK` redirections arrive here too.
    Error(String),
    /// `:integer`
    Integer(i64),
    /// `$bulk string`, including empty bulk strings; binary-safe.
    Bulk(Vec<u8>),
    /// A null bulk string (`$-1`), a null array (`*-1`), or `RESP3` `_`.
    Null,
    /// `*array`
    Array(Vec<Reply>),
    /// `RESP3` `#t` / `#f`
    Boolean(bool),
    /// `RESP3` `,double`, including `inf`, `-inf`, and `nan`.
    Double(f64),
    /// `RESP3` `%map` as ordered key/value pairs.
    Map(Vec<(Reply, Reply)>),
    /// `RESP3` `~set` as an unordered collection carried in arrival order.
    Set(Vec<Reply>),
    /// `RESP3` `>push` (out-of-band messages).
    Push(Vec<Reply>),
}
impl Reply {
    /// The `RESP` type name, used in type-mismatch messages.
    #[must_use]
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Self::Simple(_) => "simple string",
            Self::Error(_) => "error",
            Self::Integer(_) => "integer",
            Self::Bulk(_) => "bulk string",
            Self::Null => "null",
            Self::Array(_) => "array",
            Self::Boolean(_) => "boolean",
            Self::Double(_) => "double",
            Self::Map(_) => "map",
            Self::Set(_) => "set",
            Self::Push(_) => "push",
        }
    }
    /// Whether this is a null reply.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    /// Whether this is a server error reply.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
    /// Borrows the payload as text (`Simple` or UTF-8 `Bulk`).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the reply is
    /// another type, or when a bulk payload is not valid UTF-8.
    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::Simple(text) => Ok(text),
            Self::Bulk(bytes) => std::str::from_utf8(bytes).map_err(|error| {
                Error::validation(format!("redis bulk reply is not valid UTF-8: {error}"))
            }),
            other => Err(type_mismatch("a string", other)),
        }
    }
    /// Like [`Reply::as_str`], but maps [`Reply::Null`] to `None`.
    ///
    /// # Errors
    ///
    /// Same as [`Reply::as_str`] for non-null, non-string replies.
    pub fn as_opt_str(&self) -> Result<Option<&str>> {
        if self.is_null() {
            Ok(None)
        } else {
            self.as_str().map(Some)
        }
    }
    /// Owned variant of [`Reply::as_str`].
    ///
    /// # Errors
    ///
    /// Same as [`Reply::as_str`].
    pub fn as_string_owned(&self) -> Result<String> {
        self.as_str().map(ToOwned::to_owned)
    }
    /// Owned variant of [`Reply::as_opt_str`].
    ///
    /// # Errors
    ///
    /// Same as [`Reply::as_opt_str`].
    pub fn as_opt_string_owned(&self) -> Result<Option<String>> {
        self.as_opt_str().map(|text| text.map(ToOwned::to_owned))
    }
    /// Borrows the payload as bytes (`Simple` or `Bulk`).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for other types.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        match self {
            Self::Simple(text) => Ok(text.as_bytes()),
            Self::Bulk(bytes) => Ok(bytes),
            other => Err(type_mismatch("a string", other)),
        }
    }
    /// Like [`Reply::as_bytes`], but maps [`Reply::Null`] to `None`.
    ///
    /// # Errors
    ///
    /// Same as [`Reply::as_bytes`] for non-null, non-string replies.
    pub fn as_opt_bytes(&self) -> Result<Option<&[u8]>> {
        if self.is_null() {
            Ok(None)
        } else {
            self.as_bytes().map(Some)
        }
    }
    /// Borrows the payload as an integer; a `RESP3` boolean yields `1`/`0`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for other types.
    pub fn as_integer(&self) -> Result<i64> {
        match self {
            Self::Integer(value) => Ok(*value),
            Self::Boolean(value) => Ok(i64::from(*value)),
            other => Err(type_mismatch("an integer", other)),
        }
    }
    /// Interprets a boolean or integer reply as a boolean.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for other types.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Boolean(value) => Ok(*value),
            Self::Integer(value) => Ok(*value != 0),
            other => Err(type_mismatch("a boolean", other)),
        }
    }
    /// Borrows an array-shaped reply (`Array`, `RESP3` `Set`, or `Push`).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for other types.
    pub fn as_array(&self) -> Result<&[Reply]> {
        match self {
            Self::Array(items) | Self::Set(items) | Self::Push(items) => Ok(items.as_slice()),
            other => Err(type_mismatch("an array", other)),
        }
    }
    /// Borrows a `RESP3` map reply as ordered key/value pairs.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for other types.
    pub fn as_map(&self) -> Result<&[(Reply, Reply)]> {
        match self {
            Self::Map(pairs) => Ok(pairs.as_slice()),
            other => Err(type_mismatch("a map", other)),
        }
    }
}
/// Builds the type-mismatch validation error shared by the accessors.
fn type_mismatch(expected: &str, actual: &Reply) -> Error {
    Error::validation(format!(
        "expected {expected} reply, got {}",
        actual.kind_name()
    ))
}
/// Encodes `args` as a `RESP` array of bulk strings.
///
/// The encoding is byte-exact for arbitrary binary arguments, so credentials,
/// keys, and values may contain `\r`, `\n`, NUL, or non-UTF-8 bytes.
/// An empty argument list encodes as the empty array `*0\r\n`.
#[must_use]
pub fn encode_command(args: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + args.iter().map(|arg| arg.len() + 16).sum::<usize>());
    out.push(b'*');
    out.extend_from_slice(args.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    for arg in args {
        out.push(b'$');
        out.extend_from_slice(arg.len().to_string().as_bytes());
        out.extend_from_slice(b"\r\n");
        out.extend_from_slice(arg);
        out.extend_from_slice(b"\r\n");
    }
    out
}
/// Parses one reply from the front of `input`.
///
/// Returns `Ok(None)` when `input` does not yet hold a complete reply, and
/// `Ok(Some((reply, consumed)))` otherwise, where `consumed` is the number of
/// bytes belonging to that reply (any trailing bytes are preserved for the next
/// call). Invalid framing and oversized headers are reported as errors; this
/// function never panics.
///
/// A `RESP3` attribute (`|N`) is consumed as metadata and the reply that follows
/// it is returned in its place. A `RESP3` streamed string (`;N`) is returned as
/// the concatenation of its chunks in one [`Reply::Bulk`].
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Transport`] for malformed or
/// truncated framing and for a chunked aggregate that exceeds the parser bounds.
pub fn parse_reply(input: &[u8]) -> Result<Option<(Reply, usize)>> {
    parse_at(input, 0)
}
/// Recursive worker for [`parse_reply`]; `depth` bounds recursion on nested replies.
fn parse_at(input: &[u8], depth: usize) -> Result<Option<(Reply, usize)>> {
    if depth > MAX_DEPTH {
        return Err(Error::transport(format!(
            "RESP reply nesting exceeds {MAX_DEPTH} levels"
        )));
    }
    let Some(&tag) = input.first() else {
        return Ok(None);
    };
    match tag {
        b'+' | b'-' => {
            let Some((line, consumed)) = parse_line(input, 1)? else {
                return Ok(None);
            };
            let text = String::from_utf8_lossy(line).into_owned();
            let reply = if tag == b'+' {
                Reply::Simple(text)
            } else {
                Reply::Error(text)
            };
            Ok(Some((reply, consumed)))
        }
        b':' => {
            let Some((value, consumed)) = parse_i64_line(input, 1)? else {
                return Ok(None);
            };
            Ok(Some((Reply::Integer(value), consumed)))
        }
        b'$' | b'=' => parse_bulk(input, tag == b'='),
        b'*' | b'~' | b'>' => parse_aggregate(input, tag, depth),
        b'%' => parse_map(input, depth),
        b'_' => {
            let Some((line, consumed)) = parse_line(input, 1)? else {
                return Ok(None);
            };
            if !line.is_empty() {
                return Err(Error::transport(
                    "invalid RESP null: expected an empty line",
                ));
            }
            Ok(Some((Reply::Null, consumed)))
        }
        b'#' => {
            let Some((line, consumed)) = parse_line(input, 1)? else {
                return Ok(None);
            };
            match line {
                b"t" => Ok(Some((Reply::Boolean(true), consumed))),
                b"f" => Ok(Some((Reply::Boolean(false), consumed))),
                _ => Err(Error::transport(
                    "invalid RESP boolean: expected 't' or 'f'",
                )),
            }
        }
        b',' => {
            let Some((line, consumed)) = parse_line(input, 1)? else {
                return Ok(None);
            };
            let text = std::str::from_utf8(line)
                .map_err(|_| Error::transport("invalid RESP double: non-UTF-8 bytes"))?;
            let value = text.parse::<f64>().map_err(|error| {
                Error::transport(format!("invalid RESP double '{text}': {error}"))
            })?;
            Ok(Some((Reply::Double(value), consumed)))
        }
        b'(' => {
            let Some((line, consumed)) = parse_line(input, 1)? else {
                return Ok(None);
            };
            let text = std::str::from_utf8(line)
                .map_err(|_| Error::transport("invalid RESP big number: non-UTF-8 bytes"))?;
            // Fits in `i64` -> `Integer`; larger magnitudes keep the literal digits.
            Ok(Some((
                text.parse::<i64>()
                    .map_or_else(|_| Reply::Bulk(line.to_vec()), Reply::Integer),
                consumed,
            )))
        }
        b'|' => parse_attribute(input, depth),
        b';' => parse_streamed_string(input),
        other => Err(Error::transport(format!(
            "invalid RESP reply type byte 0x{other:02x}"
        ))),
    }
}
/// Parses a `RESP3` attribute (`|N` key/value pairs) and the reply that follows.
///
/// Attributes are metadata about the next reply, so the parser projects them
/// away: the returned value is the reply after the pairs, and `consumed` covers
/// both. A trailing attribute chain (`|1 ... |1 ... value`) resolves the same way
/// because the following reply is parsed with the normal grammar one level deeper,
/// which keeps the recursion inside [`MAX_DEPTH`].
fn parse_attribute(input: &[u8], depth: usize) -> Result<Option<(Reply, usize)>> {
    let Some((count, mut consumed)) = parse_i64_line(input, 1)? else {
        return Ok(None);
    };
    if !(0..=MAX_ELEMENTS).contains(&count) {
        return Err(Error::transport(format!(
            "invalid RESP attribute length {count}"
        )));
    }
    let count = usize::try_from(count)
        .map_err(|_| Error::transport(format!("invalid RESP attribute length {count}")))?;
    for _ in 0..count {
        let Some((_, used)) = parse_at(&input[consumed..], depth + 1)? else {
            return Ok(None);
        };
        consumed += used;
        let Some((_, used)) = parse_at(&input[consumed..], depth + 1)? else {
            return Ok(None);
        };
        consumed += used;
    }
    let Some((reply, used)) = parse_at(&input[consumed..], depth + 1)? else {
        return Ok(None);
    };
    Ok(Some((reply, consumed + used)))
}
/// Parses a `RESP3` streamed (chunked) string into one binary-safe [`Reply::Bulk`].
///
/// Every chunk is `;N\r\n<N bytes>\r\n`; a zero length chunk is the terminator, so
/// `;0\r\n` decodes to an empty bulk string. Both a single chunk and the total
/// payload are bounded by [`MAX_BULK_LEN`]. Chunks are scalar values, so they add
/// no nesting depth; the `;` type byte itself is already bounded by the [`MAX_DEPTH`]
/// check in [`parse_at`].
fn parse_streamed_string(input: &[u8]) -> Result<Option<(Reply, usize)>> {
    let mut consumed = 0;
    let mut payload: Vec<u8> = Vec::new();
    loop {
        let Some((length, header_end)) = parse_i64_line(&input[consumed..], 1)? else {
            return Ok(None);
        };
        let length = usize::try_from(length).map_err(|_| {
            Error::transport(format!(
                "invalid RESP streamed string chunk length {length}"
            ))
        })?;
        if length > MAX_BULK_LEN {
            return Err(Error::transport(format!(
                "RESP streamed string chunk length {length} exceeds the {MAX_BULK_LEN}-byte limit"
            )));
        }
        let payload_start = consumed
            .checked_add(header_end)
            .ok_or_else(|| Error::transport("RESP chunk length overflows the input offset"))?;
        // The zero-length chunk is the terminator: `;0\r\n` carries no payload, so the
        // header is all that is left to consume.
        if length == 0 {
            return Ok(Some((Reply::Bulk(payload), payload_start)));
        }
        let payload_end = payload_start
            .checked_add(length)
            .ok_or_else(|| Error::transport("RESP chunk length overflows the input offset"))?;
        let total = payload_end
            .checked_add(2)
            .ok_or_else(|| Error::transport("RESP chunk length overflows the input offset"))?;
        if input.len() < total {
            return Ok(None);
        }
        if &input[payload_end..total] != b"\r\n" {
            return Err(Error::transport(
                "invalid RESP streamed string framing: chunk not terminated by CRLF",
            ));
        }
        consumed = total;
        if payload.len().saturating_add(length) > MAX_BULK_LEN {
            return Err(Error::transport(format!(
                "RESP streamed string exceeds the {MAX_BULK_LEN}-byte limit"
            )));
        }
        payload.extend_from_slice(&input[payload_start..payload_end]);
    }
}
/// Parses a `RESP` bulk string (or a `RESP3` verbatim string when `verbatim`).
fn parse_bulk(input: &[u8], verbatim: bool) -> Result<Option<(Reply, usize)>> {
    let Some((length, header_end)) = parse_i64_line(input, 1)? else {
        return Ok(None);
    };
    if length == -1 {
        return Ok(Some((Reply::Null, header_end)));
    }
    let length = usize::try_from(length)
        .map_err(|_| Error::transport(format!("invalid RESP bulk length {length}")))?;
    if length > MAX_BULK_LEN {
        return Err(Error::transport(format!(
            "RESP bulk length {length} exceeds the {MAX_BULK_LEN}-byte limit"
        )));
    }
    let payload_end = header_end
        .checked_add(length)
        .ok_or_else(|| Error::transport("RESP bulk length overflows the input offset"))?;
    let total = payload_end
        .checked_add(2)
        .ok_or_else(|| Error::transport("RESP bulk length overflows the input offset"))?;
    if input.len() < total {
        return Ok(None);
    }
    if &input[payload_end..total] != b"\r\n" {
        return Err(Error::transport(
            "invalid RESP bulk framing: payload not terminated by CRLF",
        ));
    }
    let payload = &input[header_end..payload_end];
    if verbatim {
        // Verbatim strings carry a three-byte encoding tag (`txt:`, `mkd:`, ...);
        // the payload is exposed without it. A short payload is kept whole.
        let text = if payload.len() > 3 && payload[3] == b':' {
            &payload[4..]
        } else {
            payload
        };
        Ok(Some((Reply::Bulk(text.to_vec()), total)))
    } else {
        Ok(Some((Reply::Bulk(payload.to_vec()), total)))
    }
}
/// Parses an array, `RESP3` set, or `RESP3` push aggregation.
fn parse_aggregate(input: &[u8], tag: u8, depth: usize) -> Result<Option<(Reply, usize)>> {
    let Some((count, mut consumed)) = parse_i64_line(input, 1)? else {
        return Ok(None);
    };
    if count == -1 {
        return Ok(Some((Reply::Null, consumed)));
    }
    if !(0..=MAX_ELEMENTS).contains(&count) {
        return Err(Error::transport(format!(
            "invalid RESP aggregate length {count}"
        )));
    }
    let count = usize::try_from(count)
        .map_err(|_| Error::transport(format!("invalid RESP aggregate length {count}")))?;
    let mut items = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let Some((item, used)) = parse_at(&input[consumed..], depth + 1)? else {
            return Ok(None);
        };
        consumed += used;
        items.push(item);
    }
    let reply = match tag {
        b'*' => Reply::Array(items),
        b'~' => Reply::Set(items),
        _ => Reply::Push(items),
    };
    Ok(Some((reply, consumed)))
}
/// Parses a `RESP3` map (a count followed by that many key/value pairs).
fn parse_map(input: &[u8], depth: usize) -> Result<Option<(Reply, usize)>> {
    let Some((count, mut consumed)) = parse_i64_line(input, 1)? else {
        return Ok(None);
    };
    if !(0..=MAX_ELEMENTS).contains(&count) {
        return Err(Error::transport(format!("invalid RESP map length {count}")));
    }
    let count = usize::try_from(count)
        .map_err(|_| Error::transport(format!("invalid RESP map length {count}")))?;
    let mut pairs = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let Some((key, used)) = parse_at(&input[consumed..], depth + 1)? else {
            return Ok(None);
        };
        consumed += used;
        let Some((value, used)) = parse_at(&input[consumed..], depth + 1)? else {
            return Ok(None);
        };
        consumed += used;
        pairs.push((key, value));
    }
    Ok(Some((Reply::Map(pairs), consumed)))
}
/// Reads one `CRLF`-terminated line starting at `offset`, returning the line and
/// the offset just past the `CRLF`; `Ok(None)` means the terminator is missing.
fn parse_line(input: &[u8], offset: usize) -> Result<Option<(&[u8], usize)>> {
    let mut index = offset;
    while index < input.len() {
        match input[index] {
            b'\r' => {
                let Some(&next) = input.get(index + 1) else {
                    return Ok(None);
                };
                if next != b'\n' {
                    return Err(Error::transport(
                        "invalid RESP framing: carriage return not followed by line feed",
                    ));
                }
                return Ok(Some((&input[offset..index], index + 2)));
            }
            b'\n' => {
                return Err(Error::transport(
                    "invalid RESP framing: line feed without carriage return",
                ));
            }
            _ => index += 1,
        }
    }
    Ok(None)
}
/// Parses a signed integer carried on its own `CRLF`-terminated line.
fn parse_i64_line(input: &[u8], offset: usize) -> Result<Option<(i64, usize)>> {
    let Some((line, consumed)) = parse_line(input, offset)? else {
        return Ok(None);
    };
    let text = std::str::from_utf8(line)
        .map_err(|_| Error::transport("invalid RESP integer: non-UTF-8 bytes"))?;
    let value = text
        .parse::<i64>()
        .map_err(|error| Error::transport(format!("invalid RESP integer '{text}': {error}")))?;
    Ok(Some((value, consumed)))
}
#[cfg(test)]
mod tests;
