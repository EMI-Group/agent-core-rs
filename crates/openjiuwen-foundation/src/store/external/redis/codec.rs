//! A pure RESP2 encoder/decoder with RESP3 tolerance.
//!
//! The module has no I/O: [`encode_command`] turns a command's arguments into the
//! RESP "array of bulk strings" form, [`encode`] serializes any [`RespValue`], and
//! [`Decoder`] turns an arbitrarily chunked byte stream into complete frames.
//!
//! The decoder is deliberately total: an incomplete frame yields `Ok(None)` and
//! leaves the buffered bytes in place, while a protocol violation yields an
//! [`Error::External`](openjiuwen_common::error::Error) naming the violation — a
//! malformed stream can never panic the process or spin forever.
//!
//! RESP3 frames (`_`, `#`, `,`, `=`, `%`, `~`, `>`) are accepted so a server that
//! negotiated RESP3 through `HELLO` still decodes; the other RESP3-only frames
//! (attributes, streamed aggregates, big numbers) are rejected rather than
//! guessed at.

use bytes::Bytes;

use openjiuwen_common::error::{Error, Result};

/// Maximum nesting depth of an aggregate frame.
///
/// A hostile peer could otherwise drive the recursive parser into a stack
/// overflow; real replies never nest anywhere near this deep.
const MAX_DEPTH: usize = 64;

/// The largest capacity pre-allocated for an aggregate frame's element vector.
///
/// The announced element count is attacker-controlled, so the vector grows as
/// elements actually arrive instead of trusting it.
const MAX_PREALLOC: usize = 1024;

/// Builds the protocol-violation error used for malformed inbound frames.
pub(crate) fn protocol_error(message: impl Into<String>) -> Error {
    Error::external(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("redis protocol error: {}", message.into()),
    ))
}

/// One RESP frame.
///
/// The variants map one-to-one onto RESP type prefixes, with the RESP3 additions
/// kept distinct so callers can tell a null apart from an empty bulk string.
#[derive(Debug, Clone, PartialEq)]
pub enum RespValue {
    /// `+text\r\n` — a simple (unencoded) string such as `OK` or `PONG`.
    SimpleString(String),
    /// `-text\r\n` — a server error such as `ERR unknown command`.
    Error(String),
    /// `:number\r\n` — a signed integer.
    Integer(i64),
    /// `$len\r\n<bytes>\r\n` — a binary-safe string; `None` is the null bulk
    /// string (`$-1\r\n`).
    BulkString(Option<Bytes>),
    /// `*count\r\n...` — an array; `None` is the null array (`*-1\r\n`).
    Array(Option<Vec<RespValue>>),
    /// `_\r\n` — the RESP3 null.
    Null,
    /// `#t\r\n` / `#f\r\n` — the RESP3 boolean.
    Boolean(bool),
    /// `,number\r\n` — the RESP3 double (`inf`, `-inf` and `nan` included).
    Double(f64),
    /// `=len\r\n<bytes>\r\n` — a RESP3 verbatim string.
    ///
    /// The three-byte format prefix (`txt:` / `mkd:`) is retained in the payload,
    /// because the backend never needs to interpret it.
    Verbatim(Vec<u8>),
    /// `%count\r\n...` — a RESP3 map of `count` key/value pairs.
    Map(Vec<(RespValue, RespValue)>),
    /// `~count\r\n...` — a RESP3 set.
    Set(Vec<RespValue>),
    /// `>count\r\n...` — a RESP3 out-of-band push message.
    Push(Vec<RespValue>),
}

impl RespValue {
    /// The binary payload of a bulk string, simple string or verbatim string.
    ///
    /// Returns `None` for every other variant, including the null bulk string.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::BulkString(Some(bytes)) => Some(bytes),
            Self::SimpleString(text) => Some(text.as_bytes()),
            Self::Verbatim(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// The integer payload of an [`RespValue::Integer`].
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(number) => Some(*number),
            _ => None,
        }
    }

    /// The frame rendered as text when it has an obvious textual form.
    ///
    /// Simple strings, error frames, verbatim strings and integers always
    /// convert; bulk strings convert when their payload is valid UTF-8. Every
    /// other variant (arrays, nulls, maps, ...) yields `None`.
    pub fn as_string(&self) -> Option<String> {
        match self {
            Self::SimpleString(text) | Self::Error(text) => Some(text.clone()),
            Self::BulkString(Some(bytes)) => String::from_utf8(bytes.to_vec()).ok(),
            Self::Verbatim(bytes) => String::from_utf8(bytes.clone()).ok(),
            Self::Integer(number) => Some(number.to_string()),
            _ => None,
        }
    }

    /// Consumes the frame, returning its payload bytes when it has one.
    ///
    /// Equivalent to [`RespValue::as_bytes`] without borrowing; returns `None`
    /// for the null bulk string and every non-string variant.
    pub fn into_bytes(self) -> Option<Vec<u8>> {
        match self {
            Self::BulkString(Some(bytes)) => Some(bytes.to_vec()),
            Self::SimpleString(text) => Some(text.into_bytes()),
            Self::Verbatim(bytes) => Some(bytes),
            _ => None,
        }
    }
}

/// Encodes a command as a RESP array of bulk strings.
///
/// Every argument is binary-safe: its length is announced, so embedded CRLF and
/// NUL bytes pass through untouched. An empty argument list encodes to `*0\r\n`.
pub fn encode_command(args: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for arg in args {
        write_bulk(&mut out, arg);
    }
    out
}

/// Encodes one frame in RESP2 (or RESP3 for the RESP3-only variants).
///
/// Only the variants a client needs to send are supported; nested aggregates are
/// encoded recursively.
pub fn encode(value: &RespValue) -> Vec<u8> {
    let mut out = Vec::new();
    write_value(&mut out, value);
    out
}

/// Appends `$len\r\n<payload>\r\n`.
fn write_bulk(out: &mut Vec<u8>, payload: &[u8]) {
    out.push(b'$');
    out.extend_from_slice(payload.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
}

/// Appends the encoding of one frame.
fn write_value(out: &mut Vec<u8>, value: &RespValue) {
    match value {
        RespValue::SimpleString(text) => write_line(out, b'+', text),
        RespValue::Error(text) => write_line(out, b'-', text),
        RespValue::Integer(number) => write_line(out, b':', &number.to_string()),
        RespValue::BulkString(Some(bytes)) => write_bulk(out, bytes),
        RespValue::BulkString(None) => out.extend_from_slice(b"$-1\r\n"),
        RespValue::Array(Some(items)) => write_aggregate(out, b'*', items),
        RespValue::Array(None) => out.extend_from_slice(b"*-1\r\n"),
        RespValue::Null => out.extend_from_slice(b"_\r\n"),
        RespValue::Boolean(flag) => {
            out.extend_from_slice(if *flag { b"#t\r\n" } else { b"#f\r\n" });
        }
        RespValue::Double(number) => write_line(out, b',', &number.to_string()),
        RespValue::Verbatim(payload) => {
            out.push(b'=');
            out.extend_from_slice(payload.len().to_string().as_bytes());
            out.extend_from_slice(b"\r\n");
            out.extend_from_slice(payload);
            out.extend_from_slice(b"\r\n");
        }
        RespValue::Map(pairs) => {
            out.push(b'%');
            out.extend_from_slice(pairs.len().to_string().as_bytes());
            out.extend_from_slice(b"\r\n");
            for (key, value) in pairs {
                write_value(out, key);
                write_value(out, value);
            }
        }
        RespValue::Set(items) => write_aggregate(out, b'~', items),
        RespValue::Push(items) => write_aggregate(out, b'>', items),
    }
}

/// Appends `<tag>text\r\n`.
fn write_line(out: &mut Vec<u8>, tag: u8, text: &str) {
    out.push(tag);
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(b"\r\n");
}

/// Appends `<tag>count\r\n` followed by every element.
fn write_aggregate(out: &mut Vec<u8>, tag: u8, items: &[RespValue]) {
    out.push(tag);
    out.extend_from_slice(items.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    for item in items {
        write_value(out, item);
    }
}

/// An incremental RESP frame decoder.
///
/// Bytes are buffered until a frame is complete, which makes the decoder safe to
/// drive from a socket read loop: feed every chunk to [`Decoder::push`], then drain
/// complete frames with [`Decoder::try_next`] until it returns `Ok(None)`.
#[derive(Debug, Default)]
pub struct Decoder {
    /// Bytes received but not yet consumed.
    buffer: Vec<u8>,
}

impl Decoder {
    /// Creates an empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a freshly received chunk to the internal buffer.
    pub fn push(&mut self, chunk: &[u8]) {
        self.buffer.extend_from_slice(chunk);
    }

    /// Number of buffered bytes that do not yet form a complete frame.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Decodes the next complete frame, if one is buffered.
    ///
    /// Returns `Ok(None)` when the buffer holds only an incomplete frame; the
    /// bytes are preserved for the next [`Decoder::push`]. A malformed frame is a
    /// protocol error, and the offending bytes stay buffered so the caller can
    /// discard the connection.
    pub fn try_next(&mut self) -> Result<Option<RespValue>> {
        match parse(&self.buffer, 0, 0)? {
            None => Ok(None),
            Some((value, consumed)) => {
                self.buffer.drain(..consumed);
                Ok(Some(value))
            }
        }
    }
}

/// Parses one frame at `pos`.
///
/// Returns `Ok(None)` when the frame is incomplete, otherwise the value and the
/// position just past its last byte.
fn parse(input: &[u8], pos: usize, depth: usize) -> Result<Option<(RespValue, usize)>> {
    if depth > MAX_DEPTH {
        return Err(protocol_error(format!(
            "aggregate frame nests deeper than {MAX_DEPTH} levels"
        )));
    }
    let Some(&tag) = input.get(pos) else {
        return Ok(None);
    };
    let start = pos + 1;
    match tag {
        b'+' => simple(input, start),
        b'-' => error_frame(input, start),
        b':' => integer(input, start),
        b'$' => bulk(input, start),
        b'*' => aggregate(input, start, depth, |items| RespValue::Array(Some(items))),
        b'%' => map(input, start, depth),
        b'~' => aggregate(input, start, depth, RespValue::Set),
        b'>' => aggregate(input, start, depth, RespValue::Push),
        b'_' => null(input, start),
        b'#' => boolean(input, start),
        b',' => double(input, start),
        b'=' => verbatim(input, start),
        other => Err(protocol_error(format!(
            "unknown frame type byte 0x{other:02x}"
        ))),
    }
}

/// Reads one CRLF-terminated line starting at `from`.
fn line(input: &[u8], from: usize) -> Result<Option<(&str, usize)>> {
    let mut index = from;
    while index + 1 < input.len() {
        if input[index] == b'\r' && input[index + 1] == b'\n' {
            let text = std::str::from_utf8(&input[from..index])
                .map_err(|_| protocol_error("line is not valid UTF-8"))?;
            return Ok(Some((text, index + 2)));
        }
        index += 1;
    }
    Ok(None)
}

/// Reads the announced length of a bulk/aggregate frame.
///
/// `None` for the length itself means the RESP null form (`-1`).
fn length(input: &[u8], start: usize) -> Result<Option<(Option<usize>, usize)>> {
    let Some((text, next)) = line(input, start)? else {
        return Ok(None);
    };
    let announced: i64 = text
        .parse()
        .map_err(|_| protocol_error(format!("invalid length {text:?}")))?;
    if announced < -1 {
        return Err(protocol_error(format!(
            "invalid negative length {announced}"
        )));
    }
    let length = if announced == -1 {
        None
    } else {
        Some(usize::try_from(announced).map_err(|_| protocol_error("length is too large"))?)
    };
    Ok(Some((length, next)))
}

/// Decodes `+text\r\n`.
fn simple(input: &[u8], start: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((text, next)) = line(input, start)? else {
        return Ok(None);
    };
    Ok(Some((RespValue::SimpleString(text.to_string()), next)))
}

/// Decodes `-text\r\n`.
fn error_frame(input: &[u8], start: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((text, next)) = line(input, start)? else {
        return Ok(None);
    };
    Ok(Some((RespValue::Error(text.to_string()), next)))
}

/// Decodes `:number\r\n`.
fn integer(input: &[u8], start: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((text, next)) = line(input, start)? else {
        return Ok(None);
    };
    let number: i64 = text
        .parse()
        .map_err(|_| protocol_error(format!("invalid integer {text:?}")))?;
    Ok(Some((RespValue::Integer(number), next)))
}

/// Decodes `#t\r\n` / `#f\r\n`.
fn boolean(input: &[u8], start: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((text, next)) = line(input, start)? else {
        return Ok(None);
    };
    let flag = match text {
        "t" => true,
        "f" => false,
        other => return Err(protocol_error(format!("invalid boolean frame {other:?}"))),
    };
    Ok(Some((RespValue::Boolean(flag), next)))
}

/// Decodes `,number\r\n`.
fn double(input: &[u8], start: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((text, next)) = line(input, start)? else {
        return Ok(None);
    };
    let number: f64 = text
        .parse()
        .map_err(|_| protocol_error(format!("invalid double {text:?}")))?;
    Ok(Some((RespValue::Double(number), next)))
}

/// Decodes `_\r\n`, tolerating an ignored payload before the CRLF.
fn null(input: &[u8], start: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((_, next)) = line(input, start)? else {
        return Ok(None);
    };
    Ok(Some((RespValue::Null, next)))
}

/// Decodes `$len\r\n<payload>\r\n`.
fn bulk(input: &[u8], start: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((length, next)) = length(input, start)? else {
        return Ok(None);
    };
    let Some(length) = length else {
        return Ok(Some((RespValue::BulkString(None), next)));
    };
    let Some((payload, end)) = payload(input, next, length)? else {
        return Ok(None);
    };
    Ok(Some((
        RespValue::BulkString(Some(Bytes::copy_from_slice(payload))),
        end,
    )))
}

/// Decodes `=len\r\n<payload>\r\n`.
fn verbatim(input: &[u8], start: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((length, next)) = length(input, start)? else {
        return Ok(None);
    };
    let Some(length) = length else {
        return Err(protocol_error("verbatim frame announced a null length"));
    };
    let Some((payload, end)) = payload(input, next, length)? else {
        return Ok(None);
    };
    Ok(Some((RespValue::Verbatim(payload.to_vec()), end)))
}

/// Slices the `length`-byte payload starting at `from` plus its trailing CRLF.
///
/// `Ok(None)` means the payload has not arrived yet.
fn payload(input: &[u8], from: usize, length: usize) -> Result<Option<(&[u8], usize)>> {
    let Some(end) = from.checked_add(length) else {
        return Err(protocol_error("payload length overflows the address space"));
    };
    let Some(after) = end.checked_add(2) else {
        return Err(protocol_error("payload length overflows the address space"));
    };
    if input.len() < after {
        return Ok(None);
    }
    if &input[end..after] != b"\r\n" {
        return Err(protocol_error("payload is not followed by CRLF"));
    }
    Ok(Some((&input[from..end], after)))
}

/// Decodes an array/set/push frame with `count` elements.
fn aggregate(
    input: &[u8],
    start: usize,
    depth: usize,
    wrap: fn(Vec<RespValue>) -> RespValue,
) -> Result<Option<(RespValue, usize)>> {
    let Some((count, next)) = length(input, start)? else {
        return Ok(None);
    };
    let Some(count) = count else {
        return Ok(Some((RespValue::Array(None), next)));
    };
    let Some((items, end)) = elements(input, next, count, depth)? else {
        return Ok(None);
    };
    Ok(Some((wrap(items), end)))
}

/// Decodes a RESP3 map frame with `count` pairs.
fn map(input: &[u8], start: usize, depth: usize) -> Result<Option<(RespValue, usize)>> {
    let Some((count, next)) = length(input, start)? else {
        return Ok(None);
    };
    let Some(count) = count else {
        return Err(protocol_error("map frame announced a null length"));
    };
    let Some(pair_count) = count.checked_mul(2) else {
        return Err(protocol_error("map size overflows the address space"));
    };
    let Some((flat, end)) = elements(input, next, pair_count, depth)? else {
        return Ok(None);
    };
    let mut pairs = Vec::with_capacity(count.min(MAX_PREALLOC));
    let mut iter = flat.into_iter();
    while let (Some(key), Some(value)) = (iter.next(), iter.next()) {
        pairs.push((key, value));
    }
    Ok(Some((RespValue::Map(pairs), end)))
}

/// Decodes `count` consecutive frames starting at `from`.
fn elements(
    input: &[u8],
    from: usize,
    count: usize,
    depth: usize,
) -> Result<Option<(Vec<RespValue>, usize)>> {
    let mut items = Vec::with_capacity(count.min(MAX_PREALLOC));
    let mut pos = from;
    for _ in 0..count {
        let Some((value, next)) = parse(input, pos, depth + 1)? else {
            return Ok(None);
        };
        items.push(value);
        pos = next;
    }
    Ok(Some((items, pos)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes exactly one frame from `bytes`, asserting nothing is left over.
    fn decode_one(bytes: &[u8]) -> RespValue {
        let mut decoder = Decoder::new();
        decoder.push(bytes);
        let value = decoder.try_next().expect("decode").expect("complete frame");
        assert_eq!(decoder.buffered(), 0, "unexpected trailing bytes");
        value
    }

    #[test]
    fn command_encoding_is_an_array_of_bulk_strings() {
        assert_eq!(
            encode_command(&[b"SET", b"k", b"v"]),
            b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n".to_vec()
        );
        assert_eq!(encode_command(&[]), b"*0\r\n".to_vec());
        assert_eq!(
            encode_command(&[b"X", b"a\r\nb"]),
            b"*2\r\n$1\r\nX\r\n$4\r\na\r\nb\r\n".to_vec()
        );
    }

    #[test]
    fn encode_round_trips_every_writable_variant() {
        for value in [
            RespValue::SimpleString("OK".to_string()),
            RespValue::Error("ERR bad".to_string()),
            RespValue::Integer(-42),
            RespValue::BulkString(Some(Bytes::from_static(b"\0bin"))),
            RespValue::BulkString(None),
            RespValue::Array(Some(vec![
                RespValue::Integer(1),
                RespValue::BulkString(Some(Bytes::from_static(b"two"))),
            ])),
            RespValue::Array(None),
            RespValue::Null,
            RespValue::Boolean(true),
            RespValue::Boolean(false),
            RespValue::Map(vec![(
                RespValue::SimpleString("k".to_string()),
                RespValue::Integer(1),
            )]),
            RespValue::Set(vec![RespValue::Integer(7)]),
            RespValue::Push(vec![RespValue::SimpleString("message".to_string())]),
        ] {
            assert_eq!(decode_one(&encode(&value)), value);
        }
    }

    #[test]
    fn decodes_resp2_scalars() {
        assert_eq!(
            decode_one(b"+OK\r\n"),
            RespValue::SimpleString("OK".to_string())
        );
        assert_eq!(
            decode_one(b"-ERR nope\r\n"),
            RespValue::Error("ERR nope".to_string())
        );
        assert_eq!(decode_one(b":17\r\n"), RespValue::Integer(17));
        assert_eq!(
            decode_one(b"$3\r\nabc\r\n"),
            RespValue::BulkString(Some(Bytes::from_static(b"abc")))
        );
        assert_eq!(
            decode_one(b"$0\r\n\r\n"),
            RespValue::BulkString(Some(Bytes::new()))
        );
        assert_eq!(decode_one(b"$-1\r\n"), RespValue::BulkString(None));
    }

    #[test]
    fn decodes_resp2_aggregates() {
        assert_eq!(
            decode_one(b"*2\r\n$1\r\na\r\n:1\r\n"),
            RespValue::Array(Some(vec![
                RespValue::BulkString(Some(Bytes::from_static(b"a"))),
                RespValue::Integer(1),
            ]))
        );
        assert_eq!(decode_one(b"*0\r\n"), RespValue::Array(Some(vec![])));
        assert_eq!(decode_one(b"*-1\r\n"), RespValue::Array(None));
        assert_eq!(
            decode_one(b"*2\r\n*1\r\n:1\r\n*1\r\n:2\r\n"),
            RespValue::Array(Some(vec![
                RespValue::Array(Some(vec![RespValue::Integer(1)])),
                RespValue::Array(Some(vec![RespValue::Integer(2)])),
            ]))
        );
    }

    #[test]
    fn decodes_resp3_frames() {
        assert_eq!(decode_one(b"_\r\n"), RespValue::Null);
        assert_eq!(decode_one(b"#t\r\n"), RespValue::Boolean(true));
        assert_eq!(decode_one(b"#f\r\n"), RespValue::Boolean(false));
        assert_eq!(decode_one(b",1.5\r\n"), RespValue::Double(1.5));
        assert_eq!(decode_one(b",inf\r\n"), RespValue::Double(f64::INFINITY));
        assert_eq!(
            decode_one(b"=9\r\ntxt:hello\r\n"),
            RespValue::Verbatim(b"txt:hello".to_vec())
        );
        assert_eq!(
            decode_one(b"~2\r\n:1\r\n:2\r\n"),
            RespValue::Set(vec![RespValue::Integer(1), RespValue::Integer(2)])
        );
        assert_eq!(
            decode_one(b">1\r\n+hi\r\n"),
            RespValue::Push(vec![RespValue::SimpleString("hi".to_string())])
        );
        assert_eq!(
            decode_one(b"%1\r\n+k\r\n:1\r\n"),
            RespValue::Map(vec![(
                RespValue::SimpleString("k".to_string()),
                RespValue::Integer(1),
            )])
        );
    }

    #[test]
    fn accessors_never_panic_and_distinguish_null() {
        let bulk = RespValue::BulkString(Some(Bytes::from_static(b"abc")));
        assert_eq!(bulk.as_bytes(), Some(b"abc".as_slice()));
        assert_eq!(bulk.as_string().as_deref(), Some("abc"));
        assert_eq!(bulk.clone().into_bytes(), Some(b"abc".to_vec()));
        assert_eq!(bulk.as_integer(), None);

        let null = RespValue::BulkString(None);
        assert_eq!(null.as_bytes(), None);
        assert_eq!(null.as_string(), None);
        assert_eq!(null.clone().into_bytes(), None);

        assert_eq!(RespValue::Integer(9).as_integer(), Some(9));
        assert_eq!(RespValue::Integer(9).as_string().as_deref(), Some("9"));
        assert_eq!(RespValue::Array(None).as_string(), None);
    }

    #[test]
    fn partial_frames_are_buffered_until_complete() {
        let mut decoder = Decoder::new();
        decoder.push(b"$5\r\nhel");
        assert_eq!(decoder.try_next().expect("partial"), None);
        assert_eq!(decoder.buffered(), 7);

        decoder.push(b"lo\r\n");
        assert_eq!(
            decoder.try_next().expect("complete"),
            Some(RespValue::BulkString(Some(Bytes::from_static(b"hello"))))
        );
        assert_eq!(decoder.buffered(), 0);
    }

    #[test]
    fn partial_aggregate_is_buffered_until_every_element_arrives() {
        let mut decoder = Decoder::new();
        decoder.push(b"*2\r\n+one\r\n");
        assert_eq!(decoder.try_next().expect("partial"), None);
        assert_eq!(decoder.buffered(), 10);

        decoder.push(b"+two\r\n");
        assert_eq!(
            decoder.try_next().expect("complete"),
            Some(RespValue::Array(Some(vec![
                RespValue::SimpleString("one".to_string()),
                RespValue::SimpleString("two".to_string()),
            ])))
        );
    }

    #[test]
    fn several_pipelined_frames_decode_in_order() {
        let mut decoder = Decoder::new();
        decoder.push(b"+OK\r\n:1\r\n$1\r\nx\r\n");
        assert_eq!(
            decoder.try_next().expect("first"),
            Some(RespValue::SimpleString("OK".to_string()))
        );
        assert_eq!(
            decoder.try_next().expect("second"),
            Some(RespValue::Integer(1))
        );
        assert_eq!(
            decoder.try_next().expect("third"),
            Some(RespValue::BulkString(Some(Bytes::from_static(b"x"))))
        );
        assert_eq!(decoder.try_next().expect("empty"), None);
        assert_eq!(decoder.buffered(), 0);
    }

    #[test]
    fn byte_at_a_time_feeding_still_decodes() {
        let encoded = encode_command(&[b"PING"]);
        let mut decoder = Decoder::new();
        let mut value = None;
        for byte in encoded {
            decoder.push(&[byte]);
            if let Some(frame) = decoder.try_next().expect("decode") {
                value = Some(frame);
            }
        }
        assert_eq!(
            value,
            Some(RespValue::Array(Some(vec![RespValue::BulkString(Some(
                Bytes::from_static(b"PING")
            ))])))
        );
    }

    #[test]
    fn malformed_frames_report_protocol_errors() {
        for (bytes, needle) in [
            (&b"@what\r\n"[..], "unknown frame type"),
            (&b":nope\r\n"[..], "invalid integer"),
            (&b"$x\r\n"[..], "invalid length"),
            (&b"$-2\r\n"[..], "invalid negative length"),
            (&b"#maybe\r\n"[..], "invalid boolean"),
            (&b",nope\r\n"[..], "invalid double"),
            (&b"$3\r\nabcXX"[..], "not followed by CRLF"),
        ] {
            let mut decoder = Decoder::new();
            decoder.push(bytes);
            let error = decoder.try_next().expect_err("protocol violation");
            assert!(
                error.to_string().contains(needle),
                "{error} does not mention {needle:?}"
            );
        }
    }

    #[test]
    fn excessive_nesting_is_rejected_without_recursing_forever() {
        let mut bytes = Vec::new();
        for _ in 0..(MAX_DEPTH + 2) {
            bytes.extend_from_slice(b"*1\r\n");
        }
        bytes.extend_from_slice(b":1\r\n");
        let mut decoder = Decoder::new();
        decoder.push(&bytes);
        let error = decoder.try_next().expect_err("too deep");
        assert!(error.to_string().contains("nests deeper"));
    }

    #[test]
    fn empty_buffer_yields_nothing() {
        let mut decoder = Decoder::new();
        assert_eq!(decoder.try_next().expect("empty"), None);
        assert_eq!(decoder.buffered(), 0);
    }
}
