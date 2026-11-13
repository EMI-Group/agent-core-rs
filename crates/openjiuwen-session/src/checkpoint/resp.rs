//! A dependency-free RESP2 client for the Redis checkpoint backend.
//!
//! Python's checkpointer family delegates its remote variant to the `redis`
//! client library; the Rust port keeps the workspace offline and dependency-free
//! by speaking the RESP2 wire protocol directly over
//! [`tokio::net::TcpStream`]: command arrays of bulk strings out, and simple
//! strings, error replies, integers, bulk strings, and (nested) arrays in.
//!
//! Only the subset the checkpointer needs is implemented. Supported URL forms
//! are `redis://[user[:password]@]host[:port][/db]`; `rediss://` (TLS) reports
//! [`openjiuwen_common::error::ErrorKind::Unsupported`] instead of silently
//! connecting in the clear.
#![allow(clippy::module_name_repetitions)]

use openjiuwen_common::error::{Error, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

/// The largest bulk string the client accepts (512 MiB).
const MAX_BULK_LENGTH: usize = 512 * 1024 * 1024;
/// The largest number of elements accepted in one array reply.
const MAX_ARRAY_LENGTH: usize = 16 * 1024 * 1024;
/// The port used when the URL omits one.
const DEFAULT_PORT: u16 = 6379;

/// One decoded RESP2 reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RespValue {
    /// A simple string (`+OK`).
    Simple(String),
    /// A server-side error reply (`-ERR unknown command`).
    Error(String),
    /// An integer (`:42`).
    Integer(i64),
    /// A bulk string (`$3` + `foo`); `None` is the nil bulk reply (`$-1`).
    Bulk(Option<Vec<u8>>),
    /// An array (`*2` + two values); `None` is the nil array (`*-1`).
    Array(Option<Vec<RespValue>>),
}

impl RespValue {
    /// Returns the reply as text: simple strings directly, bulk strings when
    /// they are valid UTF-8.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Simple(text) => Some(text),
            Self::Bulk(Some(bytes)) => std::str::from_utf8(bytes).ok(),
            _ => None,
        }
    }

    /// Returns the reply's bytes: simple strings as their UTF-8 encoding, bulk
    /// strings as their payload.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Simple(text) => Some(text.as_bytes()),
            Self::Bulk(Some(bytes)) => Some(bytes),
            _ => None,
        }
    }

    /// Returns the reply as an integer: integer replies directly, simple and
    /// bulk strings when they parse as a number.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            Self::Simple(text) => text.trim().parse().ok(),
            Self::Bulk(Some(bytes)) => std::str::from_utf8(bytes).ok()?.trim().parse().ok(),
            _ => None,
        }
    }

    /// Returns the reply's elements when it is an array; the nil array yields
    /// `None`.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(Some(items)) => Some(items),
            _ => None,
        }
    }

    /// Returns whether the reply is a nil bulk string or the nil array.
    #[must_use]
    pub fn is_nil(&self) -> bool {
        matches!(self, Self::Bulk(None) | Self::Array(None))
    }

    /// Converts a server error reply into a
    /// [`openjiuwen_common::error::ErrorKind::Session`] error, leaving every
    /// other reply untouched.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Session`] carrying the
    /// server's message when the reply is [`RespValue::Error`].
    pub fn into_result(self) -> Result<Self> {
        match self {
            Self::Error(reply) => Err(Error::session(format!("redis error reply: {reply}"))),
            other => Ok(other),
        }
    }
}

/// Encodes `args` as one RESP array of bulk strings
/// (`*N\r\n$len\r\narg\r\n…`), the only command form the client sends.
///
/// ```
/// use openjiuwen_session::checkpoint::resp::encode_command;
///
/// assert_eq!(encode_command(&["PING"]), b"*1\r\n$4\r\nPING\r\n");
/// ```
#[must_use]
pub fn encode_command(args: &[&str]) -> Vec<u8> {
    let parts: Vec<&[u8]> = args.iter().map(|arg| arg.as_bytes()).collect();
    encode_command_bytes(&parts)
}

/// Encodes binary-safe arguments as one RESP array of bulk strings.
pub(crate) fn encode_command_bytes(args: &[&[u8]]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(16 + args.iter().map(|arg| arg.len() + 16).sum::<usize>());
    encoded.extend_from_slice(format!("*{}\r\n", args.len()).as_bytes());
    for arg in args {
        encoded.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        encoded.extend_from_slice(arg);
        encoded.extend_from_slice(b"\r\n");
    }
    encoded
}

/// Reads exactly one RESP2 reply of any shape from `reader`.
///
/// The function is generic over [`AsyncBufRead`], so the codec can be exercised
/// from an in-memory buffer; [`RespConnection`] drives it over a
/// [`BufReader<TcpStream>`].
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Session`] when the transport
/// fails mid-reply or when the peer sends something that is not RESP2.
pub async fn read_value<R>(reader: &mut R) -> Result<RespValue>
where
    R: AsyncBufRead + Unpin,
{
    read_reply(reader).await.map_err(ReadError::into_error)
}

/// A transport or protocol failure while decoding a reply.
#[derive(Debug)]
enum ReadError {
    /// The socket failed or closed mid-reply.
    Transport(std::io::Error),
    /// The peer sent a malformed or unsupported RESP2 frame.
    Protocol(String),
}

impl ReadError {
    /// Builds a protocol failure.
    fn protocol(message: impl Into<String>) -> Self {
        Self::Protocol(message.into())
    }

    /// Maps the failure onto the crate error type.
    fn into_error(self) -> Error {
        match self {
            Self::Transport(error) => Error::session(format!("redis read failed: {error}")),
            Self::Protocol(message) => Error::session(message),
        }
    }
}

/// Decodes one reply, keeping transport failures distinguishable so
/// [`RespConnection`] can mark itself broken and reconnect.
async fn read_reply<R>(reader: &mut R) -> std::result::Result<RespValue, ReadError>
where
    R: AsyncBufRead + Unpin,
{
    // Arrays are assembled from a stack of `(expected, collected)` frames, so
    // nesting never recurses.
    let mut pending: Vec<(usize, Vec<RespValue>)> = Vec::new();
    loop {
        let line = read_line(reader).await?;
        let mut value = match line.as_bytes().first().copied() {
            Some(b'+') => RespValue::Simple(line[1..].to_string()),
            Some(b'-') => RespValue::Error(line[1..].to_string()),
            Some(b':') => RespValue::Integer(parse_integer(&line[1..])?),
            Some(b'$') => match parse_length(&line[1..])? {
                None => RespValue::Bulk(None),
                Some(length) => RespValue::Bulk(Some(read_bulk(reader, length).await?)),
            },
            Some(b'*') => match parse_length(&line[1..])? {
                None => RespValue::Array(None),
                Some(0) => RespValue::Array(Some(Vec::new())),
                Some(count) => {
                    if count > MAX_ARRAY_LENGTH {
                        return Err(ReadError::protocol(format!(
                            "RESP array length {count} exceeds the {MAX_ARRAY_LENGTH}-element limit"
                        )));
                    }
                    pending.push((count, Vec::with_capacity(count.min(1024))));
                    continue;
                }
            },
            Some(other) => {
                return Err(ReadError::protocol(format!(
                    "unexpected RESP type byte {other:#04x}"
                )));
            }
            None => return Err(ReadError::protocol("empty RESP reply line")),
        };
        loop {
            let complete = match pending.last_mut() {
                Some((expected, collected)) => {
                    collected.push(value);
                    collected.len() >= *expected
                }
                None => return Ok(value),
            };
            if !complete {
                break;
            }
            let Some((_, collected)) = pending.pop() else {
                break;
            };
            value = RespValue::Array(Some(collected));
        }
    }
}

/// Reads one CRLF-terminated protocol line, without the terminator.
async fn read_line<R>(reader: &mut R) -> std::result::Result<String, ReadError>
where
    R: AsyncBufRead + Unpin,
{
    let mut buffer = Vec::new();
    let read = reader
        .read_until(b'\n', &mut buffer)
        .await
        .map_err(ReadError::Transport)?;
    if read == 0 {
        return Err(ReadError::Transport(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "redis connection closed while waiting for a reply",
        )));
    }
    if buffer.last() != Some(&b'\n') {
        return Err(ReadError::Transport(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "redis reply line is not terminated",
        )));
    }
    buffer.pop();
    if buffer.last() == Some(&b'\r') {
        buffer.pop();
    }
    String::from_utf8(buffer)
        .map_err(|error| ReadError::protocol(format!("RESP line is not UTF-8: {error}")))
}

/// Reads a bulk payload of `length` bytes plus its CRLF terminator.
async fn read_bulk<R>(reader: &mut R, length: usize) -> std::result::Result<Vec<u8>, ReadError>
where
    R: AsyncBufRead + Unpin,
{
    if length > MAX_BULK_LENGTH {
        return Err(ReadError::protocol(format!(
            "RESP bulk length {length} exceeds the {MAX_BULK_LENGTH}-byte limit"
        )));
    }
    let mut payload = vec![0_u8; length];
    reader
        .read_exact(&mut payload)
        .await
        .map_err(ReadError::Transport)?;
    let mut terminator = [0_u8; 2];
    reader
        .read_exact(&mut terminator)
        .await
        .map_err(ReadError::Transport)?;
    if &terminator != b"\r\n" {
        return Err(ReadError::protocol(
            "RESP bulk string is missing its CRLF terminator",
        ));
    }
    Ok(payload)
}

/// Parses a `$`/`*` length field; `-1` is the nil form.
fn parse_length(field: &str) -> std::result::Result<Option<usize>, ReadError> {
    let length: i64 = field
        .trim()
        .parse()
        .map_err(|_| ReadError::protocol(format!("malformed RESP length {field:?}")))?;
    if length == -1 {
        return Ok(None);
    }
    if length < 0 {
        return Err(ReadError::protocol(format!(
            "unsupported RESP length {length}"
        )));
    }
    let length = usize::try_from(length).map_err(|_| {
        ReadError::protocol(format!("RESP length {length} does not fit the platform"))
    })?;
    Ok(Some(length))
}

/// Parses a `:` integer field.
fn parse_integer(field: &str) -> std::result::Result<i64, ReadError> {
    field
        .trim()
        .parse()
        .map_err(|_| ReadError::protocol(format!("malformed RESP integer {field:?}")))
}

/// A single RESP2 connection: one buffered socket, used sequentially.
///
/// The connection is not multiplexed; [`RespConnection::command`] writes one
/// command and reads its reply, so a connection must not be shared across
/// concurrent callers (the checkpointer serializes access through a mutex).
#[derive(Debug)]
pub struct RespConnection {
    stream: BufReader<TcpStream>,
    broken: bool,
}

impl RespConnection {
    /// Connects to `url` and completes the handshake (`AUTH` when credentials
    /// are present, `SELECT` when the path names a database).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for a
    /// malformed URL, [`openjiuwen_common::error::ErrorKind::Unsupported`] for
    /// `rediss://`, and [`openjiuwen_common::error::ErrorKind::Session`] when
    /// the connection or the handshake fails.
    pub async fn connect(url: &str) -> Result<Self> {
        let target = ConnectionTarget::parse(url)?;
        let stream = TcpStream::connect((target.host.as_str(), target.port))
            .await
            .map_err(|error| {
                Error::session(format!(
                    "failed to connect to redis at {}:{}: {error}",
                    target.host, target.port
                ))
            })?;
        let mut connection = Self {
            stream: BufReader::new(stream),
            broken: false,
        };
        if let Some(password) = target.password.as_deref() {
            let reply = match target.username.as_deref() {
                Some(username) if !username.is_empty() => {
                    connection.command(&["AUTH", username, password]).await?
                }
                _ => connection.command(&["AUTH", password]).await?,
            };
            reply.into_result()?;
        }
        if let Some(database) = target.database {
            connection.select_database(database).await?;
        }
        Ok(connection)
    }

    /// Reports whether a transport failure has already invalidated this
    /// connection, in which case the caller must dial a fresh one.
    #[must_use]
    pub fn is_broken(&self) -> bool {
        self.broken
    }

    /// Sends one command given as text arguments and returns its reply.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Session`] when the write
    /// or the read fails, or when the peer's reply is not valid RESP2. Server
    /// error replies are returned as [`RespValue::Error`], not as `Err`.
    pub async fn command(&mut self, args: &[&str]) -> Result<RespValue> {
        let parts: Vec<&[u8]> = args.iter().map(|arg| arg.as_bytes()).collect();
        self.command_bytes(&parts).await
    }

    /// Sends one command with binary-safe arguments and returns its reply.
    ///
    /// # Errors
    ///
    /// Same as [`RespConnection::command`].
    pub async fn command_bytes(&mut self, args: &[&[u8]]) -> Result<RespValue> {
        let payload = encode_command_bytes(args);
        match self.stream.get_mut().write_all(&payload).await {
            Ok(()) => {}
            Err(error) => {
                self.broken = true;
                return Err(Error::session(format!("redis write failed: {error}")));
            }
        }
        match read_reply(&mut self.stream).await {
            Ok(value) => Ok(value),
            Err(ReadError::Transport(error)) => {
                self.broken = true;
                Err(ReadError::Transport(error).into_error())
            }
            Err(error @ ReadError::Protocol(_)) => Err(error.into_error()),
        }
    }

    /// Sends `SELECT <database>` and checks the reply.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Session`] when the
    /// transport fails or the server rejects the database index.
    pub async fn select_database(&mut self, database: u32) -> Result<()> {
        let index = database.to_string();
        self.command(&["SELECT", &index]).await?.into_result()?;
        Ok(())
    }
}

/// A parsed `redis://` URL.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConnectionTarget {
    host: String,
    port: u16,
    username: Option<String>,
    password: Option<String>,
    database: Option<u32>,
}

impl ConnectionTarget {
    /// Parses `redis://[user[:password]@]host[:port][/db]`.
    fn parse(url: &str) -> Result<Self> {
        let trimmed = url.trim();
        if trimmed.is_empty() {
            return Err(Error::validation(
                "redis url must not be empty; expected redis://host[:port][/db]",
            ));
        }
        let Some(rest) = trimmed.strip_prefix("redis://") else {
            if trimmed.starts_with("rediss://") {
                return Err(Error::unsupported(
                    "rediss (TLS) is not supported; use redis://",
                ));
            }
            return Err(Error::validation(format!(
                "unsupported redis url {trimmed:?}; expected redis://host[:port][/db]"
            )));
        };
        let (authority, path) = match rest.split_once('/') {
            Some((authority, path)) => (authority, Some(path)),
            None => (rest, None),
        };
        let (credentials, host_port) = match authority.rsplit_once('@') {
            Some((credentials, host_port)) => (Some(credentials), host_port),
            None => (None, authority),
        };
        let (username, password) = match credentials {
            Some(credentials) => match credentials.split_once(':') {
                Some((username, password)) => {
                    (Some(username.to_string()), Some(password.to_string()))
                }
                None => (None, Some(credentials.to_string())),
            },
            None => (None, None),
        };
        let (host, port) = parse_host_port(host_port, trimmed)?;
        let database = match path {
            Some(path) => {
                let segment = path.split('/').next().unwrap_or_default();
                if segment.is_empty() {
                    None
                } else {
                    Some(segment.parse::<u32>().map_err(|_| {
                        Error::validation(format!(
                            "invalid redis database index {segment:?} in {trimmed:?}"
                        ))
                    })?)
                }
            }
            None => None,
        };
        Ok(Self {
            host,
            port,
            username,
            password,
            database,
        })
    }
}

/// Splits a `host[:port]` or `[ipv6][:port]` authority into its parts.
fn parse_host_port(host_port: &str, url: &str) -> Result<(String, u16)> {
    let (host, port) = if let Some(bracketed) = host_port.strip_prefix('[') {
        let (host, after) = bracketed.split_once(']').ok_or_else(|| {
            Error::validation(format!("malformed IPv6 host in redis url {url:?}"))
        })?;
        let port = match after.strip_prefix(':') {
            Some(port) => Some(port),
            None if after.is_empty() => None,
            None => {
                return Err(Error::validation(format!(
                    "malformed redis url authority {host_port:?}"
                )));
            }
        };
        (host, port)
    } else {
        match host_port.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (host_port, None),
        }
    };
    if host.is_empty() {
        return Err(Error::validation(format!(
            "redis url {url:?} is missing a host"
        )));
    }
    let port = match port {
        Some(port) => port
            .parse::<u16>()
            .map_err(|_| Error::validation(format!("invalid redis port {port:?} in {url:?}")))?,
        None => DEFAULT_PORT,
    };
    Ok((host.to_string(), port))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::{encode_command, read_value, ConnectionTarget, RespConnection, RespValue};

    /// Decodes one reply from an in-memory buffer.
    async fn decode(bytes: &[u8]) -> openjiuwen_common::error::Result<RespValue> {
        let mut reader = tokio::io::BufReader::new(bytes);
        read_value(&mut reader).await
    }

    #[test]
    fn encodes_commands_as_bulk_string_arrays() {
        assert_eq!(encode_command(&["PING"]), b"*1\r\n$4\r\nPING\r\n".to_vec());
        assert_eq!(
            encode_command(&["SET", "k", "v"]),
            b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n".to_vec()
        );
        assert_eq!(
            encode_command(&["SET", "k", ""]),
            b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$0\r\n\r\n".to_vec()
        );
        assert_eq!(encode_command(&[]), b"*0\r\n".to_vec());
        assert_eq!(
            encode_command(&["SELECT", "12"]),
            b"*2\r\n$6\r\nSELECT\r\n$2\r\n12\r\n".to_vec()
        );
    }

    #[tokio::test]
    async fn decodes_scalar_replies() {
        assert_eq!(
            decode(b"+OK\r\n").await.unwrap(),
            RespValue::Simple("OK".into())
        );
        assert_eq!(
            decode(b"-ERR unknown command\r\n").await.unwrap(),
            RespValue::Error("ERR unknown command".into())
        );
        assert_eq!(decode(b":42\r\n").await.unwrap(), RespValue::Integer(42));
        assert_eq!(
            decode(b"$3\r\nfoo\r\n").await.unwrap(),
            RespValue::Bulk(Some(b"foo".to_vec()))
        );
        assert_eq!(
            decode(b"$0\r\n\r\n").await.unwrap(),
            RespValue::Bulk(Some(Vec::new()))
        );
        assert_eq!(decode(b"$-1\r\n").await.unwrap(), RespValue::Bulk(None));
        assert_eq!(decode(b"*-1\r\n").await.unwrap(), RespValue::Array(None));
        assert_eq!(
            decode(b"*0\r\n").await.unwrap(),
            RespValue::Array(Some(Vec::new()))
        );
    }

    #[tokio::test]
    async fn decodes_arrays_recursively() {
        let value = decode(b"*3\r\n$3\r\nfoo\r\n*-1\r\n*2\r\n:1\r\n*1\r\n$3\r\nbar\r\n")
            .await
            .unwrap();
        assert_eq!(
            value,
            RespValue::Array(Some(vec![
                RespValue::Bulk(Some(b"foo".to_vec())),
                RespValue::Array(None),
                RespValue::Array(Some(vec![
                    RespValue::Integer(1),
                    RespValue::Array(Some(vec![RespValue::Bulk(Some(b"bar".to_vec()))])),
                ])),
            ]))
        );
    }

    #[tokio::test]
    async fn decodes_binary_bulk_payloads() {
        let value = decode(b"$3\r\n\x00\xff\x01\r\n").await.unwrap();
        assert_eq!(value, RespValue::Bulk(Some(vec![0x00, 0xff, 0x01])));
        assert_eq!(value.as_bytes(), Some(&[0x00, 0xff, 0x01][..]));
        assert_eq!(value.as_str(), None);
    }

    #[tokio::test]
    async fn rejects_malformed_replies() {
        for malformed in [
            b"!weird\r\n".as_slice(),
            b"+OK".as_slice(),
            b"$3\r\nfoobar\r\n".as_slice(),
            b"$x\r\n".as_slice(),
            b":nope\r\n".as_slice(),
            b"*-2\r\n".as_slice(),
            b"*2\r\n$1\r\na\r\n".as_slice(),
            b"\r\n".as_slice(),
        ] {
            let error = decode(malformed).await.unwrap_err();
            assert_eq!(
                error.kind(),
                ErrorKind::Session,
                "unexpected error for {malformed:?}: {error}"
            );
        }
    }

    #[test]
    fn server_errors_become_session_errors() {
        let error = RespValue::Error("ERR boom".into())
            .into_result()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Session);
        assert!(error.to_string().contains("ERR boom"), "{error}");
        assert_eq!(
            RespValue::Simple("OK".into()).into_result().unwrap(),
            RespValue::Simple("OK".into())
        );
        assert!(RespValue::Bulk(None).is_nil());
        assert!(!RespValue::Bulk(Some(Vec::new())).is_nil());
    }

    #[tokio::test]
    async fn rejects_unsupported_and_malformed_urls() {
        assert_eq!(
            RespConnection::connect("rediss://localhost:6379")
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
        for malformed in [
            "",
            "http://localhost:6379",
            "redis://localhost:not-a-port",
            "redis://localhost:6379/not-a-database",
            "redis://:6379",
        ] {
            assert_eq!(
                RespConnection::connect(malformed).await.unwrap_err().kind(),
                ErrorKind::Validation,
                "unexpected error for {malformed:?}"
            );
        }
    }

    #[test]
    fn parses_connection_targets() {
        let target = ConnectionTarget::parse("redis://localhost").unwrap();
        assert_eq!(target.host, "localhost");
        assert_eq!(target.port, 6379);
        assert_eq!(target.username, None);
        assert_eq!(target.password, None);
        assert_eq!(target.database, None);

        let target = ConnectionTarget::parse("redis://user:secret@cache:6380/3").unwrap();
        assert_eq!(target.host, "cache");
        assert_eq!(target.port, 6380);
        assert_eq!(target.username.as_deref(), Some("user"));
        assert_eq!(target.password.as_deref(), Some("secret"));
        assert_eq!(target.database, Some(3));

        let target = ConnectionTarget::parse("redis://secret@cache/").unwrap();
        assert_eq!(target.host, "cache");
        assert_eq!(target.username, None);
        assert_eq!(target.password.as_deref(), Some("secret"));
        assert_eq!(target.database, None);

        let target = ConnectionTarget::parse("redis://[::1]:6390/1").unwrap();
        assert_eq!(target.host, "::1");
        assert_eq!(target.port, 6390);
        assert_eq!(target.database, Some(1));
    }
}
