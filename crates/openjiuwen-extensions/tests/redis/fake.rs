//! A stateful, in-process `RESP2` fake server for the crate-boundary Redis suite.
//!
//! `openjiuwen_extensions::redis::testsupport` is `#[cfg(test)] pub(crate)`, so it
//! is invisible to an integration test (which is its own crate); this module is the
//! peer the suite in `tests/redis.rs` talks to instead. It binds `127.0.0.1:0`,
//! keeps its data in memory, and is offline and deterministic.
//!
//! Requests are parsed with the crate's own `codec::parse_reply`, replies are
//! hand-encoded (`encode_command` only writes requests), and expiry is lazy, so no
//! task or clock is involved. Dropping the [`Server`] signals the accept loop,
//! which drops the listener and releases the port. [`Server::start_moved_to`]
//! answers `-MOVED 3999 <target>` so the cluster-redirection test can point it at a
//! second, live [`Server`].
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
use openjiuwen_extensions::redis::codec::{parse_reply, Reply};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::time::Instant;
/// A running fake server; dropping it stops the loop and releases the port.
pub(crate) struct Server {
    addr: SocketAddr,
    state: State,
    stop: Option<oneshot::Sender<()>>,
}
impl Server {
    /// Starts a server that answers every supported command.
    pub(crate) async fn start() -> Self {
        Self::start_with(None).await
    }
    /// Starts a server answering `-MOVED 3999 <target>` after the handshake.
    pub(crate) async fn start_moved_to(target: SocketAddr) -> Self {
        Self::start_with(Some(target)).await
    }
    async fn start_with(redirect: Option<SocketAddr>) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let data = Arc::new(Mutex::new(Data::default()));
        let (stop, mut stopped) = oneshot::channel();
        let accept_data = Arc::clone(&data);
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stopped => break,
                    accepted = listener.accept() => match accepted {
                        Ok((stream, _peer)) => {
                            let connection = Arc::clone(&accept_data);
                            tokio::spawn(serve(stream, connection, redirect));
                        }
                        Err(_) => break,
                    },
                }
            }
            // The loop owns `listener`, so leaving it releases the port.
        });
        Self {
            addr,
            state: State { data },
            stop: Some(stop),
        }
    }
    /// The loopback address this server listens on.
    pub(crate) fn addr(&self) -> SocketAddr {
        self.addr
    }
    /// A `redis://` URL pointing at this server, database `0`.
    pub(crate) fn url(&self) -> String {
        format!("redis://{}/0", self.addr)
    }
    /// A handle for asserting on server state.
    pub(crate) fn state(&self) -> &State {
        &self.state
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}
/// Assertion handle over a running server's data.
pub(crate) struct State {
    data: Arc<Mutex<Data>>,
}
impl State {
    /// The stored string payload for `key`, or `None` when absent or expired.
    pub(crate) fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.with(|data| data.values.get(key.as_bytes()).cloned())
    }
    /// Every live key across namespaces, sorted.
    pub(crate) fn keys(&self) -> Vec<String> {
        self.with(|data| {
            let keys = data.keys();
            keys.into_iter()
                .map(|key| String::from_utf8_lossy(&key).into_owned())
                .collect()
        })
    }
    /// Remaining TTL in milliseconds: `None` when absent, `Some(-1)` without expiry.
    pub(crate) fn ttl_ms(&self, key: &str) -> Option<i64> {
        self.with(|data| {
            if !data.exists(key.as_bytes()) {
                return None;
            }
            let remaining = data.expiries.get(key.as_bytes()).map_or(-1, |at| {
                i64::try_from(at.saturating_duration_since(Instant::now()).as_millis())
                    .unwrap_or(i64::MAX)
            });
            Some(remaining)
        })
    }
    fn with<T>(&self, read: impl FnOnce(&mut Data) -> T) -> T {
        let mut data = self
            .data
            .lock()
            .expect("the fake state mutex is not poisoned");
        data.purge();
        read(&mut data)
    }
}
/// Stored data with lazy TTL expiry.
#[derive(Default)]
struct Data {
    values: HashMap<Vec<u8>, Vec<u8>>,
    sets: HashMap<Vec<u8>, BTreeSet<Vec<u8>>>,
    zsets: HashMap<Vec<u8>, BTreeMap<Vec<u8>, f64>>,
    expiries: HashMap<Vec<u8>, Instant>,
}
impl Data {
    fn purge(&mut self) {
        let now = Instant::now();
        let expired: Vec<Vec<u8>> = self
            .expiries
            .iter()
            .filter(|(_, at)| **at <= now)
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            self.remove(&key);
        }
    }
    fn exists(&self, key: &[u8]) -> bool {
        self.values.contains_key(key) || self.sets.contains_key(key) || self.zsets.contains_key(key)
    }
    fn remove(&mut self, key: &[u8]) -> bool {
        let value = self.values.remove(key).is_some();
        let set = self.sets.remove(key).is_some();
        let zset = self.zsets.remove(key).is_some();
        self.expiries.remove(key);
        value || set || zset
    }
    fn keys(&self) -> Vec<Vec<u8>> {
        let keys = self
            .values
            .keys()
            .chain(self.sets.keys())
            .chain(self.zsets.keys());
        let mut keys: Vec<Vec<u8>> = keys.cloned().collect();
        keys.sort();
        keys.dedup();
        keys
    }
}
/// Accepts one connection and answers requests until it closes.
async fn serve(mut stream: TcpStream, state: Arc<Mutex<Data>>, redirect: Option<SocketAddr>) {
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match parse_reply(&buffer) {
            Ok(Some((request, consumed))) => {
                buffer.drain(..consumed);
                let reply = request_args(&request).map_or_else(
                    || error("ERR Protocol error: not an array"),
                    |args| execute(&state, &args, redirect),
                );
                if send_reply(&mut stream, &reply).await.is_err() {
                    return;
                }
            }
            Ok(None) => match stream.read(&mut chunk).await {
                Ok(read) if read > 0 => buffer.extend_from_slice(&chunk[..read]),
                _ => return,
            },
            Err(_) => return,
        }
    }
}
fn request_args(request: &Reply) -> Option<Vec<Vec<u8>>> {
    let Reply::Array(items) = request else {
        return None;
    };
    items
        .iter()
        .map(|item| match item {
            Reply::Bulk(bytes) => Some(bytes.clone()),
            Reply::Simple(text) => Some(text.as_bytes().to_vec()),
            _ => None,
        })
        .collect()
}
// Returns the `-ERR wrong number of arguments` reply from the caller.
macro_rules! exactly {
    ($args:expr, $count:expr, $name:expr) => {
        if $args.len() != $count {
            return wrong_arity(&$name);
        }
    };
}
// Returns the `-ERR wrong number of arguments` reply from the caller.
macro_rules! at_least {
    ($args:expr, $count:expr, $name:expr) => {
        if $args.len() < $count {
            return wrong_arity(&$name);
        }
    };
}
/// Executes one command against the shared state.
#[allow(clippy::too_many_lines)]
fn execute(state: &Arc<Mutex<Data>>, args: &[Vec<u8>], redirect: Option<SocketAddr>) -> Reply {
    let Some(name) = args.first().map(|arg| upper(arg)) else {
        return error("ERR Protocol error: empty command");
    };
    if let Some(target) = redirect {
        if !matches!(name.as_str(), "AUTH" | "SELECT" | "HELLO" | "ASKING") {
            return error(&format!("MOVED 3999 {target}"));
        }
    }
    let mut guard = state.lock().expect("the fake state mutex is not poisoned");
    guard.purge();
    let data = &mut *guard;
    match name.as_str() {
        "PING" => match args.len() {
            1 => simple("PONG"),
            2 => Reply::Bulk(args[1].clone()),
            _ => wrong_arity(&name),
        },
        "ECHO" => {
            exactly!(args, 2, name);
            Reply::Bulk(args[1].clone())
        }
        "AUTH" | "SELECT" => {
            if args.len() == 2 || args.len() == 3 {
                simple("OK")
            } else {
                wrong_arity(&name)
            }
        }
        "GET" => {
            exactly!(args, 2, name);
            let key = args[1].as_slice();
            if data.sets.contains_key(key) || data.zsets.contains_key(key) {
                return wrong_type();
            }
            data.values
                .get(key)
                .cloned()
                .map_or(Reply::Null, Reply::Bulk)
        }
        "SET" => set(data, args),
        "EXISTS" => {
            at_least!(args, 2, name);
            let live = args[1..].iter().filter(|key| data.exists(key)).count();
            Reply::Integer(count(live))
        }
        "DEL" | "UNLINK" => {
            at_least!(args, 2, name);
            let removed = args[1..].iter().filter(|key| data.remove(key)).count();
            Reply::Integer(count(removed))
        }
        "MGET" => {
            at_least!(args, 2, name);
            let values = args[1..].iter().map(|key| {
                data.values
                    .get(key.as_slice())
                    .cloned()
                    .map_or(Reply::Null, Reply::Bulk)
            });
            Reply::Array(values.collect())
        }
        "TTL" | "PTTL" => {
            exactly!(args, 2, name);
            ttl_reply(data, &args[1], name == "PTTL")
        }
        "EXPIRE" => {
            exactly!(args, 3, name);
            let Some(seconds) = parse_i64(&args[2]) else {
                return error("ERR value is not an integer or out of range");
            };
            let key = args[1].as_slice();
            if !data.exists(key) {
                return Reply::Integer(0);
            }
            if seconds <= 0 {
                data.remove(key);
                return Reply::Integer(1);
            }
            let lease = Duration::from_secs(u64::try_from(seconds).unwrap_or(0));
            data.expiries
                .insert(args[1].clone(), Instant::now() + lease);
            Reply::Integer(1)
        }
        "SCAN" => scan(data, args, &name),
        "KEYS" => {
            exactly!(args, 2, name);
            let pattern = String::from_utf8_lossy(&args[1]).into_owned();
            Reply::Array(
                matching(data, Some(&pattern))
                    .into_iter()
                    .map(Reply::Bulk)
                    .collect(),
            )
        }
        "SADD" | "SREM" | "SMEMBERS" => set_command(data, args, &name),
        "ZCARD" => {
            exactly!(args, 2, name);
            let length = data.zsets.get(args[1].as_slice()).map_or(0, BTreeMap::len);
            Reply::Integer(count(length))
        }
        "ZADD" | "ZRANGE" | "ZREM" => zset_command(data, args, &name),
        "FLUSHDB" => {
            data.values.clear();
            data.sets.clear();
            data.zsets.clear();
            data.expiries.clear();
            simple("OK")
        }
        "TYPE" => {
            exactly!(args, 2, name);
            let key = args[1].as_slice();
            let kind = if data.values.contains_key(key) {
                "string"
            } else if data.sets.contains_key(key) {
                "set"
            } else if data.zsets.contains_key(key) {
                "zset"
            } else {
                "none"
            };
            simple(kind)
        }
        _ => error(&format!("ERR unknown command '{name}'")),
    }
}
/// Handles `SET key value [NX] [KEEPTTL] [EX s | PX ms]`; a plain `SET` drops
/// any previous TTL, which the kernel `KvStore` contract depends on.
fn set(data: &mut Data, args: &[Vec<u8>]) -> Reply {
    if args.len() < 3 {
        return wrong_arity("SET");
    }
    let key = args[1].clone();
    let (mut nx, mut keep_ttl, mut expiry) = (false, false, None);
    let mut index = 3;
    while index < args.len() {
        let option = upper(&args[index]);
        match option.as_str() {
            "NX" => nx = true,
            "KEEPTTL" => keep_ttl = true,
            "EX" | "PX" => {
                let Some(amount) = args.get(index + 1).and_then(|raw| parse_i64(raw)) else {
                    return error("ERR value is not an integer or out of range");
                };
                let amount = u64::try_from(amount).unwrap_or(0);
                let lease = if option == "PX" {
                    Duration::from_millis(amount)
                } else {
                    Duration::from_secs(amount)
                };
                expiry = Some(Instant::now() + lease);
                index += 1;
            }
            _ => return error("ERR syntax error"),
        }
        index += 1;
    }
    if nx && data.exists(&key) {
        return Reply::Null;
    }
    data.sets.remove(&key);
    data.zsets.remove(&key);
    data.values.insert(key.clone(), args[2].clone());
    if let Some(at) = expiry {
        data.expiries.insert(key, at);
    } else if !keep_ttl {
        data.expiries.remove(&key);
    }
    simple("OK")
}
/// Handles `SCAN cursor [MATCH glob] [COUNT n]`: one batch, cursor `0`.
fn scan(data: &Data, args: &[Vec<u8>], name: &str) -> Reply {
    at_least!(args, 2, name);
    let mut pattern: Option<String> = None;
    let mut index = 2;
    while index < args.len() {
        let option = upper(&args[index]);
        match option.as_str() {
            "MATCH" | "COUNT" => {
                let Some(raw) = args.get(index + 1) else {
                    return error("ERR syntax error");
                };
                if option == "MATCH" {
                    pattern = Some(String::from_utf8_lossy(raw).into_owned());
                }
                index += 1;
            }
            _ => return error("ERR syntax error"),
        }
        index += 1;
    }
    let keys = matching(data, pattern.as_deref())
        .into_iter()
        .map(Reply::Bulk)
        .collect();
    Reply::Array(vec![Reply::Bulk(b"0".to_vec()), Reply::Array(keys)])
}
/// Handles `SADD`/`SREM`/`SMEMBERS`.
fn set_command(data: &mut Data, args: &[Vec<u8>], name: &str) -> Reply {
    if name == "SMEMBERS" {
        exactly!(args, 2, name);
        let members = data
            .sets
            .get(args[1].as_slice())
            .map_or_else(Vec::new, |set| {
                set.iter().cloned().map(Reply::Bulk).collect()
            });
        return Reply::Array(members);
    }
    at_least!(args, 3, name);
    let key = args[1].clone();
    if name == "SADD" {
        let set = data.sets.entry(key).or_default();
        let added = args[2..]
            .iter()
            .filter(|member| set.insert((*member).clone()))
            .count();
        return Reply::Integer(count(added));
    }
    let removed = data.sets.get_mut(&key).map_or(0, |set| {
        args[2..]
            .iter()
            .filter(|member| set.remove(member.as_slice()))
            .count()
    });
    if data.sets.get(&key).is_some_and(BTreeSet::is_empty) {
        data.sets.remove(&key);
    }
    Reply::Integer(count(removed))
}
/// Handles `ZADD`/`ZRANGE`/`ZREM`; ranges are ordered by score then member.
fn zset_command(data: &mut Data, args: &[Vec<u8>], name: &str) -> Reply {
    let key = args.get(1).cloned().unwrap_or_default();
    if name == "ZADD" {
        if args.len() < 4 || (args.len() - 2) % 2 != 0 {
            return wrong_arity(name);
        }
        let mut added = 0;
        for pair in args[2..].chunks(2) {
            let Some(score) = parse_f64(&pair[0]) else {
                return error("ERR value is not a valid float");
            };
            if data
                .zsets
                .entry(key.clone())
                .or_default()
                .insert(pair[1].clone(), score)
                .is_none()
            {
                added += 1;
            }
        }
        return Reply::Integer(count(added));
    }
    if name == "ZREM" {
        at_least!(args, 3, name);
        let removed = data.zsets.get_mut(&key).map_or(0, |zset| {
            args[2..]
                .iter()
                .filter(|member| zset.remove(member.as_slice()).is_some())
                .count()
        });
        if data.zsets.get(&key).is_some_and(BTreeMap::is_empty) {
            data.zsets.remove(&key);
        }
        return Reply::Integer(count(removed));
    }
    exactly!(args, 4, name);
    let (Some(start), Some(stop)) = (
        args.get(2).and_then(|raw| parse_i64(raw)),
        args.get(3).and_then(|raw| parse_i64(raw)),
    ) else {
        return error("ERR value is not an integer or out of range");
    };
    let mut entries: Vec<(&Vec<u8>, f64)> =
        data.zsets
            .get(key.as_slice())
            .map_or_else(Vec::new, |zset| {
                zset.iter()
                    .map(|(member, score)| (member, *score))
                    .collect()
            });
    entries.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(right.0))
    });
    let length = count(entries.len());
    let resolve = |index: i64| if index < 0 { length + index } else { index };
    let (start, stop) = (resolve(start).max(0), resolve(stop).min(length - 1));
    if stop < start || start >= length {
        return Reply::Array(Vec::new());
    }
    let (skip, take) = (
        usize::try_from(start).unwrap_or(0),
        usize::try_from(stop - start).unwrap_or(0),
    );
    let members = entries
        .iter()
        .skip(skip)
        .take(take + 1)
        .map(|(member, _)| Reply::Bulk((*member).clone()));
    Reply::Array(members.collect())
}
/// Upper-cases a byte argument for command/option matching.
fn upper(value: &[u8]) -> String {
    String::from_utf8_lossy(value).to_ascii_uppercase()
}
/// Parses a byte argument as an `i64`.
fn parse_i64(value: &[u8]) -> Option<i64> {
    std::str::from_utf8(value).ok()?.parse().ok()
}
/// Parses a byte argument as a finite `f64`.
fn parse_f64(value: &[u8]) -> Option<f64> {
    let parsed = std::str::from_utf8(value).ok()?.parse::<f64>().ok()?;
    parsed.is_finite().then_some(parsed)
}
/// Converts a count to `i64` without panicking.
fn count(count: usize) -> i64 {
    i64::try_from(count).unwrap_or(i64::MAX)
}
fn simple(text: &str) -> Reply {
    Reply::Simple(text.to_owned())
}
fn error(message: &str) -> Reply {
    Reply::Error(message.to_owned())
}
fn wrong_arity(name: &str) -> Reply {
    error(&format!(
        "ERR wrong number of arguments for '{name}' command"
    ))
}
fn wrong_type() -> Reply {
    error("WRONGTYPE Operation against a key holding the wrong kind of value")
}
/// The `-2`/`-1`/remaining TTL reply.
fn ttl_reply(data: &Data, key: &[u8], milliseconds: bool) -> Reply {
    if !data.exists(key) {
        return Reply::Integer(-2);
    }
    let Some(at) = data.expiries.get(key) else {
        return Reply::Integer(-1);
    };
    let remaining = at.saturating_duration_since(Instant::now());
    let value = if milliseconds {
        remaining.as_millis()
    } else {
        u128::from(remaining.as_secs())
    };
    Reply::Integer(i64::try_from(value).unwrap_or(i64::MAX))
}
/// The live keys matching an optional glob pattern.
fn matching(data: &Data, pattern: Option<&str>) -> Vec<Vec<u8>> {
    data.keys()
        .into_iter()
        .filter(|key| pattern.is_none_or(|glob| glob_match(glob.as_bytes(), key)))
        .collect()
}
/// A glob matcher for `*`, `?`, and backslash escapes, so the store's escaped
/// prefix still reaches a literal match.
fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some(b'\\') => pattern.get(1).is_some_and(|expected| {
            text.first() == Some(expected) && glob_match(&pattern[2..], &text[1..])
        }),
        Some(b'*') => {
            let rest = &pattern[1..];
            let mut index = 0;
            loop {
                if glob_match(rest, &text[index..]) {
                    return true;
                }
                if index == text.len() {
                    return false;
                }
                index += 1;
            }
        }
        Some(b'?') => !text.is_empty() && glob_match(&pattern[1..], &text[1..]),
        Some(expected) => text.first() == Some(expected) && glob_match(&pattern[1..], &text[1..]),
    }
}
/// Encodes a reply as `RESP2`.
fn encode(reply: &Reply) -> Vec<u8> {
    match reply {
        Reply::Simple(text) => line(b'+', text.as_bytes()),
        Reply::Error(message) => line(b'-', message.as_bytes()),
        Reply::Integer(value) => format!(":{value}\r\n").into_bytes(),
        Reply::Bulk(bytes) => {
            let mut out = format!("${}\r\n", bytes.len()).into_bytes();
            out.extend_from_slice(bytes);
            out.extend_from_slice(b"\r\n");
            out
        }
        Reply::Null => b"$-1\r\n".to_vec(),
        Reply::Array(items) => {
            let mut out = format!("*{}\r\n", items.len()).into_bytes();
            for item in items {
                out.extend_from_slice(&encode(item));
            }
            out
        }
        other => line(
            b'-',
            format!("ERR unsupported reply {}", other.kind_name()).as_bytes(),
        ),
    }
}
fn line(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 3);
    out.push(tag);
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
    out
}
/// Writes one encoded reply to the connection.
async fn send_reply(stream: &mut TcpStream, reply: &Reply) -> std::io::Result<()> {
    stream.write_all(&encode(reply)).await?;
    stream.flush().await
}
