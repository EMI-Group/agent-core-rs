//! A minimal in-process `RESP2` server for the Redis store tests.
//!
//! [`FakeRedis`] binds an ephemeral loopback port and speaks enough of the
//! protocol for [`RedisOnlineStore`](super::super::store) to be exercised offline:
//! `PING`, `ECHO`, `AUTH`, `SELECT`, `GET`, `SET` (with `NX`), `DEL`, `EXISTS`,
//! `SADD`, `SREM`, `SMEMBERS`, `SISMEMBER` and `FLUSHDB`. An unknown command
//! answers a `RESP` error, so a mistake in the command mapping fails loudly
//! instead of silently passing.
//!
//! The server keeps string values and sets — the two namespaces the store uses —
//! and answers a `WRONGTYPE` error when a key is read through the wrong one, so
//! the store's error surfacing can be tested too.
//!
//! Requests are parsed with the client's own codec
//! ([`openjiuwen_extensions::redis::codec`]), so the test server and the client
//! cannot drift on framing; only the reply encoder is local.
//!
//! Assertion helpers: [`FakeRedis::url`] (the `redis://` URL), [`FakeRedis::state`]
//! (`get`/`members`/`keys` snapshots), [`FakeRedis::command_count`] and
//! [`FakeRedis::shutdown`]. No external service and no fixed port is involved.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use openjiuwen_extensions::redis::codec::{parse_reply, Reply};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

/// A running fake server; call [`FakeRedis::shutdown`] to stop it.
pub(super) struct FakeRedis {
    addr: SocketAddr,
    state: StateHandle,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeRedis {
    /// Starts a server on an ephemeral loopback port.
    pub(super) async fn start() -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("fake redis binds an ephemeral loopback port");
        let addr = listener
            .local_addr()
            .expect("bound loopback listener has a local address");
        let shared = Arc::new(Shared::default());
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        let accept_shared = Arc::clone(&shared);
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _peer)) = accepted else { break };
                        let connection_shared = Arc::clone(&accept_shared);
                        tokio::spawn(serve_connection(stream, connection_shared));
                    }
                }
            }
        });
        Self {
            addr,
            state: StateHandle { inner: shared },
            shutdown: Some(shutdown_tx),
            task,
        }
    }

    /// A `redis://` URL pointing at this server with database `0`.
    pub(super) fn url(&self) -> String {
        self.url_with("redis")
    }

    /// A URL with `scheme` (`rediss`, `redis+cluster`, …) at this server, database `0`.
    ///
    /// The server speaks plain `RESP2`, so a TLS scheme connects to a peer that
    /// cannot complete a handshake and a cluster scheme finds no cluster node.
    pub(super) fn url_with(&self, scheme: &str) -> String {
        format!("{scheme}://{}/0", self.addr)
    }

    /// A handle for asserting on the server's data.
    pub(super) fn state(&self) -> StateHandle {
        self.state.clone()
    }

    /// Number of commands the server executed (handshake included).
    pub(super) fn command_count(&self) -> usize {
        self.state.inner.commands.load(Ordering::Relaxed)
    }

    /// Stops the listener and waits for the accept loop to finish.
    pub(super) async fn shutdown(mut self) {
        if let Some(sender) = self.shutdown.take() {
            let _ = sender.send(());
        }
        let _ = self.task.await;
    }
}

/// Snapshot/assertion handle over a fake server's data.
#[derive(Clone)]
pub(super) struct StateHandle {
    inner: Arc<Shared>,
}

impl StateHandle {
    /// The stored string value for `key`, or `None` when absent or a set.
    pub(super) fn get(&self, key: &str) -> Option<String> {
        self.with_data(|data| data.values.get(key).cloned())
    }

    /// The members of the set at `key`, sorted.
    pub(super) fn members(&self, key: &str) -> Vec<String> {
        self.with_data(|data| {
            data.sets
                .get(key)
                .map(|members| members.iter().cloned().collect())
                .unwrap_or_default()
        })
    }

    /// Every live key holding data, sorted.
    pub(super) fn keys(&self) -> Vec<String> {
        self.with_data(|data| {
            let mut keys: BTreeSet<String> = data.values.keys().cloned().collect();
            keys.extend(
                data.sets
                    .iter()
                    .filter(|(_, members)| !members.is_empty())
                    .map(|(key, _)| key.clone()),
            );
            keys.into_iter().collect()
        })
    }

    fn with_data<T>(&self, read: impl FnOnce(&Data) -> T) -> T {
        let data = self
            .inner
            .data
            .lock()
            .expect("fake redis state mutex is not poisoned");
        read(&data)
    }
}

/// Server state shared by every connection.
#[derive(Default)]
struct Shared {
    data: StdMutex<Data>,
    commands: AtomicUsize,
}

/// Stored data: the two namespaces the online store uses.
#[derive(Default)]
struct Data {
    values: BTreeMap<String, String>,
    sets: BTreeMap<String, BTreeSet<String>>,
}

impl Data {
    fn exists(&self, key: &str) -> bool {
        self.values.contains_key(key) || self.sets.contains_key(key)
    }

    fn remove(&mut self, key: &str) -> bool {
        let removed_value = self.values.remove(key).is_some();
        let removed_set = self.sets.remove(key).is_some();
        removed_value || removed_set
    }
}

/// Accepts commands on one connection until it closes.
async fn serve_connection(mut stream: TcpStream, shared: Arc<Shared>) {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let parsed = match parse_reply(&buffer) {
            Ok(parsed) => parsed,
            Err(_) => return,
        };
        if let Some((request, consumed)) = parsed {
            buffer.drain(..consumed);
            let reply = match request_args(&request) {
                Some(args) => {
                    shared.commands.fetch_add(1, Ordering::Relaxed);
                    execute(&args, &shared)
                }
                None => {
                    Reply::Error("ERR Protocol error: expected an array of bulk strings".to_owned())
                }
            };
            if write_reply(&mut stream, &reply).await.is_err() {
                return;
            }
            continue;
        }
        match stream.read(&mut chunk).await {
            Ok(read) if read > 0 => buffer.extend_from_slice(&chunk[..read]),
            _ => return,
        }
    }
}

/// Converts a request array of bulk/simple strings into command arguments.
fn request_args(request: &Reply) -> Option<Vec<String>> {
    let Reply::Array(items) = request else {
        return None;
    };
    let mut args = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Reply::Bulk(bytes) => args.push(String::from_utf8_lossy(bytes).into_owned()),
            Reply::Simple(text) => args.push(text.clone()),
            _ => return None,
        }
    }
    Some(args)
}

/// Executes one command.
fn execute(args: &[String], shared: &Shared) -> Reply {
    let Some(name) = args.first() else {
        return Reply::Error("ERR empty command".to_owned());
    };
    let name = name.to_ascii_uppercase();
    let mut data = shared
        .data
        .lock()
        .expect("fake redis state mutex is not poisoned");
    match name.as_str() {
        "PING" => match args.get(1) {
            Some(message) => Reply::Bulk(message.clone().into_bytes()),
            None => Reply::Simple("PONG".to_owned()),
        },
        "ECHO" => match args.get(1) {
            Some(message) => Reply::Bulk(message.clone().into_bytes()),
            None => Reply::Error("ERR wrong number of arguments for 'echo' command".to_owned()),
        },
        "AUTH" | "SELECT" | "FLUSHDB" => {
            if name == "FLUSHDB" {
                *data = Data::default();
            }
            Reply::Simple("OK".to_owned())
        }
        "GET" => match args.get(1) {
            Some(key) => {
                if data.sets.contains_key(key) {
                    return wrong_type();
                }
                match data.values.get(key) {
                    Some(value) => Reply::Bulk(value.clone().into_bytes()),
                    None => Reply::Null,
                }
            }
            None => arity_error(&name),
        },
        "SET" => write_command(args, &mut data),
        "DEL" => {
            if args.len() < 2 {
                return arity_error(&name);
            }
            let removed = args[1..].iter().filter(|key| data.remove(key)).count();
            counted(removed)
        }
        "EXISTS" => {
            if args.len() < 2 {
                return arity_error(&name);
            }
            let existing = args[1..].iter().filter(|key| data.exists(key)).count();
            counted(existing)
        }
        "SADD" | "SREM" | "SMEMBERS" | "SISMEMBER" => set_command(&name, args, &mut data),
        _ => Reply::Error(format!("ERR unknown command '{name}'")),
    }
}

/// Executes one set command (`SADD`/`SREM`/`SMEMBERS`/`SISMEMBER`).
fn set_command(name: &str, args: &[String], data: &mut Data) -> Reply {
    match name {
        "SADD" => {
            if args.len() < 3 {
                return arity_error(name);
            }
            let key = &args[1];
            if data.values.contains_key(key) {
                return wrong_type();
            }
            let set = data.sets.entry(key.clone()).or_default();
            let added = args[2..]
                .iter()
                .filter(|member| set.insert((*member).clone()))
                .count();
            counted(added)
        }
        "SREM" => {
            if args.len() < 3 {
                return arity_error(name);
            }
            let key = &args[1];
            if data.values.contains_key(key) {
                return wrong_type();
            }
            let Some(set) = data.sets.get_mut(key) else {
                return Reply::Integer(0);
            };
            let removed = args[2..]
                .iter()
                .filter(|member| set.remove(*member))
                .count();
            counted(removed)
        }
        "SMEMBERS" => match args.get(1) {
            Some(key) => {
                if data.values.contains_key(key) {
                    return wrong_type();
                }
                Reply::Array(
                    data.sets
                        .get(key)
                        .map(|members| {
                            members
                                .iter()
                                .map(|member| Reply::Bulk(member.clone().into_bytes()))
                                .collect()
                        })
                        .unwrap_or_default(),
                )
            }
            None => arity_error(name),
        },
        _ => {
            // SISMEMBER
            if args.len() < 3 {
                return arity_error(name);
            }
            let key = &args[1];
            if data.values.contains_key(key) {
                return wrong_type();
            }
            let present = data
                .sets
                .get(key)
                .is_some_and(|members| members.contains(&args[2]));
            Reply::Integer(i64::from(present))
        }
    }
}

/// Wraps a command's affected count into an integer reply.
fn counted(count: usize) -> Reply {
    Reply::Integer(i64::try_from(count).unwrap_or(i64::MAX))
}

/// `SET key value [NX]`; the online store only uses the `NX` form.
fn write_command(args: &[String], data: &mut Data) -> Reply {
    if args.len() < 3 {
        return arity_error("SET");
    }
    let key = &args[1];
    let only_if_absent = args
        .get(3)
        .is_some_and(|option| option.eq_ignore_ascii_case("NX"));
    if data.sets.contains_key(key) {
        return wrong_type();
    }
    if only_if_absent && data.values.contains_key(key) {
        return Reply::Null;
    }
    data.values.insert(key.clone(), args[2].clone());
    Reply::Simple("OK".to_owned())
}

fn wrong_type() -> Reply {
    Reply::Error("WRONGTYPE Operation against a key holding the wrong kind of value".to_owned())
}

fn arity_error(command: &str) -> Reply {
    Reply::Error(format!(
        "ERR wrong number of arguments for '{}' command",
        command.to_ascii_lowercase()
    ))
}

/// Writes one `RESP2` reply.
async fn write_reply(stream: &mut TcpStream, reply: &Reply) -> std::io::Result<()> {
    stream.write_all(&encode_reply(reply)).await?;
    stream.flush().await
}

/// Encodes a reply as `RESP2`.
fn encode_reply(reply: &Reply) -> Vec<u8> {
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
                out.extend_from_slice(&encode_reply(item));
            }
            out
        }
        Reply::Boolean(value) => format!(":{}\r\n", i32::from(*value)).into_bytes(),
        Reply::Double(value) => line(b'+', format!("{value}").as_bytes()),
        Reply::Map(_) | Reply::Set(_) | Reply::Push(_) => {
            // The online store never writes these; a client that asked for one
            // would decode an empty array rather than hang.
            b"*0\r\n".to_vec()
        }
    }
}

/// Builds a single `RESP` line.
fn line(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 3);
    out.push(tag);
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
    out
}
