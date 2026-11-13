//! Stateful in-process fake `Redis` server used by the client and adapter tests.
//!
//! [`FakeRedis`] binds an ephemeral loopback port, parses requests with
//! [`crate::redis::codec::parse_reply`], and answers `RESP2`, so no external
//! service or dependency is needed. It keeps string values, sets, and sorted
//! sets with lazy TTL expiry (checked on every command and on every state read,
//! so assertions are deterministic without background tasks).
//!
//! Scripted modes: [`FakeOptions::auth_required`] demands `AUTH` before any other
//! command, [`FakeOptions::moved_everything`] answers every non-handshake command
//! with `-MOVED 3999 127.0.0.1:7000`, and [`FakeOptions::with_command_delay`] makes
//! every command slow enough to exercise client timeouts.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use super::codec::{parse_reply, Reply};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::time::Instant;
mod commands;
mod support;

/// Scripted behavior for a [`FakeRedis`] instance.
#[derive(Debug, Clone, Default)]
pub(crate) struct FakeOptions {
    /// When set, every connection must `AUTH` with this `(username, password)`.
    pub(crate) require_auth: Option<(Option<String>, String)>,
    /// When set, answer every non-handshake command with a `MOVED` redirection.
    pub(crate) moved_everything: bool,
    /// Artificial delay applied before every command is executed.
    pub(crate) command_delay: Option<Duration>,
}
impl FakeOptions {
    /// Requires `AUTH` with the given user/password before any other command.
    #[must_use]
    pub(crate) fn auth_required(username: Option<&str>, password: &str) -> Self {
        Self {
            require_auth: Some((username.map(ToOwned::to_owned), password.to_owned())),
            ..Self::default()
        }
    }
    /// Answers every non-handshake command with a `MOVED` redirection.
    #[must_use]
    pub(crate) fn moved_everything() -> Self {
        Self {
            moved_everything: true,
            ..Self::default()
        }
    }
    /// Adds an artificial delay before every command is executed.
    #[must_use]
    pub(crate) fn with_command_delay(mut self, delay: Duration) -> Self {
        self.command_delay = Some(delay);
        self
    }
}
/// A running fake `Redis` server; call [`FakeRedis::shutdown`] to stop it.
pub(crate) struct FakeRedis {
    addr: SocketAddr,
    state: StateHandle,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}
impl FakeRedis {
    /// Starts a fake server with default behavior (no auth, no redirection).
    pub(crate) async fn start() -> Self {
        Self::start_with(FakeOptions::default()).await
    }
    /// Starts a fake server with scripted behavior.
    pub(crate) async fn start_with(options: FakeOptions) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("fake redis binds an ephemeral loopback port");
        let addr = listener
            .local_addr()
            .expect("bound loopback listener has a local address");
        let state = Arc::new(ServerState::new(options));
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        let accept_state = Arc::clone(&state);
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _peer)) = accepted else { break };
                        let connection_state = Arc::clone(&accept_state);
                        tokio::spawn(serve_connection(stream, connection_state));
                    }
                }
            }
        });
        Self {
            addr,
            state: StateHandle { inner: state },
            shutdown: Some(shutdown_tx),
            task,
        }
    }
    /// A `redis://` URL pointing at this server with database `0`.
    pub(crate) fn url(&self) -> String {
        format!("redis://{}/0", self.addr)
    }
    /// The loopback address the server listens on.
    pub(crate) fn addr(&self) -> SocketAddr {
        self.addr
    }
    /// A handle for asserting on server state.
    pub(crate) fn state(&self) -> StateHandle {
        self.state.clone()
    }
    /// Stops the listener and waits for the accept loop to finish.
    pub(crate) async fn shutdown(mut self) {
        if let Some(sender) = self.shutdown.take() {
            let _ = sender.send(());
        }
        let _ = self.task.await;
    }
}
/// Snapshot/assertion handle over a fake server's data.
#[derive(Clone)]
pub(crate) struct StateHandle {
    inner: Arc<ServerState>,
}
impl StateHandle {
    /// The stored string value for `key`, or `None` when absent or expired.
    pub(crate) fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.with_data(|data| data.values.get(key.as_bytes()).cloned())
    }
    /// All live keys, sorted.
    pub(crate) fn keys(&self) -> Vec<String> {
        self.with_data(|data| {
            data.keys()
                .into_iter()
                .map(|key| String::from_utf8_lossy(&key).into_owned())
                .collect()
        })
    }
    /// The members of the set at `key`, sorted.
    pub(crate) fn set_members(&self, key: &str) -> Vec<String> {
        self.with_data(|data| {
            data.sets.get(key.as_bytes()).map_or_else(Vec::new, |set| {
                set.iter()
                    .map(|member| String::from_utf8_lossy(member).into_owned())
                    .collect()
            })
        })
    }
    /// The `(member, score)` pairs of the sorted set at `key`, in member order.
    pub(crate) fn zset(&self, key: &str) -> Vec<(String, f64)> {
        self.with_data(|data| {
            data.zsets
                .get(key.as_bytes())
                .map_or_else(Vec::new, |zset| {
                    zset.iter()
                        .map(|(member, score)| {
                            (String::from_utf8_lossy(member).into_owned(), *score)
                        })
                        .collect()
                })
        })
    }
    /// Remaining TTL in milliseconds: `None` when the key is absent, `Some(-1)`
    /// when it has no expiry, otherwise the remaining time.
    pub(crate) fn ttl_ms(&self, key: &str) -> Option<i64> {
        self.with_data(|data| {
            if !data.exists(key.as_bytes()) {
                return None;
            }
            data.expiries
                .get(key.as_bytes())
                .map(|at| {
                    i64::try_from(at.saturating_duration_since(Instant::now()).as_millis())
                        .unwrap_or(i64::MAX)
                })
                .or(Some(-1))
        })
    }
    /// Number of accepted client connections so far.
    pub(crate) fn connection_count(&self) -> usize {
        self.inner.connections.load(Ordering::Relaxed)
    }
    /// Number of `AUTH` commands received (successful or not).
    pub(crate) fn auth_attempts(&self) -> usize {
        self.inner.auth_attempts.load(Ordering::Relaxed)
    }
    /// Upper-cased command names received so far, in order.
    pub(crate) fn commands(&self) -> Vec<String> {
        self.inner
            .commands
            .lock()
            .expect("fake redis command log mutex is not poisoned")
            .clone()
    }
    /// Runs `read` against freshly expired-entry-purged state.
    fn with_data<T>(&self, read: impl FnOnce(&mut Data) -> T) -> T {
        let mut data = self
            .inner
            .data
            .lock()
            .expect("fake redis state mutex is not poisoned");
        data.purge();
        read(&mut data)
    }
}
/// Shared server state.
struct ServerState {
    options: FakeOptions,
    data: StdMutex<Data>,
    connections: AtomicUsize,
    auth_attempts: AtomicUsize,
    commands: StdMutex<Vec<String>>,
}
/// Per-connection state.
struct ConnState {
    authed: bool,
    #[allow(dead_code)]
    db: Option<i64>,
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
    /// Drops every key whose TTL has elapsed.
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
    /// Whether `key` exists in any namespace (after purging).
    fn exists(&self, key: &[u8]) -> bool {
        self.values.contains_key(key) || self.sets.contains_key(key) || self.zsets.contains_key(key)
    }
    /// Removes `key` from every namespace, reporting whether it existed.
    fn remove(&mut self, key: &[u8]) -> bool {
        let removed_value = self.values.remove(key).is_some();
        let removed_set = self.sets.remove(key).is_some();
        let removed_zset = self.zsets.remove(key).is_some();
        self.expiries.remove(key);
        removed_value || removed_set || removed_zset
    }
    /// All live keys across namespaces, sorted and deduplicated.
    fn keys(&self) -> Vec<Vec<u8>> {
        let mut keys: Vec<Vec<u8>> = Vec::new();
        keys.extend(self.values.keys().cloned());
        keys.extend(self.sets.keys().cloned());
        keys.extend(self.zsets.keys().cloned());
        keys.sort();
        keys.dedup();
        keys
    }
}
/// Accepts connections until the listener is shut down.
async fn serve_connection(mut stream: TcpStream, state: Arc<ServerState>) {
    state.connections.fetch_add(1, Ordering::Relaxed);
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut connection = ConnState {
        authed: state.options.require_auth.is_none(),
        db: None,
    };
    loop {
        let Ok(parsed) = parse_reply(&buffer) else {
            return;
        };
        if let Some((request, consumed)) = parsed {
            buffer.drain(..consumed);
            let Some(args) = request_as_args(&request) else {
                let reply = Reply::Error(
                    "ERR Protocol error: expected an array of bulk strings".to_owned(),
                );
                if write_reply(&mut stream, &reply).await.is_err() {
                    return;
                }
                continue;
            };
            if let Some(delay) = state.options.command_delay {
                tokio::time::sleep(delay).await;
            }
            let reply = state.execute(&args, &mut connection);
            if write_reply(&mut stream, &reply).await.is_err() {
                return;
            }
            continue;
        }
        let read = match stream.read(&mut chunk).await {
            Ok(read) if read > 0 => read,
            _ => return,
        };
        buffer.extend_from_slice(&chunk[..read]);
    }
}
/// Converts a request array of bulk/simple strings into command arguments.
fn request_as_args(request: &Reply) -> Option<Vec<Vec<u8>>> {
    let Reply::Array(items) = request else {
        return None;
    };
    let mut args = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Reply::Bulk(bytes) => args.push(bytes.clone()),
            Reply::Simple(text) => args.push(text.as_bytes().to_vec()),
            _ => return None,
        }
    }
    Some(args)
}
/// Encodes a reply as `RESP2` (`RESP3` values fall back to their `RESP2` form).
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
        Reply::Array(items) => aggregate(b'*', items),
        Reply::Boolean(value) => format!(":{}\r\n", i32::from(*value)).into_bytes(),
        Reply::Double(value) => line(b'+', format!("{value}").as_bytes()),
        Reply::Map(pairs) => {
            let mut out = format!("%{}\r\n", pairs.len()).into_bytes();
            for (key, value) in pairs {
                out.extend_from_slice(&encode_reply(key));
                out.extend_from_slice(&encode_reply(value));
            }
            out
        }
        Reply::Set(items) => aggregate(b'~', items),
        Reply::Push(items) => aggregate(b'>', items),
    }
}
/// Builds a `tag`-prefixed aggregate reply.
fn aggregate(tag: u8, items: &[Reply]) -> Vec<u8> {
    let mut out = format!("{}{}\r\n", char::from(tag), items.len()).into_bytes();
    for item in items {
        out.extend_from_slice(&encode_reply(item));
    }
    out
}
/// Builds a single `RESP` line.
fn line(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 3);
    out.push(tag);
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
    out
}
/// Writes one encoded reply to the connection.
async fn write_reply(stream: &mut TcpStream, reply: &Reply) -> std::io::Result<()> {
    stream.write_all(&encode_reply(reply)).await?;
    stream.flush().await
}
