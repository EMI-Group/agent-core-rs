//! A tiny in-memory RESP2 server for the Redis checkpointer tests.
//!
//! The backend under test talks real RESP2 over a real socket, so the tests
//! drive it against this loopback server instead of mocking the client. The
//! server implements the command subset the checkpointer uses, keeps a command
//! log (so tests can assert on `EXPIRE`, `SELECT`, `AUTH`, …), mimics Redis's
//! rule that a set with no members disappears, and is aborted when
//! [`FakeRedis`] is dropped.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

/// A loopback RESP2 server with in-memory strings and sets.
pub(crate) struct FakeRedis {
    /// The bound `host:port` the server listens on.
    pub(crate) addr: String,
    commands: Arc<Mutex<Vec<Vec<String>>>>,
    hang_up: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

impl Drop for FakeRedis {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl FakeRedis {
    /// Starts a server on `127.0.0.1:0` and returns its handle.
    pub(crate) async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the fake redis listener");
        let addr = listener
            .local_addr()
            .expect("read the fake redis address")
            .to_string();
        let commands = Arc::new(Mutex::new(Vec::new()));
        let hang_up = Arc::new(AtomicBool::new(false));
        let store = Arc::new(Mutex::new(Store::default()));
        let handle = tokio::spawn(serve(
            listener,
            store,
            Arc::clone(&commands),
            Arc::clone(&hang_up),
        ));
        Self {
            addr,
            commands,
            hang_up,
            handle,
        }
    }

    /// Returns a `redis://host:port` URL for this server.
    pub(crate) fn url(&self) -> String {
        format!("redis://{}", self.addr)
    }

    /// Returns a `redis://credentials@host:port[/db]` URL for this server.
    pub(crate) fn url_with(&self, credentials: &str) -> String {
        format!("redis://{credentials}@{}", self.addr)
    }

    /// Returns a copy of the command log, oldest first; each entry is the
    /// command name followed by its arguments.
    pub(crate) fn commands(&self) -> Vec<Vec<String>> {
        self.commands.lock().expect("read the command log").clone()
    }

    /// Counts the logged commands whose name matches `name` (case-insensitive).
    pub(crate) fn count(&self, name: &str) -> usize {
        self.commands()
            .iter()
            .filter(|command| {
                command
                    .first()
                    .is_some_and(|logged| logged.eq_ignore_ascii_case(name))
            })
            .count()
    }

    /// Makes the server close every client connection right after replying,
    /// which forces clients to notice the dropped socket and reconnect.
    pub(crate) fn hang_up_after_each_reply(&self) {
        self.hang_up.store(true, Ordering::SeqCst);
    }

    /// Returns the first logged command whose name matches `name`.
    pub(crate) fn command(&self, name: &str) -> Option<Vec<String>> {
        self.commands().into_iter().find(|command| {
            command
                .first()
                .is_some_and(|logged| logged.eq_ignore_ascii_case(name))
        })
    }
}

/// The in-memory keyspace.
#[derive(Debug, Default)]
struct Store {
    strings: BTreeMap<Vec<u8>, Vec<u8>>,
    sets: BTreeMap<Vec<u8>, BTreeSet<Vec<u8>>>,
}

/// Accepts connections until the listener is dropped.
async fn serve(
    listener: TcpListener,
    store: Arc<Mutex<Store>>,
    commands: Arc<Mutex<Vec<Vec<String>>>>,
    hang_up: Arc<AtomicBool>,
) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let store = Arc::clone(&store);
        let commands = Arc::clone(&commands);
        let hang_up = Arc::clone(&hang_up);
        tokio::spawn(async move { handle_client(stream, store, commands, hang_up).await });
    }
}

/// Serves one client connection until it closes.
async fn handle_client(
    stream: TcpStream,
    store: Arc<Mutex<Store>>,
    commands: Arc<Mutex<Vec<Vec<String>>>>,
    hang_up: Arc<AtomicBool>,
) {
    let mut reader = BufReader::new(stream);
    loop {
        let Ok(Some(command)) = read_command(&mut reader).await else {
            return;
        };
        if command.is_empty() {
            continue;
        }
        commands.lock().expect("append to the command log").push(
            command
                .iter()
                .map(|argument| String::from_utf8_lossy(argument).into_owned())
                .collect(),
        );
        let reply = {
            let mut store = store.lock().expect("lock the fake store");
            execute(&mut store, &command)
        };
        if reader.get_mut().write_all(&reply).await.is_err() {
            return;
        }
        if hang_up.load(Ordering::SeqCst) {
            // Dropping the stream closes the socket, so the client's next
            // command hits a transport failure and must reconnect.
            return;
        }
    }
}

/// Reads one RESP array of bulk strings.
async fn read_command<R>(reader: &mut R) -> std::io::Result<Option<Vec<Vec<u8>>>>
where
    R: AsyncBufRead + Unpin,
{
    let mut line = Vec::new();
    if reader.read_until(b'\n', &mut line).await? == 0 {
        return Ok(None);
    }
    let header = std::str::from_utf8(strip_crlf(&line))
        .map_err(|_| invalid("command header is not UTF-8"))?;
    let Some(count) = header.strip_prefix('*') else {
        return Err(invalid("expected a RESP array command"));
    };
    let count: usize = count
        .trim()
        .parse()
        .map_err(|_| invalid("malformed array length"))?;
    let mut arguments = Vec::with_capacity(count);
    for _ in 0..count {
        let mut header = Vec::new();
        if reader.read_until(b'\n', &mut header).await? == 0 {
            return Ok(None);
        }
        let header = std::str::from_utf8(strip_crlf(&header))
            .map_err(|_| invalid("bulk header is not UTF-8"))?;
        let Some(length) = header.strip_prefix('$') else {
            return Err(invalid("expected a bulk string argument"));
        };
        let length: usize = length
            .trim()
            .parse()
            .map_err(|_| invalid("malformed bulk length"))?;
        let mut payload = vec![0_u8; length];
        reader.read_exact(&mut payload).await?;
        let mut terminator = [0_u8; 2];
        reader.read_exact(&mut terminator).await?;
        arguments.push(payload);
    }
    Ok(Some(arguments))
}

/// Builds an invalid-data I/O error with `message`.
fn invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.to_string())
}

/// Strips a trailing `\r\n` (or bare `\n`) from `line`.
fn strip_crlf(line: &[u8]) -> &[u8] {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    line.strip_suffix(b"\r").unwrap_or(line)
}

/// Executes one command and returns its encoded reply.
fn execute(store: &mut Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    let name = String::from_utf8_lossy(&arguments[0]).to_ascii_uppercase();
    match name.as_str() {
        "PING" => simple("PONG"),
        "AUTH" | "SELECT" => simple("OK"),
        "FLUSHDB" => {
            store.strings.clear();
            store.sets.clear();
            simple("OK")
        }
        "GET" => get(store, arguments),
        "SET" => set(store, arguments),
        "EXPIRE" => expire(store, arguments),
        "DEL" => delete(store, arguments),
        "SADD" => sadd(store, arguments),
        "SREM" => srem(store, arguments),
        "SMEMBERS" => smembers(store, arguments),
        "EXISTS" => exists(store, arguments),
        _ => error_reply(&format!(
            "ERR unknown command '{}'",
            String::from_utf8_lossy(&arguments[0])
        )),
    }
}

/// Handles `GET key`.
fn get(store: &Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    let Some(key) = arguments.get(1) else {
        return wrong_arity();
    };
    match store.strings.get(key) {
        Some(value) => bulk(value),
        None => nil_bulk(),
    }
}

/// Handles `SET key value [EX seconds | PX milliseconds]`.
fn set(store: &mut Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    if arguments.len() < 3 {
        return wrong_arity();
    }
    store
        .strings
        .insert(arguments[1].clone(), arguments[2].clone());
    // `EX <seconds>` / `PX <milliseconds>` are accepted and ignored: the tests
    // assert on the command log instead of on expiry.
    let mut index = 3;
    while index < arguments.len() {
        let option = String::from_utf8_lossy(&arguments[index]).to_ascii_uppercase();
        index += if option == "EX" || option == "PX" {
            2
        } else {
            1
        };
    }
    simple("OK")
}

/// Handles `EXPIRE key seconds`; the TTL itself is ignored.
fn expire(store: &Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    if arguments.len() < 3 {
        return wrong_arity();
    }
    let exists =
        store.strings.contains_key(&arguments[1]) || store.sets.contains_key(&arguments[1]);
    integer(i64::from(exists))
}

/// Handles `DEL key...`.
fn delete(store: &mut Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    let mut removed = 0;
    for key in &arguments[1..] {
        if store.strings.remove(key).is_some() {
            removed += 1;
        }
        if store.sets.remove(key).is_some() {
            removed += 1;
        }
    }
    integer(removed)
}

/// Handles `SADD key member...`.
fn sadd(store: &mut Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    if arguments.len() < 3 {
        return wrong_arity();
    }
    let members = store.sets.entry(arguments[1].clone()).or_default();
    let mut added = 0;
    for member in &arguments[2..] {
        if members.insert(member.clone()) {
            added += 1;
        }
    }
    integer(added)
}

/// Handles `SREM key member...`; Redis drops a set once its last member goes.
fn srem(store: &mut Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    if arguments.len() < 3 {
        return wrong_arity();
    }
    let mut removed = 0;
    if let Some(members) = store.sets.get_mut(&arguments[1]) {
        for member in &arguments[2..] {
            if members.remove(member) {
                removed += 1;
            }
        }
        if members.is_empty() {
            store.sets.remove(&arguments[1]);
        }
    }
    integer(removed)
}

/// Handles `SMEMBERS key`.
fn smembers(store: &Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    let Some(key) = arguments.get(1) else {
        return wrong_arity();
    };
    let items: Vec<Vec<u8>> = store
        .sets
        .get(key)
        .map(|members| members.iter().map(|member| bulk(member)).collect())
        .unwrap_or_default();
    array(&items)
}

/// Handles `EXISTS key...`.
fn exists(store: &Store, arguments: &[Vec<u8>]) -> Vec<u8> {
    let mut found = 0;
    for key in &arguments[1..] {
        if store.strings.contains_key(key) || store.sets.contains_key(key) {
            found += 1;
        }
    }
    integer(found)
}

/// Encodes a simple-string reply.
fn simple(text: &str) -> Vec<u8> {
    format!("+{text}\r\n").into_bytes()
}

/// Encodes an error reply.
fn error_reply(text: &str) -> Vec<u8> {
    format!("-{text}\r\n").into_bytes()
}

/// Encodes an integer reply.
fn integer(value: i64) -> Vec<u8> {
    format!(":{value}\r\n").into_bytes()
}

/// Encodes a bulk-string reply.
fn bulk(value: &[u8]) -> Vec<u8> {
    let mut encoded = format!("${}\r\n", value.len()).into_bytes();
    encoded.extend_from_slice(value);
    encoded.extend_from_slice(b"\r\n");
    encoded
}

/// Encodes the nil bulk reply.
fn nil_bulk() -> Vec<u8> {
    b"$-1\r\n".to_vec()
}

/// Encodes an array reply from already-encoded elements.
fn array(items: &[Vec<u8>]) -> Vec<u8> {
    let mut encoded = format!("*{}\r\n", items.len()).into_bytes();
    for item in items {
        encoded.extend_from_slice(item);
    }
    encoded
}

/// Encodes the wrong-arity error reply.
fn wrong_arity() -> Vec<u8> {
    error_reply("ERR wrong number of arguments for command")
}
