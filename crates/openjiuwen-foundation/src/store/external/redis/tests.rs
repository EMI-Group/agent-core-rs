//! Offline tests for the Redis backend.
//!
//! A scripted fake Redis server stands in for a real one: a [`TcpListener`]
//! decodes client commands with this module's own [`Decoder`] and answers with
//! hand-built RESP frames. Nothing here leaves the loopback interface, and no
//! test needs a running server.
//!
//! The `rediss://` transport is checked with [`RawProbe`], a protocol-blind
//! loopback server that records the first bytes a client writes: those must be a
//! TLS handshake record, so the TLS path is proven without completing a handshake
//! (which would need a trusted certificate chain the offline build cannot mint).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::SecretString;

use super::client::RedisConnection;
use super::codec::{Decoder, RespValue};
use super::store::RedisKvStore;
use super::RedisConfig;
use crate::store::kv::{KvStore, KvValue};

/// What the fake server does with one command.
struct Reply {
    /// RESP bytes written back before the connection decides what to do next.
    bytes: Vec<u8>,
    /// Whether the connection closes right after writing `bytes`.
    close: bool,
}

impl Reply {
    /// Replies with `bytes` and keeps the connection open.
    fn bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            close: false,
        }
    }

    /// Closes the connection without writing anything.
    fn close() -> Self {
        Self {
            bytes: Vec::new(),
            close: true,
        }
    }
}

/// One command handler shared by the fake server and the test.
type Handler = Arc<dyn Fn(&[Vec<u8>]) -> Reply + Send + Sync>;

/// A fake Redis server listening on a loopback port.
struct FakeRedis {
    /// `host:port` the server listens on.
    addr: String,
    /// Every command received, rendered as text for assertions.
    commands: Arc<Mutex<Vec<Vec<String>>>>,
    /// The accept loop; aborted when the fixture drops.
    task: tokio::task::JoinHandle<()>,
}

impl Drop for FakeRedis {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl FakeRedis {
    /// Starts a server that answers each command with `handler`.
    async fn start(handler: Handler) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr").to_string();
        let commands = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&commands);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let handler = Arc::clone(&handler);
                let recorder = Arc::clone(&recorder);
                tokio::spawn(async move { serve(stream, handler, recorder).await });
            }
        });
        Self {
            addr,
            commands,
            task,
        }
    }

    /// Every command the server received so far.
    fn commands(&self) -> Vec<Vec<String>> {
        self.commands.lock().expect("commands").clone()
    }
}

/// Serves one client connection until it closes.
async fn serve(mut stream: TcpStream, handler: Handler, recorder: Arc<Mutex<Vec<Vec<String>>>>) {
    let mut decoder = Decoder::new();
    let mut buffer = [0_u8; 4096];
    'connection: loop {
        match stream.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => decoder.push(&buffer[..read]),
        }
        loop {
            let Some(frame) = decoder.try_next().expect("client command") else {
                break;
            };
            let args = command_args(frame).expect("command array");
            recorder.lock().expect("commands").push(
                args.iter()
                    .map(|arg| String::from_utf8_lossy(arg).into_owned())
                    .collect(),
            );
            let reply = handler(&args);
            if stream.write_all(&reply.bytes).await.is_err() || reply.close {
                break 'connection;
            }
        }
    }
}

/// Converts an inbound command frame into its argument list.
fn command_args(frame: RespValue) -> Option<Vec<Vec<u8>>> {
    match frame {
        RespValue::Array(Some(items)) => items.into_iter().map(RespValue::into_bytes).collect(),
        _ => None,
    }
}

/// An in-memory stand-in for the Redis commands the backend uses.
struct MiniRedis {
    /// The keyspace.
    data: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
}

impl MiniRedis {
    /// Creates an empty shared keyspace.
    fn shared() -> Arc<Self> {
        Arc::new(Self {
            data: Mutex::new(HashMap::new()),
        })
    }

    /// Inserts one key directly, bypassing the protocol.
    fn insert(&self, key: &str, value: &[u8]) {
        self.data
            .lock()
            .expect("keyspace")
            .insert(key.as_bytes().to_vec(), value.to_vec());
    }

    /// Answers one command.
    fn handle(&self, args: &[Vec<u8>]) -> Reply {
        let name = args
            .first()
            .map(|name| String::from_utf8_lossy(name).to_ascii_uppercase())
            .unwrap_or_default();
        match name.as_str() {
            "AUTH" | "SELECT" => Reply::bytes(simple("OK")),
            "PING" => Reply::bytes(simple("PONG")),
            "SET" => self.set(args),
            "GET" => self.get(args),
            "EXISTS" => self.exists(args),
            "DEL" => self.delete(args),
            "MGET" => self.mget(args),
            "SCAN" => self.scan(args),
            other => Reply::bytes(server_error(&format!("ERR unknown command '{other}'"))),
        }
    }

    /// Handles `SET key value [NX] [EX seconds | PX milliseconds]`.
    fn set(&self, args: &[Vec<u8>]) -> Reply {
        if args.len() < 3 {
            return Reply::bytes(server_error("ERR wrong number of arguments"));
        }
        let key = args[1].clone();
        let value = args[2].clone();
        let mut exclusive = false;
        let mut index = 3;
        while index < args.len() {
            let option = args[index].to_ascii_uppercase();
            match option.as_slice() {
                b"NX" => {
                    exclusive = true;
                    index += 1;
                }
                b"EX" | b"PX" => {
                    if index + 1 >= args.len() {
                        return Reply::bytes(server_error("ERR syntax error"));
                    }
                    index += 2;
                }
                _ => return Reply::bytes(server_error("ERR syntax error")),
            }
        }
        let mut data = self.data.lock().expect("keyspace");
        if exclusive && data.contains_key(&key) {
            return Reply::bytes(nil());
        }
        data.insert(key, value);
        Reply::bytes(simple("OK"))
    }

    /// Handles `GET key`.
    fn get(&self, args: &[Vec<u8>]) -> Reply {
        let data = self.data.lock().expect("keyspace");
        match data.get(&args[1]) {
            Some(value) => Reply::bytes(bulk(value)),
            None => Reply::bytes(nil()),
        }
    }

    /// Handles `EXISTS key [key ...]`.
    fn exists(&self, args: &[Vec<u8>]) -> Reply {
        let data = self.data.lock().expect("keyspace");
        let count = args[1..]
            .iter()
            .filter(|key| data.contains_key(*key))
            .count();
        Reply::bytes(integer(i64::try_from(count).unwrap_or(i64::MAX)))
    }

    /// Handles `DEL key [key ...]`, counting keys that actually existed.
    fn delete(&self, args: &[Vec<u8>]) -> Reply {
        let mut data = self.data.lock().expect("keyspace");
        let mut deleted = 0;
        for key in &args[1..] {
            if data.remove(key).is_some() {
                deleted += 1;
            }
        }
        Reply::bytes(integer(deleted))
    }

    /// Handles `MGET key [key ...]`.
    fn mget(&self, args: &[Vec<u8>]) -> Reply {
        let data = self.data.lock().expect("keyspace");
        let values = args[1..]
            .iter()
            .map(|key| match data.get(key) {
                Some(value) => bulk(value),
                None => nil(),
            })
            .collect();
        Reply::bytes(array(values))
    }

    /// Handles `SCAN cursor MATCH pattern COUNT n`, always in one page.
    fn scan(&self, args: &[Vec<u8>]) -> Reply {
        let mut pattern = String::from("*");
        let mut index = 2;
        while index + 1 < args.len() {
            let option = args[index].to_ascii_uppercase();
            match option.as_slice() {
                b"MATCH" => {
                    pattern = String::from_utf8_lossy(&args[index + 1]).into_owned();
                    index += 2;
                }
                b"COUNT" => index += 2,
                _ => index += 1,
            }
        }
        let data = self.data.lock().expect("keyspace");
        let keys = data
            .keys()
            .filter(|key| glob_matches(&pattern, key))
            .map(|key| bulk(key))
            .collect();
        drop(data);
        Reply::bytes(array(vec![bulk(b"0"), array(keys)]))
    }
}

/// Matches a `prefix*` glob against `key`.
fn glob_matches(pattern: &str, key: &[u8]) -> bool {
    let key = String::from_utf8_lossy(key);
    match pattern.strip_suffix('*') {
        Some(prefix) => key.starts_with(prefix),
        None => key == pattern,
    }
}

/// Builds a handler over a [`MiniRedis`] keyspace.
fn mini_redis(state: &Arc<MiniRedis>) -> Handler {
    let state = Arc::clone(state);
    Arc::new(move |args: &[Vec<u8>]| state.handle(args))
}

/// Encodes `+text\r\n`.
fn simple(text: &str) -> Vec<u8> {
    format!("+{text}\r\n").into_bytes()
}

/// Encodes `-text\r\n`.
fn server_error(text: &str) -> Vec<u8> {
    format!("-{text}\r\n").into_bytes()
}

/// Encodes `:number\r\n`.
fn integer(number: i64) -> Vec<u8> {
    format!(":{number}\r\n").into_bytes()
}

/// Encodes `$len\r\n<payload>\r\n`.
fn bulk(payload: &[u8]) -> Vec<u8> {
    let mut out = format!("${}\r\n", payload.len()).into_bytes();
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
    out
}

/// Encodes the null bulk string `$-1\r\n`.
fn nil() -> Vec<u8> {
    b"$-1\r\n".to_vec()
}

/// Encodes an array of already-encoded frames.
fn array(items: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = format!("*{}\r\n", items.len()).into_bytes();
    for item in items {
        out.extend_from_slice(&item);
    }
    out
}

/// Whether the server saw exactly this command.
fn saw(commands: &[Vec<String>], expected: &[&str]) -> bool {
    commands.iter().any(|command| {
        command
            .iter()
            .map(String::as_str)
            .eq(expected.iter().copied())
    })
}

/// Counts how many commands started with `name`.
fn count_named(commands: &[Vec<String>], name: &str) -> usize {
    commands
        .iter()
        .filter(|command| command.first().is_some_and(|first| first == name))
        .count()
}

/// Connects a store to a fake server with no credentials.
async fn store_for(server: &FakeRedis) -> RedisKvStore {
    RedisConfig::new(server.addr.clone())
        .connect()
        .await
        .expect("connect")
}

#[tokio::test]
async fn connect_sends_auth_and_select() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = RedisConfig::new(server.addr.clone())
        .with_username("default")
        .with_password("s3cret")
        .with_db(2)
        .connect()
        .await
        .expect("connect");
    store.set("k", KvValue::text("v")).await.expect("set");

    let commands = server.commands();
    assert_eq!(commands[0], vec!["AUTH", "default", "s3cret"]);
    assert_eq!(commands[1], vec!["SELECT", "2"]);
    assert!(saw(&commands, &["SET", "k", "v"]));
}

#[tokio::test]
async fn url_supplies_credentials_and_database() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let config = RedisConfig::from_url(format!("redis://alice:pw@{}/4", server.addr));
    let store = config.connect().await.expect("connect");
    store.set("k", KvValue::text("v")).await.expect("set");

    let commands = server.commands();
    assert_eq!(commands[0], vec!["AUTH", "alice", "pw"]);
    assert_eq!(commands[1], vec!["SELECT", "4"]);
}

#[test]
fn rediss_url_parses_with_tls_and_fields() {
    let resolved = RedisConfig::from_url("rediss://user:pass@host:6390/2")
        .resolved()
        .expect("resolved");
    assert!(resolved.tls, "a rediss:// URL enables TLS");
    assert_eq!(resolved.addr, "host:6390");
    assert_eq!(resolved.username.as_deref(), Some("user"));
    assert_eq!(resolved.password.as_deref(), Some("pass"));
    assert_eq!(resolved.db, Some(2));
}

#[test]
fn rediss_url_without_port_keeps_the_default_port() {
    let resolved = RedisConfig::from_url("rediss://cache.internal")
        .resolved()
        .expect("resolved");
    assert!(resolved.tls);
    assert_eq!(resolved.addr, "cache.internal:6379");
    assert_eq!(resolved.db, None);
}

#[test]
fn redis_url_stays_cleartext() {
    let resolved = RedisConfig::from_url("redis://localhost:6390/3")
        .resolved()
        .expect("resolved");
    assert!(!resolved.tls);
    assert_eq!(resolved.addr, "localhost:6390");
    assert_eq!(resolved.db, Some(3));
}

#[test]
fn explicit_fields_override_a_rediss_url_without_disabling_tls() {
    let mut config = RedisConfig::from_url("rediss://user:pass@host:6390/2");
    config.addr = Some("other:7000".to_string());
    config.username = Some("alice".to_string());
    config.password = Some(SecretString::new("secret"));
    config.db = Some(5);

    let resolved = config.resolved().expect("resolved");
    assert_eq!(resolved.addr, "other:7000");
    assert_eq!(resolved.username.as_deref(), Some("alice"));
    assert_eq!(resolved.password.as_deref(), Some("secret"));
    assert_eq!(resolved.db, Some(5));
    assert!(
        resolved.tls,
        "a rediss URL is never downgraded to cleartext"
    );
}

#[test]
fn with_tls_enables_tls_for_an_explicit_addr() {
    let resolved = RedisConfig::new("127.0.0.1:6380")
        .with_tls(true)
        .resolved()
        .expect("resolved");
    assert!(resolved.tls);
    assert_eq!(resolved.addr, "127.0.0.1:6380");

    let plain = RedisConfig::new("127.0.0.1:6380")
        .resolved()
        .expect("resolved");
    assert!(!plain.tls, "an explicit addr is cleartext by default");

    let default = RedisConfig::default().resolved().expect("resolved");
    assert_eq!(default.addr, super::DEFAULT_ADDR);
    assert!(!default.tls);
}

#[test]
fn other_url_schemes_are_config_errors() {
    for url in [
        "http://localhost:6379",
        "https://localhost:6379",
        "unix:///tmp/redis.sock",
        "redis+tls://localhost:6379",
    ] {
        let error = RedisConfig::from_url(url)
            .resolved()
            .err()
            .expect("only redis:// and rediss:// are accepted");
        assert_eq!(error.kind(), ErrorKind::Config, "{url}: {error}");
        assert!(error.to_string().contains("scheme"), "{url}: {error}");
    }
}

#[tokio::test]
async fn rediss_connect_starts_a_tls_handshake_and_writes_no_resp() {
    let server = RawProbe::start(None).await;
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        RedisConnection::connect_tls(&server.addr, None, None),
    )
    .await
    .expect("the handshake must not hang")
    .expect_err("the probe closes the connection mid-handshake");

    assert!(
        matches!(error.kind(), ErrorKind::External | ErrorKind::Timeout),
        "unexpected error kind: {error}"
    );
    let received = server.received();
    assert!(
        received.len() >= 2,
        "no bytes reached the server: {received:?}"
    );
    assert_eq!(
        &received[..2],
        &[0x16, 0x03],
        "the first bytes must be a TLS handshake record, not a RESP command: {received:?}"
    );
    for marker in [b"AUTH".as_slice(), b"PING".as_slice(), b"SELECT".as_slice()] {
        assert!(
            !contains_sequence(&received, marker),
            "cleartext RESP reached the server: {received:?}"
        );
    }
}

#[tokio::test]
async fn rediss_handshake_timeout_is_a_typed_error() {
    let server = RawProbe::silent().await;
    let started = std::time::Instant::now();
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        RedisConnection::connect_tls_with_timeout(
            &server.addr,
            None,
            None,
            Duration::from_millis(300),
        ),
    )
    .await
    .expect("the configured timeout must fire well before the 5 s bound")
    .expect_err("a silent server never completes the handshake");

    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn rediss_config_connect_uses_the_tls_transport() {
    let server = RawProbe::start(None).await;
    let error = RedisConfig::from_url(format!("rediss://{}", server.addr))
        .connect()
        .await
        .expect_err("the probe closes the connection mid-handshake");
    assert!(
        matches!(error.kind(), ErrorKind::External | ErrorKind::Timeout),
        "{error}"
    );
    let received = server.received();
    assert!(
        received.starts_with(&[0x16, 0x03]),
        "RedisConfig::connect must dial TLS for a rediss:// URL: {received:?}"
    );
}

#[tokio::test]
async fn tls_host_that_is_not_a_server_name_is_a_config_error() {
    let error = RedisConnection::connect_tls("bad host:6390", None, None)
        .await
        .expect_err("a host with a space cannot become a TLS server name");
    assert_eq!(error.kind(), ErrorKind::Config, "{error}");
}

#[tokio::test]
async fn redis_connect_still_writes_plain_resp_to_the_same_probe() {
    let server = RawProbe::start(Some(simple("PONG"))).await;
    let mut connection = RedisConnection::connect(&server.addr, None, None)
        .await
        .expect("connect");
    assert!(!connection.is_tls());
    connection.ping().await.expect("ping");

    let received = server.received();
    assert!(
        received.starts_with(b"*1\r\n$4\r\nPING\r\n"),
        "redis:// must stay cleartext RESP: {received:?}"
    );
}

#[tokio::test]
async fn default_config_uses_the_documented_defaults() {
    let config = RedisConfig::default();
    assert_eq!(config.timeout, Duration::from_secs(5));
    assert_eq!(config.key_prefix, "");
    assert!(config.url.is_none());
    assert!(config.addr.is_none());
}

#[tokio::test]
async fn from_env_reads_url_then_credentials() {
    std::env::set_var("OPENJIUWEN_REDIS_URL", "redis://env-host:6390/1");
    std::env::set_var("OPENJIUWEN_REDIS_USERNAME", "env-user");
    std::env::set_var("OPENJIUWEN_REDIS_PASSWORD", "env-pass");

    let config = RedisConfig::from_env().expect("from env");
    assert_eq!(config.url.as_deref(), Some("redis://env-host:6390/1"));
    assert_eq!(config.username.as_deref(), Some("env-user"));
    assert_eq!(
        config.password.as_ref().map(SecretString::expose_secret),
        Some("env-pass")
    );

    std::env::remove_var("OPENJIUWEN_REDIS_URL");
    std::env::remove_var("OPENJIUWEN_REDIS_USERNAME");
    std::env::remove_var("OPENJIUWEN_REDIS_PASSWORD");
}

#[tokio::test]
async fn text_bytes_and_empty_values_round_trip() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = store_for(&server).await;

    for value in [
        KvValue::text("hello"),
        KvValue::text(""),
        KvValue::bytes(vec![0_u8, 1, 255]),
        KvValue::bytes(Vec::new()),
    ] {
        store.set("k", value.clone()).await.expect("set");
        assert_eq!(store.get("k").await.expect("get"), Some(value));
    }

    // A byte value is tagged with the marker plus base64; text is stored raw.
    let commands = server.commands();
    assert!(saw(&commands, &["SET", "k", "\0ojb:AAH/"]));
    assert!(saw(&commands, &["SET", "k", "hello"]));
}

#[tokio::test]
async fn set_get_exists_and_delete() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = store_for(&server).await;

    store.set("k", KvValue::text("v")).await.expect("set");
    assert_eq!(store.get("k").await.expect("get"), Some(KvValue::text("v")));
    assert!(store.exists("k").await.expect("exists"));
    store.delete("k").await.expect("delete");
    assert_eq!(store.get("k").await.expect("get"), None);
    assert!(!store.exists("k").await.expect("exists"));
    // Deleting a missing key is a no-op, not an error.
    store.delete("k").await.expect("second delete");

    let commands = server.commands();
    assert!(saw(&commands, &["GET", "k"]));
    assert!(saw(&commands, &["EXISTS", "k"]));
    assert_eq!(count_named(&commands, "DEL"), 2);
}

#[tokio::test]
async fn key_prefix_is_applied_and_stripped() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let first = RedisConfig::new(server.addr.clone())
        .with_key_prefix("app:")
        .connect()
        .await
        .expect("first");
    let second = RedisConfig::new(server.addr.clone())
        .with_key_prefix("other:")
        .connect()
        .await
        .expect("second");

    first.set("a", KvValue::text("1")).await.expect("set");
    second.set("a", KvValue::text("2")).await.expect("set");
    assert_eq!(first.get("a").await.expect("get"), Some(KvValue::text("1")));
    assert_eq!(
        second.get("a").await.expect("get"),
        Some(KvValue::text("2"))
    );

    let entries = first.get_by_prefix("").await.expect("scan");
    assert_eq!(
        entries.keys().cloned().collect::<Vec<_>>(),
        vec!["a".to_string()]
    );
    assert_eq!(entries.get("a"), Some(&KvValue::text("1")));

    let commands = server.commands();
    assert!(saw(&commands, &["SET", "app:a", "1"]));
    assert!(saw(&commands, &["SET", "other:a", "2"]));
    assert!(saw(
        &commands,
        &["SCAN", "0", "MATCH", "app:*", "COUNT", "100"]
    ));
}

#[tokio::test]
async fn exclusive_set_uses_nx_and_expiry() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = store_for(&server).await;

    assert!(store
        .exclusive_set("lock", KvValue::text("first"), None)
        .await
        .expect("first write"));
    assert!(!store
        .exclusive_set("lock", KvValue::text("second"), None)
        .await
        .expect("second write"));
    assert_eq!(
        store.get("lock").await.expect("get"),
        Some(KvValue::text("first"))
    );

    assert!(store
        .exclusive_set("ttl", KvValue::text("v"), Some(Duration::from_secs(60)))
        .await
        .expect("second ttl"));
    assert!(store
        .exclusive_set(
            "short",
            KvValue::text("v"),
            Some(Duration::from_millis(500))
        )
        .await
        .expect("sub-second ttl"));
    // A zero TTL is already expired: nothing is written and `false` comes back.
    assert!(!store
        .exclusive_set("zero", KvValue::text("v"), Some(Duration::ZERO))
        .await
        .expect("zero ttl"));
    assert!(!store.exists("zero").await.expect("exists"));

    let commands = server.commands();
    assert!(saw(&commands, &["SET", "lock", "first", "NX"]));
    assert!(saw(&commands, &["SET", "ttl", "v", "NX", "EX", "60"]));
    assert!(saw(&commands, &["SET", "short", "v", "NX", "PX", "500"]));
    assert!(!saw(&commands, &["SET", "zero", "v", "NX"]));
}

#[tokio::test]
async fn prefix_scans_and_deletes() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = store_for(&server).await;

    for key in ["p/1", "p/2", "q/1"] {
        store.set(key, KvValue::text(key)).await.expect("set");
    }

    let entries = store.get_by_prefix("p/").await.expect("scan");
    assert_eq!(
        entries.keys().cloned().collect::<Vec<_>>(),
        vec!["p/1".to_string(), "p/2".to_string()]
    );
    assert_eq!(store.delete_by_prefix("p/").await.expect("delete"), 2);
    assert_eq!(store.delete_by_prefix("p/").await.expect("again"), 0);
    assert_eq!(
        store.get_by_prefix("q/").await.expect("scan").len(),
        1,
        "the q/ entry survives"
    );
    assert_eq!(store.get_by_prefix("").await.expect("all").len(), 1);
}

#[tokio::test]
async fn mget_preserves_order_and_misses() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = store_for(&server).await;

    store.set("a", KvValue::text("1")).await.expect("set a");
    store.set("c", KvValue::text("3")).await.expect("set c");

    let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    assert_eq!(
        store.mget(&keys).await.expect("mget"),
        vec![Some(KvValue::text("1")), None, Some(KvValue::text("3"))]
    );
    assert_eq!(store.mget(&[]).await.expect("empty").len(), 0);
    assert!(saw(&server.commands(), &["MGET", "a", "b", "c"]));
}

#[tokio::test]
async fn batch_delete_counts_live_keys() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = store_for(&server).await;

    store.set("a", KvValue::text("1")).await.expect("set a");
    store.set("b", KvValue::text("2")).await.expect("set b");

    let keys = vec![
        "a".to_string(),
        "b".to_string(),
        "a".to_string(),
        "missing".to_string(),
    ];
    assert_eq!(store.batch_delete(&keys).await.expect("delete"), 2);
    assert_eq!(store.batch_delete(&[]).await.expect("empty"), 0);
}

#[tokio::test]
async fn prefix_reads_and_deletes_are_chunked() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = store_for(&server).await;
    for index in 0..150 {
        state.insert(&format!("bulk/{index:03}"), b"v");
    }

    let entries = store.get_by_prefix("bulk/").await.expect("scan");
    assert_eq!(entries.len(), 150);
    assert_eq!(
        count_named(&server.commands(), "MGET"),
        2,
        "150 keys split into two MGET batches"
    );

    assert_eq!(store.delete_by_prefix("bulk/").await.expect("delete"), 150);
    assert_eq!(count_named(&server.commands(), "DEL"), 2);
}

#[tokio::test]
async fn pipeline_applies_operations_in_order() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let store = store_for(&server).await;
    store
        .set("existing", KvValue::text("old"))
        .await
        .expect("seed");

    let mut pipeline = store.pipeline();
    pipeline.set("queued", KvValue::text("new"));
    pipeline.get("queued");
    pipeline.exists("queued");
    pipeline.exists("missing");
    pipeline.get("existing");

    let results = pipeline.execute().await.expect("execute");
    assert_eq!(
        results,
        vec![
            None,
            Some(KvValue::text("new")),
            Some(KvValue::text("true")),
            Some(KvValue::text("false")),
            Some(KvValue::text("old")),
        ]
    );
}

#[tokio::test]
async fn server_error_frames_surface_as_external_errors() {
    let handler: Handler = Arc::new(|args: &[Vec<u8>]| {
        let name = args
            .first()
            .map(|name| String::from_utf8_lossy(name).to_ascii_uppercase())
            .unwrap_or_default();
        if name == "GET" {
            Reply::bytes(server_error("ERR boom"))
        } else {
            Reply::bytes(simple("OK"))
        }
    });
    let server = FakeRedis::start(handler).await;
    let store = store_for(&server).await;

    let error = store.get("k").await.expect_err("server error");
    assert_eq!(error.kind(), ErrorKind::External);
    let message = error.to_string();
    assert!(message.contains("ERR boom"), "{message}");
}

#[tokio::test]
async fn connect_to_a_closed_port_fails() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    drop(listener);

    let error = RedisConnection::connect(&addr, None, None)
        .await
        .expect_err("no listener");
    assert!(
        matches!(error.kind(), ErrorKind::External | ErrorKind::Timeout),
        "{error}"
    );
}

#[tokio::test]
async fn reconnect_retries_a_read_only_command() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let handler = {
        let attempts = Arc::clone(&attempts);
        Arc::new(move |args: &[Vec<u8>]| -> Reply {
            let is_get = args
                .first()
                .is_some_and(|name| name.eq_ignore_ascii_case(b"GET"));
            if is_get {
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Reply::close();
                }
                return Reply::bytes(bulk(b"value"));
            }
            Reply::bytes(simple("OK"))
        }) as Handler
    };
    let server = FakeRedis::start(handler).await;
    let store = store_for(&server).await;

    assert_eq!(
        store.get("k").await.expect("retried get"),
        Some(KvValue::text("value"))
    );
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        server.commands(),
        vec![
            vec!["GET".to_string(), "k".to_string()],
            vec!["GET".to_string(), "k".to_string()],
        ]
    );
}

#[tokio::test]
async fn mutating_command_is_not_retried_after_write() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let handler = {
        let attempts = Arc::clone(&attempts);
        Arc::new(move |args: &[Vec<u8>]| -> Reply {
            let is_set = args
                .first()
                .is_some_and(|name| name.eq_ignore_ascii_case(b"SET"));
            if is_set {
                attempts.fetch_add(1, Ordering::SeqCst);
                return Reply::close();
            }
            Reply::bytes(simple("OK"))
        }) as Handler
    };
    let server = FakeRedis::start(handler).await;
    let store = store_for(&server).await;

    let error = store
        .set("k", KvValue::text("v"))
        .await
        .expect_err("lost reply");
    assert_eq!(
        attempts.load(Ordering::SeqCst),
        1,
        "a mutating command must not be re-sent"
    );
    assert!(
        error.to_string().contains("closed the connection"),
        "{error}"
    );
}

#[tokio::test]
async fn scan_cursor_loop_deduplicates_pages() {
    let scans = Arc::new(AtomicUsize::new(0));
    let handler = {
        let scans = Arc::clone(&scans);
        Arc::new(move |args: &[Vec<u8>]| -> Reply {
            let name = args
                .first()
                .map(|name| String::from_utf8_lossy(name).to_ascii_uppercase())
                .unwrap_or_default();
            match name.as_str() {
                "SCAN" => {
                    if scans.fetch_add(1, Ordering::SeqCst) == 0 {
                        Reply::bytes(array(vec![bulk(b"7"), array(vec![bulk(b"a"), bulk(b"b")])]))
                    } else {
                        Reply::bytes(array(vec![bulk(b"0"), array(vec![bulk(b"b"), bulk(b"c")])]))
                    }
                }
                "MGET" => Reply::bytes(array(args[1..].iter().map(|key| bulk(key)).collect())),
                _ => Reply::bytes(simple("OK")),
            }
        }) as Handler
    };
    let server = FakeRedis::start(handler).await;
    let store = store_for(&server).await;

    let entries = store.get_by_prefix("").await.expect("scan");
    assert_eq!(
        entries.keys().cloned().collect::<Vec<_>>(),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
    assert_eq!(scans.load(Ordering::SeqCst), 2);

    let commands = server.commands();
    assert!(saw(&commands, &["SCAN", "0", "MATCH", "*", "COUNT", "100"]));
    assert!(saw(&commands, &["SCAN", "7", "MATCH", "*", "COUNT", "100"]));
    assert!(saw(&commands, &["MGET", "a", "b", "c"]));
}

#[tokio::test]
async fn ping_round_trips() {
    let state = MiniRedis::shared();
    let server = FakeRedis::start(mini_redis(&state)).await;
    let mut connection = RedisConnection::connect(&server.addr, None, None)
        .await
        .expect("connect");
    connection.ping().await.expect("ping");
    assert!(saw(&server.commands(), &["PING"]));
}

/// A loopback server that records the raw bytes a client writes, protocol-blind.
///
/// The `rediss://` transport is proven with this fixture: the first bytes a client
/// sends must be a TLS handshake record (`0x16 0x03`), never a cleartext RESP
/// command. A real handshake cannot be completed offline because that needs a
/// certificate chain the workspace trusts (or a PEM/certificate-generation crate),
/// so the probe always closes the connection instead.
struct RawProbe {
    /// `host:port` the server listens on.
    addr: String,
    /// Every byte read from the single accepted connection.
    received: Arc<Mutex<Vec<u8>>>,
    /// The accept loop; aborted when the fixture drops.
    task: tokio::task::JoinHandle<()>,
}

impl Drop for RawProbe {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl RawProbe {
    /// Starts a server that records the first chunk, optionally answers it, and
    /// then closes the connection.
    ///
    /// With `reply: None` the connection is dropped right after the first chunk,
    /// which aborts a TLS handshake mid-flight.
    async fn start(reply: Option<Vec<u8>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr").to_string();
        let received = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&received);
        let task = tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut buffer = [0_u8; 2048];
            match stream.read(&mut buffer).await {
                Ok(0) | Err(_) => return,
                Ok(read) => recorder
                    .lock()
                    .expect("received")
                    .extend_from_slice(&buffer[..read]),
            }
            let Some(reply) = reply else {
                // Dropping the stream closes the connection mid-handshake.
                return;
            };
            if stream.write_all(&reply).await.is_err() {
                return;
            }
            // Keep the socket open until the fixture drops, discarding any input.
            let mut sink = [0_u8; 1024];
            while let Ok(read) = stream.read(&mut sink).await {
                if read == 0 {
                    break;
                }
            }
        });
        Self {
            addr,
            received,
            task,
        }
    }

    /// Starts a server that accepts one connection and then never answers it.
    async fn silent() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr").to_string();
        let received = Arc::new(Mutex::new(Vec::new()));
        let task = tokio::spawn(async move {
            let Ok((_stream, _)) = listener.accept().await else {
                return;
            };
            // Hold the socket open without reading or writing: the client must
            // give up on its own timeout.
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        Self {
            addr,
            received,
            task,
        }
    }

    /// Every byte the server received so far.
    fn received(&self) -> Vec<u8> {
        self.received.lock().expect("received").clone()
    }
}

/// Whether `haystack` contains `needle` as a contiguous byte sequence.
fn contains_sequence(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack.len() >= needle.len()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}
