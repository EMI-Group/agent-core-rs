//! Offline tests for the Redis backend.
//!
//! A scripted fake Redis server stands in for a real one: a [`TcpListener`]
//! decodes client commands with this module's own [`Decoder`] and answers with
//! hand-built RESP frames. Nothing here leaves the loopback interface, and no
//! test needs a running server.

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

#[tokio::test]
async fn rediss_urls_are_unsupported() {
    let error = RedisConfig::from_url("rediss://localhost:6379")
        .connect()
        .await
        .expect_err("TLS is not available");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[tokio::test]
async fn non_redis_url_schemes_are_config_errors() {
    let error = RedisConfig::from_url("http://localhost:6379")
        .connect()
        .await
        .expect_err("wrong scheme");
    assert_eq!(error.kind(), ErrorKind::Config);
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
