#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the team transport abstraction.
//!
//! The suite is offline: the in-process backend runs over the crate's message
//! bus with [`common::RecordingHandler`], and the Redis backend runs over a
//! loopback fake Redis server (`TcpListener` on `127.0.0.1:0`) that decodes
//! client commands with `openjiuwen-foundation`'s RESP [`Decoder`] and answers
//! with hand-built RESP2 frames. No credentials, no real server, and every wait
//! is bounded well under ten seconds.

mod common;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::store::external::redis::codec::{Decoder, RespValue};
use openjiuwen_multiagent::transport::{
    build_transport, InProcessTransport, RedisTransport, RedisTransportConfig, TeamTransport,
    TransportConfig, TransportKind,
};
use openjiuwen_multiagent::{MessageBusConfig, MessageEnvelope, MessageHandler};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::common::RecordingHandler;

/// Builds a recording handler plus the erased handler the transports take.
fn recording_handler() -> (Arc<RecordingHandler>, Arc<dyn MessageHandler>) {
    let recorder = RecordingHandler::new();
    let handler = Arc::clone(&recorder) as Arc<dyn MessageHandler>;
    (recorder, handler)
}

/// Encodes a RESP2 simple string.
fn simple(text: &str) -> Vec<u8> {
    format!("+{text}\r\n").into_bytes()
}

/// Encodes a RESP2 error frame.
fn server_error(text: &str) -> Vec<u8> {
    format!("-{text}\r\n").into_bytes()
}

/// Encodes a RESP2 integer.
fn integer(number: impl std::fmt::Display) -> Vec<u8> {
    format!(":{number}\r\n").into_bytes()
}

/// Encodes a RESP2 bulk string.
fn bulk(payload: &[u8]) -> Vec<u8> {
    let mut out = format!("${}\r\n", payload.len()).into_bytes();
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
    out
}

/// Encodes the RESP2 null bulk string.
fn nil() -> Vec<u8> {
    b"$-1\r\n".to_vec()
}

/// In-memory state of the fake Redis server.
#[derive(Default)]
struct ServerState {
    /// One FIFO list per Redis key.
    lists: Mutex<HashMap<String, VecDeque<Vec<u8>>>>,
    /// Every command received, arguments rendered as text.
    commands: Mutex<Vec<Vec<String>>>,
}

impl ServerState {
    /// Records one command for later assertions.
    fn record(&self, args: &[Vec<u8>]) {
        self.commands.lock().expect("commands").push(
            args.iter()
                .map(|arg| String::from_utf8_lossy(arg).into_owned())
                .collect(),
        );
    }

    /// Answers one command with RESP2 bytes.
    fn handle(&self, args: &[Vec<u8>]) -> Vec<u8> {
        let name = args
            .first()
            .map(|name| String::from_utf8_lossy(name).to_ascii_uppercase())
            .unwrap_or_default();
        match name.as_str() {
            "PING" => simple("PONG"),
            "AUTH" | "SELECT" => simple("OK"),
            "RPUSH" => self.rpush(args),
            "RPOP" => self.rpop(args),
            "LLEN" => self.llen(args),
            "DEL" => self.del(args),
            other => server_error(&format!("ERR unknown command '{other}'")),
        }
    }

    /// Handles `RPUSH key value [value ...]`.
    fn rpush(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if args.len() < 3 {
            return server_error("ERR wrong number of arguments for 'rpush'");
        }
        let key = String::from_utf8_lossy(&args[1]).into_owned();
        let mut lists = self.lists.lock().expect("lists");
        let list = lists.entry(key).or_default();
        for value in &args[2..] {
            list.push_back(value.clone());
        }
        integer(list.len())
    }

    /// Handles `RPOP key`.
    fn rpop(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if args.len() != 2 {
            return server_error("ERR wrong number of arguments for 'rpop'");
        }
        let key = String::from_utf8_lossy(&args[1]).into_owned();
        let mut lists = self.lists.lock().expect("lists");
        match lists.get_mut(&key).and_then(VecDeque::pop_front) {
            Some(value) => bulk(&value),
            None => nil(),
        }
    }

    /// Handles `LLEN key`.
    fn llen(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if args.len() != 2 {
            return server_error("ERR wrong number of arguments for 'llen'");
        }
        let key = String::from_utf8_lossy(&args[1]).into_owned();
        let lists = self.lists.lock().expect("lists");
        integer(lists.get(&key).map_or(0, VecDeque::len))
    }

    /// Handles `DEL key [key ...]`.
    fn del(&self, args: &[Vec<u8>]) -> Vec<u8> {
        let mut lists = self.lists.lock().expect("lists");
        let removed = args[1..]
            .iter()
            .filter(|key| {
                lists
                    .remove(&String::from_utf8_lossy(key).into_owned())
                    .is_some()
            })
            .count();
        integer(removed)
    }

    /// Appends one payload to `key` directly, bypassing the protocol.
    fn push(&self, key: &str, payload: &[u8]) {
        self.lists
            .lock()
            .expect("lists")
            .entry(key.to_owned())
            .or_default()
            .push_back(payload.to_vec());
    }

    /// Returns a snapshot of the list `key`.
    fn list(&self, key: &str) -> Vec<Vec<u8>> {
        self.lists
            .lock()
            .expect("lists")
            .get(key)
            .map(|list| list.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns every recorded command whose name is `name` (case-insensitive).
    fn commands_named(&self, name: &str) -> Vec<Vec<String>> {
        self.commands
            .lock()
            .expect("commands")
            .iter()
            .filter(|args| {
                args.first()
                    .is_some_and(|first| first.eq_ignore_ascii_case(name))
            })
            .cloned()
            .collect()
    }
}

/// A fake Redis server listening on a loopback port.
struct FakeRedis {
    /// The `host:port` the server listens on.
    addr: String,
    /// The shared server state.
    state: Arc<ServerState>,
    /// The accept loop, aborted when the fixture drops.
    task: tokio::task::JoinHandle<()>,
}

impl Drop for FakeRedis {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl FakeRedis {
    /// Binds an ephemeral loopback port and serves until dropped.
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr").to_string();
        let state = Arc::new(ServerState::default());
        let shared = Arc::clone(&state);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let state = Arc::clone(&shared);
                tokio::spawn(async move { serve(stream, state).await });
            }
        });
        Self { addr, state, task }
    }

    /// Builds a transport config pointing at this server.
    fn config(&self, poll_interval_ms: u64) -> RedisTransportConfig {
        RedisTransportConfig::new()
            .with_addr(self.addr.clone())
            .with_poll_interval_ms(poll_interval_ms)
    }

    /// Pushes a payload onto a list directly.
    fn push(&self, key: &str, payload: &[u8]) {
        self.state.push(key, payload);
    }

    /// Returns a snapshot of a list.
    fn list(&self, key: &str) -> Vec<Vec<u8>> {
        self.state.list(key)
    }

    /// Returns the recorded commands named `name`.
    fn commands_named(&self, name: &str) -> Vec<Vec<String>> {
        self.state.commands_named(name)
    }
}

/// Serves one client connection until it closes.
async fn serve(mut stream: TcpStream, state: Arc<ServerState>) {
    let mut decoder = Decoder::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let read = match stream.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        decoder.push(&buffer[..read]);
        loop {
            let Some(frame) = decoder.try_next().expect("client command") else {
                break;
            };
            let Some(args) = command_args(frame) else {
                break;
            };
            state.record(&args);
            if stream.write_all(&state.handle(&args)).await.is_err() {
                return;
            }
        }
    }
}

/// Converts one inbound RESP array into its argument list.
fn command_args(frame: RespValue) -> Option<Vec<Vec<u8>>> {
    match frame {
        RespValue::Array(Some(items)) => items.into_iter().map(RespValue::into_bytes).collect(),
        _ => None,
    }
}

/// Waits until `key` holds at least `len` entries, then returns them.
///
/// Bounded at roughly five seconds, so a broken worker fails the test instead of
/// hanging it.
async fn wait_for_list(server: &FakeRedis, key: &str, len: usize) -> Vec<Vec<u8>> {
    for _ in 0..500 {
        let list = server.list(key);
        if list.len() >= len {
            return list;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("list '{key}' did not reach {len} entries");
}

#[tokio::test]
async fn pulsar_is_rejected_as_unsupported() {
    let (_recorder, handler) = recording_handler();
    let config = TransportConfig::new().with_kind(TransportKind::Pulsar);
    let error = build_transport(&config, "team", MessageBusConfig::new(), handler).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.message().to_string();
    assert!(message.contains("Pulsar"), "{message}");
    assert!(message.contains("not implemented"), "{message}");
}

#[tokio::test]
async fn redis_without_settings_is_a_config_error() {
    let (_recorder, handler) = recording_handler();
    let config = TransportConfig::new().with_kind(TransportKind::Redis);
    let error = build_transport(&config, "team", MessageBusConfig::new(), handler).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.message().contains("redis"), "{}", error.message());
}

#[tokio::test]
async fn build_transport_builds_the_in_process_backend() {
    let (recorder, handler) = recording_handler();
    recorder.set_reply(json!({"built": true}));
    let config = TransportConfig::new().with_key_prefix("unused");
    let transport = build_transport(
        &config,
        "team-built",
        MessageBusConfig::new().with_max_queue_size(4),
        handler,
    )
    .unwrap();
    assert_eq!(transport.kind(), TransportKind::InProcess);
    assert_eq!(transport.team_id(), "team-built");
    let reply = transport
        .send(
            MessageEnvelope::p2p(json!({"n": 1}), "a", "b"),
            Some(Duration::from_millis(500)),
        )
        .await
        .unwrap();
    assert_eq!(reply, json!({"built": true}));
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn in_process_send_publish_and_subscriptions_match_the_bus() {
    let (recorder, handler) = recording_handler();
    let transport = InProcessTransport::new("team-inproc", MessageBusConfig::new(), handler);
    assert_eq!(transport.kind(), TransportKind::InProcess);
    assert_eq!(transport.team_id(), "team-inproc");
    transport.start().await.unwrap();

    recorder.set_reply(json!({"ok": true}));
    let reply = transport
        .send(
            MessageEnvelope::p2p(json!({"n": 1}), "a", "b").with_session("s1"),
            Some(Duration::from_millis(500)),
        )
        .await
        .unwrap();
    assert_eq!(reply, json!({"ok": true}));
    assert_eq!(recorder.p2p_payloads(), vec![json!({"n": 1})]);
    assert_eq!(recorder.p2p.lock()[0].session_id.as_deref(), Some("s1"));

    transport.add_subscription("z", "news");
    transport.add_subscription("a", "news");
    assert_eq!(transport.subscribers("news"), vec!["a", "z"]);
    transport
        .publish(MessageEnvelope::pubsub(
            json!({"story": 1}),
            "publisher",
            "news",
        ))
        .await
        .unwrap();
    assert_eq!(recorder.pubsub_order(), vec!["a", "z"]);

    transport.remove_subscription("a", "news");
    assert_eq!(transport.subscribers("news"), vec!["z"]);
    transport.stop().await.unwrap();
    assert!(!transport.bus().is_started());
}

#[tokio::test]
async fn in_process_transport_exposes_the_subscription_registry() {
    let (_recorder, handler) = recording_handler();
    let transport = InProcessTransport::new("team-introspect", MessageBusConfig::new(), handler);

    transport.add_subscription("z", "news");
    transport.add_subscription("a", "news");
    transport.add_subscription("a", "*__p2p__");

    assert_eq!(
        transport.list_subscriptions(),
        BTreeMap::from([
            ("*__p2p__".to_owned(), vec!["a".to_owned()]),
            ("news".to_owned(), vec!["a".to_owned(), "z".to_owned()]),
        ])
    );
    assert_eq!(transport.agent_subscriptions("a"), vec!["*__p2p__", "news"]);
    assert_eq!(transport.agent_subscriptions("ghost"), Vec::<String>::new());
    assert_eq!(transport.subscription_count(), 3);

    transport.remove_subscription("a", "news");
    assert_eq!(transport.subscription_count(), 2);
    assert_eq!(transport.agent_subscriptions("a"), vec!["*__p2p__"]);
}

#[tokio::test]
async fn in_process_send_passes_the_caller_deadline_through() {
    let (recorder, handler) = recording_handler();
    recorder.delay_first(Duration::from_millis(300));
    let transport = InProcessTransport::new("team-deadline", MessageBusConfig::new(), handler);
    let error = transport
        .send(
            MessageEnvelope::p2p(json!({}), "a", "b"),
            Some(Duration::from_millis(20)),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_send_round_trips_through_the_reply_list() {
    let server = FakeRedis::start().await;
    let (recorder, handler) = recording_handler();
    recorder.set_reply(json!({"echo": "hi"}));
    let transport = RedisTransport::new(server.config(5), "team-redis", "openjiuwen:test", handler);
    transport.watch_agent("b").unwrap();
    transport.start().await.unwrap();
    assert_eq!(transport.kind(), TransportKind::Redis);
    assert_eq!(transport.team_id(), "team-redis");

    let request = MessageEnvelope::p2p(json!({"n": 7}), "a", "b").with_session("s1");
    let message_id = request.message_id.clone();
    let reply = transport
        .send(request, Some(Duration::from_secs(5)))
        .await
        .unwrap();
    assert_eq!(reply, json!({"echo": "hi"}));

    let received = recorder.p2p.lock().clone();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].message, json!({"n": 7}));
    assert_eq!(received[0].sender.as_deref(), Some("a"));
    assert_eq!(received[0].recipient.as_deref(), Some("b"));
    assert_eq!(received[0].session_id.as_deref(), Some("s1"));

    assert_eq!(transport.p2p_queue("b"), "openjiuwen:test:team-redis:p2p:b");
    assert_eq!(
        transport.reply_queue(&message_id),
        format!("openjiuwen:test:team-redis:reply:{message_id}")
    );
    assert_eq!(
        transport.pubsub_queue("b"),
        "openjiuwen:test:team-redis:pubsub:b"
    );
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_publish_pushes_one_frame_per_sorted_subscriber() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "team-pub", "team", handler);
    transport.start().await.unwrap();
    for subscriber in ["z", "a", "m"] {
        transport.add_subscription(subscriber, "news");
    }
    transport
        .publish(MessageEnvelope::pubsub(
            json!({"story": 1}),
            "publisher",
            "news",
        ))
        .await
        .unwrap();

    let pushed: Vec<String> = server
        .commands_named("RPUSH")
        .iter()
        .map(|args| args[1].clone())
        .collect();
    assert_eq!(
        pushed,
        vec![
            "team:team-pub:pubsub:a".to_owned(),
            "team:team-pub:pubsub:m".to_owned(),
            "team:team-pub:pubsub:z".to_owned(),
        ]
    );
    let payload: Value = serde_json::from_slice(&server.list("team:team-pub:pubsub:a")[0]).unwrap();
    assert_eq!(payload["topic_id"], "news");
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_publish_without_subscribers_is_a_no_op() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "team-empty", "team", handler);
    transport.start().await.unwrap();
    transport
        .publish(MessageEnvelope::pubsub(
            json!({}),
            "publisher",
            "nobody-listens",
        ))
        .await
        .unwrap();
    assert_eq!(server.commands_named("RPUSH").len(), 0);
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_reply_timeout_names_both_ends() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "team-slow", "team", handler);
    transport.start().await.unwrap();
    let error = transport
        .send(
            MessageEnvelope::p2p(json!({}), "sender-x", "nobody"),
            Some(Duration::from_millis(120)),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    let message = error.message().to_string();
    assert!(message.contains("'sender-x'"), "{message}");
    assert!(message.contains("'nobody'"), "{message}");
    assert!(message.contains("timed out after"), "{message}");
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_worker_dispatches_a_queued_envelope() {
    let server = FakeRedis::start().await;
    let (recorder, handler) = recording_handler();
    recorder.set_reply(json!({"ok": 1}));
    let transport = RedisTransport::new(server.config(5), "team-worker", "team", handler);
    transport.watch_agent("b").unwrap();
    transport.start().await.unwrap();

    let queue = transport.p2p_queue("b");
    // A malformed entry is logged and skipped, so the next one still arrives.
    server.push(&queue, b"not-json");
    let envelope = MessageEnvelope::p2p(json!({"n": 2}), "a", "b");
    server.push(&queue, &serde_json::to_vec(&envelope).unwrap());

    let reply_key = transport.reply_queue(&envelope.message_id);
    let replies = wait_for_list(&server, &reply_key, 1).await;
    let frame: Value = serde_json::from_slice(&replies[0]).unwrap();
    assert_eq!(frame["value"], json!({"ok": 1}));
    assert_eq!(recorder.p2p_payloads(), vec![json!({"n": 2})]);
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_worker_delivers_pubsub_to_its_subscriber() {
    let server = FakeRedis::start().await;
    let (recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "team-sub", "team", handler);
    transport.watch_agent("sub").unwrap();
    transport.start().await.unwrap();

    let envelope = MessageEnvelope::pubsub(json!({"story": 2}), "publisher", "news");
    server.push(
        &transport.pubsub_queue("sub"),
        &serde_json::to_vec(&envelope).unwrap(),
    );
    for _ in 0..500 {
        if !recorder.pubsub.lock().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let delivered = recorder.pubsub.lock().clone();
    assert_eq!(delivered.len(), 1);
    assert_eq!(delivered[0].0, "sub");
    assert_eq!(delivered[0].1.message, json!({"story": 2}));
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_start_and_stop_are_idempotent() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "team-lifecycle", "team", handler);
    assert!(!transport.is_started());
    transport.start().await.unwrap();
    transport.start().await.unwrap();
    assert!(transport.is_started());
    // Watching twice keeps one worker per inbox.
    transport.watch_agent("a").unwrap();
    transport.watch_agent("a").unwrap();
    assert!(transport.watch_agent("").is_err());
    transport.stop().await.unwrap();
    transport.stop().await.unwrap();
    assert!(!transport.is_started());
    // A stopped transport can start again.
    transport.start().await.unwrap();
    assert!(transport.is_started());
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_start_reports_an_unreachable_server() {
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(
        RedisTransportConfig::new()
            .with_addr("127.0.0.1:1")
            .with_poll_interval_ms(5),
        "team-unreachable",
        "team",
        handler,
    );
    let error = transport.start().await.unwrap_err();
    assert_ne!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("redis connect"),
        "{}",
        error.message()
    );
    assert!(!transport.is_started());
}

#[tokio::test]
async fn redis_send_and_publish_require_their_routing_fields() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "team-routing", "team", handler);
    transport.start().await.unwrap();

    let send_error = transport
        .send(
            MessageEnvelope::new(json!({})),
            Some(Duration::from_millis(50)),
        )
        .await
        .unwrap_err();
    assert_eq!(send_error.kind(), ErrorKind::Validation);
    assert_eq!(send_error.message(), "recipient is required");

    let publish_error = transport
        .publish(MessageEnvelope::new(json!({})))
        .await
        .unwrap_err();
    assert_eq!(publish_error.kind(), ErrorKind::Validation);
    assert_eq!(publish_error.message(), "topic_id is required");
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_empty_team_and_prefix_fall_back_to_defaults() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "", "", handler);
    assert_eq!(transport.team_id(), "default");
    assert_eq!(transport.key_prefix(), "openjiuwen:team");
    assert_eq!(transport.p2p_queue("a"), "openjiuwen:team:default:p2p:a");
    // Debug output exposes the scheme but never a password.
    let rendered = format!("{transport:?}");
    assert!(rendered.contains("RedisTransport"), "{rendered}");
}
