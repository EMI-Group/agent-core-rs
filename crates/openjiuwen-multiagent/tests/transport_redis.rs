#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! Integration tests for the Redis Streams team transport.
//!
//! The suite is offline: every transport points at a loopback fake Redis server
//! (`TcpListener` on `127.0.0.1:0`) that decodes client commands with
//! `openjiuwen-foundation`'s RESP [`Decoder`] and keeps a small but faithful
//! Streams state machine — per-key entries, per-(key, group) delivery cursors and
//! pending lists. It answers `PING`, `AUTH`, `SELECT`, `XADD`, `XGROUP CREATE`
//! (with `MKSTREAM` and `BUSYGROUP`), `XREADGROUP` (both `>` and explicit ids),
//! `XREAD`, `XACK`, `XAUTOCLAIM` and `DEL`.
//!
//! No credentials, no real server, and every wait is bounded well under ten
//! seconds.

mod common;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::store::external::redis::codec::{Decoder, RespValue};
use openjiuwen_multiagent::transport::{
    build_transport, RedisTopology, RedisTransport, RedisTransportConfig, TeamTransport,
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

/// Encodes the RESP2 null array, what `XREAD`/`XREADGROUP` return when empty.
fn null_array() -> Vec<u8> {
    b"*-1\r\n".to_vec()
}

/// Encodes a RESP2 array header for `count` elements.
fn array(count: usize) -> Vec<u8> {
    format!("*{count}\r\n").into_bytes()
}

/// Renders one argument as text.
fn text(arg: &[u8]) -> String {
    String::from_utf8_lossy(arg).into_owned()
}

/// One entry of a fake stream.
#[derive(Clone, Debug)]
struct Entry {
    /// The entry id (`<seq>-0`).
    id: String,
    /// Field/value pairs in insertion order.
    fields: Vec<(String, Vec<u8>)>,
}

impl Entry {
    /// Returns the value of `name`, when the entry carries it.
    fn field(&self, name: &str) -> Option<&[u8]> {
        self.fields
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value.as_slice())
    }
}

/// One stream of the fake server.
#[derive(Default)]
struct Stream {
    /// The entries in append order.
    entries: Vec<Entry>,
    /// Sequence of the last generated entry id.
    last_seq: u64,
}

/// One delivered-but-unacknowledged entry of a group.
struct Pending {
    /// Id of the pending entry.
    id: String,
    /// Consumer the entry was last delivered to.
    consumer: String,
    /// When the entry was last delivered.
    delivered_at: Instant,
}

/// One consumer group of one stream.
#[derive(Default)]
struct Group {
    /// Index of the first entry the `>` cursor has not delivered yet.
    last_delivered: usize,
    /// The group's pending entries.
    pending: Vec<Pending>,
}

/// In-memory state of the fake Redis server.
#[derive(Default)]
struct ServerState {
    /// One stream per Redis key.
    streams: Mutex<HashMap<String, Stream>>,
    /// One consumer group per `(key, group)` pair.
    groups: Mutex<HashMap<(String, String), Group>>,
    /// Every command received, arguments rendered as text.
    commands: Mutex<Vec<Vec<String>>>,
    /// Connections opened before the current generation are dropped.
    generation: AtomicU64,
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
            "XADD" => self.xadd(args),
            "XGROUP" => self.xgroup(args),
            "XREADGROUP" => self.xreadgroup(args),
            "XREAD" => self.xread(args),
            "XACK" => self.xack(args),
            "XAUTOCLAIM" => self.xautoclaim(args),
            "DEL" => self.del(args),
            other => server_error(&format!("ERR unknown command '{other}'")),
        }
    }

    /// Handles `XADD key * field value [field value ...]`.
    fn xadd(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if args.len() < 5 || (args.len() - 3) % 2 != 0 {
            return server_error("ERR wrong number of arguments for 'xadd'");
        }
        let key = text(&args[1]);
        let fields: Vec<(String, Vec<u8>)> = args[3..]
            .chunks(2)
            .map(|pair| (text(&pair[0]), pair[1].clone()))
            .collect();
        bulk(self.append(&key, fields).as_bytes())
    }

    /// Appends one entry to `key`, returning its generated id.
    fn append(&self, key: &str, fields: Vec<(String, Vec<u8>)>) -> String {
        let mut streams = self.streams.lock().expect("streams");
        let stream = streams.entry(key.to_owned()).or_default();
        stream.last_seq += 1;
        let id = format!("{:010}-0", stream.last_seq);
        stream.entries.push(Entry {
            id: id.clone(),
            fields,
        });
        id
    }

    /// Handles `XGROUP CREATE key group id [MKSTREAM]`.
    fn xgroup(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if args.len() < 5 || !text(&args[1]).eq_ignore_ascii_case("CREATE") {
            return server_error("ERR unsupported XGROUP subcommand");
        }
        let key = text(&args[2]);
        let group = text(&args[3]);
        let create_stream = args
            .iter()
            .any(|arg| text(arg).eq_ignore_ascii_case("MKSTREAM"));
        let end = {
            let mut streams = self.streams.lock().expect("streams");
            if !streams.contains_key(&key) {
                if !create_stream {
                    return server_error("ERR the XGROUP subcommand requires the key to exist");
                }
                streams.insert(key.clone(), Stream::default());
            }
            streams.get(&key).map_or(0, |stream| stream.entries.len())
        };
        let mut groups = self.groups.lock().expect("groups");
        if groups.contains_key(&(key.clone(), group.clone())) {
            return server_error("BUSYGROUP Consumer Group name already exists");
        }
        let last_delivered = if text(&args[4]) == "$" { end } else { 0 };
        groups.insert(
            (key, group),
            Group {
                last_delivered,
                pending: Vec::new(),
            },
        );
        simple("OK")
    }

    /// Handles `XREADGROUP GROUP g c [COUNT n] STREAMS key id`.
    fn xreadgroup(&self, args: &[Vec<u8>]) -> Vec<u8> {
        let Some(group_pos) = position(args, "GROUP") else {
            return server_error("ERR missing GROUP");
        };
        let Some(streams_pos) = position(args, "STREAMS") else {
            return server_error("ERR missing STREAMS");
        };
        if args.len() < group_pos + 3 || args.len() < streams_pos + 3 {
            return server_error("ERR wrong number of arguments for 'xreadgroup'");
        }
        let group = text(&args[group_pos + 1]);
        let consumer = text(&args[group_pos + 2]);
        let key = text(&args[streams_pos + 1]);
        let start = text(&args[streams_pos + 2]);
        let count = count_arg(args);
        let entries = {
            let streams = self.streams.lock().expect("streams");
            match streams.get(&key) {
                Some(stream) => stream.entries.clone(),
                None => return null_array(),
            }
        };
        let mut groups = self.groups.lock().expect("groups");
        let Some(group_state) = groups.get_mut(&(key.clone(), group)) else {
            return server_error("NOGROUP No such consumer group");
        };
        let delivered = if start == ">" {
            let from = group_state.last_delivered.min(entries.len());
            let slice: Vec<Entry> = entries.iter().skip(from).take(count).cloned().collect();
            group_state.last_delivered = from + slice.len();
            for entry in &slice {
                group_state.pending.push(Pending {
                    id: entry.id.clone(),
                    consumer: consumer.clone(),
                    delivered_at: Instant::now(),
                });
            }
            slice
        } else {
            let ids: Vec<String> = group_state
                .pending
                .iter()
                .filter(|pending| {
                    pending.consumer == consumer && pending.id.as_str() > start.as_str()
                })
                .take(count)
                .map(|pending| pending.id.clone())
                .collect();
            entries
                .iter()
                .filter(|entry| ids.contains(&entry.id))
                .cloned()
                .collect()
        };
        if delivered.is_empty() {
            return null_array();
        }
        stream_reply(&key, &delivered)
    }

    /// Handles `XREAD [COUNT n] STREAMS key id`.
    fn xread(&self, args: &[Vec<u8>]) -> Vec<u8> {
        let Some(streams_pos) = position(args, "STREAMS") else {
            return server_error("ERR missing STREAMS");
        };
        if args.len() < streams_pos + 3 {
            return server_error("ERR wrong number of arguments for 'xread'");
        }
        let key = text(&args[streams_pos + 1]);
        let start = text(&args[streams_pos + 2]);
        let count = count_arg(args);
        let streams = self.streams.lock().expect("streams");
        let Some(stream) = streams.get(&key) else {
            return null_array();
        };
        let entries: Vec<Entry> = stream
            .entries
            .iter()
            .filter(|entry| entry.id.as_str() > start.as_str())
            .take(count)
            .cloned()
            .collect();
        if entries.is_empty() {
            return null_array();
        }
        stream_reply(&key, &entries)
    }

    /// Handles `XACK key group id [id ...]`.
    fn xack(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if args.len() < 4 {
            return server_error("ERR wrong number of arguments for 'xack'");
        }
        let key = text(&args[1]);
        let group = text(&args[2]);
        let mut groups = self.groups.lock().expect("groups");
        let Some(group_state) = groups.get_mut(&(key, group)) else {
            return integer(0);
        };
        let mut removed = 0;
        for id in &args[3..] {
            let id = text(id);
            let before = group_state.pending.len();
            group_state.pending.retain(|pending| pending.id != id);
            if group_state.pending.len() < before {
                removed += 1;
            }
        }
        integer(removed)
    }

    /// Handles `XAUTOCLAIM key group consumer min-idle start [COUNT n]`.
    fn xautoclaim(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if args.len() < 6 {
            return server_error("ERR wrong number of arguments for 'xautoclaim'");
        }
        let key = text(&args[1]);
        let group = text(&args[2]);
        let consumer = text(&args[3]);
        let min_idle: u64 = text(&args[4]).parse().unwrap_or(0);
        let start = text(&args[5]);
        let count = count_arg(args);
        let entries = {
            let streams = self.streams.lock().expect("streams");
            streams
                .get(&key)
                .map_or_else(Vec::new, |stream| stream.entries.clone())
        };
        let mut groups = self.groups.lock().expect("groups");
        let Some(group_state) = groups.get_mut(&(key, group)) else {
            return server_error("NOGROUP No such consumer group");
        };
        let now = Instant::now();
        let mut claimed = Vec::new();
        let mut truncated = false;
        for pending in &mut group_state.pending {
            if pending.id.as_str() <= start.as_str() {
                continue;
            }
            if claimed.len() >= count {
                truncated = true;
                break;
            }
            let idle = u64::try_from(now.duration_since(pending.delivered_at).as_millis())
                .unwrap_or(u64::MAX);
            if idle < min_idle {
                continue;
            }
            pending.consumer.clone_from(&consumer);
            pending.delivered_at = now;
            if let Some(entry) = entries.iter().find(|entry| entry.id == pending.id) {
                claimed.push(entry.clone());
            }
        }
        let cursor = if truncated {
            claimed
                .last()
                .map_or_else(|| "0-0".to_owned(), |entry| entry.id.clone())
        } else {
            "0-0".to_owned()
        };
        let mut out = array(3);
        out.extend(bulk(cursor.as_bytes()));
        out.extend(stream_entries(&claimed));
        out.extend(array(0));
        out
    }

    /// Handles `DEL key [key ...]`.
    fn del(&self, args: &[Vec<u8>]) -> Vec<u8> {
        let mut streams = self.streams.lock().expect("streams");
        let mut groups = self.groups.lock().expect("groups");
        let mut removed = 0;
        for key in &args[1..] {
            let key = text(key);
            if streams.remove(&key).is_some() {
                removed += 1;
            }
            groups.retain(|(group_key, _), _| *group_key != key);
        }
        integer(removed)
    }

    /// Appends one entry to `key` directly, bypassing the protocol.
    fn push(&self, key: &str, fields: &[(&str, Vec<u8>)]) -> String {
        let fields: Vec<(String, Vec<u8>)> = fields
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.clone()))
            .collect();
        self.append(key, fields)
    }

    /// Appends one entry and marks it pending for `consumer` of `group`.
    ///
    /// The stream and the group are updated under one lock window, so the `>`
    /// cursor can never see the seeded entry as new: only
    /// [`XAUTOCLAIM`](ServerState::xautoclaim) can recover it.
    fn seed_pending(&self, key: &str, group: &str, consumer: &str, envelope: &[u8]) -> String {
        let mut streams = self.streams.lock().expect("streams");
        let stream = streams.entry(key.to_owned()).or_default();
        stream.last_seq += 1;
        let id = format!("{:010}-0", stream.last_seq);
        stream.entries.push(Entry {
            id: id.clone(),
            fields: vec![("envelope".to_owned(), envelope.to_vec())],
        });
        let end = stream.entries.len();
        let mut groups = self.groups.lock().expect("groups");
        let group_state = groups
            .entry((key.to_owned(), group.to_owned()))
            .or_default();
        group_state.last_delivered = end;
        group_state.pending.push(Pending {
            id: id.clone(),
            consumer: consumer.to_owned(),
            delivered_at: Instant::now()
                .checked_sub(Duration::from_secs(5))
                .unwrap_or_else(Instant::now),
        });
        id
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

    /// Returns a snapshot of the stream `key`.
    fn entries(&self, key: &str) -> Vec<Entry> {
        self.streams
            .lock()
            .expect("streams")
            .get(key)
            .map_or_else(Vec::new, |stream| stream.entries.clone())
    }
}

/// Finds the position of an identifier argument (case-insensitive).
fn position(args: &[Vec<u8>], name: &str) -> Option<usize> {
    args.iter()
        .position(|arg| text(arg).eq_ignore_ascii_case(name))
}

/// Reads the optional `COUNT n` clause of a read command.
fn count_arg(args: &[Vec<u8>]) -> usize {
    position(args, "COUNT")
        .and_then(|index| args.get(index + 1))
        .and_then(|value| text(value).parse().ok())
        .unwrap_or(100)
}

/// Encodes one `[id, [field, value, ...]]` entry frame.
fn stream_entry(entry: &Entry) -> Vec<u8> {
    let mut out = array(2);
    out.extend(bulk(entry.id.as_bytes()));
    out.extend(array(entry.fields.len() * 2));
    for (name, value) in &entry.fields {
        out.extend(bulk(name.as_bytes()));
        out.extend(bulk(value));
    }
    out
}

/// Encodes an array of entry frames.
fn stream_entries(entries: &[Entry]) -> Vec<u8> {
    let mut out = array(entries.len());
    for entry in entries {
        out.extend(stream_entry(entry));
    }
    out
}

/// Encodes the `[[key, entries]]` shape of `XREAD`/`XREADGROUP`.
fn stream_reply(key: &str, entries: &[Entry]) -> Vec<u8> {
    let mut out = array(1);
    out.extend(array(2));
    out.extend(bulk(key.as_bytes()));
    out.extend(stream_entries(entries));
    out
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

    /// Appends one entry to a stream directly.
    fn push(&self, key: &str, fields: &[(&str, Vec<u8>)]) -> String {
        self.state.push(key, fields)
    }

    /// Appends an entry and marks it pending for `consumer` of `group`.
    fn seed_pending(&self, key: &str, group: &str, consumer: &str, envelope: &[u8]) -> String {
        self.state.seed_pending(key, group, consumer, envelope)
    }

    /// Returns a snapshot of a stream.
    fn entries(&self, key: &str) -> Vec<Entry> {
        self.state.entries(key)
    }

    /// Returns the recorded commands named `name`.
    fn commands_named(&self, name: &str) -> Vec<Vec<String>> {
        self.state.commands_named(name)
    }

    /// Closes every connection opened so far on its next command.
    fn drop_connections(&self) {
        self.state.generation.fetch_add(1, Ordering::SeqCst);
    }
}

/// Serves one client connection until it closes.
///
/// A connection opened before the last [`FakeRedis::drop_connections`] closes
/// without answering its next command, which is exactly what a server restart
/// looks like to the client (a read failure followed by a reconnect).
async fn serve(mut stream: TcpStream, state: Arc<ServerState>) {
    let generation = state.generation.load(Ordering::SeqCst);
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
            if state.generation.load(Ordering::SeqCst) != generation {
                return;
            }
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

/// Polls `condition` for roughly five seconds, panicking when it never holds.
async fn wait_until(mut condition: impl FnMut() -> bool, what: &str) {
    for _ in 0..500 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {what}");
}

/// Waits until `recorder` handled `count` P2P deliveries.
async fn wait_for_p2p(recorder: &RecordingHandler, count: usize) {
    for _ in 0..500 {
        if recorder.p2p.lock().len() >= count {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "recorder saw {} P2P deliveries, expected {count}",
        recorder.p2p.lock().len()
    );
}

/// Waits until `recorder` handled `count` pub/sub deliveries.
async fn wait_for_pubsub(recorder: &RecordingHandler, count: usize) {
    for _ in 0..500 {
        if recorder.pubsub.lock().len() >= count {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "recorder saw {} pub/sub deliveries, expected {count}",
        recorder.pubsub.lock().len()
    );
}

#[tokio::test]
async fn redis_send_round_trips_across_transport_instances() {
    let server = FakeRedis::start().await;
    let (recorder_a, handler_a) = recording_handler();
    recorder_a.set_reply(json!({"echo": "hi"}));
    let receiver = RedisTransport::new(server.config(5), "team-cross", "team", handler_a);
    receiver.watch_agent("agent").unwrap();
    receiver.start().await.unwrap();

    let (_recorder_b, handler_b) = recording_handler();
    let sender = RedisTransport::new(server.config(5), "team-cross", "team", handler_b);
    sender.start().await.unwrap();

    assert_eq!(receiver.kind(), TransportKind::Redis);
    assert_eq!(receiver.team_id(), "team-cross");
    assert_eq!(receiver.p2p_queue("agent"), "team:team-cross:p2p:agent");
    assert_eq!(receiver.reply_queue("m1"), "team:team-cross:reply:m1");
    assert_eq!(
        receiver.pubsub_queue("agent"),
        "team:team-cross:pubsub:agent"
    );
    assert_eq!(receiver.p2p_consumer_group(), "team:team-cross:p2p:cg");
    assert_eq!(
        receiver.pubsub_consumer_group(),
        "team:team-cross:pubsub:cg"
    );
    assert_eq!(receiver.consumer_name().len(), 36);
    assert_ne!(receiver.consumer_name(), sender.consumer_name());

    let request = MessageEnvelope::p2p(json!({"n": 7}), "sender", "agent").with_session("s1");
    let reply = sender
        .send(request, Some(Duration::from_secs(5)))
        .await
        .unwrap();
    assert_eq!(reply, json!({"echo": "hi"}));

    let delivered = recorder_a.p2p.lock().clone();
    assert_eq!(delivered.len(), 1);
    assert_eq!(delivered[0].message, json!({"n": 7}));
    assert_eq!(delivered[0].sender.as_deref(), Some("sender"));
    assert_eq!(delivered[0].recipient.as_deref(), Some("agent"));
    assert_eq!(delivered[0].session_id.as_deref(), Some("s1"));

    // The request travelled as a stream entry carrying the reply key.
    let ping = server.commands_named("XADD");
    let p2p = ping
        .iter()
        .find(|args| args.len() >= 7 && args[1] == "team:team-cross:p2p:agent")
        .expect("p2p XADD");
    assert_eq!(p2p[2], "*");
    assert_eq!(p2p[3], "envelope");
    assert_eq!(p2p[5], "reply");
    assert!(p2p[6].starts_with("team:team-cross:reply:"));

    // Destructive list commands are gone for good.
    assert!(server.commands_named("RPOP").is_empty());
    assert!(server.commands_named("RPUSH").is_empty());
    assert!(server.commands_named("LLEN").is_empty());
    assert!(!server.commands_named("XREADGROUP").is_empty());
    assert!(!server.commands_named("XACK").is_empty());

    receiver.stop().await.unwrap();
    sender.stop().await.unwrap();
}

#[tokio::test]
async fn two_transports_in_one_group_deliver_an_entry_once() {
    let server = FakeRedis::start().await;
    let (first, handler_a) = recording_handler();
    let (second, handler_b) = recording_handler();
    let receiver_a = RedisTransport::new(server.config(5), "team-shared", "team", handler_a);
    receiver_a.watch_agent("agent").unwrap();
    receiver_a.start().await.unwrap();
    let receiver_b = RedisTransport::new(server.config(5), "team-shared", "team", handler_b);
    receiver_b.watch_agent("agent").unwrap();
    receiver_b.start().await.unwrap();
    assert_eq!(
        receiver_a.p2p_consumer_group(),
        receiver_b.p2p_consumer_group()
    );
    assert_ne!(receiver_a.consumer_name(), receiver_b.consumer_name());

    let (_sender_recorder, sender_handler) = recording_handler();
    let sender = RedisTransport::new(server.config(5), "team-shared", "team", sender_handler);
    sender.start().await.unwrap();
    sender
        .send(
            MessageEnvelope::p2p(json!({"n": 1}), "sender", "agent"),
            Some(Duration::from_secs(5)),
        )
        .await
        .unwrap();

    // Give the second worker time to (wrongly) deliver a duplicate.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let deliveries = first.p2p.lock().len() + second.p2p.lock().len();
    assert_eq!(
        deliveries, 1,
        "the consumer group must hand out each entry once"
    );
    assert_eq!(server.commands_named("XACK").len(), 1);

    receiver_a.stop().await.unwrap();
    receiver_b.stop().await.unwrap();
    sender.stop().await.unwrap();
}

#[tokio::test]
async fn worker_reclaims_a_pending_entry_after_a_reconnect() {
    let server = FakeRedis::start().await;
    let (recorder, handler) = recording_handler();
    recorder.set_reply(json!({"ok": 5}));
    let transport = RedisTransport::new(
        server.config(5).with_claim_idle_ms(0),
        "team-reclaim",
        "team",
        handler,
    );
    transport.watch_agent("agent").unwrap();
    transport.start().await.unwrap();

    // Both inbox workers poll, so both have a live connection to drop.
    wait_until(
        || server.commands_named("XREADGROUP").len() >= 2,
        "the first worker polls",
    )
    .await;

    let envelope = MessageEnvelope::p2p(json!({"n": 9}), "sender", "agent");
    let key = transport.p2p_queue("agent");
    let entry_id = server.seed_pending(
        &key,
        &transport.p2p_consumer_group(),
        "crashed-consumer",
        &serde_json::to_vec(&envelope).unwrap(),
    );
    assert!(recorder.p2p.lock().is_empty());

    // Drop the live sockets: the workers fail their next read and reconnect.
    server.drop_connections();
    wait_for_p2p(&recorder, 1).await;
    assert_eq!(recorder.p2p_payloads(), vec![json!({"n": 9})]);

    // The reclaimed entry was dispatched and then acknowledged.
    wait_until(
        || {
            server
                .commands_named("XACK")
                .iter()
                .any(|args| args.len() >= 4 && args[1] == key && args[3] == entry_id)
        },
        "the reclaimed entry is acked",
    )
    .await;
    assert!(!server.commands_named("XAUTOCLAIM").is_empty());

    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_worker_dispatches_a_queued_envelope_and_tolerates_poison() {
    let server = FakeRedis::start().await;
    let (recorder, handler) = recording_handler();
    recorder.set_reply(json!({"ok": 1}));
    let transport = RedisTransport::new(server.config(5), "team-worker", "team", handler);
    transport.watch_agent("b").unwrap();
    transport.start().await.unwrap();

    let queue = transport.p2p_queue("b");
    // A malformed entry is logged and acked, so the next one still arrives.
    server.push(&queue, &[("envelope", b"not-json".to_vec())]);
    let envelope = MessageEnvelope::p2p(json!({"n": 2}), "a", "b");
    server.push(
        &queue,
        &[("envelope", serde_json::to_vec(&envelope).unwrap())],
    );

    let reply_key = transport.reply_queue(&envelope.message_id);
    wait_until(
        || {
            server
                .entries(&reply_key)
                .iter()
                .any(|entry| entry.field("frame").is_some())
        },
        "the reply frame",
    )
    .await;
    let replies = server.entries(&reply_key);
    let frame: Value = serde_json::from_slice(replies[0].field("frame").unwrap()).unwrap();
    assert_eq!(frame["value"], json!({"ok": 1}));
    assert_eq!(recorder.p2p_payloads(), vec![json!({"n": 2})]);
    wait_until(
        || server.commands_named("XACK").len() >= 2,
        "both entries (poison included) are acked",
    )
    .await;

    transport.stop().await.unwrap();
}

#[tokio::test]
async fn redis_pubsub_fans_out_to_two_watched_subscribers() {
    let server = FakeRedis::start().await;
    let (first, handler_a) = recording_handler();
    let (second, handler_b) = recording_handler();
    let sub_a = RedisTransport::new(server.config(5), "team-fanout", "team", handler_a);
    sub_a.watch_agent("sub-a").unwrap();
    sub_a.start().await.unwrap();
    let sub_b = RedisTransport::new(server.config(5), "team-fanout", "team", handler_b);
    sub_b.watch_agent("sub-b").unwrap();
    sub_b.start().await.unwrap();

    let (_publisher_recorder, publisher_handler) = recording_handler();
    let publisher = RedisTransport::new(server.config(5), "team-fanout", "team", publisher_handler);
    publisher.start().await.unwrap();
    // The subscriber registry is local to the publishing transport.
    publisher.add_subscription("sub-b", "news");
    publisher.add_subscription("sub-a", "news");
    publisher
        .publish(MessageEnvelope::pubsub(
            json!({"story": 1}),
            "publisher",
            "news",
        ))
        .await
        .unwrap();

    wait_for_pubsub(&first, 1).await;
    wait_for_pubsub(&second, 1).await;
    assert_eq!(first.pubsub_order(), vec!["sub-a"]);
    assert_eq!(second.pubsub_order(), vec!["sub-b"]);
    assert_eq!(first.pubsub.lock()[0].1.message, json!({"story": 1}));

    // One copy per sorted subscriber, each carrying only the envelope.
    let adds = server.commands_named("XADD");
    let pushed: Vec<&str> = adds
        .iter()
        .filter(|args| args.len() == 5)
        .map(|args| args[1].as_str())
        .collect();
    assert_eq!(
        pushed,
        vec![
            "team:team-fanout:pubsub:sub-a",
            "team:team-fanout:pubsub:sub-b",
        ]
    );

    sub_a.stop().await.unwrap();
    sub_b.stop().await.unwrap();
    publisher.stop().await.unwrap();
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
    assert!(server.commands_named("XADD").is_empty());
    transport.stop().await.unwrap();
}

#[tokio::test]
async fn non_single_topologies_are_rejected_as_unsupported() {
    let (_recorder, handler) = recording_handler();
    for (topology, needle) in [
        (RedisTopology::Cluster, "Redis Cluster client"),
        (RedisTopology::Sentinel, "Redis Sentinel client"),
    ] {
        let config = TransportConfig::new()
            .with_kind(TransportKind::Redis)
            .with_redis(RedisTransportConfig::new().with_topology(topology));
        let error = build_transport(
            &config,
            "team",
            MessageBusConfig::new(),
            Arc::clone(&handler),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_string();
        assert!(message.contains(needle), "{message}");
        assert!(message.contains(topology.as_str()), "{message}");
    }
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
async fn redis_watch_agent_is_idempotent_and_stop_joins_workers() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "team-lifecycle", "team", handler);
    assert!(!transport.is_started());
    transport.watch_agent("agent").unwrap();
    transport.watch_agent("agent").unwrap();
    assert!(transport.watch_agent("").is_err());
    transport.start().await.unwrap();
    transport.start().await.unwrap();
    assert!(transport.is_started());
    // Watching the same agent once more must not spawn a second worker.
    transport.watch_agent("agent").unwrap();

    let p2p_key = transport.p2p_queue("agent");
    wait_until(
        || {
            server
                .commands_named("XGROUP")
                .iter()
                .any(|args| args.get(2) == Some(&p2p_key))
        },
        "the consumer group is created",
    )
    .await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    let creates = server
        .commands_named("XGROUP")
        .iter()
        .filter(|args| args.get(2) == Some(&p2p_key))
        .count();
    assert_eq!(creates, 1, "one worker per inbox, not one per watch call");

    transport.stop().await.unwrap();
    let polls = server.commands_named("XREADGROUP").len();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        server.commands_named("XREADGROUP").len(),
        polls,
        "stop must join every worker"
    );
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
async fn redis_empty_team_and_prefix_fall_back_to_defaults() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(5), "", "", handler);
    assert_eq!(transport.team_id(), "default");
    assert_eq!(transport.key_prefix(), "openjiuwen:team");
    assert_eq!(transport.p2p_queue("a"), "openjiuwen:team:default:p2p:a");
    assert_eq!(
        transport.p2p_consumer_group(),
        "openjiuwen:team:default:p2p:cg"
    );
    // Debug output exposes the scheme but never a password.
    let rendered = format!("{transport:?}");
    assert!(rendered.contains("RedisTransport"), "{rendered}");
}
