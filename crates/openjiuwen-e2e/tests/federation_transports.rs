#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::too_many_lines
)]
//! Federation over the kernel transport seam: the Redis Streams team transport plus the
//! routing-table and inbox-filter layers that decide who receives what.
//!
//! The suite drives the kernel through the umbrella only (`openjiuwen::core::multi_agent::…`
//! and `openjiuwen::core::foundation::…`), the way a host process composes a fleet: one
//! [`build_transport`] call selects the backend, several transports of one team share a server,
//! and each one dispatches through its own [`MessageHandler`].
//!
//! # The fake server
//!
//! Everything is offline: every transport points at a loopback fake Redis server
//! (`TcpListener` on `127.0.0.1:0`) that decodes client commands with the foundation RESP
//! [`Decoder`] and keeps a small but faithful Streams state machine — per-key entries,
//! per-`(key, group)` delivery cursors and pending lists, plus the subscription-registry sets.
//!
//! It answers only the commands these scenarios need:
//!
//! - `PING`, `AUTH`, `SELECT` (handshake);
//! - `XADD`, `XGROUP CREATE` (with `MKSTREAM` and `BUSYGROUP`), `XREADGROUP` (both `>` and an
//!   explicit id), `XREAD`, `XACK`, `DEL`;
//! - `XAUTOCLAIM` with an always-empty claim pass (no scenario leaves work pending across a
//!   reconnect);
//! - the registry SET family `SADD`, `SREM`, `SMEMBERS`, `SCARD`, with a switch that makes all
//!   four answer `-ERR` so a failed registry write is observable.
//!
//! Redis delivery is polled by design, so a scenario waits for the *recorded* evidence (the
//! handler's deliveries, the server's commands and pending lists) instead of ordering events
//! with sleeps; every wait is bounded well under [`support::BOUND`], so a hang fails fast.
//!
//! # Scenarios
//!
//! 1. **Streams round trip** — a transport from the public factory publishes to a watched
//!    agent; the envelope arrives unchanged, the request travels as one `XADD` carrying its
//!    reply key, and `XACK` clears the group's pending entry once the reply was written.
//! 2. **Routing** — a [`RoutingTable`] snapshot resolves groups and topic patterns to sorted
//!    recipients; a group broadcast reaches both members, a routed topic reaches only its
//!    subscriber, and a directed send reaches only its target.
//! 3. **Inbox filters** — an [`InboxFilter`] on the recipient's handler rejects denied senders,
//!    missing metadata and off-topic publications; a P2P rejection is a typed error at the
//!    sender, a dropped publication is still acknowledged, and removing the filter restores
//!    delivery.
//! 4. **Failure paths** — a registry write/read that answers `-ERR` surfaces a typed error while
//!    the local subscriptions survive, an unreachable server fails `start` instead of hanging,
//!    and a non-single Redis topology is a typed `Unsupported`.
mod support;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen::core::common::error::{Error, ErrorKind, Result};
use openjiuwen::core::foundation::store::external::redis::codec::{Decoder, RespValue};
use openjiuwen::core::multi_agent::inbox::InboxFilter;
use openjiuwen::core::multi_agent::routing::RoutingTable;
use openjiuwen::core::multi_agent::transport::{
    build_transport, RedisTopology, RedisTransport, RedisTransportConfig, TeamTransport,
    TransportConfig, TransportKind,
};
use openjiuwen::core::multi_agent::{MessageBusConfig, MessageEnvelope, MessageHandler};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use support::{within, BOUND};

/// Team id every transport in this suite joins.
const TEAM: &str = "fleet";

/// Key prefix every transport in this suite uses (kept equal to [`TEAM`] for readable keys).
const PREFIX: &str = "fleet";

/// Poll delay of every transport, in milliseconds: short enough to be fast, never a deadline.
const POLL_MS: u64 = 5;

/// Registry refresh delay, pushed out so the explicit refresh is what a scenario observes.
const REGISTRY_MS: u64 = 60_000;

/// Builds a recording handler plus the erased handler the transports take.
fn recording_handler() -> (Arc<RecordingHandler>, Arc<dyn MessageHandler>) {
    let recorder = RecordingHandler::new();
    let handler = Arc::clone(&recorder) as Arc<dyn MessageHandler>;
    (recorder, handler)
}

/// A [`MessageHandler`] that records what arrived, answers P2P with one value, and can apply an
/// [`InboxFilter`] the way the runtime's dispatcher does.
///
/// A rejected P2P delivery is a [`Error::Validation`] the sender receives; a rejected publication
/// is dropped silently (the runtime logs it), so the worker still acknowledges the entry.
#[derive(Default)]
struct RecordingHandler {
    /// Every P2P envelope handled, in handling order.
    p2p: Mutex<Vec<MessageEnvelope>>,
    /// Every pub/sub delivery as `(subscriber, envelope)`, in handling order.
    pubsub: Mutex<Vec<(String, MessageEnvelope)>>,
    /// The reason of every rejected delivery, in rejection order.
    rejections: Mutex<Vec<String>>,
    /// The reply every accepted P2P delivery returns.
    reply: Mutex<Option<Value>>,
    /// The inbox filter applied to every inbound delivery, when any.
    filter: Mutex<Option<InboxFilter>>,
}

impl RecordingHandler {
    /// Builds a handler replying `{"ok": true}`.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            reply: Mutex::new(Some(json!({"ok": true}))),
            ..Self::default()
        })
    }

    /// Sets the value every accepted P2P delivery replies with.
    fn set_reply(&self, value: Value) {
        *self.reply.lock().expect("reply") = Some(value);
    }

    /// Applies `filter` to every inbound delivery.
    fn restrict(&self, filter: InboxFilter) {
        *self.filter.lock().expect("filter") = Some(filter);
    }

    /// Stops filtering, so every delivery is accepted again.
    fn clear_filter(&self) {
        *self.filter.lock().expect("filter") = None;
    }

    /// Returns the recorded P2P payloads, in handling order.
    fn p2p_payloads(&self) -> Vec<Value> {
        self.p2p
            .lock()
            .expect("p2p")
            .iter()
            .map(|envelope| envelope.message.clone())
            .collect()
    }

    /// Returns the recorded pub/sub subscriber ids, in handling order.
    fn pubsub_order(&self) -> Vec<String> {
        self.pubsub
            .lock()
            .expect("pubsub")
            .iter()
            .map(|(subscriber, _envelope)| subscriber.clone())
            .collect()
    }

    /// Returns the rejection reasons recorded so far.
    fn rejections(&self) -> Vec<String> {
        self.rejections.lock().expect("rejections").clone()
    }

    /// Applies the configured filter, recording the reason when it rejects `envelope`.
    fn rejection(&self, envelope: &MessageEnvelope, kind: &str) -> Option<String> {
        let filter = self.filter.lock().expect("filter").clone()?;
        if filter.accepts(envelope) {
            return None;
        }
        let destination = envelope
            .recipient
            .as_deref()
            .or(envelope.topic_id.as_deref())
            .unwrap_or("-");
        let reason = format!("inbox filter rejected the {kind} delivery for '{destination}'");
        self.rejections
            .lock()
            .expect("rejections")
            .push(reason.clone());
        Some(reason)
    }
}

#[async_trait]
impl MessageHandler for RecordingHandler {
    async fn handle_p2p(&self, envelope: &MessageEnvelope) -> Result<Value> {
        if let Some(reason) = self.rejection(envelope, "point-to-point") {
            return Err(Error::validation(reason));
        }
        self.p2p.lock().expect("p2p").push(envelope.clone());
        Ok(self
            .reply
            .lock()
            .expect("reply")
            .clone()
            .unwrap_or(Value::Null))
    }

    async fn handle_pubsub(&self, envelope: &MessageEnvelope, subscriber: &str) -> Result<()> {
        if self.rejection(envelope, "pub/sub").is_some() {
            return Ok(());
        }
        self.pubsub
            .lock()
            .expect("pubsub")
            .push((subscriber.to_owned(), envelope.clone()));
        Ok(())
    }
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

/// One consumer group of one stream.
#[derive(Default)]
struct Group {
    /// Index of the first entry the `>` cursor has not delivered yet.
    last_delivered: usize,
    /// Ids delivered to some consumer and not acknowledged yet.
    pending: Vec<String>,
}

/// In-memory state of the fake Redis server.
#[derive(Default)]
struct ServerState {
    /// One stream per Redis key.
    streams: Mutex<HashMap<String, Stream>>,
    /// One Redis SET per key (the subscription registry).
    sets: Mutex<HashMap<String, BTreeSet<String>>>,
    /// One consumer group per `(key, group)` pair.
    groups: Mutex<HashMap<(String, String), Group>>,
    /// Every command received, arguments rendered as text.
    commands: Mutex<Vec<Vec<String>>>,
    /// When set, every SET command answers an error frame.
    fail_sets: AtomicBool,
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
            "XAUTOCLAIM" => Self::xautoclaim(),
            "DEL" => self.del(args),
            "SADD" => self.sadd(args),
            "SREM" => self.srem(args),
            "SMEMBERS" => self.smembers(args),
            "SCARD" => self.scard(args),
            other => server_error(&format!("ERR unknown command '{other}'")),
        }
    }

    /// Answers every SET command with `-ERR` while the failure switch is on.
    fn registry_error(&self) -> Option<Vec<u8>> {
        self.fail_sets
            .load(Ordering::SeqCst)
            .then(|| server_error("ERR subscription registry unavailable"))
    }

    /// Handles `SADD key member [member ...]`, returning how many were added.
    fn sadd(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if let Some(error) = self.registry_error() {
            return error;
        }
        if args.len() < 3 {
            return server_error("ERR wrong number of arguments for 'sadd'");
        }
        let key = text(&args[1]);
        let mut sets = self.sets.lock().expect("sets");
        let set = sets.entry(key).or_default();
        let added = args[2..]
            .iter()
            .filter(|member| set.insert(text(member)))
            .count();
        integer(added)
    }

    /// Handles `SREM key member [member ...]`, returning how many were removed.
    fn srem(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if let Some(error) = self.registry_error() {
            return error;
        }
        if args.len() < 3 {
            return server_error("ERR wrong number of arguments for 'srem'");
        }
        let key = text(&args[1]);
        let mut sets = self.sets.lock().expect("sets");
        let Some(set) = sets.get_mut(&key) else {
            return integer(0);
        };
        let removed = args[2..]
            .iter()
            .filter(|member| set.remove(&text(member)))
            .count();
        if set.is_empty() {
            sets.remove(&key);
        }
        integer(removed)
    }

    /// Handles `SMEMBERS key`, returning the members sorted.
    fn smembers(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if let Some(error) = self.registry_error() {
            return error;
        }
        if args.len() < 2 {
            return server_error("ERR wrong number of arguments for 'smembers'");
        }
        let key = text(&args[1]);
        let members: Vec<String> = self
            .sets
            .lock()
            .expect("sets")
            .get(&key)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default();
        let mut out = array(members.len());
        for member in members {
            out.extend(bulk(member.as_bytes()));
        }
        out
    }

    /// Handles `SCARD key`, returning the cardinality.
    fn scard(&self, args: &[Vec<u8>]) -> Vec<u8> {
        if let Some(error) = self.registry_error() {
            return error;
        }
        if args.len() < 2 {
            return server_error("ERR wrong number of arguments for 'scard'");
        }
        let key = text(&args[1]);
        let count = self
            .sets
            .lock()
            .expect("sets")
            .get(&key)
            .map_or(0, BTreeSet::len);
        integer(count)
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
            group_state
                .pending
                .extend(slice.iter().map(|entry| entry.id.clone()));
            slice
        } else {
            let ids: Vec<String> = group_state
                .pending
                .iter()
                .filter(|id| id.as_str() > start.as_str())
                .take(count)
                .cloned()
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

    /// Handles `XACK key group id`, removing the pending entry.
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
            group_state.pending.retain(|pending| *pending != id);
            if group_state.pending.len() < before {
                removed += 1;
            }
        }
        integer(removed)
    }

    /// Answers an empty `XAUTOCLAIM` claim pass.
    ///
    /// No scenario here leaves work pending across a reconnect, so the cursor is always `0-0`
    /// and nothing is claimed; the worker's start-up pass still gets a well-formed reply.
    fn xautoclaim() -> Vec<u8> {
        let mut out = array(3);
        out.extend(bulk(b"0-0"));
        out.extend(array(0));
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

    /// Returns the pending entry ids of `group` on `key`.
    fn pending_ids(&self, key: &str, group: &str) -> Vec<String> {
        self.groups
            .lock()
            .expect("groups")
            .get(&(key.to_owned(), group.to_owned()))
            .map_or_else(Vec::new, |state| state.pending.clone())
    }

    /// Returns how many entries of `key` the `>` cursor of `group` has handed out.
    fn delivered_len(&self, key: &str, group: &str) -> usize {
        self.groups
            .lock()
            .expect("groups")
            .get(&(key.to_owned(), group.to_owned()))
            .map_or(0, |state| state.last_delivered)
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
    fn config(&self) -> RedisTransportConfig {
        RedisTransportConfig::new()
            .with_addr(self.addr.clone())
            .with_poll_interval_ms(POLL_MS)
            .with_registry_interval_ms(REGISTRY_MS)
    }

    /// Returns a snapshot of a stream.
    fn entries(&self, key: &str) -> Vec<Entry> {
        self.state.entries(key)
    }

    /// Returns the recorded commands named `name`.
    fn commands_named(&self, name: &str) -> Vec<Vec<String>> {
        self.state.commands_named(name)
    }

    /// Returns the pending entry ids of a group.
    fn pending_ids(&self, key: &str, group: &str) -> Vec<String> {
        self.state.pending_ids(key, group)
    }

    /// Returns how many entries a group has handed out.
    fn delivered_len(&self, key: &str, group: &str) -> usize {
        self.state.delivered_len(key, group)
    }

    /// Makes every SET command answer `-ERR` (or clears that switch).
    fn fail_sets(&self, fail: bool) {
        self.state.fail_sets.store(fail, Ordering::SeqCst);
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

/// Polls `condition`, panicking when it never holds.
///
/// Redis delivery is polled, so evidence is awaited rather than slept on; the loop bounds the
/// wait to roughly five seconds, well inside [`support::BOUND`].
async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    for _ in 0..1_000 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

/// Waits until `recorder` handled `count` P2P deliveries.
async fn wait_for_p2p(recorder: &RecordingHandler, count: usize) {
    wait_until("the P2P deliveries to arrive", || {
        recorder.p2p.lock().expect("p2p").len() >= count
    })
    .await;
}

/// Waits until `recorder` handled `count` pub/sub deliveries.
async fn wait_for_pubsub(recorder: &RecordingHandler, count: usize) {
    wait_until("the pub/sub deliveries to arrive", || {
        recorder.pubsub.lock().expect("pubsub").len() >= count
    })
    .await;
}

/// Waits until the server recorded a command named `name` matching `matches`.
async fn wait_for_command(
    server: &FakeRedis,
    name: &str,
    matches: impl Fn(&[String]) -> bool,
    what: &str,
) {
    wait_until(what, || {
        server.commands_named(name).iter().any(|args| matches(args))
    })
    .await;
}

#[tokio::test]
async fn redis_streams_round_trip_acks_the_delivered_request() {
    let server = FakeRedis::start().await;
    let (receiver_recorder, receiver_handler) = recording_handler();
    receiver_recorder.set_reply(json!({"echo": "hi"}));
    let receiver = RedisTransport::new(server.config(), TEAM, PREFIX, receiver_handler);
    receiver.watch_agent("alpha").unwrap();
    within("the receiving transport starts", receiver.start())
        .await
        .unwrap();
    assert_eq!(receiver.kind(), TransportKind::Redis);
    assert_eq!(receiver.team_id(), TEAM);
    assert_eq!(receiver.p2p_queue("alpha"), "fleet:fleet:p2p:alpha");
    assert_eq!(receiver.p2p_consumer_group(), "fleet:fleet:p2p:cg");

    // The publisher comes from the public factory: that is the seam a host process composes.
    let (_publisher_recorder, publisher_handler) = recording_handler();
    let config = TransportConfig::new()
        .with_kind(TransportKind::Redis)
        .with_key_prefix(PREFIX)
        .with_redis(server.config());
    let publisher =
        build_transport(&config, TEAM, MessageBusConfig::new(), publisher_handler).unwrap();
    assert_eq!(publisher.kind(), TransportKind::Redis);
    assert_eq!(publisher.team_id(), TEAM);
    within("the publishing transport starts", publisher.start())
        .await
        .unwrap();

    let request = MessageEnvelope::p2p(json!({"n": 7}), "sender", "alpha").with_session("s1");
    let message_id = request.message_id.clone();
    let reply = within("the P2P round trip", publisher.send(request, Some(BOUND)))
        .await
        .unwrap();
    assert_eq!(reply, json!({"echo": "hi"}));

    // The watched agent's handler saw the envelope unchanged.
    let delivered = receiver_recorder.p2p.lock().expect("p2p").clone();
    assert_eq!(delivered.len(), 1);
    assert_eq!(delivered[0].message, json!({"n": 7}));
    assert_eq!(delivered[0].sender.as_deref(), Some("sender"));
    assert_eq!(delivered[0].recipient.as_deref(), Some("alpha"));
    assert_eq!(delivered[0].session_id.as_deref(), Some("s1"));

    // The request travelled as one stream entry carrying its own reply key.
    let queue = receiver.p2p_queue("alpha");
    let reply_key = format!("fleet:fleet:reply:{message_id}");
    let adds = server.commands_named("XADD");
    let p2p = adds
        .iter()
        .find(|args| args.len() == 7 && args[1] == queue)
        .expect("the P2P XADD");
    assert_eq!(p2p[2], "*");
    assert_eq!(p2p[3], "envelope");
    assert_eq!(p2p[5], "reply");
    assert_eq!(p2p[6], reply_key);

    // The stream itself holds the envelope the sender published.
    let stored = server.entries(&queue);
    assert_eq!(stored.len(), 1);
    let sent: MessageEnvelope = serde_json::from_slice(
        stored[0]
            .field("envelope")
            .expect("the entry carries an envelope field"),
    )
    .unwrap();
    assert_eq!(sent.message_id, message_id);
    assert_eq!(sent.recipient.as_deref(), Some("alpha"));
    assert_eq!(sent.message, json!({"n": 7}));

    // Delivery is Streams, never a destructive list.
    assert!(server.commands_named("RPOP").is_empty());
    assert!(server.commands_named("RPUSH").is_empty());
    assert!(server.commands_named("LLEN").is_empty());

    // XACK cleared the pending entry, so nothing is left for the group to reclaim.
    let group = receiver.p2p_consumer_group();
    let entry_id = server.entries(&queue)[0].id.clone();
    wait_until("the delivered request to be acknowledged", || {
        server.pending_ids(&queue, &group).is_empty()
    })
    .await;
    let acks = server.commands_named("XACK");
    assert!(
        acks.iter().any(|args| args.len() == 4
            && args[1] == queue
            && args[2] == group
            && args[3] == entry_id),
        "{acks:?}"
    );
    assert_eq!(server.delivered_len(&queue, &group), 1);
    // The consumed reply stream was deleted, so a later request cannot read a stale frame.
    assert!(server
        .commands_named("DEL")
        .iter()
        .any(|args| args.len() == 2 && args[1] == reply_key));

    within("the receiving transport stops", receiver.stop())
        .await
        .unwrap();
    within("the publishing transport stops", publisher.stop())
        .await
        .unwrap();
}

#[tokio::test]
async fn broadcast_and_directed_deliveries_follow_the_routing_table() {
    let server = FakeRedis::start().await;
    let (alpha_recorder, alpha_handler) = recording_handler();
    let alpha = RedisTransport::new(server.config(), TEAM, PREFIX, alpha_handler);
    alpha.watch_agent("alpha").unwrap();
    within("the first member's transport starts", alpha.start())
        .await
        .unwrap();
    let (beta_recorder, beta_handler) = recording_handler();
    let beta = RedisTransport::new(server.config(), TEAM, PREFIX, beta_handler);
    beta.watch_agent("beta").unwrap();
    within("the second member's transport starts", beta.start())
        .await
        .unwrap();

    // The snapshot a host assembles from its group and topic registries.
    let routing = RoutingTable {
        team_id: TEAM.to_owned(),
        groups: BTreeMap::from([(
            "crew".to_owned(),
            vec!["alpha".to_owned(), "beta".to_owned()],
        )]),
        topics: BTreeMap::from([
            ("news".to_owned(), vec!["alpha".to_owned()]),
            ("news.*".to_owned(), vec!["beta".to_owned()]),
        ]),
        agents: BTreeMap::from([
            (
                "alpha".to_owned(),
                vec!["crew".to_owned(), "news".to_owned()],
            ),
            (
                "beta".to_owned(),
                vec!["crew".to_owned(), "news.*".to_owned()],
            ),
        ]),
    };
    assert_eq!(routing.route("crew"), vec!["alpha", "beta"]);
    assert_eq!(routing.route("news"), vec!["alpha"]);
    assert_eq!(routing.route("news.tech"), vec!["beta"]);
    assert!(routing.route("nobody").is_empty());
    assert_eq!(routing.destinations("beta"), vec!["crew", "news.*"]);

    let (_publisher_recorder, publisher_handler) = recording_handler();
    let publisher = RedisTransport::new(server.config(), TEAM, PREFIX, publisher_handler);
    within("the publishing transport starts", publisher.start())
        .await
        .unwrap();
    for subscriber in routing.route("crew") {
        publisher.add_subscription(&subscriber, "crew");
    }
    within(
        "the group broadcast",
        publisher.publish(MessageEnvelope::pubsub(
            json!({"broadcast": 1}),
            "coordinator",
            "crew",
        )),
    )
    .await
    .unwrap();
    wait_for_pubsub(&alpha_recorder, 1).await;
    wait_for_pubsub(&beta_recorder, 1).await;

    // A concrete topic resolves through its pattern, so only the routed member receives it.
    for subscriber in routing.route("news.tech") {
        publisher.add_subscription(&subscriber, "news.tech");
    }
    within(
        "the routed topic publish",
        publisher.publish(MessageEnvelope::pubsub(
            json!({"theme": "tech"}),
            "coordinator",
            "news.tech",
        )),
    )
    .await
    .unwrap();
    wait_for_pubsub(&beta_recorder, 2).await;
    // Each transport records its own subscriber id: only `beta` saw the themed publication.
    assert_eq!(alpha_recorder.pubsub_order(), vec!["alpha"]);
    assert_eq!(beta_recorder.pubsub_order(), vec!["beta", "beta"]);
    assert_eq!(
        alpha_recorder.pubsub.lock().expect("pubsub")[0].1.message,
        json!({"broadcast": 1})
    );
    assert_eq!(
        beta_recorder.pubsub.lock().expect("pubsub")[1].1.message,
        json!({"theme": "tech"})
    );

    // One stream copy per routed subscriber, in sorted order.
    let pushed: Vec<String> = server
        .commands_named("XADD")
        .iter()
        .filter(|args| args.len() == 5 && args[1].contains(":pubsub:"))
        .map(|args| args[1].clone())
        .collect();
    assert_eq!(
        pushed,
        vec![
            "fleet:fleet:pubsub:alpha",
            "fleet:fleet:pubsub:beta",
            "fleet:fleet:pubsub:beta",
        ]
    );

    // A directed message reaches only its target, never the whole crew.
    let reply = within(
        "the directed send",
        publisher.send(
            MessageEnvelope::p2p(json!({"dm": 1}), "coordinator", "beta"),
            Some(BOUND),
        ),
    )
    .await
    .unwrap();
    assert_eq!(reply, json!({"ok": true}));
    wait_for_p2p(&beta_recorder, 1).await;
    assert_eq!(beta_recorder.p2p_payloads(), vec![json!({"dm": 1})]);
    assert!(
        alpha_recorder.p2p.lock().expect("p2p").is_empty(),
        "a directed message reaches only its target"
    );
    let directed: Vec<String> = server
        .commands_named("XADD")
        .iter()
        .filter(|args| args.len() == 7)
        .map(|args| args[1].clone())
        .collect();
    assert_eq!(directed, vec!["fleet:fleet:p2p:beta"]);

    within("the first member's transport stops", alpha.stop())
        .await
        .unwrap();
    within("the second member's transport stops", beta.stop())
        .await
        .unwrap();
    within("the publishing transport stops", publisher.stop())
        .await
        .unwrap();
}

#[tokio::test]
async fn inbox_filters_gate_what_reaches_a_member() {
    let server = FakeRedis::start().await;
    let (recorder, handler) = recording_handler();
    let filter = InboxFilter::new()
        .allow_senders(["member"])
        .deny_senders(["spammer"])
        .allow_topics(["news.*"])
        .require_metadata("team", "blue")
        .require_session();
    recorder.restrict(filter.clone());
    let alpha = RedisTransport::new(server.config(), TEAM, PREFIX, handler);
    alpha.watch_agent("alpha").unwrap();
    within("the filtered transport starts", alpha.start())
        .await
        .unwrap();

    // The rules are pure data: the same envelope shapes decide here and at the handler.
    let accepted = MessageEnvelope::pubsub(json!({"story": 1}), "member", "news.tech")
        .with_metadata("team", json!("blue"))
        .with_session("s1");
    assert!(filter.accepts(&accepted));
    // A P2P envelope carries no topic, so a topic rule rejects it.
    assert!(!filter.accepts(
        &MessageEnvelope::p2p(json!({"n": 1}), "member", "alpha")
            .with_metadata("team", json!("blue"))
            .with_session("s1")
    ));
    assert!(!filter.accepts(&accepted.clone().with_metadata("team", json!("red"))));
    assert!(!filter.accepts(
        &MessageEnvelope::pubsub(json!({}), "member", "alerts")
            .with_metadata("team", json!("blue"))
            .with_session("s1")
    ));
    assert!(!filter.accepts(
        &MessageEnvelope::pubsub(json!({}), "spammer", "news.tech")
            .with_metadata("team", json!("blue"))
            .with_session("s1")
    ));

    let (_publisher_recorder, publisher_handler) = recording_handler();
    let publisher = RedisTransport::new(server.config(), TEAM, PREFIX, publisher_handler);
    within("the publishing transport starts", publisher.start())
        .await
        .unwrap();
    publisher.add_subscription("alpha", "news.tech");
    publisher.add_subscription("alpha", "alerts");

    // A denied sender never reaches the member: the rejection is the sender's typed error.
    let denied = MessageEnvelope::p2p(json!({"n": 1}), "spammer", "alpha")
        .with_metadata("team", json!("blue"))
        .with_session("s1");
    let error = within("the denied sender", publisher.send(denied, Some(BOUND)))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.message().contains("remote handler failed"),
        "{}",
        error.message()
    );
    assert!(
        error
            .message()
            .contains("inbox filter rejected the point-to-point delivery for 'alpha'"),
        "{}",
        error.message()
    );
    assert!(recorder.p2p.lock().expect("p2p").is_empty());

    // A missing metadata pair and a sender-only mismatch are the same typed failure.
    let anonymous = MessageEnvelope::p2p(json!({"n": 2}), "member", "alpha").with_session("s1");
    let error = within(
        "the missing metadata",
        publisher.send(anonymous, Some(BOUND)),
    )
    .await
    .unwrap_err();
    assert!(
        error.message().contains("inbox filter rejected"),
        "{}",
        error.message()
    );

    // Publications: the accepted one is delivered, the off-topic and denied ones are dropped.
    within("the accepted publish", publisher.publish(accepted))
        .await
        .unwrap();
    wait_for_pubsub(&recorder, 1).await;
    within(
        "the off-topic publish",
        publisher.publish(
            MessageEnvelope::pubsub(json!({"story": 2}), "member", "alerts")
                .with_metadata("team", json!("blue"))
                .with_session("s1"),
        ),
    )
    .await
    .unwrap();
    within(
        "the denied publisher",
        publisher.publish(
            MessageEnvelope::pubsub(json!({"story": 3}), "spammer", "news.tech")
                .with_metadata("team", json!("blue"))
                .with_session("s1"),
        ),
    )
    .await
    .unwrap();
    wait_until("both dropped publications to be recorded", || {
        recorder.rejections().len() == 4
    })
    .await;
    assert_eq!(recorder.pubsub_order(), vec!["alpha"]);
    assert_eq!(
        recorder.pubsub.lock().expect("pubsub")[0].1.message,
        json!({"story": 1})
    );
    assert!(recorder.p2p.lock().expect("p2p").is_empty());
    let rejections = recorder.rejections();
    assert_eq!(
        rejections
            .iter()
            .filter(|reason| reason.contains("point-to-point"))
            .count(),
        2,
        "{rejections:?}"
    );
    assert_eq!(
        rejections
            .iter()
            .filter(|reason| reason.contains("pub/sub"))
            .count(),
        2,
        "{rejections:?}"
    );

    // A dropped publication is still acknowledged, so the inbox never redelivers it.
    let queue = alpha.pubsub_queue("alpha");
    let group = alpha.pubsub_consumer_group();
    wait_until("every publication to be acknowledged", || {
        server
            .commands_named("XACK")
            .iter()
            .filter(|args| args.len() == 4 && args[1] == queue)
            .count()
            == 3
    })
    .await;
    assert!(server.pending_ids(&queue, &group).is_empty());

    // Removing the filter restores delivery without touching the transport.
    recorder.clear_filter();
    let reply = within(
        "the unfiltered send",
        publisher.send(
            MessageEnvelope::p2p(json!({"n": 4}), "spammer", "alpha"),
            Some(BOUND),
        ),
    )
    .await
    .unwrap();
    assert_eq!(reply, json!({"ok": true}));
    assert_eq!(recorder.p2p_payloads(), vec![json!({"n": 4})]);

    within("the filtered transport stops", alpha.stop())
        .await
        .unwrap();
    within("the publishing transport stops", publisher.stop())
        .await
        .unwrap();
}

#[tokio::test]
async fn registry_failures_and_unreachable_servers_are_typed() {
    let server = FakeRedis::start().await;
    let (_recorder, handler) = recording_handler();
    let transport = RedisTransport::new(server.config(), TEAM, PREFIX, handler);
    let index = "fleet:fleet:subs";
    // Every registry write fails while the switch is on.
    server.fail_sets(true);
    transport.add_subscription("alpha", "news");
    // The synchronous call neither fails nor hangs: the pair is local at once.
    assert_eq!(transport.agent_subscriptions("alpha"), vec!["news"]);
    wait_for_command(
        &server,
        "SADD",
        |args| args.len() == 3 && args[1] == index,
        "the failed write-through",
    )
    .await;

    let error = within("the registry refresh", transport.refresh_subscriptions())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::External);
    assert!(
        error
            .message()
            .contains("subscription registry unavailable"),
        "{}",
        error.message()
    );
    // The failed refresh kept the local pair: no delivery route was lost.
    assert_eq!(transport.agent_subscriptions("alpha"), vec!["news"]);
    assert_eq!(transport.subscription_count(), 1);

    // A later refresh reconnects and merges the owned pair back in.
    server.fail_sets(false);
    within(
        "the recovered registry refresh",
        transport.refresh_subscriptions(),
    )
    .await
    .unwrap();
    assert_eq!(transport.agent_subscriptions("alpha"), vec!["news"]);
    assert_eq!(transport.subscription_count(), 1);

    // An unreachable server fails `start` instead of hanging on the first delivery.
    let (_other_recorder, other_handler) = recording_handler();
    let unreachable = RedisTransport::new(
        RedisTransportConfig::new()
            .with_addr("127.0.0.1:1")
            .with_poll_interval_ms(POLL_MS),
        TEAM,
        PREFIX,
        other_handler,
    );
    let error = within("the unreachable start", unreachable.start())
        .await
        .unwrap_err();
    assert!(
        matches!(error.kind(), ErrorKind::External | ErrorKind::Transport),
        "{}",
        error.kind().as_str()
    );
    assert!(
        error.message().contains("redis connect"),
        "{}",
        error.message()
    );
    assert!(!unreachable.is_started());

    // A topology the workspace cannot serve is a typed `Unsupported`, never a silent fallback.
    let (_cluster_recorder, cluster_handler) = recording_handler();
    let cluster = TransportConfig::new()
        .with_kind(TransportKind::Redis)
        .with_redis(server.config().with_topology(RedisTopology::Cluster));
    let error =
        build_transport(&cluster, TEAM, MessageBusConfig::new(), cluster_handler).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.message().contains("Redis Cluster client"),
        "{}",
        error.message()
    );
}
