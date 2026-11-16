//! Cluster redirection: `MOVED`/`ASK` parsing, hash slots, and node bookkeeping.
//!
//! A [`Nodes`] holds the seed node named by [`ConnectionOptions`] plus one lazily
//! created [`Node`] per redirection target, so a redirected command runs on the
//! node that owns the key and later commands reuse that socket.
//! Redirect targets are keyed by `(host, port)` and inherit the seed's credentials,
//! database, timeout, and TLS setting.
//!
//! The slot cache maps a `u16` hash slot to the node that reported ownership of it
//! (`MOVED`, or a `CLUSTER SLOTS` pre-seed). It is consulted only for single-key
//! commands listed in [`SINGLE_KEY_COMMANDS`], because for anything else the client
//! cannot know which key decides the slot; those commands go to the seed and rely on
//! the one-shot redirect instead.
use super::connection::Connection;
use super::ConnectionOptions;
use crate::redis::codec::{encode_command, Reply};
use openjiuwen_common::error::{Error, Result};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex as StdMutex, MutexGuard};
use tokio::sync::Mutex;

/// The `host:port` of one `Redis` node.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct NodeAddr {
    /// Host name or IP literal.
    pub(super) host: String,
    /// Server port.
    pub(super) port: u16,
}

impl NodeAddr {
    /// Records a bare `(host, port)` pair.
    pub(super) fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_owned(),
            port,
        }
    }
}

impl std::fmt::Display for NodeAddr {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}:{}", self.host, self.port)
    }
}

/// One connection to a `Redis` node, serialized by its own mutex.
pub(super) struct Node {
    addr: NodeAddr,
    options: ConnectionOptions,
    connection: Mutex<Connection>,
}

impl Node {
    /// Creates a node handle; nothing is connected until the first command.
    pub(super) fn new(options: ConnectionOptions) -> Self {
        Self {
            addr: NodeAddr::new(&options.host, options.port),
            options,
            connection: Mutex::new(Connection::default()),
        }
    }

    /// The `host:port` this node connects to.
    pub(super) fn addr(&self) -> &NodeAddr {
        &self.addr
    }

    /// The seed's options retargeted at `addr` (same credentials, database, TLS).
    pub(super) fn options_for(&self, addr: &NodeAddr) -> ConnectionOptions {
        ConnectionOptions {
            host: addr.host.clone(),
            port: addr.port,
            ..self.options.clone()
        }
    }

    /// Connects and runs the handshake when the node has no open socket.
    pub(super) async fn ensure_open(&self) -> Result<()> {
        let timeout = self.options.timeout;
        let mut connection = self.lock(timeout, "connect").await?;
        connection.ensure_open(&self.options).await
    }

    /// Sends one command on this node and returns its raw reply.
    pub(super) async fn reply(&self, payload: &[u8], name: &str) -> Result<Reply> {
        let mut replies = self.exchange(payload, 1, name).await?;
        replies
            .pop()
            .ok_or_else(|| Error::transport("redis connection closed before a reply was read"))
    }

    /// Sends `ASKING` followed by the command and returns the command's reply.
    ///
    /// Both commands travel in one write and are read under one lock acquisition, so
    /// no other command can slip between `ASKING` and the redirected command.
    pub(super) async fn asking_reply(&self, payload: &[u8], name: &str) -> Result<Reply> {
        let mut combined = encode_command(&[b"ASKING"]);
        combined.extend_from_slice(payload);
        let mut replies = self.exchange(&combined, 2, name).await?;
        let reply = replies
            .pop()
            .ok_or_else(|| Error::transport("redis connection closed before a reply was read"))?;
        match replies.pop() {
            Some(Reply::Simple(text)) if text == "OK" => Ok(reply),
            Some(Reply::Error(message)) => {
                Err(Error::transport(format!("redis ASKING failed: {message}")))
            }
            Some(other) => Err(Error::transport(format!(
                "redis ASKING returned an unexpected {} reply",
                other.kind_name()
            ))),
            None => Err(Error::transport(
                "redis connection closed before the ASKING reply was read",
            )),
        }
    }

    /// Writes `payload` and reads `expected` replies on this node.
    pub(super) async fn exchange(
        &self,
        payload: &[u8],
        expected: usize,
        name: &str,
    ) -> Result<Vec<Reply>> {
        let timeout = self.options.timeout;
        let mut connection = self.lock(timeout, name).await?;
        connection.ensure_open(&self.options).await?;
        connection.exchange(payload, expected, timeout, name).await
    }

    /// Acquires the node's connection lock under the configured timeout.
    async fn lock(
        &self,
        timeout: std::time::Duration,
        name: &str,
    ) -> Result<tokio::sync::MutexGuard<'_, Connection>> {
        tokio::time::timeout(timeout, self.connection.lock())
            .await
            .map_err(|_| {
                Error::timeout(format!(
                    "timed out waiting for the redis connection to run '{name}'"
                ))
            })
    }
}

/// Locks a standard mutex, recovering the guard if a previous holder panicked.
///
/// The maps only hold node handles, so a poisoned lock still owns consistent data and
/// must not take the client down.
fn lock<T>(mutex: &StdMutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The seed node together with every redirection target opened so far.
pub(super) struct Nodes {
    seed: Arc<Node>,
    redirects: StdMutex<BTreeMap<NodeAddr, Arc<Node>>>,
    slots: StdMutex<BTreeMap<u16, NodeAddr>>,
}

impl Nodes {
    /// Wraps the seed node described by `options`.
    pub(super) fn new(options: ConnectionOptions) -> Self {
        Self {
            seed: Arc::new(Node::new(options)),
            redirects: StdMutex::new(BTreeMap::new()),
            slots: StdMutex::new(BTreeMap::new()),
        }
    }

    /// The seed node named by the connection options.
    pub(super) fn seed(&self) -> Arc<Node> {
        Arc::clone(&self.seed)
    }

    /// The node for `addr`, reusing an open connection when one exists.
    pub(super) fn node(&self, addr: &NodeAddr) -> Arc<Node> {
        if *addr == *self.seed.addr() {
            return self.seed();
        }
        let mut redirects = lock(&self.redirects);
        Arc::clone(
            redirects
                .entry(addr.clone())
                .or_insert_with(|| Arc::new(Node::new(self.seed.options_for(addr)))),
        )
    }

    /// The node that owns `slot`, when a redirection or pre-seed recorded one.
    pub(super) fn slot_node(&self, slot: u16) -> Option<Arc<Node>> {
        let addr = lock(&self.slots).get(&slot).cloned()?;
        Some(self.node(&addr))
    }

    /// Records that `addr` owns `slot` (from a `MOVED` reply or `CLUSTER SLOTS`).
    pub(super) fn remember_slot(&self, slot: u16, addr: NodeAddr) {
        lock(&self.slots).insert(slot, addr);
    }

    /// Records every slot in `start..=end` as owned by `addr`.
    fn remember_slot_range(&self, start: u16, end: u16, addr: &NodeAddr) {
        let mut slots = lock(&self.slots);
        for slot in start..=end {
            slots.insert(slot, addr.clone());
        }
    }
}

/// Which redirection a cluster node asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RedirectKind {
    /// `-MOVED <slot> <host>:<port>`: the slot moved permanently.
    Moved,
    /// `-ASK <slot> <host>:<port>`: the slot is being migrated; send `ASKING` first.
    Ask,
}

/// A parsed `MOVED` or `ASK` redirection.
#[derive(Debug, Clone)]
pub(super) struct Redirect {
    /// Whether the redirection is permanent (`MOVED`) or one-shot (`ASK`).
    pub(super) kind: RedirectKind,
    /// The hash slot the server reported.
    pub(super) slot: u16,
    /// The node the client must contact.
    pub(super) addr: NodeAddr,
    /// The server's original message, kept for error text.
    pub(super) raw: String,
}

/// Parses a `MOVED`/`ASK` error message (`"MOVED 3999 127.0.0.1:7000"`).
///
/// Returns `None` for any other error message, and for a redirection whose slot or
/// address cannot be parsed (such a message is then reported as a plain server error).
pub(super) fn parse_redirect(message: &str) -> Option<Redirect> {
    let (kind, rest) = if let Some(rest) = message.strip_prefix("MOVED ") {
        (RedirectKind::Moved, rest)
    } else {
        (RedirectKind::Ask, message.strip_prefix("ASK ")?)
    };
    let mut fields = rest.split_whitespace();
    let slot = fields.next()?.parse::<u16>().ok()?;
    let address = fields.next()?;
    let node = parse_node_address(address)?;
    Some(Redirect {
        kind,
        slot,
        addr: node,
        raw: message.to_owned(),
    })
}

/// Splits `host:port`, accepting bracketed IPv6 literals (`[::1]:7000`).
fn parse_node_address(address: &str) -> Option<NodeAddr> {
    let (host, port) = if let Some(rest) = address.strip_prefix('[') {
        let (host, rest) = rest.split_once(']')?;
        (host, rest.strip_prefix(':')?)
    } else {
        address.rsplit_once(':')?
    };
    let port = port.parse::<u16>().ok()?;
    if host.is_empty() || port == 0 {
        return None;
    }
    Some(NodeAddr::new(host, port))
}

/// Commands whose first argument is the single key that decides the cluster slot.
///
/// Only these consult the slot cache; multi-key commands (`MGET`, `MSET`, ...) and
/// keyless commands always start at the seed node and rely on the one-shot redirect,
/// because the client cannot tell which of several keys the server would report.
pub(super) const SINGLE_KEY_COMMANDS: &[&str] = &[
    "DECR",
    "DECRBY",
    "DEL",
    "EXISTS",
    "EXPIRE",
    "GET",
    "INCR",
    "INCRBY",
    "PTTL",
    "SADD",
    "SET",
    "SISMEMBER",
    "SMEMBERS",
    "SREM",
    "TTL",
    "TYPE",
    "UNLINK",
    "ZADD",
    "ZCARD",
    "ZREM",
    "ZRANGE",
    "ZSCORE",
];

/// The hash slot of the key a command addresses, when it has exactly one key.
pub(super) fn command_slot(args: &[&[u8]]) -> Option<u16> {
    let name = args.first()?;
    let key = args.get(1)?;
    let known = SINGLE_KEY_COMMANDS
        .iter()
        .any(|candidate| name.eq_ignore_ascii_case(candidate.as_bytes()));
    known.then(|| hash_slot(key))
}

/// The `Redis` cluster hash slot of `key`, honouring `{hash tags}`.
pub(super) fn hash_slot(key: &[u8]) -> u16 {
    crc16(hash_tag(key)) & 0x3fff
}

/// Extracts the `{...}` hash tag, or the whole key when there is no usable tag.
fn hash_tag(key: &[u8]) -> &[u8] {
    let Some(open) = key.iter().position(|byte| *byte == b'{') else {
        return key;
    };
    let Some(close) = key[open + 1..].iter().position(|byte| *byte == b'}') else {
        return key;
    };
    if close == 0 {
        return key;
    }
    &key[open + 1..open + 1 + close]
}

/// `CRC-16/XMODEM`, the checksum `Redis` uses for cluster hash slots.
fn crc16(bytes: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for byte in bytes {
        crc ^= u16::from(*byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Verifies that the seed node really is part of a cluster.
///
/// `CLUSTER INFO` must report `cluster_enabled:1`, or `CLUSTER SLOTS` must return a
/// slot map; a standalone node is rejected instead of being silently used.
pub(super) async fn verify_cluster(node: &Node) -> Result<()> {
    if cluster_enabled(node).await || !read_slot_ranges(node).await.is_empty() {
        return Ok(());
    }
    Err(Error::validation(format!(
        "redis cluster mode was requested but {} is not a cluster node: CLUSTER INFO does not \
         report cluster_enabled:1 and CLUSTER SLOTS returned no slot ranges; use redis:// for a \
         standalone server",
        node.addr()
    )))
}

/// Whether `CLUSTER INFO` reports `cluster_enabled:1`.
async fn cluster_enabled(node: &Node) -> bool {
    let payload = encode_command(&[b"CLUSTER", b"INFO"]);
    match node.reply(&payload, "CLUSTER INFO").await {
        Ok(reply) => {
            let text = reply.as_opt_str().unwrap_or_default().map(str::to_owned);
            let enabled = text.is_some_and(|text| text.contains("cluster_enabled:1"));
            if !enabled {
                tracing::debug!(
                    node = %node.addr(),
                    "redis node does not report cluster_enabled:1"
                );
            }
            enabled
        }
        Err(error) => {
            tracing::debug!(node = %node.addr(), %error, "redis CLUSTER INFO failed");
            false
        }
    }
}

/// Pre-seeds the slot cache from `CLUSTER SLOTS`; failures are logged and ignored.
pub(super) async fn preseed_slots(nodes: &Nodes, node: &Node) {
    let ranges = read_slot_ranges(node).await;
    if ranges.is_empty() {
        tracing::debug!(
            node = %node.addr(),
            "redis CLUSTER SLOTS returned no slot ranges; slot routing starts from the seed node"
        );
        return;
    }
    for (start, end, addr) in &ranges {
        nodes.remember_slot_range(*start, *end, addr);
    }
    tracing::debug!(
        node = %node.addr(),
        ranges = ranges.len(),
        "redis cluster slot map pre-seeded from CLUSTER SLOTS"
    );
}

/// Reads `CLUSTER SLOTS` as `(start, end, node)` ranges, tolerating junk entries.
async fn read_slot_ranges(node: &Node) -> Vec<(u16, u16, NodeAddr)> {
    let payload = encode_command(&[b"CLUSTER", b"SLOTS"]);
    let Ok(reply) = node.reply(&payload, "CLUSTER SLOTS").await else {
        return Vec::new();
    };
    let Ok(ranges) = reply.as_array() else {
        return Vec::new();
    };
    ranges.iter().filter_map(parse_slot_range).collect()
}

/// Parses one `CLUSTER SLOTS` entry: `[start, end, [host, port, id], ...]`.
fn parse_slot_range(range: &Reply) -> Option<(u16, u16, NodeAddr)> {
    let items = range.as_array().ok()?;
    let start = u16::try_from(items.first()?.as_integer().ok()?).ok()?;
    let end = u16::try_from(items.get(1)?.as_integer().ok()?).ok()?;
    let parts = items.get(2)?.as_array().ok()?;
    let host = parts.first()?.as_str().ok()?.to_owned();
    let port = u16::try_from(parts.get(1)?.as_integer().ok()?).ok()?;
    if host.is_empty() || port == 0 || start > end {
        return None;
    }
    Some((start, end, NodeAddr::new(&host, port)))
}
