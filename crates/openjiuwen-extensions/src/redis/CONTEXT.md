# redis/ — in-crate RESP2/RESP3 client and stateful test server

## Intent

Provide the Redis transport the Redis checkpointer, key-value, document and vector adapters need, without a Redis driver crate.
The workspace builds offline and vendors no `redis`/`deadpool-redis`/`bb8`, so this module speaks the Redis serialization protocol directly over `tokio`.
Scope is single-seed command execution with TLS, cluster redirection, pipelining, an optional `AUTH`/`SELECT` handshake, and typed error mapping.
TLS (`rediss://`) is real: `tokio-rustls` with the `webpki-roots` trust anchors, no insecure escape hatch.
Cluster mode is verified at connect time and `MOVED`/`ASK` redirections are followed once.
The module is compiled only under `--features redis` (declared in `src/lib.rs`), which is the only place `tokio-rustls` and `webpki-roots` are pulled in.

## API Surface

```rust
pub mod client;
pub mod codec;
#[cfg(test)] pub(crate) mod testsupport;   // stateful fake server, tests only

pub use client::{Client, ConnectionOptions};
pub use codec::Reply;
pub fn command(args: &[&[u8]]) -> Vec<Vec<u8>>;  // owned argument vector convenience

// codec
pub fn encode_command(args: &[&[u8]]) -> Vec<u8>;                  // RESP array of bulk strings
pub fn parse_reply(input: &[u8]) -> Result<Option<(Reply, usize)>>;  // incremental, total
pub enum Reply { Simple(String), Error(String), Integer(i64), Bulk(Vec<u8>), Null,
                 Array(Vec<Reply>), Boolean(bool), Double(f64), Map(Vec<(Reply, Reply)>),
                 Set(Vec<Reply>), Push(Vec<Reply>) }

// client
pub struct ConnectionOptions { pub host: String, pub port: u16, pub username: Option<String>,
    pub password: Option<String>, pub db: Option<i64>, pub timeout: Duration,
    pub cluster_mode: bool, pub tls: bool }
impl ConnectionOptions {
    pub fn from_url(url: &str) -> Result<Self>;  // redis[s][+cluster]://[user[:pass]@]host[:port][/db]
    pub fn with_connection_args(self, args: &BTreeMap<String, serde_json::Value>) -> Result<Self>;
    pub fn with_cluster_mode(self, cluster_mode: bool) -> Self;
    pub fn with_tls(self, tls: bool) -> Self;
}
pub struct Client { .. }  // Send + Sync
impl Client {
    pub async fn connect(options: ConnectionOptions) -> Result<Self>;
    pub async fn from_url(url: &str) -> Result<Self>;
    pub fn options(&self) -> &ConnectionOptions;
    pub async fn command(&self, args: &[&[u8]]) -> Result<Reply>;
    pub async fn command_str(&self, args: &[&str]) -> Result<Reply>;
    pub async fn pipeline(&self, commands: &[Vec<Vec<u8>>]) -> Result<Vec<Reply>>;
    pub async fn ping(&self) -> Result<()>;  // requires +PONG
}
```

`Reply` accessors return `Error::validation` on a type mismatch: `as_str`, `as_opt_str`, `as_string_owned`, `as_opt_string_owned`, `as_bytes`, `as_opt_bytes`, `as_integer` (boolean → 1/0), `as_bool`, `as_array` (array/set/push), `as_map`, `kind_name`, `is_null`, `is_error`.

## Protocol Behavior

- RESP2: `+`, `-`, `:`, `$`, `*`; null bulk `$-1` and null array `*-1` both decode to `Reply::Null`, while `$0`/`*0` decode to an empty bulk/array.
- RESP3: `_` → Null, `#t`/`#f` → Boolean, `,` → Double, `%` → Map, `~` → Set, `>` → Push, `=` verbatim string → Bulk without its three-byte encoding tag.
- `|N` attributes are metadata: the `N` key/value pairs are parsed with the normal grammar and projected away, so the reply that follows them is returned; an attribute chain resolves to the value after the whole chain.
- `;N` streamed strings are read chunk by chunk until the `;0` terminator and returned as one binary-safe `Reply::Bulk`.
- A big number `(` decodes to `Integer` when it fits `i64` and otherwise keeps its literal digits as `Bulk`.
- `-MOVED`/`-ASK` stay `Reply::Error` in the codec; the client parses and follows them.
- Parser limits (transport error instead of allocation/recursion blow-up): nesting depth 128, 16M elements per aggregate, 512 MiB per bulk payload and per streamed string (also per chunk); a partial reply is always `Ok(None)`.
- Attributes recurse one level per bag, so an attribute chain deeper than the depth limit is rejected; streamed chunks are scalar and add no depth.

## Constraints

- No new dependencies beyond the `redis` feature's already-declared `tokio-rustls`/`webpki-roots`; `bytes` is not a base dependency, so byte buffers are `Vec<u8>`.
- Errors come from `openjiuwen_common::error`: `Validation` for bad URLs/arguments, a `WRONGTYPE` reply, a type-mismatched accessor, or cluster mode against a non-cluster seed; `Transport` for I/O, TLS handshake failures, handshake errors, other server errors, and a second consecutive redirection; `Timeout` for a connect, TLS handshake, or round trip exceeding `ConnectionOptions::timeout`.
- Credentials never leak: `ConnectionOptions` has a redacting `Debug`, timeout messages carry only the upper-cased command name, and connect errors carry only `host:port`.
- TLS verifies certificates and host names against the web PKI (`webpki_roots::TLS_SERVER_ROOTS`); there is no insecure mode. The `ring` provider is installed as the process default when free and always used explicitly, so a host application that installed another provider still works.
- Each node connection is a `tokio::sync::Mutex`-guarded `Connection`; a transport failure or timeout closes the socket and the next command reconnects and re-runs the handshake.
- `Connection::exchange` owns the wire through an RAII `InFlight` guard: dropping the command future before its replies are read (caller-side `timeout`, `select!`, abort) closes the socket, so the abandoned reply can never be handed to the next command; bytes left over in the buffer at command entry are a typed `Transport` desynchronisation error instead of a reusable reply.
- `Connection::ensure_open` runs `AUTH`/`SELECT` on a staging connection and publishes the socket only after they succeed, so a handshake cancelled mid-way leaves the connection closed rather than marking an unauthenticated socket open.
- The seed node plus every redirection target lives in a node map keyed by `(host, port)`; redirect targets inherit the seed's credentials, database, timeout and TLS flag.
- `MOVED` opens/reuses the target node, re-issues the command exactly once, and caches `slot → node`; `ASK` sends `ASKING` and the command in one write under one lock (no interleaving) and caches nothing.
- The slot cache is consulted only for the single-key commands listed in `cluster::SINGLE_KEY_COMMANDS` (hash tags honoured, `CRC-16/XMODEM & 0x3fff`); multi-key and keyless commands start at the seed and rely on the one-shot redirect.
- `cluster_mode` is verified at connect time (`CLUSTER INFO` must report `cluster_enabled:1`, or `CLUSTER SLOTS` must return ranges) and the slot map is pre-seeded from `CLUSTER SLOTS` best-effort, with failures logged at `debug`.
- `pipeline` writes every command to the seed in one flush, reads exactly `n` replies in order, follows a redirect per command so positions are preserved, and fails the batch on any other error reply.
- Logging is `tracing` only (no `println!`/`eprintln!`); push frames from a server are surfaced as `Reply::Push` and never consumed by the client (these adapters subscribe to nothing).
- `AUTH` is sent as `AUTH user password` when a username is present and as `AUTH password` otherwise; `SELECT` runs after a successful `AUTH`.
- Files stay under ~1000 lines; `codec/`, `client/` (`mod`, `connection`, `cluster`, `tls`), and `testsupport/` split production code from their tests.

## Notes for the Store and Checkpointer Adapters

- Build a connection with `Client::from_url(&url).await?`; apply Python-style `connection_args` first via `ConnectionOptions::from_url(url)?.with_connection_args(&args)?` and then `Client::connect`.
- Use `command_str` for ASCII commands and `crate::redis::command(&[&key_bytes, &value_bytes])` to build binary-safe argument vectors for `command`/`pipeline`.
- Read values with `Reply::as_opt_bytes()` (null → `None`) and counts with `as_integer()`; `as_array()` also accepts RESP3 sets and pushes.
- Key-value TTL semantics rely on plain `SET` clearing any previous TTL (the kernel `KvStore` contract); the fake server enforces exactly that.
- `rediss://` connects over verified TLS and `redis[s]+cluster://` verifies the seed and follows redirections, so the adapters forward the URL unchanged.
- The test server is `#[cfg(test)]` only: `crate::redis::testsupport::{FakeRedis, FakeOptions, ClusterMode}`; it is `pub(crate)`, so adapter unit tests in this crate can use it, but external crates cannot.

## Test Strategy

- `codec/tests.rs`: encode vectors (empty list, binary arguments with NUL/CRLF/non-UTF-8), every RESP2/RESP3 type, null vs empty aggregates, `MOVED`/`ASK` surfaced as error replies, attributes projected onto simple strings/maps/aggregate elements/attribute chains, streamed strings with several chunks, an empty stream, binary chunk payloads, incremental parsing across `parse_reply` calls, oversized chunk header/attribute lengths, malformed chunk terminators, the attribute depth bound, every truncation length of each sample returning `Ok(None)`, malformed framing, and a deterministic garbage-input sweep proving no panic.
- `client/tests.rs`: URL parsing (credentials, percent-decoding, IPv6 brackets, database path, TLS and cluster schemes, the `tls` flag), bad-scheme errors, redirect parsing (slots, bracketed IPv6, junk), hash slots against `CLUSTER KEYSLOT` reference values with hash tags, the single-key command allowlist, `connection_args` mapping, `Send + Sync` assertion, and end-to-end round trips against `FakeRedis` for strings, TTLs, pipeline ordering, sets/sorted sets, `AUTH`, `WRONGTYPE`, timeouts, `MOVED` followed once with slot caching, a doubled `MOVED` typed as `Transport`, `ASKING` before an `ASK`ed command, pipelines that follow redirects positionally, cluster-mode rejection of a standalone node, cluster-mode connect plus `CLUSTER SLOTS` pre-seeding, and a `rediss://` handshake against a plain listener; plus the cancellation regressions `a_dropped_command_future_never_hands_its_reply_to_the_next_command` (abort mid-reply, then prove the next `GET` sees its own value on a fresh connection) and `buffered_unsolicited_replies_are_a_typed_desynchronisation_error`.
- No certificate fixture exists offline, so a successful TLS handshake is not exercised; the TLS tests prove the handshake is attempted and that a non-TLS peer is a typed `Transport` error naming TLS.
- `testsupport/`: a stateful loopback server supporting `PING`, `ECHO`, `AUTH`, `SELECT`, `ASKING`, `GET`, `SET` (`NX`/`XX`/`EX`/`PX`/`EXAT`/`PXAT`/`KEEPTTL`), `EXISTS`, `DEL`, `UNLINK`, `MGET`, `TTL`, `PTTL`, `EXPIRE`, `SCAN`, `KEYS`, `SADD`, `SREM`, `SMEMBERS`, `SISMEMBER`, `ZCARD`, `ZADD`, `ZRANGE`, `ZREM`, `FLUSHDB`, and `TYPE`; unknown commands answer `-ERR unknown command`.
- Fake server TTLs expire lazily on every command and on every `StateHandle` read, and `ttl_ms(key)` returns `None` for a missing key, `Some(-1)` for no expiry, otherwise the remaining milliseconds.
- Scripted modes: `FakeOptions::auth_required(user, password)`, `FakeOptions::moved_to(slot, addr)`, `FakeOptions::ask_to(slot, addr)`, `FakeOptions::cluster()`/`FakeOptions::cluster_owned_by(host, port)` for `CLUSTER INFO`/`CLUSTER SLOTS`, and `FakeOptions::with_command_delay(delay)` for timeout tests.
- The fake records the command log globally (`state().commands()`) and per accepted connection (`state().connection_commands()`, `state().first_connection_commands()`), so a test proves a redirection ran the command exactly once and that `ASKING` preceded it.
- A second node is just a second `FakeRedis` on its own loopback listener; a redirection target is scripted with that node's address.
- Definition of done: `cargo fmt`, `cargo clippy -p openjiuwen-extensions --all-targets --features redis -- -D warnings`, `cargo test -p openjiuwen-extensions --features redis`, and `cargo doc -p openjiuwen-extensions --no-deps --features redis` all clean offline.

## Routing Table

- `./codec/` → `encode_command`, incremental `parse_reply` (incl. attributes and streamed strings), the `Reply` enum, and its accessors (`mod.rs`; tests in `tests.rs`)
- `./client/` → `ConnectionOptions` URL/`connection_args` parsing and the `Client` API (`mod.rs`), the TCP/TLS `Connection` and reply reads (`connection.rs`), node bookkeeping with `MOVED`/`ASK`/hash slots/`CLUSTER` verification (`cluster.rs`), and the rustls configuration (`tls.rs`); tests in `tests.rs`
- `./testsupport/` → `#[cfg(test)]` fake Redis: `FakeRedis`/`FakeOptions`/`ClusterMode`/`StateHandle` (`mod.rs`), command execution including `CLUSTER` scripting (`commands.rs`), and shared helpers (`support.rs`)
