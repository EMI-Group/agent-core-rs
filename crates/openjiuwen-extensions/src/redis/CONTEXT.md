# redis/ — in-crate RESP2/RESP3 client and stateful test server

## Intent

Provide the single-node Redis transport the Redis checkpointer and key-value store adapters need, without a Redis driver crate.
The workspace builds offline and vendors no `redis`/`deadpool-redis`/`bb8`, so this module speaks the Redis serialization protocol directly over `tokio::net::TcpStream`.
Scope is single-node command execution: one serialized connection per `Client`, optional `AUTH`/`SELECT` handshake, pipelining, and typed error mapping.
TLS (`rediss://`) and cluster slot routing are deliberately out of scope and return `Error::Unsupported` with a reason.
The module is compiled only under `--features redis` (declared in `src/lib.rs`); it adds no Cargo dependency.

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
    pub cluster_mode: bool }
impl ConnectionOptions {
    pub fn from_url(url: &str) -> Result<Self>;  // redis://[user[:pass]@]host[:port][/db]
    pub fn with_connection_args(self, args: &BTreeMap<String, serde_json::Value>) -> Result<Self>;
    pub fn with_cluster_mode(self, cluster_mode: bool) -> Self;
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
- A big number `(` decodes to `Integer` when it fits `i64` and otherwise keeps its literal digits as `Bulk`.
- `-MOVED`/`-ASK` stay `Reply::Error` in the codec; the client maps them to `Error::Unsupported`.
- Parser limits (transport error instead of allocation/recursion blow-up): nesting depth 128, 16M elements per aggregate, 512 MiB per bulk payload; a partial reply is always `Ok(None)`.
- Unsupported RESP3 constructs are typed, never panics: attributes (`|`) and streamed strings (`;`) return `Error::unsupported`.

## Constraints

- No new dependencies and no edits outside `src/redis/`; `bytes` is not a base dependency, so byte buffers are `Vec<u8>`.
- Errors come from `openjiuwen_common::error`: `Validation` for bad URLs/arguments, `WRONGTYPE` replies, and type-mismatched accessors; `Transport` for I/O, handshake, and other server errors; `Timeout` for a connect or round trip exceeding `ConnectionOptions::timeout`; `Unsupported` for `rediss://`, `MOVED`/`ASK`, and unsupported RESP3 framing.
- Credentials never leak: `ConnectionOptions` has a redacting `Debug`, timeout messages carry only the upper-cased command name, and connect errors carry only `host:port`.
- One `tokio::sync::Mutex`-guarded connection per `Client`; a transport failure or timeout closes the socket and the next command reconnects and re-runs the handshake.
- `pipeline` writes every command in one flush and reads exactly `n` replies in order; an error reply fails the whole batch like `command`.
- Logging is `tracing` only (no `println!`/`eprintln!`); cluster mode logs a warning once at connect and still connects to one node.
- `AUTH` is sent as `AUTH user password` when a username is present and as `AUTH password` otherwise; `SELECT` runs after a successful `AUTH`.
- Files stay under ~700 lines; `codec/`, `client/`, and `testsupport/` each split production code from their tests.

## Notes for the Store and Checkpointer Adapters

- Build a connection with `Client::from_url(&url).await?`; apply Python-style `connection_args` first via `ConnectionOptions::from_url(url)?.with_connection_args(&args)?` and then `Client::connect`.
- Use `command_str` for ASCII commands and `crate::redis::command(&[&key_bytes, &value_bytes])` to build binary-safe argument vectors for `command`/`pipeline`.
- Read values with `Reply::as_opt_bytes()` (null → `None`) and counts with `as_integer()`; `as_array()` also accepts RESP3 sets and pushes.
- Key-value TTL semantics rely on plain `SET` clearing any previous TTL (the kernel `KvStore` contract); the fake server enforces exactly that.
- `cluster_mode` is recorded but the client always connects to one node, so a cluster URL only works for keys on the node it reaches; cluster slot routing stays deferred.
- The test server is `#[cfg(test)]` only: `crate::redis::testsupport::{FakeRedis, FakeOptions}`; it is `pub(crate)`, so adapter unit tests in this crate can use it, but external crates cannot.

## Test Strategy

- `codec/tests.rs`: encode vectors (empty list, binary arguments with NUL/CRLF/non-UTF-8), every RESP2/RESP3 type, null vs empty aggregates, `MOVED`/`ASK` surfaced as error replies, unsupported constructs, every truncation length of each sample returning `Ok(None)`, malformed framing, and a deterministic garbage-input sweep proving no panic.
- `client/tests.rs`: URL parsing (credentials, percent-decoding, IPv6 brackets, database path, cluster scheme), TLS/bad-scheme errors, `connection_args` mapping (integer/float/string timeout, null clearing, invalid types), `Send + Sync` assertion, and end-to-end round trips against `FakeRedis` for strings, TTLs, pipeline ordering, sets/sorted sets, `AUTH`, `MOVED`, `WRONGTYPE`, and timeouts.
- `testsupport/`: a stateful loopback server supporting `PING`, `ECHO`, `AUTH`, `SELECT`, `GET`, `SET` (`NX`/`XX`/`EX`/`PX`/`EXAT`/`PXAT`/`KEEPTTL`), `EXISTS`, `DEL`, `UNLINK`, `MGET`, `TTL`, `PTTL`, `EXPIRE`, `SCAN`, `KEYS`, `SADD`, `SREM`, `SMEMBERS`, `SISMEMBER`, `ZCARD`, `ZADD`, `ZRANGE`, `ZREM`, `FLUSHDB`, and `TYPE`; unknown commands answer `-ERR unknown command`.
- Fake server TTLs expire lazily on every command and on every `StateHandle` read, and `ttl_ms(key)` returns `None` for a missing key, `Some(-1)` for no expiry, otherwise the remaining milliseconds.
- Scripted modes: `FakeOptions::auth_required(user, password)`, `FakeOptions::moved_everything()`, and `FakeOptions::with_command_delay(delay)` for timeout tests.
- Definition of done: `cargo fmt`, `cargo clippy -p openjiuwen-extensions --all-targets --features redis -- -D warnings`, `cargo test -p openjiuwen-extensions --features redis`, and `cargo doc -p openjiuwen-extensions --no-deps --features redis` all clean offline.

## Routing Table

- `./codec/` → `encode_command`, incremental `parse_reply`, the `Reply` enum, and its accessors (`mod.rs`; tests in `tests.rs`)
- `./client/` → `ConnectionOptions` URL/`connection_args` parsing, the mutex-guarded `Client`, handshake, pipelining, and error mapping (`mod.rs`; tests in `tests.rs`)
- `./testsupport/` → `#[cfg(test)]` fake Redis: `FakeRedis`/`FakeOptions`/`StateHandle` (`mod.rs`), command execution (`commands.rs`), and shared helpers (`support.rs`)
