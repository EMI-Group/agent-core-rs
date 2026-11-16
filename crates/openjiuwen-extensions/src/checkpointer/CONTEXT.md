# checkpointer/ — kernel checkpointer factory and the live single-node Redis backend

## Intent

Python's `extensions/checkpointer/redis/` registers a Redis provider with the core `CheckpointerFactory`; the memory/file/sqlite backends live in the kernel, which also ships its own Redis checkpointer behind `CheckpointerKind::Redis`.
Rust keeps that split: this module dispatches on the kernel's `CheckpointerConfig` to the kernel backends, and owns a second Redis backend over the crate's in-tree RESP client.
The Redis backend here is a live single-node implementation over the crate's in-tree RESP client (`crate::redis`), because the workspace vendors no Redis driver and builds offline.
The kernel factory's `CheckpointerKind::Redis` builds the kernel's own Redis checkpointer, so `build_checkpointer` never returns this module's backend; construct it with `RedisCheckpointer::new` (lazy connect) or `connect`/`create` (eager connect plus `PING`).
Pick this node for any "checkpoint backend selection" or "Redis checkpointer" request.

## API Surface

```rust
pub use openjiuwen_core::session::checkpoint::{
    Checkpointer, CheckpointerConfig, CheckpointerKind, Checkpoint, CheckpointMeta, CheckpointNamespace,
    MemoryCheckpointer, FileCheckpointer,
};
#[cfg(feature = "sqlite")]
pub use openjiuwen_core::session::checkpoint::SqliteCheckpointer;

/// Builds the checkpointer `config.kind` selects (`InMemory`/`File`/`Redis`/`Sqlite`),
/// delegating to `openjiuwen_core::session::checkpoint::create_checkpointer`.
pub async fn build_checkpointer(config: &CheckpointerConfig) -> Result<Arc<dyn Checkpointer>>;

/// `redis` feature only.
pub mod redis {
    pub struct RedisTTLConfig { pub default_ttl_minutes: Option<f64>, pub refresh_on_read: bool }
    pub struct RedisConnectionConfig { pub url: Option<String>, pub cluster_mode: Option<bool>,
                                       pub connection_args: BTreeMap<String, Value> }
    pub struct RedisCheckpointerConfig { pub connection: RedisConnectionConfig,
                                         pub ttl: Option<RedisTTLConfig> }
    pub struct RedisCheckpointer { /* config + lazy `OnceCell<Client>` */ }
    impl RedisTTLConfig {
        pub fn new(default_ttl_minutes: Option<f64>, refresh_on_read: bool) -> Self;
    }
    impl RedisConnectionConfig {
        pub fn new(url: impl Into<String>) -> Self;
        /// Missing URL or a scheme outside the four accepted ones -> `Error::Validation`.
        pub fn validate(&self) -> Result<()>;
        /// Explicit `cluster_mode` wins, else the `+cluster` scheme is inferred, else `false`.
        pub fn is_cluster_mode(&self) -> bool;
        /// Normalises `redis+cluster://`/`rediss+cluster://` to `redis://`/`rediss://`.
        pub fn connection_url(&self) -> Option<String>;
    }
    impl RedisCheckpointerConfig {
        pub fn new(connection: RedisConnectionConfig) -> Self;
        pub fn with_ttl(self, ttl: Option<RedisTTLConfig>) -> Self;
    }
    impl RedisCheckpointer {
        /// Validates the URL scheme and returns a handle that connects on first use.
        pub fn new(config: RedisCheckpointerConfig) -> Result<Self>;
        /// Python's provider entry point: connects eagerly, verifies `PING`, returns a live checkpointer.
        pub async fn connect(config: RedisCheckpointerConfig) -> Result<Arc<dyn Checkpointer>>;
        /// Alias of `connect`.
        pub async fn create(config: &RedisCheckpointerConfig) -> Result<Arc<dyn Checkpointer>>;
        pub const fn config(&self) -> &RedisCheckpointerConfig;
    }
    #[async_trait] impl Checkpointer for RedisCheckpointer { /* live single-node implementation */ }
}
```

## Storage Layout

- Payload key `openjiuwen:checkpoint:{session_id}:{checkpoint_id}` holds the `Checkpoint` as compact JSON.
- Index key `openjiuwen:checkpoint-index:{session_id}` is a sorted set of that session's checkpoint ids, scored by save time in epoch milliseconds.
- `save` writes `SET payload` + `ZADD index` (plus `EXPIRE` on both when a default TTL is set) in one pipelined batch.
- `delete` writes `DEL payload` + `ZREM index` in one pipelined batch; deleting a missing checkpoint is a no-op.
- `release` pages `SCAN MATCH openjiuwen:checkpoint:{session_id}:* COUNT 500`, deletes payloads in chunks of 500, then deletes the index key.
- The release pattern ends in `:` and escapes glob metacharacters (`*?[]\`) in the session id, so `session-1` never matches `session-10`.
- `list` reads `ZRANGE index 0 -1`, `MGET`s the payloads in chunks of 500, drops index members whose payload vanished (and `ZREM`s them), and returns summaries newest first by `(created_at, checkpoint_id)` descending.
- `latest` stays the trait default (sort `list`, then `load` the first), so it agrees with the backend ordering.
- `session_exists` is `ZCARD index > 0`; an unknown session answers `false`, a missing payload answers `Ok(None)`, and an empty session lists no checkpoints — none of those are errors.
- Ids are minted and validated exactly like the kernel memory backend: `validate_session_id`/`validate_checkpoint_id` first, empty `checkpoint_id` -> `uuid_hex()`, empty `created_at` -> `now_rfc3339()`.
- TTL seconds are `ceil(minutes * 60)` with a floor of one second; `refresh_on_read` re-arms both the payload and index expiry after a successful read and ignores a failed refresh (a read never fails because of it).
- Undecodable payloads are `Error::Session` naming the session and checkpoint ids, never the payload bytes.

## Ported Behavior

- `RedisConnectionConfig` validation matches Python: a URL is required (the Rust client is built from a URL, there is no pre-configured `redis_client`) and must start with one of the four accepted schemes; `cluster_mode` is inferred from the `+cluster` scheme when unset; `connection_args` are carried untouched.
- `connection_url()` mirrors Python `get_connection_url`: the `+cluster` marker is stripped because the client takes the plain `redis://`/`rediss://` URL and records cluster mode separately.
- `RedisTTLConfig::default_ttl_minutes` serializes as `default_ttl` to keep Python's field name; `refresh_on_read` defaults to `false` and unknown JSON fields are ignored.
- `RedisCheckpointerConfig` round-trips through JSON with Python's field names (`connection`, `ttl`, `default_ttl`, `refresh_on_read`).
- `RedisCheckpointer` has a hand-written redacting `Debug` (it prints cluster mode, TTL, and connection state; never the URL or `connection_args`, which may carry credentials) and is not `Clone` because the client is a single shared `OnceCell`.
- `RedisCheckpointer::connect`/`create` fail fast: they validate the config, open the connection (applying `connection_args` `username`/`password`/`db`/`timeout` and cluster mode), and require a `PING` reply.
- Python's `async` signatures are kept as real async bodies; the former `clippy::unused_async` allows are gone.

## Constraints

- This module never re-implements kernel backends; `build_checkpointer` delegates to `create_checkpointer`.
- No Redis driver dependency: the `redis` feature adds no crate, only the in-crate client and this backend.
- Only `src/checkpointer/redis.rs` and the module docs change here; `src/redis/**` is consumed as-is.
- Errors: `Error::Validation` for bad ids/URLs, `Error::Session` for encode/decode failures, `Error::Transport`/`Timeout` from the client (including TLS handshake and cluster-redirection failures).
- No payload or credential is logged; refresh failures log at `debug` with the session id only.
- All public items documented; files ≤ ~1000 lines.

## Test Strategy

- Unit tests beside `mod.rs`: `CheckpointerKind::InMemory` builds a working checkpointer and a save/load round-trip succeeds; `File` without a path is `ErrorKind::Validation`; the sqlite branch is asserted only under `--features sqlite`.
- `redis` config tests (no server): all four schemes accepted, missing/`http://` rejected, cluster inference and explicit override, `connection_url` marker stripping, `connection_args` carried through, Python-field JSON round-trip, `new` URL validation, glob escaping, TTL second rounding, redacting `Debug`.
- `redis` live tests against `crate::redis::testsupport::FakeRedis`: full-field save/load round-trip, explicit ids, invalid-id validation, missing checkpoint/session results, newest-first `list` plus `latest`, `delete` vs `release` (including the `session-1`/`session-10` prefix case), stale-index cleanup, undecodable payloads, TTL on save, TTL refresh on read and its absence without `refresh_on_read`, `connect`/`create`, cluster `MOVED` following, unreachable port, and `connection_args` credentials.
- Python counterparts named in comments: `tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py`, `test_redis_checkpointer_provider.py`.

## Remaining Deferrals

- `rediss://`/`rediss+cluster://` work through the crate's client: the TLS handshake verifies the server against the web PKI (`tokio-rustls` + `webpki-roots`, both locked offline), so no TLS error is returned for a valid certificate chain.
- Cluster URLs verify the seed as a cluster node and follow one `MOVED`/`ASK` redirection per command, so slot routing is real rather than recorded-and-ignored.
- Python's pickle payload format and `{session_id}:agent:{agent_id}:agent_state_blobs` key layout are intentionally replaced by the JSON payload plus sorted-set index described above.

## Routing Table

- `./mod.rs` → kernel re-exports, `build_checkpointer`, module-level docs, backend-selection tests
- `./redis.rs` → `redis` feature backend: configuration types, live single-node `Checkpointer`, key layout helpers, tests
