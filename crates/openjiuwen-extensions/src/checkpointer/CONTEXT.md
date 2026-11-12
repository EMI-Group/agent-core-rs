# checkpointer/ — checkpointer factory and deferred Redis backend

## Intent

Python's `extensions/checkpointer/redis/` registers a Redis provider with the core `CheckpointerFactory`; the local memory/file/sqlite backends live in the kernel.
Rust keeps that split: this module dispatches on the kernel's `CheckpointerConfig` to the kernel backends, and owns only the Redis surface, which is deferred (`Error::Unsupported`) while keeping the Python constructor names.
Pick this node for any "checkpoint backend selection" or "Redis checkpointer" request.

## API Surface

```rust
pub use openjiuwen_core::session::checkpoint::{
    Checkpointer, CheckpointerConfig, CheckpointerKind, Checkpoint, CheckpointMeta, CheckpointNamespace,
    MemoryCheckpointer, FileCheckpointer,
};
#[cfg(feature = "sqlite")]
pub use openjiuwen_core::session::checkpoint::SqliteCheckpointer;

/// Builds the checkpointer `config.kind` selects.
/// `InMemory`/`File`/`Sqlite` delegate to `openjiuwen_core::session::checkpoint::create_checkpointer`.
/// `Redis` (only reachable from `RedisCheckpointerConfig`, not `CheckpointerKind`) returns `Error::Unsupported`.
pub async fn build_checkpointer(config: &CheckpointerConfig) -> Result<Arc<dyn Checkpointer>>;

/// `redis` feature only.
pub mod redis {
    pub struct RedisTTLConfig { pub default_ttl_minutes: Option<f64>, pub refresh_on_read: bool }
    pub struct RedisConnectionConfig { pub url: Option<String>, pub cluster_mode: Option<bool>,
                                       pub connection_args: BTreeMap<String, Value> }
    pub struct RedisCheckpointerConfig { pub connection: RedisConnectionConfig,
                                         pub ttl: Option<RedisTTLConfig> }
    pub struct RedisCheckpointer { .. }
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
        /// Validates the URL scheme (`redis://`, `rediss://`, `redis+cluster://`, `rediss+cluster://`).
        pub fn new(config: RedisCheckpointerConfig) -> Result<Self>;
        /// Python's provider entry point: always `Error::Unsupported` today.
        pub async fn connect(config: RedisCheckpointerConfig) -> Result<Arc<dyn Checkpointer>>;
        pub fn config(&self) -> &RedisCheckpointerConfig;
    }
    #[async_trait] impl Checkpointer for RedisCheckpointer { /* every method -> Error::Unsupported */ }
    pub async fn create(config: &RedisCheckpointerConfig) -> Result<Arc<dyn Checkpointer>>; // alias of connect
}
```

## Ported Behavior

- `RedisConnectionConfig` validation matches Python: a URL is required (Rust has no `redis_client`, so a URL is the only connection method) and must start with one of the four accepted schemes; `cluster_mode` is inferred from the `+cluster` scheme when unset; `connection_args` are carried untouched.
- `connection_url()` mirrors Python `get_connection_url`: the `+cluster` marker is stripped because a real cluster client would take the plain `redis://`/`rediss://` URL.
- `RedisTTLConfig::default_ttl_minutes` serializes as `default_ttl` to keep Python's field name; `refresh_on_read` defaults to `false` and unknown JSON fields are ignored.
- `RedisTTLConfig::default_ttl_minutes` and `refresh_on_read` are serialized/preserved; the deferred backend names them in its `Unsupported` message so operators know what was requested.
- `RedisCheckpointerConfig` round-trips through JSON with Python's field names (`connection`, `ttl`, `default_ttl`, `refresh_on_read`).
- The Redis backend is **not** implemented: pickle-encoded blobs are a deliberate non-goal (`docs/PARITY.md` intentional deviations), and key layout (`{session_id}:agent:{agent_id}:agent_state_blobs`) is documented only.
- Every `Checkpointer` method on `RedisCheckpointer` returns `Error::Unsupported("redis checkpointer backend is deferred; use the in-memory, file, or sqlite backend")`; `session_exists` also returns that error rather than `false`, so callers cannot mistake deferral for an empty session.

## Constraints

- This module never re-implements kernel backends; it delegates to `create_checkpointer`.
- No Redis client dependency: the `redis` feature adds no crate, only the API surface (the backend would need `redis` when implemented).
- Errors: `Error::Validation` for a bad URL/config, `Error::Unsupported` for the backend itself.
- All public items documented; files ≤ ~1000 lines.

## Test Strategy

- Unit tests beside `mod.rs`: `CheckpointerKind::InMemory` builds a working checkpointer and a save/load round-trip succeeds; `File` without a path is `ErrorKind::Validation`; the sqlite branch is asserted only under `--features sqlite`.
- `redis` tests: URL scheme validation (all four accepted, `http://` rejected), `cluster_mode` inference, JSON round-trip of `RedisCheckpointerConfig`, and `connect`/every trait method returning `ErrorKind::Unsupported`.
- Python counterparts named in comments: `tests/unit_tests/extensions/checkpointer/test_redis_checkpointer_provider.py`, `test_redis_checkpointer.py`.

## Routing Table

- `./mod.rs` → `build_checkpointer`, kernel re-exports, tests
- `./redis.rs` → `redis` feature surface (`RedisConnectionConfig`, `RedisTTLConfig`, `RedisCheckpointerConfig`, `RedisCheckpointer`)
