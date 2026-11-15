# external — network-backed storage backends

## Intent

Adapters that back the five storage contracts of the parent module with external services: Chroma, Milvus and Elasticsearch vector stores, an Elasticsearch document store, a Redis key-value store, an S3-compatible object store, and a name-driven registry that builds any of them from JSON configuration.
Each adapter speaks the service's own wire protocol directly — REST through `openjiuwen_common::http::HttpClient`, RESP2 through `tokio::net::TcpStream` — because no vendor SDK can be added to an offline build.
The in-memory and local-filesystem backends in the parent modules remain the behavioral reference: an adapter matches their semantics wherever the service can express them, and documents the difference on the method that cannot.

## API Surface

Each backend module exposes a `*Config` struct, a store type implementing the parent trait, and a re-export at this module's root.

- `chroma` — `ChromaConfig` (`endpoint`, `tenant`, `database`, `api_key`, `metric`, `timeout`, `schema_metadata_key`), `ChromaVectorStore::{new(config), config()}` (`VectorStore`).
- `milvus` — `MilvusConfig` (`endpoint`, `token`, `db_name`, `metric`, `timeout`, `consistency_level`), `MilvusVectorStore::{new(config), config()}` (`VectorStore`).
- `elasticsearch` — `ElasticsearchConfig` (`endpoint`, `index_prefix`, `username`, `password`, `api_key`, `metric`, `timeout`), `ElasticsearchVectorStore::{new(config), with_client(http, config), config()}` (`VectorStore`), `ElasticsearchDocumentStore::{new(config), with_client(http, config), config()}` (`DocumentStore`), and `query::translate(&Value) -> Option<Value>` (the filter → query-DSL compiler).
- `redis` — `RedisConfig` (`url`, `addr`, `username`, `password`, `db`, `timeout`, `key_prefix`) with `connect()`; `RedisKvStore` (`KvStore`); `redis::codec` (`RespValue`, `encode_command`, `encode`, `Decoder`); `redis::client` (`RedisConnection`, `RedisServerError`, `DEFAULT_TIMEOUT`, `READ_ONLY_COMMANDS`).
- `s3` — `S3Config` (`endpoint`, `region`, `bucket`, `credentials`, `path_style`, `timeout`), `S3ObjectStore` (`ObjectStore`); `s3::sigv4` (`Credentials`, `SigningInput`, `SigningResult`, `sign_request`, `hmac_sha256`, `sha256_hex`, `hex_encode`).
- `registry` — `StoreFamily` (`Kv`/`Document`/`Vector`/`Object`/`Graph`), `BackendKind` (`Memory`/`File`/`Sqlite`/`Redis`/`Chroma`/`Milvus`/`Elasticsearch`/`S3` plus the deferred `Pgvector`/`GaussVector`/`Postgres`), `StoreSpec::{parse, from_map}`, `StoreRegistry::{new, register_*_backend, available_names, create_*_store}`, `global_registry`, `available_backend_names`, the free `create_{kv,document,vector,object,graph}_store`, the free `register_*_backend`, and the factory aliases `KvStoreFactory`, `DocumentStoreFactory`, `VectorStoreFactory`, `ObjectStoreFactory`, `GraphStoreFactory`.

## Configuration Convention

Every `*Config` is `serde::Deserialize` with `#[serde(default)]` plus a `Default` impl, so a backend can be built straight from a JSON object (`serde_json::from_value`) and every missing member falls back to a documented default.
Each config also exposes named builder setters (`with_*`) for code, and `new(...)` for the one required member (the endpoint or address).
Secrets use `openjiuwen_common::security::SecretString` (serializes transparently, redacts `Debug`): `ChromaConfig::api_key`, `ElasticsearchConfig::password`/`api_key`, `MilvusConfig::token`, `RedisConfig::password`, `S3Config::credentials`.
Each config carries `timeout: Duration` — 30 s for the HTTP adapters, 5 s for Redis — applied per request.
The vector adapters carry `metric: Metric` defaulting to `Cosine` because `CollectionSchema` has no metric member; it drives both the native distance chosen at collection creation and the score normalization applied to search hits.
Only `RedisConfig::from_env` and `S3Config::from_env` read the environment; the HTTP configs are always explicit.

## Constraints

- No new crate dependency: only the workspace crates already listed in `crates/openjiuwen-foundation/Cargo.toml` — `reqwest` (via `openjiuwen_common::http`), `tokio`, `serde`, `serde_json`, `bytes`, `base64`, `sha2`, `chrono`, `url`, `tracing`, `async-trait`, `parking_lot`, `once_cell`, `uuid`.
- All HTTP goes through `openjiuwen_common::http::HttpClient`; the Redis adapter uses a raw `tokio::net::TcpStream` because RESP is not HTTP. No direct `reqwest` calls and no vendor SDK.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for an unrepresentable schema, field or collection name, `Config` for bad configuration, `NotFound` where the parent trait requires it, `Unsupported` for a capability the service cannot express, and `External`/`Transport`/`Timeout` for service and socket failures (the message names the status code, the URL and a body excerpt).
- No panics on user input; no `unwrap`/`expect` outside `#[cfg(test)]`; no `unsafe`; `tracing` only; every public item documented (`cargo clippy -- -D warnings`).
- The adapters are always compiled: `reqwest`, `tokio`, `sha2`, `base64` and the rest are plain (non-optional) dependencies of this crate, so no adapter sits behind a feature gate and every build exercises the same surface.
- Only the `sqlite` registry branches are feature-gated; the network adapters are not.
- Files stay around ≤1000 lines: split by responsibility and move large test trees into `<module>/tests.rs` (`#[cfg(test)] mod tests;`).
- Tests are offline: `wiremock` for the HTTP services, a scripted `tokio::net::TcpListener` for Redis, and the AWS SigV4 known-answer vectors for S3 signing.

## Design Decisions

- **Shared configuration convention** instead of per-backend bespoke constructors: one `serde`-with-defaults config struct per service, a `Default` impl, `with_*` builders and `SecretString` secrets, so the registry can decode any backend's `*Config` from raw JSON with `serde_json::from_value` and tests can build stores without a config file.
- **Score normalization lives in the adapter.** Every `search` returns scores in `[0, 1]` with higher = more similar, matching `Metric::score`.
  Chroma maps cosine/IP distance `d` to `1 - d/2` (the same value as `(cosine + 1) / 2`) and squared-L2 distance `d` to `1 / (1 + d)`.
  Milvus maps COSINE/IP similarity `s` to `(s + 1) / 2` and L2 distance `d` to `max(0, (4 - d) / 4)`.
  Elasticsearch `_score` is already a similarity in `[0, 1]` for cosine/dot and `1 / (1 + d)` for `l2_norm`, so it is clamped as-is; every adapter clamps into `[0, 1]`.
- **Insert-only semantics are preserved where the service allows it.** Chroma `/add` and Elasticsearch `create` bulk actions reject a duplicate id (mapped to `Validation`); Milvus insert cannot detect a duplicate cheaply, so that limitation is documented on `MilvusVectorStore::add_docs`.
- **Schema round-trip.** Chroma stores the serialized `CollectionSchema` inside the collection metadata under `schema_metadata_key` (default `openjiuwen:schema`); Milvus reconstructs the schema from `describe`; Elasticsearch stores it under `_meta.openjiuwen_schema` and otherwise rebuilds it from index mappings plus a synthesized `id` keyword primary field.
- **Redis value encoding.** `KvValue::Text` is stored as raw UTF-8; `KvValue::Bytes` is stored as the ASCII marker `\0ojb:` followed by standard base64, so a text value can never collide with a byte value. A value that is not valid UTF-8 and lacks the marker decodes as `Bytes` rather than being lossily replaced, so the encoding is lossless for every value this store writes and safe for foreign ones.
- **Redis reconnect rule.** At most one transparent reconnect and retry per command: a failure before or while writing is retried for every command (the server never saw a complete command), while a failure while reading the reply is retried only for the read-only commands in `READ_ONLY_COMMANDS` (`PING`, `GET`, `MGET`, `EXISTS`, `SCAN`).
  A mutating command is never retried once its bytes were written, because a lost reply would otherwise become a duplicate write (notably `SET ... NX`).
  Every reconnect replays the `AUTH`/`SELECT` handshake.
- **S3 Signature V4 is in-crate.** HMAC-SHA256 is built from `sha2`, and the hex/base64 helpers are hand-written (no `hex`/`hmac` crate). `sign_request` takes an explicit `now` (`SigningInput::now`) so tests are deterministic, and only the `s3` service and the caller's region are signed.
  Path segments are RFC 3986-encoded exactly once (S3 signs single-encoded paths), the canonical query sorts by key then value, and `x-amz-content-sha256` is emitted always but signed only when the caller lists it — which `S3ObjectStore` always does.
  Addressing is path-style by default (`{endpoint}/{bucket}/{key}`, what `MinIO` and most gateways expect) and virtual-host (`{scheme}://{bucket}.{host}/{key}`) when `path_style = false`.
  A 404 from `GET`/`HEAD`/`DELETE` is tolerated as "missing object" (`Ok(None)`, `false`, no-op) because S3 cannot distinguish a missing bucket from a missing key there; only `list` maps a 404 to `NotFound`.
- **Registry globals.** The plugin maps behind `register_*_backend` are the one documented mutable global of this subtree (`RwLock<HashMap<..>>` inside a `once_cell::sync::Lazy`); `global_registry()` exposes it, and `StoreRegistry::new()` builds an isolated instance over per-instance maps for tests.
  A registered name wins over a built-in of the same spelling, and a backend of the wrong family is a `Validation` error listing the names that family accepts.

## Known Issues and Gaps

- Three files exceed the ~1000-line baseline: `chroma.rs` (~1120 lines), `milvus.rs` (~1281) and `elasticsearch/mod.rs` (~1149).
  Each holds the complete wire mapping plus the client-side validation for its service, and each already moves its test tree out into `<module>/tests.rs` (`chroma/tests.rs`, `milvus/tests.rs`, `elasticsearch/{document,vector}/tests.rs`); `elasticsearch/mod.rs` keeps its inline tests for the shared client, config and mapping helpers.
- Milvus insert is an append, so `add_docs` cannot detect a primary key already stored: the batch is validated in full locally (nothing inserts if one row is bad) and only a service-side duplicate maps to `Validation`.
- Elasticsearch document `query`, `count` and `delete` with a filter the translator cannot express fall back to a bounded local scan over at most 10 000 documents (`DEFAULT_QUERY_LIMIT`, Elasticsearch's default `index.max_result_window`), paged 500 at a time, and `limit: None` uses that same 10 000-result window.
- Redis has no TLS: a `rediss://` URL returns `Unsupported` instead of being silently downgraded.
  There is no connection pool either — commands are serialized behind one connection mutex, so a single store is a single logical connection.
- Chroma stores one embedding per record, so a schema with more than one `FloatVector` field is rejected, and JSON/array metadata is rejected because Chroma metadata is scalar-only.
  The `CollectionSchema` is recovered only from the reserved `schema_metadata_key` (`openjiuwen:schema`); a collection created elsewhere yields `Unsupported`.
  Python writes the same information under the metadata keys `schema`/`fields` (plus `field_mapping`, `vector_field`, `distance_metric` — `foundation/store/vector/chroma_vector_store.py:130`), so collections are not interchangeable between the two implementations: neither reads the other's key, and Python falls back to `field_mapping` while this store returns `Unsupported`.
- S3 cannot distinguish a missing bucket from a missing key on `GET`, `HEAD` and `DELETE`, so both read as "absent"; `list_buckets` lists what the credentials can see, not this store's buckets.
- Deferred backends: `pgvector`, `postgres` and `gauss`/`gaussvector` parse as names but construction returns `Unsupported` (no PostgreSQL or `GaussDB` driver offline).
  No external graph backend exists at all: the `Graph` family registers only `memory`, `file`/`local`.

## Test Strategy

- Every HTTP adapter is tested with `wiremock` on a loopback mock server: `chroma/tests.rs`, `milvus/tests.rs`, `elasticsearch/document/tests.rs`, `elasticsearch/vector/tests.rs` and `s3/tests.rs`.
  Each test mounts exactly the requests the call may make and asserts paths, methods and bodies, so an unexpected request fails the test.
- The shared Elasticsearch client, config, mapping and schema round-trip helpers are covered by the inline tests in `elasticsearch/mod.rs`.
- The Redis backend (`redis/tests.rs`) runs against a scripted `tokio::net::TcpListener` that decodes commands with this module's own `Decoder` and replies with hand-built RESP frames, including the reconnect and retry paths; nothing leaves the loopback interface.
- The S3 signer is pinned by known-answer vectors in `s3/tests/sigv4_vectors.rs`: the AWS `aws-sig-v4-test-suite/get-vanilla` vector, the RFC 4231 HMAC-SHA256 vectors, and the S3 documentation `GET /test.txt` example, all at a fixed clock.
- The registry is tested on isolated `StoreRegistry` instances and on the global one (`registry/tests.rs`), covering aliases, family mismatches, plugin precedence and the `Unsupported` deferred names.
- No test needs a running service, the network, or credentials; the whole suite runs under `cargo test -p openjiuwen-foundation`.

## Routing Table

- `./chroma.rs` → Chroma v1 REST client: collections, add, query, get, delete, filter translation, distance normalization
- `./milvus.rs` → Milvus v2 REST client: collections create/describe/list/has/drop, insert, search, delete
- `./elasticsearch/` → Elasticsearch config/client, filter→query-DSL translation, vector store, document store (`elasticsearch/CONTEXT.md`)
- `./redis/` → RESP codec (`codec.rs`), connection client (`client.rs`), `KvStore` (`store.rs`), `RedisConfig` (`mod.rs`)
- `./s3/` → `S3Config` (`mod.rs`), Signature V4 signer (`sigv4.rs`), `ObjectStore` (`store.rs`)
- `./registry.rs` → `BackendKind`/`StoreSpec`/`StoreFamily`, `create_*_store`, plugin registration, `StoreRegistry`
