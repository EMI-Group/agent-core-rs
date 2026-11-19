# external — network-backed storage backends

## Intent

Adapters that back the five storage contracts of the parent module with external services: Chroma, Milvus and Elasticsearch vector stores, an Elasticsearch document store, a Redis key-value store, an S3-compatible object store, and a name-driven registry that builds any of them from JSON configuration.
Each adapter speaks the service's own wire protocol directly — REST through `openjiuwen_common::http::HttpClient`, RESP2 through a `tokio::net::TcpStream` (TLS-wrapped via `tokio-rustls` for `rediss://`) — because no vendor SDK can be added to an offline build.
The in-memory and local-filesystem backends in the parent modules remain the behavioral reference: an adapter matches their semantics wherever the service can express them, and documents the difference on the method that cannot.

## API Surface

Each backend module exposes a `*Config` struct, a store type implementing the parent trait, and a re-export at this module's root.

- `chroma` — `ChromaConfig` (`endpoint`, `tenant`, `database`, `api_key`, `metric`, `timeout`, `schema_metadata_key`), `ChromaVectorStore::{new(config), config()}` (`VectorStore`).
- `milvus` — `MilvusConfig` (`endpoint`, `token`, `db_name`, `metric`, `timeout`, `consistency_level`, `reject_duplicate_primary_keys`), `MilvusVectorStore::{new(config), config()}` (`VectorStore`).
- `elasticsearch` — `ElasticsearchConfig` (`endpoint`, `index_prefix`, `username`, `password`, `api_key`, `metric`, `timeout`), `ElasticsearchVectorStore::{new(config), with_client(http, config), config()}` (`VectorStore`), `ElasticsearchDocumentStore::{new(config), with_client(http, config), config()}` (`DocumentStore`), `query::translate(&Value) -> Option<Value>` (the filter → query-DSL compiler), and the private `mapping` module (`field_from_mapping`, `SUPPORTED_DTYPES`, `schema_from_index`, `create_index_body`).
- `redis` — `RedisConfig` (`url`, `addr`, `username`, `password`, `db`, `tls`, `timeout`, `key_prefix`) with `connect()` and `with_tls`; `RedisKvStore` (`KvStore`); `redis::codec` (`RespValue`, `encode_command`, `encode`, `Decoder`); `redis::client` (`RedisConnection` with `connect`/`connect_with_timeout`/`connect_tls`/`connect_tls_with_timeout`, `RedisServerError`, `DEFAULT_TIMEOUT`, `READ_ONLY_COMMANDS`).
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

- No new vendor dependency. Beyond the workspace crates already listed in `crates/openjiuwen-foundation/Cargo.toml` (`reqwest` via `openjiuwen_common::http`, `tokio`, `serde`, `serde_json`, `bytes`, `base64`, `sha2`, `chrono`, `url`, `tracing`, `async-trait`, `parking_lot`, `once_cell`, `uuid`), the Redis TLS leg uses `tokio-rustls`/`webpki-roots`, which are pinned inline in that manifest rather than in `[workspace.dependencies]`.
- All HTTP goes through `openjiuwen_common::http::HttpClient`; the Redis adapter uses a raw `tokio::net::TcpStream` (TLS-wrapped for `rediss://`) because RESP is not HTTP. No direct `reqwest` calls and no vendor SDK.
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
- **Insert-only semantics are preserved where the service allows it.** Chroma `/add` and Elasticsearch `create` bulk actions reject a duplicate id (mapped to `Validation`); Milvus has no unique constraint, so `add_docs` probes `entities/query` for the batch's primary keys first and fails with a `Validation` naming the collisions unless `reject_duplicate_primary_keys` is off.
- **Schema round-trip.** Chroma writes the serialized `CollectionSchema` in three metadata keys — Python's `schema` and `fields` (its `to_dict()` layout) and this crate's legacy `schema_metadata_key` (default `openjiuwen:schema`) — and reads `schema` → `fields` → legacy, so a collection written by either implementation is readable by the other; Milvus reconstructs the schema from `describe`; Elasticsearch stores it under `_meta.openjiuwen_schema` and otherwise rebuilds it from index mappings plus a synthesized `id` keyword primary field.
- **Redis value encoding.** `KvValue::Text` is stored as raw UTF-8; `KvValue::Bytes` is stored as the ASCII marker `\0ojb:` followed by standard base64, so a text value can never collide with a byte value. A value that is not valid UTF-8 and lacks the marker decodes as `Bytes` rather than being lossily replaced, so the encoding is lossless for every value this store writes and safe for foreign ones.
- **Redis reconnect rule.** At most one transparent reconnect and retry per command: a failure before or while writing is retried for every command (the server never saw a complete command), while a failure while reading the reply is retried only for the read-only commands in `READ_ONLY_COMMANDS` (`PING`, `GET`, `MGET`, `EXISTS`, `SCAN`).
  A mutating command is never retried once its bytes were written, because a lost reply would otherwise become a duplicate write (notably `SET ... NX`).
  Every reconnect replays the `AUTH`/`SELECT` handshake.
- **S3 Signature V4 is in-crate.** HMAC-SHA256 is built from `sha2`, and the hex/base64 helpers are hand-written (no `hex`/`hmac` crate). `sign_request` takes an explicit `now` (`SigningInput::now`) so tests are deterministic, and only the `s3` service and the caller's region are signed.
  Path segments are RFC 3986-encoded exactly once (S3 signs single-encoded paths), the canonical query sorts by key then value, and `x-amz-content-sha256` is emitted always but signed only when the caller lists it — which `S3ObjectStore` always does.
  Addressing is path-style by default (`{endpoint}/{bucket}/{key}`, what `MinIO` and most gateways expect) and virtual-host (`{scheme}://{bucket}.{host}/{key}`) when `path_style = false`.
  A 404 from `GET`/`HEAD`/`DELETE` is tolerated as "missing object" (`Ok(None)`, `false`, no-op) because S3 cannot distinguish a missing bucket from a missing key there; only `list` maps a 404 to `NotFound`.
- **Redis TLS is the URL scheme's job.** A `rediss://` URL (or `with_tls(true)` on an explicit `addr`) wraps the socket in a `tokio-rustls` client that trusts only `webpki_roots::TLS_SERVER_ROOTS`; `with_tls(false)` cannot undo a `rediss://` URL, so a TLS endpoint is never silently downgraded to cleartext.
  Custom CA bundles, certificate pinning, `insecure-skip-verify` and the platform trust store are deliberately unsupported; `external/redis/CONTEXT.md` records the handshake and SNI rules.
- **Registry globals.** The plugin maps behind `register_*_backend` are the one documented mutable global of this subtree (`RwLock<HashMap<..>>` inside a `once_cell::sync::Lazy`); `global_registry()` exposes it, and `StoreRegistry::new()` builds an isolated instance over per-instance maps for tests.
  A registered name wins over a built-in of the same spelling, and a backend of the wrong family is a `Validation` error listing the names that family accepts.

## Known Issues and Gaps

- Two files exceed the ~1000-line baseline: `chroma.rs` (~1123 lines) and `milvus.rs` (~1403).
  Each holds the complete wire mapping plus the client-side validation for its service and moves its test tree out into `<module>/tests.rs` (`chroma/tests.rs`, `milvus/tests.rs`).
  `elasticsearch/` is split by responsibility instead (`mod.rs` 692, `mapping.rs` 710, `document.rs` 668, `vector.rs` 816, `query.rs` 424), with test trees in `mapping.rs`, `vector/tests.rs` and `document/tests.rs`.
- Milvus `add_docs` probes `entities/query` for the batch's primary keys before inserting, so the batch is validated in full locally and a collision is a `Validation` naming the keys; with the probe disabled (`reject_duplicate_primary_keys = false`) an insert is the append the service offers.
- Elasticsearch document `query`/`count`/`delete` page with `search_after` (`SEARCH_PAGE_SIZE` = 500) and fail with `Unsupported` as soon as a read would need more than `MAX_SCAN_DOCUMENTS` (10 000, Elasticsearch's default `index.max_result_window`) documents; they never truncate.
  An untranslatable filter still works through a capped `match_all` scan after one `tracing::warn!`.
- Redis has no cluster/Sentinel support and no connection pool — commands are serialized behind one connection mutex, so a single store is a single logical connection.
  TLS is available for `rediss://` but trusts only the webpki roots (no custom CA, pinning, skip-verify, client certificates or platform trust store), and a `rediss://` URL can never be silently downgraded.
- Chroma stores one embedding per record, so a schema with more than one `FloatVector` field is rejected, and JSON/array metadata is rejected because Chroma metadata is scalar-only.
  The `CollectionSchema` is written under the Python `schema`/`fields` keys and the reserved `schema_metadata_key` (`openjiuwen:schema`) and read in that order, so a Python-written collection is readable; one carrying none of those keys is `Unsupported`.
  Python's extra metadata (`field_mapping`, `vector_field`, `distance_metric` — `foundation/store/vector/chroma_vector_store.py:130`) is not read.
- S3 cannot distinguish a missing bucket from a missing key on `GET`, `HEAD` and `DELETE`, so both read as "absent"; `list_buckets` lists what the credentials can see, not this store's buckets.
- Elasticsearch translates implicit equality on an array condition into a `terms` (any-of) query (`elasticsearch/query.rs` `equality_clause`), while the shared filter language requires exact array equality, and the translated path of `DocumentStore::query`/`count`/`delete` and `VectorStore::search`/`delete_docs_by_filters` does not re-check locally, so `{"tags": ["a","b"]}` over-matches and `_delete_by_query` can remove documents the reference filter rejects.
- Milvus `delete_docs_by_filters` reads at most `FILTER_DELETE_LIMIT` (16 384) primary keys and returns that partial count as a completed delete, unlike the Elasticsearch scan which turns the same situation into `Error::Unsupported`.
- Milvus `create_collection` sets `autoID` from the schema while `add_docs` always sends a primary key (generated for `auto_id` schemas), so a collection created through this adapter with `auto_id` contradicts the service rule that an auto-id collection must not receive a client-supplied primary key.
- Chroma `delete_docs_by_filters` accumulates every matched id in memory and sends them in one `/delete` request with no cap, and stops paging as soon as a page is shorter than `FILTER_DELETE_PAGE`, so a server-side page cap under-reports what was deleted.
- The Milvus duplicate-key probe treats a non-array `entities/query` reply as "no stored keys" and proceeds with the insert, silently bypassing the documented duplicate rejection.
- S3 `list` returns a partial listing as success when `IsTruncated` is set without a continuation token or after `MAX_PAGES` (1000) pages, and `create_bucket` accepts every HTTP 409 as idempotent success, including `BucketAlreadyExists` for a bucket owned by another account.
- Deferred backends: `pgvector`, `postgres` and `gauss`/`gaussvector` parse as names but construction returns `Unsupported` (no PostgreSQL or `GaussDB` driver offline).
  No external graph backend exists at all: the `Graph` family registers only `memory`, `file`/`local`.

## Test Strategy

- Every HTTP adapter is tested with `wiremock` on a loopback mock server: `chroma/tests.rs`, `milvus/tests.rs`, `elasticsearch/document/tests.rs`, `elasticsearch/vector/tests.rs` and `s3/tests.rs`.
  Each test mounts exactly the requests the call may make and asserts paths, methods and bodies, so an unexpected request fails the test.
- The shared Elasticsearch client and config are covered by the inline tests in `elasticsearch/mod.rs`; the schema ↔ mapping conversion and the exhaustive dtype table have their own test tree in `elasticsearch/mapping.rs`.
- The Redis backend (`redis/tests.rs`) runs against a scripted `tokio::net::TcpListener` that decodes commands with this module's own `Decoder` and replies with hand-built RESP frames, including the reconnect and retry paths; nothing leaves the loopback interface.
  A protocol-blind `RawProbe` server proves the TLS path without completing a handshake (the first bytes must be a TLS record and must contain no cleartext `AUTH`/`PING`/`SELECT`), and the URL/TLS resolution rules are asserted directly.
- The S3 signer is pinned by known-answer vectors in `s3/tests/sigv4_vectors.rs`: the AWS `aws-sig-v4-test-suite/get-vanilla` vector, the RFC 4231 HMAC-SHA256 vectors, and the S3 documentation `GET /test.txt` example, all at a fixed clock.
- The registry is tested on isolated `StoreRegistry` instances and on the global one (`registry/tests.rs`), covering aliases, family mismatches, plugin precedence and the `Unsupported` deferred names.
- No test needs a running service, the network, or credentials; the whole suite runs under `cargo test -p openjiuwen-foundation`.

## Routing Table

- `./chroma.rs` → Chroma v1 REST client: collections, add, query, get, delete, filter translation, distance normalization; the schema metadata read/write keys live in `chroma/schema_metadata.rs` (`chroma/CONTEXT.md`)
- `./milvus.rs` → Milvus v2 REST client: collections create/describe/list/has/drop, duplicate-key probe, insert, search, delete (`milvus/CONTEXT.md`)
- `./elasticsearch/` → Elasticsearch config/client, filter→query-DSL translation, vector store, document store (`elasticsearch/CONTEXT.md`)
- `./elasticsearch/mapping.rs` → schema ↔ `mappings.properties` conversion, the exhaustive dtype table (`SUPPORTED_DTYPES`), the `_meta.openjiuwen_schema` marker
- `./redis/` → RESP codec (`codec.rs`), connection client incl. the TLS transport (`client.rs`), `KvStore` (`store.rs`), `RedisConfig` (`mod.rs`) (`redis/CONTEXT.md`)
- `./s3/` → `S3Config` (`mod.rs`), Signature V4 signer (`sigv4.rs`), `ObjectStore` (`store.rs`)
- `./registry.rs` → `BackendKind`/`StoreSpec`/`StoreFamily`, `create_*_store`, plugin registration, `StoreRegistry`
