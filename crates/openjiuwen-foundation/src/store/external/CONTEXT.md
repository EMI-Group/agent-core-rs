# external — network-backed storage backends

## Intent

Optional adapters that back the five storage contracts of the parent module with external services: Chroma, Milvus and Elasticsearch vector stores, an Elasticsearch document store, a Redis key-value store, an S3-compatible object store, and a name-driven registry that builds any of them from JSON configuration.
Every adapter speaks the service's own wire protocol directly — REST through `openjiuwen_common::http::HttpClient`, RESP2 through `tokio::net::TcpStream` — because no vendor SDK can be added to an offline build.
The in-memory and local-filesystem backends in the parent modules remain the behavioral reference: an adapter matches their semantics wherever the service can express them, and documents the difference on the method that cannot.

## API Surface

Each module exposes a `*Config` struct, a store type implementing the parent trait, and a re-export at this module's root.

- `chroma` — `ChromaConfig`, `ChromaVectorStore` (`VectorStore`).
- `milvus` — `MilvusConfig`, `MilvusVectorStore` (`VectorStore`).
- `elasticsearch` — `ElasticsearchConfig`, `ElasticsearchVectorStore` (`VectorStore`), `ElasticsearchDocumentStore` (`DocumentStore`).
- `redis` — `RedisConfig`, `RedisKvStore` (`KvStore`); `redis::codec` (`RespValue`, `encode_command`, `Decoder`); `redis::client` (`RedisConnection`, `RedisServerError`).
- `s3` — `S3Config`, `S3ObjectStore` (`ObjectStore`); `s3::sigv4` (`Credentials`, `SigningInput`, `SigningResult`, `sign_request`).
- `registry` — `BackendKind`, `StoreSpec`, `create_kv_store`, `create_document_store`, `create_vector_store`, `create_object_store`, `create_graph_store`, `StoreRegistry`, `register_*_backend`.

## Configuration Convention

Every `*Config` is `serde::Deserialize` with `#[serde(default)]` plus a `Default` impl, so a backend can be built straight from a JSON object (`serde_json::from_value`), and exposes named builder setters for tests.
Secrets use `openjiuwen_common::security::SecretString` (serializes transparently, redacts `Debug`).
Each config carries `timeout: Duration` (default 30 s HTTP, 5 s Redis) and, for the vector adapters, `metric: Metric` defaulting to `Cosine` because `CollectionSchema` has no metric member.
The metric drives both the native distance chosen at collection creation and the score normalization applied to search hits.

## Constraints

- No new crate dependency: only the workspace crates already listed in `crates/openjiuwen-foundation/Cargo.toml` — `reqwest` (via `openjiuwen_common::http`), `tokio`, `serde`, `serde_json`, `bytes`, `base64`, `sha2`, `chrono`, `url`, `tracing`, `async-trait`, `parking_lot`, `uuid`.
- All HTTP goes through `openjiuwen_common::http::HttpClient`; the Redis adapter uses a raw `tokio::net::TcpStream` because RESP is not HTTP. No direct `reqwest` calls.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for an unrepresentable schema, field or collection name, `Config` for bad configuration, `NotFound` where the parent trait requires it, `Unsupported` for a capability the service cannot express, and `External`/`Transport`/`Timeout` for service and socket failures (the message names the status code, the URL and a body excerpt).
- No panics on user input; no `unwrap`/`expect` outside `#[cfg(test)]`; no `unsafe`; `tracing` only; every public item documented (`cargo clippy -- -D warnings`).
- Files stay around ≤1000 lines: split by responsibility and move large test trees into `<module>/tests.rs` (`#[cfg(test)] mod tests;`).
- The adapters are not feature-gated: `reqwest`, `tokio` and the rest are already default dependencies of this crate, so always-compiled modules keep the build matrix small.
- Tests are offline: `wiremock` for the HTTP services, a scripted `tokio::net::TcpListener` for Redis, and the AWS SigV4 known-answer vectors for S3 signing.

## Design Decisions

- **Score normalization lives in the adapter.** Every `search` returns scores in `[0, 1]` with higher = more similar, matching `Metric::score`: cosine distance `d` → `1 - d/2` (the same value as `(cosine + 1) / 2`), dot/IP similarity `s` → `(s + 1) / 2` clamped to `[0, 1]`, L2 distance `d` → `1 / (1 + d)` for Chroma (whose distance is squared L2) and `max(0, (4 - d) / 4)` for Milvus; Elasticsearch `_score` is already a similarity in `[0, 1]` for cosine/dot and `1 / (1 + d)` for `l2_norm`, so it is clamped as-is.
- **Insert-only semantics are preserved where the service allows it.** Chroma `/add` and Elasticsearch `create` bulk actions reject a duplicate id (mapped to `Validation`); Milvus insert cannot detect a duplicate cheaply, so that limitation is documented on `MilvusVectorStore::add_docs`.
- **Schema round-trip.** Chroma stores the serialized `CollectionSchema` inside the collection metadata under one reserved key; Milvus reconstructs the schema from `describe`; Elasticsearch reconstructs it from index mappings plus a synthesized `id` keyword primary field.
- **Redis value encoding.** `KvValue::Text` is stored as raw UTF-8; `KvValue::Bytes` is stored as the ASCII marker `\0ojb:` followed by standard base64, so a text value can never collide with a byte value. A value that is not valid UTF-8 and lacks the marker decodes as `Bytes` rather than being lossily replaced.
- **Redis reconnect rule.** One transparent reconnect and retry per command, and only for connect failures or read-only commands (`PING`, `GET`, `MGET`, `EXISTS`, `SCAN`); a mutating command is never retried once its bytes were written, because a lost reply would otherwise become a duplicate write (notably `SET ... NX`).
- **S3 Signature V4 is in-crate.** HMAC-SHA256 is built from `sha2`, and the hex/base64 helpers are hand-written (no `hex`/`hmac` crate). `sign_request` takes an explicit `now` so tests are deterministic, and only the `s3` service and the caller's region are signed.
- **Registry globals.** The plugin maps behind `register_*_backend` are the one documented mutable global of this subtree (`RwLock<HashMap<..>>` in a `once_cell::sync::Lazy`); `StoreRegistry` exposes the same API over per-instance maps for tests.

## Routing Table

- `./chroma.rs` → Chroma v1 REST client: collections, add, query, get, delete, filter translation, distance normalization
- `./milvus.rs` → Milvus v2 REST client: collections create/describe/list/has/drop, insert, search, delete
- `./elasticsearch/` → Elasticsearch config/client, filter→query-DSL translation, vector store, document store (`elasticsearch/CONTEXT.md`)
- `./redis/` → RESP codec (`codec.rs`), connection client (`client.rs`), `KvStore` (`store.rs`), `RedisConfig` (`mod.rs`)
- `./s3/` → `S3Config` (`mod.rs`), Signature V4 signer (`sigv4.rs`), `ObjectStore` (`store.rs`)
- `./registry.rs` → `BackendKind`/`StoreSpec`, `create_*_store`, plugin registration, `StoreRegistry`
