# embedding — vector representations

## Intent

Port the Python `retrieval/embedding/` abstraction and the OpenAI-compatible client, plus a deterministic fake for offline tests.
No external SDK: requests go through `openjiuwen_common::http`.

## API Surface

### `mod.rs`
- `#[async_trait] pub trait Embedding: Send + Sync`:
  - `fn model_name(&self) -> &str`
  - `fn dimension_hint(&self) -> Option<usize>` (default `None`)
  - `async fn embed_query(&self, text: &str) -> Result<Vec<f32>>`
  - `async fn embed_documents(&self, texts: &[String], batch_size: Option<usize>) -> Result<Vec<Vec<f32>>>`
  - default `async fn dimension(&self) -> Result<usize>` (hint or probe with `"test"`)
- `validate_texts`/`validate_query` helpers: empty list and blank entries → `Error::Validation`.

### `fake.rs`
- `HashEmbedding { dimension, model_name }`; `new(dimension)` (default 64), `Default`.
- Deterministic SHA-256 per whitespace token → dimension-sized signed vector, mean pooled over tokens, L2 normalised; empty text → zero vector.
- Same text always yields the same vector; identical vectors have cosine 1.0.

### `openai.rs`
- `OpenAiEmbedding { config, client, dimension: Mutex<Option<usize>> }`; `OpenAiEmbedding::new(EmbeddingConfig) -> Result<Self>`.
- Endpoint: `base_url` unchanged if it ends with `/embeddings`, otherwise trim trailing `/` and append `/embeddings`.
- Request body `{"model", "input": string|string[], "dimensions"?: n}`; `Authorization: Bearer` when `api_key` is set (never logged); extra headers from config.
- Batch size `batch_size.or(config.max_batch_size)` (capped by config `max_batch_size` when provided), requests sequential, result order preserved.
- Response parsing: `data[].embedding` sorted by `index` when all indices are present, then top-level `embedding` (single or list), then `embeddings`; count mismatch → `Error::Retrieval`.
- Failures: non-2xx → `Error::Retrieval` with status and up to 512 chars of body; transport/timeouts keep `Error::Transport`/`Error::Timeout`.
- On success, caches the vector length as `dimension_hint` (config `dimension` wins).

## Constraints

- The crate has no `openjiuwen-foundation` dependency; the `EmbeddingConfig` in `common::config` mirrors Python `EmbeddingConfig` fields and the client uses `openjiuwen_common::http`.
- No credentials in errors/logs; use `SecretString::expose_secret` only at header construction.
- Batching must preserve input order exactly.
- Unit tests for the OpenAI client use `wiremock`; no live network.

## Test Strategy

Hash embedding determinism/dimension/normalisation and empty text; OpenAI request body/headers/batching (wiremock matcher counts), response shape variants, index sorting, non-2xx and malformed payload errors, blank input validation, `dimension` caching.

## Routing Table

- `./mod.rs` → `Embedding` trait, validation helpers, re-exports
- `./fake.rs` → `HashEmbedding` deterministic fake
- `./openai.rs` → OpenAI-compatible embedding client
- `./openai/tests.rs` → wiremock tests for endpoint, request body, auth, batching, response shapes, and failures
