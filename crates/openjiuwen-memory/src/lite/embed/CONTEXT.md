# lite::embed
## Intent
The vector half of the lite coding memory: line-granular Markdown chunking, a content-hash keyed embedding cache, and a chunked embedding index with cosine similarity and the semantic/text blend.
`chunk` is a faithful port of Python `core/memory/lite/internal.py`'s `estimate_tokens` and `chunk_markdown`; `cache` and `index` are the Python lite embedding cache and vector search rewritten as idiomatic async Rust over `openjiuwen_retrieval::embedding::Embedding`.
The index is derived state beside `MEMORY.md`: it can be dropped and rebuilt from the Markdown document at any time, and no vector is ever written into the Markdown file.
Everything runs offline: the cache has an in-memory backend, `SQLite` is an optional durable backend behind the crate's `sqlite` feature, and every model call goes through a caller-supplied `Embedding` (tests use `HashEmbedding`).
## API Surface
- `crate::lite::embed::chunk_markdown(content: &str, config: &ChunkingConfig) -> Vec<MemChunk>` (line-granular, `#[must_use]`).
- `crate::lite::embed::estimate_tokens(text: &str) -> usize` — `text.chars().count() / 4`, Python's `len(text) // 4`.
- `crate::lite::embed::ChunkingConfig { target_tokens: usize, overlap: usize }` — `Default { 256, 32 }`, `Debug + Clone + Copy + PartialEq + Eq + Serialize + Deserialize`, serde `snake_case` + container `default`, `validate() -> Result<()>`.
- `crate::lite::embed::MemChunk { text: String, start_line: usize, end_line: usize }` — 1-based inclusive line range, `Debug + Clone + PartialEq + Eq + Serialize + Deserialize`, `line_count()`.
- `crate::lite::embed::content_hash(text: &str) -> String` — 16 lowercase hex characters of a 64-bit FNV-1a digest.
- `crate::lite::embed::CacheKey { model: String, hash: String }` — `Debug + Clone + PartialEq + Eq + PartialOrd + Ord + Hash + Serialize + Deserialize`, usable as a `BTreeMap`/`HashMap` key.
- `crate::lite::embed::EmbeddingCache` — `Clone` handle (`Arc`): `in_memory()`, `sqlite(SqlitePool)` (feature `sqlite`), `is_persistent()`, `get(model, hash)`, `put(model, hash, vector)`, `len()`, `is_empty()`, `clear()`.
- `crate::lite::embed::IndexedChunk { id: String, document: String, chunk: MemChunk, vector: Option<Vec<f32>> }` — `id` is `"<document>:<start_line>:<end_line>"` and `document` is the source key the chunk was cut from (a fact name, or `"<path>:<line>"` for a session record).
- `crate::lite::embed::EmbeddingStats { chunks: usize, embedded: usize, cached: usize, dimension: Option<usize> }` — what one `reindex` pass did.
- `crate::lite::embed::ScoredChunk { id: String, document: String, chunk: MemChunk, similarity: f32 }` — one recall hit with its `0.0..=1.0` similarity; `id` and `document` carry the same two keys as `IndexedChunk`.
- `crate::lite::embed::EmbeddingIndex` — `Clone`: `new(model, cache)`, `model()`, `len()`, `is_empty()`, `chunks()`, `clear()`, `reindex(&dyn Embedding, &[(String, String)], &ChunkingConfig)`, `search(&dyn Embedding, &str, limit)`.
- `crate::lite::embed::cosine_similarity(left: &[f32], right: &[f32]) -> f32` — `-1.0..=1.0`, `0.0` when not comparable.
- `crate::lite::embed::similarity_from_cosine(cosine: f32) -> f32` — `(cosine + 1) / 2` clamped to `0.0..=1.0`.
- `crate::lite::embed::blend(semantic: f32, keyword: f32, vector_weight: f32, text_weight: f32) -> f32` — the renormalized two-signal blend.
- `mod.rs` keeps the module documentation, declares `cache`/`chunk`/`index`, re-exports every item above, and holds `pub(super) const TARGET`.
## Constraints
- No new dependencies: the cache key hash is a locally implemented 64-bit FNV-1a formatted as 16 hex characters, standing in for Python's SHA-256 digest truncated to the same shape.
- The `sqlite` feature gates exactly `EmbeddingCache::sqlite`, the `lite_embedding_cache` table DDL, the vector blob codec, and the SQLite test module; every other path compiles and is tested without the feature.
- Cache key composition is `(embedding.model_name(), content_hash(chunk.text))`; `EmbeddingIndex::model()` is a label only and never a cache key part, and query vectors from `embed_query` are never cached.
- `reindex` takes `(document key, content)` pairs and keeps the key on every chunk it cuts: the local store passes a fact name for a fact and `"<path>:<line>"` for a session record, and `search` returns the same key on `ScoredChunk::document`, so a caller maps a hit back to its source without parsing the id.
- `chunk_markdown` never splits a line, an oversized line becomes its own chunk, the overlap is the longest suffix of the flushed chunk that fits the overlap budget, and `start_line` is recomputed as `line_number - overlap.len()`; empty content yields no chunks.
- `blend` clamps both scores to `0.0..=1.0`, treats non-finite or negative weights as `0.0`, renormalizes by the weight sum, returns `max(semantic, keyword)` when both weights are zero, and never returns `NaN`.
- `cosine_similarity` returns `0.0` for a dimension mismatch, an empty slice, a zero norm, or a non-finite result; `search` scores a chunk without a vector, or with one of another dimension, as `similarity = 0.0` instead of dropping it.
- `EmbeddingIndex::reindex` validates the chunking config first, embeds all cache misses in ONE `embed_documents(&texts, None)` call, writes each result back to the cache, replaces the whole chunk list only on success, and skips blank documents.
- `EmbeddingIndex::search` rejects a blank query with `Error::Validation`, answers an empty index without calling the model, and orders by similarity descending then id ascending before truncating to `limit.max(1)`.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Validation` for a bad config or blank query, `Memory` for a corrupt cache row or a model returning the wrong vector count, `External` (source preserved) for `sqlx` failures, and the model's own kind when embedding fails; nothing is swallowed.
- No `unsafe`, no `println!`/`eprintln!`; diagnostics use `tracing` with target `openjiuwen_memory::lite::embed`; no `unwrap`/`expect` outside tests.
- A `parking_lot` mutex only guards the in-memory map and is never held across an `.await`; the `SqlitePool` is cheaply cloned.
- `index.rs` is 895 lines because its 21 tests live in-file (the implementation half is about 390 lines); `cache.rs` stays under 700 and `chunk.rs` under 450.
## Routing Table
- `./mod.rs` → module documentation, submodule declarations, public re-exports, the tracing `TARGET`
- `./chunk.rs` → `ChunkingConfig`, `MemChunk`, `estimate_tokens`, `chunk_markdown`, the chunking algorithm and its 15 unit tests
- `./cache.rs` → `content_hash`, `CacheKey`, `EmbeddingCache` (in-memory + `sqlite` backends), the vector blob codec, 9 unit tests plus 7 `sqlite`-gated tests
- `./index.rs` → `IndexedChunk`, `EmbeddingStats`, `ScoredChunk`, `EmbeddingIndex`, `cosine_similarity`, `similarity_from_cosine`, `blend`, 21 unit tests
## Test Strategy
- 45 unit tests with default features (15 chunking, 9 cache, 21 index) and 52 with `--features sqlite`.
- Chunking tests pin `estimate_tokens`, the config default/serde/validation, empty and single-line input, whole-line slicing, oversized lines, overlap carry-over with exact `(start_line, end_line)` values for a multi-flush document, and zero-token lines.
- Cache tests pin the three FNV-1a vectors (`""`, `"a"`, `"hello"`), 16-lowercase-hex shape, stability and distinction, hit/miss/overwrite/clear/len, clone sharing, and key ordering/hashing.
- `sqlite` cache tests use `tempfile::tempdir()`, assert an exact vector round trip (`f32::MAX`/`MIN` included) and persistence across handles and pool reopens, and pin the typed errors for a corrupt blob, a negative dimension, and a closed pool.
- `index` tests use `HashEmbedding` wrapped in an in-test counting `Embedding` (query calls, document calls, batch sizes, scripted failure) and assert one batch for misses, ZERO model calls plus `embedded == 0`/`cached == n` on a second unchanged `reindex`, incremental re-embedding after an edit, chunk-id numbering, validation, failure rollback, search ordering/ties/limit/blank-query/empty-index behaviour, and the similarity/blend edge cases.
- Everything is deterministic and offline; gates are `cargo fmt -p openjiuwen-memory`, `cargo test -p openjiuwen-memory --offline --lib lite::embed` with and without `--features sqlite`, `cargo clippy -p openjiuwen-memory --offline --all-targets -- -D warnings` with and without the feature, and `cargo doc -p openjiuwen-memory --no-deps` warning-free.
