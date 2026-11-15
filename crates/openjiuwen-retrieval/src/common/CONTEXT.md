# common — data model, results, configs, filters

## Intent

Shared, serializable value types for the whole crate: the Python `retrieval/common` models (plus the foundation `Document` model), result types, knowledge-base configs, and metadata filtering.
No I/O and no dependencies on other modules in this crate.

## API Surface

### `document.rs`
- `pub type Metadata = serde_json::Map<String, serde_json::Value>` (also re-exported at the module root).
- `Document { id, text, metadata }` with `id` serialized as `id_` (alias `id`), default generated uuid4; `Document::new`, `with_id`, `with_metadata`, `metadata_entry`.
- `TextChunk { id, text, doc_id, metadata, embedding }` with `id` serialized as `id_`; `TextChunk::from_document(doc, text, id)` copies `doc_id` and metadata; `id` of `None` or `""` generates a `uuid4`; `with_embedding`.
- Python parity: `Document` keys `id_/text/metadata`; `TextChunk` keys `id_/text/doc_id/metadata/embedding` (embedding serialized as null when absent).

### `result.rs`
- `SearchResult { id, text, score, metadata }` — `score` is required (no `Option`).
- `RetrievalResult { text, score, metadata, doc_id, chunk_id }` — optional ids serialize as null.
- `MultiKBRetrievalResult { text, score, raw_score, raw_score_scaled, kb_ids, metadata }` — all scores required.
- Result types implement `Default` where meaningful (`metadata`/`kb_ids` empty, `score` 0.0).

### `config.rs`
- `KnowledgeBaseConfig` with `kb_id`, `index_type` (default `"hybrid"`), `use_graph`, `chunk_size` 512, `chunk_overlap` 50, `use_caption_for_images`; `KnowledgeBaseConfig::new(kb_id)`, `validate()` (`Error::Config` for an unknown `index_type`); `use_caption_for_images` is honored by the knowledge base's image embedding policy.
- `parse_index_type(value) -> Result<&'static str>`: normalizes case/whitespace and maps `vector`/`bm25`/`hybrid`; anything else is `Error::Config` (never a silent dense fallback).
- `RetrievalConfig` with `top_k` 5, `score_threshold`, `use_graph`, `agentic`, `graph_expansion`, `filters: Option<MetadataFilter>`; `RetrievalConfig::with_top_k/with_score_threshold/with_filter`.
- `IndexConfig { index_name, index_type, use_caption_for_images }` with `validate()` (blank `index_name` or unknown `index_type` → `Error::Config`) and `to_knowledge_base_config()` (`index_name` → `kb_id`, plus `index_type`/`use_caption_for_images`; default chunk window).
- `VectorStoreConfig { store_provider, database_name, collection_name, distance_metric }` (default `"cosine"`); consumed by `InMemoryVectorStore::from_config`, which rejects a non-blank `database_name` and a non-`cosine` metric with `Error::Config`.
- `EmbeddingConfig { model_name, base_url, api_key, timeout, max_retries, max_batch_size, headers, dimension, verify_ssl }` with `new(model_name, base_url)` and builders; `api_key` is `openjiuwen_common::security::SecretString`; every field is read by `OpenAiEmbedding`.
- `RerankerConfig { api_key, api_base, model_name (alias model), timeout, temperature, top_p }` with `validate()` (non-blank `api_base`/`model_name`, `timeout > 0`, `temperature` in `[0, 2]`, `top_p` in `(0, 1]`) and `require_api_key()`; consumed by `reranker::DashScopeReranker`; `model_name` accepts Python's `model` alias on input and serializes as `model_name`.
- All configs default every field on deserialization and ignore unknown fields; `RetrievalConfig` implements `Default` with `top_k = 5`.
- `Duration` fields (`EmbeddingConfig::timeout`, `RerankerConfig::timeout`) serialize as seconds (integer when whole, float otherwise) and deserialization rejects negative or non-numeric values.
- Constants: `DEFAULT_TOP_K`, `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP`, `INDEX_TYPE_VECTOR/BM25/HYBRID`, `DEFAULT_EMBEDDING_TIMEOUT`, `DEFAULT_EMBEDDING_BATCH_SIZE`.

### `filter.rs`
- `MetadataFilter` enum: `Eq/Ne/Gt/Gte/Lt/Lte/In/Nin/Exists/Contains/And/Or/Not`.
- `MetadataFilter::matches(&Metadata) -> bool`; numeric comparisons for JSON numbers, lexicographic for strings, `In`/`Contains` over arrays/strings, `Exists` checks key presence.
- `Ne`/`Nin` satisfy a missing field; `And([])` is true and `Or([])` is false; mixed JSON types never match and never panic.
- `MetadataFilter::from_value(&Value) -> Result<Self>` is the public parser behind `Deserialize`; unknown `$` operators and wrong operator operands return `Error::Validation`.
- `Serialize`/`Deserialize`: objects without `$` keys are AND-ed equality; `{"field": value}`; `{"field": {"$gt": 5}}`; `{"$and": [...]}`, `{"$or": [...]}`, `{"$not": {...}}`; round-trips to a canonical shape (a mixed top-level object becomes one `$and`).
- Builders `eq`, `ne`, `in_values`, `exists`, `and`, `or`, `not`; `std::ops::Not` is implemented as an alias for `Not(Box<_>)`.

## Constraints

- All types `Debug + Clone + PartialEq + Serialize + Deserialize`, snake_case, tolerant of unknown fields, missing collections default to empty.
- No `unwrap`/`expect` outside tests; parse failures map to `openjiuwen_common::error::Error::Validation`.
- `MetadataFilter::matches` never panics on mixed JSON types (returns `false`).
- Keep files ≤400 lines; `document.rs`, `result.rs`, `config.rs`, `filter.rs`, `mod.rs`.

## Routing Table

- `./document.rs` → `Metadata`, `Document`, `TextChunk`
- `./result.rs` → `SearchResult`, `RetrievalResult`, `MultiKBRetrievalResult`
- `./config.rs` → knowledge-base/retrieval/index/vector-store/embedding/reranker configs, their validators, `parse_index_type`, and constants
- `./filter.rs` → `MetadataFilter` parsing, serialization, evaluation
- `./mod.rs` → module exports (`pub use ...`)
