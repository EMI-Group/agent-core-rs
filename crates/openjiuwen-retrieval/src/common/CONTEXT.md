# common — data model, results, configs, filters

## Intent

Shared, serializable value types for the whole crate: the Python `retrieval/common` models (plus the foundation `Document` model), result types, knowledge-base configs, and metadata filtering.
No I/O and no dependencies on other modules in this crate.

## API Surface

### `document.rs`
- `pub type Metadata = serde_json::Map<String, serde_json::Value>` (also re-exported at the module root).
- `Document { id, text, metadata }` with `id` serialized as `id_` (alias `id`), default generated uuid4; `Document::new`, `with_id`, `with_metadata`, `metadata_entry`.
- `TextChunk { id, text, doc_id, metadata, embedding }` with `id` serialized as `id_`; `TextChunk::from_document(doc, text, id)` copies `doc_id` and metadata; `with_embedding`.
- Python parity: `Document` keys `id_/text/metadata`; `TextChunk` keys `id_/text/doc_id/metadata/embedding` (embedding serialized as null when absent).

### `result.rs`
- `SearchResult { id, text, score, metadata }` — `score` is required (no `Option`).
- `RetrievalResult { text, score, metadata, doc_id, chunk_id }` — optional ids serialize as null.
- `MultiKBRetrievalResult { text, score, raw_score, raw_score_scaled, kb_ids, metadata }` — all scores required.
- Result types implement `Default` where meaningful (`metadata`/`kb_ids` empty, `score` 0.0).

### `config.rs`
- `KnowledgeBaseConfig` with `kb_id`, `index_type` (default `"hybrid"`), `use_graph`, `chunk_size` 512, `chunk_overlap` 50, `use_caption_for_images`; `KnowledgeBaseConfig::new(kb_id)`.
- `RetrievalConfig` with `top_k` 5, `score_threshold`, `use_graph`, `agentic`, `graph_expansion`, `filters: Option<MetadataFilter>`; `RetrievalConfig::with_top_k/with_score_threshold/with_filter`.
- `IndexConfig { index_name, index_type, use_caption_for_images }`.
- `VectorStoreConfig { store_provider, database_name, collection_name, distance_metric }` (default `"cosine"`).
- `EmbeddingConfig { model_name, base_url, api_key, timeout, max_retries, max_batch_size, headers, dimension, verify_ssl }` with `new(model_name, base_url)` and builders; `api_key` is `openjiuwen_common::security::SecretString`.
- `RerankerConfig { api_key, api_base, model_name (alias model), timeout, temperature, top_p }`.
- Constants: `DEFAULT_TOP_K`, `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP`, `INDEX_TYPE_VECTOR/BM25/HYBRID`, `DEFAULT_EMBEDDING_TIMEOUT`, `DEFAULT_EMBEDDING_BATCH_SIZE`.

### `filter.rs`
- `MetadataFilter` enum: `Eq/Ne/Gt/Gte/Lt/Lte/In/Nin/Exists/Contains/And/Or/Not`.
- `MetadataFilter::matches(&Metadata) -> bool`; numeric comparisons for JSON numbers, lexicographic for strings, `In`/`Contains` over arrays/strings, `Exists` checks key presence.
- `Serialize`/`Deserialize`: objects without `$` keys are AND-ed equality; `{"field": value}`; `{"field": {"$gt": 5}}`; `{"$and": [...]}`, `{"$or": [...]}`, `{"$not": {...}}`; round-trips to a canonical shape.
- Builders `eq`, `ne`, `in_values`, `exists`, `and`, `or`, `not`.

## Constraints

- All types `Debug + Clone + PartialEq + Serialize + Deserialize`, snake_case, tolerant of unknown fields, missing collections default to empty.
- No `unwrap`/`expect` outside tests; parse failures map to `openjiuwen_common::error::Error::Validation`.
- `MetadataFilter::matches` never panics on mixed JSON types (returns `false`).
- Keep files ≤400 lines; `document.rs`, `result.rs`, `config.rs`, `filter.rs`, `mod.rs`.

## Routing Table

- `./document.rs` → `Metadata`, `Document`, `TextChunk`
- `./result.rs` → `SearchResult`, `RetrievalResult`, `MultiKBRetrievalResult`
- `./config.rs` → knowledge-base/retrieval/index/vector-store/embedding/reranker configs and constants
- `./filter.rs` → `MetadataFilter` parsing, serialization, evaluation
- `./mod.rs` → module exports (`pub use ...`)
