# knowledge_base — ingestion and retrieval pipeline

## Intent

Port `SimpleKnowledgeBase` behavior (Python `retrieval/simple_knowledge_base.py`) as the crate's composition layer: add/update/delete documents, parse files/URLs, retrieve with filters/threshold, optional rerank, statistics, and multi-KB retrieval.
The pipeline is load → chunk → embed → upsert → search → (rerank).

## API Surface

### `mod.rs`
- `SimpleKnowledgeBase` with private fields: `config`, `store: Arc<dyn VectorStore>`, `embedder: Arc<dyn Embedding>`, `chunker: Arc<dyn Chunker>`, `loader: Arc<dyn DocumentLoader>` (default `AutoLoader`), `reranker: Option<Arc<dyn Reranker>>`, `document_ids: RwLock<IndexSet<String>>`.
- Constructors/configuration: `new(config, store, embedder, chunker)`, `from_config(config, store, embedder)` (CharChunker built from config sizes), `with_reranker`, `with_loader`, `id()`, `config()`.
- Document operations:
  - `add_documents(&[Document]) -> Result<Vec<String>>`: fill blank ids with uuid4, chunk documents, embed chunk texts in batches (`embed_documents(texts, None)`), build `VectorRecord`s (`id`/`chunk_id` = chunk id, `doc_id` = document id, metadata = chunk metadata), `store.upsert`, record document ids, return ids in input order.
  - `add_texts(&[String])`: wrap in `Document::new`, delegate.
  - `delete_documents(&[String]) -> Result<usize>`: `store.delete_by_filter(MetadataFilter::In { field: "doc_id", values })`, remove ids from the tracked set.
  - `update_documents(&[Document])`: delete existing ids, then add; returns re-added ids.
- Retrieval:
  - `retrieve(query)` = `retrieve_with_config(query, &RetrievalConfig::default())`.
  - `retrieve_with_config(query, config)`: validate non-empty query and `top_k > 0`; `embed_query`, `VectorQuery { top_k, filter: config.filters, score_threshold }`, `store.search`; optional `reranker.rerank(query, hits, Some(top_k))`; map to `RetrievalResult` (`chunk_id` = hit id, `doc_id` from metadata `doc_id`), truncate to `top_k`.
- Ingestion helpers:
  - `parse_files(&[String])`: load each path; per-source error logs `warn` and skips (Python parity).
  - `parse_urls(&[String])`: same, but skips non-URL sources with a warning.
- Introspection: `count()` (chunks in store), `get_statistics()` → `KnowledgeBaseStatistics { kb_id, index_type, chunk_count, document_count, has_reranker, has_loader }`.
- Free functions `retrieve_multi_kb(kbs, query, top_k)` and `retrieve_multi_kb_with_source(kbs, query, top_k)` (in `multi.rs`): run retrievals concurrently with `futures::future::join_all`, tolerate per-KB failure (empty), dedupe by text keeping max score and merging `kb_ids`, sort descending, limit to `top_k` (default 5).

### `multi.rs`
- `pub async fn retrieve_multi_kb(...) -> Vec<RetrievalResult>` and `..._with_source(...) -> Vec<MultiKBRetrievalResult>`; merge `raw_score`/`raw_score_scaled` from metadata when present (default = score), sort `kb_ids` unique.

## Constraints

- Stage errors are wrapped with context (`"chunking documents"`, `"embedding chunks"`, `"upserting chunks"`, `"searching vector store"`, `"reranking results"`).
- No I/O in this module beyond delegating to loader/embedder/store; async throughout.
- `top_k = 0` or empty query → `Error::Validation`.
- Concurrent access is safe: only `document_ids` is behind a lock and it is never held across `.await`.
- Files ≤500 lines (`mod.rs`, `multi.rs`); shared conversions in `mod.rs`.

## Test Strategy

End-to-end with `InMemoryVectorStore` + `HashEmbedding` + `CharChunker`: add documents → expected chunk count, ids in order; retrieve returns the matching document first with `doc_id`/`chunk_id`; `score_threshold` and metadata filter; reranker applied and truncated; delete removes chunks/count; update replaces; `from_config` honors chunk sizes; `parse_files` skips a missing file but returns the good one; `parse_urls` skips non-URLs; statistics values; multi-KB dedupe/max score/limit and per-KB failure tolerance.

## Routing Table

- `./mod.rs` → `SimpleKnowledgeBase`, `KnowledgeBaseStatistics`, pipeline orchestration
- `./multi.rs` → multi-KB retrieval helpers
