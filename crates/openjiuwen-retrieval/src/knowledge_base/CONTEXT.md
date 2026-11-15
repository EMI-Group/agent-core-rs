# knowledge_base — ingestion and retrieval pipeline

## Intent

Port `SimpleKnowledgeBase` behavior (Python `retrieval/simple_knowledge_base.py`) as the crate's composition layer: add/update/delete documents, parse files/URLs, retrieve through a selectable channel (dense, BM25, hybrid), optional graph expansion, optional agentic rewriting, rerank, statistics, and multi-KB retrieval.
The pipeline is load → chunk → embed → upsert → (sparse index + triple index) → retrieve → (rerank).

## API Surface

### `mod.rs`
- `SimpleKnowledgeBase` with private fields: `config`, `store: Arc<dyn VectorStore>`, `embedder: Arc<dyn Embedding>`, `chunker: Arc<dyn Chunker>`, `loader: Arc<dyn DocumentLoader>` (default `AutoLoader`), `reranker: Option<Arc<dyn Reranker>>`, `document_ids: RwLock<IndexSet<String>>`, `sparse: Option<Arc<RwLock<Bm25Index>>>`, `rewriter: Option<Arc<dyn QueryRewriter>>`, `agentic: bool`, `agentic_iterations: usize`, `graph_store: Option<Arc<dyn GraphStore>>`, `graph_hops: usize`.
- Constructors/configuration: `new(config, store, embedder, chunker)`, `from_config(config, store, embedder)` (validates the config, then builds a `CharChunker` from the config sizes), `from_index_config(&IndexConfig, store, embedder)` (validates the index config, maps `index_name` → `kb_id` and carries `index_type`/`use_caption_for_images`), `with_reranker`, `with_loader`, `with_sparse_index(bool)`, `with_rewriter`, `with_agentic(bool)`, `with_agentic_iterations(usize)`, `with_graph_store`, `with_graph_hops`, `id()`, `config()`, `graph_store()`, `agentic_iterations()`, `has_sparse_index()`, `sparse_index_len()`.
- Sparse index: enabled at construction exactly when `config.index_type == "bm25"`; `hybrid`/`vector` start dense-only and opt in through `with_sparse_index(true)` (documented deviation from Python, which fuses by default, so existing dense results stay byte-identical). `with_sparse_index` is synchronous, so enabling it after ingestion only indexes later chunks.
- Document operations:
  - `add_documents(&[Document]) -> Result<Vec<String>>`: enforce the image embedding policy, fill blank ids with uuid4, chunk documents, embed chunk texts in batches (`embed_documents(texts, None)`), build `VectorRecord`s (`id`/`chunk_id` = chunk id, `doc_id` = document id, metadata = chunk metadata), `store.upsert`, then `index_chunks` (BM25 `add_chunks` plus `RuleTripleExtractor` triples into an attached graph store), record document ids, return ids in input order.
  - Image policy (`use_caption_for_images`): an image document (`modality = "image"` or non-empty `image_path`) is indexed from its caption for text-only embedders and when the flag is `true`; the flag `false` with an embedder whose `supports_multimodal()` is `true` is a typed `Error::Unsupported` (image+text embedding is not implemented) returned before anything is chunked or stored.
  - `add_texts(&[String])`: wrap in `Document::new`, delegate.
  - `delete_documents(&[String]) -> Result<usize>`: list the store for the chunk ids of those documents (only when a sparse index exists), `store.delete_by_filter(MetadataFilter::In { field: "doc_id", values })`, drop those chunk ids from the BM25 index, remove ids from the tracked set; graph triples are not purged (documented).
  - `update_documents(&[Document])`: delete existing ids, then add; returns re-added ids and re-indexes both auxiliary indexes.
- Retrieval:
  - `retrieve(query)` = `retrieve_with_config(query, &RetrievalConfig::default())`.
  - `retrieve_with_config(query, config)` delegates to `retrieve::retrieve_with_config` (see below); validates a non-empty query and `top_k > 0` first.
- Ingestion helpers:
  - `parse_files(&[String])`: load each path; per-source error logs `warn` and skips (Python parity).
  - `parse_urls(&[String])`: same, but skips non-URL sources with a warning.
- Introspection: `count()` (chunks in store), `get_statistics()` → `KnowledgeBaseStatistics { kb_id, index_type, chunk_count, document_count, has_reranker, has_loader, sparse_index_size, has_graph, has_rewriter, agentic_enabled }`; the four channel fields are serde-defaulted so older JSON still round-trips.
- Free functions `retrieve_multi_kb(kbs, query, top_k)` and `retrieve_multi_kb_with_source(kbs, query, top_k)` (in `multi.rs`): run retrievals concurrently with `futures::future::join_all`, tolerate per-KB failure (empty), dedupe by text keeping max score and merging `kb_ids`, sort descending, limit to `top_k` (default 5).
- Constants: `DEFAULT_AGENTIC_ITERATIONS` (2, `AgenticRetriever`'s default), `DEFAULT_GRAPH_HOPS` (2, `GraphRetriever`'s default).

### `retrieve.rs`
- `retrieve_with_config(kb, query, config)`: resolves the channel from `index_type` + sparse-index presence (`bm25`→`Sparse`, `hybrid`+index→`Hybrid`, otherwise `Dense`) and reports `Error::Config` for an unknown index type, wraps it in `GraphRetriever` when a graph store is attached and `config.use_graph.unwrap_or(kb.config.use_graph) || config.graph_expansion`, wraps that in `AgenticRetriever` when `config.agentic || kb.agentic`, runs `Retriever::retrieve` with `top_k`/`score_threshold`/`filters`, applies the reranker (converted through `SearchResult`), and truncates to `top_k`.
- Sparse-index degradation: `bm25`/`hybrid` with the sparse index explicitly disabled logs `warn` and falls back to dense (the empty index would return nothing); an unknown index type never degrades.
- Channel retrievers: `VectorRetriever` for dense, `SparseIndexRetriever` (private adapter reading the KB's live `RwLock<Bm25Index>`) for BM25, `HybridRetriever` (RRF, candidate factor 2) for hybrid.
- Graph: `GraphRetriever` with `kb.graph_hops`; triple lookups use `Sparse` for the BM25/hybrid channels and `Vector` for dense-only.
- Agentic: rewriter is the injected `Arc<dyn QueryRewriter>` or a `HeuristicQueryRewriter` configured with the document-frequency vocabulary of the KB's own chunks (`store.list`); a store failure logs `warn` and falls back to length-ranked keywords; `AgenticConfig::max_iter = kb.agentic_iterations` (0 is a `Validation` error at retrieval time).
- Graph without a store: `use_graph`/`graph_expansion` log `warn` and fall back to the plain channel.
- Hybrid `score_threshold` filters the dense channel only (BM25 scores live on another scale, matching `HybridRetriever`).

### `multi.rs`
- Compatible functions, unchanged: `pub async fn retrieve_multi_kb(kbs, query, top_k) -> Vec<RetrievalResult>` and `retrieve_multi_kb_with_source(...) -> Vec<MultiKBRetrievalResult>`; both fan out with `futures::future::join_all`, tolerate a per-KB failure (nothing contributed, `warn` logged), dedupe by text keeping the first metadata and the maximum score, sort `kb_ids` unique/ascending, sort descending and limit to `top_k` (default 5); `raw_score`/`raw_score_scaled` come from metadata when present (default = score).
- Failure reporting types: `MultiKbFailure { kb_id, error }` (the failure's `Display` text), `MultiKbReport { results: Vec<RetrievalResult>, failures }` and `MultiKbSourceReport { results: Vec<MultiKBRetrievalResult>, failures }` — all `Debug, Clone, PartialEq, Serialize, Deserialize`, the two reports also `Default`; `failures` follows the order of the supplied `kbs` slice and is empty on full success.
- Reporting variants: `retrieve_multi_kb_reporting(kbs, query, top_k) -> MultiKbReport` and `retrieve_multi_kb_with_source_reporting(...) -> MultiKbSourceReport` return exactly the compatible merge/sort/limit results plus every per-KB failure.
- Strict variants: `retrieve_multi_kb_strict(kbs, query, top_k) -> Result<Vec<RetrievalResult>>` and `retrieve_multi_kb_with_source_strict(...) -> Result<Vec<MultiKBRetrievalResult>>` return `Ok(vec![])` for an empty `kbs` slice, `Ok` with the surviving merged results on a partial failure, and only on a total failure `Err(Error::retrieval("every knowledge base failed: {kb_id}: {error}; …"))` (`ErrorKind::Retrieval`, one entry per knowledge base in `kbs` order joined with `"; "`).
- One private fan-out (`retrieve_all`) returns both the per-KB result sets and the collected failures, and keeps the single `warn!(caller, kb_id, error, "knowledge base retrieval failed; skipping its results")` log so every caller keeps today's visibility.

## Constraints

- Stage errors are wrapped with context (`"validating knowledge base configuration"`, `"validating index configuration"`, `"chunking documents"`, `"embedding chunks"`, `"upserting chunks"`, `"indexing graph triples"`, `"configuring agentic retrieval"`, `"retrieving from the knowledge base"`, `"reranking results"`).
- No I/O in this module beyond delegating to loader/embedder/store/graph store; async throughout; `parking_lot` locks are never held across `.await`.
- `top_k = 0` or empty query → `Error::Validation`.
- Files ≤900 lines (`mod.rs`, `retrieve.rs`, `multi.rs`); shared conversions live in the lowest module that needs them.

## Test Strategy

End-to-end with `InMemoryVectorStore` + `HashEmbedding`/`TableEmbedding` + `CharChunker` + `InMemoryGraphStore`: add documents → expected chunk count, ids in order; retrieve returns the matching document first with `doc_id`/`chunk_id`; `score_threshold` and metadata filter; reranker applied and truncated; delete removes chunks/count and sparse entries; update replaces; `from_config` honors chunk sizes; `parse_files` skips a missing file but returns the good one; `parse_urls` skips non-URLs; statistics values; multi-KB dedupe/max score/limit and per-KB failure tolerance; the reporting variants surface one `MultiKbFailure` per failing knowledge base in input order (with the underlying `boom` message) while their `results` match the compatible function; the strict variants return the surviving partial results, return `ErrorKind::Retrieval` naming every failed knowledge base only when all of them fail, and treat an empty slice as an empty report / `Ok(vec![])`.
`tests/knowledge_base.rs` also covers the config surface: `from_config` rejects an unknown `index_type` with `Error::Config`, `from_index_config` maps every `IndexConfig` field (bm25 sparse index, id, caption flag) and rejects blank names/unknown types, and `use_caption_for_images` runs the caption branch for text-only embedders and reports `Error::Unsupported` for a multimodal embedder with the flag off.
`tests/knowledge_base_channels.rs` covers the channel layer: BM25 lexical hits and index upkeep, hybrid RRF lifting a lexical match the dense channel misses, agentic recall improvement plus its iteration bound, graph triple indexing/expansion/fallback, per-channel filters and thresholds, reranking on the BM25/hybrid channels, and the statistics/serde fields.
Unit tests in `retrieve.rs` cover channel resolution (including case normalization and the unknown-type `Error::Config`), mode mapping, and the live sparse-index adapter.

## Routing Table

- `./mod.rs` → `SimpleKnowledgeBase`, `KnowledgeBaseStatistics`, ingestion/index upkeep, statistics
- `./retrieve.rs` → channel dispatch, sparse/graph/agentic composition, `SparseIndexRetriever`
- `./multi.rs` → multi-KB retrieval helpers
