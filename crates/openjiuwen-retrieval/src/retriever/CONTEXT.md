# retriever — unified retriever, five implementations, rank fusion

## Intent

Port Python's `openjiuwen/core/retrieval/retriever/` package, `utils/fusion.py`, and the Milvus ranker formulas behind `result_ranking.py` to Rust.
All five retrievers are implemented: dense vector, sparse BM25, hybrid, graph expansion, and agentic iteration.
The hybrid retriever can fuse with reciprocal rank fusion (the default, `k = 60`) or the configurable weighted/RRF rankers in `common::ranking`.
Filters reach the underlying index instead of being dropped, which is a documented fix over Python's base class.
Retrievers are offline by default: the agentic loop uses deterministic heuristics and only calls a model when one is supplied.

## API Surface

### `mod.rs`
- `RetrieveMode { Vector, Sparse, Hybrid }` — serde snake_case, `Default = Hybrid`, `as_str()` returning the Python wire names.
- `#[async_trait] Retriever` with `retrieve(query, top_k, score_threshold, mode, filters)` and `retrieve_search_results(query, top_k, mode, filters)`.
- `Retriever::batch_retrieve(queries, top_k, mode)` defaults to sequential retrieval and preserves query order; `Retriever::close` defaults to a no-op.
- `search_result_to_retrieval(SearchResult) -> RetrievalResult` — `chunk_id` is the hit id, `doc_id` comes from the metadata `doc_id` field; `retrieval_to_search` is its inverse.
- `ensure_query(query)` rejects blank queries with `Error::Validation` for every concrete retriever.
- Re-exports `VectorRetriever`, `SparseRetriever`, `HybridRetriever`, `GraphRetriever`, `AgenticRetriever`, `AgenticConfig`, `AgenticTrace`, `AgenticStopReason`, `rrf_fusion`, `rrf_fusion_default`, `rrf_fusion_weighted`, `rrf_fusion_channels`, `weighted_fusion`, `deduplicate`, `DEFAULT_RRF_K`.

### `vector.rs`
- `VectorRetriever { store: Arc<dyn VectorStore>, embedder: Arc<dyn Embedding> }` with `new(store, embedder)`, `store()`, `embedder()`.
- Serves only `RetrieveMode::Vector`; other modes return `Error::Validation` with `VectorRetriever only supports 'vector' mode, got <mode>`.
- Encodes with `embed_query` and searches `VectorQuery { vector, top_k, filter: filters.cloned(), score_threshold }`, projecting hits through `search_result_to_retrieval`.
- Blank query → `Error::Validation`; `top_k == 0` → empty result without touching the embedder (documented choice, mirrored by `HybridRetriever`/`GraphRetriever`; `AgenticRetriever` validates `top_k` like Python instead).

### `sparse.rs`
- `SparseRetriever { index: Arc<Bm25Index> }` with `new(index)` and `index()`.
- Only `RetrieveMode::Sparse` is accepted; other modes return `Error::Validation` with `SparseRetriever only supports 'sparse' mode, got <mode>`.
- `retrieve` applies the score threshold through the index, `retrieve_search_results` returns raw hits.

### `hybrid.rs`
- `HybridRetriever { dense, sparse, candidate_factor: usize /*2*/, rank_config: RankConfig /*Rrf k=60*/ }` with `new(dense, sparse)`, `with_candidate_factor`, `with_rrf_k`, `with_rank_config`, `from_parts(store, embedder, index)`, and accessors (`dense`, `sparse`, `candidate_factor`, `rank_config`, `rrf_k`).
- The default fusion is exactly as before: `rrf_fusion` with `DEFAULT_RRF_K = 60`; `RankConfig::default()` (`Rrf` with `k = 40`) is deliberately not used as the retriever default, so existing result ordering never changes.
- `Vector` → dense results, falling back to the sparse channel when dense returns nothing (Python parity, documented); `score_threshold` filters the dense channel.
- `Sparse` → sparse results with the threshold delegated to the sparse channel.
- `Hybrid` → over-fetch `top_k * candidate_factor` from both channels via `retrieve_search_results`, apply `score_threshold` to the dense channel only (BM25 scores live on another scale), fuse through `rank_config.apply(vec![name(empty), dense, sparse])`, truncate to `top_k`.
- `with_rank_config(RankConfig::Weighted(..))` switches the hybrid fusion to the weighted ranker; `with_rrf_k(k)` selects RRF with `k` (keeping RRF flags), and `rrf_k()` returns `DEFAULT_RRF_K` when the active config is weighted.
- `candidate_factor` is clamped to at least 1; `batch_retrieve` uses the trait default.

### `graph.rs`
- `GraphRetriever { chunk_retriever, graph_store, max_hops: usize /*2*/, expansion_limit: usize /*5*/, triple_mode: RetrieveMode /*Sparse*/ }` with `new`, `with_max_hops`, `with_expansion_limit` (0 disables expansion), `with_triple_mode`, and accessors.
- `retrieve_with_report` and `retrieve_search_results_with_report` return the hits alongside a `GraphExpansionReport`; the `Retriever` trait methods delegate to them and drop the report.
- Flow: chunk retrieval in the requested mode → entity candidates from the query plus the top `ENTITY_CHUNK_LIMIT` chunk texts via `RuleTripleExtractor::entities_in_text` → `neighbors(entity, max_hops)` triples deduplicated by normalized key and bounded to `MAX_TRIPLES` (32) → chunk query per triple text (bounded to `expansion_limit`) → `rrf_fusion(vec![expanded, chunks])` truncated to `top_k`.
- `GraphExpansionReport { entities, triple_count, expanded_queries, degradations }` with `is_degraded()`; `GraphDegradation { stage, subject, error }`; `GraphStage { EntityLookup, NeighborExpansion, ChunkQuery }` — all serde snake_case, and `EntityLookup` has no producer yet because the rule extractor is infallible.
- Real failures (`NeighborExpansion`, `ChunkQuery`) log at `warn!`, are recorded in `GraphExpansionReport::degradations`, and never fail the call: the walk continues with the remaining entities/triples and the chunk hits are kept.
- When the report holds degradations, every returned hit carries `metadata["graph_degraded"] = true` (`GRAPH_DEGRADED_FIELD`), applied after truncation on the exact vector returned, so a clean run never grows the key.
- Genuine emptiness — no entity candidates, no triples around the candidates, no new chunks, expansion disabled, or `top_k == 0` — stays `debug!` and is a clean run with empty `degradations`.
- A failing *initial* chunk retrieval still propagates as a typed error.

### `agentic.rs`
- `AgenticConfig { max_iter /*2*/, sufficiency_threshold /*0.6*/, passages_per_iteration /*5*/ }` with builders and `validate()` (`max_iter == 0`, `passages_per_iteration == 0`, and a threshold outside `[0, 1]` are `Error::Validation`; Python silently resets instead).
- `AgenticRetriever { retriever, rewriter, model: Option<Arc<dyn CompletionModel>>, extractor: Arc<dyn TripleExtractor>, config }` with `new(retriever, rewriter)`, `with_config(config) -> Result`, `with_model`, `with_extractor`, and accessors.
- `retrieve_with_trace(...) -> Result<(Vec<RetrievalResult>, AgenticTrace)>` runs ≤ `max_iter` retrieve → read triples → rewrite loops; `impl Retriever` delegates to it.
- Sufficiency: fraction of the original query's rewriter keywords that occur in the accumulated passage texts (`>= sufficiency_threshold` stops with `Sufficient`).
- Next query: `RewriteRequest::new(original)` with history = previous queries plus the `TripleMemory::triples_str()` lines; the model path uses `READ_PROMPT`/`REWRITE_PROMPT` and `extract_json_object` plus a lenient tuple parser, falling back to the rewriter/extractor on any failure.
- `AgenticTrace { iterations, queries, triples, stopped_reason }` with `AgenticStopReason { Sufficient, MaxIterations, EmptyRewrite, NoNewResults, ModelFailure }` (serde snake_case); `ModelFailure` wins when a model step degraded.
- `score_threshold` is applied to each iteration's channel scores before fusion, because fused scores are reciprocal ranks.

### `fusion.rs`
- `rrf_fusion(results_list, k)` keys results by text, uses 1-based ranks, scores `Σ 1/(k + rank)`, keeps the first-seen object, and sorts by fused score descending with stable ties.
- `rrf_fusion_default(results_list)` uses `DEFAULT_RRF_K = 60`; `deduplicate(results)` keeps the first-seen text at its position with the maximum score.
- `rrf_fusion_weighted(results_list, k, weights) -> Result<Vec<SearchResult>>` multiplies each channel's contribution by `weights[i]` (non-positive weight drops the channel) and rejects a weight/list length mismatch with `Error::Validation`.
- `rrf_fusion_channels(results_list, &RrfRankConfig)` derives the weight mask from `is_active()` and uses the config's `k`; it is the RRF path of `RankConfig::apply`.
- `weighted_fusion(lists, weights) -> Result<Vec<SearchResult>>` min-max normalises each channel's scores into `[0, 1]` (a constant channel normalises to `1.0`, an empty channel contributes nothing), multiplies by the weight, sums per text (first-seen object kept), and sorts descending with stable ties; mismatched lengths are `Error::Validation`.
- Milvus normalises distances server-side per metric (where smaller can be better); this crate's scores are similarities, so `weighted_fusion` normalises the scores directly and sorts descending — the documented deviation.
- `rrf_fusion` is implemented as `rrf_fusion_weighted` with all-ones weights, so the two formulas share one accumulation loop and keep identical behaviour.

## Constraints
- Async through `async_trait`; no blocking work, no background tasks, and `Retriever` stays object-safe (`Send + Sync`).
- No panics on user input: unsupported modes and invalid arguments surface as typed errors.
- Deterministic output: stable sorts, insertion-order tie-breaks, no wall-clock or random input.
- `Debug` is implemented manually where the wrapped trait objects are not `Debug`.
- Logging goes through `openjiuwen_common::logging`; failed graph steps log at `warn`, empty expansions at `debug`, and no graph step ever fails the call.

## Test Strategy
- Unit tests beside each module cover mode serde/defaults, result projection, mode rejection, thresholds, RRF scores, hybrid fusion/over-fetch/fallback, graph expansion and degradation, agentic config validation, trace serde, triple parsing, and sufficiency arithmetic.
- `tests/retrievers.rs` exercises all five retrievers end to end with `HashEmbedding`, `InMemoryVectorStore`, `Bm25Index`, and `InMemoryGraphStore`, plus scripted rewriters and `StaticCompletionModel`; everything runs offline.
- `tests/ranking_configs.rs` covers the ranking configs (defaults, serde wire shape, validation, zero-weight normalisation, RRF masks, `k = 40` vs `DEFAULT_RRF_K = 60`), both fusion formulas, and the hybrid retriever's default-RRF regression plus the weighted-vs-RRF ordering difference on a fixed-channel corpus.
- `tests/strictness_losses.rs` covers the graph half of that suite: a failing `neighbors` lookup and a failing triple chunk query are recorded as `NeighborExpansion`/`ChunkQuery` degradations, flag the hits with `GRAPH_DEGRADED_FIELD`, and still return the chunk hits.

## Routing Table
- `./mod.rs` → `RetrieveMode`, the `Retriever` trait, query validation, and result projection
- `./vector.rs` → `VectorRetriever`
- `./sparse.rs` → `SparseRetriever`
- `./hybrid.rs` → `HybridRetriever`
- `./graph.rs` → `GraphRetriever`, `GraphExpansionReport`, `GraphDegradation`, `GraphStage`, `GRAPH_DEGRADED_FIELD`
- `./agentic.rs` → `AgenticRetriever`, `AgenticConfig`, `AgenticTrace`, prompts
- `./fusion.rs` → reciprocal rank fusion, weighted fusion, channel-masked RRF, and deduplication
