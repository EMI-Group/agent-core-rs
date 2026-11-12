# retriever — unified retriever, sparse retrieval, rank fusion

## Intent
Port Python's `openjiuwen/core/retrieval/retriever/` base and `utils/fusion.py` to Rust, plumbing filters that Python silently drops.
Only the sparse (BM25) channel is implemented here; vector, hybrid, agentic, and graph retrievers are deferred to other work.

## API Surface
### `mod.rs`
- `RetrieveMode { Vector, Sparse, Hybrid }` — serde snake_case, `Default = Hybrid`, `as_str()` returning the Python wire names.
- `#[async_trait] Retriever` with `retrieve(query, top_k, score_threshold, mode, filters)` and `retrieve_search_results(query, top_k, mode, filters)`.
- `Retriever::batch_retrieve(queries, top_k, mode)` defaults to sequential retrieval and preserves query order; `Retriever::close` defaults to a no-op.
- `search_result_to_retrieval(SearchResult) -> RetrievalResult` — `chunk_id` is the hit id, `doc_id` comes from the metadata `doc_id` field.
- Re-exports `SparseRetriever`, `rrf_fusion`, `rrf_fusion_default`, `deduplicate`, `DEFAULT_RRF_K`.

### `sparse.rs`
- `SparseRetriever { index: Arc<Bm25Index> }` with `new(index)` and `index()` returning the shared index.
- Only `RetrieveMode::Sparse` is accepted; `Vector` and `Hybrid` return `Error::Validation` with Python's message `SparseRetriever only supports 'sparse' mode, got <mode>`.
- `retrieve` applies the score threshold through the index, `retrieve_search_results` returns raw hits, and `batch_retrieve`/`close` use the trait defaults.

### `fusion.rs`
- `rrf_fusion(results_list, k)` keys results by text, uses 1-based ranks, scores `Σ 1/(k + rank)`, keeps the first-seen object, and sorts by fused score descending with stable ties.
- `rrf_fusion_default(results_list)` uses `DEFAULT_RRF_K = 60`; `k = 0` degenerates to `1 / rank`.
- `deduplicate(results)` removes duplicate texts, keeping the first-seen result at its original position with the maximum score seen.

## Constraints
- Async through `async_trait`; no blocking work, no background tasks, and `Retriever` stays object-safe (`Send + Sync`).
- Filters reach the underlying index — a documented fix over Python's base, which accepts and drops them.
- No panics on user input: unsupported modes and invalid arguments surface as typed errors.
- Deterministic output: score ordering uses stable sorts and insertion-order tie-breaks.

## Test Strategy
- Unit tests beside the modules cover mode serde/defaults, result projection, mode rejection, filter/threshold plumbing, batch ordering, RRF scores, and deduplication.
- `tests/sparse.rs` exercises the public retriever surface end to end together with the BM25 index and fusion helpers; everything runs offline.

## Routing Table
- `./mod.rs` → `RetrieveMode`, the `Retriever` trait, and result projection
- `./sparse.rs` → `SparseRetriever`
- `./fusion.rs` → reciprocal rank fusion and deduplication
