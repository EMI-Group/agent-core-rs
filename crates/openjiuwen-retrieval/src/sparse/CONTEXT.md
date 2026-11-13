# sparse — lexical retrieval and BM25

## Intent
Provide the lexical retrieval channel Python delegates to its vector databases: an additive sparse-vector value and an in-process BM25 index.
Python ships no BM25 implementation of its own, so the scoring math and the index are new Rust capabilities while the retriever/result surface stays Python-shaped.

## API Surface
### `mod.rs`
- `SparseVector` serializes as a sorted `{"term": weight}` object and is the additive sparse representation the stores lack.
- `SparseVector` API: `new`, `from_weights(iter)` (duplicates summed, zero weights dropped, entries sorted by term), `entries`, `get`, `dot`, `l2_norm`, `top_terms(n)` (weight desc then term asc), `len`, `is_empty`, `Default`, `PartialEq`.

### `bm25.rs`
- Constants `DEFAULT_STOPWORDS` (small English list), `DEFAULT_K1 = 1.2`, `DEFAULT_B = 0.75`.
- `Bm25Config { k1, b }` with `new(k1, b)`, `Default`, and `validate()` returning `Error::Validation` for non-finite/negative `k1` or `b` outside `[0, 1]`.
- `Bm25Query { text, top_k: 5, filter, score_threshold }` with `new(text)`, `with_top_k`, `with_filter`, `with_score_threshold`.
- `Bm25Statistics { document_count, term_count, average_document_length, total_tokens }` (serde, `Default`, `Copy`).
- `Bm25Index`: `new(Bm25Config) -> Result<Self>`, `Default` (default config plus default stopwords), `config`, `with_stopwords`, `without_stopwords`, `add_document(id, text, metadata)`, `add_documents(&[Document]) -> usize`, `add_chunks(&[TextChunk]) -> usize` (records the parent `doc_id`), `upsert(id, text, metadata) -> bool` (true when the id was new), `remove(id) -> bool`, `contains`, `len`, `is_empty`, `clear`, `stats`, `idf(term)`, `search(&Bm25Query) -> Vec<SearchResult>`.
- `tokenize(text)` folds case, splits every separator, and emits CJK ideographs/kana as individual tokens; the index (not `tokenize`) applies the stopword list.

## Scoring
- `idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`; per-term contribution is `idf · tf·(k1 + 1) / (tf + k1·(1 - b + b·|d|/avgdl))`.
- `k1 = 0` removes term-frequency saturation; `b = 0` disables document-length normalisation.
- Query terms are deduplicated through a `BTreeSet`, so summation order — and therefore every bit of the score — is deterministic.
- Documents matching no query term are dropped; hits are sorted by descending score (stable, so insertion order breaks ties) and truncated to `top_k`.
- `doc_id`/`chunk_id` pseudo-fields are injected into result metadata and evaluated by `MetadataFilter`, mirroring the vector stores.
- `score_threshold` is inclusive; `top_k = 0`, an empty index, an empty query, or a stopword-only query returns no hits.
- Scores are raw, finite, non-negative BM25 values comparable only within one index.

## Constraints
- Pure CPU and synchronous: no I/O, no randomness, no panics on user input.
- `Bm25Index::new` validates the configuration; `Default` uses the validated default constants.
- Replacing an id keeps its insertion position; `remove` retracts token and document-frequency accounting.
- Stopword builders only affect documents indexed afterwards.
- Every public type is serde snake_case with defaults for missing fields.

## Test Strategy
- Unit tests beside the modules cover `SparseVector` math and serde, tokenization (case, punctuation, CJK), config validation, statistics, replacement, and removal.
- `tests/sparse.rs` covers hand-computed BM25 scores, idf effect, k1/b effects, top-k truncation, filters, thresholds, insertion-order tie-breaks, upsert/remove, empty index, unknown terms, and the sparse retriever.

## Routing Table
- `./mod.rs` → `SparseVector` and module re-exports
- `./bm25.rs` → tokenizer, configuration, query, statistics, and the `Bm25Index`
