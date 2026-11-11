# reranker — result rescoring

## Intent

Provide the Rust `Reranker` seam (Python's ABC returns `dict[str, float]`; Rust returns ordered `SearchResult`s) with a deterministic default and an LLM-backed adapter seam.
Python has no default/no-op reranker and does not wire rerankers into knowledge bases; this crate adds both for the offline pipeline.

## API Surface

### `mod.rs`
- `#[async_trait] pub trait Reranker: Send + Sync { async fn rerank(&self, query: &str, documents: Vec<SearchResult>, top_n: Option<usize>) -> Result<Vec<SearchResult>>; }`
- Re-exports `ScoreReranker`, `LlmReranker`, `LlmScorer`.

### `score.rs`
- `ScoreReranker { top_n: Option<usize> }`; `new(top_n)`, `Default` (no truncation).
- Sorts descending by existing `score` (stable), truncates to `top_n` when set; treats `top_n = Some(0)` as empty output; returns input unchanged for empty documents.

### `llm.rs`
- `#[async_trait] pub trait LlmScorer: Send + Sync { async fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>; }`.
- `LlmReranker { scorer: Arc<dyn LlmScorer>, top_n: Option<usize> }`; `new(scorer, top_n)`.
- Calls the scorer with document texts, requires exactly one finite score per document (`Error::Retrieval` otherwise), writes scores into results, sorts descending (stable), truncates.
- Empty documents → no scorer call, empty output.

## Constraints

- No dependency on `openjiuwen-foundation`; a `Model`-backed `LlmScorer` adapter belongs in an upper crate.
- Deterministic ordering; no randomness.
- No panics; invalid scorer output is a `Error::Retrieval` describing expected vs received counts.

## Test Strategy

ScoreReranker ordering/truncation/`top_n = 0`/empty; LlmReranker uses a fake scorer, verifies received query/doc texts and order, count mismatch error, empty input never calls the scorer.

## Routing Table

- `./mod.rs` → `Reranker` trait and re-exports
- `./score.rs` → `ScoreReranker` default implementation
- `./llm.rs` → `LlmScorer` seam and `LlmReranker`
