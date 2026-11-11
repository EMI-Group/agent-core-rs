# recall

## Intent

Pure ranking logic for memory recall: tokenization and BM25 keyword scoring, cosine/recency/importance scoring, the documented blend, near-duplicate detection and merge, and final candidate ordering with token budgets.
No storage or async here, so formulas are unit-testable and shared by the engine and graph modules.

## API Surface

### `keyword.rs`
```rust
pub fn tokenize(text: &str) -> Vec<String>;
pub fn bm25_scores(query: &str, documents: &[&str], k1: f32, b: f32) -> Vec<f32>; // raw, same length as documents
pub fn normalized_bm25_scores(query: &str, documents: &[&str]) -> Vec<f32>;        // scaled by max to [0,1]
pub fn keyword_score(query: &str, document: &str) -> f32;                          // single-doc coverage score in [0,1]
pub fn jaccard(a: &str, b: &str) -> f32;                                           // token-set Jaccard in [0,1]
pub fn token_count(text: &str) -> usize;
```
Tokenization is deterministic: lowercase, split on non-alphanumeric, keep CJK runs as character bigrams (a single CJK char stays unigram), drop tokens of length 1 unless CJK, and deduplicate nothing (BM25 needs term frequencies). Defaults `k1 = 1.2`, `b = 0.75`. The corpus for BM25 is the candidate set of one query; `normalized_bm25_scores` divides by the maximum raw score (all zero when max is zero). `keyword_score` is a `[0,1]` fraction of query terms present in the document, used by graph/entity recall where no corpus statistics exist.

### `scoring.rs`
```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32;   // raw cosine in [-1,1]; 0 on mismatch/empty/non-finite
pub fn normalized_similarity(a: &[f32], b: &[f32]) -> f32; // (cosine + 1) / 2, clamped [0,1]; 0 for empty/mismatch
pub fn recency_score(timestamp: DateTime<Utc>, now: DateTime<Utc>, half_life_secs: u64) -> f32;
pub fn blend(similarity: f32, keyword: f32, importance: f32, recency: f32, config: &RankingConfig) -> f32;
pub fn weighted(name: &str, weight: f32, ...)? // not public; helper
pub fn trim_to_token_budget(hits: Vec<RecallHit>, budget: usize, counter: &dyn TokenCounter) -> Vec<RecallHit>;
```
Formula (must match the crate root `CONTEXT.md`):

```text
semantic = (w_sim * similarity + w_kw * keyword) / (w_sim + w_kw)
final    = (w_sem * semantic + w_imp * importance + w_rec * recency) / (w_sem + w_imp + w_rec)
recency  = 0.5 ^ (age_secs / half_life_secs), future timestamps clamp age to 0
```
- Every input is clamped to `[0,1]`; a zero group denominator is treated as `1` so the other term dominates and no NaN escapes.
- `trim_to_token_budget` keeps hits in score order while `used + cost <= budget`; `cost = ApproxTokenCounter::count_text(content)` (foundation); a budget of `0` yields an empty vector; `None` (handled by callers) means no trim. The first hit is kept only if it fits.

### `dedupe.rs`
```rust
pub struct Candidate { pub record: MemoryRecord, pub similarity: Option<f32> } // from semantic search or keyword scope scan

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateKind { Exact, Similar, SameId }

#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateMatch { pub index: usize, pub kind: DuplicateKind, pub similarity: f32 }

pub fn normalize_text(text: &str) -> String;               // lowercase, collapse whitespace, strip edge punctuation
pub fn find_duplicate(incoming: &MemoryRecord, candidates: &[Candidate], config: &DedupeConfig) -> Option<DuplicateMatch>;
pub fn merge_records(existing: &MemoryRecord, incoming: &MemoryRecord, now: DateTime<Utc>) -> MemoryRecord;
```
`find_duplicate` checks, in order: same `id`; normalized-text equality (`Exact`); then for contents at least `min_content_len` chars, normalized cosine `>= similarity_threshold` or token Jaccard `>= jaccard_threshold` (`Similar`). Only `candidate_top_k` highest-similarity candidates are considered when the caller supplies more (callers already limit). The first match in candidate order wins; ties resolve by higher similarity.
`merge_records` keeps the existing `id`, `scope`, and `timestamp`; sets `content` to the longer content (incoming on equal length); unions metadata (incoming wins per key); `importance = max`; `access_count = existing + incoming`; `source` prefers incoming; `updated_at = now`; `embedding = None` (the engine re-embeds the merged content); `kind = incoming.kind` when the incoming kind is not `Unknown`, else existing.

### `pipeline.rs`
```rust
pub struct ScoredCandidate { pub record: MemoryRecord, pub similarity: Option<f32>, pub used_keyword_fallback: bool }

pub fn rank(candidates: Vec<ScoredCandidate>, query: &str, now: DateTime<Utc>, config: &RankingConfig) -> Vec<RecallHit>;
```
Steps: collect candidate texts, compute normalized BM25 scores over the candidate corpus, compute `recency` from `record.timestamp`, use `record.importance`, then `blend`. A candidate with `similarity: None` contributes `similarity = 0` and relies on keyword (this is the keyword-only degradation path). Output is sorted by `score` descending; ties by `record.id` ascending. The returned `RecallHit` includes the exact component values. `rank` never filters by threshold — the engine applies `threshold` after ranking so `RecallHit` components stay inspectable.

## Constraints

- Pure synchronous functions; deterministic for identical inputs (no wall clock inside; `now` is a parameter).
- No `f32` NaN/inf may escape: clamp and guard all divisions.
- BM25 handles an empty query/corpus by returning zeros.
- Dedupe never mutates the input records.
- Doc comments must state the formula and units.

## Routing Table

- `./keyword.rs` → tokenizer, BM25, Jaccard
- `./scoring.rs` → cosine/normalized similarity, recency, blend, token-budget trim
- `./dedupe.rs` → duplicate detection and merge
- `./pipeline.rs` → candidate ranking into `RecallHit`
- `./mod.rs` → declarations and re-exports

## Test Strategy

- Unit tests: tokenizer (ASCII, punctuation, CJK bigrams), BM25 ordering and normalization, Jaccard bounds, cosine edge cases (mismatch/zero/opposite), recency half-life values (0, half-life, 2× half-life), blend weights (similarity-only, keyword-only, importance-only configurations), token-budget trimming, exact/similar duplicate detection, merge field rules.
- Behavior-port note in tests: `// ports tests/unit_tests/core/memory/manage/test_mem_update_checker.py (deterministic subset)`.
