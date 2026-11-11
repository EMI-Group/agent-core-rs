//! Keyword/BM25, cosine/recency scoring, dedupe, and candidate ranking.
//!
//! Pure synchronous ranking logic shared by the engine, the graph, and the lite coding
//! memory: no storage, no async, and no wall clock (callers pass `now`), so every
//! formula is unit-testable and deterministic.
//!
//! - [`keyword`] — tokenizer, `BM25`, normalized `BM25`, coverage score, Jaccard.
//! - [`scoring`] — cosine/normalized similarity, recency decay, the documented blend,
//!   token-budget trimming.
//! - [`dedupe`] — near-duplicate detection and record merging
//!   (`// ports tests/unit_tests/core/memory/manage/test_mem_update_checker.py`).
//! - [`pipeline`] — candidate ranking into [`crate::record::RecallHit`].
//!
//! The scoring formula is fixed by the crate documentation:
//!
//! ```text
//! semantic = (w_sim * similarity + w_kw * keyword) / (w_sim + w_kw)
//! final    = (w_sem * semantic + w_imp * importance + w_rec * recency)
//!            / (w_sem + w_imp + w_rec)
//! recency  = 0.5 ^ (age_secs / half_life_secs)
//! ```
pub mod dedupe;
pub mod keyword;
pub mod pipeline;
pub mod scoring;
pub use dedupe::{find_duplicate, merge_records, Candidate, DuplicateKind, DuplicateMatch};
pub use keyword::{
    bm25_scores, jaccard, keyword_score, normalized_bm25_scores, token_count, tokenize, DEFAULT_B,
    DEFAULT_K1,
};
pub use pipeline::{rank, ScoredCandidate};
pub use scoring::{
    blend, cosine_similarity, normalized_similarity, recency_score, trim_to_token_budget,
};
