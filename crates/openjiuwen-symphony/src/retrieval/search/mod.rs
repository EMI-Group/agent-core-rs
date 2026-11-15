//! The query-time search runtime: foundation layer.
//!
//! Ports the bottom of Python `openjiuwen.symphony.retrieval.search.runtime` (v0.1.20) — the
//! value objects, the serializable traversal config, the field-weighted lexical scorer, the
//! compact code book, the score-truncation decision and (Rust-only) embedding-backed scoring.
//! The layers above it — subtree exposure and prompt rendering, the traversal protocols, and
//! the engine/retrievers/facade — are added by later waves on top of this surface.
//!
//! # What is here
//!
//! - [`flow`](crate::retrieval::search::flow) — the Python flow value objects:
//!   [`ChatMessage`](crate::retrieval::search::flow::ChatMessage),
//!   the Python-shaped [`SearchCursor`](crate::retrieval::search::flow::SearchCursor) (a
//!   node id, a depth, a branch path and the requested `top_k`),
//!   [`ExposedNode`](crate::retrieval::search::flow::ExposedNode),
//!   [`SelectableResolution`](crate::retrieval::search::flow::SelectableResolution),
//!   [`ExposedFragment`](crate::retrieval::search::flow::ExposedFragment),
//!   [`SelectableTarget`](crate::retrieval::search::flow::SelectableTarget),
//!   [`CurrentSubtree`](crate::retrieval::search::flow::CurrentSubtree),
//!   [`SelectionProtocol`](crate::retrieval::search::flow::SelectionProtocol),
//!   [`PromptBundle`](crate::retrieval::search::flow::PromptBundle),
//!   [`SelectionResult`](crate::retrieval::search::flow::SelectionResult),
//!   [`ChildSearchCursor`](crate::retrieval::search::flow::ChildSearchCursor),
//!   [`ExpansionPlan`](crate::retrieval::search::flow::ExpansionPlan) and
//!   [`NodeSearchResult`](crate::retrieval::search::flow::NodeSearchResult).
//! - [`config`](crate::retrieval::search::config) — the serializable engine config:
//!   [`ProgressiveSearchConfig`](crate::retrieval::search::config::ProgressiveSearchConfig),
//!   its disclosure, embedding and truncation sub-configs, and
//!   [`SelectionMode`](crate::retrieval::search::config::SelectionMode).
//! - [`lexical`](crate::retrieval::search::lexical) — the deterministic field-weighted BM25
//!   scorer ([`LexicalIndex`](crate::retrieval::search::lexical::LexicalIndex),
//!   [`LexicalDocument`](crate::retrieval::search::lexical::LexicalDocument),
//!   [`tokenize`](crate::retrieval::search::lexical::tokenize),
//!   [`compile_matcher`](crate::retrieval::search::lexical::compile_matcher)).
//! - [`embedding`](crate::retrieval::search::embedding) — Rust-only embedding-backed
//!   scoring ([`EmbeddingScorer`](crate::retrieval::search::embedding::EmbeddingScorer),
//!   [`blend_scores`](crate::retrieval::search::embedding::blend_scores)).
//! - [`codebooks`](crate::retrieval::search::codebooks) — the compact boundary code book
//!   ([`CompactCodebook`](crate::retrieval::search::codebooks::CompactCodebook)) and the
//!   prefix trie a constrained decoder needs
//!   ([`CodeTrie`](crate::retrieval::search::codebooks::CodeTrie)).
//! - [`truncation`](crate::retrieval::search::truncation) — the score-truncation decision
//!   ([`truncate_sorted_by_score`](crate::retrieval::search::truncation::truncate_sorted_by_score))
//!   plus the bounded char/item helpers the renderers use.
//!
//! # Two search cursors
//!
//! [`SearchCursor`](crate::retrieval::types::SearchCursor) in
//! [`types`](crate::retrieval::types) is the *build-side* resumable pre-order traversal plan.
//! The search runtime uses its own Python-shaped cursor,
//! [`flow::SearchCursor`](crate::retrieval::search::flow::SearchCursor), which names one node
//! and the path that reached it. The two are unrelated types; do not convert one into the
//! other.
//!
//! # Determinism
//!
//! Every ranking, code assignment and truncation decision is deterministic: documents and
//! hits are ordered by key, ties break on the key, no output depends on hash-map iteration
//! order, and nothing reads the wall clock. Lexical scoring needs no model and no network;
//! the embedding path is additive and reuses `openjiuwen-retrieval`'s
//! [`Embedding`](openjiuwen_retrieval::Embedding) trait.

pub mod codebooks;
pub mod config;
pub mod embedding;
pub mod flow;
pub mod lexical;
pub mod truncation;

pub use codebooks::{default_compact_boundary_codebook, CodeTrie, CompactCodebook, DEFAULT_COMPACT_CODEBOOK_SIZE};
pub use config::{DisclosureConfig, EmbeddingScoringConfig, ProgressiveSearchConfig, SelectionMode, DEFAULT_TOP_K};
pub use embedding::{blend_scores, cosine_similarity, EmbeddingScorer};
pub use flow::{
    ChatMessage, ChildSearchCursor, CurrentSubtree, ExpansionPlan, ExposedFragment, ExposedNode,
    NodeSearchResult, PromptBundle, SearchCursor, SelectableResolution, SelectableTarget,
    SelectionProtocol, SelectionResult, ABSTAIN_TOKEN,
};
pub use lexical::{
    compile_matcher, tokenize, LexicalDocument, LexicalHit, LexicalIndex, QueryMatcher, BM25_B,
    BM25_K1, FIELD_WEIGHTS, IDENTITY_BONUS, PHRASE_FIELD_WEIGHTS,
};
pub use truncation::{
    take_budget, truncate_chars, truncate_chars_with_ellipsis, truncate_sorted_by_score,
    ScoreTruncationConfig, ScoreTruncationDecision,
};

/// The default `top_k` of the progressive runtime (Python `ProgressiveRetrieverConfig.top_k`).
///
/// [`DisclosureConfig`](crate::retrieval::search::config::DisclosureConfig) users reach it
/// through [`ProgressiveSearchConfig`](crate::retrieval::search::config::ProgressiveSearchConfig);
/// the public facade in a later wave defaults to the larger
/// [`DEFAULT_TOP_K`](crate::retrieval::search::config::DEFAULT_TOP_K) instead.
pub const RUNTIME_TOP_K: usize = 5;
