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
//! - [`flow`] — the Python flow value objects:
//!   [`ChatMessage`],
//!   the Python-shaped [`SearchCursor`] (a
//!   node id, a depth, a branch path and the requested `top_k`),
//!   [`ExposedNode`],
//!   [`SelectableResolution`],
//!   [`ExposedFragment`],
//!   [`SelectableTarget`],
//!   [`CurrentSubtree`],
//!   [`SelectionProtocol`],
//!   [`PromptBundle`],
//!   [`SelectionResult`],
//!   [`ChildSearchCursor`],
//!   [`ExpansionPlan`] and
//!   [`NodeSearchResult`].
//! - [`config`] — the serializable engine config:
//!   [`ProgressiveSearchConfig`],
//!   its disclosure, embedding and truncation sub-configs, and
//!   [`SelectionMode`].
//! - [`lexical`] — the deterministic field-weighted BM25
//!   scorer ([`LexicalIndex`],
//!   [`LexicalDocument`],
//!   [`tokenize`],
//!   [`compile_matcher`]).
//! - [`embedding`] — Rust-only embedding-backed
//!   scoring ([`EmbeddingScorer`],
//!   [`blend_scores`]).
//! - [`codebooks`] — the compact boundary code book
//!   ([`CompactCodebook`]) and the
//!   prefix trie a constrained decoder needs
//!   ([`CodeTrie`]).
//! - [`truncation`] — the score-truncation decision
//!   ([`truncate_sorted_by_score`])
//!   plus the bounded char/item helpers the renderers use.
//!
//! # Two search cursors
//!
//! [`SearchCursor`](crate::retrieval::types::SearchCursor) in
//! [`types`](crate::retrieval::types) is the *build-side* resumable pre-order traversal plan.
//! The search runtime uses its own Python-shaped cursor,
//! [`flow::SearchCursor`], which names one node
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
pub mod contracts;
pub mod embedding;
pub mod expand;
pub mod flow;
pub mod lexical;
pub mod reduce;
pub mod selector;
pub mod subtree;
pub mod truncation;

pub use codebooks::{
    default_compact_boundary_codebook, CodeTrie, CompactCodebook, DEFAULT_COMPACT_CODEBOOK_SIZE,
};
pub use config::{
    DisclosureConfig, EmbeddingScoringConfig, ProgressiveSearchConfig, SelectionMode, DEFAULT_TOP_K,
};
pub use contracts::{
    BranchReducer, CurrentSubtreeProvider, FragmentBuildFn, GenerateFn, GenerateFragmentSelector,
    GenerateRequest, SelectRequest, TargetExpander, TopKSelector,
};
pub use embedding::{blend_scores, cosine_similarity, EmbeddingScorer};
pub use expand::DefaultTargetExpander;
pub use flow::{
    ChatMessage, ChildSearchCursor, CurrentSubtree, ExpansionPlan, ExposedFragment, ExposedNode,
    NodeSearchResult, PromptBundle, SearchCursor, SelectableResolution, SelectableTarget,
    SelectionProtocol, SelectionResult, ABSTAIN_TOKEN,
};
pub use lexical::{
    compile_matcher, tokenize, LexicalDocument, LexicalHit, LexicalIndex, QueryMatcher, BM25_B,
    BM25_K1, FIELD_WEIGHTS, IDENTITY_BONUS, PHRASE_FIELD_WEIGHTS,
};
pub use reduce::DefaultBranchReducer;
pub use selector::{
    is_abstain_output, query_text, DefaultTopKSelector, LexicalTopKSelector, ABSTAIN_HINTS,
    FRAGMENT_SELECTED_EVENT, MODE_GENERATE, MODE_LEXICAL,
};
pub use subtree::{
    build_progressive_branch_description, build_progressive_item_label, build_progressive_root,
    choices_cache_key, DefaultCurrentSubtreeProvider, ItemCounterFn,
};
pub use truncation::{
    take_budget, truncate_chars, truncate_chars_with_ellipsis, truncate_sorted_by_score,
    ScoreTruncationConfig, ScoreTruncationDecision,
};

/// The default `top_k` of the progressive runtime (Python `ProgressiveRetrieverConfig.top_k`).
///
/// [`DisclosureConfig`] users reach it
/// through [`ProgressiveSearchConfig`];
/// the public facade in a later wave defaults to the larger
/// [`DEFAULT_TOP_K`] instead.
pub const RUNTIME_TOP_K: usize = 5;
