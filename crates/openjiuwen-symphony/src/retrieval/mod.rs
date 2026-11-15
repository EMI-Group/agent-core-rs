//! Capability-tree index construction and progressive retrieval.
//!
//! Ports Python `openjiuwen.symphony.retrieval` (v0.1.20). This module currently
//! implements the **index BUILD half**: the item sources, the deterministic grouping that
//! places an item in the tree, the preset/schema guard that bounds a tree's shape, the full
//! and incremental builder, and the IO of the three index artifacts (`tree_index.yaml`,
//! `catalog.jsonl`, `manifest.json`).
//!
//! # The search runtime is the remaining half
//!
//! The query-time **search runtime** — subtree selection, field-weighted lexical scoring,
//! bounded disclosure rendering, and embedding-backed branch selection — has no module yet.
//! The `pub mod search;` slot below is reserved for it, and it consumes only the types
//! declared here (the tree, the choices, the catalog records and the cursor), so its surface
//! is already fixed by them.
//!
//! # Layout
//!
//! - [`types`](crate::retrieval::types) — the shared value objects:
//!   [`RetrieverItem`](crate::retrieval::types::RetrieverItem),
//!   [`RetrieverNode`](crate::retrieval::types::RetrieverNode),
//!   [`RetrieverChoice`](crate::retrieval::types::RetrieverChoice),
//!   [`RetrieverCandidate`](crate::retrieval::types::RetrieverCandidate),
//!   [`RetrieverTrace`](crate::retrieval::types::RetrieverTrace), the resumable
//!   [`SearchCursor`](crate::retrieval::types::SearchCursor), the persisted
//!   [`IndexNode`](crate::retrieval::types::IndexNode),
//!   [`IndexManifest`](crate::retrieval::types::IndexManifest),
//!   [`CatalogRecord`](crate::retrieval::types::CatalogRecord) and the loaded
//!   [`LoadedIndex`](crate::retrieval::types::LoadedIndex), plus the text helpers
//!   [`compact_text`](crate::retrieval::types::compact_text) and
//!   [`build_retrieval_text`](crate::retrieval::types::build_retrieval_text).
//! - [`artifacts`](crate::retrieval::artifacts) — the index artifact set:
//!   [`write_index`](crate::retrieval::artifacts::write_index) and
//!   [`load_index`](crate::retrieval::artifacts::load_index), the deterministic renderers
//!   and parsers behind them, and
//!   [`build_tree_root`](crate::retrieval::artifacts::build_tree_root), which turns the
//!   persisted nodes into an in-memory tree.
//! - [`build`](crate::retrieval::build) — the index-build stages, from item sources through
//!   grouping and node assembly to catalog derivation and publication.
//!
//! # Rust-only additions and gaps
//!
//! - [`SearchCursor`](crate::retrieval::types::SearchCursor) is Rust-only: Python keeps its
//!   search cursor opaque, here it is a serializable value that carries its own traversal
//!   plan.
//! - [`IndexManifest::content_hash`](crate::retrieval::types::IndexManifest) is Rust-only:
//!   a SHA-256 of the rendered `catalog.jsonl`, so a reader detects a catalog that was
//!   rewritten without its manifest.
//! - [`LoadedIndex::item_paths`](crate::retrieval::types::LoadedIndex) is Rust-only: the
//!   item-payload → ancestor-`cid` map is precomputed at load time instead of being derived
//!   per query.
//! - `tree_index.html`, Python's human-readable tree rendering, is not produced; a build
//!   writes only the three artifacts above.
//!
//! Every construction and retrieval step is deterministic and offline; the GPU/ML grouping
//! backend (LLM-guided grouping, faiss k-means) stays a typed
//! [`Error::Unsupported`](crate::error::Error).

pub mod artifacts;
pub mod build;
pub mod types;

// Reserved for the query-time search runtime (subtree selection, bounded disclosure
// rendering, embedding-backed branch selection), which has no module yet. It will be
// mounted here as `pub mod search;`.
