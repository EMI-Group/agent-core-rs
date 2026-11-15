//! Capability-tree index construction and progressive retrieval.
//!
//! Ports Python `openjiuwen.symphony.retrieval` (v0.1.20). This module implements the
//! **index BUILD half** — the item sources, the deterministic grouping that places an item
//! in the tree, the preset/schema guard that bounds a tree's shape, the full and incremental
//! builder, and the IO of the three index artifacts (`tree_index.yaml`, `catalog.jsonl`,
//! `manifest.json`) — and the **foundation layer of the query-time search runtime**: the
//! search value objects, the runtime configuration, the field-weighted lexical scorer, the
//! optional embedding scorer, the compact codebook, and score/budget truncation.
//!
//! # The search runtime is being ported in waves
//!
//! [`search`](crate::retrieval::search) currently holds the runtime's foundation.
//! Progressive disclosure rendering, the traversal protocols (subtree selection, expansion,
//! reduction), the engine, the progressive and flat retrievers, and the service facade are
//! not present yet; they are added by the following waves on top of
//! [`search::flow`](crate::retrieval::search::flow),
//! [`search::config`](crate::retrieval::search::config) and
//! [`search::lexical`](crate::retrieval::search::lexical), and they consume only the types
//! declared here (the tree, the choices, the catalog records and the cursor).
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
//! - [`search`](crate::retrieval::search) — the query-time runtime foundation: the flow
//!   value objects ([`search::flow`](crate::retrieval::search::flow)), the runtime
//!   configuration ([`search::config`](crate::retrieval::search::config)), the
//!   field-weighted lexical scorer ([`search::lexical`](crate::retrieval::search::lexical)),
//!   the embedding scorer ([`search::embedding`](crate::retrieval::search::embedding)), the
//!   compact codebook ([`search::codebooks`](crate::retrieval::search::codebooks)) and
//!   score/budget truncation ([`search::truncation`](crate::retrieval::search::truncation)).
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
pub mod search;
pub mod types;
