//! Capability-tree index construction and progressive retrieval.
//!
//! Ports the offline core of Python `openjiuwen.symphony.retrieval` (v0.1.20): the item
//! tree model, deterministic tree construction, the persisted index artifact set
//! (`tree_index.yaml`, `catalog.jsonl`, `manifest.json`), and — added by the next round —
//! the query-time retrieval runtime (field-weighted lexical scoring, subtree selection,
//! bounded disclosure rendering, and embedding-backed branch scoring).
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
//! - [`build`](crate::retrieval::build) — the index-build stages.
//!
//! # Next round
//!
//! The query-time runtime (`search`) is added in the next round as `pub mod search;`
//! alongside the modules above. It consumes only the types declared here — the tree, the
//! choices, the catalog records and the cursor — so its surface is fixed by this round.
//!
//! Every construction and retrieval step is deterministic and offline; the GPU/ML
//! grouping backend (faiss k-means) stays a typed
//! [`Error::Unsupported`].

pub mod artifacts;
pub mod build;
pub mod types;
