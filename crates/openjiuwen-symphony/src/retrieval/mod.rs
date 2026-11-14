//! Capability-tree index construction and progressive retrieval.
//!
//! Ports the offline core of Python `openjiuwen.symphony.retrieval` (v0.1.20): the item
//! tree model, deterministic tree construction, the persisted index artifact set
//! (`tree_index.yaml`, `catalog.jsonl`, `manifest.json`), and the query-time retrieval
//! runtime (field-weighted lexical scoring, subtree selection, bounded disclosure
//! rendering, and embedding-backed branch scoring).
//!
//! Every construction and retrieval step is deterministic and offline; the GPU/ML
//! grouping backend (faiss k-means) stays a typed
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported).
