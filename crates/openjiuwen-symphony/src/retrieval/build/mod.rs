//! Index construction: the build half of the capability-tree index.
//!
//! Ports the build half of Python `openjiuwen.symphony.retrieval.build` (v0.1.20). One
//! build turns item sources into the three artifact files of a single index directory:
//!
//! ```text
//! <index-dir>/
//!   tree_index.yaml   # the capability tree, one entry per branch and leaf node
//!   catalog.jsonl     # one JSON record per leaf item
//!   manifest.json     # mode, count, item paths, worker ids, item type, content hash
//! ```
//!
//! # Pipeline
//!
//! Item sources → grouping → node assembly → catalog → publication:
//!
//! 1. **Item sources** ([`scanner`]) produce the flat, sorted item set a build indexes.
//! 2. **Grouping** ([`grouping`], steered by the optional paths of [`presets`]) derives
//!    each item's branch path, honouring Python's fan-out and depth bounds.
//! 3. **Node assembly** ([`tree`], checked by [`schema`]) turns the placements into the
//!    flat, ordered node list and prunes the branches nothing landed in.
//! 4. **Catalog** ([`catalog`]) joins each placed leaf to its source item and derives the
//!    `retrieval_text` the search runtime scores.
//! 5. **Publication** ([`workflows`]) drives a full or incremental
//!    [`IndexBuilder`](crate::retrieval::build::workflows::IndexBuilder) run and writes
//!    the three artifacts through [`crate::retrieval::artifacts`].
//!
//! # Modules
//!
//! - [`catalog`] — one `CatalogRecord` per indexed leaf, with its derived retrieval text.
//! - [`grouping`] — the deterministic grouping strategies that place items in the tree.
//! - [`presets`] — the nested-or-flat `tree_index.yaml` preset reader and `cid` deriver.
//! - [`scanner`] — item sources: descriptors, skill inventories, SKILL.md trees, JSONL.
//! - [`schema`] — tree-shape validation and the fixed root taxonomy.
//! - [`tree`] — node assembly, `cid` derivation, leaf renaming and empty-branch pruning.
//! - [`tree_text`] — the `cid` and token helpers every stage above shares.
//! - [`workflows`] — `IndexBuilder`: full and incremental builds and their output.
//!
//! The shared value objects live in [`crate::retrieval::types`], and the artifact IO in
//! [`crate::retrieval::artifacts`], so a build stage only decides *what* the nodes and
//! records are — writing and reading them is already settled.
//!
//! # Deferred
//!
//! - **LLM/faiss grouping.** [`grouping`] keeps Python's strategy names
//!   (`llm_grouping`, `faiss_kmeans`, `kmeans`) and reports
//!   [`Error::Unsupported`](crate::error::Error) for them; its deterministic strategies
//!   are the offline substitute.
//! - **LLM subtree repair.** [`tree`] prunes empty branches deterministically instead of
//!   asking a model to repair the subtree, and [`schema`] validates the shape rather than
//!   having a model fix it.
//! - **`tree_index.html`.** Python renders a human-readable tree document; the port writes
//!   only the three artifacts above (the file name constant stays for compatibility).

pub mod catalog;
/// Grouping of scanned items into tree branches.
pub mod grouping;
/// The nested-or-flat `tree_index.yaml` preset reader and `cid` deriver.
pub mod presets;
/// Scanning of the items a build indexes.
pub mod scanner;
/// Validation of a tree's shape against the build's bounds and root taxonomy.
pub mod schema;
/// Assembly and repair of the persisted node list.
pub mod tree;
/// The `cid` and token helpers every build stage shares.
pub mod tree_text;
/// Full and incremental index builds ([`workflows::IndexBuilder`]).
pub mod workflows;
