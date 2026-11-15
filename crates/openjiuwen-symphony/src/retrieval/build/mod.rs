//! Index construction: the build stages that turn capability assets into an index.
//!
//! Ports the build half of Python `openjiuwen.symphony.retrieval.build` (v0.1.20). One
//! build turns a set of scanned items into three files inside one directory:
//!
//! ```text
//! <index-dir>/
//!   tree_index.yaml   # the capability tree, one entry per branch and leaf node
//!   catalog.jsonl     # one JSON record per leaf item
//!   manifest.json     # mode, count, item paths, worker ids, item type, content hash
//! ```
//!
//! The stages and the modules they live in:
//!
//! - [`scanner`] — discover the items to index.
//! - [`grouping`] — group items into tree branches.
//! - [`presets`] — deterministic preset taxonomies.
//! - [`schema`] — validate a model-written tree.
//! - [`tree`] — assemble and repair the node list.
//! - [`workflows`] — full and incremental builds.
//! - [`tree_text`] — the `cid` and token helpers the
//!   stages above share; it is implemented, the other six modules are the next stage's.
//!
//! The shared value objects live in [`crate::retrieval::types`], and the artifact IO in
//! [`crate::retrieval::artifacts`], so a build stage only has to decide *what* the nodes
//! and records are — writing and reading them is already settled.

/// Grouping of scanned items into tree branches (implemented by the next build stage).
pub mod grouping;
/// Deterministic preset taxonomies for a tree build (implemented by the next build stage).
pub mod presets;
/// Scanning of the items a build indexes (implemented by the next build stage).
pub mod scanner;
/// Validation of a model-written tree document (implemented by the next build stage).
pub mod schema;
/// Assembly and repair of the persisted node list (implemented by the next build stage).
pub mod tree;
/// The `cid` and token helpers every build stage shares.
pub mod tree_text;
/// Full and incremental index builds (implemented by the next build stage).
pub mod workflows;
