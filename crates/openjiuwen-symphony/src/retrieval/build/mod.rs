//! Index construction: the build half of the capability-tree index.
//!
//! Ports the build half of Python `openjiuwen.symphony.retrieval.build` (v0.1.20). One
//! build turns item sources into the artifact files of a single index directory:
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
//! Item sources → grouping → optional tree shaping → node assembly → catalog → publication:
//!
//! 1. **Item sources** ([`scanner`]) produce the flat, sorted item set a build indexes.
//! 2. **Grouping** ([`grouping`], steered by the curated paths of [`presets`]) derives each item's branch path, honouring Python's fan-out and depth bounds.
//!    The k-means and model-guided strategies are additionally steered by a [`GroupingRuntime`](crate::retrieval::build::grouping::GroupingRuntime) that supplies the embedder or completion model they need.
//! 3. **Tree shaping** ([`tree::shaping`], optional) refines the placements with a model or repairs a node list the schema guard rejected.
//! 4. **Node assembly** ([`tree`], checked by [`schema`]) turns the placements into the flat, ordered node list and prunes the branches nothing landed in.
//! 5. **Catalog** ([`catalog`]) joins each placed leaf to its source item and derives the `retrieval_text` the search runtime scores.
//! 6. **Publication** ([`workflows`]) drives a full or incremental [`IndexBuilder`](crate::retrieval::build::workflows::IndexBuilder) run and writes the three artifacts through [`crate::retrieval::artifacts`], plus the opt-in `tree_index.html` document from [`tree::visualizer`].
//!
//! # Modules
//!
//! - [`catalog`] — one `CatalogRecord` per indexed leaf, with its derived retrieval text.
//! - [`grouping`] — the grouping strategies that place items in the tree, with the embedding-backed [`kmeans`](crate::retrieval::build::grouping::kmeans) and model-guided [`llm`](crate::retrieval::build::grouping::llm) stages behind a shared [`GroupingRuntime`](crate::retrieval::build::grouping::GroupingRuntime).
//! - [`presets`] — the nested-or-flat `tree_index.yaml` preset reader and `cid` deriver.
//! - [`scanner`] — item sources: descriptors, skill inventories, SKILL.md trees, JSONL.
//! - [`schema`] — tree-shape validation and the fixed root taxonomy.
//! - [`tree`] — node assembly, `cid` derivation, leaf renaming and empty-branch pruning, with the opt-in [`shaping`](crate::retrieval::build::tree::shaping) stages and the [`visualizer`](crate::retrieval::build::tree::visualizer).
//! - [`tree_text`] — the `cid` and token helpers every stage above shares.
//! - [`workflows`] — `IndexBuilder`: full and incremental builds, their synchronous rule path and the [`GroupingRuntime`](crate::retrieval::build::grouping::GroupingRuntime) entry points `build_with_runtime` and `add_with_runtime`.
//!
//! The shared value objects live in [`crate::retrieval::types`], and the artifact IO in
//! [`crate::retrieval::artifacts`], so a build stage only decides *what* the nodes and
//! records are — writing and reading them is already settled.
//!
//! # Opt-in extras and runtime requirements
//!
//! The k-means (`kmeans`, `faiss_kmeans`) and model-guided (`llm_grouping`) grouping strategies, the tree-shaping stages and the `tree_index.html` document are opt-in.
//! Grouping that needs an embedder or a completion model runs only through the [`GroupingRuntime`](crate::retrieval::build::grouping::GroupingRuntime) entry points — [`group_items_with`](crate::retrieval::build::grouping::group_items_with), `IndexBuilder::build_with_runtime` and `IndexBuilder::add_with_runtime`.
//! The synchronous [`group_items`](crate::retrieval::build::grouping::group_items), [`IndexBuilder::build`](crate::retrieval::build::workflows::IndexBuilder::build) and `IndexBuilder::add` path places every item with a rule strategy and answers [`Error::Unsupported`](crate::error::Error) for a runtime strategy.
//! Shaping is configured through the `BuildOptions` field of [`BuildConfig`](crate::retrieval::build::workflows::BuildConfig): its default `shaping` is [`ShapingMode::None`](crate::retrieval::build::tree::shaping::ShapingMode::None), which keeps a build byte-identical to an unshaped run and calls no model.
//! The HTML document is configured through the same `BuildOptions::write_html`, is off by default and stays outside the three-artifact contract.

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
