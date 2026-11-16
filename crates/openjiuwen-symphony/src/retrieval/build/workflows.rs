//! Full and incremental index builds: the
//! [`IndexBuilder`](crate::retrieval::build::workflows::IndexBuilder).
//!
//! Ports Python `openjiuwen.symphony.retrieval.build.workflows`
//! (`index_builder.py`, `full_build.py`, `incremental_build.py`, `tree_ops.py`,
//! `artifacts.py`, `output_writer.py`, openJiuwen SDK v0.1.20) as one builder over a flat
//! [`IndexNode`](crate::retrieval::types::IndexNode) list plus a
//! [`CatalogRecord`](crate::retrieval::types::CatalogRecord) catalog.
//!
//! # What one full build does
//!
//! 1. Normalizes the item type: the sources' [`ItemSources::item_type`](crate::retrieval::build::scanner::ItemSources::item_type) wins, the
//!    configuration's `item_type` is the fallback, and anything but `skill`/`plugin` is a
//!    validation error.
//! 2. Places the scanned items. The [`build`](crate::retrieval::build::workflows::IndexBuilder::build)
//!    path calls the synchronous [`group_items`](crate::retrieval::build::grouping::group_items),
//!    which is where a strategy that needs a model or an embedder stays a typed
//!    deferral; [`build_with_runtime`](crate::retrieval::build::workflows::IndexBuilder::build_with_runtime)
//!    awaits [`group_items_with`](crate::retrieval::build::grouping::group_items_with) over
//!    the caller's [`GroupingRuntime`](crate::retrieval::build::grouping::GroupingRuntime), so
//!    `llm_grouping`, `faiss_kmeans` and `kmeans` place items there. Either way the
//!    provenance of the placements is reported as
//!    [`BuildReport::grouping_provenance`](crate::retrieval::build::workflows::BuildReport::grouping_provenance).
//! 3. Optionally shapes those placements with one of the model stages
//!    ([`BuildOptions::shaping`](crate::retrieval::build::workflows::BuildOptions::shaping)):
//!    `one_shot`, `expansion` and `equivalence` run *before* assembly through
//!    [`apply`](crate::retrieval::build::tree::shaping::apply), and `repair` runs *after* the
//!    schema check below. `None` — the default — calls no stage at all.
//! 4. Assembles the node list
//!    ([`build_nodes`](crate::retrieval::build::tree::build_nodes)) — the deterministic,
//!    model-free tree of Python's fallback path.
//! 5. Checks the tree ([`TreeSchema::validate`](crate::retrieval::build::schema::TreeSchema::validate)).
//!    A `blank_cid`, `duplicate_cid`, `missing_parent`, `cycle`, `excessive_depth` or
//!    `too_many_nodes` aborts the build before a byte is written, because each of them
//!    describes a tree this builder must not publish; every other code (`excessive_fan_out`,
//!    `blank_label`, `missing_worker_id`) is advisory — kept in the report and logged.
//!    With [`ShapingMode::Repair`](crate::retrieval::build::tree::shaping::ShapingMode::Repair)
//!    configured, a fatal finding is instead handed to
//!    [`repair`](crate::retrieval::build::tree::shaping::repair::repair) and the repaired node
//!    list is re-validated: a tree that passes is published, one that still fails
//!    aborts in the same `Error::Validation` shape as an un-repaired build.
//! 6. Derives one catalog record per leaf that carries a `worker_id`, joined to its scanned
//!    item by that id ([`catalog_record`](crate::retrieval::build::catalog::catalog_record)).
//! 7. Publishes the three artifacts atomically through
//!    [`write_index`](crate::retrieval::artifacts::write_index) and, only when
//!    [`BuildOptions::write_html`](crate::retrieval::build::workflows::BuildOptions::write_html)
//!    asks for it, the opt-in `tree_index.html` visualizer after them.
//!
//! # Incremental patches
//!
//! [`IndexBuilder::open`](crate::retrieval::build::workflows::IndexBuilder::open) hydrates
//! the builder from an existing index directory;
//! [`add`](crate::retrieval::build::workflows::IndexBuilder::add) places new items and
//! replaces re-added ones,
//! [`delete`](crate::retrieval::build::workflows::IndexBuilder::delete) drops items and the
//! branches they emptied, and
//! [`rebuild_subtree`](crate::retrieval::build::workflows::IndexBuilder::rebuild_subtree)
//! re-derives one branch from the records below it. Each patch reports what it changed and
//! whether the change was large enough that a full rebuild is worthwhile.
//! [`add_with_runtime`](crate::retrieval::build::workflows::IndexBuilder::add_with_runtime)
//! is the runtime counterpart of `add`, and both share one placement path.
//!
//! # The opt-in visualizer
//!
//! `tree_index.html` sits outside the three-artifact contract: a build writes it only when
//! [`BuildOptions::write_html`](crate::retrieval::build::workflows::BuildOptions::write_html)
//! is set, it is written after the three artifacts are in place, and it is replaced
//! atomically through a temp sibling plus rename, so a failed render cannot corrupt a
//! published index. No search path reads it back, and with the flag off the published bytes
//! are exactly the three files.
//!
//! # Deviations from the Python workflows
//!
//! - Python's incremental build escalates by *re-running the full build itself*, because it
//!   still holds every item source. Here a patch simply sets
//!   [`IncrementalReport::escalated`](crate::retrieval::build::workflows::IncrementalReport),
//!   the patch is applied anyway, and the caller decides whether to
//!   [`build`](crate::retrieval::build::workflows::IndexBuilder::build) again.
//! - A subtree rebuild is deterministic instead of model-driven: Python asks an LLM to
//!   repair the branch, the port re-runs the grouping and assembly stages over the branch's
//!   own catalog records, so the result is reproducible offline.
//! - A patch is placement-only: the shaping stages run on a full build, never on
//!   [`add`](crate::retrieval::build::workflows::IndexBuilder::add) or
//!   [`add_with_runtime`](crate::retrieval::build::workflows::IndexBuilder::add_with_runtime),
//!   so a patch changes the items it is given and nothing else.
//! - A repair is reported coarsely:
//!   [`repair`](crate::retrieval::build::tree::shaping::repair::repair) answers only the
//!   repaired node list, so
//!   [`BuildReport::shaping_provenance`](crate::retrieval::build::workflows::BuildReport::shaping_provenance)
//!   is `Model` when a model was supplied with a non-zero call budget and
//!   `DeterministicFallback` otherwise, without knowing which of the two paths produced the
//!   accepted tree.
//! - No clock is read and no map iteration order reaches an output: nodes sort by
//!   `(depth, cid)` and records by `cid`.

use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind as IoErrorKind;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::{debug, warn};

use crate::error::{validation, Error, Result};
use crate::llm::CompletionModel;
use crate::retrieval::artifacts::{catalog_content_hash, load_index, parse_tree_yaml, write_index};
use crate::retrieval::build::catalog::{
    catalog_record, records_for_nodes, rekey_record, sort_records, sorted_records,
};
use crate::retrieval::build::grouping::{
    self, GroupAssignment, GroupingConfig, GroupingProvenance, GroupingRuntime,
};
use crate::retrieval::build::presets::TreePreset;
use crate::retrieval::build::scanner::{
    self, ItemSources, SourceDiagnostic, SourceItem, JSONL_PATH_PREFIX,
};
use crate::retrieval::build::schema::{
    TreeDiagnostic, TreeSchema, DIAGNOSTIC_BLANK_CID, DIAGNOSTIC_CYCLE, DIAGNOSTIC_DUPLICATE_CID,
    DIAGNOSTIC_EXCESSIVE_DEPTH, DIAGNOSTIC_MISSING_PARENT, DIAGNOSTIC_TOO_MANY_NODES,
};
use crate::retrieval::build::tree;
use crate::retrieval::build::tree::shaping::repair::repair as repair_tree;
use crate::retrieval::build::tree::shaping::{self, ShapingConfig, ShapingMode, ShapingProvenance};
use crate::retrieval::build::tree::visualizer;
use crate::retrieval::build::tree_text::{slug_term, sort_nodes};
use crate::retrieval::types::{
    CatalogRecord, IndexManifest, IndexNode, RetrieverItem, DEFAULT_ITEM_TYPE, META_CONTENT,
    META_DONT_SELECT_WHEN, META_SELECT_WHEN, MODE_FULL, TREE_INDEX_FILENAME,
};

/// The default change ratio above which an incremental patch flags a full rebuild.
///
/// Mirrors Python's `incremental_max_change_ratio`: a patch that touches more than a
/// quarter of the catalog is cheap to replace wholesale, so the caller is told to.
pub const DEFAULT_ESCALATION_RATIO: f64 = 0.25;

/// The diagnostic codes that abort a build instead of merely being reported.
///
/// The first four make the tree unusable (a missing, repeated, dangling or circular `cid`),
/// and the last two describe a tree that exceeded the shape the configuration allows: a
/// node deeper than [`TreeSchema::max_depth`] and a node list longer than
/// [`TreeSchema::max_nodes`] are both bounds the builder must not publish past.
/// Everything else [`TreeSchema::validate`] can report (`excessive_fan_out`, `blank_label`,
/// `missing_worker_id`) describes a tree that can still be published, so it is carried in
/// the report and logged.
const FATAL_TREE_CODES: [&str; 6] = [
    DIAGNOSTIC_BLANK_CID,
    DIAGNOSTIC_DUPLICATE_CID,
    DIAGNOSTIC_MISSING_PARENT,
    DIAGNOSTIC_CYCLE,
    DIAGNOSTIC_EXCESSIVE_DEPTH,
    DIAGNOSTIC_TOO_MANY_NODES,
];

/// Opt-in extras beyond the three-artifact publish contract.
///
/// The default — no visualizer, [`ShapingMode::None`] — keeps a build to the deterministic
/// stages of the three artifacts, so a caller that asks for nothing publishes exactly what
/// it published before either option existed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct BuildOptions {
    /// Also render `tree_index.html` next to the three artifacts.
    ///
    /// The document is derived from the published node list and catalog, no reader opens
    /// it, and it is written after the three artifacts, so this flag only adds a file.
    pub write_html: bool,
    /// How the LLM tree-shaping stage refines the deterministic placements.
    pub shaping: ShapingConfig,
}

/// Build-time configuration: the offline knobs of Python's `BuildConfig`.
///
/// Python's full configuration also carries the LLM client, model, timeouts, worker counts
/// and the S3 upload target; the port keeps the four settings its deterministic stages read,
/// the optional curated preset and the opt-in extras [`BuildOptions`] collects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct BuildConfig {
    /// How items are grouped into branches.
    pub grouping: GroupingConfig,
    /// The shape bounds and diagnostics the assembled tree is checked against.
    pub schema: TreeSchema,
    /// What one catalog record describes: `skill` or `plugin`.
    pub item_type: String,
    /// The change ratio above which an incremental patch flags a full rebuild.
    pub escalation_ratio: f64,
    /// A caller-supplied curated taxonomy; its paths win over the derived grouping paths.
    pub preset: Option<TreePreset>,
    /// The opt-in extras: the visualizer and the tree-shaping stage.
    pub options: BuildOptions,
}

impl Default for BuildConfig {
    /// Returns a skill build with the default grouping, the default schema, a
    /// [`DEFAULT_ESCALATION_RATIO`] escalation, no preset and no opt-in extra.
    fn default() -> Self {
        Self {
            grouping: GroupingConfig::default(),
            schema: TreeSchema::default(),
            item_type: DEFAULT_ITEM_TYPE.to_string(),
            escalation_ratio: DEFAULT_ESCALATION_RATIO,
            preset: None,
            options: BuildOptions::default(),
        }
    }
}

impl BuildConfig {
    /// Builds the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the grouping configuration.
    #[must_use]
    pub fn with_grouping(mut self, grouping: GroupingConfig) -> Self {
        self.grouping = grouping;
        self
    }

    /// Replaces the tree-shape guard.
    #[must_use]
    pub fn with_schema(mut self, schema: TreeSchema) -> Self {
        self.schema = schema;
        self
    }

    /// Replaces the item type; it is validated by [`BuildConfig::validate`].
    #[must_use]
    pub fn with_item_type(mut self, item_type: impl Into<String>) -> Self {
        self.item_type = item_type.into();
        self
    }

    /// Replaces the escalation ratio; it is validated by [`BuildConfig::validate`].
    #[must_use]
    pub fn with_escalation_ratio(mut self, escalation_ratio: f64) -> Self {
        self.escalation_ratio = escalation_ratio;
        self
    }

    /// Installs a curated preset taxonomy.
    #[must_use]
    pub fn with_preset(mut self, preset: TreePreset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Replaces the opt-in extras: the visualizer flag and the shaping stage.
    #[must_use]
    pub fn with_options(mut self, options: BuildOptions) -> Self {
        self.options = options;
        self
    }

    /// Turns the `tree_index.html` visualizer on or off.
    ///
    /// The rest of [`BuildConfig::options`] — the shaping configuration — is kept.
    #[must_use]
    pub fn with_html(mut self, write_html: bool) -> Self {
        self.options.write_html = write_html;
        self
    }

    /// The grouping the build stages actually run with.
    ///
    /// The configured [`BuildConfig::grouping`] is cloned as-is, except that a preset's
    /// [`paths`](crate::retrieval::build::presets::TreePreset::paths) replace the taxonomy,
    /// so the curated branch paths win over the paths the grouping stage would derive.
    ///
    /// A preset path that starts at the configured
    /// [`root_segment`](GroupingConfig::root_segment) — compared by [`slug_term`],
    /// case-insensitively — has that leading segment stripped before it becomes a taxonomy
    /// path. The grouping stage already places every path *below* the root, so without the
    /// strip a preset rooted at `Skills` would publish doubled cids such as
    /// `Skills.skills.documents`.
    /// Paths that become empty are dropped, and the result is sorted and deduplicated.
    #[must_use]
    pub fn effective_grouping(&self) -> GroupingConfig {
        let mut grouping = self.grouping.clone();
        if let Some(preset) = self.preset.as_ref() {
            let root = slug_term(&grouping.root_segment, "root");
            let mut taxonomy: Vec<Vec<String>> = preset
                .paths()
                .into_iter()
                .filter_map(|mut path| {
                    let rooted = path.first().is_some_and(|segment| {
                        slug_term(segment, "root").eq_ignore_ascii_case(&root)
                    });
                    if rooted {
                        path.remove(0);
                    }
                    if path.is_empty() {
                        None
                    } else {
                        Some(path)
                    }
                })
                .collect();
            taxonomy.sort();
            taxonomy.dedup();
            grouping.taxonomy = taxonomy;
        }
        grouping
    }

    /// Checks the two knobs a build cannot repair.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] naming `item_type` when it is neither `skill` nor `plugin`, and
    /// naming `escalation_ratio` when it is negative or not finite.
    pub fn validate(&self) -> Result<()> {
        scanner::normalize_item_type(&self.item_type)?;
        if !self.escalation_ratio.is_finite() || self.escalation_ratio < 0.0 {
            return Err(validation(
                "escalation_ratio",
                format!("must be finite and >= 0, got {}", self.escalation_ratio),
            ));
        }
        Ok(())
    }
}

/// What one build or patch produced.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct BuildReport {
    /// The directory the artifacts were published into.
    pub output_dir: PathBuf,
    /// The normalized item type of the published index.
    pub item_type: String,
    /// How many tree nodes were published, branches included.
    pub node_count: usize,
    /// How many catalog records were published.
    pub record_count: usize,
    /// The manifest written last, once the tree and the catalog were in place.
    pub manifest: IndexManifest,
    /// The item-source diagnostics carried through from the sources.
    pub diagnostics: Vec<SourceDiagnostic>,
    /// The advisory schema findings; a fatal one aborted the build instead.
    pub tree_diagnostics: Vec<TreeDiagnostic>,
    /// How the placements were derived: the rule, the embedder, the model or a fallback.
    pub grouping_provenance: GroupingProvenance,
    /// Where the shaping stage's placements came from; `Skipped` when it never ran.
    pub shaping_provenance: ShapingProvenance,
    /// Whether the build also rendered `tree_index.html` beside the three artifacts.
    pub html_written: bool,
}

/// What one incremental patch changed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IncrementalReport {
    /// The worker ids the patch added or replaced, sorted.
    pub added: Vec<String>,
    /// The worker ids the patch removed, sorted.
    pub removed: Vec<String>,
    /// The branch cids the patch created, dropped or re-derived, sorted.
    pub rebuilt_branches: Vec<String>,
    /// Whether the change ratio exceeds the configured escalation ratio.
    ///
    /// Python answers this by rebuilding the whole index from its item sources. The port
    /// has no sources to re-read, so the patch stays applied and the caller decides whether
    /// to run [`IndexBuilder::build`] again.
    pub escalated: bool,
    /// How much of the catalog the patch touched: `changed / max(1, records_before)`.
    pub change_ratio: f64,
}

/// The index builder: a flat node list, its catalog, and the configuration both follow.
///
/// The builder is the mutable in-memory index. Build it with
/// [`IndexBuilder::new`] (empty) or hydrate it with [`IndexBuilder::open`] (from a
/// published directory), patch it with [`IndexBuilder::add`]/[`IndexBuilder::delete`]/
/// [`IndexBuilder::rebuild_subtree`], and publish it with [`IndexBuilder::write`].
#[derive(Debug, Clone, PartialEq)]
pub struct IndexBuilder {
    /// The tree nodes, kept sorted by `(depth, cid)`.
    nodes: Vec<IndexNode>,
    /// The catalog records, kept sorted by `cid`.
    records: Vec<CatalogRecord>,
    /// The scanned item paths the manifest records.
    item_paths: Vec<String>,
    /// The normalized item type.
    item_type: String,
    /// The configuration every patch follows.
    config: BuildConfig,
}

// The patch entry points are `async` because they are the same seam as `build`, `open` and
// `write`, which do touch the filesystem; a patch that only mutates memory must not force a
// caller to change signature when it is published.
#[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
impl IndexBuilder {
    /// Builds an empty builder for `config`.
    ///
    /// The item type follows the scanner's lenient rule, so a caller can assemble a builder
    /// before running [`BuildConfig::validate`]; [`IndexBuilder::write`] re-checks it.
    #[must_use]
    pub fn new(config: BuildConfig) -> Self {
        Self {
            nodes: Vec::new(),
            records: Vec::new(),
            item_paths: Vec::new(),
            item_type: scanner::lenient_item_type(&config.item_type),
            config,
        }
    }

    /// Runs a full build and publishes it as [`MODE_FULL`].
    ///
    /// The item type comes from the sources, falling back to `config`; the effective
    /// grouping places every item; the optional shaping stage refines those placements; the
    /// assembled tree is checked before anything is written; and one catalog record per leaf
    /// carrying a `worker_id` is derived from the item with that id.
    ///
    /// Placement runs the *synchronous* grouping path, which is where the deferral of the
    /// three runtime strategies lives: `llm_grouping`, `faiss_kmeans` and `kmeans` fail with
    /// [`Error::Unsupported`] here instead of silently degrading to the deterministic rule.
    /// Use [`IndexBuilder::build_with_runtime`] to run them.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an unusable item type, escalation ratio or leaf-free
    /// record set, a fatal tree diagnostic, a grouping failure, or an artifact-write
    /// failure; see [`write_index`].
    pub async fn build(
        sources: ItemSources,
        output_dir: impl AsRef<Path>,
        config: BuildConfig,
    ) -> Result<BuildReport> {
        Self::run(sources, output_dir, config, None).await
    }

    /// Runs a full build whose placement and shaping stages may use `runtime`.
    ///
    /// [`IndexBuilder::build`] with a [`GroupingRuntime`]: the placement stage awaits
    /// [`group_items_with`](crate::retrieval::build::grouping::group_items_with), so the
    /// runtime strategies (`llm_grouping`, `faiss_kmeans`, `kmeans`) place the items, and
    /// the same runtime supplies the completion model the
    /// [`BuildOptions::shaping`] stage asks. A rule strategy still places items exactly as it
    /// does in [`IndexBuilder::build`], and the runtime's fallback mode decides whether a
    /// missing or failing model degrades to the deterministic rule or surfaces an error.
    ///
    /// # Errors
    ///
    /// The same failures as [`IndexBuilder::build`], plus the runtime strategy's own errors:
    /// [`Error::Config`] for a missing model or embedder under
    /// [`FallbackMode::Error`](crate::llm::FallbackMode::Error), the model's
    /// [`Error::Llm`], and the shaping stage's errors — a repair that cannot fix the tree is
    /// [`Error::Validation`] naming [`TREE_INDEX_FILENAME`].
    pub async fn build_with_runtime(
        sources: ItemSources,
        output_dir: impl AsRef<Path>,
        config: BuildConfig,
        runtime: &GroupingRuntime,
    ) -> Result<BuildReport> {
        Self::run(sources, output_dir, config, Some(runtime)).await
    }

    /// The one full-build path both public entry points share.
    ///
    /// `runtime: None` keeps the synchronous placement path (and its typed deferral);
    /// `runtime: Some` awaits the runtime strategies and supplies the shaping model.
    async fn run(
        sources: ItemSources,
        output_dir: impl AsRef<Path>,
        config: BuildConfig,
        runtime: Option<&GroupingRuntime>,
    ) -> Result<BuildReport> {
        config.validate()?;
        let ItemSources {
            item_type: source_item_type,
            items,
            diagnostics,
            item_paths,
        } = sources;
        let item_type = resolve_item_type(&source_item_type, &config)?;
        let mut tree_diagnostics = Vec::new();
        if let Some(preset) = config.preset.as_ref() {
            tree_diagnostics.extend(check_tree(config.schema.validate_preset(preset))?);
        }
        let grouping = config.effective_grouping();
        let (mut assignments, grouping_provenance) =
            Self::place(&items, &grouping, runtime).await?;
        let shaping = &config.options.shaping;
        let model = runtime.and_then(GroupingRuntime::model);
        let mut shaping_provenance = ShapingProvenance::Skipped;
        // `Repair` refuses to run before assembly: it needs the node list and the schema
        // findings, so it runs below instead.
        if matches!(
            shaping.mode,
            ShapingMode::OneShot | ShapingMode::Expansion | ShapingMode::Equivalence
        ) {
            let outcome = shaping::apply(
                shaping.mode,
                &items,
                &assignments,
                &grouping,
                shaping,
                model,
            )
            .await?;
            assignments = outcome.assignments;
            shaping_provenance = outcome.provenance;
        }
        let mut nodes = tree::build_nodes(&items, &assignments, &grouping.root_segment)?;
        let findings = config.schema.validate(&nodes);
        let fatal = findings
            .iter()
            .any(|diagnostic| FATAL_TREE_CODES.contains(&diagnostic.code.as_str()));
        if fatal && shaping.mode == ShapingMode::Repair {
            nodes = repair_tree(&nodes, &findings, &config.schema, shaping, model)
                .await
                .map_err(|error| {
                    validation(
                        TREE_INDEX_FILENAME,
                        format!("tree shaping repair failed: {error}"),
                    )
                })?;
            shaping_provenance = repair_provenance(model, shaping);
            tree_diagnostics.extend(check_tree(config.schema.validate(&nodes))?);
        } else {
            tree_diagnostics.extend(check_tree(findings)?);
        }
        let records = records_for_nodes(&nodes, &items);

        let output_dir = output_dir.as_ref().to_path_buf();
        let html_written = config.options.write_html;
        let builder = Self {
            nodes,
            records,
            item_paths: normalized_paths(item_paths),
            item_type: item_type.clone(),
            config,
        };
        let manifest = builder.write(&output_dir, MODE_FULL).await?;
        Ok(BuildReport {
            output_dir,
            item_type,
            node_count: builder.nodes.len(),
            record_count: builder.records.len(),
            manifest,
            diagnostics,
            tree_diagnostics,
            grouping_provenance,
            shaping_provenance,
            html_written,
        })
    }

    /// Places `items` through the synchronous rule or through `runtime`, with provenance.
    ///
    /// `None` is [`group_items`](crate::retrieval::build::grouping::group_items): the rule
    /// strategies place the items and the runtime strategies keep their typed deferral.
    /// `Some` is [`group_items_with`](crate::retrieval::build::grouping::group_items_with).
    async fn place(
        items: &[SourceItem],
        grouping: &GroupingConfig,
        runtime: Option<&GroupingRuntime>,
    ) -> Result<(Vec<GroupAssignment>, GroupingProvenance)> {
        let Some(runtime) = runtime else {
            return Ok((
                grouping::group_items(items, grouping)?,
                GroupingProvenance::Rule,
            ));
        };
        let outcome = grouping::group_items_with(items, grouping, runtime).await?;
        Ok((outcome.assignments, outcome.provenance))
    }

    /// Hydrates a builder from an already published index directory.
    ///
    /// The index is read and validated with
    /// [`load_index`] (which rejects an inconsistent
    /// manifest), then `tree_index.yaml` is parsed again for the node list, because the
    /// loaded tree is already nested into
    /// [`RetrieverNode`](crate::retrieval::types::RetrieverNode)s. Records and item paths
    /// come from the loaded index. `config` stays the builder's configuration; a manifest
    /// item type the port cannot normalize falls back to it with a warning.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid `config` or an unusable document,
    /// [`Error::NotFound`] when the directory holds no index, and [`Error::External`] for an
    /// I/O failure.
    pub async fn open(base_index_dir: impl AsRef<Path>, config: BuildConfig) -> Result<Self> {
        config.validate()?;
        let dir = base_index_dir.as_ref();
        let loaded = load_index(dir).await?;
        let text = read_index_text(dir, TREE_INDEX_FILENAME).await?;
        let nodes = parse_tree_yaml(&text)?;
        let item_type =
            if let Ok(item_type) = scanner::normalize_item_type(&loaded.manifest.item_type) {
                item_type
            } else {
                warn!(
                    item_type = %loaded.manifest.item_type,
                    "the manifest item type is unusable; falling back to the configuration"
                );
                scanner::normalize_item_type(&config.item_type)?
            };
        debug!(
            nodes = nodes.len(),
            records = loaded.records.len(),
            "hydrated an index for incremental builds"
        );
        Ok(Self {
            nodes,
            records: loaded.records,
            item_paths: loaded.manifest.item_paths,
            item_type,
            config,
        })
    }

    /// The tree nodes, sorted by `(depth, cid)`.
    #[must_use]
    pub fn nodes(&self) -> &[IndexNode] {
        &self.nodes
    }

    /// The catalog records, sorted by `cid`.
    #[must_use]
    pub fn records(&self) -> &[CatalogRecord] {
        &self.records
    }

    /// The scanned item paths the manifest records, in first-seen order.
    #[must_use]
    pub fn item_paths(&self) -> &[String] {
        &self.item_paths
    }

    /// The normalized item type of the index.
    #[must_use]
    pub fn item_type(&self) -> &str {
        &self.item_type
    }

    /// The configuration every patch follows.
    #[must_use]
    pub fn config(&self) -> &BuildConfig {
        &self.config
    }

    /// Adds or replaces items, one at a time in `worker_id` order.
    ///
    /// Each item is placed with the effective grouping, its branch chain is created when
    /// missing, and its leaf plus catalog record are appended. Re-adding a worker id
    /// replaces the item: its previous leaf (and any branch only it filled) is pruned
    /// first, so no stale leaf survives the patch.
    ///
    /// A patch is all or nothing: every item is placed with the grouping stage *before*
    /// the first node or record is touched, so a strategy that cannot run offline leaves
    /// the builder exactly as it was instead of dropping the item it failed on.
    /// Placement runs the synchronous grouping path, so a runtime strategy keeps its typed
    /// [`Error::Unsupported`] here; use [`IndexBuilder::add_with_runtime`] for those.
    ///
    /// The item paths of `sources` are appended to the builder's inventory with
    /// order-preserving dedupe.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid configuration, and the grouping stage's errors.
    pub async fn add(&mut self, sources: &ItemSources) -> Result<IncrementalReport> {
        self.patch(sources, None).await
    }

    /// Adds or replaces items, placing every one of them through `runtime`.
    ///
    /// [`IndexBuilder::add`] with a [`GroupingRuntime`]: each item is placed with
    /// [`group_items_with`](crate::retrieval::build::grouping::group_items_with), so the
    /// runtime strategies work per item. The patch stays all-or-nothing — every item is
    /// placed before the first node or record is touched — and it is placement-only: the
    /// shaping stage runs on a full build, never on a patch.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid configuration, and the runtime strategy's
    /// placement errors (see [`IndexBuilder::build_with_runtime`]).
    pub async fn add_with_runtime(
        &mut self,
        sources: &ItemSources,
        runtime: &GroupingRuntime,
    ) -> Result<IncrementalReport> {
        self.patch(sources, Some(runtime)).await
    }

    /// The one patch path both add entry points share.
    ///
    /// `runtime: None` keeps the synchronous placement path (and its typed deferral);
    /// `runtime: Some` places each item through it.
    async fn patch(
        &mut self,
        sources: &ItemSources,
        runtime: Option<&GroupingRuntime>,
    ) -> Result<IncrementalReport> {
        self.config.validate()?;
        let records_before = self.records.len();
        let grouping = self.config.effective_grouping();
        let mut ordered: Vec<&SourceItem> = sources
            .items
            .iter()
            .filter(|item| !item.worker_id.trim().is_empty())
            .collect();
        ordered.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));

        // Placement is the only fallible stage of a patch, so it runs for every new item
        // first; the mutations below cannot fail.
        let mut placements: Vec<(&SourceItem, Vec<String>)> = Vec::with_capacity(ordered.len());
        for item in ordered {
            let (assignments, _) =
                Self::place(std::slice::from_ref(item), &grouping, runtime).await?;
            let path = assignments
                .first()
                .map_or_else(Vec::new, |assignment| assignment.path.clone());
            placements.push((item, path));
        }

        let mut added: Vec<String> = Vec::with_capacity(placements.len());
        for (item, path) in placements {
            let worker_id = item.worker_id.trim().to_string();
            self.drop_worker(&worker_id);
            let parent = tree::branch_chain(&mut self.nodes, &path, &grouping.root_segment);
            let mut used: BTreeSet<String> =
                self.nodes.iter().map(|node| node.cid.clone()).collect();
            let leaf = tree::leaf_node(item, &parent, &mut used);
            self.records.push(catalog_record(&leaf, item));
            self.nodes.push(leaf);
            added.push(worker_id);
        }
        sort_nodes(&mut self.nodes);
        sort_records(&mut self.records);
        for path in &sources.item_paths {
            push_path(&mut self.item_paths, path);
        }
        added.sort();
        added.dedup();

        let change_ratio = change_ratio(added.len(), records_before);
        Ok(IncrementalReport {
            added,
            removed: Vec::new(),
            rebuilt_branches: Vec::new(),
            escalated: change_ratio > self.config.escalation_ratio,
            change_ratio,
        })
    }

    /// Removes items by worker id and prunes the branches they emptied.
    ///
    /// An id that names neither a catalog record nor a leaf is ignored, so a caller can
    /// replay a deletion. The matching item paths are dropped too: one whose last path
    /// segment is a removed worker id, and the `jsonl://skill/<id>` form.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the configuration is invalid.
    pub async fn delete(&mut self, item_ids: &[String]) -> Result<IncrementalReport> {
        self.config.validate()?;
        let records_before = self.records.len();
        let mut known: BTreeSet<&str> = self
            .records
            .iter()
            .map(|record| record.worker_id.as_str())
            .collect();
        known.extend(
            self.nodes
                .iter()
                .filter(|node| node.is_leaf())
                .map(|node| node.worker_id.as_str()),
        );
        let mut removed: BTreeSet<String> = BTreeSet::new();
        for item_id in item_ids {
            let candidate = item_id.trim();
            if !candidate.is_empty() && known.contains(candidate) {
                removed.insert(candidate.to_string());
            }
        }
        let rebuilt_branches = if removed.is_empty() {
            Vec::new()
        } else {
            tree::prune_workers(&mut self.nodes, &removed)
        };
        self.records
            .retain(|record| !removed.contains(&record.worker_id));
        self.item_paths
            .retain(|path| !names_removed_item(path, &removed));
        sort_records(&mut self.records);

        let change_ratio = change_ratio(removed.len(), records_before);
        Ok(IncrementalReport {
            added: Vec::new(),
            removed: removed.into_iter().collect(),
            rebuilt_branches,
            escalated: change_ratio > self.config.escalation_ratio,
            change_ratio,
        })
    }

    /// Re-derives the children of `node_id` from the catalog records below it.
    ///
    /// Python asks a model to repair a branch; the port re-runs this stage's own grouping
    /// and assembly over the branch's records, so the answer is deterministic and offline.
    /// Each record becomes a [`SourceItem`] whose `classification` is its `cid` segments
    /// below `node_id`, grouped under `node_id` as the root; the produced subtree replaces
    /// the old one in place. Every affected record is then re-keyed from its new leaf —
    /// `cid`, `branch_path`, `category` and `retrieval_text` move with the node, while
    /// `name`, `description`, `skill_path` and `metadata` are preserved.
    ///
    /// A preset taxonomy describes whole-tree paths, so it is not applied below a nested
    /// root; the records' own path segments decide the branches instead. Rebuilding the
    /// same branch twice reproduces the same tree.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when `node_id` is not an existing branch,
    /// [`Error::Validation`] for an invalid configuration, and the grouping stage's errors.
    pub async fn rebuild_subtree(&mut self, node_id: &str) -> Result<IncrementalReport> {
        self.config.validate()?;
        let node_id = node_id.trim();
        let Some(branch) = self
            .nodes
            .iter()
            .find(|node| node.cid == node_id && !node.is_leaf())
        else {
            return Err(Error::not_found(format!("registry branch `{node_id}`")));
        };
        let branch_cid = branch.cid.clone();
        let prefix = format!("{branch_cid}.");
        let affected: Vec<CatalogRecord> = self
            .records
            .iter()
            .filter(|record| record.cid.starts_with(&prefix))
            .cloned()
            .collect();
        if affected.is_empty() {
            debug!(branch = %branch_cid, "no catalog record below this branch; nothing to rebuild");
            return Ok(IncrementalReport::default());
        }

        let items: Vec<SourceItem> = affected
            .iter()
            .map(|record| self.subtree_item(record, &branch_cid))
            .collect();
        let mut grouping = self.config.effective_grouping();
        grouping.root_segment.clone_from(&branch_cid);
        grouping.taxonomy.clear();
        let assignments = grouping::group_items(&items, &grouping)?;
        let produced: Vec<IndexNode> = tree::build_nodes(&items, &assignments, &branch_cid)?
            .into_iter()
            .filter(|node| node.cid != branch_cid)
            .collect();

        let rebuilt_branches: Vec<String> = produced
            .iter()
            .filter(|node| node.is_branch())
            .map(|node| node.cid.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let leaf_cids: BTreeMap<String, String> = produced
            .iter()
            .filter(|node| node.is_leaf())
            .map(|node| (node.worker_id.clone(), node.cid.clone()))
            .collect();
        self.nodes
            .retain(|node| node.cid == branch_cid || !node.cid.starts_with(&prefix));
        self.nodes.extend(produced);
        sort_nodes(&mut self.nodes);

        let mut rebuilt: Vec<CatalogRecord> = Vec::with_capacity(self.records.len());
        for record in std::mem::take(&mut self.records) {
            if !record.cid.starts_with(&prefix) {
                rebuilt.push(record);
                continue;
            }
            if let Some(cid) = leaf_cids.get(&record.worker_id) {
                rebuilt.push(rekey_record(&record, cid));
            } else {
                warn!(
                    worker_id = %record.worker_id,
                    branch = %branch_cid,
                    "no rebuilt leaf for this record; dropping it"
                );
            }
        }
        self.records = rebuilt;
        sort_records(&mut self.records);

        Ok(IncrementalReport {
            added: Vec::new(),
            removed: Vec::new(),
            rebuilt_branches,
            escalated: false,
            change_ratio: 0.0,
        })
    }

    /// The manifest this builder would publish for `mode`.
    ///
    /// `count` is the record count, `worker_ids` follows the records' `cid` order, and
    /// `content_hash` is the digest of the catalog those records render to.
    #[must_use]
    pub fn manifest(&self, mode: &str) -> IndexManifest {
        let records = sorted_records(&self.records);
        IndexManifest {
            mode: mode.to_string(),
            count: records.len(),
            item_paths: self.item_paths.clone(),
            worker_ids: records
                .iter()
                .map(|record| record.worker_id.clone())
                .collect(),
            item_type: self.item_type.clone(),
            content_hash: catalog_content_hash(&records),
        }
    }

    /// Sorts and publishes the artifact set, returning the manifest written.
    ///
    /// The nodes are sorted by `(depth, cid)` and the records by `cid` as part of the
    /// write, so the bytes on disk do not depend on the order the patches produced. The
    /// configuration and the item type are validated first, so a builder assembled without
    /// [`BuildConfig::validate`] cannot publish an index that
    /// [`load_index`] would refuse. [`write_index`] creates the directory,
    /// validates the set before touching it, and replaces each file through a temp sibling
    /// plus rename, so a failed write leaves the previous index in place.
    ///
    /// When [`BuildOptions::write_html`] is set, the `tree_index.html` visualizer is written
    /// after those three files, through its own temp sibling plus rename; a failure there
    /// propagates but cannot corrupt the index already in place, and with the flag off the
    /// published bytes are exactly the three files.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid configuration, an unusable item type, or a
    /// malformed node or record set, [`Error::Config`] for an unusable target path, and
    /// [`Error::External`] for an I/O failure.
    pub async fn write(&self, output_dir: impl AsRef<Path>, mode: &str) -> Result<IndexManifest> {
        self.config.validate()?;
        let item_type = scanner::normalize_item_type(&self.item_type)?;
        let mut nodes = self.nodes.clone();
        sort_nodes(&mut nodes);
        let records = sorted_records(&self.records);
        let mut manifest = self.manifest(mode);
        manifest.item_type = item_type;
        write_index(output_dir.as_ref(), &nodes, &records, &manifest).await?;
        if self.config.options.write_html {
            visualizer::write_tree_html(
                output_dir.as_ref(),
                &nodes,
                &records,
                &self.config.grouping.root_segment,
            )
            .await?;
        }
        Ok(manifest)
    }

    /// Drops the leaf and the catalog record of `worker_id`, plus the branches it emptied.
    fn drop_worker(&mut self, worker_id: &str) {
        let removed: BTreeSet<String> = std::iter::once(worker_id.to_string()).collect();
        tree::prune_workers(&mut self.nodes, &removed);
        self.records.retain(|record| record.worker_id != worker_id);
    }

    /// Synthesizes the build input of one catalog record placed below `branch_cid`.
    fn subtree_item(&self, record: &CatalogRecord, branch_cid: &str) -> SourceItem {
        let classification = record
            .cid
            .strip_prefix(branch_cid)
            .map_or("", |rest| rest.trim_start_matches('.'))
            .to_string();
        let mut metadata: BTreeMap<String, String> = BTreeMap::new();
        for key in [META_SELECT_WHEN, META_DONT_SELECT_WHEN] {
            let hint = record.metadata_of(key);
            if !hint.is_empty() {
                metadata.insert(key.to_string(), hint.to_string());
            }
        }
        SourceItem {
            worker_id: record.worker_id.clone(),
            item: RetrieverItem::new(record.worker_id.clone(), record.cid.clone()),
            name: record.name.clone(),
            description: record.description.clone(),
            content: record.metadata_of(META_CONTENT).to_string(),
            skill_path: record.skill_path.clone(),
            classification,
            tags: Vec::new(),
            item_type: self.item_type.clone(),
            content_hash: String::new(),
            metadata,
        }
    }
}

/// Splits the schema findings into the fatal ones (returned as an error) and the advisory
/// ones (logged and returned for the report).
fn check_tree(diagnostics: Vec<TreeDiagnostic>) -> Result<Vec<TreeDiagnostic>> {
    let mut advisories: Vec<TreeDiagnostic> = Vec::new();
    for diagnostic in diagnostics {
        if FATAL_TREE_CODES.contains(&diagnostic.code.as_str()) {
            return Err(validation(
                TREE_INDEX_FILENAME,
                format!(
                    "{} (cid `{}`): {}",
                    diagnostic.code, diagnostic.cid, diagnostic.detail
                ),
            ));
        }
        warn!(
            code = %diagnostic.code,
            cid = %diagnostic.cid,
            detail = %diagnostic.detail,
            "advisory tree finding"
        );
        advisories.push(diagnostic);
    }
    Ok(advisories)
}

/// The provenance a post-assembly repair reports.
///
/// [`repair`](crate::retrieval::build::tree::shaping::repair::repair) answers only the
/// repaired node list, so which of its two paths produced the accepted tree is not
/// observable here: a model that was supplied together with a non-zero call budget is
/// reported as [`ShapingProvenance::Model`], everything else as
/// [`ShapingProvenance::DeterministicFallback`].
fn repair_provenance(
    model: Option<&dyn CompletionModel>,
    shaping: &ShapingConfig,
) -> ShapingProvenance {
    if model.is_some() && shaping.max_calls > 0 {
        ShapingProvenance::Model
    } else {
        ShapingProvenance::DeterministicFallback
    }
}

/// The item type a build records: the sources' type when it names one, else `config`'s.
fn resolve_item_type(source_item_type: &str, config: &BuildConfig) -> Result<String> {
    let raw = if source_item_type.trim().is_empty() {
        config.item_type.as_str()
    } else {
        source_item_type
    };
    scanner::normalize_item_type(raw)
}

/// `changed / before.max(1)`, the ratio an incremental patch reports.
#[allow(clippy::cast_precision_loss)] // record counts are far below 2^53
fn change_ratio(changed: usize, before: usize) -> f64 {
    changed as f64 / before.max(1) as f64
}

/// Whether `path` names one of the removed items, in either inventory form.
fn names_removed_item(path: &str, removed: &BTreeSet<String>) -> bool {
    if removed
        .iter()
        .any(|item_id| path == format!("{JSONL_PATH_PREFIX}{item_id}"))
    {
        return true;
    }
    path.rsplit('/')
        .next()
        .is_some_and(|segment| removed.contains(segment))
}

/// Appends `path` to `paths` unless it is blank or already present.
fn push_path(paths: &mut Vec<String>, path: &str) {
    let path = path.trim();
    if path.is_empty() || paths.iter().any(|existing| existing == path) {
        return;
    }
    paths.push(path.to_string());
}

/// Trims, drops blanks and de-duplicates `paths`, keeping first-seen order.
fn normalized_paths(paths: Vec<String>) -> Vec<String> {
    let mut normalized: Vec<String> = Vec::new();
    for path in paths {
        push_path(&mut normalized, &path);
    }
    normalized
}

/// Reads one index file, mapping a missing file to [`Error::NotFound`] naming it.
async fn read_index_text(dir: &Path, file_name: &str) -> Result<String> {
    let path = dir.join(file_name);
    match fs::read_to_string(&path).await {
        Ok(text) => Ok(text),
        Err(error) if error.kind() == IoErrorKind::NotFound => Err(Error::not_found(file_name)),
        Err(error) => {
            Err(Error::external(error).with_context(format!("reading `{}`", path.display())))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    use std::sync::Arc;

    use openjiuwen_retrieval::{HashEmbedding, StaticCompletionModel};
    use tempfile::tempdir;

    use super::*;
    use crate::error::ErrorKind;
    use crate::llm::FallbackMode;
    use crate::retrieval::build::grouping::GroupingStrategy;
    use crate::retrieval::build::presets::PresetNode;
    use crate::retrieval::types::{
        CatalogRecord, CATALOG_FILENAME, MANIFEST_FILENAME, MODE_INCREMENTAL, NODE_TYPE_BRANCH,
        TREE_HTML_FILENAME,
    };

    fn item(worker_id: &str, name: &str, classification: &str) -> SourceItem {
        SourceItem {
            worker_id: worker_id.to_string(),
            item: RetrieverItem::new(worker_id, worker_id),
            name: name.to_string(),
            description: format!("{name} routes matching requests."),
            content: format!("{name} usage notes."),
            skill_path: format!("skills/{worker_id}/SKILL.md"),
            classification: classification.to_string(),
            tags: Vec::new(),
            item_type: "skill".to_string(),
            content_hash: String::new(),
            metadata: BTreeMap::new(),
        }
    }

    fn sources(items: Vec<SourceItem>) -> ItemSources {
        let mut item_paths: Vec<String> = Vec::new();
        for entry in &items {
            push_path(&mut item_paths, &format!("store/{}", entry.worker_id));
        }
        ItemSources {
            item_type: "skill".to_string(),
            items,
            diagnostics: Vec::new(),
            item_paths,
        }
    }

    fn three_items() -> Vec<SourceItem> {
        vec![
            item("PdfFill", "PDF Fill", "documents/pdf"),
            item("DocRead", "Doc Reader", "documents/pdf"),
            item("WebSearch", "Web Search", "web"),
        ]
    }

    fn cids(builder: &IndexBuilder) -> Vec<String> {
        builder
            .nodes()
            .iter()
            .map(|node| node.cid.clone())
            .collect()
    }

    async fn build_three(dir: &Path) -> IndexBuilder {
        IndexBuilder::build(sources(three_items()), dir, BuildConfig::new())
            .await
            .unwrap();
        IndexBuilder::open(dir, BuildConfig::new()).await.unwrap()
    }

    #[tokio::test]
    async fn full_build_publishes_the_three_artifacts() {
        let dir = tempdir().unwrap();
        let mut input = sources(three_items());
        input.diagnostics.push(SourceDiagnostic::warning(
            "blank_item",
            "DocRead",
            "only its id",
        ));
        let report = IndexBuilder::build(input, dir.path(), BuildConfig::new())
            .await
            .unwrap();

        assert_eq!(report.output_dir, dir.path());
        assert_eq!(report.item_type, "skill");
        assert_eq!(report.node_count, 7, "4 branches plus 3 leaves");
        assert_eq!(report.record_count, 3);
        assert_eq!(report.manifest.mode, MODE_FULL);
        assert_eq!(report.manifest.count, 3);
        assert_eq!(report.manifest.item_type, "skill");
        assert_eq!(
            report.manifest.worker_ids,
            vec!["DocRead", "PdfFill", "WebSearch"],
            "records are listed in cid order"
        );
        assert_eq!(
            report.manifest.item_paths,
            vec!["store/PdfFill", "store/DocRead", "store/WebSearch"]
        );
        assert_ne!(report.manifest.content_hash, "");
        assert_eq!(report.diagnostics.len(), 1);
        assert_eq!(report.tree_diagnostics, Vec::new());

        for name in [TREE_INDEX_FILENAME, CATALOG_FILENAME, MANIFEST_FILENAME] {
            assert!(dir.path().join(name).exists(), "{name} was not written");
        }

        let loaded = load_index(dir.path()).await.unwrap();
        assert_eq!(loaded.records.len(), 3);
        assert_eq!(loaded.manifest.count, 3);
        let text = fs::read_to_string(dir.path().join(TREE_INDEX_FILENAME))
            .await
            .unwrap();
        let nodes = parse_tree_yaml(&text).unwrap();
        assert_eq!(nodes.iter().filter(|node| node.is_leaf()).count(), 3);
        for record in &loaded.records {
            let node = nodes
                .iter()
                .find(|node| node.cid == record.cid)
                .expect("every record names a node");
            assert!(node.is_leaf(), "{} is not a leaf", record.cid);
            assert_eq!(node.worker_id, record.worker_id);
            assert_eq!(record.category, record.branch_path.join("."));
        }
        assert!(nodes
            .iter()
            .any(|node| node.cid == "Skills.documents.pdf.pdffill"));
        assert!(nodes.iter().any(|node| node.cid == "Skills.web"));
    }

    #[tokio::test]
    async fn two_full_builds_are_byte_identical() {
        let first = tempdir().unwrap();
        let second = tempdir().unwrap();
        IndexBuilder::build(sources(three_items()), first.path(), BuildConfig::new())
            .await
            .unwrap();
        let mut reversed = three_items();
        reversed.reverse();
        IndexBuilder::build(sources(reversed), second.path(), BuildConfig::new())
            .await
            .unwrap();

        for name in [TREE_INDEX_FILENAME, CATALOG_FILENAME] {
            let left = fs::read(first.path().join(name)).await.unwrap();
            let right = fs::read(second.path().join(name)).await.unwrap();
            assert_eq!(left, right, "{name} differs between two builds");
        }
    }

    #[tokio::test]
    async fn open_and_add_appends_a_leaf_and_a_record() {
        let dir = tempdir().unwrap();
        let mut builder = build_three(dir.path()).await;
        assert_eq!(builder.item_type(), "skill");
        assert_eq!(builder.item_paths().len(), 3);
        let original_cids: BTreeSet<String> =
            builder.nodes().iter().map(|n| n.cid.clone()).collect();

        let added = sources(vec![item("ImageGen", "Image Generator", "media/images")]);
        let patch = builder.add(&added).await.unwrap();

        assert_eq!(patch.added, vec!["ImageGen"]);
        assert_eq!(patch.removed, Vec::<String>::new());
        assert_eq!(patch.rebuilt_branches, Vec::<String>::new());
        assert!((patch.change_ratio - 1.0 / 3.0).abs() < 1e-9);
        assert!(patch.escalated, "one of three records exceeds 0.25");
        assert!(builder
            .nodes()
            .iter()
            .any(|node| node.is_leaf() && node.worker_id == "ImageGen"));
        assert_eq!(builder.records().len(), 4);
        assert!(builder
            .item_paths()
            .iter()
            .any(|path| path == "store/ImageGen"));
        for cid in &original_cids {
            assert!(
                builder.nodes().iter().any(|node| &node.cid == cid),
                "{cid} vanished"
            );
        }

        let manifest = builder.write(dir.path(), MODE_INCREMENTAL).await.unwrap();
        assert_eq!(manifest.mode, MODE_INCREMENTAL);
        assert_eq!(manifest.count, 4);
        assert_eq!(
            load_index(dir.path()).await.unwrap().records.len(),
            4,
            "the published catalog carries the new record"
        );
    }

    #[tokio::test]
    async fn add_replaces_an_existing_worker_id() {
        let dir = tempdir().unwrap();
        let mut builder = build_three(dir.path()).await;
        let previous = builder
            .records()
            .iter()
            .find(|record| record.worker_id == "PdfFill")
            .expect("PdfFill was indexed")
            .cid
            .clone();

        let patch = builder
            .add(&sources(vec![item("PdfFill", "PDF Fill", "media/images")]))
            .await
            .unwrap();

        assert_eq!(patch.added, vec!["PdfFill"]);
        assert_eq!(builder.records().len(), 3, "a re-add replaces, not appends");
        let record = builder
            .records()
            .iter()
            .find(|record| record.worker_id == "PdfFill")
            .expect("PdfFill is still indexed");
        assert_ne!(record.cid, previous);
        assert_eq!(record.cid, "Skills.media.images.pdffill");
        assert_eq!(record.branch_path, vec!["Skills", "media", "images"]);
        assert!(
            builder.nodes().iter().all(|node| node.cid != previous),
            "the old leaf was pruned"
        );
        assert_eq!(
            builder
                .nodes()
                .iter()
                .filter(|node| node.is_leaf() && node.worker_id == "PdfFill")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn delete_prunes_the_leaf_its_branch_and_the_record() {
        let dir = tempdir().unwrap();
        let mut input = sources(three_items());
        push_path(&mut input.item_paths, "jsonl://skill/WebSearch");
        IndexBuilder::build(input, dir.path(), BuildConfig::new())
            .await
            .unwrap();
        let mut builder = IndexBuilder::open(dir.path(), BuildConfig::new())
            .await
            .unwrap();

        let patch = builder.delete(&["WebSearch".to_string()]).await.unwrap();

        assert_eq!(patch.removed, vec!["WebSearch"]);
        assert_eq!(patch.added, Vec::<String>::new());
        assert_eq!(patch.rebuilt_branches, vec!["Skills.web"]);
        assert!(patch.escalated);
        assert!(builder
            .records()
            .iter()
            .all(|record| record.worker_id != "WebSearch"));
        assert_eq!(builder.records().len(), 2);
        assert!(builder.nodes().iter().all(|node| node.cid != "Skills.web"));
        assert!(builder
            .nodes()
            .iter()
            .all(|node| node.worker_id != "WebSearch"));
        assert_eq!(
            builder.item_paths(),
            &["store/PdfFill".to_string(), "store/DocRead".to_string()]
        );

        let noop = builder.delete(&["Absent".to_string()]).await.unwrap();
        assert_eq!(noop.removed, Vec::<String>::new());
        assert!(!noop.escalated);
        assert_eq!(noop.change_ratio, 0.0);
        assert_eq!(builder.records().len(), 2);
    }

    #[tokio::test]
    async fn escalation_flags_a_large_change_and_ignores_a_small_one() {
        let small = tempdir().unwrap();
        let mut builder = build_three(small.path()).await;
        let patch = builder.delete(&["DocRead".to_string()]).await.unwrap();
        assert!(patch.escalated, "one of three records exceeds 0.25");
        assert!((patch.change_ratio - 1.0 / 3.0).abs() < 1e-9);

        let big = tempdir().unwrap();
        let items: Vec<SourceItem> = (0..8)
            .map(|index| {
                item(
                    &format!("tool{index}"),
                    &format!("Reading {index}"),
                    "documents",
                )
            })
            .collect();
        IndexBuilder::build(sources(items), big.path(), BuildConfig::new())
            .await
            .unwrap();
        let mut builder = IndexBuilder::open(big.path(), BuildConfig::new())
            .await
            .unwrap();
        let patch = builder.delete(&["tool3".to_string()]).await.unwrap();
        assert!(!patch.escalated, "one of eight records stays below 0.25");
        assert!((patch.change_ratio - 0.125).abs() < 1e-9);

        let strict = BuildConfig::new().with_escalation_ratio(0.5);
        let mut builder = IndexBuilder::open(big.path(), strict).await.unwrap();
        let patch = builder.delete(&["tool4".to_string()]).await.unwrap();
        assert!(!patch.escalated, "a raised ratio tolerates a third");
    }

    #[tokio::test]
    async fn rebuild_subtree_re_derives_the_children_and_keeps_record_identity() {
        let dir = tempdir().unwrap();
        let flat = vec![
            item("PdfFill", "PDF Fill", "documents"),
            item("DocRead", "Doc Reader", "documents"),
        ];
        IndexBuilder::build(sources(flat), dir.path(), BuildConfig::new())
            .await
            .unwrap();
        let mut builder = IndexBuilder::open(dir.path(), BuildConfig::new())
            .await
            .unwrap();
        let before = builder.records().to_vec();
        assert_eq!(before.len(), 2);
        assert!(before
            .iter()
            .all(|record| record.branch_path == vec!["Skills", "documents"]));

        let patch = builder.rebuild_subtree("Skills.documents").await.unwrap();

        assert_eq!(patch.added, Vec::<String>::new());
        assert_eq!(patch.removed, Vec::<String>::new());
        assert!(!patch.escalated);
        assert_eq!(
            patch.rebuilt_branches,
            vec!["Skills.documents.docread", "Skills.documents.pdffill"]
        );

        let after = builder.records();
        assert_eq!(after.len(), 2);
        for (new, old) in after.iter().zip(before.iter()) {
            assert_eq!(new.worker_id, old.worker_id);
            assert_eq!(new.name, old.name);
            assert_eq!(new.description, old.description);
            assert_eq!(new.skill_path, old.skill_path);
            assert_eq!(new.metadata, old.metadata);
            assert_ne!(new.cid, old.cid, "the leaf moved under a new branch");
            assert!(new.cid.starts_with(&format!("{}.", old.cid)));
            assert_eq!(
                new.category,
                format!("{}.{}", old.category, old.cid.rsplit('.').next().unwrap())
            );
            assert_eq!(
                new.branch_path,
                vec!["Skills", "documents", old.cid.rsplit('.').next().unwrap()]
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<String>>()
            );
            assert!(new.retrieval_text.contains(&new.cid));
        }
        assert!(builder
            .nodes()
            .iter()
            .all(|node| !(node.is_leaf() && node.cid == before[0].cid)));

        let snapshot: Vec<CatalogRecord> = builder.records().to_vec();
        builder.rebuild_subtree("Skills.documents").await.unwrap();
        assert_eq!(
            builder.records(),
            snapshot.as_slice(),
            "a rebuild is stable"
        );
    }

    #[tokio::test]
    async fn rebuild_subtree_rejects_an_unknown_or_leaf_node() {
        let dir = tempdir().unwrap();
        let mut builder = build_three(dir.path()).await;
        let error = builder.rebuild_subtree("Skills.absent").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.message().contains("Skills.absent"));

        let error = builder
            .rebuild_subtree("Skills.web.websearch")
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound, "a leaf is not a branch");

        let empty = tempdir().unwrap();
        let error = IndexBuilder::open(empty.path(), BuildConfig::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn a_preset_taxonomy_wins_over_the_grouping_paths() {
        let dir = tempdir().unwrap();
        let preset = TreePreset {
            nodes: vec![PresetNode {
                cid: "Skills".to_string(),
                node_type: NODE_TYPE_BRANCH.to_string(),
                description: "Skill tree".to_string(),
                children: vec![PresetNode {
                    cid: "Skills.Documents".to_string(),
                    node_type: NODE_TYPE_BRANCH.to_string(),
                    description: "Documents".to_string(),
                    children: vec![PresetNode {
                        cid: "Skills.Documents.Files".to_string(),
                        node_type: NODE_TYPE_BRANCH.to_string(),
                        description: "Files".to_string(),
                        ..PresetNode::default()
                    }],
                    ..PresetNode::default()
                }],
                ..PresetNode::default()
            }],
        };
        let config = BuildConfig::new().with_preset(preset);
        assert_eq!(
            config.effective_grouping().taxonomy,
            vec![
                vec!["Documents".to_string()],
                vec!["Documents".to_string(), "Files".to_string()],
            ],
            "the preset's root segment is stripped: the grouping stage adds it back"
        );

        let report = IndexBuilder::build(
            sources(vec![item("PdfFill", "PDF Fill", "documents/pdf")]),
            dir.path(),
            config,
        )
        .await
        .unwrap();
        assert_eq!(report.record_count, 1);
        let builder = IndexBuilder::open(dir.path(), BuildConfig::new())
            .await
            .unwrap();
        let leaf = builder
            .nodes()
            .iter()
            .find(|node| node.is_leaf())
            .expect("the single item has a leaf");
        assert_eq!(leaf.cid, "Skills.documents.files.pdffill");
        assert!(cids(&builder).contains(&"Skills.documents.files".to_string()));
        assert!(
            !cids(&builder)
                .iter()
                .any(|cid| cid == "Skills.skills" || cid.starts_with("Skills.skills.")),
            "a preset rooted at Skills must not produce doubled cids: {:?}",
            cids(&builder)
        );
    }

    #[tokio::test]
    async fn a_preset_rooted_below_the_root_segment_stays_under_the_root() {
        let dir = tempdir().unwrap();
        let preset = TreePreset {
            nodes: vec![PresetNode {
                cid: "documents".to_string(),
                node_type: NODE_TYPE_BRANCH.to_string(),
                description: "Documents".to_string(),
                children: vec![PresetNode {
                    cid: "documents.pdf-fill".to_string(),
                    node_type: NODE_TYPE_BRANCH.to_string(),
                    description: "PDF forms".to_string(),
                    ..PresetNode::default()
                }],
                ..PresetNode::default()
            }],
        };
        let config = BuildConfig::new().with_preset(preset);
        assert_eq!(
            config.effective_grouping().taxonomy,
            vec![
                vec!["documents".to_string()],
                vec!["documents".to_string(), "pdf-fill".to_string()],
            ],
            "a preset that does not start at the root segment is kept verbatim"
        );

        IndexBuilder::build(
            sources(vec![item("PdfFill", "PDF Fill", "documents/pdf")]),
            dir.path(),
            config,
        )
        .await
        .unwrap();
        let builder = IndexBuilder::open(dir.path(), BuildConfig::new())
            .await
            .unwrap();
        let leaf = builder
            .nodes()
            .iter()
            .find(|node| node.is_leaf())
            .expect("the single item has a leaf");
        assert_eq!(leaf.cid, "Skills.documents.pdf-fill.pdffill");
        assert!(cids(&builder).contains(&"Skills.documents.pdf-fill".to_string()));
    }

    #[tokio::test]
    async fn a_duplicate_preset_cid_aborts_the_build() {
        let dir = tempdir().unwrap();
        let preset = TreePreset {
            nodes: vec![
                PresetNode {
                    cid: "Skills".to_string(),
                    node_type: NODE_TYPE_BRANCH.to_string(),
                    description: "one".to_string(),
                    ..PresetNode::default()
                },
                PresetNode {
                    cid: "Skills".to_string(),
                    node_type: NODE_TYPE_BRANCH.to_string(),
                    description: "two".to_string(),
                    ..PresetNode::default()
                },
            ],
        };
        let error = IndexBuilder::build(
            sources(three_items()),
            dir.path(),
            BuildConfig::new().with_preset(preset),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains(DIAGNOSTIC_DUPLICATE_CID));
        assert!(!dir.path().join(TREE_INDEX_FILENAME).exists());
    }

    #[tokio::test]
    async fn an_advisory_schema_finding_is_reported_not_fatal() {
        let dir = tempdir().unwrap();
        let config = BuildConfig::new().with_schema(TreeSchema::new().with_max_children(1));
        let report = IndexBuilder::build(sources(three_items()), dir.path(), config)
            .await
            .unwrap();
        assert!(report
            .tree_diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code
                == crate::retrieval::build::schema::DIAGNOSTIC_EXCESSIVE_FAN_OUT));
        assert!(dir.path().join(CATALOG_FILENAME).exists());
    }

    #[test]
    fn build_config_validates_its_knobs() {
        assert!(BuildConfig::new().validate().is_ok());
        assert_eq!(
            BuildConfig::new()
                .with_item_type("widget")
                .validate()
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
        let negative = BuildConfig::new().with_escalation_ratio(-0.5);
        assert_eq!(
            negative.validate().unwrap_err().kind(),
            ErrorKind::Validation
        );
        let not_finite = BuildConfig::new().with_escalation_ratio(f64::NAN);
        assert_eq!(
            not_finite.validate().unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            BuildConfig::default().escalation_ratio,
            DEFAULT_ESCALATION_RATIO
        );
        assert_eq!(
            BuildConfig::default().effective_grouping().taxonomy,
            Vec::<Vec<String>>::new()
        );
        assert_eq!(
            IndexBuilder::new(BuildConfig::default()).item_type(),
            "skill"
        );
        assert_eq!(
            IndexBuilder::new(BuildConfig::default()).nodes(),
            &[] as &[IndexNode]
        );
    }

    /// A build with a runtime the k-means strategy needs clusters through the embedder.
    #[tokio::test]
    async fn build_with_runtime_clusters_with_kmeans_and_reports_embedding_provenance() {
        let dir = tempdir().unwrap();
        let config = BuildConfig::new().with_grouping(
            GroupingConfig::default()
                .with_strategy(GroupingStrategy::Kmeans)
                .with_clusters(Some(2)),
        );
        let runtime = GroupingRuntime::new().with_embedding(Arc::new(HashEmbedding::new(8)));
        let report = IndexBuilder::build_with_runtime(
            sources(three_items()),
            dir.path(),
            config.clone(),
            &runtime,
        )
        .await
        .unwrap();

        assert_eq!(report.grouping_provenance, GroupingProvenance::Embedding);
        assert_eq!(report.record_count, 3, "every item keeps a record");
        assert_eq!(
            report.node_count, 6,
            "the root, two cluster branches, three leaves"
        );
        assert_eq!(report.shaping_provenance, ShapingProvenance::Skipped);
        assert!(!report.html_written);

        let builder = IndexBuilder::open(dir.path(), config).await.unwrap();
        let paths: BTreeSet<Vec<String>> = builder
            .records()
            .iter()
            .map(|record| record.branch_path.clone())
            .collect();
        assert_eq!(paths.len(), 2, "two clusters cluster the items: {paths:?}");
        assert!(
            paths.iter().all(|path| path.len() == 2),
            "a cluster is one branch below the root: {paths:?}"
        );

        let deterministic = tempdir().unwrap();
        IndexBuilder::build(
            sources(three_items()),
            deterministic.path(),
            BuildConfig::new(),
        )
        .await
        .unwrap();
        let reference = IndexBuilder::open(deterministic.path(), BuildConfig::new())
            .await
            .unwrap();
        for record in builder.records() {
            let expected = reference
                .records()
                .iter()
                .find(|other| other.worker_id == record.worker_id)
                .expect("the deterministic build indexed the same item");
            assert_ne!(
                record.cid, expected.cid,
                "{} kept the deterministic leaf",
                record.worker_id
            );
        }

        let loaded = load_index(dir.path()).await.unwrap();
        assert_eq!(loaded.records.len(), 3);
        assert_eq!(loaded.manifest.count, 3);
    }

    /// A build with a model runtime places every item where the reply says.
    #[tokio::test]
    async fn build_with_runtime_llm_grouping_places_items_with_the_model() {
        let dir = tempdir().unwrap();
        let model = Arc::new(StaticCompletionModel::single(
            r#"{"groups":[{"path":["tools"],"worker_ids":["DocRead","PdfFill","WebSearch"]}]}"#,
        ));
        let runtime = GroupingRuntime::new().with_model(model.clone());
        let config = BuildConfig::new()
            .with_grouping(GroupingConfig::default().with_strategy(GroupingStrategy::LlmGrouping));
        let report = IndexBuilder::build_with_runtime(
            sources(three_items()),
            dir.path(),
            config.clone(),
            &runtime,
        )
        .await
        .unwrap();

        assert_eq!(report.grouping_provenance, GroupingProvenance::Model);
        assert_eq!(report.record_count, 3);
        assert_eq!(model.call_count(), 1, "one prompt places the inventory");

        let builder = IndexBuilder::open(dir.path(), config).await.unwrap();
        for record in builder.records() {
            assert_eq!(
                record.branch_path,
                vec!["Skills".to_string(), "tools".to_string()],
                "{}",
                record.worker_id
            );
            assert!(record.cid.starts_with("Skills.tools."), "{}", record.cid);
        }
        assert!(cids(&builder).contains(&"Skills.tools".to_string()));
    }

    /// An unusable reply degrades to the deterministic placements under the default mode.
    #[tokio::test]
    async fn build_with_runtime_llm_grouping_falls_back_on_an_unusable_reply() {
        let dir = tempdir().unwrap();
        let model = Arc::new(StaticCompletionModel::single("not json"));
        let runtime = GroupingRuntime::new().with_model(model.clone());
        let config = BuildConfig::new()
            .with_grouping(GroupingConfig::default().with_strategy(GroupingStrategy::LlmGrouping));
        let report =
            IndexBuilder::build_with_runtime(sources(three_items()), dir.path(), config, &runtime)
                .await
                .unwrap();

        assert_eq!(
            report.grouping_provenance,
            GroupingProvenance::DeterministicFallback
        );
        assert_eq!(model.call_count(), 1, "the model was asked once");

        let deterministic = tempdir().unwrap();
        IndexBuilder::build(
            sources(three_items()),
            deterministic.path(),
            BuildConfig::new(),
        )
        .await
        .unwrap();
        for name in [TREE_INDEX_FILENAME, CATALOG_FILENAME, MANIFEST_FILENAME] {
            let fallback = fs::read(dir.path().join(name)).await.unwrap();
            let reference = fs::read(deterministic.path().join(name)).await.unwrap();
            assert_eq!(
                fallback, reference,
                "{name} differs from the deterministic build"
            );
        }
    }

    /// `build` keeps the synchronous placement path, so a runtime strategy stays deferred.
    #[tokio::test]
    async fn build_with_a_runtime_strategy_still_reports_the_sync_deferral() {
        for (strategy, feature) in [
            (GroupingStrategy::Kmeans, "k-means tree grouping"),
            (GroupingStrategy::FaissKmeans, "faiss k-means tree grouping"),
            (GroupingStrategy::LlmGrouping, "LLM-guided tree grouping"),
        ] {
            let dir = tempdir().unwrap();
            let config =
                BuildConfig::new().with_grouping(GroupingConfig::default().with_strategy(strategy));
            let error = IndexBuilder::build(sources(three_items()), dir.path(), config)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported, "{strategy:?}");
            let message = error.message();
            assert!(message.contains(feature), "{strategy:?}: {message}");
            assert!(message.contains("deferred"), "{strategy:?}: {message}");
            assert_eq!(
                std::fs::read_dir(dir.path()).unwrap().count(),
                0,
                "{strategy:?} published an artifact"
            );
        }
    }

    /// A patch places its item through the runtime, under a k-means index.
    #[tokio::test]
    async fn add_with_runtime_places_an_item_under_a_kmeans_index() {
        let dir = tempdir().unwrap();
        let config = BuildConfig::new()
            .with_grouping(GroupingConfig::default().with_strategy(GroupingStrategy::Kmeans));
        let runtime = GroupingRuntime::new().with_embedding(Arc::new(HashEmbedding::new(8)));
        IndexBuilder::build_with_runtime(
            sources(three_items()),
            dir.path(),
            config.clone(),
            &runtime,
        )
        .await
        .unwrap();
        let mut builder = IndexBuilder::open(dir.path(), config).await.unwrap();

        let patch = builder
            .add_with_runtime(
                &sources(vec![item("ImageGen", "Image Generator", "media/images")]),
                &runtime,
            )
            .await
            .unwrap();

        assert_eq!(patch.added, vec!["ImageGen"]);
        assert!(patch.escalated, "one of three records exceeds 0.25");
        assert_eq!(builder.records().len(), 4);
        assert_eq!(
            builder
                .nodes()
                .iter()
                .filter(|node| node.is_leaf() && node.worker_id == "ImageGen")
                .count(),
            1,
            "exactly one leaf carries the new worker id"
        );
        assert_eq!(
            builder
                .records()
                .iter()
                .filter(|record| record.worker_id == "ImageGen")
                .count(),
            1
        );

        builder.write(dir.path(), MODE_INCREMENTAL).await.unwrap();
        let loaded = load_index(dir.path()).await.unwrap();
        assert_eq!(loaded.records.len(), 4);
        assert!(loaded
            .records
            .iter()
            .any(|record| record.worker_id == "ImageGen"));
    }

    /// A pre-assembly shaping stage replaces the deterministic placements.
    #[tokio::test]
    async fn shaping_one_shot_changes_the_placements_and_keeps_every_worker_id() {
        let dir = tempdir().unwrap();
        let model = Arc::new(StaticCompletionModel::single(
            r#"{"assignments":[
                {"worker_id":"DocRead","path":["reading"]},
                {"worker_id":"PdfFill","path":["reading"]},
                {"worker_id":"WebSearch","path":["web","lookup"]}
            ]}"#,
        ));
        let runtime = GroupingRuntime::new().with_model(model);
        let config = BuildConfig::new().with_options(BuildOptions {
            write_html: false,
            shaping: ShapingConfig::new().with_mode(ShapingMode::OneShot),
        });
        let report = IndexBuilder::build_with_runtime(
            sources(three_items()),
            dir.path(),
            config.clone(),
            &runtime,
        )
        .await
        .unwrap();

        assert_eq!(report.shaping_provenance, ShapingProvenance::Model);
        assert_eq!(report.grouping_provenance, GroupingProvenance::Rule);
        assert_eq!(report.record_count, 3);

        let builder = IndexBuilder::open(dir.path(), config).await.unwrap();
        let node_cids = cids(&builder);
        assert!(
            node_cids.contains(&"Skills.reading".to_string()),
            "{node_cids:?}"
        );
        assert!(
            node_cids.contains(&"Skills.web.lookup".to_string()),
            "{node_cids:?}"
        );
        let workers: BTreeSet<String> = builder
            .records()
            .iter()
            .map(|record| record.worker_id.clone())
            .collect();
        assert_eq!(
            workers,
            ["DocRead", "PdfFill", "WebSearch"]
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<String>>(),
            "shaping must never lose an item"
        );
    }

    /// The post-assembly repair path publishes a tree the schema had rejected.
    #[tokio::test]
    async fn shaping_repair_fixes_a_tree_the_schema_rejected() {
        let schema = TreeSchema::new().with_max_depth(2);
        let fatal = tempdir().unwrap();
        let error = IndexBuilder::build(
            sources(three_items()),
            fatal.path(),
            BuildConfig::new().with_schema(schema),
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::Validation,
            "the unrepaired tree really is fatal: {}",
            error.message()
        );
        assert!(error.message().contains(TREE_INDEX_FILENAME));

        let dir = tempdir().unwrap();
        let model = Arc::new(StaticCompletionModel::single(
            r#"{"nodes":[
                {"cid":"Skills","type":"branch"},
                {"cid":"Skills.pdffill","type":"leaf","worker_id":"PdfFill"},
                {"cid":"Skills.docread","type":"leaf","worker_id":"DocRead"},
                {"cid":"Skills.websearch","type":"leaf","worker_id":"WebSearch"}
            ]}"#,
        ));
        let runtime = GroupingRuntime::new().with_model(model);
        let config = BuildConfig::new()
            .with_schema(schema)
            .with_options(BuildOptions {
                write_html: false,
                shaping: ShapingConfig::new().with_mode(ShapingMode::Repair),
            });
        let report = IndexBuilder::build_with_runtime(
            sources(three_items()),
            dir.path(),
            config.clone(),
            &runtime,
        )
        .await
        .unwrap();

        assert_eq!(report.shaping_provenance, ShapingProvenance::Model);
        assert_eq!(report.record_count, 3);
        let builder = IndexBuilder::open(dir.path(), config).await.unwrap();
        assert!(
            schema.validate(builder.nodes()).is_empty(),
            "the published tree satisfies the same schema"
        );
        assert!(cids(&builder).contains(&"Skills.pdffill".to_string()));
    }

    /// A repair the model cannot perform aborts in the established validation shape.
    #[tokio::test]
    async fn shaping_repair_aborts_typed_when_the_model_cannot_fix_the_tree() {
        let dir = tempdir().unwrap();
        let model = Arc::new(StaticCompletionModel::single("not a JSON object"));
        let runtime = GroupingRuntime::new().with_model(model);
        let config = BuildConfig::new()
            .with_schema(TreeSchema::new().with_max_depth(2))
            .with_options(BuildOptions {
                write_html: false,
                shaping: ShapingConfig::new()
                    .with_mode(ShapingMode::Repair)
                    .with_fallback(FallbackMode::Error),
            });
        let error =
            IndexBuilder::build_with_runtime(sources(three_items()), dir.path(), config, &runtime)
                .await
                .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains(TREE_INDEX_FILENAME),
            "{}",
            error.message()
        );
        assert!(
            error.message().contains("tree shaping repair failed"),
            "{}",
            error.message()
        );
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            0,
            "a failed repair publishes nothing"
        );
    }

    /// The visualizer is an opt-in extra next to an unchanged three-artifact set.
    #[tokio::test]
    async fn write_html_publishes_the_visualizer_beside_the_three_artifacts() {
        let dir = tempdir().unwrap();
        let report = IndexBuilder::build(
            sources(three_items()),
            dir.path(),
            BuildConfig::new().with_html(true),
        )
        .await
        .unwrap();
        assert!(report.html_written);

        let html = fs::read_to_string(dir.path().join(TREE_HTML_FILENAME))
            .await
            .unwrap();
        assert!(html.contains("<!DOCTYPE html>"), "{html:.80}");
        assert!(
            html.contains("PDF Fill") && html.contains("Doc Reader"),
            "the visualizer lists the items"
        );
        for name in [TREE_INDEX_FILENAME, CATALOG_FILENAME, MANIFEST_FILENAME] {
            assert!(dir.path().join(name).exists(), "{name} was not written");
        }
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().to_string();
            assert!(
                !std::path::Path::new(&name)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("tmp")),
                "{name} was left behind"
            );
        }

        let plain = tempdir().unwrap();
        let plain_report =
            IndexBuilder::build(sources(three_items()), plain.path(), BuildConfig::new())
                .await
                .unwrap();
        assert!(!plain_report.html_written);
        for name in [TREE_INDEX_FILENAME, CATALOG_FILENAME, MANIFEST_FILENAME] {
            let with_html = fs::read(dir.path().join(name)).await.unwrap();
            let without_html = fs::read(plain.path().join(name)).await.unwrap();
            assert_eq!(with_html, without_html, "{name} differs");
        }
    }

    /// The default configuration publishes exactly the three artifacts.
    #[tokio::test]
    async fn write_html_false_publishes_exactly_the_three_artifacts() {
        let dir = tempdir().unwrap();
        let report = IndexBuilder::build(sources(three_items()), dir.path(), BuildConfig::new())
            .await
            .unwrap();
        assert!(!report.html_written, "the visualizer is opt-in");

        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![CATALOG_FILENAME, MANIFEST_FILENAME, TREE_INDEX_FILENAME]
        );
    }

    /// The options round-trip through JSON and take their defaults when absent.
    #[test]
    fn build_options_round_trip_and_tolerate_a_missing_options_key() {
        let config = BuildConfig::new().with_options(BuildOptions {
            write_html: true,
            shaping: ShapingConfig::new().with_mode(ShapingMode::OneShot),
        });
        let json = serde_json::to_string(&config).unwrap();
        let parsed: BuildConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, config);
        assert!(parsed.options.write_html);
        assert_eq!(parsed.options.shaping.mode, ShapingMode::OneShot);

        let legacy: BuildConfig =
            serde_json::from_str(r#"{"item_type":"skill","escalation_ratio":0.25}"#).unwrap();
        assert_eq!(legacy.options, BuildOptions::default());
        assert!(!legacy.options.write_html);
        assert_eq!(legacy.options.shaping.mode, ShapingMode::None);
    }
}
