//! Grouping of scanned items into tree branches.
//!
//! Ports the grouping stage of Python `openjiuwen.symphony.retrieval.build.tree`
//! (`grouping.py`, `builder.py`, `preset_writer.py`, openJiuwen SDK v0.1.20) as three
//! interchangeable strategy families: inspectable rules, embedding-backed k-means and a
//! model-guided round trip.
//!
//! # Strategy families
//!
//! [`GroupingStrategy`](crate::retrieval::build::grouping::GroupingStrategy) names six
//! strategies, and every one of them is implemented; the strategy table is:
//!
//! | Family | Strategies | Needs | Implementation |
//! | --- | --- | --- | --- |
//! | rule | `Deterministic`, `CategoryPath`, `NameTokens` | nothing | this module |
//! | embedding | `Kmeans`, `FaissKmeans` | an embedding model | [`kmeans`](crate::retrieval::build::grouping::kmeans) |
//! | model | `LlmGrouping` | a completion model | [`llm`](crate::retrieval::build::grouping::llm) |
//!
//! [`requires_runtime`](crate::retrieval::build::grouping::GroupingStrategy::requires_runtime)
//! answers which family needs a supplied runtime.
//! The synchronous [`group_items`](crate::retrieval::build::grouping::group_items) runs only
//! the three rule strategies and reports a typed
//! [`Error::Unsupported`](crate::error::Error::Unsupported) naming the strategy for the three
//! runtime strategies, because a synchronous entry point cannot await a model.
//! The async [`group_items_with`](crate::retrieval::build::grouping::group_items_with) runs
//! all six through a [`GroupingRuntime`](crate::retrieval::build::grouping::GroupingRuntime)
//! and reports how the paths were derived in its
//! [`GroupingProvenance`](crate::retrieval::build::grouping::GroupingProvenance).
//!
//! # Deterministic placement pipeline
//!
//! Every strategy's candidate paths go through one shared, deterministic pipeline (no clock,
//! no hash-map order: items are iterated in `worker_id` order and every map and set here is a
//! `BTreeMap` or a `BTreeSet`):
//!
//! 1. **Candidate segments.** A rule strategy derives them: for `Deterministic` and
//!    `CategoryPath` the path comes from the item's `classification` when non-blank: it
//!    splits on `.`, `/`, `>`, `|` and `,`, every chunk is slugged with
//!    [`slug_term`](crate::retrieval::build::tree_text::slug_term), and blank or repeated
//!    chunks are dropped. A blank classification falls back to the item's first non-blank
//!    tag, then (`Deterministic` only) to the item's leading name token: the first
//!    non-empty token of
//!    [`normalize_name_key`](crate::model::naming::normalize_name_key) that is not a
//!    routing-noise word such as `tool` or `helper`. The `NameTokens` strategy applies
//!    only that last rule. An item with nothing usable is placed under
//!    [`GENERAL_SEGMENT`](crate::retrieval::build::grouping::GENERAL_SEGMENT).
//!    The embedding and model strategies have no rule of their own: their candidates are
//!    the cluster labels of [`kmeans`](crate::retrieval::build::grouping::kmeans) or the
//!    reply's paths from [`llm`](crate::retrieval::build::grouping::llm).
//! 2. **Taxonomy override.** When
//!    [`GroupingConfig::taxonomy`](crate::retrieval::build::grouping::GroupingConfig::taxonomy)
//!    is non-empty, an item is instead assigned to the preset path with the highest token
//!    overlap ([`text_tokens`](crate::retrieval::build::tree_text::text_tokens)) between
//!    its candidate segments and the preset segments, tie-broken by the longer path, then by
//!    the lexicographically smallest joined path. A preset path with no overlap at all
//!    leaves the item directly under the root.
//!    The override applies to every strategy's candidate path, so a runtime strategy is
//!    bound by the same preset as the rule strategies.
//! 3. **Depth truncation.** The path is cut to `max_depth - 1` segments, because the tree
//!    root already occupies depth 1 (Python's `MAX_DEPTH` counts the root the same way).
//! 4. **Segment tree.** Items are inserted into a segment trie keyed by the slugged
//!    segments, so items sharing a prefix share a branch.
//! 5. **Rebalancing.** Each node is fixed bottom-up: more than `max_children` child
//!    branches keep their first `max_children - 1` segments in sorted order and fold the
//!    remainder into one
//!    [`OTHER_SEGMENT`](crate::retrieval::build::grouping::OTHER_SEGMENT) child; more than
//!    `max_children` direct items are bucketed by the next name token, and a bucket that
//!    is still over the bound is chunked into
//!    [`PART_SEGMENT_PREFIX`](crate::retrieval::build::grouping::PART_SEGMENT_PREFIX)
//!    children of at most `max_children` items each. Both bounds need a free level below
//!    the node, so at `max_depth` they are soft — the same place Python's builder stops
//!    splitting.
//!
//! The result is one
//! [`GroupAssignment`](crate::retrieval::build::grouping::GroupAssignment) per item,
//! ordered by `worker_id`, whose `path` excludes the tree root and contains `cid`-ready
//! slug segments — exactly what
//! [`build_nodes`](crate::retrieval::build::tree::build_nodes) consumes.
//!
//! # Fallbacks
//!
//! [`group_items_with`](crate::retrieval::build::grouping::group_items_with) never reports a
//! model or embedding failure as a success: under
//! [`FallbackMode::Deterministic`](crate::llm::FallbackMode::Deterministic) (the default) a
//! missing or failing runtime degrades to the deterministic rule and the outcome's
//! provenance says
//! [`GroupingProvenance::DeterministicFallback`](crate::retrieval::build::grouping::GroupingProvenance::DeterministicFallback),
//! while under [`FallbackMode::Error`](crate::llm::FallbackMode::Error) the failure surfaces —
//! a missing runtime as an [`Error::Config`](crate::error::Error::Config) naming the builder
//! to use, a failing one as its own error.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use openjiuwen_retrieval::Embedding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{validation, Error, Result};
use crate::llm::{CompletionModel, FallbackMode};
use crate::model::naming::normalize_name_key;
use crate::retrieval::build::scanner::SourceItem;
use crate::retrieval::build::tree_text::{slug_term, text_tokens};

/// Embedding-backed k-means clustering of items into tree branches (`GroupingStrategy::{Kmeans, FaissKmeans}`).
pub mod kmeans;
/// Model-guided tree grouping (`GroupingStrategy::LlmGrouping`) with no silent lexical fallback.
pub mod llm;

/// Python's `BRANCHING_FACTOR`: the default number of children one branch may hold.
pub const DEFAULT_MAX_CHILDREN: usize = 8;
/// Python's `MAX_DEPTH`: the default tree depth, counting the root as depth 1.
pub const DEFAULT_MAX_DEPTH: usize = 6;
/// The segment Python's fallback root `cid` uses, and the default tree root here.
pub const DEFAULT_ROOT_SEGMENT: &str = "Skills";
/// The segment items are folded into when a node exceeds its branch fan-out.
pub const OTHER_SEGMENT: &str = "other";
/// The segment an item lands under when no classification, tag or name token applies.
pub const GENERAL_SEGMENT: &str = "general";
/// The prefix of the segments an overflowing leaf is chunked into (`part-1`, `part-2`, …).
pub const PART_SEGMENT_PREFIX: &str = "part-";

/// Name tokens that say nothing about an item's category, so they never become a branch.
///
/// Python's `preset_writer.GENERIC_TERMS` filters them out of branch keywords; the port
/// reuses the same idea to keep `tool`, `skill` and `helper` from becoming branches of
/// their own.
const NAME_NOISE_TOKENS: [&str; 13] = [
    "a", "an", "the", "tool", "tools", "skill", "skills", "plugin", "plugins", "use", "using",
    "helper", "utils",
];

/// How item groups are derived.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GroupingStrategy {
    /// The default: classification path, then tag, then leading name token.
    #[default]
    Deterministic,
    /// Classification and tags only; items with neither land under `general`.
    CategoryPath,
    /// Leading name tokens only, ignoring classification and tags.
    NameTokens,
    /// Python's two-phase LLM grouping, backed by
    /// [`llm`] — needs a completion model.
    LlmGrouping,
    /// Python's faiss k-means clustering, backed by
    /// [`kmeans`] — needs an embedding model.
    FaissKmeans,
    /// The spelling `Kmeans` maps to the same embedding-backed clustering as
    /// [`Self::FaissKmeans`].
    Kmeans,
}

impl GroupingStrategy {
    /// Parses a strategy name, tolerating case, `-`, `_` and surrounding whitespace.
    ///
    /// An unknown name is [`Self::Deterministic`], so a configuration file written for a
    /// newer build still groups items instead of failing the build.
    ///
    /// ```
    /// use openjiuwen_symphony::retrieval::build::grouping::GroupingStrategy;
    ///
    /// assert_eq!(GroupingStrategy::parse("Category-Path"), GroupingStrategy::CategoryPath);
    /// assert_eq!(GroupingStrategy::parse("faiss_kmeans"), GroupingStrategy::FaissKmeans);
    /// assert_eq!(GroupingStrategy::parse("nope"), GroupingStrategy::Deterministic);
    /// ```
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        let normalized = raw.trim().to_lowercase().replace(['-', ' '], "_");
        match normalized.as_str() {
            "category_path" => Self::CategoryPath,
            "name_tokens" => Self::NameTokens,
            "llm_grouping" => Self::LlmGrouping,
            "faiss_kmeans" => Self::FaissKmeans,
            "kmeans" => Self::Kmeans,
            _ => Self::Deterministic,
        }
    }

    /// The canonical `snake_case` name, which [`Self::parse`] accepts back.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::CategoryPath => "category_path",
            Self::NameTokens => "name_tokens",
            Self::LlmGrouping => "llm_grouping",
            Self::FaissKmeans => "faiss_kmeans",
            Self::Kmeans => "kmeans",
        }
    }

    /// Whether the crate can group items with this strategy.
    ///
    /// Every strategy has an implementation now: the three rule strategies run inside this
    /// module, [`kmeans`] backs the two k-means spellings and [`llm`] backs the model-guided
    /// one. Which of them needs a runtime supplied by the caller is answered by
    /// [`Self::requires_runtime`], not by this method.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        true
    }

    /// Whether the strategy needs a runtime supplied through
    /// [`GroupingRuntime`].
    ///
    /// [`Self::Kmeans`] and [`Self::FaissKmeans`] need an
    /// [`Embedding`] model, supplied with
    /// [`GroupingRuntime::with_embedding`];
    /// [`Self::LlmGrouping`] needs a
    /// [`CompletionModel`], supplied with
    /// [`GroupingRuntime::with_model`].
    /// The three rule strategies need nothing and run offline.
    #[must_use]
    pub fn requires_runtime(&self) -> bool {
        matches!(self, Self::LlmGrouping | Self::FaissKmeans | Self::Kmeans)
    }
}

/// How a build derives the branches of its tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GroupingConfig {
    /// How item groups are derived.
    pub strategy: GroupingStrategy,
    /// The maximum number of children one branch may hold; `0` is treated as `1`.
    pub max_children: usize,
    /// The maximum tree depth, counting the root as depth 1; `0` is treated as `1`.
    pub max_depth: usize,
    /// The `cid` segment of the tree root; a blank root yields no nodes at all.
    pub root_segment: String,
    /// Preset branch paths (`Vec<Vec<String>>`, handed over by
    /// [`presets`](crate::retrieval::build::presets)) that override the derived path.
    pub taxonomy: Vec<Vec<String>>,
    /// The cluster count for k-means grouping; `None` derives
    /// `ceil(items / max_children)` (see
    /// [`resolve_clusters`](crate::retrieval::build::grouping::kmeans::resolve_clusters)).
    pub clusters: Option<usize>,
}

impl Default for GroupingConfig {
    fn default() -> Self {
        Self {
            strategy: GroupingStrategy::Deterministic,
            max_children: DEFAULT_MAX_CHILDREN,
            max_depth: DEFAULT_MAX_DEPTH,
            root_segment: DEFAULT_ROOT_SEGMENT.to_string(),
            taxonomy: Vec::new(),
            clusters: None,
        }
    }
}

impl GroupingConfig {
    /// Sets the grouping strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: GroupingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the branch fan-out bound.
    #[must_use]
    pub fn with_max_children(mut self, max_children: usize) -> Self {
        self.max_children = max_children;
        self
    }

    /// Sets the tree depth bound.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the tree root's `cid` segment.
    #[must_use]
    pub fn with_root_segment(mut self, root_segment: impl Into<String>) -> Self {
        self.root_segment = root_segment.into();
        self
    }

    /// Sets the preset taxonomy paths that override the derived path.
    #[must_use]
    pub fn with_taxonomy(mut self, paths: Vec<Vec<String>>) -> Self {
        self.taxonomy = paths;
        self
    }

    /// Sets the k-means cluster count, or `None` to derive it from `max_children`.
    #[must_use]
    pub fn with_clusters(mut self, clusters: Option<usize>) -> Self {
        self.clusters = clusters;
        self
    }
}

/// One item's placement: `path` is the branch path below the root, segments already slugged.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GroupAssignment {
    /// The worker id of the placed item.
    pub worker_id: String,
    /// The branch segments below the root; empty when the item sits at the root.
    pub path: Vec<String>,
}

/// The models a runtime strategy may use, plus what to do when they are missing or fail.
///
/// A rule strategy never reads this runtime.
/// `Kmeans`/`FaissKmeans` read [`Self::embedding`] and `LlmGrouping` reads [`Self::model`];
/// [`Self::fallback`] decides whether a missing or failing runtime degrades to the
/// deterministic rule ([`FallbackMode::Deterministic`], the default) or surfaces an error
/// ([`FallbackMode::Error`]).
///
/// ```
/// use std::sync::Arc;
///
/// use openjiuwen_retrieval::HashEmbedding;
/// use openjiuwen_symphony::retrieval::build::grouping::GroupingRuntime;
///
/// let runtime = GroupingRuntime::new();
/// assert!(runtime.is_empty());
/// let runtime = runtime.with_embedding(Arc::new(HashEmbedding::new(8)));
/// assert!(!runtime.is_empty());
/// assert_eq!(runtime.embedding().map(|model| model.model_name()), Some("hash-embedding"));
/// ```
#[derive(Clone, Default)]
pub struct GroupingRuntime {
    /// The embedder `Kmeans`/`FaissKmeans` cluster with.
    embedding: Option<Arc<dyn Embedding>>,
    /// The completion model `LlmGrouping` asks.
    model: Option<Arc<dyn CompletionModel>>,
    /// What a missing or failing runtime does.
    fallback: FallbackMode,
}

impl GroupingRuntime {
    /// Builds an empty runtime with [`FallbackMode::Deterministic`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Supplies the embedder the k-means strategies cluster with.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Supplies the completion model the model-guided strategy asks.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets what a missing or failing runtime does.
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackMode) -> Self {
        self.fallback = fallback;
        self
    }

    /// The supplied embedder, if any.
    #[must_use]
    pub fn embedding(&self) -> Option<&dyn Embedding> {
        self.embedding.as_deref()
    }

    /// The supplied completion model, if any.
    #[must_use]
    pub fn model(&self) -> Option<&dyn CompletionModel> {
        self.model.as_deref()
    }

    /// What a missing or failing runtime does.
    #[must_use]
    pub fn fallback(&self) -> FallbackMode {
        self.fallback
    }

    /// Whether neither an embedder nor a model is supplied.
    ///
    /// The fallback mode alone does not make a runtime non-empty, because it changes what a
    /// failure does rather than supplying the machinery a strategy needs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.embedding.is_none() && self.model.is_none()
    }
}

impl fmt::Debug for GroupingRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroupingRuntime")
            .field("embedding", &self.embedding().map(Embedding::model_name))
            .field("model", &self.model().map(CompletionModel::name))
            .field("fallback", &self.fallback)
            .finish()
    }
}

/// How a [`GroupingOutcome`]'s paths were derived.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GroupingProvenance {
    /// A rule strategy derived the paths in-process, with no model call.
    #[default]
    Rule,
    /// The supplied embedder clustered the items.
    Embedding,
    /// The supplied completion model wrote the paths.
    Model,
    /// A runtime strategy degraded to the deterministic rule after a missing or failed
    /// runtime under [`FallbackMode::Deterministic`].
    DeterministicFallback,
}

/// The assignments one grouping run produced, with how they were derived.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GroupingOutcome {
    /// One placement per item the strategy placed, ordered by `worker_id`.
    pub assignments: Vec<GroupAssignment>,
    /// The strategy the caller asked for.
    pub strategy: GroupingStrategy,
    /// How `assignments` were derived.
    pub provenance: GroupingProvenance,
}

/// Derives the category segments of one item (public for tests and diagnostics).
///
/// The segments are slugged with [`slug_term`] and returned in path order, without the
/// root and without depth truncation (that is [`group_items`]'s job). An item the chosen
/// rule cannot classify gets `["general"]`, so the result is never empty. The model and
/// embedding strategies have no rule of their own, so this total function answers with the
/// deterministic rule for them, while
/// [`group_items_with`] applies the
/// strategy's own runtime.
///
/// ```
/// use openjiuwen_symphony::retrieval::build::grouping::{category_segments, GroupingStrategy};
/// use openjiuwen_symphony::retrieval::build::scanner::SourceItem;
///
/// let item = SourceItem {
///     name: "PDF Fill Helper".to_string(),
///     classification: "Documents/PDF".to_string(),
///     ..SourceItem::default()
/// };
/// let segments = category_segments(&item, GroupingStrategy::Deterministic);
/// assert_eq!(segments, vec!["documents".to_string(), "pdf".to_string()]);
/// ```
#[must_use]
pub fn category_segments(item: &SourceItem, strategy: GroupingStrategy) -> Vec<String> {
    let effective = match strategy {
        GroupingStrategy::CategoryPath => GroupingStrategy::CategoryPath,
        GroupingStrategy::NameTokens => GroupingStrategy::NameTokens,
        GroupingStrategy::Deterministic
        | GroupingStrategy::LlmGrouping
        | GroupingStrategy::FaissKmeans
        | GroupingStrategy::Kmeans => GroupingStrategy::Deterministic,
    };
    let mut segments = Vec::new();
    if matches!(
        effective,
        GroupingStrategy::Deterministic | GroupingStrategy::CategoryPath
    ) {
        segments = classification_segments(&item.classification);
        if segments.is_empty() {
            if let Some(tag) = item.tags.iter().find(|tag| !tag.trim().is_empty()) {
                segments.push(slug_term(tag, GENERAL_SEGMENT));
            }
        }
    }
    if segments.is_empty() && !matches!(effective, GroupingStrategy::CategoryPath) {
        if let Some(token) = leading_name_token(&item.name) {
            segments.push(token);
        }
    }
    if segments.is_empty() {
        segments.push(GENERAL_SEGMENT.to_string());
    }
    segments
}

/// The strategy-specific phrase naming the machinery a runtime strategy needs, or `None`
/// for a rule strategy.
///
/// The wording names what the strategy actually needs, so an `llm_grouping` build does not
/// report a k-means phrase.
fn deferral_reason(strategy: GroupingStrategy) -> Option<&'static str> {
    match strategy {
        GroupingStrategy::LlmGrouping => Some("LLM-guided tree grouping"),
        GroupingStrategy::FaissKmeans => Some("faiss k-means tree grouping"),
        GroupingStrategy::Kmeans => Some("k-means tree grouping"),
        GroupingStrategy::Deterministic
        | GroupingStrategy::CategoryPath
        | GroupingStrategy::NameTokens => None,
    }
}

/// The [`Error::Unsupported`] the synchronous entry point reports for a runtime strategy.
fn sync_deferral(strategy: GroupingStrategy) -> Error {
    let reason = deferral_reason(strategy).unwrap_or("this grouping strategy");
    Error::unsupported(format!(
        "{reason} needs an embedding or completion model, so it is available only from \
         `group_items_with` with a `GroupingRuntime`; the synchronous grouping entry point is \
         part of openjiuwen-symphony's deferred scope"
    ))
}

/// Places every item under the root; deterministic, sorted by `worker_id`.
///
/// The returned assignments are ordered by `worker_id` (ties keep the caller's item
/// order), and each `path` is the branch path below the root with slugged segments.
///
/// # Errors
///
/// [`GroupingStrategy::LlmGrouping`] needs a completion model and
/// [`GroupingStrategy::FaissKmeans`]/[`GroupingStrategy::Kmeans`] an embedding model, which a
/// synchronous call cannot await, so they return an [`Error::Unsupported`] naming that
/// strategy instead of grouping;
/// use [`group_items_with`] for them.
pub fn group_items(items: &[SourceItem], config: &GroupingConfig) -> Result<Vec<GroupAssignment>> {
    if config.strategy.requires_runtime() {
        return Err(sync_deferral(config.strategy));
    }
    let ordered = sorted_items(items);
    let candidates: Vec<GroupAssignment> = ordered
        .iter()
        .map(|item| GroupAssignment {
            worker_id: item.worker_id.clone(),
            path: category_segments(item, config.strategy),
        })
        .collect();
    place_assignments(&ordered, config, &candidates)
}

/// The items sorted by `worker_id`, so two runs over the same inventory place them alike.
fn sorted_items(items: &[SourceItem]) -> Vec<&SourceItem> {
    let mut ordered: Vec<&SourceItem> = items.iter().collect();
    ordered.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
    ordered
}

/// Places one candidate path per item through the taxonomy override, the depth bound, the
/// segment trie and the rebalancing pass.
///
/// `candidates[index]` belongs to `ordered[index]`; the returned assignments are in
/// `ordered` order.
fn place_assignments(
    ordered: &[&SourceItem],
    config: &GroupingConfig,
    candidates: &[GroupAssignment],
) -> Result<Vec<GroupAssignment>> {
    if candidates.len() != ordered.len() {
        return Err(validation(
            "grouping candidates",
            format!(
                "expected one candidate per item ({}), got {}",
                ordered.len(),
                candidates.len()
            ),
        ));
    }
    let limits = Limits::from_config(config);
    let taxonomy = normalize_taxonomy(&config.taxonomy, limits);
    let mut root = SegmentNode::default();
    for (index, candidate) in candidates.iter().enumerate() {
        let mut segments = candidate.path.clone();
        if !taxonomy.is_empty() {
            segments = match_taxonomy(&segments, &taxonomy);
        }
        segments.truncate(limits.max_depth.saturating_sub(1));
        root.insert(index, &segments);
    }
    let root = rebalance(root, 1, &mut Vec::new(), ordered, limits);
    let mut paths = vec![Vec::new(); ordered.len()];
    collect_paths(&root, &mut Vec::new(), &mut paths);
    Ok(ordered
        .iter()
        .enumerate()
        .map(|(index, item)| GroupAssignment {
            worker_id: item.worker_id.clone(),
            path: paths[index].clone(),
        })
        .collect())
}

/// Groups `items` with any strategy, awaiting whatever runtime the strategy needs.
///
/// Dispatch:
///
/// - `Deterministic`, `CategoryPath` and `NameTokens` run [`group_items`] and report
///   [`GroupingProvenance::Rule`]; they never touch the runtime's model or embedder.
/// - `Kmeans` and `FaissKmeans` cluster through
///   [`kmeans::cluster_paths`] and
///   report [`GroupingProvenance::Embedding`].
/// - `LlmGrouping` asks
///   [`llm::group_paths`] and reports
///   [`GroupingProvenance::Model`]; the items the reply left unassigned get no placement.
///
/// Whatever the strategy produced, the candidate paths pass through the same
/// `place_assignments` pipeline, so the taxonomy override, the `max_depth` cut and the
/// `max_children` rebalancing apply to every strategy and a model reply can never bypass
/// them.
///
/// # Fallbacks
///
/// A runtime strategy whose model or embedder is missing, or whose call fails, degrades to
/// the deterministic rule when the runtime's
/// [`FallbackMode`] is
/// [`Deterministic`](crate::llm::FallbackMode::Deterministic): the second outcome is
/// [`GroupingProvenance::DeterministicFallback`] with the strategy
/// [`GroupingStrategy::Deterministic`], so no path reports a model failure as a success.
/// Under [`Error`](crate::llm::FallbackMode::Error) the failure surfaces instead.
///
/// # Errors
///
/// Returns [`Error::Config`] for a missing embedder or model under
/// [`FallbackMode::Error`], the model's own
/// [`Error::Llm`] or the embedder's error when its call fails under
/// that mode, and any placement error of the shared pipeline.
pub async fn group_items_with(
    items: &[SourceItem],
    config: &GroupingConfig,
    runtime: &GroupingRuntime,
) -> Result<GroupingOutcome> {
    match config.strategy {
        GroupingStrategy::Deterministic
        | GroupingStrategy::CategoryPath
        | GroupingStrategy::NameTokens => rule_outcome(items, config),
        GroupingStrategy::Kmeans | GroupingStrategy::FaissKmeans => {
            embedding_outcome(items, config, runtime).await
        }
        GroupingStrategy::LlmGrouping => model_outcome(items, config, runtime).await,
    }
}

/// The outcome of a rule strategy.
fn rule_outcome(items: &[SourceItem], config: &GroupingConfig) -> Result<GroupingOutcome> {
    Ok(GroupingOutcome {
        assignments: group_items(items, config)?,
        strategy: config.strategy,
        provenance: GroupingProvenance::Rule,
    })
}

/// The outcome of the deterministic rule, reported as a fallback.
fn deterministic_outcome(items: &[SourceItem], config: &GroupingConfig) -> Result<GroupingOutcome> {
    let rule_config = config
        .clone()
        .with_strategy(GroupingStrategy::Deterministic);
    Ok(GroupingOutcome {
        assignments: group_items(items, &rule_config)?,
        strategy: GroupingStrategy::Deterministic,
        provenance: GroupingProvenance::DeterministicFallback,
    })
}

/// What a runtime strategy does when its model or embedder is missing.
fn missing_runtime(
    items: &[SourceItem],
    config: &GroupingConfig,
    runtime: &GroupingRuntime,
    message: &str,
) -> Result<GroupingOutcome> {
    match runtime.fallback() {
        FallbackMode::Error => Err(Error::config(message.to_string())),
        FallbackMode::Deterministic => {
            tracing::warn!("{}; falling back to the deterministic rule", message);
            deterministic_outcome(items, config)
        }
    }
}

/// What a runtime strategy does when its model or embedder fails.
fn failed_runtime(
    items: &[SourceItem],
    config: &GroupingConfig,
    runtime: &GroupingRuntime,
    error: Error,
    stage: &str,
) -> Result<GroupingOutcome> {
    match runtime.fallback() {
        FallbackMode::Error => Err(error),
        FallbackMode::Deterministic => {
            tracing::warn!(
                %error,
                "the {} runtime failed; falling back to the deterministic rule",
                stage
            );
            deterministic_outcome(items, config)
        }
    }
}

/// The message a k-means strategy reports when the runtime has no embedder.
const MISSING_EMBEDDER: &str = "grouping runtime: k-means grouping needs an embedding model; \
                                supply one with `GroupingRuntime::with_embedding`";
/// The message the model-guided strategy reports when the runtime has no model.
const MISSING_MODEL: &str = "grouping runtime: LLM-guided grouping needs a completion model; \
                             supply one with `GroupingRuntime::with_model`";

/// The outcome of `Kmeans`/`FaissKmeans`: the clustered paths, or the documented fallback.
async fn embedding_outcome(
    items: &[SourceItem],
    config: &GroupingConfig,
    runtime: &GroupingRuntime,
) -> Result<GroupingOutcome> {
    let Some(embedding) = runtime.embedding() else {
        return missing_runtime(items, config, runtime, MISSING_EMBEDDER);
    };
    match kmeans::cluster_paths(items, config, embedding).await {
        Ok(candidates) => {
            let ordered = sorted_items(items);
            place_outcome(&ordered, config, &candidates, GroupingProvenance::Embedding)
        }
        Err(error) => failed_runtime(items, config, runtime, error, "embedding grouping"),
    }
}

/// The outcome of `LlmGrouping`: the reply's paths, or the documented fallback.
async fn model_outcome(
    items: &[SourceItem],
    config: &GroupingConfig,
    runtime: &GroupingRuntime,
) -> Result<GroupingOutcome> {
    let Some(model) = runtime.model() else {
        return missing_runtime(items, config, runtime, MISSING_MODEL);
    };
    match llm::group_paths(items, config, model).await {
        Ok(candidates) => {
            let (ordered, placed) = assigned_items(items, &candidates);
            place_outcome(&ordered, config, &placed, GroupingProvenance::Model)
        }
        Err(error) => failed_runtime(items, config, runtime, error, "model grouping"),
    }
}

/// Places `candidates` over `ordered` and reports the requested strategy.
fn place_outcome(
    ordered: &[&SourceItem],
    config: &GroupingConfig,
    candidates: &[GroupAssignment],
    provenance: GroupingProvenance,
) -> Result<GroupingOutcome> {
    Ok(GroupingOutcome {
        assignments: place_assignments(ordered, config, candidates)?,
        strategy: config.strategy,
        provenance,
    })
}

/// Pairs the items a model reply assigned with their assignment, both in `worker_id` order.
fn assigned_items<'a>(
    items: &'a [SourceItem],
    candidates: &[GroupAssignment],
) -> (Vec<&'a SourceItem>, Vec<GroupAssignment>) {
    let mut by_id: BTreeMap<&str, &GroupAssignment> = BTreeMap::new();
    for candidate in candidates {
        by_id
            .entry(candidate.worker_id.as_str())
            .or_insert(candidate);
    }
    let mut ordered = Vec::new();
    let mut placed = Vec::new();
    for item in sorted_items(items) {
        if let Some(candidate) = by_id.get(item.worker_id.as_str()) {
            ordered.push(item);
            placed.push((*candidate).clone());
        }
    }
    (ordered, placed)
}

/// The branch/taxonomy bounds after normalization, so the algorithm never has to guard
/// against a zero that would make the recursion or the chunking loop diverge.
#[derive(Debug, Clone, Copy)]
struct Limits {
    /// The branch fan-out bound; at least `1`.
    max_children: usize,
    /// The tree depth bound, counting the root; at least `1`.
    max_depth: usize,
}

impl Limits {
    /// Normalizes `config`'s bounds, mapping `0` to `1`.
    fn from_config(config: &GroupingConfig) -> Self {
        Self {
            max_children: config.max_children.max(1),
            max_depth: config.max_depth.max(1),
        }
    }
}

/// One node of the intermediate segment trie.
#[derive(Debug, Clone, Default)]
struct SegmentNode {
    /// Indices into the sorted item list of the items that end exactly here.
    items: Vec<usize>,
    /// Child nodes keyed by their slugged segment.
    children: BTreeMap<String, SegmentNode>,
}

impl SegmentNode {
    /// Inserts `index` at `segments`.
    fn insert(&mut self, index: usize, segments: &[String]) {
        match segments.split_first() {
            Some((head, tail)) => self
                .children
                .entry(head.clone())
                .or_default()
                .insert(index, tail),
            None => self.items.push(index),
        }
    }

    /// Merges `other` into this node, keeping items and same-named children.
    fn absorb(&mut self, other: Self) {
        self.items.extend(other.items);
        for (name, child) in other.children {
            match self.children.entry(name) {
                Entry::Occupied(mut entry) => entry.get_mut().absorb(child),
                Entry::Vacant(entry) => {
                    entry.insert(child);
                }
            }
        }
    }
}

/// Enforces the fan-out and depth bounds bottom-up, returning the rebalanced node.
fn rebalance(
    node: SegmentNode,
    depth: usize,
    path: &mut Vec<String>,
    items: &[&SourceItem],
    limits: Limits,
) -> SegmentNode {
    let SegmentNode {
        items: node_items,
        children,
    } = node;
    let mut node = SegmentNode {
        items: node_items,
        children: BTreeMap::new(),
    };
    for (name, child) in children {
        path.push(name.clone());
        let child = rebalance(child, depth + 1, path, items, limits);
        path.pop();
        node.children.insert(name, child);
    }
    fold_children(&mut node, depth, path, items, limits);
    if node.items.len() > limits.max_children && depth < limits.max_depth {
        bucket_items(&mut node, depth, path, items, limits);
        fold_children(&mut node, depth, path, items, limits);
        // The bucketing and the fold may have created nodes that are over the item
        // bound themselves, so narrow them too before returning.
        let children = std::mem::take(&mut node.children);
        for (name, child) in children {
            path.push(name.clone());
            let child = rebalance(child, depth + 1, path, items, limits);
            path.pop();
            node.children.insert(name, child);
        }
    }
    node
}

/// Folds a node's excess child branches into one [`OTHER_SEGMENT`] child.
fn fold_children(
    node: &mut SegmentNode,
    depth: usize,
    path: &mut Vec<String>,
    items: &[&SourceItem],
    limits: Limits,
) {
    if node.children.len() <= limits.max_children || depth >= limits.max_depth {
        return;
    }
    let keep = limits.max_children - 1;
    let names: Vec<String> = node.children.keys().cloned().collect();
    let mut kept: BTreeMap<String, SegmentNode> = BTreeMap::new();
    let mut merged = SegmentNode::default();
    for (position, name) in names.into_iter().enumerate() {
        let Some(child) = node.children.remove(&name) else {
            continue;
        };
        if position < keep {
            kept.insert(name, child);
        } else {
            merged.absorb(child);
        }
    }
    // `other` is reserved for the fold, so a real segment named `other` is merged too.
    if let Some(existing) = kept.remove(OTHER_SEGMENT) {
        merged.absorb(existing);
    }
    path.push(OTHER_SEGMENT.to_string());
    let merged = rebalance(merged, depth + 1, path, items, limits);
    path.pop();
    kept.insert(OTHER_SEGMENT.to_string(), merged);
    node.children = kept;
}

/// Buckets a node's excess items into `next name token` children, chunking over-full buckets.
fn bucket_items(
    node: &mut SegmentNode,
    depth: usize,
    path: &[String],
    items: &[&SourceItem],
    limits: Limits,
) {
    node.items.sort_unstable();
    let mut buckets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for index in std::mem::take(&mut node.items) {
        let segment = match items.get(index) {
            Some(item) => next_name_token(item, path),
            None => GENERAL_SEGMENT.to_string(),
        };
        buckets.entry(segment).or_default().push(index);
    }
    for (segment, bucket) in buckets {
        let mut child = SegmentNode {
            items: bucket,
            children: BTreeMap::new(),
        };
        if child.items.len() > limits.max_children && depth + 2 <= limits.max_depth {
            let mut parts: BTreeMap<String, SegmentNode> = BTreeMap::new();
            for (position, chunk) in child.items.chunks(limits.max_children).enumerate() {
                parts.insert(
                    format!("{PART_SEGMENT_PREFIX}{}", position + 1),
                    SegmentNode {
                        items: chunk.to_vec(),
                        children: BTreeMap::new(),
                    },
                );
            }
            child = SegmentNode {
                items: Vec::new(),
                children: parts,
            };
        }
        match node.children.entry(segment) {
            Entry::Occupied(mut entry) => entry.get_mut().absorb(child),
            Entry::Vacant(entry) => {
                entry.insert(child);
            }
        }
    }
}

/// Records the final path of every item index.
fn collect_paths(node: &SegmentNode, path: &mut Vec<String>, out: &mut [Vec<String>]) {
    for index in &node.items {
        if let Some(slot) = out.get_mut(*index) {
            slot.clone_from(path);
        }
    }
    for (name, child) in &node.children {
        path.push(name.clone());
        collect_paths(child, path, out);
        path.pop();
    }
}

/// The item's first name token that its own branch path does not already use.
fn next_name_token(item: &SourceItem, path: &[String]) -> String {
    let used: BTreeSet<&str> = path.iter().map(String::as_str).collect();
    let key = normalize_name_key(&item.name);
    key.split('_')
        .find(|token| !token.is_empty() && !used.contains(token))
        .map_or_else(|| GENERAL_SEGMENT.to_string(), str::to_string)
}

/// The classification-derived segments of one item.
fn classification_segments(classification: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut seen = BTreeSet::new();
    for chunk in classification.split(['.', '/', '>', '|', ',']) {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let slug = slug_term(chunk, GENERAL_SEGMENT);
        if seen.insert(slug.clone()) {
            segments.push(slug);
        }
    }
    segments
}

/// The item's leading name token, skipping routing-noise words.
fn leading_name_token(name: &str) -> Option<String> {
    normalize_name_key(name)
        .split('_')
        .find(|token| !token.is_empty() && !NAME_NOISE_TOKENS.contains(token))
        .map(str::to_string)
}

/// Slugs and dedupes every taxonomy path once, so matching never re-slugs per item.
fn normalize_taxonomy(taxonomy: &[Vec<String>], limits: Limits) -> Vec<Vec<String>> {
    taxonomy
        .iter()
        .map(|path| {
            let mut segments = Vec::new();
            let mut seen = BTreeSet::new();
            for segment in path {
                if segment.trim().is_empty() {
                    continue;
                }
                let slug = slug_term(segment, GENERAL_SEGMENT);
                if seen.insert(slug.clone()) {
                    segments.push(slug);
                }
            }
            segments.truncate(limits.max_depth.saturating_sub(1));
            segments
        })
        .collect()
}

/// Picks the taxonomy path with the highest token overlap, or the root when none overlaps.
fn match_taxonomy(segments: &[String], taxonomy: &[Vec<String>]) -> Vec<String> {
    let derived: Vec<&str> = segments.iter().map(String::as_str).collect();
    let item_tokens = text_tokens(&derived);
    // (overlap, path length, joined path, taxonomy index)
    let mut best: Option<(usize, usize, String, usize)> = None;
    for (index, path) in taxonomy.iter().enumerate() {
        let path_segments: Vec<&str> = path.iter().map(String::as_str).collect();
        let overlap = item_tokens
            .intersection(&text_tokens(&path_segments))
            .count();
        if overlap == 0 {
            continue;
        }
        let joined = path.join(".");
        let replace = match &best {
            None => true,
            Some((best_overlap, best_len, best_joined, _)) => {
                overlap > *best_overlap
                    || (overlap == *best_overlap && path.len() > *best_len)
                    || (overlap == *best_overlap
                        && path.len() == *best_len
                        && joined < *best_joined)
            }
        };
        if replace {
            best = Some((overlap, path.len(), joined, index));
        }
    }
    match best {
        Some((_, _, _, index)) => taxonomy.get(index).cloned().unwrap_or_default(),
        None => Vec::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use openjiuwen_retrieval::{HashEmbedding, StaticCompletionModel};

    use super::*;
    use crate::error::{is_unsupported, ErrorKind};
    use crate::retrieval::types::RetrieverItem;

    fn item(worker_id: &str, name: &str, classification: &str, tags: &[&str]) -> SourceItem {
        SourceItem {
            worker_id: worker_id.to_string(),
            item: RetrieverItem::default(),
            name: name.to_string(),
            description: format!("{name} description"),
            content: String::new(),
            skill_path: String::new(),
            classification: classification.to_string(),
            tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
            item_type: "skill".to_string(),
            content_hash: String::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// The paths keyed by worker id, in assignment order.
    fn paths(assignments: &[GroupAssignment]) -> Vec<(String, Vec<String>)> {
        assignments
            .iter()
            .map(|assignment| (assignment.worker_id.clone(), assignment.path.clone()))
            .collect()
    }

    fn owned(segments: &[&str]) -> Vec<String> {
        segments
            .iter()
            .map(|segment| (*segment).to_string())
            .collect()
    }

    /// The three-item inventory the runtime tests share.
    fn inventory() -> Vec<SourceItem> {
        vec![
            item("a", "PDF Parser", "documents/pdf", &[]),
            item("b", "Spreadsheet Reader", "documents/sheets", &[]),
            item("c", "Chart Builder", "charts", &[]),
        ]
    }

    /// A runtime with both a model and an embedder, for the "must not be consulted" pins.
    fn full_runtime(model: &Arc<StaticCompletionModel>) -> GroupingRuntime {
        GroupingRuntime::new()
            .with_model(Arc::clone(model) as Arc<dyn CompletionModel>)
            .with_embedding(Arc::new(HashEmbedding::new(8)))
    }

    #[test]
    fn category_segments_derives_from_classification() {
        let cases = [
            ("Documents.PDF", vec!["documents", "pdf"]),
            ("docs/pdf fill", vec!["docs", "pdf-fill"]),
            ("a > b | c , d / e", vec!["a", "b", "c", "d", "e"]),
            ("docs..docs", vec!["docs"]),
            // A blank classification falls through to the leading name token.
            ("  ", vec!["unrelated"]),
        ];
        for (classification, expected) in cases {
            let item = item("w", "Unrelated Name", classification, &[]);
            assert_eq!(
                category_segments(&item, GroupingStrategy::Deterministic),
                owned(&expected),
                "classification {classification:?}"
            );
        }
    }

    #[test]
    fn category_segments_falls_back_to_tags_then_name_tokens() {
        let tagged = item("w", "Whatever", "", &["  ", "PDF Tools"]);
        assert_eq!(
            category_segments(&tagged, GroupingStrategy::Deterministic),
            owned(&["pdf-tools"])
        );
        let named = item("w", "Read File Helper", "", &[]);
        assert_eq!(
            category_segments(&named, GroupingStrategy::Deterministic),
            owned(&["read"])
        );
        let noisy = item("w", "Tool Helper Utils", "", &[]);
        assert_eq!(
            category_segments(&noisy, GroupingStrategy::Deterministic),
            owned(&["general"])
        );
        assert_eq!(
            category_segments(&item("w", "", "", &[]), GroupingStrategy::Deterministic),
            owned(&["general"])
        );
    }

    #[test]
    fn category_segments_category_path_skips_name_tokens() {
        let tagged = item("w", "Read File", "", &["docs"]);
        assert_eq!(
            category_segments(&tagged, GroupingStrategy::CategoryPath),
            owned(&["docs"])
        );
        let bare = item("w", "Read File", "", &[]);
        assert_eq!(
            category_segments(&bare, GroupingStrategy::CategoryPath),
            owned(&["general"])
        );
    }

    #[test]
    fn category_segments_name_tokens_ignores_classification() {
        let item = item("w", "beta reader", "docs", &["tag"]);
        assert_eq!(
            category_segments(&item, GroupingStrategy::NameTokens),
            owned(&["beta"])
        );
        // A deferred strategy has no rule of its own, so the total helper answers with
        // the deterministic rule while `group_items` reports the deferral.
        assert_eq!(
            category_segments(&item, GroupingStrategy::LlmGrouping),
            owned(&["docs"])
        );
    }

    #[test]
    fn group_items_is_deterministic_and_sorted_by_worker_id() {
        let mut items: Vec<SourceItem> = (0..12)
            .map(|index| {
                item(
                    &format!("w{index:02}"),
                    &format!("name {index}"),
                    &format!("group-{}", index % 3),
                    &[],
                )
            })
            .collect();
        items.reverse();
        let config = GroupingConfig::default();
        let first = group_items(&items, &config).unwrap();
        let second = group_items(&items, &config).unwrap();
        assert_eq!(first, second);
        let ids: Vec<&str> = first
            .iter()
            .map(|assignment| assignment.worker_id.as_str())
            .collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
        assert_eq!(first.len(), items.len());
    }

    #[test]
    fn group_items_places_items_under_classification_paths() {
        let items = vec![
            item("a", "Alpha", "Documents.PDF", &[]),
            item("b", "Beta", "Documents.PDF", &[]),
            item("c", "Gamma", "Web/Search", &[]),
        ];
        let assignments = group_items(&items, &GroupingConfig::default()).unwrap();
        assert_eq!(
            paths(&assignments),
            vec![
                ("a".to_string(), owned(&["documents", "pdf"])),
                ("b".to_string(), owned(&["documents", "pdf"])),
                ("c".to_string(), owned(&["web", "search"])),
            ]
        );
    }

    #[test]
    fn group_items_bounds_branch_fan_out() {
        let items: Vec<SourceItem> = (0..60)
            .map(|index| {
                item(
                    &format!("w{index:02}"),
                    &format!("Capability {index}"),
                    &format!("c{index:02}"),
                    &[],
                )
            })
            .collect();
        let config = GroupingConfig::default()
            .with_max_children(4)
            .with_max_depth(6);
        let assignments = group_items(&items, &config).unwrap();
        assert_eq!(assignments.len(), 60);

        let mut children: BTreeMap<Vec<String>, BTreeSet<String>> = BTreeMap::new();
        for assignment in &assignments {
            let mut parent: Vec<String> = Vec::new();
            for segment in &assignment.path {
                children
                    .entry(parent.clone())
                    .or_default()
                    .insert(segment.clone());
                parent.push(segment.clone());
            }
        }
        assert!(!children.is_empty(), "every item was placed at the root");
        for (parent, branches) in &children {
            assert!(
                branches.len() <= 4,
                "branch {parent:?} has {} children: {branches:?}",
                branches.len()
            );
        }
        // `other` collected the 57 folded classifications, so the bound had to be enforced
        // below the root as well.
        assert!(children[&Vec::<String>::new()].contains("other"));
        for assignment in &assignments {
            // Root plus at most `max_depth - 1` segments.
            assert!(assignment.path.len() <= 5, "{:?}", assignment.path);
        }
    }

    #[test]
    fn group_items_truncates_to_max_depth() {
        let items = vec![item("w", "Deep", "a.b.c.d.e.f.g", &[])];
        let config = GroupingConfig::default().with_max_depth(3);
        let assignments = group_items(&items, &config).unwrap();
        assert_eq!(assignments[0].path, owned(&["a", "b"]));
        let shallow = GroupingConfig::default().with_max_depth(1);
        assert_eq!(
            group_items(&items, &shallow).unwrap()[0].path,
            Vec::<String>::new()
        );
        // A zero bound behaves like the smallest usable depth.
        let zeroed = GroupingConfig::default().with_max_depth(0);
        assert_eq!(
            group_items(&items, &zeroed).unwrap()[0].path,
            Vec::<String>::new()
        );
    }

    #[test]
    fn group_items_buckets_overflowing_leaves_by_name_token() {
        let items = vec![
            item("w1", "alpha one", "docs", &[]),
            item("w2", "alpha two", "docs", &[]),
            item("w3", "beta three", "docs", &[]),
        ];
        let config = GroupingConfig::default()
            .with_max_children(2)
            .with_max_depth(4);
        let assignments = group_items(&items, &config).unwrap();
        assert_eq!(
            paths(&assignments),
            vec![
                ("w1".to_string(), owned(&["docs", "alpha"])),
                ("w2".to_string(), owned(&["docs", "alpha"])),
                ("w3".to_string(), owned(&["docs", "beta"])),
            ]
        );

        // A bucket that is still over the bound is chunked into `part-N` children.
        let many: Vec<SourceItem> = (0..9)
            .map(|index| item(&format!("w{index}"), "same name", "docs", &[]))
            .collect();
        let config = GroupingConfig::default()
            .with_max_children(4)
            .with_max_depth(4);
        let assignments = group_items(&many, &config).unwrap();
        let mut expected = Vec::new();
        for index in 0..9 {
            let part = index / 4 + 1;
            expected.push((
                format!("w{index}"),
                owned(&["docs", "same", &format!("part-{part}")]),
            ));
        }
        assert_eq!(paths(&assignments), expected);

        // Both bounds cannot hold at the depth limit, where they are documented as soft:
        // the builder stops splitting exactly as Python's does.
        let tight: Vec<SourceItem> = (0..7)
            .map(|index| item(&format!("w{index}"), "same name", "docs", &[]))
            .collect();
        let config = GroupingConfig::default()
            .with_max_children(2)
            .with_max_depth(4);
        let assignments = group_items(&tight, &config).unwrap();
        assert_eq!(assignments[0].path, owned(&["docs", "same", "part-1"]));
        assert!(assignments
            .iter()
            .all(|assignment| assignment.path.len() <= 3));
        let mut children: BTreeMap<Vec<String>, BTreeSet<String>> = BTreeMap::new();
        for assignment in &assignments {
            let mut parent: Vec<String> = Vec::new();
            for segment in &assignment.path {
                children
                    .entry(parent.clone())
                    .or_default()
                    .insert(segment.clone());
                parent.push(segment.clone());
            }
        }
        for (parent, branches) in &children {
            assert!(branches.len() <= 2, "{parent:?} has {branches:?}");
        }
    }

    #[test]
    fn group_items_taxonomy_override_picks_the_best_overlap() {
        let taxonomy = vec![owned(&["Documents", "Web"]), owned(&["Documents", "PDF"])];
        let items = vec![
            item("a", "Alpha", "documents.pdf.forms", &[]),
            item("b", "Beta", "documents.web", &[]),
        ];
        let config = GroupingConfig::default().with_taxonomy(taxonomy);
        let assignments = group_items(&items, &config).unwrap();
        assert_eq!(
            paths(&assignments),
            vec![
                ("a".to_string(), owned(&["documents", "pdf"])),
                ("b".to_string(), owned(&["documents", "web"])),
            ]
        );

        // Equal overlap: the longer preset path wins.
        let items = vec![item("a", "Alpha", "docs", &[])];
        let config = GroupingConfig::default()
            .with_taxonomy(vec![owned(&["docs"]), owned(&["docs", "pdf"])]);
        assert_eq!(
            group_items(&items, &config).unwrap()[0].path,
            owned(&["docs", "pdf"])
        );

        // Equal overlap and equal length: the lexicographically smallest joined path wins.
        let config = GroupingConfig::default()
            .with_taxonomy(vec![owned(&["docs", "zeta"]), owned(&["docs", "alpha"])]);
        assert_eq!(
            group_items(&items, &config).unwrap()[0].path,
            owned(&["docs", "alpha"])
        );
    }

    #[test]
    fn group_items_taxonomy_places_zero_overlap_items_at_the_root() {
        let items = vec![
            item("a", "Alpha", "audio.video", &[]),
            item("b", "Beta", "audio.video", &[]),
        ];
        let config = GroupingConfig::default().with_taxonomy(vec![owned(&["Documents", "PDF"])]);
        let assignments = group_items(&items, &config).unwrap();
        assert_eq!(
            paths(&assignments),
            vec![("a".to_string(), Vec::new()), ("b".to_string(), Vec::new()),]
        );
    }

    #[test]
    fn grouping_strategy_parses_and_round_trips() {
        let strategies = [
            GroupingStrategy::Deterministic,
            GroupingStrategy::CategoryPath,
            GroupingStrategy::NameTokens,
            GroupingStrategy::LlmGrouping,
            GroupingStrategy::FaissKmeans,
            GroupingStrategy::Kmeans,
        ];
        for strategy in strategies {
            assert_eq!(GroupingStrategy::parse(strategy.as_str()), strategy);
            assert_eq!(
                GroupingStrategy::parse(&strategy.as_str().to_uppercase()),
                strategy
            );
            assert_eq!(
                GroupingStrategy::parse(&strategy.as_str().replace('_', "-")),
                strategy
            );
        }
        assert_eq!(
            GroupingStrategy::parse("  NAME TOKENS "),
            GroupingStrategy::NameTokens
        );
        assert_eq!(GroupingStrategy::parse(""), GroupingStrategy::Deterministic);
        assert_eq!(GroupingStrategy::default(), GroupingStrategy::Deterministic);
    }

    #[test]
    fn strategy_parse_and_default() {
        for strategy in [
            GroupingStrategy::Deterministic,
            GroupingStrategy::CategoryPath,
            GroupingStrategy::NameTokens,
        ] {
            assert!(strategy.is_supported(), "{strategy:?} is implemented");
            assert!(
                !strategy.requires_runtime(),
                "{strategy:?} needs no runtime"
            );
        }
        for strategy in [
            GroupingStrategy::LlmGrouping,
            GroupingStrategy::FaissKmeans,
            GroupingStrategy::Kmeans,
        ] {
            assert!(
                strategy.is_supported(),
                "{strategy:?} is implemented by its runtime module"
            );
            assert!(
                strategy.requires_runtime(),
                "{strategy:?} needs a supplied runtime"
            );
        }
        assert_eq!(GroupingStrategy::default(), GroupingStrategy::Deterministic);
    }

    #[test]
    fn runtime_strategies_are_sync_deferrals() {
        let items = vec![item("w", "Alpha", "docs", &[])];
        for (strategy, reason) in [
            (GroupingStrategy::LlmGrouping, "LLM-guided tree grouping"),
            (GroupingStrategy::FaissKmeans, "faiss k-means tree grouping"),
            (GroupingStrategy::Kmeans, "k-means tree grouping"),
        ] {
            let config = GroupingConfig::default().with_strategy(strategy);
            let error = group_items(&items, &config).unwrap_err();
            assert!(is_unsupported(&error), "{strategy:?}: {error}");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(error.message().starts_with(reason), "{strategy:?}: {error}");
            assert!(
                error.message().contains("deferred"),
                "{strategy:?}: {error}"
            );
            assert!(
                error.message().contains("deferred scope"),
                "{strategy:?}: {error}"
            );
            assert!(
                error.message().contains("group_items_with"),
                "{strategy:?} names the async entry point: {error}"
            );
            assert!(
                !error.message().contains("LLM/faiss"),
                "{strategy:?} names one strategy: {error}"
            );
        }
        let llm = group_items(
            &items,
            &GroupingConfig::default().with_strategy(GroupingStrategy::LlmGrouping),
        )
        .unwrap_err();
        assert!(
            !llm.message().contains("k-means"),
            "an LLM strategy is not a k-means deferral: {llm}"
        );
    }

    #[test]
    fn grouping_config_builders_set_every_field() {
        let config = GroupingConfig::default()
            .with_strategy(GroupingStrategy::CategoryPath)
            .with_max_children(3)
            .with_max_depth(4)
            .with_root_segment("Capabilities")
            .with_taxonomy(vec![owned(&["docs"])]);
        assert_eq!(config.strategy, GroupingStrategy::CategoryPath);
        assert_eq!(config.max_children, 3);
        assert_eq!(config.max_depth, 4);
        assert_eq!(config.root_segment, "Capabilities");
        assert_eq!(config.taxonomy, vec![owned(&["docs"])]);
        let default = GroupingConfig::default();
        assert_eq!(default.strategy, GroupingStrategy::Deterministic);
        assert_eq!(default.max_children, DEFAULT_MAX_CHILDREN);
        assert_eq!(default.max_depth, DEFAULT_MAX_DEPTH);
        assert_eq!(default.root_segment, DEFAULT_ROOT_SEGMENT);
        assert_eq!(default.taxonomy, Vec::<Vec<String>>::new());
    }

    #[test]
    fn assignments_and_config_round_trip_through_json() {
        let assignment = GroupAssignment {
            worker_id: "w".to_string(),
            path: owned(&["docs", "pdf"]),
        };
        let encoded = serde_json::to_string(&assignment).unwrap();
        assert!(encoded.contains("\"worker_id\":\"w\""));
        let decoded: GroupAssignment = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, assignment);
        let tolerant: GroupAssignment = serde_json::from_str(r#"{"unknown": 1}"#).unwrap();
        assert_eq!(tolerant, GroupAssignment::default());

        let config = GroupingConfig::default().with_taxonomy(vec![owned(&["docs"])]);
        let decoded: GroupingConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(decoded, config);
        let tolerant: GroupingConfig =
            serde_json::from_str(r#"{"strategy": "name_tokens"}"#).unwrap();
        assert_eq!(tolerant.strategy, GroupingStrategy::NameTokens);
        assert_eq!(tolerant.max_children, DEFAULT_MAX_CHILDREN);
    }

    #[tokio::test]
    async fn rule_strategies_report_rule_provenance_and_ignore_the_runtime() {
        let items = inventory();
        let model = Arc::new(StaticCompletionModel::single(r#"{"groups":[]}"#));
        let runtime = full_runtime(&model);
        for strategy in [
            GroupingStrategy::Deterministic,
            GroupingStrategy::CategoryPath,
            GroupingStrategy::NameTokens,
        ] {
            let config = GroupingConfig::default().with_strategy(strategy);
            let outcome = group_items_with(&items, &config, &runtime).await.unwrap();
            assert_eq!(outcome.provenance, GroupingProvenance::Rule);
            assert_eq!(outcome.strategy, strategy);
            assert_eq!(outcome.assignments, group_items(&items, &config).unwrap());
        }
        assert_eq!(
            model.call_count(),
            0,
            "a rule strategy must never call the model"
        );
    }

    #[tokio::test]
    async fn kmeans_groups_through_the_embedding_runtime() {
        let items = inventory();
        let runtime = GroupingRuntime::new().with_embedding(Arc::new(HashEmbedding::new(16)));
        let config = GroupingConfig::default()
            .with_strategy(GroupingStrategy::Kmeans)
            .with_max_children(2)
            .with_max_depth(3);
        let outcome = group_items_with(&items, &config, &runtime).await.unwrap();
        assert_eq!(outcome.provenance, GroupingProvenance::Embedding);
        assert_eq!(outcome.strategy, GroupingStrategy::Kmeans);
        assert_eq!(outcome.assignments.len(), items.len());
        let ids: Vec<&str> = outcome
            .assignments
            .iter()
            .map(|assignment| assignment.worker_id.as_str())
            .collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "the assignments are ordered by worker_id");
        for assignment in &outcome.assignments {
            assert!(assignment.path.len() <= 2, "{:?}", assignment.path);
        }
    }

    #[tokio::test]
    async fn kmeans_without_an_embedder_falls_back_deterministically() {
        let items = inventory();
        let runtime = GroupingRuntime::new();
        assert!(runtime.is_empty());
        let config = GroupingConfig::default().with_strategy(GroupingStrategy::Kmeans);
        let outcome = group_items_with(&items, &config, &runtime).await.unwrap();
        assert_eq!(outcome.strategy, GroupingStrategy::Deterministic);
        assert_eq!(
            outcome.provenance,
            GroupingProvenance::DeterministicFallback
        );
        assert_eq!(
            outcome.assignments,
            group_items(&items, &GroupingConfig::default()).unwrap()
        );
    }

    #[tokio::test]
    async fn kmeans_without_an_embedder_is_a_config_error_in_error_mode() {
        let items = inventory();
        let runtime = GroupingRuntime::new().with_fallback(FallbackMode::Error);
        let config = GroupingConfig::default().with_strategy(GroupingStrategy::FaissKmeans);
        let error = group_items_with(&items, &config, &runtime)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.message().contains("with_embedding"), "{error}");
    }

    #[tokio::test]
    async fn kmeans_cluster_count_bounds_the_paths() {
        // `clusters` clamps into `[1, min(items, max_children)]`, so every case bounds the paths.
        // The inventory holds exactly `DEFAULT_MAX_CHILDREN` items: a larger one would be split by
        // this module's own fan-out rebalancing, which would hide the k-means cluster bound.
        let items: Vec<SourceItem> = (0..DEFAULT_MAX_CHILDREN)
            .map(|index| {
                item(
                    &format!("w{index:02}"),
                    &format!("name {index}"),
                    &format!("group-{}", index % 3),
                    &[],
                )
            })
            .collect();
        let runtime = GroupingRuntime::new().with_embedding(Arc::new(HashEmbedding::new(64)));
        let cases: [(Option<usize>, usize, usize); 5] = [
            (Some(1), 1, 1),
            (Some(3), 1, 3),
            (Some(999), 1, DEFAULT_MAX_CHILDREN),
            (Some(0), 1, DEFAULT_MAX_CHILDREN),
            (None, 1, DEFAULT_MAX_CHILDREN),
        ];
        for (clusters, min_paths, max_paths) in cases {
            let config = GroupingConfig::default()
                .with_strategy(GroupingStrategy::Kmeans)
                .with_clusters(clusters);
            let outcome = group_items_with(&items, &config, &runtime).await.unwrap();
            let distinct: BTreeSet<Vec<String>> = outcome
                .assignments
                .iter()
                .map(|assignment| assignment.path.clone())
                .collect();
            assert!(
                distinct.len() >= min_paths && distinct.len() <= max_paths,
                "clusters {clusters:?} produced {} distinct paths, want {min_paths}..={max_paths}: {:?}",
                distinct.len(),
                paths(&outcome.assignments)
            );
            let placed: BTreeSet<&str> = outcome
                .assignments
                .iter()
                .map(|assignment| assignment.worker_id.as_str())
                .collect();
            assert_eq!(outcome.assignments.len(), items.len(), "{clusters:?}");
            assert_eq!(placed.len(), items.len(), "{clusters:?}");
            assert_eq!(outcome.strategy, GroupingStrategy::Kmeans);
            assert_eq!(outcome.provenance, GroupingProvenance::Embedding);
        }
    }

    #[tokio::test]
    async fn llm_grouping_uses_the_model_runtime() {
        let items = inventory();
        let reply = r#"{"groups":[{"path":["Documents","PDFs","Forms"],"worker_ids":["a","b"]}]}"#;
        let runtime =
            GroupingRuntime::new().with_model(Arc::new(StaticCompletionModel::single(reply)));
        let config = GroupingConfig::default()
            .with_strategy(GroupingStrategy::LlmGrouping)
            .with_max_children(2)
            .with_max_depth(3);
        let outcome = group_items_with(&items, &config, &runtime).await.unwrap();
        assert_eq!(outcome.provenance, GroupingProvenance::Model);
        assert_eq!(outcome.strategy, GroupingStrategy::LlmGrouping);
        assert_eq!(outcome.assignments.len(), 2, "c was left unassigned");
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                ("a".to_string(), owned(&["documents", "pdfs"])),
                ("b".to_string(), owned(&["documents", "pdfs"])),
            ],
            "the reply's three segments are cut to max_depth - 1"
        );
        for assignment in &outcome.assignments {
            assert!(assignment.path.len() <= 2, "{:?}", assignment.path);
        }
    }

    #[tokio::test]
    async fn llm_grouping_falls_back_deterministically() {
        let items = inventory();
        let config = GroupingConfig::default().with_strategy(GroupingStrategy::LlmGrouping);
        let outcome = group_items_with(&items, &config, &GroupingRuntime::new())
            .await
            .unwrap();
        assert_eq!(outcome.strategy, GroupingStrategy::Deterministic);
        assert_eq!(
            outcome.provenance,
            GroupingProvenance::DeterministicFallback
        );
        assert_eq!(
            outcome.assignments,
            group_items(&items, &GroupingConfig::default()).unwrap()
        );

        let scripted =
            GroupingRuntime::new().with_model(Arc::new(StaticCompletionModel::single("not json")));
        let outcome = group_items_with(&items, &config, &scripted).await.unwrap();
        assert_eq!(outcome.strategy, GroupingStrategy::Deterministic);
        assert_eq!(
            outcome.provenance,
            GroupingProvenance::DeterministicFallback
        );
    }

    #[tokio::test]
    async fn llm_grouping_surfaces_an_unusable_reply_in_error_mode() {
        let items = inventory();
        let config = GroupingConfig::default().with_strategy(GroupingStrategy::LlmGrouping);

        let missing = GroupingRuntime::new().with_fallback(FallbackMode::Error);
        let error = group_items_with(&items, &config, &missing)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.message().contains("with_model"), "{error}");

        let unusable = GroupingRuntime::new()
            .with_model(Arc::new(StaticCompletionModel::single("not json")))
            .with_fallback(FallbackMode::Error);
        let error = group_items_with(&items, &config, &unusable)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("llm_grouping"), "{error}");
    }

    #[test]
    fn grouping_outcome_round_trips_and_provenance_is_snake_case() {
        let outcome = GroupingOutcome {
            assignments: vec![GroupAssignment {
                worker_id: "w".to_string(),
                path: owned(&["docs"]),
            }],
            strategy: GroupingStrategy::Kmeans,
            provenance: GroupingProvenance::Model,
        };
        let encoded = serde_json::to_string(&outcome).unwrap();
        assert!(encoded.contains("\"provenance\":\"model\""), "{encoded}");
        assert!(encoded.contains("\"strategy\":\"kmeans\""), "{encoded}");
        let decoded: GroupingOutcome = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, outcome);
        let tolerant: GroupingOutcome = serde_json::from_str(r#"{"unknown": 1}"#).unwrap();
        assert_eq!(tolerant, GroupingOutcome::default());
        assert_eq!(tolerant.provenance, GroupingProvenance::Rule);
        assert_eq!(tolerant.provenance, GroupingProvenance::default());
        assert_eq!(tolerant.strategy, GroupingStrategy::Deterministic);

        for (provenance, text) in [
            (GroupingProvenance::Rule, "rule"),
            (GroupingProvenance::Embedding, "embedding"),
            (GroupingProvenance::Model, "model"),
            (
                GroupingProvenance::DeterministicFallback,
                "deterministic_fallback",
            ),
        ] {
            let quoted = format!("\"{text}\"");
            assert_eq!(serde_json::to_string(&provenance).unwrap(), quoted);
            assert_eq!(
                serde_json::from_str::<GroupingProvenance>(&quoted).unwrap(),
                provenance
            );
        }
    }

    #[test]
    fn grouping_runtime_debug_names_the_model_and_embedder() {
        let model = Arc::new(StaticCompletionModel::single("{}"));
        let runtime = full_runtime(&model);
        let debug = format!("{runtime:?}");
        assert!(debug.contains("static-completion-model"), "{debug}");
        assert!(debug.contains("hash-embedding"), "{debug}");
        assert!(!runtime.is_empty());

        let empty = GroupingRuntime::new();
        assert!(empty.is_empty());
        assert!(empty.embedding().is_none());
        assert!(empty.model().is_none());
        assert_eq!(empty.fallback(), FallbackMode::Deterministic);
        assert!(!GroupingRuntime::new()
            .with_embedding(Arc::new(HashEmbedding::new(8)))
            .is_empty());
        assert_eq!(
            empty.with_fallback(FallbackMode::Error).fallback(),
            FallbackMode::Error
        );
    }

    #[test]
    fn place_assignments_requires_one_candidate_per_item() {
        let items = inventory();
        let ordered = sorted_items(&items);
        let error = place_assignments(&ordered, &GroupingConfig::default(), &[]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("grouping candidates"), "{error}");
    }
}
