//! Grouping of scanned items into tree branches.
//!
//! Ports the grouping stage of Python `openjiuwen.symphony.retrieval.build.tree`
//! (`grouping.py`, `builder.py`, `preset_writer.py`, openJiuwen SDK v0.1.20) as a
//! deterministic, inspectable substitute for its model-driven core.
//!
//! # Why a substitute
//!
//! Python derives the branch path of a skill with an LLM: `grouping.py` renders every
//! batch of skills into `GROUP_DISCOVERY_PROMPT`, asks the model for group definitions,
//! merges the discovery passes with `GROUP_MERGE_PROMPT`, and then assigns each skill
//! with `SKILL_ASSIGNMENT_PROMPT` — retried, repaired, and (optionally) clustered with
//! faiss k-means when the count is large. Neither the model nor faiss exists offline, and
//! neither is reproducible: the same skills can land in different buckets on two runs.
//!
//! The port therefore derives each item's branch path with plain, inspectable rules —
//! the classification path, then the item's first tag, then its leading name token — and
//! then enforces Python's structural bounds (`BRANCHING_FACTOR`, `MAX_DEPTH`) with a
//! bottom-up rebalancing pass. The strategies `LlmGrouping`, `FaissKmeans` and `Kmeans`
//! (see [`GroupingStrategy`](crate::retrieval::build::grouping::GroupingStrategy)) keep
//! Python's names and report [`unsupported`](crate::error::unsupported) instead of
//! pretending to group.
//!
//! # Algorithm
//!
//! [`group_items`](crate::retrieval::build::grouping::group_items) runs five steps, all
//! deterministic (no clock, no hash-map order: items are iterated in `worker_id` order and
//! every map and set here is a `BTreeMap` or a `BTreeSet`):
//!
//! 1. **Candidate segments.** For the `Deterministic` and `CategoryPath` strategies the
//!    path comes from the item's `classification` when non-blank: it splits on `.`, `/`,
//!    `>`, `|` and `,`, every chunk is slugged with
//!    [`slug_term`](crate::retrieval::build::tree_text::slug_term), and blank or repeated
//!    chunks are dropped. A blank classification falls back to the item's first non-blank
//!    tag, then (`Deterministic` only) to the item's leading name token: the first
//!    non-empty token of
//!    [`normalize_name_key`](crate::model::naming::normalize_name_key) that is not a
//!    routing-noise word such as `tool` or `helper`. The `NameTokens` strategy applies
//!    only that last rule. An item with nothing usable is placed under
//!    [`GENERAL_SEGMENT`](crate::retrieval::build::grouping::GENERAL_SEGMENT).
//! 2. **Depth truncation.** The path is cut to `max_depth - 1` segments, because the tree
//!    root already occupies depth 1 (Python's `MAX_DEPTH` counts the root the same way).
//! 3. **Segment tree.** Items are inserted into a segment trie keyed by the slugged
//!    segments, so items sharing a prefix share a branch.
//! 4. **Rebalancing.** Each node is fixed bottom-up: more than `max_children` child
//!    branches keep their first `max_children - 1` segments in sorted order and fold the
//!    remainder into one
//!    [`OTHER_SEGMENT`](crate::retrieval::build::grouping::OTHER_SEGMENT) child; more than
//!    `max_children` direct items are bucketed by the next name token, and a bucket that
//!    is still over the bound is chunked into
//!    [`PART_SEGMENT_PREFIX`](crate::retrieval::build::grouping::PART_SEGMENT_PREFIX)
//!    children of at most `max_children` items each. Both bounds need a free level below
//!    the node, so at `max_depth` they are soft — the same place Python's builder stops
//!    splitting.
//! 5. **Taxonomy override.** When
//!    [`GroupingConfig::taxonomy`](crate::retrieval::build::grouping::GroupingConfig::taxonomy)
//!    is non-empty, an item is instead assigned to the preset path with the highest token
//!    overlap ([`text_tokens`](crate::retrieval::build::tree_text::text_tokens)) between
//!    its derived segments and the preset segments, tie-broken by the longer path, then by
//!    the lexicographically smallest joined path. A preset path with no overlap at all
//!    leaves the item directly under the root.
//!
//! The result is one
//! [`GroupAssignment`](crate::retrieval::build::grouping::GroupAssignment) per item,
//! ordered by `worker_id`, whose `path` excludes the tree root and contains `cid`-ready
//! slug segments — exactly what
//! [`build_nodes`](crate::retrieval::build::tree::build_nodes) consumes.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{unsupported, Result};
use crate::model::naming::normalize_name_key;
use crate::retrieval::build::scanner::SourceItem;
use crate::retrieval::build::tree_text::{slug_term, text_tokens};

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
    /// Python's two-phase LLM grouping — a typed deferral.
    LlmGrouping,
    /// Python's faiss k-means clustering — a typed deferral.
    FaissKmeans,
    /// The spelling `Kmeans` maps to the same deferral as [`Self::FaissKmeans`].
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
    /// The two model/ML strategies need a completion model or a faiss clusterer, so
    /// [`group_items`] reports them as [`unsupported`].
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            Self::Deterministic | Self::CategoryPath | Self::NameTokens
        )
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
}

impl Default for GroupingConfig {
    fn default() -> Self {
        Self {
            strategy: GroupingStrategy::Deterministic,
            max_children: DEFAULT_MAX_CHILDREN,
            max_depth: DEFAULT_MAX_DEPTH,
            root_segment: DEFAULT_ROOT_SEGMENT.to_string(),
            taxonomy: Vec::new(),
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

/// Derives the category segments of one item (public for tests and diagnostics).
///
/// The segments are slugged with [`slug_term`] and returned in path order, without the
/// root and without depth truncation (that is [`group_items`]'s job). An item the chosen
/// rule cannot classify gets `["general"]`, so the result is never empty for a supported
/// strategy. For a deferred strategy no rule applies, so this total function answers with
/// the deterministic rule while [`group_items`] reports the deferral.
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
    let effective = if strategy.is_supported() {
        strategy
    } else {
        GroupingStrategy::Deterministic
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

/// The [`unsupported`] reason for a strategy this crate cannot run, or `None` for a
/// strategy it can.
///
/// The wording names the machinery the strategy actually needs, so an `llm_grouping` build
/// does not report a k-means deferral.
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

/// Places every item under the root; deterministic, sorted by `worker_id`.
///
/// The returned assignments are ordered by `worker_id` (ties keep the caller's item
/// order), and each `path` is the branch path below the root with slugged segments.
///
/// # Errors
///
/// [`GroupingStrategy::LlmGrouping`] needs a completion model and
/// [`GroupingStrategy::FaissKmeans`]/[`GroupingStrategy::Kmeans`] a faiss clusterer, so
/// they return [`unsupported`] (an
/// [`Error::Unsupported`](crate::error::Error)) naming that strategy instead of grouping.
pub fn group_items(items: &[SourceItem], config: &GroupingConfig) -> Result<Vec<GroupAssignment>> {
    if let Some(reason) = deferral_reason(config.strategy) {
        return Err(unsupported(reason));
    }
    let limits = Limits::from_config(config);
    let mut ordered: Vec<&SourceItem> = items.iter().collect();
    ordered.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
    let taxonomy = normalize_taxonomy(&config.taxonomy, limits);
    let mut root = SegmentNode::default();
    for (index, item) in ordered.iter().enumerate() {
        let mut segments = category_segments(item, config.strategy);
        if !taxonomy.is_empty() {
            segments = match_taxonomy(&segments, &taxonomy);
        }
        segments.truncate(limits.max_depth.saturating_sub(1));
        root.insert(index, &segments);
    }
    let root = rebalance(root, 1, &mut Vec::new(), &ordered, limits);
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

    use super::*;
    use crate::error::is_unsupported;
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
        assert!(GroupingStrategy::Deterministic.is_supported());
        assert!(GroupingStrategy::CategoryPath.is_supported());
        assert!(GroupingStrategy::NameTokens.is_supported());
        assert!(!GroupingStrategy::LlmGrouping.is_supported());
    }

    #[test]
    fn unsupported_strategies_report_the_deferral() {
        let items = vec![item("w", "Alpha", "docs", &[])];
        for (strategy, reason) in [
            (GroupingStrategy::LlmGrouping, "LLM-guided tree grouping"),
            (GroupingStrategy::FaissKmeans, "faiss k-means tree grouping"),
            (GroupingStrategy::Kmeans, "k-means tree grouping"),
        ] {
            let config = GroupingConfig::default().with_strategy(strategy);
            let error = group_items(&items, &config).unwrap_err();
            assert!(is_unsupported(&error), "{strategy:?}: {error}");
            assert!(error.message().starts_with(reason), "{strategy:?}: {error}");
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
}
