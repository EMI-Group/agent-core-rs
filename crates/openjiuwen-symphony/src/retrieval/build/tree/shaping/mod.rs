//! LLM tree shaping: the four opt-in model stages of the index build.
//!
//! Python's `openjiuwen.symphony.retrieval.build.tree` shapes the capability tree with a
//! model at four points. This module ports those four stages onto the crate's
//! [`CompletionModel`](crate::llm::CompletionModel) seam and gives each of them a
//! documented deterministic fallback, so
//! a build never needs a model:
//!
//! | Stage | What the model does | Deterministic fallback |
//! | --- | --- | --- |
//! | [`one_shot`](crate::retrieval::build::tree::shaping::one_shot) | places every item in one prompt | [`group_items`] |
//! | [`expansion`](crate::retrieval::build::tree::shaping::expansion) | splits one skeleton branch per prompt, breadth first | that branch's own subtree |
//! | [`repair`](crate::retrieval::build::tree::shaping::repair) | minimally fixes a node list the schema guard rejected | empty-branch pruning |
//! | [`equivalence`](crate::retrieval::build::tree::shaping::equivalence) | merges semantically equal sibling branches | normalized-name collapsing |
//!
//! The default build calls none of them: [`GroupingConfig`] grouping is already
//! deterministic, and a stage only runs when a caller asks for it with a
//! [`ShapingMode`](crate::retrieval::build::tree::shaping::ShapingMode) and passes a model.
//!
//! # The seam
//!
//! Every stage receives `Option<&dyn CompletionModel>` from its caller (this crate never
//! stores one) and reads its knobs from
//! [`ShapingConfig`](crate::retrieval::build::tree::shaping::ShapingConfig). One prompt is
//! rendered per call, and the reply is parsed tolerantly: unknown and repeated worker ids,
//! blank segments and unusable entries are dropped instead of failing the run, and a
//! reply with no parseable JSON object is a *soft* failure. Prompt payloads are
//! byte-stable — items are rendered in `(worker_id, name)` order, paths are joined with
//! `/`, descriptions are cut to `description_limit` characters, and at most
//! `max_items_per_prompt` entries are sent (`0` means unlimited), taking the first entries
//! of that sorted order; whatever a prompt leaves out keeps the stage's deterministic
//! answer (those items are backfilled, never dropped).
//!
//! # Provenance
//!
//! [`ShapingProvenance`](crate::retrieval::build::tree::shaping::ShapingProvenance) names
//! where a run's answer came from, and the rule is shared by every stage:
//!
//! - `Model` — the model produced at least one usable answer, so its placements are in the
//!   result.
//! - `DeterministicFallback` — a model was configured and called, but every answer was
//!   unusable (or the call failed), so the stage's deterministic rule answered. Under
//!   [`FallbackMode::Deterministic`](crate::llm::FallbackMode) exactly one
//!   `tracing::warn!` is emitted per run.
//! - `Skipped` — no model was called at all: no model configured, `max_calls == 0`, or
//!   [`ShapingMode::None`](crate::retrieval::build::tree::shaping::ShapingMode::None).
//!
//! A reply is *usable* when it carries the stage's array and that array yields at least one
//! applicable entry. Under [`FallbackMode::Error`](crate::llm::FallbackMode) an
//! unusable reply becomes
//! `Error::llm("tree shaping <stage>: the model reply contained no usable JSON object")`
//! and a model invocation failure is returned unchanged; the other stage-level failures
//! (a repaired tree that stays invalid, an equivalence pass that would lose an item) are
//! `Error::Llm`/`Error::Validation` in both modes.
//!
//! # Shared rules
//!
//! No reply is trusted, and every stage applies the same normalisation before returning:
//!
//! - every segment is slugged with
//!   [`slug_term`] and blank or repeated
//!   segments are dropped;
//! - a path keeps at most `max_depth - 1` segments, because the tree root occupies depth
//!   1;
//! - a parent with more than `max_children` child segments keeps its first
//!   `max_children - 1` segments in sorted order and folds the rest into one `other`
//!   branch. That needs a free level below the parent, so the fan-out bound is hard above
//!   `max_depth` and soft exactly at it — the same place the deterministic grouping pass
//!   stops splitting;
//! - the result holds one placement per input item (a worker id the input repeats keeps one
//!   placement per item, in item order) and is sorted by `(worker_id, path)`.
//!
//! `dropped` lists, sorted and de-duplicated, every worker id a reply named that the stage
//! could not use: an id the input does not have, or a repeat of an id already placed.
//!
//! # Dispatch
//!
//! [`apply`](crate::retrieval::build::tree::shaping::apply) is the single entry point for
//! the three stages that answer an
//! [`ShapingOutcome`](crate::retrieval::build::tree::shaping::ShapingOutcome):
//! `None` returns the caller's skeleton untouched with provenance `Skipped`, and the others
//! delegate to their own function. Repair cannot be dispatched: it needs a node list and
//! its diagnostics, so it is called as
//! [`repair`](crate::retrieval::build::tree::shaping::repair) and `apply` answers
//! `Error::Validation` naming the `mode` argument.
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{validation, Error, Result};
use crate::llm::{
    complete_json, string_field, string_list, truncate_text, CompletionModel, FallbackMode,
};
use crate::model::normalize_name_key;
use crate::retrieval::build::grouping::{
    group_items, GroupAssignment, GroupingConfig, OTHER_SEGMENT,
};
use crate::retrieval::build::scanner::SourceItem;
use crate::retrieval::build::tree_text::slug_term;

/// Merging of semantically equal sibling branches.
pub mod equivalence;
/// Breadth-first expansion of a deterministic skeleton.
pub mod expansion;
/// One-prompt placement of every item.
pub mod one_shot;
/// Minimal repair of a node list the schema guard rejected.
pub mod repair;

/// The default number of model calls one shaping run may make.
pub const DEFAULT_MAX_CALLS: usize = 8;
/// The default number of entries one shaping prompt describes; `0` means unlimited.
pub const DEFAULT_MAX_ITEMS_PER_PROMPT: usize = 24;
/// The default number of characters of one item's description a prompt carries.
pub const DEFAULT_DESCRIPTION_LIMIT: usize = 240;

/// The failure message every stage reports when its reply carries nothing usable.
const NO_USABLE_REPLY: &str = "the model reply contained no usable JSON object";

/// How a build may hand its tree to a model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShapingMode {
    /// No model stage: the build keeps its deterministic tree.
    #[default]
    None,
    /// One prompt places every item.
    OneShot,
    /// A bounded breadth-first expansion of a deterministic skeleton.
    Expansion,
    /// A minimal fix for a node list the schema guard rejected.
    Repair,
    /// Merging of semantically equal sibling branches.
    Equivalence,
}

/// The knobs one shaping stage reads.
///
/// The model itself is passed to the stage rather than stored here, so a configuration is
/// plain data that round-trips through JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ShapingConfig {
    /// Which stage [`apply`] runs.
    pub mode: ShapingMode,
    /// What a stage does when its model call fails or answers unusably.
    pub fallback: FallbackMode,
    /// The most model calls one run may make; `0` skips the stage entirely.
    pub max_calls: usize,
    /// The most entries one prompt describes; `0` means unlimited.
    pub max_items_per_prompt: usize,
    /// The most characters of one item's description a prompt carries.
    pub description_limit: usize,
}

impl Default for ShapingConfig {
    fn default() -> Self {
        Self {
            mode: ShapingMode::None,
            fallback: FallbackMode::Deterministic,
            max_calls: DEFAULT_MAX_CALLS,
            max_items_per_prompt: DEFAULT_MAX_ITEMS_PER_PROMPT,
            description_limit: DEFAULT_DESCRIPTION_LIMIT,
        }
    }
}

impl ShapingConfig {
    /// Builds the default configuration: [`ShapingMode::None`],
    /// [`FallbackMode::Deterministic`] and the `DEFAULT_*` knobs.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the stage [`apply`] runs.
    #[must_use]
    pub fn with_mode(mut self, mode: ShapingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the fallback mode.
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackMode) -> Self {
        self.fallback = fallback;
        self
    }

    /// Sets the model-call budget.
    #[must_use]
    pub fn with_max_calls(mut self, max_calls: usize) -> Self {
        self.max_calls = max_calls;
        self
    }

    /// Sets how many entries one prompt may describe (`0` means unlimited).
    #[must_use]
    pub fn with_max_items_per_prompt(mut self, max_items_per_prompt: usize) -> Self {
        self.max_items_per_prompt = max_items_per_prompt;
        self
    }

    /// Sets how many characters of an item's description a prompt carries.
    #[must_use]
    pub fn with_description_limit(mut self, description_limit: usize) -> Self {
        self.description_limit = description_limit;
        self
    }
}

/// Where a shaping run's answer came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShapingProvenance {
    /// The model produced at least one usable answer.
    Model,
    /// A model was called but answered unusably, so the deterministic rule answered.
    DeterministicFallback,
    /// No model was called at all.
    #[default]
    Skipped,
}

/// The placements a shaping stage produced, plus how it produced them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ShapingOutcome {
    /// One placement per input item, sorted by `(worker_id, path)`.
    pub assignments: Vec<GroupAssignment>,
    /// The stage that produced the placements.
    pub mode: ShapingMode,
    /// Where the placements came from.
    pub provenance: ShapingProvenance,
    /// How many model calls the stage made.
    pub model_calls: usize,
    /// The worker ids a reply named that the stage could not use, sorted and deduplicated.
    pub dropped: Vec<String>,
}

impl ShapingOutcome {
    /// Builds an outcome that dropped no worker id.
    #[must_use]
    pub fn new(
        assignments: Vec<GroupAssignment>,
        mode: ShapingMode,
        provenance: ShapingProvenance,
        model_calls: usize,
    ) -> Self {
        Self {
            assignments,
            mode,
            provenance,
            model_calls,
            dropped: Vec::new(),
        }
    }

    /// Whether the placements came from the model.
    #[must_use]
    pub fn is_model(&self) -> bool {
        self.provenance == ShapingProvenance::Model
    }
}

/// Runs the stage `mode` selects; the documented single entry point for the outcome stages.
///
/// [`ShapingMode::None`] answers the caller's `skeleton` unchanged (even when its segments
/// are not slugged) with provenance
/// [`Skipped`](crate::retrieval::build::tree::shaping::ShapingProvenance::Skipped) and no
/// model call. [`ShapingMode::OneShot`],
/// [`Expansion`](crate::retrieval::build::tree::shaping::ShapingMode::Expansion) and
/// [`Equivalence`](crate::retrieval::build::tree::shaping::ShapingMode::Equivalence)
/// delegate to their own stage, and
/// [`Repair`](crate::retrieval::build::tree::shaping::ShapingMode::Repair) is refused
/// because it needs a node list and its diagnostics.
///
/// # Errors
///
/// The delegated stage's errors — a failed determinism rule
/// ([`Error::Unsupported`](crate::error::Error) for a grouping strategy this crate cannot
/// run), the propagated model failure under [`FallbackMode::Error`], or a repaired tree
/// that stays invalid — plus `Error::Validation` on `mode` for `Repair`.
pub async fn apply(
    mode: ShapingMode,
    items: &[SourceItem],
    skeleton: &[GroupAssignment],
    config: &GroupingConfig,
    shaping: &ShapingConfig,
    model: Option<&dyn CompletionModel>,
) -> Result<ShapingOutcome> {
    match mode {
        ShapingMode::None => Ok(skipped(skeleton, ShapingMode::None)),
        ShapingMode::OneShot => one_shot::one_shot(items, config, shaping, model).await,
        ShapingMode::Expansion => {
            expansion::expansion(items, skeleton, config, shaping, model).await
        }
        ShapingMode::Equivalence => {
            equivalence::equivalence(skeleton, config, shaping, model).await
        }
        ShapingMode::Repair => Err(validation(
            "mode",
            "repair needs the node list and its diagnostics; call `repair` directly",
        )),
    }
}

/// The normalised depth and fan-out bounds a shaped path must satisfy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PathLimits {
    /// The branch fan-out bound; at least `1`.
    max_children: usize,
    /// The tree depth bound, counting the root; at least `1`.
    max_depth: usize,
}

impl PathLimits {
    /// Normalizes a configuration's bounds, mapping `0` to `1`.
    fn from_config(config: &GroupingConfig) -> Self {
        Self {
            max_children: config.max_children.max(1),
            max_depth: config.max_depth.max(1),
        }
    }

    /// How many segments a path below the root may carry.
    fn max_segments(self) -> usize {
        self.max_depth.saturating_sub(1)
    }

    /// Whether a branch at `cid_depth` may fold its overflow into an `other` child.
    ///
    /// The fold needs a free level below the branch, so a branch at the depth limit keeps
    /// its overflow — the bound is soft exactly at `max_depth`.
    fn can_fold(self, cid_depth: usize) -> bool {
        cid_depth < self.max_depth && self.max_children > 0
    }
}

/// One node of the intermediate segment trie the bound and collapse passes build.
#[derive(Debug, Clone, Default)]
struct PathTrie {
    /// Indices of the paths that end exactly here.
    leaves: Vec<usize>,
    /// Child nodes keyed by their segment.
    children: BTreeMap<String, PathTrie>,
}

impl PathTrie {
    /// Inserts `index` at `segments`.
    fn insert(&mut self, index: usize, segments: &[String]) {
        match segments.split_first() {
            Some((head, tail)) => self
                .children
                .entry(head.clone())
                .or_default()
                .insert(index, tail),
            None => self.leaves.push(index),
        }
    }

    /// Merges `other` into this node, keeping leaves and same-named children.
    fn absorb(&mut self, other: Self) {
        self.leaves.extend(other.leaves);
        for (name, child) in other.children {
            match self.children.entry(name) {
                Entry::Occupied(mut entry) => entry.get_mut().absorb(child),
                Entry::Vacant(entry) => {
                    entry.insert(child);
                }
            }
        }
    }

    /// Folds every group of sibling segments sharing a normalized name under the
    /// lexicographically smallest member of the group.
    ///
    /// The canonical segment is a property of the group, not of its first insertion, so two
    /// runs over the same input collapse identically.
    fn collapse(self) -> Self {
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for name in self.children.keys() {
            groups
                .entry(normalize_name_key(name))
                .or_default()
                .push(name.clone());
        }
        let mut children: BTreeMap<String, PathTrie> = BTreeMap::new();
        for names in groups.into_values() {
            let Some(canonical) = names.first().cloned() else {
                continue;
            };
            let mut merged = PathTrie::default();
            for name in &names {
                if let Some(child) = self.children.get(name) {
                    merged.absorb(child.clone());
                }
            }
            children.insert(canonical, merged.collapse());
        }
        PathTrie {
            leaves: self.leaves,
            children,
        }
    }

    /// Enforces the fan-out bound bottom-up, folding overflow under `other`.
    fn rebalance(mut self, cid_depth: usize, limits: PathLimits) -> Self {
        let children = std::mem::take(&mut self.children);
        for (name, child) in children {
            self.children
                .insert(name, child.rebalance(cid_depth + 1, limits));
        }
        if self.children.len() > limits.max_children && limits.can_fold(cid_depth) {
            let keep = limits.max_children.saturating_sub(1);
            let names: Vec<String> = self.children.keys().cloned().collect();
            let mut kept: BTreeMap<String, PathTrie> = BTreeMap::new();
            let mut merged = PathTrie::default();
            for (position, name) in names.into_iter().enumerate() {
                let Some(child) = self.children.remove(&name) else {
                    continue;
                };
                if position < keep {
                    kept.insert(name, child);
                } else {
                    merged.absorb(child);
                }
            }
            // `other` is reserved for the fold, so a real segment named `other` joins it.
            if let Some(existing) = kept.remove(OTHER_SEGMENT) {
                merged.absorb(existing);
            }
            kept.insert(
                OTHER_SEGMENT.to_string(),
                merged.rebalance(cid_depth + 1, limits),
            );
            self.children = kept;
        }
        self
    }

    /// Records the final path of every index.
    fn collect(&self, path: &mut Vec<String>, out: &mut [Vec<String>]) {
        for index in &self.leaves {
            if let Some(slot) = out.get_mut(*index) {
                slot.clone_from(path);
            }
        }
        for (name, child) in &self.children {
            path.push(name.clone());
            child.collect(path, out);
            path.pop();
        }
    }
}

/// Slugs `path`, dropping blank and repeated segments and cutting it to `limit` segments.
fn slug_segments(path: &[String], limit: usize) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for segment in path {
        if segments.len() >= limit {
            break;
        }
        let slug = slug_term(segment, "");
        if slug.is_empty() || !seen.insert(slug.clone()) {
            continue;
        }
        segments.push(slug);
    }
    segments
}

/// Trims `path`'s segments without slugging them, dropping blanks, repeats and overflow.
fn raw_segments(path: &[String], limit: usize) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for segment in path {
        if segments.len() >= limit {
            break;
        }
        let segment = segment.trim();
        if segment.is_empty() || !seen.insert(segment.to_string()) {
            continue;
        }
        segments.push(segment.to_string());
    }
    segments
}

/// Applies the breadth bound to already-normalized paths, folding overflow under `other`.
fn bound_paths(paths: &mut [Vec<String>], limits: PathLimits) {
    let mut trie = PathTrie::default();
    for (index, path) in paths.iter().enumerate() {
        trie.insert(index, path);
    }
    let trie = trie.rebalance(1, limits);
    let mut bounded: Vec<Vec<String>> = vec![Vec::new(); paths.len()];
    trie.collect(&mut Vec::new(), &mut bounded);
    for (path, bounded) in paths.iter_mut().zip(bounded) {
        *path = bounded;
    }
}

/// Slugs and bounds every placement's path in place.
fn normalize_paths(assignments: &mut [GroupAssignment], limits: PathLimits) {
    let max_segments = limits.max_segments();
    for assignment in assignments.iter_mut() {
        assignment.path = slug_segments(&assignment.path, max_segments);
    }
    let mut paths: Vec<Vec<String>> = assignments
        .iter()
        .map(|assignment| assignment.path.clone())
        .collect();
    bound_paths(&mut paths, limits);
    for (assignment, path) in assignments.iter_mut().zip(paths) {
        assignment.path = path;
    }
}

/// Applies the shared normalisation and sorting, then builds the outcome.
fn finalize(
    mut assignments: Vec<GroupAssignment>,
    limits: PathLimits,
    mode: ShapingMode,
    provenance: ShapingProvenance,
    model_calls: usize,
    dropped: BTreeSet<String>,
) -> ShapingOutcome {
    normalize_paths(&mut assignments, limits);
    assignments.sort_by(|left, right| {
        (left.worker_id.as_str(), left.path.as_slice())
            .cmp(&(right.worker_id.as_str(), right.path.as_slice()))
    });
    ShapingOutcome {
        assignments,
        mode,
        provenance,
        model_calls,
        dropped: dropped.into_iter().collect(),
    }
}

/// Reports `assignments` unchanged as a run that never called a model.
fn skipped(assignments: &[GroupAssignment], mode: ShapingMode) -> ShapingOutcome {
    ShapingOutcome::new(assignments.to_vec(), mode, ShapingProvenance::Skipped, 0)
}

/// Groups `items` deterministically and reports the answer as a fallback outcome.
fn deterministic_outcome(
    items: &[SourceItem],
    config: &GroupingConfig,
    mode: ShapingMode,
    model_calls: usize,
) -> Result<ShapingOutcome> {
    let limits = PathLimits::from_config(config);
    Ok(finalize(
        group_items(items, config)?,
        limits,
        mode,
        ShapingProvenance::DeterministicFallback,
        model_calls,
        BTreeSet::new(),
    ))
}

/// The deterministic grouping of `items` as a `worker_id` → path map.
fn deterministic_paths(
    items: &[SourceItem],
    config: &GroupingConfig,
) -> Result<BTreeMap<String, Vec<String>>> {
    Ok(path_map(&group_items(items, config)?))
}

/// Keys a list of placements by worker id, keeping the last placement of a repeated id.
fn path_map(assignments: &[GroupAssignment]) -> BTreeMap<String, Vec<String>> {
    assignments
        .iter()
        .map(|assignment| (assignment.worker_id.clone(), assignment.path.clone()))
        .collect()
}

/// Counts how often every worker id appears, so a pass can prove it lost nothing.
fn worker_counts(assignments: &[GroupAssignment]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for assignment in assignments {
        *counts.entry(assignment.worker_id.clone()).or_default() += 1;
    }
    counts
}

/// Splits a model-authored path string on `/`, `.` and `>`.
fn split_path(text: &str) -> Vec<String> {
    text.split(['/', '.', '>'])
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

/// Reads a model-authored path, accepting an array of segments or one joined string.
///
/// Array entries are treated as segments and additionally split on the same separators, so
/// a model that answers `["documents/pdf"]` is still understood.
fn parse_path(value: &Value) -> Vec<String> {
    match value {
        Value::Array(entries) => entries
            .iter()
            .filter_map(Value::as_str)
            .flat_map(split_path)
            .collect(),
        Value::String(text) => split_path(text),
        _ => Vec::new(),
    }
}

/// Reads one path-valued field of a JSON object.
fn path_field(value: &Value, field: &str) -> Vec<String> {
    value.get(field).map_or_else(Vec::new, parse_path)
}

/// Whether `path` starts with `prefix`.
fn starts_with(path: &[String], prefix: &[String]) -> bool {
    path.get(..prefix.len()) == Some(prefix)
}

/// Whether `path` continues below `prefix` (at least one segment deeper).
fn below(path: &[String], prefix: &[String]) -> bool {
    path.len() > prefix.len() && starts_with(path, prefix)
}

/// The items worth describing to a model, sorted by `(worker_id, name)`.
fn sorted_items(items: &[SourceItem]) -> Vec<&SourceItem> {
    let mut ordered: Vec<&SourceItem> = items.iter().collect();
    ordered.sort_by(|left, right| {
        (left.worker_id.as_str(), left.name.as_str())
            .cmp(&(right.worker_id.as_str(), right.name.as_str()))
    });
    ordered
}

/// How many of an ordered list a prompt may carry (`max_items_per_prompt == 0` is
/// unlimited).
fn prompt_len(total: usize, shaping: &ShapingConfig) -> usize {
    if shaping.max_items_per_prompt == 0 {
        total
    } else {
        total.min(shaping.max_items_per_prompt)
    }
}

/// The prompt view of one item: its identity plus a shortened description.
fn item_value(item: &SourceItem, description_limit: usize) -> Value {
    let name = if item.name.trim().is_empty() {
        item.worker_id.clone()
    } else {
        item.name.clone()
    };
    json!({
        "worker_id": item.worker_id,
        "name": truncate_text(&name, description_limit),
        "description": truncate_text(item.description.trim(), description_limit),
        "classification": item.classification,
        "tags": item.tags,
    })
}

/// A parsed assignment reply: the usable placements plus the ids the reply could not use.
type AssignmentReply = (BTreeMap<String, Vec<String>>, BTreeSet<String>);

/// Reads `{"assignments":[{"worker_id":"…","path":[…]}]}` against the known worker ids.
///
/// Returns `None` when the reply carries no `assignments` array or no usable entry in it
/// (a usable entry names a known worker id for the first time). Unknown and repeated ids
/// are returned as the second half of the answer.
fn read_assignments(value: &Value, known: &BTreeSet<&str>) -> Option<AssignmentReply> {
    let entries = value.get("assignments")?.as_array()?;
    let mut paths: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut dropped: BTreeSet<String> = BTreeSet::new();
    for entry in entries {
        let Some(worker_id) = string_field(entry, "worker_id") else {
            continue;
        };
        if !known.contains(worker_id.as_str()) || paths.contains_key(&worker_id) {
            dropped.insert(worker_id);
            continue;
        }
        paths.insert(worker_id, path_field(entry, "path"));
    }
    if paths.is_empty() {
        return None;
    }
    Some((paths, dropped))
}

/// Assembles one placement per item, backfilling from `fallback` what the model left out.
fn assemble(
    items: &[SourceItem],
    model_paths: &BTreeMap<String, Vec<String>>,
    fallback: Option<&BTreeMap<String, Vec<String>>>,
) -> Vec<GroupAssignment> {
    items
        .iter()
        .map(|item| GroupAssignment {
            worker_id: item.worker_id.clone(),
            path: model_paths
                .get(&item.worker_id)
                .or_else(|| fallback.and_then(|fallback| fallback.get(&item.worker_id)))
                .cloned()
                .unwrap_or_default(),
        })
        .collect()
}

/// Warns once per run about a failed stage and reports whether it may continue.
///
/// Under [`FallbackMode::Error`] the failure is returned unchanged, so the caller can
/// propagate it; under [`FallbackMode::Deterministic`] the run warns at most once and then
/// answers with its deterministic rule.
fn on_model_failure(
    shaping: &ShapingConfig,
    warned: &mut bool,
    stage: &str,
    error: Error,
) -> Result<()> {
    match shaping.fallback {
        FallbackMode::Error => Err(error),
        FallbackMode::Deterministic => {
            if !*warned {
                tracing::warn!(
                    %error,
                    stage = stage,
                    "tree shaping: the model answer was unusable; using the deterministic answer"
                );
                *warned = true;
            }
            Ok(())
        }
    }
}

/// The typed failure a stage reports for an unusable reply.
fn no_usable_reply(stage: &str) -> Error {
    Error::llm(format!("tree shaping {stage}: {NO_USABLE_REPLY}"))
}

/// Calls the model with one rendered prompt and reports the JSON object it answered.
///
/// Shared by every stage so the soft-failure rule stays in one place: `None` means "no
/// usable JSON object", and a model invocation failure is returned unchanged.
async fn call_model(
    model: &dyn CompletionModel,
    system: &str,
    request: &str,
) -> Result<Option<Value>> {
    complete_json(Some(model), system, request).await
}

/// The `{"worker_ids":[…]}` view of one sibling branch, used by the equivalence prompts.
fn segment_entries(groups: &BTreeMap<Vec<String>, BTreeSet<String>>) -> Vec<Value> {
    groups
        .iter()
        .filter(|(_, segments)| segments.len() > 1)
        .map(|(parent, segments)| {
            json!({
                "parent": parent,
                "segments": segments.iter().cloned().collect::<Vec<String>>(),
            })
        })
        .collect()
}

/// Collects the string entries of a JSON array field, dropping blanks and repeats.
fn strings_field(value: &Value, field: &str) -> Vec<String> {
    string_list(value.get(field))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod fixtures {
    use std::collections::BTreeMap;

    use super::*;
    use crate::retrieval::types::RetrieverItem;

    /// One scanned item with `worker_id`, `name` and `description`.
    pub(super) fn item(worker_id: &str, name: &str, description: &str) -> SourceItem {
        SourceItem {
            worker_id: worker_id.to_string(),
            item: RetrieverItem::default(),
            name: name.to_string(),
            description: description.to_string(),
            content: String::new(),
            skill_path: String::new(),
            classification: String::new(),
            tags: Vec::new(),
            item_type: "skill".to_string(),
            content_hash: String::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Owns a list of segments.
    pub(super) fn owned(segments: &[&str]) -> Vec<String> {
        segments
            .iter()
            .map(|segment| (*segment).to_string())
            .collect()
    }

    /// One placement.
    pub(super) fn assignment(worker_id: &str, path: &[&str]) -> GroupAssignment {
        GroupAssignment {
            worker_id: worker_id.to_string(),
            path: owned(path),
        }
    }

    /// The `(worker_id, path)` pairs of a list, in order.
    pub(super) fn paths(assignments: &[GroupAssignment]) -> Vec<(String, Vec<String>)> {
        assignments
            .iter()
            .map(|assignment| (assignment.worker_id.clone(), assignment.path.clone()))
            .collect()
    }

    /// The same pairs as a `worker_id` → path map.
    pub(super) fn map_of(assignments: &[GroupAssignment]) -> BTreeMap<String, Vec<String>> {
        path_map(assignments)
    }

    /// The default grouping configuration.
    pub(super) fn config() -> GroupingConfig {
        GroupingConfig::default()
    }

    /// The default shaping configuration.
    pub(super) fn shaping() -> ShapingConfig {
        ShapingConfig::default()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::fixtures::{assignment, config, item, owned, paths, shaping};
    use super::*;

    #[test]
    fn shaping_config_defaults_and_builders() {
        let default = ShapingConfig::default();
        assert_eq!(default.mode, ShapingMode::None);
        assert_eq!(default.fallback, FallbackMode::Deterministic);
        assert_eq!(default.max_calls, DEFAULT_MAX_CALLS);
        assert_eq!(default.max_items_per_prompt, DEFAULT_MAX_ITEMS_PER_PROMPT);
        assert_eq!(default.description_limit, DEFAULT_DESCRIPTION_LIMIT);
        assert_eq!(ShapingConfig::new(), default);

        let configured = ShapingConfig::new()
            .with_mode(ShapingMode::Equivalence)
            .with_fallback(FallbackMode::Error)
            .with_max_calls(3)
            .with_max_items_per_prompt(5)
            .with_description_limit(80);
        assert_eq!(configured.mode, ShapingMode::Equivalence);
        assert_eq!(configured.fallback, FallbackMode::Error);
        assert_eq!(configured.max_calls, 3);
        assert_eq!(configured.max_items_per_prompt, 5);
        assert_eq!(configured.description_limit, 80);
    }

    #[test]
    fn shaping_types_round_trip_through_json() {
        let config = ShapingConfig::new()
            .with_mode(ShapingMode::Expansion)
            .with_max_calls(2);
        let encoded = serde_json::to_string(&config).unwrap();
        assert!(encoded.contains("\"mode\":\"expansion\""));
        let decoded: ShapingConfig = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, config);
        let tolerant: ShapingConfig = serde_json::from_str(r#"{"unknown": 1}"#).unwrap();
        assert_eq!(tolerant, ShapingConfig::default());
        let partial: ShapingConfig = serde_json::from_str(r#"{"max_calls": 4}"#).unwrap();
        assert_eq!(partial.max_calls, 4);
        assert_eq!(partial.mode, ShapingMode::None);

        let outcome = ShapingOutcome::new(
            vec![assignment("w", &["docs"])],
            ShapingMode::OneShot,
            ShapingProvenance::Model,
            1,
        );
        let decoded: ShapingOutcome =
            serde_json::from_str(&serde_json::to_string(&outcome).unwrap()).unwrap();
        assert_eq!(decoded, outcome);
        assert!(outcome.is_model());
        let empty: ShapingOutcome = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, ShapingOutcome::default());
        assert!(!empty.is_model());
    }

    #[test]
    fn modes_and_provenance_use_snake_case_spellings() {
        for (mode, spelling) in [
            (ShapingMode::None, "\"none\""),
            (ShapingMode::OneShot, "\"one_shot\""),
            (ShapingMode::Expansion, "\"expansion\""),
            (ShapingMode::Repair, "\"repair\""),
            (ShapingMode::Equivalence, "\"equivalence\""),
        ] {
            assert_eq!(serde_json::to_string(&mode).unwrap(), spelling);
            assert_eq!(serde_json::from_str::<ShapingMode>(spelling).unwrap(), mode);
        }
        for (provenance, spelling) in [
            (ShapingProvenance::Model, "\"model\""),
            (
                ShapingProvenance::DeterministicFallback,
                "\"deterministic_fallback\"",
            ),
            (ShapingProvenance::Skipped, "\"skipped\""),
        ] {
            assert_eq!(serde_json::to_string(&provenance).unwrap(), spelling);
            assert_eq!(
                serde_json::from_str::<ShapingProvenance>(spelling).unwrap(),
                provenance
            );
        }
        assert_eq!(ShapingMode::default(), ShapingMode::None);
        assert_eq!(ShapingProvenance::default(), ShapingProvenance::Skipped);
    }

    #[tokio::test]
    async fn apply_none_answers_the_skeleton_unchanged() {
        let items = vec![item("w", "Alpha", "Docs")];
        let skeleton = vec![assignment("w", &["Not Slugged", "PDF"])];
        let outcome = apply(
            ShapingMode::None,
            &items,
            &skeleton,
            &config(),
            &shaping(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(outcome.assignments, skeleton);
        assert_eq!(outcome.provenance, ShapingProvenance::Skipped);
        assert_eq!(outcome.mode, ShapingMode::None);
        assert_eq!(outcome.model_calls, 0);
        assert_eq!(outcome.dropped, Vec::<String>::new());
    }

    #[tokio::test]
    async fn apply_refuses_repair_and_names_the_argument() {
        let error = apply(ShapingMode::Repair, &[], &[], &config(), &shaping(), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().starts_with("mode:"), "{error}");
        assert!(error.message().contains("repair"), "{error}");
    }

    #[test]
    fn slug_segments_slugs_drops_blanks_and_truncates() {
        assert_eq!(
            slug_segments(&owned(&["Documents", " PDF Fill ", "pdf fill"]), 8),
            owned(&["documents", "pdf-fill"])
        );
        assert_eq!(
            slug_segments(&owned(&["a", "a", "b"]), 8),
            owned(&["a", "b"])
        );
        assert_eq!(
            slug_segments(&owned(&["a", "b", "c"]), 2),
            owned(&["a", "b"])
        );
        assert_eq!(slug_segments(&owned(&["***"]), 8), Vec::<String>::new());
        assert_eq!(slug_segments(&owned(&["a"]), 0), Vec::<String>::new());
        assert_eq!(
            raw_segments(&owned(&["Doc", " doc ", "PDF"]), 8),
            owned(&["Doc", "doc", "PDF"])
        );
        assert_eq!(
            raw_segments(&owned(&["Doc", "  ", "PDF"]), 2),
            owned(&["Doc", "PDF"])
        );
    }

    #[test]
    fn bound_paths_folds_the_overflow_under_other() {
        let mut paths = vec![
            owned(&["alpha"]),
            owned(&["beta"]),
            owned(&["gamma"]),
            owned(&["delta"]),
        ];
        let limits = PathLimits {
            max_children: 2,
            max_depth: 6,
        };
        bound_paths(&mut paths, limits);
        assert_eq!(paths[0], owned(&["alpha"]));
        for path in &paths[1..] {
            assert_eq!(*path, owned(&[OTHER_SEGMENT]));
        }
    }

    #[test]
    fn bound_paths_keeps_the_overlap_of_a_folded_subtree() {
        let mut paths = vec![
            owned(&["alpha", "keep"]),
            owned(&["beta", "one"]),
            owned(&["gamma", "two"]),
        ];
        let limits = PathLimits {
            max_children: 2,
            max_depth: 6,
        };
        bound_paths(&mut paths, limits);
        assert_eq!(paths[0], owned(&["alpha", "keep"]));
        assert_eq!(paths[1], owned(&[OTHER_SEGMENT, "one"]));
        assert_eq!(paths[2], owned(&[OTHER_SEGMENT, "two"]));
    }

    #[test]
    fn bound_paths_is_soft_at_the_depth_limit() {
        let mut paths = vec![owned(&["a", "x"]), owned(&["a", "y"])];
        let limits = PathLimits {
            max_children: 1,
            max_depth: 2,
        };
        bound_paths(&mut paths, limits);
        assert_eq!(paths, vec![owned(&["a", "x"]), owned(&["a", "y"])]);

        let mut shallow = vec![owned(&["x"]), owned(&["y"])];
        bound_paths(&mut shallow, limits);
        assert_eq!(
            shallow,
            vec![owned(&[OTHER_SEGMENT]), owned(&[OTHER_SEGMENT])]
        );
    }

    #[test]
    fn parse_path_accepts_arrays_and_joined_strings() {
        assert_eq!(
            parse_path(&json!(["documents", "pdf"])),
            owned(&["documents", "pdf"])
        );
        assert_eq!(
            parse_path(&json!("documents/pdf.fill>forms")),
            owned(&["documents", "pdf", "fill", "forms"])
        );
        assert_eq!(
            parse_path(&json!(["documents/pdf", "  "])),
            owned(&["documents", "pdf"])
        );
        assert_eq!(parse_path(&json!(7)), Vec::<String>::new());
        assert_eq!(parse_path(&Value::Null), Vec::<String>::new());
    }

    #[test]
    fn read_assignments_drops_unknown_and_repeated_ids() {
        let known: BTreeSet<&str> = ["w1", "w2"].into_iter().collect();
        let value = json!({
            "assignments": [
                {"worker_id": "w1", "path": ["alpha"]},
                {"worker_id": "w1", "path": ["beta"]},
                {"worker_id": "ghost", "path": ["gamma"]},
                {"path": ["delta"]},
                {"worker_id": "w2", "path": "media/video"}
            ]
        });
        let (paths, dropped) = read_assignments(&value, &known).unwrap();
        assert_eq!(paths.get("w1"), Some(&owned(&["alpha"])));
        assert_eq!(paths.get("w2"), Some(&owned(&["media", "video"])));
        assert_eq!(
            dropped.into_iter().collect::<Vec<String>>(),
            vec!["ghost".to_string(), "w1".to_string()]
        );
    }

    #[test]
    fn read_assignments_reports_an_unusable_reply() {
        let known: BTreeSet<&str> = ["w1"].into_iter().collect();
        assert!(read_assignments(&json!({"assignments": []}), &known).is_none());
        assert!(read_assignments(&json!({"nodes": []}), &known).is_none());
        assert!(read_assignments(
            &json!({"assignments": [{"worker_id": "ghost", "path": ["a"]}]}),
            &known
        )
        .is_none());
        assert!(read_assignments(&json!("no object"), &known).is_none());
    }

    #[test]
    fn finalize_normalizes_bounds_and_sorts() {
        let assignments = vec![
            assignment("w2", &["Beta"]),
            assignment("w1", &["Alpha", "One"]),
            assignment("w1", &["Alpha", "Two"]),
        ];
        let limits = PathLimits {
            max_children: 8,
            max_depth: 6,
        };
        let outcome = finalize(
            assignments,
            limits,
            ShapingMode::OneShot,
            ShapingProvenance::Model,
            1,
            ["ghost".to_string()].into_iter().collect(),
        );
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                ("w1".to_string(), owned(&["alpha", "one"])),
                ("w1".to_string(), owned(&["alpha", "two"])),
                ("w2".to_string(), owned(&["beta"])),
            ]
        );
        assert_eq!(outcome.dropped, vec!["ghost".to_string()]);
        assert!(outcome.is_model());
    }

    #[test]
    fn path_prefix_helpers_match_prefixes_only() {
        let prefix = owned(&["docs"]);
        assert!(starts_with(&owned(&["docs"]), &prefix));
        assert!(starts_with(&owned(&["docs", "pdf"]), &prefix));
        assert!(!starts_with(&owned(&["documents"]), &prefix));
        assert!(below(&owned(&["docs", "pdf"]), &prefix));
        assert!(!below(&owned(&["docs"]), &prefix));
        assert!(!below(&owned(&["media"]), &prefix));
    }

    #[test]
    fn prompt_len_honours_zero_as_unlimited() {
        let unlimited = ShapingConfig::default().with_max_items_per_prompt(0);
        assert_eq!(prompt_len(100, &unlimited), 100);
        let bounded = ShapingConfig::default().with_max_items_per_prompt(3);
        assert_eq!(prompt_len(100, &bounded), 3);
        assert_eq!(prompt_len(2, &bounded), 2);
    }

    #[test]
    fn item_value_shortens_the_description() {
        let long = item("w", "Name", &"x".repeat(50));
        let value = item_value(&long, 4);
        assert_eq!(value["worker_id"], json!("w"));
        assert_eq!(value["name"], json!("Name"));
        assert_eq!(value["description"], json!("xxxx"));
    }

    #[test]
    fn worker_counts_counts_repeats() {
        let counts = worker_counts(&[assignment("w", &["a"]), assignment("w", &["b"])]);
        assert_eq!(counts.get("w"), Some(&2));
    }
}
