//! The pure decision layer of the consolidation pipeline.
//!
//! Everything here is clock-free and model-free: the types describe one decision set,
//! [`parse_consolidation_response`] turns a model answer into those decisions, and
//! [`deterministic_plan`] is the rule fallback. No item in this module performs I/O, reads
//! a clock, or calls a model, which is what lets [`super::LlmConsolidator`] run the same
//! code path for a model answer, a degraded answer, and an offline test.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::recall::keyword::jaccard;
use crate::record::normalize_text;

use super::super::model::DreamMetrics;
use super::{DreamCandidate, DEFAULT_FALLBACK_SIMILARITY};

/// One decision the consolidation pipeline applies.
///
/// The wire shape is internally tagged by `"action"`, for example
/// `{"action":"merge","target_id":"a","source_ids":["b"],"content":"…"}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum DreamConsolidationAction {
    /// Fold `source_ids` into `target_id`, whose content becomes `content`.
    Merge {
        /// Record that stays.
        target_id: String,
        /// Records absorbed into the target.
        source_ids: Vec<String>,
        /// Merged text for the target.
        content: String,
        /// Why the model (or fallback rule) merged them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Leave one record untouched.
    Keep {
        /// Record kept.
        id: String,
        /// Why the record is kept.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Replace `source_ids` with one summary text.
    Summarize {
        /// Records folded into the summary.
        source_ids: Vec<String>,
        /// Summary text replacing the sources.
        summary: String,
        /// Why the cluster is summarized.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Drop records.
    Forget {
        /// Records to remove.
        ids: Vec<String>,
        /// Why the records are no longer needed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

impl DreamConsolidationAction {
    /// The `snake_case` action name.
    #[must_use]
    pub const fn action_name(&self) -> &'static str {
        match self {
            Self::Merge { .. } => "merge",
            Self::Keep { .. } => "keep",
            Self::Summarize { .. } => "summarize",
            Self::Forget { .. } => "forget",
        }
    }

    /// Every record id the action touches, in declaration order.
    #[must_use]
    pub fn affected_ids(&self) -> Vec<&str> {
        match self {
            Self::Merge {
                target_id,
                source_ids,
                ..
            } => std::iter::once(target_id.as_str())
                .chain(source_ids.iter().map(String::as_str))
                .collect(),
            Self::Keep { id, .. } => vec![id.as_str()],
            Self::Summarize { source_ids, .. } => source_ids.iter().map(String::as_str).collect(),
            Self::Forget { ids, .. } => ids.iter().map(String::as_str).collect(),
        }
    }

    /// The reason, when one was recorded.
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Merge { reason, .. }
            | Self::Keep { reason, .. }
            | Self::Summarize { reason, .. }
            | Self::Forget { reason, .. } => reason.as_deref(),
        }
    }

    /// One human-readable line describing the action, used in outcome notes.
    #[must_use]
    pub fn note(&self) -> String {
        match self {
            Self::Merge {
                target_id,
                source_ids,
                ..
            } => format!("merge {} record(s) into `{target_id}`", source_ids.len()),
            Self::Keep { id, .. } => format!("keep `{id}`"),
            Self::Summarize { source_ids, .. } => {
                format!("summarize {} record(s)", source_ids.len())
            }
            Self::Forget { ids, .. } => format!("forget {} record(s)", ids.len()),
        }
    }
}

/// Where a [`DreamConsolidationPlan`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DreamConsolidationOrigin {
    /// The deterministic rule fallback.
    #[default]
    Rules,
    /// A model answered and its actions were parsed.
    Llm,
}

/// One consolidation decision set for a single job.
///
/// `metrics` is derived from the actions at construction time
/// ([`DreamMetrics::records_scanned`] from the candidate count, `records_merged` from
/// merge sources, `records_created` from summaries, `records_deleted` from forgotten ids,
/// and `tokens_saved` as `chars / 4` over the text the actions reclaim). `degraded` is
/// `Some(reason)` exactly when the plan came from the rule fallback or a model answer was
/// replaced by it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamConsolidationPlan {
    /// Id of the job the plan consolidates.
    pub job_id: String,
    /// Which pipeline produced the actions.
    pub origin: DreamConsolidationOrigin,
    /// The decisions, in model or grouping order.
    pub actions: Vec<DreamConsolidationAction>,
    /// Why the plan fell back to rules, when it did.
    pub degraded: Option<String>,
    /// Counters derived from the actions.
    pub metrics: DreamMetrics,
}

impl Default for DreamConsolidationPlan {
    fn default() -> Self {
        Self {
            job_id: String::new(),
            origin: DreamConsolidationOrigin::Rules,
            actions: Vec::new(),
            degraded: None,
            metrics: DreamMetrics::default(),
        }
    }
}

impl DreamConsolidationPlan {
    /// Builds a plan and derives its metrics from `actions` and `candidates`.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        origin: DreamConsolidationOrigin,
        actions: Vec<DreamConsolidationAction>,
        degraded: Option<String>,
        candidates: &[DreamCandidate],
    ) -> Self {
        let metrics = plan_metrics(&actions, candidates);
        Self {
            job_id: job_id.into(),
            origin,
            actions,
            degraded,
            metrics,
        }
    }

    /// Returns `true` when the plan came from the deterministic fallback.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.degraded.is_some()
    }

    /// Returns `true` when the plan decided nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// The notes recorded on the completion outcome: the degradation reason first (when
    /// present), then one line per action.
    #[must_use]
    pub fn notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        if let Some(reason) = &self.degraded {
            notes.push(format!("degraded: {reason}"));
        }
        notes.extend(self.actions.iter().map(DreamConsolidationAction::note));
        notes
    }
}

/// Computes the counters a plan reports.
fn plan_metrics(
    actions: &[DreamConsolidationAction],
    candidates: &[DreamCandidate],
) -> DreamMetrics {
    let contents: BTreeMap<&str, &str> = candidates
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate.content.as_str()))
        .collect();
    let mut metrics = DreamMetrics {
        records_scanned: candidates.len() as u64,
        ..DreamMetrics::default()
    };
    let mut reclaimed_chars = 0_usize;
    for action in actions {
        match action {
            DreamConsolidationAction::Merge { source_ids, .. } => {
                metrics.records_merged = metrics
                    .records_merged
                    .saturating_add(source_ids.len() as u64);
                reclaimed_chars += chars_of(source_ids, &contents);
            }
            DreamConsolidationAction::Summarize { source_ids, .. } => {
                metrics.records_created = metrics.records_created.saturating_add(1);
                reclaimed_chars += chars_of(source_ids, &contents);
            }
            DreamConsolidationAction::Forget { ids, .. } => {
                metrics.records_deleted = metrics.records_deleted.saturating_add(ids.len() as u64);
                reclaimed_chars += chars_of(ids, &contents);
            }
            DreamConsolidationAction::Keep { .. } => {}
        }
    }
    metrics.tokens_saved = (reclaimed_chars / 4) as u64;
    metrics
}

/// Sums the character length of the known ids in `ids`.
fn chars_of(ids: &[String], contents: &BTreeMap<&str, &str>) -> usize {
    ids.iter()
        .filter_map(|id| contents.get(id.as_str()))
        .map(|content| content.chars().count())
        .sum()
}
/// The JSON object shape the parser accepts from a model.
#[derive(Debug, Deserialize)]
struct RawPlan {
    /// Required: an answer without an action list is unusable.
    actions: Option<Vec<RawAction>>,
}

/// One tolerant action object; aliases cover the common provider paraphrases.
#[derive(Debug, Deserialize)]
struct RawAction {
    #[serde(alias = "op", alias = "type", alias = "operation")]
    action: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    target_id: Option<String>,
    #[serde(default, alias = "source_ids", alias = "record_ids")]
    ids: Vec<String>,
    #[serde(default, alias = "summary", alias = "text")]
    content: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

/// Parses a model answer into a plan, or `None` when the answer is unusable.
///
/// Accepts text wrapped in prose or code fences (the first `{` to the last `}` is
/// parsed), applies the documented action aliases, and drops ids the model invented. An
/// answer that parses but yields no valid action is unusable (`None`), while an explicit
/// `{"actions":[]}` is a valid empty plan.
#[must_use]
pub fn parse_consolidation_response(
    raw: &str,
    job_id: &str,
    candidates: &[DreamCandidate],
) -> Option<DreamConsolidationPlan> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    let body = raw.get(start..=end)?;
    let parsed: RawPlan = serde_json::from_str(body).ok()?;
    let raw_actions = parsed.actions?;
    let known: BTreeMap<&str, &DreamCandidate> = candidates
        .iter()
        .filter(|candidate| !candidate.id.is_empty())
        .map(|candidate| (candidate.id.as_str(), candidate))
        .collect();
    if raw_actions.is_empty() {
        return Some(DreamConsolidationPlan::new(
            job_id,
            DreamConsolidationOrigin::Llm,
            Vec::new(),
            None,
            candidates,
        ));
    }
    let mut actions = Vec::with_capacity(raw_actions.len());
    let mut dropped = 0_usize;
    for raw_action in &raw_actions {
        match normalize_action(raw_action, &known) {
            Some(action) => actions.push(action),
            None => dropped = dropped.saturating_add(1),
        }
    }
    if actions.is_empty() {
        return None;
    }
    if dropped > 0 {
        tracing::warn!(
            target: "openjiuwen_memory::dreaming::pipeline",
            job_id,
            dropped,
            "dropped unusable consolidation actions"
        );
    }
    Some(DreamConsolidationPlan::new(
        job_id,
        DreamConsolidationOrigin::Llm,
        actions,
        None,
        candidates,
    ))
}

/// Normalises one raw action, returning `None` when it names no usable candidate.
#[allow(clippy::too_many_lines)]
fn normalize_action(
    raw: &RawAction,
    known: &BTreeMap<&str, &DreamCandidate>,
) -> Option<DreamConsolidationAction> {
    let name = raw.action.trim().to_ascii_lowercase();
    let mut ids: Vec<String> = Vec::new();
    for id in raw.ids.iter().chain(raw.id.iter()) {
        if known.contains_key(id.as_str()) && !ids.iter().any(|seen| seen == id) {
            ids.push(id.clone());
        }
    }
    let reason = raw
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
        .map(str::to_owned);
    match name.as_str() {
        "merge" | "combine" => {
            // The documented answer shape names the kept record with `target_id` and only
            // the absorbed ones with `source_ids`, so the target joins the affected set here.
            if let Some(target_id) = raw.target_id.as_deref() {
                if known.contains_key(target_id) && !ids.iter().any(|seen| seen == target_id) {
                    ids.insert(0, target_id.to_owned());
                }
            }
            if ids.len() < 2 {
                return None;
            }
            let target_id = raw
                .target_id
                .as_deref()
                .filter(|id| known.contains_key(id))
                .or_else(|| raw.id.as_deref().filter(|id| known.contains_key(id)))
                .map_or_else(|| ids.first().cloned().unwrap_or_default(), str::to_owned);
            let source_ids: Vec<String> =
                ids.iter().filter(|id| **id != target_id).cloned().collect();
            if source_ids.is_empty() {
                return None;
            }
            let content = raw
                .content
                .as_deref()
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .map_or_else(
                    || {
                        longest_content(
                            std::iter::once(target_id.as_str())
                                .chain(source_ids.iter().map(String::as_str)),
                            known,
                        )
                    },
                    str::to_owned,
                );
            Some(DreamConsolidationAction::Merge {
                target_id,
                source_ids,
                content,
                reason,
            })
        }
        "keep" | "retain" => {
            let id = ids.first()?.clone();
            Some(DreamConsolidationAction::Keep { id, reason })
        }
        "summarize" | "summarise" | "summary" | "condense" => {
            if ids.is_empty() {
                return None;
            }
            let summary = raw
                .content
                .as_deref()
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .map_or_else(
                    || longest_content(ids.iter().map(String::as_str), known),
                    str::to_owned,
                );
            if summary.is_empty() {
                return None;
            }
            Some(DreamConsolidationAction::Summarize {
                source_ids: ids,
                summary,
                reason,
            })
        }
        "forget" | "delete" | "drop" | "remove" => {
            if ids.is_empty() {
                return None;
            }
            Some(DreamConsolidationAction::Forget { ids, reason })
        }
        _ => None,
    }
}

/// Returns the longest known candidate text among `ids`, ignoring empty texts.
fn longest_content<'a>(
    ids: impl Iterator<Item = &'a str>,
    known: &BTreeMap<&str, &DreamCandidate>,
) -> String {
    ids.filter_map(|id| known.get(id).map(|candidate| candidate.content.trim()))
        .filter(|content| !content.is_empty())
        .max_by_key(|content| content.chars().count())
        .unwrap_or_default()
        .to_owned()
}

/// The deterministic consolidation rule: merge near-duplicates, keep everything else.
///
/// Candidates are visited in order; each joins the first group whose representative is an
/// exact normalized-text match or has token Jaccard `>= threshold`. Members after the
/// first merge into it (keeping the longest text), and an unmatched candidate is kept.
/// Empty texts are never merged. The result is independent of wall-clock time and of any
/// model, which is what makes it the fallback.
#[must_use]
pub fn deterministic_plan(
    job_id: &str,
    candidates: &[DreamCandidate],
    threshold: f32,
) -> DreamConsolidationPlan {
    let threshold = if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        DEFAULT_FALLBACK_SIMILARITY
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let normalized = normalize_text(&candidate.content);
        let matching = if normalized.is_empty() {
            None
        } else {
            groups.iter_mut().find(|group| {
                let representative = &candidates[group[0]];
                let representative_text = normalize_text(&representative.content);
                (!representative_text.is_empty() && representative_text == normalized)
                    || jaccard(&representative.content, &candidate.content) >= threshold
            })
        };
        match matching {
            Some(group) => group.push(index),
            None => groups.push(vec![index]),
        }
    }
    let actions = groups
        .into_iter()
        .map(|group| {
            if let [only] = group.as_slice() {
                return DreamConsolidationAction::Keep {
                    id: candidates[*only].id.clone(),
                    reason: Some("fallback: no duplicate found".to_owned()),
                };
            }
            let target_id = candidates[group[0]].id.clone();
            let source_ids = group[1..]
                .iter()
                .map(|index| candidates[*index].id.clone())
                .collect();
            let content = group
                .iter()
                .map(|index| candidates[*index].content.trim())
                .filter(|content| !content.is_empty())
                .max_by_key(|content| content.chars().count())
                .unwrap_or_default()
                .to_owned();
            DreamConsolidationAction::Merge {
                target_id,
                source_ids,
                content,
                reason: Some(format!("fallback: content similarity >= {threshold}")),
            }
        })
        .collect();
    DreamConsolidationPlan::new(
        job_id,
        DreamConsolidationOrigin::Rules,
        actions,
        None,
        candidates,
    )
}
