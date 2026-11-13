//! Deterministic member-target selection.
//!
//! Python's `member_selector.py` scores pre-computed role/mechanism attribution
//! reports; it never calls a model. The Rust port keeps exactly that arithmetic:
//!
//! ```text
//! selection_score = severity_weight * 10
//!                 + confidence * 5
//!                 + min(eligible_actions, 5) / 5
//! ```
//!
//! and the same total order — score descending, severity weight descending, role
//! ascending — so a beam of `max_roles_per_run` roles is reproducible from the
//! same attribution input.
//!
//! Attribution itself (Python's `role_attributor`/`mechanism_attributor`, both
//! LLM prompts over a harness transcript) is the caller's job. This module
//! exposes the data types for that hand-off ([`AttributionInput`] /
//! [`AttributionResult`]) plus one deterministic default, [`SynthesizedAttributor`],
//! which reads the `role` key of each issue's metadata and is documented as the
//! offline stand-in for Python's synthesized single-harness attribution.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
/// Severity of the issues attributed to a member.
///
/// Weights match Python's `_SEVERITY_WEIGHTS`: critical `4`, high `3`, medium
/// `2`, low `1`. [`Severity::default`] is [`Severity::Low`], which is also the
/// weight Python uses for an unrecognised severity string.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Blocks the member's objective outright.
    Critical,
    /// Degrades the member's objective on most tasks.
    High,
    /// Degrades some tasks.
    Medium,
    /// Cosmetic or rare failure.
    #[default]
    Low,
}
impl Severity {
    /// Returns the numeric weight used by the selection formula.
    #[must_use]
    pub fn weight(self) -> f64 {
        match self {
            Self::Critical => 4.0,
            Self::High => 3.0,
            Self::Medium => 2.0,
            Self::Low => 1.0,
        }
    }
    /// Returns the `snake_case` name used in configuration and artifacts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }
}
impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
/// One member role proposed for optimisation, with its attribution evidence.
///
/// Python's `MemberOptimizationTarget`/`RoleScore` pair is flattened into one
/// type: the role is already scored, the issues are attributed, and the eligible
/// actions are the harness surfaces a planner may pick from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberTarget {
    /// The member role (for example `implement` or `verify`).
    pub role: String,
    /// Highest severity among the attributed issues.
    pub severity: Severity,
    /// Average attribution confidence of the attributed issues.
    pub confidence: f64,
    /// Action-definition names this role may use; empty means "nothing to do".
    pub eligible_actions: Vec<String>,
    /// Ids of the issues attributed to this role, in attribution order.
    pub issue_ids: Vec<String>,
    /// Harness member name, when the caller resolved one.
    pub member_name: Option<String>,
    /// Caller annotations, never interpreted here.
    pub metadata: BTreeMap<String, Value>,
}
/// A [`MemberTarget`] together with the score that ranked it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SelectedTarget {
    /// The target itself, unchanged.
    pub target: MemberTarget,
    /// Its [`selection_score`].
    pub score: f64,
}
/// Scores one target with Python's `severity_weight*10 + confidence*5 + issue_score`.
///
/// The issue term is `min(eligible_actions, 5) / 5`, so a role with five or more
/// eligible actions is indistinguishable from one with exactly five.
#[must_use]
#[allow(clippy::cast_precision_loss)] // the action count is tiny and exact in f64
pub fn selection_score(target: &MemberTarget) -> f64 {
    let issue_score = target.eligible_actions.len().min(5) as f64 / 5.0;
    target.severity.weight() * 10.0 + target.confidence * 5.0 + issue_score
}
/// Returns every selectable target in Python's total order.
///
/// Targets below `min_confidence` or without eligible actions are dropped first,
/// exactly like Python's `eligible` filter. Ordering is score descending,
/// severity weight descending, role ascending, which makes it total (roles are
/// unique per attribution) and therefore deterministic.
#[must_use]
pub fn rank_targets(targets: &[MemberTarget], min_confidence: f64) -> Vec<SelectedTarget> {
    let mut ranked: Vec<SelectedTarget> = targets
        .iter()
        .filter(|target| target.confidence >= min_confidence)
        .filter(|target| !target.eligible_actions.is_empty())
        .map(|target| SelectedTarget {
            target: target.clone(),
            score: selection_score(target),
        })
        .collect();
    ranked.sort_by(compare_targets);
    ranked
}
/// Selects the `beam_width` highest-ranked targets.
///
/// This is [`rank_targets`] followed by a truncation; callers that want to
/// inspect or report the tail (Python's `unselected_attributions`) call
/// [`rank_targets`] directly.
#[must_use]
pub fn select_targets(
    targets: &[MemberTarget],
    beam_width: usize,
    min_confidence: f64,
) -> Vec<SelectedTarget> {
    let mut ranked = rank_targets(targets, min_confidence);
    ranked.truncate(beam_width);
    ranked
}
/// Python's sort key: `(-score, -max_severity, role)`.
fn compare_targets(left: &SelectedTarget, right: &SelectedTarget) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| {
            right
                .target
                .severity
                .weight()
                .total_cmp(&left.target.severity.weight())
        })
        .then_with(|| left.target.role.cmp(&right.target.role))
}
/// One failing case handed to attribution.
///
/// Python keeps the role attribution report and the mechanism attribution report
/// as separate documents produced by two LLM prompts. The Rust seam takes the
/// flattened observation instead: the issue id, its severity and confidence, and
/// the metadata the attributor wrote (at minimum `role`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct IssueObservation {
    /// Stable issue id.
    pub issue_id: String,
    /// Issue severity, already classified by the attributor.
    pub severity: Severity,
    /// Attribution confidence in `0.0..=1.0`.
    pub confidence: f64,
    /// Attribution evidence; [`SynthesizedAttributor`] reads `role` here.
    pub metadata: BTreeMap<String, Value>,
}
/// Everything attribution needs to produce [`MemberTarget`]s.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AttributionInput {
    /// The failing cases to attribute.
    pub issues: Vec<IssueObservation>,
    /// Action-definition names per role; the caller's action-group policy.
    pub eligible_actions: BTreeMap<String, Vec<String>>,
}
/// The result of attribution: scored targets plus the issues left unattributed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AttributionResult {
    /// One target per attributed role, sorted by role ascending.
    pub targets: Vec<MemberTarget>,
    /// Issues that carried no role, sorted by issue id ascending.
    pub unassigned: Vec<UnassignedIssue>,
}
/// An issue the attributor could not assign to a member role.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct UnassignedIssue {
    /// The issue id.
    pub issue_id: String,
    /// Why it was left unassigned.
    pub reason: String,
}
/// The metadata key read by [`SynthesizedAttributor`].
pub const ROLE_METADATA_KEY: &str = "role";
/// The metadata key read by [`SynthesizedAttributor`] for the member name.
pub const MEMBER_METADATA_KEY: &str = "member_name";
/// Deterministic offline stand-in for Python's LLM role attribution.
///
/// Every [`IssueObservation`] whose metadata carries a non-empty `role` string is
/// attributed to that role; the resulting [`MemberTarget`] takes the maximum
/// severity, the average confidence, the union of the issue ids, and the eligible
/// actions the caller supplied for that role. `member_name` comes from the first
/// issue that carries one.
///
/// This is deliberately not an LLM: it exists so the selection, planning,
/// execution, and verification loop is testable end to end. A production caller
/// replaces it with its own attribution (Python's `role_attributor` and
/// `mechanism_attributor` prompts) and hands the result to [`select_targets`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynthesizedAttributor;
impl SynthesizedAttributor {
    /// Creates the attributor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
    /// Attributes `input`'s issues to roles.
    ///
    /// The output is deterministic: targets are emitted by role ascending and
    /// unassigned issues by issue id ascending; issue ids keep their input order
    /// inside a target.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // issue counts are tiny and exact in f64
    pub fn attribute(&self, input: &AttributionInput) -> AttributionResult {
        let mut grouped: BTreeMap<String, Vec<&IssueObservation>> = BTreeMap::new();
        let mut unassigned = Vec::new();
        for issue in &input.issues {
            match issue
                .metadata
                .get(ROLE_METADATA_KEY)
                .and_then(Value::as_str)
            {
                Some(role) if !role.trim().is_empty() => {
                    grouped
                        .entry(role.trim().to_owned())
                        .or_default()
                        .push(issue);
                }
                _ => unassigned.push(UnassignedIssue {
                    issue_id: issue.issue_id.clone(),
                    reason: format!("issue metadata carries no `{ROLE_METADATA_KEY}`"),
                }),
            }
        }
        unassigned.sort_by(|left, right| left.issue_id.cmp(&right.issue_id));
        let targets = grouped
            .into_iter()
            .map(|(role, issues)| synthesize_target(&role, &issues, input))
            .collect();
        AttributionResult {
            targets,
            unassigned,
        }
    }
}
/// Builds one [`MemberTarget`] from the issues attributed to `role`.
#[allow(clippy::cast_precision_loss)] // issue counts are tiny and exact in f64
fn synthesize_target(
    role: &str,
    issues: &[&IssueObservation],
    input: &AttributionInput,
) -> MemberTarget {
    let severity = issues
        .iter()
        .map(|issue| issue.severity)
        .max_by(|left, right| left.weight().total_cmp(&right.weight()))
        .unwrap_or_default();
    let confidence = if issues.is_empty() {
        0.0
    } else {
        issues.iter().map(|issue| issue.confidence).sum::<f64>() / issues.len() as f64
    };
    let member_name = issues
        .iter()
        .find_map(|issue| issue.metadata.get(MEMBER_METADATA_KEY))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned);
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "attribution".to_owned(),
        Value::from("synthesized_metadata_role"),
    );
    metadata.insert("issue_count".to_owned(), Value::from(issues.len()));
    MemberTarget {
        role: role.to_owned(),
        severity,
        confidence,
        eligible_actions: input
            .eligible_actions
            .get(role)
            .cloned()
            .unwrap_or_default(),
        issue_ids: issues.iter().map(|issue| issue.issue_id.clone()).collect(),
        member_name,
        metadata,
    }
}
