//! The scorer seam: turn `(case, answer)` into a graded score and a reason.
//!
//! A scorer may look at the [`ToolTrace`] behind an answer through
//! [`CaseScorer::score_with_trace`]; [`ToolUsageScorer`] uses it to require that
//! the tools a case declares were actually used.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

use crate::tune::base::Case;
use crate::tune::tool::ToolTrace;

use super::runner::render_value;

/// Scores one answer against its case.
pub trait CaseScorer: Send + Sync {
    /// Returns the score in `0.0..=1.0` and the reason for it.
    fn score(&self, case: &Case, answer: &BTreeMap<String, Value>) -> (f64, String);

    /// Scores `answer` given the tool calls that produced it. The default
    /// ignores the trace.
    fn score_with_trace(
        &self,
        case: &Case,
        answer: &BTreeMap<String, Value>,
        trace: &ToolTrace,
    ) -> (f64, String) {
        let _ = trace;
        self.score(case, answer)
    }
}

/// The deterministic default [`CaseScorer`]: compare answer and label by key.
///
/// See the module documentation for the scoring bands. The reason names the
/// first label key that is not an exact match, or states that every key matched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LabelScorer;

impl LabelScorer {
    /// Creates the scorer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl CaseScorer for LabelScorer {
    fn score(&self, case: &Case, answer: &BTreeMap<String, Value>) -> (f64, String) {
        if case.label.is_empty() {
            return (0.0, "case label has no keys to score".to_owned());
        }
        let mut total = 0.0;
        let mut first_mismatch: Option<String> = None;
        for (key, expected) in &case.label {
            let expected_text = render_value(expected);
            let actual_text = answer_text(answer, key);
            let (score, reason) = match &actual_text {
                Some(actual) => {
                    let score = score_pair(&expected_text, actual);
                    let reason = if score >= 1.0 {
                        String::new()
                    } else {
                        format!("label key `{key}` scored {score:.2}: expected {expected_text:?}, got {actual:?}")
                    };
                    (score, reason)
                }
                None => (0.0, format!("label key `{key}` is missing from the answer")),
            };
            total += score;
            if score < 1.0 && first_mismatch.is_none() {
                first_mismatch = Some(reason);
            }
        }
        let mean = total / case.label.len() as f64;
        let reason = first_mismatch
            .unwrap_or_else(|| format!("all {} label key(s) matched exactly", case.label.len()));
        (mean.clamp(0.0, 1.0), reason)
    }
}

/// The text a label key is compared with: the answer entry, or the sole answer
/// entry when the key is absent.
fn answer_text(answer: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    if let Some(value) = answer.get(key) {
        return Some(render_value(value));
    }
    if answer.len() == 1 {
        return answer.values().next().map(render_value);
    }
    None
}

/// Scores one expected/actual string pair with the bands of [`LabelScorer`].
pub(super) fn score_pair(expected: &str, actual: &str) -> f64 {
    let expected = expected.trim();
    let actual = actual.trim();
    if expected == actual {
        return 1.0;
    }
    let expected_lower = expected.to_lowercase();
    let actual_lower = actual.to_lowercase();
    if expected_lower == actual_lower {
        return 0.9;
    }
    if !expected_lower.is_empty()
        && !actual_lower.is_empty()
        && (actual_lower.contains(&expected_lower) || expected_lower.contains(&actual_lower))
    {
        return 0.75;
    }
    let expected_tokens = tokens(expected);
    let actual_tokens = tokens(actual);
    let union = expected_tokens.union(&actual_tokens).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = expected_tokens.intersection(&actual_tokens).count();
    let jaccard = intersection as f64 / union as f64;
    if jaccard >= 0.5 {
        return 0.3 + 0.3 * ((jaccard - 0.5) / 0.5);
    }
    0.0
}

/// Splits text into lower-cased alphanumeric tokens, dropping punctuation.
pub(super) fn tokens(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// A scorer that multiplies an inner score by the fraction of required tools
/// that were used.
///
/// The inner scorer (a [`LabelScorer`] unless
/// [`with_inner`](Self::with_inner) replaces it) supplies the base score, so a
/// `ToolUsageScorer` composes with any other scorer. The base is multiplied by
/// [`ToolTrace::coverage`] of the required names: an empty requirement list
/// leaves the inner score unchanged, and a run that used none of the required
/// tools scores `0.0`.
pub struct ToolUsageScorer {
    /// The scorer producing the base score.
    inner: Box<dyn CaseScorer>,
    /// The tool names a run must use, in requirement order.
    required: Vec<String>,
}

impl ToolUsageScorer {
    /// Creates a scorer requiring `required`, with [`LabelScorer`] as the inner
    /// scorer.
    ///
    /// An empty requirement list is allowed and scores the inner result
    /// unchanged.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when a required name is blank.
    pub fn new(required: Vec<String>) -> Result<Self> {
        for name in &required {
            if name.trim().is_empty() {
                return Err(Error::validation(
                    "ToolUsageScorer.required entries must be non-blank tool names",
                ));
            }
        }
        Ok(Self {
            inner: Box::new(LabelScorer),
            required,
        })
    }

    /// Creates a scorer requiring every tool the case declares
    /// ([`Case::tool_names`]).
    ///
    /// A case without declared tools yields an empty requirement list and
    /// therefore scores the inner result unchanged.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when a declared name is blank.
    pub fn from_case(case: &Case) -> Result<Self> {
        Self::new(case.tool_names().into_iter().map(str::to_owned).collect())
    }

    /// Replaces the scorer producing the base score.
    #[must_use]
    pub fn with_inner(mut self, inner: Box<dyn CaseScorer>) -> Self {
        self.inner = inner;
        self
    }

    /// The required tool names, in requirement order.
    pub fn required(&self) -> &[String] {
        &self.required
    }

    /// The scorer producing the base score.
    pub fn inner(&self) -> &dyn CaseScorer {
        self.inner.as_ref()
    }

    /// The required names the trace never used, in `required` order.
    pub fn missing(&self, trace: &ToolTrace) -> Vec<String> {
        trace.missing(&self.required_refs())
    }

    /// Scores as if no tool was used, i.e. against an empty trace.
    pub fn score(&self, case: &Case, answer: &BTreeMap<String, Value>) -> (f64, String) {
        self.score_with_trace(case, answer, &ToolTrace::default())
    }

    /// Scores `answer` against `case` using the tool calls in `trace`.
    ///
    /// The base score comes from the inner scorer's
    /// [`score_with_trace`](CaseScorer::score_with_trace), so nesting works; it
    /// is multiplied by the required-name coverage and clamped into
    /// `0.0..=1.0` (a non-finite base becomes `0.0`). The reason keeps the inner
    /// reason when every required tool was used and otherwise names the missing
    /// tools in requirement order.
    pub fn score_with_trace(
        &self,
        case: &Case,
        answer: &BTreeMap<String, Value>,
        trace: &ToolTrace,
    ) -> (f64, String) {
        let (base, inner_reason) = self.inner.score_with_trace(case, answer, trace);
        let factor = trace.coverage(&self.required_refs());
        let total = if base.is_finite() {
            (base * factor).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let missing = self.missing(trace);
        let reason = if missing.is_empty() {
            inner_reason
        } else {
            let names = missing.join(", ");
            if inner_reason.is_empty() {
                format!("missing required tool(s): {names}")
            } else {
                format!("missing required tool(s): {names}; {inner_reason}")
            }
        };
        (total, reason)
    }

    /// The required names as `&str`s, the shape [`ToolTrace`] takes.
    fn required_refs(&self) -> Vec<&str> {
        self.required.iter().map(String::as_str).collect()
    }
}

impl fmt::Debug for ToolUsageScorer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolUsageScorer")
            .field("required", &self.required)
            .field("inner", &"custom scorer")
            .finish()
    }
}

impl CaseScorer for ToolUsageScorer {
    fn score(&self, case: &Case, answer: &BTreeMap<String, Value>) -> (f64, String) {
        ToolUsageScorer::score(self, case, answer)
    }

    fn score_with_trace(
        &self,
        case: &Case,
        answer: &BTreeMap<String, Value>,
        trace: &ToolTrace,
    ) -> (f64, String) {
        ToolUsageScorer::score_with_trace(self, case, answer, trace)
    }
}
