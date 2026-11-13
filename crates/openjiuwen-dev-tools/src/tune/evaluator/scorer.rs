//! The scorer seam: turn `(case, answer)` into a graded score and a reason.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::tune::base::Case;

use super::runner::render_value;

/// Scores one answer against its case.
pub trait CaseScorer: Send + Sync {
    /// Returns the score in `0.0..=1.0` and the reason for it.
    fn score(&self, case: &Case, answer: &BTreeMap<String, Value>) -> (f64, String);
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
