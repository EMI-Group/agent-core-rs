//! Score truncation and the bounded text/item helpers the renderers use.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.runtime.truncation` (v0.1.20) exactly
//! — [`truncate_sorted_by_score`]
//! and its decision record — and adds three Rust-only bounded helpers the disclosure
//! renderers need ([`truncate_chars`], [`truncate_chars_with_ellipsis`], [`take_budget`]).
//!
//! # Decision order
//!
//! The three cut rules are applied in the reference's order — absolute `min_score`, then
//! `relative_min_score`, then the `autocut` gap — and each one may only *lower* the cutoff,
//! so the strictest rule wins and the reported `reason` names it:
//!
//! | Rule | Reason | Cut before |
//! | --- | --- | --- |
//! | `min_score` | `"min_score"` | the first item scoring below the absolute threshold |
//! | `relative_min_score` | `"relative_min_score"` | the first item scoring below `best × relative` (only when `best > 0`) |
//! | `autocut` | `"autocut"` | the item after the `autocut_jumps`-th qualifying score drop |
//!
//! Input is assumed sorted by descending score (the reference never reorders it either).
//! An empty input keeps nothing, reports `initial_count = 0`, `kept_count = 0` and no
//! cutoff. Nothing here reads the clock or allocates a hash map.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How a score-sorted result list is truncated.
///
/// Mirrors Python `ScoreTruncationConfig`. All thresholds are optional: `None` disables the
/// rule, and `autocut_jumps == 0` disables the autocut gap.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ScoreTruncationConfig {
    /// Drop everything scoring below this absolute threshold.
    pub min_score: Option<f64>,
    /// Drop everything scoring below `best_score ×` this fraction (only when `best > 0`).
    pub relative_min_score: Option<f64>,
    /// Cut after the `autocut_jumps`-th qualifying score drop; `0` disables autocut.
    pub autocut_jumps: usize,
    /// The minimum relative drop (`drop / |previous|`) autocut counts as a jump.
    pub autocut_min_relative_drop: f64,
}

/// Why and where a score-sorted list was truncated.
///
/// Mirrors Python `ScoreTruncationDecision`, so a trace or a log can explain a short list.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ScoreTruncationDecision {
    /// How many items the decision was made on.
    pub initial_count: usize,
    /// How many items were kept.
    pub kept_count: usize,
    /// The index the list was cut at, when it was cut at all.
    pub cutoff_index: Option<usize>,
    /// The rule that produced the cutoff (`"min_score"`, `"relative_min_score"`,
    /// `"autocut"`), or `""` when nothing was cut.
    pub reason: String,
    /// The best (first) score, when the input was not empty.
    pub best_score: Option<f64>,
    /// The absolute threshold the decision used.
    pub min_score: Option<f64>,
    /// The relative threshold the decision used.
    pub relative_min_score: Option<f64>,
    /// The autocut jump count the decision used.
    pub autocut_jumps: usize,
    /// The autocut relative-drop threshold the decision used.
    pub autocut_min_relative_drop: f64,
}

/// The rule name reported when the absolute threshold cut the list.
pub const REASON_MIN_SCORE: &str = "min_score";
/// The rule name reported when the relative threshold cut the list.
pub const REASON_RELATIVE_MIN_SCORE: &str = "relative_min_score";
/// The rule name reported when the score-gap rule cut the list.
pub const REASON_AUTOCUT: &str = "autocut";

/// Truncates a score-sorted list of `items` according to `config`.
///
/// `score` reads an item's score. The list is never reordered, cloned items keep their
/// order, and an empty input yields `(vec![], decision with zero counts)`.
///
/// # Example
///
/// ```
/// use openjiuwen_symphony::retrieval::search::truncation::{
///     truncate_sorted_by_score, ScoreTruncationConfig,
/// };
///
/// let items = [0.9, 0.5, 0.1];
/// let config = ScoreTruncationConfig {
///     min_score: Some(0.2),
///     ..ScoreTruncationConfig::default()
/// };
/// let (kept, decision) = truncate_sorted_by_score(&items, |score| *score, &config);
/// assert_eq!(kept, vec![0.9, 0.5]);
/// assert_eq!(decision.reason, "min_score");
/// ```
#[must_use]
pub fn truncate_sorted_by_score<T: Clone>(
    items: &[T],
    score: impl Fn(&T) -> f64,
    config: &ScoreTruncationConfig,
) -> (Vec<T>, ScoreTruncationDecision) {
    let Some(first) = items.first() else {
        return (Vec::new(), ScoreTruncationDecision::default());
    };
    let best_score = score(first);
    let mut cutoff = items.len();
    let mut reason = String::new();

    if let Some(min_score) = config.min_score {
        let candidate = first_score_below(items, &score, min_score);
        if candidate < cutoff {
            cutoff = candidate;
            REASON_MIN_SCORE.clone_into(&mut reason);
        }
    }

    if let Some(relative) = config.relative_min_score {
        if best_score > 0.0 {
            let threshold = best_score * relative.max(0.0);
            let candidate = first_score_below(items, &score, threshold);
            if candidate < cutoff {
                cutoff = candidate;
                REASON_RELATIVE_MIN_SCORE.clone_into(&mut reason);
            }
        }
    }

    if config.autocut_jumps > 0 {
        let candidate = autocut_cutoff(
            items,
            &score,
            config.autocut_jumps.max(1),
            config.autocut_min_relative_drop.max(0.0),
        );
        if candidate < cutoff {
            cutoff = candidate;
            REASON_AUTOCUT.clone_into(&mut reason);
        }
    }

    let kept = items[..cutoff].to_vec();
    let decision = ScoreTruncationDecision {
        initial_count: items.len(),
        kept_count: kept.len(),
        cutoff_index: if cutoff < items.len() {
            Some(cutoff)
        } else {
            None
        },
        reason,
        best_score: Some(best_score),
        min_score: config.min_score,
        relative_min_score: config.relative_min_score,
        autocut_jumps: config.autocut_jumps,
        autocut_min_relative_drop: config.autocut_min_relative_drop.max(0.0),
    };
    (kept, decision)
}

/// The index of the first item scoring strictly below `threshold`, or the list length.
fn first_score_below<T>(items: &[T], score: &impl Fn(&T) -> f64, threshold: f64) -> usize {
    items
        .iter()
        .position(|item| score(item) < threshold)
        .unwrap_or(items.len())
}

/// The index of the first item after the `jumps`-th qualifying drop, or the list length.
fn autocut_cutoff<T>(
    items: &[T],
    score: &impl Fn(&T) -> f64,
    jumps: usize,
    min_relative_drop: f64,
) -> usize {
    if items.len() <= 1 {
        return items.len();
    }
    let mut observed = 0_usize;
    for index in 1..items.len() {
        let previous = score(&items[index - 1]);
        let current = score(&items[index]);
        let absolute_drop = previous - current;
        if absolute_drop <= 0.0 {
            continue;
        }
        let relative_drop = absolute_drop / previous.abs().max(1e-9);
        if relative_drop < min_relative_drop {
            continue;
        }
        observed += 1;
        if observed >= jumps {
            return index;
        }
    }
    items.len()
}

/// Cuts `text` to at most `max_chars` characters, never splitting a multi-byte character.
///
/// `max_chars == 0` means "unlimited": the text comes back unchanged with `false`. The
/// returned flag reports whether anything was cut.
///
/// # Example
///
/// ```
/// use openjiuwen_symphony::retrieval::search::truncation::truncate_chars;
///
/// assert_eq!(truncate_chars("能力树", 2), ("能力".to_owned(), true));
/// assert_eq!(truncate_chars("abc", 0), ("abc".to_owned(), false));
/// ```
#[must_use]
pub fn truncate_chars(text: &str, max_chars: usize) -> (String, bool) {
    if max_chars == 0 || text.chars().count() <= max_chars {
        return (text.to_owned(), false);
    }
    // The byte offset of the `max_chars + 1`-th character is the cut point, and it is
    // always a character boundary.
    let cut = text
        .char_indices()
        .nth(max_chars)
        .map_or(text.len(), |(index, _)| index);
    (text[..cut].to_owned(), true)
}

/// Cuts `text` to at most `max_chars` characters, ending a cut text with `"..."`.
///
/// The ellipsis counts **inside** the budget: a cut result is exactly `max_chars`
/// characters long. When the budget cannot hold both text and the ellipsis
/// (`max_chars <= 3`) the text is cut plainly and still reported as truncated.
/// `max_chars == 0` means "unlimited".
#[must_use]
pub fn truncate_chars_with_ellipsis(text: &str, max_chars: usize) -> (String, bool) {
    if max_chars == 0 {
        return (text.to_owned(), false);
    }
    if text.chars().count() <= max_chars {
        return (text.to_owned(), false);
    }
    if max_chars <= 3 {
        let (cut, _) = truncate_chars(text, max_chars);
        return (cut, true);
    }
    let (head, _) = truncate_chars(text, max_chars - 3);
    let head = head.trim_end();
    (format!("{head}..."), true)
}

/// Takes at most `max_items` items from `items`.
///
/// `max_items == 0` means "unlimited": the items come back unchanged with `false`. The
/// returned flag reports whether anything was dropped.
///
/// # Example
///
/// ```
/// use openjiuwen_symphony::retrieval::search::truncation::take_budget;
///
/// assert_eq!(take_budget(vec![1, 2, 3], 2), (vec![1, 2], true));
/// assert_eq!(take_budget(vec![1], 0), (vec![1], false));
/// ```
#[must_use]
pub fn take_budget<T>(items: Vec<T>, max_items: usize) -> (Vec<T>, bool) {
    if max_items == 0 || items.len() <= max_items {
        return (items, false);
    }
    let mut items = items;
    items.truncate(max_items);
    (items, true)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn scores(values: &[f64]) -> Vec<f64> {
        values.to_vec()
    }

    fn decision_of(
        values: &[f64],
        config: &ScoreTruncationConfig,
    ) -> (Vec<f64>, ScoreTruncationDecision) {
        truncate_sorted_by_score(&scores(values), |score| *score, config)
    }

    #[test]
    fn an_empty_input_keeps_nothing_and_reports_no_cutoff() {
        let (kept, decision) = decision_of(&[], &ScoreTruncationConfig::default());
        assert_eq!(kept, Vec::<f64>::new());
        assert_eq!(decision.initial_count, 0);
        assert_eq!(decision.kept_count, 0);
        assert_eq!(decision.cutoff_index, None);
        assert_eq!(decision.reason, "");
        assert_eq!(decision.best_score, None);
    }

    #[test]
    fn a_config_without_rules_keeps_everything() {
        let (kept, decision) = decision_of(&[0.9, 0.4, 0.1], &ScoreTruncationConfig::default());
        assert_eq!(kept, vec![0.9, 0.4, 0.1]);
        assert_eq!(decision.kept_count, 3);
        assert_eq!(decision.cutoff_index, None);
        assert_eq!(decision.reason, "");
        assert_eq!(decision.best_score, Some(0.9));
    }

    #[test]
    fn min_score_cuts_at_the_first_item_below_the_threshold() {
        let config = ScoreTruncationConfig {
            min_score: Some(0.35),
            ..ScoreTruncationConfig::default()
        };
        let (kept, decision) = decision_of(&[0.9, 0.5, 0.3, 0.2], &config);
        assert_eq!(kept, vec![0.9, 0.5]);
        assert_eq!(decision.reason, REASON_MIN_SCORE);
        assert_eq!(decision.cutoff_index, Some(2));
        assert_eq!(decision.kept_count, 2);
        assert_eq!(decision.initial_count, 4);
        assert_eq!(decision.min_score, Some(0.35));
    }

    #[test]
    fn relative_min_score_uses_the_best_score_and_skips_a_non_positive_best() {
        let config = ScoreTruncationConfig {
            relative_min_score: Some(0.5),
            ..ScoreTruncationConfig::default()
        };
        let (kept, decision) = decision_of(&[1.0, 0.5, 0.4], &config);
        assert_eq!(kept, vec![1.0, 0.5]);
        assert_eq!(decision.reason, REASON_RELATIVE_MIN_SCORE);
        assert_eq!(decision.cutoff_index, Some(2));

        // 0.4 (the item at index 2) is exactly the threshold, so nothing is cut.
        let (kept, decision) = decision_of(&[1.0, 0.5, 0.5], &config);
        assert_eq!(kept.len(), 3);
        assert_eq!(decision.reason, "");

        // A non-positive best score disables the relative rule.
        let (kept, decision) = decision_of(&[0.0, -0.1], &config);
        assert_eq!(kept, vec![0.0, -0.1]);
        assert_eq!(decision.reason, "");
    }

    #[test]
    fn autocut_cuts_after_the_configured_number_of_drops() {
        let config = ScoreTruncationConfig {
            autocut_jumps: 1,
            autocut_min_relative_drop: 0.5,
            ..ScoreTruncationConfig::default()
        };
        // 1.0 -> 0.5 is a 50% drop, which counts as one jump.
        let (kept, decision) = decision_of(&[1.0, 0.5, 0.49, 0.48], &config);
        assert_eq!(kept, vec![1.0]);
        assert_eq!(decision.reason, REASON_AUTOCUT);
        assert_eq!(decision.cutoff_index, Some(1));

        // With two jumps the second qualifying drop decides.
        let config = ScoreTruncationConfig {
            autocut_jumps: 2,
            autocut_min_relative_drop: 0.5,
            ..ScoreTruncationConfig::default()
        };
        let (kept, decision) = decision_of(&[1.0, 0.5, 0.25], &config);
        assert_eq!(kept, vec![1.0, 0.5]);
        assert_eq!(decision.cutoff_index, Some(2));

        // Drops smaller than the relative threshold are ignored.
        let config = ScoreTruncationConfig {
            autocut_jumps: 1,
            autocut_min_relative_drop: 0.6,
            ..ScoreTruncationConfig::default()
        };
        let (kept, decision) = decision_of(&[1.0, 0.5, 0.4], &config);
        assert_eq!(kept.len(), 3);
        assert_eq!(decision.reason, "");
    }

    #[test]
    fn the_strictest_rule_decides_the_cutoff_and_the_first_rule_to_reach_it_names_it() {
        // min_score (0.1) cuts at 3, relative (0.8) at 1, and autocut can only confirm that
        // cutoff — the reference keeps the first rule's name when nothing moves the cutoff.
        let config = ScoreTruncationConfig {
            min_score: Some(0.1),
            relative_min_score: Some(0.8),
            autocut_jumps: 1,
            autocut_min_relative_drop: 0.1,
        };
        let (kept, decision) = decision_of(&[1.0, 0.7, 0.5, 0.2], &config);
        assert_eq!(kept, vec![1.0]);
        assert_eq!(decision.reason, REASON_RELATIVE_MIN_SCORE);
        assert_eq!(decision.cutoff_index, Some(1));
        assert_eq!(decision.autocut_jumps, 1);
        assert_eq!(decision.autocut_min_relative_drop, 0.1);
        assert_eq!(decision.relative_min_score, Some(0.8));

        // A looser relative threshold lets autocut become the strictest rule.
        let config = ScoreTruncationConfig {
            min_score: Some(0.1),
            relative_min_score: Some(0.6),
            autocut_jumps: 1,
            autocut_min_relative_drop: 0.1,
        };
        let (kept, decision) = decision_of(&[1.0, 0.7, 0.5, 0.2], &config);
        assert_eq!(kept, vec![1.0]);
        assert_eq!(decision.reason, REASON_AUTOCUT);
        assert_eq!(decision.cutoff_index, Some(1));
    }

    #[test]
    fn a_single_item_list_is_never_cut_by_autocut() {
        let config = ScoreTruncationConfig {
            autocut_jumps: 3,
            autocut_min_relative_drop: 0.0,
            ..ScoreTruncationConfig::default()
        };
        let (kept, decision) = decision_of(&[0.5], &config);
        assert_eq!(kept, vec![0.5]);
        assert_eq!(decision.cutoff_index, None);
    }

    #[test]
    fn char_truncation_is_char_boundary_safe() {
        assert_eq!(truncate_chars("能力树索引", 3), ("能力树".to_owned(), true));
        assert_eq!(
            truncate_chars("能力树索引", 5),
            ("能力树索引".to_owned(), false)
        );
        assert_eq!(
            truncate_chars("能力树索引", 9),
            ("能力树索引".to_owned(), false)
        );
        assert_eq!(truncate_chars("abc", 0), ("abc".to_owned(), false));
        assert_eq!(truncate_chars("", 3), (String::new(), false));
        assert_eq!(truncate_chars("能力树", 1), ("能".to_owned(), true));
    }

    #[test]
    fn ellipsis_counts_inside_the_budget() {
        let (text, truncated) = truncate_chars_with_ellipsis("abcdefghij", 6);
        assert_eq!(text, "abc...");
        assert_eq!(text.chars().count(), 6);
        assert!(truncated);

        let (text, truncated) = truncate_chars_with_ellipsis("ab", 6);
        assert_eq!(text, "ab");
        assert!(!truncated);

        let (text, truncated) = truncate_chars_with_ellipsis("能力树索引构建", 5);
        assert_eq!(text, "能力...");
        assert!(truncated);

        let (text, truncated) = truncate_chars_with_ellipsis("abcdef", 3);
        assert_eq!(text, "abc");
        assert!(truncated);

        assert_eq!(
            truncate_chars_with_ellipsis("abcdef", 0),
            ("abcdef".to_owned(), false)
        );
    }

    #[test]
    fn take_budget_caps_items_and_treats_zero_as_unlimited() {
        assert_eq!(take_budget(vec![1, 2, 3], 0), (vec![1, 2, 3], false));
        assert_eq!(take_budget(vec![1, 2, 3], 3), (vec![1, 2, 3], false));
        assert_eq!(take_budget(vec![1, 2, 3], 2), (vec![1, 2], true));
        assert_eq!(take_budget(Vec::<u8>::new(), 2), (vec![], false));
    }

    #[test]
    fn the_decision_serializes_with_snake_case_keys() {
        let config = ScoreTruncationConfig {
            min_score: Some(0.5),
            relative_min_score: None,
            autocut_jumps: 2,
            autocut_min_relative_drop: 0.25,
        };
        let (_, decision) = decision_of(&[1.0, 0.4], &config);
        let json = serde_json::to_value(&decision).unwrap();
        assert_eq!(json["reason"], "min_score");
        assert_eq!(json["kept_count"], 1);
        assert_eq!(json["cutoff_index"], 1);
        assert_eq!(json["autocut_min_relative_drop"], 0.25);
        let parsed: ScoreTruncationDecision = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, decision);
    }
}
