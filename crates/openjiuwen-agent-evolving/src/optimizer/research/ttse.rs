//! Test-time skill tips (TTSE): Python's tip-induction optimizer.
//!
//! Ports the deterministic core of the Python `ttse` induction step: selected
//! signals are clustered by Jaccard similarity of their normalised token sets,
//! each cluster contributes one representative tip, and the tip bank is deduped
//! and bounded.
//!
//! The `tips` parameter holds a JSON array of tip strings; the bank is capped at
//! [`TtseStrategy::capacity`] tips (default 12), existing tips keep their order,
//! and new tips are appended in cluster order (first appearance in the signal
//! list), so two runs over the same signals produce identical output.

use crate::signal::{normalize_excerpt, EvolutionSignal, EvolutionTarget, MAX_EXCERPT_CHARS};

use super::base::{
    encode_text_list, feedback_texts, jaccard, near_duplicate, normalized_tokens, parse_text_list,
    render_signals, shorten, TextualStrategy,
};
use super::params::OperatorKind;

/// Default maximum number of tips retained by [`TtseStrategy`].
pub const DEFAULT_TIP_CAPACITY: usize = 12;

/// Default Jaccard threshold for merging two signals into one cluster.
pub const DEFAULT_CLUSTER_THRESHOLD: f64 = 0.5;

optimizer_wrapper!(TtseOptimizer, TtseStrategy);

/// Clusters similar feedback into bounded tips.
///
/// A signal is consumed when its target is [`EvolutionTarget::Skill`] or
/// [`EvolutionTarget::Other`]; it updates the parameter named `tips`.
#[derive(Debug, Clone, PartialEq)]
pub struct TtseStrategy {
    /// Maximum number of tips retained.
    pub capacity: usize,
    /// Jaccard similarity at or above which two excerpts share a cluster.
    pub threshold: f64,
}

impl Default for TtseStrategy {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_TIP_CAPACITY,
            threshold: DEFAULT_CLUSTER_THRESHOLD,
        }
    }
}

impl TtseStrategy {
    /// Creates the strategy with the defaults (capacity 12, threshold 0.5).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the tip capacity.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Replaces the clustering threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

impl TextualStrategy for TtseStrategy {
    fn name(&self) -> &'static str {
        "ttse"
    }

    fn select_signals(&self, signals: &[EvolutionSignal]) -> Vec<EvolutionSignal> {
        signals
            .iter()
            .filter(|signal| {
                signal.target == EvolutionTarget::Skill || signal.target == EvolutionTarget::Other
            })
            .cloned()
            .collect()
    }

    fn parameter_name(&self, signal: &EvolutionSignal) -> Option<String> {
        let relevant =
            signal.target == EvolutionTarget::Skill || signal.target == EvolutionTarget::Other;
        relevant.then(|| "tips".to_owned())
    }

    fn rewrite(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let feedback = feedback_texts(signals, gradient);
        if feedback.is_empty() {
            return current.to_owned();
        }
        let mut representatives: Vec<String> = Vec::new();
        for (excerpt, _weight) in &feedback {
            let excerpt = excerpt.trim();
            if excerpt.is_empty() {
                continue;
            }
            let tokens = normalized_tokens(excerpt);
            let clustered = representatives.iter().any(|representative| {
                jaccard(&normalized_tokens(representative), &tokens) >= self.threshold
            });
            if !clustered {
                representatives.push(excerpt.to_owned());
            }
        }
        let mut tips = parse_text_list(current);
        for representative in representatives {
            if tips.len() >= self.capacity {
                break;
            }
            let tip = shorten(&representative, MAX_EXCERPT_CHARS);
            if tips.iter().any(|existing| {
                normalize_excerpt(existing) == normalize_excerpt(&tip)
                    || near_duplicate(existing, &tip, self.threshold)
            }) {
                continue;
            }
            tips.push(tip);
        }
        encode_text_list(&tips)
    }

    fn prompt(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let gradient = if gradient.trim().is_empty() {
            "(none)"
        } else {
            gradient
        };
        let tips = parse_text_list(current);
        let tips = if tips.is_empty() {
            "(empty)".to_owned()
        } else {
            tips.iter()
                .map(|tip| format!("- {tip}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        format!(
            "Induce at most {capacity} reusable tips from the feedback below. Merge \
             duplicate advice into one tip.\n\n\
             Current tips:\n{tips}\n\n\
             Textual gradient:\n{gradient}\n\n\
             Relevant signals:\n{signals}\n\n\
             Return the complete tip list as a JSON array of strings wrapped in \
             <improved>...</improved>.",
            capacity = self.capacity,
            signals = render_signals(signals),
        )
    }

    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Tip
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(excerpt: &str) -> EvolutionSignal {
        EvolutionSignal::new("execution_failure", EvolutionTarget::Other, excerpt).expect("signal")
    }

    #[test]
    fn similar_feedback_collapses_into_one_tip() {
        let strategy = TtseStrategy::new();
        let signals = vec![
            signal("Retry the failed step"),
            signal("Retry the failed step twice"),
            signal("Always validate the schema"),
            signal("Retry the failed step"),
        ];
        let rewritten = strategy.rewrite("[]", "", &signals);
        let tips: Vec<String> = serde_json::from_str(&rewritten).expect("json array");
        assert_eq!(
            tips,
            vec![
                "Retry the failed step".to_owned(),
                "Always validate the schema".to_owned(),
            ]
        );
        assert_eq!(strategy.rewrite(&rewritten, "", &signals), rewritten);
    }

    #[test]
    fn capacity_bounds_the_tip_bank() {
        let strategy = TtseStrategy::new().with_capacity(2);
        let signals = vec![
            signal("alpha beta gamma"),
            signal("delta epsilon zeta"),
            signal("eta theta iota"),
        ];
        let rewritten = strategy.rewrite("[]", "", &signals);
        let tips: Vec<String> = serde_json::from_str(&rewritten).expect("json array");
        assert_eq!(tips.len(), 2);
    }

    #[test]
    fn skill_signals_are_also_consumed() {
        let strategy = TtseStrategy::new();
        let skill = EvolutionSignal::new("execution_failure", EvolutionTarget::Skill, "lesson")
            .expect("signal");
        let selected = strategy.select_signals(&[skill]);
        assert_eq!(selected.len(), 1);
        assert_eq!(
            strategy.parameter_name(&selected[0]),
            Some("tips".to_owned())
        );
        let memory =
            EvolutionSignal::new("low_score", EvolutionTarget::Memory, "nope").expect("signal");
        assert_eq!(strategy.select_signals(&[memory]).len(), 0);
    }

    #[test]
    fn no_feedback_leaves_the_bank_untouched() {
        let strategy = TtseStrategy::new();
        assert_eq!(strategy.rewrite(r#"["- keep"]"#, "", &[]), r#"["- keep"]"#);
    }
}
