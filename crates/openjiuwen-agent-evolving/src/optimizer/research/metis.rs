//! Context-section selection: Python's `context_evolve_call` / Metis optimizer.
//!
//! Ports the offline decision core of the Metis context optimizer: every signal
//! is mapped to one context section, the section's utility is the sum of the
//! weights of its signals (unmatched signals default to `Body`), and the rewrite
//! keeps the highest-utility sections in a deterministic order while recording
//! what was dropped.
//!
//! The `selection` parameter holds the JSON record
//! `{"sections": [...], "dropped": [...], "budget": N}`. Section text itself is
//! stored in the other parameters of the same [`OperatorState`](super::OperatorState);
//! the selection record is the compact plan a context assembler consumes.
//!
//! Section order is utility descending then name ascending; the retained list is
//! capped at [`MetisStrategy::budget`], and `dropped` is sorted alphabetically so
//! two runs over the same signals produce byte-identical output.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::signal::{EvolutionCategory, EvolutionSignal, EvolutionTarget};

use super::base::{render_signals, TextualStrategy};
use super::params::OperatorKind;

/// Default number of context sections retained by [`MetisStrategy`].
pub const DEFAULT_SECTION_BUDGET: usize = 8;

optimizer_wrapper!(MetisOptimizer, MetisStrategy);

/// Selects context sections by feedback utility.
///
/// A signal is consumed when its target or category is
/// [`EvolutionTarget::Context`] / [`EvolutionCategory::Context`]; it updates the
/// parameter named `selection`.
#[derive(Debug, Clone, PartialEq)]
pub struct MetisStrategy {
    /// Maximum number of sections retained in one selection.
    pub budget: usize,
}

impl Default for MetisStrategy {
    fn default() -> Self {
        Self {
            budget: DEFAULT_SECTION_BUDGET,
        }
    }
}

impl MetisStrategy {
    /// Creates the strategy with the default section budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the section budget.
    #[must_use]
    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }
}

/// The section a signal contributes to.
///
/// An explicit [`EvolutionSignal::section`] always wins; otherwise the category
/// maps onto the conventional section names, with `Body` as the fallback.
fn section_name(signal: &EvolutionSignal) -> String {
    let section = signal.section.trim();
    if !section.is_empty() {
        return section.to_owned();
    }
    let name = match signal.category {
        EvolutionCategory::Context => "Context",
        EvolutionCategory::Tool => "Tools",
        EvolutionCategory::Memory => "Memory",
        EvolutionCategory::Skill => "Skills",
        EvolutionCategory::Script | EvolutionCategory::Workflow => "Workflow",
        EvolutionCategory::Description | EvolutionCategory::Body | EvolutionCategory::Other => {
            "Body"
        }
    };
    name.to_owned()
}

impl TextualStrategy for MetisStrategy {
    fn name(&self) -> &'static str {
        "metis"
    }

    fn select_signals(&self, signals: &[EvolutionSignal]) -> Vec<EvolutionSignal> {
        signals
            .iter()
            .filter(|signal| {
                signal.target == EvolutionTarget::Context
                    || signal.category == EvolutionCategory::Context
            })
            .cloned()
            .collect()
    }

    fn parameter_name(&self, signal: &EvolutionSignal) -> Option<String> {
        let relevant = signal.target == EvolutionTarget::Context
            || signal.category == EvolutionCategory::Context;
        relevant.then(|| "selection".to_owned())
    }

    fn rewrite(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        if signals.is_empty() && gradient.trim().is_empty() {
            return current.to_owned();
        }
        let mut previous: Vec<String> = Vec::new();
        if let Ok(Value::Object(record)) = serde_json::from_str::<Value>(current) {
            for key in ["sections", "dropped"] {
                if let Some(Value::Array(items)) = record.get(key) {
                    for item in items {
                        if let Some(section) = item.as_str() {
                            let section = section.trim();
                            if !section.is_empty() && !previous.iter().any(|seen| seen == section) {
                                previous.push(section.to_owned());
                            }
                        }
                    }
                }
            }
        }
        let mut utilities: BTreeMap<String, f64> = BTreeMap::new();
        for signal in signals {
            *utilities.entry(section_name(signal)).or_insert(0.0) += signal.weight;
        }
        if signals.is_empty() {
            let lines = gradient
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count();
            if lines > 0 {
                *utilities.entry("Body".to_owned()).or_insert(0.0) += lines as f64;
            }
        }
        let mut ranked: Vec<(String, f64)> = utilities
            .iter()
            .filter(|(_, utility)| **utility > 0.0)
            .map(|(name, utility)| (name.clone(), *utility))
            .collect();
        ranked.sort_by(|(left_name, left_utility), (right_name, right_utility)| {
            right_utility
                .total_cmp(left_utility)
                .then_with(|| left_name.cmp(right_name))
        });
        let sections: Vec<String> = ranked
            .into_iter()
            .take(self.budget)
            .map(|(name, _)| name)
            .collect();
        let mut known: BTreeSet<String> = previous.into_iter().collect();
        known.extend(utilities.keys().cloned());
        let dropped: Vec<String> = known
            .into_iter()
            .filter(|name| !sections.contains(name))
            .collect();
        json!({
            "sections": sections,
            "dropped": dropped,
            "budget": self.budget,
        })
        .to_string()
    }

    fn prompt(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let gradient = if gradient.trim().is_empty() {
            "(none)"
        } else {
            gradient
        };
        format!(
            "Select the context sections that should be kept. Keep at most {budget} \
             sections, most useful first.\n\n\
             Current selection:\n{current}\n\n\
             Textual gradient:\n{gradient}\n\n\
             Relevant signals:\n{signals}\n\n\
             Return a JSON object {{\"sections\": [...], \"dropped\": [...], \
             \"budget\": {budget}}} wrapped in <improved>...</improved>.",
            budget = self.budget,
            signals = render_signals(signals),
        )
    }

    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(section: &str, weight: f64) -> EvolutionSignal {
        EvolutionSignal::new("low_score", EvolutionTarget::Context, "evidence")
            .expect("signal")
            .with_section(section)
            .with_weight(weight)
            .expect("weight")
    }

    #[test]
    fn sections_are_ordered_by_utility_then_name() {
        let strategy = MetisStrategy::new();
        let rewritten = strategy.rewrite(
            "",
            "",
            &[
                signal("Body", 0.3),
                signal("Tools", 0.8),
                signal("Alpha", 0.3),
            ],
        );
        let record: Value = serde_json::from_str(&rewritten).expect("json");
        assert_eq!(record["sections"], json!(["Tools", "Alpha", "Body"]));
        assert_eq!(record["budget"], json!(DEFAULT_SECTION_BUDGET));
    }

    #[test]
    fn previous_sections_are_kept_or_dropped_explicitly() {
        let strategy = MetisStrategy::new();
        let current = r#"{"sections": ["Body", "History"], "dropped": [], "budget": 8}"#;
        let rewritten = strategy.rewrite(current, "", &[signal("Body", 0.5)]);
        let record: Value = serde_json::from_str(&rewritten).expect("json");
        assert_eq!(record["sections"], json!(["Body"]));
        assert_eq!(record["dropped"], json!(["History"]));
    }

    #[test]
    fn unmatched_signals_default_to_body() {
        let strategy = MetisStrategy::new();
        let signal = EvolutionSignal::new("low_score", EvolutionTarget::Context, "evidence")
            .expect("signal")
            .with_category(EvolutionCategory::Body);
        let rewritten = strategy.rewrite("", "", &[signal]);
        let record: Value = serde_json::from_str(&rewritten).expect("json");
        assert_eq!(record["sections"], json!(["Body"]));
    }

    #[test]
    fn budget_caps_the_retained_sections() {
        let strategy = MetisStrategy::new().with_budget(1);
        let rewritten = strategy.rewrite("", "", &[signal("Body", 0.3), signal("Tools", 0.8)]);
        let record: Value = serde_json::from_str(&rewritten).expect("json");
        assert_eq!(record["sections"], json!(["Tools"]));
        assert_eq!(record["dropped"], json!(["Body"]));
    }

    #[test]
    fn no_feedback_leaves_the_selection_untouched() {
        let strategy = MetisStrategy::new();
        let current = r#"{"sections": ["Body"], "dropped": [], "budget": 8}"#;
        assert_eq!(strategy.rewrite(current, "", &[]), current);
    }

    #[test]
    fn only_context_signals_are_consumed() {
        let strategy = MetisStrategy::new();
        let tool = EvolutionSignal::new("execution_failure", EvolutionTarget::Tool, "nope")
            .expect("signal");
        assert_eq!(strategy.select_signals(&[tool]).len(), 0);
        let selected = strategy.select_signals(&[signal("Body", 0.5)]);
        assert_eq!(
            strategy.parameter_name(&selected[0]),
            Some("selection".to_owned())
        );
    }
}
