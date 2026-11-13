//! Long-term memory optimization: Python's `memory_call` optimizer.
//!
//! Ports the outcome of the Python `memory_call` optimizer: the tunable value is
//! the ordered list of memory entries, and a rewrite adds the highest-signal
//! failures, merges near-duplicates, and drops the weakest entries once the list
//! exceeds its capacity.
//!
//! The `entries` parameter holds a JSON array of strings. Ordering is
//! deterministic: existing entries keep their position, new entries are appended
//! in signal order (newest last), and eviction removes the lowest-weight entry
//! first (ties: oldest first).

use crate::signal::{normalize_excerpt, EvolutionSignal, EvolutionTarget};

use super::base::{
    encode_text_list, feedback_texts, near_duplicate, parse_text_list, render_signals,
    TextualStrategy,
};
use super::params::OperatorKind;

optimizer_wrapper!(MemoryOptimizer, MemoryStrategy);

/// Consolidates memory signals into a bounded entry list.
///
/// Only [`EvolutionTarget::Memory`] signals are consumed; they update the
/// parameter named `entries`.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryStrategy {
    /// Maximum number of entries retained; the weakest are dropped first.
    pub capacity: usize,
    /// Maximum number of new entries appended per rewrite.
    pub top_k: usize,
    /// Jaccard similarity at or above which two entries are merged.
    pub similarity: f64,
}

impl Default for MemoryStrategy {
    fn default() -> Self {
        Self {
            capacity: 16,
            top_k: 4,
            similarity: 0.8,
        }
    }
}

impl MemoryStrategy {
    /// Creates the strategy with the defaults (capacity 16, top 4, similarity
    /// 0.8).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the capacity and per-rewrite insertion limit.
    #[must_use]
    pub fn with_limits(mut self, capacity: usize, top_k: usize) -> Self {
        self.capacity = capacity;
        self.top_k = top_k;
        self
    }

    /// Replaces the near-duplicate similarity threshold.
    #[must_use]
    pub fn with_similarity(mut self, similarity: f64) -> Self {
        self.similarity = similarity;
        self
    }
}

impl TextualStrategy for MemoryStrategy {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn select_signals(&self, signals: &[EvolutionSignal]) -> Vec<EvolutionSignal> {
        signals
            .iter()
            .filter(|signal| signal.target == EvolutionTarget::Memory)
            .cloned()
            .collect()
    }

    fn parameter_name(&self, signal: &EvolutionSignal) -> Option<String> {
        (signal.target == EvolutionTarget::Memory).then(|| "entries".to_owned())
    }

    fn rewrite(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let mut entries = parse_text_list(current);
        let mut weights: Vec<f64> = entries
            .iter()
            .map(|entry| {
                signals
                    .iter()
                    .filter(|signal| {
                        normalize_excerpt(entry) == normalize_excerpt(&signal.excerpt)
                            || near_duplicate(entry, &signal.excerpt, self.similarity)
                    })
                    .map(|signal| signal.weight)
                    .fold(0.0, f64::max)
            })
            .collect();
        let mut added = 0;
        for (excerpt, weight) in feedback_texts(signals, gradient) {
            if added >= self.top_k {
                break;
            }
            let excerpt = excerpt.trim();
            if excerpt.is_empty() {
                continue;
            }
            let normalized = normalize_excerpt(excerpt);
            if entries
                .iter()
                .any(|entry| normalize_excerpt(entry) == normalized)
            {
                continue;
            }
            if let Some(position) = entries
                .iter()
                .position(|entry| near_duplicate(entry, excerpt, self.similarity))
            {
                // Merge near-duplicates into the longer (more specific) text and
                // keep the strongest observed weight for eviction ordering.
                if excerpt.chars().count() > entries[position].chars().count() {
                    excerpt.clone_into(&mut entries[position]);
                }
                weights[position] = weights[position].max(weight);
                continue;
            }
            entries.push(excerpt.to_owned());
            weights.push(weight);
            added += 1;
        }
        while entries.len() > self.capacity {
            let victim = weights
                .iter()
                .enumerate()
                .min_by(|(left_index, left_weight), (right_index, right_weight)| {
                    left_weight
                        .total_cmp(right_weight)
                        .then_with(|| left_index.cmp(right_index))
                })
                .map(|(index, _)| index);
            let Some(victim) = victim else {
                break;
            };
            entries.remove(victim);
            weights.remove(victim);
        }
        encode_text_list(&entries)
    }

    fn prompt(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let gradient = if gradient.trim().is_empty() {
            "(none)"
        } else {
            gradient
        };
        let entries = parse_text_list(current);
        let entries = if entries.is_empty() {
            "(empty)".to_owned()
        } else {
            entries
                .iter()
                .map(|entry| format!("- {entry}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        format!(
            "Improve the long-term memory entries below. Keep at most {capacity} entries \
             and prefer short, reusable lessons.\n\n\
             Current entries:\n{entries}\n\n\
             Textual gradient:\n{gradient}\n\n\
             Relevant signals:\n{signals}\n\n\
             Return the complete entry list as a JSON array of strings wrapped in \
             <improved>...</improved>.",
            capacity = self.capacity,
            signals = render_signals(signals),
        )
    }

    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(excerpt: &str) -> EvolutionSignal {
        EvolutionSignal::new("low_score", EvolutionTarget::Memory, excerpt).expect("signal")
    }

    #[test]
    fn rewrite_appends_deduped_entries() {
        let strategy = MemoryStrategy::new();
        let signals = vec![
            signal("Always confirm the target directory exists."),
            signal("Always confirm the target directory exists."),
            signal("Prefer absolute paths."),
        ];
        let rewritten = strategy.rewrite("[]", "", &signals);
        let entries: Vec<String> = serde_json::from_str(&rewritten).expect("json array");
        assert_eq!(
            entries,
            vec![
                "Always confirm the target directory exists.".to_owned(),
                "Prefer absolute paths.".to_owned(),
            ]
        );
        assert_eq!(strategy.rewrite(&rewritten, "", &signals), rewritten);
    }

    #[test]
    fn rewrite_merges_near_duplicates() {
        let strategy = MemoryStrategy::new();
        let existing = r#"["Retry the failed step twice before giving up."]"#;
        let rewritten = strategy.rewrite(
            existing,
            "",
            &[signal("Retry the failed step twice before giving up")],
        );
        assert_eq!(rewritten, existing);
    }

    #[test]
    fn rewrite_drops_the_lowest_weight_entry_at_capacity() {
        let strategy = MemoryStrategy::new().with_limits(2, 4);
        let existing = r#"["old one", "old two"]"#;
        let rewritten = strategy.rewrite(existing, "", &[signal("new lesson")]);
        let entries: Vec<String> = serde_json::from_str(&rewritten).expect("json array");
        assert_eq!(entries, vec!["old two".to_owned(), "new lesson".to_owned()]);
    }

    #[test]
    fn targeted_entries_survive_eviction_weighting() {
        let strategy = MemoryStrategy::new().with_limits(2, 4);
        let existing = r#"["Keep this one.", "Drop this one."]"#;
        let rewritten = strategy.rewrite(
            existing,
            "",
            &[
                signal("Keep this one."),
                signal("A new lesson with weight."),
            ],
        );
        let entries: Vec<String> = serde_json::from_str(&rewritten).expect("json array");
        assert_eq!(entries.len(), 2);
        assert!(entries.contains(&"Keep this one.".to_owned()));
        assert!(entries.contains(&"A new lesson with weight.".to_owned()));
    }

    #[test]
    fn only_memory_signals_are_consumed() {
        let strategy = MemoryStrategy::new();
        let instruction = EvolutionSignal::new("low_score", EvolutionTarget::Instruction, "nope")
            .expect("signal");
        assert_eq!(strategy.select_signals(&[instruction]).len(), 0);
        let selected = strategy.select_signals(&[signal("lesson")]);
        assert_eq!(
            strategy.parameter_name(&selected[0]),
            Some("entries".to_owned())
        );
    }
}
