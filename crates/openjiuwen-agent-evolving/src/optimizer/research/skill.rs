//! Skill-experience optimization: Python's `skill_call` optimizer.
//!
//! Ports the outcome of the Python `experience_optimizer`: the tunable value is
//! the skill's experience section, stored as a JSON array of bullet strings (each
//! rendered with a leading `- `). A rewrite appends the highest-signal lessons,
//! merges near-duplicates, and keeps the list bounded so the document stays
//! prompt-sized.
//!
//! The `experience` parameter holds the JSON array of bullets. Ordering is
//! deterministic: existing bullets keep their position, new bullets are appended
//! in signal order (newest last), and eviction removes the lowest-weight bullet
//! first (ties: oldest first).

use crate::signal::{normalize_excerpt, EvolutionSignal, EvolutionTarget};

use super::base::{
    encode_text_list, feedback_texts, near_duplicate, parse_text_list, render_signals,
    strip_bullet, TextualStrategy,
};
use super::params::OperatorKind;

optimizer_wrapper!(SkillExperienceOptimizer, SkillExperienceStrategy);

/// Consolidates skill signals into bounded experience bullets.
///
/// Only [`EvolutionTarget::Skill`] signals are consumed; they update the
/// parameter named `experience`.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillExperienceStrategy {
    /// Maximum number of bullets retained; the weakest are dropped first.
    pub capacity: usize,
    /// Maximum number of new bullets appended per rewrite.
    pub top_k: usize,
    /// Jaccard similarity at or above which two bullets are merged.
    pub similarity: f64,
}

impl Default for SkillExperienceStrategy {
    fn default() -> Self {
        Self {
            capacity: 16,
            top_k: 4,
            similarity: 0.8,
        }
    }
}

impl SkillExperienceStrategy {
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

    /// Renders one entry with its bullet marker.
    fn bullet(text: &str) -> String {
        format!("- {}", strip_bullet(text.trim()))
    }
}

impl TextualStrategy for SkillExperienceStrategy {
    fn name(&self) -> &'static str {
        "skill_experience"
    }

    fn select_signals(&self, signals: &[EvolutionSignal]) -> Vec<EvolutionSignal> {
        signals
            .iter()
            .filter(|signal| signal.target == EvolutionTarget::Skill)
            .cloned()
            .collect()
    }

    fn parameter_name(&self, signal: &EvolutionSignal) -> Option<String> {
        (signal.target == EvolutionTarget::Skill).then(|| "experience".to_owned())
    }

    fn rewrite(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let mut entries = parse_text_list(current);
        let mut weights: Vec<f64> = entries
            .iter()
            .map(|entry| {
                let text = strip_bullet(entry);
                signals
                    .iter()
                    .filter(|signal| {
                        normalize_excerpt(text) == normalize_excerpt(&signal.excerpt)
                            || near_duplicate(text, &signal.excerpt, self.similarity)
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
            let normalized = normalize_excerpt(strip_bullet(excerpt));
            if entries
                .iter()
                .any(|entry| normalize_excerpt(strip_bullet(entry)) == normalized)
            {
                continue;
            }
            if let Some(position) = entries
                .iter()
                .position(|entry| near_duplicate(strip_bullet(entry), excerpt, self.similarity))
            {
                if strip_bullet(excerpt).chars().count()
                    > strip_bullet(&entries[position]).chars().count()
                {
                    entries[position] = Self::bullet(excerpt);
                }
                weights[position] = weights[position].max(weight);
                continue;
            }
            entries.push(Self::bullet(excerpt));
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
                .map(|entry| {
                    let bullet = strip_bullet(entry);
                    format!("- {bullet}")
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        format!(
            "Improve the skill experience bullets below. Keep at most {capacity} bullets, \
             one reusable lesson per bullet.\n\n\
             Current bullets:\n{entries}\n\n\
             Textual gradient:\n{gradient}\n\n\
             Relevant signals:\n{signals}\n\n\
             Return the complete bullet list as a JSON array of strings, each starting \
             with '- ', wrapped in <improved>...</improved>.",
            capacity = self.capacity,
            signals = render_signals(signals),
        )
    }

    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Skill
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(excerpt: &str) -> EvolutionSignal {
        EvolutionSignal::new("execution_failure", EvolutionTarget::Skill, excerpt).expect("signal")
    }

    #[test]
    fn rewrite_prefixes_bullets_and_dedupes() {
        let strategy = SkillExperienceStrategy::new();
        let signals = vec![
            signal("Check the schema first."),
            signal("Check the schema first."),
        ];
        let rewritten = strategy.rewrite("[]", "", &signals);
        let entries: Vec<String> = serde_json::from_str(&rewritten).expect("json array");
        assert_eq!(entries, vec!["- Check the schema first.".to_owned()]);
        assert_eq!(strategy.rewrite(&rewritten, "", &signals), rewritten);
    }

    #[test]
    fn rewrite_drops_the_lowest_weight_bullet_at_capacity() {
        let strategy = SkillExperienceStrategy::new().with_limits(2, 4);
        let rewritten =
            strategy.rewrite(r#"["- Old one", "- Old two"]"#, "", &[signal("New lesson")]);
        let entries: Vec<String> = serde_json::from_str(&rewritten).expect("json array");
        assert_eq!(
            entries,
            vec!["- Old two".to_owned(), "- New lesson".to_owned()]
        );
    }

    #[test]
    fn only_skill_signals_are_consumed() {
        let strategy = SkillExperienceStrategy::new();
        let tool = EvolutionSignal::new("execution_failure", EvolutionTarget::Tool, "nope")
            .expect("signal");
        assert_eq!(strategy.select_signals(&[tool]).len(), 0);
        let selected = strategy.select_signals(&[signal("lesson")]);
        assert_eq!(
            strategy.parameter_name(&selected[0]),
            Some("experience".to_owned())
        );
    }
}
