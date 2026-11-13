//! Instruction/prompt optimization: Python's `llm_call` optimizer.
//!
//! Ports the outcome of `InstructionOptimizer` in
//! `openjiuwen/agent_evolving/optimizer/llm_call/instruction_optimizer.py`
//! without the Python model plumbing: failure-driven signals (the ones the
//! Python optimizer selects) are consolidated into a bounded bullet list appended
//! to the instruction, and a model can rewrite the whole instruction when one is
//! injected.
//!
//! The rewrite is deterministic: bullets are added in signal order, deduplicated
//! by normalised excerpt, capped at [`InstructionStrategy::max_bullets`], and the
//! total text never grows past [`InstructionStrategy::max_chars`] characters.

use crate::signal::{normalize_excerpt, EvolutionSignal, EvolutionTarget, MAX_EXCERPT_CHARS};

use super::base::{feedback_texts, render_signals, shorten, strip_bullet, TextualStrategy};
use super::params::OperatorKind;

optimizer_wrapper!(InstructionOptimizer, InstructionStrategy);

/// Consolidates failure feedback into instruction text.
///
/// Only [`EvolutionTarget::Instruction`] signals are consumed: they update the
/// parameter named `instruction` on every registered instruction operator.
#[derive(Debug, Clone, PartialEq)]
pub struct InstructionStrategy {
    /// Maximum number of new feedback bullets appended per rewrite.
    pub max_bullets: usize,
    /// Maximum total instruction length, in characters, after the rewrite.
    pub max_chars: usize,
    /// Maximum length of one appended bullet, in characters.
    pub max_bullet_chars: usize,
}

impl Default for InstructionStrategy {
    fn default() -> Self {
        Self {
            max_bullets: 5,
            max_chars: 2_000,
            max_bullet_chars: MAX_EXCERPT_CHARS,
        }
    }
}

impl InstructionStrategy {
    /// Creates the strategy with the Python-shaped defaults (5 bullets, 2000
    /// characters).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the bullet and length bounds.
    #[must_use]
    pub fn with_limits(mut self, max_bullets: usize, max_chars: usize) -> Self {
        self.max_bullets = max_bullets;
        self.max_chars = max_chars;
        self
    }
}

impl TextualStrategy for InstructionStrategy {
    fn name(&self) -> &'static str {
        "instruction"
    }

    fn select_signals(&self, signals: &[EvolutionSignal]) -> Vec<EvolutionSignal> {
        signals
            .iter()
            .filter(|signal| signal.target == EvolutionTarget::Instruction)
            .cloned()
            .collect()
    }

    fn parameter_name(&self, signal: &EvolutionSignal) -> Option<String> {
        (signal.target == EvolutionTarget::Instruction).then(|| "instruction".to_owned())
    }

    fn rewrite(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let mut text = current.trim_end().to_owned();
        let mut seen: Vec<String> = text
            .lines()
            .map(|line| normalize_excerpt(strip_bullet(line.trim())))
            .filter(|normalized| !normalized.is_empty())
            .collect();
        let normalized_text = normalize_excerpt(&text);
        let mut added = 0;
        for (excerpt, _weight) in feedback_texts(signals, gradient) {
            if added >= self.max_bullets {
                break;
            }
            let normalized = normalize_excerpt(&excerpt);
            if normalized.is_empty()
                || seen.contains(&normalized)
                || normalized_text.contains(&normalized)
            {
                continue;
            }
            let bullet = format!("- {}", shorten(&excerpt, self.max_bullet_chars));
            if text.chars().count() + bullet.chars().count() + 1 > self.max_chars {
                break;
            }
            text.push('\n');
            text.push_str(&bullet);
            seen.push(normalized);
            added += 1;
        }
        text
    }

    fn prompt(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let gradient = if gradient.trim().is_empty() {
            "(none)"
        } else {
            gradient
        };
        format!(
            "Improve the instruction below so it avoids the observed failures.\n\
             Keep the existing intent and formatting.\n\n\
             Current instruction:\n<<<\n{current}\n>>>\n\n\
             Textual gradient:\n{gradient}\n\n\
             Relevant signals:\n{signals}\n\n\
             Return the complete rewritten instruction wrapped in \
             <improved>...</improved>.",
            signals = render_signals(signals),
        )
    }

    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Instruction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(excerpt: &str) -> EvolutionSignal {
        EvolutionSignal::new("low_score", EvolutionTarget::Instruction, excerpt).expect("signal")
    }

    #[test]
    fn rewrite_appends_deduped_bounded_bullets() {
        let strategy = InstructionStrategy::new();
        let current = "Plan the task.";
        let signals = vec![
            signal("Add a verification step."),
            signal("add a verification step."),
            signal("State the expected output."),
        ];
        let rewritten = strategy.rewrite(current, "Add a verification step.", &signals);
        assert_eq!(
            rewritten,
            "Plan the task.\n- Add a verification step.\n- State the expected output."
        );

        let again = strategy.rewrite(&rewritten, "", &signals);
        assert_eq!(again, rewritten);
    }

    #[test]
    fn rewrite_respects_the_bullet_and_length_caps() {
        let signals: Vec<EvolutionSignal> = (0..5)
            .map(|index| signal(&format!("feedback number {index}")))
            .collect();
        let bullet_capped = InstructionStrategy::new()
            .with_limits(2, 2_000)
            .rewrite("Short.", "", &signals);
        assert_eq!(bullet_capped.matches("\n- ").count(), 2);

        let length_capped = InstructionStrategy::new()
            .with_limits(5, 40)
            .rewrite("Short.", "", &signals);
        assert!(length_capped.chars().count() <= 40);
        assert_eq!(length_capped.matches("\n- ").count(), 1);
    }

    #[test]
    fn only_instruction_signals_are_consumed() {
        let strategy = InstructionStrategy::new();
        let tool_signal =
            EvolutionSignal::new("execution_failure", EvolutionTarget::Tool, "bad schema")
                .expect("signal");
        assert_eq!(strategy.select_signals(&[tool_signal]).len(), 0);
        let selected = strategy.select_signals(&[signal("fix it")]);
        assert_eq!(selected.len(), 1);
        assert_eq!(
            strategy.parameter_name(&selected[0]),
            Some("instruction".to_owned())
        );
    }

    #[test]
    fn prompt_contains_every_input() {
        let strategy = InstructionStrategy::new();
        let prompt = strategy.prompt("Do it.", "Be precise.", &[signal("missed step")]);
        assert!(prompt.contains("Do it."));
        assert!(prompt.contains("Be precise."));
        assert!(prompt.contains("missed step"));
        assert!(prompt.contains("<improved>"));
    }
}
