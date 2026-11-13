//! Tool-description optimization: Python's `tool_call` optimizer.
//!
//! Ports the outcome of the Python `tool_call` optimizer: each tool owns one
//! operator (`operator_id = "tool.{name}"`) whose `description` parameter learns
//! from failures attributed to that tool. Signals are matched by
//! [`EvolutionSignal::tool_name`], so one tool's feedback never leaks into
//! another tool's description, and the appended clause is deduplicated and
//! bounded.

use crate::signal::{normalize_excerpt, EvolutionSignal, EvolutionTarget, MAX_EXCERPT_CHARS};

use super::base::{
    feedback_texts, render_signals, shorten, strip_bullet, TextualStrategy, PARAMETER_SEPARATOR,
};
use super::params::OperatorKind;

optimizer_wrapper!(ToolOptimizer, ToolStrategy);

/// Appends a bounded "When this fails" clause to a tool description.
///
/// A signal is consumed when its target is [`EvolutionTarget::Tool`] and it
/// names a tool; the signal updates `description` on the operator whose id is
/// `tool.{tool_name}`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolStrategy {
    /// Maximum number of new failure clauses appended per rewrite.
    pub max_clauses: usize,
    /// Maximum total description length, in characters, after the rewrite.
    pub max_chars: usize,
    /// Maximum length of one appended clause, in characters.
    pub max_clause_chars: usize,
}

impl Default for ToolStrategy {
    fn default() -> Self {
        Self {
            max_clauses: 3,
            max_chars: 1_000,
            max_clause_chars: MAX_EXCERPT_CHARS,
        }
    }
}

impl ToolStrategy {
    /// Creates the strategy with the defaults (3 clauses, 1000 characters).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the clause and length bounds.
    #[must_use]
    pub fn with_limits(mut self, max_clauses: usize, max_chars: usize) -> Self {
        self.max_clauses = max_clauses;
        self.max_chars = max_chars;
        self
    }

    /// The operator id the strategy uses for `tool_name`.
    #[must_use]
    pub fn operator_id(tool_name: &str) -> String {
        format!("tool.{tool_name}")
    }
}

impl TextualStrategy for ToolStrategy {
    fn name(&self) -> &'static str {
        "tool"
    }

    fn select_signals(&self, signals: &[EvolutionSignal]) -> Vec<EvolutionSignal> {
        signals
            .iter()
            .filter(|signal| {
                signal.target == EvolutionTarget::Tool
                    && signal
                        .tool_name
                        .as_deref()
                        .is_some_and(|name| !name.trim().is_empty())
            })
            .cloned()
            .collect()
    }

    fn parameter_name(&self, signal: &EvolutionSignal) -> Option<String> {
        if signal.target != EvolutionTarget::Tool {
            return None;
        }
        let tool_name = signal.tool_name.as_deref()?.trim();
        if tool_name.is_empty() {
            return None;
        }
        Some(format!(
            "{}{PARAMETER_SEPARATOR}description",
            Self::operator_id(tool_name)
        ))
    }

    fn rewrite(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let mut text = current.trim_end().to_owned();
        let mut seen: Vec<String> = text
            .lines()
            .map(|line| normalize_excerpt(strip_bullet(line.trim())))
            .filter(|normalized| !normalized.is_empty())
            .collect();
        let normalized_text = normalize_excerpt(&text);
        let mut clauses: Vec<String> = Vec::new();
        for (excerpt, _weight) in feedback_texts(signals, gradient) {
            if clauses.len() >= self.max_clauses {
                break;
            }
            let normalized = normalize_excerpt(&excerpt);
            if normalized.is_empty()
                || seen.contains(&normalized)
                || normalized_text.contains(&normalized)
            {
                continue;
            }
            clauses.push(shorten(&excerpt, self.max_clause_chars));
            seen.push(normalized);
        }
        if clauses.is_empty() {
            return text;
        }
        let joined = clauses.join("; ");
        let terminator = if joined.ends_with(['.', '!', '?']) {
            ""
        } else {
            "."
        };
        let suffix = format!("\n\nWhen this fails: {joined}{terminator}");
        if text.chars().count() + suffix.chars().count() > self.max_chars {
            return text;
        }
        text.push_str(&suffix);
        text
    }

    fn prompt(&self, current: &str, gradient: &str, signals: &[EvolutionSignal]) -> String {
        let gradient = if gradient.trim().is_empty() {
            "(none)"
        } else {
            gradient
        };
        format!(
            "Improve the tool description below so the model uses the tool \
             correctly and recovers from the observed failures. Keep the existing \
             behaviour descriptions.\n\n\
             Current description:\n<<<\n{current}\n>>>\n\n\
             Textual gradient:\n{gradient}\n\n\
             Relevant signals:\n{signals}\n\n\
             Return the complete rewritten description wrapped in \
             <improved>...</improved>.",
            signals = render_signals(signals),
        )
    }

    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Tool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(tool: &str, excerpt: &str) -> EvolutionSignal {
        EvolutionSignal::new("execution_failure", EvolutionTarget::Tool, excerpt)
            .expect("signal")
            .with_tool(tool)
    }

    #[test]
    fn rewrite_adds_one_bounded_clause() {
        let strategy = ToolStrategy::new();
        let signals = vec![
            signal("search", "The query must be a single string."),
            signal("search", "The query must be a single string."),
        ];
        let rewritten = strategy.rewrite("Search the web.", "", &signals);
        assert_eq!(
            rewritten,
            "Search the web.\n\nWhen this fails: The query must be a single string."
        );
        assert_eq!(strategy.rewrite(&rewritten, "", &signals), rewritten);
    }

    #[test]
    fn signal_keys_are_qualified_per_tool() {
        let strategy = ToolStrategy::new();
        let selected = strategy.select_signals(&[
            signal("search", "schema mismatch"),
            signal("fetch", "timeout"),
            EvolutionSignal::new("low_score", EvolutionTarget::Instruction, "nope")
                .expect("signal"),
        ]);
        assert_eq!(selected.len(), 2);
        assert_eq!(
            strategy.parameter_name(&selected[0]),
            Some("tool.search::description".to_owned())
        );
        assert_eq!(
            strategy.parameter_name(&selected[1]),
            Some("tool.fetch::description".to_owned())
        );
        assert_eq!(ToolStrategy::operator_id("search"), "tool.search");
    }

    #[test]
    fn rewrite_respects_caps() {
        let strategy = ToolStrategy::new().with_limits(1, 60);
        let signals: Vec<EvolutionSignal> = (0..4)
            .map(|index| signal("search", &format!("failure number {index}")))
            .collect();
        let rewritten = strategy.rewrite("Search.", "", &signals);
        assert_eq!(rewritten.matches("When this fails").count(), 1);
        assert!(rewritten.chars().count() <= 60);
    }
}
