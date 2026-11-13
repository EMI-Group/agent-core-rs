//! Structured optimization history — the textual policy-update mechanism.
//!
//! Ports `history.py`. No weights are trained; each iteration's outcome is
//! recorded as a [`HistoryEntry`], then compressed into a short textual update
//! that is fed back into the next round of prompt generation. That compressed
//! text *is* the "policy gradient": it steers the next candidates without any
//! fine-tuning.
//!
//! Python has an LLM-backed [`HistoryCompressor`] with four buckets
//! (`improved_reward`, `reduced_reward`, `preserve`, `avoid`). Only the
//! deterministic `_heuristic` port is kept: a budgeted distillation of the best
//! and weakest entries. It needs no model, so a run is reproducible.

use serde::{Deserialize, Serialize};

/// How many characters of a prompt [`HistoryEntry::render`] keeps by default.
const DEFAULT_PROMPT_LIMIT: usize = 600;

/// How many recent entries [`OptimizationHistory::render`] includes.
const RECENT_LIMIT: usize = 4;

/// The default compressed-text budget of [`HistoryCompressor`].
const DEFAULT_BUDGET: usize = 1200;

/// How many observations per bucket the heuristic compressor keeps.
const OBSERVATIONS_PER_BUCKET: usize = 3;

/// One recorded iteration of the optimization loop.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HistoryEntry {
    /// The 1-based iteration number.
    pub index: usize,
    /// The reward the iteration's best candidate earned.
    pub reward: f64,
    /// The prompt that earned it.
    pub prompt: String,
    /// What the iteration observed about the prompt's behavior.
    pub observations: Vec<String>,
}

impl HistoryEntry {
    /// Creates an entry without observations.
    #[must_use]
    pub fn new(index: usize, reward: f64, prompt: impl Into<String>) -> Self {
        Self {
            index,
            reward,
            prompt: prompt.into(),
            observations: Vec::new(),
        }
    }

    /// Replaces the observations.
    #[must_use]
    pub fn with_observations(mut self, observations: Vec<String>) -> Self {
        self.observations = observations;
        self
    }

    /// Renders the entry with the default 600-character prompt limit.
    #[must_use]
    pub fn render(&self) -> String {
        self.render_with_limit(DEFAULT_PROMPT_LIMIT)
    }

    /// Renders the entry, truncating the prompt to `limit` characters.
    ///
    /// The header is `### Iteration {index} — reward {reward:.3f}` (an em dash),
    /// followed by the prompt and, when present, an `Observations:` section with
    /// one `- {observation}` line per observation. A truncated prompt ends with
    /// `…`.
    #[must_use]
    pub fn render_with_limit(&self, limit: usize) -> String {
        let prompt = truncate_prompt(&self.prompt, limit);
        let mut lines = vec![
            format!("### Iteration {} — reward {:.3}", self.index, self.reward),
            "Prompt:".to_owned(),
            prompt,
        ];
        if !self.observations.is_empty() {
            lines.push("Observations:".to_owned());
            for observation in &self.observations {
                lines.push(format!("- {observation}"));
            }
        }
        lines.join("\n")
    }
}

/// Truncates `prompt` to at most `limit` characters, appending `…` when cut.
fn truncate_prompt(prompt: &str, limit: usize) -> String {
    if prompt.chars().count() <= limit {
        return prompt.to_owned();
    }
    let mut truncated: String = prompt.chars().take(limit).collect();
    truncated.push('…');
    truncated
}

/// The recorded iterations plus the latest compressed lessons.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OptimizationHistory {
    /// The recorded iterations, oldest first.
    pub entries: Vec<HistoryEntry>,
    /// The latest compressed lessons, when a compressor has run.
    pub compressed: String,
}

impl OptimizationHistory {
    /// Appends one iteration.
    pub fn add(&mut self, entry: HistoryEntry) {
        self.entries.push(entry);
    }

    /// The best reward seen (`0.0` when the history is empty).
    #[must_use]
    pub fn best_reward(&self) -> f64 {
        let mut best = 0.0f64;
        for entry in &self.entries {
            best = best.max(entry.reward);
        }
        best
    }

    /// The prompt of the best entry (`""` when empty); ties keep the first entry.
    #[must_use]
    pub fn best_prompt(&self) -> String {
        let mut best: Option<&HistoryEntry> = None;
        for entry in &self.entries {
            if best.is_none_or(|best| entry.reward > best.reward) {
                best = Some(entry);
            }
        }
        best.map_or_else(String::new, |entry| entry.prompt.clone())
    }

    /// Renders the last `limit` entries, separated by a blank line.
    #[must_use]
    pub fn render_recent(&self, limit: usize) -> String {
        let start = self.entries.len().saturating_sub(limit);
        self.entries[start..]
            .iter()
            .map(HistoryEntry::render)
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Renders the compressed lessons and the recent iterations.
    ///
    /// The sections are `## Distilled lessons so far` (when [`Self::compressed`]
    /// is non-empty) and `## Recent iterations` (when there are entries), joined
    /// by a blank line.
    #[must_use]
    pub fn render(&self) -> String {
        let mut parts = Vec::new();
        if !self.compressed.is_empty() {
            parts.push(format!("## Distilled lessons so far\n{}", self.compressed));
        }
        if !self.entries.is_empty() {
            parts.push(format!(
                "## Recent iterations\n{}",
                self.render_recent(RECENT_LIMIT)
            ));
        }
        parts.join("\n\n")
    }

    /// Whether the history holds no entries (a compressed string may still be
    /// set by a caller).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Distills an [`OptimizationHistory`] into a short, budgeted lesson list.
///
/// Python asks a model for four buckets. This deterministic port emits two: the
/// best entry's observations to keep and, when a different entry exists, the
/// weakest entry's observations to avoid. Duplicate lesson lines are dropped and
/// the result is truncated on line boundaries to [`HistoryCompressor::budget`]
/// characters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HistoryCompressor {
    /// The maximum number of characters the compressed text may contain.
    budget: usize,
}

impl Default for HistoryCompressor {
    fn default() -> Self {
        Self {
            budget: DEFAULT_BUDGET,
        }
    }
}

impl HistoryCompressor {
    /// Creates a compressor with the default 1200-character budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the character budget.
    #[must_use]
    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    /// The configured character budget.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Compresses `history`; an empty history yields `""`.
    #[must_use]
    pub fn compress(&self, history: &OptimizationHistory) -> String {
        if history.entries.is_empty() {
            return String::new();
        }

        let mut ordered: Vec<&HistoryEntry> = history.entries.iter().collect();
        ordered.sort_by(|left, right| {
            left.reward
                .partial_cmp(&right.reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = ordered[ordered.len() - 1];
        let worst = ordered[0];
        let mut lines = Vec::new();
        lines.push(format!(
            "**Best so far (reward {:.3})** — preserve its structure.",
            best.reward
        ));
        for observation in best.observations.iter().take(OBSERVATIONS_PER_BUCKET) {
            lines.push(format!("- keep: {observation}"));
        }
        if ordered.len() > 1 {
            lines.push(format!(
                "**Weakest (reward {:.3})** — avoid its choices.",
                worst.reward
            ));
            for observation in worst.observations.iter().take(OBSERVATIONS_PER_BUCKET) {
                lines.push(format!("- avoid: {observation}"));
            }
        }

        let mut seen = std::collections::BTreeSet::new();
        let mut compressed = String::new();
        for line in lines {
            if !seen.insert(line.clone()) {
                continue;
            }
            let first = compressed.is_empty();
            let separator = usize::from(!first);
            if compressed.chars().count() + separator + line.chars().count() > self.budget {
                break;
            }
            if !first {
                compressed.push('\n');
            }
            compressed.push_str(&line);
        }
        compressed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_shape_matches_the_python_format() {
        let entry = HistoryEntry::new(2, 0.5, "do it")
            .with_observations(vec!["kept".to_owned(), "bad".to_owned()]);
        assert_eq!(
            entry.render(),
            "### Iteration 2 — reward 0.500\nPrompt:\ndo it\nObservations:\n- kept\n- bad"
        );

        let plain = HistoryEntry::new(1, 1.0, "p");
        assert_eq!(plain.render(), "### Iteration 1 — reward 1.000\nPrompt:\np");
    }

    #[test]
    fn render_truncates_the_prompt_with_an_ellipsis() {
        let entry = HistoryEntry::new(1, 0.25, "abcdefghij");
        assert_eq!(
            entry.render_with_limit(4),
            "### Iteration 1 — reward 0.250\nPrompt:\nabcd…"
        );
        assert!(entry.render_with_limit(10).contains("abcdefghij"));
        assert!(!entry.render_with_limit(10).contains('…'));
        assert_eq!(truncate_prompt("汉字汉字", 2), "汉字…");
    }

    #[test]
    fn empty_history_reports_empty_defaults() {
        let history = OptimizationHistory::default();
        assert!(history.is_empty());
        assert_eq!(history.best_reward(), 0.0);
        assert_eq!(history.best_prompt(), "");
        assert_eq!(history.render(), "");
        assert_eq!(history.render_recent(4), "");
    }

    #[test]
    fn best_reward_and_prompt_keep_the_first_tie() {
        let mut history = OptimizationHistory::default();
        history.add(HistoryEntry::new(1, 0.4, "first"));
        history.add(HistoryEntry::new(2, 0.9, "winner"));
        history.add(HistoryEntry::new(3, 0.9, "also winner"));
        assert!((history.best_reward() - 0.9).abs() < 1e-12);
        assert_eq!(history.best_prompt(), "winner");
        assert!(!history.is_empty());
    }

    #[test]
    fn render_recent_and_render_sections() {
        let mut history = OptimizationHistory::default();
        for index in 1..=5 {
            history.add(HistoryEntry::new(
                index,
                0.1 * index as f64,
                format!("p{index}"),
            ));
        }
        let recent = history.render_recent(2);
        assert!(recent.contains("Iteration 4"));
        assert!(recent.contains("Iteration 5"));
        assert!(!recent.contains("Iteration 3"));
        assert_eq!(recent.matches("### Iteration").count(), 2);

        let default_recent = history.render_recent(4);
        assert_eq!(default_recent.matches("### Iteration").count(), 4);
        assert_eq!(
            history.render(),
            "## Recent iterations\n".to_owned() + &default_recent
        );

        history.compressed = "keep it short".to_owned();
        assert_eq!(
            history.render(),
            format!(
                "## Distilled lessons so far\nkeep it short\n\n## Recent iterations\n{default_recent}"
            )
        );

        let only_compressed = OptimizationHistory {
            compressed: "lessons".to_owned(),
            ..OptimizationHistory::default()
        };
        assert_eq!(
            only_compressed.render(),
            "## Distilled lessons so far\nlessons"
        );
    }

    #[test]
    fn compressor_handles_a_single_entry() {
        let mut history = OptimizationHistory::default();
        history.add(HistoryEntry::new(1, 0.42, "p").with_observations(vec!["a".to_owned()]));
        let compressor = HistoryCompressor::new();
        assert_eq!(compressor.budget(), 1200);
        assert_eq!(
            compressor.compress(&history),
            "**Best so far (reward 0.420)** — preserve its structure.\n- keep: a"
        );
    }

    #[test]
    fn compressor_emits_best_then_weakest() {
        let mut history = OptimizationHistory::default();
        history.add(
            HistoryEntry::new(1, 0.1, "weak")
                .with_observations(vec!["vague".to_owned(), "too long".to_owned()]),
        );
        history.add(
            HistoryEntry::new(2, 0.9, "strong").with_observations(vec!["clear steps".to_owned()]),
        );

        let compressed = HistoryCompressor::new().compress(&history);
        assert_eq!(
            compressed,
            "**Best so far (reward 0.900)** — preserve its structure.\n\
             - keep: clear steps\n\
             **Weakest (reward 0.100)** — avoid its choices.\n\
             - avoid: vague\n\
             - avoid: too long"
        );
        // Deterministic: the same history always compresses the same way.
        assert_eq!(compressed, HistoryCompressor::new().compress(&history));
    }

    #[test]
    fn compressor_skips_repeated_lessons() {
        let mut history = OptimizationHistory::default();
        history.add(
            HistoryEntry::new(1, 0.1, "weak")
                .with_observations(vec!["same".to_owned(), "same".to_owned()]),
        );
        history
            .add(HistoryEntry::new(2, 0.9, "strong").with_observations(vec!["keep me".to_owned()]));
        let compressed = HistoryCompressor::new().compress(&history);
        assert_eq!(compressed.matches("- avoid: same").count(), 1);
        assert_eq!(compressed.matches("- keep: keep me").count(), 1);
    }

    #[test]
    fn compressor_truncates_on_line_boundaries() {
        let mut history = OptimizationHistory::default();
        history.add(HistoryEntry::new(1, 0.9, "p").with_observations(vec!["first".to_owned()]));
        history.add(HistoryEntry::new(2, 0.1, "p").with_observations(vec!["second".to_owned()]));

        let full = HistoryCompressor::new().compress(&history);
        assert!(full.lines().count() >= 4);

        // Room for the header plus exactly one lesson line.
        let header = "**Best so far (reward 0.900)** — preserve its structure.";
        let budget = header.chars().count() + 1 + "- keep: first".chars().count();
        let truncated = HistoryCompressor::new()
            .with_budget(budget)
            .compress(&history);
        assert_eq!(truncated, format!("{header}\n- keep: first"));
        assert!(truncated.chars().count() <= budget);

        // No room at all yields an empty distillation rather than a partial line.
        assert_eq!(
            HistoryCompressor::new().with_budget(5).compress(&history),
            ""
        );
    }

    #[test]
    fn compressor_returns_empty_for_an_empty_history() {
        assert_eq!(
            HistoryCompressor::new().compress(&OptimizationHistory::default()),
            ""
        );
    }

    #[test]
    fn history_serializes_with_python_field_names() {
        let history = OptimizationHistory {
            entries: vec![HistoryEntry::new(1, 0.5, "p")],
            compressed: "c".to_owned(),
        };
        let value = serde_json::to_value(&history).unwrap();
        assert!(value.get("entries").is_some());
        assert!(value.get("compressed").is_some());
        assert_eq!(value["entries"][0]["index"], serde_json::json!(1));
        let parsed: OptimizationHistory = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, history);
        assert_eq!(
            serde_json::from_value::<OptimizationHistory>(serde_json::json!({})).unwrap(),
            OptimizationHistory::default()
        );
    }
}
