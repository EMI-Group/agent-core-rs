//! The deterministic evaluator: score a candidate prompt from prompt-internal
//! signals, with no model and no answer.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::tune::base::{Case, EvaluatedCase};

use super::runner::render_value;
use super::scorer::tokens;
use super::CaseEvaluator;

/// Weight of the placeholder-preservation dimension.
const PLACEHOLDER_WEIGHT: f64 = 0.35;
/// Weight of the instruction-structure dimension.
const STRUCTURE_WEIGHT: f64 = 0.20;
/// Weight of the case-term coverage dimension.
const COVERAGE_WEIGHT: f64 = 0.30;
/// Weight of the length/duplication hygiene dimension.
const HYGIENE_WEIGHT: f64 = 0.15;

/// The per-dimension breakdown of a [`DeterministicEvaluator`] score.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DeterministicScore {
    /// Weighted total in `0.0..=1.0`.
    pub total: f64,
    /// Fraction of the seed prompt's `{{placeholder}}`s still present.
    pub placeholder: f64,
    /// Fraction of the instruction-structure cue categories present.
    pub structure: f64,
    /// Fraction of the case's label/input terms present in the candidate.
    pub coverage: f64,
    /// Length and duplication hygiene in `0.0..=1.0`.
    pub hygiene: f64,
}

impl Default for DeterministicScore {
    fn default() -> Self {
        Self {
            total: 0.0,
            placeholder: 0.0,
            structure: 0.0,
            coverage: 0.0,
            hygiene: 0.0,
        }
    }
}

impl DeterministicScore {
    /// A one-line explanation of the dimensions that make up `total`.
    #[must_use]
    pub fn reason(&self) -> String {
        format!(
            "deterministic score {:.3} (placeholder {:.2}, structure {:.2}, coverage {:.2}, \
             hygiene {:.2})",
            self.total, self.placeholder, self.structure, self.coverage, self.hygiene
        )
    }
}

/// Instruction-structure cue categories, one per dimension of the structure
/// score.
const STRUCTURE_CUES: &[&[&str]] = &[
    // An imperative task statement.
    &[
        "you are",
        "your task",
        "answer",
        "respond",
        "provide",
        "explain",
        "write",
        "solve",
        "think",
        "verify",
        "keep",
        "ask",
    ],
    // An explicit multi-step procedure.
    &[
        "step by step",
        "first",
        "second",
        "next",
        "then",
        "finally",
        "1.",
    ],
    // An output-format instruction.
    &[
        "output format",
        "format",
        "json",
        "markdown",
        "final answer",
        "respond with",
        "```",
    ],
    // A constraint or refusal.
    &[
        "must",
        "never",
        "do not",
        "don't",
        "only",
        "avoid",
        "instead of",
        "constraint",
    ],
];

/// An offline [`CaseEvaluator`] with no model: deterministic signals only.
///
/// Constructed with the seed prompt so the search landscape rewards candidates
/// that keep the seed's contract:
///
/// - **placeholder** (`0.35`) — the fraction of the seed's `{{placeholder}}`s
///   that survive; `1.0` when the seed has none.
/// - **structure** (`0.20`) — the fraction of the four instruction cue
///   categories (imperative, steps, output format, constraints) that appear.
/// - **coverage** (`0.30`) — the fraction of the case's label-value and
///   input-value terms that appear in the candidate, case-insensitively;
///   `1.0` when the case yields no terms.
/// - **hygiene** (`0.15`) — penalizes degenerate candidates: shorter than 20
///   characters (`×0.5`), longer than 8000 (`×0.7`), more than 30% duplicate
///   non-blank lines (`×(1 − duplicate ratio)`), and no alphabetic character
///   at all (`×0.2`).
///
/// A blank candidate scores `0.0`. The score is a pure function of the seed,
/// the candidate, and the case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicEvaluator {
    /// The prompt every candidate is compared against.
    seed: String,
    /// The `{{placeholder}}` names of the seed.
    seed_placeholders: BTreeSet<String>,
}

impl DeterministicEvaluator {
    /// Creates an evaluator whose reference prompt is `seed_prompt`.
    #[must_use]
    pub fn new(seed_prompt: impl Into<String>) -> Self {
        let seed = seed_prompt.into();
        let seed_placeholders = crate::tune::optimizer::placeholders(&seed);
        Self {
            seed,
            seed_placeholders,
        }
    }

    /// The reference prompt.
    #[must_use]
    pub fn seed(&self) -> &str {
        &self.seed
    }

    /// The `{{placeholder}}` names of the reference prompt.
    #[must_use]
    pub fn seed_placeholders(&self) -> &BTreeSet<String> {
        &self.seed_placeholders
    }

    /// Scores `candidate` for `case`, returning the dimension breakdown.
    #[must_use]
    pub fn score_parts(&self, candidate: &str, case: &Case) -> DeterministicScore {
        if candidate.trim().is_empty() {
            return DeterministicScore::default();
        }
        let placeholder = self.placeholder_score(candidate);
        let structure = structure_score(candidate);
        let coverage = coverage_score(candidate, case);
        let hygiene = hygiene_score(candidate);
        let total = PLACEHOLDER_WEIGHT * placeholder
            + STRUCTURE_WEIGHT * structure
            + COVERAGE_WEIGHT * coverage
            + HYGIENE_WEIGHT * hygiene;
        DeterministicScore {
            total: total.clamp(0.0, 1.0),
            placeholder,
            structure,
            coverage,
            hygiene,
        }
    }

    /// The fraction of the seed's placeholders that survive in `candidate`.
    fn placeholder_score(&self, candidate: &str) -> f64 {
        if self.seed_placeholders.is_empty() {
            return 1.0;
        }
        let present = crate::tune::optimizer::placeholders(candidate);
        let kept = self
            .seed_placeholders
            .iter()
            .filter(|name| present.contains(*name))
            .count();
        kept as f64 / self.seed_placeholders.len() as f64
    }
}

/// The fraction of the instruction cue categories present in `candidate`.
fn structure_score(candidate: &str) -> f64 {
    let lower = candidate.to_lowercase();
    let matched = STRUCTURE_CUES
        .iter()
        .filter(|cues| cues.iter().any(|cue| lower.contains(cue)))
        .count();
    matched as f64 / STRUCTURE_CUES.len() as f64
}

/// The fraction of the case's label and input terms present in `candidate`.
fn coverage_score(candidate: &str, case: &Case) -> f64 {
    let lower = candidate.to_lowercase();
    let mut terms = BTreeSet::new();
    for value in case.label.values().chain(case.inputs.values()) {
        terms.extend(tokens(&render_value(value)));
    }
    if terms.is_empty() {
        return 1.0;
    }
    let present = terms.iter().filter(|term| lower.contains(*term)).count();
    present as f64 / terms.len() as f64
}

/// Penalizes degenerate candidates: too short, too long, repetitive, or not
/// textual at all.
fn hygiene_score(candidate: &str) -> f64 {
    let trimmed = candidate.trim();
    if trimmed.is_empty() {
        return 0.0;
    }
    let mut score = 1.0;
    let chars = trimmed.chars().count();
    if chars < 20 {
        score *= 0.5;
    }
    if chars > 8000 {
        score *= 0.7;
    }
    let lines: Vec<&str> = trimmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if !lines.is_empty() {
        let unique = lines.iter().collect::<BTreeSet<_>>().len();
        let duplicate_ratio = 1.0 - unique as f64 / lines.len() as f64;
        if duplicate_ratio > 0.3 {
            score *= 1.0 - duplicate_ratio;
        }
    }
    if !trimmed.chars().any(char::is_alphabetic) {
        score *= 0.2;
    }
    score.clamp(0.0, 1.0)
}

#[async_trait]
impl CaseEvaluator for DeterministicEvaluator {
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        let parts = self.score_parts(prompt, case);
        EvaluatedCase::new(case.clone(), BTreeMap::new(), parts.total, parts.reason())
    }
}
