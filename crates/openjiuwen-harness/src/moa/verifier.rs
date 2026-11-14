//! Progress-verifier seam, structural and model-driven verification, and proposal ranking.
//!
//! Ports the scoring half of `openjiuwen/harness/moa/verifier/` (the LLM-as-a-verifier
//! adapter in `llm_verifier/scorer.py` and the shared `redaction.py`) and the ranking that
//! `openjiuwen/harness/rails/moa_rail.py` builds from a consultation. The Python
//! implementation reads tag logprobs and averages several repeats through a progress
//! tracker; this port keeps the seam and the score range but scores deterministically:
//!
//! - [`StructuralVerifier`] is an offline heuristic over the observed steps, so a host can
//!   run verifier-routed consultation without a verifier model.
//! - [`LlmVerifier`] asks one injected [`Model`] and parses the first numeric token it
//!   returns, clamped to `0.0..=1.0`.
//! - [`score_proposal`] ranks proposals offline; the ranking is structural in every mode,
//!   so verifier-routed and on-demand consultations order proposals identically.
//!
//! Every input is redacted through [`redact`] before analysis, so credentials in tool
//! arguments or results never influence a score and never reach a verifier model.
// The public names (`ProgressVerifier`, `score_progress`, ...) mirror the Python contract
// and intentionally repeat the module name.
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::ModelRequestConfig;
use openjiuwen_foundation::Model;

use super::config::{MoaConfig, VerifierSelection};
use super::prompts::verifier_messages;
use super::redaction::redact;
use super::state::ProposalSource;

pub use crate::moa::state::RankedProposal;

/// Shortest alphanumeric token treated as a problem keyword.
const MIN_KEYWORD_CHARS: usize = 4;
/// Number of leading steps whose per-step bonus is counted before the bonus saturates.
const PROGRESS_MAX_STEPS: usize = 4;
/// Base score of a non-empty step list.
const PROGRESS_BASE: f64 = 0.20;
/// Bonus contributed by one counted step.
const PROGRESS_STEP_BONUS: f64 = 0.06;
/// Bonus when every observed step is structured.
const PROGRESS_STRUCTURE_BONUS: f64 = 0.12;
/// Bonus when every problem keyword is covered by the steps.
const PROGRESS_COVERAGE_BONUS: f64 = 0.28;
/// Penalty applied to each consecutive near-duplicate step.
const PROGRESS_REPETITION_PENALTY: f64 = 0.08;
/// Upper bound of the repetition penalty, so a long stalled run keeps the base score.
const PROGRESS_MAX_REPETITION_PENALTY: f64 = 0.32;

/// Base score of a usable proposal.
const PROPOSAL_BASE: f64 = 0.30;
/// Length bonus of a proposal with at least 80 characters.
const PROPOSAL_SHORT_BONUS: f64 = 0.10;
/// Length bonus of a proposal with at least 300 characters.
const PROPOSAL_LONG_BONUS: f64 = 0.15;
/// Length, in characters, above which the short bonus applies.
const PROPOSAL_SHORT_CHARS: usize = 80;
/// Length, in characters, above which the long bonus applies.
const PROPOSAL_LONG_CHARS: usize = 300;
/// Bonus for a numbered or bulleted proposal.
const PROPOSAL_STRUCTURE_BONUS: f64 = 0.15;
/// Bonus for a balanced fenced code block.
const PROPOSAL_FENCE_BONUS: f64 = 0.10;
/// Bonus when every problem keyword appears in the proposal.
const PROPOSAL_COVERAGE_BONUS: f64 = 0.30;
/// Penalty per repeated hedging phrase beyond the first.
const PROPOSAL_HEDGE_PENALTY: f64 = 0.10;
/// Upper bound of the hedging penalty.
const PROPOSAL_MAX_HEDGE_PENALTY: f64 = 0.20;
/// Minimum token-set overlap at which two steps count as near-identical.
const REPETITION_OVERLAP: f64 = 0.8;

/// Hedging phrases that make a proposal less useful to the acting agent.
const HEDGES: [&str; 4] = ["maybe", "perhaps", "i think", "not sure"];

/// Scores how much progress a list of observed steps represents on a problem.
///
/// The trait is the seam the rail and hosts depend on: [`StructuralVerifier`] runs offline
/// and [`LlmVerifier`] delegates to an injected model. A host can also implement it to plug
/// in its own progress tracker.
#[async_trait]
pub trait ProgressVerifier: Send + Sync {
    /// Returns a progress score in `0.0..=1.0`, where `0.0` means no progress and `1.0`
    /// means the problem looks fully solved.
    ///
    /// `n_evaluations` is the repeat count configured on the verifier; a deterministic
    /// implementation may ignore it, a model-driven one may average repeats. Inputs must be
    /// redacted before they leave the process.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] when a model-backed verifier cannot produce a usable score.
    async fn score_progress(
        &self,
        problem: &str,
        steps: &[String],
        n_evaluations: usize,
    ) -> Result<f32>;
}

/// Deterministic, offline progress verifier.
///
/// The heuristic is documented so callers can reason about transitions, and it never
/// increases when the same steps are appended again:
///
/// - an empty step list scores `0.0`;
/// - otherwise the score starts at `0.20`, gains `0.06` per non-empty step up to four
///   steps, up to `0.12` for structured steps (`Action:`/`Observations:` markers,
///   numbered or bulleted lines), and up to `0.28` for problem keywords (alphanumeric
///   tokens of at least four characters) covered by the steps;
/// - each consecutive near-duplicate step (same normalised text or at least 80% token
///   overlap) costs `0.08`, capped at `0.32`, which offsets the per-step bonus so a
///   stalled agent cannot raise its own score by repeating work;
/// - the result is clamped to `0.0..=1.0` and is never NaN.
#[derive(Debug, Clone, Copy, Default)]
pub struct StructuralVerifier;

impl StructuralVerifier {
    /// Creates the verifier; the type is stateless, so construction cannot fail.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ProgressVerifier for StructuralVerifier {
    async fn score_progress(
        &self,
        problem: &str,
        steps: &[String],
        _n_evaluations: usize,
    ) -> Result<f32> {
        // A deterministic score is identical for every repeat, so `n_evaluations` is
        // accepted for trait parity and deliberately unused.
        Ok(structural_progress_score(problem, steps))
    }
}

/// Model-driven progress verifier.
///
/// Sends [`verifier_messages`] to the injected model with the model's own name and the
/// configured temperature (default `0.0`), then parses the first numeric token of the
/// answer. Python averages `n_evaluations` repeats through a logprob-aware tracker; a
/// caller-injected model already defines its own sampling variance, so this port issues
/// one deterministic request.
#[derive(Clone)]
pub struct LlmVerifier {
    /// Model used for the progress check.
    model: Arc<dyn Model>,
    /// Sampling temperature; `None` requests `0.0` for deterministic answers.
    temperature: Option<f32>,
}

impl LlmVerifier {
    /// Creates a verifier backed by `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            temperature: None,
        }
    }

    /// Returns this verifier with the sampling temperature replaced.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use openjiuwen_harness::moa::verifier::LlmVerifier;
    /// use openjiuwen_test_support::ScriptedModel;
    ///
    /// let verifier = LlmVerifier::new(Arc::new(ScriptedModel::text("0.5"))).with_temperature(0.2);
    /// assert_eq!(verifier.model().model_name(), "scripted-model");
    /// ```
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// The injected model.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

#[async_trait]
impl ProgressVerifier for LlmVerifier {
    async fn score_progress(
        &self,
        problem: &str,
        steps: &[String],
        _n_evaluations: usize,
    ) -> Result<f32> {
        let problem = redact(problem);
        let redacted: Vec<String> = steps.iter().map(|step| redact(step)).collect();
        let messages = verifier_messages(&problem, &redacted);
        let request = ModelRequestConfig::new(self.model.model_name())
            .with_temperature(self.temperature.unwrap_or(0.0));
        tracing::debug!(
            model = self.model.model_name(),
            steps = redacted.len(),
            "requesting a MoA progress score"
        );
        let response = self
            .model
            .invoke(&messages, &request)
            .await
            .map_err(|error| Error::llm(format!("MoA verifier call failed: {error}")))?;
        let text = response
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        parse_score(&text).ok_or_else(|| Error::llm("verifier returned no numeric score"))
    }
}

/// Structural score of one proposal, with human-readable reasons.
///
/// The text is redacted before analysis. The score starts at `0.30` and adds `0.10` for
/// at least 80 characters (or `0.15` for at least 300), `0.15` for a numbered or bulleted
/// structure, `0.10` for a balanced fenced code block, and up to `0.30` for problem-keyword
/// coverage. Repeated hedging (`maybe`, `perhaps`, `I think`, `not sure`) costs `0.10` per
/// occurrence beyond the first, capped at `0.20`. The result is clamped to `0.0..=1.0`;
/// blank proposals score `0.0`.
#[must_use]
pub fn score_proposal(problem: &str, source: &ProposalSource) -> RankedProposal {
    let problem = redact(problem);
    let text = redact(&source.text);
    if text.trim().is_empty() {
        return RankedProposal {
            index: source.index,
            score: 0.0,
            reasons: vec!["empty proposal".to_owned()],
        };
    }

    let mut score = PROPOSAL_BASE;
    // Reasons are collected with their contribution and sorted so the strongest factor is
    // the first thing a reader (or a log line) sees.
    let mut factors: Vec<(f64, String)> = Vec::new();

    let chars = text.chars().count();
    if chars >= PROPOSAL_LONG_CHARS {
        score += PROPOSAL_LONG_BONUS;
        factors.push((
            PROPOSAL_LONG_BONUS,
            format!("detailed proposal ({chars} characters)"),
        ));
    } else if chars >= PROPOSAL_SHORT_CHARS {
        score += PROPOSAL_SHORT_BONUS;
        factors.push((
            PROPOSAL_SHORT_BONUS,
            format!("substantive proposal ({chars} characters)"),
        ));
    }

    if looks_structured(&text) {
        score += PROPOSAL_STRUCTURE_BONUS;
        factors.push((
            PROPOSAL_STRUCTURE_BONUS,
            "uses a numbered or bulleted structure".to_owned(),
        ));
    }

    let fences = text.matches("```").count();
    if fences >= 2 && fences % 2 == 0 {
        score += PROPOSAL_FENCE_BONUS;
        factors.push((
            PROPOSAL_FENCE_BONUS,
            "contains a balanced fenced code block".to_owned(),
        ));
    }

    let (found, total) = keyword_coverage(&problem, &text);
    if found > 0 && total > 0 {
        let bonus = PROPOSAL_COVERAGE_BONUS * ratio(found, total);
        score += bonus;
        factors.push((bonus, format!("covers {found}/{total} problem keywords")));
    }

    let hedges = count_hedges(&text);
    if hedges > 1 {
        let penalty =
            (PROPOSAL_HEDGE_PENALTY * count_to_f64(hedges - 1)).min(PROPOSAL_MAX_HEDGE_PENALTY);
        score -= penalty;
        factors.push((
            -penalty,
            format!("hedging language repeated {hedges} times"),
        ));
    }

    factors.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut reasons: Vec<String> = factors.into_iter().map(|(_, reason)| reason).collect();
    if reasons.is_empty() {
        reasons.push("baseline score for a usable proposal with no notable signals".to_owned());
    }
    RankedProposal {
        index: source.index,
        score: clamp_score(score),
        reasons,
    }
}

/// Ranks `sources` by structural score, highest first, then by index ascending.
#[must_use]
pub fn rank_proposals(problem: &str, sources: &[ProposalSource]) -> Vec<RankedProposal> {
    let mut ranked: Vec<RankedProposal> = sources
        .iter()
        .map(|source| score_proposal(problem, source))
        .collect();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.index.cmp(&right.index))
    });
    ranked
}

/// Scores progress using the verifier selected by `config`.
///
/// [`VerifierSelection::Structural`] uses [`StructuralVerifier`];
/// [`VerifierSelection::Llm`] requires [`MoaConfig::verifier_model`] and uses an
/// [`LlmVerifier`] configured with `config.verifier.n_evaluations`.
///
/// # Errors
///
/// Returns [`Error::Config`] with the Python message when the LLM selection has no
/// `verifier_model`, and [`Error::Llm`] when the injected model fails or answers without a
/// numeric score.
pub async fn score_progress(config: &MoaConfig, problem: &str, steps: &[String]) -> Result<f32> {
    match config.verifier_selection {
        VerifierSelection::Structural => {
            StructuralVerifier::new()
                .score_progress(problem, steps, config.verifier.n_evaluations)
                .await
        }
        VerifierSelection::Llm => {
            let model = config
                .verifier_model
                .as_ref()
                .ok_or_else(|| Error::config("verifier_routed requires a verifier model"))?;
            LlmVerifier::new(Arc::clone(model))
                .score_progress(problem, steps, config.verifier.n_evaluations)
                .await
        }
    }
}

/// Ranks proposals under `config`.
///
/// Proposal ranking is deterministic and structural in every mode, so
/// [`VerifierSelection`] only changes how progress is scored. The function stays fallible
/// (and async) for the configuration seam and for a future model-driven ranking backend.
///
/// # Errors
///
/// Reserved for a future LLM-backed ranking; the structural ranking cannot fail today.
#[allow(clippy::unused_async)] // kept async for the future model-driven ranking backend
pub async fn rank_with_config(
    config: &MoaConfig,
    problem: &str,
    sources: &[ProposalSource],
) -> Result<Vec<RankedProposal>> {
    tracing::trace!(
        selection = config.verifier_selection.as_str(),
        sources = sources.len(),
        "ranking MoA proposals structurally"
    );
    Ok(rank_proposals(problem, sources))
}

/// Computes the deterministic structural progress score of `steps` on `problem`.
fn structural_progress_score(problem: &str, steps: &[String]) -> f32 {
    let problem = redact(problem);
    let redacted: Vec<String> = steps.iter().map(|step| redact(step)).collect();
    if redacted.is_empty() {
        return 0.0;
    }
    let observed: Vec<&str> = redacted
        .iter()
        .map(String::as_str)
        .filter(|step| !step.trim().is_empty())
        .collect();

    let mut score = PROGRESS_BASE;
    score += PROGRESS_STEP_BONUS * count_to_f64(observed.len().min(PROGRESS_MAX_STEPS));

    if !observed.is_empty() {
        let structured = observed
            .iter()
            .filter(|step| looks_structured(step))
            .count();
        score += PROGRESS_STRUCTURE_BONUS * ratio(structured, observed.len());
    }

    let joined = observed.join("\n");
    let (found, total) = keyword_coverage(&problem, &joined);
    if found > 0 && total > 0 {
        score += PROGRESS_COVERAGE_BONUS * ratio(found, total);
    }

    let repeats = observed
        .windows(2)
        .filter(|pair| is_repetition(pair[0], pair[1]))
        .count();
    score -=
        (PROGRESS_REPETITION_PENALTY * count_to_f64(repeats)).min(PROGRESS_MAX_REPETITION_PENALTY);

    clamp_score(score)
}

/// Whether `text` reads as a structured step or proposal.
///
/// A step is structured when any line is a bullet (`-`, `*`, `+` followed by whitespace),
/// a numbered item (`1.`, `1)`), or when the text carries the rail's
/// `Action:`/`Observations:` markers.
fn looks_structured(text: &str) -> bool {
    let lower = text.to_lowercase();
    if lower.contains("action:") || lower.contains("observations:") {
        return true;
    }
    text.lines().any(|line| {
        let line = line.trim_start();
        is_bullet(line) || starts_with_number(line)
    })
}

/// Whether `line` starts with a bullet marker followed by whitespace.
fn is_bullet(line: &str) -> bool {
    let mut chars = line.chars();
    match (chars.next(), chars.next()) {
        (Some(marker), Some(next)) => matches!(marker, '-' | '*' | '+') && next.is_whitespace(),
        _ => false,
    }
}

/// Whether `line` starts with `N.` or `N)`.
fn starts_with_number(line: &str) -> bool {
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    matches!(line.as_bytes().get(digits), Some(b'.' | b')'))
}

/// Unique lower-cased alphanumeric problem tokens of at least [`MIN_KEYWORD_CHARS`].
fn keywords(text: &str) -> Vec<String> {
    let mut words: Vec<String> = text
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.chars().count() >= MIN_KEYWORD_CHARS)
        .map(str::to_lowercase)
        .collect();
    words.sort();
    words.dedup();
    words
}

/// How many of the problem's keywords occur in `haystack`, and how many exist.
fn keyword_coverage(problem: &str, haystack: &str) -> (usize, usize) {
    let words = keywords(problem);
    let haystack = haystack.to_lowercase();
    let found = words
        .iter()
        .filter(|word| haystack.contains(word.as_str()))
        .count();
    (found, words.len())
}

/// Number of hedging phrases in `text`, counted case-insensitively.
fn count_hedges(text: &str) -> usize {
    let lower = text.to_lowercase();
    HEDGES
        .iter()
        .map(|phrase| lower.matches(phrase).count())
        .sum()
}

/// Normalises whitespace and case so two spellings of the same step compare equal.
fn normalise(text: &str) -> String {
    text.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether `right` repeats `left`, either verbatim or with at least
/// [`REPETITION_OVERLAP`] of the same tokens.
fn is_repetition(left: &str, right: &str) -> bool {
    let left = normalise(left);
    let right = normalise(right);
    if left.is_empty() || right.is_empty() {
        return left.is_empty() && right.is_empty();
    }
    if left == right {
        return true;
    }
    let left_tokens: BTreeSet<&str> = left.split_whitespace().collect();
    let right_tokens: BTreeSet<&str> = right.split_whitespace().collect();
    let shared = left_tokens.intersection(&right_tokens).count();
    let union = left_tokens.union(&right_tokens).count();
    ratio(shared, union) >= REPETITION_OVERLAP
}

/// Parses the first numeric token of `text`, clamped to `0.0..=1.0`.
///
/// A token is a maximal run of ASCII digits, dots, signs and exponent markers, so
/// `"0.75"`, `"Score: 0.75."` and `"1e-3"` all parse. Non-finite values and tokens without
/// a number are skipped; text with no number at all yields `None`.
fn parse_score(text: &str) -> Option<f32> {
    text.split(|character: char| {
        !(character.is_ascii_digit() || matches!(character, '.' | '-' | '+' | 'e' | 'E'))
    })
    .filter(|token| !token.is_empty())
    .find_map(|token| {
        let token = token.trim_matches('+').trim_end_matches('.');
        let value = token.parse::<f32>().ok()?;
        value.is_finite().then(|| value.clamp(0.0, 1.0))
    })
}

/// Converts a small count to `f64` for the heuristics.
#[allow(clippy::cast_precision_loss)] // counts are step totals, far below 2^53
fn count_to_f64(count: usize) -> f64 {
    count as f64
}

/// `part / whole` as `f64`, or `0.0` when `whole` is zero.
#[allow(clippy::cast_precision_loss)] // counts are step totals, far below 2^53
fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// Clamps a heuristic score into `0.0..=1.0`, mapping non-finite values to `0.0`.
#[allow(clippy::cast_possible_truncation)] // the clamp guarantees a representable value
fn clamp_score(score: f64) -> f32 {
    if score.is_finite() {
        score.clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::ScriptedModel;

    /// Builds a proposal source for the ranking tests.
    fn source(index: usize, text: &str) -> ProposalSource {
        ProposalSource::new(index, text)
    }

    /// Builds owned steps from literals.
    fn steps(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[tokio::test]
    async fn structural_scores_are_finite_and_bounded() {
        let verifier = StructuralVerifier::new();
        let cases = [
            steps(&[]),
            steps(&["inspected src/main.rs"]),
            steps(&[
                "Action: inspect src/auth/session.rs",
                "Observations: the token check returns early",
                "1. reproduce the failing auth test",
                "2. fix the root cause and rerun the suite",
                "",
                "done",
            ]),
        ];
        for case in &cases {
            let score = verifier
                .score_progress("fix the failing auth test", case, 3)
                .await
                .unwrap();
            assert!(score.is_finite(), "score must be finite: {score}");
            assert!((0.0..=1.0).contains(&score), "score out of range: {score}");
        }
        assert_eq!(cases[0].len(), 0);
        let empty = verifier
            .score_progress("fix the failing auth test", &cases[0], 3)
            .await
            .unwrap();
        assert!(empty.abs() < f32::EPSILON, "empty steps score {empty}");
    }

    #[tokio::test]
    async fn repeated_steps_do_not_raise_the_score() {
        let verifier = StructuralVerifier::new();
        let one = verifier
            .score_progress(
                "fix the failing auth test",
                &steps(&["inspect session.rs"]),
                3,
            )
            .await
            .unwrap();
        let repeated = steps(&["inspect session.rs"; 4]);
        let many = verifier
            .score_progress("fix the failing auth test", &repeated, 3)
            .await
            .unwrap();
        assert!(
            many <= one,
            "stalled steps must not raise the score: {one} -> {many}"
        );

        // Whitespace and case differences are near-identical, not progress.
        let near = steps(&["inspect session.rs", "  Inspect   session.rs  "]);
        let near_score = verifier
            .score_progress("fix the failing auth test", &near, 3)
            .await
            .unwrap();
        assert!(
            near_score <= one,
            "near-duplicate steps must not raise the score: {one} -> {near_score}"
        );
    }

    #[test]
    fn structured_proposals_outrank_thin_hedging_ones() {
        let problem = "Fix the failing auth test in src/auth/session.rs and explain the root cause";
        let structured = source(
            1,
            "Plan:\n\n\
             1. Reproduce the failing auth test in src/auth/session.rs.\n\
             2. Trace the root cause in the session module.\n\
             3. Fix the root cause and rerun the auth test.\n\
             4. Explain the fix and the root cause in the final answer.\n\n\
             ```rust\nlet verified = true;\n```",
        );
        let thin = source(2, "Maybe it is fine, I think. Perhaps the test is flaky.");
        let thin_ranked = score_proposal(problem, &thin);
        let structured_ranked = score_proposal(problem, &structured);
        assert!(
            structured_ranked.score > thin_ranked.score,
            "structured {} must beat thin {}",
            structured_ranked.score,
            thin_ranked.score
        );
        assert!(structured_ranked.score > 0.5);
        assert!(thin_ranked.score < structured_ranked.score);
        assert!(structured_ranked
            .reasons
            .iter()
            .any(|reason| reason.contains("structure") || reason.contains("keyword")));
    }

    #[test]
    fn ranking_is_by_score_then_index() {
        let problem = "Fix the failing auth test";
        let good = source(
            2,
            "1. Fix the failing auth test.\n2. Rerun the suite and explain.",
        );
        let bad = source(1, "not sure");
        let ranking = rank_proposals(problem, &[bad, good]);
        assert_eq!(ranking.len(), 2);
        assert_eq!(ranking[0].index, 2);
        assert_eq!(ranking[1].index, 1);
        assert!(ranking[0].score > ranking[1].score);

        // Equal scores fall back to ascending index.
        let left = source(3, "same text");
        let right = source(1, "same text");
        let tie = rank_proposals(problem, &[left, right]);
        assert_eq!(tie[0].index, 1);
        assert_eq!(tie[1].index, 3);
        assert!((tie[0].score - tie[1].score).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn llm_verifier_parses_a_scripted_score() {
        let model = Arc::new(ScriptedModel::text("0.75"));
        let as_model: Arc<dyn Model> = model.clone();
        let verifier = LlmVerifier::new(Arc::clone(&as_model)).with_temperature(0.0);
        assert!(Arc::ptr_eq(verifier.model(), &as_model));
        let score = verifier
            .score_progress("fix the test", &steps(&["ran the suite"]), 1)
            .await
            .unwrap();
        assert!((score - 0.75).abs() < f32::EPSILON, "parsed {score}");
        let call = &model.calls()[0];
        assert!(call.request.temperature.unwrap_or_default().abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn llm_verifier_rejects_output_without_a_number() {
        let verifier = LlmVerifier::new(Arc::new(ScriptedModel::text("no score available")));
        let error = verifier
            .score_progress("fix the test", &steps(&["ran the suite"]), 1)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error
            .to_string()
            .contains("verifier returned no numeric score"));
    }

    #[tokio::test]
    async fn config_dispatch_selects_the_verifier() {
        let config = MoaConfig::new(vec![Arc::new(ScriptedModel::text("reference"))]);
        let score = score_progress(&config, "fix the failing auth test", &steps(&["inspect"]))
            .await
            .unwrap();
        assert!((0.0..=1.0).contains(&score));

        let misconfigured = config.with_verifier_selection(VerifierSelection::Llm);
        let error = score_progress(&misconfigured, "fix the test", &steps(&["inspect"]))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error
            .to_string()
            .contains("verifier_routed requires a verifier model"));

        let configured = misconfigured.with_verifier_model(Arc::new(ScriptedModel::text("0.4")));
        let score = score_progress(&configured, "fix the test", &steps(&["inspect"]))
            .await
            .unwrap();
        assert!(
            (score - 0.4).abs() < f32::EPSILON,
            "dispatch scored {score}"
        );
    }

    #[test]
    fn redaction_keeps_secrets_out_of_scores() {
        let problem = "Rotate the api_key: sk-secret and rerun the build";
        let proposal = source(
            7,
            "Set api_key: sk-secret before running the build.\n\n1. Rotate the key.\n2. Rerun the build.",
        );
        let ranked = score_proposal(problem, &proposal);
        assert_eq!(ranked.index, 7);
        assert!((0.0..=1.0).contains(&ranked.score));
        assert!(ranked.reasons.iter().any(|reason| !reason.is_empty()));
        for reason in &ranked.reasons {
            assert!(
                !reason.contains("sk-secret"),
                "reason leaked a secret: {reason}"
            );
        }
        assert!(!redact(&proposal.text).contains("sk-secret"));
    }

    #[tokio::test]
    async fn llm_verifier_redacts_problem_and_steps() {
        let model = Arc::new(ScriptedModel::text("0.5"));
        let verifier = LlmVerifier::new(model.clone());
        verifier
            .score_progress(
                "use api_key: sk-secret to call the service",
                &steps(&["token: sk-other was accepted"]),
                1,
            )
            .await
            .unwrap();
        let rendered: String = model.calls()[0]
            .messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!rendered.contains("sk-secret"));
        assert!(!rendered.contains("sk-other"));
    }

    #[test]
    fn parse_score_takes_the_first_numeric_token() {
        assert_eq!(parse_score("0.75"), Some(0.75));
        assert_eq!(parse_score("Score: 0.75."), Some(0.75));
        assert_eq!(parse_score("score = 1.5"), Some(1.0));
        assert_eq!(parse_score("score = -0.5"), Some(0.0));
        assert_eq!(parse_score("no number"), None);
        assert_eq!(parse_score(""), None);
        assert_eq!(parse_score("."), None);
    }
}
