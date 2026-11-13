//! The evaluator seam: run a candidate prompt, score the answer, keep the case.
//!
//! Ports the role of Python `openjiuwen/dev_tools/tune/evaluator/evaluator.py`
//! but splits it into three small seams so every pipeline can be driven offline:
//!
//! - [`PromptRunner`] turns `(candidate prompt, case)` into an answer map.
//!   [`ModelPromptRunner`] calls a [`Model`] with the candidate as the system
//!   message and the rendered case inputs as the user message;
//!   [`ScriptedPromptRunner`] replays canned answers for offline tests.
//! - [`CaseScorer`] turns `(case, answer)` into a score plus a reason.
//!   [`LabelScorer`] is the deterministic default: it compares the answer with
//!   the case label key by key.
//! - [`CaseEvaluator`] composes the two into an [`EvaluatedCase`].
//!   [`RunnerEvaluator`] evaluates through any [`PromptRunner`],
//!   [`ModelEvaluator`] is the model-backed preset, and
//!   [`DeterministicEvaluator`] evaluates without any model so prompt search has
//!   a real, repeatable landscape to climb.
//!
//! Python's `DefaultEvaluator` asks a model for a `{"result": bool, "reason": …}`
//! verdict and gives everything the same all-or-nothing score. The port keeps
//! the seams and the `0.0..=1.0` range but makes the default scorer
//! deterministic and graded, so a search can tell a near miss from a wrong
//! answer. A model can still judge cases: [`ModelEvaluator`] plus a custom
//! [`CaseScorer`] (or [`RunnerEvaluator`] with a model-runner and a custom
//! scorer) is the equivalent hook.
//!
//! # Scoring bands of [`LabelScorer`]
//!
//! For every key of the case label the answer value is compared with the
//! expected value, after trimming:
//!
//! | Condition                              | Score |
//! | -------------------------------------- | ----- |
//! | exact match                            | 1.00  |
//! | case-insensitive match                 | 0.90  |
//! | either value contains the other        | 0.75  |
//! | token-set Jaccard in `0.5..=1.0`       | 0.30 + 0.30·(J−0.5)/0.5 |
//! | otherwise                              | 0.00  |
//!
//! The case score is the mean over the label keys. When the answer carries
//! exactly one entry (for example the `output` key a plain-text model answer
//! produces) that entry stands in for any missing label key, so a model that
//! answers in prose is still scored instead of always getting `0.0`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tune::base::{Case, EvaluatedCase};
use crate::util::extract_json_fence;

/// Renders one case's inputs as the user-facing text of a prompt.
///
/// One `[key]: value` line per input, in the input map's deterministic key
/// order. String values are rendered bare; every other JSON value is rendered
/// compactly, so a number stays `4` and an object stays `{"a":1}`.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use openjiuwen_dev_tools::tune::{render_case_inputs, Case};
/// # use serde_json::Value;
/// let mut inputs = BTreeMap::new();
/// inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
/// let mut label = BTreeMap::new();
/// label.insert("answer".to_owned(), Value::String("4".to_owned()));
/// let case = Case::new(inputs, label)?;
/// assert_eq!(render_case_inputs(&case), "[question]: 2 + 2?");
/// # Ok::<(), openjiuwen_common::error::Error>(())
/// ```
#[must_use]
pub fn render_case_inputs(case: &Case) -> String {
    case.inputs
        .iter()
        .map(|(key, value)| format!("[{key}]: {}", render_value(value)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders a JSON value as plain text: strings unquoted, everything else as
/// compact JSON.
fn render_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Turns a raw model answer into an answer map.
///
/// A fenced JSON object or a whole-answer JSON object is used as the map; any
/// other answer becomes `{"output": <trimmed text>}`.
fn answer_to_map(answer: &str) -> BTreeMap<String, Value> {
    if let Some(Value::Object(object)) = extract_json_fence(answer) {
        return object.into_iter().collect();
    }
    if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(answer.trim()) {
        return object.into_iter().collect();
    }
    let mut map = BTreeMap::new();
    map.insert("output".to_owned(), Value::String(answer.trim().to_owned()));
    map
}

/// Runs a candidate prompt over one case and returns the answer map.
#[async_trait]
pub trait PromptRunner: Send + Sync {
    /// Runs `prompt` for `case`.
    ///
    /// # Errors
    ///
    /// Implementation-defined: a model runner propagates the provider error,
    /// a scripted runner reports an unknown case.
    async fn run(&self, prompt: &str, case: &Case) -> Result<BTreeMap<String, Value>>;
}

/// A [`PromptRunner`] that calls a [`Model`] once per case.
///
/// The candidate prompt is the system message and the rendered case inputs
/// ([`render_case_inputs`]) the user message. The answer becomes an answer map:
/// a fenced or whole-answer JSON object is used as-is, any other answer becomes
/// `{"output": <trimmed text>}`.
pub struct ModelPromptRunner {
    /// The model consulted for every case.
    model: Arc<dyn Model>,
}

impl ModelPromptRunner {
    /// Creates a runner over `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }

    /// The model this runner consults.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

impl fmt::Debug for ModelPromptRunner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelPromptRunner")
            .field("provider", &self.model.provider())
            .field("model_name", &self.model.model_name())
            .finish()
    }
}

#[async_trait]
impl PromptRunner for ModelPromptRunner {
    async fn run(&self, prompt: &str, case: &Case) -> Result<BTreeMap<String, Value>> {
        let messages = [
            BaseMessage::system(prompt.to_owned()),
            BaseMessage::user(render_case_inputs(case)),
        ];
        let request = ModelRequestConfig::new(self.model.model_name());
        let completion = self.model.invoke(&messages, &request).await?;
        if completion.choices.is_empty() {
            return Err(Error::llm(
                "prompt runner model returned no choices; expected an answer message",
            ));
        }
        let answer = completion
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        Ok(answer_to_map(&answer))
    }
}

/// Locks `mutex`, recovering the guard when a previous holder panicked.
///
/// A poisoned lock only means some other caller panicked; the recorded calls are
/// still consistent, so recovering keeps this helper panic-free.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One recorded [`ScriptedPromptRunner`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PromptCall {
    /// The case the call evaluated.
    pub case_id: String,
    /// The candidate prompt that was evaluated.
    pub prompt: String,
}

/// A [`PromptRunner`] that replays canned answers keyed by case id.
///
/// Unknown case ids fall back to [`with_fallback`](Self::with_fallback) when it
/// is set and report [`Error::not_found`] otherwise. Every call is recorded, so
/// a test can assert which prompts reached the runner and in which order.
#[derive(Debug, Default)]
pub struct ScriptedPromptRunner {
    /// Answers keyed by case id.
    answers: BTreeMap<String, BTreeMap<String, Value>>,
    /// Answer used for an unscripted case, when any.
    fallback: Option<BTreeMap<String, Value>>,
    /// Calls in call order.
    calls: Mutex<Vec<PromptCall>>,
}

impl ScriptedPromptRunner {
    /// Creates a runner without answers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Answers `case_id` with `answer`.
    #[must_use]
    pub fn with_answer(
        mut self,
        case_id: impl Into<String>,
        answer: BTreeMap<String, Value>,
    ) -> Self {
        self.answers.insert(case_id.into(), answer);
        self
    }

    /// Answers `case_id` with a single `output` entry holding `text`.
    #[must_use]
    pub fn with_text_answer(self, case_id: impl Into<String>, text: impl Into<String>) -> Self {
        let mut answer = BTreeMap::new();
        answer.insert("output".to_owned(), Value::String(text.into()));
        self.with_answer(case_id, answer)
    }

    /// Answers every unscripted case with `answer`.
    #[must_use]
    pub fn with_fallback(mut self, answer: BTreeMap<String, Value>) -> Self {
        self.fallback = Some(answer);
        self
    }

    /// The calls recorded so far, oldest first.
    ///
    /// # Panics
    ///
    /// Panics only when the internal lock was poisoned by a panicking caller.
    #[must_use]
    pub fn calls(&self) -> Vec<PromptCall> {
        lock(&self.calls).clone()
    }

    /// How many calls were recorded.
    #[must_use]
    pub fn call_count(&self) -> usize {
        lock(&self.calls).len()
    }
}

#[async_trait]
impl PromptRunner for ScriptedPromptRunner {
    async fn run(&self, prompt: &str, case: &Case) -> Result<BTreeMap<String, Value>> {
        lock(&self.calls).push(PromptCall {
            case_id: case.case_id.clone(),
            prompt: prompt.to_owned(),
        });
        if let Some(answer) = self.answers.get(&case.case_id) {
            return Ok(answer.clone());
        }
        self.fallback.clone().ok_or_else(|| {
            Error::not_found(format!(
                "scripted prompt runner has no answer for case {:?}",
                case.case_id
            ))
        })
    }
}

/// Scores one answer against its case.
pub trait CaseScorer: Send + Sync {
    /// Returns the score in `0.0..=1.0` and the reason for it.
    fn score(&self, case: &Case, answer: &BTreeMap<String, Value>) -> (f64, String);
}

/// The deterministic default [`CaseScorer`]: compare answer and label by key.
///
/// See the module documentation for the scoring bands. The reason names the
/// first label key that is not an exact match, or states that every key matched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LabelScorer;

impl LabelScorer {
    /// Creates the scorer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl CaseScorer for LabelScorer {
    fn score(&self, case: &Case, answer: &BTreeMap<String, Value>) -> (f64, String) {
        if case.label.is_empty() {
            return (0.0, "case label has no keys to score".to_owned());
        }
        let mut total = 0.0;
        let mut first_mismatch: Option<String> = None;
        for (key, expected) in &case.label {
            let expected_text = render_value(expected);
            let actual_text = answer_text(answer, key);
            let (score, reason) = match &actual_text {
                Some(actual) => {
                    let score = score_pair(&expected_text, actual);
                    let reason = if score >= 1.0 {
                        String::new()
                    } else {
                        format!("label key `{key}` scored {score:.2}: expected {expected_text:?}, got {actual:?}")
                    };
                    (score, reason)
                }
                None => (0.0, format!("label key `{key}` is missing from the answer")),
            };
            total += score;
            if score < 1.0 && first_mismatch.is_none() {
                first_mismatch = Some(reason);
            }
        }
        let mean = total / case.label.len() as f64;
        let reason = first_mismatch
            .unwrap_or_else(|| format!("all {} label key(s) matched exactly", case.label.len()));
        (mean.clamp(0.0, 1.0), reason)
    }
}

/// The text a label key is compared with: the answer entry, or the sole answer
/// entry when the key is absent.
fn answer_text(answer: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    if let Some(value) = answer.get(key) {
        return Some(render_value(value));
    }
    if answer.len() == 1 {
        return answer.values().next().map(render_value);
    }
    None
}

/// Scores one expected/actual string pair with the bands of [`LabelScorer`].
fn score_pair(expected: &str, actual: &str) -> f64 {
    let expected = expected.trim();
    let actual = actual.trim();
    if expected == actual {
        return 1.0;
    }
    let expected_lower = expected.to_lowercase();
    let actual_lower = actual.to_lowercase();
    if expected_lower == actual_lower {
        return 0.9;
    }
    if !expected_lower.is_empty()
        && !actual_lower.is_empty()
        && (actual_lower.contains(&expected_lower) || expected_lower.contains(&actual_lower))
    {
        return 0.75;
    }
    let expected_tokens = tokens(expected);
    let actual_tokens = tokens(actual);
    let union = expected_tokens.union(&actual_tokens).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = expected_tokens.intersection(&actual_tokens).count();
    let jaccard = intersection as f64 / union as f64;
    if jaccard >= 0.5 {
        return 0.3 + 0.3 * ((jaccard - 0.5) / 0.5);
    }
    0.0
}

/// Splits text into lower-cased alphanumeric tokens, dropping punctuation.
fn tokens(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Evaluates one candidate prompt against one case.
#[async_trait]
pub trait CaseEvaluator: Send + Sync {
    /// Evaluates `prompt` on `case`.
    ///
    /// # Errors
    ///
    /// Propagates the runner's error and any score that cannot be represented
    /// as an [`EvaluatedCase`].
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase>;

    /// Evaluates `prompt` on every case and returns the mean score plus the
    /// per-case results.
    ///
    /// An empty `cases` slice yields `(0.0, vec![])`.
    ///
    /// # Errors
    ///
    /// Propagates the first [`evaluate`](Self::evaluate) failure.
    async fn score_cases(&self, prompt: &str, cases: &[Case]) -> Result<(f64, Vec<EvaluatedCase>)> {
        if cases.is_empty() {
            return Ok((0.0, Vec::new()));
        }
        let mut evaluated = Vec::with_capacity(cases.len());
        for case in cases {
            evaluated.push(self.evaluate(prompt, case).await?);
        }
        let total: f64 = evaluated.iter().map(|case| case.score).sum();
        Ok((total / evaluated.len() as f64, evaluated))
    }
}

/// A [`CaseEvaluator`] composing any [`PromptRunner`] with any [`CaseScorer`].
pub struct RunnerEvaluator {
    /// Produces the answer map for a candidate prompt.
    runner: Arc<dyn PromptRunner>,
    /// Turns the answer map into a score.
    scorer: Arc<dyn CaseScorer>,
}

impl RunnerEvaluator {
    /// Creates an evaluator over `runner` using the default [`LabelScorer`].
    #[must_use]
    pub fn new(runner: Arc<dyn PromptRunner>) -> Self {
        Self {
            runner,
            scorer: Arc::new(LabelScorer),
        }
    }

    /// Replaces the scorer.
    #[must_use]
    pub fn with_scorer(mut self, scorer: Arc<dyn CaseScorer>) -> Self {
        self.scorer = scorer;
        self
    }

    /// The runner in use.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn PromptRunner> {
        &self.runner
    }

    /// The scorer in use.
    #[must_use]
    pub fn scorer(&self) -> &Arc<dyn CaseScorer> {
        &self.scorer
    }
}

impl fmt::Debug for RunnerEvaluator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunnerEvaluator")
            .field("runner", &"custom runner")
            .field("scorer", &"custom scorer")
            .finish()
    }
}

#[async_trait]
impl CaseEvaluator for RunnerEvaluator {
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        let answer = self.runner.run(prompt, case).await?;
        let (score, reason) = self.scorer.score(case, &answer);
        EvaluatedCase::new(case.clone(), answer, sanitize_score(score), reason)
    }
}

/// Clamps a score into `0.0..=1.0`; a non-finite score becomes `0.0`.
fn sanitize_score(score: f64) -> f64 {
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// A [`CaseEvaluator`] that runs candidate prompts through a [`Model`].
///
/// This is the model-backed preset of [`RunnerEvaluator`]: a
/// [`ModelPromptRunner`] plus a [`CaseScorer`] that defaults to
/// [`LabelScorer`].
pub struct ModelEvaluator {
    /// The composed runner-backed evaluator.
    inner: RunnerEvaluator,
}

impl ModelEvaluator {
    /// Creates an evaluator over `model` using the default [`LabelScorer`].
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            inner: RunnerEvaluator::new(Arc::new(ModelPromptRunner::new(model))),
        }
    }

    /// Replaces the scorer.
    #[must_use]
    pub fn with_scorer(mut self, scorer: Arc<dyn CaseScorer>) -> Self {
        self.inner = self.inner.with_scorer(scorer);
        self
    }

    /// The composed evaluator.
    #[must_use]
    pub fn inner(&self) -> &RunnerEvaluator {
        &self.inner
    }
}

impl fmt::Debug for ModelEvaluator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelEvaluator")
            .field("inner", &self.inner)
            .finish()
    }
}

#[async_trait]
impl CaseEvaluator for ModelEvaluator {
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        self.inner.evaluate(prompt, case).await
    }
}

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

#[cfg(test)]
mod tests {
    use openjiuwen_test_support::ScriptedModel;
    use serde_json::json;

    use super::*;

    /// Builds a case from `(key, value)` pairs.
    fn case_of(inputs: &[(&str, &str)], label: &[(&str, &str)], id: &str) -> Case {
        let inputs = inputs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned())))
            .collect();
        let label = label
            .iter()
            .map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned())))
            .collect();
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    /// A two-case fixture with distinct answers.
    fn cases() -> Vec<Case> {
        vec![
            case_of(
                &[("question", "capital of France")],
                &[("answer", "Paris")],
                "c1",
            ),
            case_of(&[("question", "2 + 2")], &[("answer", "4")], "c2"),
        ]
    }

    fn answer(entries: &[(&str, &str)]) -> BTreeMap<String, Value> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned())))
            .collect()
    }

    #[test]
    fn render_case_inputs_uses_one_line_per_input() {
        let case = case_of(
            &[("question", "2 + 2?"), ("language", "english")],
            &[("answer", "4")],
            "c1",
        );
        assert_eq!(
            render_case_inputs(&case),
            "[language]: english\n[question]: 2 + 2?"
        );
    }

    #[test]
    fn answer_to_map_prefers_json_objects() {
        assert_eq!(
            answer_to_map("```json\n{\"answer\": \"4\"}\n```"),
            answer(&[("answer", "4")])
        );
        assert_eq!(
            answer_to_map("{\"answer\": \"4\"}"),
            answer(&[("answer", "4")])
        );
        assert_eq!(
            answer_to_map("  four  "),
            answer(&[("output", "four")]),
            "plain text becomes the output entry"
        );
        assert_eq!(
            answer_to_map("[1, 2]"),
            answer(&[("output", "[1, 2]")]),
            "a JSON array is not an answer map"
        );
    }

    #[test]
    fn label_scorer_bands() {
        let scorer = LabelScorer::new();
        let case = case_of(&[("q", "x")], &[("answer", "Paris")], "c1");
        let bands = [
            ("Paris", 1.0),
            ("  Paris  ", 1.0),
            ("paris", 0.9),
            ("PARIS", 0.9),
            ("the capital is Paris, France", 0.75),
            ("Par", 0.75),
            ("Lyon", 0.0),
        ];
        for (text, expected) in bands {
            let (score, reason) = scorer.score(&case, &answer(&[("answer", text)]));
            assert!(
                (score - expected).abs() < 1e-9,
                "{text:?} should score {expected}, got {score} ({reason})"
            );
        }
    }

    #[test]
    fn label_scorer_maps_jaccard_into_the_band() {
        let scorer = LabelScorer::new();
        // Neither value contains the other, so the Jaccard band decides:
        // |intersection| / |union| = 2 / 4 = 0.5 maps to 0.30.
        let case = case_of(&[("q", "x")], &[("answer", "alpha beta gamma")], "c1");
        let (score, _) = scorer.score(&case, &answer(&[("answer", "alpha gamma delta")]));
        assert!((score - 0.3).abs() < 1e-9, "j=0.5 maps to 0.3: {score}");

        // 3 / 5 = 0.6 maps to 0.3 + 0.3 * (0.1 / 0.5) = 0.36.
        let case = case_of(&[("q", "x")], &[("answer", "alpha beta gamma delta")], "c1");
        let (score, _) = scorer.score(&case, &answer(&[("answer", "alpha beta gamma epsilon")]));
        assert!((score - 0.36).abs() < 1e-9, "j=0.6 maps to 0.36: {score}");

        // Below 0.5 the answer scores nothing.
        let (score, _) = scorer.score(&case, &answer(&[("answer", "alpha omega")]));
        assert!(score.abs() < 1e-9, "j=0.33 maps to 0: {score}");
    }

    #[test]
    fn label_scorer_averages_keys_and_names_the_first_mismatch() {
        let scorer = LabelScorer::new();
        let case = case_of(&[("q", "x")], &[("first", "one"), ("second", "two")], "c1");
        let (score, reason) = scorer.score(&case, &answer(&[("first", "one"), ("second", "2")]));
        assert!(
            (score - 0.5).abs() < 1e-9,
            "one of two keys matched: {score}"
        );
        assert!(
            reason.contains("second"),
            "reason names the mismatch: {reason}"
        );

        let (score, reason) = scorer.score(&case, &answer(&[("first", "one"), ("second", "two")]));
        assert!((score - 1.0).abs() < 1e-9);
        assert!(reason.contains("matched exactly"), "{reason}");
    }

    #[test]
    fn label_scorer_uses_a_single_answer_entry_for_missing_keys() {
        let scorer = LabelScorer::new();
        let case = case_of(&[("q", "x")], &[("answer", "4")], "c1");
        let (score, _) = scorer.score(&case, &answer(&[("output", "4")]));
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn label_scorer_reports_missing_answers_and_scores_are_bounded() {
        let scorer = LabelScorer::new();
        let case = case_of(&[("q", "x")], &[("answer", "4")], "c1");
        let (score, reason) = scorer.score(&case, &BTreeMap::new());
        assert!(score.abs() < 1e-9);
        assert!(reason.contains("missing"), "{reason}");

        let answers = [
            answer(&[("answer", "4")]),
            answer(&[("answer", "four")]),
            answer(&[("other", "4"), ("more", "5")]),
        ];
        for answer in answers {
            let (score, _) = scorer.score(&case, &answer);
            assert!((0.0..=1.0).contains(&score), "score {score} in range");
        }
    }

    #[test]
    fn label_scorer_on_empty_label_is_zero() {
        let scorer = LabelScorer::new();
        let case = Case {
            inputs: answer(&[("q", "x")]),
            label: BTreeMap::new(),
            tools: None,
            case_id: "c1".to_owned(),
        };
        let (score, reason) = scorer.score(&case, &answer(&[("answer", "x")]));
        assert!(score.abs() < 1e-9);
        assert!(reason.contains("no keys"), "{reason}");
    }

    #[test]
    fn score_pair_handles_empty_strings() {
        assert!((score_pair("", "") - 1.0).abs() < 1e-9);
        assert!(score_pair("", "x").abs() < 1e-9);
        assert!(score_pair("x", "").abs() < 1e-9);
    }

    #[tokio::test]
    async fn model_prompt_runner_sends_prompt_and_inputs() {
        let model = ScriptedModel::text("Paris");
        let runner = ModelPromptRunner::new(Arc::new(model.clone()));
        let case = cases().remove(0);
        let answer = runner.run("You answer questions.", &case).await.unwrap();
        assert_eq!(answer, answer_map("output", "Paris"));

        let call = model.last_call().expect("one recorded call");
        assert_eq!(call.messages.len(), 2);
        assert_eq!(call.messages[0].text(), "You answer questions.");
        assert!(call.messages[0].role.is_system());
        assert_eq!(call.messages[1].text(), "[question]: capital of France");
        assert!(call.messages[1].role.is_user());
    }

    #[tokio::test]
    async fn model_prompt_runner_parses_a_fenced_json_answer() {
        let model = ScriptedModel::text("```json\n{\"answer\": \"Paris\"}\n```");
        let runner = ModelPromptRunner::new(Arc::new(model));
        let answer = runner.run("prompt", &cases().remove(0)).await.unwrap();
        assert_eq!(answer, answer_map("answer", "Paris"));
    }

    #[tokio::test]
    async fn model_prompt_runner_propagates_provider_errors() {
        let model = ScriptedModel::new([openjiuwen_test_support::ScriptedTurn::fail("boom")]);
        let runner = ModelPromptRunner::new(Arc::new(model));
        let error = runner.run("prompt", &cases().remove(0)).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
    }

    #[tokio::test]
    async fn scripted_runner_replays_answers_and_records_calls() {
        let runner = ScriptedPromptRunner::new()
            .with_text_answer("c1", "Paris")
            .with_fallback(answer(&[("output", "fallback")]));
        let cases = cases();
        assert_eq!(
            runner.run("p1", &cases[0]).await.unwrap(),
            answer_map("output", "Paris")
        );
        assert_eq!(
            runner.run("p2", &cases[1]).await.unwrap(),
            answer_map("output", "fallback")
        );
        let calls = runner.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].case_id, "c1");
        assert_eq!(calls[1].prompt, "p2");
        assert_eq!(runner.call_count(), 2);
    }

    #[tokio::test]
    async fn scripted_runner_reports_unknown_cases() {
        let runner = ScriptedPromptRunner::new();
        let error = runner.run("p", &cases().remove(0)).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
        assert!(error.to_string().contains("c1"));
    }

    #[tokio::test]
    async fn runner_evaluator_scores_through_the_runner() {
        let runner = Arc::new(
            ScriptedPromptRunner::new()
                .with_text_answer("c1", "Paris")
                .with_text_answer("c2", "5"),
        );
        let evaluator = RunnerEvaluator::new(runner);
        let cases = cases();
        let evaluated = evaluator.evaluate("prompt", &cases[0]).await.unwrap();
        assert!((evaluated.score - 1.0).abs() < 1e-9);
        assert_eq!(evaluated.answer, answer_map("output", "Paris"));
        assert_eq!(evaluated.case_id(), "c1");

        let (mean, all) = evaluator.score_cases("prompt", &cases).await.unwrap();
        assert_eq!(all.len(), 2);
        assert!(
            (mean - 0.5).abs() < 1e-9,
            "one of two cases correct: {mean}"
        );

        let (mean, empty) = evaluator.score_cases("prompt", &[]).await.unwrap();
        assert!(mean.abs() < 1e-9);
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn runner_evaluator_accepts_a_custom_scorer() {
        /// A scorer that always returns a fixed score.
        struct Fixed;

        impl CaseScorer for Fixed {
            fn score(&self, _case: &Case, _answer: &BTreeMap<String, Value>) -> (f64, String) {
                (0.42, "fixed".to_owned())
            }
        }

        let runner = Arc::new(ScriptedPromptRunner::new());
        let evaluator = RunnerEvaluator::new(runner).with_scorer(Arc::new(Fixed));
        assert_eq!(
            evaluator
                .scorer()
                .score(&cases().remove(0), &BTreeMap::new())
                .0,
            0.42
        );
    }

    #[tokio::test]
    async fn model_evaluator_runs_the_model_and_scores_the_answer() {
        let model = ScriptedModel::text("Paris");
        let evaluator = ModelEvaluator::new(Arc::new(model.clone()));
        let evaluated = evaluator
            .evaluate("prompt", &cases().remove(0))
            .await
            .unwrap();
        assert!((evaluated.score - 1.0).abs() < 1e-9);
        assert_eq!(model.call_count(), 1);
        assert!(
            evaluator
                .inner()
                .scorer()
                .score(
                    &case_of(&[("q", "x")], &[("answer", "y")], "z"),
                    &answer(&[("answer", "y")])
                )
                .0
                > 0.99
        );
    }

    #[test]
    fn deterministic_score_rewards_structure_and_placeholders() {
        let evaluator = DeterministicEvaluator::new("You answer {{question}} briefly.");
        let case = case_of(&[("question", "2 + 2")], &[("answer", "4")], "c1");

        let seed = evaluator.score_parts(evaluator.seed(), &case);
        assert!((seed.placeholder - 1.0).abs() < 1e-9);
        assert!(seed.structure > 0.0);

        let bare = evaluator.score_parts("Short.", &case);
        assert!(
            bare.placeholder.abs() < 1e-9,
            "the placeholder was dropped: {bare:?}"
        );
        assert!(bare.total < seed.total);
    }

    #[test]
    fn deterministic_coverage_tracks_case_terms() {
        let evaluator = DeterministicEvaluator::new("Answer the question.");
        let case = case_of(&[("question", "2 + 2")], &[("answer", "four")], "c1");
        let partial = evaluator.score_parts("Answer the question about 2.", &case);
        let full = evaluator.score_parts("Answer the question: 2 + 2 gives four.", &case);
        assert!(partial.coverage < full.coverage);
        assert!((full.coverage - 1.0).abs() < 1e-9, "{full:?}");
    }

    #[test]
    fn deterministic_scores_are_case_dependent() {
        let evaluator = DeterministicEvaluator::new("Answer the question.");
        // `Paris` is a term of the first case only, so coverage differs.
        let prompt = "Answer the question about Paris in the requested output format.";
        let scores: Vec<f64> = cases()
            .iter()
            .map(|case| evaluator.score_parts(prompt, case).total)
            .collect();
        assert_ne!(
            scores[0], scores[1],
            "coverage differs per case: {scores:?}"
        );
    }

    #[test]
    fn deterministic_blank_and_degenerate_candidates_score_low() {
        let evaluator = DeterministicEvaluator::new("You answer {{question}}.");
        let case = case_of(&[("question", "2 + 2")], &[("answer", "4")], "c1");
        assert!(evaluator.score_parts("", &case).total.abs() < 1e-9);
        assert!(evaluator.score_parts("   \n\t ", &case).total.abs() < 1e-9);

        let degenerate = evaluator.score_parts("!!!", &case);
        assert!(degenerate.hygiene <= 0.1, "{degenerate:?}");
        assert!(degenerate.total < 0.2, "{degenerate:?}");

        let tiny = evaluator.score_parts("ok", &case);
        assert!((tiny.hygiene - 0.5).abs() < 1e-9, "{tiny:?}");
    }

    #[test]
    fn deterministic_hygiene_penalizes_repetition_and_length() {
        let evaluator = DeterministicEvaluator::new("Answer.");
        let case = case_of(&[("q", "x")], &[("answer", "y")], "c1");
        let repeated = "Answer the question.\nAnswer the question.\nAnswer the question.\nAnswer the question.";
        let parts = evaluator.score_parts(repeated, &case);
        assert!(parts.hygiene < 1.0, "{parts:?}");

        let long = "Answer the question. ".repeat(600);
        let parts = evaluator.score_parts(&long, &case);
        assert!(parts.hygiene <= 0.7, "{parts:?}");
    }

    #[test]
    fn deterministic_reason_names_every_dimension() {
        let evaluator = DeterministicEvaluator::new("Answer {{question}}.");
        let parts = evaluator.score_parts(
            "Answer {{question}} step by step with the output format requested.",
            &case_of(&[("question", "x")], &[("answer", "y")], "c1"),
        );
        let reason = parts.reason();
        for dimension in ["placeholder", "structure", "coverage", "hygiene"] {
            assert!(reason.contains(dimension), "{reason}");
        }
    }

    #[tokio::test]
    async fn deterministic_evaluator_scores_cases_without_a_runner() {
        let evaluator = DeterministicEvaluator::new("Answer {{question}}.");
        let cases = cases();
        let evaluated = evaluator
            .evaluate("Answer {{question}} in the requested format.", &cases[0])
            .await
            .unwrap();
        assert!(evaluated.answer.is_empty());
        assert!(evaluated.reason.contains("deterministic score"));
        let (mean, all) = evaluator
            .score_cases("Answer the question.", &cases)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);
        assert!(mean > 0.0 && mean <= 1.0);
    }

    #[test]
    fn deterministic_scores_are_repeatable() {
        let evaluator = DeterministicEvaluator::new("Answer {{question}}.");
        let candidate = "Answer {{question}} step by step and verify the result.";
        let case = cases().remove(0);
        let first = evaluator.score_parts(candidate, &case);
        let second = evaluator.score_parts(candidate, &case);
        assert_eq!(first, second);
        assert_eq!(first.total, second.total);
    }

    /// Builds a one-entry answer map.
    fn answer_map(key: &str, value: &str) -> BTreeMap<String, Value> {
        let mut map = BTreeMap::new();
        map.insert(key.to_owned(), Value::String(value.to_owned()));
        map
    }

    #[test]
    fn deterministic_score_serializes_all_dimensions() {
        let score = DeterministicScore {
            total: 0.5,
            placeholder: 1.0,
            structure: 0.25,
            coverage: 0.5,
            hygiene: 1.0,
        };
        let value = serde_json::to_value(score).unwrap();
        for key in ["total", "placeholder", "structure", "coverage", "hygiene"] {
            assert!(value.get(key).is_some(), "missing {key}");
        }
        let restored: DeterministicScore = serde_json::from_value(json!({
            "total": 0.5,
            "extra": true
        }))
        .unwrap();
        assert!((restored.total - 0.5).abs() < 1e-9);
        assert!(
            restored.structure.abs() < 1e-9,
            "missing dimensions default"
        );
    }
}
