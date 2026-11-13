//! Model-backed LLM-as-judge scoring.
//!
//! Ports the model-judge part of Python `openjiuwen.agent_evolving.evaluator`
//! (`evaluator.DefaultEvaluator`, `metrics.llm_as_judge.LLMAsJudgeMetric`, and
//! `templates.LLM_METRIC_TEMPLATE`) onto this crate's own [`Metric`] and
//! [`Evaluator`] seams rather than transliterating the Python prompt-string
//! plumbing.
//!
//! The module keeps three layers apart:
//!
//! - [`JudgeRubric`] / [`JudgeCriterion`] — the criteria and score scale, as
//!   pure data.
//! - [`build_judge_messages`] and [`JudgeVerdict::parse`] — a deterministic
//!   prompt and a tolerant, deterministic response parser. Both are pure
//!   functions, so they are fully testable offline.
//! - [`LlmJudge`] / [`LlmJudgeEvaluator`] — the only code that talks to a model.
//!
//! # Determinism
//!
//! Nothing here reads the clock, a random source, or the environment: the prompt
//! is a pure function of `(case, prediction, rubric)` and parsing is a pure
//! function of the response text. Model calls are the only non-determinism, and
//! the unit tests substitute a scripted model.
//!
//! # Example
//!
//! ```
//! use std::collections::BTreeMap;
//!
//! use openjiuwen_agent_evolving::dataset::Case;
//! use openjiuwen_agent_evolving::evaluator::{build_judge_messages, JudgeRubric, JudgeVerdict};
//! use openjiuwen_core::foundation::prompt::Role;
//! use serde_json::json;
//!
//! # fn main() -> openjiuwen_common::error::Result<()> {
//! let case = Case::new(
//!     BTreeMap::from([("question".to_owned(), json!("2 + 2?"))]),
//!     BTreeMap::from([("answer".to_owned(), json!("4"))]),
//! )?;
//! let rubric = JudgeRubric::default_rubric();
//!
//! let messages = build_judge_messages(&case, &json!("4"), &rubric, 0);
//! assert_eq!(messages.len(), 2);
//! assert_eq!(messages[0].role, Role::System);
//!
//! let verdict = JudgeVerdict::parse(r#"{"score": 8, "reasoning": "close enough"}"#, &rubric)?;
//! assert!((verdict.score - 0.8).abs() < 1e-9);
//! assert!(verdict.passed);
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::metrics::{finite_or_zero, ExactMatchMetric, Metric, MetricValue};
use super::Evaluator;
use crate::dataset::{Case, EvaluatedCase};

/// The name every judge evaluator reports, matching Python's metric name.
const JUDGE_EVALUATOR_NAME: &str = "llm_as_judge";

/// The score scale used when a rubric does not state one (Python's `0..10`).
const DEFAULT_MAX_SCORE: f64 = 10.0;

/// The normalised pass threshold used by [`JudgeRubric::default_rubric`].
const DEFAULT_PASSING_SCORE: f64 = 0.6;

/// How many judge calls [`LlmJudge`] attempts before reporting a parse failure.
const DEFAULT_JUDGE_ATTEMPTS: usize = 2;

/// The upper bound [`LlmJudge::with_attempts`] clamps to.
const MAX_JUDGE_ATTEMPTS: usize = 3;

/// How many prediction characters [`LlmJudge`] sends by default.
const DEFAULT_MAX_PREDICTION_CHARS: usize = 8000;

/// How many characters of an unparseable response a parse error quotes.
const EXCERPT_CHARS: usize = 80;

/// One scoring criterion of a [`JudgeRubric`].
///
/// Python encodes its rubric in prose inside `LLM_METRIC_TEMPLATE`; here it is
/// data, so a caller can reweight or extend the judge without touching prompts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JudgeCriterion {
    /// Stable criterion name; also the key used in
    /// [`JudgeVerdict::per_criterion`] and in the prompt.
    pub name: String,
    /// What the criterion measures, rendered into the judge prompt.
    pub description: String,
    /// Relative weight used when a verdict only reports per-criterion scores;
    /// `1.0` by default.
    pub weight: f64,
}

impl Default for JudgeCriterion {
    /// An unnamed criterion with weight `1.0`, the Python template's default.
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            weight: 1.0,
        }
    }
}

impl JudgeCriterion {
    /// Creates a weight-`1.0` criterion.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            weight: 1.0,
        }
    }

    /// Replaces [`JudgeCriterion::weight`].
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Checks that the criterion can be used in a prompt.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the name is blank or the weight is not a
    /// finite, strictly positive number.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::validation("judge criterion name must not be empty"));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(Error::validation(format!(
                "judge criterion `{}` weight must be finite and greater than zero, got {}",
                self.name, self.weight
            )));
        }
        Ok(())
    }
}

/// The scoring rubric handed to the judge model.
///
/// Scores from the model are read on the `0..=max_score` scale and normalised to
/// `0.0..=1.0`; [`passing_score`](Self::passing_score) is compared against that
/// normalised score, which mirrors Python's 0/1 pass/fail metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JudgeRubric {
    /// The criteria the judge must score, in prompt order.
    pub criteria: Vec<JudgeCriterion>,
    /// Upper bound of the raw score scale; must be finite and positive.
    pub max_score: f64,
    /// Normalised score at or above which a verdict counts as passed.
    pub passing_score: f64,
}

impl Default for JudgeRubric {
    /// The Python-equivalent default rubric, see
    /// [`JudgeRubric::default_rubric`].
    fn default() -> Self {
        Self::default_rubric()
    }
}

impl JudgeRubric {
    /// Builds the default rubric: `correctness` (1.0), `completeness` (0.8), and
    /// `clarity` (0.5) on a `0..10` scale with a `0.6` pass threshold.
    #[must_use]
    pub fn default_rubric() -> Self {
        Self {
            criteria: vec![
                JudgeCriterion::new(
                    "correctness",
                    "The answer agrees with the expected label in meaning and conclusion.",
                )
                .with_weight(1.0),
                JudgeCriterion::new(
                    "completeness",
                    "The answer covers every part of the question that the expected label covers.",
                )
                .with_weight(0.8),
                JudgeCriterion::new(
                    "clarity",
                    "The answer is unambiguous and does not contradict itself.",
                )
                .with_weight(0.5),
            ],
            max_score: DEFAULT_MAX_SCORE,
            passing_score: DEFAULT_PASSING_SCORE,
        }
    }

    /// Builds a rubric with a single weight-`1.0` criterion on the default scale.
    #[must_use]
    pub fn single(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            criteria: vec![JudgeCriterion::new(name, description)],
            max_score: DEFAULT_MAX_SCORE,
            passing_score: DEFAULT_PASSING_SCORE,
        }
    }

    /// Builds a rubric from explicit parts; call [`JudgeRubric::validate`]
    /// before using it with a model.
    #[must_use]
    pub fn new(criteria: Vec<JudgeCriterion>, max_score: f64, passing_score: f64) -> Self {
        Self {
            criteria,
            max_score,
            passing_score,
        }
    }

    /// The criteria the judge must score, in prompt order.
    #[must_use]
    pub fn criteria(&self) -> &[JudgeCriterion] {
        &self.criteria
    }

    /// The upper bound of the raw score scale.
    #[must_use]
    pub fn max_score(&self) -> f64 {
        self.max_score
    }

    /// The normalised pass threshold.
    #[must_use]
    pub fn passing_score(&self) -> f64 {
        self.passing_score
    }

    /// Whether the rubric declares no criteria.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.criteria.is_empty()
    }

    /// Checks that the rubric can be rendered and used for scaling.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when there are no criteria, when a criterion is
    /// invalid, when `max_score` is not finite and positive, or when
    /// `passing_score` is not finite and inside `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        if self.criteria.is_empty() {
            return Err(Error::validation(
                "judge rubric must declare at least one criterion",
            ));
        }
        for criterion in &self.criteria {
            criterion.validate()?;
        }
        if !self.max_score.is_finite() || self.max_score <= 0.0 {
            return Err(Error::validation(format!(
                "judge rubric max_score must be finite and greater than zero, got {}",
                self.max_score
            )));
        }
        if !self.passing_score.is_finite() || !(0.0..=1.0).contains(&self.passing_score) {
            return Err(Error::validation(format!(
                "judge rubric passing_score must be finite and between 0.0 and 1.0, got {}",
                self.passing_score
            )));
        }
        Ok(())
    }
}

/// The verdict one judge call produced.
///
/// `score` and every `per_criterion` value are normalised to `0.0..=1.0`, so a
/// verdict composes with [`Metric`] and [`EvaluatedCase`] without further
/// scaling; `raw_score` keeps the model's own scale for diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JudgeVerdict {
    /// The normalised score in `0.0..=1.0`.
    pub score: f64,
    /// The score on the rubric's own `0..=max_score` scale.
    pub raw_score: f64,
    /// Whether `score` reached [`JudgeRubric::passing_score`].
    pub passed: bool,
    /// The judge's justification, verbatim (trimmed) or empty.
    pub reasoning: String,
    /// Normalised per-criterion scores, keyed by criterion name.
    pub per_criterion: BTreeMap<String, f64>,
}

impl JudgeVerdict {
    /// Parses a judge response into a verdict.
    ///
    /// The parser is deliberately tolerant, because judge models wrap their
    /// answer in prose or fences and name the score key differently:
    ///
    /// - Any balanced `{...}` object in `text` is a candidate, including one
    ///   inside a ```` ```json ```` fence or surrounded by prose; braces inside
    ///   JSON strings are ignored, so escaped content does not fool the scan.
    /// - The score is read from the first present of `score`, `total_score`,
    ///   `rating`, as a JSON number or a numeric string (`"8"`, `"8/10"`,
    ///   `"8 out of 10"`, `"80%"`).
    /// - Per-criterion values come from the first present of `per_criterion`,
    ///   `criteria`, `scores`; entries that are not numeric are skipped.
    /// - The reasoning comes from the first present of `reasoning`,
    ///   `explanation`, `comment`, `reason`; a non-string is rendered as JSON.
    /// - A raw value above `1.0` and at most `max_score` is divided by
    ///   `max_score`; everything is then clamped to `0.0..=1.0`.
    /// - With no explicit score, the rubric-weighted mean of the per-criterion
    ///   values becomes the score.
    /// - A JSON object with only `result`/`passed` (Python's boolean judge
    ///   shape) scores `1.0`/`0.0`.
    /// - Without any parseable object, `score: X`, `score = X`, `X/10`,
    ///   `X out of 10`, and `X%` are scanned case-insensitively in the text.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when no score can be recovered; the message quotes
    /// the first 80 characters of `text`. Also
    /// [`Error::validation`] when the response carries a non-finite number.
    pub fn parse(text: &str, rubric: &JudgeRubric) -> Result<Self> {
        for candidate in json_object_candidates(text) {
            let Ok(Value::Object(object)) = serde_json::from_str::<Value>(candidate) else {
                continue;
            };
            if let Some(verdict) = parse_verdict_object(&object, rubric)? {
                return Ok(verdict);
            }
        }

        if let Some((raw_score, score)) = parse_text_score(text, rubric)? {
            return Ok(Self {
                score,
                raw_score,
                passed: score >= rubric.passing_score,
                reasoning: text.trim().to_owned(),
                per_criterion: BTreeMap::new(),
            });
        }

        Err(Error::validation(format!(
            "could not parse a judge verdict from the response {:?}",
            excerpt(text, EXCERPT_CHARS)
        )))
    }

    /// Builds a verdict, deriving [`JudgeVerdict::passed`] from the rubric.
    fn from_parts(
        score: f64,
        raw_score: f64,
        reasoning: String,
        per_criterion: BTreeMap<String, f64>,
        rubric: &JudgeRubric,
    ) -> Self {
        Self {
            score,
            raw_score,
            passed: score >= rubric.passing_score,
            reasoning,
            per_criterion,
        }
    }
}

/// How many characters of `text` a diagnostic quotes, never splitting a char.
fn excerpt(text: &str, limit: usize) -> String {
    let mut excerpt = String::new();
    let mut truncated = false;
    for (count, character) in text.chars().enumerate() {
        if count >= limit {
            truncated = true;
            break;
        }
        excerpt.push(character);
    }
    if truncated {
        excerpt.push('…');
    }
    excerpt
}

/// Every balanced `{...}` substring of `text`, in order.
///
/// Braces inside JSON strings are skipped, honouring backslash escapes, so a
/// response such as `{"reasoning": "use {braces}"}` yields one candidate.
fn json_object_candidates(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut candidates = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'{' {
            index += 1;
            continue;
        }
        let start = index;
        let mut depth = 0_usize;
        let mut in_string = false;
        let mut escaped = false;
        let mut end = None;
        while index < bytes.len() {
            let byte = bytes[index];
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
            } else {
                match byte {
                    b'"' => in_string = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            end = Some(index + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            index += 1;
        }
        match end {
            Some(end) => {
                candidates.push(&text[start..end]);
                index = end;
            }
            // An unbalanced object: no later candidate can close either.
            None => break,
        }
    }
    candidates
}

/// The first present value among `keys`, in priority order.
fn first_present<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| object.get(*key))
}

/// Reads the reasoning text from a verdict object.
fn extract_reasoning(object: &Map<String, Value>) -> String {
    match first_present(object, &["reasoning", "explanation", "comment", "reason"]) {
        Some(Value::String(text)) => text.trim().to_owned(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Reads a Python-shaped boolean pass flag from a verdict object.
fn extract_boolean(object: &Map<String, Value>) -> Option<bool> {
    match first_present(object, &["result", "passed"])? {
        Value::Bool(value) => Some(*value),
        Value::String(text) if text.trim().eq_ignore_ascii_case("true") => Some(true),
        Value::String(text) if text.trim().eq_ignore_ascii_case("false") => Some(false),
        _ => None,
    }
}

/// Parses one JSON verdict object, or `None` when it carries no score.
///
/// # Errors
///
/// [`Error::validation`] when a numeric field is not finite.
fn parse_verdict_object(
    object: &Map<String, Value>,
    rubric: &JudgeRubric,
) -> Result<Option<JudgeVerdict>> {
    let reasoning = extract_reasoning(object);

    let mut per_criterion = BTreeMap::new();
    if let Some(entries) =
        first_present(object, &["per_criterion", "criteria", "scores"]).and_then(Value::as_object)
    {
        for (name, value) in entries {
            // A criterion the model could not score is reported as text; it is
            // skipped rather than aborting the whole verdict.
            if let Some(raw) = value_to_raw(value)? {
                per_criterion.insert(name.clone(), raw.normalise(rubric.max_score).1);
            }
        }
    }

    if let Some(raw) = first_present(object, &["score", "total_score", "rating"])
        .map(value_to_raw)
        .transpose()?
        .flatten()
    {
        let (raw_score, score) = raw.normalise(rubric.max_score);
        return Ok(Some(JudgeVerdict::from_parts(
            score,
            raw_score,
            reasoning,
            per_criterion,
            rubric,
        )));
    }

    if !per_criterion.is_empty() {
        let score = clamp_score(weighted_mean(&per_criterion, rubric));
        let raw_score = score * scale_of(rubric);
        return Ok(Some(JudgeVerdict::from_parts(
            score,
            raw_score,
            reasoning,
            per_criterion,
            rubric,
        )));
    }

    if let Some(passed) = extract_boolean(object) {
        let score = if passed { 1.0 } else { 0.0 };
        return Ok(Some(JudgeVerdict::from_parts(
            score,
            score * scale_of(rubric),
            reasoning,
            per_criterion,
            rubric,
        )));
    }

    Ok(None)
}

/// A numeric field of a judge response, before rubric scaling.
#[derive(Debug, Clone, Copy)]
enum RawScore {
    /// A number on the rubric's `0..=max_score` scale.
    Absolute(f64),
    /// A fraction already normalised to `0.0..=1.0`.
    Normalised(f64),
}

impl RawScore {
    /// Returns `(raw_score, normalised_score)` for this value on `max_score`.
    fn normalise(self, max_score: f64) -> (f64, f64) {
        let scale = if max_score.is_finite() && max_score > 0.0 {
            max_score
        } else {
            DEFAULT_MAX_SCORE
        };
        match self {
            Self::Absolute(value) => {
                let score = if value > 1.0 && value <= scale {
                    value / scale
                } else {
                    value
                };
                (value, clamp_score(score))
            }
            Self::Normalised(value) => (value * scale, clamp_score(value)),
        }
    }
}

/// The score scale a rubric implies, falling back to 10.
fn scale_of(rubric: &JudgeRubric) -> f64 {
    if rubric.max_score.is_finite() && rubric.max_score > 0.0 {
        rubric.max_score
    } else {
        DEFAULT_MAX_SCORE
    }
}

/// Clamps a finite score into `0.0..=1.0`.
fn clamp_score(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The rubric-weighted mean of normalised per-criterion scores.
///
/// Criteria the rubric does not name fall back to weight `1.0`, matching the
/// tolerant reading of an unconstrained model response.
fn weighted_mean(per_criterion: &BTreeMap<String, f64>, rubric: &JudgeRubric) -> f64 {
    let mut total = 0.0;
    let mut weight_sum = 0.0;
    for (name, value) in per_criterion {
        let weight = rubric
            .criteria
            .iter()
            .find(|criterion| &criterion.name == name)
            .map_or(1.0, |criterion| criterion.weight);
        let weight = if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            1.0
        };
        total += value * weight;
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        total / weight_sum
    } else {
        0.0
    }
}

/// Parses a numeric field, converting a non-finite number into a validation error.
fn parse_float(text: &str) -> Result<Option<f64>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match trimmed.parse::<f64>() {
        Ok(value) if value.is_finite() => Ok(Some(value)),
        Ok(value) => Err(Error::validation(format!(
            "judge response score must be finite, got {value}"
        ))),
        Err(_) => Ok(None),
    }
}

/// Reads a JSON value as a [`RawScore`].
///
/// # Errors
///
/// [`Error::validation`] when a string parses to a non-finite number.
fn value_to_raw(value: &Value) -> Result<Option<RawScore>> {
    match value {
        Value::Number(number) => match number.as_f64() {
            Some(number) if number.is_finite() => Ok(Some(RawScore::Absolute(number))),
            Some(number) => Err(Error::validation(format!(
                "judge response score must be finite, got {number}"
            ))),
            None => Ok(None),
        },
        Value::String(text) => parse_text_number(text),
        _ => Ok(None),
    }
}

/// Parses a numeric string, accepting plain numbers, percentages, and ratios.
///
/// # Errors
///
/// [`Error::validation`] when the string parses to a non-finite number.
fn parse_text_number(text: &str) -> Result<Option<RawScore>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    // `to_ascii_lowercase` keeps byte offsets stable, so slices stay char-safe.
    let lower = trimmed.to_ascii_lowercase();

    if let Some(percent) = lower.strip_suffix('%') {
        return Ok(parse_float(percent)?.map(|value| RawScore::Normalised(value / 100.0)));
    }
    if let Some(position) = lower.find("out of") {
        let numerator = &lower[..position];
        let denominator = &lower[position + "out of".len()..];
        return ratio(numerator, denominator);
    }
    if let Some((numerator, denominator)) = lower.split_once('/') {
        return ratio(numerator, denominator);
    }
    Ok(parse_float(&lower)?.map(RawScore::Absolute))
}

/// Builds a normalised [`RawScore`] from a numerator/denominator pair.
fn ratio(numerator: &str, denominator: &str) -> Result<Option<RawScore>> {
    let (Some(numerator), Some(denominator)) = (parse_float(numerator)?, parse_float(denominator)?)
    else {
        return Ok(None);
    };
    if denominator <= 0.0 {
        return Ok(None);
    }
    Ok(Some(RawScore::Normalised(numerator / denominator)))
}

/// Splits the leading numeric run off `text`.
fn leading_number(text: &str) -> (&str, &str) {
    let mut end = 0;
    for (index, character) in text.char_indices() {
        if character.is_ascii_digit() || character == '.' || character == '+' || character == '-' {
            end = index + character.len_utf8();
        } else {
            break;
        }
    }
    text.split_at(end)
}

/// Reads a [`RawScore`] that follows a `score`/`rating` keyword.
fn score_after_keyword(text: &str) -> Result<Option<RawScore>> {
    let trimmed = text.trim_start();
    let trimmed = trimmed
        .strip_prefix(':')
        .or_else(|| trimmed.strip_prefix('='))
        .unwrap_or(trimmed)
        .trim_start();

    let (number, rest) = leading_number(trimmed);
    if number.is_empty() {
        return Ok(None);
    }
    let rest = rest.trim_start();

    if rest.starts_with('%') {
        return Ok(parse_float(number)?.map(|value| RawScore::Normalised(value / 100.0)));
    }
    if let Some(after_slash) = rest.strip_prefix('/') {
        let (denominator, _) = leading_number(after_slash.trim_start());
        if !denominator.is_empty() {
            return ratio(number, denominator);
        }
    }
    if rest.len() >= "out of".len() && rest[.."out of".len()].eq_ignore_ascii_case("out of") {
        let (denominator, _) = leading_number(rest["out of".len()..].trim_start());
        if !denominator.is_empty() {
            return ratio(number, denominator);
        }
    }
    Ok(parse_float(number)?.map(RawScore::Absolute))
}

/// The number immediately before `index` in `text`, ignoring surrounding space.
fn number_before(text: &str, index: usize) -> Option<&str> {
    let before = text[..index].trim_end();
    let start = before
        .rfind(|character: char| {
            !(character.is_ascii_digit() || character == '.' || character == '-')
        })
        .map_or(0, |position| position + 1);
    let candidate = &before[start..];
    if candidate.is_empty() {
        None
    } else {
        Some(candidate)
    }
}

/// Scans free text for a `X/N`, `X out of N`, or `X%` score.
///
/// `text` must already be ASCII-lowercased so byte offsets align with the
/// original, which is how the caller keeps the reasoning verbatim.
fn scan_bare_ratio(text: &str) -> Result<Option<RawScore>> {
    if let Some(position) = text.find("out of") {
        if let Some(numerator) = number_before(text, position) {
            let (denominator, _) = leading_number(text[position + "out of".len()..].trim_start());
            if !denominator.is_empty() {
                if let Some(raw) = ratio(numerator, denominator)? {
                    return Ok(Some(raw));
                }
            }
        }
    }

    let mut search_from = 0;
    while let Some(offset) = text[search_from..].find(['%', '/']) {
        let index = search_from + offset;
        let separator = text[index..].chars().next().unwrap_or('%');
        let numerator = number_before(text, index);
        if let Some(numerator) = numerator {
            if separator == '%' {
                return Ok(parse_float(numerator)?.map(|value| RawScore::Normalised(value / 100.0)));
            }
            let after = text[index + separator.len_utf8()..].trim_start();
            let (denominator, _) = leading_number(after);
            if !denominator.is_empty() {
                if let Some(raw) = ratio(numerator, denominator)? {
                    return Ok(Some(raw));
                }
            }
        }
        search_from = index + separator.len_utf8();
    }
    Ok(None)
}

/// Recovers a score from prose when the response contains no JSON object.
fn parse_text_score(text: &str, rubric: &JudgeRubric) -> Result<Option<(f64, f64)>> {
    let lower = text.to_ascii_lowercase();
    for keyword in ["score", "rating"] {
        let mut search_from = 0;
        while let Some(offset) = lower[search_from..].find(keyword) {
            let keyword_end = search_from + offset + keyword.len();
            if let Some(raw) = score_after_keyword(&text[keyword_end..])? {
                return Ok(Some(raw.normalise(rubric.max_score)));
            }
            search_from = keyword_end;
        }
    }
    if let Some(raw) = scan_bare_ratio(&lower)? {
        return Ok(Some(raw.normalise(rubric.max_score)));
    }
    Ok(None)
}

/// Renders a JSON object with deterministic (sorted) key order.
fn object_text(values: &BTreeMap<String, Value>) -> String {
    Value::Object(values.clone().into_iter().collect()).to_string()
}

/// Shortens `text` to `limit` characters, appending how much was dropped.
///
/// `limit == 0` disables truncation. Truncation never splits a character.
fn truncate_text(text: &str, limit: usize) -> String {
    if limit == 0 {
        return text.to_owned();
    }
    let total = text.chars().count();
    if total <= limit {
        return text.to_owned();
    }
    let mut truncated: String = text.chars().take(limit).collect();
    let _ = write!(truncated, "... [truncated {} chars]", total - limit);
    truncated
}

/// Builds the `[system, user]` messages one judge call sends.
///
/// The prompt is a pure function of its arguments: the system message states
/// the rubric and demands exactly one JSON verdict object, the user message
/// renders the case inputs, the expected label, and the prediction as compact
/// JSON (sorted keys, so the text is stable). The prediction is truncated to
/// `max_prediction_chars` characters — `0` means no truncation — with an explicit
/// `... [truncated N chars]` marker, so a huge agent transcript cannot push the
/// rubric out of the context window silently.
#[must_use]
pub fn build_judge_messages(
    case: &Case,
    prediction: &Value,
    rubric: &JudgeRubric,
    max_prediction_chars: usize,
) -> Vec<BaseMessage> {
    let scale = scale_of(rubric);
    let mut system = String::from(
        "You are an impartial evaluator scoring one agent response against an expected label.\n",
    );
    let _ = writeln!(
        system,
        "Score every criterion on a 0..{scale} scale, where 0 means unusable and {scale} means fully correct."
    );
    system.push_str("Rubric:\n");
    for criterion in &rubric.criteria {
        let description = criterion.description.trim();
        if description.is_empty() {
            let _ = writeln!(system, "- {} (weight {})", criterion.name, criterion.weight);
        } else {
            let _ = writeln!(
                system,
                "- {} (weight {}): {description}",
                criterion.name, criterion.weight
            );
        }
    }
    system.push_str(
        "Reply with exactly ONE strict JSON object and nothing else: no prose, no markdown fence, no commentary.\n",
    );
    system.push_str(
        "{\"score\": <number>, \"per_criterion\": {\"<criterion name>\": <number>}, \"reasoning\": \"<short justification>\"}\n",
    );
    let _ = writeln!(
        system,
        "Every criterion must appear in per_criterion, and both score and the per-criterion numbers use the 0..{scale} scale."
    );

    let mut user = String::from("Inputs:\n");
    user.push_str(&object_text(&case.inputs));
    user.push_str("\n\nExpected label:\n");
    user.push_str(&object_text(&case.label));
    user.push_str("\n\nPrediction:\n");
    user.push_str(&truncate_text(
        &prediction.to_string(),
        max_prediction_chars,
    ));

    vec![BaseMessage::system(system), BaseMessage::user(user)]
}

/// Scores predictions by asking a model to judge them against the case label.
///
/// This is the live-model path; the synchronous, verdict-parsing counterpart is
/// [`LlmAsJudgeMetric`](super::metrics::LlmAsJudgeMetric).
#[derive(Clone)]
pub struct LlmJudge {
    model: Arc<dyn Model>,
    rubric: JudgeRubric,
    request: ModelRequestConfig,
    attempts: usize,
    max_prediction_chars: usize,
}

impl fmt::Debug for LlmJudge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmJudge")
            .field("model", &self.model.model_name())
            .field("rubric", &self.rubric)
            .field("request", &self.request)
            .field("attempts", &self.attempts)
            .field("max_prediction_chars", &self.max_prediction_chars)
            .finish()
    }
}

impl LlmJudge {
    /// Creates a judge over `model`, using `rubric` and `request`.
    ///
    /// Defaults: 2 attempts and an 8000-character prediction cap.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `rubric` is invalid, so an unusable rubric
    /// fails at construction instead of on every case.
    pub fn new(
        model: Arc<dyn Model>,
        rubric: JudgeRubric,
        request: ModelRequestConfig,
    ) -> Result<Self> {
        rubric.validate()?;
        Ok(Self {
            model,
            rubric,
            request,
            attempts: DEFAULT_JUDGE_ATTEMPTS,
            max_prediction_chars: DEFAULT_MAX_PREDICTION_CHARS,
        })
    }

    /// Sets how many model calls one verdict may cost, clamped to `1..=3`.
    ///
    /// A retry only happens when the response cannot be parsed; model errors are
    /// never retried.
    #[must_use]
    pub fn with_attempts(mut self, attempts: usize) -> Self {
        self.attempts = attempts.clamp(1, MAX_JUDGE_ATTEMPTS);
        self
    }

    /// Sets the prediction character cap; `0` disables truncation.
    #[must_use]
    pub fn with_max_prediction_chars(mut self, max_prediction_chars: usize) -> Self {
        self.max_prediction_chars = max_prediction_chars;
        self
    }

    /// The rubric scores are read against.
    #[must_use]
    pub fn rubric(&self) -> &JudgeRubric {
        &self.rubric
    }

    /// The request configuration sent with every judge call.
    #[must_use]
    pub fn request(&self) -> &ModelRequestConfig {
        &self.request
    }

    /// The model calls a single verdict may cost.
    #[must_use]
    pub fn attempts(&self) -> usize {
        self.attempts
    }

    /// The prediction character cap; `0` means no truncation.
    #[must_use]
    pub fn max_prediction_chars(&self) -> usize {
        self.max_prediction_chars
    }

    /// The model under test.
    #[must_use]
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Asks the model to judge `prediction` for `case`.
    ///
    /// Each attempt reuses the same messages; a response that does not parse is
    /// retried up to [`LlmJudge::attempts`] times and the last parse error is
    /// returned. There is no sleep or backoff, so the call is deterministic given
    /// a deterministic model.
    ///
    /// # Errors
    ///
    /// The model's own error, or [`Error::validation`] when every response was
    /// unparseable.
    pub async fn judge(&self, case: &Case, prediction: &Value) -> Result<JudgeVerdict> {
        let messages =
            build_judge_messages(case, prediction, &self.rubric, self.max_prediction_chars);
        let mut last_error: Option<Error> = None;
        for _ in 0..self.attempts.max(1) {
            let response = self.model.invoke(&messages, &self.request).await?;
            let content = response
                .message()
                .map_or_else(String::new, |message| message.text().into_owned());
            match JudgeVerdict::parse(&content, &self.rubric) {
                Ok(verdict) => return Ok(verdict),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error
            .unwrap_or_else(|| Error::validation("the judge model returned an empty verdict")))
    }
}

/// Scores with an [`LlmJudge`] and falls back to a deterministic [`Metric`].
///
/// The fallback is not a silent degradation: whenever it is used, the reason
/// says whether no judge was configured or the judge failed, and the fallback
/// metric's name is recorded in `per_metric`.
#[derive(Clone)]
pub struct LlmJudgeEvaluator {
    judge: Option<Arc<LlmJudge>>,
    fallback: Arc<dyn Metric>,
    rubric: JudgeRubric,
}

impl fmt::Debug for LlmJudgeEvaluator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmJudgeEvaluator")
            .field("judge", &self.judge)
            .field("fallback", &self.fallback.name())
            .field("rubric", &self.rubric)
            .finish()
    }
}

impl Default for LlmJudgeEvaluator {
    /// Equivalent to [`LlmJudgeEvaluator::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl LlmJudgeEvaluator {
    /// Creates an evaluator without a judge, scoring with exact match until a
    /// judge is installed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            judge: None,
            fallback: Arc::new(ExactMatchMetric::new()),
            rubric: JudgeRubric::default_rubric(),
        }
    }

    /// Installs the judge and adopts a copy of its rubric.
    #[must_use]
    pub fn with_judge(mut self, judge: Arc<LlmJudge>) -> Self {
        self.rubric = judge.rubric().clone();
        self.judge = Some(judge);
        self
    }

    /// Replaces the rubric.
    #[must_use]
    pub fn with_rubric(mut self, rubric: JudgeRubric) -> Self {
        self.rubric = rubric;
        self
    }

    /// Replaces the deterministic metric used when the judge cannot score.
    #[must_use]
    pub fn with_fallback(mut self, fallback: Arc<dyn Metric>) -> Self {
        self.fallback = fallback;
        self
    }

    /// The installed judge, if any.
    #[must_use]
    pub fn judge(&self) -> Option<&LlmJudge> {
        self.judge.as_deref()
    }

    /// The deterministic metric used when the judge cannot score.
    #[must_use]
    pub fn fallback(&self) -> &dyn Metric {
        self.fallback.as_ref()
    }

    /// The rubric a judge built from this evaluator would use.
    #[must_use]
    pub fn rubric(&self) -> &JudgeRubric {
        &self.rubric
    }
}

#[async_trait]
impl Evaluator for LlmJudgeEvaluator {
    fn name(&self) -> &'static str {
        JUDGE_EVALUATOR_NAME
    }

    async fn evaluate(&self, case: &Case, prediction: &Value) -> Result<EvaluatedCase> {
        let Some(judge) = &self.judge else {
            return fallback_evaluated(
                case,
                prediction,
                self.fallback.as_ref(),
                None,
                "no judge model is configured",
            );
        };
        match judge.judge(case, prediction).await {
            Ok(verdict) => judge_evaluated(case, &verdict),
            Err(error) => fallback_evaluated(
                case,
                prediction,
                self.fallback.as_ref(),
                Some(&error),
                "the judge call failed",
            ),
        }
    }
}

/// Builds an [`EvaluatedCase`] from a verdict.
///
/// The reason is the verdict's reasoning followed by a deterministic
/// `[criterion=score, ...]` breakdown, and `per_metric` carries every criterion
/// plus the overall score under `llm_as_judge`.
pub(super) fn judge_evaluated(case: &Case, verdict: &JudgeVerdict) -> Result<EvaluatedCase> {
    let breakdown = verdict
        .per_criterion
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(", ");
    let reasoning = verdict.reasoning.trim();
    let reason = match (reasoning.is_empty(), breakdown.is_empty()) {
        (true, true) => String::from("judged by the model"),
        (true, false) => format!("[{breakdown}]"),
        (false, true) => reasoning.to_owned(),
        (false, false) => format!("{reasoning} [{breakdown}]"),
    };

    let mut per_metric = verdict.per_criterion.clone();
    per_metric.insert(JUDGE_EVALUATOR_NAME.to_owned(), verdict.score);

    let mut evaluated = EvaluatedCase::new(case.clone(), None, verdict.score, reason)?;
    evaluated.per_metric = Some(per_metric);
    Ok(evaluated)
}

/// Builds an [`EvaluatedCase`] by scoring with `metric`.
///
/// `failure` is the judge error, when there was one; the reason names it and the
/// fallback metric, so a fallback score is never mistaken for a judge score. A
/// named metric value is averaged (the same aggregation
/// [`MetricEvaluator`](super::metrics::MetricEvaluator) uses by default), and
/// non-finite values are coerced to `0.0` rather than poisoning the report.
///
/// # Errors
///
/// Whatever `metric.compute` returns, or [`Error::validation`] when the score
/// cannot be turned into an [`EvaluatedCase`].
pub(super) fn fallback_evaluated(
    case: &Case,
    prediction: &Value,
    metric: &dyn Metric,
    failure: Option<&Error>,
    context: &str,
) -> Result<EvaluatedCase> {
    let value = metric.compute(prediction, &case.label)?;

    let (score, mut per_metric) = match value {
        MetricValue::Scalar(value) => (finite_or_zero(value), BTreeMap::new()),
        MetricValue::Named(named) => {
            let values = named
                .values()
                .copied()
                .map(finite_or_zero)
                .collect::<Vec<_>>();
            let score = if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            };
            (score, named)
        }
    };

    per_metric
        .entry(metric.name().to_owned())
        .or_insert(finite_or_zero(score));
    // Keep the map purely finite so `EvaluationReport` aggregation stays sane.
    for value in per_metric.values_mut() {
        *value = finite_or_zero(*value);
    }

    let reason = match failure {
        Some(error) => format!(
            "{context} ({error}); scored with fallback metric `{}`",
            metric.name()
        ),
        None => format!("{context}; scored with fallback metric `{}`", metric.name()),
    };

    let mut evaluated = EvaluatedCase::new(case.clone(), None, score, reason)?;
    evaluated.per_metric = Some(per_metric);
    Ok(evaluated)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use futures::stream::BoxStream;
    use futures::StreamExt;
    use serde_json::json;

    use super::*;
    use openjiuwen_core::foundation::llm::{ChatResponse, Choice, StreamChunk};
    use openjiuwen_core::foundation::prompt::Role;

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    fn case() -> Case {
        Case::new(
            BTreeMap::from([("question".to_owned(), json!("2 + 2?"))]),
            BTreeMap::from([("answer".to_owned(), json!("4"))]),
        )
        .unwrap()
    }

    /// A model that serves queued responses and records how often it was called.
    struct FakeModel {
        responses: Mutex<VecDeque<Result<String>>>,
        calls: AtomicUsize,
    }

    impl FakeModel {
        fn new(responses: Vec<Result<String>>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses.into()),
                calls: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Model for FakeModel {
        fn provider(&self) -> &'static str {
            "fake"
        }

        fn model_name(&self) -> &'static str {
            "fake-judge"
        }

        async fn invoke(
            &self,
            _messages: &[BaseMessage],
            _request: &ModelRequestConfig,
        ) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let next = self.responses.lock().unwrap().pop_front();
            match next {
                Some(Ok(content)) => Ok(ChatResponse {
                    choices: vec![Choice {
                        message: BaseMessage::assistant(content),
                        ..Choice::default()
                    }],
                    ..ChatResponse::default()
                }),
                Some(Err(error)) => Err(error),
                None => Err(Error::validation("the fake judge ran out of responses")),
            }
        }

        fn stream(
            &self,
            _messages: Vec<BaseMessage>,
            _request: ModelRequestConfig,
        ) -> BoxStream<'static, Result<StreamChunk>> {
            futures::stream::empty().boxed()
        }
    }

    #[test]
    fn default_rubric_is_valid_and_ordered() {
        let rubric = JudgeRubric::default();
        assert!(rubric.validate().is_ok());
        let names = rubric
            .criteria()
            .iter()
            .map(|criterion| criterion.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["correctness", "completeness", "clarity"]);
        assert!(close(rubric.max_score(), 10.0));
        assert!(close(rubric.passing_score(), 0.6));
        assert!(!rubric.is_empty());
        assert_eq!(JudgeCriterion::default().weight, 1.0);
        assert_eq!(
            JudgeRubric::single("accuracy", "is it right")
                .criteria()
                .len(),
            1
        );
    }

    #[test]
    fn rubric_validation_rejects_bad_scales() {
        let empty = JudgeRubric::new(Vec::new(), 10.0, 0.6)
            .validate()
            .unwrap_err();
        assert!(
            matches!(&empty, Error::Validation(message) if message.contains("at least one criterion"))
        );

        let weight = JudgeRubric::new(
            vec![JudgeCriterion::new("correctness", "x").with_weight(0.0)],
            10.0,
            0.6,
        )
        .validate()
        .unwrap_err();
        assert!(matches!(&weight, Error::Validation(message) if message.contains("weight")));

        let scale = JudgeRubric::single("correctness", "x");
        assert!(JudgeRubric::new(scale.criteria.clone(), 0.0, 0.6)
            .validate()
            .is_err());
        assert!(JudgeRubric::new(scale.criteria.clone(), 10.0, 1.5)
            .validate()
            .is_err());
        assert!(JudgeRubric::new(scale.criteria, 10.0, f64::NAN)
            .validate()
            .is_err());
        assert!(JudgeCriterion::new("", "x").validate().is_err());
    }

    #[test]
    fn parses_fenced_and_embedded_json() {
        let rubric = JudgeRubric::default_rubric();

        let fenced = "```json\n{\"score\": 8, \"per_criterion\": {\"correctness\": 9}, \"reasoning\": \"ok\"}\n```";
        let verdict = JudgeVerdict::parse(fenced, &rubric).unwrap();
        assert!(close(verdict.score, 0.8));
        assert!(close(verdict.raw_score, 8.0));
        assert!(verdict.passed);
        assert_eq!(verdict.reasoning, "ok");
        assert!(close(verdict.per_criterion["correctness"], 0.9));

        let embedded = "Here you go: {\"total_score\": 4} hope that helps";
        let verdict = JudgeVerdict::parse(embedded, &rubric).unwrap();
        assert!(close(verdict.score, 0.4));
        assert!(!verdict.passed);

        // Braces inside a JSON string do not end the object early.
        let nested = r#"{"reasoning": "use {braces} here", "rating": 10}"#;
        let verdict = JudgeVerdict::parse(nested, &rubric).unwrap();
        assert!(close(verdict.score, 1.0));
        assert_eq!(verdict.reasoning, "use {braces} here");
    }

    #[test]
    fn parses_numeric_strings_and_normalised_scores() {
        let rubric = JudgeRubric::default_rubric();

        let verdict = JudgeVerdict::parse(r#"{"score": "8/10"}"#, &rubric).unwrap();
        assert!(close(verdict.score, 0.8));
        assert!(close(verdict.raw_score, 8.0));

        let verdict = JudgeVerdict::parse(r#"{"score": "8 out of 10"}"#, &rubric).unwrap();
        assert!(close(verdict.score, 0.8));

        let verdict = JudgeVerdict::parse(r#"{"score": "80%"}"#, &rubric).unwrap();
        assert!(close(verdict.score, 0.8));

        // A value already inside `0..=1` is taken as normalised.
        let verdict = JudgeVerdict::parse(r#"{"score": 0.75}"#, &rubric).unwrap();
        assert!(close(verdict.score, 0.75));
        assert!(close(verdict.raw_score, 0.75));

        // A value above `max_score` clamps instead of overflowing the scale.
        let verdict = JudgeVerdict::parse(r#"{"score": 42}"#, &rubric).unwrap();
        assert!(close(verdict.score, 1.0));
    }

    #[test]
    fn weighted_mean_scores_an_ungraded_response() {
        let rubric = JudgeRubric::default_rubric();
        let verdict = JudgeVerdict::parse(
            r#"{"per_criterion": {"correctness": 10, "clarity": 5}}"#,
            &rubric,
        )
        .unwrap();
        // (1.0 * 1.0 + 0.5 * 0.5) / (1.0 + 0.5) = 1.25 / 1.5
        assert!(close(verdict.score, 1.25 / 1.5));
        assert!(close(verdict.raw_score, verdict.score * 10.0));
        assert_eq!(verdict.reasoning, "");

        // An unnamed criterion defaults to weight 1.0.
        let verdict = JudgeVerdict::parse(r#"{"criteria": {"unknown": 0.2}}"#, &rubric).unwrap();
        assert!(close(verdict.score, 0.2));

        // Non-numeric entries are skipped rather than aborting the verdict.
        let verdict = JudgeVerdict::parse(
            r#"{"scores": {"correctness": 10, "notes": "n/a"}, "score": 10}"#,
            &rubric,
        )
        .unwrap();
        assert!(close(verdict.score, 1.0));
        assert_eq!(verdict.per_criterion.len(), 1);
    }

    #[test]
    fn parses_python_boolean_verdicts() {
        let rubric = JudgeRubric::default_rubric();
        let verdict = JudgeVerdict::parse(
            r#"{"result": true, "reason": "consistent conclusion"}"#,
            &rubric,
        )
        .unwrap();
        assert!(close(verdict.score, 1.0));
        assert!(verdict.passed);
        assert_eq!(verdict.reasoning, "consistent conclusion");

        let verdict = JudgeVerdict::parse(r#"{"result": "false"}"#, &rubric).unwrap();
        assert!(close(verdict.score, 0.0));
        assert!(!verdict.passed);
    }

    #[test]
    fn falls_back_to_text_patterns() {
        let rubric = JudgeRubric::default_rubric();

        let verdict = JudgeVerdict::parse("The answer is fine. score: 7", &rubric).unwrap();
        assert!(close(verdict.score, 0.7));
        assert_eq!(verdict.reasoning, "The answer is fine. score: 7");

        let verdict = JudgeVerdict::parse("score = 9/10", &rubric).unwrap();
        assert!(close(verdict.score, 0.9));

        let verdict = JudgeVerdict::parse("I would give it 8 out of 10", &rubric).unwrap();
        assert!(close(verdict.score, 0.8));

        let verdict = JudgeVerdict::parse("definitely 75% correct", &rubric).unwrap();
        assert!(close(verdict.score, 0.75));
    }

    #[test]
    fn unparseable_and_non_finite_responses_are_rejected() {
        let rubric = JudgeRubric::default_rubric();
        let error = JudgeVerdict::parse("no verdict here at all", &rubric).unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message)
                if message.contains("could not parse a judge verdict") && message.contains("no verdict")),
            "unexpected error: {error}"
        );

        let long = "x".repeat(200);
        let error = JudgeVerdict::parse(&long, &rubric).unwrap_err();
        let Error::Validation(message) = &error else {
            panic!("expected a validation error, got {error}");
        };
        // The excerpt is capped and stays char-safe.
        assert!(message.chars().count() < 200, "message: {message}");

        assert!(JudgeVerdict::parse(r#"{"score": "inf"}"#, &rubric).is_err());
    }

    #[test]
    fn build_messages_is_deterministic_and_labelled() {
        let rubric = JudgeRubric::default_rubric();
        let prediction = json!({"output": "4"});
        let messages = build_judge_messages(&case(), &prediction, &rubric, 0);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);

        let system = messages[0].text().into_owned();
        for criterion in rubric.criteria() {
            assert!(system.contains(&criterion.name), "system: {system}");
        }
        assert!(system.contains("ONE strict JSON object"));
        assert!(system.contains("\"per_criterion\""));

        let user = messages[1].text().into_owned();
        assert!(user.starts_with("Inputs:\n{\"question\":\"2 + 2?\"}"));
        assert!(user.contains("Expected label:\n{\"answer\":\"4\"}"));
        assert!(user.contains("Prediction:\n{\"output\":\"4\"}"));

        let again = build_judge_messages(&case(), &prediction, &rubric, 0);
        assert_eq!(messages, again);
    }

    #[test]
    fn build_messages_truncates_the_prediction_on_a_char_boundary() {
        let rubric = JudgeRubric::default_rubric();
        let prediction = json!("héllo wörld");
        let messages = build_judge_messages(&case(), &prediction, &rubric, 5);
        let user = messages[1].text().into_owned();
        // The rendered prediction is the JSON string, quotes included.
        assert!(user.contains("\"héll"), "user: {user}");
        assert!(user.contains("[truncated"), "user: {user}");

        // `0` keeps everything.
        let messages = build_judge_messages(&case(), &prediction, &rubric, 0);
        assert_eq!(
            messages[1].text().into_owned(),
            build_judge_messages(&case(), &prediction, &rubric, 0)[1]
                .text()
                .into_owned()
        );
        assert!(messages[1].text().into_owned().ends_with("\"héllo wörld\""));
    }

    #[tokio::test]
    async fn judge_retries_unparseable_responses_then_succeeds() {
        let model = FakeModel::new(vec![
            Ok(String::from("I cannot answer that.")),
            Ok(String::from(r#"{"score": 9, "reasoning": "correct"}"#)),
        ]);
        let judge = LlmJudge::new(
            model.clone(),
            JudgeRubric::default_rubric(),
            ModelRequestConfig::default(),
        )
        .unwrap();
        assert_eq!(judge.attempts(), DEFAULT_JUDGE_ATTEMPTS);
        assert_eq!(judge.max_prediction_chars(), DEFAULT_MAX_PREDICTION_CHARS);
        assert_eq!(judge.model().model_name(), "fake-judge");
        assert_eq!(judge.request().model_name, "");

        let verdict = judge.judge(&case(), &json!("4")).await.unwrap();
        assert!(close(verdict.score, 0.9));
        assert_eq!(model.calls(), 2);
    }

    #[tokio::test]
    async fn judge_reports_the_last_parse_error_and_propagates_model_errors() {
        let model = FakeModel::new(vec![Ok(String::from("nope")), Ok(String::from("nope"))]);
        let judge = LlmJudge::new(
            model.clone(),
            JudgeRubric::default_rubric(),
            ModelRequestConfig::default(),
        )
        .unwrap();
        let error = judge.judge(&case(), &json!("4")).await.unwrap_err();
        assert!(
            matches!(&error, Error::Validation(_)),
            "unexpected error: {error}"
        );
        assert_eq!(model.calls(), 2);

        let failing = FakeModel::new(vec![Err(Error::llm("provider down"))]);
        let judge = LlmJudge::new(
            failing.clone(),
            JudgeRubric::default_rubric(),
            ModelRequestConfig::default(),
        )
        .unwrap();
        let error = judge.judge(&case(), &json!("4")).await.unwrap_err();
        assert!(matches!(&error, Error::Llm(_)), "unexpected error: {error}");
        assert_eq!(failing.calls(), 1);
    }

    #[test]
    fn judge_builders_clamp_and_replace() {
        let model = FakeModel::new(Vec::new());
        let judge = LlmJudge::new(
            model,
            JudgeRubric::default_rubric(),
            ModelRequestConfig::new("judge-model"),
        )
        .unwrap();
        assert_eq!(judge.clone().with_attempts(0).attempts(), 1);
        assert_eq!(
            judge.clone().with_attempts(9).attempts(),
            MAX_JUDGE_ATTEMPTS
        );
        assert_eq!(
            judge
                .clone()
                .with_max_prediction_chars(12)
                .max_prediction_chars(),
            12
        );
        let error = LlmJudge::new(
            FakeModel::new(Vec::new()),
            JudgeRubric::new(Vec::new(), 10.0, 0.6),
            ModelRequestConfig::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("criterion"), "error: {error}");
    }

    #[tokio::test]
    async fn evaluator_scores_with_the_judge_and_breaks_down_criteria() {
        let model = FakeModel::new(vec![
            Ok(String::from(r#"{"score": 6, "reasoning": "partial"}"#)),
            Ok(String::from(
                r#"{"per_criterion": {"correctness": 10, "clarity": 5}, "reasoning": "fine"}"#,
            )),
        ]);
        let judge = Arc::new(
            LlmJudge::new(
                model,
                JudgeRubric::default_rubric(),
                ModelRequestConfig::default(),
            )
            .unwrap(),
        );
        let evaluator = LlmJudgeEvaluator::new().with_judge(judge);
        assert_eq!(evaluator.name(), "llm_as_judge");
        assert!(evaluator.judge().is_some());
        assert_eq!(evaluator.fallback().name(), "exact_match");
        assert_eq!(evaluator.rubric().criteria().len(), 3);

        let evaluated = evaluator.evaluate(&case(), &json!("4")).await.unwrap();
        assert!(close(evaluated.score, 0.6));
        assert_eq!(evaluated.reason, "partial");
        let per_metric = evaluated.per_metric.unwrap();
        assert!(close(per_metric["llm_as_judge"], 0.6));

        let evaluated = evaluator.evaluate(&case(), &json!("4")).await.unwrap();
        assert!(close(evaluated.score, 1.25 / 1.5));
        // `per_criterion` is a `BTreeMap`, so the breakdown is key-sorted.
        assert_eq!(evaluated.reason, "fine [clarity=0.5, correctness=1]");
        let per_metric = evaluated.per_metric.unwrap();
        assert!(close(per_metric["correctness"], 1.0));
        assert!(close(per_metric["llm_as_judge"], 1.25 / 1.5));
    }

    #[tokio::test]
    async fn evaluator_falls_back_explicitly_when_the_judge_fails() {
        let model = FakeModel::new(vec![
            Ok(String::from("no verdict")),
            Ok(String::from("still no verdict")),
        ]);
        let judge = Arc::new(
            LlmJudge::new(
                model,
                JudgeRubric::default_rubric(),
                ModelRequestConfig::default(),
            )
            .unwrap(),
        );
        let evaluator = LlmJudgeEvaluator::new().with_judge(judge);

        let evaluated = evaluator.evaluate(&case(), &json!("4")).await.unwrap();
        assert!(close(evaluated.score, 1.0));
        assert!(
            evaluated.reason.contains("the judge call failed")
                && evaluated.reason.contains("exact_match"),
            "reason: {}",
            evaluated.reason
        );
        let per_metric = evaluated.per_metric.unwrap();
        assert!(close(per_metric["exact_match"], 1.0));
    }

    #[tokio::test]
    async fn evaluator_without_a_judge_says_so() {
        let evaluator = LlmJudgeEvaluator::default();
        assert!(evaluator.judge().is_none());
        let evaluated = evaluator.evaluate(&case(), &json!("4")).await.unwrap();
        assert!(close(evaluated.score, 1.0));
        assert!(
            evaluated.reason.contains("no judge model is configured"),
            "reason: {}",
            evaluated.reason
        );

        let evaluated = evaluator.evaluate(&case(), &json!("5")).await.unwrap();
        assert!(close(evaluated.score, 0.0));
        assert_eq!(evaluator.fallback().name(), "exact_match");
        // The debug rendering names the model rather than an opaque `dyn`.
        let judge = LlmJudge::new(
            FakeModel::new(Vec::new()),
            JudgeRubric::default_rubric(),
            ModelRequestConfig::default(),
        )
        .unwrap();
        assert!(format!("{judge:?}").contains("fake-judge"));
        assert!(format!("{evaluator:?}").contains("exact_match"));
    }

    #[test]
    fn verdict_serde_round_trips() {
        let verdict = JudgeVerdict::parse(
            r#"{"score": 8, "reasoning": "ok"}"#,
            &JudgeRubric::default(),
        )
        .unwrap();
        let value = serde_json::to_value(&verdict).unwrap();
        let object = value.as_object().unwrap();
        for key in ["score", "raw_score", "passed", "reasoning", "per_criterion"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: JudgeVerdict = serde_json::from_value(value).unwrap();
        assert_eq!(restored, verdict);
    }
}
