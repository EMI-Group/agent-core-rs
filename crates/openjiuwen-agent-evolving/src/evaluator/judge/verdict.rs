//! The judge verdict and its tolerant, deterministic response parser.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::rubric::JudgeRubric;
use super::{scale_of, DEFAULT_MAX_SCORE};

/// How many characters of an unparseable response a parse error quotes.
const EXCERPT_CHARS: usize = 80;

/// The verdict one judge call produced.
///
/// `score` and every `per_criterion` value are normalised to `0.0..=1.0`, so a
/// verdict composes with [`Metric`](crate::evaluator::metrics::Metric) and
/// [`EvaluatedCase`](crate::dataset::EvaluatedCase) without further scaling;
/// `raw_score` keeps the model's own scale for diagnostics.
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

#[cfg(test)]
mod tests {
    use super::super::test_support::close;
    use super::*;

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
