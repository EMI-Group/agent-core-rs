//! The structured judgement a judge returns, and the parser that recovers it.
//!
//! A [`Verdict`] carries one normalized score (`0.0..=1.0`), the pass/fail flag,
//! the reasoning text, the raw per-dimension scores on the rubric scale, and the
//! [`VerdictSource`] that produced it. [`Verdict::parse`] is the tolerant reader
//! for model answers: it accepts a fenced JSON object, tag sections, or a bare
//! JSON object, and it never panics on malformed input — unusable text simply
//! yields `None`.
//!
//! # Parsing rules
//!
//! 1. A fenced ```` ```json ```` object is tried first, then tag sections, then a
//!    bare JSON object anywhere in the text.
//! 2. Field aliases: `score`/`overall_score`/`overall`,
//!    `passed`/`pass`/`result`/`ok`,
//!    `reasoning`/`reason`/`explanation`/`analysis`, and
//!    `dimension_scores`/`dimensions`/`scores`.
//! 3. Numbers may arrive as JSON numbers, numeric strings, or percent strings:
//!    `"80%"` means eighty percent *of the rubric scale*.
//! 4. Per-dimension scores that match at least one rubric dimension decide the
//!    score through [`Rubric::normalize`]; otherwise a raw overall score is
//!    interpreted on the rubric scale, and a raw value of at most `1.0` on a
//!    scale above `1.0` is treated as already normalized.
//! 5. A missing `passed` flag defaults to [`Rubric::is_passing`] of the score.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::judge::rubric::{Rubric, DEFAULT_PASS_THRESHOLD};
use crate::util::extract_json_fence;

/// Which kind of judge produced a [`Verdict`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictSource {
    /// A deterministic rule, scorer, or fallback.
    #[default]
    Deterministic,
    /// A language-model judgement.
    Model,
}

/// One judgement of a subject against a [`Rubric`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Verdict {
    /// The normalized score, always `0.0..=1.0`.
    pub score: f64,
    /// Whether the score clears the pass threshold the judge applied.
    pub passed: bool,
    /// The judge's explanation; may be blank.
    pub reasoning: String,
    /// Raw per-dimension scores, on the rubric scale.
    pub dimension_scores: BTreeMap<String, f64>,
    /// Where the judgement came from.
    pub source: VerdictSource,
}

impl Verdict {
    /// Creates a deterministic verdict, clamping `score` into `0.0..=1.0`.
    ///
    /// `passed` uses [`DEFAULT_PASS_THRESHOLD`] directly rather than a rubric, so
    /// callers without a rubric still get a consistent flag.
    #[must_use]
    pub fn deterministic(score: f64, reasoning: impl Into<String>) -> Self {
        let score = clamp_normalized(score);
        Self {
            score,
            passed: score >= DEFAULT_PASS_THRESHOLD,
            reasoning: reasoning.into(),
            dimension_scores: BTreeMap::new(),
            source: VerdictSource::Deterministic,
        }
    }

    /// Creates a model verdict from an already-normalized score and flag.
    #[must_use]
    pub fn model(
        score: f64,
        passed: bool,
        reasoning: impl Into<String>,
        dimension_scores: BTreeMap<String, f64>,
    ) -> Self {
        Self {
            score,
            passed,
            reasoning: reasoning.into(),
            dimension_scores,
            source: VerdictSource::Model,
        }
    }

    /// Checks that the verdict is internally consistent.
    ///
    /// The reasoning text may be blank; an empty explanation is not an error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the score is not a finite number in
    /// `0.0..=1.0`, or when any dimension score is not finite.
    pub fn validate(&self) -> Result<()> {
        if !self.score.is_finite() || !(0.0..=1.0).contains(&self.score) {
            return Err(Error::validation(format!(
                "verdict score must be a finite number between 0.0 and 1.0, got {}",
                self.score
            )));
        }
        for (name, score) in &self.dimension_scores {
            if !score.is_finite() {
                return Err(Error::validation(format!(
                    "verdict dimension score '{name}' must be a finite number, got {score}"
                )));
            }
        }
        Ok(())
    }

    /// Parses a model answer into a verdict, returning `None` when unusable.
    ///
    /// The accepted shapes and their precedence are listed in the
    /// [module documentation](self). This never panics: malformed JSON, unknown
    /// field types, and missing scores all degrade to `None`.
    #[must_use]
    pub fn parse(text: &str, rubric: &Rubric) -> Option<Self> {
        if let Some(Value::Object(object)) = extract_json_fence(text) {
            if let Some(verdict) = from_object(&object, rubric) {
                return Some(verdict);
            }
        }
        if let Some(verdict) = from_tags(text, rubric) {
            return Some(verdict);
        }
        if let Some(object) = bare_object(text) {
            return from_object(&object, rubric);
        }
        None
    }
}

/// Clamps `score` into `0.0..=1.0`, mapping non-finite input to `0.0`.
fn clamp_normalized(score: f64) -> f64 {
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Interprets a raw overall score on the rubric scale.
///
/// A raw value of at most `1.0` on a scale above `1.0` is read as an already
/// normalized score, because models routinely answer `0.85` whatever the scale.
fn normalize_raw_score(raw: f64, scale: f64) -> f64 {
    if !raw.is_finite() {
        return 0.0;
    }
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    if scale > 1.0 && raw <= 1.0 {
        clamp_normalized(raw)
    } else {
        clamp_normalized(raw / scale)
    }
}

/// The field values collected from one answer, before score resolution.
#[derive(Debug, Default)]
struct Fields {
    /// The raw overall score, when the answer carried one.
    score: Option<f64>,
    /// The explicit pass flag, when the answer carried one.
    passed: Option<bool>,
    /// The reasoning text, when the answer carried one.
    reasoning: Option<String>,
    /// Raw per-dimension scores, in name order.
    dimension_scores: BTreeMap<String, f64>,
}

/// Turns collected fields into a verdict, or `None` when no score is usable.
fn assemble(fields: Fields, rubric: &Rubric) -> Option<Verdict> {
    let has_known_dimension = fields
        .dimension_scores
        .keys()
        .any(|name| rubric.dimension(name).is_some());

    let score = if has_known_dimension {
        rubric.normalize(&fields.dimension_scores)
    } else {
        normalize_raw_score(fields.score?, rubric.scale)
    };

    let passed = fields.passed.unwrap_or_else(|| rubric.is_passing(score));
    Some(Verdict::model(
        score,
        passed,
        fields.reasoning.unwrap_or_default(),
        fields.dimension_scores,
    ))
}

/// The first key of `aliases` present in `object` with a string value.
fn first_string(object: &Map<String, Value>, aliases: &[&str]) -> Option<String> {
    aliases
        .iter()
        .find_map(|alias| object.get(*alias).and_then(Value::as_str))
        .map(str::to_owned)
}

/// The first key of `aliases` present in `object` with a boolean value.
fn first_bool(object: &Map<String, Value>, aliases: &[&str]) -> Option<bool> {
    aliases
        .iter()
        .find_map(|alias| object.get(*alias).and_then(parse_bool_value))
}

/// The first key of `aliases` present in `object` with a numeric value.
fn first_number(object: &Map<String, Value>, aliases: &[&str], scale: f64) -> Option<f64> {
    aliases.iter().find_map(|alias| {
        object
            .get(*alias)
            .and_then(|value| parse_number_value(value, scale))
    })
}

/// The first key of `aliases` present in `object` with an object value.
fn first_object<'a>(
    object: &'a Map<String, Value>,
    aliases: &[&str],
) -> Option<&'a Map<String, Value>> {
    aliases
        .iter()
        .find_map(|alias| object.get(*alias).and_then(Value::as_object))
}

/// Builds a verdict from one JSON object.
fn from_object(object: &Map<String, Value>, rubric: &Rubric) -> Option<Verdict> {
    let mut fields = Fields {
        score: first_number(object, &["score", "overall_score", "overall"], rubric.scale),
        passed: first_bool(object, &["passed", "pass", "result", "ok"]),
        reasoning: first_string(object, &["reasoning", "reason", "explanation", "analysis"]),
        dimension_scores: BTreeMap::new(),
    };
    if let Some(dimensions) = first_object(object, &["dimension_scores", "dimensions", "scores"]) {
        for (name, value) in dimensions {
            if let Some(score) = parse_number_value(value, rubric.scale) {
                fields.dimension_scores.insert(name.clone(), score);
            }
        }
    }
    assemble(fields, rubric)
}

/// Reads a JSON value as a score, honoring percent strings.
fn parse_number_value(value: &Value, scale: f64) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|parsed| parsed.is_finite()),
        Value::String(text) => parse_number_text(text, scale),
        _ => None,
    }
}

/// Reads `text` as a finite score, honoring a trailing percent sign.
///
/// `"80%"` yields `0.8 * scale`; plain numbers are returned as written.
fn parse_number_text(text: &str, scale: f64) -> Option<f64> {
    let trimmed = text.trim();
    let (body, percent) = match trimmed.strip_suffix('%') {
        Some(body) => (body.trim(), true),
        None => (trimmed, false),
    };
    let parsed = body.parse::<f64>().ok().filter(|value| value.is_finite())?;
    if percent {
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        Some(parsed / 100.0 * scale)
    } else {
        Some(parsed)
    }
}

/// Reads a JSON value as a pass flag.
fn parse_bool_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::String(text) => parse_bool_text(text),
        Value::Number(number) => number.as_f64().map(|parsed| parsed != 0.0),
        _ => None,
    }
}

/// Reads common pass/fail wordings.
fn parse_bool_text(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "pass" | "passed" | "ok" => Some(true),
        "false" | "no" | "fail" | "failed" => Some(false),
        _ => None,
    }
}

/// One `<name attributes>body</name>` section found in an answer.
#[derive(Debug)]
struct TagSection {
    /// The lowercased tag name.
    name: String,
    /// The original attribute text between the name and `>`.
    attributes: String,
    /// The original body between the tags.
    body: String,
}

/// Scans `text` for XML-like tag sections, ignoring closing and declaration tags.
///
/// Names are matched case-insensitively; bodies keep their original casing.
/// Scanning continues after every opening tag, so nested sections are found too.
fn scan_tags(text: &str) -> Vec<TagSection> {
    let lower = text.to_ascii_lowercase();
    let mut sections = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = lower[pos..].find('<') {
        let open = pos + rel;
        let after_open = &lower[open + 1..];
        if after_open.starts_with('/') || after_open.starts_with('!') || after_open.starts_with('?')
        {
            pos = open + 1;
            continue;
        }
        let name_len = after_open
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .count();
        if name_len == 0 {
            pos = open + 1;
            continue;
        }
        let name_end = open + 1 + name_len;
        let Some(close_rel) = lower[name_end..].find('>') else {
            break;
        };
        let tag_end = name_end + close_rel;
        let attributes = text[name_end..tag_end].trim().to_owned();
        pos = tag_end + 1;
        if attributes.ends_with('/') {
            continue;
        }
        let name = lower[open + 1..name_end].to_owned();
        let closing = format!("</{name}>");
        let body_start = tag_end + 1;
        let Some(body_end_rel) = lower[body_start..].find(&closing) else {
            continue;
        };
        let body_end = body_start + body_end_rel;
        sections.push(TagSection {
            name,
            attributes,
            body: text[body_start..body_end].to_owned(),
        });
    }
    sections
}

/// Reads the `name="…"` attribute of a `<dimension>` section.
fn dimension_attribute(attributes: &str) -> Option<String> {
    let lower = attributes.to_ascii_lowercase();
    let start = lower.find("name")?;
    let rest = attributes[start + "name".len()..].trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let body = &rest[quote.len_utf8()..];
    let end = body.find(quote)?;
    let name = body[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_owned())
    }
}

/// Builds a verdict from XML-like tag sections.
fn from_tags(text: &str, rubric: &Rubric) -> Option<Verdict> {
    let mut fields = Fields::default();
    for section in scan_tags(text) {
        match section.name.as_str() {
            "score" | "overall_score" | "overall" => {
                if let Some(score) = parse_number_text(&section.body, rubric.scale) {
                    fields.score = Some(score);
                }
            }
            "passed" | "pass" | "result" | "ok" => {
                if let Some(passed) = parse_bool_text(&section.body) {
                    fields.passed = Some(passed);
                }
            }
            "reasoning" | "reason" | "explanation" | "analysis" => {
                let reasoning = section.body.trim();
                if !reasoning.is_empty() {
                    fields.reasoning = Some(reasoning.to_owned());
                }
            }
            "dimension" => {
                let Some(name) = dimension_attribute(&section.attributes) else {
                    continue;
                };
                if let Some(score) = parse_number_text(&section.body, rubric.scale) {
                    fields.dimension_scores.insert(name, score);
                }
            }
            _ => {}
        }
    }
    assemble(fields, rubric)
}

/// Finds the first bare JSON object anywhere in `text`.
fn bare_object(text: &str) -> Option<Map<String, Value>> {
    for (index, _) in text.char_indices().filter(|(_, c)| *c == '{') {
        let mut stream = serde_json::Deserializer::from_str(&text[index..]).into_iter::<Value>();
        if let Some(Ok(Value::Object(object))) = stream.next() {
            return Some(object);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::rubric::RubricDimension;
    use serde_json::json;

    fn rubric() -> Rubric {
        Rubric::new("skill quality", "Does the skill do its job?")
            .with_dimension(RubricDimension::new("clarity", "Is it clear?"))
            .with_dimension(RubricDimension::new("depth", "Is it deep?"))
    }

    fn scaled_rubric() -> Rubric {
        rubric().with_scale(10.0)
    }

    #[test]
    fn deterministic_constructor_clamps_and_applies_the_default_threshold() {
        let verdict = Verdict::deterministic(1.4, "great");
        assert_eq!(verdict.score, 1.0);
        assert!(verdict.passed);
        assert_eq!(verdict.source, VerdictSource::Deterministic);

        let verdict = Verdict::deterministic(-0.5, "bad");
        assert_eq!(verdict.score, 0.0);
        assert!(!verdict.passed);

        let boundary = Verdict::deterministic(DEFAULT_PASS_THRESHOLD, "");
        assert!(boundary.passed);
        assert!(!Verdict::deterministic(0.599, "just below").passed);
        assert_eq!(Verdict::deterministic(f64::NAN, "").score, 0.0);
    }

    #[test]
    fn model_constructor_keeps_the_source_and_dimensions() {
        let mut scores = BTreeMap::new();
        scores.insert("clarity".to_owned(), 0.5);
        let verdict = Verdict::model(0.5, false, "meh", scores.clone());
        assert_eq!(verdict.source, VerdictSource::Model);
        assert_eq!(verdict.dimension_scores, scores);
        assert!(!verdict.passed);
    }

    #[test]
    fn validate_accepts_blank_reasoning_and_rejects_bad_numbers() {
        Verdict::deterministic(0.5, "")
            .validate()
            .expect("blank ok");

        let mut broken = Verdict::deterministic(0.5, "x");
        broken.score = 1.5;
        assert!(broken.validate().is_err());

        broken.score = f64::NAN;
        assert!(broken.validate().is_err());

        let mut broken = Verdict::deterministic(0.5, "x");
        broken
            .dimension_scores
            .insert("clarity".to_owned(), f64::INFINITY);
        let error = broken.validate().expect_err("infinite dimension");
        assert!(error.to_string().contains("clarity"), "{error}");
    }

    #[test]
    fn parse_fenced_json_with_primary_keys() {
        let text = "Here you go.\n```json\n\
                    {\"score\": 0.75, \"passed\": true, \"reasoning\": \"solid\"}\n```\n";
        let verdict = Verdict::parse(text, &rubric()).expect("parses");
        assert!((verdict.score - 0.75).abs() < 1e-9);
        assert!(verdict.passed);
        assert_eq!(verdict.reasoning, "solid");
        assert_eq!(verdict.source, VerdictSource::Model);
    }

    #[test]
    fn parse_fenced_json_with_aliases() {
        let text = "```json\n{\"overall\": 0.4, \"ok\": false, \"analysis\": \"weak\"}\n```";
        let verdict = Verdict::parse(text, &rubric()).expect("parses");
        assert!((verdict.score - 0.4).abs() < 1e-9);
        assert!(!verdict.passed);
        assert_eq!(verdict.reasoning, "weak");

        let text = "```json\n{\"overall_score\": \"0.9\", \"result\": \"pass\", \"reason\": \"fine\"}\n```";
        let verdict = Verdict::parse(text, &rubric()).expect("parses");
        assert!((verdict.score - 0.9).abs() < 1e-9);
        assert!(verdict.passed);
        assert_eq!(verdict.reasoning, "fine");
    }

    #[test]
    fn parse_percent_strings() {
        let text = "```json\n{\"score\": \"80%\", \"reasoning\": \"mostly\"}\n```";
        let verdict = Verdict::parse(text, &scaled_rubric()).expect("parses");
        assert!((verdict.score - 0.8).abs() < 1e-9);
        assert!(verdict.passed);
    }

    #[test]
    fn parse_dimension_scores_normalize_through_the_rubric() {
        let text = "```json\n{\"dimension_scores\": {\"clarity\": 8, \"depth\": 4}}\n```";
        let verdict = Verdict::parse(text, &scaled_rubric()).expect("parses");
        // Equal weights: (8/10 + 4/10) / 2.
        assert!((verdict.score - 0.6).abs() < 1e-9);
        assert!(verdict.passed);
        assert_eq!(verdict.dimension_scores.len(), 2);
        assert_eq!(verdict.dimension_scores["clarity"], 8.0);
    }

    #[test]
    fn parse_dimension_scores_accept_strings_and_percents() {
        let text = "```json\n{\"scores\": {\"clarity\": \"9\", \"depth\": \"50%\"}}\n```";
        let verdict = Verdict::parse(text, &scaled_rubric()).expect("parses");
        // 50% of scale 10 is 5.0, so (0.9 + 0.5) / 2.
        assert!((verdict.score - 0.7).abs() < 1e-9);
        assert_eq!(verdict.dimension_scores["depth"], 5.0);
    }

    #[test]
    fn parse_unknown_dimension_names_fall_back_to_the_raw_score() {
        let text = "```json\n{\"score\": 3.0, \"dimensions\": {\"polish\": 9}}\n```";
        let verdict = Verdict::parse(text, &scaled_rubric()).expect("parses");
        assert!((verdict.score - 0.3).abs() < 1e-9);
    }

    #[test]
    fn parse_treats_a_small_raw_score_on_a_large_scale_as_normalized() {
        let text = "```json\n{\"score\": 0.8}\n```";
        let verdict = Verdict::parse(text, &scaled_rubric()).expect("parses");
        assert!((verdict.score - 0.8).abs() < 1e-9);
    }

    #[test]
    fn parse_passed_defaults_to_the_threshold() {
        let verdict = Verdict::parse("```json\n{\"score\": 0.7}\n```", &rubric()).expect("parses");
        assert!(verdict.passed);

        let verdict = Verdict::parse("```json\n{\"score\": 0.5}\n```", &rubric()).expect("parses");
        assert!(!verdict.passed);
    }

    #[test]
    fn parse_tag_sections() {
        let text = "<score>0.8</score>\n<reasoning>clear and complete</reasoning>\n\
                    <dimension name=\"clarity\">0.9</dimension>\n\
                    <dimension name=\"depth\">0.5</dimension>";
        let verdict = Verdict::parse(text, &rubric()).expect("parses");
        // Dimensions win over the tag score: (0.9 + 0.5) / 2.
        assert!((verdict.score - 0.7).abs() < 1e-9);
        assert_eq!(verdict.reasoning, "clear and complete");
        assert_eq!(verdict.dimension_scores.len(), 2);
        assert!(verdict.passed);
    }

    #[test]
    fn parse_tag_sections_are_case_insensitive_and_accept_reason() {
        let text = "<SCORE>0.35</SCORE>\n<REASON>needs work</REASON>";
        let verdict = Verdict::parse(text, &rubric()).expect("parses");
        assert!((verdict.score - 0.35).abs() < 1e-9);
        assert!(!verdict.passed);
        assert_eq!(verdict.reasoning, "needs work");

        let text = "<overall_score>0.65</overall_score>";
        let verdict = Verdict::parse(text, &rubric()).expect("parses");
        assert!((verdict.score - 0.65).abs() < 1e-9);
    }

    #[test]
    fn parse_tag_score_only_uses_the_raw_scale() {
        let text = "<score>7</score>";
        let verdict = Verdict::parse(text, &scaled_rubric()).expect("parses");
        assert!((verdict.score - 0.7).abs() < 1e-9);
    }

    #[test]
    fn parse_bare_json_object_without_a_fence() {
        let text = "My judgement: {\"score\": 0.55, \"passed\": false} — thanks.";
        let verdict = Verdict::parse(text, &rubric()).expect("parses");
        assert!((verdict.score - 0.55).abs() < 1e-9);
        assert!(!verdict.passed);
    }

    #[test]
    fn parse_empty_dimension_map_falls_back_to_nothing() {
        let text = "```json\n{\"dimensions\": {}, \"reasoning\": \"none\"}\n```";
        assert!(Verdict::parse(text, &rubric()).is_none());
    }

    #[test]
    fn parse_malformed_input_yields_none() {
        assert!(Verdict::parse("", &rubric()).is_none());
        assert!(Verdict::parse("no structured answer here", &rubric()).is_none());
        assert!(Verdict::parse("```json\n{not json}\n```", &rubric()).is_none());
        assert!(
            Verdict::parse("```json\n{\"reasoning\": \"scoreless\"}\n```", &rubric()).is_none()
        );
        assert!(Verdict::parse("{", &rubric()).is_none());
        assert!(Verdict::parse("<score>high</score>", &rubric()).is_none());
        assert!(Verdict::parse("{\"score\": null}", &rubric()).is_none());
        assert!(Verdict::parse("{\"score\": \"NaN\"}", &rubric()).is_none());
        assert!(Verdict::parse("{\"score\": true}", &rubric()).is_none());
    }

    #[test]
    fn parse_clamps_out_of_range_scores() {
        let verdict = Verdict::parse("{\"score\": 42}", &scaled_rubric()).expect("parses");
        assert_eq!(verdict.score, 1.0);

        let verdict = Verdict::parse("{\"score\": -7}", &rubric()).expect("parses");
        assert_eq!(verdict.score, 0.0);
    }

    #[test]
    fn parse_never_panics_on_unusual_text() {
        for text in [
            "<",
            ">",
            "<dimension",
            "<dimension name=>3</dimension>",
            "{}",
            "}{",
        ] {
            let _ = Verdict::parse(text, &rubric());
        }
    }

    #[test]
    fn serde_round_trip_uses_snake_case_source() {
        let verdict = Verdict::model(0.5, true, "ok", BTreeMap::new());
        let value = serde_json::to_value(&verdict).expect("serializes");
        assert_eq!(value["source"], json!("model"));
        assert_eq!(value["dimension_scores"], json!({}));

        let decoded: Verdict = serde_json::from_value(value).expect("deserializes");
        assert_eq!(decoded, verdict);

        let decoded: Verdict =
            serde_json::from_value(json!({"score": 0.2, "passed": false})).expect("defaults");
        assert_eq!(decoded.source, VerdictSource::Deterministic);
        assert_eq!(decoded.reasoning, "");

        assert_eq!(
            serde_json::to_value(VerdictSource::Deterministic).expect("serializes"),
            json!("deterministic")
        );
        let decoded: VerdictSource = serde_json::from_value(json!("model")).expect("deserializes");
        assert_eq!(decoded, VerdictSource::Model);
    }
}
