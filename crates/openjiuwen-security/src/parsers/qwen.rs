//! `Qwen3Guard` model-output parser.
//!
//! Ports `QwenGuardParser` from
//! `openjiuwen/core/security/guardrail/context.py`. The model answers in one of
//! several dialects: a standard `Safety:`/`Categories:` text (optionally wrapped
//! in JSON), a rich `analysis` JSON object, a simple `judgment` object, an API
//! `severity_level` object, or a `risk_level`/`level` object. All of them
//! collapse onto the same three-level map: `safe`, `controversial`, `unsafe`.
//!
//! Parsing is fallible: a reply outside those dialects (an empty object, an
//! unrelated key set, `null`, a boolean, a number, an array, or free text with
//! no verdict marker) is [`Error::validation`], never a silent `Safe` grade.
//! Inside a recognized dialect a verdict that cannot be mapped — an unknown
//! level *token* or a *missing* verdict — maps onto
//! [`QwenGuardParser::default_risk_level`], which [`QwenGuardParser::new`] sets
//! to [`RiskLevel::Critical`] so it blocks under the default
//! `block_level = Critical`. [`QwenGuardParser::with_default_risk_level`] with
//! [`RiskLevel::Safe`] is the explicit opt-in for Python's behaviour, which
//! graded every unmappable verdict `Safe`.

use std::sync::LazyLock;

use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use serde_json::{json, Map, Value};

use super::ModelOutputParser;
use crate::{RiskAssessment, RiskLevel};

/// Matches Python's `safety\s*:\s*(\w+)` extraction (case-insensitive).
#[allow(clippy::expect_used)]
static SAFETY_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)safety\s*:\s*(\w+)").expect("the safety pattern is a valid regex")
});

/// Matches Python's `categories?\s*:\s*(.+?)(?:\n|$)` extraction
/// (case-insensitive).
#[allow(clippy::expect_used)]
static CATEGORIES_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)categories?\s*:\s*(.+?)(?:\n|$)")
        .expect("the categories pattern is a valid regex")
});

/// Parser for `Qwen3Guard` moderation output.
///
/// The default configuration keeps Python's `risk_type = "content_risk"` and
/// deviates on one field: `default_risk_level` is [`RiskLevel::Critical`] rather
/// than Python's `Safe`, so an unmappable or missing verdict errs upward.
#[derive(Debug, Clone)]
pub struct QwenGuardParser {
    /// Risk name reported when the model flags content but names no category
    /// (`"content_risk"` by default).
    pub risk_type: String,
    /// Level used when the model's verdict cannot be mapped onto the level
    /// table, and when a recognized dialect carries no verdict at all.
    ///
    /// [`RiskLevel::Critical`] by default; set [`RiskLevel::Safe`] through
    /// [`Self::with_default_risk_level`] for Python's grading.
    pub default_risk_level: RiskLevel,
}

impl Default for QwenGuardParser {
    /// [`Self::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl QwenGuardParser {
    /// A parser with the crate defaults.
    ///
    /// `risk_type = "content_risk"` (Python) and `default_risk_level =
    /// RiskLevel::Critical` (a deliberate deviation from Python's `Safe`: an
    /// unrecognizable verdict token blocks under the default `block_level`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            risk_type: "content_risk".to_owned(),
            default_risk_level: RiskLevel::Critical,
        }
    }

    /// Uses `risk_type` instead of the default `"content_risk"`.
    #[must_use]
    pub fn with_risk_type(mut self, risk_type: impl Into<String>) -> Self {
        self.risk_type = risk_type.into();
        self
    }

    /// Uses `default_risk_level` for verdicts outside the risk map.
    ///
    /// Pass [`RiskLevel::Safe`] for Python's grading of an unmappable verdict.
    #[must_use]
    pub fn with_default_risk_level(mut self, default_risk_level: RiskLevel) -> Self {
        self.default_risk_level = default_risk_level;
        self
    }

    /// Maps a model verdict onto a risk level.
    ///
    /// Lowercases and trims the verdict; an exact table key wins (including the
    /// severity words `critical`/`high`/`medium`/`low`), otherwise the first
    /// keyword contained in the verdict (checked longest first, so `"unsafe"`
    /// beats `"safe"`), otherwise [`Self::default_risk_level`]. Containment is
    /// deliberately limited to `controversial`/`unsafe`/`safe`: `low` is a
    /// substring of words like `below`.
    fn map_risk_level(&self, value: &str) -> RiskLevel {
        let normalized = value.trim().to_lowercase();
        let exact = match normalized.as_str() {
            "safe" => Some(RiskLevel::Safe),
            "low" => Some(RiskLevel::Low),
            "controversial" | "medium" => Some(RiskLevel::Medium),
            "unsafe" | "high" => Some(RiskLevel::High),
            "critical" => Some(RiskLevel::Critical),
            _ => None,
        };
        if let Some(level) = exact {
            return level;
        }
        for (keyword, level) in [
            ("controversial", RiskLevel::Medium),
            ("unsafe", RiskLevel::High),
            ("safe", RiskLevel::Safe),
        ] {
            if normalized.contains(keyword) {
                return level;
            }
        }
        self.default_risk_level
    }

    /// Builds the assessment shared by every dialect.
    #[allow(clippy::unused_self)]
    fn assess(&self, risk_level: RiskLevel, risk_type: String, details: Value) -> RiskAssessment {
        let has_risk = risk_level.is_risky();
        RiskAssessment {
            has_risk,
            risk_level,
            risk_type: has_risk.then_some(risk_type),
            confidence: 0.0,
            details: Some(details),
        }
    }

    /// Dispatches an object to the dialect its fields identify.
    ///
    /// An object carrying none of the dialect markers is unrecognized, which is
    /// an error rather than a safe grade.
    fn parse_object(&self, data: &Map<String, Value>) -> Result<RiskAssessment> {
        if data.contains_key("safety") {
            return Ok(self.parse_standard_object(data));
        }
        if data.contains_key("analysis") {
            return Ok(self.parse_full_object(data));
        }
        if data.contains_key("judgment") {
            return Ok(self.parse_simple_object(data));
        }
        if data.contains_key("severity_level") {
            return Ok(self.parse_api_object(data));
        }
        if data.contains_key("risk_level") || data.contains_key("level") {
            return Ok(self.parse_generic_object(data));
        }
        Err(unrecognized_output())
    }

    /// Standard `safety` / `categories` object dialect.
    fn parse_standard_object(&self, data: &Map<String, Value>) -> RiskAssessment {
        let safety = string_or(data.get("safety"), "");
        let categories = categories_of(data.get("categories"));
        let risk_level = self.map_risk_level(&safety);
        let risk_type = categories
            .first()
            .cloned()
            .unwrap_or_else(|| self.risk_type.clone());
        self.assess(
            risk_level,
            risk_type,
            json!({"safety": safety, "categories": categories}),
        )
    }

    /// Rich `analysis` object dialect.
    fn parse_full_object(&self, data: &Map<String, Value>) -> RiskAssessment {
        let analysis = data.get("analysis").and_then(Value::as_object);
        let field = |key: &str| analysis.and_then(|map| map.get(key));

        let risk_level = self.map_risk_level(&string_or(field("risk_level"), ""));
        let risk_categories = categories_of(field("risk_categories"));
        let risk_type = risk_categories
            .first()
            .cloned()
            .unwrap_or_else(|| self.risk_type.clone());
        let evidence = field("evidence").cloned().unwrap_or(Value::Null);
        let language = string_or(field("language"), "unknown");
        let decision = string_or(data.get("decision"), "unknown");
        let version = string_or(data.get("version"), "unknown");
        self.assess(
            risk_level,
            risk_type,
            json!({
                "risk_categories": risk_categories,
                "evidence": evidence,
                "language": language,
                "decision": decision,
                "version": version,
            }),
        )
    }

    /// Legacy simple `judgment` object dialect.
    fn parse_simple_object(&self, data: &Map<String, Value>) -> RiskAssessment {
        let judgment = string_or(data.get("judgment"), "");
        let risk_level = self.map_risk_level(&judgment);
        let reason = data.get("reason").cloned().unwrap_or(Value::Null);
        let language = string_or(data.get("language"), "unknown");
        self.assess(
            risk_level,
            self.risk_type.clone(),
            json!({"reason": reason, "language": language, "judgment": judgment}),
        )
    }

    /// Legacy API `severity_level` object dialect.
    fn parse_api_object(&self, data: &Map<String, Value>) -> RiskAssessment {
        let severity = string_or(data.get("severity_level"), "");
        let risk_level = self.map_risk_level(&severity);
        let reason = data.get("reason").cloned().unwrap_or(Value::Null);
        let language = string_or(data.get("language"), "unknown");
        self.assess(
            risk_level,
            self.risk_type.clone(),
            json!({"reason": reason, "language": language, "severity_level": severity}),
        )
    }

    /// Fallback object dialect: `risk_level` (or `level`) plus `risk_type`.
    fn parse_generic_object(&self, data: &Map<String, Value>) -> RiskAssessment {
        let level_value = data.get("risk_level").or_else(|| data.get("level"));
        let risk_level = self.map_risk_level(&string_or(level_value, ""));
        let risk_type = data
            .get("risk_type")
            .map_or_else(|| self.risk_type.clone(), string_value);
        self.assess(risk_level, risk_type, Value::Object(data.clone()))
    }

    /// Parses the standard `Safety:`/`Categories:` text format.
    ///
    /// Text with neither a `Safety:` verdict nor a `Categories:` list is
    /// unrecognized, which is an error rather than a safe grade.
    fn parse_standard_text(&self, text: &str) -> Result<RiskAssessment> {
        let safety = extract_safety(text);
        let categories = extract_categories(text);
        if safety == "unknown" && categories.is_empty() {
            return Err(unrecognized_output());
        }
        let risk_level = self.map_risk_level(&safety);
        let risk_type = categories
            .first()
            .cloned()
            .unwrap_or_else(|| self.risk_type.clone());
        Ok(self.assess(
            risk_level,
            risk_type,
            json!({"safety": safety, "categories": categories}),
        ))
    }

    /// Parses a string, first as a JSON object when one is embedded.
    fn parse_string(&self, text: &str) -> Result<RiskAssessment> {
        if let Some(data) = json_object_in(text) {
            return self.parse_object(&data);
        }
        self.parse_standard_text(text)
    }
}

/// The error an output outside every accepted dialect reports.
///
/// The message names the parser and the dialects it accepts and never echoes
/// the payload, so it is safe to log and to surface in a violation.
fn unrecognized_output() -> Error {
    Error::validation(
        "qwen_guard parser: unrecognized model output dialect (expected a safety/categories \
         object, an analysis object, a judgment object, a severity_level object, a risk_level \
         object, an embedded JSON object, or a 'Safety:'/'Categories:' text block)",
    )
}

impl ModelOutputParser for QwenGuardParser {
    // The trait fixes the return type at `&str`; the name is a literal.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "qwen_guard"
    }

    fn parse(&self, output: &Value) -> Result<RiskAssessment> {
        match output {
            Value::Object(data) => self.parse_object(data),
            Value::String(text) => self.parse_string(text),
            // Python stringifies every other value and reads it as text, which
            // accepts any scalar as a "safe" reply. A scalar is not a verdict
            // dialect, so it is rejected instead.
            _ => Err(unrecognized_output()),
        }
    }
}

/// Finds and parses a JSON object embedded in `text`.
///
/// Tries the whole trimmed string when it looks like an object, then the
/// substring between the first `{` and the last `}`.
fn json_object_in(text: &str) -> Option<Map<String, Value>> {
    let trimmed = text.trim();
    if trimmed.starts_with('{') || trimmed.ends_with('}') {
        if let Ok(Value::Object(data)) = serde_json::from_str::<Value>(trimmed) {
            return Some(data);
        }
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end > start {
        if let Ok(Value::Object(data)) = serde_json::from_str::<Value>(&trimmed[start..=end]) {
            return Some(data);
        }
    }
    None
}

/// Extracts the `Safety:` verdict from the standard text format.
///
/// Mirrors Python: the regex first, then a line keyed `safety:`, then the first
/// of `unsafe`/`controversial`/`safe` contained in the lowercase text, else
/// `"unknown"`.
fn extract_safety(text: &str) -> String {
    let captures = SAFETY_PATTERN.captures(text);
    if let Some(found) = captures.as_ref().and_then(|captures| captures.get(1)) {
        return found.as_str().to_owned();
    }
    if let Some(found) = safety_from_lines(text) {
        return found;
    }
    let lower = text.to_lowercase();
    for keyword in ["unsafe", "controversial", "safe"] {
        if lower.contains(keyword) {
            return keyword.to_owned();
        }
    }
    "unknown".to_owned()
}

/// Fallback for [`extract_safety`]: the first word after a `safety:` key.
fn safety_from_lines(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if let Some(index) = lower.find("safety:") {
            let value = &trimmed[(index + "safety:".len())..];
            if let Some(token) = value.split_whitespace().next() {
                return Some(token.to_owned());
            }
        }
    }
    None
}

/// Extracts the category list from the standard text format.
///
/// Mirrors Python: the `categories?` regex first, then a line keyed
/// `categories` or `category`; values are split on commas and whitespace.
fn extract_categories(text: &str) -> Vec<String> {
    let captures = CATEGORIES_PATTERN.captures(text);
    if let Some(found) = captures.as_ref().and_then(|captures| captures.get(1)) {
        let categories = split_categories(found.as_str());
        if !categories.is_empty() {
            return categories;
        }
    }
    categories_from_lines(text)
}

/// Fallback for [`extract_categories`]: a line keyed `categories`/`category`.
fn categories_from_lines(text: &str) -> Vec<String> {
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        if key == "categories" || key == "category" {
            let categories = split_categories(value);
            if !categories.is_empty() {
                return categories;
            }
        }
    }
    Vec::new()
}

/// Splits a comma- or whitespace-separated category list.
fn split_categories(text: &str) -> Vec<String> {
    text.split(|character: char| character == ',' || character.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Reads a category list that is either a JSON array or a delimited string.
fn categories_of(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .map(string_value)
            .filter(|item| !item.is_empty())
            .collect(),
        Some(Value::String(text)) => split_categories(text),
        _ => Vec::new(),
    }
}

/// Renders a JSON value as a string without failing.
fn string_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_owned(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// [`string_value`] for an optional value, falling back to `default`.
fn string_or(value: Option<&Value>, default: &str) -> String {
    value.map_or(default.to_owned(), string_value)
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    #[allow(clippy::needless_pass_by_value)]
    fn parse(parser: &QwenGuardParser, output: Value) -> Result<RiskAssessment> {
        parser.parse(&output)
    }

    fn details(assessment: &RiskAssessment) -> Value {
        assessment.details.clone().unwrap()
    }

    #[test]
    fn standard_text_unsafe() {
        // ports tests/unit_tests/core/security/guardrail/test_parsers_and_backends.py
        let parser = QwenGuardParser::new();
        let assessment = parse(&parser, json!("Safety: Unsafe\nCategories: Violent")).unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("Violent"));
        let details = details(&assessment);
        assert_eq!(details["safety"], json!("Unsafe"));
        assert_eq!(details["categories"], json!(["Violent"]));
    }

    #[test]
    fn standard_text_safe() {
        let parser = QwenGuardParser::new();
        let assessment = parse(&parser, json!("Safety: Safe\nCategories:")).unwrap();
        assert!(!assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
        assert_eq!(assessment.risk_type, None);
        let details = details(&assessment);
        assert_eq!(details["safety"], json!("Safe"));
        assert_eq!(details["categories"], json!([]));
    }

    #[test]
    fn standard_text_controversial_is_medium() {
        let parser = QwenGuardParser::new();
        let assessment = parse(
            &parser,
            json!("Safety: Controversial\nCategories: Political"),
        )
        .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Medium);
        assert_eq!(assessment.risk_type.as_deref(), Some("Political"));
    }

    #[test]
    fn standard_text_multiple_categories() {
        let parser = QwenGuardParser::new();
        let assessment = parse(
            &parser,
            json!("Safety: Unsafe\nCategories: Violent, Sexual, Hate"),
        )
        .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("Violent"));
        assert_eq!(
            details(&assessment)["categories"],
            json!(["Violent", "Sexual", "Hate"])
        );
    }

    #[test]
    fn standard_text_is_case_insensitive() {
        let parser = QwenGuardParser::new();
        let assessment = parse(&parser, json!("safety: unsafe\ncategories: violent")).unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(details(&assessment)["categories"], json!(["violent"]));
    }

    #[test]
    fn standard_text_keyword_fallback() {
        let parser = QwenGuardParser::new();
        let assessment = parse(
            &parser,
            json!("This content is unsafe and should be blocked"),
        )
        .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(details(&assessment)["safety"], json!("unsafe"));
        assert_eq!(assessment.risk_type.as_deref(), Some("content_risk"));
    }

    #[test]
    fn standard_text_without_markers_is_an_error() {
        let parser = QwenGuardParser::new();
        let error = parse(&parser, json!("Unknown format without keywords")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("unrecognized"), "{error}");
        assert!(error.to_string().contains("qwen_guard"), "{error}");
    }

    #[test]
    fn standard_text_category_only() {
        // A `Categories:` block with no `Safety:` verdict is a recognized reply
        // without a verdict, so it takes the fail-safe default level.
        let parser = QwenGuardParser::new();
        let assessment = parse(&parser, json!("category: violence, hate")).unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Critical);
        assert_eq!(assessment.risk_type.as_deref(), Some("violence"));
        assert_eq!(
            details(&assessment)["categories"],
            json!(["violence", "hate"])
        );
        assert_eq!(details(&assessment)["safety"], json!("unknown"));

        // Python's grading is the explicit opt-in.
        let python = QwenGuardParser::new().with_default_risk_level(RiskLevel::Safe);
        let allowed = parse(&python, json!("category: violence, hate")).unwrap();
        assert!(!allowed.has_risk);
        assert_eq!(allowed.risk_level, RiskLevel::Safe);
        assert_eq!(allowed.risk_type, None);
    }

    #[test]
    fn standard_object_dialect() {
        let parser = QwenGuardParser::new();
        let assessment = parse(
            &parser,
            json!({"safety": "Unsafe", "categories": ["Violent", "Hate"]}),
        )
        .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("Violent"));
        assert_eq!(
            details(&assessment)["categories"],
            json!(["Violent", "Hate"])
        );
    }

    #[test]
    fn standard_object_accepts_string_categories() {
        let parser = QwenGuardParser::new();
        let assessment = parse(
            &parser,
            json!({"safety": "Unsafe", "categories": "Violent, Sexual"}),
        )
        .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_type.as_deref(), Some("Violent"));
        assert_eq!(
            details(&assessment)["categories"],
            json!(["Violent", "Sexual"])
        );
    }

    #[test]
    fn standard_object_risky_without_categories_falls_back_to_risk_type() {
        let parser = QwenGuardParser::new().with_risk_type("content_moderation");
        let assessment = parse(&parser, json!({"safety": "Unsafe", "categories": []})).unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_type.as_deref(), Some("content_moderation"));
    }

    #[test]
    fn full_object_dialect() {
        let parser = QwenGuardParser::new();
        let assessment = parse(
            &parser,
            json!({
                "analysis": {
                    "risk_level": "unsafe",
                    "risk_categories": ["hate_speech"],
                    "evidence": "Test evidence",
                    "language": "zh",
                },
                "decision": "block",
                "version": "qwen3guard-gen-8b-v1.0",
            }),
        )
        .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("hate_speech"));
        let details = details(&assessment);
        assert_eq!(details["risk_categories"], json!(["hate_speech"]));
        assert_eq!(details["evidence"], json!("Test evidence"));
        assert_eq!(details["language"], json!("zh"));
        assert_eq!(details["decision"], json!("block"));
        assert_eq!(details["version"], json!("qwen3guard-gen-8b-v1.0"));
        assert!((assessment.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn full_object_defaults_missing_fields() {
        let parser = QwenGuardParser::new();
        let assessment = parse(&parser, json!({"analysis": {}})).unwrap();
        assert!(assessment.has_risk, "a missing verdict is not safe");
        assert_eq!(assessment.risk_level, RiskLevel::Critical);
        assert_eq!(assessment.risk_type.as_deref(), Some("content_risk"));
        let details = details(&assessment);
        assert_eq!(details["risk_categories"], json!([]));
        assert_eq!(details["evidence"], Value::Null);
        assert_eq!(details["language"], json!("unknown"));
        assert_eq!(details["decision"], json!("unknown"));
        assert_eq!(details["version"], json!("unknown"));
    }

    #[test]
    fn simple_object_dialect() {
        let parser = QwenGuardParser::new();
        let assessment = parse(
            &parser,
            json!({"judgment": "Unsafe", "reason": "Test reason", "language": "zh"}),
        )
        .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("content_risk"));
        let details = details(&assessment);
        assert_eq!(details["reason"], json!("Test reason"));
        assert_eq!(details["language"], json!("zh"));
        assert_eq!(details["judgment"], json!("Unsafe"));
    }

    #[test]
    fn api_object_dialect() {
        let parser = QwenGuardParser::new();
        let unsafe_assessment = parse(
            &parser,
            json!({"severity_level": "unsafe", "reason": "Test reason", "language": "zh"}),
        )
        .unwrap();
        assert!(unsafe_assessment.has_risk);
        assert_eq!(unsafe_assessment.risk_level, RiskLevel::High);
        assert_eq!(
            details(&unsafe_assessment)["severity_level"],
            json!("unsafe")
        );

        // `"HIGH"` matches the severity vocabulary case-insensitively.
        let high = parse(&parser, json!({"severity_level": "HIGH"})).unwrap();
        assert!(high.has_risk);
        assert_eq!(high.risk_level, RiskLevel::High);
        assert_eq!(details(&high)["severity_level"], json!("HIGH"));
    }

    #[test]
    fn generic_object_dialect() {
        let parser = QwenGuardParser::new();
        let assessment = parse(
            &parser,
            json!({"risk_level": "unsafe", "risk_type": "jailbreak", "score": 0.9}),
        )
        .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("jailbreak"));
        assert_eq!(
            details(&assessment),
            json!({
                "risk_level": "unsafe",
                "risk_type": "jailbreak",
                "score": 0.9,
            })
        );

        let level_alias = parse(&parser, json!({"level": "controversial"})).unwrap();
        assert_eq!(level_alias.risk_level, RiskLevel::Medium);
        assert_eq!(level_alias.risk_type.as_deref(), Some("content_risk"));

        let empty = parse(&parser, json!({}));
        let error = empty.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("unrecognized"), "{error}");
    }

    #[test]
    fn json_string_wrapping_a_standard_object() {
        let parser = QwenGuardParser::new();
        let whole = json!(r#"{"safety": "Unsafe", "categories": ["Violent"]}"#);
        let assessment = parse(&parser, whole).unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("Violent"));

        let embedded = json!(
            r#"The model said: {"safety": "Controversial", "categories": ["Political"]} -- done"#
        );
        let embedded_assessment = parse(&parser, embedded).unwrap();
        assert!(embedded_assessment.has_risk);
        assert_eq!(embedded_assessment.risk_level, RiskLevel::Medium);
        assert_eq!(embedded_assessment.risk_type.as_deref(), Some("Political"));
    }

    #[test]
    fn json_scalars_and_arrays_are_errors() {
        let parser = QwenGuardParser::new();
        for output in [
            json!(0.99),
            json!(42),
            json!([1, 2, 3]),
            json!([1]),
            json!(true),
            Value::Null,
        ] {
            let error = parse(&parser, output.clone()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{output}");
            assert!(error.to_string().contains("unrecognized"), "{output}");
        }
    }

    #[test]
    fn unrecognized_objects_and_text_are_errors() {
        let parser = QwenGuardParser::new();
        for output in [
            json!({}),
            json!({"unrelated": 1}),
            json!("this reply mentions no verdict"),
        ] {
            let error = parse(&parser, output.clone()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{output}");
            let message = error.to_string();
            assert!(message.contains("unrecognized"), "{output}: {message}");
            assert!(message.contains("qwen_guard"), "{output}: {message}");
            assert!(
                !message.contains("this reply mentions no verdict"),
                "the payload must not be echoed: {message}"
            );
        }

        // A recognized shape with an unknown *value* is not an error: it errs
        // upward at the `new()` default level.
        let fallback = parse(
            &QwenGuardParser::new(),
            json!({"safety": "totally-unheard-of"}),
        )
        .unwrap();
        assert!(fallback.has_risk);
        assert_eq!(fallback.risk_level, RiskLevel::Critical);

        // Python's grading of an unmappable verdict is the explicit opt-in.
        let python = parse(
            &QwenGuardParser::new().with_default_risk_level(RiskLevel::Safe),
            json!({"safety": "totally-unheard-of"}),
        )
        .unwrap();
        assert!(!python.has_risk);
        assert_eq!(python.risk_level, RiskLevel::Safe);
    }

    #[test]
    fn custom_default_risk_level_handles_unknown_verdicts() {
        let parser = QwenGuardParser::new().with_default_risk_level(RiskLevel::Medium);
        let assessment = parse(&parser, json!("Safety: no-such-verdict")).unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Medium);
        assert_eq!(assessment.risk_type.as_deref(), Some("content_risk"));

        // The severity words now map exactly, ahead of the configured default.
        for (verdict, expected) in [
            ("high", RiskLevel::High),
            ("critical", RiskLevel::Critical),
            ("low", RiskLevel::Low),
            ("medium", RiskLevel::Medium),
            ("safe", RiskLevel::Safe),
        ] {
            let explicit = parse(&parser, json!({"risk_level": verdict})).unwrap();
            assert_eq!(explicit.risk_level, expected, "{verdict}");
        }

        // A token outside the vocabulary still takes the configured default.
        let unknown = parse(&parser, json!({"risk_level": "elevated"})).unwrap();
        assert_eq!(unknown.risk_level, RiskLevel::Medium);
    }

    #[test]
    fn risk_level_mapping_prefers_longest_keyword() {
        let parser = QwenGuardParser::new();
        assert_eq!(parser.map_risk_level(" safe "), RiskLevel::Safe);
        assert_eq!(parser.map_risk_level("UNSAFE"), RiskLevel::High);
        assert_eq!(parser.map_risk_level("Controversial"), RiskLevel::Medium);
        // `"unsafe"` contains `"safe"`; the longer keyword must win.
        assert_eq!(parser.map_risk_level("this is unsafe"), RiskLevel::High);
        assert_eq!(
            parser.map_risk_level("controversial content"),
            RiskLevel::Medium
        );

        // The severity words are exact-match only, case-insensitively.
        assert_eq!(parser.map_risk_level("Critical"), RiskLevel::Critical);
        assert_eq!(parser.map_risk_level(" HIGH "), RiskLevel::High);
        assert_eq!(parser.map_risk_level("medium"), RiskLevel::Medium);
        assert_eq!(parser.map_risk_level("low"), RiskLevel::Low);
        // Not containment matches: `low` is a substring of `below`.
        assert_eq!(
            parser.map_risk_level("below the threshold"),
            RiskLevel::Critical
        );
        assert_eq!(parser.map_risk_level("perfectly fine"), RiskLevel::Critical);
    }

    #[test]
    fn constructor_defaults_with_a_fail_safe_verdict_level() {
        // `risk_type` matches Python; the verdict fallback is intentionally
        // `Critical` instead of Python's `Safe` so an unmappable token blocks.
        let parser = QwenGuardParser::default();
        assert_eq!(parser.risk_type, "content_risk");
        assert_eq!(parser.default_risk_level, RiskLevel::Critical);
        assert_eq!(
            QwenGuardParser::new().default_risk_level,
            RiskLevel::Critical
        );
    }

    #[test]
    fn an_unmappable_verdict_or_missing_verdict_is_not_safe() {
        let parser = QwenGuardParser::new();

        // Recognized dialects carrying an unknown token or no verdict at all.
        for output in [
            json!({"safety": "totally-unheard-of"}),
            json!({"analysis": {}}),
            json!({"judgment": null}),
            json!({"severity_level": ""}),
            json!({"risk_level": "elevated"}),
        ] {
            let assessment = parse(&parser, output.clone()).unwrap();
            assert!(assessment.has_risk, "{output} must not grade safe");
            assert_eq!(assessment.risk_level, RiskLevel::Critical, "{output}");
            assert_eq!(
                assessment.risk_type.as_deref(),
                Some("content_risk"),
                "{output}"
            );
        }

        // A `Categories:`-only text reply has no `Safety:` line either.
        let categories_only = parse(&parser, json!("Categories: Violent")).unwrap();
        assert!(categories_only.has_risk);
        assert_eq!(categories_only.risk_level, RiskLevel::Critical);
        assert_eq!(categories_only.risk_type.as_deref(), Some("Violent"));

        // A genuine safe verdict still grades safe and stays finding-free.
        for output in [
            json!({"safety": "safe"}),
            json!({"severity_level": "safe"}),
            json!({"analysis": {"risk_level": "safe"}}),
            json!("Safety: Safe"),
        ] {
            let assessment = parse(&parser, output.clone()).unwrap();
            assert!(!assessment.has_risk, "{output}");
            assert_eq!(assessment.risk_level, RiskLevel::Safe, "{output}");
            assert_eq!(assessment.risk_type, None, "{output}");
        }
    }
}
