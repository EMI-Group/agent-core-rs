//! Sensitive-data/PII detection with redaction.
//!
//! Detection patterns, validators, and the [`SensitiveDataGuardrail`] live here;
//! the guardrail is a [`Guardrail`] implementation whose
//! redaction primitive [`SensitiveDataGuardrail::redact_text`] is reused by
//! tool-argument validation.
//!
//! Python's guardrail package has no sensitive-data detector at all, so this
//! module has no `builtin.py` counterpart: it is one of the detectors the Python
//! package advertises but never shipped (see the crate `CONTEXT.md`). The
//! observable contract is Python's rule-backend contract: a match is a finding
//! at the pattern's severity, the content is redacted when the config asks for
//! it, and `modified_data` carries the sanitized text.
//!
//! # Matching semantics
//!
//! Patterns are compiled once, in table order, with
//! [`CompiledRule::compile`](crate::rules::CompiledRule::compile) in
//! case-sensitive mode. Within one pattern the non-overlapping matches come from
//! [`find_all`]; across patterns the earliest pattern in
//! the table wins, so a number matched by both `credit_card` and another pattern
//! is reported and redacted once. A candidate is dropped, never reported, when
//! its [`SensitiveValidator`] rejects it, when the exact matched substring is in
//! the config's `allowlist`, or when it overlaps a range an earlier pattern
//! already claimed.
//!
//! Findings never embed matched text: `details` carries `matched_patterns` (in
//! table order, deduplicated) and `match_count`, and the sanitized text is
//! exposed only through `modified_data`.
//!
//! # Example
//!
//! ```
//! use openjiuwen_security::guardrails::sensitive_data::SensitiveDataGuardrail;
//!
//! let guardrail = SensitiveDataGuardrail::new();
//! let outcome = guardrail.redact_text("我的邮箱是 user@example.com，谢谢");
//! assert_eq!(outcome.matches.len(), 1);
//! assert_eq!(outcome.matches[0].pattern, "email");
//! assert_eq!(outcome.redacted, "我的邮箱是 <REDACTED>，谢谢");
//! ```
#![allow(clippy::module_name_repetitions)]

use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use openjiuwen_common::error::Result;
use openjiuwen_common::security::REDACTED;

use crate::guardrail::{Guardrail, DEFAULT_PRIORITY};
use crate::rules::{find_all, luhn_valid, redact_ranges, CompiledRule};
use crate::{GuardrailContext, GuardrailResult, RiskLevel, ScanStage};

/// Stable guardrail name, mirrored in findings and logs.
const NAME: &str = "sensitive_data";

/// Risk type every detection reports.
const RISK_TYPE: &str = "sensitive_data";

/// Configuration for [`SensitiveDataGuardrail`].
///
/// Every field is serde data, so the config round-trips through JSON; unknown
/// fields are ignored and missing ones fall back to [`Default`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct SensitiveDataGuardrailConfig {
    /// Whether the *pipeline builder* installs this guardrail.
    ///
    /// The flag is consumed by `SecurityConfig::pipeline()`; the guardrail
    /// itself always scans, so a directly constructed
    /// [`SensitiveDataGuardrail`] never silently does nothing.
    pub enabled: bool,
    /// Severity used by a pattern that does not carry its own
    /// [`SensitivePattern::risk_level`].
    pub risk_level: RiskLevel,
    /// Whether [`SensitiveDataGuardrail::check`] exposes the sanitized text in
    /// `modified_data["sanitized_text"]`.
    pub redact: bool,
    /// Replacement written in place of every redacted range.
    pub marker: String,
    /// The pattern table, in matching and redaction order.
    pub patterns: Vec<SensitivePattern>,
    /// Additional patterns, compiled after [`Self::patterns`] so the default
    /// table keeps precedence.
    pub extra_patterns: Vec<SensitivePattern>,
    /// Matched substrings to leave untouched (exact, case-sensitive).
    pub allowlist: Vec<String>,
}

impl Default for SensitiveDataGuardrailConfig {
    /// Detection on, [`RiskLevel::High`] as the fallback severity, redaction
    /// with the shared [`REDACTED`] marker, and the default pattern table.
    fn default() -> Self {
        Self {
            enabled: true,
            risk_level: RiskLevel::High,
            redact: true,
            marker: REDACTED.to_owned(),
            patterns: default_patterns(),
            extra_patterns: Vec::new(),
            allowlist: Vec::new(),
        }
    }
}

/// One detection pattern of a [`SensitiveDataGuardrailConfig`].
///
/// The regex source is kept verbatim so the config round-trips and so
/// `details`/reports can name the pattern without the compiled object.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct SensitivePattern {
    /// Stable pattern name, surfaced in `details["matched_patterns"]`.
    pub name: String,
    /// Regular-expression source, compiled case-sensitively.
    pub pattern: String,
    /// Severity this pattern reports; `None` means
    /// [`SensitiveDataGuardrailConfig::risk_level`].
    pub risk_level: Option<RiskLevel>,
    /// Extra acceptance check applied to the matched substring.
    pub validator: Option<SensitiveValidator>,
}

/// Acceptance check applied to a matched substring.
///
/// A validator turns a syntactic match into a semantic one: a candidate the
/// validator rejects is not a match at all, so it is neither reported nor
/// redacted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitiveValidator {
    /// The Luhn checksum, which credit-card numbers satisfy; non-digit
    /// characters are ignored before the check.
    Luhn,
}

impl SensitiveValidator {
    /// Whether `matched` satisfies the check.
    #[must_use]
    fn accepts(self, matched: &str) -> bool {
        match self {
            Self::Luhn => luhn_valid(matched),
        }
    }
}

/// One accepted match, located by byte range in the scanned text.
///
/// The matched text itself is deliberately absent: a match record is safe to
/// log, serialize into a finding, or hand to a caller that must not see the
/// original content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SensitiveMatch {
    /// Name of the pattern that matched.
    pub pattern: String,
    /// Severity attributed to this match.
    pub risk_level: RiskLevel,
    /// Start byte offset in the scanned text (inclusive).
    pub start: usize,
    /// End byte offset in the scanned text (exclusive).
    pub end: usize,
}

/// Result of redacting one text: the sanitized text plus the accepted matches,
/// in table order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RedactionOutcome {
    /// The text with every accepted range replaced by the configured marker;
    /// identical to the input when nothing matched.
    pub redacted: String,
    /// Accepted, non-overlapping matches.
    pub matches: Vec<SensitiveMatch>,
}

impl RedactionOutcome {
    /// The names of the matched patterns, in table order, without duplicates.
    #[must_use]
    fn matched_patterns(&self) -> Vec<&str> {
        let mut names: Vec<&str> = Vec::new();
        for found in &self.matches {
            if !names.contains(&found.pattern.as_str()) {
                names.push(found.pattern.as_str());
            }
        }
        names
    }
}

/// One pattern compiled together with its resolved severity and validator.
#[derive(Debug, Clone)]
struct CompiledPattern {
    /// The compiled regex plus its name.
    rule: CompiledRule,
    /// Severity after applying the config fallback.
    risk_level: RiskLevel,
    /// Optional acceptance check.
    validator: Option<SensitiveValidator>,
}

impl CompiledPattern {
    /// Compiles `pattern`, resolving its severity against `fallback`.
    ///
    /// # Errors
    ///
    /// Propagates [`CompiledRule::compile`]'s validation error unchanged; its
    /// message already reads `invalid pattern '<name>': <error>`.
    fn compile(pattern: &SensitivePattern, fallback: RiskLevel) -> Result<Self> {
        Ok(Self {
            rule: CompiledRule::compile(pattern.name.clone(), &pattern.pattern, true)?,
            risk_level: pattern.risk_level.unwrap_or(fallback),
            validator: pattern.validator,
        })
    }

    /// Whether `matched` passes the pattern's validator (`true` without one).
    #[must_use]
    fn accepts(&self, matched: &str) -> bool {
        self.validator
            .is_none_or(|validator| validator.accepts(matched))
    }
}

/// The default detection table, in matching and redaction order.
///
/// A `credit_card` candidate additionally has to pass the Luhn checksum; every
/// other pattern is purely syntactic. Kept case-sensitive so an email address is
/// found exactly as written.
#[must_use]
fn default_patterns() -> Vec<SensitivePattern> {
    /// A full table row; the risk level and validator vary per row.
    fn row(
        name: &str,
        pattern: &str,
        risk_level: RiskLevel,
        validator: Option<SensitiveValidator>,
    ) -> SensitivePattern {
        SensitivePattern {
            name: name.to_owned(),
            pattern: pattern.to_owned(),
            risk_level: Some(risk_level),
            validator,
        }
    }

    vec![
        row(
            "email",
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            RiskLevel::High,
            None,
        ),
        row("cn_mobile", r"\b1[3-9]\d{9}\b", RiskLevel::Medium, None),
        row(
            "cn_id_card",
            r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
            RiskLevel::High,
            None,
        ),
        row(
            "credit_card",
            r"\b(?:\d[ -]?){12,18}\d\b",
            RiskLevel::High,
            Some(SensitiveValidator::Luhn),
        ),
        row("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", RiskLevel::High, None),
        row(
            "api_key",
            r"\b(?:sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36})\b",
            RiskLevel::Critical,
            None,
        ),
        row(
            "private_key",
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
            RiskLevel::Critical,
            None,
        ),
    ]
}

/// PII/secret detection guardrail.
///
/// The guardrail is stateless after construction: one instance can serve many
/// concurrent scans. It binds to every stage (empty [`Guardrail::stages`]) so a
/// pipeline decides where PII matters, and it never performs I/O, so
/// [`Guardrail::check`] is infallible in practice.
///
/// [`SensitiveDataGuardrailConfig::enabled`] is *not* consulted by
/// [`Guardrail::check`]: it exists so a pipeline builder can skip installing the
/// guardrail, and a directly constructed guardrail always scans.
pub struct SensitiveDataGuardrail {
    /// The configuration the guardrail was built from.
    config: SensitiveDataGuardrailConfig,
    /// `config.patterns` followed by `config.extra_patterns`.
    patterns: Vec<CompiledPattern>,
    /// Pipeline ordering weight.
    priority: i32,
    /// Stages to scan; empty means every stage.
    stages: Vec<ScanStage>,
}

impl SensitiveDataGuardrail {
    /// A guardrail over [`SensitiveDataGuardrailConfig::default`]: the default
    /// seven-pattern table with redaction enabled.
    ///
    /// The default table cannot fail to compile, so this constructor is
    /// infallible; use [`Self::with_config`] for user-supplied patterns.
    ///
    /// # Panics
    ///
    /// Never in practice: the built-in table is a set of literals covered by the
    /// module's tests. A panic would mean a built-in pattern was edited into an
    /// invalid regex, i.e. a programming error rather than bad user input.
    #[must_use]
    #[allow(clippy::expect_used)] // the built-in pattern table is statically valid.
    pub fn new() -> Self {
        Self::with_config(SensitiveDataGuardrailConfig::default())
            .expect("the built-in detection table is always a valid regex")
    }

    /// Compiles `config.patterns`, then `config.extra_patterns`, in order.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when a
    /// pattern is not a valid regular expression, with
    /// [`CompiledRule::compile`]'s message (`invalid pattern '<name>': <error>`).
    pub fn with_config(config: SensitiveDataGuardrailConfig) -> Result<Self> {
        let mut patterns = Vec::with_capacity(config.patterns.len() + config.extra_patterns.len());
        for pattern in config.patterns.iter().chain(config.extra_patterns.iter()) {
            patterns.push(CompiledPattern::compile(pattern, config.risk_level)?);
        }
        Ok(Self {
            config,
            patterns,
            priority: DEFAULT_PRIORITY,
            stages: Vec::new(),
        })
    }

    /// Restricts the guardrail to `stages`; an empty list scans every stage.
    #[must_use]
    pub fn with_stages(mut self, stages: impl IntoIterator<Item = ScanStage>) -> Self {
        self.stages = stages.into_iter().collect();
        self
    }

    /// Overrides the pipeline priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// The configuration this guardrail was built from.
    #[must_use]
    pub fn config(&self) -> &SensitiveDataGuardrailConfig {
        &self.config
    }

    /// Finds and redacts every accepted match in `text`.
    ///
    /// Patterns are applied in table order; within a pattern the matches are the
    /// non-overlapping ones [`find_all`] returns. A match is dropped when its
    /// validator rejects it, when the exact matched substring is allowlisted, or
    /// when it overlaps a range an earlier pattern already claimed. The result's
    /// `redacted` field equals `text` and `matches` is empty when nothing
    /// matched.
    #[must_use]
    pub fn redact_text(&self, text: &str) -> RedactionOutcome {
        let mut claimed: Vec<(usize, usize)> = Vec::new();
        let mut matches = Vec::new();
        for pattern in &self.patterns {
            let rules = std::slice::from_ref(&pattern.rule);
            for found in find_all(rules, text) {
                let Some(token) = text.get(found.start..found.end) else {
                    continue;
                };
                if !pattern.accepts(token) {
                    continue;
                }
                if self.config.allowlist.iter().any(|entry| entry == token) {
                    continue;
                }
                let overlaps = claimed
                    .iter()
                    .any(|&(start, end)| found.start < end && start < found.end);
                if overlaps {
                    continue;
                }
                claimed.push((found.start, found.end));
                matches.push(SensitiveMatch {
                    pattern: pattern.rule.name.clone(),
                    risk_level: pattern.risk_level,
                    start: found.start,
                    end: found.end,
                });
            }
        }
        RedactionOutcome {
            redacted: redact_ranges(text, &claimed, &self.config.marker),
            matches,
        }
    }
}

impl Default for SensitiveDataGuardrail {
    /// Same as [`SensitiveDataGuardrail::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SensitiveDataGuardrail {
    /// Reports the name, the pattern names, priority, stages, and the redaction
    /// settings; scanned content is never printed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self
            .patterns
            .iter()
            .map(|pattern| pattern.rule.name.as_str())
            .collect();
        f.debug_struct("SensitiveDataGuardrail")
            .field("name", &NAME)
            .field("patterns", &names)
            .field("priority", &self.priority)
            .field("stages", &self.stages)
            .field("redact", &self.config.redact)
            .field("marker", &self.config.marker)
            .finish()
    }
}

#[async_trait]
impl Guardrail for SensitiveDataGuardrail {
    // The trait fixes the return type at `&str`; the name is a literal.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        NAME
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn stages(&self) -> &[ScanStage] {
        &self.stages
    }

    async fn check(&self, ctx: &GuardrailContext) -> Result<GuardrailResult> {
        // An unscannable payload is a typed failure, not a pass: the pipeline's
        // failure policy decides whether it blocks.
        let text = ctx.text_view_checked()?;
        if text.trim().is_empty() {
            return Ok(GuardrailResult::pass());
        }
        let outcome = self.redact_text(&text);
        if outcome.matches.is_empty() {
            return Ok(GuardrailResult::pass());
        }
        // Every match carries an attributed severity, so `max` is `Some` here;
        // the fallback keeps the function total without an unwrap.
        let risk_level = outcome
            .matches
            .iter()
            .map(|found| found.risk_level)
            .max()
            .unwrap_or(self.config.risk_level);
        let details = json!({
            "matched_patterns": outcome.matched_patterns(),
            "match_count": outcome.matches.len(),
        });
        let modified_data = if self.config.redact {
            Some(json!({ "sanitized_text": outcome.redacted }))
        } else {
            None
        };
        Ok(GuardrailResult::block(
            risk_level,
            RISK_TYPE,
            Some(details),
            modified_data,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::GuardrailContentType;

    fn guardrail(config: SensitiveDataGuardrailConfig) -> SensitiveDataGuardrail {
        SensitiveDataGuardrail::with_config(config).unwrap()
    }

    fn pattern(name: &str, source: &str) -> SensitivePattern {
        SensitivePattern {
            name: name.to_owned(),
            pattern: source.to_owned(),
            risk_level: None,
            validator: None,
        }
    }

    fn text_ctx(text: &str) -> GuardrailContext {
        GuardrailContext::text(text, ScanStage::UserInput)
    }

    #[test]
    fn default_table_keeps_documented_order_levels_and_validators() {
        let config = SensitiveDataGuardrailConfig::default();
        let described: Vec<(&str, RiskLevel, Option<SensitiveValidator>)> = config
            .patterns
            .iter()
            .map(|entry| {
                (
                    entry.name.as_str(),
                    entry.risk_level.unwrap(),
                    entry.validator,
                )
            })
            .collect();
        assert_eq!(
            described,
            vec![
                ("email", RiskLevel::High, None),
                ("cn_mobile", RiskLevel::Medium, None),
                ("cn_id_card", RiskLevel::High, None),
                (
                    "credit_card",
                    RiskLevel::High,
                    Some(SensitiveValidator::Luhn)
                ),
                ("us_ssn", RiskLevel::High, None),
                ("api_key", RiskLevel::Critical, None),
                ("private_key", RiskLevel::Critical, None),
            ]
        );

        assert!(config.enabled);
        assert_eq!(config.risk_level, RiskLevel::High);
        assert!(config.redact);
        assert_eq!(config.marker, REDACTED);
        assert_eq!(config.extra_patterns.len(), 0);
        assert_eq!(config.allowlist.len(), 0);

        let empty: SensitiveDataGuardrailConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, config, "serde defaults agree with Default");

        let round_trip: SensitiveDataGuardrailConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(round_trip, config);
        assert_eq!(
            serde_json::to_value(SensitiveValidator::Luhn).unwrap(),
            json!("luhn")
        );
    }

    #[tokio::test]
    async fn luhn_gates_credit_card_candidates() {
        let guardrail = SensitiveDataGuardrail::new();
        let valid = guardrail
            .check(&text_ctx("card: 4111 1111 1111 1111"))
            .await
            .unwrap();
        assert!(!valid.is_safe());
        assert_eq!(valid.risk_level, RiskLevel::High);
        assert_eq!(valid.risk_type.as_deref(), Some("sensitive_data"));
        assert_eq!(
            valid.details,
            Some(json!({"matched_patterns": ["credit_card"], "match_count": 1}))
        );
        assert_eq!(
            valid.modified_data,
            Some(json!({"sanitized_text": "card: <REDACTED>"}))
        );

        let invalid = guardrail
            .check(&text_ctx("card: 4111 1111 1111 1112"))
            .await
            .unwrap();
        assert!(invalid.is_safe(), "a failed Luhn check is not a match");
        assert_eq!(invalid.details, None);
        assert_eq!(invalid.modified_data, None);
    }

    #[tokio::test]
    async fn overlapping_patterns_report_and_redact_the_earlier_one_only() {
        let guardrail = guardrail(SensitiveDataGuardrailConfig {
            patterns: vec![pattern("wide", "abc"), pattern("narrow", "bc")],
            ..SensitiveDataGuardrailConfig::default()
        });
        let outcome = guardrail.redact_text("abc abc");
        assert_eq!(
            outcome.matches,
            vec![
                SensitiveMatch {
                    pattern: "wide".to_owned(),
                    risk_level: RiskLevel::High,
                    start: 0,
                    end: 3,
                },
                SensitiveMatch {
                    pattern: "wide".to_owned(),
                    risk_level: RiskLevel::High,
                    start: 4,
                    end: 7,
                },
            ]
        );
        assert_eq!(outcome.redacted, "<REDACTED> <REDACTED>");

        let result = guardrail.check(&text_ctx("abc abc")).await.unwrap();
        assert_eq!(
            result.details,
            Some(json!({"matched_patterns": ["wide"], "match_count": 2})),
            "pattern names are deduplicated in table order"
        );
    }

    #[test]
    fn multibyte_text_keeps_surrounding_characters_intact() {
        let guardrail = SensitiveDataGuardrail::new();
        let text = "联系人邮箱：user@example.com 谢谢";
        let outcome = guardrail.redact_text(text);
        assert_eq!(outcome.matches.len(), 1);
        let found = &outcome.matches[0];
        assert_eq!(found.pattern, "email");
        assert_eq!(&text[found.start..found.end], "user@example.com");
        assert_eq!(outcome.redacted, "联系人邮箱：<REDACTED> 谢谢");
    }

    #[tokio::test]
    async fn allowlist_skips_an_exact_matched_substring() {
        let guardrail = guardrail(SensitiveDataGuardrailConfig {
            allowlist: vec!["user@example.com".to_owned()],
            ..SensitiveDataGuardrailConfig::default()
        });
        let text = "reach me at user@example.com today";
        let outcome = guardrail.redact_text(text);
        assert_eq!(outcome.matches.len(), 0);
        assert_eq!(outcome.redacted, text);

        let result = guardrail.check(&text_ctx(text)).await.unwrap();
        assert!(result.is_safe());
    }

    #[test]
    fn marker_override_replaces_the_default_marker() {
        let guardrail = guardrail(SensitiveDataGuardrailConfig {
            marker: "[hidden]".to_owned(),
            ..SensitiveDataGuardrailConfig::default()
        });
        let outcome = guardrail.redact_text("call me at 13800138000");
        assert_eq!(outcome.matches.len(), 1);
        assert_eq!(outcome.matches[0].pattern, "cn_mobile");
        assert_eq!(outcome.matches[0].risk_level, RiskLevel::Medium);
        assert_eq!(outcome.redacted, "call me at [hidden]");
    }

    #[tokio::test]
    async fn redact_false_reports_the_finding_without_sanitized_text() {
        let guardrail = guardrail(SensitiveDataGuardrailConfig {
            redact: false,
            ..SensitiveDataGuardrailConfig::default()
        });
        let result = guardrail
            .check(&text_ctx("key sk-abcdefghijklmnopqrst"))
            .await
            .unwrap();
        assert!(!result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Critical);
        assert_eq!(
            result.details,
            Some(json!({"matched_patterns": ["api_key"], "match_count": 1}))
        );
        assert_eq!(result.modified_data, None);
    }

    #[tokio::test]
    async fn empty_and_benign_text_pass() {
        let guardrail = SensitiveDataGuardrail::new();
        for text in ["", "   ", "hello there, nothing secret"] {
            let result = guardrail.check(&text_ctx(text)).await.unwrap();
            assert!(result.is_safe(), "{text:?} must pass");
            assert_eq!(result.details, None);
            assert_eq!(result.modified_data, None);
        }
        assert_eq!(guardrail.redact_text("").matches.len(), 0);
    }

    #[tokio::test]
    async fn unscannable_payload_is_a_validation_error() {
        let guardrail = SensitiveDataGuardrail::new();
        let error = guardrail
            .check(&GuardrailContext::raw(
                "user_input",
                serde_json::Value::Null,
            ))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("unscannable content"), "{error}");

        // A scannable empty text still passes: nothing to find is not a
        // failure.
        let empty = guardrail.check(&text_ctx("")).await.unwrap();
        assert!(empty.is_safe());
        assert_eq!(empty.risk_level, RiskLevel::Safe);
    }

    #[test]
    fn extra_patterns_are_compiled_after_the_table() {
        let guardrail = guardrail(SensitiveDataGuardrailConfig {
            patterns: vec![pattern("first", "x+")],
            extra_patterns: vec![pattern("second", "xyz")],
            ..SensitiveDataGuardrailConfig::default()
        });
        let outcome = guardrail.redact_text("xyz");
        assert_eq!(outcome.matches.len(), 1);
        assert_eq!(
            outcome.matches[0].pattern, "first",
            "the table keeps precedence over extras"
        );
    }

    #[test]
    fn invalid_pattern_is_a_validation_error() {
        let error = SensitiveDataGuardrail::with_config(SensitiveDataGuardrailConfig {
            extra_patterns: vec![pattern("broken", "([a-z")],
            ..SensitiveDataGuardrailConfig::default()
        })
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid pattern 'broken'"),
            "{error}"
        );
    }

    #[test]
    fn trait_shape_and_builders() {
        let guardrail = SensitiveDataGuardrail::new();
        assert_eq!(guardrail.name(), "sensitive_data");
        assert_eq!(guardrail.priority(), DEFAULT_PRIORITY);
        assert_eq!(guardrail.stages().len(), 0);
        assert!(guardrail.handles(ScanStage::LlmInvokeOutput));
        assert_eq!(guardrail.config(), &SensitiveDataGuardrailConfig::default());

        let scoped = SensitiveDataGuardrail::new()
            .with_stages([ScanStage::ToolInvokeInput])
            .with_priority(7);
        assert_eq!(scoped.priority(), 7);
        assert_eq!(scoped.stages(), &[ScanStage::ToolInvokeInput]);
        assert!(!scoped.handles(ScanStage::UserInput));

        let debug = format!("{scoped:?}");
        assert!(debug.contains("SensitiveDataGuardrail"), "{debug}");
        assert!(debug.contains("credit_card"), "{debug}");
        assert!(debug.contains("marker"), "{debug}");
        assert!(
            !debug.contains("sk-abcdefghijklmnopqrst"),
            "content is never printed"
        );
    }

    #[tokio::test]
    async fn tool_call_contexts_are_scanned_through_text_view() {
        let guardrail = SensitiveDataGuardrail::new();
        let ctx = GuardrailContext::tool_output(
            "lookup",
            &json!({"owner": "user@example.com"}),
            ScanStage::ToolInvokeOutput,
        );
        assert_eq!(ctx.content_type, GuardrailContentType::ToolCall);
        let result = guardrail.check(&ctx).await.unwrap();
        assert!(!result.is_safe());
        let sanitized = result.modified_data.unwrap();
        assert!(
            !sanitized["sanitized_text"]
                .as_str()
                .unwrap()
                .contains("user@"),
            "{sanitized}"
        );
        assert!(sanitized["sanitized_text"]
            .as_str()
            .unwrap()
            .contains(REDACTED));
    }
}
