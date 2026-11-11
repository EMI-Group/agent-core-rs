//! Configurable regex content rules.
//!
//! Ports Python's rule-backend semantics into a standalone [`ContentRuleGuardrail`]:
//! an ordered rule table where the first matching rule decides the verdict.
//!
//! Python has no content-rule guardrail of its own — its rule backend only served
//! the six fixed prompt-injection patterns — so the rule type, its config, and the
//! per-rule risk metadata are an addition. The matching semantics are Python's:
//! rules run in configuration order, the first match produces the verdict, and
//! later rules are not consulted.
//!
//! Every pattern is compiled once at construction; a malformed pattern is a
//! construction error, never a panic and never a silently skipped rule.
//!
//! # Example
//!
//! ```
//! use openjiuwen_security::guardrails::content_rules::{ContentRule, ContentRuleGuardrail};
//! use openjiuwen_security::{Guardrail, GuardrailContext, RiskLevel, ScanStage};
//!
//! let rule = ContentRule {
//!     name: "secret_leak".to_owned(),
//!     pattern: r"leak \w+".to_owned(),
//!     risk_type: "secret_leak".to_owned(),
//!     risk_level: RiskLevel::Critical,
//!     message: Some("do not leak secrets".to_owned()),
//! };
//! let guardrail = ContentRuleGuardrail::from_patterns(vec![rule]).expect("a valid regex");
//!
//! let runtime = tokio::runtime::Builder::new_current_thread()
//!     .enable_all()
//!     .build()
//!     .expect("a tokio runtime");
//! let ctx = GuardrailContext::text("please leak passwords", ScanStage::LlmInvokeInput);
//! let result = runtime.block_on(guardrail.check(&ctx)).expect("a verdict");
//! assert!(!result.is_safe());
//! assert_eq!(result.risk_type.as_deref(), Some("secret_leak"));
//! ```
#![allow(clippy::module_name_repetitions)]

use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use openjiuwen_common::error::Result;
use openjiuwen_common::security::REDACTED;

use crate::guardrail::{Guardrail, DEFAULT_PRIORITY};
use crate::rules::{redact_ranges, CompiledRule};
use crate::{GuardrailContext, GuardrailResult, RiskLevel, ScanStage};

/// Stable guardrail name, mirrored in findings and logs.
const NAME: &str = "content_rules";

/// One user-supplied content rule.
///
/// The rule is pure configuration: [`ContentRuleGuardrail`] compiles
/// [`ContentRule::pattern`] once and reports [`ContentRule::risk_type`] and
/// [`ContentRule::risk_level`] when it matches. Deserialization is tolerant —
/// every field falls back to its [`Default`] value, so a rule written without
/// `risk_level` reports [`RiskLevel::High`] — and unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct ContentRule {
    /// Stable rule name, surfaced in `GuardrailResult::details` and in
    /// construction errors.
    pub name: String,
    /// The regular expression to search for, as written by the caller.
    pub pattern: String,
    /// Machine-readable risk name reported on a match
    /// (`"banned_word"`, `"profanity"`, ...).
    pub risk_type: String,
    /// Severity reported on a match; [`RiskLevel::High`] by default.
    pub risk_level: RiskLevel,
    /// Optional human-readable explanation carried for configuration parity.
    ///
    /// It is deliberately *not* copied into `GuardrailResult::details`: details
    /// stay machine-readable and content-free (rule name and pattern only).
    pub message: Option<String>,
}

impl Default for ContentRule {
    /// An empty rule at [`RiskLevel::High`] without a message.
    fn default() -> Self {
        Self {
            name: String::new(),
            pattern: String::new(),
            risk_type: String::new(),
            risk_level: RiskLevel::High,
            message: None,
        }
    }
}

/// Configuration for [`ContentRuleGuardrail`].
///
/// Deserialization is tolerant: every field falls back to its [`Default`] value
/// and unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct ContentRuleGuardrailConfig {
    /// The rule table in evaluation order; an empty table never blocks.
    pub rules: Vec<ContentRule>,
    /// Whether matching is case-sensitive; `false` (the default) matches
    /// case-insensitively.
    pub case_sensitive: bool,
    /// Whether a match proposes sanitized content in
    /// `GuardrailResult::modified_data` under `sanitized_text`.
    pub redact: bool,
    /// Replacement text for the matched span when [`Self::redact`] is set;
    /// defaults to [`REDACTED`] (`"<REDACTED>"`).
    pub marker: String,
}

impl Default for ContentRuleGuardrailConfig {
    /// No rules, case-insensitive matching, no redaction, [`REDACTED`] marker.
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            case_sensitive: false,
            redact: false,
            marker: REDACTED.to_owned(),
        }
    }
}

/// One config rule with its compiled pattern.
struct CompiledContentRule {
    /// Name, pattern source, compiled regex, and severity.
    compiled: CompiledRule,
    /// Machine-readable risk name reported on a match.
    risk_type: String,
}

impl CompiledContentRule {
    /// The severity this rule reports.
    ///
    /// Construction always attaches `ContentRule::risk_level`, so
    /// [`RiskLevel::High`] (the rule default) is only a fallback.
    fn risk_level(&self) -> RiskLevel {
        self.compiled.risk_level.unwrap_or(RiskLevel::High)
    }
}

/// A guardrail over an ordered table of user-supplied regex rules.
///
/// The first rule that matches [`GuardrailContext::text_view`] decides the
/// verdict, so put the most severe or most specific rules first. The guardrail
/// scans every stage by default (an empty [`Guardrail::stages`] list) and is
/// stateless after construction: one instance can serve many concurrent scans.
pub struct ContentRuleGuardrail {
    rules: Vec<CompiledContentRule>,
    config: ContentRuleGuardrailConfig,
    priority: i32,
    stages: Vec<ScanStage>,
}

impl ContentRuleGuardrail {
    /// Compiles every rule in `config.rules`, in configuration order.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when a rule
    /// pattern is not a valid regular expression; the message reads
    /// `invalid pattern '<name>': <regex error>` and names the offending rule.
    pub fn new(config: ContentRuleGuardrailConfig) -> Result<Self> {
        let mut rules = Vec::with_capacity(config.rules.len());
        for rule in &config.rules {
            let compiled =
                CompiledRule::compile(rule.name.clone(), &rule.pattern, config.case_sensitive)?
                    .with_risk_level(rule.risk_level);
            rules.push(CompiledContentRule {
                compiled,
                risk_type: rule.risk_type.clone(),
            });
        }
        Ok(Self {
            rules,
            config,
            priority: DEFAULT_PRIORITY,
            stages: Vec::new(),
        })
    }

    /// A guardrail over `rules` with the remaining defaults: case-insensitive
    /// matching and no redaction.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`]: a rule pattern that is not a valid regular
    /// expression is a validation error.
    pub fn from_patterns(rules: Vec<ContentRule>) -> Result<Self> {
        Self::new(ContentRuleGuardrailConfig {
            rules,
            ..ContentRuleGuardrailConfig::default()
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
    pub fn config(&self) -> &ContentRuleGuardrailConfig {
        &self.config
    }
}

impl fmt::Debug for ContentRuleGuardrail {
    /// Reports the name, rule names, priority, and matching settings; scanned
    /// content is never printed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rules: Vec<&str> = self
            .rules
            .iter()
            .map(|rule| rule.compiled.name.as_str())
            .collect();
        f.debug_struct("ContentRuleGuardrail")
            .field("name", &NAME)
            .field("rules", &rules)
            .field("priority", &self.priority)
            .field("stages", &self.stages)
            .field("case_sensitive", &self.config.case_sensitive)
            .field("redact", &self.config.redact)
            .finish()
    }
}

#[async_trait]
impl Guardrail for ContentRuleGuardrail {
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
        let text = ctx.text_view();
        if text.trim().is_empty() || self.rules.is_empty() {
            return Ok(GuardrailResult::pass());
        }
        for rule in &self.rules {
            let Some((start, end)) = rule.compiled.find(&text) else {
                continue;
            };
            let modified = if self.config.redact {
                Some(json!({
                    "sanitized_text": redact_ranges(&text, &[(start, end)], &self.config.marker),
                }))
            } else {
                None
            };
            return Ok(GuardrailResult::block(
                rule.risk_level(),
                rule.risk_type.clone(),
                Some(json!({
                    "rule": rule.compiled.name,
                    "matched_pattern": rule.compiled.pattern,
                })),
                modified,
            ));
        }
        Ok(GuardrailResult::pass())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    fn text_ctx(text: &str) -> GuardrailContext {
        GuardrailContext::text(text, ScanStage::LlmInvokeInput)
    }

    fn rule(name: &str, pattern: &str, risk_type: &str) -> ContentRule {
        ContentRule {
            name: name.to_owned(),
            pattern: pattern.to_owned(),
            risk_type: risk_type.to_owned(),
            ..ContentRule::default()
        }
    }

    #[tokio::test]
    async fn first_matching_rule_decides() {
        let guardrail = ContentRuleGuardrail::from_patterns(vec![
            ContentRule {
                risk_level: RiskLevel::Critical,
                ..rule("first", r"secret", "first_hit")
            },
            ContentRule {
                risk_level: RiskLevel::Low,
                ..rule("second", r"secret", "second_hit")
            },
        ])
        .unwrap();

        let result = guardrail.check(&text_ctx("a secret plan")).await.unwrap();
        assert!(!result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Critical);
        assert_eq!(result.risk_type.as_deref(), Some("first_hit"));
        assert_eq!(
            result.details,
            Some(json!({"rule": "first", "matched_pattern": "secret"}))
        );
        assert_eq!(result.modified_data, None);
    }

    #[tokio::test]
    async fn later_rules_still_match_when_the_first_does_not() {
        let guardrail = ContentRuleGuardrail::from_patterns(vec![
            rule("first", r"secret", "first_hit"),
            rule("second", r"forbidden", "second_hit"),
        ])
        .unwrap();

        let result = guardrail.check(&text_ctx("forbidden")).await.unwrap();
        assert!(!result.is_safe());
        assert_eq!(result.risk_type.as_deref(), Some("second_hit"));
        assert_eq!(
            result.details,
            Some(json!({"rule": "second", "matched_pattern": "forbidden"}))
        );
        assert_eq!(
            result.risk_level,
            RiskLevel::High,
            "the rule default severity is reported"
        );
    }

    #[tokio::test]
    async fn case_insensitive_by_default_and_case_sensitive_when_configured() {
        let rules = vec![rule("banned", "forbidden", "banned_word")];

        let insensitive = ContentRuleGuardrail::from_patterns(rules.clone()).unwrap();
        let hit = insensitive.check(&text_ctx("FORBIDDEN")).await.unwrap();
        assert!(!hit.is_safe());
        assert_eq!(hit.risk_type.as_deref(), Some("banned_word"));

        let sensitive = ContentRuleGuardrail::new(ContentRuleGuardrailConfig {
            rules,
            case_sensitive: true,
            ..ContentRuleGuardrailConfig::default()
        })
        .unwrap();
        assert!(sensitive
            .check(&text_ctx("FORBIDDEN"))
            .await
            .unwrap()
            .is_safe());
        assert!(!sensitive
            .check(&text_ctx("forbidden"))
            .await
            .unwrap()
            .is_safe());
    }

    #[tokio::test]
    async fn redact_replaces_the_matched_span_with_the_marker() {
        let configured = ContentRuleGuardrail::new(ContentRuleGuardrailConfig {
            rules: vec![rule("secret", "secret", "secret_leak")],
            redact: true,
            marker: "[gone]".to_owned(),
            ..ContentRuleGuardrailConfig::default()
        })
        .unwrap();

        let result = configured
            .check(&text_ctx("say the secret now"))
            .await
            .unwrap();
        assert_eq!(
            result.modified_data,
            Some(json!({"sanitized_text": "say the [gone] now"}))
        );

        let default_marker = ContentRuleGuardrail::new(ContentRuleGuardrailConfig {
            rules: vec![rule("secret", "secret", "secret_leak")],
            redact: true,
            ..ContentRuleGuardrailConfig::default()
        })
        .unwrap();
        let result = default_marker
            .check(&text_ctx("say the secret now"))
            .await
            .unwrap();
        assert_eq!(
            result.modified_data,
            Some(json!({"sanitized_text": "say the <REDACTED> now"}))
        );
    }

    #[tokio::test]
    async fn no_rules_always_pass() {
        let guardrail = ContentRuleGuardrail::new(ContentRuleGuardrailConfig::default()).unwrap();
        let result = guardrail.check(&text_ctx("anything at all")).await.unwrap();
        assert!(result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Safe);
        assert_eq!(result.risk_type, None);
        assert_eq!(result.details, None);
        assert_eq!(result.modified_data, None);
    }

    #[tokio::test]
    async fn empty_or_whitespace_text_passes_without_matching() {
        let guardrail =
            ContentRuleGuardrail::from_patterns(vec![rule("any", r".*", "any")]).unwrap();
        for text in ["", "   ", "\n\t"] {
            let result = guardrail.check(&text_ctx(text)).await.unwrap();
            assert!(result.is_safe(), "{text:?} must pass");
        }
    }

    #[test]
    fn invalid_pattern_is_a_validation_error() {
        let error = ContentRuleGuardrail::from_patterns(vec![rule("broken", "([a-z", "invalid")])
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid pattern 'broken'"),
            "{error}"
        );
    }

    #[test]
    fn builders_override_stages_and_priority() {
        let guardrail = ContentRuleGuardrail::new(ContentRuleGuardrailConfig::default()).unwrap();

        let configured = ContentRuleGuardrail::from_patterns(vec![rule("a", "a", "a")])
            .unwrap()
            .with_stages([ScanStage::UserInput])
            .with_priority(7);
        assert_eq!(configured.priority(), 7);
        assert_eq!(configured.stages(), &[ScanStage::UserInput]);
        assert!(configured.handles(ScanStage::UserInput));
        assert!(!configured.handles(ScanStage::LlmInvokeInput));
        assert_eq!(configured.name(), "content_rules");

        assert_eq!(guardrail.priority(), DEFAULT_PRIORITY);
        assert!(
            guardrail.stages().is_empty(),
            "empty stages scan every stage"
        );
    }

    #[test]
    fn config_defaults_are_preserved_through_serde() {
        assert_eq!(
            ContentRuleGuardrailConfig::default(),
            ContentRuleGuardrailConfig {
                rules: Vec::new(),
                case_sensitive: false,
                redact: false,
                marker: "<REDACTED>".to_owned(),
            }
        );
        assert_eq!(
            ContentRule::default(),
            ContentRule {
                name: String::new(),
                pattern: String::new(),
                risk_type: String::new(),
                risk_level: RiskLevel::High,
                message: None,
            }
        );

        let rule: ContentRule = serde_json::from_value(json!({
            "name": "banned",
            "pattern": "forbidden",
            "risk_type": "banned_word",
            "future_field": true,
        }))
        .unwrap();
        assert_eq!(rule.risk_level, RiskLevel::High, "missing level defaults");
        assert_eq!(rule.message, None);

        let config = ContentRuleGuardrailConfig {
            rules: vec![ContentRule {
                message: Some("no swearing".to_owned()),
                ..rule
            }],
            case_sensitive: true,
            redact: true,
            marker: "[gone]".to_owned(),
        };
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["rules"][0]["risk_level"], json!("high"));
        assert_eq!(value["rules"][0]["message"], json!("no swearing"));
        assert_eq!(value["case_sensitive"], json!(true));
        assert_eq!(value["marker"], json!("[gone]"));
        let back: ContentRuleGuardrailConfig = serde_json::from_value(value).unwrap();
        assert_eq!(back, config);

        let empty: ContentRuleGuardrailConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, ContentRuleGuardrailConfig::default());
    }

    #[test]
    fn debug_reports_shape_never_content() {
        let guardrail = ContentRuleGuardrail::from_patterns(vec![
            rule("first", r"secret", "first_hit"),
            rule("second", r"forbidden", "second_hit"),
        ])
        .unwrap();
        let debug = format!("{guardrail:?}");
        assert!(debug.contains("ContentRuleGuardrail"), "{debug}");
        assert!(debug.contains("content_rules"), "{debug}");
        assert!(debug.contains("first"), "{debug}");
        assert!(debug.contains("case_sensitive"), "{debug}");
        assert!(debug.contains("redact"), "{debug}");
        assert!(!debug.contains("secret"), "{debug}");

        let owner = ContentRuleGuardrail::new(ContentRuleGuardrailConfig {
            rules: vec![rule("secret", "say the secret", "secret_leak")],
            redact: true,
            ..ContentRuleGuardrailConfig::default()
        })
        .unwrap();
        let debug = format!("{owner:?}");
        assert!(!debug.contains("say the secret"), "{debug}");
    }
}
