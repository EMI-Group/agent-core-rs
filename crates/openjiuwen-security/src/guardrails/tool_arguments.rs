//! Tool-call argument validation.
//!
//! Allow/deny lists, argument deny patterns, size limits, and nested PII
//! redaction, bound to the tool-input stage.
//!
//! The Python reference has no tool-argument guardrail; this detector is an
//! addition whose semantics follow the crate's rule-backend contract: the first
//! failing check produces the verdict, `details` never embeds the matched text,
//! and sanitized content is exposed only through `modified_data`.
//!
//! Checks run in order and the first failure wins: denied tool, tool outside a
//! non-empty allowlist, oversized arguments, a deny pattern, sensitive data.
//!
//! Argument text is every string leaf of the argument JSON, recursively through
//! objects and arrays, joined with `\n`; empty or whitespace-only leaves skip
//! the pattern and PII checks.
//!
//! # Example
//!
//! ```
//! use openjiuwen_security::guardrails::tool_arguments::ToolArgumentGuardrail;
//! use openjiuwen_security::{Guardrail, GuardrailContext, RiskLevel, ScanStage};
//! use serde_json::json;
//!
//! let guardrail = ToolArgumentGuardrail::new();
//! let runtime = tokio::runtime::Builder::new_current_thread()
//!     .enable_all()
//!     .build()
//!     .expect("a tokio runtime");
//! let ctx = GuardrailContext::tool_call(
//!     "run_shell",
//!     &json!({"command": "rm -rf /"}),
//!     ScanStage::ToolInvokeInput,
//! );
//! let result = runtime.block_on(guardrail.check(&ctx)).expect("a verdict");
//! assert!(!result.is_safe());
//! assert_eq!(result.risk_level, RiskLevel::High);
//! assert_eq!(result.risk_type.as_deref(), Some("tool_argument"));
//! ```
#![allow(clippy::module_name_repetitions)]

use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;

use openjiuwen_common::error::Result;

use crate::guardrail::{Guardrail, DEFAULT_PRIORITY};
use crate::guardrails::content_rules::ContentRule;
use crate::guardrails::sensitive_data::{
    RedactionOutcome, SensitiveDataGuardrail, SensitiveDataGuardrailConfig,
};
use crate::rules::CompiledRule;
use crate::{GuardrailContext, GuardrailResult, RiskLevel, ScanStage};

/// Stable guardrail name, mirrored in findings and logs.
const NAME: &str = "tool_arguments";

/// Risk type reported for a denied tool name.
const RISK_TYPE_BLOCKED_TOOL: &str = "blocked_tool";
/// Risk type reported for a tool outside a non-empty allowlist.
const RISK_TYPE_TOOL_NOT_ALLOWED: &str = "tool_not_allowed";
/// Risk type reported when the arguments exceed the configured size limit.
const RISK_TYPE_ARGUMENTS_TOO_LARGE: &str = "arguments_too_large";
/// Risk type reported when argument deny patterns match.
const RISK_TYPE_TOOL_ARGUMENT: &str = "tool_argument";
/// Risk type reported when sensitive data is found in the arguments.
const RISK_TYPE_SENSITIVE_DATA: &str = "sensitive_data";

/// The four default argument deny rules, in evaluation order.
///
/// Every row reports [`RISK_TYPE_TOOL_ARGUMENT`] at [`RiskLevel::High`], so a
/// match is a finding a pipeline may allow (detect-only) while the risky call is
/// still reported.
#[must_use]
fn default_deny_patterns() -> Vec<ContentRule> {
    /// A full table row at [`RiskLevel::High`] without a message.
    fn row(name: &str, pattern: &str) -> ContentRule {
        ContentRule {
            name: name.to_owned(),
            pattern: pattern.to_owned(),
            risk_type: RISK_TYPE_TOOL_ARGUMENT.to_owned(),
            risk_level: RiskLevel::High,
            message: None,
        }
    }

    vec![
        row("shell_substitution", r"\$\([^)]*\)|`[^`]*`|\$\{[^}]*\}"),
        row(
            "shell_chain",
            r"(?:;|&&|\|\|)\s*(?:rm|curl|wget|nc|chmod|sh|bash)\b",
        ),
        row("path_traversal", r"\.\./\.\./"),
        row("destructive_command", r"\brm\s+-[rRf]"),
    ]
}

/// The nested sensitive-data configuration used when the field is absent.
///
/// Kept as a named function so the serde attribute and [`Default`] agree: a JSON
/// payload that omits `sensitive_data` round-trips back to this value.
#[must_use]
#[allow(clippy::unnecessary_wraps)] // serde's `default` path must yield the field type.
fn default_sensitive_data_config() -> Option<SensitiveDataGuardrailConfig> {
    Some(SensitiveDataGuardrailConfig::default())
}

/// Configuration for [`ToolArgumentGuardrail`].
///
/// Deserialization is tolerant: every field falls back to its [`Default`] value
/// and unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct ToolArgumentGuardrailConfig {
    /// Tool names allowed to run; an empty list allows every tool.
    pub allowed_tools: Vec<String>,
    /// Tool names that must never run (exact, case-sensitive match).
    pub denied_tools: Vec<String>,
    /// Severity reported for a denied tool or an allowlist violation.
    pub denied_tool_risk_level: RiskLevel,
    /// Argument deny rules, in evaluation order; the first match decides.
    #[serde(default = "default_deny_patterns")]
    pub deny_patterns: Vec<ContentRule>,
    /// Nested PII configuration; `None` disables the sensitive-data check.
    #[serde(default = "default_sensitive_data_config")]
    pub sensitive_data: Option<SensitiveDataGuardrailConfig>,
    /// Maximum size of the compact JSON arguments in bytes; `None` sets no
    /// limit.
    pub max_argument_bytes: Option<usize>,
}

impl Default for ToolArgumentGuardrailConfig {
    /// Every tool allowed, no denied tools, the default deny table at
    /// [`RiskLevel::Critical`] for tool-name violations, PII detection with
    /// redaction, and no size limit.
    fn default() -> Self {
        Self {
            allowed_tools: Vec::new(),
            denied_tools: Vec::new(),
            denied_tool_risk_level: RiskLevel::Critical,
            deny_patterns: default_deny_patterns(),
            sensitive_data: default_sensitive_data_config(),
            max_argument_bytes: None,
        }
    }
}

/// One deny rule compiled together with the risk type it reports.
struct CompiledDenyRule {
    /// The compiled regex plus its name and source.
    compiled: CompiledRule,
    /// Machine-readable risk name reported on a match.
    risk_type: String,
}

impl CompiledDenyRule {
    /// The severity this rule reports; [`RiskLevel::High`] is only a fallback.
    fn risk_level(&self) -> RiskLevel {
        self.compiled.risk_level.unwrap_or(RiskLevel::High)
    }
}

/// Validates the arguments of tool calls before the tool runs.
///
/// The guardrail binds to [`ScanStage::ToolInvokeInput`] by default and is
/// stateless after construction: one instance can serve many concurrent scans.
/// It never performs I/O, so [`Guardrail::check`] is infallible in practice.
pub struct ToolArgumentGuardrail {
    /// The configuration the guardrail was built from.
    config: ToolArgumentGuardrailConfig,
    /// `config.deny_patterns`, compiled case-insensitively in table order.
    rules: Vec<CompiledDenyRule>,
    /// The optional PII detector over the argument text.
    sensitive: Option<SensitiveDataGuardrail>,
    /// Pipeline ordering weight.
    priority: i32,
    /// Stages to scan.
    stages: Vec<ScanStage>,
}

impl ToolArgumentGuardrail {
    /// The stages the guardrail scans unless [`Self::with_stages`] overrides
    /// them: the arguments of a tool call.
    pub const DEFAULT_STAGES: &'static [ScanStage] = &[ScanStage::ToolInvokeInput];

    /// A guardrail over [`ToolArgumentGuardrailConfig::default`].
    ///
    /// # Panics
    ///
    /// Never in practice: a panic would mean the built-in deny table was edited
    /// into an invalid regex, i.e. a programming error rather than bad input.
    #[must_use]
    #[allow(clippy::expect_used)] // the built-in deny table is statically valid.
    pub fn new() -> Self {
        Self::with_config(ToolArgumentGuardrailConfig::default())
            .expect("the built-in deny table is always a valid regex")
    }

    /// Compiles `config.deny_patterns` in order, case-insensitively, and builds
    /// the nested PII detector when `config.sensitive_data` is set.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when a deny
    /// pattern (message `invalid pattern '<name>': <error>`) or a nested
    /// sensitive-data pattern is not a valid regular expression.
    pub fn with_config(config: ToolArgumentGuardrailConfig) -> Result<Self> {
        let mut rules = Vec::with_capacity(config.deny_patterns.len());
        for rule in &config.deny_patterns {
            let compiled = CompiledRule::compile(rule.name.clone(), &rule.pattern, false)?
                .with_risk_level(rule.risk_level);
            rules.push(CompiledDenyRule {
                compiled,
                risk_type: rule.risk_type.clone(),
            });
        }
        let sensitive = match &config.sensitive_data {
            Some(nested) => Some(SensitiveDataGuardrail::with_config(nested.clone())?),
            None => None,
        };
        Ok(Self {
            config,
            rules,
            sensitive,
            priority: DEFAULT_PRIORITY,
            stages: Self::DEFAULT_STAGES.to_vec(),
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
    pub fn config(&self) -> &ToolArgumentGuardrailConfig {
        &self.config
    }

    /// The verdict for a denied or non-allowlisted tool name.
    fn tool_name_verdict(&self, tool_name: &str) -> Option<GuardrailResult> {
        if self
            .config
            .denied_tools
            .iter()
            .any(|denied| denied == tool_name)
        {
            return Some(GuardrailResult::block(
                self.config.denied_tool_risk_level,
                RISK_TYPE_BLOCKED_TOOL,
                Some(json!({"tool_name": tool_name})),
                None,
            ));
        }
        if !self.config.allowed_tools.is_empty()
            && !self
                .config
                .allowed_tools
                .iter()
                .any(|allowed| allowed == tool_name)
        {
            return Some(GuardrailResult::block(
                self.config.denied_tool_risk_level,
                RISK_TYPE_TOOL_NOT_ALLOWED,
                Some(json!({"tool_name": tool_name})),
                None,
            ));
        }
        None
    }

    /// The verdict for arguments that exceed the size limit, measured on the
    /// compact JSON serialization of the arguments.
    fn size_verdict(&self, content: &Value) -> Option<GuardrailResult> {
        let limit = self.config.max_argument_bytes?;
        let size = content.to_string().len();
        (size > limit).then(|| {
            GuardrailResult::block(
                RiskLevel::High,
                RISK_TYPE_ARGUMENTS_TOO_LARGE,
                Some(json!({"size": size, "limit": limit})),
                None,
            )
        })
    }

    /// The verdict for the first matching deny rule.
    fn pattern_verdict(&self, text: &str, tool_name: Option<&str>) -> Option<GuardrailResult> {
        self.rules.iter().find_map(|rule| {
            rule.compiled.is_match(text).then(|| {
                GuardrailResult::block(
                    rule.risk_level(),
                    rule.risk_type.clone(),
                    Some(with_tool_name(
                        json!({
                            "rule": rule.compiled.name,
                            "matched_pattern": rule.compiled.pattern,
                        }),
                        tool_name,
                    )),
                    None,
                )
            })
        })
    }

    /// The verdict for sensitive data in the arguments: the highest match
    /// severity, with sanitized arguments attached only when redaction is on.
    fn sensitive_verdict(
        &self,
        content: &Value,
        text: &str,
        tool_name: Option<&str>,
    ) -> Option<GuardrailResult> {
        let guardrail = self.sensitive.as_ref()?;
        let outcome = guardrail.redact_text(text);
        if outcome.matches.is_empty() {
            return None;
        }
        let level = outcome
            .matches
            .iter()
            .map(|found| found.risk_level)
            .max()
            .unwrap_or(RiskLevel::High);
        let modified = guardrail
            .config()
            .redact
            .then(|| json!({"sanitized_arguments": redact_value(content, guardrail)}));
        Some(GuardrailResult::block(
            level,
            RISK_TYPE_SENSITIVE_DATA,
            Some(with_tool_name(
                json!({
                    "matched_patterns": matched_pattern_names(&outcome),
                    "match_count": outcome.matches.len(),
                }),
                tool_name,
            )),
            modified,
        ))
    }
}

impl Default for ToolArgumentGuardrail {
    /// Same as [`ToolArgumentGuardrail::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ToolArgumentGuardrail {
    /// Reports the name, priority, stages, tool-list sizes, rule names, and the
    /// detection settings; argument content is never printed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rules: Vec<&str> = self
            .rules
            .iter()
            .map(|rule| rule.compiled.name.as_str())
            .collect();
        f.debug_struct("ToolArgumentGuardrail")
            .field("name", &NAME)
            .field("priority", &self.priority)
            .field("stages", &self.stages)
            .field("allowed_tools", &self.config.allowed_tools.len())
            .field("denied_tools", &self.config.denied_tools.len())
            .field("rules", &rules)
            .field("max_argument_bytes", &self.config.max_argument_bytes)
            .field("sensitive_data", &self.sensitive.is_some())
            .finish()
    }
}

#[async_trait]
impl Guardrail for ToolArgumentGuardrail {
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
        // The arguments are walked structurally below, but the checked
        // projection still gates the scan: an unscannable payload (for example
        // `null` arguments) is a typed failure rather than a silent pass, and
        // the pipeline's failure policy decides whether it blocks.
        ctx.text_view_checked()?;
        let tool_name = ctx.get_tool_name();
        if let Some(name) = tool_name {
            if let Some(verdict) = self.tool_name_verdict(name) {
                return Ok(verdict);
            }
        }
        if let Some(verdict) = self.size_verdict(&ctx.content) {
            return Ok(verdict);
        }
        let text = argument_text(&ctx.content);
        if text.trim().is_empty() {
            return Ok(GuardrailResult::pass());
        }
        if let Some(verdict) = self.pattern_verdict(&text, tool_name) {
            return Ok(verdict);
        }
        if let Some(verdict) = self.sensitive_verdict(&ctx.content, &text, tool_name) {
            return Ok(verdict);
        }
        Ok(GuardrailResult::pass())
    }
}

/// Adds `tool_name` to a `details` object when the context carries one.
///
/// Details stay content-free: only the caller-supplied tool name is copied, never
/// the arguments the tool would receive.
fn with_tool_name(mut details: Value, tool_name: Option<&str>) -> Value {
    if let Some(name) = tool_name {
        if let Some(map) = details.as_object_mut() {
            map.insert("tool_name".to_owned(), Value::String(name.to_owned()));
        }
    }
    details
}

/// Collects every string leaf of `value`, in serialization order.
///
/// Objects are traversed in their serialized key order and arrays in element
/// order, so the flattened text is stable for a given argument payload.
fn collect_strings(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(text) => out.push(text.clone()),
        Value::Array(items) => {
            for item in items {
                collect_strings(item, out);
            }
        }
        Value::Object(map) => {
            for item in map.values() {
                collect_strings(item, out);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// The flattened argument text every deny rule and the PII detector scan.
///
/// String leaves are joined with `\n`; an object, array, and numeric payload
/// without strings all yield the empty string.
fn argument_text(content: &Value) -> String {
    let mut leaves = Vec::new();
    collect_strings(content, &mut leaves);
    leaves.join("\n")
}

/// Names of the matched PII patterns, in table order, without duplicates.
fn matched_pattern_names(outcome: &RedactionOutcome) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for found in &outcome.matches {
        if !names.contains(&found.pattern) {
            names.push(found.pattern.clone());
        }
    }
    names
}

/// Returns `value` with every string leaf redacted by `guardrail`.
///
/// Objects and arrays keep their shape and their non-string leaves, so the
/// sanitized arguments remain valid arguments for the same tool.
fn redact_value(value: &Value, guardrail: &SensitiveDataGuardrail) -> Value {
    match value {
        Value::String(text) => Value::String(guardrail.redact_text(text).redacted),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| redact_value(item, guardrail))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), redact_value(item, guardrail)))
                .collect(),
        ),
        Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value
)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::guardrails::sensitive_data::SensitivePattern;

    fn guardrail() -> ToolArgumentGuardrail {
        ToolArgumentGuardrail::new()
    }

    fn rule(name: &str, pattern: &str, risk_type: &str) -> ContentRule {
        ContentRule {
            name: name.to_owned(),
            pattern: pattern.to_owned(),
            risk_type: risk_type.to_owned(),
            ..ContentRule::default()
        }
    }

    fn tool_ctx(name: &str, arguments: Value) -> GuardrailContext {
        GuardrailContext::tool_call(name, &arguments, ScanStage::ToolInvokeInput)
    }

    #[test]
    fn default_deny_table_shape() {
        let expected: Vec<(&str, &str, RiskLevel)> = vec![
            (
                "shell_substitution",
                r"\$\([^)]*\)|`[^`]*`|\$\{[^}]*\}",
                RiskLevel::High,
            ),
            (
                "shell_chain",
                r"(?:;|&&|\|\|)\s*(?:rm|curl|wget|nc|chmod|sh|bash)\b",
                RiskLevel::High,
            ),
            ("path_traversal", r"\.\./\.\./", RiskLevel::High),
            ("destructive_command", r"\brm\s+-[rRf]", RiskLevel::High),
        ];
        let table = default_deny_patterns();
        assert_eq!(table.len(), expected.len());
        for (row, (name, pattern, level)) in table.iter().zip(expected) {
            assert_eq!(row.name, name);
            assert_eq!(row.pattern, pattern);
            assert_eq!(row.risk_level, level);
            assert_eq!(row.risk_type, "tool_argument");
            assert_eq!(row.message, None);
        }
    }

    #[test]
    fn default_config_values() {
        let config = ToolArgumentGuardrailConfig::default();
        assert_eq!(config.allowed_tools, Vec::<String>::new());
        assert_eq!(config.denied_tools, Vec::<String>::new());
        assert_eq!(config.denied_tool_risk_level, RiskLevel::Critical);
        assert_eq!(config.deny_patterns, default_deny_patterns());
        assert_eq!(
            config.sensitive_data,
            Some(SensitiveDataGuardrailConfig::default())
        );
        assert_eq!(config.max_argument_bytes, None);
    }

    #[test]
    fn config_round_trips_through_serde() {
        let config = ToolArgumentGuardrailConfig::default();
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["denied_tool_risk_level"], json!("critical"));
        assert_eq!(
            value["deny_patterns"][0]["name"],
            json!("shell_substitution")
        );
        assert_eq!(value["sensitive_data"]["redact"], json!(true));
        assert_eq!(value["max_argument_bytes"], Value::Null);
        let back: ToolArgumentGuardrailConfig = serde_json::from_value(value).unwrap();
        assert_eq!(back, config);

        let partial: ToolArgumentGuardrailConfig =
            serde_json::from_value(json!({"denied_tools": ["rm_rf"]})).unwrap();
        assert_eq!(partial.denied_tools, vec!["rm_rf".to_owned()]);
        assert_eq!(partial.deny_patterns, default_deny_patterns());
        assert_eq!(
            partial.sensitive_data,
            Some(SensitiveDataGuardrailConfig::default()),
            "an omitted sensitive_data field restores the default"
        );
        assert_eq!(
            partial,
            ToolArgumentGuardrailConfig {
                denied_tools: vec!["rm_rf".to_owned()],
                ..ToolArgumentGuardrailConfig::default()
            }
        );

        let disabled: ToolArgumentGuardrailConfig =
            serde_json::from_value(json!({"sensitive_data": null})).unwrap();
        assert_eq!(disabled.sensitive_data, None);
    }

    #[test]
    fn nested_leaves_are_flattened() {
        let content = json!({"a": {"b": ["x", 1]}});
        assert_eq!(argument_text(&content), "x");
        assert_eq!(
            argument_text(&json!({"first": "one", "second": {"deep": ["two", "three"]}})),
            "one\ntwo\nthree"
        );
        assert_eq!(argument_text(&json!({})), "");
        assert_eq!(argument_text(&json!([1, true, null])), "");
    }

    #[tokio::test]
    async fn every_default_pattern_is_detected() {
        let guardrail = guardrail();
        let cases = [
            ("shell_substitution", "$(whoami)"),
            ("shell_substitution", "${HOME}/secrets"),
            ("shell_substitution", "`id`"),
            ("shell_chain", "pwd; rm -rf /"),
            ("shell_chain", "true && curl example.com"),
            ("path_traversal", "../../etc/passwd"),
            ("destructive_command", "rm -rf /"),
        ];
        for (name, command) in cases {
            let ctx = tool_ctx("run_shell", json!({"command": command}));
            let result = guardrail.check(&ctx).await.unwrap();
            assert!(!result.is_safe(), "{command:?} must be blocked");
            assert_eq!(result.risk_type.as_deref(), Some("tool_argument"));
            assert_eq!(result.risk_level, RiskLevel::High);
            let details = result.details.unwrap();
            assert_eq!(details["rule"], json!(name), "{command:?}");
            assert_eq!(details["tool_name"], json!("run_shell"), "{command:?}");
            assert_eq!(
                details["matched_pattern"],
                json!(
                    default_deny_patterns()
                        .iter()
                        .find(|rule| rule.name == name)
                        .unwrap()
                        .pattern
                ),
                "{command:?}"
            );
            assert_eq!(result.modified_data, None);
        }
    }

    #[tokio::test]
    async fn deny_patterns_match_case_insensitively() {
        let result = guardrail()
            .check(&tool_ctx("run_shell", json!({"command": "RM -RF /"})))
            .await
            .unwrap();
        assert!(!result.is_safe());
        assert_eq!(
            result.details.unwrap()["rule"],
            json!("destructive_command")
        );
    }

    #[tokio::test]
    async fn nested_arguments_are_scanned() {
        let ctx = tool_ctx(
            "run_shell",
            json!({"options": {"steps": ["echo ok", "$(cat /etc/passwd)"]}}),
        );
        let result = guardrail().check(&ctx).await.unwrap();
        assert!(!result.is_safe());
        assert_eq!(
            result.details.as_ref().unwrap()["rule"],
            json!("shell_substitution")
        );
        assert!(
            !format!("{result:?}").contains("cat /etc/passwd"),
            "details never carry the matched text"
        );
    }

    #[tokio::test]
    async fn first_matching_rule_wins() {
        let configured = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            deny_patterns: vec![
                ContentRule {
                    risk_level: RiskLevel::Low,
                    ..rule("first", "secret", "first_hit")
                },
                rule("second", "secret", "second_hit"),
            ],
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap();
        let result = configured
            .check(&tool_ctx("lookup", json!({"q": "a secret"})))
            .await
            .unwrap();
        assert_eq!(result.risk_type.as_deref(), Some("first_hit"));
        assert_eq!(result.risk_level, RiskLevel::Low);
    }

    #[tokio::test]
    async fn denied_and_allowlisted_tools() {
        let denied = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            denied_tools: vec!["shell_exec".to_owned()],
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap();
        let result = denied
            .check(&tool_ctx("shell_exec", json!({"command": "ls"})))
            .await
            .unwrap();
        assert!(!result.is_safe());
        assert_eq!(result.risk_type.as_deref(), Some("blocked_tool"));
        assert_eq!(result.risk_level, RiskLevel::Critical);
        assert_eq!(result.details, Some(json!({"tool_name": "shell_exec"})));
        assert!(denied
            .check(&tool_ctx("search", json!({"q": "hello"})))
            .await
            .unwrap()
            .is_safe());

        let allowlisted = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            allowed_tools: vec!["search".to_owned()],
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap();
        let result = allowlisted
            .check(&tool_ctx("shell_exec", json!({"command": "ls"})))
            .await
            .unwrap();
        assert_eq!(result.risk_type.as_deref(), Some("tool_not_allowed"));
        assert_eq!(result.details, Some(json!({"tool_name": "shell_exec"})));
        assert!(allowlisted
            .check(&tool_ctx("search", json!({"q": "hello"})))
            .await
            .unwrap()
            .is_safe());

        let no_name = GuardrailContext::raw("_framework:tool_invoke_input", json!({"q": "hello"}));
        assert!(allowlisted.check(&no_name).await.unwrap().is_safe());

        let configurable = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            denied_tools: vec!["shell_exec".to_owned()],
            denied_tool_risk_level: RiskLevel::Medium,
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap();
        let result = configurable
            .check(&tool_ctx("shell_exec", json!({})))
            .await
            .unwrap();
        assert_eq!(result.risk_level, RiskLevel::Medium);
    }

    #[tokio::test]
    async fn argument_size_limit() {
        let limited = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            max_argument_bytes: Some(16),
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap();
        let arguments = json!({"q": "1234567890"});
        let size = arguments.to_string().len();
        assert!(size > 16);
        let result = limited.check(&tool_ctx("search", arguments)).await.unwrap();
        assert!(!result.is_safe());
        assert_eq!(result.risk_type.as_deref(), Some("arguments_too_large"));
        assert_eq!(result.risk_level, RiskLevel::High);
        assert_eq!(result.details, Some(json!({"size": size, "limit": 16})));
        assert_eq!(result.modified_data, None);

        assert!(limited
            .check(&tool_ctx("search", json!({"q": "a"})))
            .await
            .unwrap()
            .is_safe());
    }

    #[tokio::test]
    async fn empty_arguments_pass() {
        let detector = guardrail();
        for arguments in [json!({}), json!({"q": ""}), json!({"q": "   "})] {
            let result = detector
                .check(&tool_ctx("search", arguments))
                .await
                .unwrap();
            assert!(result.is_safe(), "{result:?}");
            assert_eq!(result.details, None);
            assert_eq!(result.modified_data, None);
            assert_eq!(result.risk_type, None);
        }
        let benign = detector
            .check(&tool_ctx(
                "search",
                json!({"query": "rust ownership", "limit": 3}),
            ))
            .await
            .unwrap();
        assert!(benign.is_safe());
    }

    #[tokio::test]
    async fn unscannable_payload_is_a_validation_error() {
        let detector = guardrail();
        // `null` arguments cannot be projected, so the detector reports a typed
        // failure instead of passing the call unexamined.
        let error = detector
            .check(&GuardrailContext::raw(
                "user_input",
                serde_json::Value::Null,
            ))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("unscannable content"), "{error}");
        let null_arguments = detector
            .check(&tool_ctx("search", json!(null)))
            .await
            .unwrap_err();
        assert_eq!(null_arguments.kind(), ErrorKind::Validation);
        assert!(
            null_arguments.to_string().contains("unscannable content"),
            "{null_arguments}"
        );

        // Scannable empty arguments still pass.
        assert!(detector
            .check(&tool_ctx("search", json!({})))
            .await
            .unwrap()
            .is_safe());
    }

    #[tokio::test]
    async fn sensitive_data_is_redacted_in_arguments() {
        let result = guardrail()
            .check(&tool_ctx(
                "send_mail",
                json!({"to": "user@example.com", "cc": ["ops@example.com"], "count": 2}),
            ))
            .await
            .unwrap();
        assert!(!result.is_safe());
        assert_eq!(result.risk_type.as_deref(), Some("sensitive_data"));
        assert_eq!(result.risk_level, RiskLevel::High);
        assert_eq!(
            result.details,
            Some(json!({
                "matched_patterns": ["email"],
                "match_count": 2,
                "tool_name": "send_mail",
            }))
        );
        let modified = result.modified_data.unwrap();
        let sanitized = &modified["sanitized_arguments"];
        assert_eq!(sanitized["to"], json!("<REDACTED>"));
        assert_eq!(sanitized["cc"][0], json!("<REDACTED>"));
        assert_eq!(sanitized["count"], json!(2));
        assert!(
            !sanitized.to_string().contains("user@example.com"),
            "{sanitized}"
        );

        let critical = guardrail()
            .check(&tool_ctx(
                "send_mail",
                json!({"to": "user@example.com", "secret": "sk-abcdefghijklmnopqrst"}),
            ))
            .await
            .unwrap();
        assert_eq!(critical.risk_level, RiskLevel::Critical);
        assert_eq!(
            critical.details.unwrap()["matched_patterns"],
            json!(["email", "api_key"])
        );
    }

    #[tokio::test]
    async fn sensitive_data_config_variants() {
        let redacted = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            sensitive_data: Some(SensitiveDataGuardrailConfig {
                redact: false,
                ..SensitiveDataGuardrailConfig::default()
            }),
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap();
        let result = redacted
            .check(&tool_ctx("send_mail", json!({"to": "user@example.com"})))
            .await
            .unwrap();
        assert!(!result.is_safe());
        assert_eq!(result.risk_type.as_deref(), Some("sensitive_data"));
        assert_eq!(result.modified_data, None);

        let disabled = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            sensitive_data: None,
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap();
        assert!(disabled
            .check(&tool_ctx("send_mail", json!({"to": "user@example.com"})))
            .await
            .unwrap()
            .is_safe());
    }

    #[tokio::test]
    async fn deny_patterns_win_over_sensitive_data() {
        let result = guardrail()
            .check(&tool_ctx(
                "run_shell",
                json!({"command": "rm -rf /", "owner": "user@example.com"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.risk_type.as_deref(), Some("tool_argument"));
        assert_eq!(result.modified_data, None);
    }

    #[test]
    fn stage_defaults_and_overrides() {
        let guardrail = guardrail();
        assert_eq!(guardrail.name(), "tool_arguments");
        assert_eq!(guardrail.priority(), DEFAULT_PRIORITY);
        assert_eq!(guardrail.stages(), ToolArgumentGuardrail::DEFAULT_STAGES);
        assert!(guardrail.handles(ScanStage::ToolInvokeInput));
        assert!(!guardrail.handles(ScanStage::LlmInvokeInput));
        assert_eq!(guardrail.config(), &ToolArgumentGuardrailConfig::default());

        let overridden = ToolArgumentGuardrail::new()
            .with_stages([ScanStage::ToolInvokeOutput])
            .with_priority(7);
        assert_eq!(overridden.priority(), 7);
        assert_eq!(overridden.stages(), &[ScanStage::ToolInvokeOutput]);
        assert!(overridden.handles(ScanStage::ToolInvokeOutput));
        assert!(!overridden.handles(ScanStage::ToolInvokeInput));

        let every_stage = ToolArgumentGuardrail::new().with_stages([]);
        assert_eq!(every_stage.stages(), &[]);
        assert!(every_stage.handles(ScanStage::UserInput));
    }

    #[test]
    fn invalid_deny_pattern_is_a_validation_error() {
        let error = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            deny_patterns: vec![rule("broken", "([a-z", "tool_argument")],
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid pattern 'broken'"),
            "{error}"
        );
    }

    #[test]
    fn invalid_nested_sensitive_pattern_is_a_validation_error() {
        let error = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            sensitive_data: Some(SensitiveDataGuardrailConfig {
                extra_patterns: vec![SensitivePattern {
                    name: "broken_pii".to_owned(),
                    pattern: "([a-z".to_owned(),
                    risk_level: None,
                    validator: None,
                }],
                ..SensitiveDataGuardrailConfig::default()
            }),
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid pattern 'broken_pii'"),
            "{error}"
        );
    }

    #[test]
    fn debug_reports_shape_never_arguments() {
        let configured = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
            allowed_tools: vec!["search".to_owned()],
            denied_tools: vec!["shell_exec".to_owned()],
            ..ToolArgumentGuardrailConfig::default()
        })
        .unwrap();
        let debug = format!("{configured:?}");
        assert!(debug.contains("ToolArgumentGuardrail"), "{debug}");
        assert!(debug.contains("tool_arguments"), "{debug}");
        assert!(debug.contains("shell_substitution"), "{debug}");
        assert!(debug.contains("destructive_command"), "{debug}");
        assert!(debug.contains("allowed_tools: 1"), "{debug}");
        assert!(debug.contains("denied_tools: 1"), "{debug}");
        assert!(debug.contains("stages"), "{debug}");
        assert!(!debug.contains("rm -rf"), "{debug}");
        assert!(format!("{:?}", ToolArgumentGuardrail::default()).contains("tool_arguments"));
    }
}
