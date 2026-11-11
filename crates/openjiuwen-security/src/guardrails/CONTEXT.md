# src/guardrails — built-in guardrails

## Intent

The detectors shipped with the crate: prompt-injection heuristics (Python parity), sensitive-data/PII detection with redaction, configurable regex content rules, and tool-call argument validation.
Each one is a `Guardrail` implementation: stateless after construction, async `check(&GuardrailContext) -> Result<GuardrailResult>`, with a serde config and Python-compatible defaults where Python has an equivalent (`PromptInjectionGuardrailConfig`, `RuleBasedBackendConfig`).

## API Surface

```rust
// prompt_injection.rs — port of builtin.py
pub struct PromptInjectionGuardrailConfig { pub mode: String /* "rules" */, pub model_type: Option<String>,
    pub api_url: Option<String>, pub api_key: Option<String>, pub timeout: Duration /* 30s */,
    pub model_path: Option<String>, pub device: String /* "auto" */, pub custom_patterns: Option<Vec<String>>,
    pub risk_level: RiskLevel /* High */, pub bert_thresholds: Option<Thresholds>, pub attack_class_id: i64 /* 1 */,
    pub qwen_risk_type: String /* "content_risk" */ }
pub struct PromptInjectionGuardrail { /* backend, priority, stages */ }
impl PromptInjectionGuardrail {
    pub const DEFAULT_STAGES: &'static [ScanStage];         // [LlmInvokeInput, ToolInvokeOutput]
    pub fn new() -> Self;                                   // rule backend, six Python patterns, HIGH
    pub fn with_config(config: PromptInjectionGuardrailConfig) -> Result<Self>;
    pub fn with_backend(backend: Arc<dyn GuardrailBackend>) -> Self;
    pub fn with_parser(/* config builder counterpart */) -> Result<Self>;   // parser injected for api/local mode
    pub fn with_stages(self, stages: impl IntoIterator<Item = ScanStage>) -> Self;
    pub fn with_priority(self, priority: i32) -> Self;
    pub fn backend_name(&self) -> &str;
}

// sensitive_data.rs (addition)
pub struct SensitivePattern { pub name: String, pub pattern: String, pub risk_level: Option<RiskLevel>, pub validator: Option<SensitiveValidator> }
pub enum SensitiveValidator { Luhn }
pub struct SensitiveDataGuardrailConfig { pub enabled: bool /* true */, pub risk_level: RiskLevel /* High */,
    pub redact: bool /* true */, pub marker: String /* "<REDACTED>" */, pub patterns: Vec<SensitivePattern> /* default table */,
    pub extra_patterns: Vec<SensitivePattern>, pub allowlist: Vec<String> /* matched substrings to skip */ }
pub struct SensitiveDataGuardrail { /* compiled */ }
pub struct RedactionOutcome { pub redacted: String, pub matches: Vec<SensitiveMatch> }
pub struct SensitiveMatch { pub pattern: String, pub risk_level: RiskLevel, pub range: Range<usize> }
impl SensitiveDataGuardrail {
    pub fn new() -> Self;
    pub fn with_config(config: SensitiveDataGuardrailConfig) -> Result<Self>;
    pub fn redact_text(&self, text: &str) -> RedactionOutcome;   // used by tool-argument validation too
}

// content_rules.rs (addition)
pub struct ContentRule { pub name: String, pub pattern: String, pub risk_type: String, pub risk_level: RiskLevel, pub message: Option<String> }
pub struct ContentRuleGuardrailConfig { pub rules: Vec<ContentRule> /* empty */, pub case_sensitive: bool /* false */,
    pub redact: bool /* false */, pub marker: String /* "<REDACTED>" */ }
pub struct ContentRuleGuardrail { /* compiled */ }
impl ContentRuleGuardrail { pub fn new(config: ContentRuleGuardrailConfig) -> Result<Self>; pub fn from_patterns(rules: Vec<ContentRule>) -> Result<Self>; }

// tool_arguments.rs (addition)
pub struct ToolArgumentGuardrailConfig { pub allowed_tools: Vec<String> /* empty = all */, pub denied_tools: Vec<String>,
    pub denied_tool_risk_level: RiskLevel /* Critical */, pub deny_patterns: Vec<ContentRule> /* default table, HIGH */,
    pub sensitive_data: Option<SensitiveDataGuardrailConfig> /* Some(default) */, pub max_argument_bytes: Option<usize> /* None */ }
pub struct ToolArgumentGuardrail { /* compiled */ }
impl ToolArgumentGuardrail { pub fn new() -> Self; pub fn with_config(cfg) -> Result<Self>; }
```
Stage defaults: prompt injection `[LlmInvokeInput, ToolInvokeOutput]` (Python `DEFAULT_EVENTS`); sensitive data, content rules `[]` (every stage); tool arguments `[ToolInvokeInput]`.

## Constraints

- Every rule list is compiled once at construction with `regex::RegexBuilder` (`case_insensitive(true)` unless the config says otherwise); an invalid pattern returns `Error::Validation("invalid <kind> pattern '<name>': <regex error>")` — never a panic, never a silent skip.
- All regexes run against `GuardrailContext::text_view()`; matches are found with `find_iter` and redaction is applied on the original text by byte range, last-to-first, so ranges stay valid.
- Findings never embed matched text: `details` carries pattern/rule names, `match_count`, and, for tool arguments, `tool_name`. Redacted text is exposed only through `modified_data` (`{"sanitized_text": ...}` for text, `{"sanitized_arguments": ...}` for tool arguments).
- An empty/whitespace text (or empty arguments) is always `GuardrailResult::pass()`; detectors do no work without content.
- `SensitiveDataGuardrail` matches are de-duplicated by (pattern, start) so a number matched by both `credit_card` and `phone` is redacted once; the first overlapping match wins.
- `allowlist` entries are compared to the full matched substring (exact match, case-sensitive) and skip that match.

## Default sensitive-data table

| name | pattern | risk | validator |
|---|---|---|---|
| `email` | `[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}` | `High` | — |
| `cn_mobile` | `\b1[3-9]\d{9}\b` | `Medium` | — |
| `cn_id_card` | `\b[1-9]\d{5}(?:18\|19\|20)\d{2}(?:0[1-9]\|1[0-2])(?:0[1-9]\|[12]\d\|3[01])\d{3}[\dXx]\b` | `High` | — |
| `credit_card` | `\b(?:\d[ -]?){12,18}\d\b` | `High` | `Luhn` |
| `us_ssn` | `\b\d{3}-\d{2}-\d{4}\b` | `High` | — |
| `api_key` | `\b(?:sk-[A-Za-z0-9]{16,}\|AKIA[0-9A-Z]{16}\|ghp_[A-Za-z0-9]{36})\b` | `Critical` | — |
| `private_key` | `-----BEGIN [A-Z ]*PRIVATE KEY-----` | `Critical` | — |

A `credit_card` candidate is accepted only when its digits pass the Luhn checksum (`SensitiveValidator::Luhn`); a failing candidate is not a match at all.
Orders above are the default redaction order; findings list matched pattern names in table order, de-duplicated.

## Default tool-argument deny rules

| name | pattern | risk |
|---|---|---|
| `shell_substitution` | `\$\([^)]*\)\|\`[^\`]*\`\|\$\{[^}]*\}` | `High` |
| `shell_chain` | `(?:;\|&&\|\|\|)\s*(?:rm\|curl\|wget\|nc\|chmod\|sh\|bash)\b` | `High` |
| `path_traversal` | `\.\./\.\./` | `High` |
| `destructive_command` | `\brm\s+-[rRf]` | `High` |

Risk types: `"tool_argument"` for pattern hits, `"blocked_tool"` for a denied tool name, `"tool_not_allowed"` for a tool outside a non-empty `allowed_tools`, `"sensitive_data"` for PII found in arguments.
Denied tool names and allowlist violations use `denied_tool_risk_level` (`Critical` by default) so they block; pattern hits use their rule level (detect-only by default, exactly like Python's HIGH findings).

## Behavioral contract

`PromptInjectionGuardrail::with_config` reproduces Python's construction validation and message strings:
`"invalid mode: {mode}, must be 'rules', 'api' or 'local'"`, `"api_url is required for api mode"`, `"model_path is required for local mode"`, `"either model_type or parser must be specified for api/local mode"`, `"unknown model_type: {model_type}"`, checked in that order.
Mode `rules` → `RuleBasedPromptInjectionBackend` with `custom_patterns` and `risk_level`; `api`/`local` → `ApiModelBackend`/`LocalModelBackend` with the default parser (`BertBinaryParser { risk_type: "prompt_injection", confidence_thresholds, attack_class_id }` for `bert`, `QwenGuardParser { risk_type: qwen_risk_type }` for `qwen` or a custom parser).
`check` returns the backend's assessment as a `GuardrailResult`: `is_safe = !has_risk`, level and type verbatim (`risk_type` stays `None` for safe results), `details` from the assessment; `modified_data` is `None`.
`SensitiveDataGuardrail::check` blocks with `risk_type = "sensitive_data"`, `risk_level = config.risk_level`, `details = {"matched_patterns": [..], "match_count": n}`, and `modified_data = {"sanitized_text": redacted}` when `redact` is true.
`ContentRuleGuardrail::check`: first matching rule yields `risk_type = rule.risk_type`, `risk_level = rule.risk_level`, `details = {"rule": name, "matched_pattern": pattern}` and `modified_data = {"sanitized_text": ...}` when `redact`; other rules still run for reporting when no match has been found yet is not attempted — one result per scan (first match wins, Python rule-backend semantics).
`ToolArgumentGuardrail::check`: reads `ctx.get_tool_name()` and the argument JSON; order = allow/deny list, `max_argument_bytes`, deny patterns, sensitive data. It returns the first failure as the `GuardrailResult`; otherwise `pass()`. When sensitive data is redacted it returns the finding with `modified_data = {"sanitized_arguments": <arguments with redacted strings>}`.

## Routing Table

- `./mod.rs` → re-exports
- `./prompt_injection.rs` → `PromptInjectionGuardrail`, config, backend factory + validation tests
- `./sensitive_data.rs` → `SensitiveDataGuardrail`, `SensitivePattern`, `SensitiveValidator`, `RedactionOutcome`, default table
- `./content_rules.rs` → `ContentRule`, `ContentRuleGuardrail` + config
- `./tool_arguments.rs` → `ToolArgumentGuardrail` + config + default deny rules
