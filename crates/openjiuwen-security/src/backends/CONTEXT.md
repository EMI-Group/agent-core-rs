# src/backends — detection backends

## Intent

Backends own *how* content is judged; guardrails own *what* to scan and how to react.
Ports `openjiuwen/core/security/guardrail/backends.py` (rule-based and LLM prompt-injection backends, API and local model backends) plus the transport seams the Python code inlined (`httpx`, `torch`).
Every backend implements `crate::guardrail::GuardrailBackend` and returns a `RiskAssessment`; it never decides blocking.

## API Surface

```rust
pub struct RuleBasedBackendConfig { pub patterns: Vec<String>, pub risk_level: RiskLevel }
pub struct RuleBasedPromptInjectionBackend { /* config */ }
impl RuleBasedPromptInjectionBackend {
    pub fn new() -> Self;                                  // six Python default patterns, HIGH
    pub fn with_config(cfg: RuleBasedBackendConfig) -> Result<Self>;   // invalid regex => Error::Validation
    pub fn patterns(&self) -> &[CompiledRule];
}

pub struct LlmPromptInjectionBackendConfig { pub api_endpoint: String, pub api_key: SecretString, pub model_name: String,
                                             pub system_prompt: Option<String>, pub timeout: Duration }
pub struct LlmPromptInjectionBackend { /* config */ }     // Python stub parity: analyze() delegates to RuleBasedPromptInjectionBackend
impl LlmPromptInjectionBackend { pub fn new(cfg: LlmPromptInjectionBackendConfig) -> Self; pub const DEFAULT_SYSTEM_PROMPT: &'static str }

#[async_trait] pub trait ModerationTransport: Send + Sync {
    async fn moderate(&self, request: ModerationRequest<'_>) -> Result<Value>;
}
pub struct ModerationRequest<'a> { pub url: &'a str, pub api_key: Option<&'a str>, pub text: &'a str, pub timeout: Duration }
pub struct ApiModelBackendConfig { pub api_url: String, pub parser: Option<Arc<dyn ModelOutputParser>>,
                                   pub api_key: Option<SecretString>, pub timeout: Duration, pub risk_type: String }
pub struct ApiModelBackend { /* config + optional transport */ }
impl ApiModelBackend { pub fn new(cfg: ApiModelBackendConfig) -> Self; pub fn with_transport(self, Arc<dyn ModerationTransport>) -> Self }

pub struct LocalModelBackendConfig { pub model_path: String, pub parser: Option<Arc<dyn ModelOutputParser>>,
                                     pub device: String, pub risk_type: String }
pub struct LocalModelBackend { /* config */ }
impl LocalModelBackend { pub fn model_info(&self) -> ModelInfo; }   // {"model_loaded": false, "has_model": false, "has_tokenizer": false}
```
Configs are serde-serializable with the Python field names/defaults (`timeout` 30.0 s, `device` `"auto"`, `risk_type` `"model_detection"`); secrets use `SecretString`.

## Constraints

- `GuardrailBackend` is object-safe and async (`#[async_trait]`); `name()` returns a stable identifier (`"rule_based_prompt_injection"`, `"llm_prompt_injection"`, `"api_model"`, `"local_model"`).
- Behaviour on `GuardrailContext` content: build the text with `GuardrailContext::text_view()`; empty/whitespace text short-circuits to a `SAFE` assessment **without** calling a transport or parser (Python does the same for API/local backends).
- No HTTP client, no model runtime: `ApiModelBackend` without a `ModerationTransport` and `LocalModelBackend` always return `Error::Unsupported` naming the missing piece; parsers are still validated/used when a transport exists.
- Python's rule backend hard-codes `risk_type = "prompt_injection"` and `confidence = 1.0`; keep both. The first matching pattern wins and `details = {"matched_pattern": <pattern>}`.
- Never log or return the scanned text; log pattern names and level only.

## Behavioral contract (port of `backends.py`)

Default prompt-injection patterns, matched case-insensitively and unanchored, in order:

```
ignore.*previous.*instructions
disregard.*prior.*commands
system.*prompt
you.*are.*now
act.*as
forget.*everything
```

`RuleBasedPromptInjectionBackend::with_config` compiles each `Vec<String>` entry with `regex::RegexBuilder::new(p).case_insensitive(true)`. Patterns are Rust regexes; a Python-only construct (e.g. backreference) is a construction error with a message naming the pattern index.
`RuleBasedBackendConfig::default()` = the six patterns + `RiskLevel::High` (Python default). A backend configured with `risk_level = CRITICAL` produces findings that block under the default policy — this is how Python's system test makes blocking observable.
`LlmPromptInjectionBackend` is a documented stub: Python builds a prompt, discards it, and delegates to a fresh `RuleBasedPromptInjectionBackend`; the port does the same and keeps the config for API compatibility (`DEFAULT_SYSTEM_PROMPT` preserved verbatim).
`ApiModelBackend::analyze`: text → `transport.moderate(ModerationRequest { url: api_url, api_key, text, timeout })` → parser (explicit `parser` or default `BertBinaryParser`); a transport error propagates as `Error` and the pipeline applies its failure policy.
`LocalModelBackend::analyze`: `Err(Error::unsupported("local model backend requires an inference runtime (deferred adapter)"))` after the empty-text short-circuit.

## Routing Table

- `./mod.rs` → re-exports and shared defaults
- `./rule_based.rs` → `RuleBasedBackendConfig`, `RuleBasedPromptInjectionBackend` (pattern list + unit tests)
- `./llm.rs` → `LlmPromptInjectionBackend`/config (stub parity) + `DEFAULT_SYSTEM_PROMPT`
- `./api.rs` → `ApiModelBackend`/config, `ModerationTransport`, `ModerationRequest`
- `./local.rs` → `LocalModelBackend`/config, `ModelInfo`
