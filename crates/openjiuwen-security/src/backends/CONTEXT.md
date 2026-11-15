# src/backends — detection backends

## Intent

Backends own *how* content is judged; guardrails own *what* to scan and how to react.
Ports `openjiuwen/core/security/guardrail/backends.py` (rule-based, `LLM` prompt-injection, API and local model backends), supplies the `httpx` transport Python inlined, and adds three vendor moderation adapters under `./vendors/`.
Every backend implements `crate::guardrail::GuardrailBackend` and returns a `RiskAssessment`; it never decides blocking — the pipeline turns assessments into findings and applies `SecurityPolicy`.
Local judging (rules, parser over a reply) is CPU-only; `http`, `api`, `llm` with a judge, and `vendors::*` perform HTTP through `openjiuwen_common::http::HttpClient`.
Backends never log or return the scanned text: diagnostics carry backend names, stages, levels, lengths, scores, and vendor codes only.

## Failure Contract

`SecurityPolicy::default()` is `FailurePolicy::FailClosed`: any `Err` a guardrail or backend returns becomes a blocking `CRITICAL` finding with `risk_type = "guardrail_error"`, so a broken detector can never read as "safe".
`FailurePolicy::FailOpen` is an explicit opt-in that only records the error in `GuardrailOutcome::errors` and leaves the decision unchanged.
A backend therefore has exactly two ways to say "nothing found" and one to say "cannot tell":

| Situation | Backend behaviour | Pipeline effect (default) |
|---|---|---|
| payload is blank *scannable* text (empty string, empty message array, whitespace only) | `Ok(safe_assessment())` — the only short-circuit to a safe verdict | `Allow`, no finding |
| payload is scannable and the detector finds nothing | `Ok(safe_assessment())` / a safe `RiskAssessment` | `Allow`, no finding |
| body unreachable (unscannable payload, missing transport, unusable reply, invalid verdict, missing runtime) | `Err(...)` — `Validation`, `Transport`, `Llm`, or `Unsupported` | blocking `Critical` `guardrail_error` finding, `GuardrailOutcome::enforce()` yields `Error::Security` |

`safe_assessment()` is *only* the blank-scannable-text verdict (`has_risk = false`, `Safe`, `confidence = 1.0`) and is never used to paper over a failure.
Every backend projects the payload with `GuardrailContext::text_view_checked()`, never the total `text_view()`: a payload the projection cannot read is `Error::validation("unscannable content: ...")`.
Remote replies are graded strictly: an empty body, a non-JSON body, a non-object JSON body, an unrecognized vendor schema, and an invalid judge verdict are all errors instead of defaulted "clean" assessments.

## API Surface

```rust
// rule_based.rs
pub struct RuleBasedBackendConfig { pub patterns: Vec<String>, pub risk_level: RiskLevel }
impl RuleBasedPromptInjectionBackend {
    pub fn new() -> Self;                                              // six Python default patterns, HIGH
    pub fn with_config(cfg: RuleBasedBackendConfig) -> Result<Self>;   // invalid regex => Error::Validation
    pub fn patterns(&self) -> &[CompiledRule];                         // rules named `pattern[<index>]`
    pub fn risk_level(&self) -> RiskLevel;
}

// llm.rs
pub struct LlmPromptInjectionBackendConfig { pub api_endpoint: String, pub api_key: SecretString, pub model_name: String,
                                             pub system_prompt: Option<String>, pub timeout: f64 /* seconds */ }
impl LlmPromptInjectionBackend {
    pub fn new(cfg) -> Self;
    pub fn config(&self) -> &LlmPromptInjectionBackendConfig;
    pub fn with_model(self, Arc<dyn Model>) -> Self;                   // foundation `llm::Model`; installs the judge
    pub fn with_model_request(self, ModelRequestConfig) -> Self;
    pub fn has_judge(&self) -> bool;
    pub fn request(&self) -> &ModelRequestConfig;
    pub fn parse_verdict(&self, output: &str) -> Result<RiskAssessment>;
    pub const DEFAULT_SYSTEM_PROMPT: &'static str;
}

// api.rs
#[async_trait] pub trait ModerationTransport: Send + Sync { async fn moderate(&self, &ModerationRequest) -> Result<Value>; }
pub struct ModerationRequest { pub url: String, pub api_key: Option<SecretString>, pub text: String, pub timeout: Duration }
pub struct ApiModelBackendConfig { pub api_url: String, pub api_key: Option<SecretString>, pub timeout: f64, pub risk_type: String }
impl ApiModelBackend {
    pub fn new(cfg) -> Self;
    pub fn http(cfg) -> Result<Self>;                                  // built-in HTTP transport, blank api_url => Validation
    pub fn with_http_defaults(self) -> Result<Self>;                    // keeps an injected transport and any parser
    pub fn with_parser(self, Arc<dyn ModelOutputParser>) -> Self;
    pub fn with_transport(self, Arc<dyn ModerationTransport>) -> Self;
}

// http.rs
pub enum HttpAuth { None, Bearer(SecretString), Header { name: String, value: SecretString } }
pub struct HttpModerationConfig { pub url: String, pub headers: BTreeMap<String, String>, pub auth: HttpAuth,
                                  pub timeout: f64, pub max_retries: u32, pub retry_initial_backoff: f64, pub text_field: String }
impl HttpModerationConfig { with_url / with_header / with_headers / with_auth / with_timeout / with_max_retries / with_text_field }
impl HttpModerationTransport {
    pub fn new(config) -> Result<Self>;                                // Validation for bad config, Config if the client fails
    pub fn config(&self) -> &HttpModerationConfig;
}   // Clone + Debug; implements ModerationTransport
// crate-private, shared with the vendor adapters:
// http_client(Duration, u32, Duration) -> Result<HttpClient>, json_body(HttpResponse, service) -> Result<Value>, resolve_timeout(f64) -> Duration
// crate-private, transport-only: require_object(Value, status, service) -> Result<Value>, json_shape(&Value) -> &'static str

// local.rs
pub const SUPPORTED_DEVICES: [&str; 4] = ["auto", "cpu", "cuda", "mps"];
pub struct LocalModelBackendConfig { pub model_path: String, pub device: String, pub risk_type: String }  // + validate()
pub struct ModelInfo { pub model_loaded: bool, pub has_model: bool, pub has_tokenizer: bool }
impl LocalModelBackend { new / with_parser / with_device / config / model_path / device / is_available / model_info }
```

Configs are serde-serializable with the Python field names/defaults (`timeout` 30.0 s, `device` `"auto"`, `risk_type` `"model_detection"`); secrets use `SecretString`.
`HttpModerationConfig` defaults: blank URL, no headers, `HttpAuth::None`, `30.0` s, `max_retries: 2`, `retry_initial_backoff: 0.2` s, `text_field: "text"`.
Timeouts are kept as `f64` seconds for Python parity and sanitised by `resolve_timeout` (NaN, infinities, zero, and negatives fall back to 30 s).
Parsers and judges are runtime collaborators, not configuration: they are injected (`with_parser`, `with_model`, `with_transport`) and absent from the serde structs.
`ModerationRequest` owns its payload and has a manual `Debug` printing `url`, `api_key_present`, `text_len`, and `timeout` only.
`mod.rs` also holds `safe_assessment()` (crate-private): `has_risk = false`, `Safe`, `confidence = 1.0`, no risk type, no details.

## Behavioral Contract

**Rule-based prompt injection** (port of `backends.py`, unchanged detection): default patterns, matched case-insensitively and unanchored, in order —
`ignore.*previous.*instructions`, `disregard.*prior.*commands`, `system.*prompt`, `you.*are.*now`, `act.*as`, `forget.*everything`.
`with_config` compiles each entry through `CompiledRule::compile("pattern[<index>]", p, false)`; an empty `patterns` list falls back to the six defaults (Python `config.patterns or self._default_patterns()`); a Rust-incompatible construct is a construction error naming the index.
`RuleBasedBackendConfig::default()` is the six patterns at `RiskLevel::High`; the first match wins, `risk_type = "prompt_injection"`, `confidence = 1.0`, `details = {"matched_pattern": <pattern>}`.
`analyze` projects with `text_view_checked()?`, returns `safe_assessment()` for blank scannable text, and returns `safe_assessment()` again when *no* default or configured pattern matches — that second safe verdict is a genuine negative result, not a failure.
A backend configured at `CRITICAL` blocks under the default policy — that is how Python's system test makes blocking observable.

**LLM prompt injection**: without a judge, `analyze` delegates to a `RuleBasedPromptInjectionBackend::new()` (Python stub parity; the config does not affect detection).
With `with_model`, `analyze` projects with `text_view_checked()?`, short-circuits blank scannable text, sends a system message (`config.system_prompt` or `DEFAULT_SYSTEM_PROMPT`) and the projected text as the user message through `Model::invoke`, and grades the reply with `parse_verdict`.
The request defaults to `ModelRequestConfig { model_name: config.model_name (the judge's own name when that is blank), temperature: 0.0, max_tokens: 256, stream: false }`; `with_model_request` replaces it.
`parse_verdict` accepts bare JSON, a fenced code block, or prose wrapping one object; the outermost braces are located first, so text outside the object is ignored.
`JudgeVerdict` requires a boolean `has_risk` (`has_risk: bool` carries no `#[serde(default)]`): `{}`, an object without `has_risk`, and a non-boolean `has_risk` are all `Error::llm("llm judge returned an invalid verdict: <serde reason>")` — the serde reason names the offending field, and nothing from the reply is echoed.
A reply with no JSON object at all is `Error::llm("llm judge returned no JSON verdict")`.
For a risky verdict an unknown/missing `risk_level` maps to the fail-safe `RiskLevel::High` (never `Safe`), a missing `risk_type` to `"prompt_injection"`, and `confidence` is clamped to `0.0..=1.0` (default `1.0`).
`details` carries `{backend, model, judge_risk_level?}` and nothing verbatim from the reply; judge `Model` errors and a reply with no message propagate unchanged, so the pipeline's fail-closed policy decides.
`DEFAULT_SYSTEM_PROMPT` is a *reconstructed* Chinese JSON-contract prompt (`has_risk`, `risk_type`, `risk_level` over `SAFE`…`CRITICAL`), because Python's own constant is unused by its stub.

**API model backend**: empty scannable text short-circuits to `safe_assessment()` before any transport call; an unscannable payload is `Error::validation` from `text_view_checked()?`; with no transport at all (neither injected nor HTTP) `analyze` returns `Error::Unsupported` naming `ApiModelBackend::http`.
`http` / `with_http_defaults` derive `HttpModerationConfig { url: api_url, auth: api_key => Bearer else None, timeout, max_retries: 2, retry_initial_backoff: 0.2, text_field: "text" }` and reject a blank `api_url` with `Error::Validation`.
An injected transport always wins, in either call order; the parser (default `BertBinaryParser`) is independent of the transport choice.
`analyze` builds `ModerationRequest { url: api_url, api_key, text, timeout }` (configured timeout, or 30 s for unusable values) and grades the reply with the parser.
`ModelOutputParser::parse` is fallible: an unrecognized model-output dialect is `Error::validation` naming the parser and its accepted dialects, so a mis-pointed endpoint surfaces as a guardrail failure instead of a clean verdict.
Recognized shapes keep the parsers' own fail-safe defaults (for example `QwenGuardParser::default_risk_level`); a *recognized* dialect is never an error even when it grades `Safe`.

**HTTP moderation transport**: posts `{"<text_field>": "<text>"}` with the static headers and one credential.
The per-call `ModerationRequest::url` wins when non-blank (else the configured URL; both blank is `Error::Validation`) and a non-zero per-call `timeout` wins over the configured one.
Auth precedence: `HttpAuth::Bearer`/`Header` from the configuration, else `Bearer` from `request.api_key`, else no credential; a `Header` credential is sent verbatim.
A `2xx` reply is accepted only when it is a JSON **object**; the three rejection messages (all `Error::Transport`, all naming the service and status, never the payload or a credential) are:
`"moderation service <service> returned HTTP <status> with an empty body; expected a JSON object"`,
`"moderation service <service> returned HTTP <status> with a non-JSON body of <n> bytes: <serde reason>"`, and
`"moderation service <service> returned HTTP <status> with a JSON <shape> body; expected a JSON object"` (`<shape>` is `array`, `string`, `number`, `boolean`, or `null`).
A non-2xx status is `Error::Transport` with the status and at most 200 characters of the whitespace-collapsed body.
Retries cover transport errors and the retryable statuses (`429` and every `5xx`) with exponential backoff capped at 5 s (`max_retries` attempts after the first; `0` sends one request).
An unusable `2xx` body is *final* and is never retried, so a garbage `200` fails fast instead of being re-sent.
`json_body` (shared with the vendors) keeps the status mapping, names an empty body explicitly, and decodes JSON; the object requirement lives in the transport-only `require_object`.
`HttpModerationConfig::validate` (private) rejects a blank URL or text field, a non-finite/non-positive timeout, a negative/non-finite backoff, and header names/values that are not an RFC 7230 token / contain control characters.

**Local model backend**: the backend has no inference runtime, so `is_available()` is always `false` and `model_info()` is always all-`false`; `config`, `model_path`, `device`, `with_device`, and `with_parser` keep the Python shape so a runtime can be added without an API change.
`analyze` projects with `text_view_checked()?` first, returns `safe_assessment()` for blank scannable text, and returns `Error::unsupported("local model backend requires an inference runtime (deferred adapter); use PromptInjectionGuardrail mode \"api\" with an ApiModelBackend against a local inference server instead")` for every other payload.
An unscannable payload is therefore `Error::Validation` (naming `unscannable content`), never a pass.
`LocalModelBackendConfig::validate` accepts `auto`/`cpu`/`cuda`/`mps` (trimmed, ASCII case-insensitive) and rejects anything else, including `cuda:0`; a blank `model_path` is allowed at this layer.
A `PromptInjectionGuardrail` with `mode = "local"` and no `model_path` is a construction error (`"model_path is required for local mode"`), and a complete `local` configuration builds but blocks every non-empty scan under the default policy as a `guardrail_error` configuration error.
The supported way to run a local model is `mode = "api"` with an `ApiModelBackend` (or an injected `ApiModelBackend`) pointed at a local/loopback HTTP inference server.

## Constraints

- `GuardrailBackend` is object-safe and async (`#[async_trait]`); `name()` returns a stable identifier (`"rule_based_prompt_injection"`, `"llm_prompt_injection"`, `"api_model"`, `"local_model"`, plus the vendor `NAME` constants).
- Build the examined text with `GuardrailContext::text_view_checked()?`; `text_view()` is the total projection and is not used by any backend. Empty/whitespace scannable text short-circuits to `safe_assessment()` without calling a transport, parser, or judge.
- Every credential is a `SecretString`; `Debug`, findings, and errors expose presence/kind only.
- Error mapping: `Error::Validation` for unusable configuration, an unscannable payload, or an unrecognized reply dialect; `Error::Transport` for status, empty/non-JSON/non-object body, and vendor-code failures; `Error::Timeout` on deadline expiry; `Error::Llm` for an unusable judge reply; `Error::Unsupported` for the missing local runtime or a backend with no transport.
- All HTTP goes through `openjiuwen_common::http::HttpClient`; no direct vendor SDK and no second HTTP stack.
- Logging through `tracing` (target `openjiuwen_security::backends` and `...::vendors::<name>`); never log content, only names, statuses, levels, lengths, shapes.
- Tests are offline: loopback `wiremock` servers and fakes only, no live network, no wall-clock assertions.

## Routing Table

- `./mod.rs` → module list, re-exports, shared `safe_assessment()` (blank-scannable-text verdict only)
- `./rule_based.rs` → `RuleBasedBackendConfig`, `RuleBasedPromptInjectionBackend` (pattern list + unit tests)
- `./llm.rs` → `LlmPromptInjectionBackend`/config, rule fallback, injected judge, `JudgeVerdict`, `parse_verdict`, `DEFAULT_SYSTEM_PROMPT`
- `./api.rs` → `ApiModelBackend`/config, `ModerationTransport`, `ModerationRequest`, built-in HTTP constructors
- `./http.rs` → `HttpModerationTransport`, `HttpModerationConfig`, `HttpAuth`, the object/empty-body reply contract, and the crate-private client/JSON/timeout helpers shared with the vendors
- `./local.rs` → `LocalModelBackend`/config, `ModelInfo`, `SUPPORTED_DEVICES`, device validation, the `mode = "local"` blocking-configuration contract
- `./vendors/` → OpenAI, Azure, and Aliyun moderation adapters (`vendors/CONTEXT.md`)

## Test Strategy

- Module suites behind `#[cfg(test)]` in `http`, `api`, `llm`, `openai`, `azure`, and `aliyun` run against loopback `wiremock` servers: wire path/headers/body, credential placement, retries, status and JSON error mapping, band boundaries, config defaults/serde, and redacting `Debug`.
- `http` pins the reply contract: a `200` with an empty body, a `200` with `not json`, and `200` bodies of every non-object shape (`[]`, `"clean"`, `0`, `false`, `null`) are `ErrorKind::Transport` naming the status and shape without echoing the payload; a garbage `200` is sent exactly once even with retries configured; a `500` is retried once before the following `200` is accepted, and a persistent `500` is attempted `max_retries + 1` times.
- `llm` covers the accepted dialects (plain, fenced, prose), every level mapping and the fail-safe `High` fallback, confidence clamping, `{}`/missing-`has_risk`/non-boolean-`has_risk` → `ErrorKind::Llm` naming the invalid verdict, no-JSON replies, unscannable payloads → `Validation`, blank text never calling the judge, judge errors propagating, and the fail-closed pipeline run.
- `rule_based` and `local` carry pure unit tests (pattern table, empty-text short-circuit, unscannable → `Validation`, device validation, introspection, config serde), and `local` additionally proves the guardrail wiring: an incomplete `mode = "local"` config is a construction `Validation` error, a complete one blocks a non-empty scan with `risk_type == "guardrail_error"` under the default policy, and passes only under an explicit `FailurePolicy::FailOpen`.
- The judge path is exercised end to end with `OpenAiCompatibleClient` over a mock `/v1/chat/completions`, including prompt/request defaults and a fail-closed pipeline run.
- Crate-level behaviour through `GuardrailPipeline` + `BackendGuardrail` (blocking, failure policy, stage binding, no content leak) lives in `../tests/adapters.rs`.
