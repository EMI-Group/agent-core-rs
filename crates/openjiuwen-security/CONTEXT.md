# openjiuwen-security

## Intent

Guardrails and the content-safety pipeline for the openJiuwen runtime.
Ports the Python package `openjiuwen/core/security/guardrail/` (v0.1.20) and extends it with the detectors the Python package advertises but never shipped: sensitive-data/PII redaction, configurable content rules, and tool-call argument validation.
The crate is self-contained (layer 2): it owns the models, the guardrail abstraction, the ordered pipeline, the parsers, the rule engine, and the moderation backends; it never imports `openjiuwen-runner`, so any caller (agent, workflow, tool, CLI, harness) can run a scan in-process.
Detection runs locally (regex rules, PII patterns, parser-graded replies) or remotely: a generic JSON-over-HTTP moderation transport plus first-party OpenAI, Azure Content Safety, and Aliyun Green adapters speak those wire formats over `openjiuwen_common::http::HttpClient`.
No vendor SDK and no local inference runtime are linked: the three vendor adapters are plain HTTP clients, and `LocalModelBackend` keeps the Python configuration/introspection shape while `analyze` returns `Error::Unsupported`.
`openjiuwen-foundation` supplies `prompt::BaseMessage` for the context/pipeline constructors and, for the judge, `llm::Model`/`ChatResponse`/`ModelRequestConfig`.

## Security Model

The Python contract, reproduced exactly, is a two-tier policy:
a detection returns a `GuardrailResult`; **only `CRITICAL` aborts the caller** (Python `AbortError` escaping the callback framework), every lower severity raises a block error that the framework swallows, so execution continues with a recorded finding.

| Situation | Python behavior | Rust behavior |
|---|---|---|
| every guardrail safe | callback returns `None` | `SecurityDecision::Allow`, no findings |
| unsafe `LOW`/`MEDIUM`/`HIGH` | `GuardrailError` raised, swallowed by the framework | finding recorded with `blocking = false`, `SecurityDecision::Allow`, `outcome.risk_level` = highest finding level |
| unsafe at/above `policy.block_level` (default `CRITICAL`) | `AbortError` propagates, call aborted | finding `blocking = true`, `SecurityDecision::Block`, remaining guardrails skipped; `GuardrailOutcome::enforce()` yields `Error::Security` |
| guardrail/backend/parser failure | propagated to the framework, logged, swallowed → execution continues (fail-open) | evaluation error recorded; `FailurePolicy::FailClosed` (**default**) turns it into a blocking finding (`risk_type = "guardrail_error"`, level `CRITICAL`, reason in the message and in `details.error`); the explicit opt-in `FailurePolicy::FailOpen` records it and leaves the decision unchanged (Python parity) |
| payload no detector can read (`null` content, non-string `text` content, non-array `messages` content, messages carrying no text or tool-call arguments) | Python's total text projection scans nothing and the scan passes | `GuardrailContext::text_view_checked()` returns `Error::Validation("unscannable content: ...")`, which the failure policy resolves: blocking `risk_type = "unscannable_content"` under the default, recorded under `FailOpen` |
| 2xx moderation reply of an unrecognized shape | the Python parsers grade it `Safe` | `Error::Validation` naming the expected dialect/schema (both parsers and all three vendor adapters); a recognized *not-flagged* reply is still a genuine `Safe` |
| recognized reply whose verdict is unmappable or missing (`safety: "totally-unheard-of"`, `analysis: {}`, `judgment: null`, an attack prediction with no confidence) | `default_risk_level = Safe` | `QwenGuardParser::default_risk_level` is `RiskLevel::Critical` (blocking under the default `block_level`) and `BertBinaryParser` grades a confidence-less attack prediction `High`; `with_default_risk_level(RiskLevel::Safe)` is the explicit Python-parity opt-in |
| guardrail constructed without a backend | `ValueError("No backend configured for ...")` at call time | not representable: `BackendGuardrail::new` requires an `Arc<dyn GuardrailBackend>`, so the mistake is a compile error |

Rules:
- A backend failure here includes an unusable HTTP reply, a transport error, and an invalid judge verdict; all are recorded as evaluation errors and handled by the failure policy.
- Failures are never silent: every failure is stored in `GuardrailOutcome::errors`, logged at `warn` (target `openjiuwen_security::pipeline`) and, under the default fail-closed policy, also reported as a blocking finding with its reason; `FailurePolicy::FailOpen` is an explicit opt-in for callers that prefer availability, never the default.
- Blank *scannable* text (empty string, empty message array, whitespace) is the only short-circuit to a pass: it is a legitimate payload with nothing to find, not a failure.
- A guardrail never panics on untrusted content; a malformed user-supplied regex is a construction error (`Error::Validation`), not a panic.
- `PromptInjectionGuardrail` mode `local` builds a guardrail whose backend always errors on non-empty text (`backends/local.rs`, no in-process ML runtime), so under the default policy local mode blocks every non-empty scan; `mode = "api"` with an `ApiModelBackend` pointed at a loopback inference server is the supported local-model path.
- `GuardrailContext::messages` stores `Value::Null` when the messages cannot be serialized (`context.rs`); `text_view_checked()` reports that (and every other unscannable payload) as a validation error instead of scanning nothing.
- A `SecurityConfig` that leaves security enabled but registers no guardrail is `Error::Config` unless `allow_all` is set (`enabled = false` is itself the explicit off switch and keeps building the empty pipeline).
- Findings and reports never carry raw matched text or vendor matched words: `details` holds pattern names, rule names, counts, scores, and vendor codes only; `GuardrailOutcome::redacted()` additionally runs every `details` value through `openjiuwen_common::security::redact_json`.
- Unknown/missing risk types degrade to the literal `"unknown"` in messages (Python parity).
- Finding messages: blocking at `CRITICAL` → `"Critical security risk detected: <risk_type>"` (Python `AbortError` reason); blocking at a lowered `block_level` → `"Guardrail blocked: <risk_type> risk detected"` (Python `GuardrailError` message); non-blocking findings → `"Security finding: <risk_type> (level=<LEVEL>)"`. `GuardrailOutcome::enforce()` returns `Error::Security(<message>)` for the first blocking finding.

## API Surface

All public data types are `serde` (`snake_case`, `#[serde(default)]`, unknown fields tolerated) and JSON field names match the Python dataclass field names.
`RiskLevel` serializes as its Python value (`"safe"|"low"|"medium"|"high"|"critical"`) and is `Ord` in that severity order.

```
level:      RiskLevel { Safe, Low, Medium, High, Critical } (as_str/name/is_risky/at_least/from_value), GuardrailContentType { Text, Messages, ToolCall, Raw }
context:    ScanStage { LlmInvokeInput, LlmInvokeOutput, ToolInvokeInput, ToolInvokeOutput, UserInput, Other(String) } with event()/from_event()/content_type()
            constants EVENT_LLM_INVOKE_INPUT "_framework:llm_invoke_input", EVENT_LLM_INVOKE_OUTPUT, EVENT_TOOL_INVOKE_INPUT, EVENT_TOOL_INVOKE_OUTPUT, EVENT_USER_INPUT "user_input"
            GuardrailContext { content_type, content: Value, event: String, metadata: BTreeMap<String, Value> } + text(impl Into<String>, ScanStage)/messages(&[BaseMessage], ScanStage)/tool_call(impl Into<String>, &Value, ScanStage)/tool_output(same as tool_call)/raw(impl Into<String>, Value) constructors,
            get_text()/get_messages()/get_tool_name()/text_view()/text_view_checked() -> Result<Cow<str>>/with_metadata()/stage(); manual redacting Debug
result:     GuardrailResult { is_safe, risk_level, risk_type, details, modified_data } with pass()/pass_with(details)/block(level, risk_type, details, modified_data)/message()
            RiskAssessment { has_risk, risk_level, risk_type, confidence, details } with into_result()
finding:    Finding { guardrail, risk_level, risk_type, message, blocking, details, modified_data } + finding_message(risk_type, risk_level, blocking) shared wording builder
            GuardrailEvaluation { guardrail, result: Option<GuardrailResult>, error: Option<String> }
            SecurityDecision { Allow, Block }
            GuardrailOutcome { stage, evaluations, findings, errors, risk_level, decision, blocked_by } with is_safe()/is_blocked()/enforce()/violation()/modified_data()/redacted()
            GuardrailViolation { risk_level, risk_type, message, event, details } + From<GuardrailViolation> for openjiuwen_common::error::Error
guardrail:  #[async_trait] trait Guardrail { name(), priority() = 100, stages() = &[], handles(stage), check(&GuardrailContext) -> Result<GuardrailResult> }
            #[async_trait] trait GuardrailBackend { name(), analyze(&GuardrailContext) -> Result<RiskAssessment> }
            BackendGuardrail (adapts a backend into a Guardrail under an independent name; with_priority/with_stages/backend; always owns an Arc<dyn GuardrailBackend>)
pipeline:   GuardrailPipeline { add(&mut self, impl Guardrail + 'static)/add_arc(&mut self, Arc<dyn Guardrail>)/extend(IntoIterator<Item = Arc<dyn Guardrail>>)/guardrails()/policy(), builders new()/with_policy(SecurityPolicy)/Default } running guardrails by descending priority, registration order as tiebreak
            run(&GuardrailContext) -> GuardrailOutcome (infallible; pre-checks text_view_checked() and resolves a failure by the policy before running the guardrails); scan_text(ScanStage, impl Into<String>) / scan_user_input(impl Into<String>) / scan_prompt(&[BaseMessage]) / scan_model_output(impl Into<String>) / scan_tool_input(impl Into<String>, &Value) / scan_tool_output(impl Into<String>, &Value) -> GuardrailOutcome
            constants GUARDRAIL_ERROR_RISK_TYPE "guardrail_error", UNSCANNABLE_RISK_TYPE "unscannable_content", CONTEXT_GUARDRAIL "context"
config:     SecurityConfig { enabled, allow_all = false, policy, targets, prompt_injection, sensitive_data, content_rules, tool_arguments } with default()/protected()/pipeline() -> Result<GuardrailPipeline> (an enabled config that registers no guardrail is Error::Config unless allow_all)
            SecurityPolicy { block_level = Critical, failure_policy = FailClosed, stop_on_block = true }, FailurePolicy { FailClosed (default), FailOpen (opt-in) },
            ScanTargets { prompts, user_input, model_outputs, tool_inputs, tool_outputs } (all true; stages() maps them to ScanStage)
report:     RedactedOutcome { decision, risk_level, findings, errors, guardrails, stage } (serde + Display summary)
rules:      shared regex engine (compiled rules, case-insensitive search, marker redaction) used by every built-in detector
parsers:    ModelOutputParser { name(), parse(&Value) -> Result<RiskAssessment> } (fallible: an unrecognized dialect is Error::Validation naming the expected dialects); BertBinaryParser, QwenGuardParser (verdict fallback `default_risk_level` = `RiskLevel::Critical`; `with_default_risk_level(RiskLevel::Safe)` is the Python-parity opt-in), Thresholds (see ./src/parsers/CONTEXT.md)
backends:   RuleBasedPromptInjectionBackend + RuleBasedBackendConfig (six Python default patterns; first match wins, HIGH),
            LlmPromptInjectionBackend + LlmPromptInjectionBackendConfig (deterministic rule fallback; with_model(Arc<dyn Model>) installs a real JSON-verdict judge, with_model_request/parse_verdict/has_judge),
            ApiModelBackend + ApiModelBackendConfig (new()/http(config)?/with_http_defaults()?/with_parser/with_transport; an injected transport wins), ModerationTransport + ModerationRequest,
            HttpModerationTransport + HttpModerationConfig { url, headers, auth, timeout, max_retries, retry_initial_backoff, text_field } + HttpAuth { None, Bearer, Header } (an empty, non-JSON or non-object 2xx body is Error::Transport and is not retried),
            LocalModelBackend + LocalModelBackendConfig (config()/model_path()/device()/is_available() == false/with_device/with_parser; SUPPORTED_DEVICES, ModelInfo, analyze() => Error::Unsupported),
            vendors::{OpenAiModerationBackend, AzureContentSafetyBackend, AliyunGreenBackend} with their configs and AliyunGreenAuth (see ./src/backends/CONTEXT.md and ./src/backends/vendors/CONTEXT.md)
guardrails: PromptInjectionGuardrail + PromptInjectionGuardrailConfig (Python parity, mode rules/api/local) with new()/with_config()/with_config_and_parser(config, Arc<dyn ModelOutputParser>)/with_backend(Arc<dyn GuardrailBackend>)/with_stages/with_priority/backend_name/backend/config + const DEFAULT_STAGES = [LlmInvokeInput, ToolInvokeOutput],
            SensitiveDataGuardrail + config + SensitivePattern/SensitiveValidator (PII detect + redact) with new()/with_config()/redact_text(&str) -> RedactionOutcome, SensitiveMatch/RedactionOutcome,
            ContentRuleGuardrail + ContentRule + config with new(ContentRuleGuardrailConfig)/from_patterns(Vec<ContentRule>) (no Default impl), ToolArgumentGuardrail + config with new()/with_config() + const DEFAULT_STAGES = [ToolInvokeInput];
            every built-in also has with_stages(impl IntoIterator<Item = ScanStage>), with_priority(i32), config()
```

Crate root re-exports every public type flat (`openjiuwen_security::GuardrailPipeline`, `openjiuwen_security::HttpModerationTransport`, `openjiuwen_security::OpenAiModerationBackend`, `openjiuwen_security::SUPPORTED_DEVICES`, ...); modules stay public for qualified paths and module constants (`vendors::openai::NAME`, `DEFAULT_MODEL`, ...) stay behind their module.
Python name mapping: `GuardrailResult.pass_` → `GuardrailResult::pass`, `GuardrailResult.block` → `GuardrailResult::block`, `BaseGuardrail` → `Guardrail` + `BackendGuardrail`, `GuardrailContext.get_text` → `GuardrailContext::get_text`, `RuleBasedBackendConfig` → `RuleBasedBackendConfig`.

## Constraints

- Async-first on tokio; `Guardrail`/`GuardrailBackend` are `#[async_trait]` and object-safe (`Arc<dyn Guardrail>`); the regex/PII/content-rule detectors are CPU-only, while any detector wired to a remote backend (`HttpModerationTransport`, `ApiModelBackend::http`, the LLM judge, a vendor adapter) performs real I/O through the shared HTTP client.
- Errors: `openjiuwen_common::error::{Error, Result}` only; no parallel error enum. `Error::security` marks blocking/guardrail verdicts, `Error::config` a config that registers no detector, `Error::validation` bad config, an invalid regex, an unscannable payload, or an unrecognized model-output dialect/vendor schema, `Error::transport`/`Error::timeout` remote wire failures (including a 2xx body that is empty, non-JSON or not an object), `Error::llm` an unusable judge reply, `Error::unsupported` deferred adapters. No `unwrap`/`expect` outside module-level test cfgs and `#[allow(clippy::expect_used)]` static regexes.
- No `unsafe`, no globals: a pipeline is an owned value; there is no process-global registry (Python's `Runner.callback_framework` registration is the runner's concern, wired outside this crate).
- Logging through `tracing` only (targets `openjiuwen_security::<module>`, vendor adapters `openjiuwen_security::backends::vendors::<name>`); no `stdout`. Guardrail content and credentials are never logged; log names, stages, levels, statuses, lengths.
- Every credential is a `SecretString`; configs round-trip through JSON but `Debug` shows presence/kind only.
- Files target ≤1000 lines; the Aliyun adapter is a directory module (`vendors/aliyun/{mod,signing,tests}.rs`) so the wire mapping, the ACS3 signing helpers, and their tests stay separately reviewable.
- Default features are empty and every test runs offline: loopback `wiremock` servers, no live network, no credentials.

## Routing Table

- `./src/level.rs` → `RiskLevel`, `GuardrailContentType`, severity ordering and JSON values
- `./src/context.rs` → `ScanStage`, event-name constants, `GuardrailContext` and text projection
- `./src/result.rs` → `GuardrailResult`, `RiskAssessment`
- `./src/finding.rs` → `Finding`, `GuardrailEvaluation`, `SecurityDecision`, `GuardrailOutcome`, `GuardrailViolation`, `Error` conversion
- `./src/guardrail.rs` → `Guardrail` and `GuardrailBackend` traits, `BackendGuardrail`
- `./src/pipeline.rs` → ordered pipeline, aggregation, blocking policy, scan helpers
- `./src/config.rs` → `SecurityConfig`, `SecurityPolicy`, `FailurePolicy`, `ScanTargets`, pipeline construction
- `./src/report.rs` → `RedactedOutcome` and redacted rendering of an outcome
- `./src/rules.rs` → shared regex rule engine and redaction primitives
- `./src/parsers/` → model-output parsers (`BertBinaryParser`, `QwenGuardParser`) (`src/parsers/CONTEXT.md`)
- `./src/backends/` → rule-based/LLM/API/local backends, the generic HTTP moderation transport (`http.rs`), and the vendor adapters (`src/backends/CONTEXT.md`)
- `./src/backends/vendors/` → OpenAI, Azure Content Safety, and Aliyun Green adapters with their configs and wire contracts (`src/backends/vendors/CONTEXT.md`)
- `./src/guardrails/` → built-in guardrails: prompt injection, sensitive data, content rules, tool arguments (`src/guardrails/CONTEXT.md`)
- `./tests/` → crate-level behavior suites (models, pipeline, prompt injection, sensitive data, tool arguments, config, helpers, moderation adapters) (`tests/CONTEXT.md`)

## Test Strategy

- `tests/models.rs` — `RiskLevel` values/order/serde, `GuardrailResult::pass`/`block` shapes, `RiskAssessment` defaults, serde round-trips, redacting `Debug` for `GuardrailContext`, JSON field names.
- `tests/pipeline.rs` — priority ordering (descending, stable), stage filtering, aggregation of findings, severity/blocking matrix (LOW/MEDIUM/HIGH allow, CRITICAL block, configurable `block_level`), skip-after-block, the fail-closed default and the explicit `FailOpen` opt-in (the `guardrail_error` finding shape, message, `details.error` and `enforce()`), the unscannable-context outcome (`unscannable_content`, `blocked_by` is `context`), a multimodal-only messages payload, and `modified_data` merge.
- `tests/prompt_injection.rs` — six Python default patterns (case-insensitive, one finding per scan), `matched_pattern` details, config defaults, config validation errors (exact substrings), the recognized parser dialect tables plus the unrecognized-reply error table for both parsers (`ErrorKind::Validation`, the message names the parser, no payload echo), the Qwen severity-word exact matches (`HIGH` ⇒ `High`) and the `Critical` fallback for an unmappable `severity_level`, `LlmPromptInjectionBackend` rule delegation, `ApiModelBackend` empty-content/transport-required behavior through an in-file fake transport, a loopback `200 {}` reply blocking instead of grading safe, `LocalModelBackend`'s unsupported shape and the blocking `local`-mode default, rule-backend default level `HIGH`.
- `src/parsers/{bert,qwen}.rs` (module suites) — the extraction table, the threshold bands and argmax/softmax rules, every Qwen dialect and its text format, the exact severity-word vocabulary with containment limited to `controversial`/`unsafe`/`safe`, `an_unmappable_verdict_or_missing_verdict_is_not_safe` (Critical under `new()`, Safe under `with_default_risk_level(RiskLevel::Safe)`), `a_predicted_attack_without_confidence_errs_upward` (`{"label": 1}`, a null and a non-numeric confidence ⇒ `High`, `{"label": 0}` ⇒ `Safe`), and the unrecognized-output errors with no payload echo.
- `tests/sensitive_data.rs` — email/phone/card/id/secret detection, Luhn rejection, redaction output (`<REDACTED>`), `modified_data.sanitized_text`, no false positives on benign text, custom patterns/extra patterns, marker override.
- `tests/tool_arguments.rs` — denied/allowed tool names, shell-injection patterns in arguments, nested JSON argument traversal, sensitive-data redaction inside arguments, stage binding (tool input only).
- `tests/config.rs` — every detector default preserved (Python `PromptInjectionGuardrailConfig`/`RuleBasedBackendConfig` fields and values), the fail-closed policy default, `SecurityConfig::default()` building the empty pipeline (the explicit off switch), `allow_all` as the opt-in for an enabled-but-empty config, `Error::Config` for an enabled config with no target or no detector, `protected()` registering the default set, `pipeline()` validation errors, full serde round-trip.
- `tests/helpers.rs` — `scan_prompt`/`scan_user_input`/`scan_model_output`/`scan_tool_input`/`scan_tool_output` context shapes (stages, events, metadata `tool_name`/`message_count`), all-message scanning (a secret in an earlier turn is still found), unscannable payloads blocking through every helper while an empty pipeline still allows them, `enforce()` on blocked vs finding-only outcomes, `RedactedOutcome`/`Debug` keeping the failure reason and never the payload, and `modified_data()` merge order.
- `tests/adapters.rs` — loopback `wiremock` servers driven through `GuardrailPipeline` + `BackendGuardrail`: each vendor adapter gets benign, blocking, sub-threshold (lowered `block_level`), transport-failure (explicit `FailOpen` and the default fail-closed), unrecognized-2xx (`{}`, non-JSON, JSON non-object) blocking with the vendor or dialect named and exactly one request, stage-binding, and no-content-leak cases; the generic `ApiModelBackend` + `HttpModerationTransport` path gets the recorded wire request, a lowered-`block_level` block, and the failure-policy pair.
- Module-level `#[cfg(test)]` suites cover the backends' wire requests, retries, status/JSON mapping, judge verdicts, signature vectors, config serde, and redacting `Debug` (see `src/backends/CONTEXT.md`).
- Behavior-port tests cite the Python test they mirror in a comment (`// ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py`).
- Gates: `cargo fmt -p openjiuwen-security`, `cargo test -p openjiuwen-security`, `cargo clippy -p openjiuwen-security --all-targets -- -D warnings` all clean offline.

## Design Decisions

- **`Guardrail` takes a `GuardrailContext`, not Python's `*args/**kwargs`.** Python's `BaseGuardrail.extract_context(event, *args, **kwargs)` is untyped and positional-message bugs silently degrade scans; here the pipeline builds the context from typed helpers, and `ContextExtractor` plug-ins are unnecessary because every built-in consumes `text_view()`.
- **`ScanStage` carries Python's exact event strings** (`"_framework:llm_invoke_input"`, ...) so a runner callback adapter can bind by string without a translation table; `Other(String)` accepts event names this crate does not know.
- **Aggregation and the pipeline are an addition.** Python guardrails only bind to callback events and never produce a combined verdict, so nothing existed to port; the model here is one ordered run producing evaluations, findings, and a decision. Ordering: `priority` descending (Python semantics: higher first), registration order as the tiebreak.
- **`SecurityDecision` is a decision value, not an exception.** `run` is infallible by design (guardrail failures are data); `enforce()` converts a blocked outcome into `Error::Security`, and integration helpers return `GuardrailOutcome` so callers can inspect findings or redact instead of unwinding.
- **Failures are fail-closed by default.** `FailurePolicy::FailClosed` is the default and `FailOpen` is an explicit opt-in: an unusable detector must never read as safe, because a mis-pointed endpoint, an expired credential or a broken parser would otherwise silently disable the guardrail. Python's swallow-and-continue behavior survives as `FailOpen` for callers that deliberately prefer availability, so the choice is visible in the policy instead of implicit in the crate defaults.
- **A failure is reported, not just recorded.** A fail-closed failure becomes a blocking `CRITICAL` finding whose message is `Guardrail blocked: <risk_type> risk detected (<reason>)` and whose `details.error` carries the reason (a guardrail/backend/parser error, or `unscannable content: ...`), so a caller that only inspects `enforce()`/`violation()` still sees why the call was denied. `risk_type` separates `guardrail_error` from `unscannable_content`, and `outcome.errors` keeps the `"<guardrail>: <error>"` entries.
- **The payload projection is checked, not total.** `text_view()` stays total for compatibility, but `text_view_checked()` is what the detectors and the pipeline use: `null` content, a `text` context whose content is not a string, a `messages` context that is not an array or carries no text/tool-call arguments, and unserializable content are failures. Empty scannable payloads (empty string, empty message array, whitespace) still pass, because "nothing to find" is not "could not look".
- **An empty enabled config is a configuration error.** `SecurityConfig::pipeline()` refuses to build a pipeline that scans nothing unless `allow_all` is set, so a misspelled detector table or an all-off `ScanTargets` cannot silently protect nothing. `enabled = false` remains the documented explicit off switch (and still builds the empty pipeline), so a host that constructs `SecurityConfig::default()` and replaces the pipeline later is unaffected.
- **Parsers are fallible and name the dialect.** An unrecognized model-output shape is `Error::Validation` rather than a synthetic `Safe` assessment, because a `Safe` grade from an endpoint we do not understand is indistinguishable from a clean scan.
- **A missing or unmappable verdict errs upward.** Python graded both `default_risk_level = Safe` and an attack prediction with no confidence as `Safe`; here `QwenGuardParser::new()` sets `default_risk_level` to `RiskLevel::Critical` (so an unknown token or an absent verdict blocks under the default `block_level`), and `BertBinaryParser` grades a predicted attack with no usable confidence `High`, the level a confident attack gets. `QwenGuardParser::with_default_risk_level(RiskLevel::Safe)` is the explicit opt-in for Python's grading; the exact-match severity words `critical`/`high`/`medium`/`low` map before the fallback, and containment matching stays limited to `controversial`/`unsafe`/`safe` so `below` is not read as `low`.
- **Vendor adapters validate the reply schema.** A recognized not-flagged reply is a genuine `Safe`, but a reply whose schema is not understood (missing or renamed fields, empty result arrays, a non-boolean `flagged`) is `Error::Validation` naming the vendor and the required fields; the shared HTTP transport additionally rejects an empty, non-JSON or non-object 2xx body as `Error::Transport` and never retries it.
- **`block_level` defaults to `Critical`** (Python effective behavior) and is configurable; lowering it makes `LOW`/`MEDIUM`/`HIGH` findings blocking with Python's `GuardrailError` wording.
- **Message scanning is strictly stronger than Python**: Python's `PromptInjectionGuardrail` scans only the last message of `messages`; here `text_view()` joins every message's text and tool-call arguments with `\n`, so a malicious earlier turn is caught too. Single-message prompts behave identically.
- **`SensitivePattern` matching is regex + optional validator** (Luhn for card numbers), never a bare substring, and redaction is applied left-to-right on the original text so overlapping matches cannot corrupt each other.
- **Guardrails do not chain sanitized content between each other** (Python does not either): each sees the original context, and `GuardrailOutcome::modified_data()` merges their `modified_data` objects in evaluation order (later keys win).
- **The parser is injected, not serialized**: Python's `PromptInjectionGuardrailConfig` holds a parser object; here the serde config carries only data and `PromptInjectionGuardrail::with_config_and_parser` injects an `Arc<dyn ModelOutputParser>`, while `bert`/`qwen` defaults are built from `bert_thresholds`, `attack_class_id`, and `qwen_risk_type`.
- **Backend configs keep secrets as `SecretString` and never hold trait objects in serde**: parsers arrive through `with_parser`, transports through `with_transport`, judges through `with_model`, so every config round-trips through JSON.
- **Remote moderation is a plain HTTP client, not a vendor SDK.** One generic transport (`{text_field: text}` in, JSON out) plus three thin vendor adapters share `http_client`/`json_body`/`resolve_timeout`, so request construction, retries, and error mapping live in one place.
- **An injected transport always wins over the built-in HTTP default** (in either call order), so tests and deployments can substitute a fake or a vendor-specific transport without rebuilding the backend; a missing transport stays `Error::Unsupported` rather than failing at construction.
- **The LLM judge is injected, not built.** `LlmPromptInjectionBackend::with_model` takes any `foundation::llm::Model`; without one the backend keeps Python's deterministic rule-based fallback, so the default path stays offline and reproducible.
- **Vendor adapters report what they can prove.** Aliyun `RiskWords`/`Reason` and Azure blocklist terms are dropped, only counts, category severities, scores, and request ids reach `details`.

## Known Gaps and Deferred Adapters

- Every built-in detector is implemented (`guardrails::{prompt_injection, sensitive_data, content_rules, tool_arguments}`) and `SecurityConfig::pipeline()` assembles them from configuration.
- The remaining gaps are the runtime and vendor transports below, not missing detectors.
- `LocalModelBackend` has no inference runtime: configuration, device validation, and introspection are preserved, but `analyze` returns `Error::Unsupported` (Python needs `torch`/`transformers`). Under the fail-closed default that makes `PromptInjectionGuardrail` mode `local` a blocking configuration error on every non-empty scan; the supported local-model path is `mode = "api"` with an `ApiModelBackend` (or any `ModerationTransport`) pointed at a loopback inference server, which the suites exercise against `wiremock`.
- `LlmPromptInjectionBackend` without an injected judge runs the deterministic rule-based detector (Python's stub semantics); that delegation is a detector choice, not a failure, so it is not routed through the failure policy — inject a judge with `with_model` when the LLM verdict is required.
- A blank *scannable* payload (empty string, whitespace, empty message array) is allowed even when the configured backend could not have answered: it is a legitimate empty payload, and blocking it would deny empty prompts rather than fail closed on detection.
- No vendor SDKs: the OpenAI, Azure, and Aliyun adapters speak the vendor HTTP APIs directly; there is no moderation transport for any other vendor (DashScope, Google, Bedrock, ...), and adding one means a `ModerationTransport`/`GuardrailBackend` implementation over the shared client.
- Moderation is one request per scan and one graded result (OpenAI `results[0]`); batching, streaming moderation, and multi-result aggregation are not implemented.
- No process-global registration helper: Python's `register_guardrail`/`unregister_guardrail` bind to `Runner.callback_framework`; the runner crate owns that wiring (see `docs/PARITY.md` security row).
- Python quirks deliberately not reproduced: `PromptInjectionGuardrail.with_rules()` (documented but never defined), double-registration bookkeeping bug, and `GuardrailContext(content_type=type(None))` fallback.

## Known Issues

- `backends::http::http_client` builds the shared client without `max_response_bytes` (`HttpClientConfig::default()` leaves it `None`), so a moderation endpoint can stream an unbounded body into memory, and `body_snippet` then copies it again (`from_utf8_lossy`, a `split_whitespace` vector, a `join`) before keeping 200 characters.
- The text projection is recomputed per detector and per backend: `GuardrailPipeline::run` projects once and discards it, then `PromptInjectionGuardrail::check`, its rule/API/LLM backend, `SensitiveDataGuardrail::check`, `ContentRuleGuardrail::check` and `ToolArgumentGuardrail::check` each call `text_view_checked()` again, so a `Messages` prompt scan under `SecurityConfig::protected()` joins the whole conversation four times and a tool-input scan serializes the arguments two to four times.
- `redact_ranges` considers ranges from the end backwards, so when two input ranges overlap the rightmost one wins; ranges with the same start are considered in input order (the replaced end-first implementation sorted them unstably, leaving that tie unspecified).
- Range claiming and redaction are linear-ish: `rules::RangeClaims` keeps accepted ranges in a `BTreeMap` keyed by start (two probes per candidate, `O(m log m)` for `m` matches), `redact_ranges` selects in one descending pass and rebuilds the output in one pre-sized `String::with_capacity` pass, and `SensitiveDataGuardrail::redact_text` uses the same claim set, so a tool result holding thousands of PII hits no longer costs `O(m²)` comparisons or per-range string memmoves inside the scan task.
- The vendor adapters (`vendors::{openai, azure, aliyun}`) project with `text_view_checked()?` like every local detector, so an unscannable payload is `Error::Validation("unscannable content: ...")` — a failure the policy resolves — and never a posted compact JSON answered Safe.
- `ToolArgumentGuardrail` scans argument text twice when redaction is on (the joined text in `sensitive_verdict`, then every string leaf again in `redact_value`), serializes the whole argument value just to measure `max_argument_bytes`, and redacts per leaf so PII split across two leaves is reported but not redacted in `sanitized_arguments`.
- `tool_arguments::collect_strings` and `redact_value` recurse over caller-supplied JSON without a depth cap; serde_json's default 128-level parse limit covers parsed input, but a programmatically built `Value` can overflow the stack.

## Python Mapping

| Python (`openjiuwen/core/security/guardrail/`) | Rust |
|---|---|
| `enums.py` `RiskLevel`, `GuardrailContentType` | `level::RiskLevel`, `level::GuardrailContentType` |
| `models.py` `GuardrailResult`, `RiskAssessment` | `result::GuardrailResult`, `result::RiskAssessment` |
| `context.py` `GuardrailContext` | `context::GuardrailContext` |
| `context.py` `ModelOutputParser`/`BertBinaryParser`/`QwenGuardParser` | `parsers::{ModelOutputParser, BertBinaryParser, QwenGuardParser}` |
| `backends.py` `RuleBasedPromptInjectionBackend` | `backends::rule_based::RuleBasedPromptInjectionBackend` |
| `backends.py` `LLMPromptInjectionBackend` | `backends::llm::LlmPromptInjectionBackend` (rule fallback + optional injected judge) |
| `backends.py` `APIModelBackend` + its inlined `httpx` call | `backends::api::ApiModelBackend` + `backends::http::HttpModerationTransport` |
| `backends.py` `LocalModelBackend` (`torch`/`transformers`) | `backends::local::LocalModelBackend` (config/introspection only, `Unsupported`) |
| `guardrail.py` `BaseGuardrail` | `guardrail::Guardrail` (trait) + `guardrail::BackendGuardrail` |
| `builtin.py` `PromptInjectionGuardrail`, `PromptInjectionGuardrailConfig` | `guardrails::prompt_injection::{PromptInjectionGuardrail, PromptInjectionGuardrailConfig}` |
| `__init__.py` `register_guardrail`/`unregister_guardrail` | out of scope (runner callback wiring) |
| — (no Python equivalent) | `pipeline`, `config`, `report`, `guardrails::{sensitive_data, content_rules, tool_arguments}`, `rules` |
| — (no Python equivalent) | `backends::vendors::{openai, azure, aliyun}` moderation adapters |
| — (no Python equivalent) | `backends::llm::LlmPromptInjectionBackend::with_model` real LLM judge (Python builds and discards a prompt instead) |
