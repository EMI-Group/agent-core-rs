# openjiuwen-security

## Intent

Guardrails and the content-safety pipeline for the openJiuwen runtime.
Ports the Python package `openjiuwen/core/security/guardrail/` (v0.1.20) and extends it with the detectors the Python package advertises but never shipped: sensitive-data/PII redaction, configurable content rules, and tool-call argument validation.
The crate is self-contained (layer 2): it owns the models, the guardrail abstraction, the ordered pipeline, the parsers, and the rule engine; it never imports `openjiuwen-runner`, so any caller (agent, workflow, tool, CLI, harness) can run a scan synchronously in-process.
Vendor/cloud moderation services are a trait seam only (`GuardrailBackend` + `ModerationTransport`); no HTTP transport and no local model runtime live here.

## Security Model

The Python contract, reproduced exactly, is a two-tier policy:
a detection returns a `GuardrailResult`; **only `CRITICAL` aborts the caller** (Python `AbortError` escaping the callback framework), every lower severity raises a block error that the framework swallows, so execution continues with a recorded finding.

| Situation | Python behavior | Rust behavior |
|---|---|---|
| every guardrail safe | callback returns `None` | `SecurityDecision::Allow`, no findings |
| unsafe `LOW`/`MEDIUM`/`HIGH` | `GuardrailError` raised, swallowed by the framework | finding recorded with `blocking = false`, `SecurityDecision::Allow`, `outcome.risk_level` = highest finding level |
| unsafe at/above `policy.block_level` (default `CRITICAL`) | `AbortError` propagates, call aborted | finding `blocking = true`, `SecurityDecision::Block`, remaining guardrails skipped; `GuardrailOutcome::enforce()` yields `Error::Security` |
| guardrail/backend/parser failure | propagated to the framework, logged, swallowed → execution continues (fail-open) | evaluation error recorded; `FailurePolicy::FailOpen` (default, parity) leaves the decision unchanged; `FailurePolicy::FailClosed` turns it into a blocking finding (`risk_type = "guardrail_error"`, level `CRITICAL`) |
| guardrail constructed without a backend | `ValueError("No backend configured for ...")` | `Err(Error::Security("No backend configured for <name>. Use ..."))` from `check` |

Rules:
- Failures are never silent: every failure is stored in `GuardrailOutcome::errors` and logged at `warn` (target `openjiuwen_security::pipeline`); the fail-open default is chosen for parity and is documented, not hidden.
- A guardrail never panics on untrusted content; a malformed user-supplied regex is a construction error (`Error::Validation`), not a panic.
- Findings and reports never carry raw matched text: `details` holds pattern names, rule names, counts, and scores only; `GuardrailOutcome::redacted()` additionally runs every `details` value through `openjiuwen_common::security::redact_json`.
- Unknown/missing risk types degrade to the literal `"unknown"` in messages (Python parity).
- Finding messages: blocking at `CRITICAL` → `"Critical security risk detected: <risk_type>"` (Python `AbortError` reason); blocking at a lowered `block_level` → `"Guardrail blocked: <risk_type> risk detected"` (Python `GuardrailError` message); non-blocking findings → `"Security finding: <risk_type> (level=<LEVEL>)"`. `GuardrailOutcome::enforce()` returns `Error::Security(<message>)` for the first blocking finding.

## API Surface

All public data types are `serde` (`snake_case`, `#[serde(default)]`, unknown fields tolerated) and JSON field names match the Python dataclass field names.
`RiskLevel` serializes as its Python value (`"safe"|"low"|"medium"|"high"|"critical"`) and is `Ord` in that severity order.

```
level:      RiskLevel { Safe, Low, Medium, High, Critical } (as_str/name/is_risky/at_least/from_value), GuardrailContentType { Text, Messages, ToolCall, Raw }
context:    ScanStage { LlmInvokeInput, LlmInvokeOutput, ToolInvokeInput, ToolInvokeOutput, UserInput, Other(String) } with event()/from_event()/content_type()
            constants EVENT_LLM_INVOKE_INPUT "_framework:llm_invoke_input", EVENT_LLM_INVOKE_OUTPUT, EVENT_TOOL_INVOKE_INPUT, EVENT_TOOL_INVOKE_OUTPUT, EVENT_USER_INPUT "user_input"
            GuardrailContext { content_type, content: Value, event: String, metadata: BTreeMap<String, Value> } + text()/messages()/tool_call()/raw() constructors,
            get_text()/get_messages()/get_tool_name()/text_view()/with_metadata(); manual redacting Debug
result:     GuardrailResult { is_safe, risk_level, risk_type, details, modified_data } with pass()/pass_with(details)/block(level, risk_type, details, modified_data)/message()
            RiskAssessment { has_risk, risk_level, risk_type, confidence, details } with into_result()
finding:    Finding { guardrail, risk_level, risk_type, message, blocking, details, modified_data }
            GuardrailEvaluation { guardrail, result: Option<GuardrailResult>, error: Option<String> }
            SecurityDecision { Allow, Block }
            GuardrailOutcome { stage, evaluations, findings, errors, risk_level, decision, blocked_by } with is_safe()/is_blocked()/enforce()/violation()/modified_data()/redacted()
            GuardrailViolation { risk_level, risk_type, message, event, details } + From<GuardrailViolation> for openjiuwen_common::error::Error
guardrail:  #[async_trait] trait Guardrail { name(), priority() = 100, stages() = &[], handles(stage), check(&GuardrailContext) -> Result<GuardrailResult> }
            #[async_trait] trait GuardrailBackend { name(), analyze(&GuardrailContext) -> Result<RiskAssessment> }
            BackendGuardrail (backend adapter implementing Guardrail; no backend => Error::Security)
pipeline:   GuardrailPipeline { add/add_arc/extend/guardrails/policy } running guardrails by descending priority, registration order as tiebreak
            run(&GuardrailContext) -> GuardrailOutcome; scan_text/scan_user_input/scan_prompt/scan_model_output/scan_tool_input/scan_tool_output helpers
config:     SecurityConfig { enabled, policy, targets, prompt_injection, sensitive_data, content_rules, tool_arguments } with default()/protected()/pipeline() -> Result<GuardrailPipeline>
            SecurityPolicy { block_level = Critical, failure_policy = FailOpen, stop_on_block = true }, FailurePolicy { FailOpen, FailClosed },
            ScanTargets { prompts, user_input, model_outputs, tool_inputs, tool_outputs } (all true; stages() maps them to ScanStage)
report:     RedactedOutcome { decision, risk_level, findings, errors, guardrails, stage } (serde + Display summary)
rules:      shared regex engine (compiled rules, case-insensitive search, marker redaction) used by every built-in detector
parsers:    ModelOutputParser trait; BertBinaryParser, QwenGuardParser (see ./src/parsers/CONTEXT.md)
backends:   RuleBasedPromptInjectionBackend + RuleBasedBackendConfig (six Python default patterns),
            LlmPromptInjectionBackend + LlmPromptInjectionBackendConfig (Python stub semantics: delegates to the rule backend),
            ApiModelBackend + ApiModelBackendConfig (transport seam), LocalModelBackend + LocalModelBackendConfig (unsupported seam)
guardrails: PromptInjectionGuardrail + PromptInjectionGuardrailConfig (Python parity, mode rules/api/local),
            SensitiveDataGuardrail + config + SensitivePattern (PII detect + redact),
            ContentRuleGuardrail + ContentRule + config, ToolArgumentGuardrail + config
```

Crate root re-exports every item above flat (`openjiuwen_security::GuardrailPipeline`, `openjiuwen_security::RiskLevel`, ...); modules stay public for qualified paths.
Python name mapping: `GuardrailResult.pass_` → `GuardrailResult::pass`, `GuardrailResult.block` → `GuardrailResult::block`, `BaseGuardrail` → `Guardrail` + `BackendGuardrail`, `GuardrailContext.get_text` → `GuardrailContext::get_text`, `RuleBasedBackendConfig` → `RuleBasedBackendConfig`.

## Constraints

- Async-first on tokio; `Guardrail`/`GuardrailBackend` are `#[async_trait]` and object-safe (`Arc<dyn Guardrail>`); CPU-only scanning adds no I/O.
- Errors: `openjiuwen_common::error::{Error, Result}` only; no parallel error enum. `Error::security` marks blocking/guardrail failures, `Error::validation` marks bad config or invalid regex, `Error::unsupported` marks deferred adapters. No `unwrap`/`expect` outside module-level test cfgs and `#[allow(clippy::expect_used)]` static regexes.
- No `unsafe`, no globals: a pipeline is an owned value; there is no process-global registry (Python's `Runner.callback_framework` registration is the runner's concern, wired outside this crate).
- Logging through `tracing` only (targets `openjiuwen_security::<module>`); no `stdout`. Guardrail content is never logged; log names, stages, levels.
- Files ≤1000 lines; tests live in `tests/` at crate level plus `#[cfg(test)]` unit tests for pattern tables and parsers.
- Default features are empty and every test runs offline.

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
- `./src/backends/` → prompt-injection/moderation backends and their configs (`src/backends/CONTEXT.md`)
- `./src/guardrails/` → built-in guardrails: prompt injection, sensitive data, content rules, tool arguments (`src/guardrails/CONTEXT.md`)
- `./tests/` → crate-level behavior suites (models, pipeline, prompt injection, sensitive data, tool arguments, config, integration helpers)

## Test Strategy

- `tests/models.rs` — `RiskLevel` values/order/serde, `GuardrailResult::pass`/`block` shapes, `RiskAssessment` defaults, serde round-trips, redacting `Debug` for `GuardrailContext`, JSON field names.
- `tests/pipeline.rs` — priority ordering (descending, stable), stage filtering, aggregation of findings, severity/blocking matrix (LOW/MEDIUM/HIGH allow, CRITICAL block, configurable `block_level`), skip-after-block, fail-open vs fail-closed, `modified_data` merge, `enforce()` message parity.
- `tests/prompt_injection.rs` — six Python default patterns (case-insensitive, one finding per scan), `matched_pattern` details, config defaults, config validation errors (exact substrings), parser tables (BERT thresholds/logits/probabilities, Qwen dialects), `LlmPromptInjectionBackend` rule delegation, `ApiModelBackend` empty-content/transport-required behavior, rule-backend default level `HIGH`.
- `tests/sensitive_data.rs` — email/phone/card/id/secret detection, Luhn rejection, redaction output (`<REDACTED>`), `modified_data.sanitized_text`, no false positives on benign text, custom patterns/extra patterns, marker override.
- `tests/tool_arguments.rs` — denied/allowed tool names, shell-injection patterns in arguments, nested JSON argument traversal, sensitive-data redaction inside arguments, stage binding (tool input only).
- `tests/config.rs` — every default preserved (Python `PromptInjectionGuardrailConfig`/`RuleBasedBackendConfig` fields and values), `SecurityConfig::default()` scans nothing, `protected()` enables the default set, `pipeline()` validation errors, full serde round-trip.
- `tests/helpers.rs` — `scan_prompt`/`scan_model_output`/`scan_tool_input`/`scan_tool_output` context shapes (stages, metadata `tool_name`/`message_count`), `enforce()` on blocked outcomes, `RedactedOutcome` contains no original secrets, and an `ApiModelBackend` over a fake `ModerationTransport` feeding `BertBinaryParser`.
- Behavior-port tests cite the Python test they mirror in a comment (`// ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py`).
- Gates: `cargo fmt -p openjiuwen-security`, `cargo test -p openjiuwen-security`, `cargo clippy -p openjiuwen-security --all-targets -- -D warnings` all clean offline.

## Design Decisions

- **`Guardrail` takes a `GuardrailContext`, not Python's `*args/**kwargs`.** Python's `BaseGuardrail.extract_context(event, *args, **kwargs)` is untyped and positional-message bugs silently degrade scans; here the pipeline builds the context from typed helpers, and `ContextExtractor` plug-ins are unnecessary because every built-in consumes `text_view()`.
- **`ScanStage` carries Python's exact event strings** (`"_framework:llm_invoke_input"`, ...) so a runner callback adapter can bind by string without a translation table; `Other(String)` accepts event names this crate does not know.
- **Aggregation and the pipeline are an addition.** Python guardrails only bind to callback events and never produce a combined verdict, so nothing existed to port; the model here is one ordered run producing evaluations, findings, and a decision. Ordering: `priority` descending (Python semantics: higher first), registration order as the tiebreak.
- **`SecurityDecision` is a decision value, not an exception.** `run` is infallible by design (guardrail failures are data); `enforce()` converts a blocked outcome into `Error::Security`, and integration helpers return `GuardrailOutcome` so callers can inspect findings or redact instead of unwinding.
- **`block_level` defaults to `Critical`** (Python effective behavior) and is configurable; lowering it makes `LOW`/`MEDIUM`/`HIGH` findings blocking with Python's `GuardrailError` wording.
- **Message scanning is strictly stronger than Python**: Python's `PromptInjectionGuardrail` scans only the last message of `messages`; here `text_view()` joins every message's text and tool-call arguments with `\n`, so a malicious earlier turn is caught too. Single-message prompts behave identically.
- **`SensitivePattern` matching is regex + optional validator** (Luhn for card numbers), never a bare substring, and redaction is applied left-to-right on the original text so overlapping matches cannot corrupt each other.
- **Guardrails do not chain sanitized content between each other** (Python does not either): each sees the original context, and `GuardrailOutcome::modified_data()` merges their `modified_data` objects in evaluation order (later keys win).

## Known Gaps and Deferred Adapters

- `ApiModelBackend` performs no HTTP: it requires an injected `ModerationTransport`; without one, `analyze` returns `Error::Unsupported`. The Python `httpx` call to `{"text": ...}` is the transport's job.
- `LocalModelBackend` has no inference runtime: constructor and config are preserved (parity of fields/defaults), `analyze` returns `Error::Unsupported`. Python needs `torch`/`transformers`.
- `LlmPromptInjectionBackend` keeps Python's stub semantics (delegates to the rule backend, config fields unused) instead of inventing an LLM judge; a real judge is a `Guardrail` implementation over `openjiuwen_foundation::llm::Model`.
- No process-global registration helper: Python's `register_guardrail`/`unregister_guardrail` bind to `Runner.callback_framework`; the runner crate owns that wiring (see `docs/PARITY.md` security row).
- No vendor moderation adapters (OpenAI/阿里云/Azure content safety); `GuardrailBackend` + `backends::ModerationTransport` are the seam (`ApiModelBackend` plus a parser is the adapter shape).
- Python quirks deliberately not reproduced: `PromptInjectionGuardrail.with_rules()` (documented but never defined), double-registration bookkeeping bug, and `GuardrailContext(content_type=type(None))` fallback.

## Python Mapping

| Python (`openjiuwen/core/security/guardrail/`) | Rust |
|---|---|
| `enums.py` `RiskLevel`, `GuardrailContentType` | `level::RiskLevel`, `level::GuardrailContentType` |
| `models.py` `GuardrailResult`, `RiskAssessment` | `result::GuardrailResult`, `result::RiskAssessment` |
| `context.py` `GuardrailContext` | `context::GuardrailContext` |
| `context.py` `ModelOutputParser`/`BertBinaryParser`/`QwenGuardParser` | `parsers::{ModelOutputParser, BertBinaryParser, QwenGuardParser}` |
| `backends.py` `RuleBasedPromptInjectionBackend`, `LLMPromptInjectionBackend`, `APIModelBackend`, `LocalModelBackend` | `backends::{RuleBasedPromptInjectionBackend, LlmPromptInjectionBackend, ApiModelBackend, LocalModelBackend}` |
| `guardrail.py` `BaseGuardrail` | `guardrail::Guardrail` (trait) + `guardrail::BackendGuardrail` |
| `builtin.py` `PromptInjectionGuardrail`, `PromptInjectionGuardrailConfig` | `guardrails::prompt_injection::{PromptInjectionGuardrail, PromptInjectionGuardrailConfig}` |
| `__init__.py` `register_guardrail`/`unregister_guardrail` | out of scope (runner callback wiring) |
| — (no Python equivalent) | `pipeline`, `config`, `report`, `guardrails::{sensitive_data, content_rules, tool_arguments, moderation}`, `rules` |
