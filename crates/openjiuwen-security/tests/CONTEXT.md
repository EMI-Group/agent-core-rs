# tests — crate-level behavior suites

## Intent

Integration tests for `openjiuwen-security`, one file per concern, exercising the public API only (no `pub(crate)` reach-in).
Eight test binaries (`adapters`, `models`, `pipeline`, `prompt_injection`, `sensitive_data`, `tool_arguments`, `config`, `helpers`) plus this file; wire-level backend tests stay in the module `#[cfg(test)]` suites.
Reference: the Python suites `tests/unit_tests/core/security/guardrail/{test_guardrail_models,test_guardrail_base,test_parsers_and_backends}.py` and `tests/system_tests/security/guardrail/test_guardrail.py`; each ported behavior cites the Python test in a comment.

## Constraints

- Offline and deterministic: no network, no model runtime, no wall-clock assertions; HTTP behavior is driven against loopback `wiremock` servers.
- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` so `clippy --all-targets -- -D warnings` is clean.
- Fake backends/parsers/transports/clients are small structs implementing the crate traits in-file; no shared test crate.

## Routing Table

- `./adapters.rs` → vendor moderation adapters (`OpenAI`, Azure Content Safety, Aliyun Green) and `ApiModelBackend` + `HttpModerationTransport` through `GuardrailPipeline` + `BackendGuardrail`: 21 tests covering benign, blocking, sub-threshold, transport-failure (fail-open/fail-closed), stage-binding, and no-content-leak cases, plus the generic transport's recorded wire request
- `./models.rs` → severity/model serde, `GuardrailResult`, redacting `Debug`
- `./pipeline.rs` → ordering, stage filtering, aggregation, severity/blocking matrix, failure policy
- `./prompt_injection.rs` → the six default patterns, blocking policy, config defaults/validation/serde, `BERT`/`Qwen` parser tables, `LlmPromptInjectionBackend` rule delegation, `ApiModelBackend`/`LocalModelBackend` behavior through in-file fakes, stage binding
- `./sensitive_data.rs` → PII detection, Luhn, redaction, `modified_data`
- `./tool_arguments.rs` → tool allow/deny lists, argument patterns, nested JSON redaction
- `./config.rs` → defaults, `SecurityConfig` construction/serde
- `./helpers.rs` → scan helpers, context shapes, `enforce()`, `RedactedOutcome`
