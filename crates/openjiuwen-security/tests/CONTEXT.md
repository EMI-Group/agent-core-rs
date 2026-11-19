# tests — crate-level behavior suites

## Intent

Integration tests for `openjiuwen-security`, one file per concern, exercising the public API only (no `pub(crate)` reach-in).
Eight test binaries plus this file: `adapters`, `config`, `helpers`, `models`, `pipeline`, `prompt_injection`, `sensitive_data`, `tool_arguments`; wire-level backend tests stay in the module `#[cfg(test)]` suites.
Reference: the Python suites `tests/unit_tests/core/security/guardrail/{test_guardrail_models,test_guardrail_base,test_parsers_and_backends}.py` and `tests/system_tests/security/guardrail/test_guardrail.py`; each ported behavior cites the Python test in a comment.
The ported contract is the two-tier policy — only `CRITICAL` aborts the caller, every lower severity is recorded and execution continues — extended by this crate's aggregation and by its failure handling.
Failures are fail-closed by default: a guardrail that returns an error, and a payload the checked projection cannot read, both become a blocking `CRITICAL` finding whose reason is carried in the message and in `details.error`.
Under the explicit `FailurePolicy::FailOpen` the same failure is only recorded in `GuardrailOutcome::errors` (and logged), the decision staying unchanged.
The suites assert both halves of that split: blocking under the default policy, recorded-only under the opt-in.

## Constraints

- Offline and deterministic: no network, no model runtime, no wall-clock assertions; HTTP behavior is driven against loopback `wiremock` servers.
- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` so `clippy --all-targets -- -D warnings` is clean.
- Fake backends/parsers/transports/clients/guardrails are small structs implementing the crate traits in-file; no shared test crate.
- Failure-path tests always state the policy they exercise: blocking cases build the pipeline with `GuardrailPipeline::new()` (default `FailClosed`) and continuation cases opt into `SecurityPolicy { failure_policy: FailurePolicy::FailOpen, .. }`.
- Unscannable-payload coverage goes through the public entry points (`GuardrailPipeline::run`, the scan helpers) rather than calling `GuardrailContext::text_view_checked` directly.

## Routing Table

- `./pipeline.rs` → ordering (priority descending, registration tiebreak), stage filtering, aggregation of findings, the severity/blocking matrix, `stop_on_block`, fail-closed handling of guardrail errors (`guardrail_error`, exact message/details/`enforce()` shape) and of unscannable payloads (`unscannable_content`, `blocked_by == Some("context")`, multimodal-only message arrays), the opt-in fail-open path, scan-helper context shapes, and redacted reporting
- `./config.rs` → detector defaults ported from Python, `SecurityPolicy`/`ScanTargets`/`FailurePolicy` defaults (fail-closed), `SecurityConfig`'s `enabled = false` off switch, the `allow_all` opt-in that turns a config scanning nothing (no target, or no detector) from `ErrorKind::Config` into an empty pipeline, `protected()` registration order and policy, and JSON round-trips including `allow_all`
- `./helpers.rs` → scan-helper context shapes and metadata, `enforce()`/`violation()`/`modified_data()` behavior, redaction safety of `RedactedOutcome`/`Debug`, all-message scanning, blocking of unscannable payloads through `run` and `scan_tool_input` (with the failure reason kept and the payload dropped from the redacted view), and the empty pipeline that still allows an unscannable context
- `./models.rs` → severity/model serde, `GuardrailResult` pass/block shapes and JSON field names, `RiskAssessment`, outcome serde, and redacting `Debug`/`Display`
- `./prompt_injection.rs` → the six default patterns, config defaults/validation/serde, the `BERT`/`Qwen` parser tables, rule-based and API/local backend behavior, and stage binding
- `./sensitive_data.rs` → PII detection, Luhn validation, redaction output, `modified_data`, and a 2_000-match scan that pins the linear claim selection
- `./tool_arguments.rs` → tool allow/deny lists, shell-injection and argument patterns, nested JSON traversal/redaction, and stage binding
- `./adapters.rs` → the OpenAI, Azure Content Safety, and Aliyun Green adapters plus the generic `ApiModelBackend` + `HttpModerationTransport` path, driven through `GuardrailPipeline` + `BackendGuardrail` against loopback `wiremock` servers: benign, blocking, sub-threshold, transport-failure (both failure policies), stage-binding, and no-content-leak cases, plus the fail-open regressions (`non-string`/image-only/`null` payloads are `unscannable content` failures through every adapter, never a `Safe` verdict, with no wire request, and the vendors agree with the four local detectors on that error)
