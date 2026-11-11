# tests — crate-level behavior suites

## Intent

Integration tests for `openjiuwen-security`, one file per concern, exercising the public API only (no `pub(crate)` reach-in).
Reference: the Python suites `tests/unit_tests/core/security/guardrail/{test_guardrail_models,test_guardrail_base,test_parsers_and_backends}.py` and `tests/system_tests/security/guardrail/test_guardrail.py`; each ported behavior cites the Python test in a comment.

## Constraints

- Offline and deterministic: no network, no model runtime, no wall-clock assertions.
- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` so `clippy --all-targets -- -D warnings` is clean.
- Fake backends/parsers/clients are small structs implementing the crate traits in-file; no shared test crate.

## Routing Table

- `./models.rs` → severity/model serde, `GuardrailResult`, redacting `Debug`
- `./pipeline.rs` → ordering, stage filtering, aggregation, severity/blocking matrix, failure policy
- `./prompt_injection.rs` → default patterns, config validation, parsers, backend adapters
- `./sensitive_data.rs` → PII detection, Luhn, redaction, `modified_data`
- `./tool_arguments.rs` → tool allow/deny lists, argument patterns, nested JSON redaction
- `./config.rs` → defaults, `SecurityConfig` construction/serde
- `./helpers.rs` → scan helpers, context shapes, `enforce()`, `RedactedOutcome`
