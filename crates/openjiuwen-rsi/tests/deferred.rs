//! Offline integration suite for the surfaces of `openjiuwen-rsi` that remain
//! deferred (`Error::Unsupported`).
//!
//! The implemented runtime, evaluator, and optimiser surfaces moved out of this
//! suite: the command/git/CI/Python/subprocess layers live in
//! `openjiuwen_rsi::runtime`, the case records and benchmark runtimes in
//! `openjiuwen_rsi::harness_rsi::evaluator`, member optimisation in
//! `openjiuwen_rsi::harness_rsi::member_optimizer`, and paper optimisation in
//! `openjiuwen_rsi::artifact_rsi::paper_opt`; `tests/crate_surface.rs` pins their
//! flat re-exports. What is left is the crate `CONTEXT.md` `## Deferred Scope`
//! residue, and each stub must fail loudly with `ErrorKind::Unsupported` instead
//! of returning a silent placeholder:
//!
//! - the documented constructor always fails, so a caller cannot build one of
//!   these types by accident and mistake it for a working implementation;
//! - the operation is an associated function, so the same error is reachable
//!   without a value;
//! - both carry `unsupported: <surface> is not implemented in the Rust port:
//!   <reason>` with a non-empty, descriptive reason;
//! - the one implemented type, `EvolutionPolicy`, is ordinary data and
//!   round-trips through `as_str`/`Display`/`FromStr`/serde.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Debug;
use std::str::FromStr;

use openjiuwen_rsi::{
    ErrorKind, EvolutionEngine, EvolutionPolicy, LlmJudgeEvaluator, Result, RuntimeExtensionLoader,
};
use serde_json::{json, Value};

/// `Error::Unsupported`'s `Display` prefix.
const PREFIX: &str = "unsupported: ";
/// Fixed fragment separating a surface name from its reason.
const MARKER: &str = " is not implemented in the Rust port: ";

/// Asserts `result` is the documented deferred error for `surface`, then returns
/// the reason that follows `<surface> is not implemented in the Rust port: `.
///
/// Enforces the message shape exactly: `unsupported: <surface><MARKER><reason>`,
/// with a non-empty, trimmed, multi-word reason naming why the surface is
/// deferred (a real explanation, not a placeholder).
#[track_caller]
fn assert_deferred<T: Debug>(result: Result<T>, surface: &str) -> String {
    let error = result.expect_err("deferred surface must fail");
    assert_eq!(
        error.kind(),
        ErrorKind::Unsupported,
        "`{surface}` must be reported as unsupported"
    );
    let message = error.to_string();
    let body = message
        .strip_prefix(PREFIX)
        .expect("unsupported errors display with the `unsupported: ` prefix");
    let expected = format!("{surface}{MARKER}");
    let reason = body
        .strip_prefix(&expected)
        .expect("deferred message names the surface and the port marker");
    assert_eq!(reason.trim(), reason, "`{surface}` reason must be trimmed");
    assert!(!reason.is_empty(), "`{surface}` reason must not be empty");
    assert!(
        reason.split_whitespace().count() >= 3,
        "`{surface}` reason must explain the gap, got `{reason}`"
    );
    reason.to_owned()
}

/// Asserts the constructor and the operation of one deferred surface both fail
/// with the same documented `Error::Unsupported` message.
#[track_caller]
fn assert_surface_pair<T: Debug, U: Debug>(new: Result<T>, operation: Result<U>, surface: &str) {
    let constructor_reason = assert_deferred(new, surface);
    let operation_reason = assert_deferred(operation, surface);
    assert_eq!(
        constructor_reason, operation_reason,
        "`new` and the operation must report the same reason for `{surface}`"
    );
}

/// CONTEXT.md `## Deferred Scope` row: agentdescent itself (`selection`,
/// `aggregator`, `ledger`, `staleness`, `filetree`, `evolution`).
#[test]
fn evolution_engine_is_deferred() {
    assert_surface_pair(
        EvolutionEngine::new(),
        EvolutionEngine::evolve(json!({"population": [{"candidate": "c0"}]})),
        "agentdescent evolution engine",
    );
}

/// CONTEXT.md `## Deferred Scope` row:
/// `auto_harness.infra.runtime_extension_loader` — synthetic `sys.modules`
/// hot-loading has no Rust analogue.
#[test]
fn runtime_extension_loader_is_deferred() {
    assert_surface_pair(
        RuntimeExtensionLoader::new(),
        RuntimeExtensionLoader::load(json!({"module": "openjiuwen.runtime_ext"})),
        "runtime extension loader",
    );
}

/// CONTEXT.md `## Deferred Scope` row: `harness_rsi.evaluator.judger` —
/// judge-model plumbing and prompt construction need the harness layer. The
/// benchmark runtimes are implemented and are covered by
/// `tests/evaluator_benchmarks.rs`.
#[test]
fn llm_judge_evaluator_is_deferred() {
    assert_surface_pair(
        LlmJudgeEvaluator::new(),
        LlmJudgeEvaluator::evaluate(json!({"rubric": "passes the tests", "answer": "ok"})),
        "LLM judge evaluator",
    );
}

/// `EvolutionPolicy` is the one implemented deferred-family type: ordinary data
/// that names its wire form identically through `as_str`, `Display`, `FromStr`,
/// and serde (`snake_case`), and defaults to `Full`.
#[test]
fn evolution_policy_round_trips_as_snake_case_data() {
    let policies = [
        (EvolutionPolicy::Full, "full"),
        (EvolutionPolicy::Guarded, "guarded"),
        (EvolutionPolicy::Reflective, "reflective"),
    ];
    for (policy, name) in policies {
        assert_eq!(policy.as_str(), name);
        assert_eq!(policy.to_string(), name);
        assert_eq!(
            EvolutionPolicy::from_str(name).expect("documented name parses"),
            policy
        );
        let encoded = serde_json::to_value(policy).expect("policy serializes");
        assert_eq!(encoded, json!(name), "serde uses the snake_case wire name");
        let decoded: EvolutionPolicy =
            serde_json::from_value(encoded).expect("policy deserializes");
        assert_eq!(decoded, policy);
    }
    assert_eq!(EvolutionPolicy::default(), EvolutionPolicy::Full);
    assert_eq!(EvolutionPolicy::default().as_str(), "full");
}

/// An unknown policy name is rejected with `ErrorKind::Validation`, echoing the
/// offending name and listing the accepted values.
#[test]
fn evolution_policy_rejects_unknown_names() {
    for invalid in ["chaotic", "Full", "reflective "] {
        let error = EvolutionPolicy::from_str(invalid).expect_err("unknown policy is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.to_string();
        assert!(
            message.contains("unknown evolution policy"),
            "message must identify the failure: `{message}`"
        );
        assert!(
            message.contains(invalid),
            "message must echo `{invalid}`: `{message}`"
        );
        for accepted in ["full", "guarded", "reflective"] {
            assert!(
                message.contains(accepted),
                "message must name accepted value `{accepted}`: `{message}`"
            );
        }
    }

    let empty = EvolutionPolicy::from_str("").expect_err("empty policy name is rejected");
    assert_eq!(empty.kind(), ErrorKind::Validation);
    assert!(empty.to_string().contains("full"));

    // serde's derived deserializer rejects the same unknown wire name.
    assert!(serde_json::from_value::<EvolutionPolicy>(json!("chaotic")).is_err());
}

/// The reason attached to a deferred error is per-surface, so a caller can tell
/// the three remaining deferred families apart from the message alone.
#[test]
fn deferred_reasons_are_surface_specific() {
    let surfaces = [
        (
            assert_deferred(EvolutionEngine::new(), "agentdescent evolution engine"),
            "agentdescent evolution engine",
        ),
        (
            assert_deferred(RuntimeExtensionLoader::new(), "runtime extension loader"),
            "runtime extension loader",
        ),
        (
            assert_deferred(LlmJudgeEvaluator::new(), "LLM judge evaluator"),
            "LLM judge evaluator",
        ),
    ];

    let mut reasons = std::collections::BTreeSet::new();
    for (reason, surface) in &surfaces {
        assert!(
            reasons.insert(reason.clone()),
            "`{surface}` must not reuse another surface's reason"
        );
    }
    assert_eq!(reasons.len(), 3, "all three deferred surfaces are covered");

    let rejected: Result<Value> = RuntimeExtensionLoader::load(Value::Null);
    assert_deferred(rejected, "runtime extension loader");
}
