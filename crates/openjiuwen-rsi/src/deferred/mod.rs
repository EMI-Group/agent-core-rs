//! API-shaped stubs for the Python surfaces that stay out of scope.
//!
//! Python's `openjiuwen.rsi` package spans ~97k lines across 313 files. Most of
//! it is ported: subprocess/git/CI execution and program evaluation live in
//! [`crate::runtime`], run usage accounting in [`crate::ledger`], case records,
//! judge scoring, the benchmark runtimes and metrics in
//! [`crate::harness_rsi::evaluator`], member optimisation in
//! [`crate::harness_rsi::member_optimizer`], the promotion loops in
//! [`crate::harness_rsi::single_harness`], and paper optimisation in
//! [`crate::artifact_rsi::paper_opt`].
//!
//! Three surfaces still need something the offline, deterministic Rust port
//! cannot ship. Rather than dropping them silently, each keeps its public shape
//! here so a caller can write against the eventual API and detect the gap
//! programmatically by matching
//! [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported):
//!
//! - [`agentdescent::EvolutionEngine`] — the `agentdescent==0.4.6`
//!   evolve/ledger/aggregator internals carry distributed, deep-RL training
//!   machinery with no maintained Rust equivalent.
//! - [`evaluators::LlmJudgeEvaluator`] — a guarded judge model and its prompt
//!   construction belong to the harness layer, not to this crate.
//! - [`runtime::RuntimeExtensionLoader`] — Python's `sys.modules` hot-loading
//!   has no Rust analogue.
//!
//! Every entry point returns [`Error::Unsupported`] built by
//! `not_implemented`, whose message reads
//! `<surface> is not implemented in the Rust port: <reason>`. The documented
//! constructors always fail, so a caller cannot build one of these types by
//! accident and mistake it for a working implementation; the operations are
//! associated functions (`EvolutionEngine::evolve`, …) so the error is reachable
//! without a value.
//!
//! The one implemented type is [`EvolutionPolicy`], which is ordinary data: it
//! parses, formats, and round-trips through serde like any other public type.

use openjiuwen_common::error::Error;

pub mod agentdescent;
pub mod evaluators;
pub mod runtime;

pub use agentdescent::{EvolutionEngine, EvolutionPolicy};
pub use evaluators::LlmJudgeEvaluator;
pub use runtime::RuntimeExtensionLoader;

/// Builds the explicit [`Error::Unsupported`] every deferred entry point
/// returns.
///
/// The message always starts with `<surface> is not implemented in the Rust
/// port: ` and continues with `reason`, so callers and tests can identify the
/// missing feature without matching on prose.
pub(crate) fn not_implemented(surface: &str, reason: &str) -> Error {
    Error::unsupported(format!(
        "{surface} is not implemented in the Rust port: {reason}"
    ))
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use openjiuwen_common::error::ErrorKind;
    use serde_json::{json, Value};

    use super::*;

    const MARKER: &str = " is not implemented in the Rust port: ";

    #[track_caller]
    fn assert_deferred(result: openjiuwen_common::error::Result<Value>, surface: &str) {
        let error = result.expect_err("deferred surface must fail");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        let expected = format!("{surface}{MARKER}");
        assert!(
            message.contains(&expected),
            "expected `{expected}` in `{message}`"
        );
        let reason = message
            .split_once(&expected)
            .map(|(_, reason)| reason)
            .expect("marker present");
        assert!(!reason.is_empty(), "deferred reason must not be empty");
    }

    #[test]
    fn agentdescent_engine_is_deferred() {
        let error = EvolutionEngine::new().expect_err("engine constructor must fail");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error
            .to_string()
            .starts_with("unsupported: agentdescent evolution engine is not implemented"));
        assert_deferred(
            EvolutionEngine::evolve(Value::Null),
            "agentdescent evolution engine",
        );
    }

    #[test]
    fn runtime_extension_loader_is_deferred() {
        assert!(RuntimeExtensionLoader::new().is_err());
        assert_deferred(
            RuntimeExtensionLoader::load(Value::Null),
            "runtime extension loader",
        );
    }

    #[test]
    fn llm_judge_evaluator_is_deferred() {
        assert!(LlmJudgeEvaluator::new().is_err());
        assert_deferred(
            LlmJudgeEvaluator::evaluate(Value::Null),
            "LLM judge evaluator",
        );
    }

    #[test]
    fn evolution_policy_round_trips_through_from_str() {
        for policy in [
            EvolutionPolicy::Full,
            EvolutionPolicy::Guarded,
            EvolutionPolicy::Reflective,
        ] {
            let parsed = EvolutionPolicy::from_str(policy.as_str()).expect("policy parses");
            assert_eq!(parsed, policy);
            assert_eq!(policy.to_string(), policy.as_str());
        }
        assert_eq!(EvolutionPolicy::default(), EvolutionPolicy::Full);
    }

    #[test]
    fn evolution_policy_rejects_unknown_names() {
        let error = EvolutionPolicy::from_str("chaotic").expect_err("unknown policy is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("chaotic"));
    }

    #[test]
    fn evolution_policy_round_trips_through_json() {
        let encoded = serde_json::to_value(EvolutionPolicy::Reflective).expect("policy serializes");
        assert_eq!(encoded, json!("reflective"));
        let decoded: EvolutionPolicy =
            serde_json::from_value(encoded).expect("policy deserializes");
        assert_eq!(decoded, EvolutionPolicy::Reflective);
    }

    #[test]
    fn deferred_message_helper_is_well_formed() {
        let error = not_implemented("a surface", "a reason");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.to_string(),
            "unsupported: a surface is not implemented in the Rust port: a reason"
        );
    }
}
