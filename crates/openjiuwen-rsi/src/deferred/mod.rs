//! API-shaped stubs for the Python surfaces deferred to a future milestone.
//!
//! Python's `openjiuwen.rsi` package spans ~97k lines across 313 files; part of
//! it needs an ML runtime, a vendor harness SDK, subprocess/git execution, or
//! distributed evolution, none of which the offline, deterministic Rust port
//! ships. Rather than dropping those areas silently, every one of them keeps its
//! public shape here so a caller can write against the eventual API and detect
//! the gap programmatically by matching
//! [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported).
//!
//! Every entry point returns [`Error::Unsupported`]
//! built by `not_implemented`, whose message reads
//! `<surface> is not implemented in the Rust port: <reason>` and names the
//! crate `CONTEXT.md` `## Deferred Scope` table. The documented constructors
//! always fail, so a caller cannot build one of these types by accident and
//! mistake it for a working implementation; the operations are associated
//! functions (`EvolutionEngine::evolve`, `GitBackend::apply`, …) so the error is
//! reachable without a value.
//!
//! The one implemented type is [`EvolutionPolicy`], which is ordinary data: it
//! parses, formats, and round-trips through serde like any other public type.

use openjiuwen_common::error::Error;

pub mod agentdescent;
pub mod evaluators;
pub mod member_optimizer;
pub mod paper_opt;
pub mod runtime;

pub use agentdescent::{EvolutionEngine, EvolutionPolicy};
pub use evaluators::{LlmJudgeEvaluator, SweBenchEvaluator, TerminalBenchEvaluator};
pub use member_optimizer::MemberOptimizer;
pub use paper_opt::PaperOptimizer;
pub use runtime::{CiGate, GitBackend, RuntimeExtensionLoader, SubprocessEvaluator};

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
    fn runtime_surfaces_are_deferred() {
        assert!(SubprocessEvaluator::new().is_err());
        assert_deferred(
            SubprocessEvaluator::evaluate(json!({"program": "print(1)"})),
            "subprocess program evaluator",
        );
        assert!(RuntimeExtensionLoader::new().is_err());
        assert_deferred(
            RuntimeExtensionLoader::load(Value::Null),
            "runtime extension loader",
        );
        assert!(CiGate::new().is_err());
        assert_deferred(CiGate::run(Value::Null), "CI gate runner");
        assert!(GitBackend::new().is_err());
        assert_deferred(GitBackend::apply(Value::Null), "git backend");
    }

    #[test]
    fn evaluator_surfaces_are_deferred() {
        assert!(SweBenchEvaluator::new().is_err());
        assert_deferred(
            SweBenchEvaluator::evaluate(Value::Null),
            "SWE-bench evaluator",
        );
        assert!(TerminalBenchEvaluator::new().is_err());
        assert_deferred(
            TerminalBenchEvaluator::evaluate(Value::Null),
            "terminal-bench evaluator",
        );
        assert!(LlmJudgeEvaluator::new().is_err());
        assert_deferred(
            LlmJudgeEvaluator::evaluate(Value::Null),
            "LLM judge evaluator",
        );
    }

    #[test]
    fn optimiser_surfaces_are_deferred() {
        assert!(PaperOptimizer::new().is_err());
        assert_deferred(PaperOptimizer::optimize(Value::Null), "paper optimiser");
        assert!(MemberOptimizer::new().is_err());
        assert_deferred(MemberOptimizer::optimize(Value::Null), "member optimiser");
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
