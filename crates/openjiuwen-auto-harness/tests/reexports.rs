//! Proves the facade re-exports the canonical `openjiuwen_rsi::auto_harness`
//! types instead of declaring look-alike copies: both paths must resolve to the
//! same [`TypeId`]. Values also have to flow between the two spellings.

use std::any::TypeId;
use std::time::Duration;

#[test]
fn auto_harness_config_is_the_canonical_type() {
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::AutoHarnessConfig>(),
        TypeId::of::<openjiuwen_rsi::auto_harness::AutoHarnessConfig>()
    );
    // ... and the deeply-nested canonical module path agrees too.
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::AutoHarnessConfig>(),
        TypeId::of::<openjiuwen_rsi::harness_rsi::auto_harness::AutoHarnessConfig>()
    );
}

#[test]
fn auto_harness_orchestrator_is_the_canonical_type() {
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::AutoHarnessOrchestrator>(),
        TypeId::of::<openjiuwen_rsi::auto_harness::AutoHarnessOrchestrator>()
    );
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::AutoHarnessOrchestrator>(),
        TypeId::of::<openjiuwen_rsi::harness_rsi::auto_harness::AutoHarnessOrchestrator>()
    );
}

#[test]
fn secondary_surface_is_the_canonical_type() {
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::AutoHarnessEvent>(),
        TypeId::of::<openjiuwen_rsi::auto_harness::AutoHarnessEvent>()
    );
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::Budget>(),
        TypeId::of::<openjiuwen_rsi::auto_harness::Budget>()
    );
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::SessionState>(),
        TypeId::of::<openjiuwen_rsi::auto_harness::SessionState>()
    );
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::StopReason>(),
        TypeId::of::<openjiuwen_rsi::auto_harness::StopReason>()
    );
}

#[test]
fn module_reexports_match_the_canonical_modules() {
    // `KvCheckpointStore` is also reachable through the `checkpoint` submodule;
    // the facade must expose both spellings and they must be the same type.
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::checkpoint::KvCheckpointStore>(),
        TypeId::of::<openjiuwen_rsi::auto_harness::checkpoint::KvCheckpointStore>()
    );
    assert_eq!(
        TypeId::of::<openjiuwen_auto_harness::config::PipelinePreference>(),
        TypeId::of::<openjiuwen_rsi::auto_harness::config::PipelinePreference>()
    );
}

#[test]
fn values_flow_between_the_facade_and_the_canonical_crate() {
    // A limits value built through the facade drives the canonical budget ...
    let limits = openjiuwen_auto_harness::BudgetLimits {
        wall_clock: Duration::from_secs(60),
        cost_limit_usd: 1.5,
        task_timeout: Duration::from_secs(5),
    };
    let budget = openjiuwen_auto_harness::Budget::new(limits);
    assert!(!budget.should_stop());

    // ... and a config built through the facade feeds the canonical stop
    // conditions, whose decision keeps the facade's spelling.
    let config = openjiuwen_auto_harness::AutoHarnessConfig::default();
    let conditions = openjiuwen_rsi::auto_harness::StopConditions::from_config(&config);
    let decision: openjiuwen_auto_harness::StopDecision = conditions.evaluate(&[], 0, &budget);
    assert!(!decision.stop);
    assert_eq!(decision.reason, None);
}

#[test]
fn crate_name_is_stable() {
    assert_eq!(
        openjiuwen_auto_harness::crate_name(),
        "openjiuwen-auto-harness"
    );
}
