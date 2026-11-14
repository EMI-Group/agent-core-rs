use super::*;

#[test]
fn new_uses_documented_defaults() {
    let runner = Runner::new();
    assert_eq!(runner.runner_id(), "global");
    assert!(!runner.is_started());
    assert!(runner.guardrails().is_none());
    assert!(runner.checkpointer().is_none());
    assert!(runner.resource_mgr().is_empty());
    assert!(runner.callbacks().is_empty());
    // The observation seams and the pool exist from construction, open and
    // empty, so runs do not require `start`.
    assert_eq!(runner.monitor().list_active(), Vec::new());
    assert!(runner.hitl().is_empty());
    let Some(hub_monitor) = runner.hitl().monitor() else {
        panic!("the hub mirrors its transitions into a monitor");
    };
    assert!(Arc::ptr_eq(&hub_monitor, &runner.monitor()));
    assert_eq!(runner.list_active_teams(), Vec::new());
    assert_eq!(runner.team_pool().capacity(), 8);
    assert!(!runner.team_pool().is_closed());
    assert!(runner.agent_team_status("crew", "s-1").is_none());
    let debug = format!("{runner:?}");
    assert!(debug.starts_with("Runner {"));
    assert!(debug.contains("runner_id: \"global\""));
    assert!(debug.contains("started: false"));
    assert!(debug.contains("guardrails: false"));
    assert!(debug.contains("pooled_teams: 0"));
    assert!(debug.contains("active_runs: 0"));
}

#[test]
fn runner_id_is_fixed_at_construction() {
    let runner = Runner::with_config(RunnerConfig {
        runner_id: "custom".to_owned(),
        ..RunnerConfig::default()
    });
    assert_eq!(runner.runner_id(), "custom");
    runner.set_config(RunnerConfig {
        runner_id: "other".to_owned(),
        ..RunnerConfig::default()
    });
    assert_eq!(runner.runner_id(), "custom");
    assert_eq!(runner.config().runner_id, "custom");
}
