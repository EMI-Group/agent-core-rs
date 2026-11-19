//! Capacity-leak regression suite: an unwinding executor must not consume a
//! concurrency slot.
//!
//! Without the driver's RAII guard and the panic containment around
//! `TaskExecutor::execute`, a panicking run stays in the in-flight map forever:
//! the concurrency gate keeps counting the stale entry and the completion
//! sentinel stops firing. This suite proves the slot is reusable and that the
//! task records a reason.
use super::*;
use serde_json::json;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn a_panicking_executor_never_leaks_a_concurrency_slot() {
    let gated = ControllerConfig {
        max_concurrent_tasks: 2,
        ..config()
    };
    let panicking = shared(TestExecutor::new(TASK_TYPE).with_outcome(Outcome::Panic("boom")));
    let healthy = shared(TestExecutor::new(OTHER_TASK_TYPE));
    let fixture = Fixture::new(
        gated,
        vec![
            Arc::clone(&panicking) as Arc<dyn TaskExecutor>,
            Arc::clone(&healthy) as Arc<dyn TaskExecutor>,
        ],
    );
    fixture.add("p1", TASK_TYPE);
    fixture.add("p2", TASK_TYPE);
    assert!(fixture.scheduler.start());

    wait_for_status(&fixture, "p1", TaskStatus::Failed).await;
    wait_for_status(&fixture, "p2", TaskStatus::Failed).await;
    for task_id in ["p1", "p2"] {
        let task = fixture.manager.task(task_id).expect("task is stored");
        assert_eq!(
            task.error_message.as_deref(),
            Some("task executor panicked: boom"),
            "{task_id} carries the panic reason"
        );
    }
    assert_eq!(
        panicking.state().executions.load(Ordering::SeqCst),
        2,
        "both runs were admitted before either panicked"
    );
    assert_eq!(
        fixture.scheduler.in_flight().len(),
        0,
        "both unwound runs are untracked"
    );

    // The freed capacity admits a healthy task, which completes normally.
    fixture.add("t3", OTHER_TASK_TYPE);
    wait_for_status(&fixture, "t3", TaskStatus::Completed).await;
    fixture.scheduler.wait_until_idle().await;
    fixture.scheduler.stop().await;

    assert_eq!(fixture.scheduler.in_flight().len(), 0);
    let signals = fixture.signals();
    assert_eq!(
        signals
            .iter()
            .filter(|item| item.payload["type"] == json!("all_tasks_processed"))
            .count(),
        2,
        "one sentinel per idle stretch (the panic stretch, then the healthy run): {signals:?}"
    );
}
