//! Tests for the swarmflow lane runner: bounded fan-out, retry, fail-fast fan-in, plan ordering,
//! payload delivery, validation and serde.
//!
//! Every test is deterministic and offline: lanes are closures or tiny instrumented fakes, and the
//! only waiting is the handful of milliseconds a fake sleeps to force overlapping attempts.
use super::{
    run_swarmflow, LaneClosure, LaneOutcome, LaneSpec, LaneStatus, SwarmLane, SwarmflowOutcome,
    SwarmflowPlan, SwarmflowRunner,
};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
// -----------------------------------------------------------------------------------------------
// Fakes and fixtures
// -----------------------------------------------------------------------------------------------
/// A lane that measures its own in-flight attempts, so a test can compare the runner's measured
/// `peak_concurrency` against an independent observation.
struct InstrumentedLane {
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    delay_ms: u64,
}
impl InstrumentedLane {
    fn new(delay_ms: u64) -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            delay_ms,
        }
    }
}
#[async_trait]
impl SwarmLane for InstrumentedLane {
    #[allow(clippy::unnecessary_literal_bound)] // the trait fixes the return type
    fn name(&self) -> &str {
        "instrumented"
    }
    async fn run(&self, lane: &LaneSpec, _attempt: u32) -> Result<Value> {
        let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(current, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(json!({ "lane": lane.lane_id }))
    }
}
/// Builds a plan with one default-policy lane per id, in the given order.
fn plan_with(ids: &[&str], max_concurrency: usize) -> SwarmflowPlan {
    let mut plan = SwarmflowPlan::new("plan-1").with_max_concurrency(max_concurrency);
    for id in ids {
        plan = plan.with_lane(LaneSpec::new(*id, json!({ "lane": id })));
    }
    plan
}
// -----------------------------------------------------------------------------------------------
// Bounded fan-out
// -----------------------------------------------------------------------------------------------
#[tokio::test]
async fn peak_concurrency_reaches_the_limit_without_exceeding_it() {
    let plan = plan_with(&["a", "b", "c", "d", "e", "f"], 3);
    let lane = Arc::new(InstrumentedLane::new(2));
    let outcome = SwarmflowRunner::new()
        .run(&plan, Arc::clone(&lane))
        .await
        .unwrap();
    assert_eq!(outcome.completed, 6);
    assert_eq!(outcome.total_attempts, 6);
    assert_eq!(lane.peak.load(Ordering::SeqCst), 3);
    assert_eq!(outcome.peak_concurrency, 3);
    assert!(outcome.peak_concurrency <= plan.max_concurrency);
    assert_eq!(lane.in_flight.load(Ordering::SeqCst), 0);
}
// -----------------------------------------------------------------------------------------------
// Retry and failure aggregation
// -----------------------------------------------------------------------------------------------
#[tokio::test]
async fn lane_retries_then_succeeds() {
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&attempts);
    let lane = Arc::new(LaneClosure::new(
        "flaky",
        move |spec: LaneSpec, attempt: u32| {
            let recorded = Arc::clone(&recorded);
            async move {
                recorded.lock().unwrap().push(attempt);
                if attempt == 1 {
                    Err(Error::tool(format!("transient {}", spec.lane_id)))
                } else {
                    Ok(json!({ "attempt": attempt, "payload": spec.payload }))
                }
            }
        },
    ));
    let plan = SwarmflowPlan::new("p").with_lane(
        LaneSpec::new("only", json!({ "task": 7 }))
            .with_max_attempts(3)
            .with_backoff_ms(0),
    );
    let outcome = run_swarmflow(&plan, lane).await.unwrap();
    let only = outcome.lane("only").unwrap();
    assert_eq!(only.status, LaneStatus::Completed);
    assert_eq!(only.attempts, 2);
    assert!(only.error.is_none());
    assert_eq!(only.output["attempt"], json!(2));
    assert_eq!(*attempts.lock().unwrap(), vec![1_u32, 2_u32]);
    assert_eq!(
        (outcome.completed, outcome.failed, outcome.skipped),
        (1, 0, 0)
    );
    assert_eq!(outcome.total_attempts, 2);
    assert!(outcome.is_success());
}
#[tokio::test]
async fn lane_fails_after_exhausting_attempts() {
    let lane = Arc::new(LaneClosure::new(
        "broken",
        |spec: LaneSpec, attempt: u32| async move {
            Err::<Value, Error>(Error::tool(format!("boom {} at {attempt}", spec.lane_id)))
        },
    ));
    let plan = SwarmflowPlan::new("p").with_lane(
        LaneSpec::new("only", Value::Null)
            .with_max_attempts(3)
            .with_backoff_ms(1),
    );
    let outcome = SwarmflowRunner::new().run(&plan, lane).await.unwrap();
    let only = outcome.lane("only").unwrap();
    assert_eq!(only.status, LaneStatus::Failed);
    assert_eq!(only.attempts, 3);
    assert_eq!(only.error.as_deref(), Some("tool: boom only at 3"));
    assert!(only.output.is_null());
    assert_eq!(outcome.failed, 1);
    assert_eq!(outcome.total_attempts, 3);
    assert_eq!(outcome.failures().len(), 1);
    assert!(!outcome.is_success());
}
#[tokio::test]
async fn continue_on_failure_runs_every_lane_and_counts_attempts() {
    let lane = Arc::new(LaneClosure::new(
        "mixed",
        |spec: LaneSpec, _attempt: u32| {
            let id = spec.lane_id;
            async move {
                if id == "bad" {
                    Err::<Value, Error>(Error::tool("bad lane"))
                } else {
                    Ok(json!({ "lane": id }))
                }
            }
        },
    ));
    let plan = plan_with(&["ok-1", "bad", "ok-2"], 2);
    let outcome = run_swarmflow(&plan, lane).await.unwrap();
    assert_eq!(
        (outcome.completed, outcome.failed, outcome.skipped),
        (2, 1, 0)
    );
    assert_eq!(outcome.total_attempts, 3);
    assert_eq!(outcome.failures().len(), 1);
    assert_eq!(outcome.failures()[0].lane_id, "bad");
    assert_eq!(outcome.lane("bad").unwrap().attempts, 1);
    assert_eq!(outcome.lane("ok-1").unwrap().output["lane"], json!("ok-1"));
    assert!(!outcome.is_success());
}
#[tokio::test]
async fn fail_fast_skips_unlaunched_lanes_and_keeps_running_ones() {
    let lane = Arc::new(LaneClosure::new(
        "mixed",
        |spec: LaneSpec, _attempt: u32| {
            let id = spec.lane_id;
            async move {
                if id == "boom" {
                    Err::<Value, Error>(Error::tool("boom"))
                } else if id == "slow-ok" {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                    Ok(json!({ "lane": id }))
                } else {
                    Ok(json!({ "lane": id }))
                }
            }
        },
    ));
    let plan = SwarmflowPlan::new("p")
        .with_max_concurrency(2)
        .with_continue_on_failure(false)
        .with_lane(LaneSpec::new("slow-ok", Value::Null))
        .with_lane(LaneSpec::new("boom", Value::Null))
        .with_lane(LaneSpec::new("never", Value::Null));
    let outcome = run_swarmflow(&plan, lane).await.unwrap();
    let ids: Vec<&str> = outcome
        .lanes
        .iter()
        .map(|lane| lane.lane_id.as_str())
        .collect();
    assert_eq!(ids, vec!["slow-ok", "boom", "never"]);
    assert_eq!(
        outcome.lane("slow-ok").unwrap().status,
        LaneStatus::Completed
    );
    assert_eq!(outcome.lane("slow-ok").unwrap().attempts, 1);
    assert_eq!(outcome.lane("boom").unwrap().status, LaneStatus::Failed);
    let never = outcome.lane("never").unwrap();
    assert_eq!(never.status, LaneStatus::Skipped);
    assert_eq!(never.attempts, 0);
    assert!(never.error.is_none());
    assert_eq!(
        (outcome.completed, outcome.failed, outcome.skipped),
        (1, 1, 1)
    );
    assert!(!outcome.is_success());
}
// -----------------------------------------------------------------------------------------------
// Ordering and payload delivery
// -----------------------------------------------------------------------------------------------
#[tokio::test]
async fn outcome_keeps_plan_order_when_lanes_finish_out_of_order() {
    let completion_order = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&completion_order);
    let lane = Arc::new(LaneClosure::new(
        "ordered",
        move |spec: LaneSpec, _attempt: u32| {
            let recorded = Arc::clone(&recorded);
            let id = spec.lane_id;
            async move {
                if id == "slow" {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                recorded.lock().unwrap().push(id.clone());
                Ok(json!({ "lane": id }))
            }
        },
    ));
    let plan = plan_with(&["slow", "fast"], 2);
    let outcome = run_swarmflow(&plan, lane).await.unwrap();
    assert_eq!(
        *completion_order.lock().unwrap(),
        vec!["fast".to_owned(), "slow".to_owned()]
    );
    assert_eq!(outcome.lanes[0].lane_id, "slow");
    assert_eq!(outcome.lanes[1].lane_id, "fast");
    assert_eq!(outcome.completed, 2);
}
#[tokio::test]
async fn lanes_receive_the_payload_and_one_based_attempt() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&seen);
    let lane = Arc::new(LaneClosure::new(
        "echo",
        move |spec: LaneSpec, attempt: u32| {
            let recorded = Arc::clone(&recorded);
            async move {
                recorded.lock().unwrap().push((
                    spec.lane_id.clone(),
                    spec.payload.clone(),
                    attempt,
                ));
                Ok(spec.payload)
            }
        },
    ));
    let plan = SwarmflowPlan::new("p")
        .with_lane(LaneSpec::new("alpha", json!({ "task": 7 })).with_title("Alpha lane"));
    let outcome = run_swarmflow(&plan, lane).await.unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        vec![("alpha".to_owned(), json!({ "task": 7 }), 1_u32)]
    );
    assert_eq!(outcome.lane("alpha").unwrap().output, json!({ "task": 7 }));
}
// -----------------------------------------------------------------------------------------------
// Validation and lookups
// -----------------------------------------------------------------------------------------------
#[tokio::test]
async fn invalid_plans_are_rejected_before_launching() {
    let lane = Arc::new(LaneClosure::new(
        "never",
        |_spec: LaneSpec, _attempt: u32| async { Ok::<Value, Error>(json!({})) },
    ));
    let cases: Vec<(SwarmflowPlan, &str)> = vec![
        (SwarmflowPlan::new(""), "plan_id"),
        (SwarmflowPlan::new("p"), "no lanes"),
        (
            SwarmflowPlan::new("p")
                .with_lane(LaneSpec::new("a", Value::Null))
                .with_lane(LaneSpec::new("a", Value::Null)),
            "duplicate lane id",
        ),
        (
            SwarmflowPlan::new("p")
                .with_max_concurrency(0)
                .with_lane(LaneSpec::new("a", Value::Null)),
            "max_concurrency",
        ),
        (
            SwarmflowPlan::new("p").with_lane(LaneSpec::new("a", Value::Null).with_max_attempts(0)),
            "max_attempts",
        ),
        (
            SwarmflowPlan::new("p").with_lane(LaneSpec::new("", Value::Null)),
            "empty lane_id",
        ),
    ];
    for (plan, needle) in cases {
        let error = SwarmflowRunner::new()
            .run(&plan, Arc::clone(&lane))
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::Validation,
            "case {needle}: {error}"
        );
        assert!(error.to_string().contains(needle), "case {needle}: {error}");
    }
}
#[test]
fn lookups_return_none_for_unknown_lanes() {
    let plan = plan_with(&["a"], 1);
    assert!(plan.lane("a").is_some());
    assert!(plan.lane("missing").is_none());
    let outcome = SwarmflowOutcome::default();
    assert!(outcome.lane("missing").is_none());
    assert_eq!(outcome.failures().len(), 0);
    assert!(outcome.is_success());
}
// -----------------------------------------------------------------------------------------------
// Serde
// -----------------------------------------------------------------------------------------------
#[test]
fn plan_and_lane_serde_round_trip_with_defaults() {
    let value = json!({
        "plan_id": "p",
        "max_concurrency": 4,
        "lanes": [
            { "lane_id": "a", "payload": { "task": 1 }, "unknown_field": true }
        ],
        "unknown_plan_field": "ignored"
    });
    let plan: SwarmflowPlan = serde_json::from_value(value).unwrap();
    assert_eq!(plan.max_concurrency, 4);
    assert!(plan.continue_on_failure);
    assert_eq!(plan.max_backoff_ms, 1_000);
    assert_eq!(plan.lanes.len(), 1);
    let lane = &plan.lanes[0];
    assert_eq!(lane.lane_id, "a");
    assert_eq!(lane.payload, json!({ "task": 1 }));
    assert_eq!(lane.title, "");
    assert_eq!(lane.max_attempts, 1);
    assert_eq!(lane.backoff_ms, 0);
    assert_eq!(lane.metadata.len(), 0);
    let round_trip: SwarmflowPlan =
        serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    assert_eq!(round_trip, plan);
}
#[test]
fn outcome_serde_uses_wire_status_strings() {
    let outcome = SwarmflowOutcome {
        plan_id: "p".to_owned(),
        lanes: vec![
            LaneOutcome {
                lane_id: "a".to_owned(),
                status: LaneStatus::Completed,
                attempts: 1,
                output: json!({ "ok": true }),
                error: None,
            },
            LaneOutcome {
                lane_id: "b".to_owned(),
                status: LaneStatus::Failed,
                attempts: 2,
                output: Value::Null,
                error: Some("tool: boom".to_owned()),
            },
            LaneOutcome {
                lane_id: "c".to_owned(),
                status: LaneStatus::Skipped,
                attempts: 0,
                output: Value::Null,
                error: None,
            },
        ],
        completed: 1,
        failed: 1,
        skipped: 1,
        total_attempts: 3,
        peak_concurrency: 1,
    };
    let value = serde_json::to_value(&outcome).unwrap();
    assert_eq!(value["lanes"][0]["status"], json!("completed"));
    assert_eq!(value["lanes"][1]["status"], json!("failed"));
    assert_eq!(value["lanes"][2]["status"], json!("skipped"));
    let round_trip: SwarmflowOutcome = serde_json::from_value(value).unwrap();
    assert_eq!(round_trip, outcome);
    let partial: LaneOutcome = serde_json::from_value(json!({ "lane_id": "x" })).unwrap();
    assert_eq!(partial.status, LaneStatus::Skipped);
    assert_eq!(partial.attempts, 0);
    assert!(partial.output.is_null());
    assert!(partial.error.is_none());
    let tolerant: SwarmflowOutcome =
        serde_json::from_value(json!({ "plan_id": "p", "unknown": 1 })).unwrap();
    assert_eq!(tolerant.lanes.len(), 0);
    assert_eq!(tolerant.total_attempts, 0);
}
// -----------------------------------------------------------------------------------------------
// LaneClosure as a trait object
// -----------------------------------------------------------------------------------------------
#[tokio::test]
async fn lane_closure_runs_through_a_dyn_swarm_lane() {
    let lane: Arc<dyn SwarmLane> = Arc::new(LaneClosure::new(
        "dyn",
        |spec: LaneSpec, attempt: u32| async move {
            Ok::<Value, Error>(json!({ "lane": spec.lane_id, "attempt": attempt }))
        },
    ));
    assert_eq!(lane.name(), "dyn");
    let plan = SwarmflowPlan::new("p").with_lane(LaneSpec::new("a", Value::Null));
    let outcome = run_swarmflow(&plan, Arc::clone(&lane)).await.unwrap();
    let only = outcome.lane("a").unwrap();
    assert_eq!(only.status, LaneStatus::Completed);
    assert_eq!(only.output["attempt"], json!(1));
}
