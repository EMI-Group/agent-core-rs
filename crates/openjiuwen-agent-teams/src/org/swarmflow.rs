//! Deterministic plan → bounded fan-out → fan-in lane runner.
//!
//! A `SwarmflowPlan` is an ordered list of `LaneSpec`s plus the run policy: how many lanes may run
//! at once (`max_concurrency`), whether an exhausted lane stops the rest of the plan
//! (`continue_on_failure`), and the retry/backoff rules each lane carries. `SwarmflowRunner`
//! launches every lane as its own future under that bound, retries a failing lane up to its
//! `max_attempts`, and folds the results into one `SwarmflowOutcome` in plan order.
//!
//! # How a lane maps onto the team runtime
//!
//! One lane is one unit of delegated work. In the team runtime that ordinarily means one member
//! turn: the integration layer wraps `TeamAgent::invoke` (leader-side delegation) or a
//! `MemberRuntime::send` round-trip in a `SwarmLane`, puts the member prompt into
//! `LaneSpec::payload`, and hands the plan to `SwarmflowRunner::run`. The runner itself makes
//! **no LLM call, spawns no member and touches no store** — it is pure orchestration over the
//! `SwarmLane` it is given, so the same plan runs identically over scripted fakes in tests. The
//! `TeamAgent`/`MemberRuntime` → `SwarmLane` adapter is a separate wiring step and is not part of
//! this module.
//!
//! # Semantics
//!
//! * **Validation** — `SwarmflowPlan::validate` rejects a blank plan id, an empty lane list,
//!   duplicate or blank lane ids, `max_concurrency == 0` and `max_attempts == 0` with
//!   `Error::validation`. `SwarmflowRunner::run` validates before launching anything, so an empty
//!   plan is an error rather than a silent success. An unknown lane id passed to
//!   `SwarmflowOutcome::lane` simply yields `None`.
//! * **Bounded fan-out** — at most `max_concurrency` lanes execute at a time, and every lane runs
//!   as its own future, so a slow lane never blocks a fast one. `peak_concurrency` is *measured*:
//!   an in-flight counter wraps every lane attempt and records the maximum observed, it is never
//!   derived from the plan.
//! * **Retry** — a lane returning `Err` is attempted again until it has used `max_attempts`
//!   attempts in total. The wait before attempt *n+1* is `backoff_ms << (n-1)` saturated at
//!   `max_backoff_ms`; a lane with `max_attempts == 1` never retries. Retries are logged at
//!   `debug` through `tracing` and nowhere else.
//! * **Failure aggregation / fan-in** — `LaneOutcome::error` holds the last error's `to_string()`.
//!   With `continue_on_failure == false` the first lane that exhausts its attempts stops new lanes
//!   from launching: lanes already running finish and keep their real status, lanes never started
//!   are `LaneStatus::Skipped`. With `continue_on_failure == true` every lane is attempted.
//! * **Order** — `SwarmflowOutcome::lanes` is always in plan order regardless of completion order;
//!   `completed`/`failed`/`skipped` and `total_attempts` are folded counts.
//!
//! Everything here is in-process and deterministic: no network, no wall-clock ordering beyond the
//! retry sleeps, no process-global state, and no panic on caller input. Python parity source:
//! `openjiuwen/agent_teams/workflow/runner.py` and `swarmflow_infra.py`; this module deliberately
//! covers only the lane engine, not the surrounding workflow scripting surface.
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

/// Boxed body of a [`LaneClosure`] lane.
type LaneBody = Box<dyn Fn(LaneSpec, u32) -> BoxFuture<'static, Result<Value>> + Send + Sync>;

/// One unit of delegated work inside a [`SwarmflowPlan`].
///
/// The lane carries its own policy (`max_attempts`, `backoff_ms`) so a plan can mix cheap lanes
/// that should fail fast with expensive ones worth retrying. `payload` is opaque to the runner: it
/// is cloned and handed to the [`SwarmLane`] verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LaneSpec {
    /// Caller-chosen lane id, unique within the plan.
    pub lane_id: String,
    /// Human-readable title used by prompts and logs; free-form.
    pub title: String,
    /// Opaque payload delivered to the lane on every attempt.
    pub payload: Value,
    /// Total attempt budget for this lane; `1` means "never retry".
    pub max_attempts: u32,
    /// Base retry delay in milliseconds (doubled on every further attempt).
    pub backoff_ms: u64,
    /// Free-form metadata; the runner never reads it.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for LaneSpec {
    fn default() -> Self {
        Self {
            lane_id: String::new(),
            title: String::new(),
            payload: Value::Null,
            max_attempts: 1,
            backoff_ms: 0,
            metadata: BTreeMap::new(),
        }
    }
}

impl LaneSpec {
    /// Creates a lane with the default policy (`max_attempts = 1`, no backoff).
    #[must_use]
    pub fn new(lane_id: impl Into<String>, payload: Value) -> Self {
        Self {
            lane_id: lane_id.into(),
            payload,
            ..Self::default()
        }
    }

    /// Sets the human-readable title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Sets the total attempt budget (validated by [`SwarmflowPlan::validate`]).
    #[must_use]
    pub fn with_max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts;
        self
    }

    /// Sets the base retry delay in milliseconds.
    #[must_use]
    pub fn with_backoff_ms(mut self, ms: u64) -> Self {
        self.backoff_ms = ms;
        self
    }
}

/// A complete fan-out/fan-in request: ordered lanes plus the run policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SwarmflowPlan {
    /// Caller-chosen plan id echoed by every [`SwarmflowOutcome`].
    pub plan_id: String,
    /// Lanes in plan order; this order is what the outcome preserves.
    pub lanes: Vec<LaneSpec>,
    /// Maximum number of lanes allowed to execute at the same time.
    pub max_concurrency: usize,
    /// When `false`, the first lane that exhausts its attempts stops new lanes from launching.
    pub continue_on_failure: bool,
    /// Upper bound of any single retry delay, in milliseconds.
    pub max_backoff_ms: u64,
    /// Free-form metadata; the runner never reads it.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for SwarmflowPlan {
    fn default() -> Self {
        Self {
            plan_id: String::new(),
            lanes: Vec::new(),
            max_concurrency: 1,
            continue_on_failure: true,
            max_backoff_ms: 1_000,
            metadata: BTreeMap::new(),
        }
    }
}

impl SwarmflowPlan {
    /// Creates an empty plan with the default policy (`max_concurrency = 1`,
    /// `continue_on_failure = true`, `max_backoff_ms = 1000`).
    ///
    /// The plan is invalid until at least one lane is added; see [`Self::validate`].
    #[must_use]
    pub fn new(plan_id: impl Into<String>) -> Self {
        Self {
            plan_id: plan_id.into(),
            ..Self::default()
        }
    }

    /// Appends a lane, keeping plan order.
    #[must_use]
    pub fn with_lane(mut self, lane: LaneSpec) -> Self {
        self.lanes.push(lane);
        self
    }

    /// Sets the maximum number of lanes that may execute at the same time.
    #[must_use]
    pub fn with_max_concurrency(mut self, limit: usize) -> Self {
        self.max_concurrency = limit;
        self
    }

    /// Sets whether a lane failure lets the rest of the plan run.
    #[must_use]
    pub fn with_continue_on_failure(mut self, yes: bool) -> Self {
        self.continue_on_failure = yes;
        self
    }

    /// Sets the cap applied to every retry delay, in milliseconds.
    #[must_use]
    pub fn with_max_backoff_ms(mut self, ms: u64) -> Self {
        self.max_backoff_ms = ms;
        self
    }

    /// Returns the lane with `lane_id`, if the plan has one.
    #[must_use]
    pub fn lane(&self, lane_id: &str) -> Option<&LaneSpec> {
        self.lanes.iter().find(|lane| lane.lane_id == lane_id)
    }

    /// Checks the plan is runnable.
    ///
    /// A plan is valid when its `plan_id` is non-blank, it has at least one lane, every lane id is
    /// non-blank and unique, `max_concurrency` is at least `1`, and every lane's `max_attempts` is
    /// at least `1`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] naming the first problem found: a blank `plan_id`, an empty
    /// lane list, `max_concurrency == 0`, a blank or duplicate lane id, or `max_attempts == 0`.
    pub fn validate(&self) -> Result<()> {
        if self.plan_id.trim().is_empty() {
            return Err(Error::validation(
                "swarmflow plan_id must not be empty".to_owned(),
            ));
        }
        if self.lanes.is_empty() {
            return Err(Error::validation(format!(
                "swarmflow plan '{}' has no lanes",
                self.plan_id
            )));
        }
        if self.max_concurrency == 0 {
            return Err(Error::validation(format!(
                "swarmflow plan '{}' max_concurrency must be at least 1 (got 0)",
                self.plan_id
            )));
        }
        let mut seen = std::collections::BTreeSet::new();
        for lane in &self.lanes {
            if lane.lane_id.trim().is_empty() {
                return Err(Error::validation(format!(
                    "swarmflow plan '{}' has a lane with an empty lane_id",
                    self.plan_id
                )));
            }
            if !seen.insert(lane.lane_id.as_str()) {
                return Err(Error::validation(format!(
                    "swarmflow plan '{}' declares duplicate lane id '{}'",
                    self.plan_id, lane.lane_id
                )));
            }
            if lane.max_attempts == 0 {
                return Err(Error::validation(format!(
                    "swarmflow plan '{}' lane '{}' max_attempts must be at least 1 (got 0)",
                    self.plan_id, lane.lane_id
                )));
            }
        }
        Ok(())
    }
}

/// Terminal status of one lane measured by the runner.
///
/// Serializes to the `snake_case` wire strings `completed`, `failed` and `skipped`. The default is
/// `Skipped`, the status of a lane that never launched, so a partially populated outcome never
/// claims success by omission.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneStatus {
    /// The lane returned `Ok` within its attempt budget.
    Completed,
    /// The lane used all its attempts and every one returned `Err`.
    Failed,
    /// The lane never launched because an earlier lane failed a fail-fast plan.
    #[default]
    Skipped,
}

/// The runner's verdict for one lane.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LaneOutcome {
    /// Id of the lane this outcome belongs to.
    pub lane_id: String,
    /// Terminal status.
    pub status: LaneStatus,
    /// Number of attempts the runner actually performed (`0` when skipped).
    pub attempts: u32,
    /// Value returned by the last successful attempt; `null` when the lane did not complete.
    pub output: Value,
    /// `to_string()` of the last error when the lane failed; `None` otherwise.
    pub error: Option<String>,
}

/// The folded result of one plan run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SwarmflowOutcome {
    /// Id of the plan that produced this outcome.
    pub plan_id: String,
    /// One outcome per plan lane, always in plan order.
    pub lanes: Vec<LaneOutcome>,
    /// Number of lanes that completed.
    pub completed: usize,
    /// Number of lanes that failed.
    pub failed: usize,
    /// Number of lanes that were skipped.
    pub skipped: usize,
    /// Sum of the per-lane attempt counts.
    pub total_attempts: u32,
    /// Highest number of lane attempts observed running at the same time.
    pub peak_concurrency: usize,
}

impl SwarmflowOutcome {
    /// Returns `true` when every lane completed (nothing failed and nothing was skipped).
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.failed == 0 && self.skipped == 0
    }

    /// Returns the failed lane outcomes in plan order.
    #[must_use]
    pub fn failures(&self) -> Vec<&LaneOutcome> {
        self.lanes
            .iter()
            .filter(|lane| lane.status == LaneStatus::Failed)
            .collect()
    }

    /// Returns the outcome of `lane_id`, if the plan had such a lane.
    #[must_use]
    pub fn lane(&self, lane_id: &str) -> Option<&LaneOutcome> {
        self.lanes.iter().find(|lane| lane.lane_id == lane_id)
    }
}

/// The execution backend of a swarmflow lane.
///
/// Implementors do the actual work of one delegated unit — in the team runtime an adapter wraps
/// `TeamAgent::invoke` or a `MemberRuntime` turn — while the runner owns dispatch, retry and
/// aggregation. Implementations must be cheap to call repeatedly: the runner may invoke `run` up
/// to `LaneSpec::max_attempts` times for the same lane.
#[async_trait::async_trait]
pub trait SwarmLane: Send + Sync {
    /// Name of the backend, used for tracing only.
    fn name(&self) -> &str;

    /// Executes one attempt of `lane`; `attempt` starts at `1`.
    ///
    /// # Errors
    ///
    /// Any error is treated as a retryable attempt failure until the lane's `max_attempts` budget
    /// is exhausted, after which it becomes the lane's [`LaneOutcome::error`].
    async fn run(&self, lane: &LaneSpec, attempt: u32) -> Result<Value>;
}

/// A [`SwarmLane`] built from an async closure, the seam tests and simple adapters use.
pub struct LaneClosure {
    name: String,
    run: LaneBody,
}

impl LaneClosure {
    /// Builds a lane named `name` that runs `run` for every attempt.
    ///
    /// `run` receives a clone of the plan's [`LaneSpec`] and the 1-based attempt number.
    #[must_use]
    pub fn new<F, Fut>(name: impl Into<String>, run: F) -> Self
    where
        F: Fn(LaneSpec, u32) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        Self {
            name: name.into(),
            run: Box::new(move |lane, attempt| Box::pin(run(lane, attempt))),
        }
    }
}

#[async_trait::async_trait]
impl SwarmLane for LaneClosure {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(&self, lane: &LaneSpec, attempt: u32) -> Result<Value> {
        (self.run)(lane.clone(), attempt).await
    }
}

/// The plan → fan-out → fan-in runner. Stateless; cheap to create per call.
#[derive(Debug, Clone, Copy, Default)]
pub struct SwarmflowRunner;

impl SwarmflowRunner {
    /// Creates a runner.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Runs `plan` over `lane`, bounded by the plan's concurrency limit.
    ///
    /// Lane failures do not turn into an `Err`: they are aggregated into the returned
    /// [`SwarmflowOutcome`] (see [`SwarmflowOutcome::is_success`] and
    /// [`SwarmflowOutcome::failures`]). Only an invalid plan is an error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when [`SwarmflowPlan::validate`] rejects `plan`. Nothing is
    /// launched in that case.
    #[allow(clippy::unused_self)] // pinned signature: the runner is stateless but callable on an instance
    pub async fn run<L>(&self, plan: &SwarmflowPlan, lane: Arc<L>) -> Result<SwarmflowOutcome>
    where
        L: SwarmLane + ?Sized + 'static,
    {
        plan.validate()?;
        let abort = Arc::new(AtomicBool::new(false));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let limit = plan.max_concurrency.max(1);
        let continue_on_failure = plan.continue_on_failure;
        let max_backoff_ms = plan.max_backoff_ms;
        let results: Vec<(usize, LaneOutcome)> =
            futures::stream::iter(plan.lanes.iter().cloned().enumerate())
                .map(|(index, spec)| {
                    let lane = Arc::clone(&lane);
                    let abort = Arc::clone(&abort);
                    let in_flight = Arc::clone(&in_flight);
                    let peak = Arc::clone(&peak);
                    async move {
                        let outcome = if !continue_on_failure && abort.load(Ordering::SeqCst) {
                            skipped(&spec)
                        } else {
                            run_lane(
                                &spec,
                                lane.as_ref(),
                                &in_flight,
                                &peak,
                                continue_on_failure,
                                max_backoff_ms,
                                &abort,
                            )
                            .await
                        };
                        (index, outcome)
                    }
                })
                .buffer_unordered(limit)
                .collect()
                .await;

        let mut ordered: Vec<Option<LaneOutcome>> = vec![None; plan.lanes.len()];
        for (index, outcome) in results {
            ordered[index] = Some(outcome);
        }
        let lanes: Vec<LaneOutcome> = ordered.into_iter().flatten().collect();
        let completed = lanes
            .iter()
            .filter(|outcome| outcome.status == LaneStatus::Completed)
            .count();
        let failed = lanes
            .iter()
            .filter(|outcome| outcome.status == LaneStatus::Failed)
            .count();
        let skipped = lanes
            .iter()
            .filter(|outcome| outcome.status == LaneStatus::Skipped)
            .count();
        let total_attempts = lanes
            .iter()
            .map(|outcome| outcome.attempts)
            .fold(0_u32, u32::saturating_add);
        Ok(SwarmflowOutcome {
            plan_id: plan.plan_id.clone(),
            lanes,
            completed,
            failed,
            skipped,
            total_attempts,
            peak_concurrency: peak.load(Ordering::SeqCst),
        })
    }
}

/// Convenience wrapper around [`SwarmflowRunner::new`] and [`SwarmflowRunner::run`].
///
/// # Errors
///
/// Same as [`SwarmflowRunner::run`]: an invalid plan is [`Error::validation`].
pub async fn run_swarmflow<L>(plan: &SwarmflowPlan, lane: Arc<L>) -> Result<SwarmflowOutcome>
where
    L: SwarmLane + ?Sized + 'static,
{
    SwarmflowRunner::new().run(plan, lane).await
}

/// Outcome of a lane that never launched.
fn skipped(spec: &LaneSpec) -> LaneOutcome {
    LaneOutcome {
        lane_id: spec.lane_id.clone(),
        status: LaneStatus::Skipped,
        ..LaneOutcome::default()
    }
}

/// Runs one lane, retrying within its budget.
async fn run_lane<L>(
    spec: &LaneSpec,
    lane: &L,
    in_flight: &AtomicUsize,
    peak: &AtomicUsize,
    continue_on_failure: bool,
    max_backoff_ms: u64,
    abort: &AtomicBool,
) -> LaneOutcome
where
    L: SwarmLane + ?Sized,
{
    let mut attempts: u32 = 0;
    loop {
        attempts = attempts.saturating_add(1);
        let result = {
            let _slot = InFlightGuard::enter(in_flight, peak);
            lane.run(spec, attempts).await
        };
        match result {
            Ok(output) => {
                return LaneOutcome {
                    lane_id: spec.lane_id.clone(),
                    status: LaneStatus::Completed,
                    attempts,
                    output,
                    error: None,
                };
            }
            Err(error) => {
                if attempts >= spec.max_attempts {
                    if !continue_on_failure {
                        abort.store(true, Ordering::SeqCst);
                    }
                    return LaneOutcome {
                        lane_id: spec.lane_id.clone(),
                        status: LaneStatus::Failed,
                        attempts,
                        output: Value::Null,
                        error: Some(error.to_string()),
                    };
                }
                let delay = backoff_delay(spec.backoff_ms, attempts, max_backoff_ms);
                tracing::debug!(
                    lane_id = %spec.lane_id,
                    lane = lane.name(),
                    attempt = attempts,
                    delay_ms = delay,
                    error = %error,
                    "swarmflow lane attempt failed; retrying"
                );
                if delay > 0 {
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
            }
        }
    }
}

/// Delay before attempt `attempt + 1`: `backoff_ms << (attempt - 1)`, saturated and capped.
fn backoff_delay(backoff_ms: u64, attempt: u32, max_backoff_ms: u64) -> u64 {
    backoff_ms
        .checked_shl(attempt - 1)
        .unwrap_or(u64::MAX)
        .min(max_backoff_ms)
}

/// In-flight counter for one lane attempt; decrements on drop.
struct InFlightGuard<'a> {
    in_flight: &'a AtomicUsize,
}

impl<'a> InFlightGuard<'a> {
    /// Counts one more attempt as running and records the observed maximum.
    fn enter(in_flight: &'a AtomicUsize, peak: &AtomicUsize) -> Self {
        let current = in_flight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
        peak.fetch_max(current, Ordering::SeqCst);
        Self { in_flight }
    }
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "swarmflow_tests.rs"]
mod swarmflow_tests;
