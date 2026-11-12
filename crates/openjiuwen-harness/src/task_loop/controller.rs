//! The outer task-loop controller: round driving, budget checks, queues,
//! pause/resume/abort and post-round state persistence.
//!
//! Ports `_run_task_loop` in `openjiuwen/harness/deep_agent.py` plus the
//! interaction supervisor's round bookkeeping
//! (`openjiuwen/harness/task_loop/task_loop_controller.py`).
//!
//! The controller deliberately owns no agent, model or session: one round of
//! work is executed through the object-safe [`RoundExecutor`] seam supplied by
//! `crate::agent`, so this module stays pure loop machinery.
//!
//! See `CONTEXT.md` for the full contract and the deviations from Python.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::Notify;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use openjiuwen_common::error::{Error, Result};

use crate::schema::{
    ActiveInteractionRound, DeepAgentState, InteractionPhase, RoundOutcome, RoundWorkItem,
    StopEvaluationContext,
};
use crate::task_loop::coordinator::{LoopCoordinator, DEFAULT_MAX_OUTER_ROUNDS};
use crate::task_loop::handler::TaskLoopEventHandler;
use crate::task_loop::queues::{LoopQueues, QueuedInput};

/// Stop reason recorded when the loop stops at a pause request.
const PAUSED_STOP_REASON: &str = "paused";

/// Stop reason recorded when the loop stops because of an abort.
const ABORTED_STOP_REASON: &str = "Aborted";

/// Outcome error code of an aborted round.
const ABORTED_CODE: &str = "aborted";

/// Outcome error code produced when a task-loop hook fails.
const HANDLER_ERROR_CODE: &str = "handler_error";

/// Outcome error code produced when the executor fails.
const ROUND_EXECUTION_ERROR_CODE: &str = "round_execution_error";

/// Outcome error code returned when there is no active round to wait for.
const NO_ACTIVE_ROUND_CODE: &str = "no_active_round";

/// Outcome error message returned when there is no active round to wait for.
const NO_ACTIVE_ROUND_MESSAGE: &str = "no active round";

/// Context key stamped on continuation rounds (`resume_continuation`).
const RESUME_CONTINUATION_KEY: &str = "resume_continuation";

/// Identifier of one task-loop round (1-based, monotonic per controller).
pub type RoundId = u64;

/// Persistence callback invoked after every round.
///
/// The sink owns the session write (`SESSION_STATE_KEY`); the controller performs
/// no session IO itself and treats sink failures as non-fatal.
pub type StateSink = Arc<dyn Fn(&DeepAgentState) -> Result<()> + Send + Sync>;

/// Executes exactly one round of agent work for the loop controller.
///
/// Implemented by `crate::agent` (or a thin adapter) so the controller never
/// depends on a live agent, model or session.
#[async_trait]
pub trait RoundExecutor: Send + Sync {
    /// Runs one round of work and reports the outcome.
    ///
    /// # Errors
    ///
    /// An `Err` becomes `RoundOutcome::failed("round_execution_error", message)`;
    /// expected round failures should instead be reported through
    /// [`RoundOutcome`] so the loop can keep its Python-compatible error codes.
    async fn run_round(&self, work: RoundWorkItem) -> Result<RoundOutcome>;

    /// Python's per-round result dict (`result_type`, `output`, ...) fed to the
    /// stop-condition evaluators.
    ///
    /// The default keeps the no-progress guard inert.
    #[must_use]
    fn last_round_result(&self) -> Option<Value> {
        None
    }

    /// Cumulative token usage of the session.
    ///
    /// The controller converts it into a delta for
    /// [`LoopCoordinator::observe_round`].
    #[must_use]
    fn token_usage(&self) -> u64 {
        0
    }
}

/// Loop-level knobs of the outer task loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskLoopConfig {
    /// Per-round completion timeout; [`Duration::ZERO`] disables it.
    pub completion_timeout: Duration,
    /// Cap on outer rounds before the loop stops with `MaxOuterRounds`.
    pub max_outer_rounds: usize,
    /// Stamp `context["resume_continuation"] = true` on continuation rounds.
    pub resume_continuation: bool,
    /// Invoke the [`StateSink`] after every round.
    pub persist_state: bool,
}

impl Default for TaskLoopConfig {
    fn default() -> Self {
        Self {
            completion_timeout: Duration::from_secs(600),
            max_outer_rounds: DEFAULT_MAX_OUTER_ROUNDS,
            resume_continuation: false,
            persist_state: true,
        }
    }
}

impl TaskLoopConfig {
    /// Returns this config with the completion timeout replaced.
    #[must_use]
    pub const fn with_completion_timeout(mut self, completion_timeout: Duration) -> Self {
        self.completion_timeout = completion_timeout;
        self
    }

    /// Returns this config with the outer-round cap replaced.
    #[must_use]
    pub const fn with_max_outer_rounds(mut self, max_outer_rounds: usize) -> Self {
        self.max_outer_rounds = max_outer_rounds;
        self
    }

    /// Returns this config with the continuation stamping flag replaced.
    #[must_use]
    pub const fn with_resume_continuation(mut self, resume_continuation: bool) -> Self {
        self.resume_continuation = resume_continuation;
        self
    }

    /// Returns this config with the state-persistence flag replaced.
    #[must_use]
    pub const fn with_persist_state(mut self, persist_state: bool) -> Self {
        self.persist_state = persist_state;
        self
    }
}

/// Aggregate result of one [`TaskLoopController::run_until_idle`] call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoopReport {
    /// Number of rounds executed by this call.
    pub rounds: usize,
    /// Reason the loop stopped, when a stop condition or the caller decided it.
    pub stop_reason: Option<String>,
    /// Outcome of the last executed round.
    pub last_outcome: Option<RoundOutcome>,
    /// Whether the loop stopped because of an abort or cancellation.
    pub aborted: bool,
    /// Whether the loop stopped because it was paused.
    pub paused: bool,
}

impl LoopReport {
    /// Whether the last executed round failed.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.last_outcome
            .as_ref()
            .is_some_and(RoundOutcome::is_error)
    }
}

/// One submitted-but-not-yet-executed round.
#[derive(Debug)]
struct PendingRound {
    round_id: RoundId,
    task_id: Option<String>,
    work: RoundWorkItem,
}

/// Driver of the outer task loop.
///
/// The controller never owns an agent: one round is delegated to a
/// [`RoundExecutor`]. It owns the coordinator (budgets and stop conditions), the
/// shared [`LoopQueues`], the persisted [`DeepAgentState`] and the hook registry.
pub struct TaskLoopController {
    handler: Box<dyn TaskLoopEventHandler>,
    coordinator: LoopCoordinator,
    queues: LoopQueues,
    config: TaskLoopConfig,
    state: DeepAgentState,
    state_sink: Option<StateSink>,
    pending: VecDeque<PendingRound>,
    steering_dispatch: Arc<Mutex<VecDeque<QueuedInput>>>,
    next_round_id: RoundId,
    phase: Mutex<InteractionPhase>,
    cancel: CancellationToken,
    paused: AtomicBool,
    last_outcome: Arc<Mutex<Option<RoundOutcome>>>,
    completion: Arc<Notify>,
}

impl fmt::Debug for TaskLoopController {
    /// Prints the loop machinery; the boxed handler is summarised by its type.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskLoopController")
            .field("config", &self.config)
            .field("coordinator", &self.coordinator)
            .field("phase", &self.phase())
            .field("paused", &self.is_paused())
            .field("cancelled", &self.is_cancelled())
            .field("pending_rounds", &self.pending.len())
            .field("next_round_id", &self.next_round_id)
            .field("iteration", &self.coordinator.iteration())
            .field("has_state_sink", &self.state_sink.is_some())
            .finish_non_exhaustive()
    }
}

impl TaskLoopController {
    /// Builds a controller from its handler, coordinator, queues and config.
    ///
    /// `config.max_outer_rounds` is pushed into the coordinator so the
    /// `MaxOuterRounds` stop reason has a single budget authority.
    #[must_use]
    pub fn new(
        handler: Box<dyn TaskLoopEventHandler>,
        mut coordinator: LoopCoordinator,
        queues: LoopQueues,
        config: TaskLoopConfig,
    ) -> Self {
        coordinator.set_max_outer_rounds(config.max_outer_rounds);
        Self {
            handler,
            coordinator,
            queues,
            config,
            state: DeepAgentState::default(),
            state_sink: None,
            pending: VecDeque::new(),
            steering_dispatch: Arc::new(Mutex::new(VecDeque::new())),
            next_round_id: 1,
            phase: Mutex::new(InteractionPhase::Idle),
            cancel: CancellationToken::new(),
            paused: AtomicBool::new(false),
            last_outcome: Arc::new(Mutex::new(None)),
            completion: Arc::new(Notify::new()),
        }
    }

    /// Returns this controller with the persisted agent state replaced.
    #[must_use]
    pub fn with_state(mut self, state: DeepAgentState) -> Self {
        self.state = state;
        self
    }

    /// Returns this controller with a state sink attached.
    #[must_use]
    pub fn with_state_sink(mut self, sink: StateSink) -> Self {
        self.state_sink = Some(sink);
        self
    }

    /// Replaces the state sink.
    pub fn set_state_sink(&mut self, sink: StateSink) {
        self.state_sink = Some(sink);
    }

    /// Replaces the persisted agent state.
    pub fn set_state(&mut self, state: DeepAgentState) {
        self.state = state;
    }

    /// Snapshot of the persisted agent state.
    ///
    /// Updated after every round with the iteration, the stop-condition state and
    /// the still-queued follow-ups.
    #[must_use]
    pub fn state(&self) -> DeepAgentState {
        self.state.clone()
    }

    /// Invokes the state sink with the current state.
    ///
    /// # Errors
    ///
    /// Propagates the sink's own error; the loop itself only logs it.
    pub fn persist(&self) -> Result<()> {
        if let Some(sink) = self.state_sink.as_ref() {
            sink(&self.state)?;
        }
        Ok(())
    }

    /// Queues one user round and returns its round id.
    pub fn submit_round(
        &mut self,
        inputs: BTreeMap<String, Value>,
        is_follow_up: bool,
        task_id: Option<String>,
    ) -> RoundId {
        let work = RoundWorkItem::user(None, inputs, is_follow_up, true);
        self.submit_work(work, task_id)
    }

    /// Queues arbitrary work and returns its round id.
    pub fn submit_work(&mut self, work: RoundWorkItem, task_id: Option<String>) -> RoundId {
        let round_id = self.allocate_round_id();
        self.pending.push_back(PendingRound {
            round_id,
            task_id,
            work,
        });
        round_id
    }

    /// Number of submitted rounds that have not executed yet.
    #[must_use]
    pub fn pending_rounds(&self) -> usize {
        self.pending.len()
    }

    /// Executes one round under a fresh round id.
    pub async fn run_round(
        &mut self,
        executor: &dyn RoundExecutor,
        work: RoundWorkItem,
    ) -> RoundOutcome {
        let round_id = self.allocate_round_id();
        self.execute_round(executor, round_id, None, work).await
    }

    /// Executes rounds until no work is left or a stop condition fires.
    ///
    /// One follow-up is drained per round. After every round the coordinator is
    /// observed, `state.iteration` / `state.stop_condition_state` /
    /// `state.pending_follow_ups` are refreshed, the state sink is invoked (when
    /// `config.persist_state`) and any `next_work` decided by the round is
    /// queued. Evaluation happens only when work exists, so stateful evaluators
    /// are never evaluated twice for one round.
    pub async fn run_until_idle(&mut self, executor: &dyn RoundExecutor) -> LoopReport {
        self.coordinator
            .set_max_outer_rounds(self.config.max_outer_rounds);
        let started = Instant::now();
        let mut report = LoopReport::default();
        loop {
            if self.is_paused() {
                debug!("task-loop paused between rounds");
                report.paused = true;
                report.stop_reason = Some(PAUSED_STOP_REASON.to_string());
                break;
            }
            if self.cancel.is_cancelled() {
                self.coordinator.request_abort();
                report.aborted = true;
                report.stop_reason = Some(ABORTED_STOP_REASON.to_string());
                break;
            }
            if !self.has_work() {
                report.stop_reason = self.coordinator.stop_reason().map(str::to_string);
                report.aborted = self.coordinator.is_aborted();
                break;
            }
            let mut context = StopEvaluationContext::default();
            if !self.coordinator.should_continue(&mut context).await {
                report.stop_reason = self.coordinator.stop_reason().map(str::to_string);
                report.aborted = self.coordinator.is_aborted();
                break;
            }
            let Some(pending) = self.next_work_item() else {
                break;
            };
            let mut work = pending.work;
            if self.config.resume_continuation && report.rounds > 0 {
                work.context
                    .insert(RESUME_CONTINUATION_KEY.to_string(), Value::Bool(true));
            }
            let outcome = self
                .execute_round(executor, pending.round_id, pending.task_id, work)
                .await;
            report.rounds += 1;
            report.last_outcome = Some(outcome.clone());
            self.observe_round(executor, started);
            self.persist_round_state();
            if let Some(next) = outcome.next_work.clone() {
                self.enqueue_next_work(next);
            }
            if outcome.is_error() {
                report.stop_reason.clone_from(&outcome.error_code);
                report.aborted = outcome.error_code.as_deref() == Some(ABORTED_CODE);
                break;
            }
        }
        if report.aborted {
            self.set_phase(InteractionPhase::Terminated);
        }
        report
    }

    /// Waits for the running round to finish and returns its outcome.
    ///
    /// With no active round this returns the last recorded outcome, or
    /// `RoundOutcome::failed("no_active_round", "no active round")`. A wait that
    /// exceeds `timeout` cancels the loop and returns [`RoundOutcome::timeout`].
    pub async fn wait_round_completion(&self, timeout: Option<Duration>) -> RoundOutcome {
        // Drop a permit left behind by an earlier completed round so the wait
        // below targets the round that is running now.
        let _ = self.completion.notified().now_or_never();
        if self.phase() != InteractionPhase::Running {
            return self.last_outcome_snapshot().unwrap_or_else(no_active_round);
        }
        let notified = self.completion.notified();
        let resolved = if let Some(duration) = timeout {
            tokio::time::timeout(duration, notified).await.is_ok()
        } else {
            notified.await;
            true
        };
        if resolved {
            return self.last_outcome_snapshot().unwrap_or_else(no_active_round);
        }
        warn!("wait_round_completion timed out; cancelling the task loop");
        self.cancel.cancel();
        self.set_phase(InteractionPhase::Terminated);
        RoundOutcome::timeout()
    }

    /// Appends one input to the follow-up queue.
    pub fn push_follow_up(&self, input: impl Into<QueuedInput>) {
        self.queues.push_follow_up(input);
    }

    /// Appends one input to the steering data plane and logs a dispatch record.
    ///
    /// The executor drains the data plane before each model call; the dispatch
    /// record makes `handle_interaction` fire exactly once for this input.
    pub fn push_steer(&self, input: impl Into<QueuedInput>) {
        let input = input.into();
        debug!(task_id = input.task_id(), "task-loop steering input queued");
        self.steering_dispatch.lock().push_back(input.clone());
        self.queues.push_steer(input);
    }

    /// Removes and returns every queued follow-up input.
    #[must_use]
    pub fn drain_follow_up(&self) -> Vec<QueuedInput> {
        self.queues.drain_follow_up()
    }

    /// Whether a follow-up input is queued.
    #[must_use]
    pub fn has_follow_up(&self) -> bool {
        self.queues.has_follow_up()
    }

    /// The queues shared with the agent and the event handler.
    #[must_use]
    pub fn queues(&self) -> &LoopQueues {
        &self.queues
    }

    /// Pauses the loop between rounds.
    ///
    /// # Errors
    ///
    /// Returns `Error::validation` once the controller is terminated (after
    /// [`Self::cancel`] or a completed abort).
    pub fn pause(&self) -> Result<()> {
        if self.phase() == InteractionPhase::Terminated {
            return Err(Error::validation("task loop is terminated"));
        }
        self.paused.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Resumes a paused loop.
    ///
    /// # Errors
    ///
    /// Returns `Error::validation` once the controller is terminated (after
    /// [`Self::cancel`] or a completed abort).
    pub fn resume(&self) -> Result<()> {
        if self.phase() == InteractionPhase::Terminated {
            return Err(Error::validation("task loop is terminated"));
        }
        self.paused.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Whether the loop is paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Cancels the loop and terminates the controller.
    pub fn cancel(&self) {
        self.cancel.cancel();
        self.coordinator.request_abort();
        self.set_phase(InteractionPhase::Terminated);
    }

    /// Whether cancellation was requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Latches the coordinator abort flag.
    pub fn request_abort(&self) {
        self.coordinator.request_abort();
    }

    /// Whether an abort has been requested.
    #[must_use]
    pub fn is_aborted(&self) -> bool {
        self.coordinator.is_aborted()
    }

    /// The stop reason recorded by the last stop decision.
    #[must_use]
    pub fn stop_reason(&self) -> Option<&str> {
        self.coordinator.stop_reason()
    }

    /// Number of completed outer-loop rounds.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.coordinator.iteration()
    }

    /// The current interaction phase.
    #[must_use]
    pub fn phase(&self) -> InteractionPhase {
        *self.phase.lock()
    }

    /// Whether a round is currently executing.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.phase() == InteractionPhase::Running
    }

    /// The loop configuration.
    #[must_use]
    pub fn config(&self) -> &TaskLoopConfig {
        &self.config
    }

    /// Mutable access to the loop configuration.
    ///
    /// `max_outer_rounds` is re-applied to the coordinator at the start of the
    /// next [`Self::run_until_idle`] call.
    pub fn config_mut(&mut self) -> &mut TaskLoopConfig {
        &mut self.config
    }

    /// Allocates the next monotonic round id.
    fn allocate_round_id(&mut self) -> RoundId {
        let round_id = self.next_round_id;
        self.next_round_id = self.next_round_id.saturating_add(1);
        round_id
    }

    /// Sets the interaction phase (interior mutability keeps `&self` hooks usable).
    fn set_phase(&self, phase: InteractionPhase) {
        *self.phase.lock() = phase;
    }

    /// Whether a submitted round or a follow-up is waiting.
    fn has_work(&self) -> bool {
        !self.pending.is_empty() || self.queues.has_follow_up()
    }

    /// Pops the next round: a submission first, otherwise one drained follow-up.
    fn next_work_item(&mut self) -> Option<PendingRound> {
        if let Some(pending) = self.pending.pop_front() {
            return Some(pending);
        }
        let input = self.queues.pop_follow_up()?;
        let round_id = self.allocate_round_id();
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "query".to_string(),
            Value::String(input.content().to_string()),
        );
        let mut work = RoundWorkItem::user(None, inputs, true, true);
        for (key, value) in input.metadata() {
            work.context.insert(key.clone(), value.clone());
        }
        Some(PendingRound {
            round_id,
            task_id: input.task_id().map(str::to_string),
            work,
        })
    }

    /// Queues the agent-decided `next_work` of a finished round.
    fn enqueue_next_work(&mut self, next: RoundWorkItem) {
        if !next.is_follow_up {
            self.submit_work(next, None);
            return;
        }
        let content = match next.query() {
            Some(Value::String(text)) => text.clone(),
            Some(other) => other.to_string(),
            None => serde_json::to_string(&next.inputs).unwrap_or_default(),
        };
        let mut input = QueuedInput::new(content);
        if !next.context.is_empty() {
            input = input.with_metadata(next.context);
        }
        self.queues.push_follow_up(input);
    }

    /// Feeds the completed round into the coordinator counters.
    fn observe_round(&mut self, executor: &dyn RoundExecutor, started: Instant) {
        let iteration = self.coordinator.iteration().saturating_add(1);
        let delta = executor
            .token_usage()
            .saturating_sub(self.coordinator.token_usage());
        self.coordinator.observe_round(
            iteration,
            delta,
            started.elapsed().as_secs_f64(),
            executor.last_round_result(),
        );
    }

    /// Refreshes the persisted state and invokes the state sink.
    fn persist_round_state(&mut self) {
        self.state.iteration = self.coordinator.iteration();
        self.state.stop_condition_state = Some(self.coordinator.state().get_state());
        self.state.pending_follow_ups = self.queues.follow_up_contents();
        if !self.config.persist_state {
            return;
        }
        if let Err(err) = self.persist() {
            warn!(error = %err, "task-loop state sink failed");
        }
    }

    /// Snapshot of the last recorded round outcome.
    fn last_outcome_snapshot(&self) -> Option<RoundOutcome> {
        self.last_outcome.lock().clone()
    }

    /// Executes one round: hooks, executor, terminal hook, then persistence hooks.
    async fn execute_round(
        &mut self,
        executor: &dyn RoundExecutor,
        round_id: RoundId,
        task_id: Option<String>,
        work: RoundWorkItem,
    ) -> RoundOutcome {
        self.set_phase(InteractionPhase::Running);
        let active = ActiveInteractionRound {
            work: work.clone(),
            task_id,
        };
        let mut hook_error = self.run_pre_round_hooks(round_id, &active).await;
        if hook_error.is_none() {
            hook_error = self.dispatch_steering(round_id).await;
        }
        let mut outcome = if let Some(message) = hook_error {
            Self::handler_failure(message)
        } else if self.cancel.is_cancelled() || self.coordinator.is_aborted() {
            debug!(round_id, "task-loop round aborted before execution");
            RoundOutcome::abort()
        } else {
            self.run_executor(executor, work).await
        };
        if let Err(err) = self.run_terminal_hook(round_id, &active, &outcome).await {
            warn!(round_id, error = %err, "task-loop terminal hook failed");
            if !outcome.is_error() {
                outcome = Self::handler_failure(err.to_string());
            }
        }
        if let Err(err) = self.handler.after_round(round_id, &active, &outcome).await {
            warn!(round_id, error = %err, "task-loop after_round hook failed");
        }
        *self.last_outcome.lock() = Some(outcome.clone());
        self.set_phase(InteractionPhase::Idle);
        if self.cancel.is_cancelled() || self.coordinator.is_aborted() {
            self.set_phase(InteractionPhase::Terminated);
        }
        self.completion.notify_one();
        outcome
    }

    /// Runs `prepare_round` then `handle_input`, returning the first hook error.
    async fn run_pre_round_hooks(
        &self,
        round_id: RoundId,
        active: &ActiveInteractionRound,
    ) -> Option<String> {
        if let Err(err) = self.handler.prepare_round(round_id, active).await {
            warn!(round_id, error = %err, "task-loop prepare_round hook failed");
            return Some(err.to_string());
        }
        if let Err(err) = self.handler.handle_input(round_id, active).await {
            warn!(round_id, error = %err, "task-loop handle_input hook failed");
            return Some(err.to_string());
        }
        None
    }

    /// Dispatches the round's steering inputs to `handle_interaction` once each.
    async fn dispatch_steering(&self, round_id: RoundId) -> Option<String> {
        let inputs = {
            let mut dispatch = self.steering_dispatch.lock();
            std::mem::take(&mut *dispatch)
        };
        let mut first_error = None;
        for input in inputs {
            if let Err(err) = self.handler.handle_interaction(round_id, input).await {
                warn!(round_id, error = %err, "task-loop handle_interaction hook failed");
                if first_error.is_none() {
                    first_error = Some(err.to_string());
                }
            }
        }
        first_error
    }

    /// Runs the executor under the completion timeout.
    async fn run_executor(
        &self,
        executor: &dyn RoundExecutor,
        work: RoundWorkItem,
    ) -> RoundOutcome {
        let timeout = self.config.completion_timeout;
        if timeout.is_zero() {
            return match executor.run_round(work).await {
                Ok(outcome) => outcome,
                Err(err) => Self::round_execution_failure(&err),
            };
        }
        match tokio::time::timeout(timeout, executor.run_round(work)).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(err)) => Self::round_execution_failure(&err),
            Err(_elapsed) => {
                warn!(
                    timeout_secs = timeout.as_secs_f64(),
                    "task-loop round exceeded its completion timeout"
                );
                self.coordinator.request_abort();
                RoundOutcome::timeout()
            }
        }
    }

    /// Runs one of the three terminal hooks according to the outcome.
    async fn run_terminal_hook(
        &self,
        round_id: RoundId,
        active: &ActiveInteractionRound,
        outcome: &RoundOutcome,
    ) -> Result<()> {
        if outcome.error_code.as_deref() == Some(ABORTED_CODE) {
            self.handler.on_abort(round_id, active).await
        } else if outcome.is_error() {
            self.handler
                .handle_task_failed(round_id, active, outcome)
                .await
        } else {
            self.handler
                .handle_task_completion(round_id, active, outcome)
                .await
        }
    }

    /// Logs a hook failure and converts it into the `handler_error` outcome.
    fn handler_failure(message: String) -> RoundOutcome {
        warn!(error = %message, "task-loop hook failed");
        RoundOutcome::failed(HANDLER_ERROR_CODE, message)
    }

    /// Converts an executor error into the `round_execution_error` outcome.
    fn round_execution_failure(err: &Error) -> RoundOutcome {
        RoundOutcome::failed(ROUND_EXECUTION_ERROR_CODE, err.to_string())
    }
}

/// Builds the canonical "no active round" outcome.
fn no_active_round() -> RoundOutcome {
    RoundOutcome::failed(NO_ACTIVE_ROUND_CODE, NO_ACTIVE_ROUND_MESSAGE)
}

#[cfg(test)]
#[path = "controller_tests.rs"]
mod tests;
