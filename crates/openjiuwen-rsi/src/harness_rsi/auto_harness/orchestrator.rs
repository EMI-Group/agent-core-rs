//! The bounded, resumable auto-harness session loop.
//!
//! Ported from Python `AutoHarnessOrchestrator`
//! (`openjiuwen/rsi/harness_rsi/auto_harness/orchestrator.py`, v0.1.20), whose
//! single `run_session_stream` coroutine drives one pipeline and yields chunks.
//! The Rust port splits that coroutine into an explicit lifecycle, so every step
//! is testable and a crashed run is resumable:
//!
//! - [`AutoHarnessOrchestrator::start`] opens the session (idempotent while it
//!   runs);
//! - [`AutoHarnessOrchestrator::run_iteration`] performs exactly one
//!   seed → select → mutate → evaluate → accept → checkpoint cycle;
//! - [`AutoHarnessOrchestrator::finish`] closes it and returns the
//!   [`SessionSummary`];
//! - [`AutoHarnessOrchestrator::resume`] restores a persisted [`SessionState`].
//!
//! [`AutoHarnessOrchestrator::run`] and [`AutoHarnessOrchestrator::stream`] are
//! alternative single-call entry points that drive start → iterations → finish
//! for callers that need the whole session at once rather than per-iteration
//! control.
//!
//! The loop never constructs a harness. Candidates are graded through the
//! injected [`Evaluator`] — a `DeepAgent` run behind
//! [`HarnessRunner`](super::runner::HarnessRunner), a fake, or a replay — so this
//! crate stays offline-testable and `openjiuwen-harness` stays free of RSI
//! knowledge.
//!
//! ## Concurrency and cancellation
//!
//! Session state is a [`SessionState`] behind a `std::sync::Mutex`. Locks are
//! never held across an `await`: every step snapshots or clones under the lock
//! and drops it before touching the network, the store, or the evaluator. A
//! [`CancellationToken`] is observed at each await point
//! ([`MutationStrategy::propose`], [`Evaluator::evaluate`], and
//! [`CheckpointStore::save`]) through `tokio::select!`, so
//! [`AutoHarnessOrchestrator::stop`] interrupts an in-flight evaluation without
//! leaving a lock held.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use async_stream::try_stream;
use futures::stream::BoxStream;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use openjiuwen_common::error::{Error, Result};

use crate::evolution::{
    Candidate, CandidateId, CandidateStatus, Case, Evaluator, MutationRequest, MutationStrategy,
    SelectionStrategy,
};

use super::budget::{Budget, BudgetLimits};
use super::builder::AutoHarnessOrchestratorBuilder;
use super::checkpoint::{CheckpointStore, SessionState};
use super::config::AutoHarnessConfig;
use super::event::AutoHarnessEvent;
use super::task::{IterationRecord, SessionSummary, StopConditions, StopDecision, StopReason};

/// Stable id of the seed candidate every session starts from.
///
/// The id is fixed (not a UUID) so mutation ids such as `seed/g1/v0` — and every
/// ordering derived from a candidate set — stay reproducible.
pub const SEED_CANDIDATE_ID: &str = "seed";

/// Default body of the seed candidate, used when the caller supplies none.
///
/// The body is a comment-only YAML-friendly stub: the mutator appends its marker
/// line to it, and a harness runner that receives it runs the unmodified
/// configuration.
pub const DEFAULT_SEED_BODY: &str = "# rsi auto_harness seed\n";

/// `tracing` target of the session loop.
const LOG_TARGET: &str = "openjiuwen_rsi::auto_harness::orchestrator";

/// The mutable half of the orchestrator: session state, the live budget, and the
/// one-shot terminal-event latch.
struct Inner {
    /// The session state that is checkpointed after every iteration.
    state: SessionState,
    /// The live budget; `None` until the session is started or resumed.
    budget: Option<Budget>,
    /// Whether a terminal event (`harness_session_finished`) was already emitted.
    finished_emitted: bool,
}

/// The resolved parts of an orchestrator, produced by
/// [`AutoHarnessOrchestratorBuilder::build`].
///
/// The builder lives in a sibling module, so it hands the already-defaulted and
/// already-validated parts over through this record instead of a long argument
/// list.
pub(crate) struct OrchestratorParts {
    /// Validated configuration.
    pub(crate) config: AutoHarnessConfig,
    /// Session id (`uuid_v4` unless the caller set one).
    pub(crate) session_id: String,
    /// Body of the seed candidate.
    pub(crate) seed_body: String,
    /// Number of proposals requested per iteration.
    pub(crate) mutation_count: usize,
    /// Cases every candidate is graded against.
    pub(crate) cases: Vec<Case>,
    /// Mutation strategy.
    pub(crate) mutator: Arc<dyn MutationStrategy>,
    /// Selection strategy.
    pub(crate) selector: Arc<dyn SelectionStrategy>,
    /// Evaluator (the supplied one, or a runner adapter).
    pub(crate) evaluator: Arc<dyn Evaluator>,
    /// Checkpoint backend.
    pub(crate) checkpoint_store: Arc<dyn CheckpointStore>,
    /// Budget limits derived from the configuration.
    pub(crate) limits: BudgetLimits,
    /// The iteration stamped on evaluator requests, advanced per iteration.
    pub(crate) iteration_clock: Arc<AtomicU32>,
}

/// What one iteration collected before its [`IterationRecord`] is built.
struct IterationTally {
    /// 1-based iteration number.
    iteration: u32,
    /// Candidates used as mutation parents.
    parents: Vec<CandidateId>,
    /// Candidates the mutator proposed, duplicates included.
    proposed: usize,
    /// Candidates that were evaluated.
    evaluated: usize,
    /// Evaluated candidates that produced a graded report.
    valid: usize,
    /// Ids of the candidates evaluated in this iteration.
    evaluated_ids: Vec<CandidateId>,
}

/// Drives one bounded, resumable auto-harness optimisation session.
///
/// Construct it with [`AutoHarnessOrchestrator::builder`]; every collaborator
/// (runner/evaluator, mutator, selector, checkpoint store, cases) is injected, so
/// the loop is deterministic and offline-testable. Behaviour that callers must
/// know about:
///
/// - a session that already finished (it carries a `stop_reason`) refuses further
///   iterations with [`Error::Cancelled`], from `run_iteration`, `run`, and
///   `stream`;
/// - [`AutoHarnessOrchestrator::run_iteration`] auto-starts a session that was
///   never started;
/// - an evaluation error marks the candidate [`CandidateStatus::Invalid`] and the
///   session continues — only cancellation aborts an iteration;
/// - `run` and `stream` are two alternative single-call entry points; both mark
///   the session's terminal event as emitted, so only one of them reports a
///   terminal event per session.
pub struct AutoHarnessOrchestrator {
    /// Validated configuration.
    config: AutoHarnessConfig,
    /// Session id used for checkpoints and events.
    session_id: String,
    /// Body of the seed candidate.
    seed_body: String,
    /// Number of proposals requested per iteration.
    mutation_count: usize,
    /// Cases every candidate is graded against.
    cases: Vec<Case>,
    /// Mutation strategy.
    mutator: Arc<dyn MutationStrategy>,
    /// Selection strategy.
    selector: Arc<dyn SelectionStrategy>,
    /// Evaluator used to grade every proposal.
    evaluator: Arc<dyn Evaluator>,
    /// Checkpoint backend.
    checkpoint_store: Arc<dyn CheckpointStore>,
    /// Budget limits derived from the configuration.
    limits: BudgetLimits,
    /// The iteration stamped on evaluator requests, advanced per iteration.
    iteration_clock: Arc<AtomicU32>,
    /// Cancellation token observed at every await point.
    cancel: CancellationToken,
    /// Session state, live budget, and the terminal-event latch.
    inner: Mutex<Inner>,
}

// `Arc<dyn ...>` collaborators are not `Debug`, so the orchestrator is
// summarized by its session id and state instead of every field.
impl std::fmt::Debug for AutoHarnessOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state();
        f.debug_struct("AutoHarnessOrchestrator")
            .field("session_id", &self.session_id)
            .field("iteration", &state.iteration)
            .field("running", &state.running)
            .field("stop_reason", &state.stop_reason)
            .finish_non_exhaustive()
    }
}

impl AutoHarnessOrchestrator {
    /// Starts building an orchestrator for `config`.
    ///
    /// The builder validates the configuration and requires either a
    /// [`HarnessRunner`](super::runner::HarnessRunner) or an [`Evaluator`].
    #[must_use]
    pub fn builder(config: AutoHarnessConfig) -> AutoHarnessOrchestratorBuilder {
        AutoHarnessOrchestratorBuilder::new(config)
    }

    /// Assembles the orchestrator from already-defaulted parts.
    pub(crate) fn assemble(parts: OrchestratorParts) -> Self {
        let state = SessionState::new(parts.session_id.clone());
        Self {
            config: parts.config,
            session_id: parts.session_id,
            seed_body: parts.seed_body,
            mutation_count: parts.mutation_count,
            cases: parts.cases,
            mutator: parts.mutator,
            selector: parts.selector,
            evaluator: parts.evaluator,
            checkpoint_store: parts.checkpoint_store,
            limits: parts.limits,
            iteration_clock: parts.iteration_clock,
            cancel: CancellationToken::new(),
            inner: Mutex::new(Inner {
                state,
                budget: None,
                finished_emitted: false,
            }),
        }
    }

    /// Returns the configuration the session was built with.
    #[must_use]
    pub fn config(&self) -> &AutoHarnessConfig {
        &self.config
    }

    /// Returns the session id used for checkpoints and events.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns a snapshot of the session state.
    ///
    /// The snapshot includes candidates, history, counters, the incumbent best,
    /// and the spend; mutating it has no effect on the session.
    #[must_use]
    pub fn state(&self) -> SessionState {
        self.state_guard().state.clone()
    }

    /// Returns whether the session is currently accepting iterations.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.state_guard().state.running
    }

    /// Returns whether [`Self::stop`] was called or the token was cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Returns the budget limits the session was built with (test-only view).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn limits(&self) -> &BudgetLimits {
        &self.limits
    }

    /// Returns the cases every candidate is graded against (test-only view).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn cases(&self) -> &[Case] {
        &self.cases
    }

    /// Returns how many proposals each iteration requests (test-only view).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn mutation_count(&self) -> usize {
        self.mutation_count
    }

    /// Starts the session: marks it running and opens a fresh budget.
    ///
    /// Calling it while the session is already running is a no-op, so `start`
    /// never resets the accumulated spend. Starting a session that already
    /// finished (see [`Self::finish`] and [`Self::stop`]) is a caller error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the session already carries a
    /// `stop_reason`, because a finished session cannot be restarted.
    pub fn start(&self) -> Result<()> {
        let mut guard = self.state_guard();
        if guard.state.is_finished() {
            return Err(Error::validation(format!(
                "auto-harness session `{}` has already finished",
                self.session_id
            )));
        }
        if guard.state.running {
            return Ok(());
        }
        guard.state.running = true;
        guard.state.session_id.clone_from(&self.session_id);
        let spent_usd = guard.state.spent_usd;
        guard.budget = Some(Budget::with_spent(self.limits.clone(), spent_usd));
        Ok(())
    }

    /// Restores the persisted state of this session, when there is one.
    ///
    /// `Ok(None)` means the store holds no checkpoint for the session id; the
    /// session is then left untouched. When a checkpoint exists, its candidates,
    /// history, counters, incumbent best, and spend replace the in-memory state,
    /// `running` is restored as `true`, and the budget starts with the persisted
    /// spend so cost limits continue where the previous process stopped. A
    /// checkpoint that already carries a `stop_reason` keeps it, and further
    /// iterations stay refused.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the checkpoint store cannot read (or
    /// parse) the checkpoint.
    pub async fn resume(&self) -> Result<Option<SessionState>> {
        let Some(loaded) = self.checkpoint_store.load(&self.session_id).await? else {
            return Ok(None);
        };
        let mut guard = self.state_guard();
        let spent_usd = loaded.spent_usd;
        guard.state = loaded;
        guard.state.running = true;
        guard.budget = Some(Budget::with_spent(self.limits.clone(), spent_usd));
        guard.finished_emitted = false;
        Ok(Some(guard.state.clone()))
    }

    /// Runs exactly one iteration and returns its [`IterationRecord`].
    ///
    /// A session that was never started is started implicitly. Duplicate proposal
    /// ids are skipped (they count as proposed but are not evaluated), evaluation
    /// failures mark the candidate invalid instead of aborting the session, and
    /// the state is checkpointed before the record is returned.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the session is stopped, finished, or the
    /// cancellation token fired; [`Error::Validation`] when the mutator rejects
    /// the request; and [`Error::Session`] when the checkpoint cannot be written.
    pub async fn run_iteration(&self) -> Result<IterationRecord> {
        self.run_iteration_with(&mut Vec::new()).await
    }

    /// Cancels the session.
    ///
    /// The token is cancelled first (so an in-flight iteration aborts at its next
    /// await point), then the state is marked not running. A `stop_reason` that
    /// is already set — for example from a stop condition that just fired — is
    /// kept; otherwise it becomes [`StopReason::Cancelled`].
    pub fn stop(&self) {
        self.cancel.cancel();
        let mut guard = self.state_guard();
        guard.state.running = false;
        if guard.state.stop_reason.is_none() {
            guard.state.stop_reason = Some(StopReason::Cancelled);
        }
    }

    /// Closes the session, persists it, and returns its summary.
    ///
    /// The summary counts the completed iterations, the evaluations and accepted
    /// improvements recorded in the history, and the budget spend. When no stop
    /// condition fired, the stop reason defaults to [`StopReason::MaxIterations`]
    /// if the configured iteration limit is reached, else the last record's stop
    /// reason, else [`StopReason::Failed`] (a session closed early without any
    /// stop condition reports `Failed`, mirroring Python where only real stop
    /// conditions set `stop_reason`). Calling `finish` twice is harmless: the
    /// summary is recomputed and the state re-persisted.
    ///
    /// `finish` does not emit events; [`Self::stream`] emits the single
    /// `harness_session_finished` event with this summary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the final checkpoint cannot be written.
    pub async fn finish(&self) -> Result<SessionSummary> {
        let summary = {
            let mut guard = self.state_guard();
            guard.state.running = false;
            if guard.state.stop_reason.is_none() {
                guard.state.stop_reason = Some(self.default_stop_reason(&guard.state));
            }
            if let Some(budget) = guard.budget.as_ref() {
                guard.state.spent_usd = budget.spent_usd();
            }
            summarize(&guard.state)
        };
        let snapshot = self.state();
        self.checkpoint_store.save(&snapshot).await.map_err(|err| {
            Error::session(format!(
                "failed to persist auto-harness session `{}`: {err}",
                self.session_id
            ))
        })?;
        Ok(summary)
    }

    /// Streams the session's events, driving the session to completion.
    ///
    /// This is a single-call entry point (see also [`Self::run`]): it starts the
    /// session, runs iterations until a stop condition fires, finishes, and ends
    /// with exactly one `harness_session_finished` event. Events of an iteration
    /// are yielded once that iteration completed, in the order they happened
    /// (`iteration_started`, `candidates_evaluated`, `iteration_finished`,
    /// `checkpoint_saved`). Cancellation ends the stream with `session_cancelled`
    /// instead of the terminal event.
    ///
    /// A session that already finished, a session whose terminal event was
    /// already emitted, or a cancelled session yields a single
    /// [`Error::Cancelled`].
    pub fn stream(self: &Arc<Self>) -> BoxStream<'static, Result<AutoHarnessEvent>> {
        let orchestrator = Arc::clone(self);
        let events: BoxStream<'static, Result<AutoHarnessEvent>> = Box::pin(try_stream! {
            if orchestrator.state().is_finished()
                || orchestrator.finished_emitted()
                || orchestrator.is_cancelled()
            {
                Err(Error::cancelled())?;
            }
            orchestrator.start()?;
            yield AutoHarnessEvent::SessionStarted {
                session_id: orchestrator.session_id.clone(),
                max_iterations: orchestrator.config.max_iterations,
            };
            loop {
                if orchestrator.is_cancelled() {
                    yield AutoHarnessEvent::SessionCancelled {
                        iteration: orchestrator.state().iteration,
                    };
                    return;
                }
                if orchestrator.state().is_finished() {
                    break;
                }
                let mut buffered = Vec::new();
                match orchestrator.run_iteration_with(&mut buffered).await {
                    Ok(_) => {
                        for event in buffered {
                            yield event;
                        }
                    }
                    Err(err) if err.is_cancelled() => {
                        yield AutoHarnessEvent::SessionCancelled {
                            iteration: orchestrator.state().iteration,
                        };
                        return;
                    }
                    Err(err) => Err(err)?,
                }
            }
            let summary = orchestrator.finish().await?;
            orchestrator.mark_finished_emitted();
            yield AutoHarnessEvent::SessionFinished { summary };
        });
        events
    }

    /// Runs the session to completion and returns its [`SessionSummary`].
    ///
    /// This is the event-free counterpart of [`Self::stream`]: it starts the
    /// session, runs iterations until a stop condition fires, and finishes. Both
    /// entry points mark the terminal event as emitted, so they are mutually
    /// exclusive per session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the session already finished or was
    /// cancelled, plus every error [`Self::run_iteration`] and [`Self::finish`]
    /// report.
    pub async fn run(self: &Arc<Self>) -> Result<SessionSummary> {
        if self.state().is_finished() || self.finished_emitted() || self.is_cancelled() {
            return Err(Error::cancelled());
        }
        self.start()?;
        loop {
            if self.is_cancelled() {
                return Err(Error::cancelled());
            }
            if self.state().is_finished() {
                break;
            }
            self.run_iteration().await?;
        }
        self.mark_finished_emitted();
        self.finish().await
    }

    /// One iteration, buffering its events into `events`.
    ///
    /// Kept off the public API: [`Self::run_iteration`] calls it with a scratch
    /// buffer and [`Self::stream`] yields the buffer after the iteration.
    async fn run_iteration_with(
        &self,
        events: &mut Vec<AutoHarnessEvent>,
    ) -> Result<IterationRecord> {
        let started = Instant::now();
        self.ensure_ready()?;
        self.seed_if_empty();
        let (parent_ids, parents, iteration) = self.select_parents();
        self.iteration_clock
            .store(iteration, std::sync::atomic::Ordering::Relaxed);
        events.push(AutoHarnessEvent::IterationStarted {
            iteration,
            parents: parent_ids.iter().map(|id| id.as_str().to_owned()).collect(),
        });

        let proposals = self.propose(parents, iteration).await?;
        let proposed = proposals.len();
        let fresh = self.discard_duplicates(proposals);
        let (evaluated, valid, evaluated_ids) = self.evaluate_proposals(fresh).await?;
        let tally = IterationTally {
            iteration,
            parents: parent_ids,
            proposed,
            evaluated,
            valid,
            evaluated_ids,
        };
        let record = self.complete_iteration(tally, started, events);
        events.push(AutoHarnessEvent::IterationFinished {
            record: record.clone(),
        });
        self.checkpoint(iteration, events).await?;
        Ok(record)
    }

    /// Refuses a finished/cancelled session and auto-starts a fresh one.
    fn ensure_ready(&self) -> Result<()> {
        if self.cancel.is_cancelled() || self.state_guard().state.is_finished() {
            return Err(Error::cancelled());
        }
        if !self.is_running() {
            self.start()?;
        }
        if self.cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        Ok(())
    }

    /// Seeds the candidate set so there is always something to mutate.
    fn seed_if_empty(&self) {
        let mut guard = self.state_guard();
        if guard.state.candidates.is_empty() {
            guard
                .state
                .candidates
                .push(Candidate::new(SEED_CANDIDATE_ID, self.seed_body.clone()));
        }
    }

    /// Selects the mutation parents and derives the iteration number.
    ///
    /// Selection ids that are not (or no longer) candidates are skipped rather
    /// than trusted, so a stale selector cannot panic the loop.
    fn select_parents(&self) -> (Vec<CandidateId>, Vec<Candidate>, u32) {
        let guard = self.state_guard();
        let iteration = guard.state.iteration + 1;
        let parent_ids = self.selector.select(&guard.state.candidates, 1);
        let parents: Vec<Candidate> = parent_ids
            .iter()
            .filter_map(|id| {
                guard
                    .state
                    .candidates
                    .iter()
                    .find(|candidate| &candidate.id == id)
                    .cloned()
            })
            .collect();
        if parents.len() < parent_ids.len() {
            debug!(
                target: LOG_TARGET,
                session_id = %self.session_id,
                "ignoring selection ids that are not candidates"
            );
        }
        (parent_ids, parents, iteration)
    }

    /// Asks the mutator for the next generation of `iteration`.
    async fn propose(&self, parents: Vec<Candidate>, iteration: u32) -> Result<Vec<Candidate>> {
        let request = MutationRequest {
            parents,
            generation: iteration,
            count: self.mutation_count,
            metadata: BTreeMap::new(),
        };
        self.cancellable(self.mutator.propose(request)).await
    }

    /// Drops proposals whose id the session already knows.
    ///
    /// The duplicates still count as proposed (the mutator produced them) but are
    /// never evaluated, so a strategy that repeats itself cannot burn the budget.
    fn discard_duplicates(&self, proposals: Vec<Candidate>) -> Vec<Candidate> {
        let mut seen: BTreeSet<String> = self
            .state_guard()
            .state
            .candidates
            .iter()
            .map(|candidate| candidate.id.as_str().to_owned())
            .collect();
        let mut fresh = Vec::with_capacity(proposals.len());
        for candidate in proposals {
            if !seen.insert(candidate.id.as_str().to_owned()) {
                debug!(
                    target: LOG_TARGET,
                    session_id = %self.session_id,
                    candidate = %candidate.id,
                    "skipping duplicate candidate id"
                );
                continue;
            }
            fresh.push(candidate);
        }
        fresh
    }

    /// Evaluates every proposal, returning `(evaluated, valid, ids)`.
    ///
    /// An evaluation error marks the candidate invalid and the session continues;
    /// only cancellation aborts the iteration.
    async fn evaluate_proposals(
        &self,
        proposals: Vec<Candidate>,
    ) -> Result<(usize, usize, Vec<CandidateId>)> {
        let cases = self.cases.clone();
        let mut evaluated = 0_usize;
        let mut valid = 0_usize;
        let mut evaluated_ids: Vec<CandidateId> = Vec::with_capacity(proposals.len());
        for mut candidate in proposals {
            match self
                .cancellable(self.evaluator.evaluate(&candidate, &cases))
                .await
            {
                Ok(report) => {
                    if report.score().is_some() {
                        valid += 1;
                    }
                    evaluated += 1;
                    evaluated_ids.push(candidate.id.clone());
                    candidate.evaluation = Some(report.clone());
                    candidate.status = CandidateStatus::Evaluated;
                    let mut guard = self.state_guard();
                    if let Some(budget) = guard.budget.as_mut() {
                        budget.record_cost(report.cost_usd);
                        guard.state.spent_usd = budget.spent_usd();
                    }
                    guard.state.candidates.push(candidate);
                }
                Err(err) if err.is_cancelled() => return Err(err),
                Err(err) => {
                    warn!(
                        target: LOG_TARGET,
                        session_id = %self.session_id,
                        candidate = %candidate.id,
                        error = %err,
                        "candidate evaluation failed; marking it invalid"
                    );
                    candidate.status = CandidateStatus::Invalid;
                    candidate.metadata.insert(
                        "error".to_owned(),
                        serde_json::Value::String(err.to_string()),
                    );
                    self.state_guard().state.candidates.push(candidate);
                }
            }
        }
        Ok((evaluated, valid, evaluated_ids))
    }

    /// Compares the iteration's best with the incumbent, records the iteration,
    /// and evaluates the stop conditions.
    fn complete_iteration(
        &self,
        tally: IterationTally,
        started: Instant,
        events: &mut Vec<AutoHarnessEvent>,
    ) -> IterationRecord {
        let mut guard = self.state_guard();
        let previous_best_score = guard.state.best_score;
        let best = best_of(&guard.state.candidates, &tally.evaluated_ids);
        let (best_candidate, best_score) = match &best {
            Some((id, score)) => (Some(id.clone()), Some(*score)),
            None => (None, None),
        };
        // The first scored candidate has no incumbent, so it reports no
        // improvement; a decreasing best never reports a negative one.
        let improvement = match (best_score, previous_best_score) {
            (Some(best), Some(previous)) => (best - previous).max(0.0),
            _ => 0.0,
        };
        let accepted = match (best_score, previous_best_score) {
            (Some(best), Some(previous)) => best > previous,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if accepted {
            guard.state.best_candidate.clone_from(&best_candidate);
            guard.state.best_score = best_score;
            if let Some(winner) = guard
                .state
                .candidates
                .iter_mut()
                .find(|candidate| Some(&candidate.id) == best_candidate.as_ref())
            {
                winner.status = CandidateStatus::Selected;
            }
        }
        events.push(AutoHarnessEvent::CandidatesEvaluated {
            iteration: tally.iteration,
            proposed: tally.proposed,
            evaluated: tally.evaluated,
            valid: tally.valid,
            best_score,
        });

        let mut record = IterationRecord {
            iteration: tally.iteration,
            parents: tally.parents,
            proposed: tally.proposed,
            evaluated: tally.evaluated,
            valid: tally.valid,
            best_candidate,
            best_score,
            previous_best_score,
            improvement,
            accepted,
            duration_ms: elapsed_ms(started),
            stop: None,
        };

        // The stop decision sees the iteration it just completed, so a target
        // score or the iteration cap takes effect immediately.
        guard.state.history.push(record.clone());
        guard.state.iteration = tally.iteration;
        let decision = match guard.budget.as_ref() {
            Some(budget) => StopConditions::from_config(&self.config).evaluate(
                &guard.state.history,
                tally.iteration,
                budget,
            ),
            None => StopDecision::go(),
        };
        if decision.stop {
            if let Some(reason) = decision.reason {
                record.stop = Some(reason);
                guard.state.stop_reason = Some(reason);
                if let Some(last) = guard.state.history.last_mut() {
                    last.stop = Some(reason);
                }
            }
            guard.state.running = false;
            debug!(
                target: LOG_TARGET,
                session_id = %self.session_id,
                reason = ?record.stop,
                detail = %decision.detail,
                "auto-harness session stopping"
            );
        }
        record
    }

    /// Persists the session state; a failed write is reported, never swallowed.
    async fn checkpoint(&self, iteration: u32, events: &mut Vec<AutoHarnessEvent>) -> Result<()> {
        let snapshot = self.state();
        match self
            .cancellable(self.checkpoint_store.save(&snapshot))
            .await
        {
            Ok(()) => {
                events.push(AutoHarnessEvent::CheckpointSaved {
                    session_id: self.session_id.clone(),
                    iteration,
                });
                Ok(())
            }
            Err(err) if err.is_cancelled() => Err(err),
            Err(err) => Err(Error::session(format!(
                "failed to persist auto-harness session `{}`: {err}",
                self.session_id
            ))),
        }
    }

    /// Awaits `future`, aborting with [`Error::Cancelled`] when the session's
    /// token fires first.
    async fn cancellable<T>(
        &self,
        future: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        tokio::select! {
            () = self.cancel.cancelled() => Err(Error::cancelled()),
            result = future => result,
        }
    }

    /// The stop reason `finish` records when no stop condition fired.
    fn default_stop_reason(&self, state: &SessionState) -> StopReason {
        if state.iteration >= self.config.max_iterations {
            return StopReason::MaxIterations;
        }
        state
            .history
            .last()
            .and_then(|record| record.stop)
            .unwrap_or(StopReason::Failed)
    }

    /// Returns whether a terminal event was already emitted for the session.
    fn finished_emitted(&self) -> bool {
        self.state_guard().finished_emitted
    }

    /// Latches the terminal-event emission.
    fn mark_finished_emitted(&self) {
        self.state_guard().finished_emitted = true;
    }

    /// Locks the mutable state, recovering a poisoned lock.
    ///
    /// A poisoned lock only happens if another thread panicked while holding it;
    /// the state is a plain snapshot, so recovering keeps the orchestrator usable
    /// instead of panicking the caller.
    fn state_guard(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Returns the highest-scoring candidate among `ids`, ties by id ascending.
///
/// Only candidates evaluated in the iteration are considered, and an ungraded or
/// invalid candidate never wins.
fn best_of(candidates: &[Candidate], ids: &[CandidateId]) -> Option<(CandidateId, f64)> {
    let mut best: Option<(CandidateId, f64)> = None;
    for id in ids {
        let Some(candidate) = candidates.iter().find(|candidate| &candidate.id == id) else {
            continue;
        };
        let Some(score) = candidate.score() else {
            continue;
        };
        best = match best {
            None => Some((id.clone(), score)),
            Some((best_id, best_score)) => match score.partial_cmp(&best_score) {
                Some(Ordering::Greater) => Some((id.clone(), score)),
                Some(Ordering::Equal) if id.as_str() < best_id.as_str() => {
                    Some((id.clone(), score))
                }
                _ => Some((best_id, best_score)),
            },
        };
    }
    best
}

/// Builds the [`SessionSummary`] of `state`.
fn summarize(state: &SessionState) -> SessionSummary {
    SessionSummary {
        session_id: state.session_id.clone(),
        iterations: state.iteration,
        best_candidate: state.best_candidate.clone(),
        best_score: state.best_score,
        evaluated: state
            .history
            .iter()
            .map(|record| u64::try_from(record.evaluated).unwrap_or(u64::MAX))
            .sum(),
        accepted: u64::try_from(
            state
                .history
                .iter()
                .filter(|record| record.accepted)
                .count(),
        )
        .unwrap_or(u64::MAX),
        stop_reason: state.stop_reason,
        spent_usd: state.spent_usd,
    }
}

/// Milliseconds elapsed since `started`, saturating instead of wrapping.
fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

// The suite lives in its own file so this module stays within the crate's
// ~1000-line file budget; it is still a child module of `orchestrator`, so
// `use super::*` reaches every private helper.
#[cfg(test)]
#[path = "orchestrator_tests.rs"]
mod tests;
