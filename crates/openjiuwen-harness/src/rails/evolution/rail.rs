//! The harness-side evolution rail body: per-iteration metrics and the run snapshot.
//!
//! [`MetricsEvolutionRail`] implements [`EvolutionRail`] without any dependency
//! on an evolution crate: it samples the harness context, keeps one
//! [`EvolutionMetrics`] row per outer task-loop iteration, appends the
//! proposals its host drafts, and writes one [`EvolutionSnapshot`] when the
//! invocation ends.
//!
//! The rail is deliberately un-observable by the run: every IO and
//! serialization failure is logged with `tracing::warn!` and swallowed, because
//! self-evolution bookkeeping must never abort an agent run. The fallible
//! surface ([`MetricsEvolutionRail::write_snapshot`]) stays public for hosts
//! and tests that want the error.
//!
//! | Event | What the hook records |
//! | --- | --- |
//! | `BeforeInvoke` | fresh `run_id`, start clock, emptied recorder and proposal log |
//! | `AfterModelCall` | one model call, `usage` token sums, provider finish reason |
//! | `AfterToolCall` | one tool call, its name, whether it failed |
//! | `AfterTaskIteration` | one metrics row and a reset of the per-iteration counters |
//! | `AfterInvoke` | finish clock, stop reason, the persisted snapshot |
use std::collections::BTreeMap;
use std::path::PathBuf;

use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::metrics::{sum_tokens, EvolutionMetrics, IterationRecorder, MetricsClock};
use super::proposal::{EvolutionProposal, ProposalLog};
use super::snapshot::{EvolutionSnapshot, EVOLUTION_SNAPSHOT_VERSION};
use super::EvolutionRail;
use crate::rails::HarnessRailContext;

/// Name the rail registers under.
const RAIL_NAME: &str = "metrics-evolution";

/// Tuning of [`MetricsEvolutionRail`].
///
/// The capacities are bounded by default so a long-lived agent cannot grow its
/// evolution state without limit; `0` makes the matching buffer unbounded.
/// Without a path nothing is written — the rail then only serves
/// [`MetricsEvolutionRail::snapshot`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct MetricsEvolutionRailConfig {
    /// Maximum number of per-iteration rows kept, `0` for unbounded.
    pub metrics_capacity: usize,
    /// Maximum number of proposals kept, `0` for unbounded.
    pub proposal_capacity: usize,
    /// Destination of the run snapshot; `None` disables snapshot writes.
    pub snapshot_path: Option<PathBuf>,
    /// Destination of the proposal log; `None` disables proposal writes.
    pub proposal_path: Option<PathBuf>,
}

impl Default for MetricsEvolutionRailConfig {
    fn default() -> Self {
        Self {
            metrics_capacity: 256,
            proposal_capacity: 64,
            snapshot_path: None,
            proposal_path: None,
        }
    }
}

impl MetricsEvolutionRailConfig {
    /// Replaces the per-iteration row capacity.
    #[must_use]
    pub fn with_metrics_capacity(mut self, metrics_capacity: usize) -> Self {
        self.metrics_capacity = metrics_capacity;
        self
    }

    /// Replaces the proposal capacity.
    #[must_use]
    pub fn with_proposal_capacity(mut self, proposal_capacity: usize) -> Self {
        self.proposal_capacity = proposal_capacity;
        self
    }

    /// Sets the run-snapshot destination.
    #[must_use]
    pub fn with_snapshot_path(mut self, snapshot_path: impl Into<PathBuf>) -> Self {
        self.snapshot_path = Some(snapshot_path.into());
        self
    }

    /// Sets the proposal-log destination.
    #[must_use]
    pub fn with_proposal_path(mut self, proposal_path: impl Into<PathBuf>) -> Self {
        self.proposal_path = Some(proposal_path.into());
        self
    }
}

/// The mutable state of the current invocation.
///
/// The counters hold only what happened since the last iteration closed; the
/// rows already pushed into the [`IterationRecorder`] are not counted twice.
#[derive(Debug, Default)]
struct RunState {
    /// Uuid of the run, minted at `BeforeInvoke`.
    run_id: String,
    /// Agent the run belongs to.
    agent_id: String,
    /// Session the run belongs to, when it had one.
    session_id: Option<String>,
    /// Clock reading at `BeforeInvoke`.
    started_at_ms: u64,
    /// Clock reading at `AfterInvoke`; `0` until the run ends.
    finished_at_ms: u64,
    /// Clock reading when the current iteration started.
    iteration_started_ms: u64,
    /// Iterations closed since `BeforeInvoke`, including evicted rows.
    iteration_count: usize,
    /// Model calls since the last iteration closed.
    model_calls: u64,
    /// Tool calls since the last iteration closed.
    tool_calls: u64,
    /// Failed tool calls since the last iteration closed.
    failed_tool_calls: u64,
    /// Tool calls per tool name since the last iteration closed.
    tool_calls_by_name: BTreeMap<String, u64>,
    /// Prompt tokens since the last iteration closed.
    input_tokens: Option<u64>,
    /// Completion tokens since the last iteration closed.
    output_tokens: Option<u64>,
    /// Total tokens since the last iteration closed.
    total_tokens: Option<u64>,
    /// Last provider finish reason or terminal stop reason observed.
    stop_reason: Option<String>,
}

/// Records harness metrics and proposals and snapshots them at invocation end.
///
/// Register it through [`super::evolution_adapter`] like any other evolution
/// rail, or call [`MetricsEvolutionRail::snapshot`] directly when a host wants
/// the data in memory only.
pub struct MetricsEvolutionRail {
    /// Capacities and write destinations.
    config: MetricsEvolutionRailConfig,
    /// Per-iteration metrics of the current run.
    recorder: IterationRecorder,
    /// Proposals drafted during the current run.
    proposals: ProposalLog,
    /// State of the current invocation.
    run: Mutex<RunState>,
}

impl MetricsEvolutionRail {
    /// Creates a rail with the default monotonic clock.
    #[must_use]
    pub fn new(config: MetricsEvolutionRailConfig) -> Self {
        Self::with_clock(config, super::metrics::default_metrics_clock())
    }

    /// Creates a rail reading every timestamp from `clock`.
    #[must_use]
    pub fn with_clock(config: MetricsEvolutionRailConfig, clock: MetricsClock) -> Self {
        Self {
            recorder: IterationRecorder::with_clock(config.metrics_capacity, clock),
            proposals: match &config.proposal_path {
                Some(path) => ProposalLog::new(config.proposal_capacity).with_path(path.clone()),
                None => ProposalLog::new(config.proposal_capacity),
            },
            config,
            run: Mutex::new(RunState::default()),
        }
    }

    /// The configured capacities and write destinations.
    #[must_use]
    pub fn config(&self) -> &MetricsEvolutionRailConfig {
        &self.config
    }

    /// The per-iteration recorder of the current run.
    #[must_use]
    pub fn recorder(&self) -> &IterationRecorder {
        &self.recorder
    }

    /// The proposal log of the current run.
    #[must_use]
    pub fn proposals(&self) -> &ProposalLog {
        &self.proposals
    }

    /// The current clock reading, in milliseconds.
    #[must_use]
    pub fn now_ms(&self) -> u64 {
        self.recorder.now_ms()
    }

    /// Appends a proposal to the current run.
    ///
    /// A proposal whose `created_at_ms` is still `0` is stamped with the clock
    /// reading of this call.
    pub fn push_proposal(&self, mut proposal: EvolutionProposal) {
        if proposal.created_at_ms == 0 {
            proposal.created_at_ms = self.now_ms();
        }
        self.proposals.push(proposal);
    }

    /// Builds the snapshot of the current run without touching the filesystem.
    ///
    /// The totals are the recorder rows plus the counters of an iteration that
    /// has not closed yet, so a snapshot taken mid-run is still complete.
    #[must_use]
    pub fn snapshot(&self) -> EvolutionSnapshot {
        let totals = self.recorder.totals();
        let iterations = self.recorder.list();
        let proposals = self.proposals.entries();
        let now = self.now_ms();

        let (
            run_id,
            agent_id,
            session_id,
            started_at_ms,
            finished_at_ms,
            iteration_count,
            model_calls,
            tool_calls,
            failed_tool_calls,
            stop_reason,
        ) = {
            let run = self.run.lock();
            let finished_at_ms = if run.finished_at_ms == 0 {
                now
            } else {
                run.finished_at_ms
            };
            (
                run.run_id.clone(),
                run.agent_id.clone(),
                run.session_id.clone(),
                run.started_at_ms,
                finished_at_ms,
                run.iteration_count,
                totals.model_calls.saturating_add(run.model_calls),
                totals.tool_calls.saturating_add(run.tool_calls),
                totals
                    .failed_tool_calls
                    .saturating_add(run.failed_tool_calls),
                run.stop_reason.clone().or(totals.stop_reason),
            )
        };

        EvolutionSnapshot {
            version: EVOLUTION_SNAPSHOT_VERSION,
            run_id,
            agent_id,
            session_id,
            started_at_ms,
            finished_at_ms,
            duration_ms: finished_at_ms.saturating_sub(started_at_ms),
            iteration_count,
            model_calls,
            tool_calls,
            failed_tool_calls,
            stop_reason,
            iterations,
            proposals,
        }
    }

    /// Writes the snapshot and the proposal log to their configured paths.
    ///
    /// A missing path is skipped, not an error, so a host can enable the two
    /// destinations independently.
    ///
    /// # Errors
    ///
    /// Returns the first IO or serialization failure from
    /// [`EvolutionSnapshot::save`] or [`ProposalLog::save`]; the rail's hooks
    /// log and swallow it instead of aborting the run.
    pub fn write_snapshot(&self) -> Result<()> {
        if let Some(path) = self.config.snapshot_path.as_deref() {
            self.snapshot().save(path)?;
        }
        if self.config.proposal_path.is_some() {
            self.proposals.save()?;
        }
        Ok(())
    }

    /// Starts a fresh run: new id, new clock anchor, emptied buffers.
    fn begin_run(&self, ctx: &HarnessRailContext) {
        let now = self.now_ms();
        {
            let mut run = self.run.lock();
            *run = RunState {
                run_id: uuid::Uuid::new_v4().to_string(),
                agent_id: ctx.base.agent_id.clone(),
                session_id: ctx
                    .base
                    .session
                    .as_ref()
                    .map(|session| session.id().to_owned()),
                started_at_ms: now,
                iteration_started_ms: now,
                ..RunState::default()
            };
        }
        self.recorder.clear();
        self.proposals.clear();
    }

    /// Counts one model call, its usage and its finish reason.
    fn count_model_call(&self, ctx: &HarnessRailContext) {
        let mut run = self.run.lock();
        run.model_calls = run.model_calls.saturating_add(1);
        if !ctx.base.agent_id.is_empty() {
            run.agent_id.clone_from(&ctx.base.agent_id);
        }
        if let Some(response) = ctx.base.response.as_ref() {
            run.input_tokens = sum_tokens(run.input_tokens, Some(response.usage.input_tokens));
            run.output_tokens = sum_tokens(run.output_tokens, Some(response.usage.output_tokens));
            run.total_tokens = sum_tokens(run.total_tokens, Some(response.usage.total_tokens));
            if let Some(reason) = response.finish_reason.as_ref() {
                run.stop_reason = Some(reason.as_str().to_owned());
            }
        }
    }

    /// Counts one tool call by name, and whether it failed.
    ///
    /// The name comes from the pending call, or from the result's own call when
    /// the loop fires the hook with only a result attached.
    fn count_tool_call(&self, ctx: &HarnessRailContext) {
        let name = ctx
            .base
            .tool_call
            .as_ref()
            .map(|call| call.name.clone())
            .or_else(|| {
                ctx.base
                    .tool_result
                    .as_ref()
                    .map(|result| result.call.name.clone())
            });
        let failed = ctx
            .base
            .tool_result
            .as_ref()
            .is_some_and(|result| !result.success || result.error.is_some());

        let mut run = self.run.lock();
        run.tool_calls = run.tool_calls.saturating_add(1);
        if failed {
            run.failed_tool_calls = run.failed_tool_calls.saturating_add(1);
        }
        if let Some(name) = name {
            if !name.is_empty() {
                let count = run.tool_calls_by_name.entry(name).or_insert(0);
                *count = count.saturating_add(1);
            }
        }
    }

    /// Closes the current iteration into one metrics row.
    fn close_iteration(&self, ctx: &HarnessRailContext) {
        let now = self.now_ms();
        let iteration = ctx
            .task_info()
            .map_or(ctx.base.iteration, |info| info.iteration);

        let metrics = {
            let mut run = self.run.lock();
            let metrics = EvolutionMetrics {
                iteration,
                elapsed_ms: now.saturating_sub(run.iteration_started_ms),
                model_calls: run.model_calls,
                tool_calls: run.tool_calls,
                failed_tool_calls: run.failed_tool_calls,
                tool_calls_by_name: std::mem::take(&mut run.tool_calls_by_name),
                input_tokens: run.input_tokens.take(),
                output_tokens: run.output_tokens.take(),
                total_tokens: run.total_tokens.take(),
                stop_reason: run.stop_reason.clone(),
            };
            run.iteration_count = run.iteration_count.saturating_add(1);
            run.iteration_started_ms = now;
            run.model_calls = 0;
            run.tool_calls = 0;
            run.failed_tool_calls = 0;
            metrics
        };
        self.recorder.push(metrics);
    }

    /// Closes the run: finish clock, identity and terminal stop reason.
    fn close_run(&self, ctx: &HarnessRailContext) {
        let now = self.now_ms();
        let mut run = self.run.lock();
        run.finished_at_ms = now;
        if !ctx.base.agent_id.is_empty() {
            run.agent_id.clone_from(&ctx.base.agent_id);
        }
        if let Some(session) = ctx.base.session.as_ref() {
            run.session_id = Some(session.id().to_owned());
        }
        if let Some(output) = ctx.base.output.as_ref() {
            if let Some(reason) = output
                .termination_reason
                .clone()
                .or_else(|| output.finish_reason.clone())
            {
                run.stop_reason = Some(reason);
            }
        }
    }
}

impl std::fmt::Debug for MetricsEvolutionRail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricsEvolutionRail")
            .field("config", &self.config)
            .field("recorder", &self.recorder)
            .field("proposals", &self.proposals)
            .finish_non_exhaustive()
    }
}

impl EvolutionRail for MetricsEvolutionRail {
    fn name(&self) -> &str {
        RAIL_NAME
    }

    fn on_before_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.begin_run(ctx);
        Ok(())
    }

    fn on_after_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.count_model_call(ctx);
        Ok(())
    }

    fn on_after_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.count_tool_call(ctx);
        Ok(())
    }

    fn on_after_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.close_iteration(ctx);
        Ok(())
    }

    fn on_after_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.close_run(ctx);
        if let Err(error) = self.write_snapshot() {
            tracing::warn!(
                rail = self.name(),
                error = %error,
                "failed to persist the evolution snapshot"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use openjiuwen_agent::react::model_step::ModelReply;
    use openjiuwen_agent::{AgentOutput, RailEvent, ToolCallResult};
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::{FinishReason, ToolCall, Usage};

    use super::*;
    use crate::rails::TaskIterationInfo;

    /// A clock that only moves when the test advances it.
    struct TestClock(Arc<AtomicU64>);

    impl TestClock {
        fn new() -> Self {
            Self(Arc::new(AtomicU64::new(1_000)))
        }

        fn advance(&self, delta_ms: u64) -> u64 {
            self.0.fetch_add(delta_ms, Ordering::SeqCst) + delta_ms
        }

        fn clock(&self) -> MetricsClock {
            let state = Arc::clone(&self.0);
            Arc::new(move || state.load(Ordering::SeqCst))
        }
    }

    fn ctx(event: RailEvent) -> HarnessRailContext {
        HarnessRailContext::new(openjiuwen_agent::RailContext::new(event).with_agent_id("agent-1"))
    }

    fn model_reply(reason: FinishReason, input: u64, output: u64) -> ModelReply {
        ModelReply {
            message: openjiuwen_foundation::BaseMessage::assistant("reply"),
            finish_reason: Some(reason),
            usage: Usage::new(input, output),
            reasoning_present: false,
        }
    }

    /// A context carrying one successful `bash` tool call.
    fn successful_tool_ctx() -> HarnessRailContext {
        let call = ToolCall::new("bash").with_arguments(serde_json::json!({"command": "ls"}));
        let mut ctx = ctx(RailEvent::AfterToolCall);
        ctx.base.tool_result = Some(ToolCallResult::success(
            call.clone(),
            serde_json::json!({"ok": true}),
            "done",
        ));
        ctx.base.tool_call = Some(call);
        ctx
    }

    /// A context carrying one failing `read_file` tool call.
    fn failed_tool_ctx() -> HarnessRailContext {
        let call = ToolCall::new("read_file");
        let mut ctx = ctx(RailEvent::AfterToolCall);
        ctx.base.tool_result = Some(ToolCallResult::failure(call.clone(), "not found"));
        ctx.base.tool_call = Some(call);
        ctx
    }

    /// A context carrying one model reply.
    fn model_ctx(response: ModelReply) -> HarnessRailContext {
        let mut ctx = ctx(RailEvent::AfterModelCall);
        ctx.base.response = Some(response);
        ctx
    }

    fn iteration_ctx(round_id: u64, iteration: usize) -> HarnessRailContext {
        ctx(RailEvent::BeforeInvoke).with_task(TaskIterationInfo::new(round_id, iteration))
    }

    fn finished_ctx(termination_reason: Option<&str>) -> HarnessRailContext {
        let mut ctx = ctx(RailEvent::AfterInvoke);
        let output = match termination_reason {
            Some(reason) => AgentOutput::answer("done").with_termination_reason(reason),
            None => AgentOutput::answer("done"),
        };
        ctx.base.output = Some(output);
        ctx
    }

    fn rail_with(clock: &TestClock, config: MetricsEvolutionRailConfig) -> MetricsEvolutionRail {
        MetricsEvolutionRail::with_clock(config, clock.clock())
    }

    #[test]
    fn rail_is_named_and_prioritised_like_an_evolution_rail() {
        let rail = MetricsEvolutionRail::new(MetricsEvolutionRailConfig::default());
        assert_eq!(rail.name(), "metrics-evolution");
        assert_eq!(rail.priority(), crate::rails::rail_priority::EVOLUTION);
        assert_eq!(rail.priority(), 60);
        assert_eq!(rail.config().metrics_capacity, 256);
        assert_eq!(rail.config().proposal_capacity, 64);
    }

    #[test]
    fn records_one_metrics_row_per_iteration() {
        let clock = TestClock::new();
        let rail = rail_with(&clock, MetricsEvolutionRailConfig::default());

        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();

        // First iteration: one model call, two tool calls (one of them failed).
        rail.on_after_model_call(&mut model_ctx(model_reply(FinishReason::ToolCalls, 10, 5)))
            .unwrap();
        rail.on_after_tool_call(&mut successful_tool_ctx()).unwrap();
        rail.on_after_tool_call(&mut failed_tool_ctx()).unwrap();
        clock.advance(40);
        rail.on_after_task_iteration(&mut iteration_ctx(7, 0))
            .unwrap();

        // Second iteration: one model call, no tools, no clock movement.
        rail.on_after_model_call(&mut model_ctx(model_reply(FinishReason::Stop, 2, 3)))
            .unwrap();
        rail.on_after_task_iteration(&mut iteration_ctx(7, 1))
            .unwrap();

        let rows = rail.recorder().list();
        assert_eq!(rows.len(), 2, "one row per task iteration");

        let first = &rows[0];
        assert_eq!(first.iteration, 0);
        assert_eq!(first.elapsed_ms, 40);
        assert_eq!(first.model_calls, 1);
        assert_eq!(first.tool_calls, 2);
        assert_eq!(first.failed_tool_calls, 1);
        assert_eq!(first.tool_calls_by_name.get("bash").copied(), Some(1));
        assert_eq!(first.tool_calls_by_name.get("read_file").copied(), Some(1));
        assert_eq!(first.input_tokens, Some(10));
        assert_eq!(first.output_tokens, Some(5));
        assert_eq!(first.total_tokens, Some(15));
        assert_eq!(first.stop_reason.as_deref(), Some("tool_calls"));

        let second = &rows[1];
        assert_eq!(second.iteration, 1);
        assert_eq!(second.elapsed_ms, 0);
        assert_eq!(second.model_calls, 1);
        assert_eq!(second.tool_calls, 0);
        assert_eq!(second.failed_tool_calls, 0);
        assert!(second.tool_calls_by_name.is_empty());
        assert_eq!(second.input_tokens, Some(2));
        assert_eq!(second.stop_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn counters_reset_between_iterations() {
        let clock = TestClock::new();
        let rail = rail_with(&clock, MetricsEvolutionRailConfig::default());
        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();

        rail.on_after_tool_call(&mut successful_tool_ctx()).unwrap();
        rail.on_after_task_iteration(&mut iteration_ctx(1, 0))
            .unwrap();
        rail.on_after_task_iteration(&mut iteration_ctx(1, 1))
            .unwrap();

        let rows = rail.recorder().list();
        assert_eq!(rows[0].tool_calls, 1);
        assert_eq!(rows[1].tool_calls, 0, "the counter was reset");
        assert!(rows[1].tool_calls_by_name.is_empty());
    }

    #[test]
    fn a_run_without_task_iterations_still_reports_totals() {
        let clock = TestClock::new();
        let rail = rail_with(&clock, MetricsEvolutionRailConfig::default());
        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        rail.on_after_model_call(&mut model_ctx(model_reply(FinishReason::Stop, 4, 6)))
            .unwrap();
        rail.on_after_tool_call(&mut successful_tool_ctx()).unwrap();

        let snapshot = rail.snapshot();
        assert_eq!(snapshot.iteration_count, 0);
        assert_eq!(snapshot.iterations.len(), 0);
        assert_eq!(snapshot.model_calls, 1, "unflushed counters still count");
        assert_eq!(snapshot.tool_calls, 1);
    }

    #[test]
    fn proposals_are_stamped_and_ordered() {
        let clock = TestClock::new();
        let rail = rail_with(&clock, MetricsEvolutionRailConfig::default());
        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();

        clock.advance(15);
        rail.push_proposal(EvolutionProposal::new("skill_patch", "first"));
        clock.advance(15);
        rail.push_proposal(EvolutionProposal::new("prompt_tweak", "second"));

        let proposals = rail.proposals().entries();
        assert_eq!(proposals.len(), 2);
        assert_eq!(proposals[0].summary, "first");
        assert_eq!(proposals[0].created_at_ms, 1_015);
        assert_eq!(proposals[1].summary, "second");
        assert_eq!(proposals[1].created_at_ms, 1_030);
    }

    #[test]
    fn proposals_are_capped_and_a_new_run_starts_empty() {
        let clock = TestClock::new();
        let config = MetricsEvolutionRailConfig::default().with_proposal_capacity(2);
        let rail = rail_with(&clock, config);

        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        for summary in ["first", "second", "third"] {
            rail.push_proposal(EvolutionProposal::new("skill_patch", summary));
        }
        let summaries: Vec<String> = rail
            .proposals()
            .entries()
            .into_iter()
            .map(|proposal| proposal.summary)
            .collect();
        assert_eq!(summaries, ["second", "third"]);

        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        assert!(rail.proposals().is_empty(), "a new run starts clean");
        assert!(rail.recorder().is_empty());
    }

    #[test]
    fn snapshot_round_trips_through_a_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot_path = dir.path().join("evolution").join("snapshot.json");
        let clock = TestClock::new();
        let config = MetricsEvolutionRailConfig::default().with_snapshot_path(&snapshot_path);
        let rail = rail_with(&clock, config);

        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        rail.on_after_model_call(&mut model_ctx(model_reply(FinishReason::ToolCalls, 8, 2)))
            .unwrap();
        rail.on_after_tool_call(&mut successful_tool_ctx()).unwrap();
        clock.advance(25);
        rail.on_after_task_iteration(&mut iteration_ctx(3, 0))
            .unwrap();
        rail.push_proposal(EvolutionProposal::new("skill_patch", "draft"));
        clock.advance(5);

        rail.on_after_invoke(&mut finished_ctx(Some("max_iterations")))
            .unwrap();

        let loaded = EvolutionSnapshot::load(&snapshot_path).unwrap();
        assert_eq!(loaded, rail.snapshot());
        assert_eq!(loaded.version, EVOLUTION_SNAPSHOT_VERSION);
        assert!(!loaded.run_id.is_empty(), "the run id is a fresh uuid");
        assert_eq!(loaded.agent_id, "agent-1");
        assert_eq!(loaded.session_id, None);
        assert_eq!(loaded.started_at_ms, 1_000);
        assert_eq!(loaded.finished_at_ms, 1_030);
        assert_eq!(loaded.duration_ms, 30);
        assert_eq!(loaded.iteration_count, 1);
        assert_eq!(loaded.model_calls, 1);
        assert_eq!(loaded.tool_calls, 1);
        assert_eq!(loaded.failed_tool_calls, 0);
        assert_eq!(loaded.stop_reason.as_deref(), Some("max_iterations"));
        assert_eq!(loaded.iterations.len(), 1);
        assert_eq!(loaded.proposals.len(), 1);
        assert_eq!(loaded.proposals[0].summary, "draft");
    }

    #[test]
    fn each_run_gets_a_fresh_run_id() {
        let clock = TestClock::new();
        let rail = rail_with(&clock, MetricsEvolutionRailConfig::default());

        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        let first = rail.snapshot().run_id;
        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        let second = rail.snapshot().run_id;

        assert_ne!(first, "");
        assert_ne!(first, second);
    }

    #[test]
    fn proposal_log_is_written_beside_the_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let proposal_path = dir.path().join("proposals.json");
        let clock = TestClock::new();
        let config = MetricsEvolutionRailConfig::default().with_proposal_path(&proposal_path);
        let rail = rail_with(&clock, config);

        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        rail.push_proposal(EvolutionProposal::new("skill_patch", "persisted"));
        rail.on_after_invoke(&mut finished_ctx(None)).unwrap();

        let log = ProposalLog::new(0).with_path(&proposal_path);
        let stored = log.load().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].summary, "persisted");
    }

    #[test]
    fn a_failing_write_is_swallowed_by_the_hook_but_visible_on_the_fallible_surface() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, "not a directory").unwrap();

        let clock = TestClock::new();
        let config = MetricsEvolutionRailConfig::default()
            .with_snapshot_path(blocker.join("snapshot.json"))
            .with_proposal_path(blocker.join("proposals.json"));
        let rail = rail_with(&clock, config);

        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        // The hook logs the failure and keeps the run alive.
        assert!(rail.on_after_invoke(&mut finished_ctx(None)).is_ok());

        let error = rail.write_snapshot().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("evolution snapshot"));
    }

    #[test]
    fn write_snapshot_without_a_path_writes_nothing() {
        let clock = TestClock::new();
        let rail = rail_with(&clock, MetricsEvolutionRailConfig::default());
        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();

        assert!(rail.write_snapshot().is_ok());
    }

    #[test]
    fn snapshot_before_a_run_ends_uses_the_current_clock() {
        let clock = TestClock::new();
        let rail = rail_with(&clock, MetricsEvolutionRailConfig::default());
        rail.on_before_invoke(&mut ctx(RailEvent::BeforeInvoke))
            .unwrap();
        clock.advance(12);

        let snapshot = rail.snapshot();
        assert_eq!(snapshot.started_at_ms, 1_000);
        assert_eq!(snapshot.finished_at_ms, 1_012);
        assert_eq!(snapshot.duration_ms, 12);
    }

    #[test]
    fn debug_prints_config_and_buffers() {
        let rail = MetricsEvolutionRail::new(MetricsEvolutionRailConfig::default());
        let rendered = format!("{rail:?}");
        assert!(rendered.contains("MetricsEvolutionRail"), "{rendered}");
        assert!(rendered.contains("metrics_capacity: 256"), "{rendered}");
    }
}
