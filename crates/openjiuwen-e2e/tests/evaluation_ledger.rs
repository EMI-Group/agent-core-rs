//! Integration scenario: the `openjiuwen::agent_evolving` evaluator pipeline over a
//! fake command runner, plus the `openjiuwen::rsi` run ledger, its progress
//! callbacks, and the `AutoHarnessEvent` stream over a scripted harness runner.
//!
//! Everything is offline and deterministic. The pipeline never spawns `docker` or
//! `git`: [`FakeCommandRunner`] answers the whole Docker lifecycle from a script and
//! records every command line, which is the evidence the scenarios assert on. The
//! RSI half is driven by a scripted [`HarnessRunner`] whose score queue makes the
//! session converge after a fixed number of iterations.
//!
//! What this file adds over `self_improvement.rs` (which drives the orchestrator
//! iteration by iteration and the trainer through checkpoints):
//!
//! 1. The evaluator pipeline's Docker lifecycle and `ScoreSpec::TestSummary` scoring,
//!    including batch aggregation and the typed missing-`docker` failure.
//! 2. The run ledger: bind, deduplication by `call_id`, cumulative totals, per-model
//!    and per-stage metrics, the usage snapshot, the persisted event order, and
//!    `check_delivery` while a call is still pending.
//! 3. `emit_progress` reaching an injected callback exactly once.
//! 4. The orchestrator's event stream end to end, with a ledger maintained alongside
//!    it that records one usage record per evaluated candidate.
//!
//! No assertion reads a generated identifier, a timestamp, or a duration; counts,
//! ordering, scores, and statuses are what is checked.
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen::agent_evolving::evaluator_pipeline::{
    parse_score, CommandRunner, EvaluationPipeline, EvaluationTask, ExecResult, FakeCommandRunner,
    PipelineConfig, ScoreSpec,
};
use openjiuwen::core::common::error::{ErrorKind, Result};
use openjiuwen::rsi::harness_rsi::progress::{
    emit_progress, OptimizationProgressEvent, ProgressCallback,
};
use openjiuwen::rsi::ledger::{StageRecord, UNATTRIBUTED_STAGE};
use openjiuwen::rsi::{
    AutoHarnessConfig, AutoHarnessEvent, AutoHarnessOrchestrator, Case as RsiCase, CaseResult,
    EvaluationReport, HarnessRunOutcome, HarnessRunRequest, HarnessRunner, InMemoryLedgerStore,
    LedgerStore, ModelCall, ModelCallStatus, RunEvent, RunLedger, UsageRecord, UsageTokens,
    UsageTotals,
};
use serde_json::json;
use support::{collect_bounded, within};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

/// The task id the ledger scenarios record under; it is also the store key.
const LEDGER_TASK_ID: &str = "e2e-ledger-task";

/// Session id the streamed orchestrator persists its checkpoint under.
const STREAM_SESSION_ID: &str = "e2e-evaluation-ledger-stream";

/// Model name every ledger record of the streamed session reports.
const SCRIPTED_MODEL: &str = "scripted-e2e";

/// Stage name the streamed session attributes its usage records to.
const EVALUATE_STAGE: &str = "evaluate";

// ---------------------------------------------------------------------------
// Evaluator pipeline fixtures
// ---------------------------------------------------------------------------

/// A shared fake runner whose unscripted commands succeed with empty output.
///
/// `record_only` keeps the script down to the two commands the assertions care
/// about (`docker run`, `docker exec`): the workspace copy and `docker rm -f` still
/// happen and are still recorded, but they need no queued result.
fn recorded_runner() -> Arc<FakeCommandRunner> {
    Arc::new(FakeCommandRunner::new().record_only())
}

/// The fake runner behind the trait object the pipeline constructor takes.
fn boxed(runner: &Arc<FakeCommandRunner>) -> Arc<dyn CommandRunner> {
    runner.clone()
}

/// An in-memory ledger store behind the trait object the ledger takes.
fn ledger_store(store: &Arc<InMemoryLedgerStore>) -> Arc<dyn LedgerStore> {
    store.clone()
}

/// A pipeline rooted at `root` scoring with `ScoreSpec::TestSummary`.
fn test_summary_pipeline(runner: &Arc<FakeCommandRunner>, root: &Path) -> EvaluationPipeline {
    EvaluationPipeline::new(
        PipelineConfig::new(root, "openjiuwen-eval:e2e").with_score_spec(ScoreSpec::TestSummary),
        boxed(runner),
    )
}

/// A task with one test command and the default (exit-code) score policy.
///
/// The default policy is what makes the task inherit the pipeline's
/// `TestSummary` policy, so the exec output decides the score.
fn eval_task(task_id: &str) -> EvaluationTask {
    EvaluationTask::new(task_id, "make the tests pass", ["pytest", "-q"]).expect("valid task")
}

/// `true` when both scores are within floating-point noise of each other.
fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

// ---------------------------------------------------------------------------
// RSI fixtures
// ---------------------------------------------------------------------------

/// A [`HarnessRunner`] that answers with queued scores and records every request.
///
/// Copied from `self_improvement.rs`: the orchestrator only reads the produced
/// [`EvaluationReport`], so a scripted score queue is the whole seam, and the
/// request log is the evidence for one run per iteration.
struct ScriptedRunner {
    /// Scores handed out by successive runs, oldest first.
    scores: Mutex<VecDeque<f64>>,
    /// Every request the orchestrator issued, in call order.
    requests: Mutex<Vec<HarnessRunRequest>>,
}

impl ScriptedRunner {
    /// Creates a runner answering with `scores` in order.
    fn new(scores: Vec<f64>) -> Arc<Self> {
        Arc::new(Self {
            scores: Mutex::new(scores.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        })
    }

    /// Returns a copy of the recorded requests.
    fn requests(&self) -> Vec<HarnessRunRequest> {
        self.requests
            .lock()
            .expect("request log lock")
            .iter()
            .cloned()
            .collect()
    }
}

#[async_trait]
impl HarnessRunner for ScriptedRunner {
    fn name(&self) -> &str {
        SCRIPTED_MODEL
    }

    async fn run(&self, request: HarnessRunRequest) -> Result<HarnessRunOutcome> {
        let score = {
            let mut scores = self.scores.lock().expect("score queue lock");
            scores
                .pop_front()
                .expect("scripted runner needs one score per run")
        };
        self.requests
            .lock()
            .expect("request log lock")
            .push(request);
        Ok(HarnessRunOutcome {
            report: graded_report(score),
            ..HarnessRunOutcome::default()
        })
    }
}

/// A one-case report scoring `score`; a graded report is what makes a candidate
/// eligible to become the incumbent.
fn graded_report(score: f64) -> EvaluationReport {
    EvaluationReport::from_results(vec![CaseResult {
        case_id: "case-1".to_owned(),
        passed: true,
        score: Some(score),
        weight: 1.0,
        error: None,
        details: json!({"note": "graded"}),
    }])
}

/// A configuration rooted at the scenario's temp directory, every other field defaulted.
fn rsi_config(data_dir: &Path) -> AutoHarnessConfig {
    AutoHarnessConfig {
        data_dir: data_dir.to_path_buf(),
        ..AutoHarnessConfig::default()
    }
}

/// The single case every candidate is graded against.
fn rsi_case() -> RsiCase {
    RsiCase {
        id: "case-1".to_owned(),
        ..RsiCase::default()
    }
}

/// One usage record for `call_id`, carrying `tokens` for `model`.
fn usage_record(call_id: &str, model: &str, tokens: UsageTokens) -> UsageRecord {
    UsageRecord::new(
        LEDGER_TASK_ID,
        call_id,
        ModelCall::new(model).with_tokens(tokens),
    )
}

/// The number of calls a metrics bucket covers, when the bucket exists.
fn bucket_calls(bucket: Option<&UsageTotals>) -> Option<u64> {
    bucket.map(|bucket| bucket.calls)
}

/// Records one usage record per candidate every `candidates_evaluated` event reports.
///
/// The orchestrator exposes no ledger hook, so the caller keeps the ledger alongside
/// the stream. The returned iteration numbers are the ones that reported candidates,
/// in stream order.
async fn record_evaluated_candidates(
    ledger: &mut RunLedger,
    events: &[AutoHarnessEvent],
) -> Vec<u32> {
    let mut iterations = Vec::new();
    for event in events {
        let AutoHarnessEvent::CandidatesEvaluated {
            iteration,
            evaluated,
            ..
        } = event
        else {
            continue;
        };
        iterations.push(*iteration);
        for index in 0..*evaluated {
            let record = UsageRecord::new(
                STREAM_SESSION_ID,
                format!("iteration-{iteration}-candidate-{index}"),
                ModelCall::new(SCRIPTED_MODEL).with_tokens(UsageTokens::new(
                    Some(120),
                    Some(30),
                    None,
                )),
            )
            .with_node_ref(format!("iteration-{iteration}"))
            .with_stage_ref(EVALUATE_STAGE);
            assert!(
                within("record candidate usage", ledger.record_usage(record))
                    .await
                    .expect("candidate usage persists")
            );
        }
    }
    iterations
}

// ---------------------------------------------------------------------------
// Evaluator pipeline scenarios
// ---------------------------------------------------------------------------

/// A failing test summary is a report, not an error: the score is the pass ratio, the
/// exec exit code is data, and the container is removed before the report returns.
#[tokio::test]
async fn a_failing_test_summary_scores_the_exec_and_removes_the_container() {
    let dir = TempDir::new().expect("scenario temp dir");
    let runner = recorded_runner();
    runner.push_result("docker", "run", ExecResult::ok("cid-e2e\n"));
    runner.push_result(
        "docker",
        "exec",
        ExecResult::new("", "== 1 failed, 2 passed in 0.1s ==", 1),
    );
    let pipeline = test_summary_pipeline(&runner, dir.path());
    let report = within(
        "failing task",
        pipeline.run_task(&eval_task("case-failing"), &CancellationToken::new()),
    )
    .await
    .expect("a failing test is a report");

    assert!(!report.passed);
    assert!(close(report.score, 2.0 / 3.0), "two of three tests passed");
    assert_eq!(report.attempts, 1, "the default policy runs once");
    assert_eq!(report.exec.returncode, 1);
    assert!(!report.exec.success());
    assert_eq!(report.container_id.as_deref(), Some("cid-e2e"));
    assert_eq!(report.task_id, "case-failing");
    let failure = report.failure.expect("a failing task explains itself");
    assert!(failure.contains("2/3 tests passed"), "got `{failure}`");

    // The same output scored directly through the public parser agrees with the report.
    let scored = parse_score(&report.exec, &ScoreSpec::TestSummary).expect("scorable output");
    assert!(!scored.passed);
    assert!(close(scored.score, report.score));

    let invocations = runner.invocations();
    let exec = invocations
        .iter()
        .find(|spec| {
            spec.program == "docker" && spec.args.first().map(String::as_str) == Some("exec")
        })
        .expect("the test command ran through `docker exec`");
    assert_eq!(exec.args.get(1).map(String::as_str), Some("-w"));
    assert!(exec.args.iter().any(|arg| arg == "pytest"));
    assert!(
        runner
            .command_lines()
            .iter()
            .any(|line| line.starts_with("docker rm -f cid-e2e")),
        "the container is removed before the report returns"
    );
}

/// A batch scores each task on its own exec output and aggregates pass counts.
#[tokio::test]
async fn a_batch_of_tasks_summarises_pass_count_and_mean_score() {
    let dir = TempDir::new().expect("scenario temp dir");
    let runner = recorded_runner();
    runner.push_result("docker", "run", ExecResult::ok("cid-pass\n"));
    runner.push_result("docker", "run", ExecResult::ok("cid-fail\n"));
    runner.push_result("docker", "exec", ExecResult::ok("== 3 passed in 0.1s ==\n"));
    runner.push_result(
        "docker",
        "exec",
        ExecResult::new("", "1 failed, 2 passed", 1),
    );
    let pipeline = test_summary_pipeline(&runner, dir.path());
    let reports = within(
        "batch",
        pipeline.run_all(
            &[eval_task("task-pass"), eval_task("task-fail")],
            &CancellationToken::new(),
        ),
    )
    .await
    .expect("a batch never fails on a failing task");

    assert_eq!(reports.len(), 2);
    assert!(reports[0].passed, "an all-pass summary passes");
    assert!(close(reports[0].score, 1.0));
    assert_eq!(reports[0].container_id.as_deref(), Some("cid-pass"));
    assert!(!reports[1].passed);
    assert!(close(reports[1].score, 2.0 / 3.0));
    assert_eq!(reports[1].container_id.as_deref(), Some("cid-fail"));

    let summary = EvaluationPipeline::summary(&reports);
    assert_eq!(summary.tasks, 2);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 1);
    assert!(close(summary.mean_score, f64::midpoint(1.0, 2.0 / 3.0)));

    // Half a batch is summarised the same way, so the empty/partial cases agree.
    let partial = EvaluationPipeline::summary(&reports[1..]);
    assert_eq!((partial.tasks, partial.passed, partial.failed), (1, 0, 1));
    assert!(close(partial.mean_score, 2.0 / 3.0));
}

/// A host without `docker` is a typed `Unsupported` failure raised before any command.
#[tokio::test]
async fn a_missing_docker_binary_is_unsupported_before_any_command() {
    let dir = TempDir::new().expect("scenario temp dir");
    let runner = Arc::new(FakeCommandRunner::new().with_program_missing("docker"));
    let pipeline = test_summary_pipeline(&runner, dir.path());
    let error = within(
        "missing docker",
        pipeline.run_task(&eval_task("case-no-docker"), &CancellationToken::new()),
    )
    .await
    .expect_err("a missing docker binary cannot be evaluated");

    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("docker"), "got `{message}`");
    assert!(
        runner.invocations().is_empty(),
        "availability is checked before the workspace is prepared"
    );
}

// ---------------------------------------------------------------------------
// Ledger scenarios
// ---------------------------------------------------------------------------

/// Records the ledger scenario's two calls and one stage transition.
///
/// The second call is a failed, staged call, so the ledger has per-model and
/// per-stage buckets to aggregate. The first record is returned so the caller can
/// replay its `call_id` and prove deduplication.
async fn seed_ledger(ledger: &mut RunLedger) -> UsageRecord {
    let first = usage_record(
        "call-1",
        "gpt-e2e",
        UsageTokens::new(Some(10), Some(5), Some(2)),
    );
    assert!(within("record 1", ledger.record_usage(first.clone()))
        .await
        .expect("the first call persists"));
    let mut failed = usage_record(
        "call-2",
        "test-model",
        UsageTokens::new(Some(4), Some(1), None),
    )
    .with_stage_ref(EVALUATE_STAGE)
    .with_node_ref("iteration-1")
    .with_cost_usd(0.25);
    failed.model_call.status = ModelCallStatus::Failed;
    assert!(within("record 2", ledger.record_usage(failed))
        .await
        .expect("the second call persists"));
    within(
        "record stage",
        ledger.record_stage(StageRecord::new("iteration-1", json!("verify"))),
    )
    .await
    .expect("a stage transition persists");
    first
}

/// The ledger deduplicates by call id, aggregates per model and stage, persists in
/// write order, and reports an unfinished call as a delivery failure.
#[tokio::test]
async fn the_run_ledger_deduplicates_persists_and_reports_delivery() {
    let store = Arc::new(InMemoryLedgerStore::new());
    let mut ledger = RunLedger::new(LEDGER_TASK_ID).with_store(ledger_store(&store));
    within("bind", ledger.bind())
        .await
        .expect("a fresh store replays nothing");
    assert_eq!(ledger.task_id(), LEDGER_TASK_ID);
    assert!(ledger.totals().is_empty(), "nothing was charged yet");

    let first = within("seed ledger", seed_ledger(&mut ledger)).await;
    assert!(
        !within("duplicate", ledger.record_usage(first))
            .await
            .expect("a duplicate is not an error"),
        "a re-delivered call id is ignored"
    );

    let totals = ledger.totals();
    assert_eq!(totals.calls, 2, "the duplicate was not counted twice");
    assert_eq!(
        (
            totals.input_tokens,
            totals.output_tokens,
            totals.cache_hit_tokens
        ),
        (14, 6, 2)
    );
    assert!(close(totals.cost_usd, 0.25));
    assert_eq!(ledger.records().len(), 3, "two calls and one stage");

    let metrics = ledger.metrics();
    assert_eq!(metrics.totals, totals);
    assert_eq!((metrics.failed_calls, metrics.incomplete_calls), (1, 0));
    assert_eq!(metrics.by_model.len(), 2, "one bucket per model");
    assert_eq!(
        (
            bucket_calls(metrics.by_model.get("test-model")),
            bucket_calls(metrics.by_stage.get(EVALUATE_STAGE)),
            bucket_calls(metrics.by_stage.get(UNATTRIBUTED_STAGE)),
        ),
        (Some(1), Some(1), Some(1)),
        "a call outside a stage lands in the unattributed bucket"
    );

    let snapshot = ledger.snapshot();
    assert_eq!(snapshot.call_count, Some(2));
    assert_eq!(
        snapshot.tokens,
        UsageTokens::new(Some(14), Some(6), Some(2))
    );
    assert_eq!(snapshot.cost_estimate, Some(0.25));

    let events = store.events(LEDGER_TASK_ID);
    let event_types: Vec<&str> = events.iter().map(RunEvent::event_type).collect();
    assert_eq!(
        event_types,
        ["progress.usage", "progress.usage", "node.stage"]
    );
    let call_ids: Vec<&str> = events
        .iter()
        .filter_map(|event| event.as_usage().map(|record| record.call_id.as_str()))
        .collect();
    assert_eq!(call_ids, ["call-1", "call-2"], "persisted in write order");
    let stage = events[2].as_stage().expect("the last event is the stage");
    assert_eq!(stage.node_ref, "iteration-1");
    assert_eq!(stage.stage, json!("verify"));

    // Delivery accounting: a pending call blocks it until it is finished or reports.
    assert!(ledger.check_delivery().is_ok(), "nothing is in flight");
    ledger.mark_pending("call-3");
    let error = ledger
        .check_delivery()
        .expect_err("a pending call fails delivery");
    assert_eq!(error.kind(), ErrorKind::Session);
    ledger.finish_pending(ModelCallStatus::Incomplete);
    assert!(ledger.check_delivery().is_ok(), "finish_pending settles it");
    ledger.mark_pending("call-4");
    let late = usage_record("call-4", "gpt-e2e", UsageTokens::new(Some(1), None, None));
    assert!(within("record pending", ledger.record_usage(late))
        .await
        .expect("a pending call that reports usage persists"));
    assert!(
        ledger.check_delivery().is_ok(),
        "reporting usage clears the pending marker"
    );
    assert_eq!(ledger.totals().calls, 3);
}

/// An injected progress callback sees the event once; the absent-callback case is a no-op.
#[test]
fn progress_events_reach_an_injected_callback_and_ignore_the_absent_one() {
    let seen: Arc<Mutex<Vec<OptimizationProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let callback: ProgressCallback = Arc::new(move |event: &OptimizationProgressEvent| {
        sink.lock().expect("progress log lock").push(event.clone());
    });
    let event = OptimizationProgressEvent::new("optimize")
        .with_score(0.75)
        .with_improved(true)
        .with_message("iteration 2");

    emit_progress(Some(&callback), &event);
    emit_progress(None, &event);

    let seen = seen.lock().expect("progress log lock");
    assert_eq!(seen.len(), 1, "the callback ran exactly once");
    assert_eq!(seen[0], event);
    assert_eq!(seen[0].phase, "optimize");
    assert_eq!(seen[0].score, Some(0.75));
    assert_eq!(seen[0].improved, Some(true));
    assert_eq!(seen[0].message, "iteration 2");
}

// ---------------------------------------------------------------------------
// Orchestrator stream scenario
// ---------------------------------------------------------------------------

/// The event stream is ordered, ends with exactly one terminal event whose summary
/// matches the session state, and a ledger maintained alongside it records one usage
/// record per evaluated candidate.
#[tokio::test]
async fn the_event_stream_is_ordered_and_the_ledger_matches_the_evaluated_candidates() {
    let dir = TempDir::new().expect("scenario temp dir");
    let runner = ScriptedRunner::new(vec![0.5, 0.5]);
    let orchestrator = Arc::new(
        AutoHarnessOrchestrator::builder(rsi_config(dir.path()))
            .with_runner(runner.clone())
            .with_cases(vec![rsi_case()])
            .with_session_id(STREAM_SESSION_ID)
            .build()
            .expect("orchestrator builds"),
    );

    let results = within("event stream", collect_bounded(orchestrator.stream(), 32)).await;
    let events = results
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .expect("the stream completes");
    let kinds: Vec<&str> = events.iter().map(AutoHarnessEvent::kind).collect();
    assert_eq!(
        kinds,
        [
            "session_started",
            "iteration_started",
            "candidates_evaluated",
            "iteration_finished",
            "checkpoint_saved",
            "iteration_started",
            "candidates_evaluated",
            "iteration_finished",
            "checkpoint_saved",
            "harness_session_finished",
        ],
        "two flat iterations converge, then the terminal event"
    );

    let started = events.first().expect("the stream opens with an event");
    assert_eq!(
        started,
        &AutoHarnessEvent::SessionStarted {
            session_id: STREAM_SESSION_ID.to_owned(),
            max_iterations: AutoHarnessConfig::default().max_iterations,
        }
    );
    let terminal: Vec<&AutoHarnessEvent> = events
        .iter()
        .filter(|event| event.kind() == "harness_session_finished")
        .collect();
    assert_eq!(terminal.len(), 1, "exactly one terminal event");
    let last = events.last().expect("the stream ends with an event");
    let AutoHarnessEvent::SessionFinished { summary } = last else {
        panic!(
            "the stream ends with the terminal event, got `{}`",
            last.kind()
        );
    };

    let state = orchestrator.state();
    assert_eq!(summary.session_id, STREAM_SESSION_ID);
    assert_eq!(
        summary.iterations, state.iteration,
        "the summary reports the state"
    );
    assert_eq!(summary.iterations, 2);
    assert_eq!(summary.evaluated, 2, "one candidate per iteration");
    assert_eq!(state.history.len(), 2);
    assert_eq!(
        orchestrator.state().stop_reason,
        summary.stop_reason,
        "the terminal summary is the persisted stop reason"
    );
    assert!(!orchestrator.is_running());
    assert_eq!(runner.requests().len(), 2, "one harness run per iteration");
    assert_eq!(runner.requests()[1].iteration, 2);
    assert_eq!(runner.requests()[0].session_id, STREAM_SESSION_ID);

    // The orchestrator has no ledger hook, so the caller keeps one alongside the
    // stream: one usage record per candidate the iteration reports as evaluated.
    let store = Arc::new(InMemoryLedgerStore::new());
    let mut ledger = RunLedger::new(STREAM_SESSION_ID).with_store(ledger_store(&store));
    within("ledger bind", ledger.bind())
        .await
        .expect("a fresh store replays nothing");
    let iterations = within(
        "record candidates",
        record_evaluated_candidates(&mut ledger, &events),
    )
    .await;
    assert_eq!(
        iterations,
        [1, 2],
        "the stream reports the iterations in order"
    );
    assert_eq!(ledger.totals().calls, summary.evaluated);
    assert_eq!(ledger.totals().input_tokens, 240);
    assert_eq!(
        bucket_calls(ledger.metrics().by_stage.get(EVALUATE_STAGE)),
        Some(2)
    );
    assert!(
        ledger.check_delivery().is_ok(),
        "every candidate reported usage"
    );
    assert_eq!(
        store.events(STREAM_SESSION_ID).len(),
        usize::try_from(summary.evaluated).expect("the evaluated count fits in usize")
    );
}
