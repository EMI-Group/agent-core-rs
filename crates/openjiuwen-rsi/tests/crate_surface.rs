//! Offline surface suite for the implemented flat re-exports of
//! `openjiuwen-rsi`.
//!
//! Guards the crate root: every surface that used to be an `Error::Unsupported`
//! stub and is now implemented resolves by its flat name and genuinely works —
//! the test constructs it and asserts the operation yields data instead of
//! `ErrorKind::Unsupported`. The three surfaces that are still deferred are
//! owned by `tests/deferred.rs`.
//!
//! Everything here is offline: commands run through `FakeCommandRunner`, git
//! runs through the same seam, and the benchmark evaluators only build argv (no
//! harness is executed).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::Arc;

use serde_json::json;

use openjiuwen_rsi::paper_opt::PaperOptConfig;
use openjiuwen_rsi::{
    collect_metrics, discover_python, parse_gates, parse_judge_verdict, score_verdict,
    write_summary, AutoHarnessOrchestrator, Candidate, CaseExecutionBackend, CaseExecutionResult,
    CaseRunner, CaseTrace, CiGateReport, CiGateRunner, CiGateSpec, CommandOutcome, CommandRunner,
    CommandSpec, ContinuousResult, ContinuousSingleHarness, Dataset, Error, ErrorKind,
    EvaluationCase, EvolutionEngine, EvolutionPolicy, FakeCommandRunner, FileLedgerStore, GateKind,
    GateOutcome, GitOperations, GitStatus, InMemoryLedgerStore, IterativeResult,
    IterativeSingleHarness, JudgeResult, JudgeVerdict, LedgerMetrics, LlmJudgeEvaluator,
    MemberOptimizationArtifact, MemberOptimizationRequest, MemberOptimizationStatus,
    MemberOptimizer, MemberOptimizerConfig, MemberTarget, MetricsSummary, ModelCall,
    ModelCallStatus, OptimizationAction, OptimizationPlan, PaperOptimizer, ProcessCommandRunner,
    PromotionPolicyMode, PuctTree, PytestOutcome, PytestReport, PythonToolchain, ReportFormat,
    RunEvent, RunLedger, RuntimeExtensionLoader, ScoringContract, Severity, SingleHarnessConfig,
    SingleHarnessState, SingleHarnessStatus, SubprocessConfig, SubprocessEvaluator,
    SubprocessResult, SweBenchConfig, SweBenchEvaluator, TerminalBenchConfig,
    TerminalBenchEvaluator, UsageRecord, UsageSnapshot, UsageTokens, UsageTotals,
};

/// Compile-time proof that `T` is nameable through the flat re-export set.
fn assert_type<T>() {}

/// A scripted runner whose every command succeeds with empty output.
fn successful_runner() -> Arc<FakeCommandRunner> {
    Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::success_with("")))
}

#[test]
fn command_seam_is_implemented() {
    let spec = CommandSpec::new("git").arg("status").arg("--porcelain");
    assert_eq!(spec.program, "git");
    assert_eq!(spec.args, ["status", "--porcelain"]);

    let outcome = CommandOutcome::success_with("ok");
    assert!(outcome.success());
    assert!(!outcome.timed_out);
    assert_eq!(outcome.code, Some(0));

    let fake = FakeCommandRunner::new().fallback(CommandOutcome::success_with("done"));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    let ran = runtime.block_on(fake.run(spec));
    assert_eq!(ran.expect("fake runner never fails").stdout, "done");
    assert_eq!(fake.call_count(), 1);
    assert_eq!(fake.command_lines(), ["git status --porcelain"]);

    let process = ProcessCommandRunner::new();
    assert_eq!(process.name(), "process");
    assert_type::<CommandSpec>();
}

#[test]
fn git_operations_run_through_the_command_seam() {
    let fake = Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::success_with("main\n")));
    let git = GitOperations::new(fake, ".");
    assert_eq!(git.workspace(), Path::new("."));
    let branch = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime")
        .block_on(git.current_branch());
    assert_eq!(branch.expect("git seam works"), Some("main".to_owned()));
    assert_eq!(GitStatus::default().dirty_files, Vec::<String>::new());
}

#[test]
fn ci_gate_runner_is_implemented() {
    let spec = CiGateSpec {
        name: "test".to_owned(),
        kind: GateKind::Command,
        command: "echo ok".to_owned(),
        commits: 0,
    };
    let runner = CiGateRunner::new(successful_runner(), ".")
        .with_gates(vec![spec])
        .with_timeout(std::time::Duration::from_secs(5));
    assert_eq!(runner.gates().len(), 1);
    assert_eq!(runner.gates()[0].kind, GateKind::Command);
    let report: CiGateReport = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime")
        .block_on(runner.run("test"))
        .expect("the scripted gate runs");
    assert!(report.passed);
    assert_eq!(report.gates.len(), 1);
    let gate: &GateOutcome = &report.gates[0];
    assert_eq!(gate.name, "test");
    assert_type::<CiGateSpec>();
}

#[test]
fn python_toolchain_and_subprocess_evaluator_are_implemented() {
    let toolchain = PythonToolchain::new(successful_runner(), "workspace")
        .with_python("python3")
        .with_report_format(ReportFormat::Json);
    assert_eq!(toolchain.python(), "python3");
    assert_eq!(toolchain.workspace(), Path::new("workspace"));
    assert!(toolchain.venv_dir().ends_with(".venv"));

    // `discover_python` returns an interpreter or nothing; it never panics.
    let discover: fn() -> Option<String> = discover_python;
    let _ = discover();

    let evaluator = SubprocessEvaluator::new(successful_runner());
    assert_eq!(evaluator.config().entrypoint, "candidate.py");
    assert_type::<SubprocessConfig>();
    assert_type::<SubprocessResult>();
    assert_type::<PytestOutcome>();
    assert_type::<PytestReport>();
}

#[test]
fn ledger_is_implemented() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut ledger =
        RunLedger::new("task-1").with_store(Arc::new(FileLedgerStore::new(dir.path())));
    assert_eq!(ledger.task_id(), "task-1");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    runtime.block_on(ledger.bind()).expect("file ledger loads");
    assert_eq!(ledger.totals().calls, 0);
    let metrics: LedgerMetrics = ledger.metrics();
    assert_eq!(metrics.totals.calls, 0);
    assert_eq!(metrics.by_model.len(), 0);
    let snapshot: UsageSnapshot = ledger.snapshot();
    assert_eq!(snapshot.call_count, Some(0));

    let mut in_memory = RunLedger::new("task-2").with_store(Arc::new(InMemoryLedgerStore::new()));
    runtime
        .block_on(in_memory.bind())
        .expect("in-memory ledger loads");
    assert_eq!(in_memory.records().len(), 0);

    assert_type::<RunEvent>();
    assert_type::<ModelCall>();
    assert_type::<ModelCallStatus>();
    assert_type::<UsageRecord>();
    assert_type::<UsageTokens>();
    assert_type::<UsageTotals>();
}

#[test]
fn case_records_and_benchmark_evaluators_are_implemented() {
    let dataset = Dataset::from_json_value(&json!([
        {"id": "case-1", "input": "say hi"},
        {"id": "case-2", "input": "say bye"},
    ]))
    .expect("a JSON array is a dataset");
    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.case_ids(), ["case-1", "case-2"]);

    let case = EvaluationCase::from_json_value(
        0,
        &json!({"id": "case-1", "input": "say hi", "custom": true}),
    )
    .expect("a case object parses");
    assert_eq!(case.case_id, "case-1");

    let swebench = SweBenchEvaluator::new(successful_runner(), SweBenchConfig::default());
    let command = swebench.run_command("django__django-11099", Path::new("predictions.jsonl"));
    assert_eq!(command.program, "python3");
    assert!(command
        .args
        .iter()
        .any(|arg| arg == "swebench.harness.run_evaluation"));

    let terminal = TerminalBenchEvaluator::new(successful_runner(), TerminalBenchConfig::default());
    let command = terminal.run_command("hello-world", Path::new("results.json"));
    assert_eq!(command.program, "terminal-bench");
    assert!(command.args.iter().any(|arg| arg == "--task-id"));

    assert_type::<CaseRunner>();
    assert_type::<CaseTrace>();
    assert_type::<CaseExecutionResult>();
    assert_type::<Arc<dyn CaseExecutionBackend>>();
}

#[test]
fn judge_scoring_and_metrics_are_implemented() {
    let verdict: JudgeVerdict = parse_judge_verdict(
        r#"{"status":"completed","overall_reason":"ok","behaviors":[{"id":"b1","score":1.0}]}"#,
    )
    .expect("a well-formed verdict parses");
    assert_eq!(verdict.behaviors.len(), 1);

    let result: JudgeResult = score_verdict(&ScoringContract::default(), &verdict);
    assert_eq!(result.score, 0.0, "an empty contract scores zero");
    assert!(!result.passed);

    let summary: MetricsSummary = collect_metrics(&[], "exact_match");
    assert_eq!(summary.total_cases, 0);
    assert_eq!(summary.passed_cases, 0);
    assert_eq!(summary.failed_cases, 0);
    assert_eq!(summary.average_score, None);
    assert_eq!(summary.evaluation_method, "exact_match");

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("summary.json");
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime")
        .block_on(write_summary(&path, &summary))
        .expect("summary is written");
    let written = std::fs::read_to_string(&path).expect("summary file exists");
    assert!(written.contains("exact_match"));
}

#[test]
fn optimisers_are_implemented() {
    let paper = PaperOptimizer::new(PaperOptConfig::default());
    assert_eq!(paper.config().max_iterations, 5);
    assert_eq!(paper.tree().len(), 0);

    let member = MemberOptimizer::new(MemberOptimizerConfig::default());
    assert_eq!(member.config().max_roles_per_run, 2);
    assert_type::<MemberOptimizationArtifact>();
    assert_type::<MemberOptimizationRequest>();
    assert_type::<MemberOptimizationStatus>();
    assert_type::<MemberTarget>();
    assert_type::<OptimizationAction>();
    assert_type::<OptimizationPlan>();
    assert_type::<Severity>();
}

#[test]
fn single_harness_surface_is_nameable() {
    let config = SingleHarnessConfig::default();
    config.validate().expect("the default config is valid");
    assert_eq!(config.mode, PromotionPolicyMode::ContinuousTwoStageV2);

    assert_type::<IterativeSingleHarness>();
    assert_type::<ContinuousSingleHarness>();
    assert_type::<IterativeResult>();
    assert_type::<ContinuousResult>();
    assert_type::<SingleHarnessState>();
    assert_type::<SingleHarnessStatus>();
}

#[test]
fn core_protocol_and_umbrella_names_still_resolve() {
    assert_type::<AutoHarnessOrchestrator>();
    assert_type::<PuctTree>();

    let candidate = Candidate::new("candidate-1", "body");
    assert_eq!(candidate.id.as_str(), "candidate-1");

    assert_eq!(EvolutionPolicy::default(), EvolutionPolicy::Full);
    assert_eq!(Error::unsupported("x").kind(), ErrorKind::Unsupported);

    let gates: fn(&str) -> Vec<CiGateSpec> = parse_gates;
    assert_eq!(gates("gates: []").len(), 0);
}

/// The three remaining stubs are the only flat names that still report
/// `Unsupported`; everything else at the crate root is implemented.
#[test]
fn only_the_documented_stubs_are_deferred() {
    for error in [
        EvolutionEngine::new().expect_err("engine is deferred"),
        LlmJudgeEvaluator::new().expect_err("judge is deferred"),
        RuntimeExtensionLoader::new().expect_err("loader is deferred"),
    ] {
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
}
