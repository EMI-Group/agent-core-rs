#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for `openjiuwen_agent_evolving::evaluator_pipeline`.
//!
//! Everything runs through [`FakeCommandRunner`]: no test spawns `docker` or
//! `git`, reads the environment, or depends on wall-clock time. The suites cover
//! the deterministic Docker/git command lines, stop-on-failure cleanup, the
//! typed missing-`docker` failure, workspace preparation (clone, checkout,
//! patch, failure), every score policy including malformed output and timeouts,
//! per-task isolation in a batch, summary arithmetic, and the serde contract.
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use openjiuwen_agent_evolving::evaluator_pipeline::{
    container_name, parse_score, sanitize_task_id, CommandRunner, DockerEnvironment,
    EvaluationPipeline, EvaluationTask, ExecResult, FakeCommandRunner, GitWorkspaceSpec,
    PipelineConfig, PipelineSummary, PreparedWorkspace, ScoreSpec, TaskReport, WorkspacePreparer,
    PATCH_FILE_NAME,
};
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// A shared fake runner.
fn runner() -> Arc<FakeCommandRunner> {
    Arc::new(FakeCommandRunner::new())
}

/// The same fake runner behind the trait object the pipeline constructors take.
fn boxed(runner: &Arc<FakeCommandRunner>) -> Arc<dyn CommandRunner> {
    runner.clone()
}

/// A runner that succeeds with empty output for anything not explicitly
/// scripted, so a test scripts only the commands it asserts on.
fn recorded_runner() -> Arc<FakeCommandRunner> {
    Arc::new(FakeCommandRunner::new().record_only())
}

/// A task with one test command and the default (exit-code) score policy.
fn task(id: &str) -> EvaluationTask {
    EvaluationTask::new(id, "make the tests pass", ["pytest", "-q"]).unwrap()
}

/// A pipeline rooted at `root` with the exit-code score policy.
fn pipeline(runner: &Arc<FakeCommandRunner>, root: &Path) -> EvaluationPipeline {
    EvaluationPipeline::new(
        PipelineConfig::new(root, "openjiuwen-eval:test").with_score_spec(ScoreSpec::ExitCode),
        boxed(runner),
    )
}

/// The commands `runner` recorded, in order.
fn command_lines(runner: &FakeCommandRunner) -> Vec<String> {
    runner.command_lines()
}

/// A report with `score`, passed when it is a full pass.
fn report(score: f64, passed: bool) -> TaskReport {
    TaskReport::new(
        "task",
        ExecResult::ok(""),
        score,
        passed,
        1,
        (!passed).then(|| "failed".to_owned()),
    )
}

// ---------------------------------------------------------------------------
// Docker environment
// ---------------------------------------------------------------------------

#[tokio::test]
async fn docker_lifecycle_commands_are_deterministic() {
    let directory = tempfile::tempdir().unwrap();
    let dockerfile = directory.path().join("Dockerfile");
    std::fs::write(&dockerfile, "FROM scratch\n").unwrap();
    let workspace = directory.path().join("workspace");
    let report_path = directory.path().join("report.json");
    let runner = runner();
    runner.push_result("docker", "build", ExecResult::ok(""));
    runner.push_result("docker", "run", ExecResult::ok("cid123\n"));
    runner.push_result("docker", "inspect", ExecResult::ok("true\n"));
    runner.push_result("docker", "exec", ExecResult::ok("done\n"));
    runner.push_result("docker", "cp", ExecResult::ok(""));
    runner.push_result("docker", "cp", ExecResult::ok(""));
    runner.push_result("docker", "rm", ExecResult::ok("cid123\n"));
    let mut environment = DockerEnvironment::new(
        boxed(&runner),
        "img:latest",
        "box",
        &workspace,
        "/workspace",
    )
    .with_env(BTreeMap::from([("TOKEN".to_owned(), "t".to_owned())]))
    .with_timeout_ms(1_234);

    assert_eq!(
        environment.ensure_available().await.unwrap(),
        Path::new("/fake/bin/docker")
    );
    assert_eq!(
        environment
            .build(directory.path(), &dockerfile, "img:latest")
            .await
            .unwrap(),
        "img:latest"
    );
    environment.start().await.unwrap();
    assert_eq!(environment.container_id(), Some("cid123"));
    assert!(environment.is_running().await.unwrap());
    let exec = environment
        .exec(&["pytest".to_owned(), "-q".to_owned()])
        .await
        .unwrap();
    assert!(exec.success());
    environment.copy_to(&workspace, "/workspace").await.unwrap();
    environment
        .copy_from("/workspace/report.json", &report_path)
        .await
        .unwrap();
    environment.stop().await.unwrap();
    assert!(!environment.has_container());
    assert_eq!(environment.timeout_ms(), 1_234);
    assert_eq!(environment.container_name(), "box");
    assert_eq!(environment.image(), "img:latest");
    assert_eq!(environment.container_workdir(), "/workspace");

    assert_eq!(
        command_lines(&runner),
        vec![
            format!(
                "docker build -t img:latest -f {} {}",
                dockerfile.display(),
                directory.path().display()
            ),
            format!(
                "docker run -d --name box -w /workspace -v {}:/workspace -e TOKEN=t img:latest sleep infinity",
                workspace.display()
            ),
            "docker inspect -f {{.State.Running}} cid123".to_owned(),
            "docker exec -w /workspace -e TOKEN=t cid123 pytest -q".to_owned(),
            format!("docker cp {} cid123:/workspace", workspace.display()),
            format!(
                "docker cp cid123:/workspace/report.json {}",
                report_path.display()
            ),
            "docker rm -f cid123".to_owned(),
        ]
    );
}

#[tokio::test]
async fn docker_rejects_a_missing_dockerfile_and_a_stopped_container() {
    let directory = tempfile::tempdir().unwrap();
    let runner = runner();
    let environment = DockerEnvironment::new(
        boxed(&runner),
        "img:latest",
        "box",
        directory.path().join("ws"),
        "/workspace",
    );
    let error = environment
        .build(
            directory.path(),
            &directory.path().join("missing.Dockerfile"),
            "img:latest",
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("Dockerfile not found"));
    let error = environment.exec(&["pytest".to_owned()]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("not running"));
    assert_eq!(command_lines(&runner).len(), 0);
}

#[tokio::test]
async fn docker_stop_forgets_an_already_removed_container() {
    let directory = tempfile::tempdir().unwrap();
    let runner = runner();
    runner.push_result("docker", "run", ExecResult::ok("cid\n"));
    runner.push_result(
        "docker",
        "rm",
        ExecResult::failure("Error: No such container: cid", 1),
    );
    let mut environment = DockerEnvironment::new(
        boxed(&runner),
        "img",
        container_name("t"),
        directory.path(),
        "/workspace",
    );
    environment.start().await.unwrap();
    environment.stop().await.unwrap();
    assert!(!environment.has_container());
    // A second stop is a no-op, not a second `docker rm`.
    environment.stop().await.unwrap();
    assert_eq!(
        command_lines(&runner)
            .iter()
            .filter(|line| line.starts_with("docker rm"))
            .count(),
        1
    );
}

// ---------------------------------------------------------------------------
// Whole-task pipeline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_failing_test_is_a_report_and_the_container_is_removed() {
    let directory = tempfile::tempdir().unwrap();
    let runner = recorded_runner();
    runner.push_result("docker", "run", ExecResult::ok("cid\n"));
    runner.push_result(
        "docker",
        "exec",
        ExecResult::new("", "== 1 failed, 2 passed in 0.1s ==", 1),
    );
    let pipeline = EvaluationPipeline::new(
        PipelineConfig::new(directory.path(), "img").with_score_spec(ScoreSpec::TestSummary),
        boxed(&runner),
    );
    let report = pipeline
        .run_task(&task("case-1"), &CancellationToken::new())
        .await
        .unwrap();

    assert!(!report.passed);
    assert_eq!(report.attempts, 1);
    assert!((report.score - 2.0 / 3.0).abs() < 1e-9);
    assert_eq!(report.container_id.as_deref(), Some("cid"));
    assert!(report.failure.unwrap().contains("tests passed"));
    assert_eq!(report.task_id, "case-1");
    assert!(command_lines(&runner)
        .last()
        .unwrap()
        .starts_with("docker rm -f cid"));
    assert_eq!(report.exec.returncode, 1);
}

#[tokio::test]
async fn missing_docker_is_reported_as_unsupported() {
    let directory = tempfile::tempdir().unwrap();
    let runner = Arc::new(FakeCommandRunner::new().with_program_missing("docker"));
    let pipeline = pipeline(&runner, directory.path());
    let error = pipeline
        .run_task(&task("case-1"), &CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("docker"));
    // Nothing ran: the workspace was never prepared, since availability is
    // checked first.
    assert_eq!(command_lines(&runner).len(), 0);
}

#[tokio::test]
async fn attempts_re_execute_a_failing_command_until_it_passes() {
    let directory = tempfile::tempdir().unwrap();
    let runner = recorded_runner();
    runner.push_result("docker", "run", ExecResult::ok("cid\n"));
    runner.push_result("docker", "exec", ExecResult::failure("red", 1));
    runner.push_result("docker", "exec", ExecResult::ok("green\n"));
    let pipeline = EvaluationPipeline::new(
        PipelineConfig::new(directory.path(), "img")
            .with_score_spec(ScoreSpec::ExitCode)
            .with_max_attempts(3),
        boxed(&runner),
    );
    let report = pipeline
        .run_task(&task("retry"), &CancellationToken::new())
        .await
        .unwrap();
    assert!(report.passed);
    assert_eq!(report.attempts, 2);
    assert!(report.failure.is_none());
    assert_eq!(
        command_lines(&runner)
            .iter()
            .filter(|line| line.contains("docker exec"))
            .count(),
        2
    );
}

#[tokio::test]
async fn the_pipeline_builds_an_image_only_when_a_dockerfile_is_configured() {
    let directory = tempfile::tempdir().unwrap();
    let dockerfile = directory.path().join("Dockerfile");
    std::fs::write(&dockerfile, "FROM scratch\n").unwrap();
    let runner = recorded_runner();
    runner.push_result("docker", "build", ExecResult::ok(""));
    runner.push_result("docker", "run", ExecResult::ok("cid\n"));
    runner.push_result("docker", "exec", ExecResult::ok(""));
    let pipeline = EvaluationPipeline::new(
        PipelineConfig::new(directory.path(), "img")
            .with_dockerfile(&dockerfile)
            .with_score_spec(ScoreSpec::ExitCode),
        boxed(&runner),
    );
    let report = pipeline
        .run_task(&task("build"), &CancellationToken::new())
        .await
        .unwrap();
    assert!(report.passed);
    assert_eq!(
        command_lines(&runner).first().unwrap(),
        &format!(
            "docker build -t img -f {} {}",
            dockerfile.display(),
            directory.path().display()
        )
    );
}

#[tokio::test]
async fn an_invalid_task_is_rejected_before_any_command_runs() {
    let directory = tempfile::tempdir().unwrap();
    let runner = recorded_runner();
    let pipeline = pipeline(&runner, directory.path());
    let invalid = EvaluationTask {
        task_id: "  ".to_owned(),
        ..task("placeholder")
    };
    let error = pipeline
        .run_task(&invalid, &CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(command_lines(&runner).len(), 0);
}

// ---------------------------------------------------------------------------
// Workspace preparation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn workspace_preparation_clones_checks_out_and_resolves_head() {
    let runner = runner();
    runner.push_result("git", "clone", ExecResult::ok("Cloning into 'repo'...\n"));
    runner.push_result("git", "checkout", ExecResult::ok(""));
    runner.push_result("git", "rev-parse", ExecResult::ok("abc123\n"));
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().join("repo");
    let preparer = WorkspacePreparer::new(boxed(&runner));
    let spec = GitWorkspaceSpec::from_url("https://example.test/repo.git").with_revision("v1.0");

    let prepared_tree = preparer
        .prepare(&workspace, &spec, &CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(prepared_tree.revision.as_deref(), Some("abc123"));
    assert!(!prepared_tree.applied_patch);
    assert_eq!(
        prepared_tree.workspace,
        std::path::absolute(&workspace).unwrap()
    );
    assert_eq!(
        command_lines(&runner),
        vec![
            format!(
                "git clone https://example.test/repo.git {}",
                prepared_tree.workspace.display()
            ),
            "git checkout v1.0".to_owned(),
            "git rev-parse HEAD".to_owned(),
        ]
    );
    let invocations = runner.invocations();
    assert_eq!(
        invocations[1].cwd.as_deref(),
        Some(prepared_tree.workspace.as_path())
    );
    assert_eq!(
        invocations[2].cwd.as_deref(),
        Some(prepared_tree.workspace.as_path())
    );
}

#[tokio::test]
async fn workspace_preparation_honours_shallow_clones_and_applies_a_patch() {
    let runner = runner();
    runner.push_result("git", "clone", ExecResult::ok(""));
    runner.push_result("git", "apply", ExecResult::ok(""));
    runner.push_result("git", "rev-parse", ExecResult::ok("def456\n"));
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().join("repo");
    let patch = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-a\n+b\n";
    let preparer = WorkspacePreparer::new(boxed(&runner));
    let spec = GitWorkspaceSpec::from_url("https://example.test/repo.git")
        .shallow()
        .with_patch(patch);

    let prepared_tree = preparer
        .prepare(&workspace, &spec, &CancellationToken::new())
        .await
        .unwrap();

    assert!(prepared_tree.applied_patch);
    assert_eq!(prepared_tree.revision.as_deref(), Some("def456"));
    let patch_path = prepared_tree.workspace.join(PATCH_FILE_NAME);
    assert!(!patch_path.exists());
    assert_eq!(
        command_lines(&runner),
        vec![
            format!(
                "git clone --depth 1 https://example.test/repo.git {}",
                prepared_tree.workspace.display()
            ),
            format!("git apply {}", patch_path.display()),
            "git rev-parse HEAD".to_owned(),
        ]
    );
}

#[tokio::test]
async fn workspace_preparation_fetches_an_existing_checkout_and_updates_submodules() {
    let runner = runner();
    runner.push_result("git", "fetch", ExecResult::ok(""));
    runner.push_result("git", "checkout", ExecResult::ok(""));
    runner.push_result("git", "submodule", ExecResult::ok(""));
    runner.push_result("git", "rev-parse", ExecResult::ok("f00\n"));
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().join("repo");
    std::fs::create_dir_all(workspace.join(".git")).unwrap();
    let preparer = WorkspacePreparer::new(boxed(&runner));
    let spec = GitWorkspaceSpec::from_url("https://example.test/repo.git")
        .with_revision("main")
        .with_submodules();

    let prepared_tree = preparer
        .prepare(&workspace, &spec, &CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(prepared_tree.revision.as_deref(), Some("f00"));
    assert_eq!(
        command_lines(&runner),
        vec![
            "git fetch --all --tags".to_owned(),
            "git checkout main".to_owned(),
            "git submodule update --init --recursive".to_owned(),
            "git rev-parse HEAD".to_owned(),
        ]
    );
}

#[tokio::test]
async fn a_failing_git_command_is_a_validation_error_naming_the_command() {
    let runner = runner();
    runner.push_result(
        "git",
        "clone",
        ExecResult::failure("fatal: repository not found", 128),
    );
    let directory = tempfile::tempdir().unwrap();
    let preparer = WorkspacePreparer::new(boxed(&runner));
    let spec = GitWorkspaceSpec::from_url("https://example.test/missing.git");

    let error = preparer
        .prepare(
            &directory.path().join("repo"),
            &spec,
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("git clone"));
    assert!(message.contains("repository not found"));
}

#[tokio::test]
async fn a_revision_without_a_checkout_is_rejected() {
    let runner = recorded_runner();
    let directory = tempfile::tempdir().unwrap();
    let preparer = WorkspacePreparer::new(boxed(&runner));
    let error = preparer
        .prepare(
            &directory.path().join("plain"),
            &GitWorkspaceSpec::default().with_revision("main"),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("not a git repository"));
}

#[tokio::test]
async fn a_plain_directory_prepares_without_a_revision() {
    let runner = Arc::new(FakeCommandRunner::new());
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().join("plain");
    let runner_ref = Arc::clone(&runner);
    let preparer = WorkspacePreparer::new(runner);
    let prepared_tree: PreparedWorkspace = preparer
        .prepare(
            &workspace,
            &GitWorkspaceSpec::default(),
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(prepared_tree.workspace.is_dir());
    assert_eq!(prepared_tree.revision, None);
    assert!(!prepared_tree.applied_patch);
    // The unscripted `git rev-parse` probe is tolerated, not an error.
    assert_eq!(command_lines(&runner_ref), vec!["git rev-parse HEAD"]);
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

#[test]
fn score_parsing_covers_every_policy_and_its_malformed_inputs() {
    let passed = ExecResult::ok("");
    let failed = ExecResult::failure("boom", 2);

    let exit = parse_score(&passed, &ScoreSpec::ExitCode).unwrap();
    assert_eq!((exit.score, exit.passed), (1.0, true));
    let exit = parse_score(&failed, &ScoreSpec::ExitCode).unwrap();
    assert_eq!((exit.score, exit.passed), (0.0, false));

    let json_score = parse_score(
        &ExecResult::ok("log line\n{\"score\": 90}\n"),
        &ScoreSpec::Json { path: None },
    )
    .unwrap();
    assert!((json_score.score - 0.9).abs() < 1e-9);
    assert!(!json_score.passed);
    let nested = parse_score(
        &ExecResult::ok("{\"result\": {\"score\": 1, \"passed\": true}}"),
        &ScoreSpec::Json {
            path: Some("result.score".to_owned()),
        },
    )
    .unwrap();
    assert!(nested.passed);
    assert!(parse_score(&ExecResult::ok("no json"), &ScoreSpec::Json { path: None }).is_err());
    assert!(parse_score(
        &ExecResult::ok("{\"score\": \"high\"}"),
        &ScoreSpec::Json { path: None }
    )
    .is_err());
    assert!(parse_score(
        &ExecResult::ok("{\"score\": 101}"),
        &ScoreSpec::Json { path: None }
    )
    .is_err());

    let summary = parse_score(
        &ExecResult::ok("2 failed, 6 passed"),
        &ScoreSpec::TestSummary,
    )
    .unwrap();
    assert!((summary.score - 0.75).abs() < 1e-9);
    assert!(!summary.passed);
    let tap = parse_score(
        &ExecResult::ok("1..2\nok 1 - one\nnot ok 2 - two\n"),
        &ScoreSpec::TestSummary,
    )
    .unwrap();
    assert_eq!(tap.score, 0.5);
    assert!(parse_score(&ExecResult::ok("quiet"), &ScoreSpec::TestSummary).is_err());

    let auto = parse_score(
        &ExecResult::ok("{\"score\": 0.25}\n2 passed"),
        &ScoreSpec::Auto,
    )
    .unwrap();
    assert_eq!(auto.score, 0.25);
    let auto = parse_score(&ExecResult::ok("2 passed"), &ScoreSpec::Auto).unwrap();
    assert_eq!(auto.score, 1.0);
    let auto = parse_score(&failed, &ScoreSpec::Auto).unwrap();
    assert_eq!(auto.score, 0.0);

    let timed_out = parse_score(&ExecResult::timeout("killed"), &ScoreSpec::Auto).unwrap();
    assert_eq!((timed_out.score, timed_out.passed), (0.0, false));
    assert!(timed_out.detail.contains("timed out"));
}

// ---------------------------------------------------------------------------
// Batching and summaries
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_all_isolates_per_task_failures() {
    let directory = tempfile::tempdir().unwrap();
    let runner = recorded_runner();
    runner.push_result("docker", "run", ExecResult::ok("first\n"));
    runner.push_result("docker", "exec", ExecResult::ok(""));
    runner.push_result(
        "git",
        "clone",
        ExecResult::failure("fatal: repository not found", 128),
    );
    let first = task("first");
    let mut second = task("second");
    second.environment_spec.insert(
        "repo_url".to_owned(),
        json!("https://example.test/missing.git"),
    );
    let pipeline = pipeline(&runner, directory.path());

    let reports = pipeline
        .run_all(&[first, second], &CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(reports.len(), 2);
    assert!(reports[0].passed);
    assert_eq!(reports[0].container_id.as_deref(), Some("first"));
    assert!(!reports[1].passed);
    assert_eq!(reports[1].score, 0.0);
    assert_eq!(reports[1].attempts, 0);
    assert!(reports[1].failure.as_deref().unwrap().contains("git clone"));
    let summary = EvaluationPipeline::summary(&reports);
    assert_eq!((summary.tasks, summary.passed, summary.failed), (2, 1, 1));
    assert!((summary.mean_score - 0.5).abs() < 1e-9);
}

#[tokio::test]
async fn run_all_stops_on_cancellation() {
    let directory = tempfile::tempdir().unwrap();
    let runner = recorded_runner();
    let pipeline = pipeline(&runner, directory.path());
    let cancel = CancellationToken::new();
    cancel.cancel();
    let error = pipeline.run_all(&[task("one")], &cancel).await.unwrap_err();
    assert!(error.is_cancelled());
    assert_eq!(command_lines(&runner).len(), 0);
}

#[test]
fn pipeline_summary_arithmetic_and_container_names() {
    let summary = PipelineSummary::from_reports(&[report(1.0, true), report(0.5, false)]);
    assert_eq!(
        summary,
        PipelineSummary {
            tasks: 2,
            passed: 1,
            failed: 1,
            mean_score: 0.75,
        }
    );
    assert_eq!(
        PipelineSummary::from_reports(&[]),
        PipelineSummary {
            tasks: 0,
            passed: 0,
            failed: 0,
            mean_score: 0.0,
        }
    );
    assert_eq!(container_name("Case 7/A"), "openjiuwen-eval-case-7-a");
    assert_eq!(sanitize_task_id(""), "task");
    let attempts = PipelineConfig::default().with_max_attempts(99);
    assert_eq!(attempts.effective_max_attempts(), 5);
    assert_eq!(attempts.with_max_attempts(0).effective_max_attempts(), 1);
}

// ---------------------------------------------------------------------------
// Serde contract
// ---------------------------------------------------------------------------

#[test]
fn config_task_report_and_exec_results_round_trip_through_json() {
    let config = PipelineConfig::new("/tmp/workspaces", "img:latest")
        .with_dockerfile("/tmp/Dockerfile")
        .with_container_workdir("/work")
        .with_env(BTreeMap::from([("A".to_owned(), "1".to_owned())]))
        .with_timeout_ms(1_500)
        .with_max_attempts(3)
        .with_score_spec(ScoreSpec::Json {
            path: Some("result.score".to_owned()),
        });
    let encoded = serde_json::to_value(&config).unwrap();
    assert_eq!(encoded["container_workdir"], json!("/work"));
    assert_eq!(encoded["max_attempts"], json!(3));
    assert_eq!(
        encoded["score_spec"],
        json!({"json": {"path": "result.score"}})
    );
    let decoded: PipelineConfig = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, config);

    // A partial config keeps the documented defaults.
    let partial: PipelineConfig = serde_json::from_value(json!({"docker_image": "x"})).unwrap();
    let defaults = PipelineConfig::default();
    assert_eq!(partial.docker_image, "x");
    assert_eq!(partial.workspace_root, defaults.workspace_root);
    assert_eq!(partial.container_workdir, "/workspace");
    assert_eq!(partial.timeout_ms, defaults.timeout_ms);
    assert_eq!(partial.max_attempts, 1);
    assert_eq!(partial.effective_max_attempts(), 1);
    assert_eq!(partial.score_spec, defaults.score_spec);

    let task = task("serde").with_skills(["alpha", "beta"]);
    let encoded = serde_json::to_value(&task).unwrap();
    assert_eq!(encoded["score_spec"], json!("exit_code"));
    assert_eq!(encoded["has_skills"], json!(true));
    let decoded: EvaluationTask = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, task);

    let exec = ExecResult::new("out\n", "err\n", 4);
    let decoded: ExecResult = serde_json::from_value(serde_json::to_value(&exec).unwrap()).unwrap();
    assert_eq!(decoded, exec);
    let defaulted: ExecResult = serde_json::from_value(json!({})).unwrap();
    assert_eq!(defaulted.returncode, -1);
    assert!(!defaulted.success());

    let report = TaskReport::failed("task", "no docker");
    let decoded: TaskReport =
        serde_json::from_value(serde_json::to_value(&report).unwrap()).unwrap();
    assert_eq!(decoded, report);
    assert_eq!(decoded.task_id(), "task");
    assert!(!decoded.is_passed());
}
