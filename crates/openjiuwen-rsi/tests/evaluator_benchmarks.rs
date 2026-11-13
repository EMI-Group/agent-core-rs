//! Offline suite for the SWE-bench and terminal-bench evaluators.
//!
//! Python parity: `openjiuwen/rsi/harness_rsi/evaluator/{swebench_runtime.py,
//! terminal_bench_runtime.py}` (v0.1.20). The harnesses themselves never run:
//! every test injects `FakeCommandRunner`, writes the report/results file the
//! harness would have written, and asserts the exact argv, classification,
//! failure literal, and adapter report.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_rsi::evolution::{Candidate, Case, Evaluator};
use openjiuwen_rsi::harness_rsi::evaluator::{
    classification_from_report, failure_reason, infrastructure_failure_reason, parse_results,
    per_instance_failure_reason, EvaluationCase, SweBenchClassification, SweBenchConfig,
    SweBenchDatasetEvaluator, SweBenchEvaluator, TerminalBenchConfig,
    TerminalBenchDatasetEvaluator, TerminalBenchEvaluator, COMMAND_EXCERPT_LIMIT,
    RESULTS_FILE_MISSING,
};
use openjiuwen_rsi::runtime::command::{CommandOutcome, FakeCommandRunner, FakeReply};

/// Builds a SWE-bench evaluator over a scripted runner.
fn swebench(
    runner: FakeCommandRunner,
    reports_dir: &std::path::Path,
    run_id: &str,
) -> SweBenchEvaluator {
    let mut config = SweBenchConfig {
        python: "python3".to_owned(),
        dataset: "princeton-nlp/SWE-bench_Verified".to_owned(),
        split: "test".to_owned(),
        run_id: run_id.to_owned(),
        ..SweBenchConfig::default()
    };
    config.reports_dir = reports_dir.to_path_buf();
    SweBenchEvaluator::new(Arc::new(runner), config)
}

/// A case document bound to one instance.
fn swebench_case(instance_id: &str, patch: &str) -> EvaluationCase {
    EvaluationCase::from_json_value(
        0,
        &json!({
            "case_id": instance_id,
            "swebench": {"instance_id": instance_id, "patch": patch},
        }),
    )
    .expect("case parses")
}

/// Writes the aggregate report the harness would have produced.
async fn write_report(dir: &std::path::Path, name: &str, report: Value) {
    tokio::fs::create_dir_all(dir).await.expect("reports dir");
    tokio::fs::write(dir.join(name), report.to_string())
        .await
        .expect("report written");
}

#[test]
fn swebench_argv_matches_the_official_harness() {
    let mut config = SweBenchConfig {
        python: "/usr/bin/python3".to_owned(),
        dataset: "ds".to_owned(),
        split: "dev".to_owned(),
        namespace: "ns".to_owned(),
        instance_image_tag: "tag".to_owned(),
        run_id: "run-1".to_owned(),
        model_name: "m".to_owned(),
        timeout_secs: 120,
        cache_level: "instance".to_owned(),
        extra_args: vec!["--force_rebuild".to_owned(), "true".to_owned()],
        ..SweBenchConfig::default()
    };
    config.reports_dir = std::path::PathBuf::from("reports");
    let evaluator = SweBenchEvaluator::new(Arc::new(FakeCommandRunner::new()), config);
    let spec = evaluator.run_command("django__django-1", std::path::Path::new("p.jsonl"));
    assert_eq!(spec.program, "/usr/bin/python3");
    assert_eq!(
        spec.args,
        vec![
            "-m",
            "swebench.harness.run_evaluation",
            "-d",
            "ds",
            "-s",
            "dev",
            "-i",
            "django__django-1",
            "-p",
            "p.jsonl",
            "--max_workers",
            "1",
            "-t",
            "120",
            "--cache_level",
            "instance",
            "--clean",
            "false",
            "-id",
            "run-1",
            "--namespace",
            "ns",
            "--instance_image_tag",
            "tag",
            "--force_rebuild",
            "true",
        ]
    );
    assert_eq!(spec.timeout, Some(Duration::from_secs(720)));
    assert_eq!(evaluator.config().model_name, "m");
}

#[test]
fn classification_covers_all_four_kinds() {
    let report = json!({
        "resolved_ids": ["r"],
        "error_ids": ["e"],
        "incomplete_ids": ["i"],
        "unresolved_ids": ["u"],
    });
    assert_eq!(
        classification_from_report(&report, "r"),
        SweBenchClassification::Resolved
    );
    assert_eq!(
        classification_from_report(&report, "e"),
        SweBenchClassification::Error
    );
    assert_eq!(
        classification_from_report(&report, "i"),
        SweBenchClassification::Incomplete
    );
    assert_eq!(
        classification_from_report(&report, "u"),
        SweBenchClassification::Unresolved
    );
    assert_eq!(
        classification_from_report(&json!({}), "unknown"),
        SweBenchClassification::Unresolved
    );
}

#[test]
fn failure_reasons_are_the_python_literals() {
    assert_eq!(
        failure_reason(SweBenchClassification::Error, Some(7), true).as_deref(),
        Some("official harness exited with code 7")
    );
    assert_eq!(
        failure_reason(SweBenchClassification::Error, Some(0), false).as_deref(),
        Some("official aggregate report is missing")
    );
    assert_eq!(
        failure_reason(SweBenchClassification::Error, Some(0), true).as_deref(),
        Some("official report classified the instance as an error")
    );
    assert_eq!(
        failure_reason(SweBenchClassification::Incomplete, None, true).as_deref(),
        Some("official report classified the instance as incomplete")
    );
    assert_eq!(
        per_instance_failure_reason(false).as_deref(),
        Some("official per-instance report is missing")
    );
    assert!(failure_reason(SweBenchClassification::Resolved, Some(0), true).is_none());
    assert_eq!(
        infrastructure_failure_reason(&["disk quota exceeded".to_owned()], "DISK QUOTA EXCEEDED")
            .as_deref(),
        Some("configured infrastructure failure matched: disk quota exceeded")
    );
}

#[tokio::test]
async fn resolved_instance_scores_one_and_writes_predictions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let reports = dir.path().join("reports");
    let runner = FakeCommandRunner::new().reply(FakeReply::outcome(
        "python3",
        CommandOutcome::success_with("evaluation finished"),
    ));
    let evaluator = swebench(runner, &reports, "run-1");
    write_report(
        &reports,
        "openjiuwen.run-1.json",
        json!({"resolved_ids": ["o__r-1"]}),
    )
    .await;

    let case = swebench_case("o__r-1", "diff --git a/x b/x");
    let outcome = evaluator
        .evaluate_case(&case, &dir.path().join("work"))
        .await
        .expect("evaluate");
    assert_eq!(outcome.classification, SweBenchClassification::Resolved);
    assert_eq!(outcome.score, Some(1.0));
    assert!(outcome.error.is_none());
    assert!(outcome.report_path.is_some());
    assert_eq!(outcome.metadata.get("exit_code"), Some(&json!(0)));

    let predictions = tokio::fs::read_to_string(dir.path().join("work/predictions.jsonl"))
        .await
        .expect("predictions written");
    assert_eq!(predictions.lines().count(), 1);
    let decoded: Value = serde_json::from_str(predictions.trim()).expect("prediction json");
    assert_eq!(decoded["instance_id"], json!("o__r-1"));
    assert_eq!(decoded["model_name_or_path"], json!("openjiuwen"));
    assert_eq!(decoded["model_patch"], json!("diff --git a/x b/x"));
}

#[tokio::test]
async fn unresolved_and_error_instances_classify_without_a_score() {
    let dir = tempfile::tempdir().expect("tempdir");
    let reports = dir.path().join("reports");
    let runner = FakeCommandRunner::new().fallback(CommandOutcome::success_with("done"));
    let evaluator = swebench(runner, &reports, "run-2");
    write_report(
        &reports,
        "openjiuwen.run-2.json",
        json!({"unresolved_ids": ["o__r-2"], "error_ids": ["o__r-3"]}),
    )
    .await;

    let unresolved = evaluator
        .evaluate_case(&swebench_case("o__r-2", "patch"), &dir.path().join("a"))
        .await
        .expect("evaluate");
    assert_eq!(
        unresolved.classification,
        SweBenchClassification::Unresolved
    );
    assert_eq!(unresolved.score, Some(0.0));
    assert!(unresolved.error.is_none());

    let errored = evaluator
        .evaluate_case(&swebench_case("o__r-3", "patch"), &dir.path().join("b"))
        .await
        .expect("evaluate");
    assert_eq!(errored.classification, SweBenchClassification::Error);
    assert_eq!(errored.score, None);
    assert_eq!(
        errored.error.as_deref(),
        Some("official report classified the instance as an error")
    );
}

#[tokio::test]
async fn harness_failures_and_missing_reports_are_reported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let reports = dir.path().join("reports");
    let failing = FakeCommandRunner::new().fallback(CommandOutcome::failure_with(2, "boom"));
    let evaluator = swebench(failing, &reports, "run-3");
    let outcome = evaluator
        .evaluate_case(&swebench_case("o__r-4", "patch"), &dir.path().join("a"))
        .await
        .expect("evaluate");
    assert_eq!(outcome.classification, SweBenchClassification::Error);
    assert_eq!(outcome.score, None);
    assert_eq!(
        outcome.error.as_deref(),
        Some("official harness exited with code 2")
    );

    let reports = dir.path().join("reports2");
    let good = FakeCommandRunner::new().fallback(CommandOutcome::success_with("nothing here"));
    let evaluator = swebench(good, &reports, "run-4");
    let outcome = evaluator
        .evaluate_case(&swebench_case("o__r-5", "patch"), &dir.path().join("b"))
        .await
        .expect("evaluate");
    assert_eq!(outcome.classification, SweBenchClassification::Error);
    assert_eq!(outcome.report_path, None);
    assert_eq!(
        outcome.error.as_deref(),
        Some("official aggregate report is missing")
    );
}

#[tokio::test]
async fn swebench_cases_without_an_instance_are_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let evaluator = swebench(
        FakeCommandRunner::new().fallback(CommandOutcome::success_with("done")),
        dir.path(),
        "run-5",
    );
    let case =
        EvaluationCase::from_json_value(0, &json!({"case_id": "plain"})).expect("case parses");
    let error = evaluator
        .evaluate_case(&case, dir.path())
        .await
        .expect_err("instance id is required");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("swebench.instance_id"));
}

#[tokio::test]
async fn swebench_dataset_adapter_scores_resolved_and_ungraded_cases() {
    let dir = tempfile::tempdir().expect("tempdir");
    let reports = dir.path().join("reports");
    let runner = FakeCommandRunner::new().fallback(CommandOutcome::success_with("done"));
    let evaluator = SweBenchDatasetEvaluator::new(swebench(runner, &reports, "run-6"));
    write_report(
        &reports,
        "openjiuwen.run-6.json",
        json!({"resolved_ids": ["good"], "error_ids": ["broken"]}),
    )
    .await;
    let cases = vec![
        Case {
            id: "good".to_owned(),
            weight: 2.0,
            input: json!({"swebench": {"instance_id": "good", "patch": "p"}}),
            ..Case::default()
        },
        Case {
            id: "plain".to_owned(),
            input: json!({"swebench": {"instance_id": "plain", "patch": "p"}}),
            ..Case::default()
        },
        Case {
            id: "broken".to_owned(),
            input: json!({"swebench": {"instance_id": "broken", "patch": "p"}}),
            ..Case::default()
        },
    ];
    let candidate = Candidate::new("candidate-1", "body");
    let report = evaluator
        .evaluate(&candidate, &cases)
        .await
        .expect("report");
    assert_eq!(report.total, 3);
    assert_eq!(report.results[0].score, Some(1.0));
    assert!(report.results[0].passed);
    assert_eq!(report.results[0].weight, 2.0);
    assert_eq!(report.results[1].score, Some(0.0));
    assert_eq!(report.results[2].score, None);
    assert!(report.results[2].error.is_some());
    assert_eq!(report.results[2].details["classification"], json!("error"));
    assert_eq!(Evaluator::name(&evaluator), "swebench");
}

#[test]
fn terminal_bench_argv_and_filter() {
    let config = TerminalBenchConfig {
        command: "tb".to_owned(),
        dataset: "ds".to_owned(),
        timeout_secs: 30,
        task_filter: Some("train/*".to_owned()),
        results_dir: std::path::PathBuf::from("out"),
        extra_args: vec!["--verbose".to_owned()],
    };
    let evaluator = TerminalBenchEvaluator::new(Arc::new(FakeCommandRunner::new()), config);
    let spec = evaluator.run_command("t1", std::path::Path::new("out/t1.json"));
    assert_eq!(spec.program, "tb");
    assert_eq!(
        spec.args,
        vec![
            "run",
            "--dataset",
            "ds",
            "--task-id",
            "t1",
            "--output-path",
            "out/t1.json",
            "--verbose",
            "--task-filter",
            "train/*",
        ]
    );
    assert!(spec.merge_stderr);
    assert_eq!(spec.timeout, Some(Duration::from_secs(30)));
    assert_eq!(evaluator.config().dataset, "ds");
}

#[test]
fn terminal_bench_results_shapes_parse() {
    let list = json!({"results": [
        {"task_id": "t1", "resolved": true, "commands": [
            {"command": "make", "cwd": "/app", "exit_code": 2,
             "stdout_excerpt": "partial", "stderr_excerpt": "boom",
             "timeout_sec": 5, "background": true}
        ]}
    ]});
    let outcome = parse_results(&list, "t1").expect("list shape");
    assert!(outcome.passed);
    assert_eq!(outcome.score, Some(1.0));
    assert_eq!(outcome.log.len(), 1);
    assert_eq!(outcome.log[0].command, "make");
    assert_eq!(outcome.log[0].exit_code, 2);
    assert_eq!(outcome.log[0].timeout_sec, 5);
    assert!(outcome.log[0].background);

    let tasks = json!({"tasks": {"t1": {"status": "failed", "score": 0.25}}});
    let outcome = parse_results(&tasks, "t1").expect("tasks shape");
    assert!(!outcome.passed);
    assert_eq!(outcome.score, Some(0.25));

    let ids = json!({"resolved_ids": ["a"], "unresolved_ids": ["t1"]});
    assert!(!parse_results(&ids, "t1").expect("id shape").passed);
    assert!(
        parse_results(&json!({"resolved_ids": ["t1"]}), "t1")
            .expect("id shape")
            .passed
    );
    assert!(parse_results(&tasks, "absent").is_err());
    assert!(parse_results(&json!(["not", "an", "object"]), "t1").is_err());
    assert!(parse_results(&json!({"unknown": 1}), "t1").is_err());
    assert!(parse_results(&json!({"tasks": {"t1": {"status": "weird"}}}), "t1").is_err());
}

#[test]
fn terminal_bench_command_excerpts_are_capped() {
    let long = "y".repeat(COMMAND_EXCERPT_LIMIT + 40);
    let value = json!({"tasks": {"t1": {"resolved": true, "commands": [
        {"command": "cat", "stdout_excerpt": long, "stderr_excerpt": "short"}
    ]}}});
    let outcome = parse_results(&value, "t1").expect("parses");
    assert_eq!(
        outcome.log[0].stdout_excerpt.chars().count(),
        COMMAND_EXCERPT_LIMIT
    );
    assert!(outcome.log[0].stdout_excerpt.ends_with("..."));
    assert_eq!(outcome.log[0].stderr_excerpt, "short");
}

#[tokio::test]
async fn terminal_bench_evaluate_case_reads_the_results_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let runner = FakeCommandRunner::new().fallback(CommandOutcome::success_with("ran"));
    let evaluator = TerminalBenchEvaluator::new(
        Arc::new(runner),
        TerminalBenchConfig {
            results_dir: dir.path().join("results"),
            ..TerminalBenchConfig::default()
        },
    );
    let case = EvaluationCase::from_json_value(
        0,
        &json!({"case_id": "t-1", "metadata": {"task_id": "tb/task-1"}}),
    )
    .expect("case parses");
    let output_dir = dir.path().join("case");
    let results_path = output_dir.join("tb-task-1.json");
    tokio::fs::create_dir_all(&output_dir)
        .await
        .expect("output dir");
    tokio::fs::write(
        &results_path,
        json!({"results": [{"task_id": "tb/task-1", "resolved": true}]}).to_string(),
    )
    .await
    .expect("results written");
    let outcome = evaluator
        .evaluate_case(&case, &output_dir)
        .await
        .expect("evaluate");
    assert_eq!(outcome.task_id, "tb/task-1");
    assert!(outcome.passed);
    assert_eq!(outcome.score, Some(1.0));
    assert!(outcome.error.is_none());
    assert_eq!(outcome.metadata.get("exit_code"), Some(&json!(0)));
}

#[tokio::test]
async fn terminal_bench_missing_results_are_a_case_error_not_an_err() {
    let dir = tempfile::tempdir().expect("tempdir");
    let evaluator = TerminalBenchEvaluator::new(
        Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::success_with("ran"))),
        TerminalBenchConfig::default(),
    );
    let case = EvaluationCase::from_json_value(0, &json!({"case_id": "t-2"})).expect("case parses");
    let outcome = evaluator
        .evaluate_case(&case, &dir.path().join("empty"))
        .await
        .expect("evaluate");
    assert!(!outcome.passed);
    assert_eq!(outcome.score, None);
    assert_eq!(outcome.error.as_deref(), Some(RESULTS_FILE_MISSING));
}

#[tokio::test]
async fn terminal_bench_dataset_adapter_maps_cases() {
    let dir = tempfile::tempdir().expect("tempdir");
    let runner = FakeCommandRunner::new().fallback(CommandOutcome::success_with("ran"));
    let evaluator = TerminalBenchDatasetEvaluator::new(TerminalBenchEvaluator::new(
        Arc::new(runner),
        TerminalBenchConfig {
            results_dir: dir.path().to_path_buf(),
            ..TerminalBenchConfig::default()
        },
    ));
    let case = Case {
        id: "task-a".to_owned(),
        weight: 1.5,
        input: json!({"case_id": "task-a"}),
        ..Case::default()
    };
    let output_dir = dir.path().join("task-a");
    tokio::fs::create_dir_all(&output_dir)
        .await
        .expect("output dir");
    tokio::fs::write(
        output_dir.join("task-a.json"),
        json!({"tasks": {"task-a": {"resolved": true, "commands": [{"command": "ls"}]}}})
            .to_string(),
    )
    .await
    .expect("results written");

    let candidate = Candidate::new("candidate-2", "body");
    let report = evaluator
        .evaluate(&candidate, &[case])
        .await
        .expect("report");
    assert_eq!(report.total, 1);
    assert_eq!(report.results[0].score, Some(1.0));
    assert_eq!(report.results[0].weight, 1.5);
    assert_eq!(report.results[0].details["commands"], json!(1));
    assert_eq!(Evaluator::name(&evaluator), "terminal_bench");
}
