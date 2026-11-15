//! Writer tests for the JSON and Markdown artifact layers.
//!
//! Every test works in a fresh `TempWorkspace`: the writers are the only I/O in
//! this module, so the fixtures are report values and the assertions re-read the
//! files they produced.

use openjiuwen_test_support::TempWorkspace;

use super::*;
use crate::skill_evaluator::report::{
    CaseSummary, DesignReport, EvaluatorScore, EvaluatorScores, FinalScore, FunctionalReport,
    SafetyFinding, SafetyReport, ScoreDimension, PIPELINE_BLOCKED_AT_E3, PIPELINE_COMPLETED,
};
use crate::skill_evaluator::{grade_for, AutoFailCode};

use std::collections::BTreeMap;

/// A completed report with one value in every member.
fn completed_report() -> EvaluationReport {
    let mut dimensions = BTreeMap::new();
    dimensions.insert(
        "d1_knowledge_ratio".to_string(),
        ScoreDimension {
            score: 7.0,
            max: 10.0,
        },
    );
    let mut safety_dimensions = BTreeMap::new();
    safety_dimensions.insert(
        "d1_harmful_instruction".to_string(),
        ScoreDimension {
            score: 26.0,
            max: 30.0,
        },
    );
    EvaluationReport {
        design: DesignReport {
            skill_name: "demo".to_string(),
            design_score: 74.0,
            design_score_max: 100.0,
            design_score_pct: 0.74,
            grade: grade_for(0.74),
            knowledge_delta: 0.74,
            ready_for_functional_testing: true,
            dimensions,
        },
        functional: FunctionalReport {
            skill_name: "demo".to_string(),
            skill_path: "skills/demo".to_string(),
            timestamp: 1_700_000_000_000,
            summary: CaseSummary {
                total: 4,
                passed: 3,
                partial: 1,
                failed: 0,
                pass_rate: 0.75,
            },
            results: Vec::new(),
            recommendations: vec!["Edge cases".to_string()],
        },
        safety: SafetyReport {
            skill_name: "demo".to_string(),
            score_pct: 0.83,
            gate_threshold: 0.8,
            blocked: false,
            auto_fail: None,
            dimensions: Some(safety_dimensions),
            findings: Vec::new(),
        },
        final_score: FinalScore {
            skill_name: "demo".to_string(),
            pipeline_status: PIPELINE_COMPLETED.to_string(),
            final_score_pct: 0.818,
            grade: grade_for(0.818),
            evaluators: EvaluatorScores {
                e1_llm_judge: Some(EvaluatorScore::weighted(0.74, 0.4)),
                e2_functional: EvaluatorScore::weighted(0.87, 0.6),
            },
            gates_passed: vec!["e3_safety".to_string()],
            recommendation: "ready for release".to_string(),
        },
        judge: None,
    }
}

/// The completed fixture with its safety gate closed.
///
/// Mirrors the blocked variant of the JSON writer tests, so the Markdown skip
/// rules are exercised on the same pipeline status.
fn blocked_report() -> EvaluationReport {
    let mut report = completed_report();
    report.safety.blocked = true;
    report.safety.auto_fail = Some(AutoFailCode::HardcodedCredentials);
    report.safety.score_pct = 0.0;
    report.safety.dimensions = None;
    report.safety.findings.push(SafetyFinding {
        dimension: "d3_data_handling".to_string(),
        severity: "critical".to_string(),
        message: "the skill body ships an API key".to_string(),
    });
    report.final_score.pipeline_status = PIPELINE_BLOCKED_AT_E3.to_string();
    report.final_score.final_score_pct = 0.0;
    report.final_score.grade = grade_for(0.0);
    report.final_score.gates_passed = Vec::new();
    report
}

fn keys(value: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = value.as_object().expect("object").keys().cloned().collect();
    keys.sort();
    keys
}

fn sorted(keys: &[&str]) -> Vec<String> {
    let mut keys: Vec<String> = keys.iter().map(|key| (*key).to_string()).collect();
    keys.sort();
    keys
}

#[tokio::test]
async fn each_writer_emits_the_python_artifact_shape() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let dir = workspace.path("evals/skill-tests/demo");
    let report = completed_report();

    let paths = report.write_artifacts(&dir).await.expect("artifacts");

    assert_eq!(paths.dir, dir);
    assert_eq!(paths.design, dir.join(DESIGN_ARTIFACT));
    assert_eq!(paths.safety, dir.join(SAFETY_ARTIFACT));
    assert_eq!(paths.functional, dir.join(FUNCTIONAL_ARTIFACT));
    assert_eq!(
        paths.final_score.as_ref(),
        Some(&dir.join(FINAL_SCORE_ARTIFACT))
    );

    let design: serde_json::Value = serde_json::from_str(
        &tokio::fs::read_to_string(&paths.design)
            .await
            .expect("design"),
    )
    .expect("design json");
    assert_eq!(
        keys(&design),
        sorted(&[
            "skill_name",
            "design_score",
            "design_score_max",
            "design_score_pct",
            "grade",
            "knowledge_delta",
            "dimensions",
            "ready_for_functional_testing",
        ])
    );
    assert_eq!(design, serde_json::to_value(&report.design).expect("value"));
    assert!((design["design_score_pct"].as_f64().expect("number") - 0.74).abs() < 1e-9);

    let safety: serde_json::Value = serde_json::from_str(
        &tokio::fs::read_to_string(&paths.safety)
            .await
            .expect("safety"),
    )
    .expect("safety json");
    assert_eq!(
        keys(&safety),
        sorted(&[
            "skill_name",
            "score_pct",
            "gate_threshold",
            "blocked",
            "auto_fail",
            "dimensions",
            "findings",
        ])
    );
    assert_eq!(safety, serde_json::to_value(&report.safety).expect("value"));

    let functional: serde_json::Value = serde_json::from_str(
        &tokio::fs::read_to_string(&paths.functional)
            .await
            .expect("functional"),
    )
    .expect("functional json");
    assert_eq!(
        keys(&functional),
        sorted(&[
            "skill_name",
            "skill_path",
            "timestamp",
            "summary",
            "results",
            "recommendations",
        ])
    );

    let final_score: serde_json::Value = serde_json::from_str(
        &tokio::fs::read_to_string(paths.final_score.as_ref().expect("final score path"))
            .await
            .expect("final score"),
    )
    .expect("final score json");
    assert_eq!(
        keys(&final_score),
        sorted(&[
            "skill_name",
            "pipeline_status",
            "final_score_pct",
            "grade",
            "evaluators",
            "gates_passed",
            "recommendation",
        ])
    );
    assert_eq!(
        final_score["evaluators"]["e1_llm_judge"]["contribution"],
        serde_json::json!(0.296)
    );
}

#[tokio::test]
async fn a_blocked_run_skips_the_final_score_file() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let dir = workspace.path("blocked");
    let mut report = completed_report();
    report.safety.blocked = true;
    report.safety.auto_fail = Some(AutoFailCode::HardcodedCredentials);
    report.safety.score_pct = 0.0;
    report.safety.dimensions = None;
    report.final_score.pipeline_status = "BLOCKED_AT_E3".to_string();
    report.final_score.final_score_pct = 0.0;

    let paths = report.write_artifacts(&dir).await.expect("artifacts");

    assert!(paths.final_score.is_none());
    assert!(!dir.join(FINAL_SCORE_ARTIFACT).exists());
    assert!(dir.join(DESIGN_ARTIFACT).exists());
    let safety: serde_json::Value = serde_json::from_str(
        &tokio::fs::read_to_string(&paths.safety)
            .await
            .expect("safety"),
    )
    .expect("safety json");
    assert_eq!(
        safety["auto_fail"],
        serde_json::json!("AF2: hardcoded credentials, API keys, or secrets")
    );
    assert_eq!(safety["dimensions"], serde_json::Value::Null);
}

#[tokio::test]
async fn writers_create_nested_directories() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let report = completed_report();
    let nested = workspace.path("a/b/c");

    let path = report.write_design_report(&nested).await.expect("write");

    assert_eq!(path, nested.join(DESIGN_ARTIFACT));
    assert!(path.is_file());
}

#[tokio::test]
async fn a_non_directory_target_is_a_validation_error() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let file = workspace.path("artifact.json");
    tokio::fs::write(&file, b"{}").await.expect("seed file");
    let report = completed_report();

    for result in [
        report.write_design_report(&file).await.map(|_| ()),
        report.write_safety_report(&file).await.map(|_| ()),
        report.write_functional_report(&file).await.map(|_| ()),
        report.write_final_score(&file).await.map(|_| ()),
        report.write_artifacts(&file).await.map(|_| ()),
    ] {
        let error = result.expect_err("not a directory");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("artifact.json"), "{error}");
        assert!(error.to_string().contains("not a directory"), "{error}");
    }
}

#[tokio::test]
async fn artifact_names_match_the_python_pipeline() {
    assert_eq!(FINAL_SCORE_ARTIFACT, "final_score.json");
    assert_eq!(DESIGN_ARTIFACT, "llm_judge_score.json");
    assert_eq!(SAFETY_ARTIFACT, "skill_safety_score.json");
    assert_eq!(FUNCTIONAL_ARTIFACT, "skill_tester_report.json");
}

/// Reads a written artifact back as text.
async fn read_text(path: &std::path::Path) -> String {
    tokio::fs::read_to_string(path)
        .await
        .expect("a written artifact must read back")
}

#[tokio::test]
async fn markdown_writers_emit_the_python_reports() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let dir = workspace.path("evals/skill-tests/demo/markdown");
    let report = completed_report();

    let paths = report
        .write_markdown_artifacts(&dir)
        .await
        .expect("markdown artifacts");

    assert_eq!(paths.dir, dir);
    assert_eq!(paths.llm_judge, dir.join(LLM_JUDGE_REPORT_ARTIFACT));
    assert_eq!(paths.safety, dir.join(SAFETY_REPORT_ARTIFACT));
    assert_eq!(
        paths.final_report.as_ref(),
        Some(&dir.join(FINAL_REPORT_ARTIFACT))
    );
    assert!(
        paths.pipeline_report.is_none(),
        "a completed run must not write a pipeline report"
    );
    assert!(!dir.join(PIPELINE_REPORT_ARTIFACT).exists());

    let llm_judge = read_text(&paths.llm_judge).await;
    assert_eq!(
        llm_judge,
        markdown::llm_judge_report(&report.design),
        "the writer must persist the design renderer verbatim"
    );
    assert!(
        llm_judge.starts_with("# Skill Evaluation Report: demo\n"),
        "{llm_judge}"
    );
    assert!(llm_judge.contains("## Dimension Scores"), "{llm_judge}");

    let safety = read_text(&paths.safety).await;
    assert_eq!(safety, markdown::skill_safety_report(&report.safety));
    assert!(
        safety.starts_with("# Safety Evaluation Report: demo\n"),
        "{safety}"
    );
    assert!(safety.contains("## Verdict"), "{safety}");

    let final_report = read_text(paths.final_report.as_ref().expect("final report")).await;
    assert_eq!(final_report, markdown::final_report(&report));
    assert!(
        final_report.starts_with("# Eval Pipeline Report: demo\n"),
        "{final_report}"
    );
    assert!(
        final_report.contains("## Evaluator Breakdown"),
        "{final_report}"
    );
    assert!(
        final_report.contains("## Key Issues to Fix"),
        "{final_report}"
    );
    assert!(
        final_report.contains("**Run timestamp**: 2023-11-14T22:13:20Z"),
        "{final_report}"
    );
}

#[tokio::test]
async fn a_blocked_run_writes_only_the_pipeline_report() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let dir = workspace.path("evals/skill-tests/demo/markdown");
    let report = blocked_report();

    let paths = report
        .write_markdown_artifacts(&dir)
        .await
        .expect("markdown artifacts");

    assert!(
        paths.final_report.is_none(),
        "a blocked run has no final report"
    );
    assert_eq!(
        paths.pipeline_report.as_ref(),
        Some(&dir.join(PIPELINE_REPORT_ARTIFACT))
    );
    assert!(!dir.join(FINAL_REPORT_ARTIFACT).exists());

    let pipeline = read_text(paths.pipeline_report.as_ref().expect("pipeline report")).await;
    assert_eq!(pipeline, markdown::pipeline_report(&report));
    assert!(
        pipeline.starts_with("# Eval Pipeline Report: demo\n"),
        "{pipeline}"
    );
    assert!(pipeline.contains("**Status**: BLOCKED_AT_E3"), "{pipeline}");
    assert!(pipeline.contains("## Key Issues to Fix"), "{pipeline}");
    assert!(!pipeline.contains("## Evaluator Breakdown"), "{pipeline}");
    assert!(paths.llm_judge.is_file() && paths.safety.is_file());
}

#[tokio::test]
async fn the_markdown_skip_rules_return_none_without_writing() {
    let workspace = TempWorkspace::new().expect("temp workspace");

    let completed = completed_report();
    let completed_dir = workspace.path("completed");
    assert!(completed
        .write_pipeline_report(&completed_dir)
        .await
        .expect("skip")
        .is_none());
    assert!(
        !completed_dir.exists(),
        "a skipped report must not create the artifact directory"
    );

    let blocked = blocked_report();
    let blocked_dir = workspace.path("blocked");
    assert!(blocked
        .write_final_report(&blocked_dir)
        .await
        .expect("skip")
        .is_none());
    assert!(
        !blocked_dir.exists(),
        "a skipped report must not create the artifact directory"
    );

    assert!(completed
        .write_final_report(&completed_dir)
        .await
        .expect("final")
        .is_some());
    assert!(blocked
        .write_pipeline_report(&blocked_dir)
        .await
        .expect("pipeline")
        .is_some());
}

#[tokio::test]
async fn markdown_writers_create_nested_directories() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let report = completed_report();
    let nested = workspace.path("a/b/c");

    let path = report.write_llm_judge_report(&nested).await.expect("write");

    assert_eq!(path, nested.join(LLM_JUDGE_REPORT_ARTIFACT));
    assert!(path.is_file());
}

#[tokio::test]
async fn markdown_writers_reject_a_non_directory_target() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let file = workspace.path("report.md");
    tokio::fs::write(&file, b"# seeded")
        .await
        .expect("seed file");

    let completed = completed_report();
    for result in [
        completed.write_llm_judge_report(&file).await.map(|_| ()),
        completed.write_safety_report_md(&file).await.map(|_| ()),
        completed.write_final_report(&file).await.map(|_| ()),
        completed.write_markdown_artifacts(&file).await.map(|_| ()),
    ] {
        let error = result.expect_err("not a directory");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("report.md"), "{error}");
        assert!(error.to_string().contains("not a directory"), "{error}");
    }

    let blocked = blocked_report();
    let error = blocked
        .write_pipeline_report(&file)
        .await
        .expect_err("not a directory");
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
}

#[tokio::test]
async fn markdown_artifact_names_match_the_python_pipeline() {
    assert_eq!(LLM_JUDGE_REPORT_ARTIFACT, "llm_judge_report.md");
    assert_eq!(SAFETY_REPORT_ARTIFACT, "skill_safety_report.md");
    assert_eq!(FINAL_REPORT_ARTIFACT, "final_report.md");
    assert_eq!(PIPELINE_REPORT_ARTIFACT, "pipeline_report.md");
}

#[tokio::test]
async fn both_artifact_layers_write_the_same_directory() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let dir = workspace.path("evals/skill-tests/demo");
    let report = completed_report();

    let json = report.write_artifacts(&dir).await.expect("json artifacts");
    let markdown_paths = report
        .write_markdown_artifacts(&dir)
        .await
        .expect("markdown artifacts");

    assert_eq!(json.dir, markdown_paths.dir);
    let written = workspace.files();
    for file in [
        DESIGN_ARTIFACT,
        SAFETY_ARTIFACT,
        FUNCTIONAL_ARTIFACT,
        FINAL_SCORE_ARTIFACT,
        LLM_JUDGE_REPORT_ARTIFACT,
        SAFETY_REPORT_ARTIFACT,
        FINAL_REPORT_ARTIFACT,
    ] {
        assert!(
            written.contains(&std::path::PathBuf::from(format!(
                "evals/skill-tests/demo/{file}"
            ))),
            "missing {file} in {written:?}"
        );
    }
}
