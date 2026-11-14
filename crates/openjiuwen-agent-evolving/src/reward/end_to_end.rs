//! End-to-end reward-check scenarios: evidence gathering, judging, and the
//! report → signal → pack round trip.
//!
//! These are the cross-module scenarios the reward-check work is verified with.
//! They live beside the modules (rather than in `tests/`) because this work owns
//! only `src/reward/`; they drive the public seams exactly as an integration test
//! would: `openjiuwen-test-support`'s `ScriptedModel` for the model, a temporary
//! workspace for the files, and a scripted command seam, so nothing touches the
//! network, the clock, or a real interpreter.
#![cfg(test)]

use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::json;

use super::{
    default_pack, score_criterion, to_reward_signal, CheckRunner, CheckTool, CriterionKind,
    CriterionStatus, EvidenceAgent, EvidenceCriterion, EvidenceGathered, EvidenceTask,
    ModelRewardJudge, RewardCheck, FLOOR_FAIL_CAP, REWARD_CHECK_DIMENSION, REWARD_CHECK_SOURCE,
    SCHEMA_VERSION,
};
use crate::evaluator_pipeline::FakeCommandRunner;

/// A criterion with two claims, both floor requirements.
fn task() -> EvidenceTask {
    EvidenceTask::new("Summarise the notes")
        .with_workspace_files(["notes.txt"])
        .with_final_reply("See notes.txt")
        .with_criteria(vec![EvidenceCriterion::new("c1", "Content")
            .with_claim("Names the score")
            .with_claim("Names the date")])
}

/// A workspace with one notes file.
fn workspace() -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("notes.txt"), "Score: 7\nDate: May\n").unwrap();
    temp
}

/// A check runner whose command seam is scripted and never spawns anything.
fn runner(root: &std::path::Path) -> CheckRunner {
    CheckRunner::new(root, Arc::new(FakeCommandRunner::new().record_only()))
}

/// The judge reply for the two-claim criterion: met then partial.
fn verdict() -> String {
    String::from(
        r#"{"claims": [{"index": 0, "status": "met", "reason": "the score is stated"},
                       {"index": 1, "status": "partial", "reason": "the month is vague"}],
            "summary": "Content is mostly there; the date is vague."}"#,
    )
}

#[tokio::test]
async fn evidence_gathering_then_judging_scores_the_work() {
    let temp = workspace();
    let scripted = ScriptedModel::new([
        ScriptedTurn::text(
            r#"{"checks": [{"tool": "read", "path": "notes.txt", "pattern": "score", "criteria": ["c1"]}]}"#,
        ),
        ScriptedTurn::text(r#"{"checks": []}"#),
        ScriptedTurn::text(verdict()),
    ]);
    let model: Arc<dyn Model> = Arc::new(scripted.clone());

    let agent = EvidenceAgent::new(Arc::clone(&model), ModelRequestConfig::default());
    let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
    assert_eq!(gathered.rounds, 2);
    assert_eq!(gathered.checks.len(), 1);
    assert_eq!(gathered.checks[0].tool, Some(CheckTool::Read));
    assert!(gathered.checks[0].usable());

    let judge = ModelRewardJudge::new(Arc::clone(&model), ModelRequestConfig::default());
    let report = judge
        .judge(&task(), &gathered, "See notes.txt")
        .await
        .unwrap();
    assert_eq!(report.total, Some(75));
    assert_eq!(report.used_checks, 1);
    assert_eq!(
        report.summary,
        "Content is mostly there; the date is vague."
    );
    assert_eq!(report.criteria[0].status, CriterionStatus::Partial);
    assert_eq!(SCHEMA_VERSION, "reward/3");
    assert_eq!(scripted.call_count(), 3);

    // The judge was shown the check output and the reply.
    let judge_call = &scripted.calls()[2];
    let user = judge_call.messages.last().unwrap().text().into_owned();
    assert!(user.contains("notes.txt"), "judge message: {user}");
    assert!(user.contains("Score: 7"), "judge message: {user}");
    assert!(user.contains("See notes.txt"), "judge message: {user}");
    assert!(user.contains("reward/3"), "judge message: {user}");
}

#[tokio::test]
async fn browser_and_picture_checks_are_typed_deferred() {
    let temp = workspace();
    let checks = runner(temp.path());

    let error = checks
        .run(&RewardCheck::new().with_page("index.html"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Unsupported(_)),
        "unexpected: {error}"
    );
    assert!(error.to_string().contains("playwright"));

    std::fs::write(temp.path().join("poster.png"), b"not a picture").unwrap();
    let error = checks
        .run(&RewardCheck::new().with_path("poster.png"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Unsupported(_)),
        "unexpected: {error}"
    );
    assert!(error.to_string().contains("PIL"));

    // The evidence loop records the deferred check instead of aborting the run,
    // so the model sees why the check could not run.
    let model = ScriptedModel::new([
        ScriptedTurn::text(r#"{"checks": [{"page": "index.html", "criteria": ["c1"]}]}"#),
        ScriptedTurn::text(r#"{"checks": []}"#),
    ]);
    let agent = EvidenceAgent::new(
        Arc::new(model) as Arc<dyn Model>,
        ModelRequestConfig::default(),
    );
    let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
    let error = gathered.checks[0].error.as_deref().unwrap();
    assert!(error.contains("playwright"), "recorded error: {error}");
    assert!(!gathered.checks[0].usable());
}

#[tokio::test]
async fn a_garbage_judge_reply_scores_nothing_and_yields_no_signal() {
    let temp = workspace();
    let model = ScriptedModel::new([
        ScriptedTurn::text(r#"{"checks": [{"tool": "read", "path": "notes.txt"}]}"#),
        ScriptedTurn::text(r#"{"checks": []}"#),
        ScriptedTurn::text("It looks quite good to me."),
    ]);
    let model: Arc<dyn Model> = Arc::new(model.clone());
    let agent = EvidenceAgent::new(Arc::clone(&model), ModelRequestConfig::default());
    let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
    let judge = ModelRewardJudge::new(Arc::clone(&model), ModelRequestConfig::default());
    let report = judge.judge(&task(), &gathered, "").await.unwrap();

    assert_eq!(report.total, None);
    assert_eq!(report.criteria[0].status, CriterionStatus::Unknown);
    assert_eq!(report.criteria[0].score, None);
    assert!(report.criteria[0]
        .claims
        .iter()
        .all(|(_, status, reason)| *status == CriterionStatus::Unknown
            && reason.contains("not a JSON verdict")));
    assert!(to_reward_signal(&report).is_none());
    // The default pack has nothing to aggregate, so the run passes nothing.
    let summary = default_pack().aggregate(&[]).unwrap();
    assert!(!summary.passed());
    assert!(summary.total() < default_pack().passing_score);
}

#[tokio::test]
async fn report_to_signal_to_pack_round_trip() {
    let temp = workspace();
    let model = ScriptedModel::new([
        ScriptedTurn::text(r#"{"checks": [{"tool": "read", "path": "notes.txt"}]}"#),
        ScriptedTurn::text(r#"{"checks": []}"#),
        ScriptedTurn::text(verdict()),
    ]);
    let model: Arc<dyn Model> = Arc::new(model);
    let agent = EvidenceAgent::new(Arc::clone(&model), ModelRequestConfig::default());
    let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
    let judge = ModelRewardJudge::new(Arc::clone(&model), ModelRequestConfig::default());
    let report = judge.judge(&task(), &gathered, "").await.unwrap();

    let signal = to_reward_signal(&report).unwrap();
    assert_eq!(signal.dimension, REWARD_CHECK_DIMENSION);
    assert_eq!(signal.dimension, "correctness");
    assert_eq!(signal.source.as_deref(), Some(REWARD_CHECK_SOURCE));
    assert_eq!(signal.evidence.as_deref(), Some(report.summary.as_str()));
    assert!((signal.value - f64::from(report.total.unwrap()) / 100.0).abs() < 1e-12);
    assert!(signal.validate().is_ok());

    let pack = default_pack();
    let summary = pack.aggregate(&[signal]).unwrap();
    assert!((summary.total() - 0.75).abs() < 1e-9);
    assert!(summary.passed());
    let dimension = summary.dimension("correctness").unwrap();
    assert!((dimension.normalised - 0.75).abs() < 1e-9);
    // The pack's other dimensions were not measured and are reported as missing.
    assert_eq!(summary.missing(), ["clarity", "efficiency"]);
}

#[tokio::test]
async fn a_floor_failure_caps_the_reported_total() {
    let temp = workspace();
    let model = ScriptedModel::new([
        ScriptedTurn::text(r#"{"checks": [{"tool": "read", "path": "notes.txt"}]}"#),
        ScriptedTurn::text(r#"{"checks": []}"#),
        ScriptedTurn::text(
            r#"{"claims": [{"index": 0, "status": "met", "reason": "stated"},
                           {"index": 1, "status": "unmet", "reason": "no date at all"}],
                "summary": "The date is missing."}"#,
        ),
    ]);
    let model: Arc<dyn Model> = Arc::new(model);
    let agent = EvidenceAgent::new(Arc::clone(&model), ModelRequestConfig::default());
    let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
    let judge = ModelRewardJudge::new(Arc::clone(&model), ModelRequestConfig::default());
    let report = judge.judge(&task(), &gathered, "").await.unwrap();

    // met + unmet would be 50, and the failed floor claim caps it at 25.
    assert_eq!(
        score_criterion(
            CriterionKind::Floor,
            &[CriterionStatus::Met, CriterionStatus::Unmet]
        ),
        Some(FLOOR_FAIL_CAP)
    );
    assert_eq!(report.criteria[0].score, Some(FLOOR_FAIL_CAP));
    assert_eq!(report.total, Some(FLOOR_FAIL_CAP));
    let signal = to_reward_signal(&report).unwrap();
    assert!((signal.value - 0.25).abs() < 1e-9);
    let summary = default_pack().aggregate(&[signal]).unwrap();
    assert!(!summary.passed());
}

#[tokio::test]
async fn an_evidence_run_that_gathers_nothing_still_judges_the_reply() {
    let temp = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new([
        ScriptedTurn::text(r#"{"checks": []}"#),
        ScriptedTurn::text(
            r#"{"claims": [{"index": 0, "status": "met", "reason": "the reply states it"},
                           {"index": 1, "status": "met", "reason": "the reply dates it"}],
                "summary": "The reply carries the content."}"#,
        ),
    ]);
    let model: Arc<dyn Model> = Arc::new(model);
    let agent = EvidenceAgent::new(Arc::clone(&model), ModelRequestConfig::default());
    let gathered: EvidenceGathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
    assert_eq!(gathered.rounds, 1);
    assert_eq!(gathered.checks, Vec::new());

    let judge = ModelRewardJudge::new(Arc::clone(&model), ModelRequestConfig::default());
    let report = judge.judge(&task(), &gathered, "").await.unwrap();
    assert_eq!(report.total, Some(100));
    assert_eq!(report.used_checks, 0);
    assert!((to_reward_signal(&report).unwrap().value - 1.0).abs() < f64::EPSILON);
}

#[test]
fn check_model_json_is_tolerant_and_names_its_tool() {
    let check: RewardCheck = serde_json::from_value(json!({
        "path": "notes.txt",
        "pattern": "score",
        "criteria": ["c1"],
        "unknown_field": true,
    }))
    .unwrap();
    assert_eq!(check.infer_tool(), Some(CheckTool::Read));
    assert_eq!(check.criteria, ["c1"]);
    assert_eq!(
        check.dedupe_key(),
        RewardCheck::new()
            .with_tool(CheckTool::Read)
            .with_path("notes.txt")
            .with_pattern("score")
            .dedupe_key()
    );
}
