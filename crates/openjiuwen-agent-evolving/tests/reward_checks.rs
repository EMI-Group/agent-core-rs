//! Public-API scenarios for the reward check/judge pipeline.
//!
//! The `read` tool runs against a `tempfile` workspace, the `python` tool through
//! `FakeCommandRunner`, and the evidence agent and judge through `ScriptedModel`.
//! Nothing here spawns an interpreter, a browser or a network request.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use openjiuwen_agent_evolving::{
    default_pack, score_criterion, to_reward_signal, CheckRunner, CheckTool, CriterionKind,
    CriterionStatus, EvidenceAgent, EvidenceCriterion, EvidenceGathered, EvidenceTask, ExecResult,
    FakeCommandRunner, ModelRewardJudge, RewardCheck,
};
use openjiuwen_common::error::Error;
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::json;
use std::sync::Arc;
use tempfile::TempDir;

fn write(dir: &TempDir, name: &str, contents: &str) {
    let path = dir.path().join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("fixture directory");
    }
    std::fs::write(path, contents).expect("fixture file");
}

fn runner(dir: &TempDir) -> CheckRunner {
    CheckRunner::new(dir.path(), Arc::new(FakeCommandRunner::new()))
}

fn scripted(turns: impl IntoIterator<Item = ScriptedTurn>) -> Arc<dyn Model> {
    Arc::new(ScriptedModel::new(turns))
}

fn model_config() -> ModelRequestConfig {
    ModelRequestConfig::default()
}

fn approx(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

fn evidence_criterion() -> EvidenceCriterion {
    EvidenceCriterion::new("c1", "Parser works")
        .with_claim("the parser accepts the fixture")
        .with_claim("the parser has a regression test")
}

fn evidence_task(criterion: EvidenceCriterion) -> EvidenceTask {
    EvidenceTask::new("Fix the parser")
        .with_workspace_files(["notes.txt"])
        .with_final_reply("the parser is fixed and tested")
        .with_criteria(vec![criterion])
}

fn verdict(rows: &[(&str, &str)], summary: &str) -> String {
    let claims: Vec<_> = rows
        .iter()
        .enumerate()
        .map(
            |(index, (status, reason))| json!({"index": index, "status": status, "reason": reason}),
        )
        .collect();
    json!({"claims": claims, "summary": summary}).to_string()
}

#[tokio::test]
async fn read_range_shows_whole_lines_and_the_line_count() {
    let dir = TempDir::new().expect("temp dir");
    write(&dir, "notes.txt", "one\ntwo\nthree\nfour\n");
    let outcome = runner(&dir)
        .run(&RewardCheck::new().with_path("notes.txt").with_count(2))
        .await
        .expect("read check");
    assert!(outcome.succeeded());
    assert_eq!(
        outcome.result,
        "path: notes.txt\nlines: 1-2 of 4\n1: one\n2: two"
    );
}

#[tokio::test]
async fn read_pattern_is_case_insensitive_and_numbers_the_hits() {
    let dir = TempDir::new().expect("temp dir");
    write(&dir, "notes.txt", "Score: 3\nnothing\nSCORE: 7\n");
    let outcome = runner(&dir)
        .run(
            &RewardCheck::new()
                .with_path("notes.txt")
                .with_pattern("score"),
        )
        .await
        .expect("read check");
    assert!(outcome.succeeded());
    assert_eq!(
        outcome.result,
        "path: notes.txt\nmatches: 2\n1: Score: 3\n3: SCORE: 7"
    );
}

#[tokio::test]
async fn read_pattern_without_hits_reports_no_line_matches() {
    let dir = TempDir::new().expect("temp dir");
    write(&dir, "notes.txt", "alpha\nbeta\n");
    let outcome = runner(&dir)
        .run(
            &RewardCheck::new()
                .with_path("notes.txt")
                .with_pattern("absent"),
        )
        .await
        .expect("read check");
    assert!(outcome.succeeded());
    assert_eq!(
        outcome.result,
        "path: notes.txt\nmatches: 0\n(no line matches)"
    );
}

#[tokio::test]
async fn read_pattern_stops_after_the_hit_cap() {
    let dir = TempDir::new().expect("temp dir");
    let text: String = (1..=10).fold(String::new(), |mut text, number| {
        text.push_str(&format!("hit {number}\n"));
        text
    });
    write(&dir, "notes.txt", &text);
    let outcome = CheckRunner::new(dir.path(), Arc::new(FakeCommandRunner::new()))
        .with_max_hits(2)
        .run(
            &RewardCheck::new()
                .with_path("notes.txt")
                .with_pattern("hit"),
        )
        .await
        .expect("read check");
    assert_eq!(
        outcome.result,
        "path: notes.txt\nmatches: 2\n1: hit 1\n2: hit 2"
    );
}

#[tokio::test]
async fn a_folder_listing_is_sorted_and_relative() {
    let dir = TempDir::new().expect("temp dir");
    write(&dir, "b.txt", "b");
    write(&dir, "a.txt", "a");
    write(&dir, "sub/c.txt", "c");
    let outcome = runner(&dir)
        .run(&RewardCheck::new().with_path("."))
        .await
        .expect("folder check");
    assert!(outcome.succeeded());
    assert_eq!(
        outcome.result,
        "path: .\nfolder; files:\na.txt\nb.txt\nsub/c.txt"
    );
}

#[tokio::test]
async fn a_folder_listing_honours_the_limit() {
    let dir = TempDir::new().expect("temp dir");
    write(&dir, "a.txt", "a");
    write(&dir, "b.txt", "b");
    let outcome = CheckRunner::new(dir.path(), Arc::new(FakeCommandRunner::new()))
        .with_listing_limit(1)
        .run(&RewardCheck::new().with_path("."))
        .await
        .expect("folder check");
    assert_eq!(outcome.result, "path: .\nfolder; files:\na.txt");
}

#[tokio::test]
async fn a_path_that_leaves_the_workspace_is_rejected() {
    let dir = TempDir::new().expect("temp dir");
    let error = runner(&dir)
        .run(&RewardCheck::new().with_path("../escape.txt"))
        .await
        .expect_err("an escape is rejected before any read");
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    assert!(
        error.to_string().contains("stay inside the workspace"),
        "{error}"
    );
}

#[tokio::test]
async fn a_picture_check_is_typed_unsupported() {
    let dir = TempDir::new().expect("temp dir");
    write(&dir, "shot.png", "not really a png");
    let error = runner(&dir)
        .run(&RewardCheck::new().with_path("shot.png"))
        .await
        .expect_err("pictures need PIL");
    assert!(matches!(error, Error::Unsupported(_)), "got {error:?}");
    assert!(error.to_string().contains("PIL"), "{error}");
}

#[tokio::test]
async fn a_browser_check_names_playwright_as_the_missing_dependency() {
    let dir = TempDir::new().expect("temp dir");
    let error = runner(&dir)
        .run(
            &RewardCheck::new()
                .with_tool(CheckTool::Browser)
                .with_page("index.html"),
        )
        .await
        .expect_err("the browser tool is deferred");
    assert!(matches!(error, Error::Unsupported(_)), "got {error:?}");
    let message = error.to_string();
    assert!(message.contains("playwright"), "{message}");
    assert!(message.contains("browser"), "{message}");
}

#[tokio::test]
async fn an_unidentifiable_check_is_skipped_instead_of_run() {
    let dir = TempDir::new().expect("temp dir");
    let outcome = runner(&dir)
        .run(&RewardCheck::new())
        .await
        .expect("a skipped check is not an error");
    assert!(!outcome.ok);
    assert!(outcome.skipped.is_some());
    assert!(outcome.error.is_none());
    assert!(outcome.result.is_empty());

    let outcome = runner(&dir)
        .run(&RewardCheck::new().with_tool(CheckTool::Read))
        .await
        .expect("a read without a path is skipped");
    assert!(!outcome.ok);
    assert!(outcome.skipped.unwrap().contains("needs a \"path\""));
}

#[tokio::test]
async fn the_python_tool_runs_through_the_command_seam() {
    let dir = TempDir::new().expect("temp dir");
    let fake = Arc::new(FakeCommandRunner::new());
    for program in ["python3", "python"] {
        fake.push_result(program, "-I", ExecResult::ok("hello\n"));
    }
    let runner = CheckRunner::new(dir.path(), Arc::clone(&fake) as Arc<dyn CommandRunner>);
    let outcome = runner
        .run(&RewardCheck::new().with_code("print('hello')"))
        .await
        .expect("python check");
    assert!(outcome.succeeded(), "{outcome:?}");
    assert!(outcome.result.contains("exit_code: 0"), "{outcome:?}");
    assert!(outcome.result.contains("hello"), "{outcome:?}");
    let invocations = fake.invocations();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].args.first().map(String::as_str), Some("-I"));
}

#[tokio::test]
async fn a_failing_python_check_reports_its_exit_code() {
    let dir = TempDir::new().expect("temp dir");
    let fake = Arc::new(FakeCommandRunner::new());
    for program in ["python3", "python"] {
        fake.push_result(program, "-I", ExecResult::new("", "boom", 1));
    }
    let outcome = CheckRunner::new(dir.path(), fake as Arc<dyn CommandRunner>)
        .run(&RewardCheck::new().with_code("raise SystemExit(1)"))
        .await
        .expect("python check");
    assert!(!outcome.ok);
    assert_eq!(outcome.error.as_deref(), Some("exit code 1"));
    assert!(outcome.result.contains("exit_code: 1"), "{outcome:?}");
}

#[tokio::test]
async fn evidence_gathering_and_judging_feed_a_reward_pack() {
    let dir = TempDir::new().expect("temp dir");
    write(&dir, "notes.txt", "Score: 3\n");
    let runner = runner(&dir);
    let task = evidence_task(evidence_criterion());
    let checks = json!({
        "checks": [{
            "tool": "read",
            "path": "notes.txt",
            "pattern": "score",
            "criteria": ["c1"],
        }]
    })
    .to_string();
    let evidence = EvidenceAgent::new(
        scripted([
            ScriptedTurn::text(checks),
            ScriptedTurn::text("{\"checks\": []}"),
        ]),
        model_config(),
    );
    let gathered = evidence.gather(&task, &runner).await.expect("gather");
    assert_eq!(gathered.checks.len(), 1);
    assert!(gathered.checks[0].ok);
    assert!(gathered.checks[0].result.contains("matches: 1"));

    let judge = ModelRewardJudge::new(
        scripted([ScriptedTurn::text(verdict(
            &[
                ("met", "the fixture parses"),
                ("partial", "only one regression test exists"),
            ],
            "the parser works; the regression coverage is thin",
        ))]),
        model_config(),
    );
    let report = judge
        .judge(&task, &gathered, &task.final_reply)
        .await
        .expect("judge");
    assert_eq!(report.total, Some(75));
    assert_eq!(report.used_checks, 1);
    let criterion = report.criterion("c1").expect("judged criterion");
    assert_eq!(criterion.status, CriterionStatus::Partial);
    assert_eq!(criterion.score, Some(75));
    assert_eq!(criterion.claims.len(), 2);
    assert_eq!(criterion.claims[0].1, CriterionStatus::Met);
    assert_eq!(criterion.claims[1].1, CriterionStatus::Partial);
    assert!(report.summary.contains("regression coverage"));

    let signal = to_reward_signal(&report).expect("a scored report has a signal");
    assert_eq!(signal.dimension, "correctness");
    assert!(approx(signal.value, 0.75));
    assert_eq!(signal.source.as_deref(), Some("reward_check"));
    assert_eq!(signal.evidence.as_deref(), Some(report.summary.as_str()));

    let summary = default_pack()
        .aggregate(std::slice::from_ref(&signal))
        .expect("aggregate");
    let correctness = summary.per_dimension.get("correctness").expect("scored");
    assert!(approx(correctness.normalised, 0.75));
    assert!(approx(summary.total, 0.45));
    assert!(summary.missing.contains(&String::from("efficiency")));
    assert!(summary.missing.contains(&String::from("clarity")));
    assert!(!summary.passed);
}

#[tokio::test]
async fn a_met_and_unmet_floor_pair_is_capped_at_25() {
    assert_eq!(
        score_criterion(
            CriterionKind::Floor,
            &[CriterionStatus::Met, CriterionStatus::Unmet],
        ),
        Some(openjiuwen_agent_evolving::reward::FLOOR_FAIL_CAP)
    );
    assert_eq!(
        score_criterion(
            CriterionKind::ProactiveChallenge,
            &[CriterionStatus::Met, CriterionStatus::Unmet],
        ),
        Some(50),
        "a challenge claim has no floor cap"
    );

    let task = evidence_task(evidence_criterion());
    let judge = ModelRewardJudge::new(
        scripted([ScriptedTurn::text(verdict(
            &[
                ("met", "the fixture parses"),
                ("unmet", "no regression test exists"),
            ],
            "parsing works; the test is missing",
        ))]),
        model_config(),
    );
    let report = judge
        .judge(&task, &EvidenceGathered::default(), "")
        .await
        .expect("judge");
    assert_eq!(report.total, Some(25));
    assert_eq!(report.criterion("c1").expect("criterion").score, Some(25));
    assert!(to_reward_signal(&report).is_some());
}

#[tokio::test]
async fn an_all_unknown_criterion_contributes_no_score() {
    assert_eq!(
        score_criterion(
            CriterionKind::Floor,
            &[CriterionStatus::Unknown, CriterionStatus::Unknown],
        ),
        None
    );
    let task = evidence_task(evidence_criterion());
    let judge = ModelRewardJudge::new(
        scripted([ScriptedTurn::text(verdict(
            &[
                ("unknown", "no evidence was gathered"),
                ("unknown", "no evidence was gathered"),
            ],
            "",
        ))]),
        model_config(),
    );
    let report = judge
        .judge(&task, &EvidenceGathered::default(), "done")
        .await
        .expect("judge");
    assert_eq!(report.total, None);
    assert_eq!(
        report.criterion("c1").expect("criterion").status,
        CriterionStatus::Unknown
    );
    assert_eq!(report.criterion("c1").expect("criterion").score, None);
    assert!(to_reward_signal(&report).is_none());
}

#[tokio::test]
async fn garbage_judge_output_is_unknown_and_never_panics() {
    let task = evidence_task(evidence_criterion());
    let judge = ModelRewardJudge::new(
        scripted([ScriptedTurn::text("I think it is probably fine")]),
        model_config(),
    );
    let report = judge
        .judge(&task, &EvidenceGathered::default(), "done")
        .await
        .expect("a non-JSON verdict is reported, not raised");
    assert_eq!(report.total, None);
    let criterion = report.criterion("c1").expect("criterion");
    assert_eq!(criterion.status, CriterionStatus::Unknown);
    assert_eq!(criterion.score, None);
    assert!(criterion.claims.iter().all(|(_, status, reason)| {
        *status == CriterionStatus::Unknown && reason.contains("was not a JSON verdict")
    }));
    assert!(to_reward_signal(&report).is_none());
}

#[tokio::test]
async fn a_judge_without_criteria_is_a_validation_error() {
    let judge = ModelRewardJudge::new(
        scripted([ScriptedTurn::text(verdict(&[], ""))]),
        model_config(),
    );
    let task = EvidenceTask::new("nothing to judge");
    let error = judge
        .judge(&task, &EvidenceGathered::default(), "")
        .await
        .expect_err("no criteria to judge");
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    assert!(error.to_string().contains("at least one criterion"));
}

#[tokio::test]
async fn a_check_without_an_explicit_tool_infers_it_from_its_payload() {
    assert_eq!(
        RewardCheck::new().with_path("notes.txt").infer_tool(),
        Some(CheckTool::Read)
    );
    assert_eq!(
        RewardCheck::new().with_code("print(1)").infer_tool(),
        Some(CheckTool::Python)
    );
    assert_eq!(
        RewardCheck::new()
            .with_action(json!({"click": "#run"}))
            .infer_tool(),
        Some(CheckTool::Browser)
    );
    assert_eq!(RewardCheck::new().infer_tool(), None);
}

#[tokio::test]
async fn a_read_check_infers_read_and_still_runs() {
    let dir = TempDir::new().expect("temp dir");
    write(&dir, "notes.txt", "one\ntwo\n");
    let check = RewardCheck::new().with_path("notes.txt");
    assert_eq!(check.infer_tool(), Some(CheckTool::Read));
    let outcome = runner(&dir).run(&check).await.expect("inferred read");
    assert!(outcome.succeeded());
    assert!(outcome.result.contains("lines: 1-2 of 2"), "{outcome:?}");
}
