//! Offline suite for evaluation case records, the judge scoring contract, the
//! case runner bridge, and metrics.
//!
//! Python parity: `openjiuwen/rsi/harness_rsi/evaluator/{case_backend.py,
//! judger/scoring.py, transport.py, requirement_results.py,
//! metrics_collector.py}` plus `data_loader/grading_contract.py` (v0.1.20).
//! Every assertion is hand-computed or structural: no model, no network, no
//! clock.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_rsi::harness_rsi::evaluator::{
    classify_judge_failure, collect_metrics, is_transient_error, parse_judge_verdict,
    requirement_results, score_verdict, verdict_from_case, write_summary, AnswerRole, BehaviorSpec,
    CaseCommandBuilder, CaseExecutionBackend, CaseExecutionRequest, CaseExecutionResult,
    CaseReference, CaseRunner, Dataset, EvaluationCase, ExecutionStatus, ForbiddenBehaviorSpec,
    JudgeFailure, JudgeVerdict, PenaltyMode, ScoringContract, SweBenchCase,
};
use openjiuwen_rsi::runtime::command::{CommandOutcome, CommandSpec};

fn approx(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

fn verdict(a: f64, b: f64) -> JudgeVerdict {
    parse_judge_verdict(&format!(
        r#"{{"overall_reason":"graded","behaviors":[{{"id":"a","score":{a}}},{{"id":"b","score":{b}}}]}}"#
    ))
    .expect("verdict parses")
}

#[test]
fn cases_read_aliases_and_preserve_unknown_keys() {
    let case = EvaluationCase::from_json_value(
        2,
        &json!({
            "id": "alpha",
            "prompt": "solve it",
            "extra_field": {"keep": true},
            "assets": ["a.txt"],
            "reference": {"answer": "42", "required_behaviors": ["correct"]},
        }),
    )
    .expect("case parses");
    assert_eq!(case.case_id, "alpha");
    assert_eq!(case.input, json!("solve it"));
    assert_eq!(
        case.metadata.get("extra_field"),
        Some(&json!({"keep": true}))
    );
    assert_eq!(case.assets, vec!["a.txt".to_owned()]);
    assert_eq!(case.reference.answer, json!("42"));
    assert_eq!(case.reference.required_behaviors[0].id, "correct");
}

#[test]
fn input_alias_precedence_is_fixed() {
    let case = EvaluationCase::from_json_value(
        0,
        &json!({"input": 1, "inputs": 2, "query": 3, "prompt": 4}),
    )
    .expect("case parses");
    assert_eq!(case.input, json!(1));
    let case = EvaluationCase::from_json_value(0, &json!({"question": 9})).expect("case parses");
    assert_eq!(case.input, json!(9));
    let case =
        EvaluationCase::from_json_value(0, &json!({"name": "no input"})).expect("case parses");
    assert_eq!(case.input, Value::Null);
}

#[test]
fn swebench_binding_parses_nested_and_alias_ids() {
    let case = EvaluationCase::from_json_value(
        0,
        &json!({"swebench": {"instance": "django__django-1", "fail_to_pass": ["t1"], "future": 1}}),
    )
    .expect("case parses");
    let swebench: SweBenchCase = case.swebench.expect("swebench present");
    assert_eq!(swebench.instance_id, "django__django-1");
    assert_eq!(swebench.fail_to_pass, vec!["t1".to_owned()]);
    assert_eq!(swebench.extra, json!({"future": 1}));
    let missing = EvaluationCase::from_json_value(0, &json!({"swebench": {"repo": "x"}}));
    assert!(missing.is_err(), "instance_id is required");
}

#[test]
fn datasets_reject_duplicates_and_name_the_bad_index() {
    let duplicate = Dataset::from_json_value(&json!([{"case_id": "x"}, {"case_id": "x"}]));
    let error = duplicate.expect_err("duplicate ids rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("duplicate case id `x`"));

    let malformed = EvaluationCase::from_json_value(5, &json!({"reference": "not-an-object"}));
    let message = malformed.expect_err("malformed reference").to_string();
    assert!(message.contains("case 5"), "{message}");
    assert!(message.contains("reference must be an object"), "{message}");

    let object = Dataset::from_json_value(&json!({"dataset_id": "ds", "cases": [{"id": "a"}]}))
        .expect("dataset parses");
    assert_eq!(object.dataset_id, "ds");
    assert_eq!(object.case_ids(), vec!["a"]);
    assert_eq!(object.len(), 1);
    assert!(!object.is_empty());
}

#[test]
fn jsonl_datasets_skip_blank_lines_and_report_bad_lines() {
    let dataset =
        Dataset::from_jsonl_str("{\"id\":\"a\"}\n\n  \n{\"id\":\"b\"}\n", "lines").expect("parses");
    assert_eq!(dataset.dataset_id, "lines");
    assert_eq!(dataset.case_ids(), vec!["a", "b"]);

    let broken = Dataset::from_jsonl_str("{\"id\":\"a\"}\n{oops}\n", "lines");
    let message = broken.expect_err("bad line rejected").to_string();
    assert!(message.contains("case 1"), "{message}");
}

#[tokio::test]
async fn datasets_load_from_json_and_jsonl_paths() {
    let dir = tempfile::tempdir().expect("tempdir");
    let jsonl = dir.path().join("grades.jsonl");
    tokio::fs::write(&jsonl, "{\"id\":\"a\"}\n{\"id\":\"b\"}\n")
        .await
        .expect("write jsonl");
    let dataset = Dataset::load(&jsonl).await.expect("jsonl loads");
    assert_eq!(dataset.dataset_id, "grades");
    assert_eq!(dataset.len(), 2);

    let json_path = dir.path().join("plain.json");
    tokio::fs::write(&json_path, "[{\"id\":\"only\"}]")
        .await
        .expect("write json");
    let dataset = Dataset::load(&json_path).await.expect("json loads");
    assert_eq!(dataset.dataset_id, "plain");
    assert_eq!(dataset.case_ids(), vec!["only"]);

    let missing = Dataset::load(&dir.path().join("absent.json")).await;
    assert_eq!(
        missing.expect_err("missing file").kind(),
        ErrorKind::Session
    );
}

#[test]
fn verdict_parsing_accepts_fences_and_rejects_bad_payloads() {
    let valid = parse_judge_verdict(
        "verdict follows\n```json\n{\"status\":\"completed\",\"overall_reason\":\"ok\",\"behaviors\":[{\"id\":\"a\",\"score\":1.0}]}\n```\n",
    )
    .expect("fenced verdict parses");
    assert_eq!(valid.behaviors[0].score, 1.0);

    let non_finite = parse_judge_verdict(r#"{"behaviors":[{"id":"a","score":NaN}]}"#);
    let message = non_finite.expect_err("NaN rejected").to_string();
    assert!(message.contains("non-finite"), "{message}");
    assert!(parse_judge_verdict(r#"{"behaviors":[],"score":Infinity}"#).is_err());

    let duplicate =
        parse_judge_verdict(r#"{"behaviors":[{"id":"a","score":1.0},{"id":"a","score":0.0}]}"#);
    assert!(duplicate
        .expect_err("duplicate ids rejected")
        .to_string()
        .contains("unique"));

    let root = parse_judge_verdict("[1, 2]").expect_err("non-object root rejected");
    assert!(root.to_string().contains("must be an object"), "{root}");

    let status = parse_judge_verdict(r#"{"status":"unavailable","behaviors":[]}"#);
    assert!(status
        .expect_err("unknown status rejected")
        .to_string()
        .contains("unavailable"));

    let malformed = parse_judge_verdict(r#"{"behaviors": [{"id": "#);
    assert_eq!(
        malformed.expect_err("truncated").kind(),
        ErrorKind::Validation
    );

    let prose = parse_judge_verdict("no verdict here");
    assert!(prose.is_err());
}

#[test]
fn weighted_mean_and_penalties_are_hand_computable() {
    // base = (1.0*3 + 0.5*1 + 0.0*1) / 5 = 3.5 / 5 = 0.7
    let contract = ScoringContract {
        required: vec![
            BehaviorSpec {
                id: "a".to_owned(),
                weight: 3.0,
            },
            BehaviorSpec::new("b"),
            BehaviorSpec::new("c"),
        ],
        forbidden: vec![ForbiddenBehaviorSpec::new("leak")],
        ..ScoringContract::default()
    };
    let graded = verdict(1.0, 0.5);
    let clean = score_verdict(&contract, &graded);
    assert!(approx(clean.score, 0.7));
    assert!(!clean.passed, "0.7 is below the 0.8 threshold");
    assert_eq!(clean.metadata.get("base_score"), Some(&json!(0.7)));

    // Ceiling: min(0.7, 1.0 - 0.3) = 0.7; subtract: max(0.0, 0.7 - 0.3) = 0.4
    let triggered = parse_judge_verdict(
        r#"{"behaviors":[{"id":"a","score":1.0},{"id":"b","score":0.5}],"forbidden_hits":[{"id":"leak","triggered":true}]}"#,
    )
    .expect("verdict parses");
    let ceiling = score_verdict(&contract, &triggered);
    assert!(approx(ceiling.score, 0.7));
    let mut subtract_contract = contract.clone();
    subtract_contract.penalty_mode = PenaltyMode::Subtract;
    let subtract = score_verdict(&subtract_contract, &triggered);
    assert!(approx(subtract.score, 0.4));
    assert_eq!(
        subtract.metadata.get("penalty_mode"),
        Some(&json!("subtract"))
    );
    assert_eq!(
        subtract.metadata.get("triggered_forbidden_behaviors"),
        Some(&json!(["leak"]))
    );
}

#[test]
fn threshold_boundary_is_inclusive_and_missing_behaviors_score_zero() {
    let contract = ScoringContract {
        required: vec![BehaviorSpec::new("a"), BehaviorSpec::new("b")],
        success_threshold: 0.5,
        ..ScoringContract::default()
    };
    // b is absent from the verdict: (1.0 + 0.0) / 2 = 0.5
    let exact = verdict(1.0, 0.0);
    let graded = score_verdict(&contract, &exact);
    assert!(approx(graded.score, 0.5));
    assert!(graded.passed, "the threshold is inclusive");

    let mut stricter = contract.clone();
    stricter.success_threshold = 0.500_000_1;
    assert!(!score_verdict(&stricter, &exact).passed);
}

#[test]
fn binary_mode_collapses_the_score_but_keeps_the_raw_value() {
    let contract = ScoringContract {
        required: vec![BehaviorSpec::new("a")],
        binary: true,
        success_threshold: 0.8,
        ..ScoringContract::default()
    };
    let passed = score_verdict(&contract, &verdict(0.9, 0.0));
    assert!(passed.passed);
    assert!(approx(passed.score, 1.0));
    assert_eq!(passed.metadata.get("raw_score"), Some(&json!(0.9)));

    let failed = score_verdict(&contract, &verdict(0.4, 0.0));
    assert!(!failed.passed);
    assert!(approx(failed.score, 0.0));
    assert_eq!(failed.metadata.get("raw_score"), Some(&json!(0.4)));
}

#[test]
fn empty_contract_scores_zero() {
    let empty = score_verdict(&ScoringContract::default(), &verdict(1.0, 1.0));
    assert!(approx(empty.score, 0.0));
    assert!(!empty.passed);
}

#[test]
fn contracts_expand_rubric_and_reference_answer() {
    let reference = CaseReference {
        answer: json!("42"),
        rubric: json!(["state the result", "show the work"]),
        required_behaviors: vec![BehaviorSpec::new("explicit")],
        forbidden_behaviors: vec![ForbiddenBehaviorSpec::new("invented")],
        answer_role: AnswerRole::Criterion,
        ..CaseReference::default()
    };
    let contract = verdict_from_case(&reference, 0.9, false);
    let ids: Vec<&str> = contract
        .required
        .iter()
        .map(|spec| spec.id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["reference_answer", "explicit", "rubric_001", "rubric_002"]
    );
    assert_eq!(contract.forbidden.len(), 1);
    assert!(approx(contract.success_threshold, 0.9));
    assert!(!contract.binary);

    let referenced = CaseReference {
        answer_role: AnswerRole::Reference,
        ..reference
    };
    let ids: Vec<String> = verdict_from_case(&referenced, 0.8, false)
        .required
        .into_iter()
        .map(|spec| spec.id)
        .collect();
    assert!(!ids.contains(&"reference_answer".to_owned()));
}

#[test]
fn judge_failures_classify_into_five_kinds() {
    assert_eq!(
        classify_judge_failure(r#"{"behaviors": [{"id": "a""#),
        JudgeFailure::Truncated
    );
    assert_eq!(
        classify_judge_failure("```json\n{\"behaviors\": []}"),
        JudgeFailure::Truncated
    );
    assert_eq!(
        classify_judge_failure(r#"{"status":"unavailable"}"#),
        JudgeFailure::Unavailable
    );
    assert_eq!(
        classify_judge_failure("repair attempt returned an invalid payload"),
        JudgeFailure::RepairInvalid
    );
    assert_eq!(
        classify_judge_failure("repair model unavailable"),
        JudgeFailure::RepairUnavailable
    );
    assert_eq!(
        classify_judge_failure(r#"{"unexpected": true}"#),
        JudgeFailure::InvalidSchema
    );
    assert_eq!(JudgeFailure::Truncated.as_str(), "truncated");
    assert_eq!(JudgeFailure::InvalidSchema.as_str(), "invalid_schema");
}

#[test]
fn requirement_results_are_sorted_by_id_and_cover_both_groups() {
    let contract = ScoringContract {
        required: vec![BehaviorSpec::new("b"), BehaviorSpec::new("a")],
        forbidden: vec![ForbiddenBehaviorSpec::new("c")],
        ..ScoringContract::default()
    };
    let judged = parse_judge_verdict(
        r#"{"behaviors":[{"id":"a","score":1.0},{"id":"b","score":0.25}],"forbidden_hits":[{"id":"c","triggered":true}]}"#,
    )
    .expect("verdict parses");
    let results = requirement_results(&contract, &judged);
    let ids: Vec<&str> = results.iter().map(|item| item.id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b", "c"]);
    assert!(results[0].met);
    assert!(!results[1].met);
    assert!(!results[2].met);
    assert!(approx(results[1].score, 0.25));
    assert!(results[2].weight.abs() < f64::EPSILON);
}

#[test]
fn transient_markers_match_the_python_list() {
    for marker in [
        "Connection reset by peer",
        "HTTP 429 Too Many Requests",
        "Service Unavailable",
        "request timed out",
        "Temporary failure in name resolution",
        "incomplete chunked read",
    ] {
        assert!(is_transient_error(marker), "{marker} should be transient");
    }
    for marker in [
        "judge output must contain behaviors",
        "duplicate case id `x`",
        "invalid_schema",
    ] {
        assert!(
            !is_transient_error(marker),
            "{marker} should not be transient"
        );
    }
}

#[test]
fn metrics_average_graded_scores_only() {
    let cases = [
        ("a", "passed", Some(1.0)),
        ("b", "failed", Some(0.5)),
        ("c", "error", None),
    ];
    let mut traces = Vec::new();
    for (case_id, status, score) in cases {
        let mut trace = openjiuwen_rsi::harness_rsi::evaluator::CaseTrace {
            case_id: case_id.to_owned(),
            status: status.to_owned(),
            ..openjiuwen_rsi::harness_rsi::evaluator::CaseTrace::default()
        };
        trace.result.case_id = case_id.to_owned();
        trace.result.score = score;
        if status == "error" {
            trace.result.error = Some("invalid_schema".to_owned());
        }
        trace
            .artifacts
            .insert("kind".to_owned(), Value::String("judged".to_owned()));
        traces.push(trace);
    }
    let summary = collect_metrics(&traces, "llm_as_judge");
    assert_eq!(summary.total_cases, 3);
    assert_eq!(summary.passed_cases, 1);
    assert_eq!(summary.failed_cases, 2);
    assert_eq!(summary.average_score, Some(0.75));
    assert_eq!(summary.evaluation_method, "llm_as_judge");
    assert_eq!(summary.case_results[0].metadata.len(), 1);
    assert_eq!(
        summary.case_results[2].error.as_deref(),
        Some("invalid_schema")
    );

    let ungraded = collect_metrics(&traces[2..], "llm_as_judge");
    assert_eq!(ungraded.average_score, None);
    assert!(collect_metrics(&[], "llm_as_judge").average_score.is_none());
}

#[tokio::test]
async fn write_summary_round_trips_as_pretty_json() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("nested/summary.json");
    let summary = openjiuwen_rsi::harness_rsi::evaluator::MetricsSummary {
        total_cases: 2,
        passed_cases: 1,
        failed_cases: 1,
        average_score: Some(0.5),
        evaluation_method: "judge".to_owned(),
        ..openjiuwen_rsi::harness_rsi::evaluator::MetricsSummary::default()
    };
    write_summary(&path, &summary)
        .await
        .expect("summary writes");
    let text = std::fs::read_to_string(&path).expect("summary reads");
    assert!(text.ends_with('\n'));
    assert!(text.contains("\"total_cases\": 2"));
    let decoded: Value = serde_json::from_str(&text).expect("summary is JSON");
    assert_eq!(decoded["evaluation_method"], json!("judge"));
}

/// A backend that returns a scripted execution result.
struct ScriptedBackend {
    result: std::result::Result<CaseExecutionResult, &'static str>,
}

#[async_trait]
impl CaseExecutionBackend for ScriptedBackend {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "scripted"
    }

    async fn execute(&self, _request: CaseExecutionRequest) -> Result<CaseExecutionResult> {
        match &self.result {
            Ok(result) => Ok(result.clone()),
            Err(message) => Err(Error::tool(*message)),
        }
    }
}

fn runner_with(result: std::result::Result<CaseExecutionResult, &'static str>) -> CaseRunner {
    CaseRunner::new(Arc::new(ScriptedBackend { result }))
}

fn execution_request() -> (CaseExecutionRequest, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let case = EvaluationCase::from_json_value(
        0,
        &json!({"case_id": "c1", "reference": {"required_behaviors": ["a"]}}),
    )
    .expect("case parses");
    (
        CaseExecutionRequest {
            case,
            output_dir: dir.path().to_path_buf(),
            session_id: "session-1".to_owned(),
        },
        dir,
    )
}

#[tokio::test]
async fn case_runner_grades_a_verdict_response() {
    let (request, _dir) = execution_request();
    let response = json!({
        "status": "completed",
        "behaviors": [{"id": "a", "score": 1.0}],
    });
    let runner = runner_with(Ok(CaseExecutionResult::passed(response)));
    let trace = runner.run(request).await.expect("trace");
    assert_eq!(trace.status, "passed");
    assert_eq!(trace.result.score, Some(1.0));
    assert!(trace.result.error.is_none());
    assert!(!trace.is_infrastructure_error());
    assert_eq!(runner.backend_name(), "scripted");
}

#[tokio::test]
async fn case_runner_marks_backend_errors_as_ungraded_infrastructure() {
    let (request, _dir) = execution_request();
    let runner = runner_with(Err("connection reset by peer"));
    let trace = runner.run(request).await.expect("trace");
    assert_eq!(trace.status, "error");
    assert_eq!(trace.result.score, None);
    assert!(trace.execution.is_none());
    assert!(trace.is_infrastructure_error());
}

#[tokio::test]
async fn case_runner_marks_failed_executions_unless_retryable() {
    let (request, _dir) = execution_request();
    let runner = runner_with(Ok(CaseExecutionResult::failed("exit code 2")));
    let trace = runner.run(request).await.expect("trace");
    assert_eq!(trace.status, "failed");
    assert_eq!(trace.result.score, None);
    assert_eq!(trace.result.error.as_deref(), Some("exit code 2"));
    assert!(!trace.is_infrastructure_error());

    let (request, _dir) = execution_request();
    let runner = runner_with(Ok(CaseExecutionResult::failed("HTTP 503 from the judge")));
    let trace = runner.run(request).await.expect("trace");
    assert!(trace.is_infrastructure_error());
}

#[tokio::test]
async fn case_runner_keeps_an_unavailable_verdict_ungraded() {
    let (request, _dir) = execution_request();
    let runner = runner_with(Ok(CaseExecutionResult::passed(
        json!({"status": "unavailable"}),
    )));
    let trace = runner.run(request).await.expect("trace");
    assert_eq!(trace.status, "failed");
    assert_eq!(trace.result.score, None);
    assert_eq!(trace.result.error.as_deref(), Some("unavailable"));
    assert_eq!(trace.result.details["classification"], json!("unavailable"));
}

/// Builds the `echo <payload>` command a test backend runs.
struct JsonEcho {
    payload: Value,
}

impl CaseCommandBuilder for JsonEcho {
    fn build(&self, _case: &EvaluationCase) -> Result<CommandSpec> {
        Ok(CommandSpec::new("echo").arg(self.payload.to_string()))
    }
}

#[tokio::test]
async fn candidate_runner_maps_evolution_cases() {
    let (request, _dir) = execution_request();
    // A scripted backend that echoes a fixed verdict regardless of the case.
    let backend = Arc::new(ScriptedBackend {
        result: Ok(CaseExecutionResult::passed(json!({
            "behaviors": [{"id": "a", "score": 1.0}],
        }))),
    });
    let runner = CaseRunner::new(backend);
    let trace = runner.run(request).await.expect("trace");
    assert!(trace.result.passed);

    let candidate_runner = openjiuwen_rsi::harness_rsi::evaluator::CandidateCaseRunner::new(runner)
        .with_output_dir(std::env::temp_dir());
    assert_eq!(
        openjiuwen_rsi::evolution::Evaluator::name(&candidate_runner),
        "case_runner"
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("origin".to_owned(), json!("test"));
    let case = openjiuwen_rsi::evolution::Case {
        id: "evo-1".to_owned(),
        weight: 2.0,
        input: json!({"reference": {"required_behaviors": ["a"]}}),
        metadata,
        ..openjiuwen_rsi::evolution::Case::default()
    };
    let candidate = openjiuwen_rsi::evolution::Candidate::new("cand-1", "body");
    let report = openjiuwen_rsi::evolution::Evaluator::evaluate(
        &candidate_runner,
        &candidate,
        std::slice::from_ref(&case),
    )
    .await
    .expect("report");
    assert_eq!(report.total, 1);
    assert_eq!(report.results[0].case_id, "evo-1");
    assert_eq!(report.results[0].weight, 2.0);
    assert_eq!(report.results[0].score, Some(1.0));
}

#[tokio::test]
async fn command_backend_reads_a_fenced_response() {
    let runner = openjiuwen_rsi::runtime::command::FakeCommandRunner::new().reply(
        openjiuwen_rsi::runtime::command::FakeReply::outcome(
            "echo",
            CommandOutcome::success_with(
                "```json\n{\"response\": {\"score\": 0.5, \"passed\": false}}\n```",
            ),
        ),
    );
    let backend = openjiuwen_rsi::harness_rsi::evaluator::CommandCaseBackend::new(
        Arc::new(runner),
        Arc::new(JsonEcho { payload: json!({}) }),
    );
    assert_eq!(backend.field(), "response");
    let (request, _dir) = execution_request();
    let execution = backend.execute(request).await.expect("execution");
    assert_eq!(execution.execution_status, ExecutionStatus::Passed);
    assert_eq!(execution.response, json!({"score": 0.5, "passed": false}));
    assert_eq!(execution.metadata.get("exit_code"), Some(&json!(0)));
}
