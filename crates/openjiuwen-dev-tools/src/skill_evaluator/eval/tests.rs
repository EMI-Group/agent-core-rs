use std::collections::BTreeMap;
use std::sync::Mutex;

use openjiuwen_foundation::prompt::Role;
use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
use serde_json::{json, Value};

use super::*;
use crate::skill_evaluator::ScriptedCaseRunner;

/// An [`EvalAgent`] answering from a prompt table or a fallback, recording
/// every prompt it saw.
#[derive(Default)]
struct TestAgent {
    answers: BTreeMap<String, String>,
    fallback: Option<String>,
    calls: Mutex<Vec<String>>,
}

impl TestAgent {
    fn answering_all(answer: impl Into<String>) -> Self {
        Self {
            fallback: Some(answer.into()),
            ..Self::default()
        }
    }

    fn table(pairs: &[(&str, &str)]) -> Self {
        Self {
            answers: pairs
                .iter()
                .map(|(prompt, answer)| ((*prompt).to_string(), (*answer).to_string()))
                .collect(),
            ..Self::default()
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls lock").clone()
    }
}

#[async_trait]
impl EvalAgent for TestAgent {
    async fn run(&self, prompt: &str) -> Result<String> {
        self.calls
            .lock()
            .expect("calls lock")
            .push(prompt.to_string());
        self.answers
            .get(prompt)
            .or(self.fallback.as_ref())
            .cloned()
            .ok_or_else(|| Error::validation(format!("no canned answer for prompt {prompt:?}")))
    }
}

fn workspace() -> TempWorkspace {
    TempWorkspace::new().expect("temp workspace")
}

fn error_text<T: std::fmt::Debug>(result: Result<T>) -> String {
    result.expect_err("expected an error").to_string()
}

/// A JSON case object; every one carries an unknown field, which loading must
/// tolerate.
fn case_object(id: &str) -> Value {
    json!({
        "id": id,
        "name": format!("Case {id}"),
        "case_category": "functional",
        "prompt": format!("Prompt {id}"),
        "expected_output": "",
        "files": [],
        "expectations": ["contains:answer"],
        "unknown_field": {"nested": [1, 2, 3]},
    })
}

fn outcome(text: &str, passed: bool) -> AssertionOutcome {
    AssertionOutcome {
        text: text.to_string(),
        passed,
        evidence: "evidence".to_string(),
    }
}

fn sample_run() -> EvalRun {
    let cases = vec![
        CaseResult {
            id: "case-1".to_string(),
            name: "Pass".to_string(),
            status: CaseStatus::Passed,
            assertions: vec![outcome("contains:alpha", true)],
        },
        CaseResult {
            id: "case-2".to_string(),
            name: "Fail".to_string(),
            status: CaseStatus::Failed,
            assertions: vec![outcome("contains:beta", false)],
        },
    ];
    EvalRun {
        summary: summarize(&cases),
        cases,
    }
}

#[test]
fn load_cases_json_reads_all_three_shapes() {
    let array = json!([case_object("case-1"), case_object("case-2")]);
    let cases = load_cases_json(&array.to_string()).expect("json array");
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0].id, "case-1");
    assert_eq!(cases[1].prompt, "Prompt case-2");
    assert_eq!(cases[0].expectations, vec!["contains:answer"]);

    let wrapped = json!({"cases": [case_object("case-3")]});
    let cases = load_cases_json(&wrapped.to_string()).expect("wrapped array");
    assert_eq!(cases[0].id, "case-3");

    let single = case_object("case-4");
    assert_eq!(
        load_cases_json(&single.to_string()).expect("single case")[0].name,
        "Case case-4"
    );
    assert_eq!(load_cases_json("[]").expect("empty array").len(), 0);
}

#[test]
fn load_cases_json_rejects_bad_documents() {
    assert!(error_text(load_cases_json(" \n\t ")).contains("blank"));
    assert!(error_text(load_cases_json("{not json")).contains("not valid JSON"));
    assert!(error_text(load_cases_json("42")).contains("a number"));
    assert!(error_text(load_cases_json(r#"{"cases": 7}"#)).contains("`cases`"));

    let invalid = json!([case_object("case-1"), {
        "id": "case-2", "name": "No expectations", "prompt": "Do something."
    }]);
    assert!(error_text(load_cases_json(&invalid.to_string())).contains("expectation"));

    let malformed = json!([{"id": 7}]);
    assert!(error_text(load_cases_json(&malformed.to_string())).contains("index 0"));
}

#[tokio::test]
async fn load_cases_file_reads_jsonl_and_names_bad_lines() {
    let workspace = workspace();
    let text = format!("{}\n\n{}\n", case_object("case-1"), case_object("case-2"));
    workspace.write("cases.jsonl", &text).expect("write jsonl");
    let cases = load_cases_file(workspace.path("cases.jsonl"))
        .await
        .expect("load jsonl");
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[1].id, "case-2");

    workspace
        .write("broken.jsonl", "{not a case\n")
        .expect("write");
    let error = error_text(load_cases_file(workspace.path("broken.jsonl")).await);
    assert!(error.contains("line 1"), "{error}");

    let text = format!(
        "{}\n\n{}\n",
        case_object("case-1"),
        json!({"id": "case-2", "name": "No expectations", "prompt": "Do it."})
    );
    workspace.write("invalid.jsonl", text).expect("write");
    let error = error_text(load_cases_file(workspace.path("invalid.jsonl")).await);
    assert!(error.contains("line 3"), "{error}");
    assert!(error.contains("expectation"), "{error}");
}

#[tokio::test]
async fn load_cases_file_sniffs_content_and_reports_missing_files() {
    let workspace = workspace();
    // One case per line behind a `.json` extension: the document starts with
    // `{`, so the JSON parse fails and the JSONL retry has to rescue it.
    let text = format!("{}\n{}\n", case_object("case-1"), case_object("case-2"));
    workspace.write("cases.json", text).expect("write");
    let cases = load_cases_file(workspace.path("cases.json"))
        .await
        .expect("sniffed jsonl");
    assert_eq!(cases.len(), 2);

    workspace
        .write("comments.json", "# comment\n")
        .expect("write");
    assert!(error_text(load_cases_file(workspace.path("comments.json")).await).contains("line 1"));

    let error = error_text(load_cases_file(workspace.path("absent.json")).await);
    assert!(error.contains("absent.json"), "{error}");

    workspace.write("blank.jsonl", "  \n\n").expect("write");
    assert!(error_text(load_cases_file(workspace.path("blank.jsonl")).await).contains("blank"));
}

#[tokio::test]
async fn harness_scores_pass_fail_and_partial_cases() {
    let cases = [
        EvalCase::new("case-1", "Pass", "prompt-1")
            .with_expectations(["contains:alpha", "not_contains:beta"]),
        EvalCase::new("case-2", "Fail", "prompt-2").with_expectation("contains:missing"),
        EvalCase::new("case-3", "Partial", "prompt-3")
            .with_expectations(["contains:alpha", "contains:missing"]),
    ];
    let agent = Arc::new(TestAgent::table(&[
        ("prompt-1", "alpha only"),
        ("prompt-2", "gamma"),
        ("prompt-3", "alpha"),
    ]));
    let run = EvalHarness::new(agent.clone())
        .run(&cases)
        .await
        .expect("run");

    assert_eq!(run.cases.len(), 3);
    assert_eq!(run.cases[0].status, CaseStatus::Passed);
    assert_eq!(run.cases[1].status, CaseStatus::Failed);
    assert_eq!(run.cases[2].status, CaseStatus::Partial);
    assert_eq!(run.cases[0].assertions.len(), 2);
    assert_eq!(run.cases[1].name, "Fail");
    assert_eq!(run.summary, summarize(&run.cases));
    assert_eq!(
        (run.summary.total, run.summary.passed, run.summary.partial),
        (3, 1, 1)
    );
    assert_eq!(run.summary.failed, 1);
    assert!((run.summary.pass_rate - 0.5).abs() < 1e-9);
    assert_eq!(agent.calls(), ["prompt-1", "prompt-2", "prompt-3"]);
}

#[tokio::test]
async fn expected_output_only_cases_use_a_normalized_comparison() {
    let cases = [EvalCase::new("case-1", "Reference", "prompt-1")
        .with_expected_output("  alpha \n\t beta  ")];
    let run = EvalHarness::new(Arc::new(TestAgent::answering_all("alpha beta")))
        .run(&cases)
        .await
        .expect("run");
    assert_eq!(run.cases[0].status, CaseStatus::Passed);
    assert_eq!(
        run.cases[0].assertions[0].text,
        "equals_normalized:alpha beta"
    );
    assert_eq!(run.cases[0].assertions[0].evidence, "alpha beta");

    let failing =
        [EvalCase::new("case-1", "Reference", "prompt-1").with_expected_output("alpha beta")];
    let run = EvalHarness::new(Arc::new(TestAgent::answering_all("alpha gamma")))
        .run(&failing)
        .await
        .expect("run");
    assert_eq!(run.cases[0].status, CaseStatus::Failed);

    let long = "a".repeat(500);
    let cases = [EvalCase::new("case-1", "Long", "prompt").with_expected_output(&long)];
    let run = EvalHarness::new(Arc::new(TestAgent::answering_all(long)))
        .run(&cases)
        .await
        .expect("run");
    let evidence = &run.cases[0].assertions[0].evidence;
    assert_eq!(evidence.chars().count(), EVIDENCE_LIMIT);
    assert!(evidence.ends_with('…'));
}

#[tokio::test]
async fn harness_rejects_cases_without_checks_and_blank_fields() {
    let harness = EvalHarness::new(Arc::new(TestAgent::answering_all("answer")));

    let error = error_text(
        harness
            .run(&[EvalCase::new("case-1", "Bare", "prompt")])
            .await,
    );
    assert!(error.contains("case-1"), "{error}");
    assert!(error.contains("expectation"), "{error}");

    let blank_id = [EvalCase {
        id: "  ".to_string(),
        ..EvalCase::new("case-1", "Name", "prompt").with_expectation("contains:x")
    }];
    assert!(error_text(harness.run(&blank_id).await).contains("id"));

    let blank_name = [EvalCase {
        name: String::new(),
        ..EvalCase::new("case-1", "Name", "prompt").with_expectation("contains:x")
    }];
    let error = error_text(harness.run(&blank_name).await);
    assert!(error.contains("name"), "{error}");
    assert!(error.contains("case-1"), "{error}");

    let blank_prompt = [EvalCase {
        prompt: " \n ".to_string(),
        ..EvalCase::new("case-1", "Name", "prompt").with_expectation("contains:x")
    }];
    assert!(error_text(harness.run(&blank_prompt).await).contains("prompt"));
}

#[tokio::test]
async fn harness_validates_the_configured_skill_and_exposes_it() {
    let agent: Arc<dyn EvalAgent> = Arc::new(TestAgent::answering_all("answer"));
    let harness = EvalHarness::new(agent.clone()).with_skill(SkillDefinition::new(
        "demo",
        "Demo skill",
        "# Body",
    ));
    assert!(Arc::ptr_eq(harness.agent(), &agent));
    assert_eq!(
        harness.skill().map(|skill| skill.name.as_str()),
        Some("demo")
    );

    let error = error_text(
        EvalHarness::new(agent)
            .with_skill(SkillDefinition::new("", "Demo skill", "# Body"))
            .run(&[])
            .await,
    );
    assert!(error.contains("name"), "{error}");
}

#[tokio::test]
async fn case_runner_agent_delegates_to_the_runner() {
    let skill = SkillDefinition::new("demo", "Demo skill", "# Body");
    let runner = Arc::new(ScriptedCaseRunner::new().respond("query", "runner answer"));
    let agent = CaseRunnerAgent::new(runner.clone(), skill);

    assert_eq!(agent.skill().name, "demo");
    assert_eq!(runner.responses().len(), 1);
    assert_eq!(agent.run("any prompt").await.expect("run"), "runner answer");
}

#[tokio::test]
async fn model_agent_sends_a_system_and_a_user_message() {
    let model = Arc::new(ScriptedModel::text("the answer"));
    assert_eq!(
        ModelAgent::new(model.clone())
            .run("hello")
            .await
            .expect("run"),
        "the answer"
    );
    assert_eq!(model.call_count(), 1);
    let calls = model.calls();
    assert_eq!(calls[0].messages.len(), 2);
    assert_eq!(calls[0].messages[0].role, Role::System);
    assert_eq!(
        calls[0].messages[0].text().to_string(),
        DEFAULT_SYSTEM_PROMPT
    );
    assert_eq!(calls[0].messages[1].role, Role::User);
    assert_eq!(calls[0].messages[1].text().to_string(), "hello");

    let model = Arc::new(ScriptedModel::text("ok"));
    ModelAgent::new(model.clone())
        .with_system_prompt("Be terse.")
        .run("hi")
        .await
        .expect("run");
    assert_eq!(model.calls()[0].messages[0].text().to_string(), "Be terse.");
}

#[tokio::test]
async fn model_agent_rejects_empty_answers() {
    for answer in ["", "   \n\t"] {
        let error = ModelAgent::new(Arc::new(ScriptedModel::text(answer)))
            .run("a long prompt")
            .await
            .expect_err("empty answer");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
        assert!(error.to_string().contains("a long prompt"), "{error}");
        assert!(error.to_string().contains("13 characters"), "{error}");
    }
}

#[test]
fn to_json_uses_the_run_artifact_wire_keys() {
    let json = sample_run().to_json().expect("to_json");
    assert_eq!(json["cases"].as_array().map(Vec::len), Some(2));

    let first = &json["cases"][0];
    for key in [
        "case_id",
        "name",
        "status",
        "passed",
        "reason",
        "assertions",
    ] {
        assert!(first.get(key).is_some(), "missing key {key}");
    }
    assert_eq!(first["case_id"], "case-1");
    assert_eq!(first["name"], "Pass");
    assert_eq!(first["status"], "passed");
    assert_eq!(first["passed"], true);
    assert_eq!(first["reason"], "");
    assert_eq!(first["assertions"].as_array().map(Vec::len), Some(1));

    let second = &json["cases"][1];
    assert_eq!(second["status"], "failed");
    assert_eq!(second["passed"], false);
    assert_eq!(second["reason"], "contains:beta");
    assert_eq!(json["summary"]["total"], 2);
    assert_eq!(json["summary"]["passed"], 1);
    assert_eq!(json["summary"]["failed"], 1);
}

#[test]
fn to_json_string_round_trips_and_jsonl_is_line_terminated() {
    let run = sample_run();
    let text = run.to_json_string().expect("to_json_string");
    let back: Value = serde_json::from_str(&text).expect("parse");
    assert_eq!(back, run.to_json().expect("to_json"));

    let text = run.to_jsonl().expect("to_jsonl");
    assert_eq!(text.matches('\n').count(), 2);
    assert!(text.ends_with('\n'));
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    let first: Value = serde_json::from_str(lines[0]).expect("line 1");
    assert_eq!(first["case_id"], "case-1");
    let second: Value = serde_json::from_str(lines[1]).expect("line 2");
    assert_eq!(second["case_id"], "case-2");

    assert_eq!(EvalRun::default().to_jsonl().expect("empty"), "");
    let empty = EvalRun::default().to_json().expect("empty json");
    assert_eq!(empty["cases"].as_array().map(Vec::len), Some(0));
}

#[tokio::test]
async fn write_json_and_write_jsonl_create_parents() {
    let workspace = workspace();
    let run = sample_run();

    run.write_json(workspace.path("reports/nested/run.json"))
        .await
        .expect("write json");
    let text = workspace
        .read("reports/nested/run.json")
        .expect("read json");
    let value: Value = serde_json::from_str(&text).expect("parse json");
    assert_eq!(value["cases"].as_array().map(Vec::len), Some(2));

    run.write_jsonl(workspace.path("reports/nested/run.jsonl"))
        .await
        .expect("write jsonl");
    let text = workspace.read("reports/nested/run.jsonl").expect("read");
    assert_eq!(text, run.to_jsonl().expect("jsonl"));
    assert_eq!(text.lines().count(), 2);
}

#[test]
fn resolve_output_file_decides_between_directory_and_file() {
    let workspace = workspace();
    let directory = workspace.create_dir("existing.dir").expect("create dir");
    let generated = format!("{RESULT_FILE_PREFIX}fixed.txt");
    assert_eq!(
        resolve_output_file(&directory, "fixed"),
        directory.join(&generated),
        "an existing directory receives the generated name"
    );

    let missing = workspace.path("not-created");
    assert_eq!(
        resolve_output_file(&missing, "fixed"),
        missing.join(&generated),
        "an extensionless path is treated as a directory"
    );

    let trailing = PathBuf::from(format!("{}/", missing.display()));
    assert_eq!(
        resolve_output_file(&trailing, "fixed"),
        trailing.join(&generated),
        "a trailing separator marks a directory"
    );

    let explicit = workspace.path("result.txt");
    assert_eq!(resolve_output_file(&explicit, "fixed"), explicit);
}

#[tokio::test]
async fn run_query_writes_into_an_existing_directory() {
    let workspace = workspace();
    let directory = workspace.create_dir("results").expect("create dir");
    let agent = Arc::new(TestAgent::answering_all("final answer"));
    let request = QueryRequest {
        prompt: "Summarize the document.".to_string(),
        output_path: directory.clone(),
        files_base_dir: None,
        timestamp: "20240102_030405".to_string(),
        max_iterations: Some(3),
    };

    let outcome = EvalHarness::new(agent.clone())
        .run_query(&request)
        .await
        .expect("run query");
    assert_eq!(
        outcome.output_file,
        directory.join("skill_test_result_20240102_030405.txt")
    );
    assert_eq!(
        outcome.report_file,
        directory.join("skill_test_result_20240102_030405.json")
    );
    assert_eq!(outcome.answer, "final answer");
    assert_eq!(
        std::fs::read_to_string(&outcome.output_file).expect("read answer"),
        "final answer"
    );

    let text = std::fs::read_to_string(&outcome.report_file).expect("read report");
    let report: Value = serde_json::from_str(&text).expect("parse report");
    assert_eq!(report["timestamp"], "20240102_030405");
    assert_eq!(report["answer"], "final answer");
    assert_eq!(report["max_iterations"], 3);
    assert_eq!(
        report["output_file"],
        outcome.output_file.to_string_lossy().as_ref()
    );
    assert_eq!(
        report["files_base_dir"],
        directory.to_string_lossy().as_ref()
    );

    let prompt = report["prompt"].as_str().expect("prompt");
    assert!(prompt.contains("Summarize the document."), "{prompt}");
    assert!(prompt.contains("Maximum agent iterations: 3"), "{prompt}");
    assert!(
        prompt.contains(&format!("'{}'", directory.display())),
        "{prompt}"
    );
    assert_eq!(agent.calls(), vec![prompt.to_string()]);
}

#[tokio::test]
async fn run_query_resolves_other_output_shapes() {
    let workspace = workspace();
    let target = workspace.path("fresh-results");
    let request = QueryRequest {
        prompt: "Do it.".to_string(),
        output_path: target.clone(),
        files_base_dir: Some(workspace.path("inputs")),
        timestamp: "ts".to_string(),
        max_iterations: None,
    };
    let outcome = EvalHarness::new(Arc::new(TestAgent::answering_all("answer")))
        .run_query(&request)
        .await
        .expect("run query");
    assert_eq!(
        outcome.output_file,
        target.join(format!("{RESULT_FILE_PREFIX}ts.txt"))
    );
    assert_eq!(
        std::fs::read_to_string(&outcome.output_file).expect("read"),
        "answer"
    );

    let text = std::fs::read_to_string(&outcome.report_file).expect("read report");
    let report: Value = serde_json::from_str(&text).expect("parse report");
    assert_eq!(report["max_iterations"], Value::Null);
    assert_eq!(
        report["files_base_dir"],
        workspace.path("inputs").to_string_lossy().as_ref()
    );
    assert!(!report["prompt"]
        .as_str()
        .expect("prompt")
        .contains("Maximum agent iterations"));
    assert_eq!(
        report["report_file"],
        outcome.report_file.to_string_lossy().as_ref()
    );

    let explicit = workspace.path("nested/deep/result.txt");
    let request = QueryRequest {
        prompt: "Do it.".to_string(),
        output_path: explicit.clone(),
        timestamp: "ts".to_string(),
        ..QueryRequest::default()
    };
    let outcome = EvalHarness::new(Arc::new(TestAgent::answering_all("answer")))
        .run_query(&request)
        .await
        .expect("run query");
    assert_eq!(outcome.output_file, explicit);
    assert_eq!(
        outcome.report_file,
        workspace.path("nested/deep/result.json")
    );
    assert_eq!(
        std::fs::read_to_string(&explicit).expect("read answer"),
        "answer"
    );
    assert!(outcome.report_file.exists());
}

#[tokio::test]
async fn run_query_rejects_blank_expectations_and_agent_failures() {
    let workspace = workspace();
    let harness = EvalHarness::new(Arc::new(TestAgent::default()));

    let blank_prompt = QueryRequest {
        prompt: "   ".to_string(),
        timestamp: "ts".to_string(),
        ..QueryRequest::default()
    };
    assert!(error_text(harness.run_query(&blank_prompt).await).contains("prompt"));

    let blank_timestamp = QueryRequest {
        prompt: "Do it.".to_string(),
        ..QueryRequest::default()
    };
    assert!(error_text(harness.run_query(&blank_timestamp).await).contains("timestamp"));

    let request = QueryRequest {
        prompt: "Do it.".to_string(),
        output_path: workspace.path("out"),
        timestamp: "ts".to_string(),
        ..QueryRequest::default()
    };
    let error = error_text(harness.run_query(&request).await);
    assert!(error.contains("no canned answer"), "{error}");
}
