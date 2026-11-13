//! Cross-module integration suite for the judge, eval harness, and builder pipeline.
//!
//! Three public seams are exercised together: the rubric-driven
//! [`ModelJudge`] and its deterministic fallback, the `skill_evaluator` report
//! provenance that consumes that judge, the runnable [`EvalHarness`] with its
//! JSON/JSONL artifacts, and the staged [`AgentBuilder`] pipeline that emits a
//! validated workflow document plus its Mermaid rendering.
//!
//! Everything runs offline on [`ScriptedModel`], writes only under a
//! [`TempWorkspace`], and uses fixed timestamps.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::agent_builder::{
    render_mermaid, AgentBuilder, AgentType, BuildOutput, BuildState, DlReflector, DslEdge,
    DslNode, ReflectionIssueKind, WorkflowDsl,
};
use openjiuwen_dev_tools::judge::{JudgeRequest, ModelJudge, Rubric, Verdict, VerdictSource};
use openjiuwen_dev_tools::skill_creator::SkillDefinition;
use openjiuwen_dev_tools::skill_evaluator::{
    load_cases_file, load_cases_json, score_design, CaseStatus, EvalCase, EvalHarness, ModelAgent,
    ScriptedCaseRunner, SkillEvaluator,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn, TempWorkspace};
use serde_json::{json, Value};

/// The fixed timestamp every report assertion uses.
const STAMP: i64 = 1_700_000_000_000;

/// The query the staged builder pipeline is fed.
const QUERY: &str = "Create a workflow that summarizes the document and returns a summary.";

/// A clarification answer declaring the request complete.
const CLARIFY_READY: &str = r#"```json
{"ready": true, "question": "", "summary": "goal, output"}
```"#;

/// A clarification answer asking for the expected output.
const CLARIFY_NEEDED: &str = r#"```json
{"ready": false, "question": "What should the agent produce?", "summary": "goal"}
```"#;

/// A model answer holding a complete workflow design.
const DESIGN_FENCE: &str = r#"```json
{"name": "demo", "description": "demo design",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "llm_1", "type": "llm", "title": "Draft", "description": "draft it"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "llm_1"}, {"source": "llm_1", "target": "end"}]}
```"#;

/// A model answer holding a valid workflow document.
const GENERATED_FENCE: &str = r#"```json
{"name": "generated", "description": "generated workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"},
           {"id": "llm_1", "type": "llm", "title": "Draft"},
           {"id": "end", "type": "end", "title": "End"}],
 "edges": [{"source": "start", "target": "llm_1"}, {"source": "llm_1", "target": "end"}]}
```"#;

/// Coerces a scripted model into the trait object each seam takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// A two-dimension rubric used by every judge assertion.
fn rubric() -> Rubric {
    Rubric::uniform(
        "skill quality",
        "Is the skill usable?",
        ["clarity", "depth"],
    )
}

/// A fenced verdict answer the parser accepts.
fn fenced_verdict(score: f64, passed: bool) -> String {
    format!("```json\n{{\"score\": {score}, \"passed\": {passed}, \"reasoning\": \"ok\"}}\n```")
}

/// A version-valid demonstration skill.
fn skill() -> SkillDefinition {
    let body = concat!(
        "# Demo Skill\n\n",
        "## Overview\n\n",
        "This skill summarizes documents end to end. It explains the input ",
        "format, the processing steps, and the expected result structure so a ",
        "reader can reproduce the work without extra guidance.\n\n",
        "## Steps\n\n",
        "1. Read the input document.\n",
        "2. Extract the key facts.\n",
        "3. Write the summary.\n\n",
        "## Output\n\n",
        "The result format is plain text.\n"
    );
    SkillDefinition::new("demo", "Summarize documents", body)
}

/// One passing case whose answer must contain `word`.
fn passing_case(id: &str, word: &str) -> EvalCase {
    EvalCase::new(id, format!("Case {id}"), format!("Respond with {word}"))
        .with_expectation(format!("contains:{word}"))
}

/// An evaluator with the scripted answer for `case-1`.
fn evaluator() -> SkillEvaluator {
    SkillEvaluator::new(Arc::new(
        ScriptedCaseRunner::new().respond("case-1", "alpha answer"),
    ))
}

/// A `BuildStatus`-agnostic assertion that the pipeline produced a workflow.
fn workflow_of(output: &BuildOutput) -> &WorkflowDsl {
    match output {
        BuildOutput::Workflow(workflow) => workflow,
        other => panic!("expected a workflow, got {other:?}"),
    }
}

#[tokio::test]
async fn the_model_judge_parses_a_fenced_verdict_and_degrades_cleanly() {
    let rubric = rubric();
    let request = JudgeRequest::new(&rubric, "the subject", "score it");

    let model = Arc::new(ScriptedModel::new([ScriptedTurn::text(fenced_verdict(
        0.75, true,
    ))]));
    let verdict = ModelJudge::new(dynamic(&model))
        .judge_with_fallback(&request, || Verdict::deterministic(0.1, "unused fallback"))
        .await;
    assert_eq!(verdict.source, VerdictSource::Model);
    assert!((verdict.score - 0.75).abs() < 1e-9);
    assert!(verdict.passed);
    assert_eq!(model.call_count(), 1);

    let garbage = Arc::new(ScriptedModel::new([ScriptedTurn::text("not a verdict")]));
    let verdict = ModelJudge::new(dynamic(&garbage))
        .judge_with_fallback(&request, || {
            Verdict::deterministic(0.42, "deterministic fallback")
        })
        .await;
    assert_eq!(verdict.source, VerdictSource::Deterministic);
    assert!((verdict.score - 0.42).abs() < 1e-9);
    assert_eq!(garbage.call_count(), 1);
}

#[tokio::test]
async fn a_judged_evaluation_records_model_provenance() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(fenced_verdict(0.8, true)),
        ScriptedTurn::text(fenced_verdict(0.95, true)),
    ]));
    let report = evaluator()
        .with_judge(dynamic(&model))
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect("the judged run succeeds");

    let record = report.judge.as_ref().expect("a judge record");
    assert_eq!(record.design_source, VerdictSource::Model);
    assert_eq!(record.safety_source, VerdictSource::Model);
    assert!((report.design.design_score - 80.0).abs() < 1e-9);
    assert!((report.safety.score_pct - 95.0).abs() < 1e-9);
    assert_eq!(report.functional.timestamp, STAMP);
    assert_eq!(report.final_score.pipeline_status, "COMPLETED");
    assert_eq!(model.call_count(), 2, "one design and one safety call");
}

#[tokio::test]
async fn a_garbage_judge_answer_falls_back_to_the_deterministic_report() {
    let garbage = Arc::new(ScriptedModel::new([
        ScriptedTurn::text("no design verdict"),
        ScriptedTurn::text("no safety verdict"),
    ]));
    let report = evaluator()
        .with_judge(dynamic(&garbage))
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect("a degraded judgement still scores");

    let record = report.judge.as_ref().expect("a judge record");
    assert_eq!(record.design_source, VerdictSource::Deterministic);
    assert_eq!(record.safety_source, VerdictSource::Deterministic);
    assert_eq!(report.design, score_design(&skill()));
    assert!((report.safety.score_pct - 100.0).abs() < 1e-9);
    assert_eq!(report.final_score.pipeline_status, "COMPLETED");
    assert_eq!(garbage.call_count(), 2);
}

#[tokio::test]
async fn the_eval_harness_runs_model_cases_and_writes_its_artifacts() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let cases_path = workspace
        .write(
            "cases/model.jsonl",
            concat!(
                "{\"id\": \"case-1\", \"name\": \"Alpha\", \"prompt\": \"Say alpha\", ",
                "\"expectations\": [\"contains:alpha\"]}\n",
                "{\"id\": \"case-2\", \"name\": \"Beta\", \"prompt\": \"Say beta\", ",
                "\"expectations\": [\"contains:beta\"]}\n"
            ),
        )
        .expect("the case file is written");
    let cases = load_cases_file(&cases_path)
        .await
        .expect("the JSONL case file loads");
    assert_eq!(cases.len(), 2);

    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text("alpha answer"),
        ScriptedTurn::text("beta answer"),
    ]));
    let harness = EvalHarness::new(Arc::new(ModelAgent::new(dynamic(&model))));
    let run = harness.run(&cases).await.expect("the harness runs");

    assert_eq!(run.summary.total, 2);
    assert_eq!(run.summary.passed, 2);
    assert_eq!(run.summary.failed, 0);
    assert_eq!(run.summary.partial, 0);
    assert!((run.summary.pass_rate - 1.0).abs() < 1e-9);
    assert!(run
        .cases
        .iter()
        .all(|case| case.status == CaseStatus::Passed));

    let json = run.to_json_string().expect("the run serializes");
    let document: Value = serde_json::from_str(&json).expect("the artifact is JSON");
    assert_eq!(document["cases"].as_array().map(Vec::len), Some(2));
    assert_eq!(document["summary"]["passed"], json!(2));

    let jsonl = run.to_jsonl().expect("the run serializes as JSONL");
    let lines: Vec<&str> = jsonl.lines().collect();
    assert_eq!(lines.len(), 2, "one object per case");
    for line in lines {
        let case: Value = serde_json::from_str(line).expect("each line is JSON");
        assert!(case.get("case_id").is_some());
        assert!(case.get("status").is_some());
    }

    run.write_json(workspace.path("reports/run.json"))
        .await
        .expect("the JSON artifact is written");
    let stored: Value = serde_json::from_str(
        &workspace
            .read("reports/run.json")
            .expect("the JSON artifact is readable"),
    )
    .expect("the stored artifact parses");
    assert_eq!(stored["summary"]["total"], json!(2));

    run.write_jsonl(workspace.path("reports/run.jsonl"))
        .await
        .expect("the JSONL artifact is written");
    let stored = workspace
        .read("reports/run.jsonl")
        .expect("the JSONL artifact is readable");
    assert_eq!(stored.lines().count(), 2);
    assert!(stored.ends_with('\n'), "every JSONL line is terminated");
}

#[test]
fn the_case_loaders_accept_every_documented_shape() {
    let case = r#"{"id": "case-1", "name": "Alpha", "prompt": "Say alpha",
                   "expectations": ["contains:alpha"]}"#;

    let array = load_cases_json(&format!("[{case}]")).expect("a JSON array loads");
    assert_eq!(array.len(), 1);
    assert_eq!(array[0].id, "case-1");

    let object = load_cases_json(&format!("{{\"cases\": [{case}]}}"))
        .expect("a {\"cases\": [...]} object loads");
    assert_eq!(object.len(), 1);

    let single = load_cases_json(case).expect("a single case object loads");
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].prompt, "Say alpha");

    let blank = load_cases_json("   ").expect_err("blank input is rejected");
    assert_eq!(blank.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn a_malformed_jsonl_line_reports_its_number() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let path = workspace
        .write(
            "cases/bad.jsonl",
            concat!(
                "{\"id\": \"case-1\", \"name\": \"Alpha\", \"prompt\": \"Say alpha\", ",
                "\"expectations\": [\"contains:alpha\"]}\n",
                "this is not a case\n"
            ),
        )
        .expect("the malformed file is written");

    let error = load_cases_file(&path)
        .await
        .expect_err("a malformed line is fatal");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("line 2"), "got {error}");
}

#[tokio::test]
async fn the_builder_pipeline_generates_a_mermaid_workflow() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_READY),
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text(GENERATED_FENCE),
    ]));
    let mut builder = AgentBuilder::new(dynamic(&model));
    let result = builder
        .build_with_pipeline(QUERY, "workflow-1", AgentType::Workflow, STAMP)
        .await
        .expect("the staged pipeline succeeds");

    assert_eq!(result.status, "completed");
    assert_eq!(result.agent_type, AgentType::Workflow);
    let workflow = workflow_of(&result.output);
    workflow
        .validate()
        .expect("the generated document validates");
    assert_eq!(workflow.name, "generated");

    let mermaid = render_mermaid(workflow);
    let lines: Vec<&str> = mermaid.lines().collect();
    assert_eq!(lines.first(), Some(&"flowchart TD"));
    assert_eq!(
        lines.len(),
        1 + workflow.nodes.len() + workflow.edges.len(),
        "one line per node then one per edge"
    );
    for node in &workflow.nodes {
        assert!(
            lines
                .iter()
                .any(|line| line.trim_start().starts_with(node.id.as_str())),
            "missing node {}",
            node.id
        );
    }
    assert_eq!(model.call_count(), 3, "clarify, design, and generate");
    assert_eq!(
        builder
            .status("workflow-1")
            .expect("the session exists")
            .state,
        BuildState::Completed
    );
}

#[tokio::test]
async fn a_pipeline_question_is_returned_as_a_response() {
    let model = Arc::new(ScriptedModel::text(CLARIFY_NEEDED));
    let mut builder = AgentBuilder::new(dynamic(&model));
    let result = builder
        .build_with_pipeline("do something", "session-1", AgentType::LlmAgent, STAMP)
        .await
        .expect("an incomplete request still succeeds");

    assert_eq!(result.status, "clarifying");
    match &result.output {
        BuildOutput::Response(text) => {
            assert_eq!(text, "What should the agent produce?");
        }
        other => panic!("expected a clarifying response, got {other:?}"),
    }
    assert_eq!(model.call_count(), 1);
    assert_eq!(
        builder
            .status("session-1")
            .expect("the session exists")
            .state,
        BuildState::Clarifying
    );
}

#[tokio::test]
async fn the_pipeline_falls_back_without_any_model_answer() {
    let model = Arc::new(ScriptedModel::new([]));
    let mut builder = AgentBuilder::new(dynamic(&model));
    let result = builder
        .build_with_pipeline(QUERY, "workflow-2", AgentType::Workflow, STAMP)
        .await
        .expect("the deterministic stages succeed");

    assert_eq!(result.status, "completed");
    let workflow = workflow_of(&result.output);
    workflow
        .validate()
        .expect("the fallback document validates");
    assert_eq!(
        workflow.nodes.first().map(|node| node.id.as_str()),
        Some("start")
    );
    assert_eq!(
        workflow.nodes.last().map(|node| node.id.as_str()),
        Some("end")
    );
    assert!(workflow.nodes.iter().any(|node| node.id == "llm_1"));
}

#[test]
fn the_reflector_repairs_a_dangling_edge() {
    let document = WorkflowDsl {
        name: "broken".to_string(),
        description: "has a dangling edge".to_string(),
        nodes: vec![
            DslNode {
                id: "start".to_string(),
                node_type: "start".to_string(),
                title: "Start".to_string(),
                ..DslNode::default()
            },
            DslNode {
                id: "end".to_string(),
                node_type: "end".to_string(),
                title: "End".to_string(),
                ..DslNode::default()
            },
        ],
        edges: vec![
            DslEdge {
                source: "start".to_string(),
                target: "end".to_string(),
                condition: None,
            },
            DslEdge {
                source: "end".to_string(),
                target: "ghost".to_string(),
                condition: None,
            },
        ],
    };

    let reflection = DlReflector::new()
        .reflect(&document)
        .expect("reflection repairs rather than failing");
    assert!(!reflection.passed);
    assert!(reflection
        .issues
        .iter()
        .any(|issue| issue.kind == ReflectionIssueKind::DanglingEdge));
    reflection
        .repaired
        .validate()
        .expect("the repaired document validates");
    assert_eq!(reflection.repaired.edges.len(), 1);
    assert_eq!(reflection.repaired.edges[0].target, "end");
}
