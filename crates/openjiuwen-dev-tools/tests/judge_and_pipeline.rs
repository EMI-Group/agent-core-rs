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

use std::path::Path;
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::agent_builder::{
    render_mermaid, AgentBuilder, AgentType, BuildOutput, BuildState, DlReflector, DslEdge,
    DslNode, ReflectionIssueKind, WorkflowDsl,
};
use openjiuwen_dev_tools::judge::{JudgeRequest, ModelJudge, Rubric, Verdict, VerdictSource};
use openjiuwen_dev_tools::skill_creator::SkillDefinition;
use openjiuwen_dev_tools::skill_evaluator::{
    iso8601_utc, load_cases_file, load_cases_json, score_design, CaseStatus, EvalCase, EvalHarness,
    EvaluatorConfig, ModelAgent, ScriptedCaseRunner, SkillEvaluator, SkillJudge,
    PIPELINE_BLOCKED_AT_E3, PIPELINE_COMPLETED,
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
    assert!((report.safety.score_pct - 0.95).abs() < 1e-9);
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
    assert!((report.safety.score_pct - 1.0).abs() < 1e-9);
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
        ScriptedTurn::text(DESIGN_FENCE),
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
    assert_eq!(lines.first(), Some(&"graph TD"));
    assert_eq!(
        lines.len(),
        1 + workflow.nodes.len() + workflow.edges.len(),
        "one line per node then one per edge"
    );
    assert_eq!(
        &lines[1..=workflow.nodes.len()],
        [
            "  start[Nodestart: start]",
            "  llm_1[Nodellm_1: llm_1]",
            "  end[Nodeend: end]",
        ],
        "every node renders in document order as `  {{id}}[Node{{id}}: {{label}}]`"
    );
    assert_eq!(
        &lines[1 + workflow.nodes.len()..],
        ["  start --> llm_1", "  llm_1 --> end"],
        "a single-successor edge renders unlabelled as `  {{src}} --> {{dst}}`"
    );
    assert_eq!(
        model.call_count(),
        5,
        "clarify, three designer passes, and generate"
    );
    assert_eq!(
        builder
            .status("workflow-1")
            .expect("the session exists")
            .state,
        BuildState::Completed
    );
}

#[test]
fn a_mermaid_branch_labels_only_the_multi_successor_edges() {
    let document = WorkflowDsl {
        name: "branch".to_string(),
        description: "two paths".to_string(),
        nodes: vec![
            DslNode {
                id: "start".to_string(),
                node_type: "start".to_string(),
                title: "Start".to_string(),
                ..DslNode::default()
            },
            DslNode {
                id: "yes".to_string(),
                node_type: "end".to_string(),
                title: "Yes".to_string(),
                description: "Yes path".to_string(),
                ..DslNode::default()
            },
            DslNode {
                id: "no".to_string(),
                node_type: "end".to_string(),
                title: "No".to_string(),
                description: "No path".to_string(),
                ..DslNode::default()
            },
        ],
        edges: vec![
            DslEdge {
                source: "start".to_string(),
                target: "yes".to_string(),
                condition: Some("score > 0.5".to_string()),
            },
            DslEdge {
                source: "start".to_string(),
                target: "no".to_string(),
                condition: Some("score <= 0.5".to_string()),
            },
        ],
    };

    let mermaid = render_mermaid(&document);
    let lines: Vec<&str> = mermaid.lines().collect();
    assert_eq!(
        lines,
        vec![
            "graph TD",
            "  start[Nodestart: start]",
            "  yes[Nodeyes: Yes path]",
            "  no[Nodeno: No path]",
            "  start -- score > 0.5 --> yes",
            "  start -- score <= 0.5 --> no",
        ],
        "a multi-successor source renders `  {{src}} -- {{label}} --> {{dst}}` per edge"
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

// ---------------------------------------------------------------------------
// The configurable safety gate, the Markdown report writers, and the ISO-8601
// rendering of the report timestamp.
// ---------------------------------------------------------------------------

/// Reads one written Markdown report back as text.
async fn read_report(path: &Path) -> String {
    tokio::fs::read_to_string(path)
        .await
        .expect("a written report must be readable")
}

/// The file names under `dir`, sorted, relative to the temporary workspace.
fn markdown_names(workspace: &TempWorkspace, dir: &str) -> Vec<String> {
    let mut names: Vec<String> = workspace
        .files()
        .into_iter()
        .filter(|path| path.starts_with(dir))
        .filter_map(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    names
}

/// An evaluator over [`skill`] whose scripted design verdict is `0.80` and whose
/// scripted safety verdict is `0.85`, gated by `judge_gate` on the judge and by
/// `config_gate` in the configuration.
fn gated_evaluator(judge_gate: f64, config_gate: f64) -> SkillEvaluator {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(fenced_verdict(0.80, true)),
        ScriptedTurn::text(fenced_verdict(0.85, true)),
    ]));
    evaluator()
        .with_skill_judge(SkillJudge::new(dynamic(&model)).with_gate_threshold(judge_gate))
        .with_config(EvaluatorConfig {
            safety_gate_threshold: config_gate,
            ..EvaluatorConfig::default()
        })
}

#[tokio::test]
async fn the_safety_gate_threshold_is_configurable_per_judge_and_per_call() {
    // One verdict scoring 0.85: above the 0.80 default, below a 0.90 strict gate.
    let verdict = fenced_verdict(0.85, true);

    let lenient_model = Arc::new(ScriptedModel::new([ScriptedTurn::text(verdict.clone())]));
    let lenient_judge = SkillJudge::new(dynamic(&lenient_model));
    assert!(
        (lenient_judge.gate_threshold() - 0.80).abs() < 1e-9,
        "a fresh judge starts at the documented 0.80 gate, got {}",
        lenient_judge.gate_threshold()
    );
    let lenient = lenient_judge
        .safety(&skill(), &[])
        .await
        .expect("the safety judgement runs");
    assert!(
        !lenient.used_fallback,
        "the scripted verdict is the model's"
    );
    assert!((lenient.report.score_pct - 0.85).abs() < 1e-9);
    assert!((lenient.report.gate_threshold - 0.80).abs() < 1e-9);
    assert!(
        lenient.report.passes_gate() && !lenient.report.blocked,
        "0.85 clears the default gate: {}",
        lenient.report.score_pct
    );

    let strict_model = Arc::new(ScriptedModel::new([ScriptedTurn::text(verdict.clone())]));
    let strict_judge = SkillJudge::new(dynamic(&strict_model)).with_gate_threshold(0.90);
    assert!((strict_judge.gate_threshold() - 0.90).abs() < 1e-9);
    let strict = strict_judge
        .safety(&skill(), &[])
        .await
        .expect("a closed gate is a verdict, not an error");
    assert!(
        (strict.report.score_pct - 0.85).abs() < 1e-9,
        "the stricter gate must not rewrite the score: {}",
        strict.report.score_pct
    );
    assert!((strict.report.gate_threshold - 0.90).abs() < 1e-9);
    assert!(strict.report.blocked, "0.85 misses the strict gate");
    assert!(!strict.report.passes_gate());

    // A per-call override gates one judgement without touching the judge.
    let override_model = Arc::new(ScriptedModel::new([ScriptedTurn::text(verdict)]));
    let overriding = SkillJudge::new(dynamic(&override_model));
    let one_off = overriding
        .safety_with_threshold(&skill(), &[], 0.86)
        .await
        .expect("the one-off threshold is applied");
    assert!(one_off.report.blocked, "0.85 misses the 0.86 override");
    assert!((one_off.report.gate_threshold - 0.86).abs() < 1e-9);
    assert!(
        (overriding.gate_threshold() - 0.80).abs() < 1e-9,
        "the per-call override leaves the judge's own gate alone"
    );
}

#[tokio::test]
async fn a_gate_threshold_outside_the_unit_interval_is_a_validation_error() {
    for threshold in [1.5_f64, -0.1, f64::NAN] {
        let model = Arc::new(ScriptedModel::new([ScriptedTurn::text(fenced_verdict(
            0.9, true,
        ))]));
        let error = SkillJudge::new(dynamic(&model))
            .with_gate_threshold(threshold)
            .safety(&skill(), &[])
            .await
            .expect_err("an out-of-range gate is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "{threshold}");
        assert!(
            error.to_string().contains("safety gate threshold"),
            "{threshold}: {error}"
        );
        assert_eq!(
            model.call_count(),
            0,
            "the gate is validated before the model is consulted: {threshold}"
        );
    }

    let model = Arc::new(ScriptedModel::new([ScriptedTurn::text(fenced_verdict(
        0.9, true,
    ))]));
    let error = SkillJudge::new(dynamic(&model))
        .safety_with_threshold(&skill(), &[], 2.0)
        .await
        .expect_err("the per-call override is validated too");
    assert_eq!(error.kind(), ErrorKind::Validation);

    // The configured fraction reaches the judged pipeline through the same check.
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(fenced_verdict(0.80, true)),
        ScriptedTurn::text(fenced_verdict(0.90, true)),
    ]));
    let error = evaluator()
        .with_skill_judge(SkillJudge::new(dynamic(&model)))
        .with_config(EvaluatorConfig {
            safety_gate_threshold: 1.5,
            ..EvaluatorConfig::default()
        })
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect_err("the configured fraction is validated by the judged pipeline");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("safety gate threshold"),
        "{error}"
    );
}

#[tokio::test]
async fn with_skill_judge_applies_the_configured_gate_fraction_to_the_report() {
    assert!(
        (EvaluatorConfig::default().safety_gate_threshold - 0.80).abs() < 1e-9,
        "the default configuration gates safety at 0.80"
    );

    // The judge's own gate is lenient here, so the strict configured fraction
    // must be the one that reaches the report.
    let strict = gated_evaluator(0.50, 0.90)
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect("a closed gate is a verdict, not an error");
    assert!(
        (strict.safety.gate_threshold - 0.90).abs() < 1e-9,
        "the configuration's fraction, not the judge's 0.50, is reported: {}",
        strict.safety.gate_threshold
    );
    assert!((strict.safety.score_pct - 0.85).abs() < 1e-9);
    assert!(!strict.safety.passes_gate(), "0.85 misses the 0.90 gate");
    assert_eq!(strict.final_score.pipeline_status, PIPELINE_BLOCKED_AT_E3);
    assert!(!strict.final_score.is_completed());
    assert!(
        strict.final_score.gates_passed.is_empty(),
        "a blocked pipeline claims no passed gate"
    );
    assert!((strict.final_score.final_score_pct - 0.0).abs() < 1e-9);
    assert_eq!(
        strict
            .judge
            .as_ref()
            .expect("a judged run records provenance")
            .safety_source,
        VerdictSource::Model
    );

    let lenient = gated_evaluator(0.95, 0.70)
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect("the lenient configuration completes the pipeline");
    assert!((lenient.safety.gate_threshold - 0.70).abs() < 1e-9);
    assert!(lenient.safety.passes_gate(), "0.85 clears the 0.70 gate");
    assert_eq!(lenient.final_score.pipeline_status, PIPELINE_COMPLETED);
    assert!(lenient.final_score.is_completed());
    assert_eq!(
        lenient.final_score.gates_passed,
        vec!["e3_safety".to_owned()]
    );

    // Without a configuration the default 0.80 fraction is the reported gate.
    let default_model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(fenced_verdict(0.80, true)),
        ScriptedTurn::text(fenced_verdict(0.85, true)),
    ]));
    let defaulted = evaluator()
        .with_skill_judge(SkillJudge::new(dynamic(&default_model)))
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect("the default gate completes the pipeline");
    assert!((defaulted.safety.gate_threshold - 0.80).abs() < 1e-9);
    assert!(defaulted.safety.passes_gate());
    assert_eq!(defaulted.final_score.pipeline_status, PIPELINE_COMPLETED);
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn a_completed_judged_report_writes_the_markdown_reports() {
    use openjiuwen_dev_tools::skill_evaluator::artifacts::{
        FINAL_REPORT_ARTIFACT, LLM_JUDGE_REPORT_ARTIFACT, SAFETY_REPORT_ARTIFACT,
    };

    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(fenced_verdict(0.80, true)),
        ScriptedTurn::text(fenced_verdict(0.95, true)),
    ]));
    let report = evaluator()
        .with_judge(dynamic(&model))
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect("the judged run scores");
    // 0.40 · 0.80 design + 0.60 · 1.0 pass rate = 0.92, which grades A.
    assert!((report.final_score.final_score_pct - 0.92).abs() < 1e-9);
    assert_eq!(report.final_score.pipeline_status, PIPELINE_COMPLETED);

    let dir = workspace.path("evals/skill-tests/demo");
    let paths = report
        .write_markdown_artifacts(&dir)
        .await
        .expect("a completed run writes its markdown reports");
    assert_eq!(paths.dir, dir);
    assert_eq!(paths.llm_judge, dir.join(LLM_JUDGE_REPORT_ARTIFACT));
    assert_eq!(paths.safety, dir.join(SAFETY_REPORT_ARTIFACT));
    assert_eq!(
        paths.final_report.as_ref(),
        Some(&dir.join(FINAL_REPORT_ARTIFACT)),
        "a completed run writes final_report.md"
    );
    assert!(
        paths.pipeline_report.is_none(),
        "a completed run writes no pipeline_report.md"
    );
    assert_eq!(
        markdown_names(&workspace, "evals"),
        vec![
            FINAL_REPORT_ARTIFACT.to_owned(),
            LLM_JUDGE_REPORT_ARTIFACT.to_owned(),
            SAFETY_REPORT_ARTIFACT.to_owned(),
        ],
        "a completed run writes exactly the three surviving reports"
    );

    let design = read_report(&paths.llm_judge).await;
    assert!(
        design.starts_with("# Skill Evaluation Report: demo"),
        "the design report opens with the Python title: {design}"
    );
    assert!(
        design.contains("- **Total Score**: 80/100")
            && design.contains("- **Grade**: B")
            && design.contains("- **Ready for functional testing**: Yes"),
        "the model verdict renders in points and clears the readiness gate: {design}"
    );

    let safety = read_report(&paths.safety).await;
    assert!(
        safety.starts_with("# Safety Evaluation Report: demo"),
        "the safety report opens with the Python title: {safety}"
    );
    assert!(
        safety.contains("- **Score**: 95/100 (95%)")
            && safety.contains("- **Gate**: PASS")
            && safety.contains("- **Blocked**: No")
            && safety.contains("- **Auto-fail triggered**: None"),
        "the judged safety verdict and the open gate are rendered: {safety}"
    );

    let final_report = read_report(
        paths
            .final_report
            .as_ref()
            .expect("the completed-report path"),
    )
    .await;
    assert!(
        final_report.starts_with("# Eval Pipeline Report: demo"),
        "the pipeline report opens with the Python title: {final_report}"
    );
    assert!(
        final_report.contains("- **Score**: 0.92 (A)")
            && final_report.contains("- **Status**: COMPLETED")
            && final_report.contains("**Ship-ready**: Yes"),
        "the final verdict carries the weighted score, status, and ship-readiness: {final_report}"
    );
    assert!(
        final_report.contains("| E1 Design (skill_llm_judge) | 80% | 0.40 | 0.32 |")
            && final_report.contains("| E2 Functional (skill_tester) | 100% | 0.60 | 0.6 |")
            && final_report.contains("| E3 Safety (skill_safety_judge) | PASS | gate | - |"),
        "the evaluator breakdown carries the judged design, functional, and gate rows: {final_report}"
    );
    assert!(
        final_report.contains("- [final_score.json](final_score.json)"),
        "a completed run links the final_score.json artifact: {final_report}"
    );
    assert!(
        final_report.contains("**Run timestamp**: 2023-11-14T22:13:20Z"),
        "the timestamp line renders the caller-supplied epoch milliseconds: {final_report}"
    );
}

#[tokio::test]
async fn a_blocked_judged_report_writes_only_the_pipeline_report() {
    use openjiuwen_dev_tools::skill_evaluator::artifacts::{
        FINAL_REPORT_ARTIFACT, LLM_JUDGE_REPORT_ARTIFACT, PIPELINE_REPORT_ARTIFACT,
        SAFETY_REPORT_ARTIFACT,
    };

    let workspace = TempWorkspace::new().expect("a temporary workspace");
    // 0.85 safety sits below the configured 0.90 gate, so E3 blocks the pipeline.
    let report = gated_evaluator(0.50, 0.90)
        .evaluate_with_judge_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect("a closed gate is a verdict, not an error");
    assert_eq!(report.final_score.pipeline_status, PIPELINE_BLOCKED_AT_E3);

    let dir = workspace.path("evals/blocked");
    let paths = report
        .write_markdown_artifacts(&dir)
        .await
        .expect("a blocked run writes its markdown reports");
    assert_eq!(paths.llm_judge, dir.join(LLM_JUDGE_REPORT_ARTIFACT));
    assert_eq!(paths.safety, dir.join(SAFETY_REPORT_ARTIFACT));
    assert!(
        paths.final_report.is_none(),
        "a blocked run has no final verdict to render"
    );
    assert_eq!(
        paths.pipeline_report.as_ref(),
        Some(&dir.join(PIPELINE_REPORT_ARTIFACT)),
        "a blocked run writes pipeline_report.md"
    );
    assert_eq!(
        markdown_names(&workspace, "evals"),
        vec![
            LLM_JUDGE_REPORT_ARTIFACT.to_owned(),
            PIPELINE_REPORT_ARTIFACT.to_owned(),
            SAFETY_REPORT_ARTIFACT.to_owned(),
        ],
        "the blocked run writes three reports and skips final_report.md"
    );

    // The individual writers agree with the aggregate one.
    assert!(
        report
            .write_final_report(&dir)
            .await
            .expect("the writer runs")
            .is_none(),
        "final_report.md is only written for a completed run"
    );
    assert_eq!(
        report
            .write_pipeline_report(&dir)
            .await
            .expect("the writer runs"),
        Some(dir.join(PIPELINE_REPORT_ARTIFACT))
    );
    assert!(!dir.join(FINAL_REPORT_ARTIFACT).exists());

    let pipeline = read_report(
        paths
            .pipeline_report
            .as_ref()
            .expect("the blocked-report path"),
    )
    .await;
    assert!(
        pipeline.starts_with("# Eval Pipeline Report: demo"),
        "the block report opens with the Python title: {pipeline}"
    );
    assert!(
        pipeline.contains("- **Status**: BLOCKED_AT_E3")
            && pipeline.contains(
                "- **Blocked by**: E3 safety gate (safety score below its gate threshold \
                 or a critical finding)"
            ),
        "the block report names the gate that closed: {pipeline}"
    );
    assert!(
        pipeline.contains("- **Design score**: 80/100 (80%)")
            && pipeline.contains("- **Safety score**: 85/100 (85%), gate FAIL"),
        "the block report carries the evidence that closed the gate: {pipeline}"
    );
    for heading in ["## Key Issues to Fix", "## Full Reports"] {
        assert!(
            pipeline.contains(heading),
            "the block report carries {heading}: {pipeline}"
        );
    }
    assert!(
        pipeline.contains("- Gate: stopped at BLOCKED_AT_E3"),
        "the issue list leads with the blocking gate: {pipeline}"
    );
    assert!(
        !pipeline.contains("[final_score.json](final_score.json)"),
        "a blocked run writes no final_score.json to link: {pipeline}"
    );
}

#[tokio::test]
async fn the_functional_timestamp_renders_as_iso8601_utc() {
    let report = evaluator()
        .evaluate_at(&skill(), &[passing_case("case-1", "alpha")], STAMP)
        .await
        .expect("the report is scored");
    assert_eq!(
        report.functional.timestamp, STAMP,
        "the epoch field is kept"
    );
    assert_eq!(report.functional.timestamp_iso(), iso8601_utc(STAMP));
    assert_eq!(
        report.functional.timestamp_iso(),
        "2023-11-14T22:13:20Z",
        "the rendering is UTC at second precision with no fractional seconds"
    );

    let precise = evaluator()
        .evaluate_at(&skill(), &[passing_case("case-1", "alpha")], STAMP + 999)
        .await
        .expect("the report is scored");
    assert_eq!(
        precise.functional.timestamp,
        STAMP + 999,
        "sub-second milliseconds stay in the numeric field"
    );
    assert_eq!(
        precise.functional.timestamp_iso(),
        "2023-11-14T22:13:20Z",
        "the rendering truncates them"
    );

    let document = precise.to_json().expect("the report serializes");
    assert_eq!(document["functional"]["timestamp"], json!(STAMP + 999));
    assert!(
        document["functional"].get("timestamp_iso").is_none(),
        "timestamp_iso is a rendering method, not a serialized field"
    );
}
