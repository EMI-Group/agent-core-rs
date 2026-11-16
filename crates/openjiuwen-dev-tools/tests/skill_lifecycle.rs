//! Cross-module lifecycle of a skill: authored by `skill_creator`, persisted as a
//! bundle, reloaded from disk, and evaluated by `skill_evaluator`.
//!
//! The module unit tests already cover each half in isolation; this suite only
//! asserts the seam between them — that the artifact `skill_creator` writes is
//! exactly the artifact `skill_evaluator` reads and scores, and that the Python
//! report field names survive serialization of a real run.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::skill_creator::{
    SkillBundle, SkillCreator, SkillDefinition, SkillResource, SkillSpec,
};
use openjiuwen_dev_tools::skill_evaluator::{
    CaseStatus, EvalCase, EvaluationReport, EvaluatorConfig, ModelCaseRunner, ReactCaseRunner,
    SafetyRule, ScriptedCaseRunner, SkillEvaluator, DEFAULT_MAX_REACT_ROUNDS,
};
use openjiuwen_dev_tools::tune::{ToolStub, ToolStubs};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::Role;
use openjiuwen_test_support::{MockCompletion, ScriptedModel, ScriptedTurn, TempWorkspace};
use serde_json::json;

/// The instruction template: every design-rubric dimension is satisfiable from it
/// once rendered, and it references the shipped resource by path.
const INSTRUCTION_TEMPLATE: &str = "\
# Document Brief

## Overview

Turn `{{document_type}}` into a concise brief for {{audience}}. This skill explains the input
format, the processing steps, and the expected result structure so that a reader can reproduce
the work without extra guidance.

## Inputs

- A markdown or plain-text document supplied by the user.
- The audience named in the request.

## Steps

1. Read the document from start to finish.
2. Extract the claims that change a decision.
3. Group the claims using the sections of `references/guide.md`.
4. Draft one sentence per claim in the brief.

## Output

Return a markdown brief with a title, three to five bullets, and a closing risk note.";

/// Authors the shared fixture skill through the deterministic creator path.
///
/// Used by both tests so the evaluated skill is byte-for-byte the authored one.
fn author_skill() -> SkillDefinition {
    let mut spec = SkillSpec::new(
        "document-brief",
        "Turn a document into a decision-ready brief",
        INSTRUCTION_TEMPLATE,
    )
    .with_variable("document_type", "meeting transcript")
    .with_variable("audience", "release engineers")
    .with_resource(SkillResource::new(
        "references/guide.md",
        "# Guide\n\nName each section of the brief.\n",
    ))
    .with_version("0.1.0");
    spec.metadata.insert("owner".to_owned(), json!("docs-team"));
    spec.metadata
        .insert("stability".to_owned(), json!("experimental"));

    SkillCreator::new()
        .create(&spec)
        .expect("a valid spec must create deterministically without a model")
}

/// The three functional cases: two answers satisfy their expectation, one does not.
fn evaluation_cases() -> Vec<EvalCase> {
    vec![
        EvalCase::new(
            "case-brief",
            "Brief structure",
            "Draft a brief from the attached transcript.",
        )
        .with_expectation("contains:brief"),
        EvalCase::new(
            "case-risk",
            "Risk note",
            "Draft a brief and close it with the risk note.",
        )
        .with_expectation("contains:risk note"),
        EvalCase::new(
            "case-audience",
            "Audience targeting",
            "Draft a brief for release engineers.",
        )
        .with_expectation("contains:release engineers"),
    ]
}

/// The scripted answers behind [`evaluation_cases`]: pass, pass, fail.
fn scripted_runner() -> ScriptedCaseRunner {
    ScriptedCaseRunner::new()
        .respond(
            "case-brief",
            "This brief lists the decisions in three bullets.",
        )
        .respond(
            "case-risk",
            "The brief closes with a risk note: none recorded.",
        )
        .respond("case-audience", "A brief written for the leadership team.")
}

/// Mirrors the evaluator's four-decimal rounding for fraction values.
fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// Asserts that the Python artifact field names are present at their documented
/// nesting and agree with the in-memory report.
fn assert_python_json_fields(report: &EvaluationReport) {
    let json = report
        .to_json()
        .expect("a completed report must serialize to JSON");

    let design = &json["design"];
    assert!(
        design["design_score_pct"].is_number(),
        "design.design_score_pct must serialize as a number for Python report parity"
    );
    assert_eq!(
        design["ready_for_functional_testing"],
        json!(report.design.ready_for_functional_testing),
        "design.ready_for_functional_testing must agree with the in-memory report"
    );

    let pass_rate = &json["functional"]["summary"]["pass_rate"];
    assert!(
        pass_rate.is_number(),
        "functional.summary.pass_rate must serialize as a number"
    );
    assert!(
        (pass_rate.as_f64().expect("pass rate is a number") - report.functional.summary.pass_rate)
            .abs()
            < 1e-9,
        "functional.summary.pass_rate must agree with the in-memory report"
    );

    let final_score = &json["final_score"];
    assert!(
        final_score["final_score_pct"].is_number(),
        "final_score.final_score_pct must serialize as a number"
    );
    assert_eq!(
        final_score["pipeline_status"],
        json!(report.final_score.pipeline_status),
        "final_score.pipeline_status must serialize under its Python name"
    );
    assert_eq!(
        final_score["gates_passed"],
        json!(report.final_score.gates_passed),
        "final_score.gates_passed must serialize as the list of passed gates"
    );
}

#[tokio::test]
async fn skill_definition_survives_a_bundle_write_read_cycle() {
    let workspace = TempWorkspace::new().expect("temp workspace must be creatable");
    let root = workspace.path("skills");
    let authored = author_skill();

    assert!(
        authored.instructions.contains("release engineers"),
        "create() must render {{audience}} into the instruction body before writing"
    );
    assert!(
        !authored.instructions.contains("{{"),
        "create() must leave no unresolved placeholder in the rendered instructions"
    );

    let directory = SkillBundle::write(&authored, &root)
        .await
        .expect("writing a valid definition must succeed");
    assert_eq!(
        directory,
        root.join("document-brief"),
        "write must return <root>/<name> for the authored skill"
    );
    assert!(
        workspace.exists("skills/document-brief/SKILL.md"),
        "the bundle document must be written at <root>/<name>/SKILL.md"
    );
    assert!(
        workspace.exists("skills/document-brief/references/guide.md"),
        "nested resources must be written at their relative path inside the bundle"
    );

    let files = workspace.files();
    assert_eq!(
        files,
        vec![
            PathBuf::from("skills/document-brief/SKILL.md"),
            PathBuf::from("skills/document-brief/references/guide.md"),
        ],
        "the bundle must contain exactly the document and its one resource"
    );

    let reloaded = SkillBundle::read(&directory)
        .await
        .expect("a freshly written bundle must read back");
    assert_eq!(
        reloaded, authored,
        "read() must reconstruct the authored definition exactly, including version and metadata"
    );

    let document = tokio::fs::read_to_string(directory.join("SKILL.md"))
        .await
        .expect("the written SKILL.md must be readable");
    assert!(
        document.contains("release engineers"),
        "the persisted SKILL.md body must hold the rendered template value"
    );
    assert!(
        !document.contains("{{document_type}}") && !document.contains("{{audience}}"),
        "the persisted SKILL.md must not contain a literal {{var}} placeholder"
    );

    let reparsed = SkillDefinition::from_markdown(&document)
        .expect("the persisted SKILL.md must parse as a skill document");
    assert_eq!(
        reparsed.name, authored.name,
        "the frontmatter must preserve the skill name across the disk round trip"
    );
    assert_eq!(
        reparsed.description, authored.description,
        "the frontmatter must preserve the skill description across the disk round trip"
    );
    assert_eq!(
        reparsed.version, authored.version,
        "the frontmatter must preserve the optional version across the disk round trip"
    );
    assert_eq!(
        reparsed.metadata, authored.metadata,
        "metadata keys must survive flattening into and out of the YAML frontmatter"
    );
    assert_eq!(
        reparsed.instructions, authored.instructions,
        "the markdown body must round trip unchanged through the frontmatter document"
    );
    assert!(
        reparsed.resources.is_empty(),
        "from_markdown parses the document only; resources belong to the bundle reader"
    );
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn evaluated_skill_reports_pipeline_status_and_python_json_fields() {
    let skill = author_skill();
    let cases = evaluation_cases();

    let report = SkillEvaluator::new(Arc::new(scripted_runner()))
        .evaluate_at(&skill, &cases, 1_700_000_000_000)
        .await
        .expect("a valid skill and valid cases must evaluate");

    assert_eq!(
        report.functional.skill_name, skill.name,
        "the functional report must name the evaluated skill"
    );
    assert_eq!(
        report.functional.timestamp, 1_700_000_000_000,
        "evaluate_at must stamp the report with the supplied timestamp"
    );

    let statuses: Vec<CaseStatus> = report
        .functional
        .results
        .iter()
        .map(|result| result.status)
        .collect();
    assert_eq!(
        statuses,
        vec![CaseStatus::Passed, CaseStatus::Passed, CaseStatus::Failed],
        "results must follow input order with per-case all/none status derivation"
    );

    let summary = &report.functional.summary;
    assert_eq!(
        (
            summary.total,
            summary.passed,
            summary.partial,
            summary.failed
        ),
        (3, 2, 0, 1),
        "the summary must count the three executed cases without partial credit"
    );
    assert!(
        (summary.pass_rate - 2.0 / 3.0).abs() < 1e-9,
        "the pass rate must be passed / total, i.e. 2 of 3 here"
    );
    assert_eq!(
        report.functional.recommendations,
        vec!["Audience targeting".to_owned()],
        "recommendations must name only the failed case, in input order"
    );

    assert!(
        report.design.ready_for_functional_testing,
        "a rubric-complete skill must clear the design gate so the pipeline can complete"
    );
    assert_eq!(
        report.final_score.pipeline_status, "COMPLETED",
        "an open design and safety gate must complete the pipeline"
    );
    assert_eq!(
        report.final_score.gates_passed,
        vec!["e3_safety".to_owned()],
        "a completed pipeline reports the single gate-role evaluator (E3 safety)"
    );

    let expected_final =
        round4(0.40 * report.design.design_score_pct + 0.60 * report.functional.summary.pass_rate);
    assert!(
        (report.final_score.final_score_pct - expected_final).abs() < 1e-4,
        "the final score must be round4(0.40 * design_score_pct + 0.60 * pass_rate) over fractions, \
         got {}",
        report.final_score.final_score_pct
    );
    assert!(
        (report.final_score.evaluators.e2_functional.contribution
            - round4(0.60 * report.functional.summary.pass_rate))
        .abs()
            < 1e-4,
        "the functional contribution must be the weighted pass rate fraction"
    );

    assert_python_json_fields(&report);

    let blocked_config = EvaluatorConfig {
        safety_rules: vec![SafetyRule::new(
            "(?i)document brief",
            "data_leakage",
            "critical",
            "the skill body leaks the internal brief codename",
        )],
        ..EvaluatorConfig::default()
    };
    let blocked = SkillEvaluator::new(Arc::new(scripted_runner()))
        .with_config(blocked_config)
        .evaluate_at(&skill, &cases, 1_700_000_000_000)
        .await
        .expect("a matching safety rule blocks the verdict without failing the run");

    assert!(
        blocked.safety.blocked && blocked.safety.auto_fail.is_none(),
        "a critical safety finding must close the gate without claiming a coded auto-fail"
    );
    assert_eq!(
        blocked.safety.findings.len(),
        1,
        "exactly the one matching rule must produce a finding"
    );
    assert_eq!(
        blocked.final_score.pipeline_status, "BLOCKED_AT_E3",
        "the blocked pipeline must report the BLOCKED_AT_E3 status"
    );
    assert!(
        blocked.final_score.gates_passed.is_empty(),
        "a safety-blocked pipeline must not claim any passed gate"
    );
    assert!(
        (blocked.final_score.final_score_pct - 0.0).abs() < 1e-9,
        "a safety-blocked pipeline must zero the weighted final score"
    );
    assert!(
        (blocked.functional.summary.pass_rate - summary.pass_rate).abs() < 1e-9,
        "the safety gate must not rewrite the functional report of the same cases"
    );
    assert_eq!(
        blocked.functional.timestamp, 1_700_000_000_000,
        "the blocked run must keep the caller-supplied timestamp"
    );
}

#[tokio::test]
async fn model_runner_evaluates_the_authored_skill() {
    let skill = author_skill();
    let cases = evaluation_cases();

    // The same pass/pass/fail answers the scripted runner serves, but queued on
    // a model so the evaluator's own runner seam produces them.
    let scripted = ScriptedModel::builder()
        .respond_text("This brief lists the decisions in three bullets.")
        .respond_text("The brief closes with a risk note: none recorded.")
        .respond_text("A brief written for the leadership team.")
        .build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());

    let report = SkillEvaluator::new(Arc::new(ModelCaseRunner::new(model)))
        .evaluate_at(&skill, &cases, 1_700_000_000_000)
        .await
        .expect("a model-backed runner must evaluate the authored skill");

    assert_eq!(
        scripted.call_count(),
        3,
        "the model runner seam must make exactly one non-streaming call per evaluation case"
    );

    let statuses: Vec<CaseStatus> = report
        .functional
        .results
        .iter()
        .map(|result| result.status)
        .collect();
    assert_eq!(
        statuses,
        vec![CaseStatus::Passed, CaseStatus::Passed, CaseStatus::Failed],
        "the model answers must be judged in case order: pass, pass, fail"
    );
    assert!(
        (report.functional.summary.pass_rate - 2.0 / 3.0).abs() < 1e-9,
        "the pass rate must reflect 2 of the 3 model answers, got {}",
        report.functional.summary.pass_rate
    );
    assert_eq!(
        (
            report.functional.summary.passed,
            report.functional.summary.failed
        ),
        (2, 1),
        "the summary must count exactly the two passing and one failing model answer"
    );

    let calls = scripted.calls();
    assert_eq!(
        calls.len(),
        3,
        "the recorded call list must hold one call per evaluation case"
    );
    for (index, call) in calls.iter().enumerate() {
        assert_eq!(
            call.messages.len(),
            2,
            "case {index} must send exactly a system and a user message"
        );
        assert_eq!(
            call.messages[0].role,
            Role::System,
            "case {index} must open with the skill-document system message"
        );
        let system = call.messages[0].text();
        assert!(
            system.contains(&skill.name),
            "the system message of case {index} must name the authored skill: {system}"
        );
        assert!(
            system.contains(&skill.description),
            "the system message of case {index} must carry the authored description: {system}"
        );
        assert!(
            system.contains(&skill.instructions),
            "the system message of case {index} must carry the rendered instructions: {system}"
        );
        assert_eq!(
            call.messages[1].role,
            Role::User,
            "case {index} must send the case prompt as the user message"
        );
    }
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(
            calls[index].messages[1].text().as_ref(),
            case.prompt.as_str(),
            "the user message of case {index} must be the case prompt verbatim"
        );
    }
}

/// Asserts that none of the rendered Markdown lines carry trailing whitespace.
fn assert_markdown_is_tidy(document: &str, name: &str) {
    assert!(
        document.ends_with('\n') && !document.ends_with("\n\n"),
        "the {name} report must end with exactly one newline"
    );
    assert!(
        !document.contains("\n\n\n"),
        "the {name} report must separate its sections with one blank line"
    );
    for (index, line) in document.lines().enumerate() {
        assert_eq!(
            line.trim_end(),
            line,
            "line {} of the {name} report carries trailing whitespace: {line:?}",
            index + 1
        );
    }
}

/// The evaluator configuration that closes the E3 safety gate.
fn blocking_config() -> EvaluatorConfig {
    EvaluatorConfig {
        safety_rules: vec![SafetyRule::new(
            "(?i)document brief",
            "data_leakage",
            "critical",
            "the skill body leaks the internal brief codename",
        )],
        ..EvaluatorConfig::default()
    }
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn a_completed_run_writes_the_four_markdown_reports() {
    use openjiuwen_dev_tools::skill_evaluator::artifacts::{
        FINAL_REPORT_ARTIFACT, LLM_JUDGE_REPORT_ARTIFACT, PIPELINE_REPORT_ARTIFACT,
        SAFETY_REPORT_ARTIFACT,
    };

    let workspace = TempWorkspace::new().expect("temp workspace must be creatable");
    let skill = author_skill();
    let report = SkillEvaluator::new(Arc::new(scripted_runner()))
        .evaluate_at(&skill, &evaluation_cases(), 1_700_000_000_000)
        .await
        .expect("a valid skill and valid cases must evaluate");
    let dir = workspace.path("evals/skill-tests/document-brief");

    let paths = report
        .write_markdown_artifacts(&dir)
        .await
        .expect("a completed run must write its markdown reports");

    assert_eq!(
        paths.dir, dir,
        "the paths struct must name the target directory"
    );
    assert_eq!(
        paths.llm_judge,
        dir.join(LLM_JUDGE_REPORT_ARTIFACT),
        "the design report must be written as llm_judge_report.md"
    );
    assert_eq!(
        paths.safety,
        dir.join(SAFETY_REPORT_ARTIFACT),
        "the safety report must be written as skill_safety_report.md"
    );
    assert_eq!(
        paths.final_report.as_ref(),
        Some(&dir.join(FINAL_REPORT_ARTIFACT)),
        "a completed run must write final_report.md"
    );
    assert!(
        paths.pipeline_report.is_none(),
        "a completed run must not write pipeline_report.md"
    );
    assert!(
        !dir.join(PIPELINE_REPORT_ARTIFACT).exists(),
        "pipeline_report.md must be absent for a completed run"
    );

    let llm_judge = tokio::fs::read_to_string(&paths.llm_judge)
        .await
        .expect("the design report must be readable");
    assert!(
        llm_judge.starts_with("# Skill Evaluation Report: document-brief"),
        "the design report must open with the Python title: {llm_judge}"
    );
    for heading in [
        "## Summary",
        "## Dimension Scores",
        "## Critical Issues",
        "## Detailed Analysis",
    ] {
        assert!(
            llm_judge.contains(heading),
            "the design report must carry the {heading} section: {llm_judge}"
        );
    }
    assert!(
        llm_judge.contains("| Dimension | Score | Max |"),
        "the design report must render the rubric table: {llm_judge}"
    );
    assert_markdown_is_tidy(&llm_judge, "design");

    let safety = tokio::fs::read_to_string(&paths.safety)
        .await
        .expect("the safety report must be readable");
    assert!(
        safety.starts_with("# Safety Evaluation Report: document-brief"),
        "the safety report must open with the Python title: {safety}"
    );
    for heading in [
        "## Verdict",
        "## Dimension Scores",
        "## Findings",
        "## Cleared Dimensions",
    ] {
        assert!(
            safety.contains(heading),
            "the safety report must carry the {heading} section: {safety}"
        );
    }
    assert!(
        safety.contains("- **Gate**: PASS") && safety.contains("- **Auto-fail triggered**: None"),
        "a passing scan must open the gate and claim no auto-fail: {safety}"
    );
    assert_markdown_is_tidy(&safety, "safety");

    let final_report =
        tokio::fs::read_to_string(paths.final_report.as_ref().expect("final report path"))
            .await
            .expect("the final report must be readable");
    assert!(
        final_report.starts_with("# Eval Pipeline Report: document-brief"),
        "the final report must open with the Python title: {final_report}"
    );
    for heading in [
        "## Final Verdict",
        "## Evaluator Breakdown",
        "## Key Issues to Fix",
        "## Full Reports",
    ] {
        assert!(
            final_report.contains(heading),
            "the final report must carry the {heading} section: {final_report}"
        );
    }
    assert!(
        final_report.contains("**Status**: COMPLETED"),
        "the final report must state the completed pipeline status: {final_report}"
    );
    assert!(
        final_report.contains("| E1 Design (skill_llm_judge) |")
            && final_report.contains("| E2 Functional (skill_tester) |")
            && final_report.contains("| E3 Safety (skill_safety_judge) | PASS |"),
        "the evaluator breakdown must list E1, E2, and the passing E3 gate: {final_report}"
    );
    assert!(
        final_report.contains("[final_score.json](final_score.json)"),
        "the full-report links must point at the JSON artifacts: {final_report}"
    );
    assert!(
        final_report.contains("**Run timestamp**: 2023-11-14T22:13:20Z"),
        "the timestamp line must render the caller-supplied epoch milliseconds: {final_report}"
    );
    assert!(
        final_report.contains(&format!(
            "**Run timestamp**: {}",
            report.functional.timestamp_iso()
        )),
        "the timestamp line must agree with FunctionalReport::timestamp_iso"
    );
    assert_markdown_is_tidy(&final_report, "final");
}

#[tokio::test]
async fn a_blocked_run_writes_a_pipeline_report_instead_of_a_final_report() {
    use openjiuwen_dev_tools::skill_evaluator::artifacts::{
        FINAL_REPORT_ARTIFACT, PIPELINE_REPORT_ARTIFACT,
    };

    let workspace = TempWorkspace::new().expect("temp workspace must be creatable");
    let skill = author_skill();
    let report = SkillEvaluator::new(Arc::new(scripted_runner()))
        .with_config(blocking_config())
        .evaluate_at(&skill, &evaluation_cases(), 1_700_000_000_000)
        .await
        .expect("a matching safety rule blocks the verdict without failing the run");
    assert_eq!(
        report.final_score.pipeline_status, "BLOCKED_AT_E3",
        "the fixture must be blocked at the E3 safety gate"
    );
    let dir = workspace.path("evals/skill-tests/document-brief");

    let paths = report
        .write_markdown_artifacts(&dir)
        .await
        .expect("a blocked run must write its markdown reports");

    assert!(
        paths.final_report.is_none(),
        "a blocked run must not claim a completed-run report"
    );
    assert_eq!(
        paths.pipeline_report.as_ref(),
        Some(&dir.join(PIPELINE_REPORT_ARTIFACT)),
        "a blocked run must write pipeline_report.md"
    );
    assert!(
        !dir.join(FINAL_REPORT_ARTIFACT).exists(),
        "final_report.md must be absent for a blocked run"
    );

    let pipeline =
        tokio::fs::read_to_string(paths.pipeline_report.as_ref().expect("pipeline path"))
            .await
            .expect("the pipeline report must be readable");
    assert!(
        pipeline.starts_with("# Eval Pipeline Report: document-brief"),
        "the pipeline report must open with the Python title: {pipeline}"
    );
    assert!(
        pipeline.contains("**Status**: BLOCKED_AT_E3"),
        "the pipeline report must name the blocking gate: {pipeline}"
    );
    assert!(
        pipeline.contains("**Blocked by**: E3 safety gate"),
        "the pipeline report must describe the blocking gate: {pipeline}"
    );
    assert!(
        pipeline.contains("## Key Issues to Fix") && pipeline.contains("## Full Reports"),
        "the pipeline report must keep the Python sections: {pipeline}"
    );
    assert!(
        pipeline.contains(
            "- Safety: data_leakage (critical): the skill body leaks the internal brief codename"
        ),
        "the blocking finding must appear in the issue list: {pipeline}"
    );
    assert!(
        !pipeline.contains("[final_score.json](final_score.json)"),
        "a blocked run writes no final_score.json to link: {pipeline}"
    );
    assert_markdown_is_tidy(&pipeline, "pipeline");

    let safety = tokio::fs::read_to_string(&paths.safety)
        .await
        .expect("the safety report must be readable");
    assert!(
        safety.contains("- **Gate**: FAIL")
            && safety.contains("- **Blocked**: Yes")
            && safety.contains("## Findings\n- data_leakage (critical)"),
        "the safety report must show the closed gate and the finding: {safety}"
    );
    assert_markdown_is_tidy(&safety, "safety");
}

/// The `lookup` tool stub the `ReAct` runs share: it answers the audience.
fn react_stubs() -> ToolStubs {
    ToolStubs::new().with_stub(
        ToolStub::constant("lookup", json!({"audience": "release engineers"}))
            .expect("a non-blank stub name"),
    )
}

/// A model that looks the audience up once and then answers with `answer`.
fn tool_calling_model(answer: &str) -> ScriptedModel {
    ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "lookup",
            json!({"topic": "audience"}),
        ))
        .respond_text(answer)
        .build()
}

#[tokio::test]
async fn a_tool_using_case_runs_through_the_react_runner() {
    let skill = author_skill();
    let case = EvalCase::new("case-brief", "Brief structure", "Draft a brief.")
        .with_expectation("contains:release engineers");
    let answer = "This brief is written for release engineers.";

    // The traced entry point reports the tool round behind the answer.
    let model = tool_calling_model(answer);
    let runner = ReactCaseRunner::new(Arc::new(model.clone())).with_stubs(react_stubs());
    assert_eq!(
        runner.max_rounds(),
        DEFAULT_MAX_REACT_ROUNDS,
        "a fresh runner carries the documented tool-round budget"
    );

    let outcome = runner
        .run_traced(&case, &skill)
        .await
        .expect("the model answers after one tool round");
    assert_eq!(outcome.answer, answer);
    assert_eq!(outcome.rounds, 1, "one assistant turn requested tools");
    assert!(!outcome.truncated, "the budget did not cut the run short");
    assert!(outcome.used_tools(), "the model asked for a tool");
    assert!(outcome.used("lookup"));
    let record = &outcome.trace.calls()[0];
    assert_eq!(record.name, "lookup");
    assert_eq!(record.arguments, json!({"topic": "audience"}));
    assert_eq!(record.result, json!({"audience": "release engineers"}));
    assert!(!record.is_error(), "{record:?}");

    let calls = model.calls();
    assert_eq!(calls.len(), 2, "the tool result is fed back to the model");
    assert_eq!(
        calls[0]
            .request
            .tools
            .as_ref()
            .and_then(|tools| tools.as_array())
            .map(Vec::len),
        Some(1),
        "the registered stub is declared to the model"
    );
    let tool_message = calls[1]
        .messages
        .iter()
        .find(|message| message.role == Role::Tool)
        .expect("the second turn carries the tool result");
    assert_eq!(tool_message.text(), "{\"audience\":\"release engineers\"}");

    // The same runner satisfies the evaluator's `CaseRunner` seam, so the
    // functional report scores the tool-backed answer.
    let evaluator_model = tool_calling_model(answer);
    let report = SkillEvaluator::new(Arc::new(
        ReactCaseRunner::new(Arc::new(evaluator_model.clone())).with_stubs(react_stubs()),
    ))
    .evaluate_at(&skill, &[case], 1_700_000_000_000)
    .await
    .expect("a tool-backed case evaluates");
    assert_eq!(
        report.functional.results[0].status,
        CaseStatus::Passed,
        "the tool-backed answer satisfies `contains:release engineers`"
    );
    assert_eq!(report.functional.summary.passed, 1);
    assert!(
        (report.functional.summary.pass_rate - 1.0).abs() < 1e-9,
        "the assertion-based pass rate must stay 1.0, got {}",
        report.functional.summary.pass_rate
    );
    assert_eq!(
        evaluator_model.call_count(),
        2,
        "the evaluator seam drives the same tool round and answer"
    );
}

#[tokio::test]
async fn with_max_rounds_bounds_an_always_tool_calling_script() {
    let skill = author_skill();
    let case =
        EvalCase::new("case-loop", "Loop", "Draft a brief.").with_expectation("contains:brief");
    // Every turn asks for the tool, so only the budget can end the loop.
    let unbounded_turn = ScriptedTurn::Response(
        MockCompletion::tool_call("lookup", json!({"topic": "audience"})).to_chat_response(""),
    );
    let model = ScriptedModel::builder().fallback(unbounded_turn).build();
    let runner = ReactCaseRunner::new(Arc::new(model.clone()))
        .with_stubs(react_stubs())
        .with_max_rounds(2);

    let outcome = runner
        .run_traced(&case, &skill)
        .await
        .expect("the budget ends the loop instead of failing the run");
    assert!(
        outcome.truncated,
        "a caller must be able to tell a cut-short answer from a real one"
    );
    assert_eq!(outcome.rounds, 2, "at most `max_rounds` tool rounds run");
    assert_eq!(outcome.trace.count("lookup"), 2);
    assert!(outcome.used_tools());
    assert_eq!(
        model.call_count(),
        3,
        "a bounded loop costs at most `max_rounds + 1` model invocations"
    );

    let error = ReactCaseRunner::new(Arc::new(ScriptedModel::text("never reached")))
        .with_stubs(react_stubs())
        .with_max_rounds(0)
        .run_traced(&case, &skill)
        .await
        .expect_err("a zero budget is rejected before any model call");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("max_rounds"),
        "the error must name the budget: {error}"
    );
}
