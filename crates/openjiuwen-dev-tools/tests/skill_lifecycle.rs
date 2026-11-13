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

use openjiuwen_dev_tools::skill_creator::{
    SkillBundle, SkillCreator, SkillDefinition, SkillResource, SkillSpec,
};
use openjiuwen_dev_tools::skill_evaluator::{
    CaseStatus, EvalCase, EvaluationReport, EvaluatorConfig, ModelCaseRunner, SafetyRule,
    ScriptedCaseRunner, SkillEvaluator,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::Role;
use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
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
