//! Test-only suite for the experience draft pipeline.
//!
//! Mirrors what would otherwise live in `tests/experience_draft.rs`: it drives
//! the model path through `ScriptedModel` (happy path,
//! fenced/commented/trailing-comma tolerance, repair retry, malformed output),
//! the deterministic fallback, summary truncation, script filename rejection,
//! simplify merge-id validation, the orchestrator status matrix and
//! auto-approval persistence.
//!
//! Everything is deterministic: no clock, no network, no environment reads.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assert_is_empty
)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_core::foundation::llm::ModelRequestConfig;
use openjiuwen_test_support::{ScriptedError, ScriptedModel, ScriptedTurn};
use serde_json::json;

use super::{
    normalize_evolve_draft, normalize_simplify_draft, normalize_subject, parse_draft_text,
    validate_simplify_record_refs, DraftExperience, DraftInput, DraftSection, EvolutionRequest,
    EvolutionStatus, EvolutionSubject, EvolutionTarget, EvolveDraft, ExperienceDrafter,
    ExperienceRecord, ExperienceSection, OnlineEvolutionOrchestrator, SimplifyActionKind,
    SUMMARY_LIMIT,
};
use crate::signal::{EvolutionSignal, EvolutionTarget as SignalTarget};
use crate::trajectory::{StepKind, StepStatus, Trajectory, TrajectoryOutcome, TrajectoryStep};

// ---- helpers ---------------------------------------------------------------

fn drafter(
    turns: impl IntoIterator<Item = ScriptedTurn>,
) -> (ExperienceDrafter, Arc<ScriptedModel>) {
    let model = Arc::new(ScriptedModel::new(turns));
    let drafter = ExperienceDrafter::new().with_model(model.clone(), ModelRequestConfig::default());
    (drafter, model)
}

fn subject() -> EvolutionSubject {
    EvolutionSubject::for_skill("pdf-report").expect("valid subject")
}

fn signal(excerpt: &str) -> EvolutionSignal {
    EvolutionSignal::new("execution_failure", SignalTarget::Body, excerpt).expect("valid signal")
}

fn typed_signal(kind: &str, excerpt: &str, confidence: f64) -> EvolutionSignal {
    EvolutionSignal::new(kind, SignalTarget::Body, excerpt)
        .expect("valid signal")
        .with_confidence(confidence)
        .expect("valid confidence")
}

fn failed_trajectory(id: &str) -> Trajectory {
    let mut trajectory = Trajectory::new(id)
        .expect("valid id")
        .with_outcome(TrajectoryOutcome::Failure);
    trajectory.push(
        TrajectoryStep::new(StepKind::ToolCall, "call the pdf reader").with_tool_name("pdf_reader"),
    );
    trajectory.push(
        TrajectoryStep::new(StepKind::ToolResult, "timeout after 30s")
            .with_status(StepStatus::Error),
    );
    trajectory
}

fn request(signals: &[EvolutionSignal]) -> EvolutionRequest<'_> {
    EvolutionRequest::new(
        "pdf-report",
        "# PDF report\n\nRead and summarise PDFs.",
        signals,
    )
}

fn draft_input(signals: &[EvolutionSignal]) -> DraftInput<'_> {
    DraftInput::new(
        "pdf-report",
        "# PDF report\n\nRead and summarise PDFs.",
        signals,
    )
}

// ---- draft schema ----------------------------------------------------------

#[test]
fn subject_normalization_and_payload_omit_an_empty_scope() {
    let normalized = normalize_subject(&json!({
        "kind": "team-skill",
        "name": "  squad  ",
        "unknown": true
    }))
    .expect("valid subject");
    assert_eq!(normalized.kind.as_str(), "swarm-skill");
    assert_eq!(
        normalized.payload(),
        json!({"kind": "swarm-skill", "name": "squad"})
    );
    let empty_scope = subject().with_scope(BTreeMap::new()).payload();
    assert_eq!(empty_scope, json!({"kind": "skill", "name": "pdf-report"}));
    assert!(normalize_subject(&json!({"kind": "agent", "name": "x"})).is_err());
}

#[test]
fn script_filename_accepts_a_bare_name_and_rejects_paths() {
    let accepted = normalize_evolve_draft(
        subject(),
        &[json!({
            "summary": "s",
            "content": "c",
            "target": "script",
            "script_filename": "fix_parser.py",
            "script_language": "python"
        })],
    )
    .expect("valid draft");
    assert_eq!(
        accepted.experiences[0].script_filename.as_deref(),
        Some("fix_parser.py")
    );
    assert_eq!(accepted.experiences[0].section, DraftSection::Scripts);

    for filename in [
        ".",
        "..",
        "/tmp/fix.py",
        "scripts/fix.py",
        "scripts\\fix.py",
        "C:\\fix.py",
        "\\\\server\\share\\fix.py",
    ] {
        let error = normalize_evolve_draft(
            subject(),
            &[json!({
                "summary": "s",
                "content": "c",
                "script_filename": filename
            })],
        )
        .expect_err("path rejected");
        assert!(
            error
                .to_string()
                .contains("script_filename must be a file name"),
            "{filename}: {error}"
        );
    }
}

#[test]
fn drafts_round_trip_and_the_views_differ_only_in_source_refs() {
    let draft = normalize_evolve_draft(
        subject(),
        &[json!({
            "summary": "s",
            "content": "c",
            "source_refs": ["run-1", "—"]
        })],
    )
    .expect("valid draft");
    let encoded = serde_json::to_string(&draft).expect("serializable");
    let decoded: EvolveDraft = serde_json::from_str(&encoded).expect("deserializable");
    assert_eq!(decoded, draft);
    assert_eq!(
        draft.approval_view()["experiences"][0]["source_refs"],
        json!(["run-1"])
    );
    assert_eq!(
        draft.persistence_view()["experiences"][0].get("source_refs"),
        None
    );
    let minimal: EvolveDraft =
        serde_json::from_value(json!({"subject": {"name": "x"}})).expect("tolerant");
    assert!(minimal.experiences.is_empty());
}

#[test]
fn simplify_merge_ids_are_validated_against_existing_records() {
    let draft = normalize_simplify_draft(
        subject(),
        &[json!({
            "action": "merge",
            "record_id": "ev-1",
            "new_content": "merged",
            "merge_remove_ids": ["ev-2"]
        })],
    )
    .expect("valid draft");
    let known: BTreeSet<String> = ["ev-1", "ev-2"].into_iter().map(str::to_owned).collect();
    validate_simplify_record_refs(&draft, &known).expect("known ids");
    let missing: BTreeSet<String> = ["ev-1"].into_iter().map(str::to_owned).collect();
    let error = validate_simplify_record_refs(&draft, &missing).expect_err("missing id");
    assert_eq!(error.to_string(), "validation: record not found: ev-2");
    assert!(normalize_simplify_draft(
        subject(),
        &[json!({
            "action": "MERGE",
            "record_id": "ev-1",
            "new_content": "x",
            "merge_remove_ids": ["ev-1"]
        })]
    )
    .is_err());
}

// ---- model-driven drafting -------------------------------------------------

#[tokio::test]
async fn model_happy_path_normalizes_the_draft() {
    let (drafter, model) = drafter([ScriptedTurn::text(
        "[{\"action\":\"append\",\"target\":\"description\",\"summary\":\"  describe the PDF flow \",\"content\":\"lead with the flow\",\"source_refs\":[\"run-1\"]},{\"action\":\"append\",\"target\":\"script\",\"section\":\"Instructions\",\"summary\":\"ship the fixer\",\"content\":\"use the fixer\",\"script_filename\":\"fix.py\",\"script_language\":\"python\",\"keywords\":[\"pdf\"]}]",
    )]);
    let signals = [signal("the parser timed out")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");
    assert_eq!(draft.subject.name, "pdf-report");
    assert_eq!(draft.experiences.len(), 2);
    assert_eq!(draft.experiences[0].summary, "describe the PDF flow");
    assert_eq!(draft.experiences[0].target, EvolutionTarget::Description);
    assert_eq!(draft.experiences[0].section, DraftSection::Instructions);
    assert_eq!(draft.experiences[0].source_refs, vec!["run-1"]);
    // A script target forces the Scripts section even when the model names one.
    assert_eq!(draft.experiences[1].target, EvolutionTarget::Script);
    assert_eq!(draft.experiences[1].section, DraftSection::Scripts);
    assert_eq!(
        draft.experiences[1].script_filename.as_deref(),
        Some("fix.py")
    );
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn fenced_commented_and_trailing_comma_replies_are_tolerated() {
    let (drafter, _) = drafter([ScriptedTurn::text(
        "```json\n[\n  {\"action\": \"append\", \"summary\": \"s\", \"content\": \"c\", // note\n   \"source_refs\": [\"https://example.test/run\"],},\n]\n```",
    )]);
    let signals = [signal("the parser timed out")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");
    assert_eq!(draft.experiences.len(), 1);
    assert_eq!(
        draft.experiences[0].source_refs,
        vec!["https://example.test/run"]
    );
}

#[tokio::test]
async fn repair_retry_recovers_a_parseable_answer() {
    let (drafter, model) = drafter([
        ScriptedTurn::text("[{\"summary\": ]"),
        ScriptedTurn::text(
            "[{\"action\":\"append\",\"summary\":\"repaired\",\"content\":\"do it\"}]",
        ),
    ]);
    let signals = [signal("the parser timed out")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");
    assert_eq!(draft.experiences[0].summary, "repaired");
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn malformed_output_falls_back_to_a_non_empty_deterministic_draft() {
    let (drafter, model) = drafter([
        ScriptedTurn::text("not json at all"),
        ScriptedTurn::text("still not json"),
    ]);
    let signals = [signal("the parser timed out on large PDFs")];
    let first = drafter.draft(&draft_input(&signals)).await.expect("first");
    assert!(
        !first.is_empty(),
        "never an empty draft for usable evidence"
    );
    assert_eq!(first.experiences[0].target, EvolutionTarget::Body);
    assert_eq!(first.experiences[0].section, DraftSection::Troubleshooting);
    assert!(first.experiences[0].reason.contains("confidence 1.00"));
    assert_eq!(model.call_count(), 2, "one answer and one repair attempt");

    // An offline drafter produces the same draft (determinism).
    let offline = ExperienceDrafter::new()
        .draft(&draft_input(&signals))
        .await
        .expect("offline");
    assert_eq!(offline, first);
}

#[tokio::test]
async fn model_errors_fall_back_and_explicit_skips_are_honoured() {
    let (failing, _) = drafter([
        ScriptedTurn::Fail(ScriptedError::llm("provider down")),
        ScriptedTurn::Fail(ScriptedError::llm("provider down")),
    ]);
    let signals = [signal("the parser timed out")];
    assert!(!failing
        .draft(&draft_input(&signals))
        .await
        .expect("fallback")
        .is_empty());

    let (skipping, _) = drafter([ScriptedTurn::text(
        "[{\"action\":\"skip\",\"skip_reason\":\"irrelevant\"}]",
    )]);
    assert!(skipping
        .draft(&draft_input(&signals))
        .await
        .expect("skip")
        .is_empty());
}

#[tokio::test]
async fn no_model_draft_uses_signals_and_trajectories() {
    let signals = [signal("the parser timed out on large PDFs")];
    let trajectories = [failed_trajectory("run-1")];
    let drafter = ExperienceDrafter::new();
    assert!(!drafter.has_model());
    let draft = drafter
        .draft(
            &draft_input(&signals)
                .with_trajectories(&trajectories)
                .with_user_query("harden the parser"),
        )
        .await
        .expect("draft");
    assert_eq!(draft.experiences.len(), 2);
    assert!(draft.experiences[0]
        .reason
        .starts_with("derived from signal"));
    assert!(draft.experiences[1]
        .reason
        .starts_with("derived from trajectory"));
    assert!(draft
        .experiences
        .iter()
        .all(|item| item.target == EvolutionTarget::Body));
}

#[tokio::test]
async fn summary_truncation_respects_the_configured_limit() {
    let signals = [signal(&"a very long lesson ".repeat(30))];
    let drafter = ExperienceDrafter::new().with_max_summary_chars(24);
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");
    let summary = &draft.experiences[0].summary;
    assert_eq!(summary.chars().count(), 24);
    assert_eq!(SUMMARY_LIMIT, 160);

    // The schema itself rejects an over-long model summary instead of truncating.
    let too_long = "x".repeat(SUMMARY_LIMIT + 1);
    let error = DraftExperience::normalize(&json!({
        "summary": too_long,
        "content": "c"
    }))
    .expect_err("over limit");
    assert!(error.to_string().contains("at most 160 characters"));
}

// ---- simplify --------------------------------------------------------------

#[tokio::test]
async fn simplify_accepts_a_model_answer_and_falls_back_on_unknown_ids() {
    let records = [
        ExperienceRecord::new(
            "ev-1",
            "pdf-report",
            ExperienceSection::Body,
            "first lesson",
        )
        .expect("valid record"),
        ExperienceRecord::new(
            "ev-2",
            "pdf-report",
            ExperienceSection::Body,
            "second lesson",
        )
        .expect("valid record"),
    ];
    let (known, model) = drafter([ScriptedTurn::text(
        "[{\"action\":\"MERGE\",\"record_id\":\"ev-1\",\"new_content\":\"merged lesson\",\"merge_remove_ids\":[\"ev-2\"],\"reason\":\"duplicate\"}]",
    )]);
    assert!(known.has_model());
    let draft = known
        .simplify(&subject(), &records, "tighten the library")
        .await
        .expect("simplify");
    assert_eq!(model.call_count(), 1);
    assert_eq!(draft.actions.len(), 1);
    assert_eq!(draft.actions[0].action, SimplifyActionKind::Merge);
    assert_eq!(
        draft.actions[0].new_content.as_deref(),
        Some("merged lesson")
    );

    let (unknown, _) = drafter([ScriptedTurn::text(
        "[{\"action\":\"DELETE\",\"record_id\":\"ev-99\",\"reason\":\"gone\"}]",
    )]);
    let fallback = unknown
        .simplify(&subject(), &records, "")
        .await
        .expect("fallback");
    assert!(fallback
        .actions
        .iter()
        .all(|action| action.action == SimplifyActionKind::Keep));
}

// ---- orchestrator ----------------------------------------------------------

#[tokio::test]
async fn orchestrator_status_matrix_covers_every_status() {
    let signals = [signal("the parser timed out on large PDFs")];
    let none: [EvolutionSignal; 0] = [];
    let mut orchestrator = OnlineEvolutionOrchestrator::new();

    let no_input = orchestrator.evolve(request(&none)).await.expect("result");
    assert_eq!(no_input.status, EvolutionStatus::SkippedNoInput);
    assert!(no_input.subject.is_none());

    let missing = orchestrator
        .evolve(request(&signals).with_skill_exists(false))
        .await
        .expect("result");
    assert_eq!(missing.status, EvolutionStatus::SkippedSkillNotFound);
    assert!(missing.subject.is_some());

    let undefined = orchestrator
        .evolve(request(&signals).with_definition(false))
        .await
        .expect("result");
    assert_eq!(
        undefined.status,
        EvolutionStatus::SkippedSkillDefinitionNotFound
    );

    let generated = orchestrator.complete(&request(&signals), Err(Error::llm("provider down")));
    assert_eq!(generated.status, EvolutionStatus::GenerationFailed);
    assert_eq!(generated.reason, "llm: provider down");

    let weak = [typed_signal("execution_failure", "weak hint", 0.1)];
    let no_records = orchestrator
        .evolve(request(&weak).with_requires_approval(false))
        .await
        .expect("result");
    assert_eq!(no_records.status, EvolutionStatus::NoEvolutionNoRecords);

    let staged = orchestrator
        .evolve(request(&signals))
        .await
        .expect("result");
    assert_eq!(staged.status, EvolutionStatus::Staged);
    assert_eq!(staged.staged.len(), 1);
    assert!(orchestrator.archive().is_empty());

    let auto = orchestrator
        .evolve(request(&signals).with_requires_approval(false))
        .await
        .expect("result");
    assert_eq!(auto.status, EvolutionStatus::AutoApproved);
    assert_eq!(auto.committed.len(), 1);

    let blank = EvolveDraft::new(
        EvolutionSubject::for_skill("pdf-report").expect("subject"),
        vec![DraftExperience {
            summary: "summary".to_owned(),
            content: "   ".to_owned(),
            ..DraftExperience::default()
        }],
    );
    let broken = orchestrator.complete(&request(&signals).with_requires_approval(false), Ok(blank));
    assert_eq!(broken.status, EvolutionStatus::PersistenceFailed);
    assert!(broken.committed.is_empty());
}

#[tokio::test]
async fn auto_approval_persists_records_with_the_source_initial_score() {
    let signals = [typed_signal(
        "user_intent",
        "users want the parser to keep tables",
        1.0,
    )];
    let mut orchestrator = OnlineEvolutionOrchestrator::new();
    let result = orchestrator
        .evolve(request(&signals).with_requires_approval(false))
        .await
        .expect("result");
    assert_eq!(result.status, EvolutionStatus::AutoApproved);
    let records = orchestrator.archive().records();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(
        Some(record.record_id.as_str()),
        result.committed.first().map(String::as_str)
    );
    assert_eq!(record.skill_name, "pdf-report");
    assert_eq!(record.source, "user_intent");
    assert_eq!(record.confidence, 0.70);
    assert_eq!(record.section, ExperienceSection::Body);
    assert_eq!(
        record
            .metadata
            .get("summary")
            .and_then(|value| value.as_str()),
        Some("users want the parser to keep tables")
    );
    assert!(!record.keywords.is_empty());

    // The same evidence again merges instead of duplicating.
    let repeat = orchestrator
        .evolve(request(&signals).with_requires_approval(false))
        .await
        .expect("result");
    assert_eq!(repeat.committed, result.committed);
    assert_eq!(orchestrator.archive().len(), 1);
}

#[test]
fn draft_parsing_is_available_without_a_model() {
    let parse = parse_draft_text(&subject(), "[{\"summary\":\"s\",\"content\":\"c\"}]")
        .expect("pure parse");
    assert_eq!(parse.appended, 1);
    assert_eq!(parse.skipped, 0);
    assert_eq!(
        parse.draft.experiences[0].section,
        DraftSection::Troubleshooting
    );
    assert!(parse_draft_text(&subject(), "prose only").is_err());
}
