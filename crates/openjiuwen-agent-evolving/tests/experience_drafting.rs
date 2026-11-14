//! Integration tests for the experience drafting pipeline, driven through the
//! crate's public re-exports.
//!
//! Covers the model reply tolerance, the Python field defaults, the repair retry,
//! the deterministic fallback (including explicit `skip` handling), the draft
//! schema validation, and the online orchestrator's status matrix and
//! auto-approval persistence. Only `ScriptedModel` touches a model seam; nothing
//! here performs I/O beyond one temporary archive.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::similar_names,
    clippy::assert_is_empty,
    clippy::match_wildcard_for_single_variants
)]

use std::sync::Arc;

use openjiuwen_agent_evolving::signal::{EvolutionSignal, EvolutionTarget as SignalTarget};
use openjiuwen_agent_evolving::{
    DraftExperience, DraftInput, DraftSection, EvolutionRequest, EvolutionStatus, EvolutionSubject,
    EvolveDraft, ExperienceArchive, ExperienceDrafter, ExperienceRecord, ExperienceSection,
    OnlineEvolutionOrchestrator,
};
use openjiuwen_common::error::Error;
use openjiuwen_core::foundation::llm::ModelRequestConfig;
use openjiuwen_test_support::{ScriptedError, ScriptedModel, ScriptedTurn};
use serde_json::json;

const SKILL: &str = "pdf-report";
const SKILL_CONTENT: &str = "# PDF report\n\nRead and summarise PDFs.";

/// An offline drafter or a model-backed one over `turns`.
fn drafter(
    turns: impl IntoIterator<Item = ScriptedTurn>,
) -> (ExperienceDrafter, Arc<ScriptedModel>) {
    let model = Arc::new(ScriptedModel::new(turns));
    let drafter = ExperienceDrafter::new().with_model(model.clone(), ModelRequestConfig::default());
    (drafter, model)
}

/// A deterministic signal with confidence `1.0`.
fn signal(excerpt: &str) -> EvolutionSignal {
    EvolutionSignal::new("execution_failure", SignalTarget::Body, excerpt).expect("valid signal")
}

/// A signal with an explicit source and confidence.
fn typed_signal(kind: &str, excerpt: &str, confidence: f64) -> EvolutionSignal {
    EvolutionSignal::new(kind, SignalTarget::Body, excerpt)
        .expect("valid signal")
        .with_confidence(confidence)
        .expect("valid confidence")
}

/// The borrowed input for `signals`.
fn draft_input(signals: &[EvolutionSignal]) -> DraftInput<'_> {
    DraftInput::new(SKILL, SKILL_CONTENT, signals)
}

/// An orchestrator request that stages its draft for approval.
fn request(signals: &[EvolutionSignal]) -> EvolutionRequest<'_> {
    EvolutionRequest::new(SKILL, SKILL_CONTENT, signals)
}

/// One append item with a summary and body.
fn item(summary: &str, content: &str) -> String {
    json!({"action": "append", "summary": summary, "content": content}).to_string()
}

// ---- reply tolerance and defaults ------------------------------------------

#[tokio::test]
async fn a_fenced_commented_reply_is_normalised_with_the_python_defaults() {
    let (drafter, model) = drafter([ScriptedTurn::text(
        "```json\n[\n  {\"action\": \"append\", \"target\": \"description\", \
         \"summary\": \"  describe   the PDF flow \", \"content\": \"lead with the flow\", \
         \"source_refs\": [\"run-1\"]}, // note\n  {\"action\": \"append\", \
         \"target\": \"script\", \"section\": \"Instructions\", \"summary\": \"ship the fixer\", \
         \"content\": \"use the fixer\", \"script_filename\": \"fix.py\", \
         \"script_language\": \"python\",},\n]\n```",
    )]);
    let signals = [signal("the parser timed out")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");

    assert_eq!(draft.subject.name, SKILL);
    assert_eq!(draft.experiences.len(), 2);
    let first = &draft.experiences[0];
    assert_eq!(first.summary, "describe the PDF flow");
    assert_eq!(first.target.as_str(), "description");
    assert_eq!(first.section, DraftSection::Instructions);
    assert_eq!(first.source_refs, vec!["run-1"]);
    // A script target forces the Scripts section even when the model names one.
    let second = &draft.experiences[1];
    assert_eq!(second.target.as_str(), "script");
    assert_eq!(second.section, DraftSection::Scripts);
    assert_eq!(second.script_filename.as_deref(), Some("fix.py"));
    assert_eq!(second.script_language.as_deref(), Some("python"));
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn a_reply_without_a_target_or_section_uses_body_and_troubleshooting() {
    let (drafter, _) = drafter([ScriptedTurn::text(format!(
        "[{}]",
        item("a lesson", "do the thing")
    ))]);
    let signals = [signal("the parser timed out")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");

    let experience = &draft.experiences[0];
    assert_eq!(experience.target.as_str(), "body");
    assert_eq!(experience.section, DraftSection::Troubleshooting);
    assert!(experience.reason.is_empty());
    assert!(experience.source_refs.is_empty());
}

#[tokio::test]
async fn an_object_envelope_with_an_experiences_array_is_accepted() {
    let (drafter, _) = drafter([ScriptedTurn::text(
        r#"{"experiences": [{"summary": "s", "content": "c"}]}"#,
    )]);
    let signals = [signal("the parser timed out")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");

    assert_eq!(draft.experiences.len(), 1);
    assert_eq!(draft.experiences[0].summary, "s");
}

#[tokio::test]
async fn a_repair_retry_recovers_a_parseable_answer() {
    let (drafter, model) = drafter([
        ScriptedTurn::text("[{\"summary\": ]"),
        ScriptedTurn::text(format!("[{}]", item("repaired", "do it"))),
    ]);
    let signals = [signal("the parser timed out")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");

    assert_eq!(draft.experiences[0].summary, "repaired");
    assert_eq!(model.call_count(), 2);
}

// ---- fallback and skip semantics ------------------------------------------

#[tokio::test]
async fn malformed_output_falls_back_to_a_non_empty_deterministic_draft() {
    let (drafter, model) = drafter([
        ScriptedTurn::text("not json at all"),
        ScriptedTurn::text("still not json"),
    ]);
    let signals = [signal("the parser timed out on large PDFs")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");

    assert!(
        !draft.is_empty(),
        "usable evidence must never draft nothing"
    );
    let experience = &draft.experiences[0];
    assert_eq!(experience.target.as_str(), "body");
    assert_eq!(experience.section, DraftSection::Troubleshooting);
    assert!(
        experience.reason.starts_with("derived from signal"),
        "{}",
        experience.reason
    );
    assert_eq!(interface_calls(&model), 2, "one answer and one repair");

    let offline = ExperienceDrafter::new()
        .draft(&draft_input(&signals))
        .await
        .expect("offline draft");
    assert_eq!(offline, draft);
}

#[tokio::test]
async fn an_explicit_all_skip_answer_is_honoured_without_a_fallback() {
    let (drafter, model) = drafter([ScriptedTurn::text(
        r#"[{"action": "skip", "skip_reason": "not reusable"}]"#,
    )]);
    let signals = [signal("the parser timed out")];
    let draft = drafter.draft(&draft_input(&signals)).await.expect("draft");

    assert!(draft.is_empty(), "a deliberate rejection must survive");
    assert_eq!(interface_calls(&model), 1);
    // The fallback would have produced a lesson, so the empty draft proves the
    // skip was not overridden.
    assert!(!ExperienceDrafter::new()
        .draft(&draft_input(&signals))
        .await
        .expect("offline draft")
        .is_empty());
}

#[tokio::test]
async fn model_errors_fall_back_to_the_deterministic_draft() {
    let (drafter, _) = drafter([ScriptedTurn::Fail(ScriptedError::llm("provider down"))]);
    let signals = [signal("the parser timed out")];
    let draft = drafter
        .draft(&draft_input(&signals))
        .await
        .expect("fallback");

    assert!(!draft.is_empty());
    assert_eq!(draft.experiences[0].target.as_str(), "body");
}

// ---- draft schema validation ----------------------------------------------

#[test]
fn a_script_filename_with_a_path_separator_is_a_validation_error() {
    for filename in ["scripts/fix.py", "scripts\\fix.py", "C:\\fix.py", ".."] {
        let error = DraftExperience::normalize(&json!({
            "summary": "s",
            "content": "c",
            "script_filename": filename,
        }))
        .expect_err("a path is not a file name");
        assert!(
            matches!(error, Error::Validation(_)),
            "{filename}: {error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("script_filename must be a file name"),
            "{filename}: {error}"
        );
    }
}

#[test]
fn an_over_long_model_summary_is_rejected_by_the_schema() {
    let too_long = "x".repeat(161);
    let error = DraftExperience::normalize(&json!({
        "summary": too_long,
        "content": "c",
    }))
    .expect_err("over the summary limit");

    assert!(
        error.to_string().contains("at most 160 characters"),
        "{error}"
    );
}

#[tokio::test]
async fn the_deterministic_summary_is_bounded_by_the_configured_limit() {
    let signals = [signal(&"a very long lesson ".repeat(30))];
    let draft = ExperienceDrafter::new()
        .with_max_summary_chars(24)
        .draft(&draft_input(&signals))
        .await
        .expect("draft");

    assert_eq!(draft.experiences[0].summary.chars().count(), 24);
}

#[tokio::test]
async fn max_items_caps_the_model_draft() {
    let reply = format!(
        "[{}, {}, {}]",
        item("one", "first body"),
        item("two", "second body"),
        item("three", "third body")
    );
    let (drafter, _) = drafter([ScriptedTurn::text(reply)]);
    let signals = [signal("the parser timed out")];
    let draft = drafter
        .with_max_items(2)
        .draft(&draft_input(&signals))
        .await
        .expect("draft");

    assert_eq!(draft.experiences.len(), 2);
    assert_eq!(draft.experiences[1].summary, "two");
}

#[tokio::test]
async fn a_blank_skill_name_is_a_validation_error() {
    let signals = [signal("the parser timed out")];
    let error = ExperienceDrafter::new()
        .draft(&DraftInput::new("   ", SKILL_CONTENT, &signals))
        .await
        .expect_err("a blank skill is rejected");

    assert!(matches!(error, Error::Validation(_)), "{error:?}");
}

// ---- orchestrator ----------------------------------------------------------

#[tokio::test]
async fn the_orchestrator_prechecks_return_skipped_statuses() {
    let signals = [signal("the parser timed out")];
    let none: [EvolutionSignal; 0] = [];
    let mut orchestrator =
        OnlineEvolutionOrchestrator::new().with_archive(ExperienceArchive::new());

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
    assert!(orchestrator.archive().is_empty());
}

#[tokio::test]
async fn the_orchestrator_stages_then_auto_approves_the_same_evidence() {
    let signals = [signal("the parser timed out on large PDFs")];
    let mut orchestrator = OnlineEvolutionOrchestrator::new();

    let staged = orchestrator
        .evolve(request(&signals))
        .await
        .expect("result");
    assert_eq!(staged.status, EvolutionStatus::Staged);
    assert_eq!(staged.staged.len(), 1);
    assert!(staged.committed.is_empty());
    assert!(orchestrator.archive().is_empty());

    let auto = orchestrator
        .evolve(request(&signals).with_requires_approval(false))
        .await
        .expect("result");
    assert_eq!(auto.status, EvolutionStatus::AutoApproved);
    assert_eq!(auto.committed.len(), 1);
    assert_eq!(auto.staged.len(), 0);
}

#[tokio::test]
async fn low_confidence_evidence_produces_no_records() {
    let signals = [typed_signal("execution_failure", "weak hint", 0.1)];
    let mut orchestrator = OnlineEvolutionOrchestrator::new();

    let result = orchestrator
        .evolve(request(&signals).with_requires_approval(false))
        .await
        .expect("result");

    assert_eq!(result.status, EvolutionStatus::NoEvolutionNoRecords);
    assert!(result.reason.contains("no applied updates"));
    assert!(orchestrator.archive().is_empty());
}

#[tokio::test]
async fn auto_approval_persists_into_the_experience_archive() {
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
    assert_eq!(record.record_id, result.committed[0]);
    assert_eq!(record.skill_name, SKILL);
    assert_eq!(record.source, "user_intent");
    assert_eq!(record.confidence, 0.70);
    assert_eq!(record.section, ExperienceSection::Body);
    assert!(!record.keywords.is_empty());
    assert_eq!(
        record
            .metadata
            .get("summary")
            .and_then(|value| value.as_str()),
        Some("users want the parser to keep tables")
    );

    // The same evidence again merges rather than duplicating.
    let repeat = orchestrator
        .evolve(request(&signals).with_requires_approval(false))
        .await
        .expect("result");
    assert_eq!(repeat.committed, result.committed);
    assert_eq!(orchestrator.archive().len(), 1);
}

#[tokio::test]
async fn the_orchestrator_drafts_with_a_model_and_stages_the_result() {
    let (drafter, model) = drafter([ScriptedTurn::text(format!(
        "[{}]",
        item("model lesson", "model body")
    ))]);
    let signals = [signal("the parser timed out")];
    let mut orchestrator = OnlineEvolutionOrchestrator::with_drafter(drafter);

    let result = orchestrator
        .evolve(request(&signals))
        .await
        .expect("result");

    assert_eq!(result.status, EvolutionStatus::Staged);
    assert_eq!(result.staged[0].experiences[0].summary, "model lesson");
    assert_eq!(interface_calls(&model), 1);
    assert!(orchestrator.archive().is_empty());
}

#[tokio::test]
async fn auto_approval_persists_a_model_draft_with_its_reason_metadata() {
    let (drafter, _) = drafter([ScriptedTurn::text(format!(
        "[{}]",
        json!({"action": "append", "summary": "s", "content": "c", "reason": "from the run"})
    ))]);
    let signals = [signal("the parser timed out")];
    let mut orchestrator = OnlineEvolutionOrchestrator::with_drafter(drafter);

    let result = orchestrator
        .evolve(request(&signals).with_requires_approval(false))
        .await
        .expect("result");

    assert_eq!(result.status, EvolutionStatus::AutoApproved);
    let record = &orchestrator.archive().records()[0];
    assert_eq!(
        record
            .metadata
            .get("reason")
            .and_then(|value| value.as_str()),
        Some("from the run")
    );
}

#[tokio::test]
async fn generation_and_persistence_failures_are_statuses() {
    let signals = [signal("the parser timed out")];
    let request = request(&signals);
    let mut orchestrator = OnlineEvolutionOrchestrator::new();

    let generated = orchestrator.complete(&request, Err(Error::llm("provider down")));
    assert_eq!(generated.status, EvolutionStatus::GenerationFailed);
    assert_eq!(generated.reason, "llm: provider down");

    let blank = EvolveDraft::new(
        EvolutionSubject::for_skill(SKILL).expect("subject"),
        vec![DraftExperience {
            summary: String::from("summary"),
            content: String::from("   "),
            ..DraftExperience::default()
        }],
    );
    let persisted =
        orchestrator.complete(&request.clone().with_requires_approval(false), Ok(blank));
    assert_eq!(persisted.status, EvolutionStatus::PersistenceFailed);
    assert!(persisted.committed.is_empty());
}

/// Seeds an archive with one record, proving the seeded record still counts.
#[tokio::test]
async fn a_seeded_archive_is_preserved_across_an_auto_approved_run() {
    let seeded =
        ExperienceRecord::new("ev-seed", SKILL, ExperienceSection::Body, "a seeded lesson")
            .expect("valid record");
    let mut archive = ExperienceArchive::new();
    archive.add(seeded).unwrap();
    let mut orchestrator = OnlineEvolutionOrchestrator::new().with_archive(archive);

    let signals = [typed_signal("user_intent", "keep the tables", 1.0)];
    let result = orchestrator
        .evolve(request(&signals).with_requires_approval(false))
        .await
        .expect("result");

    assert_eq!(result.status, EvolutionStatus::AutoApproved);
    assert_eq!(orchestrator.archive().len(), 2);
    assert!(orchestrator
        .archive()
        .records()
        .iter()
        .any(|record| record.record_id == "ev-seed"));
}

/// The number of model turns consumed, read without touching the record internals.
fn interface_calls(model: &ScriptedModel) -> usize {
    model.call_count()
}
