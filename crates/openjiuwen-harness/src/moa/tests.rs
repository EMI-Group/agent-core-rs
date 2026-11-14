//! Offline end-to-end suite for the mixture-of-agents module.
//!
//! Every scenario is deterministic and network-free: the reference, verifier and aggregator
//! models are [`openjiuwen_test_support::ScriptedModel`] handles, the skill root lives in a
//! `tempfile` directory, and cancellation is driven by an explicit
//! [`CancellationToken`]. The suite walks the whole module the way a host uses it —
//! consultation fan-out, structural ranking, aggregation, the harness rail, redaction,
//! configuration validation, persistence and the consultation skill — instead of testing one
//! module in isolation (each module owns its own unit tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_agent::RailEvent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::{BaseMessage, Model};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use tokio_util::sync::CancellationToken;

use crate::rails::{prompt_sections, HarnessRail, HarnessRailContext};
use crate::skills::SkillLibrary;

use super::aggregate::{AggregationStrategy, Aggregator};
use super::builtin_skills::{materialize_builtin_skill, CONSULT_SKILL_NAME};
use super::config::{JevConfig, MoaConfig, MoaMode, VerifierSelection};
use super::consultation::consult;
use super::prompts::AGGREGATOR_SYSTEM_PROMPT;
use super::rail::{MoaRail, MOA_OUTCOME_EXTRA_KEY, MOA_PROMPT_SECTION, MOA_STATE_EXTRA_KEY};
use super::redaction::redact;
use super::skill_mount::ConsultationSkillMount;
use super::state::{
    ConsultationHistory, ConsultationOutcome, ConsultationRecord, MoaSnapshot, MoaState,
    ProposalSource,
};
use super::verifier::{rank_with_config, score_progress, score_proposal};

/// Wraps a scripted model as an injected reference handle.
fn reference(model: ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model)
}

/// Builds a user message.
fn user(text: &str) -> BaseMessage {
    BaseMessage::user(text)
}

/// Runs consultation, ranking and labelled aggregation as a host would.
#[tokio::test]
async fn full_pipeline_ranks_then_integrates_the_retained_proposals() {
    let problem = "fix the failing build in src/main.rs";
    let config = MoaConfig::new(vec![
        reference(ScriptedModel::text(
            "1. Read src/main.rs and fix the failing build.\n- verify with cargo test",
        )),
        reference(ScriptedModel::text("Short answer: rebuild the project.")),
        reference(ScriptedModel::new([ScriptedTurn::fail(
            "reference timed out",
        )])),
    ]);
    let cancel = CancellationToken::new();

    let batch = consult(&config, &[user(problem)], &cancel).await.unwrap();
    assert_eq!(batch.total_models, 3);
    assert_eq!(batch.success_count, 2);
    assert_eq!(batch.failures.len(), 1);
    assert_eq!(batch.failures[0].index, 3);
    assert_eq!(batch.sources.len(), 2);
    assert_eq!(batch.sources[0].index, 1);
    assert_eq!(batch.sources[1].index, 2);

    let ranking = rank_with_config(&config, problem, &batch.sources)
        .await
        .unwrap();
    assert_eq!(ranking.len(), batch.sources.len());
    for entry in &ranking {
        assert!(
            (0.0..=1.0).contains(&entry.score),
            "score out of range: {}",
            entry.score
        );
        assert_ne!(entry.reasons, Vec::<String>::new());
    }
    for pair in ranking.windows(2) {
        let ordering = pair[0].score.total_cmp(&pair[1].score);
        assert!(
            ordering.is_gt() || (ordering.is_eq() && pair[0].index < pair[1].index),
            "ranking must be descending by score, then by index: {ranking:?}"
        );
    }

    let aggregate = Aggregator::new(AggregationStrategy::Labelled)
        .aggregate(&config, problem, &batch.sources, &cancel)
        .await
        .unwrap();
    assert!(aggregate.starts_with("## Integrated answer (mixture of agents)"));
    let first = aggregate.find(&batch.sources[0].text).unwrap();
    let second = aggregate.find(&batch.sources[1].text).unwrap();
    assert!(first < second, "the aggregate keeps the configured order");
}

/// A batch in which every reference fails is an LLM error, not an empty outcome.
#[tokio::test]
async fn all_references_failing_is_an_llm_error() {
    let config = MoaConfig::new(vec![
        reference(ScriptedModel::new([ScriptedTurn::fail("first boom")])),
        reference(ScriptedModel::new([ScriptedTurn::fail("second boom")])),
    ]);

    let error = consult(&config, &[user("fix the build")], &CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.to_string().contains("All reference models failed"));
    assert!(error.to_string().contains("first boom"));
}

/// Cancellation stops the fan-out before any proposal is retained.
#[tokio::test]
async fn cancellation_is_reported_as_cancelled() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    let config = MoaConfig::new(vec![reference(ScriptedModel::text("unused"))]);

    let error = consult(&config, &[user("fix the build")], &cancel)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}

/// Synthesis uses the injected aggregator model and sends it the proposals.
#[tokio::test]
async fn synthesis_aggregation_goes_through_the_scripted_model() {
    let aggregator_model = ScriptedModel::text("Synthesised answer");
    let config = MoaConfig::new(vec![reference(ScriptedModel::text("proposal"))])
        .with_aggregator_model(Arc::new(aggregator_model.clone()));
    let sources = vec![
        ProposalSource::new(1, "first proposal body"),
        ProposalSource::new(2, "second proposal body"),
    ];

    let answer = Aggregator::new(AggregationStrategy::Synthesis)
        .aggregate(
            &config,
            "fix the build",
            &sources,
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(answer, "Synthesised answer");

    let calls = aggregator_model.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].messages.len(), 2);
    assert_eq!(calls[0].messages[0].text(), AGGREGATOR_SYSTEM_PROMPT);
    let request = calls[0].messages[1].text();
    assert!(request.contains("fix the build"));
    assert!(request.contains("first proposal body"));
    assert!(request.contains("second proposal body"));
}

/// The rail publishes its context, stores the outcome, and retires its state.
#[tokio::test]
async fn rail_end_to_end_publishes_consults_and_clears() {
    let rail = MoaRail::new(MoaConfig::new(vec![
        reference(ScriptedModel::text(
            "- Fix src/main.rs\n- Run cargo test to verify",
        )),
        reference(ScriptedModel::text("Rebuild the project.")),
    ]))
    .unwrap();
    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
    ctx.base.messages = vec![user("fix the failing build")];

    rail.before_invoke(&mut ctx).await.unwrap();
    let sections = prompt_sections(&ctx);
    assert_eq!(sections.len(), 1);
    assert_eq!(sections[0].name, MOA_PROMPT_SECTION);
    assert!(ctx.extra().contains_key(MOA_STATE_EXTRA_KEY));
    assert!(rail.snapshot().is_some());
    assert_eq!(rail.history(), Vec::new());

    let outcome = rail
        .consult(&ctx.base.messages, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(outcome.success_count, 2);
    assert_eq!(outcome.ranking.len(), 2);
    assert!(outcome
        .aggregate
        .as_deref()
        .unwrap()
        .contains("Integrated answer"));
    assert_eq!(rail.history().len(), 1);
    assert_eq!(rail.history()[0].problem, "fix the failing build");
    assert_eq!(rail.last_outcome(), Some(outcome.clone()));
    assert_eq!(rail.snapshot().unwrap().reference_calls, 2);

    rail.before_model_call(&mut ctx).await.unwrap();
    assert_eq!(prompt_sections(&ctx).len(), 1);
    let stored: ConsultationOutcome =
        serde_json::from_value(ctx.extra()[MOA_OUTCOME_EXTRA_KEY].clone()).unwrap();
    assert_eq!(stored, outcome);

    rail.after_invoke(&mut ctx).await.unwrap();
    assert!(rail.snapshot().is_none());
    assert_eq!(rail.last_outcome(), Some(outcome));
    assert_eq!(rail.history().len(), 1);
}

/// Credentials never reach the verifier request or the ranking reasons.
#[tokio::test]
async fn redaction_keeps_secrets_out_of_verification_but_keeps_ranking() {
    let problem = "rotate api_key=abc123def now";
    let source = ProposalSource::new(
        1,
        "Use sk-secret-123 and api_key=abc123def to fix the failing build in src/main.rs",
    );

    let ranked = score_proposal(problem, &source);
    assert!(ranked.score > 0.0, "the proposal must still be ranked");
    for reason in &ranked.reasons {
        assert!(!reason.contains("sk-secret"), "reason leaked: {reason}");
        assert!(!reason.contains("abc123def"), "reason leaked: {reason}");
    }

    let redacted = redact(&source.text);
    assert!(!redacted.contains("sk-secret"));
    assert!(!redacted.contains("abc123def"));
    assert!(redacted.contains("[redacted]"));

    let verifier_model = ScriptedModel::text("0.5");
    let config = MoaConfig::new(vec![reference(ScriptedModel::text("unused"))])
        .with_mode(MoaMode::VerifierRouted)
        .with_verifier_selection(VerifierSelection::Llm)
        .with_verifier_model(Arc::new(verifier_model.clone()));
    let score = score_progress(&config, problem, std::slice::from_ref(&source.text))
        .await
        .unwrap();
    assert!((0.0..=1.0).contains(&score));

    let calls = verifier_model.calls();
    assert_eq!(calls.len(), 1);
    for message in &calls[0].messages {
        let text = message.text();
        assert!(!text.contains("sk-secret"), "prompt leaked the key: {text}");
        assert!(!text.contains("abc123def"), "prompt leaked the key: {text}");
    }
}

/// Validation reproduces the Python diagnostics a migrating caller sees.
#[test]
fn configuration_validation_reports_the_python_messages() {
    let error = MoaConfig::new(Vec::new()).validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(error.message(), "MoA requires at least one reference model");

    let references = vec![reference(ScriptedModel::text("proposal"))];

    let llm_routed = MoaConfig::new(references.clone())
        .with_mode(MoaMode::VerifierRouted)
        .with_verifier_selection(VerifierSelection::Llm);
    assert_eq!(
        llm_routed.validate().unwrap_err().message(),
        "verifier_routed requires a verifier model"
    );

    let jev_outside_jev_mode = MoaConfig::new(references.clone()).with_jev(JevConfig::default());
    assert_eq!(
        jev_outside_jev_mode.validate().unwrap_err().message(),
        "JEV and verifier routing are mutually exclusive; jev is only supported in jev_routed"
    );

    let zero_concurrency = MoaConfig::new(references.clone()).with_max_concurrency(0);
    assert_eq!(
        zero_concurrency.validate().unwrap_err().message(),
        "max_concurrency must be a positive integer"
    );

    assert!(MoaConfig::new(references).validate().is_ok());
}

/// Diagnostic snapshots and the bounded history round-trip through JSON.
#[tokio::test]
async fn snapshot_and_history_round_trip_through_json() {
    let mut state = MoaState::new("fix the build");
    state.reference_calls = 2;
    state.verifier_calls = 1;
    state.previous_score = Some(0.4);
    let snapshot = state.snapshot();
    let json = serde_json::to_string(&snapshot).unwrap();
    assert!(
        !json.contains("fix the build"),
        "snapshot carries no task text"
    );
    let restored: MoaSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, snapshot);

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("moa-history.json");
    let mut history = ConsultationHistory::new(2);
    history.push(ConsultationRecord::new(
        "first",
        ConsultationOutcome::default(),
    ));
    history.push(ConsultationRecord::new(
        "second",
        ConsultationOutcome::default(),
    ));
    history.save(&path).await.unwrap();

    let loaded = ConsultationHistory::load(&path).await.unwrap();
    assert_eq!(loaded, history);
    assert_eq!(loaded.records()[0].problem, "first");
    assert_eq!(loaded.records()[1].problem, "second");
}

/// The embedded consultation skill materializes, mounts, discovers and unmounts.
#[tokio::test]
async fn consultation_skill_mounts_and_unmounts_through_the_library() {
    let temp = tempfile::tempdir().unwrap();
    let skill = materialize_builtin_skill(temp.path()).await.unwrap();
    assert_eq!(skill.name, CONSULT_SKILL_NAME);

    let mut library = SkillLibrary::new();
    let mut mount = ConsultationSkillMount::from_dir(&skill.directory);
    mount.validate().unwrap();
    let mounted = mount.mount(&mut library).unwrap();
    assert_eq!(mounted, skill);
    assert!(mount.is_mounted());
    assert_eq!(library.roots, vec![skill.directory.clone()]);

    let discovered = library.discover().await.unwrap();
    assert!(
        discovered
            .iter()
            .any(|found| found.name == CONSULT_SKILL_NAME && found.directory == skill.directory),
        "discovered: {discovered:?}"
    );

    assert!(mount.unmount(&mut library));
    assert!(!mount.is_mounted());
    assert_eq!(library.roots, Vec::<std::path::PathBuf>::new());
}
