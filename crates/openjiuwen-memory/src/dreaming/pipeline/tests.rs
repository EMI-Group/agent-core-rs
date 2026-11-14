//! Unit tests for the model-backed consolidation pipeline and its pure decision layer.
//!
//! Every test is offline and deterministic: [`ScriptedModel`] stands in for the provider,
//! candidates are built inline, the job clock is a fixed instant, and nothing sleeps,
//! touches the network, or depends on wall-clock ordering.
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedError, ScriptedModel, ScriptedTurn};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::super::model::{DreamJob, DreamKind, DreamMetrics, DreamState, DreamTrigger};
use super::super::planner::DreamConsolidator;
use super::*;
use crate::record::{CallOptions, MemoryScope, MemoryType};

/// A model answer the parser accepts: one merge of `b` into `a`, one keep of `c`.
const MERGE_AND_KEEP: &str = r#"{"actions":[
    {"action":"merge","target_id":"a","source_ids":["b"],"content":"merged","reason":"duplicate"},
    {"action":"keep","id":"c"}
]}"#;

/// A fixed instant, so no test depends on the wall clock.
fn ts() -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).expect("a valid timestamp")
}

/// A pending consolidation job in a fixed scope with a stable id.
fn job() -> DreamJob {
    DreamJob::new(
        MemoryScope::new("alice", "workspace"),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        ts(),
    )
    .with_id("job-1")
}

/// Three candidates: `a`/`b` are near-duplicates, `c` is unrelated.
fn candidates() -> Vec<DreamCandidate> {
    vec![
        DreamCandidate::new("a", "alice prefers dark mode in the editor"),
        DreamCandidate::new("b", "alice prefers dark mode"),
        DreamCandidate::new("c", "the weather is nice today"),
    ]
}

/// A consolidator over `list`, with the default record-only sink.
fn consolidator(
    model: Option<Arc<dyn Model>>,
    list: Vec<DreamCandidate>,
    config: LlmConsolidatorConfig,
) -> LlmConsolidator {
    LlmConsolidator::new(model, Arc::new(StaticCandidateSource::new(list)), config)
        .expect("a valid consolidator")
}

/// Wraps one JSON object as a dream-job payload.
fn payload(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    value.as_object().cloned().unwrap_or_default()
}

/// A sink that always fails, used to prove a sink error aborts the completion.
#[derive(Debug)]
struct FailingSink;

#[async_trait]
impl ConsolidationSink for FailingSink {
    async fn apply(
        &self,
        _job: &DreamJob,
        _plan: &DreamConsolidationPlan,
        _ctx: &CallOptions,
    ) -> Result<DreamMetrics> {
        Err(Error::memory("the sink refused the plan"))
    }
}

// --- the LLM path -------------------------------------------------------------------

#[tokio::test]
async fn a_valid_model_answer_becomes_an_llm_plan() {
    let model = ScriptedModel::builder()
        .respond_text(MERGE_AND_KEEP)
        .build();
    let pipeline = consolidator(
        Some(Arc::new(model.clone())),
        candidates(),
        LlmConsolidatorConfig::default(),
    );

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    assert_eq!(plan.origin, DreamConsolidationOrigin::Llm);
    assert_eq!(plan.actions.len(), 2, "{:?}", plan.actions);
    assert!(!plan.is_degraded());
    assert_eq!(plan.metrics.records_merged, 1);
    assert_eq!(plan.metrics.records_scanned, 3);
    assert_eq!(model.call_count(), 1, "a usable answer is not retried");

    let prompt = model
        .last_call()
        .expect("one recorded call")
        .last_user_text()
        .expect("a user message");
    for id in ["a", "b", "c"] {
        assert!(prompt.contains(&format!("\"id\":\"{id}\"")), "{prompt}");
    }
    assert!(
        prompt.contains("alice prefers dark mode in the editor"),
        "the candidate text travels in the prompt: {prompt}"
    );
}

#[tokio::test]
async fn an_unparseable_answer_retries_then_falls_back() {
    let model = ScriptedModel::new([
        ScriptedTurn::text("sorry, not json"),
        ScriptedTurn::text("still not json"),
    ]);
    let config = LlmConsolidatorConfig::default().with_model_retries(1);
    let pipeline = consolidator(Some(Arc::new(model.clone())), candidates(), config);

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    assert_eq!(plan.origin, DreamConsolidationOrigin::Rules);
    let reason = plan.degraded.as_deref().expect("a degradation reason");
    assert!(reason.contains("model answer"), "{reason}");
    assert_eq!(model.call_count(), 2, "one call plus one retry");
}

#[tokio::test]
async fn a_provider_error_falls_back_and_names_the_error() {
    let model = ScriptedModel::new([
        ScriptedTurn::fail_with(ScriptedError::llm("provider exploded")),
        ScriptedTurn::fail_with(ScriptedError::llm("provider exploded")),
    ]);
    let config = LlmConsolidatorConfig::default();
    let pipeline = consolidator(Some(Arc::new(model.clone())), candidates(), config);

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    assert_eq!(plan.origin, DreamConsolidationOrigin::Rules);
    let reason = plan.degraded.as_deref().expect("a degradation reason");
    assert!(reason.contains("provider exploded"), "{reason}");
    assert_eq!(model.call_count(), 2);

    let expected = deterministic_plan(
        "job-1",
        &candidates(),
        LlmConsolidatorConfig::default().fallback_similarity,
    );
    assert_eq!(
        plan.actions, expected.actions,
        "the rule plan is used verbatim"
    );
}

#[tokio::test]
async fn a_missing_model_uses_the_rules_without_a_call() {
    let pipeline = consolidator(None, candidates(), LlmConsolidatorConfig::default());

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    assert_eq!(plan.origin, DreamConsolidationOrigin::Rules);
    assert_eq!(
        plan.degraded.as_deref(),
        Some("no consolidation model is configured")
    );
    assert_eq!(plan.metrics.records_scanned, 3);
}

#[tokio::test]
async fn a_disabled_configuration_never_calls_the_model() {
    let model = ScriptedModel::builder()
        .respond_text(MERGE_AND_KEEP)
        .build();
    let config = LlmConsolidatorConfig::default().with_enabled(false);
    let pipeline = consolidator(Some(Arc::new(model.clone())), candidates(), config);

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    assert_eq!(plan.origin, DreamConsolidationOrigin::Rules);
    let reason = plan.degraded.as_deref().expect("a degradation reason");
    assert!(reason.contains("disabled"), "{reason}");
    assert_eq!(model.call_count(), 0);
}

#[tokio::test]
async fn no_candidates_short_circuits_to_an_empty_rule_plan() {
    let model = ScriptedModel::builder()
        .respond_text(MERGE_AND_KEEP)
        .build();
    let pipeline = consolidator(
        Some(Arc::new(model.clone())),
        Vec::new(),
        LlmConsolidatorConfig::default(),
    );

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    assert_eq!(plan.origin, DreamConsolidationOrigin::Rules);
    assert!(plan.is_empty());
    assert_eq!(
        plan.degraded.as_deref(),
        Some("no candidate memories to consolidate")
    );
    assert_eq!(model.call_count(), 0);
}

// --- the deterministic fallback -----------------------------------------------------

#[tokio::test]
async fn the_fallback_merges_near_duplicates_and_keeps_the_rest() {
    let list = vec![
        DreamCandidate::new("a", "alice prefers dark mode in the editor"),
        DreamCandidate::new("b", "alice prefers dark mode in the editor today"),
        DreamCandidate::new("c", "the weather is nice today"),
    ];
    let pipeline = consolidator(None, list, LlmConsolidatorConfig::default());

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    assert_eq!(plan.origin, DreamConsolidationOrigin::Rules);
    assert_eq!(plan.actions.len(), 2, "{:?}", plan.actions);
    match &plan.actions[0] {
        DreamConsolidationAction::Merge {
            target_id,
            source_ids,
            content,
            ..
        } => {
            assert_eq!(target_id, "a");
            assert_eq!(source_ids.as_slice(), ["b".to_owned()]);
            assert_eq!(content, "alice prefers dark mode in the editor today");
        }
        other => panic!("expected a merge, got {other:?}"),
    }
    assert_eq!(
        plan.actions[1],
        DreamConsolidationAction::Keep {
            id: "c".to_owned(),
            reason: Some("fallback: no duplicate found".to_owned()),
        }
    );
    assert_eq!(plan.metrics.records_merged, 1);
    assert_eq!(plan.metrics.records_scanned, 3);
}

#[tokio::test]
async fn invented_ids_make_an_answer_unusable() {
    let model = ScriptedModel::new([
        ScriptedTurn::text(
            r#"{"actions":[{"action":"merge","target_id":"zzz","source_ids":["b"]}]}"#,
        ),
        ScriptedTurn::text(
            r#"{"actions":[{"action":"merge","target_id":"zzz","source_ids":["b"]}]}"#,
        ),
    ]);
    let config = LlmConsolidatorConfig::default().with_model_retries(1);
    let pipeline = consolidator(Some(Arc::new(model.clone())), candidates(), config);

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    assert_eq!(plan.origin, DreamConsolidationOrigin::Rules);
    let reason = plan.degraded.as_deref().expect("a degradation reason");
    assert!(
        reason.contains("no usable consolidation actions"),
        "{reason}"
    );
    assert_eq!(model.call_count(), 2, "each unusable answer is retried");
}

#[test]
fn deterministic_plan_clamps_its_threshold_and_handles_no_candidates() {
    let empty = deterministic_plan("job-1", &[], 0.75);
    assert!(empty.is_empty());
    assert_eq!(empty.origin, DreamConsolidationOrigin::Rules);
    assert_eq!(empty.metrics.records_scanned, 0);

    // A non-finite threshold falls back to the documented default instead of panicking.
    let list = vec![
        DreamCandidate::new("a", "alice prefers dark mode"),
        DreamCandidate::new("b", "alice prefers dark mode"),
    ];
    let plan = deterministic_plan("job-1", &list, f32::NAN);
    assert_eq!(plan.actions.len(), 1, "{:?}", plan.actions);
    assert_eq!(plan.metrics.records_merged, 1);

    // Threshold 0 merges everything (every pair is at least 0-similar).
    let plan = deterministic_plan("job-1", &candidates(), 0.0);
    assert_eq!(plan.actions.len(), 1, "{:?}", plan.actions);
    assert_eq!(plan.metrics.records_merged, 2);
}

// --- plan application ---------------------------------------------------------------

#[tokio::test]
async fn an_empty_plan_is_a_skipped_outcome() {
    let answer = r#"{"actions":[]}"#;
    let planning = ScriptedModel::builder().respond_text(answer).build();
    let pipeline = consolidator(
        Some(Arc::new(planning)),
        candidates(),
        LlmConsolidatorConfig::default(),
    );

    let plan = pipeline.plan(&job(), &CallOptions::new()).await.unwrap();
    assert_eq!(plan.origin, DreamConsolidationOrigin::Llm);
    assert!(plan.is_empty());

    let applying = ScriptedModel::builder().respond_text(answer).build();
    let pipeline = consolidator(
        Some(Arc::new(applying)),
        candidates(),
        LlmConsolidatorConfig::default(),
    );
    let outcome = pipeline
        .consolidate(&job(), &CallOptions::new())
        .await
        .unwrap();

    assert_eq!(outcome.state, DreamState::Skipped);
    assert_eq!(outcome.notes, vec!["no consolidation actions".to_owned()]);
    assert_eq!(outcome.job_id, "job-1");
}

#[tokio::test]
async fn consolidate_applies_the_plan_through_the_sink() {
    let model = ScriptedModel::builder()
        .respond_text(MERGE_AND_KEEP)
        .build();
    let pipeline = consolidator(
        Some(Arc::new(model)),
        candidates(),
        LlmConsolidatorConfig::default(),
    )
    .with_sink(Arc::new(RecordOnlySink));

    let outcome = pipeline
        .consolidate(&job(), &CallOptions::new())
        .await
        .unwrap();

    assert_eq!(outcome.state, DreamState::Succeeded);
    assert_eq!(outcome.metrics.records_merged, 1);
    assert_eq!(outcome.metrics.records_scanned, 3);
    assert_eq!(
        outcome.notes,
        vec![
            "merge 1 record(s) into `a`".to_owned(),
            "keep `c`".to_owned(),
        ],
    );
}

#[tokio::test]
async fn a_sink_error_fails_the_consolidation() {
    let model = ScriptedModel::builder()
        .respond_text(MERGE_AND_KEEP)
        .build();
    let pipeline = consolidator(
        Some(Arc::new(model)),
        candidates(),
        LlmConsolidatorConfig::default(),
    )
    .with_sink(Arc::new(FailingSink));

    let error = pipeline
        .consolidate(&job(), &CallOptions::new())
        .await
        .expect_err("the sink failed");

    assert_eq!(error.kind(), ErrorKind::Memory);
    assert!(error.message().contains("the sink refused the plan"));
}

#[tokio::test]
async fn a_cancelled_call_returns_cancelled_without_calling_the_model() {
    let model = ScriptedModel::builder()
        .respond_text(MERGE_AND_KEEP)
        .build();
    let pipeline = consolidator(
        Some(Arc::new(model.clone())),
        candidates(),
        LlmConsolidatorConfig::default(),
    );
    let token = CancellationToken::new();
    token.cancel();
    let ctx = CallOptions::new().with_cancellation(token);

    let error = pipeline
        .plan(&job(), &ctx)
        .await
        .expect_err("a cancelled plan");

    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(model.call_count(), 0);
}

// --- configuration bounds -----------------------------------------------------------

#[test]
fn config_validation_rejects_every_out_of_range_knob() {
    let cases = [
        (
            LlmConsolidatorConfig::default().with_max_candidates(0),
            "max_candidates",
        ),
        (
            LlmConsolidatorConfig::default().with_max_content_chars(0),
            "max_content_chars",
        ),
        (
            LlmConsolidatorConfig::default().with_model_retries(11),
            "model_retries",
        ),
        (
            LlmConsolidatorConfig::default().with_fallback_similarity(f32::NAN),
            "fallback_similarity",
        ),
        (
            LlmConsolidatorConfig::default().with_fallback_similarity(1.5),
            "fallback_similarity",
        ),
        (
            LlmConsolidatorConfig::default().with_fallback_similarity(-0.1),
            "fallback_similarity",
        ),
    ];
    for (config, field) in cases {
        let error = config.validate().expect_err(field);
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains(field),
            "{field}: {}",
            error.message()
        );
    }

    // The constructor surfaces the same failures.
    let error = LlmConsolidator::new(
        None,
        Arc::new(StaticCandidateSource::empty()),
        LlmConsolidatorConfig::default().with_max_candidates(0),
    )
    .expect_err("an invalid config");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn max_candidates_bounds_the_prompt() {
    let model = ScriptedModel::builder()
        .respond_text(r#"{"actions":[{"action":"keep","id":"a"}]}"#)
        .build();
    let config = LlmConsolidatorConfig::default().with_max_candidates(2);
    let pipeline = consolidator(Some(Arc::new(model.clone())), candidates(), config);

    pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    let prompt = model
        .last_call()
        .expect("one recorded call")
        .last_user_text()
        .expect("a user message");
    assert!(prompt.contains("\"id\":\"a\""), "{prompt}");
    assert!(prompt.contains("\"id\":\"b\""), "{prompt}");
    assert!(!prompt.contains("\"id\":\"c\""), "{prompt}");
    assert!(prompt.contains("\"max_actions\":2"), "{prompt}");
}

#[tokio::test]
async fn max_content_chars_truncates_each_candidate() {
    let model = ScriptedModel::builder()
        .respond_text(r#"{"actions":[{"action":"keep","id":"a"}]}"#)
        .build();
    let config = LlmConsolidatorConfig::default().with_max_content_chars(5);
    let list = vec![DreamCandidate::new("a", "alice prefers dark mode")];
    let pipeline = consolidator(Some(Arc::new(model.clone())), list, config);

    pipeline.plan(&job(), &CallOptions::new()).await.unwrap();

    let prompt = model
        .last_call()
        .expect("one recorded call")
        .last_user_text()
        .expect("a user message");
    assert!(prompt.contains("\"text\":\"alice…\""), "{prompt}");
}

#[test]
fn bounded_text_marks_a_cut_with_an_ellipsis() {
    assert_eq!(bounded_text("alice", 5), "alice");
    assert_eq!(bounded_text("alice", 10), "alice");
    assert_eq!(bounded_text("alice prefers", 5), "alice…");
    assert_eq!(bounded_text("alice", 0), "…");
    assert_eq!(bounded_text("", 0), "");
}

// --- candidate sources --------------------------------------------------------------

#[tokio::test]
async fn payload_source_reads_an_absent_key_as_no_candidates() {
    let source = PayloadCandidateSource;
    let found = source
        .candidates(&job(), &CallOptions::new())
        .await
        .unwrap();
    assert_eq!(found, Vec::<DreamCandidate>::new());
    assert_eq!(PAYLOAD_CANDIDATES_KEY, "candidates");
}

#[tokio::test]
async fn payload_source_reads_a_valid_candidate_list() {
    let job = job().with_payload(payload(json!({
        "candidates": [{"id": "a", "text": "x", "type": "semantic_memory"}]
    })));
    let source = PayloadCandidateSource;

    let found = source.candidates(&job, &CallOptions::new()).await.unwrap();

    assert_eq!(found.len(), 1);
    assert_eq!(found[0].id, "a");
    assert_eq!(found[0].content, "x");
    assert_eq!(found[0].kind, MemoryType::SemanticMemory);
}

#[tokio::test]
async fn payload_source_rejects_a_malformed_candidate_list() {
    let job = job().with_payload(payload(json!({"candidates": "not a list"})));
    let source = PayloadCandidateSource;

    let error = source
        .candidates(&job, &CallOptions::new())
        .await
        .expect_err("a malformed payload");

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("job-1"), "{}", error.message());
}

#[tokio::test]
async fn static_source_returns_its_candidates_in_order() {
    let source = StaticCandidateSource::new(candidates());

    let found = source
        .candidates(&job(), &CallOptions::new())
        .await
        .unwrap();

    let ids: Vec<&str> = found
        .iter()
        .map(|candidate| candidate.id.as_str())
        .collect();
    assert_eq!(ids, vec!["a", "b", "c"]);
    let empty = StaticCandidateSource::empty()
        .candidates(&job(), &CallOptions::new())
        .await
        .unwrap();
    assert_eq!(empty, Vec::<DreamCandidate>::new());
}

// --- candidate model ----------------------------------------------------------------

#[test]
fn candidate_json_accepts_the_record_aliases_and_defaults() {
    let candidate: DreamCandidate =
        serde_json::from_value(json!({"id": "a", "text": "hello", "type": "user_profile"}))
            .unwrap();

    assert_eq!(candidate.id, "a");
    assert_eq!(candidate.content, "hello");
    assert_eq!(candidate.kind, MemoryType::UserProfile);
    assert!(
        (candidate.importance - 0.5).abs() < f32::EPSILON,
        "{}",
        candidate.importance
    );
    assert_eq!(
        candidate.timestamp,
        DateTime::from_timestamp(0, 0).unwrap(),
        "a missing timestamp is the epoch, never now"
    );
    assert_eq!(candidate.access_count, 0);

    let value = serde_json::to_value(&candidate).unwrap();
    assert_eq!(value["text"], json!("hello"));
    assert_eq!(value["type"], json!("user_profile"));
    let restored: DreamCandidate = serde_json::from_value(value).unwrap();
    assert_eq!(restored, candidate);

    // The record-side spellings are accepted too.
    let aliased: DreamCandidate = serde_json::from_value(
        json!({"id": "b", "content": "kept", "memory_type": "coding", "importance": 0.25}),
    )
    .unwrap();
    assert_eq!(aliased.content, "kept");
    assert_eq!(aliased.kind, MemoryType::Coding);
}

#[test]
fn candidate_validation_rejects_an_empty_id_and_a_bad_importance() {
    DreamCandidate::new("a", "text").validate().unwrap();

    let error = DreamCandidate::new("", "text")
        .validate()
        .expect_err("no id");
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = DreamCandidate::new("a", "text")
        .with_importance(1.5)
        .validate()
        .expect_err("importance out of range");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("importance"),
        "{}",
        error.message()
    );

    let error = DreamCandidate::new("a", "text")
        .with_importance(f32::NAN)
        .validate()
        .expect_err("non-finite importance");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn candidate_projection_copies_the_decision_fields() {
    let mut record = crate::record::MemoryRecord::new(
        MemoryScope::new("alice", "workspace"),
        "remember the milk",
        MemoryType::EpisodicMemory,
    )
    .with_id("rec-1")
    .with_importance(0.9);
    record.access_count = 3;

    let candidate = DreamCandidate::from_record(&record);

    assert_eq!(candidate.id, "rec-1");
    assert_eq!(candidate.content, "remember the milk");
    assert_eq!(candidate.kind, MemoryType::EpisodicMemory);
    assert!((candidate.importance - 0.9).abs() < f32::EPSILON);
    assert_eq!(candidate.access_count, 3);
    assert_eq!(candidate.timestamp, record.timestamp);
}

// --- the pure decision layer --------------------------------------------------------

#[test]
fn actions_expose_their_names_ids_and_notes() {
    let merge = DreamConsolidationAction::Merge {
        target_id: "a".to_owned(),
        source_ids: vec!["b".to_owned(), "c".to_owned()],
        content: "merged".to_owned(),
        reason: Some("duplicate".to_owned()),
    };
    assert_eq!(merge.action_name(), "merge");
    assert_eq!(merge.affected_ids(), vec!["a", "b", "c"]);
    assert_eq!(merge.reason(), Some("duplicate"));
    assert_eq!(merge.note(), "merge 2 record(s) into `a`");

    let keep = DreamConsolidationAction::Keep {
        id: "a".to_owned(),
        reason: None,
    };
    assert_eq!(keep.action_name(), "keep");
    assert_eq!(keep.affected_ids(), vec!["a"]);
    assert_eq!(keep.reason(), None);
    assert_eq!(keep.note(), "keep `a`");

    let summarize = DreamConsolidationAction::Summarize {
        source_ids: vec!["a".to_owned()],
        summary: "short".to_owned(),
        reason: None,
    };
    assert_eq!(summarize.action_name(), "summarize");
    assert_eq!(summarize.note(), "summarize 1 record(s)");

    let forget = DreamConsolidationAction::Forget {
        ids: vec!["a".to_owned(), "b".to_owned()],
        reason: None,
    };
    assert_eq!(forget.action_name(), "forget");
    assert_eq!(forget.note(), "forget 2 record(s)");
}

#[test]
fn plan_metrics_count_summaries_and_forgets() {
    let list = candidates();
    let actions = vec![
        DreamConsolidationAction::Summarize {
            source_ids: vec!["a".to_owned(), "b".to_owned()],
            summary: "short".to_owned(),
            reason: None,
        },
        DreamConsolidationAction::Forget {
            ids: vec!["c".to_owned()],
            reason: None,
        },
    ];

    let plan =
        DreamConsolidationPlan::new("job-1", DreamConsolidationOrigin::Llm, actions, None, &list);

    assert_eq!(plan.metrics.records_created, 1);
    assert_eq!(plan.metrics.records_deleted, 1);
    assert_eq!(plan.metrics.records_merged, 0);
    assert_eq!(plan.metrics.records_scanned, 3);
    let reclaimed: usize = list
        .iter()
        .take(3)
        .map(|candidate| candidate.content.chars().count())
        .sum();
    assert_eq!(plan.metrics.tokens_saved, (reclaimed / 4) as u64);
}

#[test]
fn plan_notes_lead_with_the_degradation_reason() {
    let actions = vec![DreamConsolidationAction::Keep {
        id: "a".to_owned(),
        reason: None,
    }];
    let plan = DreamConsolidationPlan::new(
        "job-1",
        DreamConsolidationOrigin::Rules,
        actions,
        Some("because".to_owned()),
        &candidates(),
    );

    assert!(plan.is_degraded());
    assert_eq!(plan.notes(), vec!["degraded: because", "keep `a`"]);

    let mut plan = plan;
    plan.degraded = None;
    assert!(!plan.is_degraded());
    assert_eq!(plan.notes(), vec!["keep `a`"]);
}

#[test]
fn parsing_tolerates_prose_fences_and_action_aliases() {
    let raw = "Here you go:\n```json\n{\"actions\":[\
        {\"op\":\"combine\",\"id\":\"a\",\"ids\":[\"b\"],\"reason\":\"same\"},\
        {\"type\":\"retain\",\"id\":\"c\"}\
    ]}\n```";
    let plan = parse_consolidation_response(raw, "job-1", &candidates()).expect("a usable plan");

    assert_eq!(plan.origin, DreamConsolidationOrigin::Llm);
    assert_eq!(plan.actions.len(), 2, "{:?}", plan.actions);
    match &plan.actions[0] {
        DreamConsolidationAction::Merge {
            target_id,
            source_ids,
            content,
            reason,
        } => {
            assert_eq!(target_id, "a");
            assert_eq!(source_ids.as_slice(), ["b".to_owned()]);
            assert_eq!(
                content, "alice prefers dark mode in the editor",
                "the longest source text fills a missing content"
            );
            assert_eq!(reason.as_deref(), Some("same"));
        }
        other => panic!("expected a merge, got {other:?}"),
    }
    assert_eq!(
        plan.actions[1],
        DreamConsolidationAction::Keep {
            id: "c".to_owned(),
            reason: None,
        }
    );
}

#[test]
fn parsing_rejects_unusable_answers() {
    for raw in [
        "",
        "no json here",
        "{}",
        r#"{"actions": "nope"}"#,
        r#"{"actions":[{"action":"keep","id":"zzz"}]}"#,
        r#"{"actions":[{"action":"nonsense","id":"a"}]}"#,
        r#"{"actions":[{"action":"forget","ids":[]}]}"#,
    ] {
        assert!(
            parse_consolidation_response(raw, "job-1", &candidates()).is_none(),
            "`{raw}` should be unusable"
        );
    }

    let plan = parse_consolidation_response(r#"{"actions":[]}"#, "job-1", &candidates())
        .expect("an explicit empty list is valid");
    assert!(plan.is_empty());
    assert_eq!(plan.origin, DreamConsolidationOrigin::Llm);
}

#[test]
fn parsing_accepts_the_forget_and_summarize_shapes() {
    let raw = r#"{"actions":[
        {"action":"forget","ids":["a"],"reason":"stale"},
        {"action":"summarize","source_ids":["b","c"],"text":"everything known"}
    ]}"#;
    let plan = parse_consolidation_response(raw, "job-1", &candidates()).expect("a usable plan");

    assert_eq!(
        plan.actions,
        vec![
            DreamConsolidationAction::Forget {
                ids: vec!["a".to_owned()],
                reason: Some("stale".to_owned()),
            },
            DreamConsolidationAction::Summarize {
                source_ids: vec!["b".to_owned(), "c".to_owned()],
                summary: "everything known".to_owned(),
                reason: None,
            },
        ]
    );
    assert_eq!(plan.metrics.records_deleted, 1);
    assert_eq!(plan.metrics.records_created, 1);
}

#[test]
fn normalizing_keeps_only_known_and_distinct_ids() {
    let raw = r#"{"actions":[{"action":"FOrget","ids":["a","a","invented","c"],"reason":"  "}]}"#;

    let plan = parse_consolidation_response(raw, "job-1", &candidates()).expect("a usable plan");

    assert_eq!(
        plan.actions,
        vec![DreamConsolidationAction::Forget {
            ids: vec!["a".to_owned(), "c".to_owned()],
            reason: None,
        }],
        "duplicates collapse, invented ids vanish, and a blank reason is absent"
    );
}

#[tokio::test]
async fn the_default_sink_records_the_plan_metrics_and_writes_nothing() {
    let plan = deterministic_plan("job-1", &candidates(), 0.75);

    let metrics = RecordOnlySink
        .apply(&job(), &plan, &CallOptions::new())
        .await
        .unwrap();

    assert_eq!(metrics, plan.metrics);
}
