//! Tests for the bounded clarification loop, its provenance, and its stream.

use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

use super::super::{Clarifier, GOAL_QUESTION, INPUT_QUESTION, OUTPUT_QUESTION};
use super::*;
use crate::agent_builder::progress::ProgressStage;

/// A model answer naming the missing input.
const NEEDS_INPUT_FENCE: &str = r#"```json
{"ready": false, "question": "What input will the agent receive?", "summary": "goal"}
```"#;

/// A model answer declaring the request ready.
const READY_FENCE: &str = r#"```json
{"ready": true, "question": "", "summary": "goal, input, output"}
```"#;

/// Coerces a scripted model into the trait object the clarifier takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// The questions a dialogue asked, oldest first.
fn questions(dialogue: &ClarificationDialogue) -> Vec<&str> {
    dialogue
        .rounds
        .iter()
        .map(|round| round.question.as_str())
        .collect()
}

/// The sources a dialogue recorded, oldest first.
fn sources(dialogue: &ClarificationDialogue) -> Vec<GenerationSource> {
    dialogue.rounds.iter().map(|round| round.source).collect()
}

#[tokio::test]
async fn answers_accumulate_until_the_request_is_ready() {
    let history = HistoryManager::new(20);
    let mut answers = [
        "The input is a product document.",
        "It should return a summary.",
    ]
    .into_iter();

    let dialogue = Clarifier::new()
        .clarify_rounds_with(
            "Create an assistant.",
            &history,
            &ClarifyConfig::default(),
            |_| answers.next().map(str::to_owned),
        )
        .await
        .expect("a dialogue succeeds");

    assert_eq!(dialogue.termination, ClarificationTermination::Ready);
    assert_eq!(dialogue.rounds.len(), 3);
    assert_eq!(
        questions(&dialogue),
        vec![INPUT_QUESTION, OUTPUT_QUESTION, ""]
    );
    assert_eq!(
        sources(&dialogue),
        vec![
            GenerationSource::Deterministic,
            GenerationSource::Deterministic,
            GenerationSource::Deterministic,
        ]
    );
    assert_eq!(dialogue.provenance.len(), dialogue.rounds.len());
    assert!(dialogue.clarification.ready);
    assert_eq!(dialogue.clarification.question, "");
    assert_eq!(
        dialogue
            .clarification
            .provenance
            .stages_matching(GenerationSource::Deterministic)
            .len(),
        3
    );
    assert_eq!(
        dialogue.rounds[0].answer.as_deref(),
        Some("The input is a product document.")
    );
    assert_eq!(
        dialogue.rounds[1].answer.as_deref(),
        Some("It should return a summary.")
    );
    assert_eq!(dialogue.rounds[2].answer, None);
    assert_eq!(dialogue.rounds[1].summary, "goal, input");
    assert_eq!(dialogue.rounds[0].index, 0);
    assert_eq!(dialogue.rounds[2].index, 2);
    assert!(!dialogue.provenance.has_fallback());
    assert!(history.is_empty(), "the caller's history is never mutated");
}

#[tokio::test]
async fn no_answer_stops_after_one_round() {
    let history = HistoryManager::new(20);
    let dialogue = Clarifier::new()
        .clarify_rounds("Create an assistant.", &history, &ClarifyConfig::default())
        .await
        .expect("a dialogue succeeds");

    assert_eq!(dialogue.termination, ClarificationTermination::NoAnswer);
    assert_eq!(dialogue.rounds.len(), 1);
    assert_eq!(dialogue.rounds[0].question, INPUT_QUESTION);
    assert_eq!(dialogue.rounds[0].answer, None);
    assert_eq!(dialogue.rounds[0].source, GenerationSource::Deterministic);
    assert_eq!(dialogue.clarification.question, INPUT_QUESTION);
    assert_eq!(dialogue.clarification.provenance.len(), 1);
    assert_eq!(dialogue.provenance, dialogue.clarification.provenance);
}

#[tokio::test]
async fn a_ready_request_stops_on_the_first_round() {
    let history = HistoryManager::new(20);
    let dialogue = Clarifier::new()
        .clarify_rounds(
            "Create an assistant that answers product questions.",
            &history,
            &ClarifyConfig::default(),
        )
        .await
        .expect("a dialogue succeeds");

    assert_eq!(dialogue.termination, ClarificationTermination::Ready);
    assert_eq!(dialogue.rounds.len(), 1);
    assert_eq!(dialogue.rounds[0].question, "");
    assert!(dialogue.rounds[0].ready);
    assert!(dialogue.clarification.ready);
}

#[tokio::test]
async fn the_round_cap_keeps_the_last_question() {
    let history = HistoryManager::new(20);
    let mut answers = [
        "Create an agent.",
        "It reads a document.",
        "Use the search tool.",
    ]
    .into_iter();

    let dialogue = Clarifier::new()
        .clarify_rounds_with(
            "Something with a database.",
            &history,
            &ClarifyConfig::default(),
            |_| answers.next().map(str::to_owned),
        )
        .await
        .expect("a dialogue succeeds");

    assert_eq!(dialogue.termination, ClarificationTermination::RoundCap);
    assert_eq!(dialogue.rounds.len(), DEFAULT_MAX_CLARIFY_ROUNDS);
    assert_eq!(
        questions(&dialogue),
        vec![GOAL_QUESTION, INPUT_QUESTION, OUTPUT_QUESTION]
    );
    assert_eq!(dialogue.clarification.question, OUTPUT_QUESTION);
    assert!(!dialogue.clarification.ready);
    assert_eq!(dialogue.provenance.len(), DEFAULT_MAX_CLARIFY_ROUNDS);
    assert_eq!(
        dialogue.rounds[2].answer.as_deref(),
        Some("Use the search tool.")
    );
}

#[tokio::test]
async fn a_repeated_question_ends_the_dialogue() {
    let history = HistoryManager::new(20);
    let dialogue = Clarifier::new()
        .clarify_rounds_with(
            "Create an assistant.",
            &history,
            &ClarifyConfig::default(),
            |_| Some("Not sure.".to_string()),
        )
        .await
        .expect("a dialogue succeeds");

    assert_eq!(
        dialogue.termination,
        ClarificationTermination::RepeatedQuestion
    );
    assert_eq!(dialogue.rounds.len(), 2);
    assert_eq!(questions(&dialogue), vec![INPUT_QUESTION, INPUT_QUESTION]);
    assert_eq!(dialogue.rounds[0].answer.as_deref(), Some("Not sure."));
    assert_eq!(dialogue.rounds[1].answer.as_deref(), Some("Not sure."));
    assert_eq!(dialogue.clarification.question, INPUT_QUESTION);
}

#[tokio::test]
async fn a_blank_answer_counts_as_no_answer() {
    let history = HistoryManager::new(20);
    let dialogue = Clarifier::new()
        .clarify_rounds_with(
            "Create an assistant.",
            &history,
            &ClarifyConfig::default(),
            |_| Some("   ".to_string()),
        )
        .await
        .expect("a dialogue succeeds");

    assert_eq!(dialogue.termination, ClarificationTermination::NoAnswer);
    assert_eq!(dialogue.rounds.len(), 1);
    assert_eq!(dialogue.rounds[0].answer, None);
}

#[tokio::test]
async fn the_model_path_accumulates_answers_across_rounds() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(NEEDS_INPUT_FENCE),
        ScriptedTurn::text(READY_FENCE),
    ]));
    let clarifier = Clarifier::with_model(dynamic(&model));

    let mut history = HistoryManager::new(20);
    history.add_user_message("Build a helper.");

    let dialogue = clarifier
        .clarify_rounds_with(
            "It summarizes documents.",
            &history,
            &ClarifyConfig::default(),
            |round| {
                if round.index == 0 {
                    Some("Use gpt-4o.".to_string())
                } else {
                    None
                }
            },
        )
        .await
        .expect("a dialogue succeeds");

    assert_eq!(dialogue.termination, ClarificationTermination::Ready);
    assert_eq!(dialogue.rounds.len(), 2);
    assert_eq!(
        sources(&dialogue),
        vec![GenerationSource::Model, GenerationSource::Model]
    );
    assert_eq!(dialogue.rounds[0].question, INPUT_QUESTION);
    assert_eq!(dialogue.rounds[0].answer.as_deref(), Some("Use gpt-4o."));
    assert!(dialogue.rounds[1].ready);
    assert_eq!(dialogue.clarification.summary, "goal, input, output");
    assert!(!dialogue.provenance.has_deviation());
    assert_eq!(
        dialogue
            .provenance
            .stages_matching(GenerationSource::Model)
            .len(),
        2
    );

    assert_eq!(model.call_count(), 2);
    let second = model.calls()[1].clone();
    assert_eq!(second.messages.len(), 4);
    assert_eq!(second.messages[1].text(), "Build a helper.");
    assert_eq!(second.messages[2].text(), "Use gpt-4o.");
    assert_eq!(second.messages[3].text(), "It summarizes documents.");

    let contents: Vec<&str> = history
        .get_history()
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    assert_eq!(contents, vec!["Build a helper."]);
}

#[tokio::test]
async fn a_model_failure_is_recorded_as_a_fallback() {
    let model = Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")]));
    let clarifier = Clarifier::with_model(dynamic(&model));
    let history = HistoryManager::new(20);

    let dialogue = clarifier
        .clarify_rounds("Create an assistant.", &history, &ClarifyConfig::default())
        .await
        .expect("a model failure is not an error");

    assert_eq!(dialogue.rounds.len(), 1);
    assert_eq!(dialogue.rounds[0].source, GenerationSource::Fallback);
    assert_eq!(dialogue.rounds[0].question, INPUT_QUESTION);
    assert!(dialogue.provenance.has_deviation());
    assert!(dialogue.provenance.has_fallback());
    assert_eq!(dialogue.provenance.fallback_count(), 1);
    assert_eq!(
        dialogue.provenance.stages[0].stage,
        ProgressStage::Clarifying
    );
    assert_eq!(dialogue.provenance.stages[0].detail, "model call failed");
    assert_eq!(dialogue.clarification.question, INPUT_QUESTION);

    let heuristic = Clarifier::new()
        .clarify("Create an assistant.", &history)
        .await
        .expect("clarify succeeds");
    assert_eq!(dialogue.clarification.ready, heuristic.ready);
    assert_eq!(dialogue.clarification.question, heuristic.question);
    assert_eq!(dialogue.clarification.summary, heuristic.summary);
    assert_eq!(
        heuristic
            .provenance
            .stages_matching(GenerationSource::Deterministic)
            .len(),
        1
    );
    assert!(!heuristic.provenance.has_fallback());
}

#[tokio::test]
async fn a_model_answer_without_the_contract_records_a_reason() {
    let model = Arc::new(ScriptedModel::text("I think it is probably fine."));
    let clarifier = Clarifier::with_model(dynamic(&model));

    let dialogue = clarifier
        .clarify_rounds(
            "Create an assistant.",
            &HistoryManager::new(20),
            &ClarifyConfig::default(),
        )
        .await
        .expect("a bad answer is not an error");

    assert_eq!(dialogue.rounds[0].source, GenerationSource::Fallback);
    assert_eq!(dialogue.provenance.fallback_count(), 1);
    assert_eq!(
        dialogue.provenance.stages[0].detail,
        "answer had no JSON fence"
    );
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn clarify_matches_the_one_round_dialogue() {
    let mut history = HistoryManager::new(20);
    history.add_user_message("Build an agent.");

    let single = Clarifier::new()
        .clarify("It reads a document.", &history)
        .await
        .expect("clarify succeeds");
    let dialogue = Clarifier::new()
        .clarify_rounds("It reads a document.", &history, &ClarifyConfig::new(1))
        .await
        .expect("a dialogue succeeds");

    assert_eq!(single, dialogue.clarification);
    assert_eq!(dialogue.rounds.len(), 1);
    assert_eq!(single.question, OUTPUT_QUESTION);
    assert_eq!(dialogue.rounds[0].question, single.question);
    assert_eq!(dialogue.rounds[0].source, GenerationSource::Deterministic);
    assert_eq!(dialogue.termination, ClarificationTermination::NoAnswer);
}

#[tokio::test]
async fn a_blank_query_is_rejected_before_any_model_call() {
    let model = Arc::new(ScriptedModel::text(NEEDS_INPUT_FENCE));
    let clarifier = Clarifier::with_model(dynamic(&model));
    let mut asked = false;

    let error = clarifier
        .clarify_rounds_with(
            "  \n\t",
            &HistoryManager::new(20),
            &ClarifyConfig::default(),
            |_| {
                asked = true;
                None
            },
        )
        .await
        .expect_err("a blank query is rejected");

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("query must not be blank"));
    assert!(!asked, "no round runs for a blank query");
    assert_eq!(model.call_count(), 0);
}

#[tokio::test]
async fn streaming_yields_one_round_per_round_then_the_dialogue() {
    let history = HistoryManager::new(20);
    let expected = Clarifier::new()
        .clarify_rounds("Create an assistant.", &history, &ClarifyConfig::default())
        .await
        .expect("a dialogue succeeds");
    let stream = Clarifier::new().stream_clarify("Create an assistant.", &history);
    assert_stream_type(&stream);

    let events: Vec<_> = stream.collect().await;

    assert_eq!(events.len(), expected.rounds.len() + 1);
    let mut rounds = Vec::new();
    for event in events {
        match event.expect("the stream yields events") {
            ClarifyEvent::Round(round) => rounds.push(round),
            ClarifyEvent::Completed(dialogue) => {
                assert_eq!(dialogue, expected);
            }
        }
    }
    assert_eq!(rounds, expected.rounds);
}

#[tokio::test]
async fn streaming_a_ready_request_yields_one_round() {
    let history = HistoryManager::new(20);
    let events: Vec<_> = Clarifier::new()
        .stream_clarify("Create an assistant that answers questions.", &history)
        .collect()
        .await;

    assert_eq!(events.len(), 2);
    match &events[0] {
        Ok(ClarifyEvent::Round(round)) => {
            assert!(round.ready);
            assert_eq!(round.question, "");
        }
        other => panic!("expected a round, got {other:?}"),
    }
    match &events[1] {
        Ok(ClarifyEvent::Completed(dialogue)) => {
            assert_eq!(dialogue.termination, ClarificationTermination::Ready);
        }
        other => panic!("expected a completed dialogue, got {other:?}"),
    }
}

#[tokio::test]
async fn streaming_a_blank_query_yields_one_error() {
    let events: Vec<_> = Clarifier::new()
        .stream_clarify("   ", &HistoryManager::new(20))
        .collect()
        .await;

    assert_eq!(events.len(), 1);
    let error = events
        .into_iter()
        .next()
        .expect("one event")
        .expect_err("a blank query is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

/// Fails to compile when the stream is not `Send + 'static`.
fn assert_stream_type(stream: &BoxStream<'static, Result<ClarifyEvent>>) {
    fn send_static<T: Send + 'static>(_: &T) {}
    send_static(stream);
}

#[test]
fn config_zero_rounds_normalizes_to_the_default() {
    assert_eq!(ClarifyConfig::new(0).max_rounds, DEFAULT_MAX_CLARIFY_ROUNDS);
    assert_eq!(
        ClarifyConfig::default().max_rounds,
        DEFAULT_MAX_CLARIFY_ROUNDS
    );
    assert_eq!(ClarifyConfig::new(2).max_rounds, 2);
    assert_eq!(ClarifyConfig::new(2).resolved_max_rounds(), 2);

    let raw = ClarifyConfig { max_rounds: 0 };
    assert_eq!(raw.resolved_max_rounds(), 1);
}

#[test]
fn terminations_have_wire_names() {
    for (termination, name) in [
        (ClarificationTermination::Ready, "ready"),
        (ClarificationTermination::NoAnswer, "no_answer"),
        (
            ClarificationTermination::RepeatedQuestion,
            "repeated_question",
        ),
        (ClarificationTermination::RoundCap, "round_cap"),
    ] {
        assert_eq!(termination.as_str(), name);
        assert_eq!(termination.to_string(), name);
        assert_eq!(
            serde_json::to_value(termination).expect("serialize"),
            serde_json::json!(name)
        );
    }
    assert_eq!(
        ClarificationTermination::default(),
        ClarificationTermination::Ready
    );
}

#[test]
fn the_dialogue_types_round_trip_through_json() {
    let mut provenance = GenerationProvenance::new();
    provenance.record_model(ProgressStage::Clarifying);
    let clarification = Clarification {
        ready: false,
        question: "What input will the agent receive?".to_string(),
        summary: "goal".to_string(),
        provenance: provenance.clone(),
    };
    let round = ClarificationRound {
        index: 0,
        question: clarification.question.clone(),
        answer: Some("A document.".to_string()),
        ready: false,
        summary: "goal".to_string(),
        source: GenerationSource::Deterministic,
    };
    let dialogue = ClarificationDialogue {
        rounds: vec![round.clone()],
        clarification: clarification.clone(),
        termination: ClarificationTermination::NoAnswer,
        provenance,
    };

    let value = serde_json::to_value(&clarification).expect("serialize");
    assert_eq!(
        value,
        serde_json::json!({
            "ready": false,
            "question": "What input will the agent receive?",
            "summary": "goal",
            "provenance": {"stages": [{"stage": "clarifying", "source": "model", "detail": ""}]},
        })
    );
    assert_eq!(
        serde_json::from_value::<Clarification>(value).expect("deserialize"),
        clarification
    );

    let value = serde_json::to_value(&round).expect("serialize");
    assert_eq!(
        serde_json::from_value::<ClarificationRound>(value).expect("deserialize"),
        round
    );

    let value = serde_json::to_value(&dialogue).expect("serialize");
    assert_eq!(value["termination"], serde_json::json!("no_answer"));
    assert_eq!(
        serde_json::from_value::<ClarificationDialogue>(value).expect("deserialize"),
        dialogue
    );

    let event = ClarifyEvent::Completed(dialogue);
    let value = serde_json::to_value(&event).expect("serialize");
    assert_eq!(
        serde_json::from_value::<ClarifyEvent>(value).expect("deserialize"),
        event
    );
}

#[test]
fn legacy_clarification_json_without_provenance_still_parses() {
    let legacy: Clarification = serde_json::from_value(serde_json::json!({
        "ready": true,
        "question": "",
        "summary": "goal, output",
    }))
    .expect("legacy JSON parses");

    assert!(legacy.ready);
    assert!(legacy.provenance.is_empty());
    assert_eq!(legacy.provenance, GenerationProvenance::new());
    assert_eq!(
        Clarification::default(),
        Clarification {
            ready: false,
            question: String::new(),
            summary: String::new(),
            provenance: GenerationProvenance::new(),
        }
    );
}
