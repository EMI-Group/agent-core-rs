#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for the research optimizers
//! (`openjiuwen_agent_evolving::optimizer::research`).
//!
//! Covers the six textual-gradient optimizers end to end: deterministic
//! proposals, backward gradients, best-so-far restoration, proposal/limit
//! exhaustion, checkpoint round-trips, signal routing per target/tool, Metis
//! utility ordering and TTSE clustering. The model-driven paths use
//! `ScriptedModel` for a successful `<improved>` answer, an unparseable answer and
//! a provider error; every failure must fall back to the deterministic rewrite
//! with a recorded provenance reason.
//!
//! Everything here is offline and deterministic: no network, no environment
//! reads, no clocks, and assertions never depend on test execution order.

use std::sync::Arc;

use openjiuwen_agent_evolving::optimizer::{
    Candidate, InstructionOptimizer, InstructionStrategy, MemoryOptimizer, MemoryStrategy,
    MetisOptimizer, MetisStrategy, OperatorKind, OperatorState, Optimizer,
    SkillExperienceOptimizer, SkillExperienceStrategy, TextualStrategy, ToolOptimizer,
    ToolStrategy, TtseOptimizer, TtseStrategy,
};
use openjiuwen_agent_evolving::signal::{EvolutionSignal, EvolutionTarget};
use openjiuwen_core::foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::{json, Value};

/// An instruction operator with the standard `instruction` parameter.
fn instruction_operator() -> OperatorState {
    OperatorState::new("planner", OperatorKind::Instruction)
        .with_parameter("instruction", "Plan the task.")
}

/// A signal with the default weight and no extra provenance.
fn signal(target: EvolutionTarget, excerpt: &str) -> EvolutionSignal {
    EvolutionSignal::new("low_score", target, excerpt).expect("signal")
}

/// Reads `operator.parameter.value` out of a candidate's `parameters` object.
fn parameter_value(candidate: &Candidate, operator: &str, parameter: &str) -> String {
    candidate
        .parameters
        .get(operator)
        .and_then(|state| state.get(parameter))
        .and_then(|state| state.get("value"))
        .and_then(Value::as_str)
        .expect("candidate parameter value")
        .to_owned()
}

/// Reads the value a proposal stored for one operator parameter.
fn live_value(operator: &OperatorState, parameter: &str) -> String {
    operator
        .parameter(parameter)
        .expect("parameter")
        .value()
        .to_owned()
}

#[tokio::test]
async fn instruction_optimizer_is_deterministic_and_tracks_feedback() {
    let mut first = InstructionOptimizer::new().with_operator(instruction_operator());
    let mut second = InstructionOptimizer::new().with_operator(instruction_operator());

    assert_eq!(first.propose(0).await.expect("limit 0").len(), 0);

    let before_first = first.propose(1).await.expect("propose");
    let before_second = second.propose(1).await.expect("propose");
    assert_eq!(before_first, before_second);
    assert_eq!(before_first[0].id, "instruction-0-0");
    assert_eq!(before_first[0].generation, 0);
    assert_eq!(
        before_first[0].parameters["candidate_meta"],
        json!({"generation": 0, "strategy": "instruction"})
    );

    let feedback = signal(
        EvolutionTarget::Instruction,
        "Plans omit a verification step.",
    );
    assert_eq!(
        first
            .backward(&[feedback.clone(), feedback.clone()])
            .unwrap(),
        1
    );
    assert_eq!(
        second
            .backward(&[feedback.clone(), feedback.clone()])
            .unwrap(),
        1
    );
    // The tracker deduplicates the repeated feedback.
    assert_eq!(first.signal_tracker().len(), 1);

    let after_first = first.propose(1).await.expect("propose");
    let after_second = second.propose(1).await.expect("propose");
    assert_eq!(after_first, after_second);
    assert_eq!(after_first[0].generation, 1);
    assert_eq!(after_first[0].id, "instruction-1-0");

    let text = parameter_value(&after_first[0], "planner", "instruction");
    assert!(text.starts_with("Plan the task."));
    assert_eq!(text.matches("Plans omit a verification step.").count(), 1);
    assert_eq!(
        first.provenance(&after_first[0].id),
        Some("planner.instruction: deterministic")
    );
}

#[tokio::test]
async fn instruction_rewrite_respects_caps_and_dedupes() {
    let strategy = InstructionStrategy::new().with_limits(2, 60);
    let signals: Vec<EvolutionSignal> = (0..5)
        .map(|index| {
            signal(
                EvolutionTarget::Instruction,
                &format!("feedback item {index}"),
            )
        })
        .collect();
    let rewritten = strategy.rewrite("Do the task.", "", &signals);
    assert!(rewritten.chars().count() <= 60);
    assert_eq!(rewritten.matches("\n- ").count(), 2);
    // Re-running the rewrite over its own output is idempotent.
    assert_eq!(strategy.rewrite(&rewritten, "", &signals), rewritten);
}

#[tokio::test]
async fn observe_promotes_the_best_parameters_for_the_next_proposal() {
    let mut optimizer = InstructionOptimizer::new().with_operator(instruction_operator());
    optimizer
        .backward(&[signal(
            EvolutionTarget::Instruction,
            "Plans omit a verification step.",
        )])
        .unwrap();

    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    let improved = parameter_value(&candidate, "planner", "instruction");
    assert!(improved.contains("verification"));
    // Before `observe` the live state is still the incumbent.
    let operator = &optimizer.operators()["planner"];
    assert_eq!(live_value(operator, "instruction"), "Plan the task.");

    optimizer.observe(&candidate, 0.8);
    assert_eq!(
        optimizer.best().map(|best| best.id.as_str()),
        Some(candidate.id.as_str())
    );
    assert_eq!(optimizer.best().and_then(|best| best.score), Some(0.8));
    let operator = &optimizer.operators()["planner"];
    assert_eq!(live_value(operator, "instruction"), improved);
    assert_eq!(optimizer.history()[0].score, Some(0.8));
    assert!(optimizer.history()[0].provenance.contains("deterministic"));

    // The next generation starts from the accepted value.
    let next = optimizer.propose(1).await.expect("propose").remove(0);
    assert_eq!(parameter_value(&next, "planner", "instruction"), improved);

    // A worse candidate keeps the incumbent live state and best record.
    optimizer.observe(&next, 0.1);
    assert_eq!(
        optimizer.best().map(|best| best.id.as_str()),
        Some(candidate.id.as_str())
    );
    let operator = &optimizer.operators()["planner"];
    assert_eq!(live_value(operator, "instruction"), improved);
}

#[tokio::test]
async fn proposal_budget_exhausts_and_zero_limit_is_empty() {
    let mut optimizer = InstructionOptimizer::new()
        .with_operator(instruction_operator())
        .with_max_proposals(2);
    assert_eq!(optimizer.propose(0).await.expect("propose").len(), 0);
    assert_eq!(optimizer.propose(1).await.expect("propose").len(), 1);
    assert_eq!(optimizer.propose(1).await.expect("propose").len(), 1);
    assert_eq!(optimizer.propose(1).await.expect("propose").len(), 0);
    assert_eq!(optimizer.history().len(), 2);
    assert_eq!(optimizer.max_proposals(), 2);
}

#[tokio::test]
async fn optimizer_without_operators_proposes_nothing() {
    let mut optimizer = InstructionOptimizer::new();
    assert_eq!(optimizer.propose(1).await.expect("propose").len(), 0);
    assert!(optimizer
        .backward(&[signal(EvolutionTarget::Instruction, "unused")])
        .is_ok());
}

#[tokio::test]
async fn operator_state_round_trips_through_checkpoint_json() {
    let mut optimizer = InstructionOptimizer::new().with_operator(instruction_operator());
    optimizer
        .backward(&[signal(EvolutionTarget::Instruction, "Add verification.")])
        .unwrap();
    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    optimizer.observe(&candidate, 0.9);

    let state = optimizer.operators_state();
    let mut checkpoint = state.as_object().expect("state object").clone();
    checkpoint.insert(
        "future_operator".to_owned(),
        json!({"note": {"value": "extra", "unknown_field": 1}}),
    );
    checkpoint.insert("candidate_meta".to_owned(), json!({"generation": 7}));
    let checkpoint = Value::Object(checkpoint);

    // A fresh optimizer with no declared operators restores from the checkpoint.
    let mut restored = InstructionOptimizer::new();
    restored.restore_state(&checkpoint).expect("restore");
    assert_eq!(restored.operators_state()["planner"], state["planner"]);
    assert_eq!(
        restored_parameter(&restored, "future_operator", "note"),
        "extra"
    );
    assert!(!restored.operators().contains_key("candidate_meta"));

    // A `{"operators": ...}` wrapper is accepted too.
    let mut wrapped = InstructionOptimizer::new();
    wrapped
        .restore_state(&json!({"operators": state.clone()}))
        .expect("restore wrapped");
    assert_eq!(wrapped.operators_state(), state);

    // Malformed input is rejected instead of silently ignored.
    let mut broken = InstructionOptimizer::new();
    assert!(broken
        .restore_state(&json!(["not", "an", "object"]))
        .is_err());
    assert!(broken
        .restore_state(&json!({"operators": "not-an-object"}))
        .is_err());
}

/// Reads one restored parameter value from a wrapper optimizer.
fn restored_parameter(optimizer: &InstructionOptimizer, operator: &str, parameter: &str) -> String {
    live_value(
        optimizer.operators().get(operator).expect("operator"),
        parameter,
    )
}

#[tokio::test]
async fn tool_feedback_is_scoped_to_one_tool() {
    let mut optimizer = ToolOptimizer::new()
        .with_operator(
            OperatorState::new("tool.search", OperatorKind::Tool)
                .with_parameter("description", "Search the web."),
        )
        .with_operator(
            OperatorState::new("tool.fetch", OperatorKind::Tool)
                .with_parameter("description", "Fetch a URL."),
        );
    let search =
        signal(EvolutionTarget::Tool, "The query must be a single string.").with_tool("search");
    let fetch = signal(EvolutionTarget::Tool, "Timeouts must be retried.").with_tool("fetch");
    let unrelated = signal(EvolutionTarget::Instruction, "ignored");
    assert_eq!(optimizer.backward(&[search, fetch, unrelated]).unwrap(), 2);

    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    let search_description = parameter_value(&candidate, "tool.search", "description");
    let fetch_description = parameter_value(&candidate, "tool.fetch", "description");
    assert!(search_description.contains("The query must be a single string."));
    assert!(!search_description.contains("Timeouts must be retried."));
    assert!(fetch_description.contains("Timeouts must be retried."));
    assert!(!fetch_description.contains("single string"));
    assert_eq!(
        ToolStrategy::operator_id("search"),
        "tool.search".to_owned()
    );
}

#[tokio::test]
async fn memory_entries_stay_bounded_with_newest_last() {
    let existing: Vec<String> = (0..16).map(|index| format!("lesson {index}")).collect();
    let mut optimizer = MemoryOptimizer::new().with_operator(
        OperatorState::new("memory", OperatorKind::Memory)
            .with_parameter("entries", serde_json::to_string(&existing).unwrap()),
    );
    let feedback = signal(EvolutionTarget::Memory, "new lesson");
    assert_eq!(
        optimizer.backward(&[feedback.clone(), feedback]).unwrap(),
        1
    );
    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    let entries: Vec<String> =
        serde_json::from_str(&parameter_value(&candidate, "memory", "entries"))
            .expect("entry json");
    assert_eq!(entries.len(), MemoryStrategy::new().capacity);
    assert_eq!(entries.last().map(String::as_str), Some("new lesson"));
    // The lowest-weight (unmatched) entry is the one dropped.
    assert!(!entries.contains(&"lesson 0".to_owned()));
    assert!(entries.contains(&"lesson 15".to_owned()));
}

#[tokio::test]
async fn skill_experience_bullets_are_prefixed_and_deduped() {
    let mut optimizer = SkillExperienceOptimizer::new().with_operator(
        OperatorState::new("skill.docs", OperatorKind::Skill).with_parameter("experience", "[]"),
    );
    optimizer
        .backward(&[
            signal(EvolutionTarget::Skill, "Check the schema first."),
            signal(EvolutionTarget::Skill, "Check the schema first."),
        ])
        .unwrap();
    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    let bullets: Vec<String> =
        serde_json::from_str(&parameter_value(&candidate, "skill.docs", "experience"))
            .expect("bullet json");
    assert_eq!(bullets, vec!["- Check the schema first.".to_owned()]);
}

#[tokio::test]
async fn metis_orders_sections_by_utility() {
    let mut optimizer = MetisOptimizer::new().with_operator(
        OperatorState::new("context", OperatorKind::Context)
            .with_parameter("selection", "")
            .with_parameter("Body", "body text")
            .with_parameter("Tools", "tool text"),
    );
    let body = signal(EvolutionTarget::Context, "body evidence")
        .with_section("Body")
        .with_weight(0.3)
        .unwrap();
    let tools = signal(EvolutionTarget::Context, "tool evidence")
        .with_section("Tools")
        .with_weight(0.8)
        .unwrap();
    assert_eq!(optimizer.backward(&[body, tools]).unwrap(), 1);
    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    let selection: Value =
        serde_json::from_str(&parameter_value(&candidate, "context", "selection"))
            .expect("selection json");
    assert_eq!(selection["sections"], json!(["Tools", "Body"]));
    assert_eq!(selection["budget"], json!(8));
    assert_eq!(selection["dropped"], json!([]));
}

#[tokio::test]
async fn ttse_clusters_similar_feedback() {
    let mut optimizer = TtseOptimizer::new()
        .with_operator(OperatorState::new("tips", OperatorKind::Tip).with_parameter("tips", "[]"));
    optimizer
        .backward(&[
            signal(EvolutionTarget::Other, "Retry the failed step"),
            signal(EvolutionTarget::Other, "Retry the failed step twice"),
            signal(EvolutionTarget::Other, "Always validate the schema"),
        ])
        .unwrap();
    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    let tips: Vec<String> =
        serde_json::from_str(&parameter_value(&candidate, "tips", "tips")).expect("tip json");
    assert_eq!(
        tips,
        vec![
            "Retry the failed step".to_owned(),
            "Always validate the schema".to_owned(),
        ]
    );
    assert_eq!(TtseStrategy::new().capacity, 12);
}

#[tokio::test]
async fn model_proposal_uses_the_improved_tag() {
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::text(
        "<improved>Better instruction</improved>",
    ));
    let mut optimizer = InstructionOptimizer::new()
        .with_model(model)
        .with_operator(instruction_operator());
    optimizer
        .backward(&[signal(
            EvolutionTarget::Instruction,
            "Plans omit verification.",
        )])
        .unwrap();

    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    assert_eq!(
        parameter_value(&candidate, "planner", "instruction"),
        "Better instruction"
    );
    assert_eq!(
        optimizer.provenance(&candidate.id),
        Some("planner.instruction: model")
    );
    assert!(optimizer.history()[0].provenance.contains("model"));
}

#[tokio::test]
async fn unparseable_model_response_falls_back_to_the_rewrite() {
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("no tags in this answer"));
    let mut optimizer = InstructionOptimizer::new()
        .with_model(model)
        .with_operator(instruction_operator());
    optimizer
        .backward(&[signal(
            EvolutionTarget::Instruction,
            "Plans omit verification.",
        )])
        .unwrap();

    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    let value = parameter_value(&candidate, "planner", "instruction");
    assert!(value.contains("Plans omit verification."));
    let provenance = optimizer.provenance(&candidate.id).expect("provenance");
    assert!(
        provenance.contains("fallback: unparseable model response"),
        "unexpected provenance: {provenance}"
    );
}

#[tokio::test]
async fn failing_model_falls_back_to_the_rewrite() {
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::new([ScriptedTurn::fail(
        "provider exploded",
    )]));
    let mut optimizer = SkillExperienceOptimizer::new()
        .with_model(model)
        .with_operator(
            OperatorState::new("skill.docs", OperatorKind::Skill)
                .with_parameter("experience", "[]"),
        );
    optimizer
        .backward(&[signal(EvolutionTarget::Skill, "Check the schema first.")])
        .unwrap();

    let candidate = optimizer.propose(1).await.expect("propose").remove(0);
    let bullets: Vec<String> =
        serde_json::from_str(&parameter_value(&candidate, "skill.docs", "experience"))
            .expect("bullet json");
    assert_eq!(bullets, vec!["- Check the schema first.".to_owned()]);
    let provenance = optimizer.provenance(&candidate.id).expect("provenance");
    assert!(
        provenance.contains("fallback: model error"),
        "unexpected provenance: {provenance}"
    );
    assert!(provenance.contains("provider exploded"));
}

#[test]
fn strategy_defaults_match_the_documented_bounds() {
    assert_eq!(InstructionStrategy::new().max_bullets, 5);
    assert_eq!(InstructionStrategy::new().max_chars, 2_000);
    assert_eq!(ToolStrategy::new().max_clauses, 3);
    assert_eq!(MemoryStrategy::new().capacity, 16);
    assert_eq!(MemoryStrategy::new().top_k, 4);
    assert_eq!(SkillExperienceStrategy::new().capacity, 16);
    assert_eq!(MetisStrategy::new().budget, 8);
    assert_eq!(TtseStrategy::new().capacity, 12);
    assert_eq!(TtseStrategy::new().threshold, 0.5);
}
