#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for `openjiuwen_agent_evolving::signal`.
//!
//! The suite is fully offline and deterministic: signal records are pure serde
//! data, the tracker uses a monotonic counter instead of a clock, and every
//! extractor is a pure function of its inputs. No model, environment or file is
//! touched.

use std::collections::BTreeMap;

use openjiuwen_agent_evolving::dataset::{Case, EvaluatedCase};
use openjiuwen_agent_evolving::signal::{
    normalize_excerpt, signals_from_conversation, signals_from_evaluated_cases,
    signals_from_failure, signals_from_review_feedback, signals_from_team_feedback,
    truncate_excerpt, EvolutionCategory, EvolutionSignal, EvolutionTarget, SignalTracker,
    TrackedSignal, MAX_EXCERPT_CHARS, MAX_SIGNALS_PER_INPUT, NEGATIVE_MARKERS,
};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::prompt::BaseMessage;
use serde_json::{json, Value};

/// A tolerant float comparison, used instead of direct `==` on `f64`.
fn approx(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

/// Builds a `BTreeMap` from string keys and JSON values.
fn map(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

/// A valid case with a fixed id, one input key and one label key.
fn case(id: &str, input_key: &str, label_key: &str) -> Case {
    Case::new(
        map(&[(input_key, json!("input"))]),
        map(&[(label_key, json!("label"))]),
    )
    .unwrap()
    .with_case_id(id)
}

/// A case evaluated with a fixed score, no answer and a fixed reason.
fn evaluated(id: &str, input_key: &str, label_key: &str, score: f64) -> EvaluatedCase {
    EvaluatedCase::new(case(id, input_key, label_key), None, score, "reason").unwrap()
}

/// A freshly constructed signal.
fn signal(kind: &str, target: EvolutionTarget, excerpt: &str) -> EvolutionSignal {
    EvolutionSignal::new(kind, target, excerpt).unwrap()
}

/// Records `signal` `times` times, exercising the duplicate path.
fn record_times(tracker: &mut SignalTracker, candidate: &EvolutionSignal, times: usize) {
    for _ in 0..times {
        tracker.record(candidate.clone());
    }
}

/// The accumulated weight stored under `key`.
fn tracked_weight(tracker: &SignalTracker, key: &str) -> f64 {
    tracker.seen.get(key).expect("tracked entry").weight
}

// ---------------------------------------------------------------------------
// 1. Construction and validation
// ---------------------------------------------------------------------------

#[test]
fn new_rejects_empty_signal_type_and_excerpt() {
    for kind in ["", "   ", "\t\n"] {
        let error = EvolutionSignal::new(kind, EvolutionTarget::Tool, "excerpt").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
    for excerpt in ["", "   ", "\t\n"] {
        let error = EvolutionSignal::new("kind", EvolutionTarget::Tool, excerpt).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}

#[test]
fn new_trims_and_fills_defaults() {
    let signal = EvolutionSignal::new("  kind  ", EvolutionTarget::Tool, "  excerpt  ").unwrap();
    assert_eq!(signal.signal_type, "kind");
    assert_eq!(signal.excerpt, "excerpt");
    assert_eq!(signal.category, EvolutionCategory::Tool);
    assert_eq!(signal.target, EvolutionTarget::Tool);
    assert_eq!(signal.section, "");
    assert!(approx(signal.confidence, 1.0));
    assert!(approx(signal.weight, 1.0));
    assert!(signal.skill_name.is_none());
    assert!(signal.tool_name.is_none());
    assert!(signal.source.is_none());
    assert!(signal.context.is_empty());
}

#[test]
fn category_is_projected_from_the_target() {
    let instruction = EvolutionSignal::new("k", EvolutionTarget::Instruction, "e").unwrap();
    assert_eq!(instruction.category(), EvolutionCategory::Description);
    let tool = EvolutionSignal::new("k", EvolutionTarget::Tool, "e").unwrap();
    assert_eq!(tool.category(), EvolutionCategory::Tool);
}

#[test]
fn confidence_and_weight_are_unit_interval_bounded() {
    let base = signal("k", EvolutionTarget::Body, "e");
    for value in [1.5, -0.1, f64::NAN] {
        assert_eq!(
            base.clone().with_confidence(value).unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            base.clone().with_weight(value).unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
    for value in [0.0, 0.25, 1.0] {
        let signal = base.clone().with_confidence(value).unwrap();
        assert!(approx(signal.confidence, value));
        let signal = base.clone().with_weight(value).unwrap();
        assert!(approx(signal.weight, value));
    }
}

#[test]
fn validate_catches_mutated_bounds() {
    let mut signal = signal("k", EvolutionTarget::Body, "e");
    assert!(signal.validate().is_ok());
    signal.confidence = 2.0;
    assert_eq!(signal.validate().unwrap_err().kind(), ErrorKind::Validation);
    signal.confidence = 0.5;
    signal.weight = -1.0;
    assert_eq!(signal.validate().unwrap_err().kind(), ErrorKind::Validation);
}

#[test]
fn with_category_overrides_the_projection() {
    let overridden =
        signal("k", EvolutionTarget::Tool, "e").with_category(EvolutionCategory::Context);
    assert_eq!(overridden.category(), EvolutionCategory::Context);
    assert_eq!(overridden.target(), EvolutionTarget::Tool);
}

// ---------------------------------------------------------------------------
// 2. Dedupe keys, normalisation and truncation
// ---------------------------------------------------------------------------

#[test]
fn dedupe_key_normalises_whitespace_and_case() {
    let left = signal("a", EvolutionTarget::Tool, "Bad   Output\n");
    let right = signal("b", EvolutionTarget::Tool, "bad output");
    assert_eq!(left.dedupe_key(), right.dedupe_key());
    assert_eq!(left.dedupe_key(), "tool:tool:bad output");
}

#[test]
fn dedupe_key_changes_with_target_or_category() {
    let base = signal("a", EvolutionTarget::Tool, "same");
    let other_target = signal("a", EvolutionTarget::Body, "same");
    assert_ne!(base.dedupe_key(), other_target.dedupe_key());
    let other_category = base.clone().with_category(EvolutionCategory::Context);
    assert_ne!(base.dedupe_key(), other_category.dedupe_key());
}

#[test]
fn normalize_and_truncate_are_deterministic() {
    assert_eq!(normalize_excerpt("  A\t B\nC "), "a b c");
    assert_eq!(truncate_excerpt("héllo", 2), "hé");
    assert_eq!(truncate_excerpt("short", 240), "short");
    let chinese = "错".repeat(300);
    let cut = truncate_excerpt(&chinese, MAX_EXCERPT_CHARS);
    assert_eq!(cut.chars().count(), MAX_EXCERPT_CHARS);
    assert_eq!(cut, "错".repeat(MAX_EXCERPT_CHARS));
}

// ---------------------------------------------------------------------------
// 3. Tracker merging
// ---------------------------------------------------------------------------

#[test]
fn tracker_merges_duplicates_by_normalised_excerpt() {
    let mut tracker = SignalTracker::new();
    let first = EvolutionSignal::new("first_kind", EvolutionTarget::Tool, "Same   Text")
        .unwrap()
        .with_confidence(0.4)
        .unwrap()
        .with_weight(1.0)
        .unwrap();
    let key = first.dedupe_key();
    assert_eq!(tracker.record(first), 1);

    let second = EvolutionSignal::new("second_kind", EvolutionTarget::Tool, " same\ntext ")
        .unwrap()
        .with_confidence(0.8)
        .unwrap()
        .with_weight(0.5)
        .unwrap();
    assert_eq!(second.dedupe_key(), key);
    assert_eq!(tracker.record(second), 2);

    assert_eq!(tracker.len(), 1);
    let entry: &TrackedSignal = tracker.seen.get(&key).unwrap();
    assert_eq!(entry.hits, 2);
    assert_eq!(entry.signal.signal_type, "first_kind");
    assert!(approx(entry.signal.confidence, 0.8));
    assert!(approx(entry.weight, 1.5));
    assert_eq!(entry.first_seq, 1);
    assert_eq!(entry.last_seq, 2);

    let third = signal("third_kind", EvolutionTarget::Body, "another");
    assert_eq!(tracker.record(third), 1);
    let third_key = tracker
        .signals()
        .iter()
        .find(|candidate| candidate.signal_type == "third_kind")
        .unwrap()
        .dedupe_key();
    let entry = tracker.seen.get(&third_key).unwrap();
    assert_eq!(entry.first_seq, 3);
    assert_eq!(entry.last_seq, 3);
}

#[test]
fn record_all_returns_the_number_recorded() {
    let mut tracker = SignalTracker::new();
    let count = tracker.record_all(vec![
        signal("a", EvolutionTarget::Body, "x"),
        signal("b", EvolutionTarget::Body, "y"),
        signal("a", EvolutionTarget::Body, "x"),
    ]);
    assert_eq!(count, 3);
    assert_eq!(tracker.len(), 2);
}

// ---------------------------------------------------------------------------
// 4. Eviction and determinism
// ---------------------------------------------------------------------------

#[test]
fn tracker_evicts_the_lightest_entry() {
    let mut tracker = SignalTracker::new().with_capacity(2);
    let light = signal("light", EvolutionTarget::Body, "alpha");
    let medium = signal("medium", EvolutionTarget::Body, "bravo");
    let heavy = signal("heavy", EvolutionTarget::Body, "charlie");
    let light_key = light.dedupe_key();
    record_times(&mut tracker, &light, 1);
    record_times(&mut tracker, &medium, 2);
    record_times(&mut tracker, &heavy, 3);

    assert_eq!(tracker.len(), 2);
    assert!(!tracker.seen.contains_key(&light_key));
    assert!(approx(tracked_weight(&tracker, &medium.dedupe_key()), 2.0));
    assert!(approx(tracked_weight(&tracker, &heavy.dedupe_key()), 3.0));
}

#[test]
fn equal_weights_evict_the_oldest_entry() {
    let mut tracker = SignalTracker::new().with_capacity(2);
    let first = signal("same", EvolutionTarget::Body, "ccc");
    let second = signal("same", EvolutionTarget::Body, "aaa");
    let third = signal("same", EvolutionTarget::Body, "bbb");
    let first_key = first.dedupe_key();
    tracker.record(first);
    tracker.record(second);
    tracker.record(third);

    assert_eq!(tracker.len(), 2);
    assert!(!tracker.seen.contains_key(&first_key));
}

#[test]
fn zero_capacity_stores_nothing() {
    let mut tracker = SignalTracker::new().with_capacity(0);
    assert_eq!(tracker.record(signal("k", EvolutionTarget::Body, "e")), 1);
    assert_eq!(tracker.len(), 0);
    assert!(tracker.is_empty());
}

#[test]
fn identical_input_sequences_order_identically() {
    let inputs = vec![
        signal("a", EvolutionTarget::Tool, "zulu"),
        signal("b", EvolutionTarget::Body, "alpha"),
        signal("c", EvolutionTarget::Memory, "mike"),
    ];
    let mut left = SignalTracker::new();
    let mut right = SignalTracker::new();
    left.record_all(inputs.clone());
    right.record_all(inputs);

    let collect = |tracker: &SignalTracker| -> Vec<String> {
        tracker
            .signals()
            .iter()
            .map(|candidate| candidate.dedupe_key())
            .collect()
    };
    assert_eq!(collect(&left), collect(&right));
}

// ---------------------------------------------------------------------------
// 5. Decay, ordering and drain
// ---------------------------------------------------------------------------

#[test]
fn decay_halves_weights_and_drops_zeroed_entries() {
    let mut tracker = SignalTracker::new().with_decay(0.5).unwrap();
    let heavy = EvolutionSignal::new("heavy", EvolutionTarget::Body, "heavy")
        .unwrap()
        .with_weight(1.0)
        .unwrap();
    let light = EvolutionSignal::new("light", EvolutionTarget::Body, "light")
        .unwrap()
        .with_weight(0.8)
        .unwrap();
    let heavy_key = heavy.dedupe_key();
    let light_key = light.dedupe_key();
    tracker.record(heavy);
    tracker.record(light);
    tracker.decay_all();

    assert!(approx(tracked_weight(&tracker, &heavy_key), 0.5));
    assert!(approx(tracked_weight(&tracker, &light_key), 0.4));

    let mut zero = SignalTracker::new().with_decay(0.0).unwrap();
    zero.record(signal("k", EvolutionTarget::Body, "e"));
    zero.decay_all();
    assert!(zero.is_empty());
}

#[test]
fn invalid_decay_is_rejected() {
    for value in [1.5, -0.5, f64::NAN] {
        let error = SignalTracker::new().with_decay(value).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}

#[test]
fn equal_weights_order_by_dedupe_key() {
    let mut tracker = SignalTracker::new();
    tracker.record_all(vec![
        signal("k", EvolutionTarget::Tool, "bbb"),
        signal("k", EvolutionTarget::Tool, "aaa"),
        signal("k", EvolutionTarget::Tool, "ccc"),
    ]);
    let excerpts: Vec<&str> = tracker
        .signals()
        .iter()
        .map(|candidate| candidate.excerpt.as_str())
        .collect();
    assert_eq!(excerpts, vec!["aaa", "bbb", "ccc"]);
    let top = tracker.top(1);
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].excerpt, "aaa");
}

#[test]
fn group_by_target_preserves_signal_order() {
    let mut tracker = SignalTracker::new();
    let zebra = EvolutionSignal::new("k", EvolutionTarget::Tool, "zebra").unwrap();
    let apple = EvolutionSignal::new("k", EvolutionTarget::Body, "apple")
        .unwrap()
        .with_weight(0.9)
        .unwrap();
    let alpha = EvolutionSignal::new("k", EvolutionTarget::Tool, "alpha")
        .unwrap()
        .with_weight(0.8)
        .unwrap();
    tracker.record(zebra);
    tracker.record(apple);
    tracker.record(alpha);

    let grouped = tracker.group_by_target();
    let keys: Vec<EvolutionTarget> = grouped.keys().copied().collect();
    assert_eq!(keys, vec![EvolutionTarget::Body, EvolutionTarget::Tool]);

    let body: Vec<&str> = grouped[&EvolutionTarget::Body]
        .iter()
        .map(|candidate| candidate.excerpt.as_str())
        .collect();
    assert_eq!(body, vec!["apple"]);
    let tool: Vec<&str> = grouped[&EvolutionTarget::Tool]
        .iter()
        .map(|candidate| candidate.excerpt.as_str())
        .collect();
    assert_eq!(tool, vec!["zebra", "alpha"]);
}

#[test]
fn drain_returns_signals_and_resets() {
    let mut tracker = SignalTracker::new();
    tracker.record(signal("x", EvolutionTarget::Tool, "bbb"));
    tracker.record(signal("y", EvolutionTarget::Body, "aaa"));

    let expected: Vec<String> = tracker
        .signals()
        .iter()
        .map(|candidate| candidate.excerpt.clone())
        .collect();
    let drained: Vec<String> = tracker
        .drain()
        .iter()
        .map(|candidate| candidate.excerpt.clone())
        .collect();
    assert_eq!(drained, expected);
    assert!(tracker.is_empty());
    assert_eq!(tracker.seq, 0);

    assert_eq!(
        tracker.record(signal("z", EvolutionTarget::Body, "fresh")),
        1
    );
    let fresh_key = tracker.signals()[0].dedupe_key();
    assert_eq!(tracker.seen.get(&fresh_key).unwrap().first_seq, 1);
}

#[test]
fn clear_resets_contents_and_sequence() {
    let mut tracker = SignalTracker::new();
    tracker.record(signal("x", EvolutionTarget::Tool, "bbb"));
    tracker.clear();
    assert!(tracker.is_empty());
    assert_eq!(tracker.seq, 0);

    tracker.record(signal("z", EvolutionTarget::Body, "again"));
    let key = tracker.signals()[0].dedupe_key();
    assert_eq!(tracker.seen.get(&key).unwrap().first_seq, 1);
}

// ---------------------------------------------------------------------------
// 6. Serde
// ---------------------------------------------------------------------------

#[test]
fn full_signal_round_trips_through_json() {
    let mut context = BTreeMap::new();
    context.insert("k".to_owned(), json!(3));
    let signal = EvolutionSignal::new("kind", EvolutionTarget::Tool, "Excerpt")
        .unwrap()
        .with_section("Section")
        .with_skill("skill")
        .with_tool("tool")
        .with_source("source")
        .with_confidence(0.3)
        .unwrap()
        .with_weight(0.7)
        .unwrap()
        .with_context(context);
    let value = serde_json::to_value(&signal).unwrap();
    let restored: EvolutionSignal = serde_json::from_value(value).unwrap();
    assert_eq!(restored, signal);
}

#[test]
fn minimal_json_fills_defaults_and_ignores_unknown_fields() {
    let restored: EvolutionSignal =
        serde_json::from_str(r#"{"signal_type":"x","excerpt":"e","mystery":1}"#).unwrap();
    assert_eq!(restored.signal_type, "x");
    assert_eq!(restored.excerpt, "e");
    assert_eq!(restored.category, EvolutionCategory::Other);
    assert_eq!(restored.target, EvolutionTarget::Other);
    assert_eq!(restored.section, "");
    assert!(approx(restored.confidence, 1.0));
    assert!(approx(restored.weight, 1.0));
    assert!(restored.skill_name.is_none());
    assert!(restored.tool_name.is_none());
    assert!(restored.source.is_none());
    assert!(restored.context.is_empty());
}

#[test]
fn enum_wire_strings_are_snake_case() {
    assert_eq!(
        serde_json::to_value(EvolutionCategory::Tool).unwrap(),
        json!("tool")
    );
    assert_eq!(
        serde_json::to_value(EvolutionTarget::Instruction).unwrap(),
        json!("instruction")
    );
    assert_eq!(
        serde_json::to_value(EvolutionCategory::Other).unwrap(),
        json!("other")
    );
}

#[test]
fn tracker_round_trips_through_json() {
    let mut tracker = SignalTracker::new().with_capacity(4);
    tracker.record(signal("a", EvolutionTarget::Tool, "one"));
    tracker.record(signal("b", EvolutionTarget::Body, "two"));
    let value = serde_json::to_value(&tracker).unwrap();
    let restored: SignalTracker = serde_json::from_value(value).unwrap();
    assert_eq!(restored, tracker);
}

// ---------------------------------------------------------------------------
// 7. Evaluated cases
// ---------------------------------------------------------------------------

#[test]
fn only_cases_strictly_below_threshold_yield_signals() {
    let cases = vec![
        evaluated("case-tool", "question", "tool", 0.2),
        evaluated("case-ok", "question", "answer", 0.9),
        evaluated("case-edge", "question", "answer", 0.5),
    ];
    let signals = signals_from_evaluated_cases(&cases, 0.5);
    assert_eq!(signals.len(), 1);

    let signal = &signals[0];
    assert_eq!(signal.signal_type, "evaluated");
    assert_eq!(signal.target, EvolutionTarget::Tool);
    assert_eq!(signal.category, EvolutionCategory::Tool);
    assert_eq!(signal.source.as_deref(), Some("evaluator"));
    assert_eq!(signal.section, "Troubleshooting");
    assert!(approx(signal.confidence, 0.8));
    assert!(signal.excerpt.contains("case-tool"));
    assert!(signal.excerpt.contains("0.20"));
    assert_eq!(signal.context["case_id"], json!("case-tool"));
    assert_eq!(signal.context["score"], json!(0.2));
    assert_eq!(signal.context["label"], json!({"tool": "label"}));
    assert_eq!(signal.context["prediction"], Value::Null);
    assert_eq!(signal.context["reason"], json!("reason"));
}

#[test]
fn zero_score_is_reported_as_low_score() {
    let signals =
        signals_from_evaluated_cases(&[evaluated("case-zero", "question", "answer", 0.0)], 0.5);
    assert_eq!(signals.len(), 1);
    assert_eq!(signals[0].signal_type, "low_score");
    assert!(approx(signals[0].confidence, 1.0));
}

#[test]
fn evaluation_prediction_carries_the_answer_map() {
    let answer = map(&[("answer", json!("out"))]);
    let evaluated = EvaluatedCase::new(
        case("case-answer", "question", "answer"),
        Some(answer),
        0.1,
        "reason",
    )
    .unwrap();
    let signals = signals_from_evaluated_cases(&[evaluated], 0.5);
    assert_eq!(signals.len(), 1);
    assert_eq!(signals[0].context["prediction"], json!({"answer": "out"}));
}

#[test]
fn evaluation_target_follows_label_keyword_priority() {
    let signals = signals_from_evaluated_cases(
        &[evaluated("case-instr", "question", "instruction", 0.1)],
        0.5,
    );
    assert_eq!(signals[0].target, EvolutionTarget::Instruction);
    assert_eq!(signals[0].category, EvolutionCategory::Description);

    let signals =
        signals_from_evaluated_cases(&[evaluated("case-body", "question", "answer", 0.1)], 0.5);
    assert_eq!(signals[0].target, EvolutionTarget::Body);
    assert_eq!(signals[0].category, EvolutionCategory::Body);
}

#[test]
fn zero_threshold_and_repeat_calls() {
    let cases = vec![evaluated("case-tool", "question", "tool", 0.2)];
    assert_eq!(signals_from_evaluated_cases(&cases, 0.0), Vec::new());
    assert_eq!(
        signals_from_evaluated_cases(&cases, 0.5),
        signals_from_evaluated_cases(&cases, 0.5)
    );
}

// ---------------------------------------------------------------------------
// 8. Conversation
// ---------------------------------------------------------------------------

#[test]
fn conversation_extracts_user_and_tool_feedback_only() {
    let messages = vec![
        BaseMessage::user("This is wrong, please fix the tool usage"),
        BaseMessage::tool("call-1", "Tool failed to run: timeout"),
        BaseMessage::assistant("wrong answer"),
        BaseMessage::user("all good"),
    ];
    let signals = signals_from_conversation(&messages, None);
    assert_eq!(signals.len(), 2);

    let user = &signals[0];
    assert_eq!(user.signal_type, "negative_feedback");
    assert_eq!(user.section, "Instructions");
    assert_eq!(user.source.as_deref(), Some("conversation"));
    assert_eq!(user.target, EvolutionTarget::Tool);
    assert!(approx(user.confidence, 0.6));
    assert!(user.skill_name.is_none());
    assert_eq!(user.excerpt, "This is wrong, please fix the tool usage");

    let tool = &signals[1];
    assert_eq!(tool.signal_type, "execution_failure");
    assert_eq!(tool.section, "Troubleshooting");
    assert_eq!(tool.source.as_deref(), Some("conversation"));
    assert_eq!(tool.target, EvolutionTarget::Tool);
    assert!(approx(tool.confidence, 0.6));
    assert_eq!(tool.excerpt, "Tool failed to run: timeout");
}

#[test]
fn conversation_excerpts_are_bounded() {
    let long = format!("{}失败", "错".repeat(300));
    let messages = vec![BaseMessage::user(long)];
    let signals = signals_from_conversation(&messages, Some("skill-x"));
    assert_eq!(signals.len(), 1);
    assert_eq!(signals[0].excerpt.chars().count(), MAX_EXCERPT_CHARS);
    assert_eq!(signals[0].skill_name.as_deref(), Some("skill-x"));
}

#[test]
fn conversation_is_repeatable() {
    let messages = vec![BaseMessage::user("this is wrong")];
    assert_eq!(
        signals_from_conversation(&messages, None),
        signals_from_conversation(&messages, None)
    );
}

// ---------------------------------------------------------------------------
// 9. Review feedback
// ---------------------------------------------------------------------------

#[test]
fn review_feedback_segments_on_markers() {
    let text = "Looks wrong.\nGreat job!\n请修复这个问题。\nAnother 失败 line";
    let signals = signals_from_review_feedback(text, Some("skill-x"));
    // Three segments carry markers ("wrong", "修复", "失败"); "Great job" does not.
    assert_eq!(signals.len(), 3);
    for signal in &signals {
        assert_eq!(signal.signal_type, "review_feedback");
        assert_eq!(signal.section, "Review");
        assert_eq!(signal.source.as_deref(), Some("review"));
        assert_eq!(signal.skill_name.as_deref(), Some("skill-x"));
        assert!(approx(signal.confidence, 0.7));
    }
    assert_eq!(signals[0].excerpt, "Looks wrong");
    assert_eq!(signals[1].excerpt, "请修复这个问题");
    assert_eq!(signals[2].excerpt, "Another 失败 line");
}

#[test]
fn review_feedback_matches_case_insensitively() {
    let signals = signals_from_review_feedback("FAILED hard", None);
    assert_eq!(signals.len(), 1);
    assert_eq!(signals[0].excerpt, "FAILED hard");
}

#[test]
fn review_feedback_without_markers_is_empty() {
    assert_eq!(
        signals_from_review_feedback("all good, thanks", None),
        Vec::new()
    );
    assert_eq!(
        signals_from_review_feedback("Looks wrong.", None),
        signals_from_review_feedback("Looks wrong.", None)
    );
}

// ---------------------------------------------------------------------------
// 10. Team feedback and failure convenience
// ---------------------------------------------------------------------------

#[test]
fn team_feedback_records_the_agent() {
    let signals = signals_from_team_feedback("agent-7", "The tool failed.\nAll good.");
    assert_eq!(signals.len(), 1);
    let signal = &signals[0];
    assert_eq!(signal.signal_type, "team_feedback");
    assert_eq!(signal.section, "Collaboration");
    assert_eq!(signal.source.as_deref(), Some("team"));
    assert_eq!(signal.skill_name.as_deref(), Some("agent-7"));
    assert_eq!(signal.target, EvolutionTarget::Tool);
    assert!(approx(signal.confidence, 0.7));
    assert_eq!(signal.context["agent_id"], json!("agent-7"));
}

#[test]
fn failure_extracts_matching_segments() {
    let signals = signals_from_failure("Timeout failed. Then it worked.", EvolutionTarget::Body);
    assert_eq!(signals.len(), 1);
    let signal = &signals[0];
    assert_eq!(signal.signal_type, "execution_failure");
    assert_eq!(signal.section, "Troubleshooting");
    assert!(signal.source.is_none());
    assert!(approx(signal.confidence, 0.6));
    assert_eq!(signal.excerpt, "Timeout failed");
}

#[test]
fn failure_falls_back_to_the_whole_text_and_truncates() {
    let text = "z".repeat(400);
    let signals = signals_from_failure(&text, EvolutionTarget::Tool);
    assert_eq!(signals.len(), 1);
    assert_eq!(signals[0].excerpt.chars().count(), MAX_EXCERPT_CHARS);
    assert_eq!(signals[0].signal_type, "execution_failure");
}

#[test]
fn failure_on_empty_text_is_empty() {
    assert_eq!(
        signals_from_failure("   ", EvolutionTarget::Body),
        Vec::new()
    );
}

// ---------------------------------------------------------------------------
// 11. Constants and cap
// ---------------------------------------------------------------------------

#[test]
fn documented_constants_are_stable() {
    assert_eq!(MAX_EXCERPT_CHARS, 240);
    assert_eq!(MAX_SIGNALS_PER_INPUT, 64);
    assert_eq!(NEGATIVE_MARKERS.len(), 13);
    for marker in [
        "wrong",
        "incorrect",
        "not correct",
        "failed",
        "doesn't work",
        "does not work",
        "please fix",
        "fix this",
        "错误",
        "不对",
        "失败",
        "修复",
        "重试",
    ] {
        assert!(
            NEGATIVE_MARKERS.contains(&marker),
            "missing marker {marker}"
        );
    }
}

#[test]
fn conversation_output_is_capped() {
    let messages: Vec<BaseMessage> = (0..70)
        .map(|index| BaseMessage::user(format!("wrong usage {index}")))
        .collect();
    let signals = signals_from_conversation(&messages, None);
    assert_eq!(signals.len(), MAX_SIGNALS_PER_INPUT);
}
