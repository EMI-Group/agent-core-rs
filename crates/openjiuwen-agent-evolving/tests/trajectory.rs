//! Offline integration tests for the `trajectory` module.
//!
//! Everything here is deterministic: no clock, no model, no network.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assert_is_empty
)]

use openjiuwen_agent_evolving::trajectory::{
    first_error, from_messages, last_n, longest_span, render_digest, span_by_name, spans,
    window_around, windows, StepKind, StepStatus, Trajectory, TrajectoryBuilder,
    TrajectoryGroupSummary, TrajectoryOutcome, TrajectoryStats, TrajectoryStep, TrajectoryStore,
};
use openjiuwen_common::error::Error;
use openjiuwen_core::foundation::prompt::{BaseMessage, ContentPart, MessageContent};
use serde_json::{json, Value};

fn step_trajectory(id: &str, count: usize) -> Trajectory {
    let mut trajectory = Trajectory::new(id).expect("valid id");
    for position in 0..count {
        trajectory.push(TrajectoryStep::new(
            StepKind::Message,
            format!("step {position}"),
        ));
    }
    trajectory
}

#[test]
fn new_validates_the_identifier() {
    assert!(matches!(Trajectory::new(""), Err(Error::Validation(_))));
    assert!(matches!(Trajectory::new("   "), Err(Error::Validation(_))));
    let trajectory = Trajectory::new("traj-1").expect("valid id");
    assert!(trajectory.is_empty());
    assert_eq!(trajectory.outcome, TrajectoryOutcome::Unknown);
    assert_eq!(trajectory.score, None);
    assert_eq!(trajectory.elapsed_ms, None);
}

#[test]
fn push_and_extend_assign_contiguous_indices() {
    let mut trajectory = Trajectory::new("traj-1").expect("valid id");
    // Explicit indices are overwritten on insertion.
    trajectory.push(TrajectoryStep::new(StepKind::Message, "a"));
    trajectory.extend([
        TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("search"),
        TrajectoryStep::new(StepKind::ToolResult, "hit"),
    ]);
    let indices: Vec<usize> = trajectory.steps().iter().map(|step| step.index).collect();
    assert_eq!(indices, vec![0, 1, 2]);
    assert_eq!(trajectory.len(), 3);
    assert_eq!(trajectory.tool_names(), vec!["search"]);
    assert_eq!(trajectory.error_count(), 0);
}

#[test]
fn score_must_be_finite_and_metadata_is_stored() {
    let trajectory = Trajectory::new("traj-1")
        .expect("valid id")
        .with_metadata("source", json!("offline"));
    assert!(matches!(
        trajectory.clone().with_score(f64::INFINITY),
        Err(Error::Validation(_))
    ));
    let scored = trajectory.with_score(0.75).expect("finite score");
    assert_eq!(scored.score, Some(0.75));
    assert_eq!(scored.metadata.get("source"), Some(&json!("offline")));
}

#[test]
fn builder_records_messages_and_tool_flow() {
    let builder = TrajectoryBuilder::new("traj-1")
        .expect("valid id")
        .with_session("session-1")
        .with_case("case-1")
        .push_message(&BaseMessage::system("be brief"))
        .push_message(&BaseMessage::user("what is 2 + 2?"))
        .push_message(&BaseMessage::assistant("calling calc"))
        .push_tool_call("calc", "call-1", "{\"expr\":\"2+2\"}")
        .push_tool_result("call-1", "4", StepStatus::Ok)
        .push_message(&BaseMessage::tool("call-1", "4"))
        .push_model_output("The answer is 4.");

    assert_eq!(builder.len(), 7);
    assert!(!builder.is_empty());
    let trajectory = builder
        .with_metadata("model", json!("fake"))
        .finish(TrajectoryOutcome::Success);

    let kinds: Vec<StepKind> = trajectory.steps().iter().map(|step| step.kind).collect();
    assert_eq!(
        kinds,
        vec![
            StepKind::System,
            StepKind::Message,
            StepKind::Message,
            StepKind::ToolCall,
            StepKind::ToolResult,
            StepKind::ToolResult,
            StepKind::Model,
        ]
    );
    let roles: Vec<Option<&str>> = trajectory
        .steps()
        .iter()
        .map(|step| step.role.as_deref())
        .collect();
    assert_eq!(
        roles,
        vec![
            Some("system"),
            Some("user"),
            Some("assistant"),
            None,
            None,
            Some("tool"),
            Some("assistant"),
        ]
    );
    assert_eq!(trajectory.steps()[3].tool_name.as_deref(), Some("calc"));
    assert_eq!(
        trajectory.steps()[3].tool_call_id.as_deref(),
        Some("call-1")
    );
    assert_eq!(
        trajectory.steps()[5].tool_call_id.as_deref(),
        Some("call-1")
    );
    assert_eq!(trajectory.session_id.as_deref(), Some("session-1"));
    assert_eq!(trajectory.case_id.as_deref(), Some("case-1"));
    assert!(trajectory.is_successful());
}

#[test]
fn non_text_message_content_is_marked() {
    let message = BaseMessage::user(MessageContent::Parts(vec![ContentPart::ImageUrl {
        image_url: openjiuwen_core::foundation::prompt::ImageUrl {
            url: "https://example.invalid/picture.png".to_owned(),
            detail: None,
        },
    }]));
    let trajectory = from_messages("traj-1", Some("session-1"), &[message]).expect("valid id");
    assert_eq!(trajectory.steps()[0].content, "[non-text content]");
    assert_eq!(trajectory.steps()[0].role.as_deref(), Some("user"));
}

#[test]
fn from_messages_assigns_deterministic_indices() {
    let messages = [
        BaseMessage::system("rules"),
        BaseMessage::user("hello"),
        BaseMessage::assistant("hi"),
    ];
    let first = from_messages("traj-1", None, &messages).expect("valid id");
    let second = from_messages("traj-1", None, &messages).expect("valid id");
    assert_eq!(first, second);
    let indices: Vec<usize> = first.steps().iter().map(|step| step.index).collect();
    assert_eq!(indices, vec![0, 1, 2]);
    assert_eq!(first.session_id, None);
}

#[test]
fn spans_pair_results_and_mark_unclosed_calls() {
    let mut trajectory = Trajectory::new("traj-1").expect("valid id");
    trajectory.push(
        TrajectoryStep::new(StepKind::ToolCall, "{\"q\":\"rust\"}")
            .with_tool_name("search")
            .with_tool_call_id("call-1"),
    );
    trajectory.push(
        TrajectoryStep::new(StepKind::ToolResult, "a result")
            .with_tool_call_id("call-1")
            .with_status(StepStatus::Ok),
    );
    trajectory.push(
        TrajectoryStep::new(StepKind::ToolCall, "{}")
            .with_tool_name("calc")
            .with_tool_call_id("call-2")
            .with_status(StepStatus::Error),
    );
    // A result recorded before its call must not close it.
    trajectory.push(TrajectoryStep::new(StepKind::ToolResult, "early").with_tool_call_id("call-3"));
    trajectory.push(
        TrajectoryStep::new(StepKind::ToolCall, "{}")
            .with_tool_name("fetch")
            .with_tool_call_id("call-3"),
    );

    let projected = spans(&trajectory);
    assert_eq!(projected.len(), 3);

    assert_eq!(projected[0].span_id, "span-0");
    assert_eq!(projected[0].name, "search");
    assert_eq!((projected[0].start, projected[0].end), (0, 1));
    assert!(!projected[0].open);
    assert_eq!(projected[0].status, StepStatus::Ok);

    assert_eq!(projected[1].name, "calc");
    assert!(projected[1].open);
    assert_eq!((projected[1].start, projected[1].end), (2, 2));
    assert_eq!(projected[1].status, StepStatus::Error);

    assert_eq!(projected[2].name, "fetch");
    assert!(projected[2].open);
    assert_eq!((projected[2].start, projected[2].end), (4, 4));

    let longest = longest_span(&trajectory).expect("at least one span");
    assert_eq!(longest.name, "search");
    let failed: Vec<String> = openjiuwen_agent_evolving::trajectory::failed_spans(&trajectory)
        .into_iter()
        .map(|span| span.name)
        .collect();
    assert_eq!(failed, vec!["calc".to_owned()]);
    assert!(span_by_name(&trajectory, "calc").is_some_and(|span| span.open));
    assert!(span_by_name(&trajectory, "missing").is_none());
    assert!(spans(&Trajectory::new("empty").expect("valid id")).is_empty());
}

#[test]
fn spans_propagate_result_status() {
    let trajectory = TrajectoryBuilder::new("traj-1")
        .expect("valid id")
        .push_tool_call("search", "call-1", "{}")
        .push_tool_result("call-1", "boom", StepStatus::Error)
        .build();
    let projected = spans(&trajectory);
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].status, StepStatus::Error);
    assert!(!projected[0].open);
    assert_eq!(
        openjiuwen_agent_evolving::trajectory::failed_spans(&trajectory).len(),
        1
    );
}

#[test]
fn windows_split_with_and_without_overlap() {
    let trajectory = step_trajectory("traj-1", 5);

    let disjoint = windows(&trajectory, 2, 0).expect("valid windows");
    let ranges: Vec<(usize, usize)> = disjoint
        .iter()
        .map(|window| (window.start, window.end))
        .collect();
    assert_eq!(ranges, vec![(0, 2), (2, 4), (4, 5)]);
    assert_eq!(disjoint[2].len(), 1);
    assert_eq!(disjoint[0].steps[0].content, "step 0");

    let overlapping = windows(&trajectory, 3, 1).expect("valid windows");
    let ranges: Vec<(usize, usize)> = overlapping
        .iter()
        .map(|window| (window.start, window.end))
        .collect();
    assert_eq!(ranges, vec![(0, 3), (2, 5)]);

    assert!(windows(&trajectory, 0, 0).is_err());
    assert!(windows(&trajectory, 2, 2).is_err());
    assert!(windows(&Trajectory::new("empty").expect("valid id"), 2, 0)
        .expect("valid windows")
        .is_empty());
}

#[test]
fn window_around_clamps_and_rejects_out_of_bounds() {
    let trajectory = step_trajectory("traj-1", 5);

    let middle = window_around(&trajectory, 2, 1).expect("in bounds");
    assert_eq!((middle.start, middle.end), (1, 4));
    assert_eq!(middle.steps[1].index, 2);

    let edge = window_around(&trajectory, 0, 3).expect("in bounds");
    assert_eq!((edge.start, edge.end), (0, 4));

    let single = window_around(&trajectory, 4, 0).expect("in bounds");
    assert_eq!((single.start, single.end), (4, 5));

    assert!(matches!(
        window_around(&trajectory, 5, 1),
        Err(Error::Validation(_))
    ));
    assert!(window_around(&Trajectory::new("empty").expect("valid id"), 0, 1).is_err());
}

#[test]
fn digest_rendering_is_bounded_utf8_safe_and_deterministic() {
    let mut trajectory = Trajectory::new("traj-1").expect("valid id");
    trajectory.push(TrajectoryStep::new(StepKind::Message, "日本語のテキスト").with_role("user"));
    trajectory.push(TrajectoryStep::new(StepKind::Model, "ok"));

    let bounded = render_digest(&trajectory, 12);
    assert_eq!(bounded.chars().count(), 12);
    assert!(bounded.ends_with('…'));
    assert_eq!(render_digest(&trajectory, 12), bounded);
    assert_eq!(render_digest(&trajectory, 0), "");
    assert_eq!(render_digest(&trajectory, 1), "…");

    let whole = render_digest(&trajectory, 1_000);
    assert!(whole.starts_with("#0 message user: "));
    assert!(whole.contains("\n#1 model -: ok"));

    let digest = trajectory.digest(1_000);
    assert_eq!(digest.trajectory_id, "traj-1");
    assert_eq!(digest.step_count, 2);
    assert_eq!(digest.tool_names, Vec::<String>::new());
    assert_eq!(digest.summary, whole);
    assert_eq!(trajectory.digest(4).summary.chars().count(), 4);

    let empty = Trajectory::new("empty").expect("valid id");
    assert_eq!(render_digest(&empty, 10), "");
    assert_eq!(empty.digest(10).summary, "");
    assert_eq!(empty.digest(10).step_count, 0);
}

#[test]
fn digest_excerpts_individual_steps() {
    let mut trajectory = Trajectory::new("traj-1").expect("valid id");
    trajectory.push(TrajectoryStep::new(StepKind::Message, "x".repeat(500)).with_role("user"));
    let rendered = render_digest(&trajectory, 10_000);
    // One line, prefix plus an 80-character excerpt ending in the marker.
    assert!(rendered.starts_with("#0 message user: xxx"));
    assert!(rendered.ends_with('…'));
    assert!(rendered.chars().count() < 200);
}

#[test]
fn digest_carries_outcome_score_and_tools() {
    let mut trajectory = Trajectory::new("traj-1")
        .expect("valid id")
        .with_outcome(TrajectoryOutcome::Failure)
        .with_score(0.25)
        .expect("finite score");
    trajectory.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("search"));
    trajectory.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("calc"));
    let digest = trajectory.digest(1_000);
    assert_eq!(digest.outcome, TrajectoryOutcome::Failure);
    assert_eq!(digest.score, Some(0.25));
    assert_eq!(
        digest.tool_names,
        vec!["calc".to_owned(), "search".to_owned()]
    );
}

#[test]
fn error_and_tail_helpers() {
    let mut trajectory = step_trajectory("traj-1", 3);
    trajectory
        .push(TrajectoryStep::new(StepKind::ToolResult, "boom").with_status(StepStatus::Error));
    assert_eq!(first_error(&trajectory).map(|step| step.index), Some(3));
    assert_eq!(last_n(&trajectory, 2).len(), 2);
    assert_eq!(last_n(&trajectory, 0).len(), 0);
    assert_eq!(last_n(&trajectory, 100).len(), 4);
    assert!(first_error(&Trajectory::new("empty").expect("valid id")).is_none());
}

fn store_fixture() -> TrajectoryStore {
    let mut store = TrajectoryStore::new();

    let mut first = Trajectory::new("t-1")
        .expect("valid id")
        .with_session("s-1")
        .with_case("case-a")
        .with_outcome(TrajectoryOutcome::Success)
        .with_score(1.0)
        .expect("finite score");
    first.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("search"));
    first.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("calc"));
    store.add(first);

    let mut second = Trajectory::new("t-2")
        .expect("valid id")
        .with_session("s-1")
        .with_case("case-a")
        .with_outcome(TrajectoryOutcome::Failure)
        .with_score(0.0)
        .expect("finite score");
    second.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("search"));
    store.add(second);

    let third = Trajectory::new("t-3")
        .expect("valid id")
        .with_session("s-2")
        .with_case("case-b")
        .with_outcome(TrajectoryOutcome::Unknown);
    store.add(third);

    store
}

#[test]
fn store_groups_and_computes_hand_checked_statistics() {
    let mut store = store_fixture();
    assert_eq!(store.len(), 3);
    assert_eq!(store.by_session("s-1").len(), 2);
    assert_eq!(store.by_case("case-b").len(), 1);
    assert_eq!(store.by_outcome(TrajectoryOutcome::Failure).len(), 1);

    let sessions = store.group_by_session();
    assert_eq!(
        sessions.keys().cloned().collect::<Vec<_>>(),
        vec!["s-1", "s-2"]
    );
    assert_eq!(sessions["s-1"].len(), 2);
    assert_eq!(sessions["s-1"][0].trajectory_id, "t-1");
    let cases = store.group_by_case();
    assert_eq!(
        cases.keys().cloned().collect::<Vec<_>>(),
        vec!["case-a", "case-b"]
    );

    let summary = store.summary();
    assert_eq!(
        summary,
        TrajectoryStats {
            count: 3,
            success_count: 1,
            failure_count: 1,
            success_rate: 1.0 / 3.0,
            mean_score: 0.5,
            tool_usage: [("calc".to_owned(), 1), ("search".to_owned(), 2)]
                .into_iter()
                .collect(),
        }
    );

    let groups = store.group_summaries();
    assert_eq!(
        groups,
        vec![
            TrajectoryGroupSummary {
                key: "s-1".to_owned(),
                count: 2,
                success_rate: 0.5,
                mean_score: 0.5,
            },
            TrajectoryGroupSummary {
                key: "s-2".to_owned(),
                count: 1,
                success_rate: 0.0,
                mean_score: 0.0,
            },
        ]
    );

    // Replacement keeps the insertion position and the entry count.
    let replacement = Trajectory::new("t-2")
        .expect("valid id")
        .with_session("s-3")
        .with_outcome(TrajectoryOutcome::Success);
    assert!(store.add(replacement));
    assert_eq!(store.len(), 3);
    assert_eq!(
        store.get("t-2").and_then(|entry| entry.session_id.clone()),
        Some("s-3".to_owned())
    );
    assert!(!store.add(Trajectory::new("t-4").expect("valid id")));

    let removed = store.remove("t-4").expect("present");
    assert_eq!(removed.trajectory_id, "t-4");
    assert!(store.remove("t-4").is_none());
}

#[test]
fn empty_store_has_zero_statistics() {
    let store = TrajectoryStore::new();
    assert!(store.is_empty());
    assert_eq!(store.iter().count(), 0);
    assert_eq!(store.summary(), TrajectoryStats::default());
    assert_eq!(store.summary().success_rate, 0.0);
    assert_eq!(store.summary().mean_score, 0.0);
    assert!(store.group_summaries().is_empty());
    assert!(store.group_by_session().is_empty());
    assert!(store.group_by_case().is_empty());
}

#[test]
fn trajectory_serde_round_trips_with_unknown_fields() {
    let mut trajectory = Trajectory::new("traj-1")
        .expect("valid id")
        .with_session("session-1")
        .with_case("case-1")
        .with_outcome(TrajectoryOutcome::Success)
        .with_score(0.5)
        .expect("finite score")
        .with_elapsed_ms(1_234)
        .with_metadata("source", json!("offline"));
    trajectory.push(
        TrajectoryStep::new(StepKind::ToolCall, "{}")
            .with_tool_name("search")
            .with_tool_call_id("call-1")
            .with_metadata("tokens", json!(12)),
    );
    trajectory.push(TrajectoryStep::new(StepKind::ToolResult, "hit").with_tool_call_id("call-1"));

    let mut payload = serde_json::to_value(&trajectory).expect("serialize");
    payload["future_top_level"] = json!({"version": 2});
    payload["steps"][0]["future_step_field"] = json!("ignored");
    let restored: Trajectory = serde_json::from_value(payload).expect("deserialize");
    assert_eq!(restored, trajectory);
    assert_eq!(restored.len(), 2);
    assert_eq!(restored.elapsed_ms, Some(1_234));
}

#[test]
fn trajectory_deserializes_with_missing_fields() {
    let restored: Trajectory =
        serde_json::from_value(json!({"trajectory_id": "solo"})).expect("deserialize");
    assert_eq!(restored.trajectory_id, "solo");
    assert!(restored.steps.is_empty());
    assert_eq!(restored.outcome, TrajectoryOutcome::Unknown);
    assert_eq!(restored.score, None);

    let empty: Trajectory = serde_json::from_value(json!({})).expect("deserialize");
    assert_eq!(empty.trajectory_id, "");
    assert!(empty.is_empty());
}

#[test]
fn store_serde_round_trips_with_unknown_fields() {
    let store = store_fixture();
    let mut payload = serde_json::to_value(&store).expect("serialize");
    assert_eq!(payload["trajectories"][0]["trajectory_id"], json!("t-1"));
    payload["future_field"] = json!(true);
    payload["trajectories"][1]["future_field"] = json!("ignored");
    let restored: TrajectoryStore = serde_json::from_value(payload).expect("deserialize");
    assert_eq!(restored, store);

    let empty: TrajectoryStore = serde_json::from_value(json!({})).expect("deserialize");
    assert!(empty.is_empty());
    let unknown: Value = json!({"trajectories": [], "extra": 1});
    let empty: TrajectoryStore = serde_json::from_value(unknown).expect("deserialize");
    assert!(empty.is_empty());
}

#[test]
fn empty_trajectory_helpers_are_safe() {
    let empty = Trajectory::new("empty").expect("valid id");
    assert!(spans(&empty).is_empty());
    assert!(longest_span(&empty).is_none());
    assert!(span_by_name(&empty, "any").is_none());
    assert!(windows(&empty, 3, 1).expect("valid").is_empty());
    assert!(window_around(&empty, 0, 2).is_err());
    assert!(first_error(&empty).is_none());
    assert!(last_n(&empty, 5).is_empty());
    assert_eq!(render_digest(&empty, 100), "");
    assert!(TrajectoryBuilder::new("empty")
        .expect("valid id")
        .is_empty());
}
