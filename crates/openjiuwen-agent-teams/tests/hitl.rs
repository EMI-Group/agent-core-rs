//! HITL surface of `openjiuwen-agent-teams`, driven through the public API only.
//!
//! The suite pins the six pieces an embedding host builds against:
//!
//! 1. the `#` / `$` / `@` routing grammar ([`parse_interact_str`]) over a real roster, including
//!    the fold-back rules for unknown mentions and unknown `$` channels;
//! 2. roster resolution ([`resolve_targets`]) — multicast order, de-duplication, and the fact that
//!    broadcast tokens and role placeholders are *not* roster members;
//! 3. the `@mention` primitive and the reserved-name table ([`parse_mention`], `is_reserved_name`);
//! 4. the run/interact admission gate ([`InteractGate`]) with its wire tokens `not_active` and
//!    `gate_closed`;
//! 5. the tool-approval bridge ([`ToolApprovalRequest`] → `InteractionRequest` →
//!    [`parse_confirm`] → `tool_approval_result`) plus the human-agent inbox enablement check;
//! 6. the relayed tool-call contract of `TeamAgent::deliver` over a live team: the interact gate
//!    first (so the team must be started), then the enabled human-agent inbox (`enable_hitt`), then
//!    a roster member whose role is exactly `passive_human`, then a known team tool —
//!    only then is the call dispatched, and a refusal carries one stable token
//!    (`human_agent_not_enabled`, `unknown_human_agent:<name>`,
//!    `tool_passthrough_avatar_not_supported`, `unknown_tool:<tool>`, `invalid_arguments:<tool>`,
//!    `tool_failed:<tool>`); an [`InteractPayload::ExternalTeam`] payload is always rejected as
//!    `unsupported:external_team_transport`.
//!
//! Everything here is deterministic and offline: no network, no model, no wall-clock ordering.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

mod support;

use std::sync::Arc;

use openjiuwen_agent_teams::constants::{
    is_reserved_name, parse_mention, DEFAULT_LEADER_MEMBER_NAME, HUMAN_AGENT_MEMBER_NAME,
    RESERVED_MEMBER_NAMES, USER_PSEUDO_MEMBER_NAME,
};
use openjiuwen_agent_teams::interaction::{
    approval_result_event, parse_confirm, parse_interact_str, resolve_targets, ConfirmPayload,
    DeliverResult, ExternalTeamEvent, GodViewMessage, GroupChatMessage, HumanAgentInbox,
    HumanAgentMessage, HumanAgentToolCall, InteractGate, InteractPayload, OperatorMessage, Route,
    ToolApprovalRequest, HUMAN_AGENT_NOT_ENABLED,
};
use openjiuwen_agent_teams::runtime::TeamAgent;
use openjiuwen_agent_teams::schema::{
    EventPayload, MemberSpecBase, PredefinedMemberSpec, TaskGraphSpec, TeamAgentSpec,
    ToolApprovalResultEvent,
};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::InteractionKind;
use serde_json::{json, Value};

use support::{sample_spec, Fixture};

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/// The roster of behaviour 1: the leader plus two workers, no human agent.
fn roster() -> Vec<String> {
    names(&["team_leader", "dev-1", "dev-2"])
}

/// A roster that also registers a human-agent avatar (`human`) and one member (`ai`).
fn roster_with_human() -> Vec<String> {
    names(&["team_leader", "dev-1", "dev-2", "human", "ai"])
}

/// Owned copy of a name list, so assertions read like the grammar spec.
fn names(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

fn member(targets: &[&str], body: &str) -> Route {
    Route::Member {
        targets: names(targets),
        body: body.to_owned(),
    }
}

fn broadcast(body: &str) -> Route {
    Route::Broadcast {
        body: body.to_owned(),
    }
}

fn god_view(body: &str) -> Route {
    Route::GodView {
        body: body.to_owned(),
    }
}

fn human_agent(name: &str, body: &str) -> Route {
    Route::HumanAgent {
        name: name.to_owned(),
        body: body.to_owned(),
    }
}

fn god_view_payload(body: &str) -> InteractPayload {
    InteractPayload::GodView(GodViewMessage {
        body: body.to_owned(),
        sender: None,
    })
}

fn operator_payload(target: Option<&str>, body: &str) -> InteractPayload {
    InteractPayload::Operator(OperatorMessage {
        target: target.map(str::to_owned),
        body: body.to_owned(),
        sender: None,
    })
}

/// The interrupted tool call used by every approval test.
fn approval_request() -> ToolApprovalRequest {
    ToolApprovalRequest {
        team_name: "team".to_owned(),
        member_name: "dev-1".to_owned(),
        tool_call_id: "call_7".to_owned(),
        tool_name: "shell".to_owned(),
        tool_args: json!({"command": "ls"}),
    }
}

/// One instance of every [`InteractPayload`] variant.
fn all_payloads() -> Vec<InteractPayload> {
    vec![
        InteractPayload::GodView(GodViewMessage {
            body: "hello leader".to_owned(),
            sender: Some("ops".to_owned()),
        }),
        InteractPayload::Operator(OperatorMessage {
            target: Some("dev-1".to_owned()),
            body: "start".to_owned(),
            sender: None,
        }),
        InteractPayload::HumanAgent(HumanAgentMessage {
            human_name: "human".to_owned(),
            target: None,
            body: "on it".to_owned(),
            sender: Some("alice".to_owned()),
        }),
        InteractPayload::HumanAgentToolCall(HumanAgentToolCall {
            human_name: "passive-1".to_owned(),
            tool_name: "member_complete_task".to_owned(),
            tool_args: json!({"task_id": "t1"}),
        }),
        InteractPayload::GroupChat(GroupChatMessage {
            body: "standup in 5".to_owned(),
            sender: Some("pm".to_owned()),
            mentions: names(&["dev-1"]),
        }),
        InteractPayload::ExternalTeam(ExternalTeamEvent {
            event: json!({"event_type": "team_completed", "payload": {}}),
        }),
    ]
}

// ---------------------------------------------------------------------------
// 1. mention grammar — `parse_interact_str`
// ---------------------------------------------------------------------------

#[test]
fn single_mention_routes_to_the_named_member() {
    assert_eq!(
        parse_interact_str("@dev-1 please deploy", &roster()),
        member(&["dev-1"], "please deploy")
    );
    // Body keeps internal whitespace and the mention prefix is consumed exactly once.
    assert_eq!(
        parse_interact_str("@dev-1   please   deploy", &roster()),
        member(&["dev-1"], "please   deploy")
    );
    // A dangling `@dev-1` has no body, so `parse_mention` declines it and it stays prose.
    assert_eq!(parse_interact_str("@dev-1", &roster()), broadcast("@dev-1"));
}

#[test]
fn multiple_mentions_fan_out_one_shared_body() {
    assert_eq!(
        parse_interact_str("@dev-1 @dev-2 standup now", &roster()),
        member(&["dev-1", "dev-2"], "standup now")
    );
    // Mention order is preserved and a repeated recipient collapses to one target.
    assert_eq!(
        parse_interact_str("@dev-2 @dev-1 @dev-2 standup", &roster()),
        member(&["dev-2", "dev-1"], "standup")
    );
    // Known recipients win; the unknown name is dropped from `targets`.
    assert_eq!(
        parse_interact_str("@ghost @dev-1 hello", &roster()),
        member(&["dev-1"], "hello")
    );
}

#[test]
fn unknown_mention_folds_back_into_the_leader_query() {
    // Every mention is unknown, so nothing is addressed: the original text is preserved verbatim
    // and the runtime delivers it as a leader query (a broadcast).
    assert_eq!(
        parse_interact_str("@nonexistent body", &roster()),
        broadcast("@nonexistent body")
    );
    assert_eq!(
        parse_interact_str("@ghost @phantom body", &roster()),
        broadcast("@ghost @phantom body")
    );
}

#[test]
fn broadcast_tokens_supersede_any_co_listed_recipient() {
    for input in [
        "@all status",
        "@* status",
        "@all @dev-1 status",
        "@dev-1 @all status",
    ] {
        assert_eq!(
            parse_interact_str(input, &roster()),
            broadcast("status"),
            "input {input}"
        );
    }
}

#[test]
fn hash_prefix_is_god_view_and_hashtag_is_prose() {
    // `# ` (with the separator) is the only explicit god-view marker.
    assert_eq!(
        parse_interact_str("# status", &roster()),
        god_view("status")
    );
    assert_eq!(
        parse_interact_str("#   status", &roster()),
        god_view("status")
    );
    // `#text` lacks the separator the grammar requires, so it is prose → broadcast (leader query).
    assert_eq!(parse_interact_str("#text", &roster()), broadcast("#text"));
    assert_eq!(
        parse_interact_str("#hashtag one", &roster()),
        broadcast("#hashtag one")
    );
}

#[test]
fn bare_text_routes_to_broadcast() {
    assert_eq!(
        parse_interact_str("hello team", &roster()),
        broadcast("hello team")
    );
    // Neither `$variable` nor a mid-body `@dev-1` is a prefix, so the text stays whole.
    assert_eq!(
        parse_interact_str("$variable", &roster()),
        broadcast("$variable")
    );
    assert_eq!(
        parse_interact_str("please ask @dev-1", &roster()),
        broadcast("please ask @dev-1")
    );
}

#[test]
fn dollar_channel_drives_one_human_agent() {
    assert_eq!(
        parse_interact_str("$human on it", &roster_with_human()),
        human_agent("human", "on it")
    );
}

#[test]
fn dollar_channel_with_named_recipient_collapses_to_member_without_leader_query() {
    // `$human @ai body`: the human agent speaks *to* `ai`. The route is `Member`, never a
    // `GodView`, so the leader is not queried.
    assert_eq!(
        parse_interact_str("$human @ai body", &roster_with_human()),
        member(&["ai"], "body")
    );
    // Documented collapse: `$human@ai` (no space) is the same channel.
    assert_eq!(
        parse_interact_str("$human@ai body", &roster_with_human()),
        member(&["ai"], "body")
    );
}

#[test]
fn unknown_dollar_channel_folds_back_into_broadcast() {
    // `human` is not on the plain roster: nobody to drive, so the input is unrouted.
    assert_eq!(
        parse_interact_str("$human @ai body", &roster()),
        broadcast("$human @ai body")
    );
    assert_eq!(
        parse_interact_str("$ghost hello", &roster()),
        broadcast("$ghost hello")
    );
    // A known channel whose only recipient is unknown also folds back.
    assert_eq!(
        parse_interact_str("$human @ghost hello", &roster_with_human()),
        broadcast("$human @ghost hello")
    );
}

#[test]
fn empty_and_whitespace_inputs_are_deterministic() {
    assert_eq!(parse_interact_str("", &roster()), broadcast(""));
    assert_eq!(parse_interact_str("   ", &roster()), broadcast("   "));
    assert_eq!(parse_interact_str("\n\t ", &roster()), broadcast("\n\t "));
    // Prefix-only inputs keep their (empty) bodies.
    assert_eq!(parse_interact_str("# ", &roster()), god_view(""));
    assert_eq!(
        parse_interact_str("$human ", &roster_with_human()),
        human_agent("human", "")
    );
}

#[test]
fn routes_serialize_with_the_documented_kind_tags() {
    for (route, kind) in [
        (god_view("a"), "god_view"),
        (human_agent("human", "b"), "human_agent"),
        (member(&["dev-1"], "c"), "member"),
        (broadcast("d"), "broadcast"),
    ] {
        let encoded = serde_json::to_value(&route).unwrap();
        assert_eq!(
            encoded.get("kind").and_then(Value::as_str),
            Some(kind),
            "route {route:?}"
        );
        let decoded: Route = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, route);
    }
}

// ---------------------------------------------------------------------------
// 2. `resolve_targets` — pure roster membership
// ---------------------------------------------------------------------------

#[test]
fn resolve_targets_multicasts_roster_names_in_mention_order() {
    let (resolved, unknown) = resolve_targets(&names(&["dev-1", "dev-2"]), &roster());
    assert_eq!(resolved, names(&["dev-1", "dev-2"]));
    assert_eq!(unknown, names(&[]));

    // Order follows the mentions, not the roster.
    let (resolved, _) = resolve_targets(&names(&["dev-2", "team_leader"]), &roster());
    assert_eq!(resolved, names(&["dev-2", "team_leader"]));
}

#[test]
fn resolve_targets_deduplicates_repeated_targets() {
    let (resolved, unknown) = resolve_targets(&names(&["dev-2", "dev-1", "dev-2"]), &roster());
    assert_eq!(resolved, names(&["dev-2", "dev-1"]));
    assert_eq!(unknown, names(&[]));

    let (resolved, unknown) = resolve_targets(&names(&["ghost", "ghost"]), &roster());
    assert_eq!(resolved, names(&[]));
    assert_eq!(unknown, names(&["ghost"]));
}

#[test]
fn resolve_targets_reports_unknown_names_in_order() {
    let (resolved, unknown) =
        resolve_targets(&names(&["dev-1", "ghost", "dev-2", "phantom"]), &roster());
    assert_eq!(resolved, names(&["dev-1", "dev-2"]));
    assert_eq!(unknown, names(&["ghost", "phantom"]));

    let (resolved, unknown) = resolve_targets(&[], &roster());
    assert_eq!(resolved, names(&[]));
    assert_eq!(unknown, names(&[]));
}

#[test]
fn broadcast_tokens_and_role_placeholders_are_not_roster_members() {
    // `*` is a broadcast token handled before resolution: it is not a member name.
    let (resolved, unknown) = resolve_targets(&names(&["*"]), &roster());
    assert_eq!(resolved, names(&[]));
    assert_eq!(unknown, names(&["*"]));

    // `leader` and `user` are role placeholders resolved at delivery, not roster rows.
    let (resolved, unknown) = resolve_targets(&names(&["leader", "user"]), &roster());
    assert_eq!(resolved, names(&[]));
    assert_eq!(unknown, names(&["leader", "user"]));

    // Consequently `@*` is a broadcast and `@leader` / `@user` fold back into the leader query.
    assert_eq!(parse_interact_str("@* go", &roster()), broadcast("go"));
    assert_eq!(
        parse_interact_str("@leader status", &roster()),
        broadcast("@leader status")
    );
    assert_eq!(
        parse_interact_str("@user status", &roster()),
        broadcast("@user status")
    );
}

// ---------------------------------------------------------------------------
// 3. `parse_mention` and the reserved-name table
// ---------------------------------------------------------------------------

#[test]
fn parse_mention_splits_target_and_body() {
    assert_eq!(parse_mention("@dev-1 hello"), Some(("dev-1", "hello")));
    assert_eq!(
        parse_mention("@dev-1   hello  there "),
        Some(("dev-1", "hello  there "))
    );
    // The body keeps newlines verbatim (Python `[\s\S]+`).
    assert_eq!(
        parse_mention("@dev-1 line one\nline two"),
        Some(("dev-1", "line one\nline two"))
    );
    // Non-ASCII bodies are safe (char boundaries, not byte indices).
    assert_eq!(
        parse_mention("@dev-1 你好 世界"),
        Some(("dev-1", "你好 世界"))
    );
}

#[test]
fn parse_mention_rejects_non_mentions() {
    for input in [
        "no mention here",
        " @dev-1 hello",
        "@dev-1",
        "@dev-1   ",
        "@ dev-1 body",
        "@",
        "@\nbody",
    ] {
        assert_eq!(parse_mention(input), None, "input {input:?}");
    }
}

#[test]
fn reserved_names_match_python() {
    assert_eq!(
        RESERVED_MEMBER_NAMES,
        [
            USER_PSEUDO_MEMBER_NAME,
            DEFAULT_LEADER_MEMBER_NAME,
            HUMAN_AGENT_MEMBER_NAME
        ]
    );
    for name in ["user", "team_leader", "human_agent"] {
        assert!(is_reserved_name(name), "{name} must be reserved");
    }
    for name in ["dev-1", "leader", "Team_Leader", "", "user_2"] {
        assert!(!is_reserved_name(name), "{name} must not be reserved");
    }
}

// ---------------------------------------------------------------------------
// 4. `InteractGate` — admission control for one run cycle
// ---------------------------------------------------------------------------

#[test]
fn fresh_gate_is_open_and_queues_when_the_team_is_active() {
    let gate = InteractGate::new();
    assert!(!gate.is_closed(), "a fresh gate is open");
    assert_eq!(gate.len(), 0);
    assert!(gate.is_empty());

    let result = gate.dispatch(god_view_payload("hello"), true);
    assert_eq!(result, DeliverResult::ok(None));
    assert!(result.ok);
    assert_eq!(result.message_id, None, "the gate mints no message id");
    assert_eq!(result.reason, None);
    assert_eq!(gate.len(), 1);
    assert_eq!(gate.try_admit().unwrap(), Some(god_view_payload("hello")));
    assert_eq!(gate.try_admit().unwrap(), None);
}

#[test]
fn open_gate_without_a_run_cycle_reports_not_active() {
    let gate = InteractGate::new();
    let result = gate.dispatch(god_view_payload("hello"), false);
    assert!(!result.ok);
    assert_eq!(result.reason.as_deref(), Some("not_active"));
    assert_eq!(result.message_id, None);
    assert!(!gate.is_closed(), "rejecting leaves the gate open");
    assert_eq!(gate.len(), 0, "a rejected payload is never queued");
}

#[test]
fn closed_gate_reports_gate_closed_and_wins_over_not_active() {
    let gate = InteractGate::new();
    gate.close();
    assert!(gate.is_closed());

    for active in [true, false] {
        let result = gate.dispatch(god_view_payload("hello"), active);
        assert!(!result.ok);
        assert_eq!(
            result.reason.as_deref(),
            Some("gate_closed"),
            "active={active}"
        );
        assert_eq!(gate.len(), 0);
    }
}

#[test]
fn closed_gate_rejects_admit_and_try_admit_with_validation() {
    let gate = InteractGate::new();
    gate.admit(god_view_payload("queued")).unwrap();
    gate.close();

    let error = gate.admit(god_view_payload("later")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.to_string(), "validation: interact gate is closed");

    let error = gate.try_admit().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.to_string(), "validation: interact gate is closed");
}

#[test]
fn close_is_idempotent_and_drain_still_returns_queued_payloads() {
    let gate = InteractGate::new();
    assert!(gate.dispatch(god_view_payload("first"), true).ok);
    assert!(
        gate.dispatch(operator_payload(Some("dev-1"), "second"), true)
            .ok
    );
    gate.close();
    gate.close();
    assert!(gate.is_closed());

    assert_eq!(
        gate.drain(),
        vec![
            god_view_payload("first"),
            operator_payload(Some("dev-1"), "second"),
        ]
    );
    assert!(gate.is_empty());
    assert_eq!(gate.drain(), Vec::<InteractPayload>::new());

    // Closing does not reopen: a post-drain dispatch is still rejected.
    assert_eq!(
        gate.dispatch(god_view_payload("after"), true)
            .reason
            .as_deref(),
        Some("gate_closed")
    );
}

// ---------------------------------------------------------------------------
// 5. tool-approval bridge
// ---------------------------------------------------------------------------

#[test]
fn to_interaction_renders_a_team_tool_approval_request() {
    let request = approval_request();
    let interaction = request.to_interaction();

    assert_eq!(interaction.kind, InteractionKind::TeamToolApproval);
    assert_eq!(
        interaction.id,
        format!("intr_team_tool_approval_{}", request.tool_call_id),
        "the id is derived from the tool call, so re-rendering is idempotent"
    );
    assert_eq!(
        interaction.question,
        json!({
            "tool_name": "shell",
            "tool_args": {"command": "ls"},
            "member_name": "dev-1",
        })
    );
    assert_eq!(interaction.options.len(), 0);
    assert_eq!(interaction.component_id, None);
    assert!(interaction.metadata.is_empty());
    // The team name has no place in the harness question; it travels on the result event.
    assert!(!interaction.question.to_string().contains("team_name"));

    let again = request.to_interaction();
    assert_eq!(again.id, interaction.id);
    assert_eq!(again.question, interaction.question);
}

#[test]
fn parse_confirm_accepts_the_documented_object_answers() {
    assert_eq!(
        parse_confirm("intr_1", &json!({"approved": true})).unwrap(),
        ConfirmPayload {
            approved: true,
            feedback: String::new(),
            auto_confirm: false,
        }
    );
    assert_eq!(
        parse_confirm("intr_1", &json!({"approved": false})).unwrap(),
        ConfirmPayload {
            approved: false,
            feedback: String::new(),
            auto_confirm: false,
        }
    );
    let full = parse_confirm(
        "intr_1",
        &json!({"approved": false, "feedback": "use rsync", "auto_confirm": true}),
    )
    .unwrap();
    assert!(!full.approved);
    assert_eq!(full.feedback, "use rsync");
    assert!(full.auto_confirm);

    // Unknown fields are tolerated so a newer leader client can add knobs.
    assert!(
        parse_confirm("intr_1", &json!({"approved": true, "persist_allow": false}))
            .unwrap()
            .approved
    );
}

#[test]
fn parse_confirm_rejects_malformed_and_bare_string_answers() {
    // Deviation from Python: there is no `"yes"` / `"no"` string form — only the object shape is
    // accepted, and a bare string is a validation error naming the interaction.
    for answer in [
        json!("yes"),
        json!("no"),
        json!(true),
        json!(null),
        json!([]),
        json!({}),
        json!({"approved": "yes"}),
        json!({"approved": 1}),
        json!({"approved": true, "feedback": 5}),
        json!({"approved": true, "auto_confirm": "no"}),
    ] {
        let error = parse_confirm("intr_9", &answer).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "answer {answer}");
        let message = error.to_string();
        assert!(message.starts_with("validation:"), "message {message}");
        assert!(
            message.contains("intr_9"),
            "error names the interaction: {message}"
        );
    }
}

#[test]
fn approval_result_event_carries_the_verdict_and_wire_shape() {
    for approved in [true, false] {
        let payload = ConfirmPayload {
            approved,
            feedback: "ignored by the schema event".to_owned(),
            auto_confirm: true,
        };
        let event = approval_result_event(&approval_request(), &payload);
        assert_eq!(
            event,
            EventPayload::ToolApprovalResult(ToolApprovalResultEvent {
                team_name: "team".to_owned(),
                member_name: "dev-1".to_owned(),
                tool_call_id: "call_7".to_owned(),
                approved,
            })
        );
        assert_eq!(event.event_type(), "tool_approval_result");
        assert_eq!(
            event.payload_value().unwrap(),
            json!({
                "team_name": "team",
                "member_name": "dev-1",
                "tool_call_id": "call_7",
                "approved": approved,
            }),
            "approved={approved}"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. payload round-trip and `DeliverResult`
// ---------------------------------------------------------------------------

#[test]
fn every_payload_variant_round_trips_with_its_snake_case_kind() {
    for payload in all_payloads() {
        let encoded = serde_json::to_value(&payload).unwrap();
        assert_eq!(
            encoded.get("kind").and_then(Value::as_str),
            Some(payload.kind())
        );
        let decoded: InteractPayload = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, payload, "kind {}", payload.kind());
    }
}

#[test]
fn payloads_tolerate_unknown_fields() {
    let decoded: InteractPayload = serde_json::from_value(json!({
        "kind": "operator",
        "target": "dev-1",
        "body": "go",
        "future_field": [1, 2, 3],
    }))
    .unwrap();
    assert_eq!(decoded, operator_payload(Some("dev-1"), "go"));

    let decoded: InteractPayload = serde_json::from_value(json!({
        "kind": "group_chat",
        "body": "hi",
        "sender": null,
        "mentions": ["dev-1"],
        "reactions": {"dev-1": "👍"},
    }))
    .unwrap();
    assert_eq!(
        decoded,
        InteractPayload::GroupChat(GroupChatMessage {
            body: "hi".to_owned(),
            sender: None,
            mentions: names(&["dev-1"]),
        })
    );

    let decoded: InteractPayload = serde_json::from_value(json!({
        "kind": "external_team",
        "event": {"event_type": "team_completed"},
        "transport": "pyzmq",
    }))
    .unwrap();
    assert_eq!(
        decoded,
        InteractPayload::ExternalTeam(ExternalTeamEvent {
            event: json!({"event_type": "team_completed"}),
        })
    );
}

#[test]
fn only_the_human_agent_channels_require_hitt() {
    for payload in all_payloads() {
        let expected = matches!(
            payload,
            InteractPayload::HumanAgent(_) | InteractPayload::HumanAgentToolCall(_)
        );
        assert_eq!(payload.requires_hitt(), expected, "kind {}", payload.kind());
    }
}

#[test]
fn deliver_result_serializes_with_its_documented_fields() {
    let result = DeliverResult {
        ok: true,
        message_id: Some("msg_1".to_owned()),
        reason: None,
        output: json!("done"),
        data: json!({"status": "completed"}),
    };
    let encoded = serde_json::to_value(&result).unwrap();
    assert_eq!(
        encoded,
        json!({
            "ok": true,
            "message_id": "msg_1",
            "reason": null,
            "output": "done",
            "data": {"status": "completed"},
        })
    );
    let decoded: DeliverResult = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, result);

    // The failure tokens are pinned by the gate contract.
    assert_eq!(
        DeliverResult::not_active().reason.as_deref(),
        Some("not_active")
    );
    assert_eq!(
        DeliverResult::gate_closed().reason.as_deref(),
        Some("gate_closed")
    );
    assert_eq!(
        DeliverResult::unsupported("external_team")
            .reason
            .as_deref(),
        Some("unsupported:external_team")
    );
    assert!(!DeliverResult::failed("send_failed:dev-1").ok);

    // Unknown fields in an inbound result are tolerated too.
    let decoded: DeliverResult =
        serde_json::from_value(json!({"ok": false, "reason": "not_active", "trace": "x"})).unwrap();
    assert_eq!(decoded, DeliverResult::not_active());
}

// ---------------------------------------------------------------------------
// 7. human-agent inbox enablement (`enable_hitt` opt-in)
// ---------------------------------------------------------------------------

#[test]
fn human_agent_inbox_is_disabled_until_hitt_is_enabled() {
    let inbox = HumanAgentInbox::new();
    assert!(!inbox.is_enabled());

    let error = inbox.require_enabled().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        format!("not_found: {HUMAN_AGENT_NOT_ENABLED}")
    );

    inbox.set_enabled(true);
    assert!(inbox.is_enabled());
    assert!(inbox.require_enabled().is_ok());

    inbox.set_enabled(false);
    assert_eq!(
        inbox.require_enabled().unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

#[tokio::test]
async fn human_agent_inbox_queues_and_drains_in_arrival_order() {
    let inbox = HumanAgentInbox::new();
    inbox.set_enabled(true);

    let first = inbox.send("first", None, None);
    let second = inbox.send("second", Some("dev-1"), Some("human"));
    assert_eq!(first.human_name, HUMAN_AGENT_MEMBER_NAME);
    assert_eq!(second.human_name, "human");
    assert_eq!(second.target.as_deref(), Some("dev-1"));

    let first_id = inbox.push(first.clone()).await.unwrap();
    let second_id = inbox.push(second.clone()).await.unwrap();
    assert!(first_id.starts_with("msg_"));
    assert_ne!(first_id, second_id);
    assert_eq!(inbox.len(), 2);

    assert_eq!(
        inbox.drain().await,
        vec![(first_id, first), (second_id, second)]
    );
    assert!(inbox.is_empty());
    assert_eq!(inbox.drain().await, Vec::new());
}

// ---------------------------------------------------------------------------
// 8. relayed tool calls — `deliver` → guard chain → `tool_dispatch`
// ---------------------------------------------------------------------------

/// The sample blueprint with the HITL channel on and one passive human on the roster.
fn hitl_passive_human() -> TeamAgentSpec {
    let mut spec = sample_spec();
    spec.enable_hitt = true;
    spec.predefined_members
        .push(PredefinedMemberSpec::PassiveHuman(MemberSpecBase {
            member_name: "alice".to_owned(),
            display_name: "Alice".to_owned(),
            desc: "a human on the roster".to_owned(),
            ..MemberSpecBase::default()
        }));
    spec
}

/// Builds and starts one live team over the HITL blueprint; the fixture keeps its fakes alive.
async fn started_hitl_team() -> (Fixture, Arc<TeamAgent>) {
    let fixture = Fixture::new();
    let agent = fixture
        .activate(&hitl_passive_human(), "hitl-deliver")
        .await;
    agent.start().await.expect("the team did not start");
    (fixture, agent)
}

/// One relayed tool-call payload, acting as `human`.
fn tool_call(human: &str, tool: &str, arguments: Value) -> InteractPayload {
    InteractPayload::HumanAgentToolCall(HumanAgentToolCall {
        human_name: human.to_owned(),
        tool_name: tool.to_owned(),
        tool_args: arguments,
    })
}

/// Pins the refusal side of a relayed call: `ok == false`, no payload, and that exact token.
fn assert_refused(answer: &DeliverResult, reason: &str) {
    assert!(!answer.ok, "{answer:?}");
    assert_eq!(answer.reason.as_deref(), Some(reason), "{answer:?}");
    assert_eq!(answer.output, Value::Null, "a refusal carries no output");
    assert_eq!(answer.data, Value::Null, "a refusal carries no data");
    assert_eq!(
        answer.message_id, None,
        "a relayed call writes no mailbox row of its own"
    );
}

#[tokio::test]
async fn a_relayed_view_task_answers_its_listing_and_its_rendered_text() {
    let (_fixture, agent) = started_hitl_team().await;
    agent
        .create_task(TaskGraphSpec {
            title: "ship it".to_owned(),
            task_id: Some("t-1".to_owned()),
            ..TaskGraphSpec::default()
        })
        .await
        .expect("the task was not created");

    let answer = agent
        .deliver(tool_call("alice", "view_task", json!({"action": "list"})))
        .await;
    assert!(answer.ok, "{answer:?}");
    assert_eq!(answer.reason, None, "a successful relay carries no token");
    assert_eq!(answer.message_id, None);

    let rows = answer
        .data
        .as_array()
        .expect("view_task answers a list of task rows");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["task_id"], json!("t-1"));
    assert_eq!(rows[0]["title"], json!("ship it"));

    // The dispatch contract: a JSON string answer passes through verbatim, every other value is
    // rendered as its compact JSON text — so `output` here is exactly the text of `data`.
    let text = answer
        .output
        .as_str()
        .expect("the rendered output is text, never a raw value");
    assert_ne!(text, "", "the rendered output is never blank");
    assert_eq!(
        text,
        serde_json::to_string(&answer.data).expect("the listing is serializable")
    );
    assert!(text.contains("\"task_id\":\"t-1\""), "{text}");

    agent.stop().await.expect("the team did not stop");
}

#[tokio::test]
async fn a_relayed_unknown_tool_is_refused_by_its_stable_token() {
    let (_fixture, agent) = started_hitl_team().await;
    let answer = agent
        .deliver(tool_call("alice", "nope", json!({"action": "list"})))
        .await;
    assert_refused(&answer, "unknown_tool:nope");
    agent.stop().await.expect("the team did not stop");
}

#[tokio::test]
async fn a_relayed_call_from_off_the_roster_names_the_unknown_human() {
    let (_fixture, agent) = started_hitl_team().await;
    let answer = agent
        .deliver(tool_call("ghost", "view_task", json!({"action": "list"})))
        .await;
    assert_refused(&answer, "unknown_human_agent:ghost");
    agent.stop().await.expect("the team did not stop");
}

#[tokio::test]
async fn a_relayed_call_from_a_non_passive_roster_member_is_refused() {
    let (_fixture, agent) = started_hitl_team().await;
    // `dev-1` is a real teammate of `sample_spec`, so the roster knows it but its role is not
    // `passive_human`: an avatar keeps its own governed LLM tool loop, so passthrough is refused.
    let answer = agent
        .deliver(tool_call("dev-1", "view_task", json!({"action": "list"})))
        .await;
    assert_refused(&answer, "tool_passthrough_avatar_not_supported");
    agent.stop().await.expect("the team did not stop");
}

#[tokio::test]
async fn an_external_team_payload_is_rejected_before_the_gate() {
    let fixture = Fixture::new();
    let agent = fixture
        .activate(&hitl_passive_human(), "hitl-deliver")
        .await;
    // The transport rejection precedes the interact gate, so a team that never started answers the
    // same token instead of `not_active`.
    let answer = agent
        .deliver(InteractPayload::ExternalTeam(ExternalTeamEvent {
            event: json!({"event_type": "team_completed", "payload": {}}),
        }))
        .await;
    assert_refused(&answer, "unsupported:external_team_transport");
}
