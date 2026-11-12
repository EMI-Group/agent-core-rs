//! Pure translations between harness-protocol vocabulary and the team vocabulary.
//!
//! Python parity source: `openjiuwen/agent_teams/harness/` — the `NativeHarness` supervisor
//! publishes `harness.state` / `harness.round` callbacks and forwards `OutputSchema` chunks; this
//! module holds the equivalent conversions as pure functions so the adapter and its tests never
//! restate the mapping.
//!
//! Nothing here performs I/O or reads a clock: every function takes a protocol value and returns
//! the team value, or `None` when the input has no team-level meaning.
use crate::schema::{ExecutionStatus, MemberStatus, RoundOutcome};
use openjiuwen_harness_protocol::{
    HarnessEvent, HarnessEventPayload, HarnessState, ItemEventKind, OutputChannel, TurnEventKind,
};
use serde_json::{json, Value};
/// Chunk type the team stream uses for answer-channel output.
pub const CHUNK_LLM_OUTPUT: &str = "llm_output";
/// Chunk type the team stream uses for reasoning-channel output.
pub const CHUNK_LLM_REASONING: &str = "llm_reasoning";
/// Chunk type the team stream uses for a started tool call.
pub const CHUNK_TOOL_CALL: &str = "tool_call";
/// Chunk type the team stream uses for a finished tool call.
pub const CHUNK_TOOL_RESULT: &str = "tool_result";
/// Maps a harness lifecycle state onto the persisted member status.
///
/// `Idle` means "up and accepting input", `Running` "busy with a turn", both pause phases
/// "paused", and `Terminated` "stopped" (the runtime was torn down but the row survives, so
/// [`MemberStatus::Stopped`] is used instead of the terminal [`MemberStatus::ShutDown`]).
/// Every state maps, so the result is `Option` only to mirror a Python mapping table that could
/// gain an unmapped state.
#[must_use]
pub fn state_to_member_status(state: HarnessState) -> Option<MemberStatus> {
    match state {
        HarnessState::Idle => Some(MemberStatus::Ready),
        HarnessState::Running => Some(MemberStatus::Busy),
        HarnessState::Pausing | HarnessState::Paused => Some(MemberStatus::Paused),
        HarnessState::Terminated => Some(MemberStatus::Stopped),
    }
}
/// Maps a harness lifecycle state onto the execution status of the current round.
///
/// `Idle` and `Running` are direct. `Pausing` maps to [`ExecutionStatus::CancelRequested`]
/// because the round is on its way out. `Paused` maps to [`ExecutionStatus::Idle`] rather than
/// `Cancelled`: the round is suspended, not terminated, and a resumed round must not have to
/// leave a terminal status first. `Terminated` is `None` — the runtime is gone, so there is no
/// execution to describe (the member status carries that fact).
#[must_use]
pub fn state_to_execution_status(state: HarnessState) -> Option<ExecutionStatus> {
    match state {
        HarnessState::Idle | HarnessState::Paused => Some(ExecutionStatus::Idle),
        HarnessState::Running => Some(ExecutionStatus::Running),
        HarnessState::Pausing => Some(ExecutionStatus::CancelRequested),
        HarnessState::Terminated => None,
    }
}
/// Extracts the round outcome from a turn-lifecycle event.
///
/// `started`/`finished`/`aborted`/`failed` map directly; `paused` and `resumed` do not close a
/// round and yield `None`. Non-turn events yield `None`.
#[must_use]
pub fn round_from_event(event: &HarnessEvent) -> Option<RoundOutcome> {
    let HarnessEventPayload::TurnLifecycle(turn) = &event.event else {
        return None;
    };
    match turn.kind {
        TurnEventKind::Started => Some(RoundOutcome::Started),
        TurnEventKind::Finished => Some(RoundOutcome::Finished),
        TurnEventKind::Aborted => Some(RoundOutcome::Aborted),
        TurnEventKind::Failed => Some(RoundOutcome::Failed),
        TurnEventKind::Paused | TurnEventKind::Resumed => None,
    }
}
/// Extracts the state a `state_changed` observation entered.
///
/// Other payloads yield `None`. The adapter uses this to forward transitions to observers in
/// the exact order the harness published them.
#[must_use]
pub fn state_from_event(event: &HarnessEvent) -> Option<HarnessState> {
    match &event.event {
        HarnessEventPayload::StateChanged(change) => Some(change.new),
        _ => None,
    }
}
/// Projects an output-bearing observation into the team stream chunk shape.
///
/// The team stream consumes `{"type": ..., "payload": ...}` frames (Python `OutputSchema`).
/// Answer-channel output becomes [`CHUNK_LLM_OUTPUT`], reasoning-channel output
/// [`CHUNK_LLM_REASONING`], a started tool item [`CHUNK_TOOL_CALL`] and a completed tool item
/// [`CHUNK_TOOL_RESULT`]; the payload is the payload model's own wire JSON, with the envelope's
/// `item_id` added for tool items. Every other event yields `None`.
#[must_use]
pub fn chunk_from_event(event: &HarnessEvent) -> Option<Value> {
    match &event.event {
        HarnessEventPayload::Output(output) => {
            let chunk_type = match output.channel {
                OutputChannel::Reasoning => CHUNK_LLM_REASONING,
                OutputChannel::Answer | OutputChannel::System => CHUNK_LLM_OUTPUT,
            };
            Some(chunk(chunk_type, &serde_json::to_value(output).ok()?))
        }
        HarnessEventPayload::ItemLifecycle(item) if item.item_type == "tool" => {
            let chunk_type = match item.kind {
                ItemEventKind::Started => CHUNK_TOOL_CALL,
                ItemEventKind::Completed => CHUNK_TOOL_RESULT,
                ItemEventKind::Updated => return None,
            };
            let mut payload = serde_json::to_value(item).ok()?;
            if let (Some(object), Some(item_id)) = (payload.as_object_mut(), event.item_id.as_ref())
            {
                object.insert("item_id".to_owned(), json!(item_id));
            }
            Some(chunk(chunk_type, &payload))
        }
        _ => None,
    }
}
/// Returns whether the observation closes a turn.
///
/// Only turn-lifecycle events with a terminal kind ([`TurnEventKind::Finished`],
/// [`TurnEventKind::Aborted`], [`TurnEventKind::Failed`]) qualify; `paused` and `resumed` keep
/// the turn open, matching [`openjiuwen_harness_protocol::TERMINAL_TURN_EVENT_KINDS`].
#[must_use]
pub fn is_terminal_event(event: &HarnessEvent) -> bool {
    matches!(
        &event.event,
        HarnessEventPayload::TurnLifecycle(turn) if turn.kind.is_terminal()
    )
}
/// Builds one team stream chunk frame.
fn chunk(chunk_type: &str, payload: &Value) -> Value {
    json!({"type": chunk_type, "payload": payload})
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use openjiuwen_harness_protocol::{
        HarnessEvent, ItemLifecycleEvent, OutputEvent, OutputKind, OutputOperation,
        StateChangedEvent, TurnLifecycleEvent,
    };
    use serde_json::json;
    /// Builds a turn-lifecycle event carrying `kind`.
    fn turn_event(kind: TurnEventKind) -> HarnessEvent {
        HarnessEvent::new(
            1,
            1.0,
            TurnLifecycleEvent { kind, result: None },
            "session-1",
            "member-1",
        )
        .with_turn_id("turn-1")
    }
    /// Builds an output event on `channel`.
    fn output_event(channel: OutputChannel) -> HarnessEvent {
        let output = OutputEvent {
            output_id: "out-1".to_owned(),
            kind: OutputKind::Text,
            content: json!("hello"),
            operation: OutputOperation::Delta,
            channel,
            content_index: 0,
            content_type: None,
            data: std::collections::BTreeMap::new(),
        };
        HarnessEvent::new(2, 2.0, output, "session-1", "member-1").with_turn_id("turn-1")
    }
    #[test]
    fn member_status_mapping_covers_every_state() {
        assert_eq!(
            state_to_member_status(HarnessState::Idle),
            Some(MemberStatus::Ready)
        );
        assert_eq!(
            state_to_member_status(HarnessState::Running),
            Some(MemberStatus::Busy)
        );
        assert_eq!(
            state_to_member_status(HarnessState::Pausing),
            Some(MemberStatus::Paused)
        );
        assert_eq!(
            state_to_member_status(HarnessState::Paused),
            Some(MemberStatus::Paused)
        );
        assert_eq!(
            state_to_member_status(HarnessState::Terminated),
            Some(MemberStatus::Stopped)
        );
    }
    #[test]
    fn execution_status_mapping_is_documented_by_value() {
        assert_eq!(
            state_to_execution_status(HarnessState::Idle),
            Some(ExecutionStatus::Idle)
        );
        assert_eq!(
            state_to_execution_status(HarnessState::Running),
            Some(ExecutionStatus::Running)
        );
        assert_eq!(
            state_to_execution_status(HarnessState::Pausing),
            Some(ExecutionStatus::CancelRequested)
        );
        assert_eq!(
            state_to_execution_status(HarnessState::Paused),
            Some(ExecutionStatus::Idle)
        );
        assert_eq!(state_to_execution_status(HarnessState::Terminated), None);
    }
    #[test]
    fn round_mapping_handles_each_turn_kind() {
        assert_eq!(
            round_from_event(&turn_event(TurnEventKind::Started)),
            Some(RoundOutcome::Started)
        );
        assert_eq!(
            round_from_event(&turn_event(TurnEventKind::Finished)),
            Some(RoundOutcome::Finished)
        );
        assert_eq!(
            round_from_event(&turn_event(TurnEventKind::Aborted)),
            Some(RoundOutcome::Aborted)
        );
        assert_eq!(
            round_from_event(&turn_event(TurnEventKind::Failed)),
            Some(RoundOutcome::Failed)
        );
        assert_eq!(round_from_event(&turn_event(TurnEventKind::Paused)), None);
        assert_eq!(round_from_event(&turn_event(TurnEventKind::Resumed)), None);
        assert_eq!(round_from_event(&output_event(OutputChannel::Answer)), None);
    }
    #[test]
    fn terminal_events_are_only_finished_aborted_failed() {
        assert!(is_terminal_event(&turn_event(TurnEventKind::Finished)));
        assert!(is_terminal_event(&turn_event(TurnEventKind::Aborted)));
        assert!(is_terminal_event(&turn_event(TurnEventKind::Failed)));
        assert!(!is_terminal_event(&turn_event(TurnEventKind::Started)));
        assert!(!is_terminal_event(&turn_event(TurnEventKind::Paused)));
        assert!(!is_terminal_event(&output_event(OutputChannel::Answer)));
    }
    #[test]
    fn state_changed_projects_the_entered_state() {
        let event = HarnessEvent::new(
            3,
            3.0,
            StateChangedEvent {
                old: HarnessState::Idle,
                new: HarnessState::Running,
            },
            "session-1",
            "member-1",
        );
        assert_eq!(state_from_event(&event), Some(HarnessState::Running));
        assert_eq!(state_from_event(&turn_event(TurnEventKind::Started)), None);
    }
    #[test]
    fn output_chunks_carry_type_and_payload() {
        let answer = chunk_from_event(&output_event(OutputChannel::Answer)).unwrap();
        assert_eq!(answer["type"], json!(CHUNK_LLM_OUTPUT));
        assert_eq!(answer["payload"]["output_id"], json!("out-1"));
        assert_eq!(answer["payload"]["content"], json!("hello"));
        let reasoning = chunk_from_event(&output_event(OutputChannel::Reasoning)).unwrap();
        assert_eq!(reasoning["type"], json!(CHUNK_LLM_REASONING));
        assert_eq!(chunk_from_event(&turn_event(TurnEventKind::Started)), None);
    }
    #[test]
    fn tool_item_chunks_include_the_item_id() {
        let event = HarnessEvent::new(
            4,
            4.0,
            ItemLifecycleEvent {
                kind: ItemEventKind::Started,
                item_type: "tool".to_owned(),
                data: std::collections::BTreeMap::new(),
            },
            "session-1",
            "member-1",
        )
        .with_item_id("call-1");
        let chunk = chunk_from_event(&event).unwrap();
        assert_eq!(chunk["type"], json!(CHUNK_TOOL_CALL));
        assert_eq!(chunk["payload"]["item_id"], json!("call-1"));
        assert_eq!(chunk["payload"]["item_type"], json!("tool"));
    }
}
