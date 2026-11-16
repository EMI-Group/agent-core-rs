//! Turn boundaries taken from the Claude CLI's delivery receipts.
//!
//! Ports Python `openjiuwen.harness_providers.claudecode.lifecycle` (v0.1.20). The CLI
//! acknowledges every user message it is handed with `command_lifecycle` frames — `queued`,
//! `started`, `completed` — keyed by the `uuid` the message carried. [`TurnCycleTracker`] records
//! the id of every message the turn wrote, follows those receipts, and decides when the turn is
//! over.
//!
//! A turn is over once a `result` arrived **and** no message it submitted is still outstanding.
//! That is exactly what separates a steered message the CLI folds into the running cycle (one
//! result for both messages) from one it answers as a new cycle (a result of its own): after the
//! first result the steered message is still outstanding, so the turn stays open and that cycle's
//! output is projected into it.
//!
//! A message the CLI never acknowledges at all stops blocking the turn after
//! `lifecycle_ack_timeout_s`; the abandonment is reported through [`drain_diagnostics`] (the driver
//! publishes it as a `warning` diagnostic). A message the CLI took (`queued`/`started`) is never
//! timed out: however long the answer runs, it is this turn's work.
//!
//! Python wraps the SDK transport in a `LifecycleTap` that appends a synthetic
//! [`SETTLED_SUBTYPE`] `system` frame once the turn settles. This crate drives the CLI directly, so
//! the turn loop asks the tracker for the verdict instead of parsing a synthetic frame; the two
//! frame-kind constants are kept because they are the wire vocabulary a host may see in provider
//! events.
//!
//! [`drain_diagnostics`]: TurnCycleTracker::drain_diagnostics

use std::collections::BTreeMap;
use std::time::Duration;

use openjiuwen_harness_protocol::JsonValue;
use tokio::time::Instant;

/// Subtype of the synthetic settled frame Python's tap appends (`SETTLED_SUBTYPE`).
pub const SETTLED_SUBTYPE: &str = "openjiuwen.turn.settled";

/// Wire type of the CLI's delivery-receipt frames (`LIFECYCLE_FRAME_TYPE`).
pub const LIFECYCLE_FRAME_TYPE: &str = "command_lifecycle";

/// How far the CLI has taken one submitted message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandState {
    /// Written to the CLI, not acknowledged yet.
    Submitted,
    /// Accepted into the CLI's queue.
    Queued,
    /// Being worked on; never timed out.
    Started,
}

/// Decides when a Claude turn is over, from the CLI's delivery receipts.
#[derive(Debug)]
pub struct TurnCycleTracker {
    /// How long, after a result, an unacknowledged message may block the turn.
    ack_timeout: Duration,
    /// Turn currently tracked; `None` outside a turn.
    turn_id: Option<String>,
    /// Submitted messages the CLI has not completed, with their last known state.
    outstanding: BTreeMap<String, CommandState>,
    /// Number of `result` frames seen in this turn.
    results: u64,
    /// Whether the current turn already ended.
    settled: bool,
    /// Session-wide: whether the CLI has ever reported a delivery receipt.
    lifecycle_seen: bool,
    /// When an unacknowledged message stops blocking the turn.
    ack_deadline: Option<Instant>,
    /// What went wrong while settling the turn.
    diagnostics: Vec<String>,
}

impl TurnCycleTracker {
    /// Binds how long a message may go unacknowledged before it is dropped.
    ///
    /// Python clamps a negative float to zero; a [`Duration`] cannot be negative, so the value is
    /// the whole contract.
    #[must_use]
    pub fn new(ack_timeout: Duration) -> Self {
        Self {
            ack_timeout,
            turn_id: None,
            outstanding: BTreeMap::new(),
            results: 0,
            settled: true,
            lifecycle_seen: false,
            ack_deadline: None,
            diagnostics: Vec::new(),
        }
    }

    /// Returns whether the CLI has ever reported a delivery receipt.
    ///
    /// A CLI build that reports no receipts leaves the first result as the only turn boundary
    /// there is; the flag is sticky for the whole cycle (like Python's session-wide tracker).
    #[must_use]
    pub fn lifecycle_seen(&self) -> bool {
        self.lifecycle_seen
    }

    /// Returns how many messages are still outstanding in the current turn.
    #[must_use]
    pub fn outstanding(&self) -> usize {
        self.outstanding.len()
    }

    /// Starts tracking the messages submitted for `turn_id`.
    pub fn begin_turn(&mut self, turn_id: &str) {
        self.turn_id = Some(turn_id.to_owned());
        self.outstanding.clear();
        self.results = 0;
        self.settled = false;
        self.ack_deadline = None;
        self.diagnostics.clear();
    }

    /// Stops tracking; frames arriving between turns are ignored.
    pub fn end_turn(&mut self) {
        self.turn_id = None;
        self.settled = true;
        self.outstanding.clear();
        self.ack_deadline = None;
    }

    /// Returns and clears what went wrong while settling the turn.
    pub fn drain_diagnostics(&mut self) -> Vec<String> {
        std::mem::take(&mut self.diagnostics)
    }

    /// Records the id of a user message written to the CLI.
    ///
    /// Sub-agent traffic (`parent_tool_use_id` set) is skipped: the CLI answers it inside a message
    /// of ours, so it never carries a receipt of its own.
    pub fn note_outbound(&mut self, frame: &serde_json::Map<String, JsonValue>) {
        if self.turn_id.is_none() || self.settled {
            return;
        }
        if frame.get("type").and_then(JsonValue::as_str) != Some("user") {
            return;
        }
        if frame.get("parent_tool_use_id").is_some_and(is_truthy_value) {
            return;
        }
        if let Some(JsonValue::String(message_id)) = frame.get("uuid") {
            if !message_id.is_empty() {
                self.outstanding
                    .insert(message_id.clone(), CommandState::Submitted);
            }
        }
    }

    /// Follows one inbound frame; returns whether the turn has settled.
    ///
    /// Frames other than a receipt or a `result` change nothing. A failed or interrupted cycle
    /// abandons its receipts at once, because the CLI stops working its queue there and they never
    /// arrive.
    pub fn note_inbound(&mut self, frame: &serde_json::Map<String, JsonValue>) -> bool {
        if self.turn_id.is_none() || self.settled {
            return false;
        }
        match frame.get("type").and_then(JsonValue::as_str) {
            Some(LIFECYCLE_FRAME_TYPE) => {
                self.lifecycle_seen = true;
                self.note_receipt(frame);
            }
            Some("result") => {
                self.results += 1;
                let subtype = frame
                    .get("subtype")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("success");
                let is_error = frame
                    .get("is_error")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(subtype != "success");
                if subtype != "success" || is_error {
                    self.abandon("ended the turn before answering");
                }
            }
            _ => return false,
        }
        self.settle_if_ready()
    }

    /// Settles a turn whose last messages never drew a receipt.
    pub fn abandon_unacknowledged(&mut self) -> bool {
        if self.turn_id.is_none() || self.settled {
            return false;
        }
        self.abandon("never acknowledged");
        self.settle_if_ready()
    }

    /// Returns how long to wait for a receipt that has not arrived yet.
    ///
    /// `None` means "no deadline": either no result has arrived yet, or every outstanding message
    /// was taken by the CLI (`queued`/`started`) and must not be timed out.
    #[must_use]
    pub fn straggler_timeout(&self) -> Option<Duration> {
        let deadline = self.ack_deadline?;
        Some(deadline.saturating_duration_since(Instant::now()))
    }

    /// Applies one receipt to the outstanding set.
    fn note_receipt(&mut self, frame: &serde_json::Map<String, JsonValue>) {
        let Some(JsonValue::String(message_id)) = frame.get("command_uuid") else {
            return;
        };
        if !self.outstanding.contains_key(message_id) {
            return;
        }
        match frame.get("state").and_then(JsonValue::as_str) {
            Some("completed") => {
                self.outstanding.remove(message_id);
            }
            Some("queued") => {
                self.outstanding
                    .insert(message_id.clone(), CommandState::Queued);
            }
            Some("started") => {
                self.outstanding
                    .insert(message_id.clone(), CommandState::Started);
            }
            _ => {}
        }
    }

    /// Drops the outstanding set, reporting why.
    fn abandon(&mut self, reason: &str) {
        if self.outstanding.is_empty() {
            return;
        }
        let ids: Vec<&str> = self.outstanding.keys().map(String::as_str).collect();
        self.diagnostics.push(format!(
            "{reason} {} message(s): {}",
            self.outstanding.len(),
            ids.join(", ")
        ));
        self.outstanding.clear();
        self.ack_deadline = None;
    }

    /// Settles the turn when a result arrived and nothing is outstanding any more.
    fn settle_if_ready(&mut self) -> bool {
        if self.results < 1 {
            return false;
        }
        if !self.lifecycle_seen {
            // A CLI build that reports no receipts leaves the first result as the only turn
            // boundary there is.
            self.outstanding.clear();
            return self.settle();
        }
        if !self.outstanding.is_empty() {
            self.arm_ack_deadline();
            return false;
        }
        self.ack_deadline = None;
        self.settle()
    }

    /// Arms the acknowledgement deadline when the CLI has taken none of the outstanding messages.
    fn arm_ack_deadline(&mut self) {
        let working = self
            .outstanding
            .values()
            .any(|state| *state != CommandState::Submitted);
        if working {
            // The CLI took the message; however long the answer runs, it is this turn's work and
            // must not be timed out.
            self.ack_deadline = None;
            return;
        }
        if self.ack_deadline.is_none() {
            self.ack_deadline = Some(Instant::now() + self.ack_timeout);
        }
    }

    /// Marks the turn settled.
    fn settle(&mut self) -> bool {
        self.settled = true;
        true
    }
}

/// Returns whether a decoded JSON value is truthy in Python's sense (non-null, non-empty).
fn is_truthy_value(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null => false,
        JsonValue::Bool(flag) => *flag,
        JsonValue::String(text) => !text.is_empty(),
        JsonValue::Array(items) => !items.is_empty(),
        JsonValue::Object(object) => !object.is_empty(),
        JsonValue::Number(_) => true,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of the receipt tracker over decoded frames.
    //!
    //! The tracker is pure bookkeeping over JSON frames plus one monotonic deadline, so these tests
    //! drive it directly: no child process, no transport, no wall-clock ordering beyond the one
    //! deliberately short acknowledgement timeout.
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::time::Duration;

    use openjiuwen_harness_protocol::JsonValue;
    use serde_json::json;

    use super::{TurnCycleTracker, LIFECYCLE_FRAME_TYPE};

    /// Acknowledgement timeout the timeout tests run with.
    const ACK: Duration = Duration::from_millis(40);

    /// Decodes one frame object.
    fn frame(value: JsonValue) -> serde_json::Map<String, JsonValue> {
        value
            .as_object()
            .cloned()
            .expect("the test frame is a JSON object")
    }

    /// Builds the user frame the driver writes for `message_id`.
    fn outbound(message_id: &str) -> serde_json::Map<String, JsonValue> {
        frame(json!({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            "parent_tool_use_id": JsonValue::Null,
            "uuid": message_id,
        }))
    }

    /// Builds one receipt frame for `message_id`.
    fn receipt(message_id: &str, state: &str) -> serde_json::Map<String, JsonValue> {
        frame(json!({
            "type": LIFECYCLE_FRAME_TYPE,
            "command_uuid": message_id,
            "state": state,
        }))
    }

    /// Builds a `result` frame.
    fn result(subtype: &str, is_error: bool) -> serde_json::Map<String, JsonValue> {
        frame(json!({"type": "result", "subtype": subtype, "is_error": is_error}))
    }

    /// A tracker whose turn submitted `message_ids`.
    fn tracking(ack: Duration, message_ids: &[&str]) -> TurnCycleTracker {
        let mut tracker = TurnCycleTracker::new(ack);
        tracker.begin_turn("turn-1");
        for message_id in message_ids {
            tracker.note_outbound(&outbound(message_id));
        }
        tracker
    }

    /// A turn with no receipt history settles on its first result, whatever it submitted.
    #[test]
    fn a_cli_without_receipts_settles_on_the_first_result() {
        let mut tracker = tracking(ACK, &["message-1"]);
        assert!(!tracker.lifecycle_seen());
        assert!(tracker.note_inbound(&result("success", false)));
        assert_eq!(tracker.outstanding(), 0);
        assert!(tracker.straggler_timeout().is_none());
    }

    /// A completed receipt and a result settle the turn in either order.
    #[test]
    fn an_answered_message_and_its_result_settle_the_turn() {
        let mut tracker = tracking(ACK, &["message-1"]);
        assert!(!tracker.note_inbound(&receipt("message-1", "started")));
        assert!(!tracker.note_inbound(&receipt("message-1", "completed")));
        assert!(tracker.note_inbound(&result("success", false)));
        assert!(tracker.lifecycle_seen());
    }

    /// A receipt that follows the result keeps the turn open; the next result settles it.
    #[test]
    fn a_receipt_after_the_result_keeps_the_turn_open() {
        let mut tracker = tracking(ACK, &["message-1", "message-2"]);
        assert!(!tracker.note_inbound(&receipt("message-1", "completed")));
        assert!(
            !tracker.note_inbound(&result("success", false)),
            "the steered message is still outstanding"
        );
        assert!(!tracker.note_inbound(&receipt("message-2", "started")));
        assert!(tracker.straggler_timeout().is_none(), "a taken message");
        assert!(!tracker.note_inbound(&result("success", false)));
        assert!(tracker.note_inbound(&receipt("message-2", "completed")));
        assert!(tracker.drain_diagnostics().is_empty());
    }

    /// A failed result abandons the outstanding receipts and settles at once.
    #[test]
    fn a_failed_result_settles_without_waiting_for_receipts() {
        let mut tracker = tracking(ACK, &["message-1"]);
        assert!(!tracker.note_inbound(&receipt("message-1", "started")));
        assert!(tracker.note_inbound(&result("error_during_execution", true)));
        let diagnostics = tracker.drain_diagnostics();
        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        assert!(
            diagnostics[0].contains("message-1") && diagnostics[0].contains("1 message(s)"),
            "{diagnostics:?}"
        );
        assert!(tracker.drain_diagnostics().is_empty(), "drained once");
    }

    /// An unacknowledged message stops blocking the turn once the deadline elapses.
    #[tokio::test]
    async fn an_unacknowledged_message_stops_blocking_the_turn() {
        let mut tracker = tracking(ACK, &["message-1"]);
        tracker.note_inbound(&receipt("a-different-command", "queued"));
        assert!(!tracker.note_inbound(&result("success", false)));
        let wait = tracker
            .straggler_timeout()
            .expect("an unacknowledged message arms the deadline");
        assert!(wait <= ACK, "{wait:?}");
        tokio::time::sleep(wait).await;
        assert!(tracker.abandon_unacknowledged());
        let diagnostics = tracker.drain_diagnostics();
        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        assert!(
            diagnostics[0].starts_with("never acknowledged"),
            "{diagnostics:?}"
        );
    }

    /// A message the CLI took is never timed out, however long its answer runs.
    #[tokio::test]
    async fn a_started_message_never_times_out() {
        let mut tracker = tracking(ACK, &["message-1"]);
        tracker.note_inbound(&receipt("message-1", "started"));
        assert!(!tracker.note_inbound(&result("success", false)));
        assert!(tracker.straggler_timeout().is_none());
        tokio::time::sleep(ACK * 3).await;
        assert!(!tracker.abandon_unacknowledged(), "still working");
        assert!(tracker.drain_diagnostics().is_empty());
    }

    /// Sub-agent messages are never tracked, and neither are frames outside a turn.
    #[test]
    fn sub_agent_traffic_and_out_of_turn_frames_are_ignored() {
        let mut tracker = TurnCycleTracker::new(ACK);
        tracker.note_outbound(&outbound("before-any-turn"));
        tracker.begin_turn("turn-1");
        let mut subagent = outbound("subagent-1");
        subagent.insert("parent_tool_use_id".to_owned(), json!("tool-1"));
        tracker.note_outbound(&subagent);
        tracker.note_outbound(&frame(json!({"type": "assistant"})));
        tracker.note_outbound(&frame(json!({"type": "user", "uuid": ""})));
        assert_eq!(tracker.outstanding(), 0);
        assert!(!tracker.note_inbound(&frame(json!({"type": "assistant"}))));
        tracker.end_turn();
        tracker.note_outbound(&outbound("after-the-turn"));
        assert_eq!(tracker.outstanding(), 0);
    }

    /// Ending a turn clears its bookkeeping but keeps the session's receipt history.
    #[test]
    fn lifecycle_seen_is_sticky_across_turns() {
        let mut tracker = tracking(ACK, &["message-1"]);
        tracker.note_inbound(&receipt("message-1", "completed"));
        assert!(tracker.note_inbound(&result("success", false)));
        tracker.end_turn();
        tracker.begin_turn("turn-2");
        assert!(tracker.lifecycle_seen());
        tracker.note_outbound(&outbound("message-2"));
        assert_eq!(tracker.outstanding(), 1);
        assert!(!tracker.note_inbound(&result("success", false)));
        assert_eq!(
            tracker.straggler_timeout().map(|wait| wait <= ACK),
            Some(true)
        );
    }

    /// The queue state is tracked from the receipts the CLI reports.
    #[test]
    fn receipts_advance_the_command_state() {
        let mut tracker = tracking(ACK, &["message-1"]);
        tracker.note_inbound(&receipt("message-1", "queued"));
        tracker.note_inbound(&receipt("message-1", "started"));
        tracker.note_inbound(&receipt("message-1", "unknown-state"));
        assert_eq!(tracker.outstanding(), 1);
        tracker.note_inbound(&receipt("message-1", "completed"));
        assert_eq!(tracker.outstanding(), 0);
    }
}
