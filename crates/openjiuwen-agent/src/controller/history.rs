//! Per-conversation conversation history kept by the controller.
//!
//! The controller records what the user asked and what its task executors
//! answered, so intent recognition can see the recent conversation and an
//! executor can be handed the same context. Python bounds that window with a
//! message-indexed slice (`chat_history[-2 * max_turns:]`), so the knob counts
//! **messages**, not turns; [`crate::controller::ControllerConfig::history_limit`]
//! carries the same meaning here.
//!
//! Two trims share that one setting:
//!
//! - [`trim_history`] runs when a turn appends its entries. It drops the oldest
//!   entries until the log fits, but never touches the entries being recorded by
//!   the turn in progress ([`trim_history`]'s `turn_start`), so a single turn
//!   that alone produces more messages than the limit keeps them all.
//! - [`window`] runs on every read, so lowering the limit through
//!   [`crate::controller::ControllerConfig`] takes effect on the very next
//!   read even though the stored log was trimmed against the older limit.
//!
//! `limit == 0` retains nothing: [`trim_history`] clears the log and [`window`]
//! returns an empty slice.
//!
//! Entries are plain data ([`HistoryEntry`]) with a `snake_case` wire form
//! ([`HistoryRole`] serializes to `"user"` / `"assistant"`), so a host can
//! persist and inspect the history without touching the controller.

use serde::{Deserialize, Serialize};

/// Who produced one [`HistoryEntry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryRole {
    /// A message the user (or the incoming event) contributed.
    User,
    /// A message a task executor produced.
    Assistant,
}

impl HistoryRole {
    /// The `snake_case` wire string of this role.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

impl Default for HistoryRole {
    /// A default entry is a user message, so an all-default entry is a blank
    /// prompt rather than a blank answer.
    fn default() -> Self {
        Self::User
    }
}

impl std::fmt::Display for HistoryRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One recorded conversation message.
///
/// `task_id` is set for entries a task executor produced, so a host can tell
/// which task answered; unknown JSON fields are tolerated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HistoryEntry {
    /// Who produced the message.
    pub role: HistoryRole,
    /// The message text.
    pub text: String,
    /// The task that produced the message, when one did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

impl HistoryEntry {
    /// Builds a user message.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: HistoryRole::User,
            text: text.into(),
            task_id: None,
        }
    }

    /// Builds an assistant message.
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: HistoryRole::Assistant,
            text: text.into(),
            task_id: None,
        }
    }

    /// Attributes this entry to `task_id`.
    #[must_use]
    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }
}

/// Appends-time trim: drops the oldest entries until `entries.len() <= limit`.
///
/// `turn_start` is the index where the turn currently being recorded begins; no
/// entry at or after it is ever removed, so a turn that alone produces more
/// entries than `limit` leaves the log longer than `limit` (the next append of a
/// later turn trims it). A `turn_start` beyond the end is clamped to
/// `entries.len()`.
///
/// `limit == 0` disables retention entirely: every entry is dropped, including
/// the current turn's.
pub fn trim_history(entries: &mut Vec<HistoryEntry>, limit: usize, turn_start: usize) {
    if limit == 0 {
        entries.clear();
        return;
    }
    let turn_start = turn_start.min(entries.len());
    let removable = turn_start.min(entries.len().saturating_sub(limit));
    if removable > 0 {
        entries.drain(..removable);
    }
}

/// Read-time trim: the newest `limit` entries.
///
/// `limit == 0` yields an empty slice and `limit >= entries.len()` yields all of
/// them, so a configuration change applies to the next read without waiting for
/// the next append.
#[must_use]
pub fn window(entries: &[HistoryEntry], limit: usize) -> &[HistoryEntry] {
    if limit == 0 {
        return &entries[..0];
    }
    if limit >= entries.len() {
        entries
    } else {
        &entries[entries.len() - limit..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entries(texts: &[&str]) -> Vec<HistoryEntry> {
        texts.iter().map(|text| HistoryEntry::user(*text)).collect()
    }

    fn texts(entries: &[HistoryEntry]) -> Vec<&str> {
        entries.iter().map(|entry| entry.text.as_str()).collect()
    }

    #[test]
    fn roles_round_trip_on_the_wire() {
        assert_eq!(
            serde_json::to_value(HistoryRole::User).unwrap(),
            json!("user")
        );
        assert_eq!(
            serde_json::to_value(HistoryRole::Assistant).unwrap(),
            json!("assistant")
        );
        let user: HistoryRole = serde_json::from_value(json!("user")).unwrap();
        let assistant: HistoryRole = serde_json::from_value(json!("assistant")).unwrap();
        assert_eq!(user, HistoryRole::User);
        assert_eq!(assistant, HistoryRole::Assistant);
        assert_eq!(HistoryRole::User.as_str(), "user");
        assert_eq!(HistoryRole::Assistant.as_str(), "assistant");
        assert_eq!(HistoryRole::User.to_string(), "user");
        assert_eq!(HistoryRole::Assistant.to_string(), "assistant");
        assert_eq!(HistoryRole::default(), HistoryRole::User);
    }

    #[test]
    fn entries_round_trip_with_and_without_a_task_id() {
        let user = HistoryEntry::user("hello");
        assert_eq!(user.role, HistoryRole::User);
        assert_eq!(user.task_id, None);
        assert_eq!(
            serde_json::to_value(&user).unwrap(),
            json!({"role": "user", "text": "hello"})
        );

        let assistant = HistoryEntry::assistant("hi").with_task_id("t1");
        assert_eq!(assistant.role, HistoryRole::Assistant);
        assert_eq!(assistant.task_id.as_deref(), Some("t1"));
        let wire = serde_json::to_value(&assistant).unwrap();
        assert_eq!(
            wire,
            json!({"role": "assistant", "text": "hi", "task_id": "t1"})
        );

        let parsed: HistoryEntry = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, assistant);
        let parsed: HistoryEntry = serde_json::from_value(json!({"text": "hello"})).unwrap();
        assert_eq!(parsed, user);
        assert_eq!(HistoryEntry::default().role, HistoryRole::User);
        assert_eq!(HistoryEntry::default().text, "");
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let parsed: HistoryEntry = serde_json::from_value(json!({
            "role": "assistant",
            "text": "done",
            "task_id": "t2",
            "extra": {"ignored": true}
        }))
        .unwrap();
        assert_eq!(parsed, HistoryEntry::assistant("done").with_task_id("t2"));
    }

    #[test]
    fn window_keeps_the_newest_entries() {
        let log = entries(&["a", "b", "c", "d"]);
        assert_eq!(window(&log, 0).len(), 0);
        assert_eq!(texts(window(&log, 2)), ["c", "d"]);
        assert_eq!(texts(window(&log, 4)), ["a", "b", "c", "d"]);
        assert_eq!(texts(window(&log, 99)), ["a", "b", "c", "d"]);
        assert_eq!(window(&[], 3).len(), 0);
    }

    #[test]
    fn trim_history_drops_only_the_oldest() {
        let mut log = entries(&["a", "b", "c", "d", "e"]);
        trim_history(&mut log, 3, 5);
        assert_eq!(texts(&log), ["c", "d", "e"]);
        trim_history(&mut log, 2, 3);
        assert_eq!(texts(&log), ["d", "e"]);
    }

    #[test]
    fn trim_history_protects_the_current_turn() {
        let mut log = entries(&["old-1", "old-2", "new-1", "new-2", "new-3"]);
        trim_history(&mut log, 2, 2);
        assert_eq!(texts(&log), ["new-1", "new-2", "new-3"]);
        assert!(log.len() > 2, "the current turn outgrew the limit");

        let mut single = entries(&["only"]);
        trim_history(&mut single, 1, 0);
        assert_eq!(texts(&single), ["only"]);
    }

    #[test]
    fn trim_history_at_the_exact_limit_is_a_no_op() {
        let mut log = entries(&["a", "b", "c"]);
        trim_history(&mut log, 3, 3);
        assert_eq!(texts(&log), ["a", "b", "c"]);
        trim_history(&mut log, 10, 3);
        assert_eq!(texts(&log), ["a", "b", "c"]);
    }

    #[test]
    fn trim_history_clamps_a_turn_start_past_the_end() {
        let mut log = entries(&["a", "b", "c"]);
        trim_history(&mut log, 1, 99);
        assert_eq!(texts(&log), ["c"]);
        trim_history(&mut entries(&[]), 5, 7);
    }

    #[test]
    fn zero_limit_retains_nothing() {
        let mut log = entries(&["a", "b", "c"]);
        trim_history(&mut log, 0, 3);
        assert_eq!(log.len(), 0);

        let mut current_turn = entries(&["new-1", "new-2"]);
        trim_history(&mut current_turn, 0, 0);
        assert!(current_turn.is_empty(), "the current turn is dropped too");
    }
}
