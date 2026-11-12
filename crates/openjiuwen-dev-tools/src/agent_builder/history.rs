//! Bounded dialogue history for the interactive agent builder.
//!
//! Ports `openjiuwen/dev_tools/agent_builder/executor/history_manager.py`:
//! [`HistoryManager`] keeps the most recent [`DialogueMessage`]s, evicting the
//! oldest message whenever the configured capacity is exceeded.
//!
//! Timestamps from the Python `DialogueMessage` are not ported — only `role`
//! and `content` are observable — and `get_latest_k` treats `k == 0` as "no
//! messages" (see its documentation for the deviation).

use serde::{Deserialize, Serialize};

/// The history capacity used when [`HistoryManager::new`] is given `0`.
pub const DEFAULT_MAX_HISTORY_SIZE: usize = 20;

/// One turn of a builder dialogue, reduced to its wire shape.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DialogueMessage {
    /// The speaker: `"user"` or `"assistant"`.
    pub role: String,
    /// The message text.
    pub content: String,
}

/// A bounded, oldest-first dialogue history.
#[derive(Debug, Clone)]
pub struct HistoryManager {
    /// The retained messages, oldest first.
    messages: Vec<DialogueMessage>,
    /// The maximum number of retained messages; always at least one.
    max_history_size: usize,
}

impl HistoryManager {
    /// Creates a manager retaining at most `max_history_size` messages.
    ///
    /// A capacity of `0` is replaced by [`DEFAULT_MAX_HISTORY_SIZE`].
    #[must_use]
    pub fn new(max_history_size: usize) -> Self {
        let max_history_size = if max_history_size == 0 {
            DEFAULT_MAX_HISTORY_SIZE
        } else {
            max_history_size
        };
        Self {
            messages: Vec::new(),
            max_history_size,
        }
    }

    /// Appends a message, evicting the oldest ones until the history fits.
    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.messages.push(DialogueMessage {
            role: role.into(),
            content: content.into(),
        });
        while self.messages.len() > self.max_history_size {
            self.messages.remove(0);
        }
    }

    /// Appends a `"user"` message.
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.add_message("user", content);
    }

    /// Appends an `"assistant"` message.
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.add_message("assistant", content);
    }

    /// Returns the retained messages, oldest first.
    #[must_use]
    pub fn get_history(&self) -> &[DialogueMessage] {
        &self.messages
    }

    /// Returns the last `k` messages, oldest first.
    ///
    /// `k == 0` yields an empty slice and `k` larger than the history yields the
    /// whole history.
    ///
    /// Python's `get_messages(k)` treats `k <= 0` as "all messages"; this port
    /// deliberately maps `0` to "none" so the accessor has an unambiguous range
    /// and callers ask for everything with [`Self::get_history`].
    #[must_use]
    pub fn get_latest_k(&self, k: usize) -> &[DialogueMessage] {
        let start = self.messages.len().saturating_sub(k);
        &self.messages[start..]
    }

    /// The number of retained messages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether no message is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Drops every retained message, keeping the configured capacity.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// The configured maximum number of retained messages.
    #[must_use]
    pub fn max_history_size(&self) -> usize {
        self.max_history_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_to_max_and_drops_oldest() {
        let mut history = HistoryManager::new(3);
        for turn in 1..=5 {
            history.add_user_message(format!("m{turn}"));
        }
        assert_eq!(history.len(), 3);
        let contents: Vec<&str> = history
            .get_history()
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(contents, vec!["m3", "m4", "m5"]);
    }

    #[test]
    fn zero_capacity_uses_default() {
        let history = HistoryManager::new(0);
        assert_eq!(history.max_history_size(), DEFAULT_MAX_HISTORY_SIZE);
        assert_eq!(history.max_history_size(), 20);

        let custom = HistoryManager::new(4);
        assert_eq!(custom.max_history_size(), 4);
    }

    #[test]
    fn get_latest_k_below_len() {
        let mut history = HistoryManager::new(10);
        history.add_user_message("a");
        history.add_assistant_message("b");
        history.add_user_message("c");

        let latest = history.get_latest_k(2);
        assert_eq!(latest.len(), 2);
        assert_eq!(latest[0].content, "b");
        assert_eq!(latest[1].content, "c");
    }

    #[test]
    fn get_latest_k_above_len_returns_all() {
        let mut history = HistoryManager::new(10);
        history.add_user_message("only");

        assert_eq!(history.get_latest_k(9), history.get_history());
        assert_eq!(history.get_latest_k(1), history.get_history());
    }

    #[test]
    fn get_latest_k_zero_is_empty() {
        let mut history = HistoryManager::new(10);
        history.add_user_message("a");
        assert_eq!(history.get_latest_k(0).len(), 0);
        assert_eq!(HistoryManager::new(10).get_latest_k(0).len(), 0);
    }

    #[test]
    fn empty_manager_has_no_history() {
        let history = HistoryManager::new(5);
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.get_history().len(), 0);
        assert_eq!(history.get_latest_k(3).len(), 0);
    }

    #[test]
    fn clear_empties_but_keeps_capacity() {
        let mut history = HistoryManager::new(5);
        history.add_user_message("a");
        history.add_assistant_message("b");
        history.clear();

        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.max_history_size(), 5);

        history.add_user_message("c");
        assert_eq!(history.len(), 1);
        assert_eq!(history.get_history()[0].content, "c");
    }

    #[test]
    fn role_helpers_set_expected_roles() {
        let mut history = HistoryManager::new(5);
        history.add_user_message("hello");
        history.add_assistant_message("hi");
        history.add_message("assistant", "note");

        let roles: Vec<&str> = history
            .get_history()
            .iter()
            .map(|message| message.role.as_str())
            .collect();
        assert_eq!(roles, vec!["user", "assistant", "assistant"]);
    }

    #[test]
    fn get_history_is_oldest_first() {
        let mut history = HistoryManager::new(5);
        history.add_user_message("first");
        history.add_assistant_message("second");
        history.add_user_message("third");

        let contents: Vec<&str> = history
            .get_history()
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(contents, vec!["first", "second", "third"]);
    }

    #[test]
    fn messages_serialize_with_role_and_content() {
        let message = DialogueMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        };
        let value = serde_json::to_value(&message).expect("serialize");
        assert_eq!(value, serde_json::json!({"role": "user", "content": "hi"}));
        let parsed: DialogueMessage = serde_json::from_value(value).expect("deserialize");
        assert_eq!(parsed, message);
    }
}
