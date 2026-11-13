//! The `/compact` context-compaction slash command.
//!
//! Python's `/compact` routes the conversation through the kernel
//! `ContextEngine` and asks a model to summarise the transcript. That engine is
//! not reachable from the `DeepAgent` facade in this port (it is owned by the
//! inner agent, not exposed to the CLI), so `/compact` prunes the CLI's own
//! transcript with a bounded, deterministic truncation instead — documented
//! behind this module and reported to the user as a message/character count.
//!
//! [`ConversationHistory`] keeps the transcript the REPL mirrors from each turn.
//! [`ConversationHistory::compact`] retains the newest
//! [`Limits::keep_messages`] messages that fit inside [`Limits::max_chars`]
//! characters, and truncates a single oversized message from the front so the
//! newest text — usually the part the next turn refers to — survives.
//!
//! Everything counts characters, never bytes, so multi-byte text is never split
//! and nothing panics.

/// Number of newest messages [`ConversationHistory::compact`] keeps by default.
pub const DEFAULT_KEEP_MESSAGES: usize = 8;

/// Character budget [`ConversationHistory::compact`] targets by default.
pub const DEFAULT_MAX_CHARS: usize = 12_000;

/// Marker prepended to the kept tail of a truncated message.
pub const TRUNCATION_MARKER: &str = "\n[...truncated]";

/// Retention bounds applied by [`ConversationHistory::compact`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Number of newest messages to retain before the character budget applies.
    pub keep_messages: usize,
    /// Total character budget of the retained transcript.
    pub max_chars: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            keep_messages: DEFAULT_KEEP_MESSAGES,
            max_chars: DEFAULT_MAX_CHARS,
        }
    }
}

/// One message of the CLI-side transcript mirror.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HistoryMessage {
    /// Message role: `user` or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// What one [`ConversationHistory::compact`] call changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionReport {
    /// Message count before compaction.
    pub before: usize,
    /// Message count after compaction.
    pub after: usize,
    /// Number of messages removed.
    pub dropped: usize,
    /// Total content characters before compaction.
    pub chars_before: usize,
    /// Total content characters after compaction.
    pub chars_after: usize,
}

impl CompactionReport {
    /// Returns whether compaction changed the transcript at all.
    ///
    /// True when a message was dropped or when a single oversized message was
    /// truncated, so a no-op report is distinguishable from a real reduction.
    #[must_use]
    pub fn dropped_any(&self) -> bool {
        self.dropped > 0 || self.chars_after < self.chars_before
    }
}

/// The CLI-side transcript mirror plus the retention bounds of `/compact`.
#[derive(Debug, Clone, Default)]
pub struct ConversationHistory {
    /// Retained messages, oldest first.
    messages: Vec<HistoryMessage>,
    /// Retention bounds applied by [`ConversationHistory::compact`].
    limits: Limits,
}

impl ConversationHistory {
    /// Creates an empty transcript using the [`Limits::default`] bounds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty transcript using `limits`.
    #[must_use]
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            messages: Vec::new(),
            limits,
        }
    }

    /// Returns the retention bounds.
    #[must_use]
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Returns the retained messages, oldest first.
    #[must_use]
    pub fn messages(&self) -> &[HistoryMessage] {
        &self.messages
    }

    /// Appends one message.
    pub fn push(&mut self, role: &str, content: &str) {
        self.messages.push(HistoryMessage {
            role: role.to_owned(),
            content: content.to_owned(),
        });
    }

    /// Returns the number of retained messages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns whether the transcript is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Drops every retained message.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Drops the oldest messages down to the retained window.
    ///
    /// Compaction runs in three documented steps:
    ///
    /// 1. keep only the newest [`Limits::keep_messages`] messages;
    /// 2. while the total content characters still exceed [`Limits::max_chars`]
    ///    and more than one message remains, drop the oldest message;
    /// 3. if the single remaining message still exceeds the budget, keep its
    ///    newest characters plus [`TRUNCATION_MARKER`], reserving the marker's
    ///    width inside the budget so the reported character total stays within
    ///    [`Limits::max_chars`] whenever that budget is at least as wide as the
    ///    marker.
    ///
    /// Counting is by `chars().count()`, so multi-byte text is measured and cut
    /// safely. The returned [`CompactionReport`] records the message counts, the
    /// number of dropped messages and the character totals; a transcript that
    /// already fits yields a no-op report.
    pub fn compact(&mut self) -> CompactionReport {
        let before = self.messages.len();
        let chars_before = self.total_chars();

        if self.messages.len() > self.limits.keep_messages {
            let excess = self.messages.len() - self.limits.keep_messages;
            self.messages.drain(..excess);
        }
        while self.messages.len() > 1 && self.total_chars() > self.limits.max_chars {
            self.messages.remove(0);
        }
        let marker_chars = TRUNCATION_MARKER.chars().count();
        let budget = self.limits.max_chars.saturating_sub(marker_chars);
        if let Some(last) = self.messages.last_mut() {
            if last.content.chars().count() > self.limits.max_chars {
                last.content = truncate_message(&last.content, budget);
            }
        }

        let after = self.messages.len();
        CompactionReport {
            before,
            after,
            dropped: before - after,
            chars_before,
            chars_after: self.total_chars(),
        }
    }

    /// Returns the total number of content characters of the transcript.
    fn total_chars(&self) -> usize {
        self.messages
            .iter()
            .map(|message| message.content.chars().count())
            .sum()
    }
}

/// Character-safe truncation keeping the newest `max_chars` characters plus the
/// marker.
///
/// Content that already fits is returned unchanged. Otherwise the newest
/// `max_chars` characters are kept and [`TRUNCATION_MARKER`] is prepended, so the
/// result may be up to the marker's width longer than `max_chars`; callers with a
/// hard budget subtract that width first (as [`ConversationHistory::compact`]
/// does). Multi-byte text is cut on character boundaries, never inside one.
#[must_use]
pub fn truncate_message(content: &str, max_chars: usize) -> String {
    let total = content.chars().count();
    if total <= max_chars {
        return content.to_owned();
    }
    let kept: String = content.chars().skip(total - max_chars).collect();
    format!("{TRUNCATION_MARKER}{kept}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Returns limits with a one-message and generous character window.
    fn limits(keep_messages: usize, max_chars: usize) -> Limits {
        Limits {
            keep_messages,
            max_chars,
        }
    }

    #[test]
    fn defaults_match_the_documented_constants() {
        assert_eq!(
            Limits::default(),
            Limits {
                keep_messages: DEFAULT_KEEP_MESSAGES,
                max_chars: DEFAULT_MAX_CHARS,
            }
        );
        let history = ConversationHistory::new();
        assert_eq!(history.limits(), Limits::default());
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.messages(), &[] as &[HistoryMessage]);
    }

    #[test]
    fn compact_drops_older_messages_down_to_the_window() {
        let mut history = ConversationHistory::with_limits(limits(3, 10_000));
        for n in 0..6 {
            history.push("user", &format!("message {n}"));
        }
        let report = history.compact();

        assert_eq!(report.before, 6);
        assert_eq!(report.after, 3);
        assert_eq!(report.dropped, 3);
        assert_eq!(report.chars_before, 54);
        assert_eq!(report.chars_after, 27);
        assert!(report.dropped_any());
        assert_eq!(
            history
                .messages()
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["message 3", "message 4", "message 5"]
        );
    }

    #[test]
    fn compact_trims_by_character_budget_keeping_the_newest() {
        let mut history = ConversationHistory::with_limits(limits(10, 10));
        history.push("user", "aaaa"); // 4
        history.push("assistant", "bbbb"); // 4, total 8
        history.push("user", "cccc"); // 4, total 12 -> over budget
        history.push("assistant", "dd"); // 2, total 14

        let report = history.compact();
        // Dropping "aaaa" leaves 10 characters, which fits exactly.
        assert_eq!(report.before, 4);
        assert_eq!(report.after, 3);
        assert_eq!(report.dropped, 1);
        assert_eq!(report.chars_before, 14);
        assert_eq!(report.chars_after, 10);
        assert_eq!(
            history
                .messages()
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["bbbb", "cccc", "dd"]
        );
    }

    #[test]
    fn compact_truncates_a_single_oversized_message_with_multi_byte_text() {
        let mut history = ConversationHistory::with_limits(limits(4, 40));
        history.push("assistant", &"é".repeat(200));

        let report = history.compact();
        assert_eq!(report.after, 1);
        assert_eq!(report.dropped, 0);
        assert!(report.dropped_any());
        assert!(report.chars_after <= 40);

        let content = &history.messages()[0].content;
        assert!(content.starts_with(TRUNCATION_MARKER));
        assert_eq!(content.chars().count(), 40);
        assert!(content.ends_with(&"é".repeat(40 - TRUNCATION_MARKER.chars().count())));
    }

    #[test]
    fn compact_of_a_short_history_is_a_no_op() {
        let mut history = ConversationHistory::with_limits(limits(8, 12_000));
        history.push("user", "hi");
        history.push("assistant", "hello");

        let report = history.compact();
        assert_eq!(report.before, 2);
        assert_eq!(report.after, 2);
        assert_eq!(report.dropped, 0);
        assert!(!report.dropped_any());
        assert_eq!(report.chars_before, report.chars_after);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn compact_preserves_order_and_clears_on_request() {
        let mut history = ConversationHistory::with_limits(limits(2, 12_000));
        for n in 0..4 {
            history.push("user", &format!("{n}"));
        }
        history.compact();
        assert_eq!(
            history
                .messages()
                .iter()
                .map(|m| m.role.as_str())
                .collect::<Vec<_>>(),
            vec!["user", "user"]
        );
        assert_eq!(
            history
                .messages()
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["2", "3"]
        );

        history.clear();
        assert!(history.is_empty());
    }

    #[test]
    fn truncate_message_keeps_short_and_cuts_long_content() {
        assert_eq!(truncate_message("short", 10), "short");
        assert_eq!(truncate_message("", 0), "");
        assert_eq!(
            truncate_message("0123456789", 4),
            format!("{TRUNCATION_MARKER}6789")
        );
        // Multi-byte content is never split.
        let cut = truncate_message("日本語テキスト", 3);
        assert_eq!(cut, format!("{TRUNCATION_MARKER}キスト"));
    }
}
