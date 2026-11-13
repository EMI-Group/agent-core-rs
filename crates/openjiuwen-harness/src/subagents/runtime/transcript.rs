//! Bounded per-subagent event transcripts and their live feeds.
//!
//! Ports the durable transcript shape of Python
//! `openjiuwen/harness/subagent_runtime/transcript.py` (a bounded per-instance message log)
//! without a stream-chunk projector.
//! [`TranscriptBuffer`] is the retained, sequence-numbered history of one sub-agent;
//! [`TranscriptChannel`] is the bounded live feed a monitoring caller subscribes to.
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::broadcast;
/// Entries retained per sub-agent transcript when no capacity is given.
pub const DEFAULT_TRANSCRIPT_CAPACITY: usize = 256;
/// Live-feed buffer when no capacity is given; a lagging receiver drops the oldest entries.
pub const DEFAULT_TRANSCRIPT_CHANNEL_CAPACITY: usize = 256;
/// Largest capacity [`TranscriptBuffer::new`] accepts; larger requests are clamped.
const MAX_TRANSCRIPT_CAPACITY: usize = 4096;
/// Longest digest [`TranscriptEntry::summary`] keeps before appending `...`.
const SUMMARY_LIMIT: usize = 120;
/// What one transcript entry records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptKind {
    /// The sub-agent was spawned with a task description.
    Spawn,
    /// The parent sent the sub-agent an input query.
    Input,
    /// The sub-agent produced a turn's assistant text.
    AssistantTurn,
    /// The sub-agent invoked a tool.
    ToolCall,
    /// The sub-agent finished its work and reported an output.
    Completion,
    /// The sub-agent's turn failed.
    Error,
    /// The sub-agent's lifecycle status changed.
    #[default]
    Status,
    /// The sub-agent was closed.
    Closed,
    /// The entry was reported evicted from the ring.
    Evicted,
}
impl TranscriptKind {
    /// The Python wire string of this kind (`spawn`, `assistant_turn`, ...).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Spawn => "spawn",
            Self::Input => "input",
            Self::AssistantTurn => "assistant_turn",
            Self::ToolCall => "tool_call",
            Self::Completion => "completion",
            Self::Error => "error",
            Self::Status => "status",
            Self::Closed => "closed",
            Self::Evicted => "evicted",
        }
    }
}
/// One message of a sub-agent's transcript.
///
/// `seq` is assigned by [`TranscriptBuffer::push`]; the constructors leave it at `0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TranscriptEntry {
    /// Buffer-assigned sequence number (1-based, monotone).
    pub seq: u64,
    /// The sub-agent instance this entry belongs to.
    pub subagent_id: String,
    /// What this entry records.
    pub kind: TranscriptKind,
    /// Message role: `system`, `user` or `assistant`.
    pub role: String,
    /// Human-readable payload (task description, query, answer, status, ...).
    pub content: String,
    /// Tool name, for [`TranscriptKind::ToolCall`] entries.
    pub tool_name: Option<String>,
    /// Tool or turn outcome, when known.
    pub success: Option<bool>,
    /// Milliseconds since the Unix epoch, supplied by the caller's clock.
    pub at_ms: i64,
}
impl Default for TranscriptEntry {
    fn default() -> Self {
        Self {
            seq: 0,
            subagent_id: String::new(),
            kind: TranscriptKind::Status,
            role: "system".to_owned(),
            content: String::new(),
            tool_name: None,
            success: None,
            at_ms: 0,
        }
    }
}
impl TranscriptEntry {
    /// A spawn entry carrying the delegated task description.
    #[must_use]
    pub fn spawn(
        subagent_id: impl Into<String>,
        task_description: impl Into<String>,
        at_ms: i64,
    ) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::Spawn,
            role: "system".to_owned(),
            content: task_description.into(),
            at_ms,
            ..Self::default()
        }
    }
    /// An entry for a query the parent sent to the sub-agent.
    #[must_use]
    pub fn input(subagent_id: impl Into<String>, query: impl Into<String>, at_ms: i64) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::Input,
            role: "user".to_owned(),
            content: query.into(),
            at_ms,
            ..Self::default()
        }
    }
    /// An entry for the assistant text one turn produced.
    #[must_use]
    pub fn assistant_turn(
        subagent_id: impl Into<String>,
        text: impl Into<String>,
        at_ms: i64,
    ) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::AssistantTurn,
            role: "assistant".to_owned(),
            content: text.into(),
            at_ms,
            ..Self::default()
        }
    }
    /// An entry for one tool invocation and its outcome (`success` is `None` while pending).
    #[must_use]
    pub fn tool_call(
        subagent_id: impl Into<String>,
        tool_name: impl Into<String>,
        detail: impl Into<String>,
        success: Option<bool>,
        at_ms: i64,
    ) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::ToolCall,
            role: "assistant".to_owned(),
            content: detail.into(),
            tool_name: Some(tool_name.into()),
            success,
            at_ms,
            ..Self::default()
        }
    }
    /// An entry for the final output of the sub-agent's work.
    #[must_use]
    pub fn completion(
        subagent_id: impl Into<String>,
        output: impl Into<String>,
        at_ms: i64,
    ) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::Completion,
            role: "assistant".to_owned(),
            content: output.into(),
            at_ms,
            ..Self::default()
        }
    }
    /// An entry for a failed turn.
    #[must_use]
    pub fn error(subagent_id: impl Into<String>, message: impl Into<String>, at_ms: i64) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::Error,
            role: "assistant".to_owned(),
            content: message.into(),
            success: Some(false),
            at_ms,
            ..Self::default()
        }
    }
    /// An entry for a lifecycle status change.
    #[must_use]
    pub fn status(subagent_id: impl Into<String>, status: &str, at_ms: i64) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::Status,
            role: "system".to_owned(),
            content: status.to_owned(),
            at_ms,
            ..Self::default()
        }
    }
    /// An entry for the sub-agent being closed.
    #[must_use]
    pub fn closed(subagent_id: impl Into<String>, reason: impl Into<String>, at_ms: i64) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::Closed,
            role: "system".to_owned(),
            content: reason.into(),
            at_ms,
            ..Self::default()
        }
    }
    /// An entry for a transcript that was evicted from its ring.
    #[must_use]
    pub fn evicted(subagent_id: impl Into<String>, reason: impl Into<String>, at_ms: i64) -> Self {
        Self {
            subagent_id: subagent_id.into(),
            kind: TranscriptKind::Evicted,
            role: "system".to_owned(),
            content: reason.into(),
            at_ms,
            ..Self::default()
        }
    }
    /// One-line digest: `"<kind>: <content>"`, newlines collapsed, truncated to 120 chars
    /// with a trailing `...`.
    #[must_use]
    pub fn summary(&self) -> String {
        let collapsed = self
            .content
            .split(['\n', '\r'])
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let digest = format!("{}: {}", self.kind.as_str(), collapsed.trim());
        if digest.chars().count() <= SUMMARY_LIMIT {
            return digest;
        }
        let head = digest.chars().take(SUMMARY_LIMIT).collect::<String>();
        format!("{head}...")
    }
}
/// Bounded first-in-first-out ring buffer of one sub-agent's transcript entries.
///
/// Sequence numbers are assigned on [`TranscriptBuffer::push`] and keep counting across
/// evictions, so a caller holding a cursor can ask for [`TranscriptBuffer::after`] and tell
/// how much history the ring dropped.
#[derive(Debug, Clone)]
pub struct TranscriptBuffer {
    /// Maximum number of retained entries (`1..=4096`).
    capacity: usize,
    /// Retained entries, oldest first.
    entries: VecDeque<TranscriptEntry>,
    /// Sequence number the next pushed entry receives.
    next_seq: u64,
    /// Entries evicted by the ring cap.
    dropped: u64,
}
impl TranscriptBuffer {
    /// Creates a ring holding at most `capacity` entries.
    ///
    /// `0` selects [`DEFAULT_TRANSCRIPT_CAPACITY`]; larger requests are clamped to `4096`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = if capacity == 0 {
            DEFAULT_TRANSCRIPT_CAPACITY
        } else {
            capacity.min(MAX_TRANSCRIPT_CAPACITY)
        };
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
            next_seq: 1,
            dropped: 0,
        }
    }
    /// Appends `entry`, assigning it the next sequence number (1-based, monotone), evicting the
    /// oldest entry when the ring is full. Returns the stored entry (with its assigned `seq`).
    pub fn push(&mut self, mut entry: TranscriptEntry) -> TranscriptEntry {
        entry.seq = self.next_seq;
        self.next_seq += 1;
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
            self.dropped += 1;
        }
        self.entries.push_back(entry.clone());
        entry
    }
    /// Appends every entry of `entries` in iteration order.
    pub fn extend(&mut self, entries: impl IntoIterator<Item = TranscriptEntry>) {
        for entry in entries {
            self.push(entry);
        }
    }
    /// Maximum number of retained entries.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Number of currently retained entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether no entry is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Number of entries evicted by the ring cap.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
    /// Sequence number the next pushed entry receives.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
    /// All retained entries, oldest first.
    #[must_use]
    pub fn entries(&self) -> Vec<TranscriptEntry> {
        self.entries.iter().cloned().collect()
    }
    /// The last `limit` retained entries, oldest first; `0` yields an empty vector.
    #[must_use]
    pub fn recent(&self, limit: usize) -> Vec<TranscriptEntry> {
        if limit == 0 {
            return Vec::new();
        }
        let skip = self.entries.len().saturating_sub(limit);
        self.entries.iter().skip(skip).cloned().collect()
    }
    /// The retained entry with sequence number `seq`, if it is still in the ring.
    #[must_use]
    pub fn get(&self, seq: u64) -> Option<&TranscriptEntry> {
        self.entries.iter().rev().find(|entry| entry.seq == seq)
    }
    /// Retained entries with `seq > cursor`, oldest first, at most `limit` of them; `0` yields
    /// an empty vector.
    #[must_use]
    pub fn after(&self, cursor: u64, limit: usize) -> Vec<TranscriptEntry> {
        if limit == 0 {
            return Vec::new();
        }
        self.entries
            .iter()
            .filter(|entry| entry.seq > cursor)
            .take(limit)
            .cloned()
            .collect()
    }
    /// A digest of the retained history.
    #[must_use]
    pub fn summary(&self) -> TranscriptSummary {
        TranscriptSummary::from_buffer(self)
    }
    /// Drops every retained entry and returns how many were removed.
    ///
    /// This is not a ring eviction: `next_seq` keeps counting and `dropped()` is unchanged, so a
    /// caller holding a cursor still sees new entries through
    /// [`TranscriptBuffer::after`].
    pub fn clear(&mut self) -> usize {
        let removed = self.entries.len();
        self.entries.clear();
        removed
    }
}
/// Digest of a [`TranscriptBuffer`]: size, evictions, and the newest entry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TranscriptSummary {
    /// Number of retained entries.
    pub count: usize,
    /// Number of entries evicted by the ring cap.
    pub dropped: u64,
    /// Kind of the newest retained entry.
    pub last_kind: Option<TranscriptKind>,
    /// Content of the newest retained entry.
    pub last_content: Option<String>,
}
impl TranscriptSummary {
    /// Digests `buffer`.
    #[must_use]
    pub fn from_buffer(buffer: &TranscriptBuffer) -> Self {
        let last = buffer.entries.back();
        Self {
            count: buffer.len(),
            dropped: buffer.dropped(),
            last_kind: last.map(|entry| entry.kind),
            last_content: last.map(|entry| entry.content.clone()),
        }
    }
    /// `"3 entries, last: completion: done"` (Python-ish human digest; singular "1 entry").
    ///
    /// A summary without a newest entry renders only the count.
    #[must_use]
    pub fn display(&self) -> String {
        let noun = if self.count == 1 { "entry" } else { "entries" };
        match (&self.last_kind, &self.last_content) {
            (Some(kind), Some(content)) if !content.is_empty() => {
                format!("{} {noun}, last: {}: {content}", self.count, kind.as_str())
            }
            (Some(kind), _) => format!("{} {noun}, last: {}", self.count, kind.as_str()),
            _ => format!("{} {noun}", self.count),
        }
    }
}
/// Live broadcast feed of one sub-agent's transcript entries; bounded, lagging receivers drop
/// entries.
///
/// A channel with no subscriber is valid: publishing then is a no-op returning no error.
#[derive(Debug)]
pub struct TranscriptChannel {
    /// Bounded fan-out of published entries.
    sender: broadcast::Sender<TranscriptEntry>,
}
impl TranscriptChannel {
    /// Creates a feed buffering at most `capacity` entries per receiver; `0` selects
    /// [`DEFAULT_TRANSCRIPT_CHANNEL_CAPACITY`].
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = if capacity == 0 {
            DEFAULT_TRANSCRIPT_CHANNEL_CAPACITY
        } else {
            capacity
        };
        let (sender, _receiver) = broadcast::channel(capacity);
        Self { sender }
    }
    /// Publishes `entry`; a channel without subscribers is not an error.
    pub fn publish(&self, entry: &TranscriptEntry) {
        let _ = self.sender.send(entry.clone());
    }
    /// Subscribes to entries published from now on.
    #[must_use]
    pub fn subscribe(&self) -> TranscriptSubscription {
        self.sender.subscribe()
    }
    /// Number of live subscriptions.
    #[must_use]
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}
/// One live subscription to a [`TranscriptChannel`].
///
/// Lagging receivers get `RecvError::Lagged` rather than blocking the publisher.
pub type TranscriptSubscription = broadcast::Receiver<TranscriptEntry>;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        TranscriptBuffer, TranscriptChannel, TranscriptEntry, TranscriptKind, TranscriptSummary,
        DEFAULT_TRANSCRIPT_CAPACITY,
    };
    fn ids(entries: &[TranscriptEntry]) -> Vec<u64> {
        entries.iter().map(|entry| entry.seq).collect()
    }
    #[test]
    fn push_assigns_monotone_sequence_numbers() {
        let mut buffer = TranscriptBuffer::new(4);
        assert_eq!(buffer.next_seq(), 1);
        let first = buffer.push(TranscriptEntry::spawn("sub-1", "explore", 10));
        let second = buffer.push(TranscriptEntry::input("sub-1", "go on", 20));
        assert_eq!((first.seq, second.seq), (1, 2));
        assert_eq!(buffer.next_seq(), 3);
        assert_eq!(ids(&buffer.entries()), vec![1, 2]);
        assert_eq!(buffer.dropped(), 0);
    }
    #[test]
    fn ring_evicts_oldest_and_counts_drops() {
        let mut buffer = TranscriptBuffer::new(2);
        buffer.push(TranscriptEntry::spawn("sub-1", "a", 1));
        buffer.push(TranscriptEntry::input("sub-1", "b", 2));
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.dropped(), 0);
        buffer.push(TranscriptEntry::assistant_turn("sub-1", "c", 3));
        assert_eq!(buffer.len(), buffer.capacity());
        assert_eq!(buffer.dropped(), 1);
        assert_eq!(ids(&buffer.entries()), vec![2, 3]);
        assert!(buffer.get(1).is_none());
        assert_eq!(buffer.get(2).expect("seq 2 retained").content, "b");
    }
    #[test]
    fn capacity_zero_falls_back_and_large_caps_clamp() {
        let buffer = TranscriptBuffer::new(0);
        assert_eq!(buffer.capacity(), DEFAULT_TRANSCRIPT_CAPACITY);
        assert!(buffer.is_empty());
        let buffer = TranscriptBuffer::new(usize::MAX);
        assert_eq!(buffer.capacity(), 4096);
        assert_eq!(TranscriptBuffer::new(1).capacity(), 1);
    }
    #[test]
    fn recent_returns_the_tail_oldest_first() {
        let mut buffer = TranscriptBuffer::new(8);
        for index in 0..5 {
            buffer.push(TranscriptEntry::status("sub-1", "running", index));
        }
        assert_eq!(ids(&buffer.recent(2)), vec![4, 5]);
        assert_eq!(ids(&buffer.recent(99)), vec![1, 2, 3, 4, 5]);
        assert_eq!(ids(&buffer.recent(0)), Vec::<u64>::new());
    }
    #[test]
    fn after_returns_entries_past_a_cursor() {
        let mut buffer = TranscriptBuffer::new(8);
        for index in 0..5 {
            buffer.push(TranscriptEntry::status("sub-1", "running", index));
        }
        assert_eq!(ids(&buffer.after(0, 99)), vec![1, 2, 3, 4, 5]);
        assert_eq!(ids(&buffer.after(2, 99)), vec![3, 4, 5]);
        assert_eq!(ids(&buffer.after(2, 1)), vec![3]);
        assert_eq!(ids(&buffer.after(5, 99)), Vec::<u64>::new());
        assert_eq!(ids(&buffer.after(0, 0)), Vec::<u64>::new());
    }
    #[test]
    fn extend_pushes_every_entry_in_order() {
        let mut buffer = TranscriptBuffer::new(8);
        buffer.extend([
            TranscriptEntry::spawn("sub-1", "task", 1),
            TranscriptEntry::completion("sub-1", "done", 2),
        ]);
        buffer.extend(Vec::new());
        assert_eq!(ids(&buffer.entries()), vec![1, 2]);
        assert_eq!(buffer.len(), 2);
    }
    #[test]
    fn clear_drops_the_entries_and_keeps_the_sequence() {
        let mut buffer = TranscriptBuffer::new(4);
        buffer.push(TranscriptEntry::spawn("sub-1", "task", 1));
        buffer.push(TranscriptEntry::input("sub-1", "more", 2));
        assert_eq!(buffer.clear(), 2);
        assert!(buffer.is_empty());
        assert_eq!(buffer.clear(), 0);
        assert_eq!(buffer.next_seq(), 3);
        let stored = buffer.push(TranscriptEntry::completion("sub-1", "done", 3));
        assert_eq!(stored.seq, 3);
        assert_eq!(buffer.summary().count, 1);
        assert_eq!(buffer.dropped(), 0);
    }
    #[test]
    fn summary_reports_count_drops_and_the_newest_entry() {
        let mut buffer = TranscriptBuffer::new(2);
        assert_eq!(buffer.summary(), TranscriptSummary::default());
        assert_eq!(buffer.summary().display(), "0 entries");
        buffer.push(TranscriptEntry::spawn("sub-1", "task", 1));
        assert_eq!(buffer.summary().display(), "1 entry, last: spawn: task");
        buffer.push(TranscriptEntry::completion("sub-1", "done", 2));
        buffer.push(TranscriptEntry::error("sub-1", "boom", 3));
        let summary = buffer.summary();
        assert_eq!(summary.count, 2);
        assert_eq!(summary.dropped, 1);
        assert_eq!(summary.last_kind, Some(TranscriptKind::Error));
        assert_eq!(summary.last_content.as_deref(), Some("boom"));
        assert_eq!(summary.display(), "2 entries, last: error: boom");
        assert_eq!(
            TranscriptSummary::from_buffer(&buffer).display(),
            "2 entries, last: error: boom"
        );
        assert_eq!(TranscriptSummary::default().display(), "0 entries");
    }
    #[test]
    fn summary_display_omits_empty_content() {
        let buffer = TranscriptBuffer {
            capacity: 4,
            entries: std::collections::VecDeque::from([TranscriptEntry::status("sub-1", "", 1)]),
            next_seq: 2,
            dropped: 0,
        };
        assert_eq!(buffer.summary().display(), "1 entry, last: status");
    }
    #[test]
    fn entry_constructors_set_kind_role_and_fields() {
        let spawn = TranscriptEntry::spawn("sub-1", "task", 7);
        assert_eq!(spawn.kind, TranscriptKind::Spawn);
        assert_eq!(spawn.role, "system");
        assert_eq!(spawn.content, "task");
        assert_eq!(spawn.at_ms, 7);
        assert_eq!(spawn.seq, 0);
        let input = TranscriptEntry::input("sub-1", "query", 8);
        assert_eq!(
            (input.kind, input.role.as_str()),
            (TranscriptKind::Input, "user")
        );
        assert_eq!(input.content, "query");
        let turn = TranscriptEntry::assistant_turn("sub-1", "text", 9);
        assert_eq!(
            (turn.kind, turn.role.as_str()),
            (TranscriptKind::AssistantTurn, "assistant")
        );
        let call = TranscriptEntry::tool_call("sub-1", "bash", "ls", Some(true), 10);
        assert_eq!(call.kind, TranscriptKind::ToolCall);
        assert_eq!(call.role, "assistant");
        assert_eq!(call.content, "ls");
        assert_eq!(call.tool_name.as_deref(), Some("bash"));
        assert_eq!(call.success, Some(true));
        let pending = TranscriptEntry::tool_call("sub-1", "bash", "ls", None, 10);
        assert_eq!(pending.success, None);
        let completion = TranscriptEntry::completion("sub-1", "done", 11);
        assert_eq!(completion.kind, TranscriptKind::Completion);
        assert_eq!(completion.role, "assistant");
        let error = TranscriptEntry::error("sub-1", "boom", 12);
        assert_eq!(error.kind, TranscriptKind::Error);
        assert_eq!(error.success, Some(false));
        let status = TranscriptEntry::status("sub-1", "running", 13);
        assert_eq!(status.kind, TranscriptKind::Status);
        assert_eq!(
            (status.role.as_str(), status.content.as_str()),
            ("system", "running")
        );
        let closed = TranscriptEntry::closed("sub-1", "parent closed", 14);
        assert_eq!(closed.kind, TranscriptKind::Closed);
        let evicted = TranscriptEntry::evicted("sub-1", "ring full", 15);
        assert_eq!(evicted.kind, TranscriptKind::Evicted);
        assert_eq!(TranscriptEntry::default().kind, TranscriptKind::Status);
        assert_eq!(TranscriptEntry::default().role, "system");
    }
    #[test]
    fn entry_summary_collapses_newlines_and_truncates() {
        let entry = TranscriptEntry::assistant_turn("sub-1", "line one\nline two\r\nline three", 1);
        assert_eq!(
            entry.summary(),
            "assistant_turn: line one line two line three"
        );
        let long = TranscriptEntry::completion("sub-1", "x".repeat(400), 1);
        let summary = long.summary();
        assert!(summary.ends_with("..."));
        assert_eq!(summary.chars().count(), 120 + 3);
        assert_eq!(
            summary.chars().take(14).collect::<String>(),
            "completion: xx"
        );
    }
    #[test]
    fn transcript_entry_json_round_trip_uses_snake_case_keys() {
        let entry =
            TranscriptEntry::tool_call("sub-1", "bash", "ls -la", Some(true), 1_700_000_000_000);
        let value = serde_json::to_value(&entry).expect("entry serializes");
        let object = value.as_object().expect("entry is an object");
        for key in [
            "seq",
            "subagent_id",
            "kind",
            "role",
            "content",
            "tool_name",
            "success",
            "at_ms",
        ] {
            assert!(object.contains_key(key), "missing key {key}");
        }
        assert_eq!(value["kind"], serde_json::json!("tool_call"));
        assert_eq!(value["at_ms"], serde_json::json!(1_700_000_000_000_i64));
        let decoded: TranscriptEntry = serde_json::from_value(value).expect("entry decodes");
        assert_eq!(decoded, entry);
        let partial: TranscriptEntry =
            serde_json::from_value(serde_json::json!({ "content": "hi" })).expect("defaults fill");
        assert_eq!(partial.content, "hi");
        assert_eq!(partial.kind, TranscriptKind::Status);
        assert_eq!(partial.role, "system");
    }
    #[test]
    fn transcript_kind_json_uses_wire_strings() {
        for (kind, wire) in [
            (TranscriptKind::Spawn, "spawn"),
            (TranscriptKind::Input, "input"),
            (TranscriptKind::AssistantTurn, "assistant_turn"),
            (TranscriptKind::ToolCall, "tool_call"),
            (TranscriptKind::Completion, "completion"),
            (TranscriptKind::Error, "error"),
            (TranscriptKind::Status, "status"),
            (TranscriptKind::Closed, "closed"),
            (TranscriptKind::Evicted, "evicted"),
        ] {
            assert_eq!(kind.as_str(), wire);
            assert_eq!(
                serde_json::to_value(kind).expect("kind serializes"),
                serde_json::json!(wire)
            );
            let decoded: TranscriptKind =
                serde_json::from_value(serde_json::json!(wire)).expect("kind decodes");
            assert_eq!(decoded, kind);
        }
    }
    #[test]
    fn transcript_summary_json_round_trip_uses_snake_case_keys() {
        let mut buffer = TranscriptBuffer::new(4);
        buffer.push(TranscriptEntry::completion("sub-1", "done", 1));
        let summary = buffer.summary();
        let value = serde_json::to_value(&summary).expect("summary serializes");
        let object = value.as_object().expect("summary is an object");
        for key in ["count", "dropped", "last_kind", "last_content"] {
            assert!(object.contains_key(key), "missing key {key}");
        }
        assert_eq!(value["last_kind"], serde_json::json!("completion"));
        let decoded: TranscriptSummary = serde_json::from_value(value).expect("summary decodes");
        assert_eq!(decoded, summary);
        let empty: TranscriptSummary =
            serde_json::from_value(serde_json::json!({})).expect("defaults fill");
        assert_eq!(empty, TranscriptSummary::default());
        assert_eq!(empty.display(), "0 entries");
    }
    #[test]
    fn channel_publishes_without_subscribers() {
        let channel = TranscriptChannel::new(0);
        assert_eq!(channel.receiver_count(), 0);
        channel.publish(&TranscriptEntry::spawn("sub-1", "task", 1));
        assert_eq!(channel.receiver_count(), 0);
    }
    #[tokio::test]
    async fn channel_delivers_published_entries_in_order() {
        let channel = TranscriptChannel::new(8);
        channel.publish(&TranscriptEntry::spawn("sub-1", "before subscribe", 1));
        let mut subscription = channel.subscribe();
        assert_eq!(channel.receiver_count(), 1);
        channel.publish(&TranscriptEntry::input("sub-1", "second", 2));
        channel.publish(&TranscriptEntry::completion("sub-1", "third", 3));
        let first = subscription.recv().await.expect("first entry");
        let second = subscription.recv().await.expect("second entry");
        assert_eq!(
            (first.kind, first.content.as_str()),
            (TranscriptKind::Input, "second")
        );
        assert_eq!(
            (second.kind, second.content.as_str()),
            (TranscriptKind::Completion, "third")
        );
        let drained = subscription.try_recv().expect_err("no more entries");
        assert!(drained.to_string().contains("empty"));
        drop(subscription);
        assert_eq!(channel.receiver_count(), 0);
    }
}
