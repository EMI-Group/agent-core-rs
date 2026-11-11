//! Events and reports emitted by the context engine.
//!
//! Every processor that changes the history returns a [`ContextEvent`]; the engine
//! aggregates them into a [`CompressReport`]. Python returns loose dictionaries
//! from `ContextEngine.compress`; the rust port types them so callers can match
//! on the outcome without parsing strings.
use serde::{Deserialize, Serialize};

/// What a processor did to the history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextEventKind {
    /// Whole dialogue rounds were dropped.
    Truncated,
    /// Older rounds were replaced by a summary.
    Summarized,
    /// Tool output was written to offload storage.
    Offloaded,
    /// A tool result was shortened in place.
    ToolResultTrimmed,
}

/// One history change reported by a processor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextEvent {
    /// Name of the processor that made the change.
    pub processor: String,
    /// What kind of change it was.
    pub kind: ContextEventKind,
    /// Indices of the messages that changed, in the history *before* the change.
    pub messages_changed: Vec<usize>,
    /// The summary text, for [`ContextEventKind::Summarized`].
    pub summary: Option<String>,
    /// The offload handle, for [`ContextEventKind::Offloaded`].
    pub offload_handle: Option<String>,
    /// Token count before the processor ran.
    pub tokens_before: usize,
    /// Token count after the processor ran.
    pub tokens_after: usize,
}

impl Default for ContextEvent {
    fn default() -> Self {
        Self {
            processor: String::new(),
            // The first kind is Python's `truncated`; an event without an
            // explicit kind is a truncation report.
            kind: ContextEventKind::Truncated,
            messages_changed: Vec::new(),
            summary: None,
            offload_handle: None,
            tokens_before: 0,
            tokens_after: 0,
        }
    }
}

/// Outcome of an explicit [`ContextEngine::compress`] call.
///
/// [`ContextEngine::compress`]: crate::engine::ContextEngine::compress
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionStatus {
    /// Nothing was over budget, or no processor changed the history.
    #[default]
    Noop,
    /// The history was compressed and persisted.
    Compressed,
    /// Another compression was already in flight for this context.
    Busy,
}

/// Result of one compression pass.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompressReport {
    /// Whether anything happened.
    pub status: CompressionStatus,
    /// One entry per processor that changed the history.
    pub events: Vec<ContextEvent>,
    /// Token count before the pass.
    pub tokens_before: usize,
    /// Token count after the pass.
    pub tokens_after: usize,
    /// Number of messages after the pass.
    pub messages: usize,
}

impl CompressReport {
    /// Whether the report describes a real change.
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.status == CompressionStatus::Compressed
    }
}

/// Result of recovering from a provider context-overflow error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecoveryOutcome {
    /// Whether recovery was allowed to try at all.
    pub attempted: bool,
    /// Whether the history is now smaller and the request may be retried.
    pub recovered: bool,
    /// The compression report, when a pass ran.
    pub report: Option<CompressReport>,
}

/// Result of appending a message through [`ContextEngine::observe`].
///
/// [`ContextEngine::observe`]: crate::engine::ContextEngine::observe
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ObserveOutcome {
    /// How many messages were appended.
    pub added: usize,
    /// How many messages the rolling limit evicted.
    pub evicted: usize,
    /// Total messages after the append.
    pub total_messages: usize,
    /// Total tokens after the append.
    pub total_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_serialize_to_python_names() {
        assert_eq!(
            serde_json::to_string(&CompressionStatus::Noop).unwrap(),
            "\"noop\""
        );
        assert_eq!(
            serde_json::to_string(&CompressionStatus::Compressed).unwrap(),
            "\"compressed\""
        );
        assert_eq!(
            serde_json::to_string(&CompressionStatus::Busy).unwrap(),
            "\"busy\""
        );
        assert_eq!(
            serde_json::to_string(&ContextEventKind::ToolResultTrimmed).unwrap(),
            "\"tool_result_trimmed\""
        );
    }

    #[test]
    fn reports_round_trip_and_tolerate_missing_fields() {
        let report = CompressReport {
            status: CompressionStatus::Compressed,
            events: vec![ContextEvent {
                processor: "truncation".to_owned(),
                kind: ContextEventKind::Truncated,
                messages_changed: vec![0, 1],
                tokens_before: 100,
                tokens_after: 20,
                ..ContextEvent::default()
            }],
            tokens_before: 100,
            tokens_after: 20,
            messages: 4,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<CompressReport>(&json).unwrap(),
            report
        );

        let empty: CompressReport = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, CompressReport::default());
        assert!(!empty.is_compressed());

        let outcome: RecoveryOutcome = serde_json::from_str("{}").unwrap();
        assert_eq!(outcome, RecoveryOutcome::default());
        assert!(!outcome.attempted);

        let observe: ObserveOutcome =
            serde_json::from_str(r#"{"added": 1, "total_tokens": 7}"#).unwrap();
        assert_eq!(observe.added, 1);
        assert_eq!(observe.total_tokens, 7);
        assert_eq!(observe.evicted, 0);
    }
}
