//! Value enums of the harness event vocabulary.
//!
//! Mirrors the enum section of the Python `openjiuwen.harness_protocol.events` module
//! (v0.1.20). Every variant serializes to its Python `.value` string via `snake_case`
//! renaming, and [`as_str`](TurnEventKind::as_str) exposes the same string for validation
//! messages and sorting.
//!
//! These are plain data enums: variants are values, not callable API surface.

use serde::{Deserialize, Serialize};

/// Lifecycle transition for one external-input-driven turn.
///
/// Wire values are `started`, `paused`, `resumed`, `finished`, `aborted`, and `failed`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TurnEventKind {
    /// The turn started.
    Started,
    /// The turn was paused.
    Paused,
    /// A paused turn was resumed.
    Resumed,
    /// The turn finished normally.
    Finished,
    /// The turn was aborted.
    Aborted,
    /// The turn failed.
    Failed,
}

/// The turn event kinds that end a turn, in Python's declared order.
///
/// Python stores these in a `frozenset`; the fixed array preserves a deterministic order for
/// Rust callers.
pub const TERMINAL_TURN_EVENT_KINDS: [TurnEventKind; 3] = [
    TurnEventKind::Finished,
    TurnEventKind::Aborted,
    TurnEventKind::Failed,
];

impl TurnEventKind {
    /// Returns the Python `.value` string of this turn event kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Paused => "paused",
            Self::Resumed => "resumed",
            Self::Finished => "finished",
            Self::Aborted => "aborted",
            Self::Failed => "failed",
        }
    }

    /// Returns whether this kind ends the turn.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Aborted | Self::Failed)
    }
}

/// Portable representation of output content.
///
/// Wire values are `text` and `structured`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    /// Free-form text content.
    Text,
    /// Structured (JSON) content.
    Structured,
}

impl OutputKind {
    /// Returns the Python `.value` string of this output kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Structured => "structured",
        }
    }
}

/// Semantic channel carrying an output block.
///
/// Wire values are `answer`, `reasoning`, and `system`; the default is `answer`.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OutputChannel {
    /// The user-facing answer channel.
    #[default]
    Answer,
    /// The model's reasoning channel.
    Reasoning,
    /// The system-instruction channel.
    System,
}

impl OutputChannel {
    /// Returns the Python `.value` string of this output channel.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Answer => "answer",
            Self::Reasoning => "reasoning",
            Self::System => "system",
        }
    }
}

/// How an output update changes the identified content block.
///
/// Wire values are `delta`, `snapshot`, and `final`; the default is `snapshot`.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OutputOperation {
    /// Append the carried content to the block.
    Delta,
    /// Replace the block content with the carried snapshot.
    #[default]
    Snapshot,
    /// Mark the block final with the carried content.
    Final,
}

impl OutputOperation {
    /// Returns the Python `.value` string of this output operation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Delta => "delta",
            Self::Snapshot => "snapshot",
            Self::Final => "final",
        }
    }
}

/// Whether a usage update is incremental or cumulative.
///
/// Wire values are `delta` and `cumulative`; the default is `cumulative`.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum UsageUpdateMode {
    /// The carried usage is an increment over the previous update.
    Delta,
    /// The carried usage replaces all previously reported usage.
    #[default]
    Cumulative,
}

impl UsageUpdateMode {
    /// Returns the Python `.value` string of this usage update mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Delta => "delta",
            Self::Cumulative => "cumulative",
        }
    }
}

/// Backpressure retention class derived from an event payload.
///
/// Wire values are `required`, `coalescible`, and `best_effort`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum EventRetention {
    /// The event must be delivered.
    Required,
    /// The event may be coalesced with a later event for the same block.
    Coalescible,
    /// The event may be dropped under pressure.
    BestEffort,
}

impl EventRetention {
    /// Returns the Python `.value` string of this retention class.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Required => "required",
            Self::Coalescible => "coalescible",
            Self::BestEffort => "best_effort",
        }
    }
}

/// Allowed behavior when the bounded observation buffer is full.
///
/// Wire values are `block`, `coalesce_or_block`, and `drop_best_effort_or_block`; the default
/// is `coalesce_or_block`.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum EventOverflowPolicy {
    /// The producer waits until the buffer has room.
    Block,
    /// Coalesce a coalescible event into an earlier one, otherwise block.
    #[default]
    CoalesceOrBlock,
    /// Drop a best-effort event, otherwise coalesce or block.
    DropBestEffortOrBlock,
}

impl EventOverflowPolicy {
    /// Returns the Python `.value` string of this overflow policy.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::CoalesceOrBlock => "coalesce_or_block",
            Self::DropBestEffortOrBlock => "drop_best_effort_or_block",
        }
    }
}

/// Lifecycle transition for a provider item such as a tool call.
///
/// Wire values are `started`, `updated`, and `completed`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ItemEventKind {
    /// The item was created.
    Started,
    /// The item changed.
    Updated,
    /// The item reached its terminal state.
    Completed,
}

impl ItemEventKind {
    /// Returns the Python `.value` string of this item event kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Updated => "updated",
            Self::Completed => "completed",
        }
    }
}

/// Observable phase of a hook invocation.
///
/// Wire values are `started` and `finished`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HookEventPhase {
    /// The hook is about to run.
    Started,
    /// The hook returned.
    Finished,
}

impl HookEventPhase {
    /// Returns the Python `.value` string of this hook phase.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Finished => "finished",
        }
    }
}

/// Severity of a diagnostic event.
///
/// Wire values are `debug`, `info`, `warning`, and `error`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticLevel {
    /// Verbose diagnostic detail.
    Debug,
    /// Informational diagnostic.
    Info,
    /// A recoverable problem worth surfacing.
    Warning,
    /// A failure.
    Error,
}

impl DiagnosticLevel {
    /// Returns the Python `.value` string of this diagnostic level.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// Terminal status of one physical model request made inside a turn.
///
/// Wire values are `completed`, `failed`, and `cancelled`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ModelRequestStatus {
    /// The request completed.
    Completed,
    /// The request failed.
    Failed,
    /// The request was cancelled.
    Cancelled,
}

impl ModelRequestStatus {
    /// Returns the Python `.value` string of this model request status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{
        DiagnosticLevel, EventOverflowPolicy, EventRetention, HookEventPhase, ItemEventKind,
        ModelRequestStatus, OutputChannel, OutputKind, OutputOperation, TurnEventKind,
        UsageUpdateMode, TERMINAL_TURN_EVENT_KINDS,
    };

    fn assert_wire<T>(value: T, wire: &str)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Copy + PartialEq + std::fmt::Debug,
    {
        assert_eq!(
            serde_json::to_value(value).expect("serializes"),
            json!(wire)
        );
        let decoded: T = serde_json::from_value(json!(wire)).expect("deserializes");
        assert_eq!(decoded, value);
    }

    #[test]
    fn turn_event_kind_wire_values_and_terminals() {
        for (kind, wire) in [
            (TurnEventKind::Started, "started"),
            (TurnEventKind::Paused, "paused"),
            (TurnEventKind::Resumed, "resumed"),
            (TurnEventKind::Finished, "finished"),
            (TurnEventKind::Aborted, "aborted"),
            (TurnEventKind::Failed, "failed"),
        ] {
            assert_wire(kind, wire);
            assert_eq!(kind.as_str(), wire);
        }

        assert_eq!(
            TERMINAL_TURN_EVENT_KINDS,
            [
                TurnEventKind::Finished,
                TurnEventKind::Aborted,
                TurnEventKind::Failed
            ]
        );
        for kind in [
            TurnEventKind::Started,
            TurnEventKind::Paused,
            TurnEventKind::Resumed,
        ] {
            assert!(!kind.is_terminal());
        }
        for kind in TERMINAL_TURN_EVENT_KINDS {
            assert!(kind.is_terminal());
        }
    }

    #[test]
    fn output_enums_match_python_values() {
        for (kind, wire) in [
            (OutputKind::Text, "text"),
            (OutputKind::Structured, "structured"),
        ] {
            assert_wire(kind, wire);
            assert_eq!(kind.as_str(), wire);
        }
        for (operation, wire) in [
            (OutputOperation::Delta, "delta"),
            (OutputOperation::Snapshot, "snapshot"),
            (OutputOperation::Final, "final"),
        ] {
            assert_wire(operation, wire);
            assert_eq!(operation.as_str(), wire);
        }
        for (channel, wire) in [
            (OutputChannel::Answer, "answer"),
            (OutputChannel::Reasoning, "reasoning"),
            (OutputChannel::System, "system"),
        ] {
            assert_wire(channel, wire);
            assert_eq!(channel.as_str(), wire);
        }
    }

    #[test]
    fn usage_retention_and_overflow_wire_values() {
        for (mode, wire) in [
            (UsageUpdateMode::Delta, "delta"),
            (UsageUpdateMode::Cumulative, "cumulative"),
        ] {
            assert_wire(mode, wire);
            assert_eq!(mode.as_str(), wire);
        }
        for (retention, wire) in [
            (EventRetention::Required, "required"),
            (EventRetention::Coalescible, "coalescible"),
            (EventRetention::BestEffort, "best_effort"),
        ] {
            assert_wire(retention, wire);
            assert_eq!(retention.as_str(), wire);
        }
        for (policy, wire) in [
            (EventOverflowPolicy::Block, "block"),
            (EventOverflowPolicy::CoalesceOrBlock, "coalesce_or_block"),
            (
                EventOverflowPolicy::DropBestEffortOrBlock,
                "drop_best_effort_or_block",
            ),
        ] {
            assert_wire(policy, wire);
            assert_eq!(policy.as_str(), wire);
        }
    }

    #[test]
    fn item_hook_diagnostic_and_request_wire_values() {
        for (kind, wire) in [
            (ItemEventKind::Started, "started"),
            (ItemEventKind::Updated, "updated"),
            (ItemEventKind::Completed, "completed"),
        ] {
            assert_wire(kind, wire);
            assert_eq!(kind.as_str(), wire);
        }
        for (phase, wire) in [
            (HookEventPhase::Started, "started"),
            (HookEventPhase::Finished, "finished"),
        ] {
            assert_wire(phase, wire);
            assert_eq!(phase.as_str(), wire);
        }
        for (level, wire) in [
            (DiagnosticLevel::Debug, "debug"),
            (DiagnosticLevel::Info, "info"),
            (DiagnosticLevel::Warning, "warning"),
            (DiagnosticLevel::Error, "error"),
        ] {
            assert_wire(level, wire);
            assert_eq!(level.as_str(), wire);
        }
        for (status, wire) in [
            (ModelRequestStatus::Completed, "completed"),
            (ModelRequestStatus::Failed, "failed"),
            (ModelRequestStatus::Cancelled, "cancelled"),
        ] {
            assert_wire(status, wire);
            assert_eq!(status.as_str(), wire);
        }
    }

    #[test]
    fn defaults_match_python_dataclass_defaults() {
        assert_eq!(OutputOperation::default(), OutputOperation::Snapshot);
        assert_eq!(OutputChannel::default(), OutputChannel::Answer);
        assert_eq!(UsageUpdateMode::default(), UsageUpdateMode::Cumulative);
        assert_eq!(
            EventOverflowPolicy::default(),
            EventOverflowPolicy::CoalesceOrBlock
        );
    }
}
