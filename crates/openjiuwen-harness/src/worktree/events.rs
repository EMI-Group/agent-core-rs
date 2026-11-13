//! Worktree lifecycle events and the in-process sink seam.
//!
//! Port of `openjiuwen/harness/tools/worktree/events.py`: the helper emits pure
//! data events and the caller decides where they go. Two sinks ship here — an
//! in-process [`tokio::sync::broadcast`] channel for hosts that fan events out to
//! a UI or a team runtime, and a [`RecordingWorktreeEvents`] sink for tests and
//! diagnostics.
//!
//! Events are emitted after the operation succeeded (or, for
//! [`WorktreeEvent::Error`], after it failed) and never change the outcome: a
//! sink cannot fail, so worktree handling cannot be aborted by an observer.
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Capacity of a [`BroadcastWorktreeEvents`] channel when none is configured.
pub const DEFAULT_EVENT_CAPACITY: usize = 64;

/// One worktree lifecycle event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorktreeEvent {
    /// A worktree was created, or recovered because it already existed.
    Created {
        /// Requested name.
        name: String,
        /// Absolute path of the worktree directory.
        path: std::path::PathBuf,
        /// Branch checked out in the worktree.
        branch: String,
        /// Whether the worktree already existed.
        existed: bool,
        /// Owner id, when the caller supplied one.
        owner: Option<String>,
    },
    /// A worktree was removed.
    Removed {
        /// Requested name.
        name: String,
        /// Absolute path the worktree used to live at.
        path: std::path::PathBuf,
        /// Whether removal skipped the dirty-worktree check.
        forced: bool,
    },
    /// A worktree was locked.
    Locked {
        /// Requested name.
        name: String,
        /// Absolute path of the worktree directory.
        path: std::path::PathBuf,
        /// Lock reason, when one was supplied.
        reason: Option<String>,
    },
    /// A worktree was unlocked.
    Unlocked {
        /// Requested name.
        name: String,
        /// Absolute path of the worktree directory.
        path: std::path::PathBuf,
    },
    /// Stale registry entries and prunable git worktrees were cleaned up.
    Pruned {
        /// Number of registry records dropped.
        removed: usize,
    },
    /// An operation failed; the error was already returned to the caller.
    Error {
        /// Operation that failed (`create`, `remove`, `lock`, …).
        action: String,
        /// Failure message.
        message: String,
    },
}

impl WorktreeEvent {
    /// The event's `type` tag, matching its serde spelling.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Created { .. } => "created",
            Self::Removed { .. } => "removed",
            Self::Locked { .. } => "locked",
            Self::Unlocked { .. } => "unlocked",
            Self::Pruned { .. } => "pruned",
            Self::Error { .. } => "error",
        }
    }
}

/// A sink for [`WorktreeEvent`]s.
///
/// Implementations must be non-blocking: the manager awaits `emit` inline and a
/// slow sink would slow worktree operations down.
#[async_trait]
pub trait WorktreeEventSink: Send + Sync {
    /// Receives one event.
    async fn emit(&self, event: WorktreeEvent);
}

/// A sink that drops every event; the default of a fresh manager.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopWorktreeEvents;

#[async_trait]
impl WorktreeEventSink for NoopWorktreeEvents {
    async fn emit(&self, _event: WorktreeEvent) {}
}

/// An in-process broadcast sink.
///
/// Every [`subscribe`](Self::subscribe) caller gets its own receiver; with no
/// receiver the event is dropped, like any `tokio` broadcast channel.
#[derive(Debug)]
pub struct BroadcastWorktreeEvents {
    sender: broadcast::Sender<WorktreeEvent>,
}

impl BroadcastWorktreeEvents {
    /// Builds a channel retaining the last `capacity` events (`0` is raised to
    /// one, which is what `tokio` accepts).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self { sender }
    }

    /// Subscribes to future events.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<WorktreeEvent> {
        self.sender.subscribe()
    }

    /// Number of active receivers.
    #[must_use]
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for BroadcastWorktreeEvents {
    fn default() -> Self {
        Self::new(DEFAULT_EVENT_CAPACITY)
    }
}

#[async_trait]
impl WorktreeEventSink for BroadcastWorktreeEvents {
    async fn emit(&self, event: WorktreeEvent) {
        // A channel with no receiver is not an error: nobody listens.
        let _ = self.sender.send(event);
    }
}

/// A sink that records every event in memory, for tests and diagnostics.
#[derive(Debug, Default)]
pub struct RecordingWorktreeEvents {
    events: Mutex<Vec<WorktreeEvent>>,
}

impl RecordingWorktreeEvents {
    /// Builds an empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The recorded events, oldest first.
    #[must_use]
    pub fn events(&self) -> Vec<WorktreeEvent> {
        self.events.lock().clone()
    }

    /// Drops every recorded event.
    pub fn clear(&self) {
        self.events.lock().clear();
    }
}

#[async_trait]
impl WorktreeEventSink for RecordingWorktreeEvents {
    async fn emit(&self, event: WorktreeEvent) {
        self.events.lock().push(event);
    }
}

impl fmt::Display for WorktreeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "worktree.{}", self.kind())
    }
}

/// Convenience alias for a shared sink.
pub type SharedWorktreeEvents = Arc<dyn WorktreeEventSink>;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn recording_sink_keeps_order_and_clears() {
        let sink = RecordingWorktreeEvents::new();
        sink.emit(WorktreeEvent::Pruned { removed: 2 }).await;
        sink.emit(WorktreeEvent::Unlocked {
            name: "feature".to_owned(),
            path: PathBuf::from("/repo/.worktrees/feature"),
        })
        .await;

        let events = sink.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind(), "pruned");
        assert_eq!(events[1].kind(), "unlocked");
        assert_eq!(events[0].to_string(), "worktree.pruned");
        sink.clear();
        assert_eq!(sink.events(), []);
    }

    #[tokio::test]
    async fn broadcast_sink_delivers_to_every_receiver() {
        let sink = Arc::new(BroadcastWorktreeEvents::new(4));
        let mut first = sink.subscribe();
        let mut second = sink.subscribe();
        assert_eq!(sink.receiver_count(), 2);

        sink.emit(WorktreeEvent::Pruned { removed: 1 }).await;

        assert_eq!(first.recv().await.unwrap().kind(), "pruned");
        assert_eq!(second.recv().await.unwrap().kind(), "pruned");
    }

    #[tokio::test]
    async fn broadcast_sink_without_receivers_drops_events() {
        let sink = BroadcastWorktreeEvents::default();
        sink.emit(WorktreeEvent::Pruned { removed: 0 }).await;
        assert_eq!(sink.receiver_count(), 0);
    }

    #[test]
    fn events_serialize_with_a_type_tag() {
        let event = WorktreeEvent::Created {
            name: "feature".to_owned(),
            path: PathBuf::from("/repo/.worktrees/feature"),
            branch: "worktree-feature".to_owned(),
            existed: false,
            owner: None,
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "created",
                "name": "feature",
                "path": "/repo/.worktrees/feature",
                "branch": "worktree-feature",
                "existed": false,
                "owner": null,
            })
        );
        let decoded: WorktreeEvent = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, event);
    }

    #[tokio::test]
    async fn noop_sink_accepts_everything() {
        let sink = NoopWorktreeEvents;
        sink.emit(WorktreeEvent::Error {
            action: "create".to_owned(),
            message: "boom".to_owned(),
        })
        .await;
    }

    #[test]
    fn shared_alias_wraps_a_sink() {
        let sink: SharedWorktreeEvents = Arc::new(NoopWorktreeEvents);
        assert_eq!(Arc::strong_count(&sink), 1);
    }
}
