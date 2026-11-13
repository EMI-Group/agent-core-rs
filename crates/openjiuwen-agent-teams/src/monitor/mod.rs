//! Plain-text status stream for a live team.
//!
//! Ports the status-stream half of `openjiuwen/agent_teams/monitor/` (Python v0.1.20) as a
//! **text** monitor: one greppable record per line, no TUI, no colour, no cursor control. The
//! monitor is a pure renderer — it reads a snapshot or an event and returns a `String`; it never
//! talks to the store, so Python's monitor database (event history, queries, retention) is
//! deferred.
//!
//! Module map:
//!
//! | Module | Types |
//! | --- | --- |
//! | [`event`] | [`MonitorEvent`], [`StampedEvent`], [`snapshot_events`] — what the stream carries |
//! | [`render`] | [`TeamMonitor`], [`TeamMonitorConfig`], [`TimestampPolicy`] — the text shape |
//!
//! Typical use: a live feed turns bus payloads into [`MonitorEvent`]s through the `from_*`
//! constructors and prints [`TeamMonitor::render_event`]; an on-demand status view passes a
//! `TeamSnapshot` to [`TeamMonitor::render_snapshot`].
pub mod event;
pub mod render;

pub use event::{snapshot_events, MonitorEvent, StampedEvent};
pub use render::{TeamMonitor, TeamMonitorConfig, TimestampPolicy};
