//! Plain-text status stream for a live team.
//!
//! Ports the status-stream half of `openjiuwen/agent_teams/monitor/` (Python v0.1.20) as a
//! **text** monitor: one greppable record per line, no TUI, no colour, no cursor control. The
//! monitor never talks to the store, so Python's monitor database (event history, queries,
//! retention) is deferred; every query here is answered by a
//! [`TeamSnapshot`](crate::schema::TeamSnapshot) the caller provides.
//!
//! The node has three layers:
//!
//! | Module | Layer | Types |
//! | --- | --- | --- |
//! | [`event`] | renderable event model | [`MonitorEvent`], [`StampedEvent`], [`snapshot_events`] |
//! | [`render`] | deterministic one-line renderer | [`TeamMonitor`], [`TeamMonitorConfig`], [`TimestampPolicy`] |
//! | [`stream`] | bounded async emitter with backpressure | [`StatusStream`], [`MonitorEmitter`], [`MonitorReceiver`], [`StatusStreamHandle`] |
//!
//! ## Flows
//!
//! - **Live feed**: bus payloads and runtime hooks become [`MonitorEvent`]s (through the `from_*`
//!   constructors), are stamped, pushed through a [`MonitorEmitter`], and rendered one line per
//!   event by the writer task spawned by [`StatusStream::spawn`].
//! - **On-demand status**: a `TeamSnapshot` goes to [`TeamMonitor::render_snapshot`], or to
//!   [`snapshot_events`] plus [`TeamMonitor::render_event`] when the host wants the same lines as
//!   the live feed.
//!
//! The output is plain text on purpose: it is meant to be `grep`ped, `tail -f`ed and diffed, and
//! the CLI owns all presentation beyond it (no TUI here). No layer performs a database query: the
//! monitor renders what the runtime hands it.
pub mod event;
pub mod render;
pub mod stream;

pub use event::{snapshot_events, MonitorEvent, StampedEvent};
pub use render::{TeamMonitor, TeamMonitorConfig, TimestampPolicy};
pub use stream::{
    BackpressurePolicy, EmitOutcome, MonitorEmitter, MonitorReceiver, StatusStream,
    StatusStreamHandle, StreamSummary,
};
