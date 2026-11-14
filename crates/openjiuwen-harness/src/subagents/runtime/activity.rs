//! Throttled per-subagent activity streams.
//!
//! Ports the bounded, throttled activity pipeline of Python
//! `openjiuwen/harness/subagent_runtime/activity.py`, `activity_events.py` and the
//! `SubagentActivity` model.
//! Python projects stream chunks into activity milestones, queues them in a bounded
//! `asyncio.Queue` and drains that queue into the parent session stream; a Rust runtime has no
//! parent stream writer, so the drain loop becomes a bounded `tokio::sync::broadcast` feed that a
//! monitoring caller subscribes to.
//! [`ActivityStream`] is the per-instance recorder: it retains a bounded ring of
//! [`SubagentActivityFrame`]s, coalesces reasoning text against the caller's clock, folds the
//! per-turn overflow into one [`ActivityKind::Truncated`] frame and publishes every frame it
//! records.
//! Frames keep the Python `SubagentActivity` field names and add the lifecycle kinds
//! (`spawn`/`queued`/`running`/`idle`/`completed`/`failed`/`evicted`/`closed`) that the
//! in-process runtime reports instead of Python's parent-session stream writer.
//! The recorder holds no clock: every publisher takes `at_ms`, the value the runtime read from
//! its injected `Clock`, so throttling stays deterministic in tests.
//! Every [`ActivityStream`] method is synchronous and non-blocking, because the runtime owns one
//! stream per instance and mutates it under its state lock.
// Size note: the implementation is ~800 lines; the file exceeds the crate's 1000-line guideline
// only because the inline `#[cfg(test)]` module below covers throttling, truncation, ring drops,
// terminal ordering, lag reporting and serde round-trips.
// The runtime wires this module in a follow-up change: until `subagents/mod.rs` re-exports it,
// nothing in the crate constructs a stream, so both reachability lints would fire on every item.
#![allow(dead_code, unreachable_pub)]

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Weak};

use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Frames retained per activity ring, and the per-turn frame cap (Python `activity_queue_size`).
pub const DEFAULT_ACTIVITY_CAPACITY: usize = 256;
/// Longest frame summary kept, in characters (Python `activity_text_max_len`).
pub const DEFAULT_ACTIVITY_TEXT_MAX_LEN: usize = 2000;
/// Window a coalesced thinking frame waits for, in milliseconds (Python `activity_throttle_ms`).
pub const DEFAULT_ACTIVITY_THROTTLE_MS: i64 = 500;
/// Largest throttle value [`ActivityThrottle::new`] accepts; larger requests are clamped.
const MAX_ACTIVITY_CAPACITY: usize = 4096;
/// Largest summary length [`ActivityThrottle::new`] accepts; larger requests are clamped.
const MAX_ACTIVITY_TEXT_MAX_LEN: usize = 65536;
/// Summary of the single frame that marks a truncated turn (Python text).
const TRUNCATED_SUMMARY: &str = "activity stream truncated for this turn";

/// What one activity frame records.
///
/// The vocabulary is Python's milestone set (`thinking`, `tool_call`, `tool_result`, `error`,
/// `truncated`) plus the lifecycle kinds the in-process runtime reports for an instance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityKind {
    /// The sub-agent was spawned with a task description.
    Spawn,
    /// The instance reserved a slot and waits for its first turn.
    #[default]
    Queued,
    /// A turn of the sub-agent started.
    Running,
    /// Coalesced reasoning text of the current thinking phase.
    Thinking,
    /// The sub-agent invoked a tool.
    ToolCall,
    /// A tool returned, with its outcome when known.
    ToolResult,
    /// The sub-agent reported an error without ending its work.
    Error,
    /// The instance finished a turn and is idle again.
    Idle,
    /// The sub-agent finished its work and reported an output.
    Completed,
    /// The sub-agent's work failed.
    Failed,
    /// The instance was evicted by the LRU policy.
    Evicted,
    /// The instance was closed.
    Closed,
    /// The turn hit the per-turn frame cap; later frames of that turn are lost.
    Truncated,
}

impl ActivityKind {
    /// The Python wire string of this kind (`tool_call`, `queued`, ...).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Spawn => "spawn",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Thinking => "thinking",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::Error => "error",
            Self::Idle => "idle",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Evicted => "evicted",
            Self::Closed => "closed",
            Self::Truncated => "truncated",
        }
    }

    /// Whether this kind ends the sub-agent's activity stream: only the four lifecycle outcomes
    /// are terminal, while [`Self::Error`] is a recoverable milestone.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Evicted | Self::Closed
        )
    }
}

/// One activity milestone of a sub-agent (Python `SubagentActivity`).
///
/// The kind default is [`ActivityKind::Queued`], so a frame decoded from a partial payload is
/// never terminal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubagentActivityFrame {
    /// The sub-agent instance this frame belongs to.
    pub subagent_id: String,
    /// The task the frame was recorded under; [`ActivityStream::begin_turn`] sets it.
    pub task_id: String,
    /// Ring-assigned sequence number (1-based, monotone across turns and evictions).
    pub seq: u64,
    /// What this frame records.
    pub kind: ActivityKind,
    /// One-line payload, truncated to the throttle's `text_max_len`.
    pub summary: String,
    /// Tool name, for tool frames.
    pub tool_name: Option<String>,
    /// Tool call id; publishers leave it unset, decoded Python payloads may carry it.
    pub tool_call_id: Option<String>,
    /// Tool or turn outcome, when known.
    pub ok: Option<bool>,
    /// Milliseconds since the Unix epoch, supplied by the caller's clock.
    pub at_ms: i64,
    /// Losses this frame reports: the ring drops so far, or the truncated turn's lost frames.
    pub dropped: Option<u64>,
    /// Identifier of the thinking phase the frame belongs to; `0` before the first phase.
    pub phase_id: u64,
}

impl SubagentActivityFrame {
    /// Whether this frame ends the sub-agent's activity stream.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.kind.is_terminal()
    }
}

/// One item a subscriber of an [`ActivityStream`] receives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivityEvent {
    /// A recorded frame, in publication order.
    Frame(SubagentActivityFrame),
    /// The subscriber fell behind the bounded feed and lost `dropped` frames.
    Lagged {
        /// Number of frames the lagging subscriber missed.
        dropped: u64,
    },
}

/// Digest of an [`ActivityStream`]: size, losses, and the newest frame.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ActivitySummary {
    /// Number of retained frames.
    pub count: usize,
    /// Number of frames evicted by the ring cap.
    pub dropped: u64,
    /// Whether a terminal frame was recorded.
    pub terminal: bool,
    /// Kind of the newest retained frame.
    pub last_kind: Option<ActivityKind>,
    /// Sequence number of the newest retained frame (`0` when the ring is empty).
    pub last_seq: u64,
}

impl ActivitySummary {
    /// `"3 frames, last: tool_call"`; singular `"1 frame, last: spawn"`, empty `"0 frames"`.
    #[must_use]
    pub fn display(&self) -> String {
        let noun = if self.count == 1 { "frame" } else { "frames" };
        match self.last_kind {
            Some(kind) => format!("{} {noun}, last: {}", self.count, kind.as_str()),
            None => format!("{} {noun}", self.count),
        }
    }
}

/// Tuning of one [`ActivityStream`]: Python `activity_queue_size`, `activity_text_max_len` and
/// `activity_throttle_ms`, clamped by every constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ActivityThrottle {
    /// Frames retained by the ring, and the per-turn frame cap.
    capacity: usize,
    /// Longest summary kept, in characters.
    text_max_len: usize,
    /// Window a thinking frame waits for, in milliseconds.
    throttle_ms: i64,
}

impl Default for ActivityThrottle {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_ACTIVITY_CAPACITY,
            text_max_len: DEFAULT_ACTIVITY_TEXT_MAX_LEN,
            throttle_ms: DEFAULT_ACTIVITY_THROTTLE_MS,
        }
    }
}

impl ActivityThrottle {
    /// Builds a throttle, clamping `capacity` to `1..=4096`, `text_max_len` to `1..=65536` and
    /// `throttle_ms` to `>= 0`.
    #[must_use]
    pub fn new(capacity: usize, text_max_len: usize, throttle_ms: i64) -> Self {
        Self {
            capacity: capacity.clamp(1, MAX_ACTIVITY_CAPACITY),
            text_max_len: text_max_len.clamp(1, MAX_ACTIVITY_TEXT_MAX_LEN),
            throttle_ms: throttle_ms.max(0),
        }
    }

    /// Same throttle with a different ring capacity and per-turn cap.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.clamp(1, MAX_ACTIVITY_CAPACITY);
        self
    }

    /// Same throttle with a different summary length limit.
    #[must_use]
    pub fn with_text_max_len(mut self, text_max_len: usize) -> Self {
        self.text_max_len = text_max_len.clamp(1, MAX_ACTIVITY_TEXT_MAX_LEN);
        self
    }

    /// Same throttle with a different thinking window.
    #[must_use]
    pub fn with_throttle_ms(mut self, throttle_ms: i64) -> Self {
        self.throttle_ms = throttle_ms.max(0);
        self
    }

    /// Frames retained by the ring, and the per-turn frame cap.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Longest summary kept, in characters.
    #[must_use]
    pub fn text_max_len(&self) -> usize {
        self.text_max_len
    }

    /// Window a thinking frame waits for, in milliseconds.
    #[must_use]
    pub fn throttle_ms(&self) -> i64 {
        self.throttle_ms
    }
}

/// One live subscription to an [`ActivityStream`]'s frame feed.
///
/// The feed is a bounded broadcast buffer, so a slow subscriber loses the oldest frames and is
/// told how many through [`ActivityEvent::Lagged`]; the publisher never blocks.
/// The subscription holds the feed weakly, so it never keeps the channel open: once the stream is
/// dropped every wait reports the end of the feed.
pub struct ActivitySubscription {
    /// Frames published from now on.
    receiver: broadcast::Receiver<SubagentActivityFrame>,
    /// The stream's feed, held weakly.
    sender: Weak<broadcast::Sender<SubagentActivityFrame>>,
}

impl ActivitySubscription {
    /// Waits for the next event, reporting a lag as [`ActivityEvent::Lagged`]; `None` once the
    /// activity stream (and with it the feed) was dropped.
    pub async fn recv(&mut self) -> Option<ActivityEvent> {
        match self.receiver.recv().await {
            Ok(frame) => Some(ActivityEvent::Frame(frame)),
            Err(broadcast::error::RecvError::Lagged(dropped)) => {
                Some(ActivityEvent::Lagged { dropped })
            }
            Err(broadcast::error::RecvError::Closed) => None,
        }
    }

    /// Takes the next event without waiting; `None` when nothing is buffered or the feed closed.
    #[must_use]
    pub fn try_recv(&mut self) -> Option<ActivityEvent> {
        match self.receiver.try_recv() {
            Ok(frame) => Some(ActivityEvent::Frame(frame)),
            Err(broadcast::error::TryRecvError::Lagged(dropped)) => {
                Some(ActivityEvent::Lagged { dropped })
            }
            Err(broadcast::error::TryRecvError::Empty | broadcast::error::TryRecvError::Closed) => {
                None
            }
        }
    }

    /// Turns the subscription into a `Send + 'static` event stream that yields the buffered and
    /// freshly published events in order and ends once the activity stream is dropped.
    pub fn into_stream(self) -> BoxStream<'static, ActivityEvent> {
        futures::stream::unfold(self, |mut subscription| async move {
            let event = subscription.recv().await?;
            Some((event, subscription))
        })
        .boxed()
    }

    /// Number of live subscriptions, including this one (`0` once the stream was dropped).
    #[must_use]
    pub fn receiver_count(&self) -> usize {
        self.sender
            .upgrade()
            .map_or(0, |sender| sender.receiver_count())
    }
}

impl fmt::Debug for ActivitySubscription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActivitySubscription")
            .field("receiver_count", &self.receiver_count())
            .finish_non_exhaustive()
    }
}

/// The parts of a frame a publisher sets; identity, sequence, phase and ring accounting belong to
/// the stream.
struct FrameSpec {
    kind: ActivityKind,
    summary: String,
    tool: Option<String>,
    ok: Option<bool>,
    lost: Option<u64>,
}

impl FrameSpec {
    /// A frame of `kind` carrying `summary`.
    fn new(kind: ActivityKind, summary: &str) -> Self {
        Self {
            kind,
            summary: summary.to_owned(),
            tool: None,
            ok: None,
            lost: None,
        }
    }

    /// The same frame, named after the tool it came from.
    fn named(mut self, tool_name: &str) -> Self {
        self.tool = Some(tool_name.to_owned());
        self
    }

    /// The same frame, reporting `lost` losses instead of the ring's cumulative count.
    fn losing(mut self, lost: u64) -> Self {
        self.lost = Some(lost);
        self
    }
}

/// The bounded, throttled activity recorder of one sub-agent instance.
///
/// Recording assigns the next sequence number, appends to the bounded ring (dropping the oldest
/// frame when full and counting the loss) and publishes the frame on the broadcast feed without
/// blocking; a feed without subscribers is not an error.
/// Frame accounting follows Python's projector: a per-turn cap of [`ActivityThrottle::capacity`]
/// non-terminal frames ends the turn with exactly one [`ActivityKind::Truncated`] frame and
/// suppresses the rest of that turn; thinking text is coalesced until the pending buffer reaches
/// [`ActivityThrottle::text_max_len`] characters or [`ActivityThrottle::throttle_ms`]
/// milliseconds pass; a terminal frame flushes pending thinking and makes every later
/// non-terminal publisher return an empty vector.
pub struct ActivityStream {
    /// Identity stamped on every frame.
    subagent_id: String,
    /// Ring size, per-turn cap, summary limit and thinking window.
    throttle: ActivityThrottle,
    /// Bounded live feed; the subscription side holds it weakly.
    sender: Arc<broadcast::Sender<SubagentActivityFrame>>,
    /// Retained frames, oldest first.
    frames: VecDeque<SubagentActivityFrame>,
    /// Sequence number the next frame receives.
    next_seq: u64,
    /// Frames evicted by the ring cap.
    dropped: u64,
    /// Task id of the current turn.
    task_id: String,
    /// Non-terminal frames recorded in the current turn.
    turn_frames: usize,
    /// Whether the current turn hit its frame cap.
    truncated: bool,
    /// Whether a terminal frame was recorded.
    terminal: bool,
    /// Reasoning text waiting for the next thinking emission.
    thinking_buffer: String,
    /// Clock value of the last thinking emission (or of `begin_turn`).
    last_emit_ms: i64,
    /// Identifier of the newest thinking phase; `0` before the first phase.
    phase_id: u64,
    /// Whether the current thinking phase is still open.
    phase_open: bool,
}

impl ActivityStream {
    /// Creates the stream of `subagent_id`; the throttle is re-clamped, so a decoded out-of-range
    /// throttle cannot shrink the ring below one frame.
    #[must_use]
    pub fn new(subagent_id: impl Into<String>, throttle: ActivityThrottle) -> Self {
        let throttle = ActivityThrottle::new(
            throttle.capacity(),
            throttle.text_max_len(),
            throttle.throttle_ms(),
        );
        let (sender, _receiver) = broadcast::channel(throttle.capacity());
        Self {
            subagent_id: subagent_id.into(),
            throttle,
            sender: Arc::new(sender),
            frames: VecDeque::with_capacity(throttle.capacity()),
            next_seq: 1,
            dropped: 0,
            task_id: String::new(),
            turn_frames: 0,
            truncated: false,
            terminal: false,
            thinking_buffer: String::new(),
            last_emit_ms: 0,
            phase_id: 0,
            phase_open: false,
        }
    }

    /// The sub-agent instance this stream belongs to.
    #[must_use]
    pub fn subagent_id(&self) -> &str {
        &self.subagent_id
    }

    /// The (clamped) tuning of this stream.
    #[must_use]
    pub fn throttle(&self) -> ActivityThrottle {
        self.throttle
    }

    /// Starts a turn for `task_id` (Python `reset_for_turn`): resets the per-turn frame count, the
    /// truncation flag, the pending thinking buffer and the throttle window to `at_ms`, while
    /// sequence numbers, ring losses and the thinking phase counter keep counting across turns.
    pub fn begin_turn(&mut self, task_id: &str, at_ms: i64) {
        task_id.clone_into(&mut self.task_id);
        self.turn_frames = 0;
        self.truncated = false;
        self.thinking_buffer.clear();
        self.last_emit_ms = at_ms;
        self.phase_open = false;
    }

    /// Records the spawn milestone carrying the delegated task description.
    pub fn spawn(&mut self, summary: &str, at_ms: i64) -> Vec<SubagentActivityFrame> {
        self.record(FrameSpec::new(ActivityKind::Spawn, summary), at_ms)
    }

    /// Records that the instance reserved a slot and waits for its first turn.
    pub fn queued(&mut self, at_ms: i64) -> Vec<SubagentActivityFrame> {
        self.record(lifecycle(ActivityKind::Queued), at_ms)
    }

    /// Records that a turn of the sub-agent started.
    pub fn running(&mut self, at_ms: i64) -> Vec<SubagentActivityFrame> {
        self.record(lifecycle(ActivityKind::Running), at_ms)
    }

    /// Records that the instance finished a turn and is idle again.
    pub fn idle(&mut self, at_ms: i64) -> Vec<SubagentActivityFrame> {
        self.record(lifecycle(ActivityKind::Idle), at_ms)
    }

    /// Buffers reasoning `text` and emits a thinking frame when the phase is due.
    ///
    /// The frame is emitted when the pending buffer reached the summary limit or when at least the
    /// throttle window passed since the previous emission (or since [`Self::begin_turn`]); an
    /// empty or whitespace-only chunk never emits, and returning to a thinking phase after a tool
    /// frame opens a new phase id.
    pub fn thinking(&mut self, text: &str, at_ms: i64) -> Vec<SubagentActivityFrame> {
        if self.terminal || self.truncated || text.trim().is_empty() {
            return Vec::new();
        }
        if self.turn_frames >= self.throttle.capacity {
            return vec![self.enter_truncated(at_ms)];
        }
        if !self.phase_open {
            self.phase_id += 1;
            self.phase_open = true;
        }
        self.thinking_buffer.push_str(text);
        let full = self.thinking_buffer.chars().count() >= self.throttle.text_max_len;
        let elapsed = at_ms.saturating_sub(self.last_emit_ms) >= self.throttle.throttle_ms;
        if !full && !elapsed {
            return Vec::new();
        }
        let pending = std::mem::take(&mut self.thinking_buffer);
        vec![self.emit_thinking(&pending, at_ms)]
    }

    /// Records a tool invocation: pending thinking is flushed first, then the tool frame.
    ///
    /// `detail` renders as `<tool_name>(<detail>)`; an empty detail leaves the bare tool name.
    pub fn tool_call(
        &mut self,
        tool_name: &str,
        detail: &str,
        ok: Option<bool>,
        at_ms: i64,
    ) -> Vec<SubagentActivityFrame> {
        let summary = if detail.trim().is_empty() {
            tool_name.to_owned()
        } else {
            format!("{tool_name}({detail})")
        };
        let mut spec = FrameSpec::new(ActivityKind::ToolCall, &summary).named(tool_name);
        spec.ok = ok;
        self.boundary(spec, at_ms)
    }

    /// Records a tool result: pending thinking is flushed first, then the result frame.
    pub fn tool_result(
        &mut self,
        tool_name: &str,
        summary: &str,
        ok: Option<bool>,
        at_ms: i64,
    ) -> Vec<SubagentActivityFrame> {
        let summary = if summary.trim().is_empty() {
            tool_name
        } else {
            summary
        };
        let mut spec = FrameSpec::new(ActivityKind::ToolResult, summary).named(tool_name);
        spec.ok = ok;
        self.boundary(spec, at_ms)
    }

    /// Records a recoverable error: pending thinking is flushed first, then the error frame.
    pub fn error(&mut self, message: &str, at_ms: i64) -> Vec<SubagentActivityFrame> {
        let summary = if message.trim().is_empty() {
            "subagent error"
        } else {
            message
        };
        self.boundary(FrameSpec::new(ActivityKind::Error, summary), at_ms)
    }

    /// Records the successful end of the sub-agent's work (`ok == Some(true)`); pending thinking
    /// is flushed first, and the stream then suppresses non-terminal frames.
    pub fn completed(&mut self, output: &str, at_ms: i64) -> Vec<SubagentActivityFrame> {
        let mut spec = FrameSpec::new(ActivityKind::Completed, output);
        spec.ok = Some(true);
        self.terminal(spec, at_ms)
    }

    /// Records the failed end of the sub-agent's work (`ok == Some(false)`).
    pub fn failed(&mut self, message: &str, at_ms: i64) -> Vec<SubagentActivityFrame> {
        let mut spec = FrameSpec::new(ActivityKind::Failed, message);
        spec.ok = Some(false);
        self.terminal(spec, at_ms)
    }

    /// Records that the instance was evicted by the LRU policy.
    pub fn evicted(&mut self, reason: &str, at_ms: i64) -> Vec<SubagentActivityFrame> {
        self.terminal(FrameSpec::new(ActivityKind::Evicted, reason), at_ms)
    }

    /// Records that the instance was closed.
    pub fn closed(&mut self, reason: &str, at_ms: i64) -> Vec<SubagentActivityFrame> {
        self.terminal(FrameSpec::new(ActivityKind::Closed, reason), at_ms)
    }

    /// Emits the thinking tail still buffered and closes the current phase; empty when nothing is
    /// pending or the turn already truncated.
    pub fn flush_thinking(&mut self, at_ms: i64) -> Vec<SubagentActivityFrame> {
        let pending = std::mem::take(&mut self.thinking_buffer);
        self.phase_open = false;
        if pending.trim().is_empty() || self.truncated {
            return Vec::new();
        }
        vec![self.emit_thinking(&pending, at_ms)]
    }

    /// All retained frames, oldest first.
    #[must_use]
    pub fn frames(&self) -> Vec<SubagentActivityFrame> {
        self.frames.iter().cloned().collect()
    }

    /// The last `limit` retained frames, oldest first; `0` yields an empty vector.
    #[must_use]
    pub fn recent(&self, limit: usize) -> Vec<SubagentActivityFrame> {
        if limit == 0 {
            return Vec::new();
        }
        let skip = self.frames.len().saturating_sub(limit);
        self.frames.iter().skip(skip).cloned().collect()
    }

    /// Retained frames with `seq > cursor`, oldest first, at most `limit` of them; `0` means no
    /// limit.
    #[must_use]
    pub fn after(&self, cursor: u64, limit: usize) -> Vec<SubagentActivityFrame> {
        let frames = self.frames.iter().filter(|frame| frame.seq > cursor);
        if limit == 0 {
            frames.cloned().collect()
        } else {
            frames.take(limit).cloned().collect()
        }
    }

    /// Number of currently retained frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no frame is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Number of frames evicted by the ring cap.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Sequence number the next recorded frame receives.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// A digest of the retained frames.
    #[must_use]
    pub fn summary(&self) -> ActivitySummary {
        let last = self.frames.back();
        ActivitySummary {
            count: self.frames.len(),
            dropped: self.dropped,
            terminal: self.terminal,
            last_kind: last.map(|frame| frame.kind),
            last_seq: last.map_or(0, |frame| frame.seq),
        }
    }

    /// Whether a terminal frame was recorded; the stream then suppresses non-terminal frames.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// Subscribes to every frame published from now on.
    #[must_use]
    pub fn subscribe(&self) -> ActivitySubscription {
        ActivitySubscription {
            receiver: self.sender.subscribe(),
            sender: Arc::downgrade(&self.sender),
        }
    }

    /// Number of live subscriptions.
    #[must_use]
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Drops every retained frame and returns how many were removed; this is not a ring eviction,
    /// so the sequence, the drop counter and the terminal flag are kept.
    pub fn clear(&mut self) -> usize {
        let removed = self.frames.len();
        self.frames.clear();
        removed
    }

    /// Records a non-terminal frame unless the turn is truncated or the stream is terminal.
    fn record(&mut self, spec: FrameSpec, at_ms: i64) -> Vec<SubagentActivityFrame> {
        if self.terminal || self.truncated {
            return Vec::new();
        }
        if self.turn_frames >= self.throttle.capacity {
            return vec![self.enter_truncated(at_ms)];
        }
        self.turn_frames += 1;
        vec![self.push_frame(spec, at_ms)]
    }

    /// Flushes pending thinking, then records a frame that ends the current thinking phase.
    fn boundary(&mut self, spec: FrameSpec, at_ms: i64) -> Vec<SubagentActivityFrame> {
        if self.terminal {
            return Vec::new();
        }
        let mut frames = self.flush_thinking(at_ms);
        frames.extend(self.record(spec, at_ms));
        frames
    }

    /// Flushes pending thinking, then records a frame that always passes and ends the stream.
    fn terminal(&mut self, spec: FrameSpec, at_ms: i64) -> Vec<SubagentActivityFrame> {
        let mut frames = self.flush_thinking(at_ms);
        frames.push(self.push_frame(spec, at_ms));
        self.terminal = true;
        frames
    }

    /// Enters the per-turn truncation state and returns the single marker frame, which reports
    /// `turn_frames - capacity + 1`: the frame that hit the cap is lost together with the rest.
    fn enter_truncated(&mut self, at_ms: i64) -> SubagentActivityFrame {
        let cap = self.throttle.capacity;
        let lost = u64::try_from(self.turn_frames.saturating_sub(cap) + 1).unwrap_or(u64::MAX);
        self.truncated = true;
        self.thinking_buffer.clear();
        self.phase_open = false;
        let spec = FrameSpec::new(ActivityKind::Truncated, TRUNCATED_SUMMARY).losing(lost);
        self.push_frame(spec, at_ms)
    }

    /// Emits one thinking frame for `text` and restarts the throttle window.
    fn emit_thinking(&mut self, text: &str, at_ms: i64) -> SubagentActivityFrame {
        self.last_emit_ms = at_ms;
        self.turn_frames += 1;
        self.push_frame(FrameSpec::new(ActivityKind::Thinking, text), at_ms)
    }

    /// Appends `spec` to the ring and publishes the resulting frame.
    ///
    /// The frame carries `spec`'s own loss count, otherwise the cumulative ring drops once the ring
    /// has dropped anything; a feed without subscribers is not an error.
    fn push_frame(&mut self, spec: FrameSpec, at_ms: i64) -> SubagentActivityFrame {
        if self.frames.len() >= self.throttle.capacity {
            self.frames.pop_front();
            self.dropped += 1;
        }
        let dropped = match spec.lost {
            Some(lost) => Some(lost),
            None if self.dropped > 0 => Some(self.dropped),
            None => None,
        };
        let frame = SubagentActivityFrame {
            subagent_id: self.subagent_id.clone(),
            task_id: self.task_id.clone(),
            seq: self.next_seq,
            kind: spec.kind,
            summary: truncate(&spec.summary, self.throttle.text_max_len),
            tool_name: spec.tool,
            tool_call_id: None,
            ok: spec.ok,
            at_ms,
            dropped,
            phase_id: self.phase_id,
        };
        self.next_seq += 1;
        self.frames.push_back(frame.clone());
        let _ = self.sender.send(frame.clone());
        frame
    }
}

impl fmt::Debug for ActivityStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActivityStream")
            .field("subagent_id", &self.subagent_id)
            .field("throttle", &self.throttle)
            .field("frames", &self.frames.len())
            .field("next_seq", &self.next_seq)
            .field("dropped", &self.dropped)
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

/// A lifecycle frame carrying the kind's wire string as its summary.
fn lifecycle(kind: ActivityKind) -> FrameSpec {
    FrameSpec::new(kind, kind.as_str())
}

/// Truncates `text` to `max_len` characters, appending `...` when it was cut.
fn truncate(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        return text.to_owned();
    }
    if max_len <= 3 {
        return text.chars().take(max_len).collect();
    }
    let head = text.chars().take(max_len - 3).collect::<String>();
    format!("{head}...")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        ActivityEvent, ActivityKind, ActivityStream, ActivitySummary, ActivityThrottle,
        SubagentActivityFrame,
    };
    use futures::StreamExt;
    use std::time::Duration;

    /// A recorder with `capacity` ring slots (and per-turn cap) and a `throttle_ms` window.
    fn recorder(capacity: usize, text_max_len: usize, throttle_ms: i64) -> ActivityStream {
        ActivityStream::new(
            "sub-1",
            ActivityThrottle::new(capacity, text_max_len, throttle_ms),
        )
    }

    fn kinds(frames: &[SubagentActivityFrame]) -> Vec<ActivityKind> {
        frames.iter().map(|frame| frame.kind).collect()
    }

    fn seqs(frames: &[SubagentActivityFrame]) -> Vec<u64> {
        frames.iter().map(|frame| frame.seq).collect()
    }

    /// Asserts that a publisher emitted nothing (its frame was suppressed or still throttled).
    #[allow(clippy::assert_is_empty)] // the assertion message prints the unexpected frames
    fn silent(frames: &[SubagentActivityFrame]) {
        assert!(frames.is_empty(), "expected no frames, got {frames:?}");
    }

    #[test]
    fn throttle_defaults_clamps_and_the_stream_normalizes_it() {
        let throttle = ActivityThrottle::default();
        assert_eq!(throttle.capacity(), 256);
        assert_eq!(throttle.text_max_len(), 2_000);
        assert_eq!(throttle.throttle_ms(), 500);
        let clamped = ActivityThrottle::new(0, 0, -5);
        assert_eq!(
            (
                clamped.capacity(),
                clamped.text_max_len(),
                clamped.throttle_ms()
            ),
            (1, 1, 0)
        );
        let clamped = ActivityThrottle::new(usize::MAX, usize::MAX, 10);
        assert_eq!((clamped.capacity(), clamped.text_max_len()), (4096, 65536));
        let tuned = ActivityThrottle::default()
            .with_capacity(4)
            .with_text_max_len(8);
        assert_eq!((tuned.capacity(), tuned.text_max_len()), (4, 8));
        let tuned = tuned
            .with_capacity(0)
            .with_text_max_len(0)
            .with_throttle_ms(-1);
        assert_eq!(
            (tuned.capacity(), tuned.text_max_len(), tuned.throttle_ms()),
            (1, 1, 0)
        );
        let raw: ActivityThrottle =
            serde_json::from_value(serde_json::json!({"capacity": 0, "text_max_len": 0})).unwrap();
        let stream = ActivityStream::new("sub-1", raw.with_throttle_ms(-1));
        assert_eq!(stream.subagent_id(), "sub-1");
        assert_eq!(stream.throttle().capacity(), 1);
        assert_eq!(stream.throttle().text_max_len(), 1);
        assert_eq!(stream.throttle().throttle_ms(), 0);
        assert_eq!(stream.frames(), Vec::new());
        assert_eq!(stream.next_seq(), 1);
        assert_eq!(stream.receiver_count(), 0);
    }

    #[test]
    fn lifecycle_publishers_record_sequenced_frames() {
        let mut stream = recorder(8, 2000, 500);
        stream.begin_turn("task-1", 0);
        let spawned = stream.spawn("explore the repo", 1);
        let queued = stream.queued(2);
        let running = stream.running(3);
        assert_eq!(kinds(&queued), vec![ActivityKind::Queued]);
        assert_eq!(kinds(&running), vec![ActivityKind::Running]);
        let frame = &spawned[0];
        assert_eq!(frame.subagent_id, "sub-1");
        assert_eq!(frame.task_id, "task-1");
        assert_eq!((frame.seq, frame.summary.as_str()), (1, "explore the repo"));
        assert_eq!((frame.at_ms, frame.phase_id), (1, 0));
        assert!(frame.dropped.is_none());
        assert!(frame.tool_name.is_none() && frame.tool_call_id.is_none());
        assert!(!frame.is_terminal());
        assert_eq!(queued[0].summary, "queued");
        assert_eq!(running[0].summary, "running");
        assert_eq!(seqs(&stream.frames()), vec![1, 2, 3]);
        assert_eq!(stream.next_seq(), 4);
        assert_eq!(stream.summary().display(), "3 frames, last: running");
    }

    #[test]
    fn thinking_is_throttled_coalesced_and_truncated() {
        let mut stream = recorder(16, 2000, 500);
        stream.begin_turn("task-1", 1_000);
        silent(&stream.thinking("Plan: ", 1_000));
        silent(&stream.thinking("read ", 1_400));
        let emitted = stream.thinking("it", 1_500);
        assert_eq!(kinds(&emitted), vec![ActivityKind::Thinking]);
        assert_eq!(emitted[0].summary, "Plan: read it");
        assert_eq!((emitted[0].phase_id, emitted[0].seq), (1, 1));
        silent(&stream.thinking("d", 1_600));
        assert_eq!(stream.flush_thinking(1_700)[0].summary, "d");
        silent(&stream.flush_thinking(1_800));
        silent(&stream.thinking("   ", 2_000));
        silent(&stream.thinking("", 2_100));

        let mut truncating = recorder(16, 10, 10_000);
        truncating.begin_turn("t1", 0);
        let full = truncating.thinking("0123456789abcd", 1);
        assert_eq!(full[0].summary, "0123456...");
        assert_eq!(full[0].summary.chars().count(), 10);
        let exact = truncating.thinking("0123456789", 2);
        assert_eq!(kinds(&exact), vec![ActivityKind::Thinking]);
        assert_eq!(exact[0].summary, "0123456789");
    }

    #[test]
    fn tool_frames_flush_pending_thinking_first() {
        let mut stream = recorder(16, 2000, 10_000);
        stream.begin_turn("task-1", 0);
        silent(&stream.thinking("about to run", 1));
        let frames = stream.tool_call("bash", "ls -la", None, 2);
        assert_eq!(
            kinds(&frames),
            vec![ActivityKind::Thinking, ActivityKind::ToolCall]
        );
        assert_eq!(frames[0].summary, "about to run");
        assert_eq!(frames[1].summary, "bash(ls -la)");
        assert_eq!(frames[1].tool_name.as_deref(), Some("bash"));
        assert!(frames[1].ok.is_none());
        assert_eq!((frames[0].phase_id, frames[1].phase_id), (1, 1));
        let bare = stream.tool_call("bash", "", Some(true), 3);
        assert_eq!((bare[0].summary.as_str(), bare[0].ok), ("bash", Some(true)));
        silent(&stream.thinking("then summarise", 4));
        let flushed = stream.flush_thinking(5);
        assert_eq!(
            (flushed[0].phase_id, flushed[0].summary.as_str()),
            (2, "then summarise")
        );
        let result = stream.tool_result("bash", "done", Some(false), 6);
        assert_eq!(kinds(&result), vec![ActivityKind::ToolResult]);
        assert_eq!(
            (result[0].tool_name.as_deref(), result[0].ok),
            (Some("bash"), Some(false))
        );
        assert_eq!(stream.tool_result("bash", " ", None, 7)[0].summary, "bash");
        let error = stream.error("boom", 8);
        assert_eq!(
            (error[0].kind, error[0].summary.as_str()),
            (ActivityKind::Error, "boom")
        );
        assert!(error[0].ok.is_none() && !error[0].is_terminal());
        assert_eq!(stream.error("", 9)[0].summary, "subagent error");
        assert_eq!(stream.summary().count, 8);
    }

    #[test]
    fn ring_overflow_counts_drops_and_stamps_later_frames() {
        let mut stream = recorder(2, 2000, 0);
        stream.begin_turn("t1", 0);
        assert!(stream.spawn("task", 1)[0].dropped.is_none());
        assert!(stream.running(2)[0].dropped.is_none());
        assert_eq!(stream.dropped(), 0);
        stream.begin_turn("t2", 3);
        let third = stream.spawn("again", 4);
        assert_eq!(stream.dropped(), 1);
        assert_eq!(third[0].dropped, Some(1));
        assert_eq!(seqs(&stream.frames()), vec![2, 3]);
        assert_eq!(seqs(&stream.after(0, 0)), vec![2, 3]);
        assert_eq!(seqs(&stream.after(2, 0)), vec![3]);
        assert_eq!(seqs(&stream.after(2, 1)), vec![3]);
        assert_eq!(seqs(&stream.recent(1)), vec![3]);
        assert_eq!(seqs(&stream.recent(9)), vec![2, 3]);
        silent(&stream.recent(0));
        assert_eq!(stream.clear(), 2);
        assert_eq!(stream.frames(), Vec::new());
        assert_eq!((stream.next_seq(), stream.dropped()), (4, 1));
    }

    #[test]
    fn per_turn_cap_emits_one_truncated_frame_and_suppresses_the_rest() {
        let mut stream = recorder(3, 2000, 0);
        stream.begin_turn("t1", 0);
        assert_eq!(stream.spawn("task", 1).len(), 1);
        assert_eq!(stream.running(2).len(), 1);
        assert_eq!(stream.tool_call("bash", "", None, 3).len(), 1);
        let truncated = stream.idle(4);
        assert_eq!(kinds(&truncated), vec![ActivityKind::Truncated]);
        assert_eq!(truncated[0].dropped, Some(1));
        assert_eq!(
            truncated[0].summary,
            "activity stream truncated for this turn"
        );
        assert!(!truncated[0].is_terminal());
        silent(&stream.thinking("later", 5));
        silent(&stream.tool_call("bash", "", None, 6));
        silent(&stream.spawn("more", 7));
        silent(&stream.flush_thinking(8));
        let summary = stream.summary();
        assert_eq!((summary.count, summary.dropped), (3, 1));
        assert_eq!(summary.last_kind, Some(ActivityKind::Truncated));
        stream.begin_turn("t2", 9);
        let fresh = stream.spawn("again", 10);
        assert_eq!(kinds(&fresh), vec![ActivityKind::Spawn]);
        assert_eq!(fresh[0].task_id, "t2");
    }

    #[test]
    fn terminal_frames_flush_thinking_and_suppress_later_frames() {
        let mut stream = recorder(16, 2000, 10_000);
        stream.begin_turn("t1", 0);
        silent(&stream.thinking("almost done", 1));
        let frames = stream.completed("the answer", 2);
        assert_eq!(
            kinds(&frames),
            vec![ActivityKind::Thinking, ActivityKind::Completed]
        );
        assert_eq!(frames[0].summary, "almost done");
        assert_eq!(frames[1].summary, "the answer");
        assert_eq!(frames[1].ok, Some(true));
        assert!(frames[1].is_terminal() && stream.is_terminal());
        assert!(stream.summary().terminal);
        silent(&stream.spawn("never", 3));
        silent(&stream.running(4));
        silent(&stream.thinking("never", 5));
        silent(&stream.idle(6));
        silent(&stream.flush_thinking(7));
        assert_eq!(
            kinds(&stream.closed("manual", 8)),
            vec![ActivityKind::Closed]
        );

        let mut failing = recorder(16, 2000, 0);
        failing.begin_turn("t1", 0);
        let failed = failing.failed("boom", 1);
        assert_eq!(kinds(&failed), vec![ActivityKind::Failed]);
        assert_eq!(failed[0].ok, Some(false));
        assert!(failed[0].is_terminal());

        let mut evicting = recorder(16, 2000, 0);
        evicting.begin_turn("t1", 0);
        let evicted = evicting.evicted("evicted", 2);
        assert_eq!(kinds(&evicted), vec![ActivityKind::Evicted]);
        assert_eq!(
            (evicted[0].summary.as_str(), evicted[0].ok),
            ("evicted", None)
        );
        assert!(evicting.is_terminal());
    }

    #[test]
    fn sequence_numbers_and_phase_ids_are_monotone_across_turns() {
        let mut stream = recorder(16, 2000, 10_000);
        stream.begin_turn("t1", 0);
        assert_eq!(stream.spawn("task", 1)[0].seq, 1);
        silent(&stream.thinking("phase one", 2));
        assert_eq!(stream.flush_thinking(3)[0].phase_id, 1);
        stream.begin_turn("t2", 100);
        let second = stream.spawn("again", 101);
        assert_eq!((second[0].seq, second[0].task_id.as_str()), (3, "t2"));
        silent(&stream.thinking("phase two", 102));
        assert_eq!(stream.flush_thinking(103)[0].phase_id, 2);
        assert_eq!(stream.next_seq(), 5);
        assert_eq!(seqs(&stream.frames()), vec![1, 2, 3, 4]);
    }

    #[test]
    fn summary_digest_reports_count_losses_and_the_newest_frame() {
        let mut single = recorder(8, 2000, 0);
        assert_eq!(single.summary(), ActivitySummary::default());
        assert_eq!(single.summary().display(), "0 frames");
        single.begin_turn("t1", 0);
        single.spawn("task", 1);
        assert_eq!(single.summary().display(), "1 frame, last: spawn");

        let mut stream = recorder(2, 2000, 0);
        stream.begin_turn("t1", 0);
        stream.spawn("task", 1);
        stream.queued(2);
        stream.begin_turn("t2", 3);
        stream.running(4);
        let summary = stream.summary();
        assert_eq!((summary.count, summary.dropped), (2, 1));
        assert_eq!(
            (summary.last_kind, summary.last_seq),
            (Some(ActivityKind::Running), 3)
        );
        assert!(!summary.terminal);
        assert_eq!(summary.display(), "2 frames, last: running");
    }

    #[test]
    fn subscription_delivers_frames_and_reports_lag_explicitly() {
        let mut stream = recorder(2, 2000, 0);
        let mut subscription = stream.subscribe();
        assert_eq!(
            (stream.receiver_count(), subscription.receiver_count()),
            (1, 1)
        );
        stream.begin_turn("t1", 0);
        let published = stream.spawn("task", 1);
        assert_eq!(
            subscription.try_recv(),
            Some(ActivityEvent::Frame(published[0].clone()))
        );
        assert!(subscription.try_recv().is_none());
        for (turn, at_ms) in [("t2", 2), ("t3", 3), ("t4", 4)] {
            stream.begin_turn(turn, at_ms);
            stream.spawn("task", at_ms);
            stream.running(at_ms);
        }
        assert_eq!(
            subscription.try_recv(),
            Some(ActivityEvent::Lagged { dropped: 4 })
        );
        let survivors = [subscription.try_recv(), subscription.try_recv()]
            .into_iter()
            .filter_map(|event| match event {
                Some(ActivityEvent::Frame(frame)) => Some(frame.seq),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(survivors, vec![6, 7]);
        assert!(subscription.try_recv().is_none());
        drop(stream);
        assert!(subscription.try_recv().is_none());
        assert_eq!(subscription.receiver_count(), 0);
        assert!(format!("{subscription:?}").contains("ActivitySubscription"));
    }

    #[tokio::test]
    async fn into_stream_yields_frames_and_is_send_static() {
        fn assert_send_static<T: Send + 'static>(_: &T) {}

        let mut stream = recorder(8, 2000, 0);
        stream.begin_turn("t1", 0);
        let mut events = stream.subscribe().into_stream();
        assert_send_static(&events);
        stream.spawn("task", 1);
        stream.running(2);
        let mut seen = Vec::new();
        for _ in 0..2 {
            let event = tokio::time::timeout(Duration::from_secs(5), events.next())
                .await
                .expect("an event arrives")
                .expect("an event");
            if let ActivityEvent::Frame(frame) = event {
                seen.push(frame.kind);
            }
        }
        assert_eq!(seen, vec![ActivityKind::Spawn, ActivityKind::Running]);
        drop(stream);
        let ended = tokio::time::timeout(Duration::from_secs(5), events.next())
            .await
            .expect("the feed ends");
        assert!(ended.is_none());
    }

    #[test]
    fn kinds_frames_and_summary_round_trip_through_snake_case_json() {
        for (kind, wire, terminal) in [
            (ActivityKind::Spawn, "spawn", false),
            (ActivityKind::Queued, "queued", false),
            (ActivityKind::Running, "running", false),
            (ActivityKind::Thinking, "thinking", false),
            (ActivityKind::ToolCall, "tool_call", false),
            (ActivityKind::ToolResult, "tool_result", false),
            (ActivityKind::Error, "error", false),
            (ActivityKind::Idle, "idle", false),
            (ActivityKind::Completed, "completed", true),
            (ActivityKind::Failed, "failed", true),
            (ActivityKind::Evicted, "evicted", true),
            (ActivityKind::Closed, "closed", true),
            (ActivityKind::Truncated, "truncated", false),
        ] {
            assert_eq!((kind.as_str(), kind.is_terminal()), (wire, terminal));
            assert_eq!(serde_json::to_value(kind).unwrap(), serde_json::json!(wire));
            assert_eq!(
                serde_json::from_value::<ActivityKind>(serde_json::json!(wire)).unwrap(),
                kind
            );
        }
        assert_eq!(ActivityKind::default(), ActivityKind::Queued);

        let mut stream = recorder(8, 2000, 0);
        stream.begin_turn("t1", 0);
        let mut frames = stream.tool_call("bash", "ls", Some(true), 7);
        let frame = frames.remove(0);
        let value = serde_json::to_value(&frame).unwrap();
        for key in [
            "subagent_id",
            "task_id",
            "seq",
            "kind",
            "summary",
            "tool_name",
            "tool_call_id",
            "ok",
            "at_ms",
            "dropped",
            "phase_id",
        ] {
            assert!(value.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(value["kind"], serde_json::json!("tool_call"));
        assert_eq!(value["tool_name"], serde_json::json!("bash"));
        assert_eq!(value["ok"], serde_json::json!(true));
        let decoded: SubagentActivityFrame = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, frame);
        let partial: SubagentActivityFrame =
            serde_json::from_value(serde_json::json!({"summary": "hi", "extra": 1})).unwrap();
        assert_eq!(
            (partial.summary.as_str(), partial.kind),
            ("hi", ActivityKind::Queued)
        );
        assert_eq!((partial.seq, partial.dropped), (0, None));
        assert!(!partial.is_terminal());

        let summary = stream.summary();
        let value = serde_json::to_value(summary.clone()).unwrap();
        for key in ["count", "dropped", "terminal", "last_kind", "last_seq"] {
            assert!(value.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(value["last_kind"], serde_json::json!("tool_call"));
        assert_eq!(
            serde_json::from_value::<ActivitySummary>(value).unwrap(),
            summary
        );
        let empty: ActivitySummary = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(empty, ActivitySummary::default());
    }
}
