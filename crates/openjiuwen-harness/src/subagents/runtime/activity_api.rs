//! Read, subscribe, and append access to the per-instance sub-agent activity streams.
//!
//! Mirrors the transcript accessors of [`super::transcript_api`] one level up: the runtime owns
//! one [`ActivityStream`] per instance, the lifecycle transitions publish their frames, and the
//! accessors here let a monitoring caller read the bounded ring, subscribe to the live feed, or
//! record progress/tool events it observes outside the runtime (the child's agent/rails layer).
//!
//! Every reader takes the runtime state lock briefly and is synchronous; a subscription is an
//! [`ActivitySubscription`], a `tokio::sync::broadcast::Receiver` that yields the frames published
//! *after* the call, while the retained ring is the way to read the history that predates it.
//! A runtime built with [`SubAgentQuota::enable_activity_stream`](super::SubAgentQuota::enable_activity_stream)
//! disabled reports an empty history and hands out feeds that never yield a frame; the stream
//! itself reports `is_enabled() == false`, so a caller can tell "recording disabled" apart from
//! "no activity yet".

use futures::stream::BoxStream;
use openjiuwen_common::error::Result;

use super::activity::{
    ActivityEvent, ActivityKind, ActivitySubscription, ActivitySummary, SubagentActivityFrame,
};
use super::{not_found, SubAgentRuntime};

impl SubAgentRuntime {
    /// Returns the retained activity frames of `id`, oldest first; `None` for unknown ids.
    ///
    /// The vector holds at most the ring capacity
    /// ([`SubAgentQuota::activity_capacity`](super::SubAgentQuota::activity_capacity)) frames;
    /// [`ActivitySummary::dropped`] reports how many older frames were evicted. Closed instances
    /// keep their frames (the tombstone survives), so this still answers after
    /// [`SubAgentRuntime::close`].
    #[must_use]
    pub fn activity(&self, id: &str) -> Option<Vec<SubagentActivityFrame>> {
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.activity_stream.frames())
    }

    /// Returns the activity frames after `cursor` (exclusive), oldest first, at most `limit`.
    ///
    /// `cursor` is the `seq` of the newest frame the caller has already seen, so `0` asks for the
    /// whole retained history; `limit == 0` means "no limit". `None` for unknown ids.
    #[must_use]
    pub fn activity_after(
        &self,
        id: &str,
        cursor: u64,
        limit: usize,
    ) -> Option<Vec<SubagentActivityFrame>> {
        let limit = if limit == 0 { usize::MAX } else { limit };
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.activity_stream.after(cursor, limit))
    }

    /// Returns the compact digest of `id`'s activity stream; `None` for unknown ids.
    ///
    /// A disabled stream (see
    /// [`SubAgentQuota::enable_activity_stream`](super::SubAgentQuota::enable_activity_stream))
    /// reports a zero count and `terminal == false`, because nothing was recorded.
    #[must_use]
    pub fn activity_summary(&self, id: &str) -> Option<ActivitySummary> {
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.activity_stream.summary())
    }

    /// Subscribes to the live activity feed of `id`; `None` for unknown ids.
    ///
    /// The subscription yields every frame published after the call and reports a lag instead of
    /// blocking the publisher. A runtime with activity recording disabled yields no frame.
    #[must_use]
    pub fn subscribe_activity(&self, id: &str) -> Option<ActivitySubscription> {
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.activity_stream.subscribe())
    }

    /// Turns the live activity feed of `id` into a `Send + 'static` event stream.
    ///
    /// The stream yields the frames published after the call in order and ends once the instance's
    /// activity stream is dropped; `None` for unknown ids.
    #[must_use]
    pub fn activity_stream(&self, id: &str) -> Option<BoxStream<'static, ActivityEvent>> {
        self.subscribe_activity(id)
            .map(ActivitySubscription::into_stream)
    }

    /// Records a caller-supplied activity frame of `kind` on `id`.
    ///
    /// The runtime owns the envelope: `subagent_id`, `task_id` (the current turn), `seq` and
    /// `at_ms` (the injected clock) are overwritten. Returns the sequence number of the recorded
    /// frame, or `Ok(None)` when the frame was suppressed — the stream is terminal, the current
    /// turn hit its frame cap, or activity recording is disabled. Terminal kinds
    /// ([`ActivityKind::is_terminal`]) end the stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// (`subagent not found: <id>`) for unknown ids.
    pub fn record_activity(
        &self,
        id: &str,
        kind: ActivityKind,
        summary: &str,
        tool_name: Option<&str>,
        ok: Option<bool>,
    ) -> Result<Option<u64>> {
        let now_ms = self.clock.now_ms();
        let mut state = self.shared.state.lock();
        let Some(instance) = state.instances.get_mut(id) else {
            return Err(not_found(id));
        };
        let frames = instance
            .activity_stream
            .record(kind, summary, tool_name, ok, now_ms);
        Ok(frames.last().map(|frame| frame.seq))
    }
}
