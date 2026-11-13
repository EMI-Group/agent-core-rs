//! Read, subscribe, and append access to the per-instance sub-agent transcripts.
//!
//! The runtime observes whole `Agent::invoke` turns only, so it records the lifecycle events it
//! can see itself: `spawn` on reservation, `input` on every query it accepts, `status` on every
//! phase change, `completion`/`error` for a finished turn, and `closed`/`evicted` when an
//! instance is closed or recycled.
//! Tool calls inside a child agent are invisible at this level, so
//! [`SubAgentRuntime::record_tool_call`] and [`SubAgentRuntime::record_transcript`] are the
//! honest seam for them: the [`SubAgentFactory`](super::SubAgentFactory) or the child's rails
//! layer appends what it observes, and the entry lands in the same ring buffer and live feed as
//! the runtime's own events.
//!
//! Every accessor takes the runtime state lock briefly and is synchronous. A subscription is a
//! `tokio::sync::broadcast::Receiver` that yields the entries published *after* the call; the
//! retained ring is the way to read the history that predates it.

use openjiuwen_common::error::Result;

use super::transcript::{TranscriptEntry, TranscriptSubscription, TranscriptSummary};
use super::{not_found, SubAgentRuntime};

impl SubAgentRuntime {
    /// Returns the transcript entries of `id`, oldest first; `None` for unknown ids.
    ///
    /// The vector holds at most the ring capacity
    /// ([`SubAgentQuota::transcript_capacity`](super::SubAgentQuota::transcript_capacity))
    /// entries; older ones were dropped, as reported by
    /// [`TranscriptSummary::dropped`](super::transcript::TranscriptSummary::dropped).
    /// Closed instances keep their transcript (the tombstone survives), so this still answers
    /// after [`SubAgentRuntime::close`].
    #[must_use]
    pub fn transcript(&self, id: &str) -> Option<Vec<TranscriptEntry>> {
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.transcript.entries())
    }

    /// Returns the entries after `cursor` (exclusive), oldest first, at most `limit` of them.
    ///
    /// `cursor` is the `seq` of the newest entry the caller has already seen, so `0` asks for the
    /// whole retained history; `limit == 0` means "no limit". `None` for unknown ids.
    #[must_use]
    pub fn transcript_after(
        &self,
        id: &str,
        cursor: u64,
        limit: usize,
    ) -> Option<Vec<TranscriptEntry>> {
        let limit = if limit == 0 { usize::MAX } else { limit };
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.transcript.after(cursor, limit))
    }

    /// Returns the compact digest of `id`'s transcript; `None` for unknown ids.
    ///
    /// The same digest is reported as `transcript` on [`SubagentMetadata`](super::SubagentMetadata)
    /// and [`SubAgentOutcome`](super::SubAgentOutcome).
    #[must_use]
    pub fn transcript_summary(&self, id: &str) -> Option<TranscriptSummary> {
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.transcript.summary())
    }

    /// Subscribes to the live transcript feed of `id`; `None` for unknown ids.
    ///
    /// The receiver yields every entry appended after the call, runtime events and
    /// [`SubAgentRuntime::record_transcript`] entries alike. A lagging receiver gets
    /// `RecvError::Lagged` instead of blocking the publisher.
    #[must_use]
    pub fn subscribe_transcript(&self, id: &str) -> Option<TranscriptSubscription> {
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.channel.subscribe())
    }

    /// Appends a tool-call event observed outside the runtime (the child's agent/rails layer).
    ///
    /// Returns the sequence number assigned to the entry. The entry is stamped with the injected
    /// clock and published on [`SubAgentRuntime::subscribe_transcript`], so a monitoring caller
    /// cannot tell runtime-recorded events from these.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// (`subagent not found: <id>`) for unknown ids.
    pub fn record_tool_call(
        &self,
        id: &str,
        tool_name: &str,
        detail: &str,
        success: Option<bool>,
    ) -> Result<u64> {
        self.append_transcript(id, |now_ms| {
            TranscriptEntry::tool_call(id, tool_name, detail, success, now_ms)
        })
    }

    /// Appends a caller-supplied entry to `id`'s transcript.
    ///
    /// The runtime owns the envelope: `subagent_id`, `seq` and `at_ms` are overwritten (the
    /// sequence number is assigned by the ring, the timestamp comes from the injected clock), so
    /// a caller only decides the kind, role, content and outcome fields. The entry is published
    /// on [`SubAgentRuntime::subscribe_transcript`] like any other.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// (`subagent not found: <id>`) for unknown ids.
    pub fn record_transcript(&self, id: &str, entry: TranscriptEntry) -> Result<u64> {
        self.append_transcript(id, move |_now_ms| entry)
    }

    /// Drops every buffered transcript entry of `id` and returns how many were removed.
    ///
    /// The sequence counter keeps counting, so a cursor held by a caller still works; `None` for
    /// unknown ids. The instance's [`SubAgentRuntime::transcript_summary`] then reports
    /// `count = 0`.
    #[must_use]
    pub fn clear_transcript(&self, id: &str) -> Option<usize> {
        let mut state = self.shared.state.lock();
        state
            .instances
            .get_mut(id)
            .map(|instance| instance.transcript.clear())
    }

    /// Stamps the entry built by `entry` with the runtime clock and appends it to `id`.
    fn append_transcript(
        &self,
        id: &str,
        entry: impl FnOnce(i64) -> TranscriptEntry,
    ) -> Result<u64> {
        let now_ms = self.clock.now_ms();
        let mut state = self.shared.state.lock();
        let Some(instance) = state.instances.get_mut(id) else {
            return Err(not_found(id));
        };
        let mut entry = entry(now_ms);
        id.clone_into(&mut entry.subagent_id);
        entry.at_ms = now_ms;
        Ok(instance.record(entry))
    }
}
