//! The member observer that feeds the team: status, rounds, chunks and lifecycle signals.
//!
//! One observer is registered per member runtime (leader at build, teammates at spawn). It does
//! three things, in this order of importance:
//!
//! 1. forwards the harness state onto [`TeamMember::mark`] and the activity registry, so the roster
//!    projection agrees with the harness without the leader polling;
//! 2. forwards round outcomes onto [`TeamMember::note_round`] and the observability/monitor sinks,
//!    counting rounds per member and timing them for turn metrics;
//! 3. fans `llm_output` chunks out to every live [`TeamAgent::subscribe_chunks`] receiver.
//!
//! The state writes are spawned: the [`MemberObserver`] callbacks are synchronous while the store
//! writes are async, and a slow store must never block the harness. Dropping a member handle (the
//! [`Weak`] upgrade fails) simply stops the roster writes.
//!
//! [`TeamAgent::subscribe_chunks`]: super::team_agent::TeamAgent::subscribe_chunks
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use openjiuwen_harness_protocol::HarnessState;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, warn};

use crate::harness::state_to_member_status;
use crate::member::{MemberActivityRegistry, MemberObserver, TeamMember};
use crate::schema::{RoundOutcome, TeamRole};

use super::observability_bind::TeamSignals;

/// Forwards one member runtime's phases, rounds and chunks into the team.
pub(super) struct MemberStreamObserver {
    /// Leader-side activity projection.
    activity: Arc<MemberActivityRegistry>,
    /// Member whose observations this instance carries.
    member_name: String,
    /// Role the member plays (carried onto observability events).
    role: TeamRole,
    /// Weak handle to the member row, so marking never extends its lifetime.
    member: Weak<TeamMember>,
    /// Team-wide observability and monitor sinks.
    signals: TeamSignals,
    /// Chunk fan-out of the handle that registered this observer.
    chunk_sinks: Arc<Mutex<Vec<UnboundedSender<Value>>>>,
    /// Rounds observed so far (`started` increments, terminal outcomes close the current one).
    rounds: AtomicU32,
    /// When the current round started, for the turn-metrics duration.
    round_started_at: Mutex<Option<Instant>>,
}
impl MemberStreamObserver {
    /// Builds the observer for `member`.
    pub(super) fn new(
        activity: Arc<MemberActivityRegistry>,
        member: Weak<TeamMember>,
        role: TeamRole,
        member_name: String,
        signals: TeamSignals,
        chunk_sinks: Arc<Mutex<Vec<UnboundedSender<Value>>>>,
    ) -> Self {
        Self {
            activity,
            member_name,
            role,
            member,
            signals,
            chunk_sinks,
            rounds: AtomicU32::new(0),
            round_started_at: Mutex::new(None),
        }
    }
    /// Returns the member name this observer carries.
    #[allow(dead_code)]
    pub(super) fn member_name(&self) -> &str {
        &self.member_name
    }
    /// Applies a status write without blocking the harness callback.
    fn mark(&self, status: crate::schema::MemberStatus) {
        let Some(member) = self.member.upgrade() else {
            return;
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            debug!(
                member = %self.member_name,
                status = status.as_str(),
                "no tokio runtime; skipping roster status write"
            );
            return;
        };
        let name = self.member_name.clone();
        handle.spawn(async move {
            if let Err(error) = member.mark(status).await {
                warn!(member = %name, %error, "failed to mark member status from the harness");
            }
        });
    }
    /// Applies a round note without blocking the harness callback.
    fn note_round(&self, outcome: RoundOutcome) {
        let Some(member) = self.member.upgrade() else {
            return;
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let name = self.member_name.clone();
        handle.spawn(async move {
            if let Err(error) = member.note_round(outcome).await {
                warn!(member = %name, %error, "failed to apply a round note");
            }
        });
    }
}
impl MemberObserver for MemberStreamObserver {
    /// Marks the member busy while a round runs and idle otherwise, then writes the status.
    fn on_state(&self, state: HarnessState) {
        match state {
            HarnessState::Running | HarnessState::Pausing => {
                self.activity.mark_busy(&self.member_name);
            }
            _ => self.activity.mark_idle(&self.member_name),
        }
        if let Some(status) = state_to_member_status(state) {
            self.signals
                .member_status(&self.member_name, self.role, status);
            self.mark(status);
        }
    }
    /// Counts and times rounds, signals them and mirrors them onto the execution machine.
    fn on_round(&self, outcome: RoundOutcome) {
        if matches!(outcome, RoundOutcome::Started) {
            let round = self.rounds.fetch_add(1, Ordering::SeqCst) + 1;
            *self.round_started_at.lock() = Some(Instant::now());
            self.signals.round_started(&self.member_name, round);
            return;
        }
        let round = self.rounds.load(Ordering::SeqCst).max(1);
        let elapsed = self.round_started_at.lock().take().map_or(0, |start| {
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
        });
        debug!(
            member = %self.member_name,
            round,
            outcome = ?outcome,
            "member round finished"
        );
        self.signals
            .round_finished(&self.member_name, round, outcome, elapsed);
        self.note_round(outcome);
    }
    /// Fans one chunk out to every live subscriber, dropping dead ones.
    fn on_chunk(&self, chunk: Value) {
        let mut sinks = self.chunk_sinks.lock();
        sinks.retain(|sink| sink.send(chunk.clone()).is_ok());
    }
}
