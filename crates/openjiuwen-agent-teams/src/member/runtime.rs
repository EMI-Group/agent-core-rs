//! The member "brain" seam: [`MemberRuntime`], its observer, and the two observer helpers.
//!
//! Python parity source: `openjiuwen/agent_teams/agent/member_runtime.py`. Python declares a
//! `Protocol` because `NativeHarness` *and* the external-CLI runtimes both satisfy it
//! structurally; Rust needs an object-safe trait because [`crate::member::TeamMember`] stores one
//! `Arc<dyn MemberRuntime>` per member and hands it to the coordination layer.
//!
//! The surface mirrors the harness-protocol interaction set (`start` / `stop` / `send` / `abort` /
//! `pause` / `resume` / `subscribe` / `state` / `session_id`) narrowed to what the in-process
//! member lifecycle actually drives. One runtime instance serves one run cycle: started per
//! coordination `start`, torn down at `finalize_round`. Rail/memory/background-controller hooks
//! from Python are intentionally absent — the harness adapter owns them and nothing above this
//! seam needs them (documented deviation in `src/member/CONTEXT.md`).
//!
//! ```
//! use std::sync::Arc;
//! use openjiuwen_agent_teams::member::{MemberObserver, ObserverEvent, RecordingObserver};
//!
//! let observer = Arc::new(RecordingObserver::new());
//! observer.on_round(openjiuwen_agent_teams::schema::RoundOutcome::Finished);
//! assert_eq!(
//!     observer.take(),
//!     vec![ObserverEvent::Round(openjiuwen_agent_teams::schema::RoundOutcome::Finished)]
//! );
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_harness_protocol::HarnessState;
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::Value;
use tracing::trace;

use crate::schema::RoundOutcome;

/// Observer of one member runtime's phase changes, rounds and output chunks.
///
/// The coordination layer registers one observer per member; the observer maps phases and round
/// outcomes onto [`crate::schema::MemberStatus`] / [`crate::schema::ExecutionStatus`]. A sink that
/// panics or blocks is the runtime's problem, not a reason to fail the round — implementations log
/// and continue.
pub trait MemberObserver: Send + Sync {
    /// Called whenever the runtime's lifecycle phase changes.
    fn on_state(&self, state: HarnessState);

    /// Called when a round starts, finishes, is aborted or fails.
    fn on_round(&self, outcome: RoundOutcome);

    /// Called for every output chunk the member produces.
    fn on_chunk(&self, chunk: Value);
}

/// The brain a team member's coordination layer drives.
///
/// `state`/`session_id` are synchronous snapshots so status decisions never need an `await`; every
/// mutating call is async and fallible. `start` is the only call that receives the team session —
/// a member derives its own child session from it and persists its harness state beneath it.
#[async_trait]
pub trait MemberRuntime: Send + Sync {
    /// Returns the runtime's session id, or `""` before the first `start`.
    fn session_id(&self) -> &str;

    /// Returns the current lifecycle phase.
    fn state(&self) -> HarnessState;

    /// Starts the runtime's round lifecycle for one run cycle.
    ///
    /// `team_session` is the team-level session the member derives its own session from; `None`
    /// means the member runs against its default session (tests, standalone use).
    ///
    /// # Errors
    ///
    /// Returns an error when the harness cannot be materialised (missing model, bad checkpoint,
    /// already running). The caller rolls the member's status back to `unstarted`.
    async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()>;

    /// Stops the runtime, cancels in-flight work and closes the output stream.
    ///
    /// # Errors
    ///
    /// Returns an error when teardown fails; the member's row keeps whatever status it had.
    async fn stop(&self) -> Result<()>;

    /// Submits input for the next round; `immediate` steers the round in flight.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtime has been stopped or the harness rejects the input.
    async fn send(&self, content: String, immediate: bool) -> Result<()>;

    /// Aborts the in-flight round: graceful when `immediate` is `false`, hard with rollback when
    /// `true`.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtime has no round to abort and cannot record the request.
    async fn abort(&self, immediate: bool) -> Result<()>;

    /// Pauses the in-flight round; the next [`Self::send`] continues it.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtime cannot enter the paused phase.
    async fn pause(&self) -> Result<()>;

    /// Resumes a warm round or restores a cold checkpoint, optionally with a `query`.
    ///
    /// # Errors
    ///
    /// Returns an error when there is nothing to resume or the restore fails.
    async fn resume(&self, query: Option<String>) -> Result<()>;

    /// Registers an observer; a runtime may hold any number of them.
    fn subscribe(&self, observer: Arc<dyn MemberObserver>);
}

/// One event captured by [`RecordingObserver`].
#[derive(Debug, Clone, PartialEq)]
pub enum ObserverEvent {
    /// A lifecycle phase change.
    State(HarnessState),
    /// A round outcome.
    Round(RoundOutcome),
    /// An output chunk.
    Chunk(Value),
}

/// An observer that records every callback so a test (or a diagnostic rail) can replay them.
///
/// Recording is lock-per-event and lock-free of `.await`, so it is safe to use from an async
/// runtime. [`RecordingObserver::take`] drains the buffer, which keeps assertions readable.
#[derive(Debug, Default)]
pub struct RecordingObserver {
    events: Mutex<Vec<ObserverEvent>>,
}

impl RecordingObserver {
    /// Creates an empty recording observer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drains and returns every event recorded so far, oldest first.
    pub fn take(&self) -> Vec<ObserverEvent> {
        std::mem::take(&mut *self.events.lock())
    }

    /// Returns the number of buffered events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.lock().len()
    }

    /// Returns `true` when nothing has been recorded since the last [`Self::take`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }
}

impl MemberObserver for RecordingObserver {
    fn on_state(&self, state: HarnessState) {
        self.events.lock().push(ObserverEvent::State(state));
    }

    fn on_round(&self, outcome: RoundOutcome) {
        self.events.lock().push(ObserverEvent::Round(outcome));
    }

    fn on_chunk(&self, chunk: Value) {
        self.events.lock().push(ObserverEvent::Chunk(chunk));
    }
}

/// A thread-safe set of observers a runtime fans every callback out to.
///
/// The fan-out clones the observer list before invoking anyone, so an observer that registers a
/// sibling (or drops one) never deadlocks the runtime. A registered observer is never removed for
/// the lifetime of the set; runtimes are per-cycle objects, so that is enough.
#[derive(Default)]
pub struct ObserverSet {
    observers: Mutex<Vec<Arc<dyn MemberObserver>>>,
}

impl ObserverSet {
    /// Creates an empty observer set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an observer.
    pub fn add(&self, observer: Arc<dyn MemberObserver>) {
        self.observers.lock().push(observer);
    }

    /// Returns the number of registered observers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.observers.lock().len()
    }

    /// Returns `true` when no observer is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.observers.lock().is_empty()
    }

    /// Forwards a phase change to every observer.
    pub fn state(&self, state: HarnessState) {
        for observer in self.snapshot() {
            observer.on_state(state);
        }
    }

    /// Forwards a round outcome to every observer.
    pub fn round(&self, outcome: RoundOutcome) {
        for observer in self.snapshot() {
            observer.on_round(outcome);
        }
    }

    /// Forwards an output chunk to every observer.
    pub fn chunk(&self, chunk: Value) {
        let observers = self.snapshot();
        let Some((last, rest)) = observers.split_last() else {
            return;
        };
        for observer in rest {
            observer.on_chunk(chunk.clone());
        }
        last.on_chunk(chunk);
    }

    /// Clones the observer list so callbacks run without holding the lock.
    fn snapshot(&self) -> Vec<Arc<dyn MemberObserver>> {
        let observers = self.observers.lock();
        trace!(count = observers.len(), "fanning out to member observers");
        observers.clone()
    }
}

impl std::fmt::Debug for ObserverSet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObserverSet")
            .field("observers", &self.len())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn recording_observer_keeps_order_and_drains() {
        let observer = RecordingObserver::new();
        observer.on_state(HarnessState::Running);
        observer.on_round(RoundOutcome::Started);
        observer.on_chunk(json!({"delta": "hi"}));
        assert_eq!(observer.len(), 3);
        assert!(!observer.is_empty());

        let events = observer.take();
        assert_eq!(
            events,
            vec![
                ObserverEvent::State(HarnessState::Running),
                ObserverEvent::Round(RoundOutcome::Started),
                ObserverEvent::Chunk(json!({"delta": "hi"})),
            ]
        );
        assert_eq!(observer.take(), Vec::new());
    }

    #[test]
    fn observer_set_fans_out_and_reports_its_size() {
        let set = ObserverSet::new();
        assert!(set.is_empty());
        let first = Arc::new(RecordingObserver::new());
        let second = Arc::new(RecordingObserver::new());
        set.add(first.clone());
        set.add(second.clone());
        assert_eq!(set.len(), 2);

        set.state(HarnessState::Paused);
        set.round(RoundOutcome::Aborted);
        set.chunk(json!(1));

        for observer in [&first, &second] {
            assert_eq!(
                observer.take(),
                vec![
                    ObserverEvent::State(HarnessState::Paused),
                    ObserverEvent::Round(RoundOutcome::Aborted),
                    ObserverEvent::Chunk(json!(1)),
                ]
            );
        }
    }

    #[test]
    fn observer_set_debug_reports_the_count() {
        let set = ObserverSet::new();
        set.add(Arc::new(RecordingObserver::new()));
        assert!(format!("{set:?}").contains("observers: 1"));
    }
}
