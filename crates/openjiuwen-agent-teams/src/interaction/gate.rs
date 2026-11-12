//! The run/interact concurrency gate.
//!
//! Python parity source: `openjiuwen/agent_teams/runtime/gate.py` (v0.1.20), reshaped for this
//! crate's synchronous dispatch seam.
//!
//! One [`InteractGate`] belongs to one active team run cycle. While the run is active, external
//! interact payloads are admitted and queued in FIFO order; when the run is about to exit the
//! gate is closed, so further payloads are rejected and the run cycle can drain whatever is
//! still queued.
//!
//! State machine:
//!
//! ```text
//! OPEN   --admit(payload)--> OPEN (queue tail)
//! OPEN   --close()--------> CLOSED
//! CLOSED --admit()--------> rejected (gate_closed)
//! CLOSED --try_admit()----> rejected (gate_closed)
//! *      --drain()--------> queue emptied FIFO (allowed after close)
//! ```
//!
//! Python's gate counts *in-flight* payloads with an `AdmissionTicket` and awaits
//! `close_and_drain`; this port keeps the observable decisions (`gate_closed`, `not_active`)
//! but models admission as a queue, because the Rust run loop pulls payloads itself instead of
//! holding a ticket across an await. The gate is synchronous and lock-guarded, so it is safe to
//! share across tasks.

use std::collections::VecDeque;

use parking_lot::Mutex;

use openjiuwen_common::error::{Error, Result};

use super::payload::{DeliverResult, InteractPayload};

/// Gate state guarded by one mutex so a check-and-enqueue is atomic.
#[derive(Debug, Default)]
struct GateState {
    /// Whether the gate stopped admitting payloads.
    closed: bool,
    /// Admitted payloads awaiting consumption, oldest first.
    queue: VecDeque<InteractPayload>,
}

/// Admission control for one active team run cycle.
///
/// Cheap to construct and safe to share (`Send + Sync`); all methods take `&self`.
#[derive(Debug, Default)]
pub struct InteractGate {
    /// Closed flag plus the FIFO queue.
    state: Mutex<GateState>,
}

impl InteractGate {
    /// Builds an open gate with an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Admits one payload by appending it to the queue.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the gate is already closed. Callers that need the
    /// stable [`DeliverResult`] token use [`InteractGate::dispatch`] instead.
    pub fn admit(&self, payload: InteractPayload) -> Result<()> {
        let mut state = self.state.lock();
        if state.closed {
            return Err(gate_closed_error());
        }
        state.queue.push_back(payload);
        Ok(())
    }

    /// Pops the oldest queued payload, or returns `None` when the queue is empty.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the gate is closed; a closed gate never hands out
    /// payloads, even ones queued before it closed (use [`InteractGate::drain`] for those).
    pub fn try_admit(&self) -> Result<Option<InteractPayload>> {
        let mut state = self.state.lock();
        if state.closed {
            return Err(gate_closed_error());
        }
        Ok(state.queue.pop_front())
    }

    /// Removes every queued payload, preserving FIFO order.
    ///
    /// Draining is allowed after the gate closes so the exiting run cycle can finish the work
    /// it already admitted.
    pub fn drain(&self) -> Vec<InteractPayload> {
        let mut state = self.state.lock();
        state.queue.drain(..).collect()
    }

    /// Closes the gate to new admissions; idempotent.
    ///
    /// Queued payloads are left in place for [`InteractGate::drain`].
    pub fn close(&self) {
        self.state.lock().closed = true;
    }

    /// Whether the gate has been closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.state.lock().closed
    }

    /// Number of payloads waiting in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.lock().queue.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.lock().queue.is_empty()
    }

    /// The run/interact entry point: decides and queues in one atomic step.
    ///
    /// - Gate closed → [`DeliverResult::gate_closed`], regardless of `active`.
    /// - Team not active → [`DeliverResult::not_active`].
    /// - Otherwise the payload is queued and the result is a success with no message id (the
    ///   run loop, not the gate, mints the message when it delivers the payload).
    #[must_use]
    pub fn dispatch(&self, payload: InteractPayload, active: bool) -> DeliverResult {
        let mut state = self.state.lock();
        if state.closed {
            return DeliverResult::gate_closed();
        }
        if !active {
            return DeliverResult::not_active();
        }
        state.queue.push_back(payload);
        DeliverResult::ok(None)
    }
}

/// The error returned by [`InteractGate::admit`] and [`InteractGate::try_admit`] once closed.
fn gate_closed_error() -> Error {
    Error::validation("interact gate is closed")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::interaction::{GodViewMessage, OperatorMessage};

    fn god_view(body: &str) -> InteractPayload {
        InteractPayload::GodView(GodViewMessage {
            body: body.to_owned(),
            sender: None,
        })
    }

    fn operator(target: &str, body: &str) -> InteractPayload {
        InteractPayload::Operator(OperatorMessage {
            target: Some(target.to_owned()),
            body: body.to_owned(),
            sender: None,
        })
    }

    #[test]
    fn admit_and_try_admit_preserve_fifo_order() {
        let gate = InteractGate::new();
        gate.admit(god_view("first")).unwrap();
        gate.admit(operator("dev-1", "second")).unwrap();
        assert_eq!(gate.len(), 2);
        assert!(!gate.is_empty());

        assert_eq!(gate.try_admit().unwrap(), Some(god_view("first")));
        assert_eq!(gate.try_admit().unwrap(), Some(operator("dev-1", "second")));
        assert_eq!(gate.try_admit().unwrap(), None);
        assert!(gate.is_empty());
    }

    #[test]
    fn drain_empties_the_queue_in_order() {
        let gate = InteractGate::new();
        gate.admit(god_view("a")).unwrap();
        gate.admit(god_view("b")).unwrap();
        assert_eq!(gate.drain(), vec![god_view("a"), god_view("b")]);
        assert_eq!(gate.len(), 0);
        assert_eq!(gate.drain().len(), 0);
    }

    #[test]
    fn dispatch_rejects_when_team_is_not_active() {
        let gate = InteractGate::new();
        let result = gate.dispatch(god_view("hello"), false);
        assert!(!result.ok);
        assert_eq!(result.reason.as_deref(), Some("not_active"));
        assert_eq!(gate.len(), 0);
    }

    #[test]
    fn dispatch_queues_and_succeeds_when_active() {
        let gate = InteractGate::new();
        let result = gate.dispatch(god_view("hello"), true);
        assert!(result.ok);
        assert_eq!(result.message_id, None);
        assert_eq!(gate.len(), 1);
        assert_eq!(gate.try_admit().unwrap(), Some(god_view("hello")));
    }

    #[test]
    fn closed_gate_rejects_admit_and_try_admit() {
        let gate = InteractGate::new();
        gate.admit(god_view("queued")).unwrap();
        gate.close();
        assert!(gate.is_closed());

        assert!(gate.admit(god_view("later")).is_err());
        assert!(gate.try_admit().is_err());
    }

    #[test]
    fn closed_gate_returns_gate_closed_even_when_inactive() {
        let gate = InteractGate::new();
        gate.close();
        let result = gate.dispatch(god_view("hello"), false);
        assert!(!result.ok);
        assert_eq!(result.reason.as_deref(), Some("gate_closed"));
    }

    #[test]
    fn queued_payloads_survive_close_until_drained() {
        let gate = InteractGate::new();
        assert!(gate.dispatch(god_view("hello"), true).ok);
        gate.close();

        let drained = gate.drain();
        assert_eq!(drained, vec![god_view("hello")]);
        assert_eq!(gate.drain(), Vec::<InteractPayload>::new());

        let result = gate.dispatch(god_view("after"), true);
        assert_eq!(result.reason.as_deref(), Some("gate_closed"));
    }

    #[test]
    fn close_is_idempotent_and_queue_len_stays_stable() {
        let gate = InteractGate::new();
        gate.close();
        gate.close();
        assert!(gate.is_closed());
        assert_eq!(gate.len(), 0);
        assert!(gate.is_empty());
    }
}
