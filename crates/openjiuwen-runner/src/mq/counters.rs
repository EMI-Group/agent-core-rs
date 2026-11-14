//! The monotonic counters [`InMemoryMessageQueue`](crate::mq::InMemoryMessageQueue)
//! maintains and reports through [`QueueStats`](crate::mq::QueueStats).
//!
//! They live here rather than in `memory.rs` so the queue's topic-table logic and
//! its bookkeeping stay separately readable; nothing else in the crate touches
//! them.

use std::sync::atomic::{AtomicU64, Ordering};

/// The counters every topic worker and the queue itself increment.
///
/// They are shared behind one `Arc`, so a worker updates the queue's own report
/// without touching the topic table (and therefore without a lock).
#[derive(Default)]
pub(super) struct Counters {
    /// Deliveries a worker actually started (a handler was present).
    pub(super) deliveries: AtomicU64,
    /// Messages dropped because their topic had no worker.
    pub(super) dropped_no_worker: AtomicU64,
    /// Messages dropped because their topic had no handler.
    pub(super) dropped_no_handler: AtomicU64,
    /// Handler invocations that returned an error.
    pub(super) handler_errors: AtomicU64,
    /// Operations refused because the queue was stopped.
    pub(super) rejected_not_started: AtomicU64,
}

impl Counters {
    /// Records one increment of `counter`.
    pub(super) fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Reads `counter`.
    pub(super) fn read(counter: &AtomicU64) -> u64 {
        counter.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;

    #[test]
    fn counters_start_at_zero_and_are_monotonic() {
        let counters = Counters::default();
        for counter in [
            &counters.deliveries,
            &counters.dropped_no_worker,
            &counters.dropped_no_handler,
            &counters.handler_errors,
            &counters.rejected_not_started,
        ] {
            assert_eq!(Counters::read(counter), 0);
        }

        Counters::bump(&counters.deliveries);
        Counters::bump(&counters.deliveries);
        Counters::bump(&counters.dropped_no_handler);
        assert_eq!(Counters::read(&counters.deliveries), 2);
        assert_eq!(Counters::read(&counters.dropped_no_handler), 1);
        assert_eq!(Counters::read(&counters.dropped_no_worker), 0);
        assert_eq!(Counters::read(&counters.handler_errors), 0);
        assert_eq!(Counters::read(&counters.rejected_not_started), 0);
    }
}
