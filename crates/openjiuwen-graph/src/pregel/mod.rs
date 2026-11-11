//! Crate-private Pregel/BSP executor internals.
//!
//! - [`channel`] — CNF barrier channels, the accept-then-activate buffer, and
//!   channel snapshots.
//! - [`dispatch`] — compilation of static edges and conditional routers into a
//!   target resolver.
//! - [`executor`] — the superstep loop: scheduling, merging, retries, timeouts,
//!   cancellation, and checkpointing.

pub(crate) mod channel;
pub(crate) mod dispatch;
pub(crate) mod executor;
