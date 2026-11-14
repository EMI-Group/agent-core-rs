//! The incremental skill-notice session layer over [`crate::discovery::notice`].
//!
//! Ports Python `openjiuwen.symphony.discovery.toolkit`: a per-session-scope tool id,
//! card normalization, the bounded notice-cursor registry and the
//! [`IncrementalNoticeSession`] that only acknowledges a reminder once it fits the
//! delivered output budget.
