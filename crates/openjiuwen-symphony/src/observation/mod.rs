//! Graph observation: caller-supplied evidence distilled into derived graph revisions.
//!
//! Ports Python `openjiuwen.symphony.observation` (v0.1.20): the canonical
//! `GraphEvolutionInput` contract, an append-only evidence log, immutable observation and
//! merged revision manifests whose ids are content hashes, and the overlay of
//! `runtime_weight` / `binding_status` that only the planner reads.
//!
//! The static `can_feed` capability graph stays authoritative; nothing here rewrites it.

pub mod contracts;
pub mod identity;
pub mod store;
pub(crate) mod timestamp;
