//! Offline "dreaming": clustering, merging and pruning of the tip bank.
//!
//! Ports the deterministic part of Python `openjiuwen.agent_evolving.ttse.dream`:
//! near-duplicate tips are clustered by keyword overlap and collapsed onto a
//! representative, then stale or negligible tips are pruned. No model call is
//! involved in the offline path.

/// Clusters, merges and prunes the tips of a [`TipBank`](super::TipBank).
pub struct Dreamer;
