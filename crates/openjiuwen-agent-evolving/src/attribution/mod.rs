//! Attribution: turning a run's outcome into per-node credit.
//!
//! Ports the offline part of Python `openjiuwen.agent_evolving.attribution`: the
//! diagnostic-signal records (`attribution/model.py`), the dependency graph and
//! closure (`attribution/nodes.py`, `attribution/closure.py`), the deterministic
//! keyword detector (`attribution/keyword_detect.py`) and the reward-stage adapter
//! (`attribution/reward_stage/`).
//!
//! - [`AttributionGraph`] — nodes, edges and the transitive closure algorithms.
//! - [`CreditAssigner`] — backward propagation of a run's reward to the steps that
//!   caused it (decay, normalisation, deterministic top contributors).
//! - [`DiagnosticSignal`] / [`SignalSource`] — the Python-shaped records.
//! - [`KeywordAttributor`] — deterministic keyword-to-target matching.
//!
//! The LLM-driven diagnosis agent and its prompt packs are deferred; every
//! algorithm here runs without a model.

mod credit;
mod graph;
mod keyword;
mod model;
mod reward_stage;

pub use credit::{CreditAssigner, CreditRank, CreditReport};
pub use graph::{AttributionEdge, AttributionGraph, AttributionNode};
pub use keyword::{KeywordAttributor, KeywordMatch, KeywordRule};
pub use model::{reward_tier, DiagnosticSignal, SignalSource, SIGNAL_TIERS};
pub use reward_stage::RewardStageAdapter;
