//! Reward records, aggregation, and normalisation.
//!
//! Ports the offline part of Python `openjiuwen.agent_evolving.reward`: the
//! scoring records (`reward/models.py`), the aggregation rules
//! (`reward/scoring.py`) and the reward-pack description (`reward/rewardpack.py`).
//!
//! - [`RewardSignal`] / [`RewardDimension`] — one measured dimension of quality.
//! - [`RewardAggregator`] — weighted aggregation with explicit normalisation and
//!   missing-dimension policy, deterministic and unit-testable.
//! - [`RewardSummary`] / [`RewardPack`] — the persisted results and the pack that
//!   describes which dimensions a run scores.
//!
//! Everything here is pure data plus arithmetic: no model calls, no container
//! runtime, no wall clock.

mod models;
mod pack;
mod scoring;

pub use models::{DimensionScore, RewardDimension, RewardSignal, RewardSummary};
pub use pack::{default_pack, reward_from_score, score_evaluated_cases, RewardPack};
pub use scoring::{
    aggregate_scores, MissingDimensionPolicy, RewardAggregator, ScoreNormalisation, ScoreStats,
};
