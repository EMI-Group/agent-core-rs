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
