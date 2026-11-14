//! Reward records, aggregation, normalisation, and model-based reward checks.
//!
//! Ports the offline part of Python `openjiuwen.agent_evolving.reward`: the
//! scoring records (`reward/models.py`), the aggregation rules
//! (`reward/scoring.py`), the reward-pack description (`reward/rewardpack.py`),
//! the evidence tools (`reward/checks.py`), the evidence agent
//! (`reward/evidence_agent.py`) and the judged scoring layer (`reward/scoring.py`
//! and the judge stage of `reward/pipeline.py`).
//!
//! - `RewardSignal` / `RewardDimension` — one measured dimension of quality.
//! - `RewardAggregator` — weighted aggregation with explicit normalisation and
//!   missing-dimension policy, deterministic and unit-testable.
//! - `RewardSummary` / `RewardPack` — the persisted results and the pack that
//!   describes which dimensions a run scores.
//! - `RewardCheck` / `CheckRunner` — the `read` and `python` checks a model asks
//!   for; browser and picture checks are deferred with a typed error.
//! - `EvidenceAgent` — the model-driven loop that gathers check evidence.
//! - `ModelRewardJudge` — per-criterion judging, the rubric arithmetic
//!   (`score_criterion`, `FLOOR_FAIL_CAP`) and `to_reward_signal`, which feeds a
//!   judged report into the existing pack.
//!
//! The records and the arithmetic are pure data; only `EvidenceAgent` and
//! `ModelRewardJudge` call a model, and both take it through the shared `Model`
//! seam so the offline tests can script it.
mod check;
mod evidence;
mod model_check;
mod models;
mod pack;
mod scoring;

#[cfg(test)]
mod end_to_end;

pub use check::{
    inside, CheckOutcome, CheckRunner, CheckTool, RewardCheck, DEFAULT_LISTING_LIMIT,
    DEFAULT_MAX_HITS, DEFAULT_MAX_OUTPUT_CHARS, DEFAULT_RANGE_COUNT, IMAGE_SUFFIXES,
    MAX_RANGE_COUNT, PAGE_SUFFIXES, PROXY_SINK, PYTHON_TIMEOUT_MS,
};
pub use evidence::{
    CriterionKind, EvidenceAgent, EvidenceClaim, EvidenceCriterion, EvidenceGathered, EvidenceTask,
    GatheredCheck, DEFAULT_MAX_RESULT_CHARS, DEFAULT_ROUNDS, EVIDENCE_AGENT_PROMPT,
};
pub use model_check::{
    score_criterion, to_reward_signal, total, CheckRewardReport, CriterionReport, CriterionStatus,
    ModelRewardJudge, FLOOR_FAIL_CAP, FLOOR_WEIGHT, JUDGE_PROMPT, REWARD_CHECK_DIMENSION,
    REWARD_CHECK_SOURCE, SCHEMA_VERSION,
};
pub use models::{DimensionScore, RewardDimension, RewardSignal, RewardSummary};
pub use pack::{default_pack, reward_from_score, score_evaluated_cases, RewardPack};
pub use scoring::{
    aggregate_scores, MissingDimensionPolicy, RewardAggregator, ScoreNormalisation, ScoreStats,
};
