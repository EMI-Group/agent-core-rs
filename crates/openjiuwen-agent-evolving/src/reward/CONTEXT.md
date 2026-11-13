# reward

## Intent
Reward records, normalisation, missing-dimension policy, weighted aggregation, and the reward pack a run is scored against.
Ports the offline part of Python `openjiuwen.agent_evolving.reward`: `reward/models.py`, `reward/scoring.py` and `reward/rewardpack.py`.
Everything here is pure data plus arithmetic: no model calls, no container runtime, no wall clock.

## API Surface
- `RewardDimension { name, weight, max_score, higher_is_better }` — `new`, `with_weight`, `with_max_score`, `with_higher_is_better`, `validate`.
- `RewardSignal { dimension, value, source, evidence, confidence }` — one measured dimension value; `validate` enforces a declared name and a finite value.
- `DimensionScore`, `RewardSummary` — the persisted per-dimension results and the aggregate total.
- `ScoreNormalisation` — `Raw` (divide by `max_score`, unclamped), `Clamp` (the default, clamped to `0.0..=1.0`) and `MinMax { min, max }` (rescaled from a declared range); `apply` never yields `NaN` or infinity.
- `MissingDimensionPolicy` — `Zero` (count an unmeasured dimension as `0.0`), `Skip` (the default: leave it out and list it in `RewardSummary::missing`) or `Fail` (`Error::validation` naming every missing dimension).
- `RewardAggregator` — `new`, `with_normalisation`, `with_policy`, `with_threshold`, `aggregate(signals)`; duplicate signals for one dimension combine by confidence, undeclared dimensions are counted but ignored.
- `RewardPack` — `name`, `version`, `dimensions`, `passing_score`, `validate`, `aggregator`; `default_pack` weights correctness `0.6`, efficiency `0.2`, clarity `0.2` with a passing score of `0.6`.
- `reward_from_score`, `score_evaluated_cases`, `aggregate_scores`/`ScoreStats` — score-to-signal and batch-statistics helpers.

## Constraints
- All arithmetic is deterministic and stated: weighted mean of the declared dimensions, `0.0..=1.0` totals under the default normalisation, no silent `NaN` propagation.
- Validation happens before arithmetic (`RewardDimension::validate`, `RewardSignal::validate`), so a malformed dimension fails with `Error::validation` instead of skewing a total.
- Field names are the Python snake_case names and unknown JSON fields are tolerated.

## Design Decisions
- Signals are supplied rather than extracted from model replies, which is what makes the scoring path testable offline.
- A pack is a serde record rather than Python's on-disk directory layout, so a pack can travel with a run's results in one JSON document.
- Packs and ad-hoc aggregators share one implementation: a pack builds a `RewardAggregator` instead of duplicating the weighted mean.
- The default pack divides the Python `0..=100` dimension values down to `0..=1`, matching the scale `openjiuwen_agent_evolving::attribution::reward_tier` and `EvaluatedCase::score` use.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./models.rs` → `RewardDimension`, `RewardSignal`, `DimensionScore`, `RewardSummary`
- `./scoring.rs` → `ScoreNormalisation`, `MissingDimensionPolicy`, `RewardAggregator`, `ScoreStats`
- `./pack.rs` → `RewardPack`, `default_pack`, `reward_from_score`, `score_evaluated_cases`
