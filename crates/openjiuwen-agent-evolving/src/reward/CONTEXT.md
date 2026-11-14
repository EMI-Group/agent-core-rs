# reward

## Intent
Reward records, normalisation, missing-dimension policy, weighted aggregation, and the reward pack a run is scored against.
Ports the offline part of Python `openjiuwen.agent_evolving.reward`: `reward/models.py`, `reward/scoring.py`, `reward/rewardpack.py`, `reward/checks.py`, `reward/evidence_agent.py` and the judge stage of `reward/pipeline.py`.
The records and the arithmetic are pure data; the model-based checks are the only paths that call a model (`EvidenceAgent`, `ModelRewardJudge`), both through the shared `Model` seam so tests can script them.
Everything here is deterministic offline: no container runtime, no browser, no wall clock.

## API Surface
- `RewardDimension { name, weight, max_score, higher_is_better }` — `new`, `with_weight`, `with_max_score`, `with_higher_is_better`, `validate`.
- `RewardSignal { dimension, value, source, evidence, confidence }` — one measured dimension value; `validate` enforces a declared name and a finite value.
- `DimensionScore`, `RewardSummary` — the persisted per-dimension results and the aggregate total.
- `ScoreNormalisation` — `Raw` (divide by `max_score`, unclamped), `Clamp` (the default, clamped to `0.0..=1.0`) and `MinMax { min, max }` (rescaled from a declared range); `apply` never yields `NaN` or infinity.
- `MissingDimensionPolicy` — `Zero` (count an unmeasured dimension as `0.0`), `Skip` (the default: leave it out and list it in `RewardSummary::missing`) or `Fail` (`Error::validation` naming every missing dimension).
- `RewardAggregator` — `new`, `with_normalisation`, `with_policy`, `with_threshold`, `aggregate(signals)`; duplicate signals for one dimension combine by confidence, undeclared dimensions are counted but ignored.
- `RewardPack` — `name`, `version`, `dimensions`, `passing_score`, `validate`, `aggregator`, `aggregate`; `default_pack` weights correctness `0.6`, efficiency `0.2`, clarity `0.2` with a passing score of `0.6`.
- `RewardSummary::from_signals(signals, per_dimension, missing, threshold)` — builds a summary directly; `RewardSummary::zero()`/`Default` is the empty summary.
- `RewardAggregator` defaults: `ScoreNormalisation::Clamp`, `MissingDimensionPolicy::Skip`, threshold `0.6`; duplicate dimension names keep the last occurrence; it derives `Debug + Clone + PartialEq` only (deliberately not serde).
- `reward_from_score`, `score_evaluated_cases`, `aggregate_scores`/`ScoreStats` — score-to-signal and batch-statistics helpers.
- `CheckTool { Read, Python, Browser }` — serde `read|python|browser`; `as_str` is the Python spelling.
- `RewardCheck { tool, path, code, pattern, count, page, actions, criteria, note }` — one check; `infer_tool()` applies Python's `TOOL_BY_FIELD` rule (`actions`/`page` → browser, `code` → python, `path` → read) and `dedupe_key()` is canonical JSON of the check without `criteria`/`note`, with the tool filled in.
- `CheckOutcome { ok, result, screenshots, skipped, error }` — constructors `ok`, `skipped`, `failed`, plus `succeeded()`.
- `CheckRunner { root, runner, max_output_chars (6000), max_hits (60), listing_limit (200) }` — `new(root, runner)`, the three `with_*` limit setters, `root()`, `async run(&RewardCheck) -> Result<CheckOutcome>`.
- `inside(root, relative)` — lexical containment check: the target must equal `root` or sit under it; `Error::validation` otherwise.
- `CheckRunner` constants: `DEFAULT_MAX_OUTPUT_CHARS` `6000`, `DEFAULT_MAX_HITS` `60`, `DEFAULT_LISTING_LIMIT` `200`, `PYTHON_TIMEOUT_MS` `60000`, `PROXY_SINK` `http://127.0.0.1:9`, `DEFAULT_RANGE_COUNT` `200`, `MAX_RANGE_COUNT` `400`, `PAGE_SUFFIXES`, `IMAGE_SUFFIXES`.
- `EvidenceAgent { model, request, max_rounds (4), max_result_chars (7500) }` — `new`, `with_max_rounds`, `with_max_result_chars`, `async gather(&EvidenceTask, &CheckRunner) -> Result<EvidenceGathered>`.
- `EVIDENCE_AGENT_PROMPT` — the agent's instructions with documented placeholders `{task}`, `{criteria}`, `{workspace_files}`, `{inputs_excerpt}`, `{final_reply}`, `{rounds}`.
- `EvidenceTask { task, workspace_files, inputs_excerpt, final_reply, criteria }`, `EvidenceCriterion { id, name, kind, claims }`, `EvidenceClaim { index, text }`, `CriterionKind { Floor, ProactiveChallenge }` (serde `floor|proactive_challenge`), each with `new`/`with_*` builders.
- `GatheredCheck { id, tool, criteria, ok, repeat, result, screenshots, skipped, error }` and `EvidenceGathered { checks, rounds, deduped }` with `checks_for(criterion)`, `distinct()`, `GatheredCheck::usable()`.
- `CriterionStatus { Met, Partial, Unmet, Unknown }` — `value()` is `Some(1.0)`, `Some(0.5)`, `Some(0.0)`, `None`; `parse` tolerates case and quotes.
- `score_criterion(kind, &[CriterionStatus]) -> Option<u32>`, `total(&[Option<u32>]) -> Option<u32>`, `FLOOR_FAIL_CAP` `25`, `FLOOR_WEIGHT` `0.5`, `SCHEMA_VERSION` `"reward/3"`.
- `CriterionReport { id, name, kind, rubric, status, score, claims }` (claims are `(text, status, reason)` triples) and `CheckRewardReport { total, criteria, summary, used_checks }` with `criterion(id)`.
- `ModelRewardJudge { model, request }` — `new`, `model()`, `request()`, `async judge(&EvidenceTask, &EvidenceGathered, final_reply) -> Result<CheckRewardReport>`; `JUDGE_PROMPT` is the per-criterion prompt.
- `to_reward_signal(&CheckRewardReport) -> Option<RewardSignal>` — dimension `REWARD_CHECK_DIMENSION` (`"correctness"`), value `total / 100`, source `REWARD_CHECK_SOURCE` (`"reward_check"`), evidence the report summary.
- No `RewardFn`/`RewardRegistry` here: those live in `crate::rl` (`rl::policy`, `rl::task`); model-scored values reach this module as `EvaluatedCase::score` or as a judged `CheckRewardReport`.

## Rubric arithmetic
- A claim status is `met` `1.0`, `partial` `0.5`, `unmet` `0.0`; `unknown` is left out of the arithmetic entirely.
- A criterion with no settled claim (all unknown, or no claim) is unscored: `score_criterion` returns `None`.
- The criterion score is `round(100 * mean(settled claims))`; a criterion carries one `CriterionKind`, so Python's `FLOOR_WEIGHT : 1 - FLOOR_WEIGHT` blend of floor and challenge means reduces to the single populated group's mean.
- Any `unmet` claim of a `Floor` criterion caps the score at `FLOOR_FAIL_CAP` (`25`); `ProactiveChallenge` criteria are never capped.
- The report total is the mean of the non-`None` criterion scores, and `None` when no criterion was scored.
- Scores round half away from zero (`f64::round`, so `12.5 → 13`), where Python's `round` is half-to-even (`12.5 → 12`); the difference is at most one point on an exact half.

## Check execution
- `read` with `path` resolves it against the workspace root lexically (a symlink inside the workspace can still point outside it, unlike Python's `Path.resolve`); a folder becomes a sorted listing capped at `listing_limit`, a file becomes UTF-8 text (invalid UTF-8 becomes `[binary file]`).
- `read` with a `pattern` matches case-insensitively as a substring (no regular-expression dependency), reports at most `max_hits` hits, shows whole lines only and names a cut when the output cap truncates them; Python's `context` knob (0–5 lines) is not part of this reduced check model, so context is always `0`.
- `read` without a pattern shows `count` lines from line 1 (clamped `1..=400`) and reports `lines: 1-N of M`.
- `read` without a `path`, and a check that names no resolvable tool, are `CheckOutcome::skipped`, not errors.
- `python` writes the script into a unique scratch directory outside the workspace (`std::env::temp_dir()`, process id plus a counter, removed afterwards), runs `python3 -I <script>` (or `python`) through `CommandRunner` with `cwd` = the workspace so the check can measure the work, `TMPDIR`/`TEMP`/`TMP` = the scratch directory, `HTTP_PROXY`/`HTTPS_PROXY` = `PROXY_SINK`, `PYTHONIOENCODING=utf-8`, `PYTHONDONTWRITEBYTECODE=1` and the 60 s timeout.
- The `python` result keeps 6000 stdout characters and 3000 stderr characters; a non-zero exit or a timeout sets `error` and `ok = false` instead of discarding the output.
- A missing interpreter is `Error::Unsupported` naming `python3`/`python`; there is no `GUARD` audit hook in this port, so a script runs unguarded inside its scratch directory.

## Deferred surface
- `browser` checks return `Error::Unsupported` with the message `browser checks are deferred: they need playwright (headless browser automation), which this port does not bundle`.
- Reading a picture (`png`, `jpg`, `jpeg`, `webp`, `gif`, `bmp`) returns `Error::Unsupported` with the message `reading <path> as a picture is deferred: it needs PIL (Pillow) image processing, which this port does not bundle`.
- Inside `EvidenceAgent::gather` such a deferred check is recorded on its `GatheredCheck` (`error` names playwright or PIL) instead of aborting the run, mirroring Python's `run_one` catch so the model can plan around it.
- Not ported from Python's evidence agent: the harness `run_checks`/`read_evidence` tool loop, saved-result paging, idle stopping, restart after a dropped stream, the default screenshot walk, screenshot-carrying messages and image judging.
- Not ported from Python's pipeline: the criteria writer, feedback placement, the art judge, `cap_by_critique`, `claim_gains` and the per-run result document.

## Constraints
- All arithmetic is deterministic and stated: weighted mean of the declared dimensions, `0.0..=1.0` totals under the default normalisation, no silent `NaN` propagation.
- Validation happens before arithmetic (`RewardDimension::validate`, `RewardSignal::validate`), so a malformed dimension fails with `Error::validation` instead of skewing a total.
- Field names are the Python snake_case names and unknown JSON fields are tolerated.
- A judge call that fails or answers prose leaves every claim `unknown` with the failure as its reason; it never panics and never invents a score, and `to_reward_signal` then returns `None`.
- `ModelRewardJudge::judge` returns `Error::validation` for a task with no criterion, and `EvidenceAgent::gather` propagates the model's own error.
- Every side effect of a check goes through `CommandRunner`, so the whole module is testable offline with `FakeCommandRunner`, `tempfile` and `ScriptedModel`.
- Files stay under ~1000 lines: `check.rs` carries the read/python tools, `evidence.rs` the gathering loop, `model_check.rs` the arithmetic and the judge.

## Design Decisions
- Signals are supplied rather than extracted from model replies, which is what makes the scoring path testable offline.
- A pack is a serde record rather than Python's on-disk directory layout, so a pack can travel with a run's results in one JSON document.
- Packs and ad-hoc aggregators share one implementation: a pack builds a `RewardAggregator` instead of duplicating the weighted mean.
- The default pack divides the Python `0..=100` dimension values down to `0..=1`, matching the scale `openjiuwen_agent_evolving::attribution::reward_tier` and `EvaluatedCase::score` use.
- The check model is a reduced record of the fields the offline tools use; `context`, `start` and `snapshots` are dropped, so a check the model writes for a deferred tool is still parsed and reported rather than rejected.
- The evidence agent talks to the model in plain JSON messages (`{"checks": [...]}`) instead of the harness tool loop, which keeps it offline and bounded by `max_rounds`.
- An unparsable model reply becomes an empty check batch and ends the run, so a confused model cannot loop or crash the reward.
- `score_criterion` takes one `CriterionKind` for the whole criterion because this model records Python's per-claim level per criterion; the floor/challenge weighting is kept in the code so the port reads like the Python source.
- The judged layer maps onto the existing pack through `to_reward_signal` (`correctness` in `0.0..=1.0`) instead of adding a second aggregation path.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./models.rs` → `RewardDimension`, `RewardSignal`, `DimensionScore`, `RewardSummary`
- `./scoring.rs` → `ScoreNormalisation`, `MissingDimensionPolicy`, `RewardAggregator`, `ScoreStats`
- `./pack.rs` → `RewardPack`, `default_pack`, `reward_from_score`, `score_evaluated_cases`
- `./check.rs` → `CheckTool`, `RewardCheck`, `CheckOutcome`, `CheckRunner`, `inside` and the read/python tools
- `./evidence.rs` → `EvidenceAgent`, `EvidenceTask`, `EvidenceCriterion`, `EvidenceClaim`, `CriterionKind`, `GatheredCheck`, `EvidenceGathered`, `EVIDENCE_AGENT_PROMPT`
- `./model_check.rs` → `CriterionStatus`, `score_criterion`, `total`, `CriterionReport`, `CheckRewardReport`, `ModelRewardJudge`, `to_reward_signal`, `JUDGE_PROMPT`
- `./end_to_end.rs` → test-only end-to-end scenarios for gathering, judging, the deferred tools and the report → signal → pack round trip
