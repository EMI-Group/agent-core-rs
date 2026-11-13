# tests

## Intent
Offline integration suites for the crate's public API: one binary per module area, 17 in total, plus the inline unit tests beside each module.
Nothing here touches a network, the environment, or wall-clock time.

## Constraints
- Deterministic and offline: fakes implement `CaseRunner`/`TrainableTarget`/`Policy`/`RolloutEngine`/`CommandRunner`/`Agent` locally.
- `tempfile` supplies checkpoint, bundle and memory-store directories; `openjiuwen-test-support` (`ScriptedModel`, fixtures) is used wherever a model path is exercised.
- Service tests drive `build_router` with `tower::ServiceExt::oneshot` instead of binding a port.
- Pipeline tests never spawn `docker` or `git`: every side effect goes through `FakeCommandRunner`.
- Each binary carries `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`.

## Routing Table
- `./dataset.rs` → JSON/JSONL parsing, async file loading, validation, seeded shuffle/split, serde
- `./evaluator.rs` → exact match, metric aggregation, batch validation, reports, `MapCaseRunner`, `AgentCaseRunner`, `evaluate_cases`
- `./judge.rs` → rubric validation, tolerant verdict parsing, judge prompt construction, `LlmJudge` retry policy over `ScriptedModel`, `LlmJudgeEvaluator` fallback, optional judge in `DefaultEvaluator`
- `./checkpoint.rs` → grid/parameter space, best-so-far, file and memory stores, manager save policy, resume
- `./optimizer_research.rs` → the six textual-gradient optimizers, backward gradients, routing per target/tool, Metis utility order, TTSE clustering, model-driven fallback
- `./trainer.rs` → baseline/epochs, early stop, checkpoint saves on improvement, resume, cancellation, budgets, candidate exhaustion, callback order
- `./pipeline.rs` → Docker/git command lines, stop-on-failure cleanup, missing-`docker` typed failure, workspace preparation, every score policy, batch summary, serde
- `./service.rs` → `/health`, `/v1/evaluate`, validation errors, 501 error envelope on RL routes
- `./rails.rs` → event translation from `HarnessRailContext`, spec defaults, observer error propagation, `registerable_rail`
- `./signal.rs` → signal validation, trackers (dedupe/decay/eviction), every extractor, excerpt bounding
- `./reward.rs` → dimensions, normalisation variants, missing-dimension policies, weighted aggregation, statistics, packs
- `./attribution.rs` → graph validation/traversal/closure, credit assignment (decay, weights, depth, normalisation, ranking), keyword attributor, reward-stage round trip
- `./trajectory.rs` → builder and message projection, span pairing, windows, digests, store grouping and statistics
- `./experience.rs` → records, archive dedupe/merge/ranking, extractor, lifecycle promote/decay/prune, rebuild
- `./sharing.rs` → keyword extraction, deterministic bundle ids, stager gates, both backends and their ranking
- `./ttse.rs` → config, tip schema and ids, bank dedupe/prune, classifier, success detection, parser, induction, dreaming, consultation, engine persistence
- `./rl.rs` → packing, reward normalisation, policy/rollout seams, trainer loop, checkpoints, optimizer wiring
