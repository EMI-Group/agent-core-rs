# tests

## Intent
Offline integration suites for the crate's public API: one binary per module area, 22 in total, plus the inline unit tests beside each module.
Nothing here touches a network, the environment, or wall-clock time.

## Constraints
- Deterministic and offline: fakes implement `CaseRunner`/`TrainableTarget`/`Policy`/`RolloutEngine`/`CommandRunner`/`Agent` locally.
- `tempfile` supplies checkpoint, bundle and memory-store directories; `openjiuwen-test-support` (`ScriptedModel`, fixtures, the `wiremock` re-export) is used wherever a model or HTTP path is exercised.
- Service tests drive `build_router` / `build_router_with_state` with `tower::ServiceExt::oneshot` instead of binding a port.
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
- `./service.rs` → `/health`, `/v1/evaluate`, validation errors, and the online RL route table's status/code envelopes
- `./online_service.rs` → task lifecycle with header validation and 201-vs-200 replay, reward ranges, capture before/after/discard, delayed-feedback turn gating, finish/abort reasons, upload→claim→training run→checkpoint, trajectory read model, health 200 vs 503, error envelopes
- `./rails.rs` → event translation from `HarnessRailContext`, spec defaults, observer error propagation, `registerable_rail`
- `./signal.rs` → signal validation, trackers (dedupe/decay/eviction), every extractor, excerpt bounding
- `./reward.rs` → dimensions, normalisation variants, missing-dimension policies, weighted aggregation, statistics, packs
- `./attribution.rs` → graph validation/traversal/closure, credit assignment (decay, weights, depth, normalisation, ranking), keyword attributor, reward-stage round trip
- `./trajectory.rs` → builder and message projection, span pairing, windows, digests, store grouping and statistics
- `./experience.rs` → records, archive dedupe/merge/ranking, extractor, lifecycle promote/decay/prune, rebuild
- `./sharing.rs` → keyword extraction, deterministic bundle ids, stager gates, both local backends and their ranking
- `./ttse.rs` → config, tip schema and ids, bank dedupe/prune, classifier, success detection, parser, induction, dreaming, consultation, engine persistence
- `./rl.rs` → packing, reward normalisation, policy/rollout seams, trainer loop, checkpoints, optimizer wiring
- `./hub_remote.rs` → remote hub protocol v1 over loopback `wiremock`: bearer auth, publish/get/remove/list/search, 404 mapping, error envelope, retry, approval-gated stager publish, `ftp://` rejection
- `./experience_drafting.rs` → model reply tolerance and defaults, repair retry, deterministic fallback, draft schema validation, orchestrator status matrix, auto-approval persistence
- `./ttse_embeddings.rs` → embedding cold fill, cosine ranking, retrieval-mode fallbacks, index persistence and mismatch rejection, scripted merge/keep-distinct/malformed dreaming
- `./reward_checks.rs` → read/python checks, browser and picture `Unsupported`, evidence gathering and judging over `ScriptedModel`, floor cap, unknown claims, garbage verdicts, reward-pack aggregation
