# updater

## Intent
The single conversion point from evolution evidence to applied parameter updates: update contracts, the apply helpers, the single/multi-dimension updaters, and the RL parameter-update pipeline.
Ports Python `openjiuwen/agent_evolving/updater/` (protocol, `single_dim`, `multi_dim`), `agent_evolving/types.py`, `agent_evolving/update_execution.py`, and the portable half of the RL update loop the crate previously listed as unported.
No persistence, no approval, no commit: `execute_updates` is the last write and the caller owns whatever store is behind the target.

## API Surface
- Contracts (`types.rs`) — `UpdateKey` (`(operator_id, target)`), `UpdateMode` (`replace`/`append`/`merge`), `UpdateEffect` (`state`/`pending_change`), `UpdateDomain` (`llm`/`tool`/`memory`), `LifecycleStage` (`local_apply_completed`), `UpdateValue`, `Update` (`Raw`/`Structured`), `UpdateEntry`, `Updates` (insertion-ordered map), `ApplyResult` (`ok()`), `ApplySummary` (`is_ok()`), `normalize_update_value`, `normalize_updates`, `EXPERIENCES_TARGET`, `SKILL_EXPERIENCE_ENTRY`.
- Apply (`apply.rs`) — `UpdateTarget` (`set_parameter`, `domain` defaulting to `llm`), `OperatorMap` (`BTreeMap<String, Arc<dyn UpdateTarget>>`), `execute_updates`, `apply_updates`, `summarize_apply_results`, `RecordingTarget` (test double: `writes`, `written_targets`, `value`, `with_rejected`).
- Routing (`updater.rs`) — `Updater` (`bind`, `requires_forward_data`, `process`, `update`, `get_state`, `load_state`), `UpdateConfig` (`score_threshold`, `extra`), `signals_from_cases`, `TextualOptimizer` (`bind`, `add_trajectory`, `backward`, `step`, `requires_forward_data`, `state`, `load_state`) plus its `Box<T>` forwarding impl, `SingleDimUpdater<O>`, `MultiDimUpdater`, `domain_for_signal`, `RecordingOptimizer` (test double: `bound`, `bound_targets`, `trajectory_ids`, `backward_calls`, `signals`, `steps`, `with_updates`, `with_requires_forward_data`).
- RL (`rl_update.rs`) — `AdvantageReport`, `AdvantageGroupStats`, `compute_advantages`, `GradientOptimizer`, `OfflinePolicyOptimizer`, `GpuGradientOptimizer`, `RlUpdatePipeline` (`new`, `with_packer`, `with_run_id`, `run_id`, `epoch`, `optimizer`, `requires_gpu`, `apply`, `save_checkpoint`), `ParameterUpdate`, `POLICY_TARGET` (`"policy"`), `policy_update_key`.

## Constraints
- `execute_updates` is two-pass, exactly like Python: every non-`None` value first in map order, then every `None`-valued key as a failure (`errors = ["update value is None"]`). A missing operator yields `errors = ["operator not found: …"]`; nothing is dropped.
- The seam is replace-only, mirroring Python's `Operator.apply_update` compatibility default: `replace` + `state` reaches `set_parameter`, any other mode/effect pair is rejected with Python's own message (`unsupported update mode/effect for compatibility operator: …`). `experiences` updates therefore normalize correctly but fail on this seam by design.
- `TextualOptimizer` has no default for `bind`/`add_trajectory`/`backward`/`step`/`state`/`load_state`; a `backward` before `bind` is `Error::validation` ("requires bind first"), not a silent success.
- `Updater::update` is a provided method over `process` and is the only place `score_threshold` is read; the comparison is inclusive (`score >= threshold` is skipped) and `None` means "no filter". A non-finite threshold is `Error::validation` instead of silently filtering everything.
- `bind` returning `0` is the Python soft-exit value, never an error.
- `MultiDimUpdater` order: domains in `UpdateDomain::ALL` order (the sorted `BTreeMap` key order), each domain sees the trajectories in input order, and operators/targets keep the optimizer's write order. A domain without an attributed signal is neither fed nor stepped; a signal whose domain has no optimizer is `Error::validation` before any domain steps; a key produced by two domains is a `Updates::merge` validation error.
- No clock, no randomness, no environment, no network, no new dependency.
- No `unsafe`; `-D warnings` clippy (pedantic) clean; files stay far below the size limit.

## Design Decisions
- `Updates` is an insertion-ordered `Vec<UpdateEntry>`, not a `BTreeMap`: Python dicts preserve insertion order, a `(String, String)` tuple key cannot be a JSON object key, and serde_json refuses to serialize tuple-keyed maps. The ordered list gives Python's iteration order and a serializable array shape (`[{operator_id, target, value}]`).
- `Update::Structured` vs `Update::Raw` replaces Python's `isinstance(UpdateValue)` check with a typed distinction; deserialization detects a structured value by its `payload` member because `#[serde(untagged)]` would turn every JSON object into an empty `UpdateValue` (all of its fields are optional).
- `UpdateTarget` is one write, not Python's `apply_update`: mode/effect interpretation stays in `execute_updates`, so every target gets the same compatibility gate and the trait stays implementable by a plain parameter map.
- `TextualOptimizer` is the crate's own `BaseOptimizer` seam: the existing `optimizer::research::TextualGradientOptimizer` is a *proposal* engine (it yields candidates, not update maps), so it is not adapted here; `RecordingOptimizer` is the deterministic double used by tests, docs and examples.
- `signals_from_cases` reuses `signal::signals_from_evaluated_cases` instead of re-deriving signals, so the offline path keeps one signal shape (and inherits that extractor's `MAX_SIGNALS_PER_INPUT` cap and its `source = "evaluator"` marker).
- `MultiDimUpdater` routes signals by `EvolutionTarget` (`tool` → tool domain, `memory` → memory domain, everything else → llm) rather than porting Python's placeholder selector, since Python ships no concrete `MultiDimUpdater` subclass at all.
- `compute_advantages` replaces each rollout's reward with its advantage (the value GRPO feeds the policy update) and keeps the raw rewards in the report, so the transformation is reversible for reporting; that mutation is also why it takes `&mut [RolloutWithReward]`.
- Ungrouped rollouts go through `rl::RewardNormaliser` (`GroupCentered`) rather than a bespoke formula: it is the existing scalar normalisation, it keeps one lone ungrouped sample at advantage `0.0` instead of an error, and it reuses the crate's finite-reward sanitisation.
- `RlUpdatePipeline::apply` packs with `rl::BatchPacker` for validation and truncation, then flattens the chunks back into one gradient batch: one gradient pass is one optimizer step, and splitting the input would silently multiply the steps.
- A failed apply is `Error::workflow` naming the count and the first failure, so a partially applied batch never looks like success; a failed step does not advance the epoch.
- `GpuGradientOptimizer::gradients` is the only caller of `crate::deferred::unsupported`, giving the deferred table a real dispatch path and naming `torch`/`verl`/`ray` in the message.

## Test Strategy
- Inline unit tests beside each module (`types.rs` normalisation/ordering/serde, `apply.rs` two-pass order and failure modes, `updater.rs` single/multi-dim routing and state, `rl_update.rs` advantage arithmetic) — all offline.
- `tests/updater.rs` drives the public surface end to end: apply ordering and `None`-last, single-dim `process`/`update` with an inclusive `score_threshold`, `Box<dyn TextualOptimizer>` forwarding, state round-trip, multi-dim domain order/attribution/collisions, group-relative advantages, the `OfflinePolicyOptimizer` batch order and written values, checkpoint contents plus `save_checkpoint`/`RlCheckpoint::load_from_path`, the `None`-valued failure path, an empty batch, and the typed `Unsupported` from `GpuGradientOptimizer`.
- No model, no network, no wall clock: the policy is a local `Policy` double that records its batches.

## Routing Table
- `./types.rs` → update contracts, ordering and normalisation
- `./apply.rs` → `UpdateTarget`, `execute_updates`, summaries, `RecordingTarget`
- `./updater.rs` → `Updater`, `TextualOptimizer`, single/multi-dimension updaters, `RecordingOptimizer`
- `./rl_update.rs` → advantages, gradient seam, `RlUpdatePipeline`, `ParameterUpdate`
