# src/online

## Intent

The online-RL service state machines, ported from Python `openjiuwen/agent_evolving/agent_rl/`.
Four cooperating machines live here: the RL task/capture registry (`task_registry.py` + the registry-owned halves of `capture_pipeline.py`), the sample ledger (`storage/trajectory_store.py`), the training-run runner (`online/training_runner.py`, with the executor seam of `online/abstract/trainer.py` + `online/scheduler/ppo_executor.py`) and the LoRA artifact repository (`storage/lora_repo.py`).
Every machine is store-agnostic: the in-memory store used offline and the future Redis store share one implementation of every rule.
What stays deferred is typed: the GPU PPO training path and the vendor AI-gateway activation path return `Error::Unsupported` with the missing capability named.
The HTTP routes and the Redis backend are later steps; nothing here reads the wall clock, a random number or a filesystem mtime.

## API Surface

- `mod.rs` — module docs (key layout table, state machines, determinism rules), `mod` declarations, re-exports, and the `json` encode/decode helpers used by the persisted records.
- `error.rs` — `OnlineError` (`TaskNotFound`, `TaskConflict`, `TurnClosed`, `InvalidRequest`, `InsufficientSamples`, `TrainingRunNotFound`, `Execution`, `Unsupported`, `Store`), `code()`, `http_status()`, `Result<T, OnlineError>`, `From<OnlineError> for Error` and the seam-side `From<Error> for OnlineError`.
- `store.rs` — `OnlineStore` seam (`get`, `set`, `set_if_absent`, `del`, `set_add`, `set_remove`, `set_members`, plus the defaulted `set_contains`), `MemoryOnlineStore` (`new`, `Default`, `keys`, `snapshot`), the `keys` namespace module, and the `Arc<S>` / `&S` delegation impls that let several machines share one backend.
- `task.rs` — `RewardMode`, `TaskStatus`, `FinishReason`, `TaskSpec`, `TaskRecord`, `Clock`/`FixedClock`/`EPOCH_STAMP`, `TaskRegistry`.
- `capture.rs` — `CaptureMeta`, `CaptureStatus`, `PublishState`, `TurnTransition`, and the `TaskRegistry` capture/turn methods.
- `sample.rs` — `SampleStatus`, `SampleRecord` (+ `pending`, `policy_version`, `validate`, `to_json`/`from_json`), `SampleLedger<S>` (`new`, `store`, `save_once`, `get`, `list_by_status`, `count_by_status`, `pending_count`, `fetch_and_mark_training`, `mark_trained`, `mark_failed`, `reset_to_pending`, `stats`) and the `SampleStore<S>` parity alias.
- `executor.rs` — `TrainingExecutor` seam (`name`, `train_batch`, `cancel`, `check_health`), `TrainingRequest`, `TrainingArtifact` (+ `validate_name`), `artifact_name`/`artifact_version`/`is_artifact_name`, `LocalTrainingExecutor` (+ `root`, `run_dir`, `descriptor_path`), `LocalArtifactDescriptor`, `PpoTrainingExecutor` (+ `device`).
- `training.rs` — `RunStatus`, `RunStage`, `TrainingRunRecord` (+ `public`, `to_public_json`, `to_json`/`from_json`, `parent_lora_name`/`parent_lora_path`), `TrainingRunnerConfig` (+ `new`, `validate`), `LoraActivator`/`LoraActivation`, `NoopLoraActivator`, `UnsupportedLoraActivator`, `TrainingCheckpoint`, `TrainingRunner<S>` (`new`, `with_clock`, `with_parent_policy`, `store`, `config`, `executor`, `claim`, `get`, `active_run`, `run`, `stop`, `recover`, `stats`, `export_checkpoint`), `SERVICE_RESTARTED`, `TRAINING_CHECKPOINT_VERSION`.
- `lora.rs` — `LoraAvailability`, `LoraVersion`, `LoraRepository` (`new`, `root`, `model_dir`, `version_dir`, `publish`, `list_versions`, `get_latest`, `get_latest_available`, `get_version`, `artifact`, `set_availability`, `list_models`), `version_literal`, `parse_version_literal`.
- `tests/` — 92 unit tests across `mod.rs`, `capture.rs`, `sample.rs`, `executor.rs`, `training.rs`, `lora.rs`.

## State Machines

The task lifecycle table (`active` → `finalized`/`aborted`) and the capture/turn tables are unchanged from step 1; see the `mod.rs`/`capture.rs` tables.

| Sample from | Event | To |
| --- | --- | --- |
| — | `save_once` (new id) | `pending` |
| — | `save_once` (existing id) | unchanged, not returned |
| `pending` | `fetch_and_mark_training` | `training` |
| `training` | `mark_trained` / `mark_failed` / `reset_to_pending` | `trained` / `failed` / `pending` |
| any | the same transition replayed | unchanged (idempotent) |

| Run from | Event | To | Sample disposition |
| --- | --- | --- | --- |
| — | `claim`, no active run, enough pending samples | `pending/queued` | `pending` → `training` |
| — | `claim`, fewer than `min_samples_for_training` pending | unchanged | untouched (`InsufficientSamples`) |
| — | `claim`, active run present | unchanged (`created = false`) | untouched |
| `pending/queued` | `run` | `running/training` → `running/activating` → `succeeded` | `training` → `trained` |
| `running/training` | executor error or artifact-name violation | `failed` | `training` → `pending` |
| `running/activating` | activation error | `failed` | stays `trained` |
| `pending`/`running` | `stop`, stage `queued`/`training` | `canceled` | `training` → `pending` |
| `pending`/`running` | `stop`, stage `activating` | `canceled` | stays `trained` |
| terminal | `stop`, `run` | unchanged | untouched |
| `queued`/`training` | `recover` | `failed("service_restarted")` | `training` → `pending` |
| `activating` | `recover`, auto-activation off | `succeeded` | stays `trained` |
| `activating` | `recover`, activation ok / failed | `succeeded` / `failed` | stays `trained` |

`claim` is single flight through `rl:v1:training_run:active` plus `set_if_absent`; the loser deletes its own orphan record and returns the winner with `created = false`.
Every step of `run` that awaits re-reads the durable record, so a concurrent `stop` or `recover` always wins over the in-flight run and never gets overwritten.
`stop` issues `TrainingExecutor::cancel` exactly while the stage is `training` — the only stage in which the executor owns the batch.

## Store Seam and Key Layout

`OnlineStore` is primitive by contract: string keys, string values, sets of strings; no domain types cross it, and `set_members` is sorted so scans are deterministic on both backends.
`set_if_absent` is the only atomicity the state machines need; every create/claim path uses it and re-reads on a lost race.
The literal layout (documented once in `mod.rs`): task/capture keys as in step 1, plus `rl:v1:sample:{id}`, `rl:v1:samples:{status}`, `rl:v1:sample:seq`, `rl:v1:training_run:{id}`, `rl:v1:training_run:active`, `rl:v1:training_runs`, `rl:v1:training_run:seq`, `rl:v1:training_run:{id}:artifact`, `rl:v1:training_run:{id}:activation`.
A training-run id must never be the literal `active` because the marker shares the prefix; run ids are allocated as `run-{seq:012x}` from the store counter.
The LoRA repository is filesystem-backed (`{root}/{model_id}/v{N}/{version.json,artifact.json}` plus `{root}/{model_id}/latest`) and writes no store key.

## Determinism

Timestamps come from the injected `Clock` (`FixedClock` by default); no code here calls `SystemTime`, `Instant` or `rand`.
Queue order is `(created_seq, sample_id)` — a monotonic store counter, never a ZSET score derived from the wall clock; a duplicate id that loses the first-wins reservation burns one sequence number (a documented gap).
Run ids come from `rl:v1:training_run:seq`; LoRA versions from the largest existing `vN` plus one, claimed by an atomic directory creation, and `created_seq` is the version number itself.
Tests hold a run inside a stage with a `Barrier`+`Semaphore` gate instead of sleeping, so the whole suite is deterministic.

## Deferred Paths

`PpoTrainingExecutor` (`train_batch`, `cancel`, `check_health`) returns `Error::Unsupported` naming `torch`, `transformers`, `verl`, `ray` and the CUDA device; no capability probe exists, so `/health` must answer 503 rather than claim a training capability.
`UnsupportedLoraActivator` returns `Error::Unsupported` naming the vendor AI gateway (AIGW) HTTP service; the artifact is still exported and recorded, only the switch is missing.
`NoopLoraActivator` records the activation under `rl:v1:training_run:{id}:activation` — deterministic, for tests and local runs.

## Known Issues

Cross-process claims are not fully atomic: two writers can both fetch pending samples before either marks them, so the loser re-queues ids the winner may already be training. Deployments run one writable service process per store (or one store connection), as Python's single scheduler does.
`fetch_and_mark_training` moves the set membership before rewriting the record; a crash in between leaves a sample indexed as `training` with a `pending` record, which `reset_to_pending` repairs.
Sample sequence allocation is read-modify-write (Redis `INCR` is the atomic upgrade); concurrent allocation can produce equal `created_seq` values, which ordering then breaks on `sample_id`.
The LoRA repository's version allocator is atomic across processes through directory creation; a network filesystem with weak directory atomicity and two writers is outside that contract, and `latest` is last-writer-wins.
`TrainingRunRecord` still carries `sample_ids` and the parent policy in its durable form; the public HTTP shape is the same struct with those fields skipped.

## Routing Table

- `./error.rs` → `OnlineError`, `code()`, `http_status()`, kernel conversions
- `./store.rs` → `OnlineStore` seam, `MemoryOnlineStore`, `keys`, shared-store delegation
- `./task.rs` → task identity, record, clock, lifecycle and terminal reward
- `./capture.rs` → capture reservations, commit/discard, turn publication and handover
- `./sample.rs` → sample records, status indexes, training fetch and counters
- `./executor.rs` → executor seam, local executor, typed PPO stub, artifact-name contract
- `./training.rs` → training-run records and runner, activation seam, checkpoint export
- `./lora.rs` → filesystem artifact repository, versions and availability
- `./tests/` → the unit suite (compiled only under `cfg(test)`): `mod.rs`, `capture.rs`, `sample.rs`, `executor.rs`, `training.rs`, `lora.rs`
- `./mod.rs` → module docs, re-exports, JSON helpers
