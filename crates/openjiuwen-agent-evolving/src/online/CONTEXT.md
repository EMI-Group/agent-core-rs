# src/online

## Intent

The online-RL task state machine: the durable lifecycle of an RL task, its capture reservations, its delayed-feedback turn handovers, and its terminal reward projection.
Python parity: `openjiuwen/agent_evolving/agent_rl/online/task_registry.py` plus the registry-owned halves of `capture_pipeline.py`.
This directory implements the state machine only; the training runner/executors/LoRA repository, the Redis backend, and the HTTP routes are later steps.
The module is store-agnostic on purpose, so the in-memory store used offline and the future Redis store share exactly one implementation of every rule.

## API Surface

- `mod.rs` — module docs (key layout table, determinism rules), `mod` declarations, re-exports, and the `json` encode/decode helpers used by the persisted records.
- `error.rs` — `OnlineError` (`TaskNotFound`, `TaskConflict`, `TurnClosed`, `InvalidRequest`, `InsufficientSamples`, `Store`), `code()`, `http_status()`, `Result<T, OnlineError>`, and `From<OnlineError> for openjiuwen_common::error::Error`.
- `store.rs` — `OnlineStore` seam (`get`, `set`, `set_if_absent`, `del`, `set_add`, `set_remove`, `set_members`, plus the defaulted `set_contains`), `MemoryOnlineStore` (`new`, `Default`, `keys`, `snapshot`), and the `keys` namespace module.
- `task.rs` — `RewardMode`, `TaskStatus`, `FinishReason`, `TaskSpec` (+ `validate`, `policy_model`), `TaskRecord` (+ `matches_spec`, `from_json`, `to_json`), `Clock`/`FixedClock`/`EPOCH_STAMP`, and `TaskRegistry` (`new`, `with_clock`, `store`, `start`, `get`, `finish`, `abort`, `active_task_for_session`, `record_reward`).
- `capture.rs` — `CaptureMeta`, `CaptureStatus`, `PublishState`, `TurnTransition`, and the `TaskRegistry` capture/turn methods (`begin_capture`, `commit_capture`, `discard_capture`, `claim_turn_publish`, `complete_turn_publish`, `fail_turn_publish`, `advance_turn`, `open_capture_count`, `pending_sample_ids`).
- `tests/` — 46 unit tests: `tests/mod.rs` covers the store, specs, task lifecycle and rewards; `tests/capture.rs` covers captures, turn publication and the written key layout.

## State Machines

| From | Event | To | Error on refusal |
| --- | --- | --- | --- |
| absent | `start` (new id, session has no active task) | `active` (created) | — |
| absent | `start` (same id + same spec) | unchanged (`created = false`) | — |
| absent | `start` (same id, other spec) | unchanged | `TaskConflict("rl_task_id already exists: {id}")` |
| absent | `start` (other id, session task active) | unchanged, that task returned | — |
| `active` | `finish`, no open captures | `finalized` | — |
| `active` | `finish`, open captures present | unchanged | `TaskConflict("RL Task still has open captures")` |
| `active` | `abort` | `aborted` (open captures and unpublished samples dropped) | — |
| `finalized` / `aborted` | `finish` / `abort` | unchanged, first reason wins | — |

| Capture from | Event | Capture to | Error on refusal |
| --- | --- | --- | --- |
| absent | `begin_capture` | `open` | `TaskConflict("RL Task is not active: {id}")`, `TaskNotFound` |
| absent | `begin_capture` (same id, same turn + request fingerprint) | unchanged, `None` returned | — |
| `open` | `commit_capture` | `committed`, sample staged, `true` | — |
| `committed` | `commit_capture` (same response fingerprint) | unchanged, `false` returned | — |
| `committed` | `commit_capture` (other response fingerprint) | unchanged | `TaskConflict("capture_id already has a different response: {cid}")` |
| `open` | `discard_capture` | deleted (also silent when the task already ended) | — |
| `committed` | `discard_capture` | unchanged | `TaskConflict("capture is already committed: {cid}")` |

| Turn publish from | Event | To | Error on refusal |
| --- | --- | --- | --- |
| absent, no open captures | `claim_turn_publish` | `publishing` (returns `Claimed`) | — |
| absent, open captures present | `claim_turn_publish` | unchanged (returns `Waiting`) | — |
| `publishing` | `complete_turn_publish` | `published`, turn added to `:turns:closed`, its pending samples dropped | `TaskConflict("turn is not publishing: {turn}")` |
| `publishing` | `fail_turn_publish` | `failed` (terminal) | — |
| `published` / `failed` | `claim_turn_publish` | unchanged (returns `Published` / `Failed`) | — |
| `published` | `advance_turn` (`:turn:current` is the previous turn, `:turn:transition` is the next) | next turn becomes current, transition key deleted | `TaskConflict` for an unpublished previous turn, a changed current turn, or a changed transition |
| absent / `publishing` / `failed` | `advance_turn` | unchanged | `TaskConflict("turn is not published: {turn}")` |

Delayed-feedback turn guards in `begin_capture`: a closed turn raises `TurnClosed("turn is already closed: {turn}")`, a turn with a publish key raises `TurnClosed("turn is already closing: {turn}")`, an open capture on the current turn while a successor is parked raises `TurnClosed("turn is closing: {turn}")`, a third turn while one is parked raises `TaskConflict("turn {next} is already waiting for {current}")`, and a capture of a new turn returns a `TurnTransition` without reserving the capture.

## Store Seam and Key Layout

`OnlineStore` is primitive by contract: string keys, string values, sets of strings; no domain types cross it, and `set_members` is sorted so scans are deterministic on both backends.
`set_if_absent` is the only atomicity the state machine needs; `start` and `record_reward` claim through it and re-read on a lost race.
The literal layout (documented once in `mod.rs`): `rl:v1:task:{id}`, `rl:v1:active_session:{session}`, `rl:v1:tasks:active`, `rl:v1:task:{id}:captures`, `:captures:open`, `:captures:committed`, `:capture:{cid}`, `:turn:{turn}:captures`, `:turn:current`, `:turn:transition`, `:turns:closed`, `:turn:{turn}:publish`, `:pending:{cid}`, `:reward`.
`MemoryOnlineStore` keeps values in a `BTreeMap` and sets in `BTreeSet`s behind one `tokio::sync::Mutex`; the guard is never held across an `.await`.
An empty set drops its key in the in-memory store and stays in Redis; both read as empty, so no rule depends on the difference.

## Clock and Determinism

`Clock` is injected: `TaskRegistry::new` installs `FixedClock` (stamp `EPOCH_STAMP`), and a caller wanting wall-clock stamps injects its own implementation through `with_clock`.
Nothing here calls `SystemTime`, `Instant`, or `rand`; fingerprints and identifiers are caller-supplied strings, so a test scripts an entire session.
`record_reward` stores `{reward, status, sample_count}` under `:reward`, publishes once per value, resumes a `projecting` record after a crash, and returns the sample count (`u64`).

## Known Issues

`OnlineError::Store` is an addition to the five Python-named errors: storage failures must not be swallowed, so they keep the kernel `Error` as their source and report `internal_error` / `500`.
`complete_turn_publish` requires the publish key to read `publishing`, so a crash between the claim and the completion leaves the turn `publishing` until a retry completes it; Python behaves identically.
`begin_capture` stores the empty string as the current turn when a `delayed_feedback` capture has no `agent_turn_id` (Python stores `"None"`); the capture pipeline validates the field before reaching the registry.
`record_reward` performs the projection and the completion in one call because there is no trajectory store yet; step 2 can split it when samples must be written before `status` becomes `published`.
`claim_turn_publish` has no task guard (it reads captures and the publish key only), matching Python; callers that need existence checks must `get` first.

## Routing Table

- `./error.rs` → `OnlineError`, `code()`, `http_status()`, `Result`, kernel `Error` conversion
- `./store.rs` → `OnlineStore` seam, `MemoryOnlineStore`, `keys`
- `./task.rs` → task identity, record, clock, lifecycle and terminal reward
- `./capture.rs` → capture reservations, commit/discard, turn publication and handover
- `./tests/` → the unit suite (compiled only under `cfg(test)`): `./tests/mod.rs` and `./tests/capture.rs`
- `./mod.rs` → module docs, re-exports, JSON helpers
