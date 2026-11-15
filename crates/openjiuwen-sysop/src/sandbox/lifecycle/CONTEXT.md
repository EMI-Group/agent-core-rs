# src/sandbox/lifecycle

## Intent

The managed-container half of the sandbox seam: one record per isolation key plus the launcher calls that keep a container running, mirroring Python's `SandboxGateway` endpoint logic (`_get_endpoint`, `_create_new_sandbox`, `_evict_idle`, `release_sandbox`) over `InMemorySandboxStore`.
The public path is unchanged: `crate::sandbox::lifecycle::{SandboxRecord, Clock, SystemClock, FixedClock, SandboxLifecycle, now_epoch_secs, global}`.
`lifecycle::global()` binds the process-global `launcher::global()`; local registries and `FixedClock` are the test norm.

## API Surface

- `mod.rs` — `SandboxLifecycle` (`new`, `with_clock`, `launchers`, `now`, `record`, `records`, `len`, `is_empty`, `ensure_endpoint`, `ensure_endpoint_now`, `release`, `evict_idle`, `evict_idle_now`, `evict_idle_for`, `clear`) with the private helpers `revive`, `launch_container`, `touch_running`, `mark_running`, `restore`, `drop_record`, `take`, `map_launcher_failure`, plus `global()` and the module-level state-machine/time/concurrency docs.
- `record.rs` — `SandboxRecord { isolation_key, sandbox_id, endpoint, status, launcher_type, created_epoch_secs, last_used_epoch_secs, container_config_hash, metadata }` (serde snake_case, tolerant of unknown fields, `JsonSchema`) plus `container_config_hash(&SandboxLauncherConfig)`, `container_metadata(&SandboxLauncherConfig)` and `refresh_container_config(&mut SandboxRecord, &SandboxLauncherConfig)`.
- `clock.rs` — `trait Clock`, `SystemClock` (the only `SystemTime` reader together with `now_epoch_secs()`), `FixedClock` (`new`, `set`, `advance`), `now_epoch_secs()`.
- `tests.rs` — manager behaviour ports (launch/reuse, resume, vanished container, `on_stop` refusals that preserve the `Unsupported` kind, idle-stop boundaries and `FixedClock` sweeps, error mapping, concurrent duplicate launch, config-hash/metadata refresh, serde round-trip) over a `RecordingLauncher` fake in a local `LauncherRegistry`.
- `mod.rs` re-exports `clock` and `record` items, so the submodules are private and the public surface is exactly the pre-split one.

## Constraints

- Time is injected: every decision takes `now_epoch_secs` or reads the injected `Clock`; never call `SystemTime` outside `clock.rs`.
- One `tokio::sync::Mutex<BTreeMap<String, SandboxRecord>>` guards the map, and the lock is never held across a launcher await.
- `revive`/`launch_container` split `ensure_endpoint` into steps 1–2 and 3–4; keep the documented ordering (reuse → status → evict → launch → store, first running record wins).
- A launcher that exists but fails while doing its job is 199008 through `launcher_error`; `map_launcher_failure` passes `Unsupported`/`Validation`/`Config` through unchanged, because those describe what the launcher *is*, not how the call went.
- `container_config_hash` is the SHA-256 of the launcher config's canonical JSON and `metadata` is `launcher_type` + `sandbox_type` + string-valued `extra_params`; both are refreshed after a successful launcher call only, never on a failure.
- Tests live in `tests.rs` declared by `mod.rs`, use local registries only, and never touch `lifecycle::global()` state.
- Cross-module behaviour ports stay in `../tests/sandbox_lifecycle.rs`; do not modify that file when refactoring internals.

## Routing Table

- `./mod.rs` → `SandboxLifecycle` manager, `global()`, state-machine/time/concurrency documentation
- `./record.rs` → `SandboxRecord`
- `./clock.rs` → `Clock`, `SystemClock`, `FixedClock`, `now_epoch_secs`
- `./tests.rs` → manager unit tests with the `RecordingLauncher` fake
