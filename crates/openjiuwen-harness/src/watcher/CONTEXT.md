# watcher

## Intent

Watch the workspace roots that the harness reloads from — skill libraries, agent-spec templates, instruction files and the MCP configuration — and emit debounced, coalesced, typed change events to subscribers plus named reload handlers.
This is a Rust-native addition: Python `openjiuwen.harness` has no watcher, so there is no parity contract for its shape; it must only compose existing public harness APIs.
The OS notification layer is owned by the optional `notify` feature (`notify = ["dep:notify"]` in `Cargo.toml`); every other type compiles and behaves identically without it, and only `WorkspaceWatcher::start` reports the missing backend.
Both internal hops are bounded and both losses are counted, logged and announced, so no producer can grow memory behind a slow consumer.
Reload handlers run on a dispatcher task of their own, so a slow handler delays neither debouncing nor publication.
The module is a leaf of the harness: it reads `openjiuwen_common` (errors, `OpenJiuwenHome`), `tokio`, `tokio_util`, `tokio-stream`, `futures`, `async-trait`, `tracing`, and the public surfaces of `crate::skills`, `crate::cli::{instruction, mcp_config, paths}` and `crate::spec`; nothing may depend on `watcher` except callers.

## API Surface

Everything public is re-exported flat from `crate::watcher`.

### `event.rs` — typed event model

- `WatchTargetKind { Skills, Spec, InstructionFile, McpConfig, Other }` — `Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize`, `#[serde(rename_all = "snake_case")]`, `as_str()`.
- `ChangeKind { Created, Modified, Removed, Renamed }` — same derives, `as_str()`, `merge(self, incoming) -> ChangeKind` (coalescing table below).
- `WatchEvent { sequence: u64, kind: WatchTargetKind, path: PathBuf, change: ChangeKind, previous_path: Option<PathBuf>, label: Option<String> }` — `Debug, Clone, PartialEq, Eq, Serialize, Deserialize`; `sequence` is monotonic per watcher starting at 1; `previous_path` is set only for `Renamed`; `label` names the concrete target (`"skills:workspace"`, `"mcp:home"`, …).
- `EffectKind` is not introduced: `WatchEvent` stays a pure data record.
- `ReloadFailure { sequence: u64, handler: String, kind: WatchTargetKind, path: PathBuf, error_kind: String, message: String }` — serde; `sequence` is the triggering event's sequence; `error_kind` is `ErrorKind::as_str()`.
- `WatchFailure { error_kind: String, message: String }` — serde; a runtime OS-watcher error.
- `DropStage { Intake, Dispatch }` — `Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize`, `#[serde(rename_all = "snake_case")]`, `as_str()` → `"intake"` / `"dispatch"`; which bounded queue lost notifications.
- `DroppedUpdates { stage: DropStage, dropped: u64, total: u64 }` — serde; `dropped` is what this notice covers since the previous one, `total` is the stage's running total.
- `WatchUpdate { Event(WatchEvent), ReloadFailed(ReloadFailure), Dropped(DroppedUpdates), WatcherFailed(WatchFailure) }` — `#[serde(tag = "type", rename_all = "snake_case")]` (`event`, `reload_failed`, `dropped`, `watcher_failed`), `Debug, Clone, PartialEq, Eq`, `sequence()` accessor (`None` for `Dropped` and `WatcherFailed`).

Coalescing table (`ChangeKind::merge(existing, incoming)`), applied per path inside one debounce window:

| existing \ incoming | Created | Modified | Removed | Renamed |
| --- | --- | --- | --- | --- |
| Created | Created | Created | Removed | Renamed |
| Modified | Modified | Modified | Removed | Renamed |
| Removed | Modified | Removed | Removed | Renamed |
| Renamed | Renamed | Renamed | Removed | Renamed |

`Removed + Created` is `Modified` (in-place rewrite); `Created + Removed` nets to `Removed` (transient file).

### `debounce.rs` — coalescing state machine

- `DebounceInput { kind, path, change, previous_path, label }` — the producer-side event before sequence assignment.
- `Debouncer::new(window: Duration)`; `window()`, `pending_len()`, `next_deadline() -> Option<Instant>`, `dropped() -> u64`.
- `push(&mut self, input: DebounceInput, now: Instant) -> ()` — coalesces into a `BTreeMap<PathBuf, Pending>` and moves that path's deadline to `now + window` (trailing debounce).
- `flush(&mut self, now: Instant) -> Vec<WatchEvent>` — drains every entry whose deadline is `<= now`, assigning increasing sequence numbers in path order.
- `take_all(&mut self) -> Vec<WatchEvent>` — drains everything (shutdown drain).
- `max_pending` default 4096: inserting a new path when full drops the input, logs `tracing::warn!` and increments `dropped()`; existing paths still coalesce.
- Pure `std::time::Instant` arithmetic passed in by the caller, so the logic is unit-tested without timers.

### `targets.rs` — targets, plans and classification

- `WatchTarget` — `directory(kind, path)` (recursive), `file(kind, path)` (single file), `with_label(label)`; accessors `kind()`, `path()`, `label()`, `is_recursive()`, `watch_path()` and `matches(&Path) -> bool`.
- A `file` target watches its **parent directory** non-recursively and filters to the exact path, so a delete→recreate or a not-yet-created `OPENJIUWEN.md` is still observed; `watch_path()` returns the parent.
- `WatchPlan { targets, skipped }` — `new()`, `is_empty()`, `len()`, `targets()`, `skipped()`, `push()`, `with_target()`, `workspace(workspace, home, env)`, `existing_only()`, `validate() -> Result<()>`.
- `WatchPlan::workspace` targets, each with a stable label:
  - `<workspace>/skills` — `Skills`, recursive, label `skills:workspace`;
  - `<workspace>/.openjiuwen/agents` (`crate::spec::AGENTS_DIR`) — `Spec`, recursive, label `spec:workspace`;
  - `<home>/.openjiuwen/agents` — `Spec`, recursive, label `spec:home`;
  - `<workspace>/.openjiuwen/plugins` — `Spec`, recursive, label `spec:plugins`;
  - `<workspace>/IDENTITY.md`, `<home>/IDENTITY.md`, `<home>/OPENJIUWEN.md`, `<workspace>/OPENJIUWEN.md` — `InstructionFile` file targets, labels `instruction:workspace:IDENTITY.md`, `instruction:home:IDENTITY.md`, `instruction:home:OPENJIUWEN.md`, `instruction:workspace:OPENJIUWEN.md` (the filename is part of the label);
  - every `crate::cli::paths::mcp_config_paths(home, workspace, env)` entry — `McpConfig` file targets, labels `mcp:home` / `mcp:workspace` / `mcp:override`.
  - `existing_only()` keeps targets whose `watch_path()` exists and records every dropped path in `skipped()` (transparent, never silent).
- `validate()` maps a missing watch path to `Error::not_found`, a permission failure to `Error::sys_op`, a file/directory mismatch to `Error::config`; the path and the target label are always in the message.
- `pub(crate) RawChange { kind: RawChangeKind, paths: Vec<PathBuf> }` and `pub(crate) classify(&[WatchTarget], &RawChange) -> Vec<DebounceInput>` — pure, unit-tested: only paths that match a target survive, a file target outranks a directory target, the longest matching directory wins, `Renamed` with `(from, to)` emits `Renamed` at `to` with `previous_path = from` (a move out of the tree is `Removed` at `from`, a move in is `Created` at `to`), `Other` maps to `Modified`, `Access` is rejected by the adapter.

### `hub.rs` — reload handlers

- `#[async_trait] trait ReloadHandler: Send + Sync { fn name(&self) -> &str; fn target_kinds(&self) -> &[WatchTargetKind]; async fn reload(&self, event: &WatchEvent) -> Result<()>; }`
- `ReloadHub::new()`, `is_empty()`, `len()`, `handler_names() -> Vec<&str>`, `with_handler<H: ReloadHandler + 'static>(self, H)`, `add_handler(Arc<dyn ReloadHandler>)`, `async dispatch(&WatchEvent) -> Vec<ReloadFailure>` (async because `ReloadHandler::reload` is async).
- Dispatch runs only handlers whose `target_kinds` contains the event kind, sequentially, and converts every `Error` into a `ReloadFailure` (also logged at `warn`); a failing handler never stops the others.
- `FnReloadHandler::new(name, kinds, closure)` — closure `for<'a> Fn(&'a WatchEvent) -> BoxFuture<'a, Result<()>>`, the seam for callers and tests.

### `handlers.rs` — concrete reload bodies (no edits outside this module)

- `SkillLibraryReloader::new(SkillLibrary)`; `reload() -> Result<Vec<Skill>>` (write-lock, `load_all_state()`, `discover()`, cache the snapshot), `current() -> Vec<Skill>` (cached snapshot).
- `SkillsReloadHandler::new(Arc<SkillLibraryReloader>)` — `WatchTargetKind::Skills`, name `skills`.
- `InstructionReloader::load(workspace: &Path, home: &OpenJiuwenHome) -> Result<Self>`; `reload() -> Result<InstructionFiles>` (re-runs `InstructionFiles::load`), `current() -> InstructionFiles`, `system_prompt()`.
- `InstructionReloadHandler::new(Arc<InstructionReloader>)` — `WatchTargetKind::InstructionFile`, name `instruction`.
- `McpConfigReloader::new(paths: Vec<PathBuf>) -> Result<Self>` (initial merged load), `reload() -> Result<Option<McpConfiguration>>` (merge later paths over earlier by server name, `None` when no file exists), `current() -> Option<McpConfiguration>`, `servers() -> Result<Vec<McpServerConfig>>`.
- `McpConfigReloadHandler::new(Arc<McpConfigReloader>)` — `WatchTargetKind::McpConfig`, name `mcp`.

### `intake.rs` — bounded queues and the drop policy

- `pub(crate) DropCounters` — `new()`, `intake()`, `dispatch()`, `total(stage)`, `record_intake()`, `record_dispatch()`; an `Arc<AtomicU64>` pair, so the OS callback thread, the intake task and the handle share one total per stage.
- `pub(crate) DropReporter` — `new(interval)`, `pending(&DropCounters)`, `next_deadline(&DropCounters, now) -> Option<Instant>` (never in the past), `report(..) -> bool` (throttled to one notice per stage per interval) and `flush(..) -> bool` (ignores the throttle; used on shutdown).
- Both bounded queues use **drop-newest**: an item arriving while the queue is full is discarded, so admitted items keep FIFO order; a synchronous OS callback cannot implement drop-oldest without blocking.
- Every drop is counted, logged at `debug` with its reason, logged once at `warn` per notice, and published as `WatchUpdate::Dropped`; a sustained overflow yields one notice per interval, never an error storm.
- The notice interval is the configured debounce window with a 50 ms floor (`MIN_DROP_NOTICE_INTERVAL` in `backend.rs`), so a tiny window cannot raise the notice rate without bound.
- A drop notice means the received deltas are incomplete: consumers must reload from disk instead of applying them.
- `#[cfg(feature = "notify")] pub(crate) push_raw(&mpsc::Sender<notify::Result<notify::Event>>, &DropCounters, notify::Result<notify::Event>) -> bool` — the callback's only work: `try_send`, counting a `Full` error and ignoring a `Closed` one (shutdown, not a drop).

### `handle.rs` — subscriptions and shutdown

- `Subscription` — `Stream<Item = WatchUpdate>` (`Send + 'static`) plus `recv() -> Option<WatchUpdate>` and `lagged_updates() -> u64`; a lagging consumer skips the oldest updates (`broadcast` ring buffer, capacity from `WorkspaceWatcher::event_capacity`, default 256), logs one `warn` per lag and never blocks the producer.
- `WatcherHandle` — `subscribe()`, `target_count()`, `is_shutdown()`, `dropped_notifications() -> u64`, `dropped_dispatches() -> u64`, `shutdown() -> Result<()>` (cancel token, drain pending events, await the intake task and then the dispatcher, idempotent); dropping the handle also cancels cooperatively.
- `shutdown()` consumes the stored `JoinHandle`s, so a panicked task surfaces as `Error::sys_op("workspace watcher task failed: …")` (the `JoinError` message names the panic) and a second call returns `Ok(())`.

### `mod.rs` — builder

- `WatcherConfig { debounce: Duration, event_capacity: usize, intake_capacity: usize, dispatch_capacity: usize }` — `Default` = 250 ms / 256 / 1024 / 256 (`DEFAULT_DEBOUNCE`, `DEFAULT_EVENT_CAPACITY`, `DEFAULT_INTAKE_CAPACITY`, `DEFAULT_DISPATCH_CAPACITY`); a zero capacity is raised to 1 before the channel is built.
- `WatcherConfig: FromStr<Err = Error>` parses `debounce=…,event_capacity=…,intake_capacity=…,dispatch_capacity=…`: missing keys keep their `Default`, items/keys/values are whitespace-trimmed, durations take `ns`/`us`/`ms`/`s`/`m` with an optional decimal fraction (a bare number is milliseconds), capacities are unsigned integers (`0` included), and an empty spec, empty item, unknown or duplicate key, or unparseable value is `Error::validation` naming the offending token.
- A spec without any `=` is the duration-only shorthand for `debounce=…` (`"250ms"`), so a single `--debounce` flag is one `parse()` call.
- `WatcherConfig: Display` emits the canonical `debounce=250ms,event_capacity=256,intake_capacity=1024,dispatch_capacity=256` form (`format_watch_duration` picks the largest exact unit), so `parse().to_string()` round-trips and the rendered form is stable.
- `WorkspaceWatcher::new()`, `with_config`, `debounce(Duration)`, `event_capacity(usize)`, `intake_capacity(usize)`, `dispatch_capacity(usize)`, `watch_target(WatchTarget)`, `watch_dir(kind, path)`, `watch_file(kind, path)`, `watch_plan(WatchPlan)`, `for_workspace(workspace, home, env)`, `on_reload<H: ReloadHandler + 'static>(H)`, `targets()`, `start() -> Result<WatcherHandle>`.
- `start` validates the plan (an empty plan is `Error::config`), builds the OS watcher, and spawns two tasks: the intake task (owns the backend, debounces, publishes) and the reload dispatcher (drains its bounded queue and runs the handlers).

## Consumption from outside the crate

- `WorkspaceWatcher::for_workspace(workspace: &Path, home: &OpenJiuwenHome, env: &dyn EnvSource)` and `WatchPlan::workspace` take `&Path`, not `&Workspace`; `OpenJiuwenHome` is `openjiuwen_common::config::OpenJiuwenHome` and `EnvSource` is `openjiuwen_harness::cli::EnvSource`, and neither is re-exported at the harness crate root.
- `WorkspaceWatcher`, `WatcherConfig`, `WatcherHandle` and `WatchUpdate` are re-exported flat from both `crate::watcher` and `crate::` (`src/lib.rs`), so `use openjiuwen_harness::watcher::…` and `use openjiuwen_harness::…` are equivalent.
- `WorkspaceWatcher::start` consumes `self`, is `async`, and returns `Result<WatcherHandle>`; without the `notify` feature it returns a typed `Error::Unsupported`.
- Construction is crate-private: `Subscription::new` and `WatcherHandle::new` are `pub(crate)`; a consumer only ever gets a `Subscription` from `WatcherHandle::subscribe`.
- `WatcherConfig` derives `Debug, Clone, PartialEq, Eq` only — no `Serialize`/`Deserialize`; a host that persists watcher settings maps them itself, and a host that carries them in a flag or env var uses `str::parse::<WatcherConfig>()` (`FromStr`) and `to_string()` (`Display`).
- `Subscription::recv().await` is an inherent method, so consuming updates needs no `futures` import; only `StreamExt`/`BoxStream` usage does (the `futures::Stream` impl at `handle.rs:121` is public).
- `pub mod watcher` is not feature-gated (`lib.rs:45`), so every type compiles without `notify`; enabling the backend is the single feature line `notify = ["dep:notify"]` (`Cargo.toml:63`) and pulls only already-locked, already-cached crates.
- The only consumer in the workspace is the `openjiuwen-cli` binary crate (`crates/openjiuwen-cli/src/watch.rs`), which takes `openjiuwen-harness = { workspace = true, features = ["notify"] }` and exposes `openjiuwen watch [--json|--list|--max-events N|--debounce-ms N]` plus `chat --watch` (report on stderr) through `WatchPlan::existing_only`, the three reload handlers and one `Subscription` select loop with `ctrl_c`.
- `openjiuwen` (umbrella) forwards the feature as `notify = ["openjiuwen-harness/notify"]` (`crates/openjiuwen/Cargo.toml:46`); `openjiuwen-e2e` and `openjiuwen-test-support` do not enable it.

## Constraints

- No panics, no `unwrap`/`expect`, no blocking calls on an async task; the `notify` callback only does a `try_send` into the bounded intake queue, so it neither blocks the backend thread nor unwinds inside `notify`.
- Both queues are bounded (`intake_capacity`, `dispatch_capacity`) and both report their loss: exact counters on the handle, one `warn` plus one `WatchUpdate::Dropped` per notice interval, and a `debug` line per dropped item. Silent drops are not acceptable.
- Reload handlers never run on the intake task: publication and debouncing stay independent of handler latency, and handlers of one event still run sequentially so two bodies never race for the same file.
- A panicking reload handler kills only the dispatcher task; its events keep being published, the dispatch queue reports `WatchUpdate::Dropped(Dispatch)`, and `shutdown()` reports the panic as `Error::sys_op`.
- Errors: `openjiuwen_common::error::{Error, Result}`; OS-watcher mapping is `PathNotFound → Error::not_found`, `Io(PermissionDenied) → Error::sys_op`, `MaxFilesWatch → Error::sys_op`, `InvalidConfig`/`Generic`/backend construction → `Error::config`; every message names the path and the target label.
- Feature-off: the whole module compiles and every type above exists; `WorkspaceWatcher::start` returns `Error::unsupported` naming the `notify` feature. Nothing else is gated, so callers (and tests) can build plans and units without the feature.
- Runtime OS errors are logged at `warn` and published as `WatchUpdate::WatcherFailed`, with consecutive identical messages collapsed so a removed subtree cannot produce an error storm; a missing path observed at runtime is downgraded to `debug`.
- Watch coverage follows the OS backend: parents named by a `WatchPlan` are registered up front, but a directory created *after* `start()` is watched only once the backend delivers its creation event, so a file written into it immediately afterwards can be missed (the directory event itself is still delivered). Pre-create watched parents before `start()` or wait for the directory event when a target subtree is built at runtime.
- Secrets: no configuration content is logged; only paths, target labels and handler names.
- Files stay ≲1000 lines; unit tests live beside the logic, integration tests in `tests.rs` behind `#![cfg(feature = "notify")]`.
- Documentation: `cargo doc -p openjiuwen-harness --no-deps` stays warning-free with and without the feature; module-level docs link feature-gated items as code spans.

## Routing Table

- `./event.rs` → watcher kinds, change kinds and merge table, `WatchEvent`, failures, `WatchUpdate`
- `./debounce.rs` → per-path trailing debounce, sequence assignment, pending cap
- `./targets.rs` → `WatchTarget`, `WatchPlan` workspace discovery, classification of raw changes
- `./hub.rs` → `ReloadHandler` trait, `ReloadHub` dispatch, `FnReloadHandler`
- `./handlers.rs` → skills/instruction/MCP reload bodies and their named handlers
- `./intake.rs` → `DropCounters`, `DropReporter` (notice throttling) and the notify-gated `push_raw` drop-and-count policy of both bounded queues
- `./handle.rs` → `Subscription`, `WatcherHandle`, drop counters, broadcast plumbing and task shutdown
- `./backend.rs` → `notify`-gated OS adapter: plan validation, watch-path registration and deduplication, the intake task (debounce/publish/queue), the reload dispatcher task, the notify-error mapping table and the runtime-failure quieting policy
- `./mod.rs` → `WatcherConfig`, `WorkspaceWatcher` builder, `start`, flat re-exports, module docs
- `./tests.rs` → `notify`-gated end-to-end watcher tests (the feature-off `Unsupported` assertion lives in `backend.rs`)

## Test Strategy

- `mod.rs` unit tests (`config_tests`, feature-independent) cover the `FromStr`/`Display` pair: the full and partial specs, the duration-only shorthand, every unit suffix and fraction, whitespace trimming, zero capacities, the canonical rendering, the stable round-trip, and every error path (empty spec, empty item, item without a value, unknown key, duplicate key, unparseable capacity, unparseable duration).
- `debounce.rs`, `event.rs` and `targets.rs` unit tests are pure and deterministic: merge table, per-path coalescing, deadline extension, ordered sequence numbers, pending-cap drops, target filtering precedence, rename mapping, plan discovery and validation errors, plus the `dropped` wire form of `DropStage`/`DroppedUpdates`.
- `intake.rs` unit tests are deterministic and timer-free: shared counters across clones, first-drop announcement, throttle/coalescing inside one interval, the shutdown flush, an unsubscribed watcher still counting, and `push_raw` dropping the newest into a one-slot queue (a closed queue is not counted).
- `handle.rs` unit tests cover ordering, lag accounting, fused streams, cooperative cancellation, idempotent shutdown, a failed intake task, a failed dispatcher task and the shared drop counters.
- `tests.rs` drives a real OS watcher on a `tempfile` directory with bounded waits (`tokio::time::timeout`, ≤10 s): create/modify/delete delivery, bursty writes collapsing to one event per window, instruction-file / MCP-config / skills reload handlers observing the new state, a removed directory not error-storming, clean idempotent shutdown (subscription stays open while the handle lives, ends with `None` after it is dropped), a panicking handler surfacing as a typed `SysOp` shutdown error, a blocked handler not delaying publication for another target, and a one-slot intake / dispatch queue reporting its drops through the counter and the `WatchUpdate::Dropped` notice.
- Slow-handler tests block the handler on a `CancellationToken` instead of sleeping, so "the handler is still running" is a fact rather than a timing guess.
- Never assert exact timings or sleep durations; gate on a bounded total deadline and poll/await the condition instead.
- Definition of done: `cargo fmt`, `cargo test -p openjiuwen-harness --offline` (default, `--no-default-features`, `--all-features`), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps`.
