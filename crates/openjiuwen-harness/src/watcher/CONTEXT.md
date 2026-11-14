# watcher

## Intent

Watch the workspace roots that the harness reloads from — skill libraries, agent-spec templates, instruction files and the MCP configuration — and emit debounced, coalesced, typed change events to subscribers plus named reload handlers.
This is a Rust-native addition: Python `openjiuwen.harness` has no watcher, so there is no parity contract for its shape; it must only compose existing public harness APIs.
The OS notification layer is owned by the optional `notify` feature (`notify = ["dep:notify"]` in `Cargo.toml`); every other type compiles and behaves identically without it, and only `WorkspaceWatcher::start` reports the missing backend.
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
- `WatchUpdate { Event(WatchEvent), ReloadFailed(ReloadFailure), WatcherFailed(WatchFailure) }` — `#[serde(tag = "type", rename_all = "snake_case")]`, `Debug, Clone, PartialEq, Eq`, `sequence()` accessor (`None` for `WatcherFailed`).

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
  - `<workspace>/IDENTITY.md`, `<home>/IDENTITY.md`, `<home>/OPENJIUWEN.md`, `<workspace>/OPENJIUWEN.md` — `InstructionFile` file targets, labels `instruction:<source>`;
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

### `handle.rs` — subscriptions and shutdown

- `Subscription` — `Stream<Item = WatchUpdate>` (`Send + 'static`) plus `recv() -> Option<WatchUpdate>` and `lagged_updates() -> u64`; a lagging consumer skips the oldest updates (`broadcast` ring buffer, capacity from `WorkspaceWatcher::event_capacity`, default 256), logs one `warn` per lag and never blocks the producer.
- `WatcherHandle` — `subscribe()`, `target_count()`, `is_shutdown()`, `shutdown() -> Result<()>` (cancel token, drain pending events, await the OS-watcher task, idempotent); dropping the handle also cancels cooperatively.

### `mod.rs` — builder

- `WatcherConfig { debounce: Duration, event_capacity: usize }` — `Default` = 250 ms / 256.
- `WorkspaceWatcher::new()`, `with_config`, `debounce(Duration)`, `event_capacity(usize)`, `watch_target(WatchTarget)`, `watch_dir(kind, path)`, `watch_file(kind, path)`, `watch_plan(WatchPlan)`, `for_workspace(workspace, home, env)`, `on_reload<H: ReloadHandler + 'static>(H)`, `targets()`, `start() -> Result<WatcherHandle>`.
- `start` validates the plan (an empty plan is `Error::config`), builds the OS watcher, and spawns one task that owns it.

## Constraints

- No panics, no `unwrap`/`expect`, no blocking calls on the async task; the `notify` callback only forwards into a tokio `mpsc` channel.
- Errors: `openjiuwen_common::error::{Error, Result}`; OS-watcher mapping is `PathNotFound → Error::not_found`, `Io(PermissionDenied) → Error::sys_op`, `MaxFilesWatch → Error::sys_op`, `InvalidConfig`/`Generic`/backend construction → `Error::config`; every message names the path and the target label.
- Feature-off: the whole module compiles and every type above exists; `WorkspaceWatcher::start` returns `Error::unsupported` naming the `notify` feature. Nothing else is gated, so callers (and tests) can build plans and units without the feature.
- Runtime OS errors are logged at `warn` and published as `WatchUpdate::WatcherFailed`, with consecutive identical messages collapsed so a removed subtree cannot produce an error storm; a missing path observed at runtime is downgraded to `debug`.
- Secrets: no configuration content is logged; only paths, target labels and handler names.
- Files stay ≲1000 lines; unit tests live beside the logic, integration tests in `tests.rs` behind `#![cfg(feature = "notify")]`.
- Documentation: `cargo doc -p openjiuwen-harness --no-deps` stays warning-free with and without the feature; module-level docs link feature-gated items as code spans.

## Routing Table

- `./event.rs` → watcher kinds, change kinds and merge table, `WatchEvent`, failures, `WatchUpdate`
- `./debounce.rs` → per-path trailing debounce, sequence assignment, pending cap
- `./targets.rs` → `WatchTarget`, `WatchPlan` workspace discovery, classification of raw changes
- `./hub.rs` → `ReloadHandler` trait, `ReloadHub` dispatch, `FnReloadHandler`
- `./handlers.rs` → skills/instruction/MCP reload bodies and their named handlers
- `./handle.rs` → `Subscription`, `WatcherHandle`, broadcast plumbing
- `./backend.rs` → `notify`-gated OS adapter: plan validation, watch-path registration and deduplication, the debounce/publish/dispatch task, the notify-error mapping table and the runtime-failure quieting policy
- `./mod.rs` → `WatcherConfig`, `WorkspaceWatcher` builder, `start`, flat re-exports, module docs
- `./tests.rs` → `notify`-gated end-to-end watcher tests and the feature-off `Unsupported` test

## Test Strategy

- `debounce.rs`, `event.rs` and `targets.rs` unit tests are pure and deterministic: merge table, per-path coalescing, deadline extension, ordered sequence numbers, pending-cap drops, target filtering precedence, rename mapping, plan discovery and validation errors.
- `tests.rs` drives a real OS watcher on a `tempfile` directory with bounded waits (`tokio::time::timeout`, ≤10 s): create/modify/delete delivery, bursty writes collapsing to one event per window, instruction-file / MCP-config / skills reload handlers observing the new state, a removed directory not error-storming, and clean idempotent shutdown.
- Never assert exact timings or sleep durations; gate on a bounded total deadline and poll/await the condition instead.
- Definition of done: `cargo fmt`, `cargo test -p openjiuwen-harness --offline` (default, `--no-default-features`, `--all-features`), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps`.
