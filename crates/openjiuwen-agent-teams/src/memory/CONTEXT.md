# src/memory — team memory wiring

## Intent

The runtime half of team memory: turn the `openjiuwen_memory::team` engine into something a
`TeamAgent` drives — a per-team wiring handle, the metadata config that selects the backend, the
model-facing tools each member mounts, the `TEAM_MEMORY.md` export/import, and the observability
bridge that makes every operation visible through the crate's own span seam.
Python parity source: `openjiuwen/agent_teams/memory/` (v0.1.20) — `TeamMemoryManager`,
`MemberMemoryToolkit` and the shared `TEAM_MEMORY.md` summary.
The engine (records, store, facade, extractors, hooks, toolkit, observer seam, document format) stays in
`openjiuwen_memory::team`; this node never re-implements it.

## API Surface

Crate path: `openjiuwen_agent_teams::memory`.

```rust
pub const MEMORY_METADATA_KEY: &str = "memory";

pub struct MemoryConfig {              // Clone, Serialize, Deserialize; Debug + PartialEq by hand
    pub enabled: bool,                 // default true
    pub store: MemoryStoreConfig,      // default Memory
    pub extractor: MemoryExtractorKind,// default Rule
    pub export_on_stop: bool,          // default true
    pub import_on_build: bool,         // default true
    pub trace: bool,                   // default true — attach the team-memory observer
    pub otel_sink: Option<Arc<OtelSpanSink>>, // `#[serde(skip)]`, default None
    pub observer: Option<Arc<dyn TeamMemoryObserver>>, // `#[serde(skip)]`, default None
    pub hooks: TeamMemoryHookConfig,   // TeamMemoryHookConfig::default()
}
impl Default for MemoryConfig;         // the fully working configuration
impl MemoryConfig { pub fn validate(&self) -> Result<()>; }

pub enum MemoryStoreConfig { Memory /* default */, File { dir: Option<PathBuf> } }
impl MemoryStoreConfig {
    pub fn is_file(&self) -> bool;
    pub fn dir(&self) -> Option<&Path>;
}
// Serialize: "memory" | {"kind":"file"} | {"kind":"file","dir":"…"};
// Deserialize accepts a bare name, a bare directory path, or a {kind, dir} object (tolerant).

pub enum MemoryExtractorKind { Rule /* default */, Llm }   // Display: "rule" | "llm"

pub fn memory_config(spec: &TeamAgentSpec) -> MemoryConfig;

pub const TEAM_MEMORY_FILE: &str = "TEAM_MEMORY.md";

// trace.rs — the observability bridge, attached where the facade is built
#[cfg(feature = "otel")] pub use observability::otel::OtelSpanSink;
#[cfg(not(feature = "otel"))] pub enum OtelSpanSink {}    // uninhabited stand-in
pub fn observer_for(team_id: &str, config: &MemoryConfig) -> Option<Arc<dyn TeamMemoryObserver>>;
pub fn facade(team_id: &str, config: &MemoryConfig) -> (TeamMemory, Option<Arc<dyn TeamMemoryObserver>>);

pub struct TeamMemoryWiring;                      // Send + Sync, manual Debug, share as Arc<Self>
impl TeamMemoryWiring {
    pub async fn open(team_id: impl Into<String>, config: MemoryConfig,
                      workspace_root: impl Into<PathBuf>) -> Result<Arc<Self>>;
    pub async fn open_with_model(team_id: impl Into<String>, config: MemoryConfig,
                                 workspace_root: impl Into<PathBuf>,
                                 model: Arc<dyn Model>) -> Result<Arc<Self>>;
    pub async fn open_with_observer(team_id: impl Into<String>, config: MemoryConfig,
                                    workspace_root: impl Into<PathBuf>,
                                    observer: Arc<dyn TeamMemoryObserver>) -> Result<Arc<Self>>;
    pub fn with_extractor(&self, extractor: Arc<dyn TeamMemoryExtractor>) -> Arc<Self>;
    pub fn team_id(&self) -> &str;
    pub fn memory(&self) -> &Arc<TeamMemory>;
    pub fn hooks(&self) -> &TeamMemoryHooks;
    pub fn config(&self) -> &MemoryConfig;
    pub fn document_path(&self) -> &Path;
    pub fn is_enabled(&self) -> bool;
    pub fn trace_enabled(&self) -> bool;
    pub fn observer(&self) -> Option<&Arc<dyn TeamMemoryObserver>>;
    pub fn toolkit(&self, member: &str, session: Option<&str>) -> Result<Arc<MemberMemoryToolkit>>;
    pub fn tools(&self, member: &str, session: Option<&str>) -> Result<Vec<Arc<dyn Tool>>>;
    pub async fn on_session_start(&self, session_id: &str) -> Result<()>;
    pub async fn on_member_join(&self, member: &str) -> Result<Option<TeamMemoryRecord>>;
    pub async fn on_member_leave(&self, member: &str) -> Result<Option<TeamMemoryRecord>>;
    pub async fn on_turn(&self, turn: TeamTurn) -> Result<()>;
    pub async fn on_turn_message(&self, from: &str, to: Option<&str>, content: &str,
                                 at: DateTime<Utc>) -> Result<()>;
    pub async fn flush(&self) -> Result<TeamMemoryFlush>;
    pub async fn export(&self) -> Result<Option<String>>;
    pub async fn import(&self) -> Result<usize>;
    pub async fn import_from(&self, content: &str) -> Result<usize>;
}

pub struct TeamMemoryTool;                       // Debug, one Tool per toolkit tool name
impl TeamMemoryTool {
    pub fn new(toolkit: Arc<MemberMemoryToolkit>, name: &'static str) -> Result<Self>;
    pub fn all(toolkit: Arc<MemberMemoryToolkit>) -> Vec<Arc<dyn Tool>>;
    pub fn name(&self) -> &'static str;
    pub fn toolkit(&self) -> &Arc<MemberMemoryToolkit>;
}
pub fn team_memory_tools(toolkit: Arc<MemberMemoryToolkit>) -> Vec<Arc<dyn Tool>>;
```

Flat re-exports at `crate::memory`: `MEMORY_METADATA_KEY`, `MemoryConfig`, `MemoryStoreConfig`,
`MemoryExtractorKind`, `memory_config`, `TeamMemoryWiring`, `TEAM_MEMORY_FILE`, `TeamMemoryTool`,
`team_memory_tools`, `OtelSpanSink`.
The bridge functions live under `crate::memory::trace`; `OtelSpanSink` is the observability crate's
sink on the `otel` build and an uninhabited stand-in otherwise.
Re-exported engine items are *not* re-exported here; callers reach them through
`openjiuwen_memory::team`.

## Metadata contract

`spec.metadata["memory"]` is optional and tolerant, in the style of
`runtime::spec_meta::checkpoint_config`:

| value | effect |
| --- | --- |
| absent, `null`, blank string | `MemoryConfig::default()` |
| non-object (bool, number, array) | one `tracing::warn!`, `MemoryConfig::default()` |
| object | parsed key by key; each malformed key logs one warning and keeps that key's default |

- `enabled`, `export_on_stop`, `import_on_build`, `trace` are booleans; anything else keeps the default.
- `store` is `"memory"`, `"file"`, a bare directory path (any other string is the directory), or
  `{"kind": "memory"|"file", "dir": "…"}`; an unknown kind, a non-string `kind`/`dir`, or a
  non-string value degrades to `Memory`.
- `extractor` is `"rule"` or `"llm"`; anything else keeps `Rule`.
- `hooks` is deserialized straight into `TeamMemoryHookConfig`; a malformed block keeps the
  engine defaults.
- `otel_sink` is never read from metadata: it is `#[serde(skip)]` and `memory_config` always sets it
  to `None`, so a host that wants the in-memory sink assigns it on the parsed `MemoryConfig`.
- `observer` is never read from metadata either: it is `#[serde(skip)]` and `memory_config` always
  sets it to `None`, so a host that wants its own observer assigns it on the parsed `MemoryConfig`
  or opens the wiring through `TeamMemoryWiring::open_with_observer`.

## Observability bridge

`TeamMemoryWiring::build` asks `trace::facade(team_id, config)` for the facade, which selects the
store, then chains `.with_observer(observer)` when `trace::observer_for` answered one.
`openjiuwen_memory::team::TeamMemory::with_observer` consumes the facade, so the observer is only
attachable at construction — never later.
The observer is `observability::memory::memory_observer(team_id)`, plus
`MemorySpanObserver::with_sink(Arc::clone(sink))` when `config.otel_sink` is `Some` on the `otel` build.
Every store selection (in-memory, and the file store that falls back to in-memory) funnels through
`trace::facade`, so "traced or not" is one decision.
`TeamMemoryWiring::observer` holds a second handle on the same object the facade owns, and
`trace_enabled()` reports whether one is attached (true only when `enabled` and `trace` are both set).
`TeamMemoryWiring::with_extractor` rebuilds the hooks over the same facade and carries the observer
over unchanged.

What one traced operation yields: one `memory.{operation}` child span of the observer's own
`team.{team_id}` root (attributes `team_id`, `operation`, `outcome`, `count`, plus `member_id`,
`kind`, `error` when present), one `tracing` event at target `openjiuwen_agent_teams::memory`
(`debug!` on success, `warn!` on failure), and — on the `otel` build — one `SpanRecord` in the
configured sink.

A caller-supplied observer (`MemoryConfig::observer`, or `TeamMemoryWiring::open_with_observer`)
replaces that built-in bridge: `observer_for` clones the supplied `Arc` and the crate's
`MemorySpanObserver` is not built at all, so a configured `otel_sink` stays unused (log line at
target `openjiuwen_agent_teams::memory`).
The precedence is `enabled` → `trace` → `observer` → built-in bridge, so an injected observer is
**not** applied to a `trace: false` wiring: the explicit switch is the last word.
There is deliberately no post-construction setter — `TeamMemory::with_observer` consumes the
facade, so the observer can only be chosen while the facade is built.
`with_extractor` carries whichever observer the wiring holds (injected or built-in) over unchanged.

Disabled: `enabled: false` or `trace: false` attaches no observer at all, so no span, no event and
no sink record is produced; the facade, hooks and document behave exactly as they otherwise would.

Caveats, stated because they are visible from this node:

- The bridge mints its trace from the team name (`TeamSpan::root_for`), so `memory.*` spans are a
  **sibling trace** of the runtime's `team.{name}` trace, not children of it — the wiring never sees
  the runtime root span. Joining the two needs a `MemorySpanObserver::from_root(TeamSpan)`
  constructor in the observability node; hosts correlate by the `team.{name}` attributes both roots carry.
- Real OTLP export stays deferred: the `opentelemetry` SDK and exporter live in
  `openjiuwen-extensions`, which is in this crate's dependency layer and cannot be depended on here,
  so the `otel` feature only fills the in-memory sink and `OTLP_EXPORT_AVAILABLE` stays `false`.

## Constraints

- Only `crates/openjiuwen-agent-teams/src/memory/**` is edited here.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::Validation` for user-supplied ids,
  knobs and documents, `Error::Memory` for document I/O failures; user input never panics.
- Hooks are all-or-nothing: a failing pass logs one `warn!` with the team and member and returns the
  error unchanged, the engine writes each extraction pass with one batch call, `on_session_start`
  installs the session only after the engine returned `Ok`, and the toolkit cache is filled only
  after the context validated.
- A disabled config (`enabled: false`) touches nothing: no hook call, no buffered turn, no session,
  no export, no observer, and `flush` answers an empty `TeamMemoryFlush`.
- `tracing` only; no `println!`/`eprintln!`; log target `openjiuwen_agent_teams::memory`.
- Files around ≤1000 lines; `wiring.rs` (1281 lines, about two thirds inline tests) is the one justified exception: it owns a single contract (the per-team handle), and splitting the tests out would scatter the lifecycle ordering assertions from the code they pin.

## Routing Table

- `./config.rs` → `MEMORY_METADATA_KEY`, `MemoryConfig` (with `trace`, the skipped `otel_sink` and
  the skipped caller-supplied `observer`), `MemoryStoreConfig`, `MemoryExtractorKind`, `validate()`
  and the tolerant `metadata["memory"]` parser `memory_config`
- `./trace.rs` → `observer_for` and `facade`: the one place the team-memory observer is chosen and
  attached to the facade (`enabled` → `trace` → supplied observer → built-in bridge), plus the
  `OtelSpanSink` re-export and its uninhabited stand-in
- `./wiring.rs` → `TeamMemoryWiring` and `TEAM_MEMORY_FILE`: facade + hooks + per-member toolkit
  cache + lifecycle pass-throughs + `TEAM_MEMORY.md` export/import + the observer handle and its
  `open_with_observer` constructor
- `./tools.rs` → `TeamMemoryTool` and `team_memory_tools`: the four `MemberMemoryToolkit` tools as
  `openjiuwen_foundation::tool::Tool` handles

## Notes for Agents

- Durability: this crate does not enable the `openjiuwen-memory` `sqlite` feature, so a `File` store
  keeps the in-memory facade and persists through `TEAM_MEMORY.md`; that is the documented
  workaround, not a silent fallback.
- Document path: `File { dir: Some(dir) }` puts `TEAM_MEMORY.md` in `dir`, every other store at
  `<workspace_root>/TEAM_MEMORY.md`.
- `open` never fails on a document: an unreadable or malformed file is logged and the store starts
  empty; `import()` and `import_from()` report the error to an explicit caller.
- `export()` answers `Ok(None)` when the config is disabled, when `export_on_stop` is `false`, or
  when the rendered document is blank; it creates the parent directories and returns the markdown
  it wrote.
- The extractor: `open` builds `RuleTeamExtractor` (an `extractor: "llm"` config without a model
  logs at debug and uses the rules), `open_with_model` builds `LlmTeamExtractor` from the supplied
  model, and `with_extractor` rebuilds the hooks over the same facade with any extractor and fresh
  lifecycle state.
- The toolkit cache is keyed by member name alone and built once, so pass the session on the first
  `toolkit(member, session)` call for that member; `tools(member, session)` returns four handles
  whose cards carry the toolkit's own id, name, description and `input_schema`.
- `Tool::invoke` reports `ToolContext` cancellation as `Error::Cancelled` before delegating to
  `MemberMemoryToolkit::handle_tool_call`, and never wraps a failure in the toolkit's `ok: false`
  envelope — the harness or agent tool layer owns that envelope.
- Lifecycle pass-throughs supply `CallOptions::new()` because the runtime has no per-call memory
  cancellation token; callers that need a deadline drive `hooks()` directly.
- `on_turn_message(from, to, content, at)` is the `TeamTurn` shorthand; `to: None` addresses the
  whole team.
- The wiring installs no `TeamMemoryObserver` of its own beyond the trace bridge; a caller that
  builds its own `TeamMemory` keeps the observer seam available, and a caller that wants its own
  observer in a wiring supplies `MemoryConfig::observer` or uses `open_with_observer`.
- `MemoryConfig` hand-writes `Debug` (the sink and the observer are rendered as
  `has_otel_sink: bool` / `has_observer: bool`, never dumped) and `PartialEq` (both handles compare
  by `Arc::ptr_eq`), and carries
  `#[allow(clippy::struct_excessive_bools)]` because its four switches are independent options.
- The `otel` sink type is feature-shaped, not the field: `MemoryConfig::otel_sink` exists on every
  build, but without `otel` it is the uninhabited `trace::OtelSpanSink` and can only be `None`.
- Nothing here calls a model, reads the network, or needs `git`; the suites use temp directories,
  fixed `DateTime` values and `openjiuwen_test_support::ScriptedModel` for the LLM-extractor
  construction path.
- The trace tests are offline and deterministic: the default build proves the observer is attached
  (same `Arc` the facade holds), that `trace: false` attaches none, that a supplied observer wins
  over the bridge (and is itself not attached to an untraced or disabled wiring), and that
  `open_with_observer` replaces the bridge while still honouring `trace: false`; the `otel` build
  drives `on_member_join` and asserts the sink holds one `memory.write` record per wiring, with one
  trace per open (two opens of the same team mint two traces — no global registry).
- `MEMORY_METADATA_KEY` is owned here; the runtime node reads it through `memory_config` only.
- The runtime needs no new call to get the default tracing: it passes the parsed `MemoryConfig`
  (from `memory_config`) straight into `TeamMemoryWiring::open`, whose sink and observer slots are
  `None` unless it assigns one; to collect span records it sets
  `MemoryConfig { otel_sink: Some(Arc::new(OtelSpanSink::new(capacity))), .. }` before `open`, and
  to route the operations into its own pipeline it sets `observer` (the runtime's
  `TeamBuildDeps::with_memory_observer` does exactly that).
