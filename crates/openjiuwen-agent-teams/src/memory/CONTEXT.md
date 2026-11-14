# src/memory — team memory wiring

## Intent

The runtime half of team memory: turn the `openjiuwen_memory::team` engine into something a
`TeamAgent` drives — a per-team wiring handle, the metadata config that selects the backend, the
model-facing tools each member mounts, and the `TEAM_MEMORY.md` export/import.
Python parity source: `openjiuwen/agent_teams/memory/` (v0.1.20) — `TeamMemoryManager`,
`MemberMemoryToolkit` and the shared `TEAM_MEMORY.md` summary.
The engine (records, store, facade, extractors, hooks, toolkit, observer seam, document format) stays in
`openjiuwen_memory::team`; this node never re-implements it.

## API Surface

Crate path: `openjiuwen_agent_teams::memory`.

```rust
pub const MEMORY_METADATA_KEY: &str = "memory";

pub struct MemoryConfig {              // Debug, Clone, PartialEq, Serialize, Deserialize
    pub enabled: bool,                 // default true
    pub store: MemoryStoreConfig,      // default Memory
    pub extractor: MemoryExtractorKind,// default Rule
    pub export_on_stop: bool,          // default true
    pub import_on_build: bool,         // default true
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

pub struct TeamMemoryWiring;                      // Send + Sync, manual Debug, share as Arc<Self>
impl TeamMemoryWiring {
    pub async fn open(team_id: impl Into<String>, config: MemoryConfig,
                      workspace_root: impl Into<PathBuf>) -> Result<Arc<Self>>;
    pub async fn open_with_model(team_id: impl Into<String>, config: MemoryConfig,
                                 workspace_root: impl Into<PathBuf>,
                                 model: Arc<dyn Model>) -> Result<Arc<Self>>;
    pub fn with_extractor(&self, extractor: Arc<dyn TeamMemoryExtractor>) -> Arc<Self>;
    pub fn team_id(&self) -> &str;
    pub fn memory(&self) -> &Arc<TeamMemory>;
    pub fn hooks(&self) -> &TeamMemoryHooks;
    pub fn config(&self) -> &MemoryConfig;
    pub fn document_path(&self) -> &Path;
    pub fn is_enabled(&self) -> bool;
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
`team_memory_tools`.
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

- `enabled`, `export_on_stop`, `import_on_build` are booleans; anything else keeps the default.
- `store` is `"memory"`, `"file"`, a bare directory path (any other string is the directory), or
  `{"kind": "memory"|"file", "dir": "…"}`; an unknown kind, a non-string `kind`/`dir`, or a
  non-string value degrades to `Memory`.
- `extractor` is `"rule"` or `"llm"`; anything else keeps `Rule`.
- `hooks` is deserialized straight into `TeamMemoryHookConfig`; a malformed block keeps the
  engine defaults.

## Constraints

- Only `crates/openjiuwen-agent-teams/src/memory/**` is edited here.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::Validation` for user-supplied ids,
  knobs and documents, `Error::Memory` for document I/O failures; user input never panics.
- Hooks are all-or-nothing: a failing pass logs one `warn!` with the team and member and returns the
  error unchanged, the engine writes each extraction pass with one batch call, `on_session_start`
  installs the session only after the engine returned `Ok`, and the toolkit cache is filled only
  after the context validated.
- A disabled config (`enabled: false`) touches nothing: no hook call, no buffered turn, no session,
  no export, and `flush` answers an empty `TeamMemoryFlush`.
- `tracing` only; no `println!`/`eprintln!`; log target `openjiuwen_agent_teams::memory`.
- Files around ≤1000 lines; `wiring.rs` is the only one that approaches it.

## Routing Table

- `./config.rs` → `MEMORY_METADATA_KEY`, `MemoryConfig`, `MemoryStoreConfig`,
  `MemoryExtractorKind`, `validate()` and the tolerant `metadata["memory"]` parser `memory_config`
- `./wiring.rs` → `TeamMemoryWiring` and `TEAM_MEMORY_FILE`: facade + hooks + per-member toolkit
  cache + lifecycle pass-throughs + `TEAM_MEMORY.md` export/import
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
- The wiring installs no `TeamMemoryObserver`; the facade keeps the engine's `tracing` lines, and
  the observer seam stays available to a caller that builds its own `TeamMemory`.
- Nothing here calls a model, reads the network, or needs `git`; the suites use temp directories,
  fixed `DateTime` values and `openjiuwen_test_support::ScriptedModel` for the LLM-extractor
  construction path.
- `MEMORY_METADATA_KEY` is owned here; the runtime node reads it through `memory_config` only.
