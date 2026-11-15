# openjiuwen-agent-teams/src/harness

## Intent

Adapt one team member to the provider-neutral harness protocol, drive it through the member-runtime seam, and persist/restore that member's checkpoints.
Python parity source: `openjiuwen/agent_teams/harness/` (`protocol_adapter.py`, `native_harness.py`, `checkpoint.py`).
The Python adapter downgrades `NativeHarness` to `HarnessProtocol`; in this port `openjiuwen_harness::NativeHarness` already implements `HarnessProtocol`, so this node builds handles, forwards commands, pumps observations, supplies the checkpoint sink and resume envelope, and persists a portable record.
The team runtime sees a `MemberRuntime`; this module is the only place that knows about `HarnessProtocol`.
Checkpointing is per runtime: a runtime built without a `MemberCheckpointStore` does no checkpoint work at all, while the team runtime supplies one to every in-process member by default (in-memory).

## API Surface

```rust
pub struct MemberHarnessContext { pub team_name: String, pub member_name: String, pub session_id: String,
                                  pub workspace_root: PathBuf, pub language: String, pub mode: MemberMode,
                                  pub card_id: String }   // card_id empty by default
impl MemberHarnessContext {
    pub fn new(team_name, member_name, session_id, workspace_root, language, mode) -> Self;  // six args, card_id = "", no toolkit
    pub fn with_card_id(self, card_id: impl Into<String>) -> Self;
    pub fn with_memory_toolkit(self, toolkit: Arc<MemberMemoryToolkit>) -> Self;   // the member's team-memory handle
    pub fn memory_toolkit(&self) -> Option<&Arc<MemberMemoryToolkit>>;
    pub fn memory_tools(&self) -> Vec<Arc<dyn Tool>>;   // the four team_memory_* tools, empty without a toolkit
    pub fn member_session_id(host_session_id: &str, member_name: &str) -> String;  // "<host>:<member>"
}
// Debug/PartialEq/Eq are hand-written: the private `memory_toolkit` field prints as present/absent and
// compares by Arc::ptr_eq, so two contexts with distinct handles for the same member are unequal.

#[async_trait] pub trait MemberHarnessFactory: Send + Sync {
    async fn create(&self, spec: &DeepAgentSpec, ctx: &MemberHarnessContext) -> Result<Arc<dyn HarnessProtocol>>;
    fn name(&self) -> &'static str;
}

pub struct HarnessMemberRuntime;   // impl crate::member::MemberRuntime over Arc<dyn HarnessProtocol>
impl HarnessMemberRuntime {
    pub fn new(harness: Arc<dyn HarnessProtocol>, session_id: impl Into<String>) -> Self;  // unchanged signature
    pub fn with_card_id(self, card_id: impl Into<String>) -> Self;
    pub fn with_checkpoint_store(self, store: Arc<dyn MemberCheckpointStore>) -> Self;
    pub fn with_checkpoint_policy(self, policy: MemberCheckpointPolicy) -> Self;
    pub fn with_cwd(self, cwd: impl Into<String>) -> Self;
    pub fn harness(&self) -> &Arc<dyn HarnessProtocol>;
    pub fn card_id(&self) -> &str;
    pub fn checkpoint_store(&self) -> Option<&Arc<dyn MemberCheckpointStore>>;
    pub fn checkpoint_policy(&self) -> &MemberCheckpointPolicy;
    pub fn cwd(&self) -> Option<&str>;
    pub async fn latest_checkpoint(&self) -> Option<NativeCheckpoint>;       // newest captured payload
    pub async fn capture_checkpoint(&self) -> Result<Option<NativeCheckpoint>>;  // forced capture, errors reported
}

pub struct NativeCheckpoint { pub session_id: String, pub card_id: String, pub cwd: Option<String>,
                              pub state: HarnessState, pub turn_id: Option<String>, pub outputs: Value }
impl NativeCheckpoint {
    pub fn new(session_id, card_id) -> Self;
    pub fn with_cwd(self, cwd) -> Self;  pub fn with_state(self, state) -> Self;
    pub fn with_turn_id(self, turn_id) -> Self;  pub fn with_outputs(self, outputs: Value) -> Self;
    pub fn to_json(&self) -> Result<Value>;
    pub fn from_json(value: Value) -> Result<Self>;
    pub fn validate_compatible(&self, session_id: &str, card_id: &str) -> Result<()>;
}

// checkpoint_store module (one child file per concern)
pub enum CheckpointResume { New, IfAvailable, Require }        // snake_case wire: new / if_available / require
pub struct MemberCheckpointPolicy { pub enabled: bool, pub on_turn: bool, pub on_state: bool,
                                    pub on_stop: bool, pub resume: CheckpointResume }
impl MemberCheckpointPolicy {
    pub fn disabled() -> Self;
    pub fn is_enabled(&self) -> bool;
    pub fn resume_policy(&self) -> openjiuwen_harness_protocol::ResumePolicy;
    pub fn captures_on_turn(&self) -> bool;  pub fn captures_on_state(&self) -> bool;
    pub fn captures_on_stop(&self) -> bool;
}
pub struct MemberCheckpointRecord { pub checkpoint: NativeCheckpoint, pub protocol: Option<HarnessCheckpoint>,
                                    pub unrecognised_state: Option<String> }   // serde: skip when None
impl MemberCheckpointRecord {
    pub fn new(checkpoint: NativeCheckpoint) -> Self;
    pub fn with_protocol(self, protocol) -> Self;
    pub fn with_unrecognised_state(self, raw: impl Into<String>) -> Self;
    pub fn has_unrecognised_state(&self) -> bool;
}

#[async_trait] pub trait MemberCheckpointStore: Send + Sync {
    fn name(&self) -> &'static str;                                  // "memory" / "file"
    async fn save(&self, record: MemberCheckpointRecord) -> Result<String>;   // returns a storage revision
    async fn load(&self, session_id: &str, card_id: &str) -> Result<Option<MemberCheckpointRecord>>;
}
pub struct InMemoryMemberCheckpointStore;   // new / shared() -> Arc<dyn MemberCheckpointStore> / len / is_empty / get / revisions
pub struct FileMemberCheckpointStore;       // new(root) / root / path_for; JSON under <root>/<session>/<card>/checkpoint.json
pub struct MemberCheckpointSink;            // impl HarnessCheckpointSink over a store + identity (new / with_cwd)

#[async_trait] pub trait ModelResolver: Send + Sync { async fn resolve(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>>; }
pub struct EnvModelResolver;                    // MODEL_PROVIDER / MODEL_NAME + ModelClientConfig::from_env
pub struct NativeMemberFactory;                 // impl MemberHarnessFactory
impl NativeMemberFactory { pub fn new(resolver: Arc<dyn ModelResolver>) -> Self; pub fn from_env() -> Self; }
pub const NATIVE_FACTORY_NAME: &str = "native";

// pure projections
pub fn state_to_member_status(HarnessState) -> Option<MemberStatus>;
pub fn state_to_execution_status(HarnessState) -> Option<ExecutionStatus>;
pub fn state_from_event(&HarnessEvent) -> Option<HarnessState>;
pub fn round_from_event(&HarnessEvent) -> Option<RoundOutcome>;
pub fn chunk_from_event(&HarnessEvent) -> Option<Value>;
pub fn is_terminal_event(&HarnessEvent) -> bool;
pub const CHUNK_LLM_OUTPUT / CHUNK_LLM_REASONING / CHUNK_TOOL_CALL / CHUNK_TOOL_RESULT: &str;
```

Runtime behaviour:
- `start(team_session)` closes any previous pump, sends `HarnessContext` (agent name/id = the runtime's session id; `host_session_id` = the team session id when supplied, else the runtime session id — the team runtime always supplies `None`; empty system prompt because the prompt lives in the constructed agent), merges `cwd` and the checkpoint wiring, publishes the settled state to observers (this is the member's `Ready`), then spawns the pump.
- `send(content, true)` → `DeliveryMode::Steer`; `send(content, false)` → `DeliveryMode::FollowUp`.
- `abort(true)` → `AbortMode::Force`; `abort(false)` → `AbortMode::Graceful`.
- `resume(Some(query))` → `HarnessInput::text(query)`; `resume(None)` → `None`.
- `stop()` captures the final checkpoint (when configured), cancels/joins the pump, calls `harness.stop()`, then publishes `Terminated`.
- `HarnessError` becomes `Error::External(Box::new(e))`; the protocol has no timeout variant, so there is no timeout special case.

## Checkpoint wiring

`MemberCheckpointPolicy` decides whether a member captures and how it consumes a stored record; defaults are `enabled`, `on_turn`, `on_stop` on, `on_state` off, `resume = IfAvailable`.

| Trigger | Condition | Payload |
| --- | --- | --- |
| `start` restore | a store is configured and the policy is enabled | `store.load(session_id, card_id)`; a compatible record's `protocol` envelope becomes `context.checkpoint` |
| turn boundary (pump) | `captures_on_turn()` and the event is `turn_lifecycle` `finished`/`aborted`/`failed` | `turn_id` = the event's turn id, `outputs` = the exported envelope's `data` (else `{}`) |
| state change (pump) | `captures_on_state()` and the event is `state_changed` | `turn_id = None`, `outputs` as above |
| `stop` | `captures_on_stop()` | `turn_id = None`, `state` = the harness state before `stop()` |
| `capture_checkpoint()` | store present and policy enabled | forced capture, `turn_id = None`, export/save errors returned |

- With a store configured a `MemberCheckpointSink` is **always** attached (even under `disabled()`), because a provider may publish on its own schedule; capture and restore happen only when the policy is enabled.
- The resume policy always comes from the policy when a store is configured, so `CheckpointResume::New` still starts cold with an envelope in the context — that is protocol semantics, not a bug.
- Restore is opportunistic: a record for another session/card (`NativeCheckpoint::validate_compatible`), a record without an envelope, or an unreadable store logs `warn!` and starts cold; `start` never fails because of checkpoint state.
- Capture is best effort in the pump and in `stop` (failures logged, never failing the command or ending the pump); the recorded payload is cached so `latest_checkpoint()` reports it.
- A record always carries the portable payload; `protocol` is `Some` only when `HarnessProtocol::export_checkpoint` returned an envelope — a payload-only record is still a valid snapshot.
- The sink decodes the payload's state from the envelope's `state` key: a known value is used verbatim, and an absent `state` is `Idle` with no flag, because it is not a degradation.
- A `state` value this build does not know (a foreign provider, a newer format, a wrong type) still records `Idle` so the restore stays tolerant, but the sink logs one `warn!` naming `member`, `card`, `provider`, `checkpoint_id` and the raw value.
- That degradation is persisted on the record as `MemberCheckpointRecord::unrecognised_state` (`Some(raw)`, `has_unrecognised_state()`), so a caller can tell a fresh idle member from one restored out of a corrupt or foreign record.
- `unrecognised_state` is snake_case, `#[serde(default, skip_serializing_if = "Option::is_none")]` and tolerates unknown fields, so a healthy record's document is unchanged and an older document decodes to `None`.
- `restore_into` re-logs the same fact with the member identity when a loaded record carries `unrecognised_state`, and still restores the envelope and payload unchanged.
- Identity convention: records are keyed on `(member session id, card id)`, where the member session is `MemberHarnessContext::member_session_id(team_session, member_name)` (`"<team_session>:<member_name>"`) and the card id is the effective spec's card id, else the member name.
- Every member starts through `start(None)`, so that derived id is also the harness `HarnessContext::host_session_id`; the team session is never forwarded into a live cycle.
- The runtime wires every in-process member through `runtime::checkpoints::wire_runtime` — `build_with` for the leader, `TeamMemberSpawner::build_harness_runtime` for each teammate — with the derived session id, the effective card id, the policy from `metadata["checkpoint"]`, the member workspace root as `cwd` and the resolved store.
- Store resolution: an injected store wins (`TeamBuildDeps::with_checkpoint_store`); otherwise `memory` mints one `InMemoryMemberCheckpointStore` shared by the whole team, while `file` roots each member at its own directory (`dir` when pinned, else `<workspace root>/.openjiuwen/checkpoints`).
- A disabled policy resolves no store at either step, so no sink is attached and no record is written.
- Only in-process runtimes are wired: a `team-member serve` child builds a bare `HarnessMemberRuntime` with no store, so a process member checkpoints nothing.
- Storage is last-write-wins JSON (never pickle); the file backend sanitises both path components (alnum, `-`, `_` kept, everything else `_`) so no identity can traverse out of the root.

## State mapping

| `HarnessState` | `MemberStatus` | `ExecutionStatus` | rationale |
| --- | --- | --- | --- |
| `Idle` | `Ready` | `Idle` | up and accepting input |
| `Running` | `Busy` | `Running` | a turn is executing |
| `Pausing` | `Paused` | `CancelRequested` | pause accepted, round still winding down |
| `Paused` | `Paused` | `Idle` | suspended, not cancelled — a resumed round must not leave a terminal status first |
| `Terminated` | `Stopped` | `None` | runtime gone; there is no execution left to describe, and the row is revivable as `Stopped` (not terminal `ShutDown`) |

## Event mapping

| Event | Projection |
| --- | --- |
| `turn_lifecycle` `started`/`finished`/`aborted`/`failed` | `RoundOutcome::{Started, Finished, Aborted, Failed}` |
| `turn_lifecycle` `paused`/`resumed` | none (does not close a round) |
| `state_changed` | `on_state(new)` |
| `output` (channel `answer`/`system`) | chunk `{"type": "llm_output", "payload": <OutputEvent>}` |
| `output` (channel `reasoning`) | chunk `{"type": "llm_reasoning", "payload": <OutputEvent>}` |
| `item_lifecycle` `item_type="tool"` `started` | chunk `{"type": "tool_call", "payload": {..., "item_id": …}}` |
| `item_lifecycle` `item_type="tool"` `completed` | chunk `{"type": "tool_result", "payload": {..., "item_id": …}}` |
| everything else | none |

Every other event payload is forwarded only through the state/round/chunk projections above; the pump does not re-publish raw protocol events.

## Native build mapping (`NativeMemberFactory`)

| Spec field | Harness counterpart |
| --- | --- |
| `card` | builder card, else `AgentCard::new(member_name)` |
| `system_prompt` | `DeepAgentConfig::system_prompt` |
| `model` (`model_name`/`name`) | `ModelResolver::resolve` → `DeepAgentBuilder::model`; the effective name (spec, else the resolved model's) → `DeepAgentConfig::model_name` so the inner `ReAct` request carries it |
| `model.provider` | `DeepAgentConfig::model_provider` (pinned spec provider, else the resolved model's `provider()`; never invented) |
| `enable_task_loop`, `max_iterations` | same-named `DeepAgentConfig` fields |
| `enable_security_rail`, `completion_timeout` | same-named `DeepAgentConfig` fields |
| `cwd`, `project_root`, `skills`, `language`, `prompt_mode` | same-named `DeepAgentConfig` fields |
| context `workspace_root` | `Workspace` (builder + config) |
| context `mode` | `default_mode` (`plan_mode` ⇒ `AgentMode::Plan`) and `enable_plan_mode` |
| context `memory_toolkit` | `DeepAgentBuilder::tools` — the four `team_memory_remember`/`search`/`forget`/`stats` tools, mounted next to the built-in coding tools |

Ignored at factory level (logged once at `debug` when present): `tools`, `mcps`, `subagents`, `rails`, `approval_required_tools`, and model `params`.
Mounting live tools/MCP servers/rails belongs to the runtime wiring that owns those handles.
The member's team-memory tools travel on `MemberHarnessContext`, never on the spec, because the runtime owns the live toolkit: without one the built agent is exactly the agent without them.
Model resolution is always required: a missing provider/model is `Error::config`.

The model identity must live on the config, not only on the model handle: `DeepAgentBuilder` copies `DeepAgentConfig::{model_name, model_provider}` onto the inner `ReActAgentConfig`, and the agent loop inherits a blank config name into every `ModelRequestConfig` whose request name is empty — an empty name makes provider requests and `list_models` fail (Python parity: `native_harness.py` reads `self._react_agent.config.model_name`).
`api_base`/`api_key` are deliberately left unset by the factory: they belong to the resolver's `ModelClientConfig`, and `DeepAgentConfig` needs them only on the lazy-model path, which never runs because the factory always supplies a live model.

## Injection seam for tests and embedders

`MemberHarnessFactory` is the only way a member's brain is replaced: `TeamBuildDeps::with_harness_factory` swaps the default `NativeMemberFactory::from_env()` for any `Arc<dyn MemberHarnessFactory>`, and `TeamAgentSpec::build_with` creates the leader through it while `TeamMemberSpawner::spawn` creates every teammate through it.
Both calls pass a `&DeepAgentSpec` (leader: `agents["leader"]`; teammate: `agents[name]` else `agents["teammate"]`) plus a `MemberHarnessContext` whose `session_id` is that member's derived `member_session_id(team_session, member_name)` and whose `card_id` is the effective card id.
A scripted factory can therefore key its per-member answers on either `member_name` or the derived id; the leader receives the routed `agents["leader"]` spec and a teammate its forked/router-merged blueprint.
The team runtime attaches each member's own `MemberMemoryToolkit` to that context through `with_memory_toolkit`, so the `team_memory_*` tools reach the member's harness while the toolkit stays owned by the runtime.
The engine handle is named `openjiuwen_memory::team::MemberMemoryToolkit` (`MemberMemoryToolkit::new(Arc<TeamMemory>, MemberMemoryContext)`) and is never re-exported by this crate; `memory_tools` is a private module, so the toolkit surface is only `MemberHarnessContext::{memory_toolkit, memory_tools}`.
The tools are the `crate::memory::team_memory_tools(toolkit: Arc<MemberMemoryToolkit>) -> Vec<Arc<dyn Tool>>` projection (`TeamMemoryTool::all`, engine order remember/search/forget/stats), each dispatching `MemberMemoryToolkit::handle_tool_call(name: &str, args: Value) -> Result<Value>`.
`HarnessMemberRuntime` has no memory accessor: its only accessors are `harness`, `card_id`, `checkpoint_store`, `checkpoint_policy`, `cwd`, `latest_checkpoint` and `capture_checkpoint`.
No public fake harness exists: `tests/support/mod.rs::ScriptedHarness`/`FakeHarnessFactory` are crate-private test helpers, so an out-of-crate integration test copies a `HarnessProtocol` impl (13 required methods; `send_auto` and `abort_graceful` have defaults).
The minimum viable fake: `state()` returns `Idle` after `start`, `events()` returns `VecEventCursor::new(Vec::new())` (an empty cursor ends the pump cleanly) or a scripted one, and `send` emits `turn_lifecycle` started → `output` (channel `answer`) → `turn_lifecycle` finished before settling back to `Idle`.
All protocol types used by such a fake are reachable from an out-of-crate test through `openjiuwen::harness_protocol`; the crate itself is `openjiuwen::agent_teams`.
A fake-LLM member is also possible without a fake protocol: `NativeMemberFactory::new(Arc<dyn ModelResolver>)` accepts a resolver returning any `Arc<dyn Model>` (for example `openjiuwen_test_support::ScriptedModel`), because the spec only carries a `ModelRef` name and `DeepAgentBuilder::model` takes the trait object.

## Constraints

- Only `crates/openjiuwen-agent-teams/src/harness/**` is edited here; the `member` seam, `lib.rs`, manifests and sibling modules are out of scope (the crate-root `pub use harness::{…}` list is maintained from the parent node).
- `openjiuwen_common::error::{Error, Result}` at the boundary; `HarnessResult` stays inside the protocol calls (the sink adapter maps store failures to `HarnessError::protocol`).
- Async-first tokio; `#[async_trait]` for object-safe async seams; no lock is held across `.await` (observers are cloned out of the `RwLock` before callbacks, the payload cache lock is released before any I/O).
- Exactly one pump task per started runtime, owned by a `PumpHandle`; it ends on `stop`, on drop of the runtime, on cursor exhaustion/error, or when the harness reports `Terminated`. No detached tasks survive `stop`.
- `tracing` only; no `println!`/`eprintln!`; no `unsafe`; no `unwrap`/`expect` outside tests.
- Serde on every public data type (snake_case, unknown fields tolerated); JSON checkpoint only (never pickle).
- Files around ≤1000 lines, split by responsibility (`adapter`/`mapping`/`checkpoint`/`memory_tools`/`native`/`checkpoint_store/*`); `adapter.rs` is the documented exception — it pairs the adapter with its unit tests, roughly half of it `#[cfg(test)]`, and carries a top-of-file length note.

## Routing Table

- `./mod.rs` → public surface and module map
- `./adapter.rs` → `MemberHarnessContext`, `MemberHarnessFactory`, `HarnessMemberRuntime` (command forwarding, observation pump, checkpoint restore/capture; long file: adapter + its unit tests)
- `./mapping.rs` → `HarnessState`/`HarnessEvent` → team status, round and chunk projections
- `./checkpoint.rs` → `NativeCheckpoint` JSON round-trip and compatibility validation
- `./checkpoint_store/` → per-member checkpoint persistence (`mod.rs` module map, `policy.rs` policy/resume, `store.rs` record + memory/file backends, `sink.rs` protocol sink adapter)
- `./native.rs` → `NativeMemberFactory`, `ModelResolver`, `EnvModelResolver`, spec → `DeepAgentConfig` mapping, team-memory tool mounting
- `./memory_tools.rs` → the context's toolkit → `Tool` projection and its `Arc::ptr_eq` identity rule

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)]`), all offline and deterministic; no real model is constructed.
- `mapping.rs` pins every `HarnessState` for both status tables and every `TurnEventKind` for rounds/terminality, plus output/tool chunk shapes.
- `checkpoint.rs` covers JSON round-trip, unknown-field tolerance, missing identity fields, and the session/card incompatibility messages.
- `checkpoint_store/` (27 tests) pins policy defaults/`disabled()`/serde/unknown fields/wire strings/resume mapping; memory round-trip, newest-wins, identity independence, revisions; file round-trip, sanitised `path_for` containment, missing file, malformed JSON; sink payload/cwd/state/outputs/envelope/receipt projection and store-failure mapping.
- The sink's state projection is pinned separately: a known state round-trips, an absent state stays unflagged, an unknown string and a wrong-typed value both restore as `Idle` with the raw value on `unrecognised_state`, the flag survives memory and file round-trips, and a record document with (or without) the field still deserialises.
- `adapter.rs` drives `HarnessMemberRuntime` with a local `FakeHarness` (command recording, recorded `HarnessContext`s, channel-backed cursor, fixed state, scriptable `export_checkpoint`/failure): start publishes `Ready`, delivery/abort/pause/resume argument mapping, observer callback order, cursor drain/close, pump cancellation on `stop`, error translation, sink/restore/capture/stop wiring, a disabled policy writing nothing, and an export error leaving the pump alive. Waits are bounded `yield_now`/1 ms polls.
- `native.rs` covers factory identity, resolver error propagation, the spec → `DeepAgentConfig` mapping (plan mode, card/language overrides), and model-identity propagation: a spec name/provider wins over the resolved one, a resolver-supplied default name reaches `NativeHarness::agent().config()`, and blank resolved names/providers neither erase a spec name nor invent a provider.
- `native.rs` also drives one scripted turn per mounting case and reads the tool names the model request carried (the mounted tools' only observable surface): a context without a toolkit mounts no `team_memory_*` tool, a context with one mounts all four next to the built-in coding tools, and the built-in tools survive the mount.
- `memory_tools.rs` pins the four engine tool names in toolkit order and the identity rule (absent matches only absent, a handle matches itself and no fresh handle).
- `adapter.rs` also pins the context identity: equal `new(...)` arguments build equal contexts, a card id or member name difference breaks equality, attaching a toolkit breaks it, a clone of the same handle keeps it, and the `Debug` view reports presence only.
- Default build: 1470 lib tests (plus the crate's 219 integration tests across 15 binaries, `full_team_sqlite` behind `--features sqlite`, and 37 doctests plus one `ignore`).
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo clippy -p openjiuwen-agent-teams --all-targets --all-features -- -D warnings`, `cargo test -p openjiuwen-agent-teams` in the default, `--no-default-features`, `--features sqlite` and `--all-features` configurations, and `cargo doc -p openjiuwen-agent-teams --no-deps` warning-free.

## Deferred

- Vendor CLI harness providers (Codex, Claude Code) and distributed transports: the factory seam is generic and the vendor adapters stay `Unsupported` in `openjiuwen-harness-providers`, while a process member builds its harness in its own child process.
- Mounting live tools, MCP servers, sub-agents, rails and approval policy from a cold spec — runtime wiring owns those handles.
- Model-pool/IntelliRouter selection — `ModelResolver` is the seam; only the environment resolver ships.

## Notes for Agents

- `adapter.rs` imports `crate::member::{MemberObserver, MemberRuntime}` from the implemented `../member/` seam; import those traits, never add a second definition of them here.
- `NativeCheckpoint` is reachable as `openjiuwen_agent_teams::harness::checkpoint::NativeCheckpoint`, `…::harness::NativeCheckpoint`, `openjiuwen_agent_teams::NativeCheckpoint` (crate root re-export) and `openjiuwen::agent_teams::…` through the umbrella; the only non-test construction sites in this crate are `CheckpointCapture` (`adapter.rs`) and `MemberCheckpointSink::save` (`checkpoint_store/sink.rs`).
- Real persistence is asserted on the store, not on the payload type: drive a runtime with `InMemoryMemberCheckpointStore::shared()` (or `FileMemberCheckpointStore::new(tempdir)`) and read `store.get(session, card)` / `FileMemberCheckpointStore::path_for`; the payload is a different type from `NativeHarness`'s `NativeCheckpointState`, which sits inside `HarnessCheckpoint::data`.
- `HarnessState` comes from `openjiuwen_harness_protocol`; re-export it (crate root does) — never define a second enum.
- The pump deliberately does not end on a terminal turn event: `events()` is the cycle-long view, and a member runs many rounds per cycle. Turn boundaries are where checkpoints are captured, not where the cycle ends.
- Observer callbacks are invoked outside the observer lock; a panicking observer ends the pump task (it is not caught), so observers must be cheap and non-panicking. Checkpoint capture runs after the observer callbacks, so observer order is unchanged.
- Python `on_state`/`on_round` fire from one supervisor writer; here they fire from the single pump task, preserving event order.
- `HarnessMemberRuntime::state()` returns the protocol state verbatim; the team-visible `MemberStatus`/`ExecutionStatus` translations are the pure functions in `mapping.rs`.
- The crate implements `HarnessProtocol` only in `#[cfg(test)]` fakes: it is a protocol *consumer*, and `TeamAgent::runtime()` yields `Arc<dyn MemberRuntime>` with no downcast to `HarnessMemberRuntime`, so the leader's protocol handle cannot be reached for wrapping.
- A team-backed protocol harness lives in `openjiuwen-harness-providers` (`native_v2`, built on `TeamRuntimeManager`), not here; the `NativeV2HarnessProvider` this crate once lacked is that sibling's export, and this crate still implements no `HarnessProtocol` outside `#[cfg(test)]`.
- The crate-root `pub use harness::{…}` list is maintained from the parent node: a new public harness type is reachable as `openjiuwen_agent_teams::harness::…` immediately and from the crate root only after that list is extended.
