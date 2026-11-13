# openjiuwen-agent-teams/src/harness

## Intent

Adapt one team member to the provider-neutral harness protocol and drive it through the member-runtime seam.
Python parity source: `openjiuwen/agent_teams/harness/` (`protocol_adapter.py`, `native_harness.py`, `checkpoint.py`).
The Python adapter downgrades `NativeHarness` to `HarnessProtocol`; in this port `openjiuwen_harness::NativeHarness` already implements `HarnessProtocol`, so this node only builds handles, forwards commands, pumps observations, and persists a portable checkpoint.
The team runtime sees a `MemberRuntime`; this module is the only place that knows about `HarnessProtocol`.

## API Surface

```rust
pub struct MemberHarnessContext { pub team_name: String, pub member_name: String, pub session_id: String,
                                  pub workspace_root: PathBuf, pub language: String, pub mode: MemberMode }
impl MemberHarnessContext { pub fn new(team_name, member_name, session_id, workspace_root, language, mode) -> Self; }

#[async_trait] pub trait MemberHarnessFactory: Send + Sync {
    async fn create(&self, spec: &DeepAgentSpec, ctx: &MemberHarnessContext) -> Result<Arc<dyn HarnessProtocol>>;
    fn name(&self) -> &'static str;
}

pub struct HarnessMemberRuntime;   // impl crate::member::MemberRuntime over Arc<dyn HarnessProtocol>
impl HarnessMemberRuntime {
    pub fn new(harness: Arc<dyn HarnessProtocol>, session_id: impl Into<String>) -> Self;
    pub fn harness(&self) -> &Arc<dyn HarnessProtocol>;
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
- `start(team_session)` closes any previous pump, sends `HarnessContext` (agent ids = the runtime's session id; `host_session_id` = the team session id when supplied, else the runtime session id; empty system prompt because the prompt lives in the constructed agent), publishes the settled state to observers (this is the member's `Ready`), then spawns the pump.
- `send(content, true)` → `DeliveryMode::Steer`; `send(content, false)` → `DeliveryMode::FollowUp`.
- `abort(true)` → `AbortMode::Force`; `abort(false)` → `AbortMode::Graceful`.
- `resume(Some(query))` → `HarnessInput::text(query)`; `resume(None)` → `None`.
- `stop()` cancels/joins the pump, calls `harness.stop()`, then publishes `Terminated`.
- `HarnessError` becomes `Error::External(Box::new(e))`; the protocol has no timeout variant, so there is no timeout special case.

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

Ignored at factory level (logged once at `debug` when present): `tools`, `mcps`, `subagents`, `rails`, `approval_required_tools`, and model `params`.
Mounting live tools/MCP servers/rails belongs to the runtime wiring that owns those handles.
Model resolution is always required: a missing provider/model is `Error::config`.

The model identity must live on the config, not only on the model handle: `DeepAgentBuilder` copies `DeepAgentConfig::{model_name, model_provider}` onto the inner `ReActAgentConfig`, and the agent loop inherits a blank config name into every `ModelRequestConfig` whose request name is empty — an empty name makes provider requests and `list_models` fail (Python parity: `native_harness.py` reads `self._react_agent.config.model_name`).
`api_base`/`api_key` are deliberately left unset by the factory: they belong to the resolver's `ModelClientConfig`, and `DeepAgentConfig` needs them only on the lazy-model path, which never runs because the factory always supplies a live model.

## Injection seam for tests and embedders

`MemberHarnessFactory` is the only way a member's brain is replaced: `TeamBuildDeps::with_harness_factory` swaps the default `NativeMemberFactory::from_env()` for any `Arc<dyn MemberHarnessFactory>`, and `TeamAgentSpec::build_with` creates the leader through it while `TeamMemberSpawner::spawn` creates every teammate through it.
Both calls pass a `&DeepAgentSpec` (leader: `agents["leader"]`; teammate: `agents[name]` else `agents["teammate"]`) plus a `MemberHarnessContext`.
`MemberHarnessContext::session_id` is the **team** session id for every member (leader and teammate alike), so a scripted factory must key its per-member answers on `member_name`, not on `session_id`.
No public fake harness exists: `tests/support/mod.rs::ScriptedHarness`/`FakeHarnessFactory` are crate-private test helpers, so an out-of-crate integration test copies a `HarnessProtocol` impl (13 required methods; `send_auto` and `abort_graceful` have defaults).
The minimum viable fake: `state()` returns `Idle` after `start`, `events()` returns `VecEventCursor::new(Vec::new())` (an empty cursor ends the pump cleanly) or a scripted one, and `send` emits `turn_lifecycle` started → `output` (channel `answer`) → `turn_lifecycle` finished before settling back to `Idle`.
All protocol types used by such a fake are reachable from an out-of-crate test through `openjiuwen::harness_protocol`; the crate itself is `openjiuwen::agent_teams`.
A fake-LLM member is also possible without a fake protocol: `NativeMemberFactory::new(Arc<dyn ModelResolver>)` accepts a resolver returning any `Arc<dyn Model>` (for example `openjiuwen_test_support::ScriptedModel`), because the spec only carries a `ModelRef` name and `DeepAgentBuilder::model` takes the trait object.

## Constraints

- Only `crates/openjiuwen-agent-teams/src/harness/**` is edited here; the `member` seam, `lib.rs`, manifests and sibling modules are out of scope.
- `openjiuwen_common::error::{Error, Result}` at the boundary; `HarnessResult` stays inside the protocol calls.
- Async-first tokio; `#[async_trait]` for object-safe async seams; no lock is held across `.await` (observers are cloned out of the `RwLock` before callbacks).
- Exactly one pump task per started runtime, owned by a `PumpHandle`; it ends on `stop`, on drop of the runtime, on cursor exhaustion/error, or when the harness reports `Terminated`. No detached tasks survive `stop`.
- `tracing` only; no `println!`/`eprintln!`; no `unsafe`; no `unwrap`/`expect` outside tests.
- Serde on every public data type; unknown fields tolerated; JSON checkpoint only (never pickle).
- Files ≤1000 lines; split by responsibility (`adapter`/`mapping`/`checkpoint`/`native`).

## Routing Table

- `./mod.rs` → public surface and module map
- `./adapter.rs` → `MemberHarnessContext`, `MemberHarnessFactory`, `HarnessMemberRuntime` (command forwarding, observation pump)
- `./mapping.rs` → `HarnessState`/`HarnessEvent` → team status, round and chunk projections
- `./checkpoint.rs` → `NativeCheckpoint` JSON round-trip and compatibility validation
- `./native.rs` → `NativeMemberFactory`, `ModelResolver`, `EnvModelResolver`, spec → `DeepAgentConfig` mapping

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)]`), all offline and deterministic; no real model is constructed.
- `mapping.rs` pins every `HarnessState` for both status tables and every `TurnEventKind` for rounds/terminality, plus output/tool chunk shapes.
- `checkpoint.rs` covers JSON round-trip, unknown-field tolerance, missing identity fields, and the session/card incompatibility messages.
- `adapter.rs` drives `HarnessMemberRuntime` with a local `FakeHarness` (command recording, channel-backed cursor, fixed state): start publishes `Ready`, delivery/abort/pause/resume argument mapping, observer callback order, cursor drain/close, pump cancellation on `stop`, and error translation. Waits are bounded `yield_now`/1 ms polls.
- `native.rs` covers factory identity, resolver error propagation, the spec → `DeepAgentConfig` mapping (plan mode, card/language overrides), and model-identity propagation: a spec name/provider wins over the resolved one, a resolver-supplied default name reaches `NativeHarness::agent().config()`, and blank resolved names/providers neither erase a spec name nor invent a provider.
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings`, `cargo test -p openjiuwen-agent-teams [--no-default-features]` green.

## Deferred

- Vendor CLI harness providers (Codex, Claude Code) and distributed transports: the factory seam is generic and the vendor adapters stay `Unsupported` in `openjiuwen-harness-providers`, while a process member builds its harness in its own child process.
- Provider-side checkpoint export/import plumbing (`HarnessProtocol::export_checkpoint` is called only by callers that opt in; the runtime does not poll it).
- Mounting live tools, MCP servers, sub-agents, rails and approval policy from a cold spec — runtime wiring owns those handles.
- Model-pool/IntelliRouter selection — [`ModelResolver`] is the seam; only the environment resolver ships.

## Notes for Agents

- `adapter.rs` imports `crate::member::{MemberObserver, MemberRuntime}` from the implemented `../member/` seam; import those traits, never add a second definition of them here.
- `HarnessState` comes from `openjiuwen_harness_protocol`; re-export it (crate root does) — never define a second enum.
- The pump deliberately does not end on a terminal turn event: `events()` is the cycle-long view, and a member runs many rounds per cycle.
- Observer callbacks are invoked outside the observer lock; a panicking observer ends the pump task (it is not caught), so observers must be cheap and non-panicking.
- Python `on_state`/`on_round` fire from one supervisor writer; here they fire from the single pump task, preserving event order.
- `HarnessMemberRuntime::state()` returns the protocol state verbatim; the team-visible `MemberStatus`/`ExecutionStatus` translations are the pure functions in `mapping.rs`.
