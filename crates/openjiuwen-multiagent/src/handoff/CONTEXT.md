# openjiuwen-multiagent :: handoff

## Intent

Sequential agent-to-agent control transfer, ported from Python `openjiuwen/core/multi_agent/teams/handoff/` (v0.1.20).
A handoff chain runs exactly one member at a time; the member either answers (chain ends) or emits a transfer signal (next member continues with the accumulated history).
Everything is in-process and tokio-native: endpoints are ordinary bus subscribers and the coordinator is a per-session rendezvous object.

## API Surface

```rust
pub struct HandoffRoute { pub source: String, pub target: String }
pub struct HandoffSnapshot { pub current_agent_id: String, pub handoff_count: usize }
pub type TerminationCondition = Arc<dyn Fn(HandoffSnapshot) -> BoxFuture<'static, Result<bool>> + Send + Sync>;
pub struct HandoffConfig { pub start_agent: Option<AgentCard>, pub max_handoffs: usize /*10*/,
                           pub routes: Vec<HandoffRoute>, #[serde(skip)] pub termination_condition: Option<TerminationCondition> }
pub struct HandoffTeamConfig { #[serde(flatten)] pub team: TeamConfig, pub handoff: HandoffConfig }
pub fn handoff_team(card: TeamCard, config: HandoffTeamConfig) -> BaseTeam;
pub struct HandoffStrategy { /* config, registry, ready, setup_lock */ }
impl HandoffStrategy { pub fn new(config: HandoffTeamConfig) -> Arc<Self>; }
pub struct ContainerAgent { /* target_card, target_provider, allowed_targets, team_card_id,
                              endpoint_id, runtime: Weak<TeamRuntime>, registry, tools_injected */ }
impl ContainerAgent {
    pub fn new(target_card: AgentCard, target_provider: AgentProvider, allowed_targets: Vec<String>,
               team_card_id: String, endpoint_id: String, runtime: Weak<TeamRuntime>,
               registry: Arc<HandoffRegistry>) -> Self;
    pub fn endpoint_id(&self) -> &str;
    pub fn allowed_targets(&self) -> &[String];
}
pub struct HandoffHistoryEntry { pub agent: String, pub output: Value }
pub struct HandoffRequest { pub input_message: Value, pub history: Vec<HandoffHistoryEntry> }
impl HandoffRequest { pub fn new(input_message: Value, history: Vec<HandoffHistoryEntry>) -> Self;
                      pub fn to_value(&self) -> Value; pub fn from_value(value: &Value) -> Result<Self>; }
pub struct HandoffRegistry { /* session id -> coordinator */ }
impl HandoffRegistry { pub fn new() -> Self; pub fn insert(&self, session_id: impl Into<String>, coordinator: Arc<HandoffOrchestrator>);
                       pub fn get(&self, session_id: &str) -> Option<Arc<HandoffOrchestrator>>;
                       pub fn remove(&self, session_id: &str) -> Option<Arc<HandoffOrchestrator>>; }
pub struct HandoffOrchestrator { .. }
impl HandoffOrchestrator { pub fn new(..); pub fn build_route_graph(..); pub fn current_agent_id(&self) -> String;
                           pub fn handoff_count(&self) -> usize; pub async fn request_handoff(&self, target: &str, reason: Option<&str>) -> Result<bool>;
                           pub async fn complete(&self, result: Value) -> bool; pub async fn error(&self, error: Error) -> bool;
                           pub async fn wait(&self, timeout: Option<Duration>) -> Result<Value>;
                           pub async fn save_to_session(&self, session: &Arc<AgentTeamSession>) -> Result<()>;
                           pub fn restore_from_session(snapshot: Option<&Value>, start: impl Into<String>,
                                                       members: &[String], config: &HandoffConfig) -> Result<Self>; }
pub struct HandoffTool { .. }        // `transfer_to_<id>`, never errors
pub struct HandoffSignal { pub target: String, pub message: Option<String>, pub reason: Option<String> }
pub struct TeamInterruptSignal { pub result: Value, pub message: Option<String> }
pub fn extract_handoff_signal(result: &Value) -> Option<HandoffSignal>;
pub fn extract_handoff_signal_from_observations(observations: &[Observation]) -> Option<HandoffSignal>;
pub async fn extract_handoff_signal_from_session(result: &Value, session: &Arc<dyn Session>) -> Option<HandoffSignal>;
pub fn extract_interrupt_signal(result: &Value) -> Option<TeamInterruptSignal>;
pub async fn flush_team_session(session: Option<&Arc<AgentTeamSession>>);
pub const HANDOFF_TARGET_KEY: &str = "__handoff_to__";
pub const HANDOFF_MESSAGE_KEY: &str = "__handoff_message__";
pub const HANDOFF_REASON_KEY: &str = "__handoff_reason__";
pub const COORDINATOR_STATE_KEY: &str = "__handoff_coordinator__";
pub const HISTORY_STATE_KEY: &str = "__handoff_history__";
pub const CONTEXT_HISTORY_STATE_KEY: &str = "__handoff_ctx_history__";
```

`HandoffStrategy` implements `TeamStrategy`: `kind() == "handoff"`, `on_member_added`/`on_member_removed` clear the readiness flag, `setup` wires one endpoint per member, `invoke` drives a chain to completion, and `stream` drives it while the containers write the frames.
`handoff_team` is `BaseTeam::with_strategy(card, config.team.clone(), HandoffStrategy::new(config))`.

## Constraints

- The `config`, `request`, `signal`, `tool`, `interrupt`, and `orchestrator` modules are the stable data/parse layer the endpoint and strategy build on; their public behaviour is fixed.
- Files stay under ~1000 lines: `container.rs` holds the endpoint only, its unit tests live in `container_tests.rs` (declared with `#[cfg(test)] #[path = "container_tests.rs"] mod tests;`).
- No `unsafe`, no stdout logging (`tracing` only), no `unwrap`/`expect`/`panic` outside `#[cfg(test)]`.
- Locks are never held across `.await`; the registry uses `parking_lot::RwLock` for short sections and `HandoffStrategy::setup_lock` is a `tokio::sync::Mutex`.
- Every public item and field is documented; the crate runs clippy with `-D warnings` including `missing_docs`.
- The crate root re-exports the whole handoff surface, so every public item here is reachable from `openjiuwen_multiagent`.
- Tests are offline and deterministic, only short durations (`message_timeout` ≈ 10 ms against a 50 ms member).

## Design Decisions

- **Next-hop publish is spawned (re-entrancy)**: a container runs inside the session's per-topic FIFO bus worker, which awaits the pub/sub handler inline; awaiting `runtime.publish(..)` for the next hop there would enqueue onto the topic the worker cannot drain, i.e. deadlock. The first publish comes from the caller task (a normal `.await`), every follow-up hop is issued from `tokio::spawn` and a failure is logged at `warn` instead of raised (the chain then ends through its deadline).
- **The registry is keyed by team session id**, so an endpoint resolving `session.parent_id()` finds the live `HandoffOrchestrator` of its chain; `drive` removes the entry and calls `runtime.cleanup_session(sid)` on every exit path.
- **Endpoints are ordinary runtime members**: id `__handoff_ep_{team_card_id}_{agent_id}`, subscribed to the pub/sub topic `container_{agent_id}`; every endpoint is registered before the start publish, because `publish` validates subscribers' registration.
- **`ContainerAgent::invoke` always returns `Ok(AgentOutput::answer("{}"))`**; results reach consumers only through frames written to the team stream, matching the Python design where the container is a control shuttle, not a result carrier.
- **Tool injection happens at most once per endpoint**: the `tools_injected` flag is set before the tool host is probed, so a host-less member is probed once and a rejected mount is only logged (Python parity).
- **Signal extraction is observations-first**: `AgentOutput.answer(..)` carries no top-level signal key, so the order is result object, then `observations` (newest first), then the session `context` tool messages.
- **Session signal extraction is JSON-only** (no Python `ast.literal_eval` fallback); a non-JSON tool message is ignored.
- **Restore tolerates partial snapshots**: a missing/invalid coordinator snapshot starts a fresh coordinator, and a malformed history array is logged and treated as empty; on resume, entries whose output has `result_type == "interrupt"` are dropped because the interrupted hop is replayed.
- **Context mirroring is best effort**: before a hop the team history is copied into the child session's `context.default_context_id.messages`; after the hop the cleaned child messages (no `tool` role, no assistant `tool_calls`) are deduped against the team history and appended. State errors are logged, never fatal.
- **The stream strategy writes no frames itself**: `stream` delegates to `crate::session::team_stream` and the containers produce every frame (`{"type":"message","payload":..}`), so `invoke` and `stream` share one code path.
- **`message_timeout` is the chain deadline**: `0.0` means unbounded; a deadline expiry is rewritten to `handoff chain timeout after {seconds}s, team='{card_id}'` (`ErrorKind::Timeout`).
- **A caller-provided team session must already be bound** (`runtime.bind_team_session(..)`) before `invoke`/`stream`, otherwise containers cannot resolve `runtime.team_session(sid)` and the chain fails with `coordinator not found ...`; `SessionLease::finish` only ends the stream for a caller session and never unbinds it.

## Routing Table

- `./mod.rs` → module docs (flow + re-entrancy), the six state/signal keys, module declarations, the re-exports, `HandoffStrategy` (setup/wiring/drive/invoke/stream) and `handoff_team`.
- `./config.rs` → `HandoffConfig`, `HandoffRoute`, `HandoffSnapshot`, `TerminationCondition`, `HandoffTeamConfig` and their serde defaults.
- `./request.rs` → `HandoffRequest` (wire form), `HandoffHistoryEntry`.
- `./signal.rs` → `HandoffSignal` and the three extractors (result, observations, session).
- `./tool.rs` → `HandoffTool`, the `transfer_to_<id>` tool a model calls to hand off.
- `./interrupt.rs` → `TeamInterruptSignal`, `extract_interrupt_signal`, `flush_team_session`.
- `./orchestrator.rs` → `HandoffOrchestrator` (route graph, hop cap, termination, rendezvous) and `HandoffRegistry`.
- `./container.rs` → `ContainerAgent`, the per-hop endpoint: tool injection, input shaping, target invocation, context mirroring, stream writes, interrupt forwarding, next-hop publish.
- `./container_tests.rs` → `ContainerAgent` unit tests (12), declared from `container.rs` as `mod tests`.

## Test Strategy

- `container_tests.rs` (12 tests, local fakes from `handoff::support`): missing session ⇒ External `session_id is empty`; no coordinator ⇒ External `coordinator not found`; a successful hop returns `"{}"`, writes a `message` frame and lets `coordinator.wait(None)` yield the target output; an observation-driven handoff advances `current_agent_id`, an unknown/blocked target completes the chain; an interrupt output completes the coordinator and persists `COORDINATOR_STATE_KEY`/`HISTORY_STATE_KEY`; a failing target reports the error and still returns `Ok`; tool injection mounts `transfer_to_<target>` exactly once on a stable `ReActAgent` and skips a host-less member; history shaping for object/string/empty history.
- `mod.rs` (9 tests) end to end through `handoff_team` + `BaseTeam::add_agent` + `Team::invoke`/`Team::stream`: single hop, two-hop chain via a spawned publish, blocked route, `max_handoffs = 0`, streamed frames, `message_timeout` timeout rewrite, and a skipped unregistered provider.
- A fake agent emits a handoff signal by returning an `Observation` whose `content` is the JSON signal (`{"__handoff_to__": ..}`), not through `AgentOutput::answer`.
- Run the crate tests offline: `cargo test -p openjiuwen-multiagent --offline`; the handoff unit tests are `cargo test -p openjiuwen-multiagent --lib --offline handoff`.

## Known Deviations

- Session identity travels on the bus envelope (the child session's `parent_id()`), not inside `HandoffRequest` as in Python's object-carrying dataclass.
- Signal extraction reads `AgentOutput.observations` before the session context; Python relies on the context engine's tool messages alone.
- Python's `ast.literal_eval` repr fallback is not ported: session signals are JSON only.
- `setup` skips a member whose provider or card vanished between registration and wiring (logged at `warn`), where Python would raise on attribute access.
- `max_handoffs`/termination are enforced by the shared `HandoffOrchestrator`, so a blocked hop completes with the current result instead of returning Python's error string.
- The endpoint provider returns a fresh `TeamAgent` per resolution but the same `ContainerAgent` prototype clone, so `tools_injected` is shared across resolutions.
