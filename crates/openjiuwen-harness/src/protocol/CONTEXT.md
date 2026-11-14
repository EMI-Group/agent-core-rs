# protocol

## Intent

The native implementation of the provider-neutral harness SPI (`openjiuwen_harness_protocol`) over `DeepAgent`: one `NativeHarness` per agent exposing start/stop, ordered events, turns, send/abort/pause/resume, checkpoints and model control, plus `NativeHarnessProvider` for discovery and creation.
Ports the observable behavior of the Python native harness providers (`openjiuwen/harness_providers/base.py` `SerializedTurnHarness` + `native/`), not their internals — the kernel `DeepAgent` owns the model/tool loop, this node owns the protocol contract.

## API Surface

- `HARNESS_NAME` (`"openjiuwen-native"`), `PROVIDER_NAME` (`"native"`), `PROVIDER_VERSION`.
- `native_card(name, version) -> HarnessCard` — capability set below; the provider and every harness instance share it.
- `NativeHarness` (`Clone`, thin wrapper over `Arc<inner>`): `new(Arc<DeepAgent>, NativeHarnessOptions)` returns an **unstarted** harness (`state() == Terminated`), `agent()`, `options()`, `started()`.
- Implements `HarnessProtocol`: `card`, `state`, `provider_session_id`, `event_buffer_config`, `start`, `stop`, `events`, `turn_events`, `send`, `abort`, `pause`, `resume`, `export_checkpoint`.
- Implements `HarnessModelControl`: `list_models`, `set_model`.
- `NativeHarnessOptions { event_buffer_capacity: u64 /*1024*/, overflow: EventOverflowPolicy /*CoalesceOrBlock*/ }` with `new()` and `with_*` builders.
- `NativeHarnessProvider` (`Default`, `new()`, `with_options(NativeHarnessOptions)`) implements `HarnessProvider` (`card`, `create(JsonObject)`) and `ModelConfigProvider` (`model_config(choice, fallback)`).
- `NativeCheckpointState` — serde payload stored in `HarnessCheckpoint::data` (schema version, session scope, iteration, pending follow-ups, plan mode, turn counter, current model/effort); `new(session_id, agent_id, host_session_id)`, `to_json() -> JsonObject`, `from_json(&JsonObject) -> HarnessResult<Self>` (rejects a foreign schema with `Validation("unsupported native checkpoint version")`).
- Everything else (`inner`, `cursor`, `events`, `bridge`, `turn`) is crate-private.
- Module file map: `mod.rs` (struct, lifecycle, `HarnessProtocol`/`HarnessModelControl` impls), `inner.rs` (shared protocol state and lock-scoped helpers), `turn.rs` (one turn task), `events.rs` (kernel frame/output → protocol payload mapping), `cursor.rs` (`EventChannel` + `ChannelEventCursor`), `bridge.rs` (sequence/envelope/turn ids, checkpoints), `provider.rs` (`NativeHarnessProvider`), `tests.rs` + `tests/` (offline suite).
- `provider_session_id()` is `None` while `Terminated`; the id is the first successful `start`'s host session id and is stable for the harness lifetime (`OnceLock`), so restarting with a different host session id is `State("native harness cannot be restarted with a different host session")`.

## Capabilities

Declared `HarnessCapability`: `steer`, `graceful_abort`, `force_abort`, `pause_resume`, `persistent_session`, `checkpoint`, `native_tools`, `mcp_tools`, `model_selection`, `model_discovery`.
Not declared: `hooks`, `endpoint_models`, `auth_fallback` — the matching operations/choices return `HarnessError::UnsupportedCapability`.
`required_host_capabilities`: none; `optional_host_capabilities`: `checkpoint_sink`, `tool_approval`, `user_input`, `telemetry`.
`start` therefore never rejects a host for missing capabilities, but always runs `HarnessCard::validate_host` (protocol version + required caps) first.

## Lifecycle semantics

- Unstarted / stopped state is `Terminated` (Python base parity); `start` emits `StateChanged(Terminated → Idle)`, opens a fresh event cycle and resets the envelope sequence to 1.
- `start` twice without `stop` → `HarnessError::State("native harness is already started")`.
- `start` rejects non-empty `HarnessContext::mcp_servers` with `UnsupportedCapability("declare MCP servers in the agent manifest; the native harness does not mount context MCP servers")`.
- `start` creates one `AgentSession` (`session_id == context.host_session_id`, `agent_id == context.agent_id`) reused by every turn of the cycle; `provider_session_id()` is that session id (`Some` only between start and stop).
- `start` applies `ResumePolicy`: `New` ignores a supplied checkpoint; `RequireResume` requires one (`Protocol("native harness requires a checkpoint to resume")`); `ResumeIfAvailable` restores when decodable and cold-starts with a `warn!` otherwise.
  A restored checkpoint must match `agent_id` + `host_session_id` (`Protocol("native checkpoint scope does not match the context")`) and `schema_version == "1"` (`Protocol("unsupported native checkpoint version")`); undecodable data is `Protocol("invalid native checkpoint")`.
- `start` calls `agent.ensure_initialized()` so a model-less agent fails at start, not at the first turn (`openjiuwen_common::Error` → `HarnessError::Protocol(message)`).
- `stop` is idempotent, aborts the active turn, awaits the turn task, emits a synthetic `Started` + `Aborted` pair for every still-queued turn (`"native harness stopped before the queued turn started"`), emits `StateChanged(→ Terminated)` and closes the event cycle (cursors drain then end).
- `send` never waits for a turn: a non-steer input either queues behind the active turn or spawns the turn task; the receipt is returned immediately (`message-<n>`, `turn-<n>`).
- Delivery modes: explicit `Steer` requires a running turn (`State("there is no active turn to steer")`) and returns the active turn id; `Auto` becomes steer while a turn runs, otherwise follow-up; a queued input is reported as `FollowUp`.
- Turn ids are `turn-<n>` and message ids `message-<n>` from monotonic per-cycle counters (documented deviation from Python's uuid hex — deterministic tests).
- Pause/resume are harness-level (the kernel exposes no pause API): `pause()` with no active turn emits `StateChanged(→ Paused)` and defers queued turns; with an active turn the turn task emits `StateChanged(→ Pausing)`, `TurnLifecycle(Paused)` at its next frame boundary, parks on a `Notify`, and `resume` emits `TurnLifecycle(Resumed)` + `StateChanged(→ Running)` and continues.
  `resume(Some(input))` also enqueues and starts that input; `resume` on a non-paused harness is `State("native harness is not paused")`.
- `abort(mode)` gates on `force_abort`/`graceful_abort`, is a no-op without an active turn, otherwise records the mode, calls `DeepAgent::abort()` and lets the turn terminate as `Aborted` with `Interrupted`/`UserAbort` (`HarnessStop` for a stop-driven abort).
- Only one observation cursor is active at a time: a second `events()`/`turn_events()` returns a cursor whose `next_event` is `HarnessError::State("the harness observation stream already has an active consumer")`; a cycle that was never started / already stopped yields `State("native harness has no active event cycle")`.

## Event mapping

Emitted payloads (all envelopes carry monotonic `sequence`, `f64` Unix-epoch-second `timestamp`, host session id, agent id, provider session id, turn id, `item_id`, `correlation_id = message id`, empty `causation_ids`):
- `StateChangedEvent` on every state transition.
- `TurnLifecycleEvent` `Started` (after `StateChanged(→ Running)`), optional `Paused`/`Resumed`, terminal `Finished`/`Aborted`/`Failed` carrying the `TurnResult`.
- `OutputEvent` for `llm_output` (`kind: Text`, `operation: Delta`, `channel: Answer`, `output_id = "native-output:<turn_id>:answer"`) and `llm_reasoning` (`Reasoning`, `...:reasoning`); `content_index` counts output events per channel.
- `UsageUpdatedEvent` from each `llm_usage` frame (`mode: Delta`), parsed from the kernel `usage_metadata` object.
- `ItemLifecycleEvent` `Started` + `Completed` per recorded tool observation at turn end (`item_type: "tool"`, `item_id` = tool call id, data carries tool name and content) — documented coarse granularity: the kernel stream has no live tool frames.
- `DiagnosticEvent` (`level: Error`) when a turn fails with an `AgentOutput::error`.
- `ProviderEvent` for any other kernel frame kind (`provider: "native"`, `event_type` = frame kind, `schema_version: "1"`, `payload` = frame payload); `answer`, `__interaction__` and `context.usage` frames produce no event.

## Turn result

`TurnResult` is built from the terminal `AgentOutput`: `Answer` → `Completed` (`stop_reason` = finish reason else `"answer"`), `Interrupt` → `Completed` with `stop_reason: "interrupt"` and `provider_data.pending_interrupt_ids`, `Error` → `Failed` with `TurnError { message, code: "sdk_error", category: "sdk_error" }`, abort/stop → `Interrupted` + `TurnTermination { UserAbort | HarnessStop }`.
Messages: one assistant message (`"<turn_id>:assistant"`) with text/reasoning blocks plus one tool message per observation (`"<turn_id>:tool:<call_id>"`).
Usage: accumulated `llm_usage` frames (`None` when the model reported nothing). `provider_data` carries `iterations_used`, `result_type` and pending interrupt ids. Timestamps come from `SystemTime`.

## Checkpoints

`NativeCheckpointState` is the only checkpoint payload (schema version `"1"`): `session_id`, `agent_id`, `host_session_id`, `iteration`, `pending_follow_ups`, `plan_mode`, `turn_counter`, optional `model`/`effort`.
Checkpoints are emitted best-effort (`SessionActivated` after start, `TurnCompleted` after each turn, `StateChanged` on pause) through `HarnessContext::checkpoint_sink` with monotonic `sequence` and the last successful `storage_revision` as the compare-and-set expectation; a `CheckpointConflict` is logged, never fatal, and does not replace the latest local checkpoint.
`start` applies a resumed checkpoint before publishing `SessionActivated`, so that checkpoint already reports the restored state, and it raises the checkpoint writer to at least the resumed checkpoint's `sequence` so the sequence never restarts at 1 on the same host session (event sequencing still resets per cycle).
`export_checkpoint` returns the latest locally built checkpoint (`None` before the first one).
`stop` publishes no checkpoint; the only reasons emitted are `SessionActivated`, `TurnCompleted` and `StateChanged`.
Each checkpoint's `revision` is the previous successful storage revision (the compare-and-set expectation), not the revision the pending write produces.

## Model control

`list_models` returns one `ModelOption` (`is_default: true`) for the configured `model_name`, or an empty list when the agent has no model name.
`set_model` validates the selection, writes `model_name` through `DeepAgent::configure`, and records `effort`; recognized efforts (`low`/`medium`/`high`) also map onto `ModelRequestConfig::temperature` (`0.2`/`0.6`/`1.0`) when a model name is known, and an unrecognized effort is `Validation("unsupported reasoning effort '<x>'")`.
The provider's `model_config` returns the create-config fragment (`model`, `temperature` from effort) and rejects an endpoint choice (`UnsupportedCapability("openjiuwen-native does not take endpoint models")`) or a fallback (`UnsupportedCapability("openjiuwen-native has no authentication fallback")`).

## Provider create config

Known keys, all optional: `model`, `provider`, `api_key`, `api_base`, `temperature` (`0.0..=2.0` → `ModelRequestConfig::temperature`), `workspace` (root + cwd), `enable_task_loop`, `max_iterations` (positive), `system_prompt`, `permissions` (`PermissionsSection`), `language`.
Unknown keys → `Validation("unknown native harness configuration fields: <sorted, ', '>")`; wrong types → `Validation("<key> must be a ...")`; blank string options → `Validation("native harness string options must be non-empty strings")`.
`create` builds `DeepAgent` synchronously with `DeepAgentBuilder` (card `AgentCard::new("deep_agent")`, no model → lazily resolved by `ensure_initialized`) and returns an unstarted `NativeHarness`.
`create` cannot take a pre-built `Model`/`DeepAgent`: a caller that already holds one skips the provider and calls `NativeHarness::new(Arc<DeepAgent>, NativeHarnessOptions)` directly, building the agent with `DeepAgentBuilder::model(Arc<dyn Model>)`.
`temperature` is written to `ModelRequestConfig::temperature` only when `model` is also set; without a model name it is dropped with a `warn!` (a `ModelRequestConfig` always needs a model name).

## Offline driving

- `NativeHarness::new` is the offline path: build the `DeepAgent` with `DeepAgentBuilder` + `ScriptedModel` + temp `Workspace`, then `start(HarnessContext::new(...))` (there is no `HarnessContext::default`).
- A provider-created (`HarnessProvider::create`) harness carries no model handle: `start` resolves one lazily from `model`/`provider`/`api_base`/`api_key`, so turns need a real model even though `start` itself stays offline (a provider test starts against `https://example.invalid/v1`).
- `create` hands back `Box<dyn HarnessProtocol>`: that handle exposes neither `agent()` nor `HarnessModelControl`.
- `HarnessInput.content` is a JSON string used verbatim; a JSON object contributes its `query` string; anything else is stringified (`content_text`).
- One observation cursor at a time: take `harness.events()` once and read `next_event()` until `None`; a second cursor reports `State`.
- Answer text arrives as `HarnessEventPayload::Output` deltas (`channel == OutputChannel::Answer`, `content` a `String`, `operation == Delta`) and the turn ends with a `TurnLifecycle` payload whose `kind` is in `TERMINAL_TURN_EVENT_KINDS` and whose `result` is the `TurnResult`.

## Host service usage

- `start` consumes only `protocol_version`, `host_capabilities`, `agent_name`, `agent_id`, `host_session_id`, `system_prompt`, `resume_policy`/`checkpoint`, `checkpoint_sink` and `mcp_servers` from `HarnessContext`.
- `interactions`, `hooks`, `tools`, `telemetry`, `cwd`, `env` and `metadata` are ignored: `ToolApproval`/`UserInput`/`Telemetry` are advertised as optional host capabilities but never called.
- There is no seam for seeding prior conversation messages or restoring a `persistent_agent`/`ResumePlan`/`PersistentSessionRecord`: each cycle builds a fresh `AgentSession` and a turn is one `HarnessInput`, while `NativeCheckpointState` carries no messages.
- The module has no occurrences of `persistent_agent`, `ResumePlan`, `PersistentSessionRecord` or `WindowAssembly`; `openjiuwen_harness::persistent_agent` is a sibling tree the protocol node does not consume.
- A permission `ask` surfaces as a kernel interrupt (`PermissionInterruptRail`) and reaches the host only through the turn's `TurnResult` (`stop_reason: "interrupt"`, `provider_data.pending_interrupt_ids`); the `__interaction__` frame maps to no event and no `HarnessInteractionRequest` is ever issued.

## Constraints

- Harness methods take `&self`; all mutable state lives behind `parking_lot::Mutex`/atomics on the shared inner struct, never held across `.await` (a dedicated `tokio::sync::Mutex` serializes turn start).
- `send` spawns a tracked `tokio::task::JoinHandle` (joined by `stop`); event emission is `async` because the bounded channel applies the `EventBufferConfig::overflow` policy (`Block` awaits, coalescible/best-effort events are dropped when full — documented approximation of coalescing).
- No duplicate event/result types: everything comes from `openjiuwen-harness-protocol`.
- Out-of-crate visibility is exactly `NativeHarness`, `NativeHarnessOptions`, `native_card`, `HARNESS_NAME`/`PROVIDER_NAME`/`PROVIDER_VERSION`, `NativeCheckpointState` and `NativeHarnessProvider`; `inner`, `cursor`, `events`, `bridge` and `turn` are private modules (their helpers are `pub(crate)`), so integration tests in `crates/openjiuwen-harness/tests/` must assert via emitted events, `export_checkpoint`, and `NativeHarness::agent()` — never via the mapping helpers.
- No network, no `unsafe`, `tracing` only, public items documented, files ≤1000 lines, no silent `unwrap`/`expect` outside tests.

## Routing Table

- `./mod.rs` → `NativeHarness`, `NativeHarnessOptions`, `native_card`, capability constants, `HarnessProtocol` + `HarnessModelControl` impls
- `./inner.rs` → `NativeHarnessInner`, `ProtocolState`, `PendingInput`, `ActiveTurn`, emission/state/checkpoint helpers
- `./turn.rs` → one turn task: spawn, frame consumption, pause/abort handling, chained queued turns, terminal events
- `./events.rs` → kernel chunk/`AgentOutput` → protocol payload mapping and `TurnResult` construction
- `./cursor.rs` → `EventChannel`, `ChannelEventCursor`, single-consumer lease, overflow policy, cycle/turn views
- `./bridge.rs` → envelope builder (sequence/timestamp/ids), `NativeCheckpointState`, checkpoint build/persist/restore
- `./provider.rs` → `NativeHarnessProvider`, create-config parsing, `ModelConfigProvider`
- `./tests.rs`, `./tests/` → offline suite driving `DeepAgent` over `ScriptedModel` and temp workspaces (`lifecycle`, `turns`, `checkpoints`, `provider`, `model_control`)

## Test Strategy

- Start/stop: protocol-version and required-capability rejection, no initial state change before start, `Terminated → Idle` event, double start, MCP-server rejection, idempotent stop with cursor exhaustion.
- Send/turns: receipt returned before the turn completes, `Started`/`Finished` lifecycle, `turn_events` finiteness (terminal included, `Paused`/`Resumed` not terminal), queued follow-up chained into a second turn, steering requires a running turn.
- Cursors: second cursor rejected with `State`, cycle view survives turn boundaries, turn view skips unrelated events, closed cycle ends cursors.
- Capability gating and state errors: abort modes, pause/resume (queued-turn deferral), `send` after stop.
- Checkpoints: `SessionActivated` emission, `export_checkpoint` shape, restore under `RequireResume` (scope mismatch, bad schema, undecodable data), `ResumeIfAvailable` cold start.
- Provider: `create` from JSON (all keys), unknown key/type errors, `model_config` fragment + endpoint/fallback gating, `card()` identity.
- Model control: `list_models` with/without a model name, `set_model` model swap + effort mapping + invalid effort.
- All tests are offline, deterministic, `#[tokio::test]`, `ScriptedModel`, temp workspaces; `#![allow(clippy::unwrap_used, clippy::expect_used)]` at the top of `tests.rs`.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings`.
