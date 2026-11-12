# protocol

## Intent

The native implementation of the provider-neutral harness SPI (`openjiuwen_harness_protocol`): one `NativeHarness` per `DeepAgent` exposing start/send/stop, ordered events, turns, abort/pause/resume, and checkpoints, plus `NativeHarnessProvider` for discovery/creation.
Ports the observable behavior of the Python native harness provider on top of `DeepAgent` (see `openjiuwen/harness_protocol/` DESIGN.md for the wire contract).

## API Surface

- `NativeHarness`: implements `harness_protocol::HarnessProtocol` (`card`, `state`, `provider_session_id`, `event_buffer_config`, `start`, `stop`, `events`, `turn_events`, `send`, `abort`, `pause`, `resume`, `export_checkpoint`) and `HarnessModelControl` (`list_models`, `set_model`).
- `NativeHarness::new(agent: Arc<DeepAgent>, options) -> Self` — unstarted harness; `started()` validates `HarnessCard::validate_host` and applies `ResumePolicy` to the supplied checkpoint.
- `NativeHarnessProvider`: implements `HarnessProvider` (`card`, `create(JsonObject) -> Box<dyn HarnessProtocol>`) and `ModelConfigProvider` (`model_config(choice, fallback)`).
- Declared capabilities: `steer`, `graceful_abort`, `force_abort`, `pause_resume`, `persistent_session`, `checkpoint`, `native_tools`, `mcp_tools`, `model_selection`, `model_discovery`. Required host capabilities: none; optional: `checkpoint_sink`, `tool_approval`, `user_input`, `telemetry`.
- `events.rs`: mapping of `AgentChunk`/`AgentOutput`/`DeepLoopEvent` to `HarnessEventPayload` (`TurnLifecycleEvent`, `ItemLifecycleEvent`, `OutputEvent`, `StateChangedEvent`, `UsageUpdatedEvent`, `DiagnosticEvent`, `HookObservedEvent`, `ProviderEvent`).
- `cursor.rs`: `ChannelEventCursor` — tokio mpsc-backed `HarnessEventCursor` with a single-consumer lease, bounded buffer and `EventBufferConfig::overflow` policy; `turn_events(turn_id)` filters one finite turn.
- `bridge.rs`: turn ids, sequence counter (monotonic per host session), timestamps (Unix epoch seconds), and `TurnResult` construction from `AgentOutput`.

## Constraints

- Harness methods take `&self`; all mutable state is synchronized internally (`parking_lot::Mutex`/`tokio::sync::Mutex`, never held across `.await` except a dedicated run lock).
- `send` returns a `SendReceipt` immediately; delivery modes map to `DeepAgent::{follow_up, steer}`; `DeliveryMode::Auto` picks steer while a turn runs, follow-up otherwise.
- Only one active event cursor: a second `events()`/`turn_events()` cursor returns `HarnessError::State`; `stop` is idempotent and closes cursors.
- Capability gating: undeclared operations return `HarnessError::UnsupportedCapability`; pause/resume map to the task-loop pause/resume; `abort` maps to `DeepAgent::abort`.
- Checkpoints are opaque JSON (`NativeCheckpointState`) emitted to `HarnessContext::checkpoint_sink` with monotonically increasing `sequence`; `export_checkpoint` returns the latest.
- Protocol models are exactly those of `openjiuwen-harness-protocol`; no duplicate event types.
- No network, no clocks beyond `SystemTime` for timestamps, no uuid crate use where the protocol leaves ids to the provider (uuid is allowed for ids).

## Routing Table

- `./mod.rs` → `NativeHarness`, `NativeHarnessOptions`, re-exports
- `./provider.rs` → `NativeHarnessProvider` (`HarnessProvider` + `ModelConfigProvider`), create-config parsing
- `./events.rs` → chunk/output → `HarnessEventPayload` mapping and turn results
- `./cursor.rs` → channel-backed event cursors, buffering, single-consumer lease
- `./bridge.rs` → turn/sequence/timestamp bookkeeping, checkpoint state

## Test Strategy

- Integration tests (`tests/protocol.rs`): start validates host protocol/capabilities, send→events→stop lifecycle, turn cursor finiteness, second cursor rejected, abort/pause/resume, capability-gated unsupported error, checkpoint export, provider create from JSON, model selection.
- Unit tests: event mapping shapes (payload variant + required ids), sequence monotonicity, buffer overflow policy.
- Parity reference: `tests/unit_tests/harness_protocol/` plus the native provider system tests.
