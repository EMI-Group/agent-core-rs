# openjiuwen-harness-protocol

## Intent

Provider-neutral SPI between a host (the openJiuwen runtime) and a harness provider (native harness, Claude Code, Codex, DSH, …).
Mirrors the Python `openjiuwen.harness_protocol` package (v0.1.20): the vocabulary for harness handles, events, turn results, checkpoints, hooks, interactions, tools, and model control.
`DESIGN.md` in this directory is the normative, field-by-field parity contract; read it before changing any public type, JSON key, or validation message.

## API Surface

- Traits: `HarnessProtocol`, `HarnessModelControl`, `HarnessProvider`, `ModelConfigProvider`, `HarnessEventCursor`, `HarnessCheckpointSink`, `HarnessHookDispatcher`, `HarnessInteractionHandler`, `ToolGateway`, `HarnessTelemetry`, `Validate`.
- Models: `HarnessCard`, `HarnessContext`, `HarnessInput`, `SendReceipt`, `HarnessEvent` + `HarnessEventPayload` (10 payload types), `TurnResult` and its parts, `HarnessCheckpoint`, `CheckpointSaveReceipt`, hook contexts/results, five interaction requests/responses, `ToolDefinition`/`ToolInvocation`/`ToolExecutionResult`, `McpServerConfig`, `ModelSelection`/`ModelEndpoint`/`ModelChoice`/`ModelOption`.
- Enums: `HarnessState`, `HarnessCapability`, `HostCapability`, `DeliveryMode`, `AbortMode`, `ResumePolicy`, `TurnStatus`, `MessageRole`, `TurnTerminationKind`, `TurnEventKind`, `Output*`, `UsageUpdateMode`, `EventRetention`, `EventOverflowPolicy`, `ItemEventKind`, `HookEventPhase`, `DiagnosticLevel`, `ModelRequestStatus`, `CheckpointReason`, `ToolDecisionKind`, `ToolApprovalDecision`, `InteractionResponseStatus`, `InteractionCancelReason`, `McpTransport`.
- Constants: `PROTOCOL_VERSION` (`"1.0"`), `EVENT_WIRE_SCHEMA_VERSION` (`"1"`), `MAX_CHECKPOINT_BYTES`, `AUTH_FALLBACK_REQUEST_TYPE`, `AUTH_FALLBACK_EVENT_TYPE`, `TERMINAL_TURN_EVENT_KINDS`.
- Functions: `harness_event_to_dict`/`harness_event_from_dict` (+ JSON string forms), `event_retention`, `validate_interaction_response`, `event_cursor_stream`, JSON helpers.
- Aliases: `JsonValue`/`JsonObject`, `HarnessResult`, `HarnessEventStream`, `DynHarness`, `DynHarnessProvider`, `DynModelControl`.
- Everything is re-exported flat from the crate root (Python's eager `__init__.__all__`); modules stay public for qualified paths.

## Constraints

- Zero workspace dependencies: only `serde`, `serde_json`, `schemars`, `thiserror`, `async-trait`, `futures`.
- No runtime dependency: no `tokio`, no clock, no id generation — providers supply `sequence` and Unix-epoch-second `f64` timestamps.
- Async traits use `async-trait` and `Send + Sync` supertraits so `Arc<dyn Trait>`/`Box<dyn Trait>` work everywhere; harness methods take `&self` (implementations synchronize internally).
- Serde: snake_case, unknown fields tolerated, no `skip_serializing_if` (wire keys always present, `null`/`[]` included), enums serialize to their Python `.value` strings.
- Errors: `HarnessError` (`Protocol`, `State`, `UnsupportedCapability`, `CheckpointConflict`, `Validation`) with Python message text preserved; no panics on provider input.
- Validation: Python `__post_init__` invariants live in the `Validate` trait; decode paths validate automatically, struct literals must call `validate()`.
- Unsigned integers (`u64`) encode Python's non-negative int checks; timestamps stay `f64` and are checked for finiteness.
- `HarnessContext` is not serde (it holds live host services); `ModelEndpoint::api_key` is redacted in `Debug`.
- Deliberate deviations from Python are listed in `DESIGN.md` §7 (freezing → value semantics, `Protocol` → traits, unions → tagged enums).
- Files ≤ ~1000 lines; all public items documented (workspace `missing_docs` + `-D warnings`).

## Routing Table

- `./DESIGN.md` → normative field-by-field parity contract, wire codec spec, test plan, deviations (read first)
- `./src/lib.rs` → crate docs, module declarations, flat public surface, type aliases
- `./src/errors.rs` → `HarnessError`, `HarnessErrorKind`, `HarnessResult`
- `./src/json.rs` → `JsonValue`/`JsonObject` aliases and conversions
- `./src/validation.rs` → `Validate` trait and shared field validators
- `./src/state.rs` → `HarnessState` (shared with the agent-teams runtime)
- `./src/models/` → harness card + capability negotiation, delivery modes, input/receipt, run context, model selection/discovery
- `./src/results.rs` → turn outcomes: statuses, content blocks, messages, termination, usage, cost, error
- `./src/events/` → event enums, payload models, `HarnessEvent` envelope, retention and buffer configuration
- `./src/serialization.rs` → `EVENT_WIRE_SCHEMA_VERSION` and the event wire codec (single source of truth)
- `./src/checkpoints.rs` → checkpoint model, save receipt, size limit, `HarnessCheckpointSink`
- `./src/hooks.rs` → synchronous hook contexts/results and `HarnessHookDispatcher`
- `./src/interactions.rs` → awaited host interactions, validation, `HarnessInteractionHandler`
- `./src/tools.rs` → tool gateway SPI and MCP server configuration
- `./src/stream.rs` → event cursor SPI, cursor→`BoxStream` adapter, in-memory cursor
- `./src/protocol.rs` → the four main SPI traits and shared aliases
- `./tests/models.rs`, `./tests/results.rs` → card/context/model-control and turn-result behavior parity
- `./tests/events.rs`, `./tests/serialization.rs` → event validation and wire-codec parity (round-trip, unknown events, schema gate)
- `./tests/services.rs`, `./tests/protocol.rs` → checkpoints/hooks/tools/interactions and trait-object/cursor/stream behavior

## Test Strategy

- Python test parity: `tests/unit_tests/harness_protocol/test_protocol.py` (29 cases) and `test_models_fork_modes.py` (2 cases) are ported one-to-one where the Rust API has an analogue; each ported test names its Python counterpart in a comment.
- Unit tests for pure helpers live beside modules; behavior/parity tests are integration tests under `./tests/`.
- Async SPI tests run on `futures::executor::block_on`; no network, no tokio, no sleeps, fully deterministic.
- Fakes (`FakeHarness`, `FakeProvider`, `FakeCheckpointSink`, `FakeInteractionHandler`, `FakeToolGateway`, `FakeHookDispatcher`, `FakeTelemetry`) live inside the test files and double as usage examples.
- Wire tests assert exact key presence and exact `event_type`/payload key sets, so accidental shape changes fail loudly.
- Definition of done: `cargo fmt -p openjiuwen-harness-protocol`, `cargo test -p openjiuwen-harness-protocol`, `cargo clippy -p openjiuwen-harness-protocol --all-targets -- -D warnings` all clean.

- `./src/interactions.rs` is intentionally ~1070 lines (five request/response families plus the tagged dispatch enums and inline tests); split it only when a new interaction family is added

## Notes for Agents

- `HarnessProtocol` (`#[async_trait]`, `Send + Sync`, all `&self`): sync `card`, `state`, `provider_session_id`, `event_buffer_config`, `events() -> Box<dyn HarnessEventCursor>`, `turn_events(Option<&str>)`; async `start(HarnessContext)`, `stop`, `send(HarnessInput, DeliveryMode)`, `abort(AbortMode)`, `pause`, `resume(Option<HarnessInput>)`, `export_checkpoint`; provided defaults `send_auto`, `abort_graceful`.
- There is no `import_checkpoint` (restore flows through `HarnessContext::checkpoint` + `ResumePolicy`) and no `send_steer` (steering is `DeliveryMode::Steer`).
- `HarnessProvider` is synchronous (no `#[async_trait]`): `fn card(&self) -> &HarnessCard` and `fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>>`; `ModelConfigProvider::model_config(&self, &ModelChoice, Option<&ModelChoice>) -> HarnessResult<JsonObject>` is sync too.
- `HarnessModelControl` (`#[async_trait]`): `list_models() -> HarnessResult<Vec<ModelOption>>`, `set_model(ModelSelection) -> HarnessResult<()>`.
- The provider model-catalog type is `ModelOption`; there is no `ModelInfo`.
- `ModelSelection`/`ModelChoice`/`ModelEndpoint`/`ModelOption` derive no `Default`: build them with struct literals naming every field (or deserialize).
- `HarnessEventCursor` (`src/stream.rs`): `#[async_trait]`, `Send` but not `Sync`, `&mut self`; `next_event() -> HarnessResult<Option<HarnessEvent>>`, `close() -> HarnessResult<()>`.
- The crate declares no cargo features (`[features] default = []`); nothing in the public surface is feature-gated.
- `openjiuwen-harness` implements the native `HarnessProtocol`/`HarnessProvider`; `openjiuwen-harness-providers` registers providers (`native` implemented, vendor providers deferred).
- Implementor obligations: all harness methods take `&self` (synchronize internally); `create` returns an unstarted harness; `start` validates through `HarnessCard::validate_host`; `send` never waits for a turn; `stop` is idempotent; only one active event cursor (a second must yield `HarnessError::State`); capability-gated operations return `HarnessError::UnsupportedCapability`.
- Checkpoint sinks must keep `sequence` monotonic per provider/agent/host-session, treat a repeated `checkpoint_id` as idempotent, and answer stale sequences or failed compare-and-set with `HarnessError::CheckpointConflict`.
- Redaction: `ModelEndpoint::api_key` and `HarnessContext::{env, metadata}` are hidden in `Debug`; never log them.
- `HarnessError` derives only `Debug` + `thiserror::Error` (no `Clone`, no `PartialEq`) and both it and `HarnessErrorKind` are `#[non_exhaustive]`: compare through `kind()`/`message()` and add a wildcard arm when matching on an error.
- `HarnessEventCursor` is `Send` but not `Sync`, both methods take `&mut self`, and the cycle/turn cursors are returned as `Box<dyn HarnessEventCursor>` (`'static`).
- `openjiuwen-agent-teams` re-exports `HarnessState` from here — never define a second copy.
- Changing a field name, enum value, or validation message is a protocol break: update `DESIGN.md`, this file, and every dependent crate.
- Additive changes within the `"1.0"` major version must keep the `EVENT_WIRE_SCHEMA_VERSION` `"1"` decode path backward compatible; bump the wire schema version for incompatible payload changes.
