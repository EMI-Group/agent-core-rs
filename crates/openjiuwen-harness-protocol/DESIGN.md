# openjiuwen-harness-protocol — normative design & parity contract

Authoritative implementation contract for this crate.
Parity source: `/home/bill/Source/agent-core/openjiuwen/harness_protocol/` (13 modules, v0.1.20, stdlib-only, read-only).
Python test parity source: `/home/bill/Source/agent-core/tests/unit_tests/harness_protocol/`.
Read this file before changing any public type, JSON key, validation rule, or message string — downstream crates (`openjiuwen-harness`, `openjiuwen-harness-providers`, `openjiuwen-agent-teams`) depend on the exact shapes.

## 1. Purpose

Provider-neutral SPI between a *host* (openJiuwen agent runtime) and a *harness provider* (native openJiuwen harness, Claude Code, Codex, DSH, …).
The wire vocabulary (events, results, checkpoints, interactions, hooks, tools) is defined here and nowhere else.

## 2. Dependency and style rules

- Dependencies: `serde`, `serde_json`, `schemars`, `thiserror`, `async-trait`, `futures` only.
  No `openjiuwen-*` crate, no `tokio`, no `uuid`, no `chrono` (the protocol has no clock and no id generation: providers supply `sequence` and Unix-epoch-second timestamps).
- No `unsafe`. All public items documented (`missing_docs` is a workspace lint and CI runs `-D warnings`).
- `#![forbid(unsafe_code)]`-equivalent comes from the workspace lint; do not add `allow(unsafe_code)`.
- Files ≤ ~1000 lines, one responsibility per file. Split further rather than growing a file.
- `async-trait` for every async trait (dyn-safe SPI); traits are `Send + Sync` supertraits so `Arc<dyn Trait>` / `Box<dyn Trait>` are directly usable.
- Async traits must not require a specific runtime: no `tokio::*` in the crate; tests drive futures with `futures::executor::block_on`.
- Clippy runs with `pedantic` + `-D warnings`. Data-model crates legitimately trip `must_use_candidate`, `struct_excessive_bools`, `trivially_copy_pass_by_ref`, `doc_markdown`, `module_name_repetitions`, `missing_errors_doc`; fix what is meaningful, and where a lint is structurally inapplicable use a *module-level* `#![allow(clippy::…)]` with a one-line justification comment. Never a blanket crate-wide `-A clippy::all`-style escape.

## 3. Module map

| File | Responsibility |
|---|---|
| `src/lib.rs` | Crate docs, module declarations, flat re-export of the full public surface (Python `__init__.__all__` equivalent), type aliases. |
| `src/errors.rs` | `HarnessError`, `HarnessErrorKind`, `HarnessResult`. |
| `src/json.rs` | `JsonValue`, `JsonObject` aliases and conversion helpers (`json_object_from_value`, `json_object_to_value`). |
| `src/validation.rs` | `Validate` trait + shared field validators used by every model. |
| `src/state.rs` | `HarnessState`. |
| `src/models/mod.rs` | Re-exports of the `models` submodules. |
| `src/models/card.rs` | `PROTOCOL_VERSION`, `HarnessCapability`, `HostCapability`, `HarnessCard` (+ `supports`, `validate_host`). |
| `src/models/delivery.rs` | `DeliveryMode`, `AbortMode`, `HarnessInput`, `SendReceipt`. |
| `src/models/context.rs` | `ResumePolicy`, `HarnessTelemetry`, `HarnessContext`. |
| `src/models/model_control.rs` | `ModelSelection`, `ModelEndpoint`, `ModelChoice`, `ModelOption`, `AUTH_FALLBACK_REQUEST_TYPE`, `AUTH_FALLBACK_EVENT_TYPE`. |
| `src/results.rs` | Turn outcome vocabulary: `TurnStatus`, `MessageRole`, `TurnTerminationKind`, `ContentBlock`, `TurnMessage`, `TurnTermination`, `MonetaryAmount`, `TurnUsage`, `TurnError`, `TurnResult`. |
| `src/events/mod.rs` | Re-exports of the `events` submodules. |
| `src/events/enums.rs` | `TurnEventKind` + `TERMINAL_TURN_EVENT_KINDS`, `OutputKind`, `OutputChannel`, `OutputOperation`, `UsageUpdateMode`, `EventRetention`, `EventOverflowPolicy`, `ItemEventKind`, `HookEventPhase`, `DiagnosticLevel`, `ModelRequestStatus`. |
| `src/events/payloads.rs` | `OutputEvent`, `ItemLifecycleEvent`, `UsageUpdatedEvent`, `ModelRequestEvent`, `StateChangedEvent`, `TurnLifecycleEvent`, `HookObservedEvent`, `DiagnosticEvent`, `ProviderEvent`, `UnknownEvent`, `HarnessEventPayload`, `event_retention`. |
| `src/events/envelope.rs` | `HarnessEvent` (manual serde = the Python wire shape), envelope validation. |
| `src/events/buffer.rs` | `EventBufferConfig`. |
| `src/serialization.rs` | `EVENT_WIRE_SCHEMA_VERSION`, `harness_event_to_dict`, `harness_event_from_dict`, `harness_event_to_json_string`, `harness_event_from_json_str`. |
| `src/checkpoints.rs` | `MAX_CHECKPOINT_BYTES`, `CheckpointReason`, `HarnessCheckpoint`, `CheckpointSaveReceipt`, `HarnessCheckpointSink`. |
| `src/hooks.rs` | `ToolDecisionKind`, `BeforePromptContext/Result`, `BeforeToolContext`, `ToolDecision`, `AfterToolContext/Result`, `StopHookContext`, `HarnessHookDispatcher`. |
| `src/interactions.rs` | Interaction enums, five request types, five response types, `HarnessInteractionRequest`, `HarnessInteractionResponse`, `validate_interaction_response`, `HarnessInteractionHandler`. |
| `src/tools.rs` | `ToolDefinition`, `ToolInvocation`, `ToolExecutionResult`, `ToolGateway`, `McpTransport`, `McpInstance`, `McpServerConfig`. |
| `src/stream.rs` | `HarnessEventCursor`, `HarnessEventStream`, `EventCursorStream` adapter, `VecEventCursor`. |
| `src/protocol.rs` | `HarnessProtocol`, `HarnessModelControl`, `HarnessProvider`, `ModelConfigProvider`, aliases. |

Tests: `tests/models.rs`, `tests/results.rs`, `tests/events.rs`, `tests/serialization.rs`, `tests/services.rs`, `tests/protocol.rs` (see §10).

## 4. Global conventions

### 4.1 Errors

```rust
pub type HarnessResult<T> = Result<T, HarnessError>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessError {
    #[error("harness protocol error: {0}")]
    Protocol(String),              // Python HarnessProtocolError
    #[error("harness state error: {0}")]
    State(String),                 // Python HarnessStateError
    #[error("unsupported harness capability: {0}")]
    UnsupportedCapability(String), // Python UnsupportedHarnessCapabilityError
    #[error("checkpoint conflict: {0}")]
    CheckpointConflict(String),    // Python CheckpointConflictError
    #[error("invalid harness value: {0}")]
    Validation(String),            // Python ValueError / TypeError in __post_init__
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HarnessErrorKind { Protocol, State, UnsupportedCapability, CheckpointConflict, Validation }
```
- `HarnessError` exposes constructors `protocol`, `state`, `unsupported_capability`, `checkpoint_conflict`, `validation` (all `impl Into<String>`), `kind()`, and `message()` (the inner string only).
- `impl From<serde_json::Error> for HarnessError` → `Protocol(format!("invalid harness JSON: {e}"))`.
- Message bodies reproduce the Python wording *verbatim* (minus Python quoting) because tests and provider adapters match substrings; the `Display` prefix is additive.
- Library code never panics on user/provider input; `unwrap_used`/`expect_used` are denied by lint.

### 4.2 JSON

```rust
pub type JsonValue = serde_json::Value;
pub type JsonObject = std::collections::BTreeMap<String, JsonValue>;
```
- `BTreeMap` (not `serde_json::Map`) gives deterministic key order, `Ord`, cheap equality, and a guaranteed `schemars` impl.
- Helpers: `json_object_from_value(JsonValue) -> Option<JsonObject>`, `json_object_to_value(&JsonObject) -> JsonValue`, `empty_json_object()`.
- Python's `freeze_json_value` / `freeze_json_object` / `json_value_to_builtin` have no Rust analogue: `JsonValue` is already JSON-compatible, owned, and finite-numbers-only. Deep immutability is replaced by value semantics. Record this in `CONTEXT.md` as a deliberate deviation — do **not** port those functions.

### 4.3 Validation

```rust
pub trait Validate {
    /// Checks the invariants the Python model enforces in `__post_init__`.
    fn validate(&self) -> HarnessResult<()>;
}
```
- Every model with Python `__post_init__` rules implements `Validate`; models without rules implement it as `Ok(())` only where a parent composes them.
- Constructors *do not* validate implicitly (Rust structs cannot); **decode paths do**: `harness_event_from_dict` validates. For other models provide `json::from_value_validated` / `from_str_validated` helpers (`T: DeserializeOwned + Validate`).
- Shared validators in `validation.rs`: `require_non_empty(field_name, value)`, `require_optional_non_empty(field_name, value)`, `require_finite_timestamp(field_name, value)`, `require_finite_non_negative(field_name, value)`, `require_unique_non_empty(field_name, values)`, `require_subset(...)`, `join_sorted(...)`. They return `HarnessError::Validation` with the exact Python messages.
- Unsigned-integer fields (`u64`) replace Python's `>= 0` int checks for `sequence`, counters, `duration_ms`, `micros`, `capacity`; a negative wire value is rejected by serde (parity: Python rejects it with `ValueError`). Timestamps stay `f64` and are validated for finiteness/range, because JSON `f64` can hold `inf`/`NaN` in memory.

### 4.4 Serde

- Every serializable type: `#[derive(Serialize, Deserialize, JsonSchema)]` with `#[serde(rename_all = "snake_case")]` on structs, unknown fields tolerated (serde default — never `deny_unknown_fields`).
- Field-level `#[serde(default)]` on exactly those fields that carry a Python default; a field Python requires must error when absent (never a container-level `#[serde(default)]` on a struct with required fields — that would silently invent values). All-default structs (`EventBufferConfig`, `TurnUsage`, `AfterToolResult`, …) may derive `Default` and use container-level defaults.
- No `skip_serializing_if` anywhere: the wire shape is "all keys always present", `null` for `None`, `[]` for empty sequences — exactly like the Python codec.
- Enums: `#[serde(rename_all = "snake_case")]`, variant values equal to the Python `.value` strings. Derive `Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema` for value enums (needed for `BTreeSet` membership and sorting).
- Sets: `BTreeSet<…>` for Python `frozenset` (serializes as a sorted JSON array). Sequences: `Vec<…>` for Python `tuple`.
- `HarnessContext` is **not** serde: it carries live host services (`Arc<dyn …>`), mirroring Python's Card/Config split. `Validate` is implemented for it.
- `McpServerConfig.instance` is `Option<McpInstance>` wrapping `Arc<dyn Any + Send + Sync>`; `#[serde(skip)]`, `#[schemars(skip)]`, manual `Debug` printing `<opaque instance>`, `PartialEq` compares only `is_some()`.
- `ModelEndpoint.api_key` is `Option<String>`; the manual `Debug` impl must render it as `Some("<redacted>")` (Python `field(repr=False)`), and it must never be logged.

## 5. Type catalogue (exact fields, defaults, rules)

Defaults marked with `=` are `Default::default()`/serde defaults.

### 5.1 `state.rs`

```rust
pub enum HarnessState { Idle, Running, Pausing, Paused, Terminated }  // "idle","running","pausing","paused","terminated"
```
Shared vocabulary with `openjiuwen-agent-teams` (that crate re-exports this type; do not define a second copy).

### 5.2 `models/card.rs`

```rust
pub const PROTOCOL_VERSION: &str = "1.0";

pub enum HarnessCapability {          // 13 values
  Steer="steer", GracefulAbort="graceful_abort", ForceAbort="force_abort", PauseResume="pause_resume",
  PersistentSession="persistent_session", Checkpoint="checkpoint", NativeTools="native_tools",
  McpTools="mcp_tools", Hooks="hooks", ModelSelection="model_selection",
  ModelDiscovery="model_discovery", EndpointModels="endpoint_models", AuthFallback="auth_fallback",
}

pub enum HostCapability {             // 11 values
  CheckpointSink="checkpoint_sink", NativeToolGateway="native_tool_gateway", McpServers="mcp_servers",
  Hooks="hooks", ToolApproval="tool_approval", UserInput="user_input", McpElicitation="mcp_elicitation",
  DynamicToolCall="dynamic_tool_call", ProviderInteraction="provider_interaction", Telemetry="telemetry",
  ModelRequestObservation="model_request_observation",
}

pub struct HarnessCard {
  pub name: String,
  pub implementation_version: String,
  pub protocol_version: String,                              // = PROTOCOL_VERSION
  pub compatible_protocol_versions: BTreeSet<String>,       // = {"1.0"}
  pub capabilities: BTreeSet<HarnessCapability>,            // empty
  pub required_host_capabilities: BTreeSet<HostCapability>, // empty
  pub optional_host_capabilities: BTreeSet<HostCapability>, // empty
  pub fork_modes: BTreeSet<String>,                         // empty; documented values "full", "before"
}
```
- `Default` → `name`/`implementation_version` empty (invalid until set); provide `HarnessCard::new(name, implementation_version)`.
- `PROTOCOL_VERSION` is also re-exported at the crate root.
- `default_compatible_protocol_versions() -> BTreeSet<String>`.
- `supports(&self, HarnessCapability) -> bool`.
- `validate_host(&self, protocol_version: &str, capabilities: &BTreeSet<HostCapability>) -> HarnessResult<()>`:
  1. version not in `compatible_protocol_versions` → `HarnessError::Protocol("host protocol '{v}' is not supported by '{name}'")`
  2. `missing = required_host_capabilities - capabilities`; non-empty → `Protocol("host is missing required capabilities: {sorted, ', '}")`
  3. optional capabilities are never validated.
- `Validate for HarnessCard`: identity/versions non-empty (`"harness card identity and versions must not be empty"`); `protocol_version ∈ compatible_protocol_versions` (`"card protocol_version must be included in compatible_protocol_versions"`); required ∩ optional empty (`"host capabilities cannot be both required and optional: {sorted, ', '}"`).

### 5.3 `models/delivery.rs`

```rust
pub enum DeliveryMode { Auto="auto", Steer="steer", FollowUp="follow_up" }        // Default::default() = Auto
pub enum AbortMode { Graceful="graceful", Force="force" }                          // default Graceful
pub struct HarnessInput { pub content: JsonValue, pub metadata: JsonObject }        // content required, metadata empty
pub struct SendReceipt { pub message_id: String, pub turn_id: String, pub accepted_mode: DeliveryMode }
```
- `HarnessInput::text(impl Into<String>) -> Self` convenience.
- `SendReceipt::validate`: `message_id`/`turn_id` non-empty (`"send receipt message_id must not be empty"`, `"send receipt turn_id must not be empty"`).

### 5.4 `models/model_control.rs`

```rust
pub const AUTH_FALLBACK_REQUEST_TYPE: &str = "auth_fallback";
pub const AUTH_FALLBACK_EVENT_TYPE: &str = "auth_fallback_activated";

pub struct ModelSelection { pub model: Option<String>, pub effort: Option<String> }
pub struct ModelEndpoint { pub api_base: Option<String>, pub api_key: Option<String>, pub provider: Option<String> }
pub struct ModelChoice { pub model: Option<String>, pub effort: Option<String>, pub endpoint: Option<ModelEndpoint> }
pub struct ModelOption {
  pub model_id: String,
  pub display_name: String,       // ""
  pub description: String,        // ""
  pub efforts: Vec<String>,       // []
  pub default_effort: Option<String>,
  pub is_default: bool,           // false
  pub extensions: JsonObject,     // {}
}
```
- `ModelSelection` `Validate`: both `model` and `effort` `None` → `"model selection must set a model, an effort, or both"`; a provided value must be non-empty after trim → `"model selection {field} must be a non-empty string when provided"`.
- `ModelChoice` `Validate` (Python parity): an all-`None` choice is **valid** (it asks the provider for its default); provided `model`/`effort` must be non-empty after trim → `"model choice {field} must be a non-empty string when provided"`; a present `endpoint` is validated.
- `ModelChoice::is_builtin(&self) -> bool` = `endpoint.is_none()`.
- `ModelEndpoint` `Validate`: `api_base`/`provider` both `None` → `"model endpoint must set an api_base, a provider, or both"`; provided values non-empty → `"model endpoint {field} must be a non-empty string when provided"`. Manual `Debug` redacts `api_key`.
- `ModelOption` `Validate`: `model_id` non-empty (`"model option model_id must not be empty"`); every effort non-empty (`"model option efforts must be non-empty strings"`); when `efforts` non-empty, `default_effort` (if set) must be a member (`"model option default_effort must be one of its efforts"`).

### 5.5 `models/context.rs`

```rust
pub enum ResumePolicy { New="new", ResumeIfAvailable="resume_if_available", RequireResume="require_resume" } // default New

pub trait HarnessTelemetry: Send + Sync {
    fn record(&self, name: &str, attributes: &JsonObject);
}

pub struct HarnessContext {
  pub agent_name: String,
  pub agent_id: String,
  pub host_session_id: String,
  pub system_prompt: String,
  pub protocol_version: String,                              // = PROTOCOL_VERSION
  pub host_capabilities: BTreeSet<HostCapability>,           // empty
  pub resume_policy: ResumePolicy,                           // New
  pub cwd: Option<String>,
  pub env: BTreeMap<String, String>,                         // empty
  pub checkpoint: Option<HarnessCheckpoint>,
  pub checkpoint_sink: Option<Arc<dyn HarnessCheckpointSink>>,
  pub tools: Option<Arc<dyn ToolGateway>>,
  pub mcp_servers: Vec<McpServerConfig>,                     // empty
  pub hooks: Option<Arc<dyn HarnessHookDispatcher>>,
  pub interactions: Option<Arc<dyn HarnessInteractionHandler>>,
  pub telemetry: Option<Arc<dyn HarnessTelemetry>>,
  pub metadata: JsonObject,                                  // empty
}
```
- `HarnessContext::new(agent_name, agent_id, host_session_id, system_prompt)` + chainable `with_*` builder methods for every remaining field. No `Default` (identity is mandatory).
- `Validate`: `agent_name`, `agent_id`, `host_session_id`, `protocol_version` non-empty in that order (`"harness context {field} must not be empty"`). Nothing else.

### 5.6 `results.rs`

```rust
pub enum TurnStatus { Completed="completed", Interrupted="interrupted", Failed="failed" }
pub enum MessageRole { User="user", Assistant="assistant", System="system", Tool="tool" }
pub enum TurnTerminationKind { UserAbort="user_abort", Timeout="timeout", Policy="policy", Provider="provider", HarnessStop="harness_stop" }

pub struct ContentBlock { pub block_id: String, pub kind: String, pub content: JsonValue, pub data: JsonObject }
pub struct TurnMessage { pub message_id: String, pub role: MessageRole, pub content: Vec<ContentBlock>, pub data: JsonObject }
pub struct TurnTermination { pub kind: TurnTerminationKind, pub message: Option<String>, pub code: Option<String>, pub provider_data: JsonObject }
pub struct MonetaryAmount { pub micros: u64, pub currency: String }          // currency default "USD"
pub struct TurnUsage {
  pub input_tokens: Option<u64>, pub output_tokens: Option<u64>,
  pub cached_input_tokens: Option<u64>, pub reasoning_output_tokens: Option<u64>,
  pub total_tokens: Option<u64>, pub provider_data: JsonObject,
}
pub struct TurnError { pub message: String, pub code: Option<String>, pub category: Option<String>,
                       pub retryable: Option<bool>, pub provider_data: JsonObject }
pub struct TurnResult {
  pub status: TurnStatus,
  pub messages: Vec<TurnMessage>,          // []
  pub final_output: JsonValue,             // Null
  pub structured_output: JsonValue,        // Null
  pub stop_reason: Option<String>,
  pub termination: Option<TurnTermination>,
  pub error: Option<TurnError>,
  pub usage: Option<TurnUsage>,
  pub cost: Option<MonetaryAmount>,
  pub started_at: Option<f64>,
  pub completed_at: Option<f64>,
  pub duration_ms: Option<u64>,
  pub provider_data: JsonObject,           // {}
}
```
- `ContentBlock::validate`: id and kind non-empty (`"content block id and kind must not be empty"`).
- `TurnMessage::validate`: `message_id` non-empty (`"turn message_id must not be empty"`); each block validated.
- `MonetaryAmount`: `validate` requires the uppercased currency to be 3 ASCII letters (`"monetary currency must be a three-letter code"`); `MonetaryAmount::new(micros, currency)` stores the uppercased value; `validate` accepts lowercase input (Python uppercases first). `micros` unsigned removes Python's `>= 0` check.
- `TurnUsage::validate`: nothing to check beyond unsigned counters; implement `Validate` as `Ok(())` (documented reason: unsigned fields subsume Python's non-negative checks).
- `TurnError::validate`: `message` non-empty (`"turn error message must not be empty"`).
- `TurnResult::validate` (order matters, messages verbatim):
  1. `Failed` requires `error` (`"failed turn result requires error"`); non-`Failed` forbids it (`"only failed turn result may contain error"`).
  2. `Interrupted` requires `termination` (`"interrupted turn result requires termination"`); non-`Interrupted` forbids it (`"only interrupted turn result may contain termination"`).
  3. `started_at`/`completed_at` finite and `>= 0` (`"turn {field} must be a finite Unix timestamp"`).
  4. `completed_at >= started_at` when both present (`"turn completed_at must not precede started_at"`).
  5. nested `messages`, `error`, `termination`, `usage`, `cost` validated.

### 5.7 `events/enums.rs`

```rust
pub enum TurnEventKind { Started, Paused, Resumed, Finished, Aborted, Failed }   // "started","paused","resumed","finished","aborted","failed"
pub const TERMINAL_TURN_EVENT_KINDS: [TurnEventKind; 3] = [Finished, Aborted, Failed];
impl TurnEventKind { pub fn is_terminal(self) -> bool }
pub enum OutputKind { Text, Structured }
pub enum OutputChannel { Answer, Reasoning, System }
pub enum OutputOperation { Delta, Snapshot, Final }
pub enum UsageUpdateMode { Delta, Cumulative }
pub enum EventRetention { Required, Coalescible, BestEffort }
pub enum EventOverflowPolicy { Block, CoalesceOrBlock, DropBestEffortOrBlock }   // default CoalesceOrBlock
pub enum ItemEventKind { Started, Updated, Completed }
pub enum HookEventPhase { Started, Finished }
pub enum DiagnosticLevel { Debug, Info, Warning, Error }
pub enum ModelRequestStatus { Completed, Failed, Cancelled }
```
All `snake_case` string values exactly as listed (`drop_best_effort_or_block`, `best_effort`, …).

### 5.8 `events/payloads.rs`

Payload structs (field order as given; all `#[serde(default)]` structs, all keys always serialized):

```rust
pub struct OutputEvent {
  pub output_id: String, pub kind: OutputKind, pub content: JsonValue,
  pub operation: OutputOperation /*=Snapshot*/, pub channel: OutputChannel /*=Answer*/,
  pub content_index: u64 /*=0*/, pub content_type: Option<String>, pub data: JsonObject,
}
pub struct ItemLifecycleEvent { pub kind: ItemEventKind, pub item_type: String, pub data: JsonObject }
pub struct UsageUpdatedEvent { pub usage: TurnUsage, pub mode: UsageUpdateMode /*=Cumulative*/ }
pub struct ModelRequestEvent {
  pub request_id: String, pub status: ModelRequestStatus,
  pub started_at: f64, pub ended_at: f64,
  pub model: Option<String>, pub provider_name: Option<String>,
  pub system_instructions: Vec<ContentBlock>, pub input_messages: Vec<TurnMessage>,
  pub input_observed: bool /*=false*/, pub output_message: Option<TurnMessage>,
  pub tool_definitions: JsonValue /*=Null*/, pub request_parameters: JsonObject,
  pub response_id: Option<String>, pub time_to_first_chunk: Option<f64>,
  pub finish_reasons: Vec<String>, pub usage: Option<TurnUsage>,
  pub cost: Option<MonetaryAmount>, pub error: Option<TurnError>, pub data: JsonObject,
}
pub struct StateChangedEvent { pub old: HarnessState, pub new: HarnessState }
pub struct TurnLifecycleEvent { pub kind: TurnEventKind, pub result: Option<TurnResult> }
pub struct HookObservedEvent { pub phase: HookEventPhase, pub hook_name: String, pub hook_id: String, pub data: JsonObject }
pub struct DiagnosticEvent { pub level: DiagnosticLevel, pub message: String, pub data: JsonObject }
pub struct ProviderEvent { pub provider: String, pub event_type: String, pub schema_version: String, pub payload: JsonObject }
pub struct UnknownEvent { pub event_type: String, pub schema_version: String, pub payload: JsonObject }

#[derive(Debug, Clone, PartialEq, JsonSchema)]   // NO derived serde: the wire shape is produced by serialization.rs (§6)
pub enum HarnessEventPayload { Output(OutputEvent), ItemLifecycle(ItemLifecycleEvent), UsageUpdated(UsageUpdatedEvent),
  ModelRequest(ModelRequestEvent), StateChanged(StateChangedEvent), TurnLifecycle(TurnLifecycleEvent),
  HookObserved(HookObservedEvent), Diagnostic(DiagnosticEvent), Provider(ProviderEvent), Unknown(UnknownEvent) }
```
`HarnessEventPayload` is deliberately **not** serde-derived: an internally tagged enum would collide with `ProviderEvent::event_type`, and the Python wire shape keeps `event_type` at envelope level with the payload nested. `serialization.rs` is the only codec.
Required helpers: `wire_event_type(&self) -> &str` (`"output"`, `"item_lifecycle"`, `"usage_updated"`, `"model_request"`, `"state_changed"`, `"turn_lifecycle"`, `"hook_observed"`, `"diagnostic"`, `"provider"`, and verbatim `event_type` for `Unknown`), `wire_schema_version(&self) -> &str` (`EVENT_WIRE_SCHEMA_VERSION` except `Unknown`), `Validate`, `From<OutputEvent> for HarnessEventPayload` (and one `From` per payload type).

Validation rules (verbatim messages):
- `OutputEvent`: `output_id` non-empty (`"output_id must not be empty"`); `content_index` unsigned (subsumes Python's non-negative check).
- `ItemLifecycleEvent`: `item_type` non-empty (`"item_type must not be empty"`).
- `ModelRequestEvent`: `request_id` non-empty (`"model request_id must not be empty"`); `started_at`/`ended_at` finite `>= 0` (`"model request {field} must be a finite Unix timestamp"`); `ended_at >= started_at` (`"model request ended_at must not precede started_at"`); `Completed` with `error` → `"completed model request must not contain error"`; `time_to_first_chunk` finite `>= 0` (`"model request time_to_first_chunk must be a non-negative duration"`); nested messages/blocks/usage/cost/error validated.
- `TurnLifecycleEvent`: non-terminal with `result` → `"non-terminal {kind} turn event must not contain a result"`; terminal without `result` → `"terminal turn event requires a result"`; kind/status agreement → `"{kind} turn event requires {expected_status} result"` with `Finished→Completed`, `Aborted→Interrupted`, `Failed→Failed`; nested result validated.
- `HookObservedEvent`: `"hook observation name and id must not be empty"`.
- `DiagnosticEvent`: `"diagnostic message must not be empty"`.
- `ProviderEvent`: `"provider event namespace, type, and schema version must not be empty"`.
- `UnknownEvent`: `"unknown event type and schema version must not be empty"`.
- `StateChangedEvent`, `UsageUpdatedEvent`: `Validate` delegates to `TurnUsage`.

`pub fn event_retention(event: &HarnessEventPayload) -> EventRetention` — first match wins:
`Output{operation: Snapshot}` → `Coalescible`; `UsageUpdated{mode: Cumulative}` → `Coalescible`; `HookObserved` → `BestEffort`; `Diagnostic{level: Debug|Info}` → `BestEffort`; otherwise `Required`.

### 5.9 `events/envelope.rs`

```rust
pub struct HarnessEvent {
  pub sequence: u64,
  pub timestamp: f64,                 // Unix epoch seconds
  pub event: HarnessEventPayload,
  pub host_session_id: String,
  pub agent_id: String,
  pub provider_session_id: Option<String>,
  pub turn_id: Option<String>,
  pub item_id: Option<String>,
  pub correlation_id: Option<String>,
  pub causation_ids: Vec<String>,     // []
}
```
`Serialize`/`Deserialize` are **manual**, delegating to the wire shape in §6 (single source of truth).
`Validate` order: `timestamp` finite and `>= 0` (`"event timestamp must be a finite Unix timestamp"`) — `sequence` unsigned subsumes Python's `>= 0`; `host_session_id`/`agent_id` non-empty (`"event host_session_id and agent_id must not be empty"`); optional ids must not be empty strings (`"optional event IDs must not be empty strings"`); `causation_ids` no empty entries (`"event causation_ids must not contain empty values"`) and no duplicates (`"event causation_ids must not contain duplicates"`); `TurnLifecycle` payload requires `turn_id` (`"turn lifecycle event requires turn_id"`); `ModelRequest` payload requires `turn_id` (`"model request event requires turn_id"`); then `event.validate()`.
`HarnessEvent::new(sequence, timestamp, event, host_session_id, agent_id)` constructor; chainable `with_turn_id`, `with_provider_session_id`, `with_item_id`, `with_correlation_id`, `with_causation_ids`.

### 5.10 `events/buffer.rs`

```rust
pub struct EventBufferConfig { pub capacity: u64 /*=1024*/, pub overflow: EventOverflowPolicy /*=CoalesceOrBlock*/ }
```
`Validate`: `capacity > 0` (`"event buffer capacity must be positive"`).

### 5.11 `checkpoints.rs`

```rust
pub const MAX_CHECKPOINT_BYTES: usize = 4 * 1024 * 1024;

pub enum CheckpointReason { SessionActivated="session_activated", TurnCompleted="turn_completed",
  StateChanged="state_changed", Periodic="periodic", ProviderRequested="provider_requested" }

pub struct HarnessCheckpoint {
  pub provider: String, pub schema_version: String, pub agent_id: String,
  pub host_session_id: String, pub checkpoint_id: String, pub sequence: u64,
  pub data: JsonObject,
  pub provider_session_id: Option<String>, pub revision: Option<String>,
}
pub struct CheckpointSaveReceipt { pub checkpoint_id: String, pub sequence: u64, pub storage_revision: String }
```
`HarnessCheckpoint::validate`: the five identity fields non-empty (`"checkpoint {field} must not be empty"`, order provider, schema_version, agent_id, host_session_id, checkpoint_id); optional ids not empty (`"optional checkpoint ids must not be empty strings"`); encoded size of `data` (`serde_json::to_vec(&self.data)` compact, UTF-8) `> MAX_CHECKPOINT_BYTES` → `"checkpoint data exceeds {MAX_CHECKPOINT_BYTES} bytes"`.
`CheckpointSaveReceipt::validate`: ids non-empty (`"checkpoint save receipt ids must not be empty"`).
`HarnessCheckpoint::size_bytes(&self) -> usize` exposes the encoded size (used by hosts and tests).

```rust
#[async_trait]
pub trait HarnessCheckpointSink: Send + Sync {
    async fn save(&self, checkpoint: &HarnessCheckpoint, reason: CheckpointReason,
                  expected_storage_revision: Option<&str>) -> HarnessResult<CheckpointSaveReceipt>;
}
```
Documented obligations (doc comments): monotonic `sequence` per provider/agent/host-session; retry of the same `checkpoint_id` is idempotent and returns the original receipt; a write at an existing-or-lower sequence, or a failed `expected_storage_revision` compare-and-set, returns `HarnessError::CheckpointConflict`.

### 5.12 `hooks.rs`

```rust
pub enum ToolDecisionKind { Allow="allow", Deny="deny", Rewrite="rewrite", Ask="ask", ProviderPolicy="provider_policy" }

pub struct BeforePromptContext { pub agent_name: String, pub provider_session_id: Option<String>,
                                 pub turn_id: Option<String>, pub prompt: String }
pub struct BeforePromptResult { pub prompt: String, pub continue_execution: bool /*=true*/,
                                pub reason: Option<String>, pub additional_context: Option<String> }
pub struct BeforeToolContext { pub agent_name: String, pub provider_session_id: Option<String>,
                               pub turn_id: Option<String>, pub call_id: String, pub tool_name: String,
                               pub arguments: JsonObject }
pub struct ToolDecision { pub decision: ToolDecisionKind, pub reason: Option<String>,
                          pub updated_arguments: Option<JsonObject>, pub additional_context: Option<String> }
pub struct AfterToolContext { pub agent_name: String, pub provider_session_id: Option<String>,
                              pub turn_id: Option<String>, pub call_id: String, pub tool_name: String,
                              pub arguments: JsonObject, pub result: JsonValue /*=Null*/, pub error: Option<String> }
pub struct AfterToolResult { pub replace_result: bool /*=false*/, pub updated_result: JsonValue /*=Null*/,
                             pub additional_context: Option<String> }
pub struct StopHookContext { pub agent_name: String, pub provider_session_id: Option<String>,
                             pub turn_id: Option<String>, pub reason: Option<String>, pub metadata: JsonObject }
```
`ToolDecision::validate`: `Rewrite` without `updated_arguments` → `"rewrite tool decision requires updated_arguments"`; non-`Rewrite` with `updated_arguments` → `"updated_arguments is only valid for a rewrite tool decision"`.
All hook contexts/results are `Serialize + Deserialize + JsonSchema` (pure JSON), no other rules.

```rust
#[async_trait]
pub trait HarnessHookDispatcher: Send + Sync {
    async fn before_prompt(&self, context: &BeforePromptContext) -> HarnessResult<BeforePromptResult>;
    async fn before_tool(&self, context: &BeforeToolContext) -> HarnessResult<ToolDecision>;
    async fn after_tool(&self, context: &AfterToolContext) -> HarnessResult<AfterToolResult>;
    async fn on_stop(&self, context: &StopHookContext) -> HarnessResult<()>;
}
```

### 5.13 `interactions.rs`

Enums: `InteractionResponseStatus{Completed,Declined,Cancelled,Failed}`; `ToolApprovalDecision{Allow,AllowForSession,Deny,Abort}`; `InteractionCancelReason{ProviderWithdrew,HarnessStopped,TurnAborted,DeadlineExceeded}` with values `provider_withdrew`,`turn_aborted`,`harness_stopped`,`deadline_exceeded`.

Requests (all `Serialize + Deserialize + JsonSchema + Validate`):
```rust
pub struct ToolApprovalRequest { pub request_id: String, pub call_id: String, pub tool_name: String,
  pub arguments: JsonObject, pub provider_session_id: Option<String>, pub turn_id: Option<String>,
  pub title: Option<String>, pub description: Option<String>, pub reason: Option<String>,
  pub deadline_at: Option<f64>, pub provider_data: JsonObject }
pub struct UserInputRequest { pub request_id: String, pub prompt: String,
  pub provider_session_id: Option<String>, pub turn_id: Option<String>,
  pub choices: Vec<String>, pub deadline_at: Option<f64>, pub provider_data: JsonObject }
pub struct McpElicitationRequest { pub request_id: String, pub server_name: String, pub prompt: String,
  pub schema: Option<JsonObject>, pub provider_session_id: Option<String>, pub turn_id: Option<String>,
  pub deadline_at: Option<f64>, pub provider_data: JsonObject }
pub struct DynamicToolCallRequest { pub request_id: String, pub call_id: String, pub tool_name: String,
  pub arguments: JsonObject, pub namespace: Option<String>, pub provider_session_id: Option<String>,
  pub turn_id: Option<String>, pub deadline_at: Option<f64>, pub provider_data: JsonObject }
pub struct ProviderInteractionRequest { pub request_id: String, pub provider: String, pub request_type: String,
  pub schema_version: String, pub payload: JsonObject, pub provider_session_id: Option<String>,
  pub turn_id: Option<String>, pub deadline_at: Option<f64> }
```
Responses:
```rust
pub struct ToolApprovalResponse { pub request_id: String, pub decision: ToolApprovalDecision,
  pub updated_arguments: Option<JsonObject>, pub reason: Option<String>, pub interrupt: bool, pub provider_data: JsonObject }
pub struct UserInputResponse { pub request_id: String, pub status: InteractionResponseStatus,
  pub content: JsonValue, pub provider_data: JsonObject }
pub struct McpElicitationResponse { pub request_id: String, pub status: InteractionResponseStatus,
  pub values: JsonObject, pub provider_data: JsonObject }
pub struct DynamicToolCallResponse { pub request_id: String, pub status: InteractionResponseStatus,
  pub result: JsonValue, pub is_error: bool, pub error_message: Option<String>, pub provider_data: JsonObject }
pub struct ProviderInteractionResponse { pub request_id: String, pub status: InteractionResponseStatus,
  pub payload: JsonObject }
```
`request_id` non-empty on every request and response: `"interaction request_id must not be empty"` / `"interaction response request_id must not be empty"`; `provider_session_id`/`turn_id` not `""` → `"interaction {field} must not be empty"`; `deadline_at` finite `>= 0` → `"interaction deadline_at must be a finite Unix timestamp"`; `ToolApprovalRequest`: `"tool approval call_id and tool_name must not be empty"`; `UserInputRequest`: `"user input prompt must not be empty"`; `McpElicitationRequest`: `"MCP elicitation server_name and prompt must not be empty"`; `DynamicToolCallRequest`: `"dynamic tool call_id and tool_name must not be empty"`; `ProviderInteractionRequest`: `"provider interaction namespace, type, and version must not be empty"`.

Dispatch enums (additive Rust extension — Python used `typing` unions):
```rust
#[serde(tag = "interaction_type", rename_all = "snake_case")]
pub enum HarnessInteractionRequest { ToolApproval(..), UserInput(..), McpElicitation(..), DynamicToolCall(..), Provider(..) }
#[serde(tag = "interaction_type", rename_all = "snake_case")]
pub enum HarnessInteractionResponse { ToolApproval(..), UserInput(..), McpElicitation(..), DynamicToolCall(..), Provider(..) }
```
Both enums: `request_id() -> &str`, `validate()`, serde, and `HarnessInteractionRequest::validate_response(&self, response: &HarnessInteractionResponse) -> HarnessResult<()>`.

```rust
pub fn validate_interaction_response(request: &HarnessInteractionRequest,
                                    response: HarnessInteractionResponse)
                                    -> HarnessResult<HarnessInteractionResponse>;
```
Rules (verbatim): id mismatch → `Protocol("interaction response id '{response}' does not match request '{request}'")`; kind mismatch → `Protocol("{RequestType} requires {ExpectedResponseType}, got {ResponseType}")` using the Rust type names (`ToolApprovalRequest requires ToolApprovalResponse, got UserInputResponse`).

```rust
#[async_trait]
pub trait HarnessInteractionHandler: Send + Sync {
    async fn handle(&self, request: &HarnessInteractionRequest) -> HarnessResult<HarnessInteractionResponse>;
    async fn cancel(&self, request_id: &str, reason: InteractionCancelReason) -> HarnessResult<()>;
    /// Convenience for Python's `reason=InteractionCancelReason.PROVIDER_WITHDREW` default.
    async fn cancel_default(&self, request_id: &str) -> HarnessResult<()> { self.cancel(request_id, InteractionCancelReason::ProviderWithdrew).await }
}
```

### 5.14 `tools.rs`

```rust
pub struct ToolDefinition { pub name: String, pub description: String, pub input_schema: JsonObject }
pub struct ToolInvocation { pub call_id: String, pub name: String, pub arguments: JsonObject }
pub struct ToolExecutionResult { pub content: JsonValue, pub is_error: bool /*=false*/ }
pub enum McpTransport { Stdio="stdio", Http="http", InProcess="in_process" }
pub struct McpInstance(/* Arc<dyn Any + Send + Sync> */);
pub struct McpServerConfig { pub name: String, pub transport: McpTransport, pub command: Vec<String>,
  pub url: Option<String>, pub instance: Option<McpInstance>, pub env: BTreeMap<String, String>,
  pub headers: BTreeMap<String, String> }
```
- `ToolDefinition::validate`: `"tool definition name must not be empty"` (description may be empty).
- `ToolInvocation::validate`: `"tool invocation call_id and name must not be empty"`.
- `McpInstance::new<T: Any + Send + Sync>(T)`, `downcast_ref::<T>() -> Option<&T>`, `Debug` → `<opaque instance>`.
- `McpServerConfig::validate` (order matters): name non-empty (`"MCP server name must not be empty"`); then transport target checks: `Stdio` + empty `command` → `"stdio MCP server requires a non-empty command"`; `Http` + no `url` → `"HTTP MCP server requires url"`; `InProcess` + no `instance` → `"in-process MCP server requires instance"`; `env` non-empty on non-`Stdio` → `"MCP env is only valid for stdio transport"`; `headers` non-empty on non-`Http` → `"MCP headers are only valid for HTTP transport"`; exactly one of (`command` non-empty, `url`, `instance`) → `"MCP server requires exactly one of command, url, or instance"`.

```rust
#[async_trait]
pub trait ToolGateway: Send + Sync {
    async fn definitions(&self) -> HarnessResult<Vec<ToolDefinition>>;
    async fn invoke(&self, invocation: &ToolInvocation) -> HarnessResult<ToolExecutionResult>;
}
```

### 5.15 `stream.rs`

```rust
#[async_trait]
pub trait HarnessEventCursor: Send {
    /// Yields `None` once the view is exhausted or closed.
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>>;
    /// Releases the cursor lease; idempotent.
    async fn close(&mut self) -> HarnessResult<()>;
}

pub type HarnessEventStream = futures::stream::BoxStream<'static, HarnessResult<HarnessEvent>>;

pub struct EventCursorStream { /* cursor + close state machine */ }
impl EventCursorStream { pub fn new(cursor: Box<dyn HarnessEventCursor>) -> Self }
impl Stream for EventCursorStream { type Item = HarnessResult<HarnessEvent>; }   // closes the cursor when it returns None
pub fn event_cursor_stream(cursor: Box<dyn HarnessEventCursor>) -> HarnessEventStream;

pub struct VecEventCursor { /* VecDeque<HarnessEvent>, closed flag */ }
impl VecEventCursor { pub fn new(events: Vec<HarnessEvent>) -> Self; pub fn remaining(&self) -> usize; pub fn is_closed(&self) -> bool }
```
- Documented contract (from Python): `PAUSED`/`RESUMED` do not end a turn view; `FINISHED`/`ABORTED`/`FAILED` do; `events()` is cycle-long; a second concurrently active cursor raises `HarnessStateError` (provider obligation); `aclose()` is idempotent and does not rewind; dropping a cursor must release the lease (document that `Drop` is the release path when a caller abandons a cursor without `close`).
- `VecEventCursor` is the reference implementation used by fakes/tests.

### 5.16 `protocol.rs`

```rust
#[async_trait]
pub trait HarnessProtocol: Send + Sync {
    fn card(&self) -> &HarnessCard;
    fn state(&self) -> HarnessState;
    fn provider_session_id(&self) -> Option<&str>;
    fn event_buffer_config(&self) -> &EventBufferConfig;
    async fn start(&self, context: HarnessContext) -> HarnessResult<()>;
    async fn stop(&self) -> HarnessResult<()>;                       // idempotent; settles Terminated
    fn events(&self) -> Box<dyn HarnessEventCursor>;
    fn turn_events(&self, turn_id: Option<&str>) -> Box<dyn HarnessEventCursor>;
    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt>;
    async fn send_auto(&self, input: HarnessInput) -> HarnessResult<SendReceipt> { self.send(input, DeliveryMode::Auto).await }
    async fn abort(&self, mode: AbortMode) -> HarnessResult<()>;
    async fn abort_graceful(&self) -> HarnessResult<()> { self.abort(AbortMode::Graceful).await }
    async fn pause(&self) -> HarnessResult<()>;
    async fn resume(&self, query: Option<HarnessInput>) -> HarnessResult<()>;
    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>>;
}

#[async_trait]
pub trait HarnessModelControl: Send + Sync {
    async fn list_models(&self) -> HarnessResult<Vec<ModelOption>>;   // requires MODEL_DISCOVERY
    async fn set_model(&self, selection: ModelSelection) -> HarnessResult<()>; // requires MODEL_SELECTION
}

pub trait HarnessProvider: Send + Sync {
    fn card(&self) -> &HarnessCard;
    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>>;   // unstarted
}

pub trait ModelConfigProvider: Send + Sync {
    fn model_config(&self, choice: &ModelChoice, fallback: Option<&ModelChoice>) -> HarnessResult<JsonObject>;
}

pub type DynHarness = Arc<dyn HarnessProtocol>;
pub type DynHarnessProvider = Arc<dyn HarnessProvider>;
pub type DynModelControl = Arc<dyn HarnessModelControl>;
```
- All harness methods take `&self`: impls use interior synchronization (documented). This keeps `Arc<dyn HarnessProtocol>` shareable, matching Python's freely shared instance.
- `send` does not wait for the turn to finish (doc comment).
- `HarnessModelControl` is deliberately separate from `HarnessProtocol` so existing implementors stay conformant.
- `create` returns an unstarted harness; sync because providers only construct.

## 6. Event wire codec (parity-critical)

`serialization.rs` is the single source of truth; `HarnessEvent`'s manual serde delegates to it.

```rust
pub const EVENT_WIRE_SCHEMA_VERSION: &str = "1";
pub fn harness_event_to_dict(event: &HarnessEvent) -> HarnessResult<JsonObject>;
pub fn harness_event_from_dict(data: JsonObject) -> HarnessResult<HarnessEvent>;
pub fn harness_event_to_json_string(event: &HarnessEvent) -> HarnessResult<String>;
pub fn harness_event_from_json_str(text: &str) -> HarnessResult<HarnessEvent>;
```

Encode — envelope with **all 12 keys, always** (`None` → `null`, empty → `[]`):
```
schema_version, event_type, sequence, timestamp, host_session_id, agent_id,
provider_session_id, turn_id, item_id, correlation_id, causation_ids, payload
```
`event_type`/`payload` per payload variant:
| variant | event_type | payload keys |
|---|---|---|
| Output | `"output"` | `output_id, kind, content, operation, channel, content_index, content_type, data` |
| ItemLifecycle | `"item_lifecycle"` | `kind, item_type, data` |
| UsageUpdated | `"usage_updated"` | `usage, mode` |
| ModelRequest | `"model_request"` | `request_id, status, started_at, ended_at, model, provider_name, system_instructions, input_messages, input_observed, output_message, tool_definitions, request_parameters, response_id, time_to_first_chunk, finish_reasons, usage, cost, error, data` |
| StateChanged | `"state_changed"` | `old, new` |
| TurnLifecycle | `"turn_lifecycle"` | `kind, result` |
| HookObserved | `"hook_observed"` | `phase, hook_name, hook_id, data` |
| Diagnostic | `"diagnostic"` | `level, message, data` |
| Provider | `"provider"` | `provider, event_type, schema_version, payload` |
| Unknown | `event.event_type` (verbatim) | `event.payload` (raw) |

Envelope `schema_version` = `EVENT_WIRE_SCHEMA_VERSION` for every known variant; for `UnknownEvent` it is the unknown event's own `schema_version`.
Nested shapes: `usage {input_tokens, output_tokens, cached_input_tokens, reasoning_output_tokens, total_tokens, provider_data}`; `cost {micros, currency}`; `message {message_id, role, content:[...], data}`; `block {block_id, kind, content, data}`; `termination {kind, message, code, provider_data}`; `error {message, code, category, retryable, provider_data}`; `result {status, messages, final_output, structured_output, stop_reason, termination, error, usage, cost, started_at, completed_at, duration_ms, provider_data}`.
All enum fields are bare value strings. Values must be plain JSON (never `serde_json::Value::Number` non-finite).

Decode:
1. `event_type` and `schema_version` must be present, string, non-empty → otherwise `Protocol("{field} must be a non-empty string")`.
2. `schema_version != "1"` → `UnknownEvent{event_type, schema_version, payload}` **before** any dispatch (never partially parse a future schema).
3. `event_type` not in the table → `UnknownEvent` (schema is `"1"`).
4. Envelope fields read with typed readers; missing/invalid → wrapped as `Protocol("invalid harness event wire object: {detail}")`. Detail wording: `"{field} must be a non-empty string"`, `"{field} must be an integer"`, `"{field} must be a number"`, `"{field} must be a JSON array"`, `"{field} must be a JSON object"`, `"unknown {field}: {value}"`, `"{field} must be a string enum value"`.
5. `causation_ids` defaults `[]`; must be a JSON array of strings.
6. Build the payload, build `HarnessEvent`, then `event.validate()`; validation errors propagate as-is (Python raises the same `ValueError` text).

`payload` absent → error (`"payload must be a JSON object"`). Extra unknown envelope keys are ignored (forward compatible). Extra unknown keys inside a *known* payload are ignored too (documented tolerance).

## 7. Deliberate deviations from Python (record in `CONTEXT.md`)

1. `ValueError`/`TypeError` → `HarnessError::Validation` (message text preserved); the four Python exception classes → `HarnessError` variants.
2. Dataclass frozen/slots + `__post_init__` → public fields + `Validate` (decode paths validate automatically; struct literals must call `validate()`).
3. Deep JSON freezing (`MappingProxyType`, list→tuple) → owned `JsonValue`; no `freeze_json_value`/`json_value_to_builtin`.
4. Python `Protocol` (duck typing) → Rust traits with `Send + Sync` supertraits, `async-trait`, and `Arc<dyn …>` in `HarnessContext`.
5. Python `typing` unions for interactions → tagged enums `HarnessInteractionRequest`/`Response` (`interaction_type`), keeping the five struct types exported individually.
6. `tuple` → `Vec`, `frozenset` → `BTreeSet`, `Mapping` → `BTreeMap`.
7. Non-negative ints → unsigned (`u64`/`usize`); negatives rejected by serde instead of a validator.
8. `HarnessContext` is not serde-serializable (holds live host services) — Python did not serialize it either.
9. `HarnessModelControl` convenience methods `send_auto`/`abort_graceful` replace Python keyword defaults.
10. No runtime `isinstance` protocol checks (Python `@runtime_checkable`); Rust trait bounds provide the guarantee at compile time.
11. Capability enums expose `as_str()` (the Python `.value`) so error messages can be sorted by wire value exactly like Python's `sorted(c.value …)`.
12. `ModelSelection` rejects an all-`None` selection (Python rule, ported); `ModelChoice` accepts it (Python parity — the provider chooses), while still rejecting blank strings.

## 8. Public surface checklist (must exist and be re-exported from `lib.rs`)

Constants: `PROTOCOL_VERSION`, `EVENT_WIRE_SCHEMA_VERSION`, `MAX_CHECKPOINT_BYTES`, `AUTH_FALLBACK_REQUEST_TYPE`, `AUTH_FALLBACK_EVENT_TYPE`, `TERMINAL_TURN_EVENT_KINDS`.
Aliases: `JsonValue`, `JsonObject`, `HarnessResult`, `HarnessErrorKind`, `HarnessEventStream`, `DynHarness`, `DynHarnessProvider`, `DynModelControl`.
Functions: `harness_event_to_dict`, `harness_event_from_dict`, `harness_event_to_json_string`, `harness_event_from_json_str`, `event_retention`, `validate_interaction_response`, `json_object_from_value`, `json_object_to_value`, `default_compatible_protocol_versions`, `event_cursor_stream`.
Traits: `Validate`, `HarnessTelemetry`, `HarnessCheckpointSink`, `HarnessHookDispatcher`, `HarnessInteractionHandler`, `ToolGateway`, `HarnessEventCursor`, `HarnessProtocol`, `HarnessModelControl`, `HarnessProvider`, `ModelConfigProvider`.
Types: every struct/enum in §5.
Crate root re-exports everything flat (like Python's eager `__init__.__all__`); module paths remain public for qualified use.

## 9. Implementation order (3 delegable rounds)

- **Round A (data core)** — `errors.rs`, `json.rs`, `validation.rs`, `state.rs`, `results.rs`, `models/{mod,card,delivery,model_control}.rs`, `lib.rs` with those re-exports; tests `tests/models.rs`, `tests/results.rs`.
- **Round B (events + wire)** — `events/*.rs`, `serialization.rs`, `lib.rs` re-export additions; tests `tests/events.rs`, `tests/serialization.rs`.
- **Round C (SPI + services)** — `checkpoints.rs`, `hooks.rs`, `interactions.rs`, `tools.rs`, `stream.rs`, `protocol.rs`, `models/context.rs`, full `lib.rs` surface; tests `tests/services.rs`, `tests/protocol.rs`.

Each round must end with `cargo fmt -p openjiuwen-harness-protocol`, `cargo test -p openjiuwen-harness-protocol`, and `cargo clippy -p openjiuwen-harness-protocol --all-targets -- -D warnings` green.

## 10. Test plan (mirrors the 33 Python cases)

`tests/models.rs` — fork_modes default empty; card capability reporting + immutability-by-value; `validate_host` (supported version, `"is not supported"`, `"missing required capabilities"`); card self-consistency errors; context retains checkpoint/sink/interactions/mcp/metadata and validates identity; model selection/option validation and defaults (`effort` only, both-none error, blank rejected, `default_effort` membership); `ModelChoice::is_builtin`; endpoint redaction in `Debug`; JSON helpers round-trip.
`tests/results.rs` — content block/message/termination/cost/usage/error validation; `MonetaryAmount` default currency `"USD"` + normalisation + three-letter check; `TurnResult` interrupted requires termination, failed requires error, cross-field prohibitions, timestamp ordering, duration; serde round-trip of a full `TurnResult`.
`tests/events.rs` — every payload validation rule (one test per message); turn lifecycle kind/status agreement; envelope validation (finite timestamp, duplicate causation ids, `turn_id` requirements); `event_retention` matrix; `EventBufferConfig` default `coalesce_or_block` + positive capacity; enum wire values (`model_selection`, `model_discovery`, `model_request_observation`, …).
`tests/serialization.rs` — full round-trip equality for `Output`(FINAL/STRUCTURED), `TurnLifecycle`, `ModelRequest` (the Python `_model_request_event` fixture shape), `Provider`, and `Unknown`; all 12 envelope keys always present; `json.dumps`-equivalence via `serde_json::to_string`; no legacy aliases (`team_session_id`, `member_agent_id`, `session_id` absent); schema-version gate (`"2"` → `UnknownEvent`, lossless re-encode); unknown `event_type` with schema `"1"` → `UnknownEvent` preserving payload; decode rejects bad envelope values with the Python messages; decode of a known payload with extra unknown keys succeeds.
`tests/services.rs` — checkpoint field emptiness + negative-sequence rejection at decode + oversize data (`"exceeds"`) + `size_bytes`; save receipt; checkpoint sink idempotency/CAS contract exercised through a fake (returns cached receipt, `CheckpointConflict` for stale sequence and CAS mismatch); hook decision rewrite rules; MCP transport exclusivity matrix (3 missing-target cases, multiple-target, misplaced env/headers); tool definition/invocation validation; interaction request/response validation, `validate_interaction_response` matching + mismatch cases, cancel recording.
`tests/protocol.rs` — trait-object safety/`isinstance`-equivalents: `Arc<dyn HarnessProtocol>`, `Box<dyn HarnessProvider>`, `Arc<dyn HarnessCheckpointSink>`, `Arc<dyn HarnessInteractionHandler>`, `Arc<dyn HarnessModelControl>`, `Arc<dyn ToolGateway>`, `Arc<dyn HarnessHookDispatcher>`, `Arc<dyn HarnessTelemetry>`; fake harness exercising `start/stop/send/turn_events/events/export_checkpoint`; cursor ordering (`[2,3,4,5,6]` for a turn view, finite, terminal-event inclusive), idempotent `close()`, fresh `events()` continues after a consumed turn view; `event_cursor_stream` yields the same order and closes on exhaustion; `VecEventCursor` semantics.

Tests use `futures::executor::block_on`; no network, no sleeps, no tokio.
