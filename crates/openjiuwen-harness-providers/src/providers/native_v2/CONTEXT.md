# native_v2 provider (agent-teams-hosted adapter)

## Intent

The `native_v2` registry entry: the Rust counterpart of Python's `NativeHarnessProtocolAdapter` (Python `openjiuwen.harness_providers`, v0.1.20).
It implements no harness of its own: `NativeV2Driver` hosts one `openjiuwen-agent-teams` team and drives it turn by turn, while `vendor_cli::VendorHarness` supplies the whole `HarnessProtocol` surface (lifecycle, turn queue, event sequence, capability gating, bounded teardown).
The team runtime stays in `openjiuwen-agent-teams` (`TeamRuntimeManager`, `TeamAgent`, `TeamBuildDeps`, `MemberHarnessFactory`, `schema::TeamAgentSpec`); this module only maps a create mapping onto it and projects its stream.
The module exists only behind the opt-in `native-v2` feature (`dep:openjiuwen-agent-teams`, default off), so a default build never compiles the team runtime.
This provider lives here, not in `openjiuwen-agent-teams`: that crate supplies the runtime and owns no provider.

## API Surface

Constants: `PROVIDER_NAME` (`"native_v2"`), `CARD_NAME` (`"native_v2"`), `IMPLEMENTATION_VERSION` (`"0.2.0"`), `DEFAULT_EVENT_BUFFER_CAPACITY` (`1024`).
`NativeV2Config { team: TeamAgentSpec, session_id: Option<String>, language: Option<String>, event_buffer_capacity: u64 }` — serde `snake_case`; the plain serde decoder tolerates unknown keys (crate-wide rule) while `from_json(&JsonObject)` rejects them (the provider SPI path).
`NativeV2Config::from_json` validation, all `Error::Validation`:

| Key | Default | Rejections |
| --- | --- | --- |
| `team` | mandatory | `"native_v2 requires a team specification"` (absent or null), `"team must be a JSON object"`, `"team is not a team specification: <serde error>"`, or the blueprint's own `TeamAgentSpec::validate` message (e.g. `"agents dict must contain a 'leader' key"`) |
| `session_id` | host session id of `start` | `"native_v2 session_id must not be empty"` (blank after trim), wrong type |
| `language` | team-runtime default | `"native_v2 language must not be empty"`, wrong type |
| `event_buffer_capacity` | `DEFAULT_EVENT_BUFFER_CAPACITY` (1024) | `"native_v2 event_buffer_capacity must be positive"` (0), wrong type |

The only accepted keys are `CONFIG_FIELDS` = `["team", "session_id", "language", "event_buffer_capacity"]`; anything else is `"unknown native_v2 configuration fields: <comma list>"`.
`NativeV2Config::event_buffer_config() -> EventBufferConfig { capacity: event_buffer_capacity, ..default }`.
`card() -> HarnessCard`; `provider() -> NativeV2HarnessProvider` (also `Default`), which implements `HarnessProvider` (`card`, `create`) and is registered as `crate::provider::plain_provider(native_v2::provider())` in `../mod.rs`.
`NativeV2Driver::{new(config), with_deps(config, deps), config()}`, re-exported from `driver`; it implements `vendor_cli::VendorDriver` only.
`new` takes `TeamBuildDeps::default()` (member models resolved from the environment through `NativeMemberFactory::from_env`, in-memory team store, temp-dir workspaces), so an offline host injects its own through `with_deps`.
`with_deps` writes a configured `language` onto the deps via `TeamBuildDeps::with_language`, matching the team crate's precedence (deps language wins over the spec).
The driver's `Debug` prints the team name, language, buffer capacity and the live team session id, and is `finish_non_exhaustive`: never the blueprint (it may carry model credentials) and never the injected deps.

### Card

`capabilities` is exactly `PersistentSession`, `GracefulAbort`, `ForceAbort`, `PauseResume`, `Steer`, `NativeTools` — what the driver reaches through `TeamAgent::runtime()` and `TeamAgent::stop`.
`optional_host_capabilities` is `UserInput` + `CheckpointSink`; `required_host_capabilities` and `fork_modes` are empty.
Absent by design: `Checkpoint`, `McpTools`, `ModelSelection`, `ModelDiscovery`, `EndpointModels`, `AuthFallback`, `Hooks` — the adapter has no portable team snapshot, never reconfigures a member's model and never answers a user-input request itself.

### Turn projection

| Team chunk kind | Observation |
| --- | --- |
| `llm_output` | answer-channel `OutputEvent` (Delta, `content_index` 0, `output_id` `native-v2-answer`), non-empty text also accumulated into the final answer |
| `llm_reasoning` | reasoning-channel `OutputEvent` (`output_id` `native-v2-reasoning`) |
| `tool_call` | `tool_call` item `Started` (`tool_call_id`, `name` default `"tool"`, `arguments`) |
| `tool_result` | `tool_call` item `Completed` (`tool_call_id`, `tool_name` default `"tool"`, `result` default the whole payload, `rendered_result`) |
| anything else | `ProviderEvent` under the `"native_v2"` namespace with the unwrapped payload |

The leader forwards each frame as `{"source_member", "role", "chunk": {"type", "payload"}}`; the projection strips both wrappers, and a frame with no `chunk` key is used as-is.
Tool fields are read from a `data` object when present and from the payload itself otherwise; a chunk without `item_id` gets `item-<n>` (the completion fallback is `max(item_index, 1)`).

### Driver commands

`open` activates the team (`TeamRuntimeManager::activate(team, session_id, {})`), stores manager + team, clears any recorded abort, and returns `team.session_id()` as the provider session id; the activation session is `config.session_id` or `context.host_session_id`.
`close` clears the abort, takes both handles and calls `TeamAgent::stop`, which never fails (teardown warnings are logged inside the team runtime); the manager is then dropped.
`run_turn` streams `team.stream(turn.content, None)`, selects between the abort watch and the stream, and ends `Completed` with the accumulated answer (`final_output` = the answer string, one assistant text message — none when the answer is empty), `Interrupted` on an observed/contended abort, or `Failed` (`code` `team_failure`, `category` `team`, `retryable` false, message `"the native_v2 team run failed: <error>"`).
`steer` renders the steer payload through `TurnRequest::new("", "", content, {}).text()` and calls `team.runtime().send(text, true)`.
`interrupt` calls `team.runtime().abort(mode == AbortMode::Force)` and then records the mode for the running turn.
`pause` calls `team.runtime().pause()`; `resume` calls `team.runtime().resume(query.map(harness_input_text))`.
Every command except `close`/`open` needs an open cycle: no activated team is `HarnessError::State("the native_v2 harness is not started")`.

## Constraints

- Feature gating: without `native-v2` the module does not exist, `providers::builtin_provider("native_v2")` returns `Error::Unsupported` (`"harness provider 'native_v2' is disabled; rebuild with the `native-v2` feature to register …"`), and the descriptor reports `implemented: true`, `configures_models: false`, `requires: Some("openjiuwen-agent-teams")`; with the feature `requires` is `None` and resolution yields this provider.
- The descriptor's `implemented` is `true` in both builds — the name is declared metadata either way; only resolution and `requires` differ.
- The provider is deliberately **not** a `ModelConfigProvider` and is registered through `plain_provider`: a team member's model comes from the injected `MemberHarnessFactory`, never from the create mapping, so `provider_model_config("native_v2", …)` reports the capability gate.
- The driver must never implement `HarnessProtocol` and never spawns a process: `VendorHarness` owns the lifecycle, the turn queue, the event sequence, capability gating and teardown.
- Capability gating happens in the shell before the driver is consulted, so a checkpoint/MCP/model command on this card is `UnsupportedCapability` without touching the team.
- One team per start/stop cycle: `open` builds a fresh `TeamRuntimeManager` over the driver's deps, so a new cycle activates the blueprint again rather than resuming the previous live team.
- Errors: `config_error_to_harness` maps every config error to `HarnessError::Validation` (a `Validation` message verbatim, anything else rendered with its workspace prefix); `team_error` keeps `Validation` → `Validation` and `Unsupported` → `UnsupportedCapability`, and reports every other team failure as `HarnessError::Protocol` with the rendered text.
- The driver's `state: Mutex<DriverState>` is only held for handle reads/writes (never across `.await`), and a poisoned lock is recovered (`PoisonError::into_inner`) so one panicking team callback cannot brick the driver.
- The harness never creates, selects or injects a model, and this module depends on no `openjiuwen-core` symbol; its only team surfaces are `runtime::{TeamAgent, TeamBuildDeps, TeamRuntimeManager}` and `schema::TeamAgentSpec`.
- Tests never start a network or vendor process: the driver is exercised through in-process `openjiuwen-agent-teams` fakes.

## Design Decisions

- No interruption channel in `TeamAgent::stream`, so the driver keeps one: a `tokio::sync::watch` of `Option<AbortMode>` (`aborts`), biased-selected ahead of the stream; `interrupt` records the mode after the leader runtime's own abort, and the running turn returns `interrupted_turn` as soon as it observes the record.
- The recorded mode is cleared at every turn start (and on `open`/`close`), so a stale abort can never end a later turn; a stream error after an observed abort still reports `Interrupted`, and a closed watch channel stops the abort branch and lets the run drain.
- Both abort modes terminate `TurnTerminationKind::UserAbort` with message `"native_v2 interrupted the turn"`: the shell already publishes `Aborted(Graceful|Force)` from the requested mode, so the result only needs to say the host asked.
- No `deep_agent`/`agent_template` tree in the create mapping: the Rust team runtime takes a `TeamAgentSpec` directly and members get their harnesses from the injected `MemberHarnessFactory`, so Python's spec-tree keys are rejected as unknown fields.
- Activation uses an empty payload (`{}`) because the driver consumes no run inputs at activation time; the provider session id is the team's session id and stays sticky through the shell's first-reported-id rule.
- The card is the single gate for `Checkpoint`: the driver has no portable team snapshot and `VendorDriver::export_checkpoint` keeps its default `UnsupportedCapability` body, which the shell reports as `does not support checkpoints`.
- The `UserInput` and `CheckpointSink` optional host capabilities are advertised as available seams, not as implemented interactions: the projection has no interaction branch, so an interaction-shaped team chunk leaves as a `ProviderEvent`.
- `language` is applied to the deps instead of the spec because `TeamBuildDeps::language` wins over the blueprint in `openjiuwen-agent-teams`.
- The local `config_error_to_harness` (rather than the vendor helper of the same name) keeps this module free of the vendor configuration helpers: its inputs are a workspace `Error`, not a vendor config error.

## Deferred

- Checkpoint export: the card declares no `Checkpoint` and the driver exports no snapshot.
- MCP tools and model selection/discovery: no capability, no `HarnessModelControl`, no model-config hook.
- Fork modes: none declared, none implemented.
- Provider events beyond the four projected kinds: any other team chunk is forwarded verbatim as a `ProviderEvent` rather than normalised into lifecycle, usage or interaction observations; token usage from the team stream is not turned into `usage` observations.
- Host user input and checkpoint sinks: advertised as optional host capabilities but never requested by the driver.
- Persisting/restoring a live team across a stop/start pair: each cycle reactivates the blueprint over the injected deps.
- Python adapter behaviors with no Rust counterpart here: manifest-driven provider config, per-member model reconfiguration from the create mapping, and org/observability layers of the team runtime.

## Test Strategy

- `crates/openjiuwen-harness-providers/tests/native_v2.rs` (feature `native-v2`, 12 tests, offline, 10 s `BOUND` on every await) drives the provider through the public surface only: `provider().create`, the returned `VendorHarness`, and `NativeV2Driver::with_deps` over a local `FakeHarnessFactory`/`ScriptedHarness` copy (another crate's `tests/` module is not importable).
- Coverage: unstarted create with the v2 card, missing-team/unknown-key rejection, one full turn (Started → answer → Finished, team session id as provider session id, one created leader harness), a second turn keeping the same team session and one leader harness, checkpoint export gated by the card, graceful and forced abort reaching the member runtime with a `UserAbort` termination, pause/resume reaching the runtime plus `stop` on close, steering rejected with `State` before a turn and delivered during one, tool chunks projected as `tool_call` Started+Completed items, and create-mapping decoding of a validated blueprint.
- Module unit tests in `mod.rs` (9): name/version constants, the exact card (capabilities, optional host caps, empty required caps and fork modes), provider identity without a model hook, unstarted create with default and explicit buffer capacity, and the create rejections (missing team, unknown key, non-object team, invalid blueprint, zero capacity, blank `session_id`/`language`).
- `cargo test -p openjiuwen-harness-providers --features native-v2` also builds the crate's default suites; this binary needs no network, no vendor CLI and no credentials.

## Routing Table

- `./mod.rs` → constants, `CONFIG_FIELDS`, `NativeV2Config` (+ `from_json`, `event_buffer_config`), `decode_team`, `card()`, `provider()`, `NativeV2HarnessProvider` + its `create`, `config_error_to_harness`, unit tests
- `./driver.rs` → `NativeV2Driver`, `DriverState`, the `VendorDriver` impl (`open`/`close`/`run_turn`/`steer`/`interrupt`/`pause`/`resume`), abort watch helpers, `project_chunk` + payload helpers, `termination_kind`, `team_error`, `lock`
- `../mod.rs` (outside this node) → the `#[cfg(feature = "native-v2")]` module gate, `native_v2_provider()` via `plain_provider`, and the compiled-out resolution message
- `../Cargo.toml` (outside this node) → `native-v2 = ["dep:openjiuwen-agent-teams"]`
- `../../factory.rs` (outside this node) → the `native_v2` descriptor (`CARD_NAME`, `0.2.0`, `implemented: true`, feature-dependent `requires`)
- `crates/openjiuwen-harness-providers/tests/native_v2.rs` (outside this node) → the end-to-end suite listed under Test Strategy
