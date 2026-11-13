# openjiuwen-harness-providers

## Intent

Provider registry and adapters for the provider-neutral harness SPI (`openjiuwen-harness-protocol`).
Ports the Python package `openjiuwen.harness_providers` (v0.1.20, ~13.6k lines): the process-global provider registry/factory, the `native` provider (in-process DeepAgent harness), the `native_v2` provider (a team runtime hosted by `openjiuwen-agent-teams`), the three subprocess vendor providers (`claudecode`, `codex`, `dsh`), the `HarnessIOAdapter` that projects a harness onto the kernel IO surface, provider-specific config and card data types, and the portable skill-bundle installer.
The crate owns *provider selection and adaptation*; it never reimplements a harness — the native harness lives in `openjiuwen-harness`, the team harness in `openjiuwen-agent-teams`, and the vendor harnesses are the vendors' own CLIs driven over stdio by the shared `vendor_cli` transport and the generic `VendorHarness` shell.
No vendor SDK is linked and no vendor binary is bundled: a vendor harness is a child process plus a wire-protocol mapping.

## API Surface

Registry and factory (module `factory`, flat re-exports):
- `HARNESS_PROVIDER_ENTRY_POINT_GROUP` (`"openjiuwen.harness_providers"`), `PROVIDER_NAMES` (`["native", "native_v2", "claudecode", "codex", "dsh"]`), `provider_names::{NATIVE, NATIVE_V2, CLAUDE_CODE, CODEX, DSH}`.
- `ProviderSpec` — one resolvable provider: `card()`, `create(config)`, `configures_models()`, `model_config(choice, fallback)`; wrappers `plain_provider(p)` and `model_configuring_provider(p)` adapt any `HarnessProvider` (+ optional `ModelConfigProvider`).
- `ProviderFactory` (`Arc<dyn Fn() -> Result<Arc<dyn ProviderSpec>> + Send + Sync>`).
- `register_harness_provider(name, factory) -> Result<()>`, `unregister_harness_provider(name)`, `available_providers() -> Vec<String>`, `resolve_provider(name) -> Result<Arc<dyn ProviderSpec>>`.
- `create_harness(provider, config) -> Result<Box<dyn HarnessProtocol>>` (never starts the harness).
- `provider_model_config(provider, choice, fallback) -> Result<JsonObject>` with the Python capability gates and message text.
- `ProviderDescriptor { name, card_name, implementation_version, builtin, implemented, configures_models, requires }`, `provider_descriptors()`, `describe_provider(name)`; vendor entries report `implemented: true` / `requires: None` exactly when the `vendor` feature is compiled in, and `native_v2` reports `requires: Some("openjiuwen-agent-teams")` only without `native-v2`.
- `HarnessContextOptions` (`new(host_session_id, system_prompt)` + public fields, `src/factory.rs:394`, `:431`) + `build_harness_context(provider, options) -> Result<HarnessContext>` — manifest-free variant of Python `build_harness_context` (defaults, capability injection, native MCP refusal).

Providers (module `providers`, one directory or file per registry name):
- `providers::native` — `provider() -> NativeHarnessProvider`, `card()`, `PROVIDER_NAME` (`"native"`); re-exports `NativeHarness`, `NativeHarnessProvider`, `NativeHarnessOptions` from `openjiuwen_harness`; gated by the `native` feature.
- `providers::native_v2` (`native-v2`) — `provider()`, `card()`, `PROVIDER_NAME` (`"native_v2"`), `CARD_NAME` (`"native_v2"`), `IMPLEMENTATION_VERSION` (`"0.2.0"`), `DEFAULT_EVENT_BUFFER_CAPACITY` (`1024`), `NativeV2Config { team, session_id, language, event_buffer_capacity }` (+ `from_json`, `event_buffer_config`), `NativeV2Driver` (+ `with_deps`).
- `providers::claudecode` (`vendor`) — `ClaudeCodeHarnessProvider`, `ClaudeDriver`, `ClaudeCodeHarnessConfig`, `ClaudeModelConfig`, `ClaudePermissionMode`, `ClaudeSystemPromptMode`, `DEFAULT_CLAUDE_MAX_BUFFER_SIZE` (32 MiB), `PROVIDER_NAME`, `CARD_NAME` (`"claude-code"`), `IMPLEMENTATION_VERSION` (`"0.1.0"`), `card()`, `provider()`, `model_config`.
- `providers::codex` (`vendor`) — `CodexHarnessProvider`, `CodexDriver`, `CodexHarnessConfig`, `CodexModelConfig`, `CodexApprovalMode`, `CodexSystemPromptMode`, `PROVIDER_NAME`, `CARD_NAME` (`"codex"`), `IMPLEMENTATION_VERSION` (`"0.1.0"`), `card()`, `provider()`, `model_config`.
- `providers::dsh` (`vendor`) — `DshHarnessProvider`, `DshDriver`, `DshHarnessConfig`, `DshSystemPromptMode`, `PROVIDER_NAME` (`"dsh"`), `CARD_NAME` (`"deepseek-harness"`), `IMPLEMENTATION_VERSION` (`"0.3.0"`), `card()`, `provider()`, `model_config`.
- `providers::vendor` (`vendor`) — `config_error_to_harness`, `model_fragment`, `nested_model_config`, `vendor_card` (crate-internal helpers only; no harness behavior).

Shared vendor transport (module `vendor_cli`, re-exported flat from the crate root; details in `./src/vendor_cli/CONTEXT.md`):
- `binary` — `BinarySpec`, `resolve_binary`, `is_available`.
- `jsonl` — `parse_line`, `parse_line_labelled`, `encode_line`, `get_str`, `get_object`, `get_array`, `get_bool`, `get_u64`, `get_f64`, `string_list`, `snippet`.
- `child` — `ChildProcessConfig`, `ChildProcess` (`spawn`, `write_json`, `write_line`, `writer`, `read_json`, `stderr_tail`, `close_stdin`, `kill`, `wait`, `pid`), `StdinWriter` / `LineWriter` (a cloneable writer that keeps a parked reader from blocking a control message), `DEFAULT_MAX_LINE_BYTES`, `STDERR_TAIL_LINES`, `STDERR_TAIL_CHARS`, `EXIT_TIMEOUT`.
- `rpc` — JSON-RPC 2.0 over that transport: `RpcTransport`, `JsonRpcClient` (`spawn`, `request`, `request_with_timeout`, `notify`, `next_incoming`, `respond`, `respond_error`, `close`, `is_closed`, `label`, `stderr_tail`), `request_id_value`.
- `events` — `EventBus` (`new`, `emit`, `events`, `turn_events`, `close`, `is_closed`, `len`, `is_empty`, `DEFAULT_CAPACITY`), `EnvelopeBuilder` (`new`, `set_provider_session_id`, `next`, `next_for_turn`, `next_for_item`).
- `permissions` — `PermissionPolicy` (`from_context`, `interactive`), `ApprovalDecision`, `ask_tool_approval`, `ask_user_input`.
- `session` — `derive_session_id`, `resolve_resume_id`, `checkpoint_str`.
- `driver` — the adapter seam `#[async_trait] VendorDriver { label, open, close, run_turn, steer, interrupt, pause, resume, export_checkpoint }` (unsupported commands keep default bodies returning `HarnessError::UnsupportedCapability`), `TurnRequest` (`new`, `text`), `TurnContext` (`new`, `emit`, `emit_correlated`, `emit_event`, `output`, `tool_started`, `tool_completed`, `usage`, `provider_event`, `diagnostic`), the result constructors `completed_turn`/`interrupted_turn`/`failed_turn`/`turn_error`, `TOOL_CALL_ITEM_TYPE`, `PROVIDER_SCHEMA_VERSION`.
- `harness` — `VendorHarness::new(HarnessCard, EventBufferConfig, Arc<dyn VendorDriver>)` + `label()`, the complete `HarnessProtocol` shell: lifecycle, turn queue + single supervisor task, per-turn event sequence, capability gating, abort/pause/resume, checkpoint delegation, bounded teardown.

IO adapter (module `io_adapter`):
- `HarnessIOAdapter`, `HarnessIOAdapterConfig`, `EventObserver`, `ProviderInteractionHandler`.
- `HarnessIOAdapter::{new, harness, config, state, session_id, pending_interrupt_ids, has_pending_interrupt, is_pending_interrupt_resume_valid, prepare_context, start, stop, outputs, delivery_mode, send, abort, pause, resume}` and `impl HarnessInteractionHandler` (`handle`, `cancel`); `start`/`outputs`/`prepare_context` take `self: &Arc<Self>` because the adapter installs itself as the context's interaction handler and owns its pump task.
- Chunk kinds: `llm_output`, `llm_reasoning`, `tool_call`, `tool_result`, `__interaction__`, `turn_lifecycle`, `task_failed` — projected as `openjiuwen_agent::AgentChunk`.

Skills (module `skills`; details in `./src/skills/CONTEXT.md`):
- `SkillSource` (`dir`, `mode`, `enabled_skills`; `new`, `with_mode`, `with_enabled_skills`, `from_json`), `SkillMode` (`all`/`auto_list`), `SkillConflict` (`skip`/`replace`/`append`, `FromStr`), `normalize_skills`, `install_skills`, `skill_scan_dir(provider)`.
- `SkillInstaller` / `SkillLayout` / `install_for_provider`: `for_provider("claudecode"|"codex"|"dsh", conflict)`, `provider()`, `scan_dir()`, `layout()`, `install(sources, cwd)`, `validate_name(name)`; scan dirs are `.claude/skills`, `.agents/skills`, `.dsh/skills`.

Helpers (flat re-exports):
- `harness_input_text(&HarnessInput) -> String`, `to_harness_input(JsonValue) -> HarnessInput`, `INTERACTIVE_INPUT_KIND` (`"interactive_input"`), `INTERACTION_KIND` (`"__interaction__"`), `TURN_LIFECYCLE_KIND` (`"turn_lifecycle"`), `TASK_FAILED_KIND` (`"task_failed"`).
- `to_json_object(&JsonValue) -> JsonObject`, `to_json_safe(&JsonValue) -> JsonValue`.
- `mcp_tool_naming_preamble(&BTreeMap<String, String>) -> String`, `TOOL_PLACEHOLDER` (`"{tool}"`).
- `providers_error_to_common(HarnessError) -> openjiuwen_common::error::Error` (module `error`).

## Constraints

- Depends only on `openjiuwen-harness`, `openjiuwen-harness-protocol`, `openjiuwen-common`, `openjiuwen-agent` + `openjiuwen-session` (the IO adapter's shared IO types) and, behind `native-v2` only, `openjiuwen-agent-teams`; never on `openjiuwen-core`, and never on a vendor SDK. `which` is used only by `vendor_cli::binary`.
- `openjiuwen-harness` owns every native harness behavior, `openjiuwen-agent-teams` owns the team runtime, and the vendor CLIs own their protocol; this crate wires, selects, validates and adapts — no harness logic is duplicated here.
- **Vendor transport rules**: a vendor adapter implements `VendorDriver` only and is wrapped by `VendorHarness`; it must never implement `HarnessProtocol` itself, and `VendorHarness` is the only place that publishes lifecycle events.
- One child process per `start`/`stop` cycle, spawned by the driver in `open` and reaped in `close`; nothing spawns in `create` (it only validates the config and resolves the executable) and no process may outlive its cycle.
- A missing or non-executable vendor binary is `HarnessError::Protocol` naming the binary, the install hint and the `PATH`/`cli_path` fix — never `UnsupportedCapability`, which is reserved for commands the pinned vendor protocol cannot express.
- Deterministic session ids: `derive_session_id` hashes `[host_session_id, agent_name]` into a UUID-shaped id (Claude Code), the Codex thread id is the provider session id, and DSH mints `dsh-<uuid>`; `VendorHarness` keeps the first reported id sticky and treats a later differing id as `HarnessError::State`.
- An abort with no active turn is a no-op on the wire: `VendorHarness` drains the queue as `Started`+`Aborted(HarnessStop)` pairs and ignores the requested mode.
- Capability gating happens in the shell before the driver is consulted, so a card that omits `Steer`, `GracefulAbort`, `ForceAbort`, `PauseResume` or `Checkpoint` (DSH omits all five, `native_v2` omits `Checkpoint`) yields `UnsupportedCapability` without touching the process.
- Vendor tests run exclusively against `tests/support::FakeCli` — a POSIX shell script speaking the provider's wire protocol — so `cargo test` needs no vendor CLI, no SDK and no network; `live-tests` is a reserved flag with no gated code in `src/`.
- Features: `native` (default), `vendor` (the three vendor modules + registration), `native-v2` (`dep:openjiuwen-agent-teams`, default off), `live-tests`; every feature is an empty toggle and every dependency except `openjiuwen-agent-teams` is non-optional, so `--no-default-features` still compiles `openjiuwen-harness`.
- Registry functions are synchronous and return `openjiuwen_common::error::Result`; provider SPI calls return `HarnessResult`; `error::providers_error_to_common` is the single translation point (UnsupportedCapability → `Error::Unsupported`, Validation → `Error::Validation`, State/Protocol → `Error::Config`, CheckpointConflict → `Error::Session`, wildcard → `Error::External`).
- Global registry state is process-wide and documented (mirrors Python's module dict): `std::sync::RwLock` + `indexmap` for insertion order, never held across `.await` or while a provider factory runs.
- Errors keep Python message text and Python `repr` quoting (`unknown harness provider 'x'; expected one of …`, `unknown Claude Code configuration fields: …`).
- Async-first: `HarnessIOAdapter` and `VendorHarness` each own exactly the tasks they spawn (one pump task, one supervisor task) and stop them in `stop`; no detached task outlives its owner.
- Serde: config types are `snake_case` and tolerate unknown keys when deserializing from JSON, while `from_json`/`from_mapping` rejects them; `api_key` is `SecretString` and the vendor configs' manual `Debug` redacts `env`/`settings_env`.
- No `unsafe`, no stdout logging (`tracing` only), files ≤1000 lines, all public items documented.

## Provider Matrix

| Registry name | Card name | Impl version | Status | Model-config hook | External requirement |
|---|---|---|---|---|---|
| `native` | `openjiuwen-native` (Python card: `deepagent`) | `openjiuwen-harness` crate version | implemented (`native`) | yes (`model`, effort→`temperature`) | none |
| `native_v2` | `native_v2` | `0.2.0` | implemented behind `native-v2` | no (members get their model from the injected harness factory) | none with the feature; otherwise `requires: openjiuwen-agent-teams` |
| `claudecode` | `claude-code` | `0.1.0` | implemented behind `vendor` | yes (`{"model": {…}}` + `fallback_model`; the endpoint `provider` label is dropped) | `claude` CLI on `PATH` or `cli_path` |
| `codex` | `codex` | `0.1.0` | implemented behind `vendor` | yes (`{"model": {…}}` + `fallback_model`; the endpoint `provider` label is kept) | `codex` CLI (`codex app-server --listen stdio://`) or `codex_bin` |
| `dsh` | `deepseek-harness` | `0.3.0` | implemented behind `vendor` | yes (flat `model`/`base_url`/`api_key`; rejects effort and fallback) | `dsh` runtime or `dsh_bin`, plus a mandatory `DSH_HOME` |

Cards: `claudecode` declares Steer, GracefulAbort, PersistentSession, Checkpoint, McpTools, ModelSelection, ModelDiscovery, EndpointModels, AuthFallback, optional host caps ToolApproval/UserInput/CheckpointSink/McpServers/ProviderInteraction/ModelRequestObservation and fork modes `full`/`before`.
`codex` declares Steer, GracefulAbort, PersistentSession, Checkpoint, McpTools, EndpointModels, AuthFallback, optional host caps ToolApproval/UserInput/CheckpointSink/McpServers/NativeToolGateway/DynamicToolCall/ProviderInteraction/ModelRequestObservation and fork modes `full`/`before`.
`dsh` declares PersistentSession, McpTools, EndpointModels with optional host cap McpServers and no fork modes.
`native_v2` declares PersistentSession, GracefulAbort, ForceAbort, PauseResume, Steer, NativeTools with optional host caps UserInput/CheckpointSink and no fork modes.
`native` reports the card owned by `openjiuwen_harness::protocol` (`HARNESS_NAME`).

## Routing Table

- `./Cargo.toml` → features (`native` default, `vendor`, `native-v2`, `live-tests`) and the optional `openjiuwen-agent-teams` dependency
- `./src/lib.rs` → crate docs, module declarations, flat re-exports
- `./src/provider.rs` → `ProviderSpec`, wrappers, `ProviderDescriptor`, provider-name constants, `is_builtin_provider`
- `./src/factory.rs` → process-global registry, resolution order, `create_harness`, `provider_model_config`, `provider_descriptors`, `build_harness_context`, `HarnessContextOptions`; unit tests in `./src/factory/tests.rs`
- `./src/providers/` → the built-in provider table: `mod.rs`, `native.rs`, `vendor.rs`, and one directory each for `claudecode/`, `codex/`, `dsh/` (`vendor`) and `native_v2/` (`native-v2`), with their own CONTEXT.md
- `./src/vendor_cli/` → the shared subprocess transport and the generic `VendorHarness` shell (`binary`, `jsonl`, `child`, `rpc`, `events`, `permissions`, `session`, `driver`, `harness`; own CONTEXT.md)
- `./src/skills/` → `mod.rs` (bundle discovery, conflict policies, portable install) + `installer.rs` (`SkillInstaller`, `SkillLayout`, per-vendor name rules); own CONTEXT.md
- `./src/io_adapter/` → `HarnessIOAdapter`, `HarnessIOAdapterConfig`, `EventObserver`, `ProviderInteractionHandler` (`mod.rs`), event→chunk projection (`projection.rs`), interaction routing and the pending map (`interactions.rs`), host-answer parsing (`answers.rs`)
- `./src/decode.rs` → strict JSON mapping helpers shared by factory + vendor configs (unknown-key rejection, typed option decoding)
- `./src/error.rs` → `HarnessError` → `openjiuwen_common::error::Error` translation
- `./src/input.rs` → `harness_input_text`, `to_harness_input`, chunk/interaction kind constants
- `./src/json.rs` → `to_json_safe` / `to_json_object`
- `./src/mcp_naming.rs` → `<mcp-tools>` naming preamble
- `./tests/` → offline suites and the `FakeCli` fixture (`tests/CONTEXT.md`)

## Design Decisions

- Python's factory keyed providers by name and probed `isinstance(provider, HarnessProvider)`/`ModelConfigProvider` at runtime; Rust cannot specialize on optional trait impls, so `ProviderSpec` is the single registered interface and the two wrapper constructors pick the model-config behavior (documented deviation from Python's duck typing).
- Python's `native` provider is *not* a `ModelConfigProvider`; the Rust native harness implements `ModelConfigProvider` (effort→temperature), so the Rust `native` provider registers with the hook. Documented deviation. `native_v2` is deliberately registered through `plain_provider`: it configures no models.
- `create_harness("native", …)` yields a `DeepAgent` with **no injected model**: the harness stays `Terminated` and `start` fails with `Protocol("DeepAgent requires a model: …")` unless the config supplies `model` (plus `provider`/`api_base`/`api_key` for lazy `init_model` resolution). Offline turn-driving tests therefore bypass the factory (`NativeHarness::new(DeepAgentBuilder::model(...))`) or point `api_base` at a local mock server.
- `native` config keys are the Rust `NativeHarnessProvider` key set (`model`, `provider`, `api_key`, `api_base`, `temperature`, `workspace`, `enable_task_loop`, `max_iterations`, `system_prompt`, `permissions`, `language`); Python's `deep_agent`/`agent_template`/`session_id`/`event_buffer_capacity` keys are not accepted because the Rust DeepAgent takes live handles through `DeepAgentBuilder` instead of a spec tree.
- `native_v2` is the Rust counterpart of Python's `NativeHarnessProtocolAdapter`, hosted here instead of in `openjiuwen-agent-teams`: the team crate supplies `TeamRuntimeManager`/`TeamAgent`/`MemberHarnessFactory`, this crate supplies the provider, the create mapping and the driver. Its create schema is `{team, session_id, language, event_buffer_capacity}` — no `deep_agent`/`agent_template` tree, because the Rust team runtime takes a `TeamAgentSpec` and members get harnesses from the injected factory.
- `native_v2` interruption rides a `watch` channel: `TeamAgent::stream` has no interruption channel, so `interrupt` records the requested `AbortMode` (after delegating to the leader runtime's `abort`) and the running turn returns `interrupted_turn` as soon as it observes the record; the value is reset at every turn start so a stale abort cannot end a later one.
- Manifest helpers (`load_manifest`, `manifest_provider_config`, `manifest_mcp_servers`) are not ported: the agent-template spec and manifest loader do not exist in `openjiuwen-harness`; `create_harness` therefore takes `(provider, config)` and `build_harness_context` takes explicit options instead of a manifest.
- `available_providers` keeps built-in order first, then registration order, de-duplicated; re-registering a non-built-in name overwrites in place (Python dict semantics), registering an empty or built-in name is `Error::Validation`.
- Unknown provider errors use `Error::NotFound`; known-but-compiled-out providers use `Error::Unsupported` naming the feature to rebuild with (distinct from Python's single `ValueError`, documented). The `native` entry stays `implemented: true` in the descriptor even in a build without the `native` feature, while `resolve_provider` reports the missing feature.
- The registry is global (like Python's module dict) to keep CLI/agent-teams call sites identical; tests use unique provider names instead of a reset hook.
- The Claude Code adapter matches Python's stream-json contract but drives it as a plain child process instead of the Claude Agent SDK: one user frame per turn, `system/init` for the session id, `control_request/can_use_tool` answered by the host policy, `result` as the turn terminator, and every other frame forwarded as a provider event.
- Codex and DSH speak JSON-RPC 2.0 over the same child transport; their `-c` overrides and `DSH_HOME`/`--profile`/`--patch` argv are built before anything spawns, so a bad configuration path fails in `create` rather than in `start`.
- `HarnessIOAdapter::outputs` is single-consumer and returns `HarnessError::State` on a second call — the Rust equivalent of Python's one async generator fed by the pump task; `vendor_cli::EventBus` exposes the same single-consumer lease to the driver-facing side.
- The event pump is exactly one `tokio::spawn` task holding a `Weak<HarnessIOAdapter>` (no reference cycle with the stored join handle); it ends on `None`, on a cursor error (logged with `tracing::warn!`), or when `stop` aborts it, and always closes the cursor. Chunk indexes are allocated lazily so events that project to nothing consume no index.
- `start` applies `prepare_context` itself (Python parity) and installs the adapter as `context.interactions` only when the host supplied none; `stop` aborts the pump, cancels every pending interaction, then stops the harness and is terminal for that adapter instance.
- `send` resolves pending interactions from `interrupt_answers` first and then `user_inputs`, keyed by request id; an interactive-shaped payload matching no pending id is forwarded with `metadata["kind"] = "interactive_input"`. Python's extra "sole pending request answered from `raw_inputs`" rule is not ported: `raw_inputs` only marks the payload as interactive-shaped.
- Approval answers are ported as data: `"yes"`/`"always"`/`"no"` words, `{"action": …}`, `{"approved": …, "feedback": …}`, `{"updated_arguments": …}` and `{"remember": true}`; every unrecognized shape becomes `Deny`.
- `build_harness_context` **drops** `options.mcp_servers` for the `native` provider and never injects `HostCapability::McpServers` there (the native harness owns its tools, Python raises); every other provider (vendor and `native_v2`) gets the servers copied plus the capability when the list is non-empty, and a checkpoint sink always injects `HostCapability::CheckpointSink`.
- Declared provider descriptors are static metadata: `vendor` and `native-v2` never rewrite a name, `resolve_provider` reports the missing feature instead, and a registered factory that panics or fails is skipped in the listing.
- `to_json_safe` is an identity in Rust (values are already JSON-safe); `to_json_object` wraps non-objects as `{"value": …}`. Documented deviation.
- `install_skills` keeps Python's safety rules (reject escaping paths, reject non-regular files, atomic stage-then-rename) but materializes internal symlinks by copying through them instead of recreating them.
- `install_skills` discovers bundles recursively like Python `_discover` (a source root that is itself a bundle stops the walk; symlinked directories are not traversed) and rewrites the appended copy's front-matter `name:` line in place instead of round-tripping YAML.
- Portable skill names accept `[A-Za-z0-9._-]+`, reject empty and `..` (Python's regex is stricter: it requires a leading alphanumeric and at most 128 bytes). `SkillInstaller` layers the per-vendor rule on top: the shared portable rule for Claude Code and Codex, kebab-case for `dsh`.
- `mcp_tool_naming_preamble` emits only the `<mcp-tools>` block; Python's trailing explanatory sentence is not appended (documented deviation from `mcp_naming.py`).
- `harness_input_text` joins an array when every element carries a string `text` field; Python additionally requires each block's `type` to equal `"text"`.

## Deferred Integrations

- `claudecode`: Claude lifecycle receipts and OTLP model-request observation (`observation.py`) — `request_observation_wait_s` is decoded, defaulted and `Debug`-printed but no observation wait is issued, and `HostCapability::ModelRequestObservation` is only advertised; Python's `failure_classifier.py` (failure classification beyond the normalized turn errors).
- `codex`: rollout-trace tailing and dynamic host tools (`experimental_raw_events` is accepted but `rawResponseItem/completed`/`rawResponse/completed` are dropped, and `item/tool/call` answers JSON-RPC "method not found"); model control (`ModelSelection`/`ModelDiscovery` are absent from the card because the shell exposes no `HarnessModelControl` seam); pausing (card has no `PauseResume`).
- `dsh`: Cordis plugin overlays/composition beyond `--profile`/`--patch`, provider-notification mapping beyond the documented session events, and abort/steer/pause/checkpoint — the pinned protocol has no interrupt request, so `interrupt` keeps the default `UnsupportedCapability` body.
- Vendor fork modes: the Claude Code and Codex cards declare `full`/`before`, but neither the shell nor the protocol implements a fork; the entry is metadata only.
- `native_v2`: checkpoint export (the card declares no `Checkpoint`, so `VendorHarness` rejects it), MCP tools and model selection, and passthrough of team lifecycle events beyond `llm_output`/`llm_reasoning`/`tool_call`/`tool_result` (projected as `ProviderEvent`).
- `trajectory.py` (OTel span recorder) — belongs to `openjiuwen-extensions`; `telemetry/otlp_receiver.py` — external receiver, not ported.
- Manifest/agent-template loading and `HarnessContext` persona rendering — deferred with the agent-template spec.
- Installed entry-point discovery (`importlib.metadata`) has no Rust analogue; providers are registered in-process or compiled in.

## Test Strategy

- `tests/registry.rs` — name resolution for every declared name, unknown-provider message, duplicate/overwrite semantics, built-in-name rejection, unregister, `available_providers` ordering/dedup, `provider_model_config` gates and fragments, `build_harness_context` defaults and capability injection, `create_harness` returns an unstarted harness, the built-in descriptors and their feature-dependent `implemented`/`requires`.
- `tests/native.rs` — native provider card/descriptor, create from a valid and an invalid mapping, unstarted state, start/stop through `NativeHarness` driven by `openjiuwen_test_support::ScriptedModel`, send/event cursor/`turn_events`/stop, model discovery + selection, checkpoint export, capability gating.
- `tests/io_adapter.rs` — chunk projection order and kinds, turn-lifecycle projection off by default, interactive-input round trip, unmatched input forwarding, auto-approve vs host approval, dynamic tool call routed through `ToolGateway`, delivery-mode selection, abort fallback, failed start cleanup; driven by an in-test `FakeHarness`.
- `tests/vendor.rs` (`vendor`) — vendor cards (declared and deliberately undeclared capabilities), config `from_json` strictness/defaults/cross-field rules, `model_config` fragments and rejections, serde round-trip snapshots, and the `create` path (typed `Protocol` error for a missing binary, unstarted harness for a fake executable).
- `tests/claudecode.rs` (`vendor`, `#![cfg(unix)]`) — full turn over `FakeCli` (init/session capture, text, tool call/result pair, lifecycle), permission round trip (allow/deny), interrupt path, `--session-id`/`--resume` argv, missing-result failure.
- `tests/codex.rs` (`vendor`, `#![cfg(unix)]`) — app-server handshake, `-c` overrides and argv, answer deltas, tool items, approval round trip (accept/decline), steering, interrupt, resumed thread, stream ending without `turn/completed`.
- `tests/dsh.rs` (`vendor` + `unix`) — launch contract (`DSH_HOME`, argv, patches), `initialize`/`session/prompt`, chunk projection, turn endings, subagent items vs. `child.session.event` provider events, capability-gated commands.
- `tests/native_v2.rs` (`native-v2`) — config/card, an end-to-end turn over `TeamAgent` with `NativeV2Driver::with_deps` fakes, steering/abort/pause while a run hangs, checkpoint deferral.
- `tests/skills.rs` — normalize modes/errors, skip/replace/append conflict policies, enabled-skill selection, traversal and non-regular-file rejection, per-vendor scan dirs, `SkillInstaller` name rules and the `dsh` flat-`*.md` normalization.
- `tests/vendor_cli.rs` / `tests/vendor_rpc.rs` (`#![cfg(unix)]`) — the real subprocess transport over `tests/support::FakeCli`: NDJSON round trip, ordered replies, EOF, stderr tail, broken pipe, malformed/over-long lines, `kill` reaping, argv/cwd/env, and the JSON-RPC request/notification/server-request/EOF paths.
- `tests/vendor_supervisor.rs` — `VendorHarness` over an in-test `FakeDriver` (`Answer`/`Fail`/`Error`/`Hang`): host validation, the single `Terminated→Idle` transition, the per-turn event sequence, serialized queued turns, steering/abort reaching the driver, capability gating on a bare card, pause/resume, checkpoint delegation, the single-consumer event lease, driver errors as `Failed`, idempotent `stop`.
- `tests/support/mod.rs` — `FakeCli::{script, replies, path, lines, env}`: a temp-directory POSIX shell script with a stdin transcript (`mod support;` per binary).
- Unit tests beside modules for pure helpers (`input`, `json`, `mcp_naming`, `decode`, `error`, factory and projection/answer unit tests), the JSON-RPC client (`src/vendor_cli/rpc/tests.rs`), the event bus, permissions, session-id derivation, driver constructors, each provider's config/card decoding and frame projection.
- Suite counts: default (`native`) lib 198 + `io_adapter` 20 + `native` 9 + `registry` 23 + `skills` 33 + `vendor_cli` 11 + `vendor_rpc` 6 + `vendor_supervisor` 14 = 314; `--features vendor` lib 274 + `claudecode` 12 + `codex` 11 + `dsh` 14 + the above 20/9/26/33/11/6/14 + `vendor` 37 = 467; `--features "vendor,native-v2"` adds `native_v2` 12 and 9 more lib tests = 488; `--no-default-features` lib 190 + `io_adapter` 20 + `native` 4 + `registry` 18 + `skills` 33 + `vendor_cli` 11 + `vendor_rpc` 6 + `vendor_supervisor` 14 = 296.
- The suites that need a child process or a Unix-only syscall are `#![cfg(unix)]`; `claudecode.rs` and `codex.rs` currently gate on `vendor` only (see Known Issues).
- Definition of done: `cargo fmt -p openjiuwen-harness-providers`, `cargo test -p openjiuwen-harness-providers`, `--features vendor`, `--features "vendor,native-v2"`, `--no-default-features`, and `cargo clippy -p openjiuwen-harness-providers --all-targets --features "vendor,native-v2" -- -D warnings` all clean; `cargo doc -p openjiuwen-harness-providers --no-deps` warning-free.

## Known Issues

- `tests/claudecode.rs` and `tests/codex.rs` spawn child processes but are gated on `feature = "vendor"` only, so a non-Unix build would compile them and fail at runtime; `tests/dsh.rs`, `tests/vendor_cli.rs` and `tests/vendor_rpc.rs` correctly use `#![cfg(unix)]`.
- `src/lib.rs` crate docs and the `providers::vendor::vendor_card` doc comment still describe the vendor providers as deferred ("creating one returns `Error::Unsupported`", "the eventual implementation"), which no longer matches the implementation.

## Notes for Agents

- The Python reference is read-only; consult it through `subagent_investigator` only.
- Do not add vendor SDK crates or vendor binaries: a vendor adapter drives an operator-installed CLI through `vendor_cli` and must return `UnsupportedCapability` for anything the pinned protocol cannot express.
- Pick the driver seam first: new vendor work goes into `src/providers/<name>/` as a `VendorDriver` (`mod.rs` config/card/provider, `harness.rs` transport, `frames.rs` projection, `wire.rs` argv/env) plus a `tests/<name>.rs` suite over `FakeCli`; reuse `vendor_cli` before writing new transport code.
- `VendorHarness` is the only publisher of lifecycle events: a driver publishes through its `TurnContext` and returns the `TurnResult`, nothing else, and it never overrides a normalized result.
- `HarnessCard` is not serde-optional: build cards with `HarnessCard::new(name, version)` (or `providers::vendor::vendor_card`) and fill `capabilities`, `optional_host_capabilities`, `fork_modes`.
- A harness created through `create_harness` cannot be handed an injected `Model`: the native provider builds its client lazily inside `start` (`openjiuwen-harness/src/agent/builder.rs:446`), so offline scripted-model tests construct `NativeHarness::new(agent, options)` over a `DeepAgentBuilder::model(Arc<dyn Model>)` and use the registry only for identity/create/descriptor assertions.
- Feature reachability: the `openjiuwen` umbrella re-exports this crate ungated as `openjiuwen::harness_providers` but forwards only `native` (and `live-tests`), so `vendor` and `native-v2` are reachable only by depending on `openjiuwen-harness-providers` directly; `openjiuwen-core` has no harness-provider symbols.
- `native_v2` lives here, not in `openjiuwen-agent-teams`: that crate supplies the runtime and `MemberHarnessFactory` and owns no provider. `openjiuwen-agent-teams/src/harness/CONTEXT.md` still claims a team-backed protocol harness is impossible, which the `native_v2` driver contradicts.
- `HarnessError` is `#[non_exhaustive]`; match with a wildcard arm and translate through `error::providers_error_to_common`.
- `openjiuwen-session::interaction::InteractiveInput` is the shared HITL answer batch — never define a second copy.
- Registry-mutating tests share a per-binary `Mutex` guard (`registry_guard()`) because the registry is process-global; without it, positional `available_providers()` assertions race with a concurrent `unregister_harness_provider`.
