# openjiuwen-harness-providers

## Intent

Provider registry and adapters for the provider-neutral harness SPI (`openjiuwen-harness-protocol`).
Ports the Python package `openjiuwen.harness_providers` (v0.1.20, ~13.6k lines): the process-global provider registry/factory, the `native` provider (in-process DeepAgent harness), the `HarnessIOAdapter` that projects a harness onto the kernel IO surface, provider-specific config/spec data types, the portable skill-bundle installer, and the deferred vendor providers (`claudecode`, `codex`, `dsh`).
The crate owns *provider selection and adaptation*; it never reimplements a harness — the native harness lives in `openjiuwen-harness` and the vendor harnesses stay deferred.

## API Surface

Registry and factory (module `factory`, flat re-exports):
- `HARNESS_PROVIDER_ENTRY_POINT_GROUP` (`"openjiuwen.harness_providers"`), `PROVIDER_NAMES` (`["native", "native_v2", "claudecode", "codex", "dsh"]`), `provider_names::{NATIVE, NATIVE_V2, CLAUDE_CODE, CODEX, DSH}`.
- `ProviderSpec` — one resolvable provider: `card()`, `create(config)`, `configures_models()`, `model_config(choice, fallback)`; wrappers `plain_provider(p)` and `model_configuring_provider(p)` adapt any `HarnessProvider` (+ optional `ModelConfigProvider`).
- `ProviderFactory` (`Arc<dyn Fn() -> Result<Arc<dyn ProviderSpec>> + Send + Sync>`).
- `register_harness_provider(name, factory) -> Result<()>`, `unregister_harness_provider(name)`, `available_providers() -> Vec<String>`, `resolve_provider(name) -> Result<Arc<dyn ProviderSpec>>`.
- `create_harness(provider, config) -> Result<Box<dyn HarnessProtocol>>` (never starts the harness).
- `provider_model_config(provider, choice, fallback) -> Result<JsonObject>` with the Python capability gates and message text.
- `ProviderDescriptor { name, card_name, implementation_version, builtin, implemented, configures_models, requires }`, `provider_descriptors()`, `describe_provider(name)`.
- `HarnessContextOptions` (`new(host_session_id, system_prompt)` + public fields, `src/factory.rs:377`, `:414`) + `build_harness_context(provider, options) -> Result<HarnessContext>` — manifest-free variant of Python `build_harness_context` (defaults, capability injection, native MCP refusal).

Providers (module `providers`):
- `providers::native` — `provider() -> NativeHarnessProvider`, `card()`, `PROVIDER_NAME` (`"native"`); re-exports `NativeHarness`, `NativeHarnessProvider`, `NativeHarnessOptions` from `openjiuwen_harness`.
- `providers::claudecode` (`vendor`) — `ClaudeCodeHarnessConfig`, `ClaudeModelConfig`, `ClaudePermissionMode`, `ClaudeSystemPromptMode`, `DEFAULT_CLAUDE_MAX_BUFFER_SIZE`, `PROVIDER_NAME`, `CARD_NAME` (`"claude-code"`), `card()`, `provider()`, `model_config`.
- `providers::codex` (`vendor`) — `CodexHarnessConfig`, `CodexModelConfig`, `CodexApprovalMode`, `PROVIDER_NAME`, `CARD_NAME` (`"codex"`), `card()`, `provider()`, `model_config`.
- `providers::dsh` (`vendor`) — `DshHarnessConfig`, `PROVIDER_NAME`, `CARD_NAME` (`"deepseek-harness"`), `card()`, `provider()`, `model_config`.

IO adapter (module `io_adapter`):
- `HarnessIOAdapter`, `HarnessIOAdapterConfig`, `EventObserver`, `ProviderInteractionHandler`.
- `HarnessIOAdapter::{new, harness, config, state, session_id, pending_interrupt_ids, has_pending_interrupt, is_pending_interrupt_resume_valid, prepare_context, start, stop, outputs, delivery_mode, send, abort, pause, resume}` and `impl HarnessInteractionHandler` (`handle`, `cancel`); `start`/`outputs`/`prepare_context` take `self: &Arc<Self>` because the adapter installs itself as the context's interaction handler and owns its pump task.
- Chunk kinds: `llm_output`, `llm_reasoning`, `tool_call`, `tool_result`, `__interaction__`, `turn_lifecycle`, `task_failed` — projected as `openjiuwen_agent::AgentChunk`.

Helpers (flat re-exports):
- `harness_input_text(&HarnessInput) -> String`, `to_harness_input(JsonValue) -> HarnessInput`, `INTERACTIVE_INPUT_KIND` (`"interactive_input"`), `INTERACTION_KIND` (`"__interaction__"`), `TURN_LIFECYCLE_KIND` (`"turn_lifecycle"`), `TASK_FAILED_KIND` (`"task_failed"`).
- `to_json_object(&JsonValue) -> JsonObject`, `to_json_safe(&JsonValue) -> JsonValue`.
- `mcp_tool_naming_preamble(&BTreeMap<String, String>) -> String`, `TOOL_PLACEHOLDER` (`"{tool}"`).
- Skills: `SkillSource`, `SkillMode` (`all`/`auto_list`), `SkillConflict` (`skip`/`replace`/`append`), `normalize_skills`, `install_skills`, `skill_scan_dir(provider)`.
- `SkillInstaller` / `SkillLayout` / `install_for_provider` — the per-vendor policy over `install_skills`: `for_provider("claudecode"|"codex"|"dsh", conflict)`, `provider()`, `scan_dir()`, `layout()`, `install(sources, cwd)`, `validate_name(name)`; `SkillLayout::{Directory, DirectoryOrFlat}` (`.dsh` accepts a flat `*.md` source root and normalizes it into `<scan>/<name>/SKILL.md`), and the name rule is the shared portable rule for Claude/Codex but kebab-case for `dsh`.
- `providers_error_to_common(HarnessError) -> openjiuwen_common::error::Error` (module `error`).

Shared vendor CLI transport (module `vendor_cli`, foundation for the three CLI adapters):
- `binary` — `BinarySpec`, `resolve_binary`, `is_available` (explicit path validated, `PATH` fallback, actionable `Protocol` errors).
- `jsonl` — `parse_line`, `parse_line_labelled`, `encode_line`, `get_str`, `get_object`, `get_array`, `get_bool`, `get_u64`, `get_f64`, `string_list`, `snippet`.
- `child` — `ChildProcessConfig`, `ChildProcess` (`spawn`, `write_json`, `write_line`, `read_json`, `stderr_tail`, `close_stdin`, `kill`, `wait`, `pid`), `DEFAULT_MAX_LINE_BYTES`, `STDERR_TAIL_LINES`, `STDERR_TAIL_CHARS`, `EXIT_TIMEOUT`.
- `rpc` — JSON-RPC 2.0 over that transport: `RpcTransport` (`write_line`, `read_json`, `stderr_tail`; implemented for `ChildProcess`), `JsonRpcClient` (`spawn`, `request`, `request_with_timeout`, `notify`, `next_incoming`, `respond`, `respond_error`, `close`, `is_closed`, `label`, `stderr_tail`), `request_id_value`.
- `events` — `EventBus` (`new`, `emit`, `events`, `turn_events`, `close`, `is_closed`, `len`, `is_empty`), `EnvelopeBuilder` (`new`, `set_provider_session_id`, `next`, `next_for_turn`, `next_for_item`).
- `permissions` — `PermissionPolicy` (`from_context`, `interactive`), `ApprovalDecision`, `ask_tool_approval`, `ask_user_input`.
- `session` — `derive_session_id`, `resolve_resume_id`, `checkpoint_str`.
- `driver` — the adapter seam `#[async_trait] VendorDriver { label, open, close, run_turn, steer, interrupt, pause, resume, export_checkpoint }` (unsupported commands have default bodies returning `HarnessError::UnsupportedCapability`), `TurnRequest`, `TurnContext` (`output`, `tool_started`, `tool_completed`, `usage`, `provider_event`, `diagnostic`, `emit`), the result constructors `completed_turn`/`interrupted_turn`/`failed_turn`/`turn_error`, `TOOL_CALL_ITEM_TYPE`, `PROVIDER_SCHEMA_VERSION`.
- `harness` — `VendorHarness::new(HarnessCard, EventBufferConfig, Arc<dyn VendorDriver>)`, the complete `HarnessProtocol` shell: lifecycle, turn queue + single supervisor task, per-turn event sequence, capability gating, abort/pause/resume, bounded teardown.

## Constraints

- Depends only on `openjiuwen-harness`, `openjiuwen-harness-protocol`, `openjiuwen-common`, and (for the IO adapter's shared IO types) `openjiuwen-agent` + `openjiuwen-session`; never on `openjiuwen-core`, `openjiuwen-agent-teams`, or vendor SDKs. `which` is used only by `vendor_cli::binary`.
- `openjiuwen-harness` owns every harness behavior; this crate wires, selects, validates and adapts — no harness logic is duplicated here.
- Registry functions are synchronous and return `openjiuwen_common::error::Result`; provider SPI calls return `HarnessResult`; `error::providers_error_to_common` is the single translation point (UnsupportedCapability → `Error::Unsupported`, Validation → `Error::Validation`, State/Protocol → `Error::Config`, CheckpointConflict → `Error::Session`, wildcard → `Error::External`).
- Global registry state is process-wide and documented (mirrors Python's module dict): `std::sync::RwLock` + `indexmap` for insertion order, never held across `.await`.
- Errors keep Python message text and Python `repr` quoting (`unknown harness provider 'x'; expected one of …`, `unknown Claude Code configuration fields: …`).
- Vendor providers are registered but not implemented: `create` returns `HarnessError::UnsupportedCapability` with the Python "SDK required" text, mapped to `Error::Unsupported` at the factory; cards stay behind the same `vendor` feature.
- Features: `native` (default) enables the native provider; `vendor` enables the three vendor modules and their registration; `live-tests` is opt-in and never enabled by default; the crate builds with `--no-default-features` and with `--features vendor`.
- All three features are empty toggles (`default = ["native"]`, `native = []`, `vendor = []`, `live-tests = []`) and every dependency is non-optional — `openjiuwen-harness` is compiled even with `--no-default-features`, where the feature only compiles out the `providers::native` module.
- No source file references `feature = "live-tests"`; it is a reserved manifest flag with no gated code (the delivered end-to-end tests live in `tests/vendor.rs` under `vendor`).
- Async-first: `HarnessIOAdapter` spawns exactly one event-pump task per `start`, owns it through the adapter lifetime, and stops it in `stop`; no detached task outlives the adapter.
- Serde: config types are `snake_case`, tolerate unknown keys when deserializing from JSON, and reject unknown keys in `from_json` mapping (Python `from_mapping`); `api_key` is `SecretString` and redacted in `Debug`.
- No `unsafe`, no stdout logging (`tracing` only), files ≤1000 lines, all public items documented.

## Provider Matrix

| Registry name | Card name | Impl version | Status | Model config hook | External requirement |
|---|---|---|---|---|---|
| `native` | `openjiuwen-native` (Python: `deepagent`) | `openjiuwen-harness` crate version | implemented | yes (`model`, effort→`temperature`) | none |
| `native_v2` | `native_v2` | `0.2.0` | implemented behind `native-v2` (hosts an `openjiuwen-agent-teams` team through `VendorHarness`) | no | none with the feature; `openjiuwen-agent-teams` when compiled out |
| `claudecode` | `claude-code` | `0.1.0` | deferred stub (`vendor`) | yes (`{"model": {…}}`, `fallback_model`) | `claude` CLI + Claude Agent SDK over NDJSON stdio |
| `codex` | `codex` | `0.1.0` | deferred stub (`vendor`) | yes (`{"model": {…}}`, `fallback_model`) | `codex` app-server + `openai-codex` SDK over JSON-RPC stdio |
| `dsh` | `deepseek-harness` | `0.3.0` | deferred stub (`vendor`) | yes (flat `model`/`base_url`/`api_key`; rejects effort + fallback) | `deepseek-harness-sdk` + DSH runtime subprocess |

Capability cards: `claudecode`/`codex` declare steer, graceful abort, persistent session, checkpoint, MCP tools, model selection/discovery, endpoint models, auth fallback (codex adds native tools); `dsh` declares only MCP tools and endpoint models.

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, feature flags, flat re-exports
- `./src/provider.rs` → `ProviderSpec`, wrappers, `ProviderDescriptor`, provider-name constants
- `./src/factory.rs` → process-global registry, resolution order, `create_harness`, `provider_model_config`, `provider_descriptors`, `build_harness_context`, `HarnessContextOptions`; unit tests in `./src/factory/tests.rs`
- `./src/providers/` → built-in provider table (`mod.rs`), `native.rs`, team-hosted `native_v2/` (`mod.rs` provider+config+card, `driver.rs` `VendorDriver` over `TeamAgent`), vendor stubs (`claudecode.rs`, `codex.rs`, `dsh.rs`) behind `vendor`, shared vendor helpers in `vendor.rs`
- `./src/io_adapter/` → `HarnessIOAdapter`, `HarnessIOAdapterConfig`, `EventObserver`, `ProviderInteractionHandler` (`mod.rs`), event→chunk projection (`projection.rs`), interaction routing and the pending map (`interactions.rs`), host-answer parsing (`answers.rs`)
- `./src/vendor_cli/` → shared vendor CLI transport and harness shell: binary resolution, NDJSON helpers, the child process, the JSON-RPC client, the observation bus, host permission translation, session-id derivation, the `VendorDriver` seam and `VendorHarness`
- `./src/skills/` → `mod.rs` (bundle discovery, conflict policies, portable install) + `installer.rs` (`SkillInstaller`, `SkillLayout`, per-vendor name rules, flat-`*.md` normalization)
- `./src/decode.rs` → strict JSON mapping helpers shared by factory + vendor configs (unknown-key rejection, typed option decoding)
- `./src/error.rs` → `HarnessError` → `openjiuwen_common::error::Error` translation
- `./src/input.rs` → `harness_input_text`, `to_harness_input`, chunk/interaction kind constants
- `./src/json.rs` → `to_json_safe` / `to_json_object`
- `./src/mcp_naming.rs` → `<mcp-tools>` naming preamble
- `./tests/` → offline suites (`tests/CONTEXT.md`)

## Design Decisions

- Python's factory keyed providers by name and probed `isinstance(provider, HarnessProvider)`/`ModelConfigProvider` at runtime; Rust cannot specialize on optional trait impls, so `ProviderSpec` is the single registered interface and the two wrapper constructors pick the model-config behavior (documented deviation from Python's duck typing).
- Python's `native` provider is *not* a `ModelConfigProvider`; the Rust native harness implements `ModelConfigProvider` (effort→temperature), so the Rust `native` provider registers with the hook. Documented deviation.
- `create_harness("native", …)` yields a `DeepAgent` with **no injected model**: the harness stays `Terminated` and `start` fails with `Protocol("DeepAgent requires a model: …")` unless the config supplies `model` (plus `provider`/`api_base`/`api_key` for lazy `init_model` resolution). Offline turn-driving tests therefore bypass the factory (`NativeHarness::new(DeepAgentBuilder::model(...))`) or point `api_base` at a local mock server.
- `native` config keys are the Rust `NativeHarnessProvider` key set (`model`, `provider`, `api_key`, `api_base`, `temperature`, `workspace`, `enable_task_loop`, `max_iterations`, `system_prompt`, `permissions`, `language`); Python's `deep_agent`/`agent_template`/`session_id`/`event_buffer_capacity` keys are not accepted because the Rust DeepAgent takes live handles through `DeepAgentBuilder` instead of a spec tree.
- Manifest helpers (`load_manifest`, `manifest_provider_config`, `manifest_mcp_servers`) are not ported: the agent-template spec and manifest loader do not exist in `openjiuwen-harness`; `create_harness` therefore takes `(provider, config)` and `build_harness_context` takes explicit options instead of a manifest.
- `available_providers` keeps built-in order first, then registration order, de-duplicated; re-registering a non-built-in name overwrites in place (Python dict semantics), registering an empty or built-in name is `Error::Validation`.
- Unknown provider errors use `Error::NotFound`; deferred-but-known providers and missing features use `Error::Unsupported` (distinct from Python's single `ValueError`, documented).
- The registry is global (like Python's module dict) to keep CLI/agent-teams call sites identical; tests use unique provider names instead of a reset hook.
- `HarnessIOAdapter::outputs` is single-consumer and returns `HarnessError::State` on a second call — the Rust equivalent of Python's one async generator fed by the pump task.
- The event pump is exactly one `tokio::spawn` task holding a `Weak<HarnessIOAdapter>` (no reference cycle with the stored join handle); it ends on `None`, on a cursor error (logged with `tracing::warn!`), or when `stop` aborts it, and always closes the cursor. Chunk indexes are allocated lazily so events that project to nothing consume no index.
- `start` applies `prepare_context` itself (Python parity) and installs the adapter as `context.interactions` only when the host supplied none; `stop` aborts the pump, cancels every pending interaction, then stops the harness and is terminal for that adapter instance.
- `send` resolves pending interactions from `interrupt_answers` first and then `user_inputs`, keyed by request id; an interactive-shaped payload matching no pending id is forwarded with `metadata["kind"] = "interactive_input"`. Python's extra "sole pending request answered from `raw_inputs`" rule is not ported: `raw_inputs` only marks the payload as interactive-shaped.
- Approval answers are ported as data: `"yes"`/`"always"`/`"no"` words, `{"action": …}`, `{"approved": …, "feedback": …}`, `{"updated_arguments": …}` and `{"remember": true}`; every unrecognized shape becomes `Deny`.
- `build_harness_context` **drops** `options.mcp_servers` for the `native` provider and never injects `HostCapability::McpServers` there (the native harness owns its tools, Python raises); every other provider gets the servers copied plus the capability when the list is non-empty, and a checkpoint sink always injects `HostCapability::CheckpointSink`.
- Declared provider descriptors are static metadata: vendor names stay `implemented: false` even with the `vendor` feature, and feature flags never rewrite an entry (`resolve_provider` reports the missing feature instead).
- Registry names are unique per test because the registry is process-global; `src/factory/tests.rs` uses `uuid`-derived names and never resets the map.
- `to_json_safe` is an identity in Rust (values are already JSON-safe); `to_json_object` wraps non-objects as `{"value": …}`. Documented deviation.
- `install_skills` keeps Python's safety rules (reject escaping paths, reject non-regular files, atomic stage-then-rename) but materializes internal symlinks by copying through them instead of recreating them.
- `install_skills` discovers bundles recursively like Python `_discover` (a source root that is itself a bundle stops the walk; symlinked directories are not traversed) and rewrites the appended copy's front-matter `name:` line in place instead of round-tripping YAML.
- Portable skill names accept `[A-Za-z0-9._-]+`, reject empty and `..` (Python's regex is stricter: it requires a leading alphanumeric and at most 128 bytes).
- `mcp_tool_naming_preamble` emits only the `<mcp-tools>` block; Python's trailing explanatory sentence is not appended (documented deviation from `mcp_naming.py`).
- `harness_input_text` joins an array when every element carries a string `text` field; Python additionally requires each block's `type` to equal `"text"`.
- Vendor `from_json` type errors use the shared `decode` wording (`{key} must be …`); Python's `TypeError` text is kept for the cross-field rules and every unknown-key message keeps the Python prefix.
- `VendorHarness` collapses `DeliveryMode::Auto` to `FollowUp` (the supervisor serializes turns, so an accepted input never starts synchronously) and reports an abort with no active turn by draining the queue as `Started`+`Aborted(HarnessStop)` pairs, ignoring the requested mode because no turn is left to interrupt.
- `VendorHarness::resume` settles in `Running` while a turn is still active and in `Idle` otherwise; `Pausing`/`Resuming` are transient states only. The provider session id is sticky: a later cycle reporting a different id is `HarnessError::State` instead of a silent identity switch.
- A `VendorDriver` error from `run_turn` becomes a `Failed` terminal event with category `"protocol"`; a driver-*returned* failure keeps its own `TurnError` category. `VendorHarness` never overrides a normalized result.
- `VendorHarness::stop` returns the `driver.close` error but still completes teardown (drain, `Terminated`, bus close, supervisor join); a close timeout is reported as `State("the {label} harness did not stop cleanly")` after the same full teardown.

## Deferred Integrations

- `claudecode`: Claude Agent SDK / `claude` CLI (NDJSON over stdio), `lifecycle.py` delivery receipts, `observation.py` OTLP model-request logs, `failure_classifier.py`.
- `codex`: `codex` app-server JSON-RPC, rollout-trace tailing, raw-event observation, dynamic-tool registration, MCP override rendering.
- `dsh`: `deepseek-harness-sdk` runtime, Cordis plugin overlays, notification mapping.
- `native_v2` checkpoint export: the card declares no `Checkpoint` capability, so `VendorHarness` rejects `export_checkpoint`; the team runtime exposes no portable snapshot.
- `native_v2` passthrough of team lifecycle events beyond `llm_output`/`llm_reasoning`/`tool_call`/`tool_result` (projected as `ProviderEvent`).
- `trajectory.py` (OTel span recorder) — belongs to `openjiuwen-extensions`; `telemetry/otlp_receiver.py` — external receiver, not ported.
- Manifest/agent-template loading and `HarnessContext` persona rendering — deferred with the agent-template spec.
- Installed entry-point discovery (`importlib.metadata`) has no Rust analogue; providers are registered in-process or compiled in.

## Test Strategy

- `tests/registry.rs` — name resolution for every declared name, unknown-provider message, duplicate/overwrite semantics, built-in-name rejection, unregister, `available_providers` ordering/dedup, `provider_model_config` gates and fragments, `build_harness_context` defaults and capability injection, `create_harness` returns an unstarted harness.
- `tests/native.rs` — native provider card/descriptor, create from a valid and an invalid mapping, unstarted state, start/stop through `NativeHarness` driven by `openjiuwen_test_support::ScriptedModel`, send/event cursor/`turn_events`/stop, model discovery + selection, checkpoint export, capability gating.
- `tests/io_adapter.rs` — chunk projection order and kinds, turn-lifecycle projection off by default, interactive-input round trip, unmatched input forwarding, auto-approve vs host approval, dynamic tool call routed through `ToolGateway`, delivery-mode selection, abort fallback, failed start cleanup; driven by an in-test `FakeHarness`.
- `tests/vendor.rs` (`#![cfg(feature = "vendor")]`) — vendor cards/capabilities, config `from_json` strictness and defaults, `model_config` fragments and rejections, `create` returns `Error::Unsupported` with the SDK-required message, serde round-trip snapshots.
- `tests/skills.rs` — normalize modes/errors, skip/replace/append conflict policies, `enabled_skills` selection, traversal rejection, non-regular-file rejection.
- `tests/vendor_cli.rs` (`#![cfg(unix)]`) — the real subprocess path over `tests/support::FakeCli`: NDJSON round trip and transcript, ordered replies, EOF as `Ok(None)`, stderr tail, broken-pipe message, malformed and over-long output lines, `kill` reaping, arguments/cwd, cleared environment.
- `tests/vendor_rpc.rs` (`#![cfg(unix)]`) — the real subprocess JSON-RPC path: request round trips and transcript, an unsolicited notification through `next_incoming`, a peer error with its code, notifications written without an id, EOF failing the pending request and closing the client, an unreadable line skipped.
- `tests/support/mod.rs` — `FakeCli::{script, replies, path, lines, env}`, a temp-directory POSIX shell script with a stdin transcript (included per test binary with `mod support;`); the script is staged as `fake-cli.staged`, chmodded, renamed onto `fake-cli`, and its bytes are written through a helper `cat`, so no write descriptor for an executable ever exists in the test process (an inherited one makes `exec` fail with `ETXTBSY`).
- `tests/vendor_supervisor.rs` — `VendorHarness` over an in-test `FakeDriver` (`Answer`/`Fail`/`Error`/`Hang` modes): host validation, the single `Terminated→Idle` transition, the per-turn event sequence, serialized queued turns, steering and abort reaching the driver, capability gating on a bare card, pause/resume, checkpoint delegation, the single-consumer event lease, driver errors as `Failed`, and idempotent `stop` draining the queue.
- Unit tests beside modules for pure helpers (`input`, `json`, `mcp_naming`, `decode`, `error`, plus factory and projection/answer unit tests) and for the JSON-RPC client (`src/vendor_cli/rpc/tests.rs`).
- All tests are offline and deterministic: no network, no sleeps beyond bounded timeouts, temporary directories for filesystem work, Python behavior tests named after their counterpart in `tests/unit_tests/harness_providers/`.
- Registry-mutating tests share a per-binary `Mutex` guard (`registry_guard()`) because the registry is process-global; without it, positional `available_providers()` assertions race with a concurrent `unregister_harness_provider`.
- Current suites: lib 188 unit tests, `tests/io_adapter.rs` 20, `tests/native.rs` 9, `tests/registry.rs` 23 (18 without `native`), `tests/skills.rs` 26, `tests/vendor.rs` 36, `tests/vendor_cli.rs` 11, `tests/vendor_rpc.rs` 6, `tests/vendor_supervisor.rs` 14.
- Definition of done: `cargo fmt -p openjiuwen-harness-providers`, `cargo test -p openjiuwen-harness-providers`, `cargo test -p openjiuwen-harness-providers --features vendor`, `cargo test -p openjiuwen-harness-providers --no-default-features`, `cargo clippy -p openjiuwen-harness-providers --all-targets --features vendor -- -D warnings` all clean.

## Notes for Agents

- The Python reference is read-only; consult it through `subagent_investigator` only.
- Do not add vendor SDK crates or spawn external CLIs: the vendor surface is data + explicit `Unsupported` errors. A vendor adapter implements `VendorDriver` and is wrapped by `VendorHarness`; it must never implement `HarnessProtocol` itself.
- `VendorHarness` is the only place that publishes lifecycle events: an adapter publishes observations through its `TurnContext` and returns the `TurnResult`, nothing else.
- `HarnessCard` is not serde-optional: build cards with `HarnessCard::new(name, version)` and fill `capabilities`, `optional_host_capabilities`, `fork_modes`.
- A harness created through `create_harness` cannot be handed an injected `Model`: the native provider builds its client lazily inside `start` (`openjiuwen-harness/src/agent/builder.rs:446`), so offline scripted-model tests construct `NativeHarness::new(agent, options)` over a `DeepAgentBuilder::model(Arc<dyn Model>)` and use the registry only for identity/create/descriptor assertions.
- Registry reachability: the umbrella exposes it ungated as `openjiuwen::harness_providers` (`crates/openjiuwen/src/lib.rs:67`); `openjiuwen-core` has no harness-provider symbols.
- `native` is default-on in this crate (`Cargo.toml:32`) and the umbrella's non-optional dependency inherits those default features, so `--no-default-features` on `openjiuwen` still builds the native provider; no workspace crate enables `vendor`.
- `HarnessError` is `#[non_exhaustive]`; match with a wildcard arm and translate through `error::providers_error_to_common`.
- `openjiuwen-session::interaction::InteractiveInput` is the shared HITL answer batch — never define a second copy.
