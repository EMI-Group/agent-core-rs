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
- `HarnessContextOptions` + `build_harness_context(provider, options) -> Result<HarnessContext>` — manifest-free variant of Python `build_harness_context` (defaults, capability injection, native MCP refusal).

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
- `providers_error_to_common(HarnessError) -> openjiuwen_common::error::Error` (module `error`).

## Constraints

- Depends only on `openjiuwen-harness`, `openjiuwen-harness-protocol`, `openjiuwen-common`, and (for the IO adapter's shared IO types) `openjiuwen-agent` + `openjiuwen-session`; never on `openjiuwen-core`, `openjiuwen-agent-teams`, or vendor SDKs.
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
| `native_v2` | `native_v2` | — | deferred (lives in `openjiuwen-agent-teams`) | no | `openjiuwen-agent-teams` wiring |
| `claudecode` | `claude-code` | `0.1.0` | deferred stub (`vendor`) | yes (`{"model": {…}}`, `fallback_model`) | `claude` CLI + Claude Agent SDK over NDJSON stdio |
| `codex` | `codex` | `0.1.0` | deferred stub (`vendor`) | yes (`{"model": {…}}`, `fallback_model`) | `codex` app-server + `openai-codex` SDK over JSON-RPC stdio |
| `dsh` | `deepseek-harness` | `0.3.0` | deferred stub (`vendor`) | yes (flat `model`/`base_url`/`api_key`; rejects effort + fallback) | `deepseek-harness-sdk` + DSH runtime subprocess |

Capability cards: `claudecode`/`codex` declare steer, graceful abort, persistent session, checkpoint, MCP tools, model selection/discovery, endpoint models, auth fallback (codex adds native tools); `dsh` declares only MCP tools and endpoint models.

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, feature flags, flat re-exports
- `./src/provider.rs` → `ProviderSpec`, wrappers, `ProviderDescriptor`, provider-name constants
- `./src/factory.rs` → process-global registry, resolution order, `create_harness`, `provider_model_config`, `provider_descriptors`, `build_harness_context`, `HarnessContextOptions`; unit tests in `./src/factory/tests.rs`
- `./src/providers/` → built-in provider table (`mod.rs`), `native.rs`, vendor stubs (`claudecode.rs`, `codex.rs`, `dsh.rs`) behind `vendor`, shared vendor helpers in `vendor.rs`
- `./src/io_adapter/` → `HarnessIOAdapter`, `HarnessIOAdapterConfig`, `EventObserver`, `ProviderInteractionHandler` (`mod.rs`), event→chunk projection (`projection.rs`), interaction routing and the pending map (`interactions.rs`), host-answer parsing (`answers.rs`)
- `./src/skills.rs` → `SkillSource`, conflict policies, portable skill-bundle install
- `./src/decode.rs` → strict JSON mapping helpers shared by factory + vendor configs (unknown-key rejection, typed option decoding)
- `./src/error.rs` → `HarnessError` → `openjiuwen_common::error::Error` translation
- `./src/input.rs` → `harness_input_text`, `to_harness_input`, chunk/interaction kind constants
- `./src/json.rs` → `to_json_safe` / `to_json_object`
- `./src/mcp_naming.rs` → `<mcp-tools>` naming preamble
- `./tests/` → offline suites (`tests/CONTEXT.md`)

## Design Decisions

- Python's factory keyed providers by name and probed `isinstance(provider, HarnessProvider)`/`ModelConfigProvider` at runtime; Rust cannot specialize on optional trait impls, so `ProviderSpec` is the single registered interface and the two wrapper constructors pick the model-config behavior (documented deviation from Python's duck typing).
- Python's `native` provider is *not* a `ModelConfigProvider`; the Rust native harness implements `ModelConfigProvider` (effort→temperature), so the Rust `native` provider registers with the hook. Documented deviation.
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

## Deferred Integrations

- `claudecode`: Claude Agent SDK / `claude` CLI (NDJSON over stdio), `lifecycle.py` delivery receipts, `observation.py` OTLP model-request logs, `failure_classifier.py`.
- `codex`: `codex` app-server JSON-RPC, rollout-trace tailing, raw-event observation, dynamic-tool registration, MCP override rendering.
- `dsh`: `deepseek-harness-sdk` runtime, Cordis plugin overlays, notification mapping.
- `native_v2`: `openjiuwen-agent-teams::harness::protocol_adapter::NativeV2HarnessProvider`.
- `trajectory.py` (OTel span recorder) — belongs to `openjiuwen-extensions`; `telemetry/otlp_receiver.py` — external receiver, not ported.
- Manifest/agent-template loading and `HarnessContext` persona rendering — deferred with the agent-template spec.
- Installed entry-point discovery (`importlib.metadata`) has no Rust analogue; providers are registered in-process or compiled in.

## Test Strategy

- `tests/registry.rs` — name resolution for every declared name, unknown-provider message, duplicate/overwrite semantics, built-in-name rejection, unregister, `available_providers` ordering/dedup, `provider_model_config` gates and fragments, `build_harness_context` defaults and capability injection, `create_harness` returns an unstarted harness.
- `tests/native.rs` — native provider card/descriptor, create from a valid and an invalid mapping, unstarted state, start/stop through `NativeHarness` driven by `openjiuwen_test_support::ScriptedModel`, send/event cursor/`turn_events`/stop, model discovery + selection, checkpoint export, capability gating.
- `tests/io_adapter.rs` — chunk projection order and kinds, turn-lifecycle projection off by default, interactive-input round trip, unmatched input forwarding, auto-approve vs host approval, dynamic tool call routed through `ToolGateway`, delivery-mode selection, abort fallback, failed start cleanup; driven by an in-test `FakeHarness`.
- `tests/vendor.rs` (`#![cfg(feature = "vendor")]`) — vendor cards/capabilities, config `from_json` strictness and defaults, `model_config` fragments and rejections, `create` returns `Error::Unsupported` with the SDK-required message, serde round-trip snapshots.
- `tests/skills.rs` — normalize modes/errors, skip/replace/append conflict policies, `enabled_skills` selection, traversal rejection, non-regular-file rejection.
- Unit tests beside modules for pure helpers (`input`, `json`, `mcp_naming`, `decode`, `error`, plus factory and projection/answer unit tests).
- All tests are offline and deterministic: no network, no sleeps beyond bounded timeouts, temporary directories for filesystem work, Python behavior tests named after their counterpart in `tests/unit_tests/harness_providers/`.
- Registry-mutating tests share a per-binary `Mutex` guard (`registry_guard()`) because the registry is process-global; without it, positional `available_providers()` assertions race with a concurrent `unregister_harness_provider`.
- Current suites: lib 109 unit tests, `tests/io_adapter.rs` 20, `tests/native.rs` 9, `tests/registry.rs` 23 (18 without `native`), `tests/skills.rs` 26, `tests/vendor.rs` 36.
- Definition of done: `cargo fmt -p openjiuwen-harness-providers`, `cargo test -p openjiuwen-harness-providers`, `cargo test -p openjiuwen-harness-providers --features vendor`, `cargo test -p openjiuwen-harness-providers --no-default-features`, `cargo clippy -p openjiuwen-harness-providers --all-targets --features vendor -- -D warnings` all clean.

## Notes for Agents

- The Python reference is read-only; consult it through `subagent_investigator` only.
- Do not add vendor SDK crates or spawn external CLIs: the vendor surface is data + explicit `Unsupported` errors.
- `HarnessCard` is not serde-optional: build cards with `HarnessCard::new(name, version)` and fill `capabilities`, `optional_host_capabilities`, `fork_modes`.
- `HarnessError` is `#[non_exhaustive]`; match with a wildcard arm and translate through `error::providers_error_to_common`.
- `openjiuwen-session::interaction::InteractiveInput` is the shared HITL answer batch — never define a second copy.
