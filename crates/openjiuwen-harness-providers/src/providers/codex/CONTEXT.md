# Codex provider (app-server subprocess adapter)

## Intent

The `codex` registry entry: configuration data, capability card, model mapping, and a real subprocess adapter for the operator-installed Codex CLI's JSON-RPC `app-server`.
Ports Python `openjiuwen.harness_providers.codex` (v0.1.20): `CodexHarnessConfig`/`CodexModelConfig` from `config.py`, the provider's `model_config` classmethod, and the `codex` adapter itself.
`create` decodes and validates the mapping, resolves the executable and returns an **unstarted** `vendor_cli::VendorHarness` wrapping a `CodexDriver`; nothing spawns until `start`.
The driver speaks the vendor protocol only — lifecycle, turn queue, capability gating and event publication belong to the shared `VendorHarness` shell.

## API Surface

Constants: `PROVIDER_NAME` (`"codex"`), `CARD_NAME` (`"codex"`), `IMPLEMENTATION_VERSION` (`"0.1.0"`).
`provider() -> CodexHarnessProvider` (also `Default`), `card() -> HarnessCard`, `model_config(choice, fallback) -> HarnessResult<JsonObject>`.
`CodexHarnessProvider` implements `HarnessProvider` (`card`, `create`) and `ModelConfigProvider` (`model_config`).
`CodexHarnessConfig` — 24 decoded fields: `system_prompt_mode`, `skills`, `skill_conflict`, `cwd`, `env`, `inherit_process_env`, `codex_bin`, `model`, `fallback_model`, `config_overrides`, `thread_config`, `bypass_approvals_and_sandbox`, `turn_idle_timeout_s`, `turn_idle_retries`, `max_will_retry_count`, `mcp_env_passthrough`, `mcp_startup_timeout_s`, `mcp_required`, `mcp_default_tools_approval_mode`, `client_name`, `client_title`, `experimental_raw_events`, `event_buffer_capacity`, `request_observation_wait_s`; `from_json(&JsonObject)` is the strict SPI decoder, plus `Default` and a manual redacting `Debug`.
`CodexModelConfig { model, provider, api_base, api_key: Option<SecretString>, effort }` — `from_json` (unknown-key and empty-string rejection), `is_external()` (provider or `api_base` set).
`CodexApprovalMode` (`auto` default / `prompt` / `writes` / `approve`) — only used by `mcp_default_tools_approval_mode`.
`CodexSystemPromptMode` (`append` default / `replace`).
`CodexDriver::new(CodexHarnessConfig, PathBuf)` with `PROVIDER_NAME`-independent `label()` = `"Codex"`; re-exported at `providers::codex::CodexDriver`.

## Constraints

- `create` order: decode config (`HarnessError::Validation` with Python message text) → `resolve_binary` → `EventBufferConfig { capacity: event_buffer_capacity, overflow: CoalesceOrBlock }` → `VendorHarness::new(card(), buffer, driver)`; `create` never spawns and the harness reports `Terminated` until `start`.
- Executable: `codex_bin` when set, else `codex` on `PATH`, resolved with `BinarySpec { name: "codex", label: "Codex", install_hint: "install the Codex CLI (npm install -g @openai/codex)" }`; a missing binary is `HarnessError::Protocol` naming `codex`, never `UnsupportedCapability`.
- Exactly one child process per start/stop cycle: `open` spawns (`ChildProcessConfig { max_line_bytes: 0, inherit_env: inherit_process_env, label: "Codex" }`), `close` closes the `JsonRpcClient`, which aborts the reader and drops the child (`kill_on_drop`).
- argv is `app-server --listen stdio://` followed by ordered `-c key=value` pairs: the external endpoint's `model_provider` / `model_providers.<key>.{name,base_url,env_key}` group, `features.default_mode_request_user_input=true` when the host declares `HostCapability::UserInput`, one `mcp_servers.<key>.{command,args,url}` group per host MCP server, then `config_overrides` verbatim; values are JSON-encoded and `-` decays to `_` in dotted keys.
- Only the **primary** `model` renders endpoint overrides and the endpoint API key; `fallback_model` is decoded and can be produced through `model_config`, but it is not rendered into argv or the thread.
- Environment layering: inherited process env (when `inherit_process_env`), then `config.env`, then `context.env`, then `OPENJIUWEN_CODEX_API_KEY` from the endpoint `api_key` (the generated `env_key` override reads exactly that name); working directory is `config.cwd`, else `context.cwd`.
- Skills are installed before the spawn, inside `spawn_blocking`, through `install_for_provider(sources, "codex", Some(cwd or "."), skill_conflict)`; an install failure is `HarnessError::Protocol`.
- Stdio MCP servers without a `command`, HTTP servers without a `url`, and all in-process servers are skipped with a `tracing::warn!` instead of failing the cycle.
- Strict decoding: `from_json` rejects unknown keys (`unknown Codex configuration fields: …`, sorted), non-object `thread_config`, unknown enum values and empty model-config strings; cross-field validation requires `turn_idle_timeout_s > 0` and positive `mcp_startup_timeout_s`/`event_buffer_capacity`.
- Plain serde deserialization tolerates unknown keys (crate-wide rule) with defaults `thread_config = {"model_reasoning_summary": "detailed"}`, `turn_idle_timeout_s = 180`, `turn_idle_retries = 1`, `max_will_retry_count = 5`, `mcp_startup_timeout_s = 120`, `mcp_required = true`, `client_name = "openjiuwen_harness"`, `client_title = "OpenJiuwen Harness"`, `experimental_raw_events = true`, `event_buffer_capacity = 1024`, `request_observation_wait_s = 5`.
- `api_key` is `SecretString` and the manual `Debug` prints `env` as `<redacted>`; no `unsafe`, no stdout logging, files stay under 1000 lines.
- Decoded fields with no transport effect: `system_prompt_mode` (the host prompt is always sent as `developerInstructions`), `turn_idle_timeout_s`/`turn_idle_retries`/`max_will_retry_count` (no retry loop), `mcp_env_passthrough`/`mcp_startup_timeout_s`/`mcp_required`/`mcp_default_tools_approval_mode`, `client_name` (the handshake uses the fixed `"openjiuwen-harness"`), `experimental_raw_events` (raw notifications are dropped), `request_observation_wait_s` (no observation wait).

## Wire Protocol

Handshake: `initialize` with `clientInfo { name: "openjiuwen-harness", title: config.client_title, version: "1" }`, then the `initialized` notification, then `thread/start` — or `thread/resume` when a checkpoint supplies a thread — and the thread id is read from `result.thread.id` (`Protocol` if absent, `Protocol("Codex resumed unexpected thread '<id>'")` if a resumed echo differs).
Thread options: `threadId` only on resume, `ephemeral: false`, `cwd`, `model`, `modelProvider` (when the endpoint names a provider), `developerInstructions` = non-empty host system prompt, `approvalPolicy`/`sandbox` = `never`/`danger-full-access` when approvals are bypassed and `on-request`/`workspace-write` otherwise, and `config` = `thread_config` plus `model_reasoning_effort` from `model.effort`.
Approvals are bypassed when `bypass_approvals_and_sandbox` is set or the model has an endpoint (provider, base URL or API key).
Turns: one `turn/start { threadId, input: [{type: "text", text}] }`, where the text is `harness_input_text` of the turn content; the native turn id from `result.turn.id` is held in driver state for steer/interrupt and cleared when the turn ends.
Notifications: `item/agentMessage/delta` → answer delta, `item/reasoning/summaryTextDelta`/`textDelta` → reasoning delta, `item/started` → `tool_started` for `commandExecution`/`dynamicToolCall`/`fileChange`/`mcpToolCall`, `item/completed` → `tool_completed` plus a tool-result message for those item types, and for `agentMessage`/`reasoning` a text/reasoning message plus a `Final` output (`last_text` is the turn's `final_output`); another completed item type or any other method becomes a `codex` provider event, except `rawResponseItem/completed` and `rawResponse/completed`, which are dropped.
Tool names and arguments: `commandExecution` → `shell` (`{command, cwd}`, result `{output, exit_code}`), `fileChange` → `apply_patch` (`{changes}`), `mcpToolCall` → `<server>.<tool>` (`arguments`, result `result`/`output`), `dynamicToolCall` → the item's `tool` field.
Usage: `thread/tokenUsage/updated` publishes a `Delta` `UsageUpdatedEvent` from `tokenUsage.last` (else `tokenUsage`) and accumulates the turn totals; camelCase and snake_case field spellings are both accepted.
Errors: the `error` notification records the most specific `TurnError` (a generic `sdk_error` never replaces a specific one) and publishes a `Warning` diagnostic when `willRetry` is set, else an `Error` diagnostic; categories are `auth_required`, `quota_exceeded`, `server_unavailable`, `request_rejected`, `rate_limited`, `network_timeout`, `sdk_error`, with an HTTP status (400/401/403/429/500/529) overriding the `codexErrorInfo` variant and `retryable` true only for `rate_limited`/`server_unavailable`/`network_timeout`.
Terminal `turn/completed`: status `completed`/`complete` → completed result (`last_text`, messages, merged usage), `interrupted` → provider interruption, `failed`/`failure` → the pending error or `CODEX_TURN_FAILED`, anything else → `CODEX_MISSING_TURN_COMPLETED` merged with the pending error; a host abort wins over the reported status (`UserAbort`, "Codex interrupted the turn").
A stream that ends before `turn/completed` returns a `HarnessStop` interruption while closing, a `UserAbort` interruption when the host aborted, and otherwise a failed turn with code `CODEX_MISSING_TURN_COMPLETED` and the client's stderr tail appended.
Any incoming message with a non-null `id` is a server-to-client request and is answered: `item/commandExecution/requestApproval` and `item/fileChange/requestApproval` through the host tool-approval policy (tool `shell` / `apply_patch`, request id `codex-approval:<itemId>`, params minus `threadId`/`turnId`) as `{"decision": "accept"|"decline"}`; `item/tool/requestUserInput` through the host user-input policy (request id `codex-ask:<itemId>`, rendered questions, first question's labels as choices) as `{"answers": {<id>: {"answers": [...]}}}`; `item/tool/call` with JSON-RPC `-32601 "dynamic tools are not available"`; every unknown method is reported as a `server_request` provider event and answered `-32601`.
Steering requires an active native turn id (otherwise `State("there is no active Codex turn")`) and sends `turn/steer { threadId, turnId, input }`; a peer error mentioning "no active turn" becomes `State("the Codex turn ended before the steer was accepted")`.
Interrupt: both `AbortMode` values behave identically (the app-server offers nothing harder than `turn/interrupt`), the host-abort flag is set first, and the `turn/interrupt` request is bounded to 5 s; a missing thread/turn id or a failed request is not an error (the failure is logged).
Sessions and checkpoints: the thread id is the provider session id; the checkpoint is `{ provider: "codex", schema_version: "1", checkpoint_id: <thread id>, sequence: <monotonic u64>, data: {"thread_id", "resumed"}, provider_session_id: <thread id>, revision: None }`, published at `open` through `context.checkpoint_sink` with `CheckpointReason::SessionActivated` (a sink failure is a logged warning); `export_checkpoint` returns the live thread or `None`.
Resume policy: `New` ignores the checkpoint, `ResumeIfAvailable` resumes its `thread_id` when present, `RequireResume` without one is `Protocol("Codex checkpoint does not carry a thread id to resume")`.
`close` marks the cycle closing, clears the active turn and thread id, and closes the client, so `stop` joins the supervisor promptly.

## Card

Declared capabilities: `Steer`, `GracefulAbort`, `PersistentSession`, `Checkpoint`, `McpTools`, `EndpointModels`, `AuthFallback`.
Optional host capabilities: `ToolApproval`, `UserInput`, `CheckpointSink`, `McpServers`, `NativeToolGateway`, `DynamicToolCall`, `ProviderInteraction`, `ModelRequestObservation`; no host capability is required.
Fork modes `["full", "before"]` are declared as metadata only.
Deliberately absent: `ForceAbort` and `PauseResume` (the app-server has no pause and nothing stronger than `turn/interrupt`), `ModelSelection`/`ModelDiscovery` (the shared shell exposes no `HarnessModelControl` seam), `NativeTools` (host dynamic tools are refused).
`model_config(choice, fallback)` returns `vendor::nested_model_config(choice, fallback, true)`: `{"model": {model, effort, provider, api_base, api_key}}` plus `{"fallback_model": …}`, keeping the endpoint `provider` label (unlike the Claude Code mapping, which drops it).

## Design Decisions

- The transport drives the vendor CLI directly instead of linking an SDK: no vendor crate, no bundled binary, and a missing CLI is an operator error rather than "unsupported".
- An external endpoint forces `bypass_approvals_and_sandbox` because Codex's auto-review approval reviewer uses a built-in model that cannot be redirected to another provider, so every review against an endpoint would fail; the host's own policy still governs the approvals the adapter answers.
- Provider-specific wire spellings are accepted in both cases (camelCase payload members and snake_case equivalents) so a renamed field never silently drops an observation.
- Server requests are identified structurally by a non-null `id` rather than by a method allowlist, so a new app-server prompt is answered (with `-32601`) instead of stalling the turn.
- Interrupt and stream end are reconciled through the host-abort flag: an abort marks the turn host-aborted first, so the terminal result is an `Aborted` turn even when the stream closes before `turn/completed`.
- Tool call results are keyed by item id, and the completion repeats the name recorded at `item/started` so a renamed field at completion cannot change the tool identity.

## Deferred

- Rollout tracing and `experimental_raw_events`: the flag is decoded but `rawResponseItem/completed`/`rawResponse/completed` are dropped, and `HostCapability::ModelRequestObservation` is only advertised — `request_observation_wait_s` issues no wait.
- Host dynamic tools (`item/tool/call` answered `-32601`), and Python's rollout-derived dynamic tool registration.
- Model control — no `HarnessModelControl` seam, so selection/discovery stay off the card.
- Pausing, `ForceAbort`, and fork execution: the two fork-mode strings are metadata only.
- MCP ergonomics from Python: `mcp_env_passthrough`, `mcp_startup_timeout_s`, `mcp_required` and `mcp_default_tools_approval_mode` are decoded but not applied to the app-server.
- Python's `failure_classifier.py` beyond the normalized categories/codes above, and `system_prompt_mode = replace` semantics (the host prompt is always sent as `developerInstructions`).
- The idle-turn and `will` retry policies (`turn_idle_timeout_s`, `turn_idle_retries`, `max_will_retry_count`) are configuration only.

## Test Strategy

- `crates/openjiuwen-harness-providers/tests/codex.rs` (`#![cfg(feature = "vendor")]`, 11 tests) drives a real `tests/support::FakeCli` shell script over pipes: creation (fake binary unstarted, missing binary `Protocol`, invalid mapping `Validation`), a full turn through `HarnessIOAdapter` (answer delta, shell call/result, final text, lifecycle, thread id as session id), argv/`-c` rendering, approval round trip (accept/decline), steering, interruption, resumed thread, and a stream ending without `turn/completed`; every await is bounded by a 10 s `WAIT`.
- Module unit tests: 13 in `mod.rs` (identity, card, defaults, strict decoding and error wording, model fragment, create errors, `Debug` redaction), 7 in `frames.rs` (tool names/args/results, error classification, answer wire shape), 9 in `wire.rs` (argv groups, endpoint table, user-input flag, MCP overrides, env layering, thread params, bypass).
- The suite needs no `codex` binary, no SDK and no network; it is not `#![cfg(unix)]` (see the crate's Known Issues).

## Known Issues

- `./frames.rs` is legitimately larger than the crate's ~1000-line target: it is the single exhaustive app-server notification/server-request projection table (a one-to-one mirror of Python's `mapping.py`), so it is kept in one file instead of splitting one protocol table across several.

## Routing Table

- `./mod.rs` → constants, `CodexApprovalMode`/`CodexSystemPromptMode`, `CodexModelConfig`, `CodexHarnessConfig` (+ `from_json`/`validate`/`Debug`), `card()`, `model_config()`, `CodexHarnessProvider` and its `create`
- `./harness.rs` → `CodexDriver`: `VendorDriver` impl, spawn/handshake/thread open, turn loop, steer, interrupt, close, checkpoint export, `DriverState`
- `./frames.rs` → notification and server-request projection (`dispatch`, deltas, tool items, usage, error classification, terminal result, approval/user-input answers), plus `TurnAccumulator`
- `./wire.rs` → argv builders (`build_args`, endpoint and MCP overrides), environment layering (`build_env`, `OPENJIUWEN_CODEX_API_KEY`), `thread_params`
- `crates/openjiuwen-harness-providers/tests/codex.rs` (outside this node) → the offline app-server transport suite over `FakeCli`
