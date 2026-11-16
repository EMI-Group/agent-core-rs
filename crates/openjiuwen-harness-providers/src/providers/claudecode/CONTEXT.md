# providers/claudecode — Claude Code provider

## Intent

The `claudecode` registry entry: a real subprocess adapter over the operator-installed `claude` CLI in stream-json mode.
Ports Python `openjiuwen.harness_providers.claudecode` (v0.1.20: `config.py`, the provider class, `model_config`, the harness transport, `lifecycle.py` and `failure_classifier.py`).
Owns only provider selection, configuration, argv/environment, and the CLI wire mapping; turn queue, lifecycle events, capability gating and teardown live in `crate::vendor_cli::VendorHarness`.
`ClaudeCodeHarnessProvider::create` decodes `ClaudeCodeHarnessConfig`, resolves `claude` via `vendor_cli::resolve_binary`, and returns an unstarted `VendorHarness` wrapping `ClaudeDriver`; it never spawns the CLI.
No vendor SDK is linked: the CLI is a child process speaking NDJSON on stdin/stdout.

## API Surface

`mod.rs` (public):
- Constants `PROVIDER_NAME` (`"claudecode"`), `CARD_NAME` (`"claude-code"`), `IMPLEMENTATION_VERSION` (`"0.1.0"`), `DEFAULT_CLAUDE_MAX_BUFFER_SIZE` (`32 * 1024 * 1024`).
- `ClaudePermissionMode` (`default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`, `auto`; `Default` = `BypassPermissions`) and `ClaudeSystemPromptMode` (`append`, `replace`; `Default` = `Append`) — typed in place of Python `Literal` strings, wire values and error text preserved.
- `ClaudeModelConfig { model, api_base, api_key: Option<SecretString>, effort }` + `from_json` (rejects unknown keys, rejects provided-but-empty strings).
- `ClaudeCodeHarnessConfig` — 20 keys (`skills`, `skill_conflict`, `cwd`, `add_dirs`, `env`, `inherit_process_env`, `cli_path`, `model`, `fallback_model`, `session_id`, `permission_mode`, `system_prompt_mode`, `include_partial_messages`, `max_turns`, `max_buffer_size`, `settings`, `settings_env`, `event_buffer_capacity`, `request_observation_wait_s`, `lifecycle_ack_timeout_s`) + `from_json` + manual `Debug` that redacts `env`/`settings_env`.
- `provider() -> ClaudeCodeHarnessProvider`, `card() -> HarnessCard`, `model_config(&ModelChoice, Option<&ModelChoice>) -> HarnessResult<JsonObject>`.
- `pub use harness::ClaudeDriver`; `ClaudeDriver::new(config, binary)`.
- The `../vendor.rs` helpers (`config_error_to_harness`, `nested_model_config`, `vendor_card`) are used by this module but stay crate-internal and are not re-exported.

Card (`vendor::vendor_card("claude-code", "0.1.0", …)`):
- Capabilities: `Steer`, `GracefulAbort`, `PersistentSession`, `Checkpoint`, `McpTools`, `ModelSelection`, `ModelDiscovery`, `EndpointModels`, `AuthFallback` — deliberately *not* `ForceAbort`, `PauseResume`, so the shell rejects those before the driver runs.
- Optional host capabilities: `ToolApproval`, `UserInput`, `CheckpointSink`, `McpServers`, `ProviderInteraction`, `ModelRequestObservation`; no required host capabilities.
- Fork modes `full`, `before` (metadata only; nothing implements a fork).

`model_config` maps a choice onto `{"model": {…}, …}` (+ `{"fallback_model": {…}}`) and drops the endpoint's `provider` label, because Claude Code speaks the Anthropic protocol only.

Transport contract (implemented by `harness.rs` + `wire.rs`):
- One child process per `start`/`stop` cycle: spawned in `open`, reaped in `close`; `create` only validates the config and resolves the executable.
- argv in this order: `--print --input-format stream-json --output-format stream-json --verbose`; `--include-partial-messages` when enabled; `--permission-mode <wire>`; `--permission-prompt-tool stdio` when the host is interactive; `--model`/`--effort` from `config.model`; `--resume <id>` or `--session-id <id>`; `--max-turns <n>`; one `--add-dir` per entry; `--append-system-prompt <text>` or `--system-prompt <text>` when the host supplied a prompt; `--settings <compact json>`.
- `--settings` carries `settings_env` plus `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` from `config.model` under `{"env": {…}}`, and merges a `config.settings` object over it (its top-level keys and its own `env` string entries win).
- Environment: inherited when `inherit_process_env`, then `config.env`, then `context.env`; working directory is `config.cwd`, else `context.cwd`; label in errors is `Claude Code`.
- Per-line stdout ceiling is `max_buffer_size` (default `DEFAULT_CLAUDE_MAX_BUFFER_SIZE`); the buffer is a transient per-line string.
- Skills from `config.skills` are installed into the project directory (`.claude/skills`) on a blocking thread before the spawn; a failure is `HarnessError::Protocol`.

Wire frames (`frames.rs`) and their projection:
- `system` — captures `session_id` and emits provider event `system/<subtype>`.
- `assistant` — `text` → `Answer`/`Final` output, `thinking` → `Reasoning`/`Final`, `tool_use` → `tool_started` plus a `tool_call` content block, other block kinds kept under their own kind; `usage` (`input_tokens`, `output_tokens`, `cache_read_input_tokens`) is merged and emitted; a reported `message.error` is classified (`classify_assistant_error`), kept as `TurnAccumulator::pending_error` and published as an `Error` diagnostic; the message is recorded when at least one block survives. Subagent frames (`parent_tool_use_id` non-null) are recorded but publish no outputs or tool events.
- `user` — only `tool_result` blocks: `tool_completed` (its `tool_name` resolved from the matching `tool_use` block recorded in `TurnAccumulator::tool_names`, falling back to the `tool_use_id` for a call the turn never announced) plus `tool_result` content blocks in a `Tool` role message carrying the call id.
- `stream_event` — only when `include_partial_messages` and not a subagent frame: `message_start`/`message_stop` bracket an index, `content_block_delta` `text_delta`/`thinking_delta` emit `Delta` outputs; everything else ignored.
- `control_request/can_use_tool` — answered from the host policy (see interactions); any other `control_request` subtype is forwarded as a provider event.
- `control_response` — ignored.
- `command_lifecycle` — forwarded as a provider event and fed to the turn's receipt tracker (see the turn boundary below).
- `result` — records the turn's terminal outcome in `TurnAccumulator` (`TerminalOutcome`), it does not terminate the turn by itself: a `success` result stores its `result` text, an `is_error`/non-`success` result stores `merge_pending_error(pending, classify_result_message(frame))` and publishes an `Error` diagnostic carrying `subtype`/`http_status`. The last result of the turn states how it ended.
- Anything else — provider event named after the frame type (`unknown` for a non-object line); nothing is silently dropped.

Turn boundary (`lifecycle.rs`, Python's `TurnCycleTracker` + `LifecycleTap`):
- The turn loop feeds every inbound frame to `TurnCycleTracker::note_inbound` and stops when it returns `true`: a `result` arrived **and** no user message the turn wrote is still outstanding.
- Every write is recorded first (`note_outbound`, keyed by the frame `uuid`; subagent frames with a non-null `parent_tool_use_id` are skipped), including steered messages, which get their own id (`{turn message id}-steer-{n}`) so a receipt always names one message.
- A receipt that follows the result keeps the turn open, so a message the CLI answers in a *new* cycle stays inside the same turn and is projected into it; the last cycle's result is the turn's outcome.
- A receipt is `queued`/`started`/`completed` on `command_uuid`; `completed` removes the message, the other two only advance its state.
- A failed or interrupted cycle (`subtype != "success"` or `is_error`) abandons the outstanding messages at once, because the CLI stops working its queue there.
- `lifecycle_seen` is sticky per cycle: once the CLI reported any receipt, an unanswered message blocks the turn; when it never reports one, the first result settles the turn (the pre-tracking behaviour).
- The acknowledgement deadline is armed only when every outstanding message is still `submitted`; a message the CLI took (`queued`/`started`) is never timed out. When it fires, the messages are dropped from the outstanding set, the turn settles, and each abandonment is published as a `warning` diagnostic (`"Claude Code never acknowledged N message(s): …"`).
- `SETTLED_SUBTYPE` (`openjiuwen.turn.settled`) belongs to Python's synthetic sentinel frame; this adapter drives the CLI directly, so the loop reads the tracker's verdict and never injects that frame.

Failure classification (`failure.rs`, Python's `failure_classifier.py`), public API re-exported from `mod.rs`:
- `FAILURE_CATEGORIES` (9: `auth_required`, `quota_exceeded`, `rate_limited`, `server_unavailable`, `network_timeout`, `process_start_failed`, `request_rejected`, `sdk_error`, `unknown`) and `MAX_FAILURE_DETAIL_CHARS` (8000, plus the `...[truncated]` suffix).
- `classify_assistant_error(error, message)` — the identifier maps to a category (`authentication_failed` → `auth_required`, `billing_error` → `quota_exceeded`, `invalid_request` → `request_rejected`, `rate_limit` → `rate_limited`, `server_error` → `server_unavailable`, anything else → `sdk_error`), the identifier is the `code`, and the owning message's text blocks are the message detail.
- `classify_result_message(frame)` — `api_error_status` decides the category (400 `request_rejected`, 401/403 `auth_required`, 429 `rate_limited`, 500/529 `server_unavailable`), the `subtype` decides it otherwise; the message is the joined informative `errors`, else the `result` text, else `Claude turn failed: HTTP <status>` / `Claude turn failed (<subtype>)`; the `code` is the subtype and `provider_data` carries `http_status`.
- `merge_pending_error(pending, terminal)` — the specific pending category/`retryable` wins over a generic terminal one (no HTTP status, or `sdk_error`); the pending message is appended when it adds text the terminal message lacks.
- `classify_cli_failure(phase, error_type, message, stderr_tail)` — `FailurePhase::{Startup, Turn}`; a timeout type is `network_timeout`, `CLINotFoundError`/`CLIConnectionError`/`ProcessError` at `Startup` is `process_start_failed`, anything else is `sdk_error`; `provider_data.sdk_error_type` names the type and the CLI's error output is appended as `Error output: …`.
- `retryable` is true exactly for `rate_limited`, `server_unavailable` and `network_timeout`.
- The driver uses it for its two stream failures: a stream that ends before the turn is `classify_cli_failure(Turn, "CLIConnectionError", …)` with code `CLAUDE_MISSING_RESULT`, a read/decode error `classify_cli_failure(Turn, "ProcessError", …)` with code `CLAUDE_TRANSPORT`; both stay `sdk_error`, non-retryable, and append the process's stderr tail.

Sessions and checkpoints:
- Session id is `config.session_id`, else `vendor_cli::derive_session_id(host_session_id, agent_name)` (SHA-256-seeded UUID shape).
- The resume id comes from `resolve_resume_id`: `New` never resumes, `ResumeIfAvailable` resumes only the supplied checkpoint, `RequireResume` prefers the checkpoint and otherwise the derived id (which always exists, so the driver's missing-resume message is unreachable).
- A supplied checkpoint's `checkpoint_id` is replaced by `data.session_id` when present (this adapter's own checkpoints put the id in both places; older payloads only carry `data.session_id`).
- `--resume <id>` is passed instead of `--session-id` whenever a resume id resolved.
- `open` publishes a `SessionActivated` checkpoint to `context.checkpoint_sink` (`provider` `claude-code`, `schema_version` `"1"`, `checkpoint_id` = provider session id, monotonic `sequence` from 1, `data {session_id, resumed}`, `provider_session_id` set, `revision` `None`); sink failures are logged with `tracing::warn!` and never fail the start.
- `export_checkpoint` returns that checkpoint with `resumed: false`, or `None` when no session is known.
- A `session_id` reported later (in `system/init` or `result`) that differs from the chosen one is adopted with one `tracing::warn!`; the shell keeps the id from `open` and rejects a restart with a different provider session as `HarnessError::State`.

Interactions:
- `PermissionPolicy::from_context` reads `HostCapability::{UserInput, ToolApproval, DynamicToolCall, NativeToolGateway}`; `interactive()` is `user_input || tool_approval`.
- A non-interactive host keeps the configured `--permission-mode`, gets no `--permission-prompt-tool`, and any stray `can_use_tool` prompt is answered allow-without-input so it can never wedge the turn.
- An interactive host forces `--permission-mode default` plus `--permission-prompt-tool stdio`, so the CLI asks the host instead of its bypass mode.
- `can_use_tool` with `tool_name == "AskUserQuestion"` becomes `ask_user_input` (request id `claude-ask:{tool_use_id}`, prompt from the joined `questions[].question` texts, choices from the first question's `options[].label`, `provider_data {tool_use_id, tool_name}`); answers are returned as `updatedInput` with an `answers` field, a missing answer denies with `"user input was not provided"`.
- Any other tool becomes `ask_tool_approval` (request id `claude-approval:{tool_use_id}`, call id `tool_use_id`); `Allow` → `{"behavior":"allow"}` (plus `updatedInput` when the host rewrote arguments), `Deny` → `{"behavior":"deny","message":…,"interrupt":…}`.
- The answer is written as `{"type":"control_response","response":{"subtype":"success","request_id":…,"response":…}}`.
- Steering writes the steered text as another `user` frame with the steer's message id.
- Abort: both `AbortMode`s send the same `control_request {"request":{"subtype":"interrupt"}}` with a fresh UUID request id; the turn ends `UserAbort` on a terminal `result`, on EOF/read error, or on close, and `Timeout` (`"Claude Code did not acknowledge the interrupt"`) when `lifecycle_ack_timeout_s` (default 10 s) elapses first.
- An interrupt recorded while the turn has not yet polled its control channel is still honoured: a flag is checked before the first read and after every loop iteration.
- `close` sets the closing flag, sends `Close` to the running turn, waits at most 4 s for the turn to release the child, closes stdin, waits (`ChildProcess::wait`, 5 s) and then kills it; a turn that never releases the child is `HarnessError::State("the Claude Code harness did not stop cleanly")`.

## Constraints

- The driver implements `VendorDriver` only; it must never implement `HarnessProtocol`, and `VendorHarness` is the only publisher of lifecycle events.
- Capability gating happens in the shell: force abort and pause/resume are rejected without touching the process because the card omits `ForceAbort`/`PauseResume`.
- No vendor SDK and no bundled binary: the resolved program is `claude` (from `PATH` or `cli_path`), and a missing/non-executable binary is `HarnessError::Protocol` naming `claude`, `not executable`, and the install hint — never `UnsupportedCapability`, which is reserved for commands the pinned protocol cannot express.
- Config decode errors are mapped to `HarnessError::Validation` through `vendor::config_error_to_harness` and surface before the binary is resolved.
- Plain serde decoding tolerates unknown keys (crate rule) while `from_json` rejects them; `skill_conflict`/`permission_mode`/`system_prompt_mode` accept Python's string wire values only through `from_json`.
- Cross-field validation: empty `session_id`, `max_turns == 0`, `max_buffer_size == 0`, `event_buffer_capacity == 0`, `lifecycle_ack_timeout_s <= 0` are `Error::Validation`.
- Files ≤1000 lines; no `unsafe`; `tracing` only (no stdout); unit tests live beside the module they test (`mod.rs`, `lifecycle.rs`, `frames.rs`).

## Design Decisions

- The Claude CLI is driven as a plain child process instead of through the Claude Agent SDK, so the adapter stays toolchain-free while matching Python's stream-json contract: one user frame per turn, `system/init` for the session id, `control_request/can_use_tool` for approvals, and the receipt-tracked `result` as terminator.
- The turn boundary is owned by `TurnCycleTracker` rather than by the first `result`, because the CLI acknowledges queued messages with `command_lifecycle` frames and may answer a steered message in a *new* cycle after the first `result`; ending at the first `result` would truncate that cycle's text. The tracker is only authoritative once the CLI proved it emits receipts (`lifecycle_seen`), so a CLI that never does keeps the old behaviour.
- The acknowledgement deadline (`lifecycle_ack_timeout_s`) is armed only while every outstanding message is still `submitted`: a message the CLI took (`queued`/`started`) is being worked on, so timing it out would end a live turn.
- Failure classification is a pure module (`failure.rs`) so the shell-facing `TurnError` can be built without a driver: the category vocabulary, `retryable` flags and `provider_data` (`sdk_error_type`, `http_status`) are Python's, and the 8000-character detail cap keeps a runaway CLI dump from flooding the host.
- An interactive host downgrades `--permission-mode` to `default` and adds `--permission-prompt-tool stdio`, otherwise the CLI's own bypass mode would never ask the host.
- The 32 MiB line ceiling exists because tool results that embed large payloads (a base64 image read by the CLI `Read` tool is ~4/3 of the file) otherwise kill the turn with a decode error.
- Endpoint injection rides `--settings` rather than process environment so the adapter's API base/key win over the user's `settings.json`; `settings_env` is the base layer and an explicit `settings.env` wins.
- `open` normalizes the checkpoint id from `data.session_id` before resolving the resume id, so checkpoints written by this adapter and by Python both resume.
- A turn multiplexes its stdout read against a control channel (`biased` toward the channel), so steer/interrupt/close win over a blocked read; the read is resumable but a partially read line is discarded, which is safe because commands normally arrive while the CLI is idle.
- The `interrupt_requested` flag covers an abort issued before the turn's control channel is observed, and is reset at every turn start so a stale abort cannot end a later turn.
- Session-id adoption is warn-and-adopt (drive the CLI's own identity) instead of failing, because the CLI owns the conversation; only the shell's cross-cycle identity check is strict.
- Permission requests always get an answer: a non-interactive host, a missing handler, an invalid request or a handler failure become allow-without-input (approval) or deny (user input), because a vendor CLI must never hang on a prompt nobody can answer.

## Deferred

- `request_observation_wait_s` (default 5 s) is decoded, defaulted and `Debug`-printed but never used: no model-request observation wait is issued and `HostCapability::ModelRequestObservation` is advertised only.
- OTLP model-request observation and the trace receipts of `observation.py`; delivery receipts (`lifecycle.py`) and failure classification (`failure_classifier.py`) are implemented in `lifecycle.rs`/`failure.rs`.
- Python's `LifecycleTap` plumbing around the tracker: the tracker's abandonment warnings are published as `warning` diagnostics and its verdict closes the turn, but the synthetic `SETTLED_SUBTYPE` frame is not injected (this adapter drives the CLI directly, not Python's `Query` wrapper).
- Fork modes `full`/`before`: declared on the card, not implemented by the protocol or the shell.
- MCP: `McpTools` and `HostCapability::McpServers` are declared, but `context.mcp_servers` is never forwarded (no `--mcp-config`/permission bridge), unlike the Codex adapter.
- Model control: `ModelSelection`/`ModelDiscovery` are card metadata; `HarnessProtocol` exposes no model-control seam and the driver implements none.
- `AuthFallback` is represented only by `fallback_model` in the `model_config` fragment; no runtime switch after an authentication failure.
- Pausing and force abort are intentionally undeclared (the CLI protocol has no such command), so the shell returns `UnsupportedCapability`.

## Known Issues

- `wire.rs` has no unit tests; argv/env coverage comes from `crates/openjiuwen-harness-providers/tests/claudecode.rs` (20 tests over `support::FakeCli`: creation, full turn projection, allow/deny approvals, stdio permission argv, interrupt, `--session-id`, `--resume`, missing result, error result, the receipt-driven turn boundary and the classifier end-to-end), while `frames.rs` covers only the `user`-frame tool-result projection in its 2 unit tests.
- The receipt tracker identifies a message by the `uuid` of the `user` frame it wrote; a CLI that reports `command_lifecycle` for a *different* id (e.g. its own internal queue slot) is treated as never having acknowledged the message, so the turn ends on the acknowledgement deadline instead of the receipt.

## Routing Table

- `./mod.rs` → constants, config types + `from_json`/`validate`, permission/system-prompt enums, card, `provider()`, `model_config`, `ClaudeCodeHarnessProvider`, the `failure`/`lifecycle` re-exports, and 13 unit tests (identity constants, card, defaults, decode rules, redaction, create errors)
- `./harness.rs` → `ClaudeDriver` (`VendorDriver`: `open`, `run_turn`, `steer`, `interrupt`, `close`, `export_checkpoint`), turn/control multiplexing, the receipt-tracked turn loop, interrupt acknowledgement, checkpoint payload, session adoption; module docs hold the transport contract
- `./frames.rs` → `dispatch` and per-frame projection (`system`, `assistant`, `user`, `stream_event`, `control_request`, `control_response`, `result`, unknown), `TurnAccumulator`/`TerminalOutcome`, usage mapping, `report_assistant_error`, `with_tail`/`compact_json`, and 2 unit tests of the tool-result name resolution
- `./lifecycle.rs` → `TurnCycleTracker` (outbound recording, `queued`/`started`/`completed` receipts, `lifecycle_seen`, the acknowledgement deadline, failed-cycle abandonment, diagnostics), `CommandState`, `LIFECYCLE_FRAME_TYPE`, `SETTLED_SUBTYPE`, and 9 unit tests of the settlement rules
- `./failure.rs` → `FAILURE_CATEGORIES`, `MAX_FAILURE_DETAIL_CHARS`, `FailurePhase`, `classify_assistant_error`, `classify_result_message`, `merge_pending_error`, `classify_cli_failure` and the private category/detail helpers
- `./wire.rs` → `build_args` (argv order), `settings_json` (`--settings` env layer), `build_env`, permission-mode wire values
- `../vendor_cli/` → shared transport this adapter depends on: `ChildProcess`, `VendorHarness`, `PermissionPolicy`, `ask_tool_approval`/`ask_user_input`, `derive_session_id`/`resolve_resume_id`/`checkpoint_str`, turn constructors
- `crates/openjiuwen-harness-providers/tests/claudecode.rs` → offline end-to-end suite over `support::FakeCli` (`#![cfg(all(unix, feature = "vendor"))]`, 20 tests)
