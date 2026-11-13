# DSH provider (`dsh` / DeepSeek Harness)

## Intent

The `DeepSeek` Harness provider: identity card, configuration data, model mapping, the subprocess launch contract, the session-event projection and the JSON-RPC driver.
Ports Python `openjiuwen.harness_providers.dsh` (v0.1.20) as a real subprocess adapter: `create` validates the config, resolves the `dsh` runtime and `DSH_HOME`, builds the argv, and returns an unstarted `vendor_cli::VendorHarness` wrapping `DshDriver`.
No DSH SDK is linked: the runtime is an operator-installed child process speaking newline-delimited JSON-RPC 2.0 over stdio.
The provider-independent shell (`VendorHarness`) owns the turn queue, lifecycle events, capability gating and process ownership policy; this module only speaks the vendor protocol.

## API Surface

`mod.rs` — configuration, card, provider, model mapping:
- `PROVIDER_NAME` (`"dsh"`), `CARD_NAME` (`"deepseek-harness"`), `IMPLEMENTATION_VERSION` (`"0.3.0"`).
- `DshSystemPromptMode` (`Append` / `Replace`, default `Replace`).
- `DshHarnessConfig` — 22 fields: `provider` (`deepseek-official`), `model` (`deepseek-v4-flash`), `reasoning_effort`, `max_tokens`, `system_prompt_mode`, `skills`, `skill_conflict` (`skip`), `cwd`, `runtime_cwd`, `dsh_bin`, `dsh_home`, `profile`, `patches`, `env`, `launch_args_override`, `initialize_timeout_seconds`, `request_timeout_seconds`, `shutdown_timeout_seconds` (1.0), `base_url`, `api_key` (`SecretString`), `system_prompt_env_var`, `event_buffer_capacity` (1024); `Default` and `from_json(&JsonObject)` (strict, provider-SPI path) plus a manual `Debug` that prints `env` as `<redacted>`.
- `provider() -> DshHarnessProvider`, `card() -> HarnessCard`, `model_config(&ModelChoice, Option<&ModelChoice>) -> HarnessResult<JsonObject>`.
- `DshHarnessProvider` implements `HarnessProvider` (`card`, `create`) and `ModelConfigProvider` (forwards to `model_config`); the registry entry is `model_configuring_provider(dsh::provider())`.
- Re-export: `DshDriver`.

`wire.rs` (module-private) — the launch contract:
- `require_dsh_home`, `build_args`, `build_env`, `resolve_cwd`, `skills_cwd`, `DSH_HOME_ENV`, `DEEPSEEK_BASE_URL_ENV`, `DEEPSEEK_API_KEY_ENV`, `MISSING_DSH_HOME_MESSAGE`.

`frames.rs` (module-private) — `DshAccumulator`:
- `new(turn_id, root_session_id)`, `is_root(session_id)`, `consume(session, event)`, `consume_child(session, session_id, event)`, `consume_subagent(session, method, payload)`, `forward(session, event_type, payload)`, `build_open_ended(stopped)`, `build_transport_failure(message)`.

`harness.rs` (module-private) — the driver:
- `DshDriver::new(config, binary, dsh_home, launch)` implementing `VendorDriver` (`label` → `"DeepSeek Harness"`, `open`, `close`, `run_turn`).
- Internal `DriverState { session_id, closing }`, `SharedProcess` (`RpcTransport` over the shared child), `PROVIDER_EVENT_NAME` (`"dsh"`), the turn messages, the failure codes `DSH_MISSING_TURN_END`, `DSH_UNKNOWN_FINISH_REASON`, `DSH_TRANSPORT`, and the protocol constants `METHOD_NOT_FOUND` (-32601), `SHUTDOWN_REQUEST_TIMEOUT` (1 s), `REAP_LOCK_TIMEOUT` (5 s).

## Launch Contract

Binary: `config.dsh_bin` or `dsh` on `PATH` via `vendor_cli::resolve_binary` (`label` "DeepSeek Harness", install hint `npm install -g deepseek-harness`); a missing or non-executable runtime is `HarnessError::Protocol` naming `dsh`.
`DSH_HOME` is mandatory, resolved in order `config.dsh_home`, non-blank `config.env["DSH_HOME"]`, non-blank host `DSH_HOME`, and reported as `HarnessError::Protocol` with `MISSING_DSH_HOME_MESSAGE` otherwise — the runtime never falls back to `~/.dsh`.
Argv is `dsh [--profile <profile>] [--patch <absolute path>]…`; `launch_args_override` replaces that vector verbatim.
Patch paths are made absolute against `config.cwd`, else `config.runtime_cwd`, else the process directory, so an unresolvable path is a `create`-time validation error rather than a failed start.
Environment: the host environment is inherited, then `config.env`, then `context.env`, then the adapter-owned values `DSH_HOME`, `DEEPSEEK_BASE_URL` / `DEEPSEEK_API_KEY` (from `base_url` / `api_key`), and `context.system_prompt` under `system_prompt_env_var` when one is declared and the prompt is non-empty; the last writer wins, so the host can never override the home or endpoint.
Working directory: `config.runtime_cwd`, else `context.cwd`, else `config.cwd`.
Skills (`config.skills`, `config.skill_conflict`) are installed through `install_for_provider(…, "dsh", project, …)` on a blocking task before the spawn, into `context.cwd`, else `config.cwd`, else `.`.
The child is spawned with `inherit_env: true` and `max_line_bytes: 0` (the transport default, 32 MiB).
The event buffer is `EventBufferConfig { capacity: event_buffer_capacity, overflow: CoalesceOrBlock }`.

## Wire Contract

`initialize` — `{"cwd"?, "provider", "model", "reasoningEffort"?, "maxTokens"?}`, awaited with `initialize_timeout_seconds` when configured and without a timeout otherwise; a failed initialize closes the client and reaps the process before returning the error.
No session-creation call exists: the server creates the session lazily on the first prompt, so `open` mints the provider session id `dsh-<uuid>` locally and reuses it for every turn of the cycle.
`session/prompt` — `{"sessionId", "contentBlocks": [{"type": "text", "text": …}]}`, one text block rendered from the turn's content by `harness_input_text`.
Responses are matched by JSON-RPC id; anything without an id is a notification and is dispatched by `method`.
A server-to-client request (a message carrying an `id`) is answered with error `-32601` (`unsupported DeepSeek Harness server request '<method>'`), because the adapter exposes no client-side capability.
`shutdown` — sent by `close` with a fixed one-second timeout; a runtime that does not acknowledge is closed and killed.
Commands: the pinned protocol has no interrupt request, so `interrupt`, `steer`, `pause`, `resume` and `export_checkpoint` keep the default `VendorDriver` `UnsupportedCapability` bodies; `close` is the only way to stop a running turn.

## Session-Event Projection

`session.event` with `params.{sessionId, event}`: an event whose `sessionId` is missing or equal to the root session id drives the turn, a child session's event is forwarded as a `child.session.event` provider event (payload `{event, sessionId}`, `turn/end` redacted) and changes nothing else.
`subagent.started` / `subagent.finished` become a `subagent` item (`Started` / `Completed`) identified by `childSessionId`, else `sessionId`, else `unknown-subagent`.
`session.event` without an `event` object, and every other notification method, is forwarded verbatim as a provider event (payload `{"params": …}`; an empty method becomes `unknown-notification`).
Root events by `type`:
- `assistant/chunk` — `text-delta` → answer delta, `reasoning-delta` → reasoning delta (output id `dsh-output:{turn}:{step}:{index}:{channel}`); `block-end` of a `text`/`reasoning` block → `Final` (finalized once); `usage` → the step's usage; any other chunk type → provider event `assistant/chunk`.
- `assistant/message` — one `Final` output per text/reasoning block (block id from the message, else `{message_id}:block:{index}`), one retained assistant `TurnMessage` (`id`, else `dsh-message:{turn}:{step}:{seq}`) with `native_turn`/`native_step`/`source` data, and the message's `usage`; `last_text_output` becomes the concatenated answer parts.
- `tool/call` → `tool_started` (call id from `callId`, name from `name`, `arguments` parsed as JSON when it is a string) and the name is remembered for the matching result.
- `tool/result` → `tool_completed` (call id from `message.source.callId`, else `callId`; content from `message.content`, else `content`); a non-null `error` becomes the completion text plus a `Warning` diagnostic.
- `step/start` / `step/end` → `step` item (`Started` / `Completed`) with item id `dsh-step:{turn}:{step}` and data `{provider_turn, provider_step}`.
- `turn/end` → records the reason, increments `native_turn_end_count`, publishes the redacted event as a provider event `turn/end`, and returns the terminal result.
Usage is kept per `(turn, step)`, published as a cumulative total on every change, and summed into `input_tokens`/`output_tokens`/`cached_input_tokens`/`reasoning_output_tokens` with `total_tokens = input + output + cached + cache_write` (`cache_write_tokens` also kept in `provider_data`).

## Turn Endings

`completed` and `max-tokens` → `Completed` with the last answer text, the retained messages and the usage.
`blocked` → `Interrupted` with `TurnTerminationKind::Policy` ("DeepSeek Harness blocked the turn").
`aborted` / `interrupted` → `Interrupted`: `interrupted` is always `Provider`; `aborted` maps `reason.reason.kind` `user` → `UserAbort`, `disposed` → `HarnessStop`, `hook` → `Policy`, anything else → `Provider`, with the redacted reason attached to `termination.provider_data`.
`error` → `Failed` with the Python message "DeepSeek Harness reported a failed turn", `code` from `reason.error.code` (string or number), category `provider`, `retryable: false`.
An unrecognized `kind` → `Failed` with `DSH_UNKNOWN_FINISH_REASON` and `provider_data.finish_reason`; a `turn/end` without a `kind`, or a stream that ends without one, → `Failed` with `DSH_MISSING_TURN_END`.
A `session/prompt` request failure → `Failed` with `DSH_TRANSPORT` and the transport message.
Every terminal result carries the retained messages, `final_output`, the usage and `provider_data.native_turn_end_count`.
Redaction (Python `_redact_turn_end_event` / `_safe_reason_data`): a surfaced `turn/end` keeps `kind`, the cause minus `message`, and an error limited to `{code, status, requestId, retryable}`.

## Card

Capabilities: `PersistentSession`, `McpTools`, `EndpointModels` only.
Optional host capability: `McpServers`; no required host capabilities and no fork modes.
`Steer`, `GracefulAbort`, `ForceAbort`, `PauseResume` and `Checkpoint` are deliberately absent because the pinned DSH protocol can express none of them, so the shell gates `abort`, `pause` and `export_checkpoint` with `UnsupportedCapability` before the driver is consulted.
`model_config` returns the flat `{"model", "base_url", "api_key"}` fragment, rejects a reasoning effort (`Validation`, "DSH has no reasoning effort setting") and any fallback (`UnsupportedCapability`, "DSH has no authentication fallback").

## Constraints

- `DshDriver` implements `VendorDriver` only; it never implements `HarnessProtocol` and never publishes lifecycle events — `VendorHarness` does.
- One child process per `start`/`stop` cycle: `create` spawns nothing, `open` spawns, `close` reaps (bounded `shutdown`, client close, stdin close, `wait`, then `kill` after 5 s).
- `open` refuses `ResumePolicy::RequireResume` and any supplied checkpoint with `UnsupportedCapability` before anything is spawned.
- The child is shared as `Arc<AsyncMutex<Option<ChildProcess>>>` between the `JsonRpcClient` transport (`SharedProcess`) and the driver's teardown, so exactly one owner closes, waits and kills it; after teardown a read yields `None` (EOF) instead of touching a dead process.
- A provider-level failure (blocked/failed turn, early stream end, prompt transport failure) is an `Ok(failed_turn)` carrying its normalized error; only "no open cycle" and "no open session" are `HarnessError::Protocol`.
- `run_turn` on a cycle whose `close` already started returns `build_open_ended(true)` (`HarnessStop`) without writing a prompt.
- Config decoding is strict on the `from_json` path (unknown keys rejected, sorted in the message), tolerant on plain serde; `api_key` is a `SecretString` and the manual `Debug` redacts `env`.
- Cross-field validation: non-empty `provider` / `model` / `dsh_home` / `profile` / `system_prompt_env_var`, positive `max_tokens`, `event_buffer_capacity` and every timeout, `launch_args_override` a non-empty array of non-empty strings, and `append` mode incompatible with `system_prompt_env_var`.
- A missing runtime or home is always `HarnessError::Protocol`, never `UnsupportedCapability`.
- No `unsafe`, no stdout logging (`tracing` only), files ≤1000 lines (`frames.rs` ~924, `harness.rs` ~537), all public items documented.

## Design Decisions

- The provider session id is minted locally (`dsh-<uuid>`) because the protocol has no session-creation call and the runtime reports no id; it stays sticky for the cycle and is cleared by `close`.
- Because the protocol has no interrupt request, the card declares no abort capability and `close` (bounded `shutdown`, then client close, which unblocks the running turn with `HarnessStop`) is the documented way to stop a turn; the IO adapter's abort fallback therefore terminates the harness rather than the turn.
- `turn/end` is both the turn terminator and a provider event: the accumulator returns the terminal result and publishes the redacted event in the same step, so a host can observe the native ending without seeing provider diagnostics.
- Output ids are content-position based (`dsh-output:{turn}:{step}:{index}:{channel}`), which makes a `block-end` chunk and the later `assistant/message` finalize the same output exactly once.
- `last_text_output` is the `final_output` projection, refreshed by each finalized answer block and each assistant message, so an `error`/unknown ending still reports whatever answer text arrived.
- The turn loop is a biased `select!` between the pinned `session/prompt` request (polled only until it answers) and `next_incoming`, so notifications cannot starve the response and the response cannot hide a buffered notification.
- Skills install before the spawn so a failed install is a start error, not a partially launched runtime.
- `close` is best effort and never fails; the 1 s shutdown wait and 5 s process-lock wait are fixed constants rather than `shutdown_timeout_seconds`.
- `skills_cwd` falls back to `.` so the installer never has to invent a destination.

## Deferred / Known Gaps

- Cordis plugin overlays and composition: only `--profile` / `--patch` argv pass-through is implemented, Python's composition builder is not ported.
- Notification-mapping parity beyond the documented events: unmapped notifications are forwarded verbatim as provider events instead of being interpreted.
- MCP servers from the host context are not transmitted over the wire — the card advertises `McpTools` / `HostCapability::McpServers` but `initialize` carries no MCP configuration, so the runtime's own tool configuration is authoritative.
- `request_timeout_seconds` and `shutdown_timeout_seconds` are decoded, validated and `Debug`-printed but not applied: the prompt request is unbounded and shutdown uses the fixed 1 s constant.
- `interrupt` / `steer` / `pause` / `resume` / `export_checkpoint` are unsupported by the protocol (default `UnsupportedCapability` bodies), and the card declares no `ModelSelection` / `ModelDiscovery`.
- No OTel/trajectory observation and no failure classifier beyond the normalized turn errors.

## Test Strategy

Integration suite `crates/openjiuwen-harness-providers/tests/dsh.rs` — `#![cfg(all(feature = "vendor", unix))]`, 14 tests, run offline against `tests/support::FakeCli` (a POSIX shell script answering `initialize`, `session/prompt` and `shutdown` and emitting `session.event` frames; it records argv, `DSH_HOME`, `PWD` and every request line in a transcript).
Cases: cycle start + `dsh-` session id + prompt payload, missing runtime, missing home, unknown key (nothing spawned), argv/`DSH_HOME`/cwd propagation, chunk projection through `HarnessIOAdapter` (lifecycle, deltas, final, tool pair), prompt/usage contract, blocked → policy abort, failed turn keeping the provider code with the diagnostic redacted, stream without `turn/end`, unknown ending, subagent items plus forwarded child events, required resume refusal (unspawned), and abort/pause/export gated while a turn runs.
All awaits are bounded (`BOUND` 10 s) and ordering is observed through the event cursor or the adapter's chunk stream.
Unit tests beside `mod.rs` (13: identity constants, card, defaults, `from_json` strictness/validation, `model_config`, missing runtime, `Debug` redaction) and beside `wire.rs` (10: home resolution, argv, override, patch absolutization, env layering, cwd/skills resolution).
Run with `cargo test -p openjiuwen-harness-providers --features vendor --test dsh`.

## Routing Table

- `./mod.rs` → card, constants, `DshHarnessConfig` + `from_json` + validation, `DshSystemPromptMode`, `model_config`, `DshHarnessProvider` (create path); unit tests
- `./wire.rs` → binary-independent launch contract: `DSH_HOME` resolution, argv (profile/patches/override), environment layering, cwd and skills-cwd resolution; unit tests
- `./frames.rs` → `DshAccumulator`: session/subagent/notification dispatch, output/tool/step/usage projection, turn endings and `turn/end` redaction
- `./harness.rs` → `DshDriver`: spawn/initialize/prompt/shutdown over `vendor_cli::JsonRpcClient`, notification dispatch, process sharing and reaping
- `../../../tests/dsh.rs` → the offline end-to-end suite over `FakeCli` (`tests/support/mod.rs`)
