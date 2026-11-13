# vendor_cli

## Intent

Shared subprocess plumbing for the vendor CLI harnesses (`claudecode`, `codex`, `dsh`).
Every adapter needs the same mechanics: resolve the CLI binary, spawn it, speak newline-delimited JSON over stdin/stdout, retain a bounded stderr tail, publish harness SPI events through a single-consumer cursor, derive a stable provider session id and translate host permissions.
This module owns those pieces so the three follow-up adapters only contribute their own wire-message mapping.
It is plumbing only — no harness behavior and no vendor SDK.

## API Surface

- `binary` — `BinarySpec { name, explicit, label, install_hint }`, `resolve_binary(&BinarySpec) -> HarnessResult<PathBuf>`, `is_available(&BinarySpec) -> bool`; explicit paths are validated (exists, regular file, execute bit on Unix) and a missing binary reports the label, the name, the install hint and the `PATH`/`cli_path` fix as `HarnessError::Protocol`.
- `jsonl` — `parse_line`, `parse_line_labelled`, `encode_line`, `get_str`, `get_object`, `get_array`, `get_bool`, `get_u64`, `get_f64`, `string_list`, `snippet`.
- `child` — `ChildProcessConfig { program, args, cwd, env, inherit_env, label, max_line_bytes }`, `ChildProcess::{spawn, write_json, write_line, read_json, stderr_tail, close_stdin, kill, wait, pid}`; `DEFAULT_MAX_LINE_BYTES`, `STDERR_TAIL_LINES`, `STDERR_TAIL_CHARS`, `EXIT_TIMEOUT`.
- `events` — `EventBus::{new, emit, events, turn_events, close, is_closed, len, is_empty}`, `EnvelopeBuilder::{new, set_provider_session_id, next, next_for_turn, next_for_item}`, `DEFAULT_CAPACITY`.
- `permissions` — `PermissionPolicy { user_input, tool_approval, dynamic_tools, native_tools }` with `from_context` / `interactive`, `ApprovalDecision::{Allow, Deny}`, `ask_tool_approval`, `ask_user_input`.
- `session` — `derive_session_id`, `resolve_resume_id`, `checkpoint_str`.
- Crate-root flat re-exports mirror every item above under `openjiuwen_harness_providers::vendor_cli`.

## Constraints

- Files stay ≤1000 lines; `tracing` only; no `unsafe`; no panics on user input.
- Errors are `HarnessError::{protocol, state, validation, unsupported_capability}` — a missing CLI is always `Protocol` (operator configuration), never `UnsupportedCapability`.
- Async-first: no guard is held across an `.await`; every wait on a child is bounded by `EXIT_TIMEOUT` (5 s); nested waits are bounded with `tokio::time::timeout`.
- The stderr drain is exactly one task per child, joined on exit and aborted when a wait times out.
- `EventBus` is single-consumer: a second cursor while a lease is held is `HarnessError::State("the harness observation stream already has an active consumer")`; overflow drops the oldest event and warns once per overflow episode.
- `get_object` returns an owned `JsonObject` because `JsonObject` is a `BTreeMap` while serde stores objects as `serde_json::Map`; no borrowed view exists.
- `parse_line` has no provider label in its signature, so it uses the neutral label `harness`; adapters call `parse_line_labelled` when the message must name the vendor.
- `derive_session_id` is `sha256` over the compact JSON seed `[host_session_id, agent_name]` formatted as `8-4-4-4-12` (Python uses a UUID5 over `NAMESPACE_URL`).
- Auto-approval: an undeclared capability or a missing handler allows the tool call / returns `None` for user input instead of blocking a turn.

## Routing Table

- `./binary.rs` → executable resolution and its actionable failures
- `./jsonl.rs` → pure NDJSON parse/encode and typed object accessors
- `./child.rs` → one spawned CLI process: piped NDJSON stdin/stdout, bounded stderr tail, bounded waits
- `./events.rs` → the observation bus, its single-consumer cursor/turn filter, and the envelope builder
- `./permissions.rs` → host permission policy and the approval/user-input translation
- `./session.rs` → deterministic session ids, resume-policy resolution, checkpoint field access
