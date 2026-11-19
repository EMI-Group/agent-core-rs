# vendor_cli

## Intent

Shared subprocess plumbing plus the generic harness shell for the vendor CLI harnesses (`claudecode`, `codex`, `dsh`) and the in-process `native_v2`.
Every adapter needs the same mechanics: resolve the CLI binary, spawn it, speak newline-delimited JSON over stdin/stdout, retain a bounded stderr tail, publish harness SPI events through a single-consumer cursor, derive a stable provider session id and translate host permissions.
Adapters on the JSON-RPC wire (the Codex app-server, the DeepSeek Harness runtime) additionally need id-correlated request/response traffic with notifications and server-to-client requests, which `rpc` supplies over the same child process.
An adapter then implements one trait (`VendorDriver`) and is wrapped by `VendorHarness`, which supplies the whole `HarnessProtocol`: lifecycle, turn queue and supervisor, event sequence, capability gating, abort/pause/resume and teardown.
It is plumbing and protocol scaffolding only — no harness behavior and no vendor SDK.

## API Surface

- `binary` — `BinarySpec { name, explicit, label, install_hint }`, `resolve_binary(&BinarySpec) -> HarnessResult<PathBuf>`, `is_available(&BinarySpec) -> bool`; explicit paths are validated (exists, regular file, execute bit on Unix) and a missing binary reports the label, the name, the install hint and the `PATH`/`cli_path` fix as `HarnessError::Protocol`.
- `jsonl` — `parse_line`, `parse_line_labelled`, `encode_line`, `get_str`, `get_object`, `get_array`, `get_bool`, `get_u64`, `get_f64`, `string_list`, `snippet`.
- `child` — `ChildProcessConfig { program, args, cwd, env, inherit_env, label, max_line_bytes }`, `ChildProcess::{spawn, write_json, write_line, writer, read_json, stderr_tail, close_stdin, kill, wait, pid}`, `StdinWriter` (a cloneable, per-line-flushing handle onto the same pipe so a control message can be written while another task is parked on a read) and the `LineWriter` trait the JSON-RPC client holds; `DEFAULT_MAX_LINE_BYTES` (32 MiB), `STDERR_TAIL_LINES` (40), `STDERR_TAIL_CHARS` (8000), `EXIT_TIMEOUT` (5 s). `read_json` enforces `max_line_bytes` *while* it reads (`fill_buf`/`consume`): at most the ceiling is retained, the surplus of an over-long line is drained to its newline (or to EOF) without being buffered, and the line is reported as `Protocol("a {label} CLI output line exceeded {n} bytes")`.
- `rpc` — JSON-RPC 2.0 over NDJSON: `RpcTransport { write_line, read_json, stderr_tail }` (implemented for `ChildProcess`) and `JsonRpcClient` (`spawn(transport, label)`, `request`, `request_with_timeout`, `notify`, `next_incoming`, `respond`, `respond_error`, `close`, `is_closed`, `label`, `stderr_tail`), plus `request_id_value(u64)`.
  Wire rules: a request is `{"id": <u64>, "method": <str>, "params": <value>}`, a notification omits `id`, a response is `{"id": …, "result": …}` or `{"id": …, "error": {"code": …, "message": …}}`, and a server-to-client request carries both `id` and `method` (delivered by `next_incoming`, answered with the same id echoed verbatim).
- `events` — `EventBus::{new, emit, events, turn_events, close, is_closed, len, is_empty}`, `EnvelopeBuilder::{new, set_provider_session_id, next, next_for_turn, next_for_item}`, `DEFAULT_CAPACITY`.
- `permissions` — `PermissionPolicy { user_input, tool_approval, dynamic_tools, native_tools }` with `from_context` / `interactive`, `ApprovalDecision::{Allow, Deny}`, `ask_tool_approval`, `ask_user_input`.
- `session` — `derive_session_id`, `resolve_resume_id`, `checkpoint_str`.
- `driver` — the adapter seam: `TurnRequest { turn_id, message_id, content, metadata }` (`new`, `text`), `TurnContext` (`new(bus, envelope)`, `emit`, `emit_correlated`, `emit_event`, `output`, `tool_started`, `tool_completed`, `usage`, `provider_event`, `diagnostic`), `#[async_trait] VendorDriver { label, open, close, run_turn, steer, interrupt, pause, resume, export_checkpoint }` (the last five have capability default bodies returning `HarnessError::UnsupportedCapability("the {label} harness does not support …")`; `export_checkpoint` defaults to `Ok(None)`), the result constructors `completed_turn`, `interrupted_turn`, `failed_turn`, `turn_error`, and `TOOL_CALL_ITEM_TYPE` (`"tool_call"`), `PROVIDER_SCHEMA_VERSION` (`"1"`).
  `tool_started`/`tool_completed` publish `ItemLifecycleEvent { item_type: "tool_call", data { tool_call_id, name|tool_name, arguments, result, rendered_result } }`, which is exactly what `io_adapter` projects into tool items.
- `harness` — `VendorHarness::new(card, EventBufferConfig, Arc<dyn VendorDriver>)` + `with_open_timeout(Duration)` + `label()`, `DEFAULT_OPEN_TIMEOUT` (30 s), a complete `HarnessProtocol` implementation: `card`, `state`, `provider_session_id`, `event_buffer_config`, `start`, `stop`, `events`, `turn_events`, `send`, `abort`, `pause`, `resume`, `export_checkpoint`.
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
- `JsonRpcClient` runs exactly one task, and that task both reads and writes: writes are queued and flushed between reads, because one `&mut` transport cannot be read and written at the same time and a buffered read must never be abandoned mid-line (that would drop bytes).
- A JSON-RPC write issued while the reader is parked is flushed immediately; one issued while the reader is blocked waiting for input is flushed as soon as the peer sends anything, and an answer to a server-to-client request is always flushed because the reader parks after handing such a request to the application.
- `JsonRpcClient::close` aborts the reader within `READER_STOP_TIMEOUT`, fails pending requests with `the {label} CLI closed its JSON-RPC stream`, drops the transport and is idempotent; the reader holds only a `Weak` handle, so dropping the last client handle aborts it too.
- End of file fails every pending request with `the {label} CLI exited before answering '{method}'` (plus the stderr tail), closes the incoming channel and rejects later writes with the closed message; a malformed line is logged and skipped, and only 32 consecutive unreadable lines declare the stream unusable.
- `VendorHarness` holds one `start`/`stop` cycle per handle and serializes the two on a `tokio::sync::Mutex`; `start` requires `Terminated`, validates the host with `card.validate_host`, bounds `driver.open` at `DEFAULT_OPEN_TIMEOUT` (30 s, overridable with `with_open_timeout`; a timeout is `Protocol("the {label} harness did not open within {secs}s")`), and only then publishes exactly one `StateChanged(Terminated→Idle)`. Every failed `start` path (open error, open timeout, refused provider-session switch) closes the driver best-effort under `CLOSE_TIMEOUT` and releases the start lock, so `stop` and a later `start` stay reachable. `stop` is idempotent, bounds `driver.close` at 10 s (`CLOSE_TIMEOUT`, timeout → `State("the {label} harness did not stop cleanly")` while teardown continues), bounds the supervisor join at 5 s (`SUPERVISOR_TIMEOUT`) before aborting it, and returns the close error after the full teardown.
- Exactly one supervisor task per harness (spawned on demand, stored on the handle) pops the pending turns and runs them strictly one at a time; `send` only enqueues and returns the receipt, minting `turn-<uuid>`/`message-<uuid>`.
- Per-turn event sequence: `StateChanged(Idle→Running)`, `TurnLifecycle(Started)` with `correlation_id = message_id`, the driver's own observations, one terminal `TurnLifecycle` whose kind matches the result status (`Completed→Finished`, `Interrupted→Aborted`, `Failed→Failed`), then `StateChanged(Running→Idle)`.
- A driver *error* from `run_turn` becomes a `Failed` terminal event carrying `turn_error(error.message(), code_from_kind, "protocol", false)`; a driver *result* keeps its own normalized `TurnError`. Either way the harness stays usable for the next turn.
- Capability gating happens before the driver is consulted: `Steer`, `GracefulAbort`, `ForceAbort`, `PauseResume` and `Checkpoint` are all checked on the card. An abort with no active turn drains the queue as `Started`+`Aborted(HarnessStop)` pairs and ignores the requested mode; `DeliveryMode::Auto` is reported as `FollowUp` because the supervisor never starts a turn synchronously.
- The provider session id is sticky: the id of the first successful `start` is reported forever, and a later cycle reporting a different id is `HarnessError::State`. `resume` settles in `Running` while a turn is active and in `Idle` otherwise; `Pausing`/`Resuming` are never resting states.
- All protocol state lives behind one `std::sync::Mutex` and no guard is held across an `.await`. A driver that publishes during `open` emits before the `Terminated→Idle` event, so that event is only the first one for a quiet driver.

## Routing Table

- `./binary.rs` → executable resolution and its actionable failures
- `./jsonl.rs` → pure NDJSON parse/encode and typed object accessors
- `./child.rs` → one spawned CLI process: piped NDJSON stdin/stdout, bounded stderr tail, bounded waits
- `./rpc.rs` → the JSON-RPC 2.0 client (`RpcTransport`, `JsonRpcClient`), its reader task, the write queue and the pending-request map
- `./rpc/tests.rs` → unit tests over an in-memory `RpcTransport` peer: out-of-order correlation, notification ordering, server requests, peer errors, timeouts, end of stream, malformed lines, idempotent close
- `./events.rs` → the observation bus, its single-consumer cursor/turn filter, and the envelope builder
- `./permissions.rs` → host permission policy and the approval/user-input translation
- `./session.rs` → deterministic session ids, resume-policy resolution, checkpoint field access
- `./driver.rs` → the `VendorDriver` seam an adapter implements, `TurnRequest`/`TurnContext` observation helpers, and the `completed_turn`/`interrupted_turn`/`failed_turn`/`turn_error` constructors
- `./harness.rs` → `VendorHarness`: the generic `HarnessProtocol` shell, its lifecycle locks (bounded driver `open`/`close`), turn queue and single supervisor task
