# react/wiring/

## Intent

`#[cfg(test)]` behaviour suites for the six seams wired into the `ReActAgent` loop: lifecycle callbacks, steering, tool exposure/MCP allowlist/resilience, client tools, and the final-request check.
These live under `src/` (rather than the crate's `tests/`) so they can observe crate-internal state (`AgentRuntime`, `hooks`, `exposure`, `pause`) and drive the loop with fake tools and `ScriptedModel` without a network.
Each file is a self-contained scenario module imported by `mod.rs`; nothing here is compiled into the library.

## API Surface

- `mod.rs` — declares the five suites; no logic.
- `callbacks.rs` — a recording `AgentCallback` collects `(event, payload)` pairs: asserts the six-event order and payload contents across a two-iteration run with one tool call, that a failing run still fires `InvokeEnd`, that a callback returning `Err` is logged and does not abort, and that an empty framework short-circuits.
- `steering.rs` — pushes text before and during a run: the message is observed as a `user` message and appears in the next model request; an empty queue leaves the request sequence unchanged.
- `policy.rs` — exposure allowlist/denylist and MCP scoping hide a tool from the model request while an allowed tool still executes; a flaky tool succeeds after the configured retries, exhausts its attempts with a failed result, and a cancelled call is never retried.
- `client_tools.rs` — a model call to a registered client tool pauses with a `client_tool` interrupt; a resume answer runs the handler and feeds the value back as the tool result; a handler error becomes a failed result.
- `final_check.rs` — `Reject { inject }` adds a user message and continues to a real answer, `Finish` returns the supplied output with progress fields, and `DefaultFinalRequestCheck` rejects an empty answer.

## Constraints

- Test-only: these modules may reach `pub(crate)` items but must never leak into the public API.
- Offline and deterministic: `ScriptedModel`, in-test `Tool` impls, no sockets or wall-clock ordering beyond short timeouts.
- Assertions target observable loop behaviour (requests the model saw, frames published, `AgentOutput`), not private data structures, so refactors inside the loop do not force rewrites.
- Files stay well under 500 lines; one concern per file.

## Routing Table

- `./callbacks.rs` → callback event order/payloads and failure policy
- `./steering.rs` → steering drain into the model window
- `./policy.rs` → exposure, MCP allowlist, per-tool resilience
- `./client_tools.rs` → client-tool pause and resume
- `./final_check.rs` → final-request check allow/reject/finish
