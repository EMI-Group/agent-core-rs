# rails/

## Intent

Agent rails: the hook trait, per-event context, priority-ordered registry, and the built-in rails that need no upper layer.
Ports the hook model of Python `core/single_agent/rail/base.py` (`AgentRail`, `AgentCallbackContext`, event firing, force-finish/retry control) and the rails the kernel can host on its own.

## API Surface

- `mod.rs` — `RailEvent`, `RailContext` (per-event data + control flags: `force_finish`, `model_continue`, `retry`, `skip_tool`, `interrupts`, `model_override`), `AgentRail` (name, priority=50, async hooks `before_invoke`, `after_invoke`, `before_model_call`, `after_model_call`, `on_model_exception`, `before_tool_call`, `after_tool_call`, `on_tool_exception`, `after_react_iteration`), `RailRegistry` (`register`, `unregister`, `rails`, `fire`), `RetryRequest`, `DEFAULT_RAIL_PRIORITY`.
- `guardrail.rs` — `GuardrailRail` over `openjiuwen_security::GuardrailPipeline`: user input on `BeforeInvoke`, model output on `AfterModelCall`, tool input/output on `BeforeToolCall`/`AfterToolCall`; blocking returns `Error::Security` (from `GuardrailOutcome::enforce`).
- `memory.rs` — `MemoryRail` over `openjiuwen_memory::LongTermMemory`: recall injected as a system message before the model call, message write on `AfterInvoke`; failures are logged only.
- `model_backup.rs` — `ModelBackupRail`: on `OnModelException`, advance to the next backup model via `ctx.model_override` and `ctx.request_retry`.
- `interrupt/mod.rs` — `ToolFilter`, `BaseInterruptRail`, `ConfirmInterruptRail`, `AskUserRail` (see `interrupt/CONTEXT.md`).

## Constraints

- Hook dispatch is by `ctx.event` with a stable descending-priority order; `fire` stops at the first `Err`, which aborts the run.
- A rail never mutates the agent directly; control flows through `RailContext`.
- `RailContext.extra` persists for the whole invoke (Python `ctx.extra`) and is where resume answers/auto-confirm flags are seeded by the loop.
- Built-in rails must be constructible without a `Runner` and must not depend on harness crates.

## Test Strategy

`../../tests/rails.rs`: registration order (priority desc, ties keep insertion order), force-finish short-circuit on model and tool phases, model retry via `ModelBackupRail`, guardrail blocking, memory rail injection, skip-tool replacement.

## Routing Table

- `./mod.rs` → trait, context, registry
- `./guardrail.rs` → security pipeline rail
- `./memory.rs` → long-term memory rail
- `./model_backup.rs` → model failover rail
- `./interrupt/` → interrupt/HITL rails
