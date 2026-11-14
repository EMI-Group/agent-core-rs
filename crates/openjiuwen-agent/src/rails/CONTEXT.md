# rails/

## Intent

Agent rails: the hook trait, per-event context, priority-ordered registry, and the built-in rails that need no upper layer.
Ports the hook model of Python `core/single_agent/rail/base.py` (`AgentRail`, `AgentCallbackContext`, event firing, force-finish/retry control) and the rails the kernel can host on its own.

## API Surface

- `mod.rs` — `RailEvent`, `RailContext` (per-event data + control flags: `force_finish`, `model_continue`, `retry`, `skip_tool`, `interrupts`, `model_override`), `AgentRail` (name, priority=50, async hooks `before_invoke`, `after_invoke`, `before_model_call`, `after_model_call`, `on_model_exception`, `before_tool_call`, `after_tool_call`, `on_tool_exception`, `after_react_iteration`), `RailRegistry` (`register`, `unregister`, `rails`, `fire`), `RetryRequest`, `DEFAULT_RAIL_PRIORITY`.
- `guardrail.rs` — `GuardrailRail` over `openjiuwen_security::GuardrailPipeline`: user input on `BeforeInvoke`, model output on `AfterModelCall`, tool input/output on `BeforeToolCall`/`AfterToolCall`, gated by `ScanTargets` flags; blocking returns `Error::Security` naming the stage, risk type, and level (from `GuardrailOutcome::enforce` + `violation`).
- `guardrail.rs` keeps the `prompts` flag for `ScanTargets` parity but never scans the prepared window; user and tool content are scanned at their own stages.
- `memory.rs` — `MemoryRail` over `openjiuwen_memory::LongTermMemory`: prompt variables on `BeforeInvoke`, recall injected as a system message after the leading system messages before the model call (at most once per invoke, flagged in `ctx.extra`), message write on `AfterInvoke`; `with_top_k`/`top_k` control recall size (default 5); failures are logged only.
- `BeforeInvoke` publishes `ctx.extra["memory_variables"]` (constant `MEMORY_VARIABLES_KEY`) only when the invocation's `AgentInput::user_id` is non-empty, so without a user id both placeholders stay verbatim.
- That object holds at most two string→string keys: `sys_memory_variables` (JSON object of the user's variables filtered to `AgentMemoryConfig::mem_variables`; absent when nothing is stored or the fetch fails, `"{}"` when every stored variable is filtered out) and `sys_long_term_memory` (JSON array; `"[]"` when nothing matches or a search fails; present whenever `enable_long_term_mem` is on).
- `sys_long_term_memory` starts with the literal `用户画像记忆：` followed by the fragment hits when `enable_user_profile || enable_semantic_memory || enable_episodic_memory`, then `摘要记忆：` followed by the summary hits when `enable_summary_memory`; the searches request 10 and 5 hits at threshold `0.3`, matching Python.
- The lookups use the invocation's `user_id` over the rail's `scope_id`; the recall-message mechanism keeps using the rail's own `scope`, and both mechanisms coexist in one invocation.
- The loop renders these keys into the system prompt; unknown placeholders stay literal and a value does not have to be valid JSON.
- The recall lands in `ctx.messages`, and the loop sends that rail-mutated window (not the freshly prepared one) to the model.
- `model_backup.rs` — `ModelBackupRail`: on `OnModelException`, advance a per-rail cursor to the next backup model via `ctx.model_override` and `ctx.request_retry(None)`; exhausted cursor leaves the context untouched.
- `interrupt/mod.rs` — `ToolFilter`, `BaseInterruptRail`, `ConfirmInterruptRail`, `AskUserRail` (see `interrupt/CONTEXT.md`).

## Constraints

- Hook dispatch is by `ctx.event` with a stable descending-priority order; `fire` stops at the first `Err`, which aborts the run.
- A rail never mutates the agent directly; control flows through `RailContext`.
- `RailContext.extra` persists for the whole invoke (Python `ctx.extra`) and is where resume answers/auto-confirm flags are seeded by the loop.
- Built-in rails must be constructible without a `Runner` and must not depend on harness crates.

## Test Strategy

- Unit tests beside each module: registry dispatch order, first-error stop, per-event hook targeting, guardrail stage blocking, memory prompt-variable filtering/absence/labels, memory recall injection/write-back, model-backup ordering and exhaustion, interrupt pause/approve/reject/ask-user/auto-confirm.
- `../../tests/rails.rs` proves the hooks are wired into the running loop: priority ordering for every event, force-finish before the model and before a tool, `ModelBackupRail` failover, `GuardrailRail` blocking (`ErrorKind::Security`), `MemoryRail` recall reaching the model plus write-back, and `skip_tool` replacing a result.

## Routing Table

- `./mod.rs` → trait, context, registry
- `./guardrail.rs` → security pipeline rail
- `./memory.rs` → long-term memory rail
- `./model_backup.rs` → model failover rail
- `./interrupt/` → interrupt/HITL rails
