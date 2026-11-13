# abilities/

## Intent

Per-agent tool registry and tool execution.
Ports Python `core/single_agent/ability_manager.py` minus the `Runner.resource_mgr` resolution: the manager owns `Arc<dyn Tool>` instances directly.

## API Surface

- `mod.rs` — `AbilityManager` (`new(owner_id)`, `add_tool`, `remove_tool`, `get`, `cards`, `tool_info`, `revision`, `reorder`, `len`, `is_empty`, `execute`), `AddAbilityResult { name, added, reason }` (`added_tool | refreshed_tool | duplicate_tool`), `ToolExecutionOptions`, `ToolCallResult { call, output, text, success, error }` with `ToolCallResult::message()`.
- `arguments.rs` — `parse_tool_arguments(&str) -> Result<Value>`: empty input is `{}`; truncated JSON is repaired (bracket/quote balancing); unrepairable input is `Error::Tool`.
- `execute.rs` — execution policy: model order, contiguous `parallel_safe` batches run concurrently (`futures::future::join_all`), non-parallel-safe tools act as barriers, per-call `tokio::time::timeout`, cancellation via `CancellationToken`; failures/timeouts/unknown tools become error results, never `Err`.
- `execute.rs` — `execute_calls` delegates to `execute_calls_with(tools, calls, options, &ToolResiliencePolicy)`; the policy's `for_tool(name)` supplies a per-tool timeout (overrides `ToolExecutionOptions::timeout`) and an attempt budget, retrying tool errors/timeouts with `backoff_for(attempt)` and never retrying `Cancelled`/`Validation` failures.

## Constraints

- No rails here: the ReAct loop fires `BeforeToolCall`/`AfterToolCall` around `execute`.
- `tool_info()` order equals registration order (the model-visible tool order), `cards()` returns clones.
- `revision` increments only on successful mutations; `reorder` keeps unknown names out.
- Registry is keyed by tool name; a duplicate name with the same `ToolCard.id` is a refresh, a different id is rejected.
- Rendering for the LLM uses `Tool::render_for_llm`.

## Test Strategy

Inline unit tests: registration/refresh/duplicate/removal, revision bumping, reorder, argument repair cases, parallel vs barrier ordering (deterministic tools recording invocation order), timeout and unknown-tool error results.

## Routing Table

- `./mod.rs` → registry types
- `./arguments.rs` → argument parsing/repair
- `./execute.rs` → execution engine
