# policy/

## Intent

Runtime tool policies that Python keeps next to `AbilityManager`: which tools the model may see, MCP tool allowlists, handlers for tools the connected client executes, per-tool resilience budgets, and the check run on a round's final answer.
The module is data plus decision helpers; the `ReAct` loop owns every wiring point (see `react/hooks.rs`, `react/exposure.rs`, `react/resume.rs`), and no policy type calls a model or a tool itself.

## API Surface

- `mod.rs` — re-exports `ClientToolHandler`, `ClientToolRegistry`, `FnClientToolHandler`, `McpToolAllowlist`, `ToolExposureMode`, `ToolExposurePolicy`, `ALL_TOOLS`, `DefaultFinalRequestCheck`, `FinalRequestCheck`, `FinalRequestContext`, `FinalRequestDecision`, `ToolResilience`, `ToolResiliencePolicy`.
- `exposure.rs` — `ToolExposureMode` (allow-all / allowlist / denylist), `ToolExposurePolicy::{allow_all, deny_all, allowlist, allow, deny, with_hide_deferred, is_visible, resolve, filter_cards, filter_tool_info}`; `McpToolAllowlist::{new, allow, allow_all, is_empty, allows, tools}`. `filter_tool_info` keeps `tool_info` and `cards` aligned by one keep-mask. `ALL_TOOLS` (`"*"`) allows every MCP tool of a server.
- `client_tools.rs` — `ClientToolHandler` (`name()`, `async invoke(&ToolCall, &Value) -> Result<Value>`), `FnClientToolHandler::new(name, closure)`, `ClientToolRegistry::{new, register, unregister, is_client_tool, handler, names, invoke}`; `invoke` looks the handler up by tool name and returns `Error::NotFound` when unregistered.
- `resilience.rs` — `ToolResilience` (`new`, `with_timeout`, `with_attempts`, `with_backoff`, `attempts()`, `backoff_for(attempt) -> Option<Duration>`), `ToolResiliencePolicy::{new, set_default, set, for_tool, is_empty}`; defaults are no timeout, one attempt, no backoff.
- `final_check.rs` — `FinalRequestContext::{new, agent_id, iteration, output_text, messages, tools}`, `FinalRequestDecision { Allow, Reject { reason, inject }, Finish(Box<AgentOutput>) }`, `FinalRequestCheck::check(&FinalRequestContext)`, `DefaultFinalRequestCheck::{new, with_inject, inject}` (rejects an empty/whitespace answer with a notice that asks for a real answer or a tool call).

## Constraints

- Policies are pure: no I/O beyond the injected client-tool handler, no logging, no panics on user input.
- `ToolExposurePolicy` only filters what the model sees; execution always uses the full tool map, so hiding a tool never breaks a call the model makes anyway.
- `ToolResiliencePolicy::for_tool` returns a complete budget (policy entry, else default), never `Option`.
- `FinalRequestCheck` is installed as `Arc<dyn FinalRequestCheck>` on `ReActAgent` (default: absent, so behavior is unchanged).
- Files stay well under 500 lines; each policy is independent of the others.

## Notes for Agents

- The loop applies the check when a model reply carries no tool calls, i.e. once the round's answer text is known (`react/hooks.rs::check_final_request`). Python's `FinalRequestCheck` runs before every model call; this crate documents the placement deviation because `FinalRequestContext::output_text` only has a meaningful value after a reply.
- `Reject { inject }` observes `inject` (or `reason`) as a `user` message and runs another iteration while the budget remains; a rejection at the iteration limit falls through to the normal answer.
- MCP tool names are scoped from `ToolCard.metadata` (`server_name`/`server_id`/`mcp_server`, as `load_mcp_tools` writes); `react/exposure.rs` strips an optional `mcp_<server>_` prefix before consulting the allowlist.
- A tool card with `execution == ToolExecution::Client` is treated as a client tool even when the registry has no handler for it; the resume then falls back to a failed result.

## Test Strategy

Unit tests beside each policy file cover the decision tables: exposure masks (allow-all, allowlist, denylist, deferred hiding, aligned tool info), MCP allowlist wildcards, registry registration/lookup/`NotFound`, resilience budgets and backoff schedules, and the default check's accept/reject split.
Loop-level behaviour (a hidden tool missing from the request, retries succeeding then exhausting, a client-tool pause and resume, reject/finish outcomes) is proven by `react/wiring/`.

## Routing Table

- `./exposure.rs` → exposure modes and MCP allowlists
- `./client_tools.rs` → client-tool handlers and registry
- `./resilience.rs` → per-tool timeout/attempt/backoff budgets
- `./final_check.rs` → final-answer check trait, context, decision
