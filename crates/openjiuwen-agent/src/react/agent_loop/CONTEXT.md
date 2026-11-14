# agent_loop/

## Intent

One invoke of the `ReAct` loop: the owned runtime snapshot, interruption load/resume, the iteration loop, rails, and termination.
Ports Python `react_agent.py`'s `_inner_invoke`/`_finish_max_iterations` control flow, split out of `react/` to keep each file under the crate's ~1000-line guideline.

## API Surface

- `mod.rs` — `AgentRuntime` (owned snapshot: card id/name, config, model, request config, rails, context engine, tool map, callback framework, steering queue, exposure policy, MCP allowlist, client-tool registry, resilience policy, optional final check, projected `visible_tool_info`/`visible_cards`), `snapshot`, `run`, `run_with`, `run_inner`, `request_for`, `reset_ctx`, `with_progress`, and the `#[cfg(test)] mod tests;` declaration.
- `tests.rs` — the module's unit tests (tool call → answer, empty-response retry, iteration limit, `FinalAnswer` policy, confirm pause/resume, ask-user answer, multimodal parts, memory variables reaching every model call, `request_for`, `reset_ctx`, `with_progress`).

## Constraints

- The module path stays `crate::react::agent_loop`; `mod.rs` re-exports nothing beyond what it defines, matching the previous single-file module.
- `tests.rs` is a plain child module, so `use super::*;` resolves to the items in `mod.rs`.
- No behaviour lives only here that callers reach without going through `run`/`run_with`.
