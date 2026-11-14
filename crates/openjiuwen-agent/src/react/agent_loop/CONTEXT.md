# agent_loop/

## Intent

One invoke of the `ReAct` loop: the owned runtime snapshot, interruption load/resume, the iteration loop, rails, and termination.
Ports Python `react_agent.py`'s `_inner_invoke`/`_finish_max_iterations` control flow, split out of `react/` to keep each file under the crate's ~1000-line guideline.

## API Surface

- `mod.rs` — `AgentRuntime` (owned snapshot: card id/name, config, model, request config, rails, context engine, tool map, callback framework, steering queue, exposure policy, MCP allowlist, client-tool registry, resilience policy, optional final check, optional KV-cache hook, projected `visible_tool_info`/`visible_cards`), `snapshot`, `run`, `run_with`, `run_inner`, `request_for`, `reset_ctx`, `with_progress`, and the `#[cfg(test)] mod tests;` declaration.
- `kv_cache.rs` — the loop's only KV-cache touch point: `prepare(runtime, session, messages, request) -> Option<KvCacheCall>` (identity, `prepare`, `begin_inference`, `affinity`) and `finish(Option<KvCacheCall>, succeeded, Option<u64>)` (`end_inference`, `note_usage` on success); both are exported to `mod.rs` only as `pub(super)`, and `None`/no-hook runs do no work.
- `tests.rs` — the module's unit tests (tool call → answer, empty-response retry, iteration limit, `FinalAnswer` policy, confirm pause/resume, ask-user answer, multimodal parts, memory variables reaching every model call, `request_for`, `reset_ctx`, `with_progress`).
`kv_cache.rs` carries its own two unit tests (`finish` with no call is a no-op; usage is recorded only for a successful call).

## Constraints

- The module path stays `crate::react::agent_loop`; `mod.rs` re-exports nothing beyond what it defines, matching the previous single-file module.
- `tests.rs` is a plain child module, so `use super::*;` resolves to the items in `mod.rs`.
- No behaviour lives only here that callers reach without going through `run`/`run_with`.
