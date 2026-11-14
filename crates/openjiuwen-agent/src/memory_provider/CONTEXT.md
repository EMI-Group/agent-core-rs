# memory_provider/
## Intent
Bridge the `openjiuwen-memory` external provider SPI (`MemoryProvider`) into the ReAct agent: one rail for prompt block, prefetch, and turn write-back, plus a `Tool` wrapper for the provider's model-facing tool schemas.
Ports the tool-rail half of Python's memory provider callback; the record SPI itself stays in `openjiuwen-memory`.
## API Surface
- `ProviderMemoryRail` — `AgentRail` over an `Arc<dyn MemoryProvider>`: `new(provider)` (name `"memory_provider"`, `PROVIDER_RAIL_NAME`), `with_name`, `provider()`.
- `ProviderMemoryRail::before_invoke` publishes `MemoryProvider::system_prompt_block` (when non-blank) under `ctx.extra["memory_provider_block"]` (`MEMORY_PROVIDER_BLOCK_KEY`); `injected_prompt_block(ctx)` reads it back and the loop appends it to the identity system message.
- `ProviderMemoryRail::before_model_call` calls `MemoryProvider::prefetch` once per invocation (guarded by `ctx.extra["openjiuwen.rails.memory_provider.prefetched"]`) with the last user message as the query and inserts a non-empty result as a system message directly after the leading system messages.
- `ProviderMemoryRail::after_invoke` calls `MemoryProvider::sync_turn(query, output)` when the output is an `Answer` with non-empty text and the invocation has a text query; a failure is logged with `tracing::warn!` and never aborts the run.
- `ProviderTool` — `openjiuwen_foundation::Tool` bound to one schema: `new(provider, card)`, `from_schema(provider, &Value)`, `provider()`; `invoke` dispatches to `MemoryProvider::handle_tool_call(name, args)` and returns the provider payload verbatim (a `{"error": …}` object is a successful tool result), checking cancellation first.
- `provider_tools(provider) -> Vec<Arc<dyn Tool>>` — one tool per `MemoryProvider::get_tool_schemas()` entry in provider order; `provider_tool_cards(&Arc<dyn MemoryProvider>) -> Vec<ToolCard>` — the same derivation as cards only.
- `finalize_session(provider, turns, owned, output)` / `finalize_turns(turns)` — the crate-internal owned-session `on_session_end` call (skipped for borrowed sessions and interrupt pauses, logged on failure).
## Cards
- `ToolCard::new(function.name, function.description, function.parameters)` with metadata `provider = <provider name>`; a missing description becomes `""` and missing parameters `{}`.
- A schema without an object `function` and a non-blank `name` is skipped with a `tracing::warn!` — never a panic and never an empty placeholder card.
## Constraints
- No new dependencies and no network: the provider is the caller's trait object, so every test drives an in-process fake.
- Best-effort by contract: `prefetch` and `handle_tool_call` are the SPI's fail-soft entry points; only `sync_turn`/`on_session_end` return errors, and both are logged rather than propagated into the run's output.
- The provider is not part of `ReActAgentConfig` (that type derives `Serialize`/`Deserialize`/`PartialEq`); it lives in its own `parking_lot::RwLock<Option<Arc<dyn MemoryProvider>>>` on `ReActAgent` and in the immutable `AgentRuntime` snapshot.
- `set_memory_provider` mounts the rail and the tools even when `MemoryProvider::is_available()` answers `false` (one `tracing::warn!`); `clear_memory_provider` drops the rail and leaves mounted tools in place.
- The rail never executes a tool and the tool never touches the rail; each goes through its own SPI entry point.
## Test Strategy
- `tests.rs` covers the rail and tool surface directly: prompt-block publication, prefetch placement and once-per-invoke guard, blank-query/empty-prefetch no-ops, answer-only write-back, failing `sync_turn`, schema-derived cards (including malformed-schema skipping), tool dispatch, and the `finalize_session` gate.
- `../../tests/memory_provider.rs` drives the whole path through `ReActAgent::invoke` on a `ScriptedModel`.
## Routing Table
- `../react/` → `ReActAgent::set_memory_provider`/`clear_memory_provider`/`end_memory_session` and the loop's `on_session_end` call
- `../rails/` → the `AgentRail`/`RailContext` seam this rail implements
