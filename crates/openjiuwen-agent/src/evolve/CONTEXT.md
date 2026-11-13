# evolve

## Intent

`ReActAgentEvolve`, the self-evolution variant of `ReActAgent` that ports `openjiuwen/core/single_agent/agents/react_agent_evolve.py`.
Execution is fully inherited: the type owns an `Arc<crate::react::ReActAgent>` and implements `crate::agent::Agent` by delegation.
The module also owns the operator synchronization: parameter updates on the `react_llm` / `react_tool` operators are pushed back into the live agent through `ParameterCallback`s.

## API Surface

- `ReActAgentEvolve` — `new(card)`, `react()`, `configure(config) -> &Self`, `set_model`, `add_tool`/`add_tools`/`remove_tool`, `register_rail`, `set_client_tools`, `set_callback_framework`.
- `ReActAgentEvolve::operators()` / `get_operators()` — type-erased handles keyed by `operator_id`; `llm_operator()` / `tool_operator()` return the boxed handles and `llm_call_operator()` / `tool_call_operator()` the typed `Arc<RwLock<…>>` handles over the same state.
- `SharedOperator = Arc<RwLock<Box<dyn Operator>>>`, `SharedLlmOperator`, `SharedToolOperator` — the shared handle aliases.
- `LLM_OPERATOR_ID` (`react_llm`) and `TOOL_OPERATOR_ID` (`react_tool`) — the Python operator ids.
- `DescriptionOverrideTool` — a `Tool` view whose `ToolCard` description is replaced while `invoke`/`stream`/`render_for_llm`/`is_parallel_safe` delegate to the wrapped tool.

## Constraints

- Only public crate APIs are used; `src/react/`, `src/operator/`, and other modules are read-only consumers.
- `system_prompt` updates rewrite `prompt_template` through `ReActAgent::configure` with the rest of `config()` copied unchanged.
- `tool_description` updates take every mounted tool out (capturing card order), re-wrap only the changed tools, and mount them back in the original order; every card field except the description is preserved.
- Python mutates `ToolCard.description` in place; the Rust port swaps in `DescriptionOverrideTool` because `ReActAgent` exposes no card mutation.
- `add_tool`/`add_tools` do not refresh the tool operator, matching Python; a later `configure` re-seeds it.
- Tests are inline, offline, and driven by `openjiuwen_test_support::ScriptedModel` plus `FunctionTool` fixtures.

## Routing Table

- `./mod.rs` → module docs and re-exports.
- `./agent.rs` → `ReActAgentEvolve`, operator construction, parameter callbacks, tool-description synchronization, unit tests.
- `./description_override.rs` → `DescriptionOverrideTool`.
