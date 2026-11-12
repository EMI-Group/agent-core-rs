# react/

## Intent

The ReAct agent: configuration, the model → tools → observation loop, and the streaming model step.
Ports Python `core/single_agent/agents/react_agent.py` (`ReActAgent`, `ReActAgentConfig`, `_inner_invoke`, `_railed_model_call`, `_finish_max_iterations`) without the legacy controller stack.

## API Surface

- `mod.rs` — `ReActAgent` (card + interior-mutable config/model/abilities/rails/prompt builder/context engine), `ReActAgent::configure/set_model/model/add_tool/add_tools/remove_tool/tools/register_rail/unregister_rail/rails/prompt_builder`, `impl Agent`.
- `config.rs` — `ReActAgentConfig` with Python field names/defaults plus `MaxIterationsPolicy` (`Error | FinalAnswer | ReturnObservations`, wire `error|final_answer|return_observations`).
- `agent_loop.rs` — one invoke: session resolution, interruption load/resume, iteration loop, rails firing, max-iteration/token-budget termination, context persistence, span tracing.
- `model_step.rs` — a single model call: window preparation, `invoke` vs `stream` (accumulate `StreamAccumulator`, emit `llm_reasoning`/`llm_output`/`llm_usage` chunks), `ModelReply { message, finish_reason, usage, reasoning_present }`, model-error recovery hook.
- `resume.rs` — replay of interrupted tool calls with `InteractiveInput` answers, producing `RESUME_START_ITERATION`.
- `prompt.rs` — identity section rendering from `prompt_template` and the run input variables.

## Constraints

- The loop never constructs tools or models itself: tools come from `AbilityManager`, the model from `ReActAgent::model()`.
- Rails are fired through `rails::RailRegistry`; `RailContext.force_finish` short-circuits the loop, `retry` retries the failing step once per request, `interrupts` pause the run.
- Tool rails fire around each call in model order; execution itself is `abilities::AbilityManager::execute`.
- Model calls use `ContextEngine::prepare` for the window; the assistant/tool messages are appended with `ContextEngine::observe`.
- `invoke` returns `AgentOutput`; `stream` yields `AgentChunk`s and mirrors frames into the session when present.
- Files ≤1000 lines: loop orchestration and model-call mechanics stay in separate files.

## Test Strategy

Behaviour tests in `../../tests/react_agent.rs` and `../../tests/streaming.rs` with `ScriptedModel`:
no-tool answer, single tool, parallel tools, tool error, max-iteration policies, empty-response retry, window/prompt rendering, streaming frame assembly.

## Routing Table

- `./config.rs` → configuration type and iteration policy
- `./agent_loop.rs` → control flow
- `./model_step.rs` → model invocation and stream accumulation
- `./resume.rs` → interrupt resume replay
- `./prompt.rs` → system prompt assembly
