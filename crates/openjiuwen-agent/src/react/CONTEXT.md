# react/

## Intent

The ReAct agent: configuration, the model → tools → observation loop, and the streaming model step.
Ports Python `core/single_agent/agents/react_agent.py` (`ReActAgent`, `ReActAgentConfig`, `_inner_invoke`, `_railed_model_call`, `_finish_max_iterations`) without the legacy controller stack.

## API Surface

- `mod.rs` — `ReActAgent` (card + interior-mutable config/model/abilities/rails/prompt builder/context engine), `ReActAgent::configure/set_model/model/add_tool/add_tools/remove_tool/tools/register_rail/unregister_rail/rails/prompt_builder/context_engine`, `impl Agent`.
- `Agent::invoke` parses the payload, resolves the session (`run::resolve_session`), runs the loop, and finishes an owned session; `Agent::stream` snapshots the agent, spawns the loop, and forwards frames through an unbounded channel (one terminal `answer` frame, or one terminal `Err` for invalid input, an unconfigured model, or cancellation).
- `config.rs` — `ReActAgentConfig` with Python field names/defaults plus `MaxIterationsPolicy` (`Error | FinalAnswer | ReturnObservations`, wire `error|final_answer|return_observations`).
- `agent_loop.rs` — `AgentRuntime::snapshot(&ReActAgent)`, `run(agent, ..)`, `run_with(&AgentRuntime, ..)`, `run_inner(..)`, and the pause helper `pause_output(..)`.
- `model_step.rs` — a single model call: `call_model` (non-streaming, used by the `FinalAnswer` policy) and `call_model_streaming` driving `Model::stream` through a `StreamAccumulator`, emitting `llm_reasoning`/`llm_output`/`llm_usage` frames through `DeltaSink`, and building `ModelReply { message, finish_reason, usage, reasoning_present }`.
- `resume.rs` — `replay_interrupted(state, interactive, abilities, options)` replays interrupted tool calls: approvals re-execute via `AbilityManager`, rejections carry `reject_feedback`, ask-user answers become the result text, and unmatched requests are returned for re-pausing.
- `prompt.rs` — `build_identity` renders the system-role template; `build_system` wraps it in one system message.

## Loop Semantics

- `AgentRuntime` is a synchronous owned snapshot (card id/name, config, model, request config, rails, context engine, tool map, ability manager, tool schemas, tool cards); no `parking_lot` guard is ever held across an `.await`.
- `run_with` opens one root span (`InvokeType::Chain`), builds one `RailContext` for the run, and always finalizes: `AfterInvoke` rails (errors logged), `ContextEngine::save_contexts`, the terminal `answer` frame on success, and the span status; a failing run leaves the terminal frame to the streaming caller so the stream never carries two terminal frames.
- `run_inner` creates the context (`"main"`), fires `BeforeInvoke`, then either replays a persisted `InterruptionState` (clearing it, seeding `_interrupt_auto_confirm` and `_resume_interrupt_answers` into `ctx.extra`, observing one tool message per replayed result, and starting at `state.iteration + 1`) or observes the user query and starts at `1`; an interactive payload with no pending interruption is `Error::Validation`.
- The effective limit is `max_iterations.max(1)` lowered by `_host_max_iterations` and `max_iterations_hard_limit`; each iteration prepares the window (`ContextEngine::prepare` with the rendered system prompt and tool schemas), fires `BeforeModelCall`, streams the model call (at most two attempts when a rail requests a retry after `OnModelException`), fires `AfterModelCall`, and observes the assistant message.
- No tool calls: `model_continue` keeps iterating, empty text retries once after a notice (`_openjiuwen_message_source_kind = "empty_response"`) and then yields `ResultType::Error` with `模型未返回有效内容（空响应），请重试或检查上下文。`, and non-empty text yields the answer with finish reason, reasoning flag, iteration count, and observations.
- Tool calls: `BeforeToolCall` fires per call in model order (force-finish, interrupt pause, and `skip_tool` are honoured), the queued calls run in one `execute_calls` batch, then `AfterToolCall` fires per result, each tool message is observed, and observations are recorded in model order.
- Termination: exceeding `max_total_tokens` yields `termination_reason = "token_budget"`; loop exhaustion yields `termination_reason = "max_iterations"`, `truncated_by_max_iterations = true`, and applies `on_max_iterations` (`FinalAnswer` makes one tool-free `call_model` and falls back to `finalization_error`).
- A pause saves `InterruptionState` (requests keyed by tool-call id, answers, assistant message, iteration, original query) and returns `AgentOutput::interrupt` with `__interaction__` frames delivered to the publisher and mirrored into the session.
- `BeforeInvoke`-time `force_finish` is returned untouched; later force-finishes get `iterations_used` and observations attached.

## Constraints

- The loop never constructs tools or models itself: tools come from the snapshot, the model from `ReActAgent::model()`.
- Rails are fired through `rails::RailRegistry`; every firing point resets the event-specific fields (`reset_ctx`) while `extra`/`session`/`agent_id`/`input`/`iteration` survive.
- Model calls use `ContextEngine::prepare` for the window; assistant/tool messages are appended with `ContextEngine::observe`.
- `stream` always runs the loop in a spawned task and mirrors frames into the session; `invoke` passes no publisher.
- Files ≤1000 lines: loop orchestration and model-call mechanics stay in separate files.

## Test Strategy

Behaviour tests in `agent_loop::tests` (tool call → answer, empty-response retry, iteration limit, `FinalAnswer` policy, confirm pause/resume, ask-user answer) and `mod.rs` tests (input/model errors, invoke answer, streaming frames) with `ScriptedModel`.
Cross-crate ReAct suites live in `../../tests/` and are maintained by the integration-test owner.

## Routing Table

- `./config.rs` → configuration type and iteration policy
- `./agent_loop.rs` → runtime snapshot and control flow
- `./model_step.rs` → model invocation and stream accumulation
- `./resume.rs` → interrupt resume replay
- `./prompt.rs` → system prompt assembly
