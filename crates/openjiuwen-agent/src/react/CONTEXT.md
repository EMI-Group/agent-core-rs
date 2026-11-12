# react/

## Intent

The ReAct agent: configuration, the model → tools → observation loop, and the streaming model step.
Ports Python `core/single_agent/agents/react_agent.py` (`ReActAgent`, `ReActAgentConfig`, `_inner_invoke`, `_railed_model_call`, `_finish_max_iterations`) without the legacy controller stack.

## API Surface

- `mod.rs` — `ReActAgent` (card + interior-mutable config/model/abilities/rails/prompt builder/context engine), `ReActAgent::configure/set_model/model/add_tool/add_tools/remove_tool/tools/register_rail/unregister_rail/rails/prompt_builder/context_engine`, `impl Agent`.
- `Agent::invoke` parses the payload, resolves the session (`run::resolve_session`), runs the loop, and finishes an owned session; `Agent::stream` snapshots the agent, spawns the loop, and forwards frames through an unbounded channel (one terminal `answer` frame, or one terminal `Err` for invalid input, an unconfigured model, or cancellation).
- `config.rs` — `ReActAgentConfig` with Python field names/defaults plus `MaxIterationsPolicy` (`Error | FinalAnswer | ReturnObservations`, wire `error|final_answer|return_observations`).
- `agent_loop.rs` — `AgentRuntime::snapshot(&ReActAgent)`, `run(agent, ..)`, `run_with(&AgentRuntime, ..)`, `run_inner(..)`, and the pause helper `pause_output(..)`.
- `model_step.rs` — a single model call: `call_model` (non-streaming, used only by the tool-free `FinalAnswer` finalization) and `call_model_streaming` driving `Model::stream` through a `StreamAccumulator`, emitting `llm_reasoning`/`llm_output`/`llm_usage` frames through `DeltaSink`, and building `ModelReply { message, finish_reason, usage, reasoning_present }`.
- `resume.rs` — `replay_interrupted(state, interactive, abilities, options)` replays interrupted tool calls: approvals re-execute via `AbilityManager`, rejections carry `reject_feedback`, ask-user answers become the result text, and unmatched requests are returned for re-pausing.
- `prompt.rs` — `build_identity` renders the system-role template; `build_system` wraps it in one system message.

## Loop Semantics

- `AgentRuntime` is a synchronous owned snapshot (card id/name, config, model, request config, rails, context engine, tool map, ability manager, tool schemas, tool cards); no `parking_lot` guard is ever held across an `.await`.
- `run_with` opens one root span (`InvokeType::Chain`), builds one `RailContext` for the run, and always finalizes: `AfterInvoke` rails (errors logged), `ContextEngine::save_contexts`, the terminal `answer` frame on success, and the span status; a failing run leaves the terminal frame to the streaming caller so the stream never carries two terminal frames.
- `run_inner` creates the context (`"main"`), fires `BeforeInvoke`, then either replays a persisted `InterruptionState` (clearing it, seeding `_resume_interrupt_answers` into `ctx.extra`, observing one tool message per replayed result, and starting at `state.iteration + 1`) or observes the user query and starts at `1`; an interactive payload with no pending interruption is `Error::Validation`.
- Auto-confirm is seeded from the session's `__interrupt_auto_confirm__` on every invoke — the fresh branch seeds it before observing the query, the resumed branch before replaying.
- Replay happens without a model call; requests the caller did not answer re-pause the run with only the remaining requests, reusing their `interrupt_id`s.
- The effective limit is `max_iterations.max(1)` lowered by `_host_max_iterations` and `max_iterations_hard_limit`; each iteration prepares the window (`ContextEngine::prepare` with the rendered system prompt and tool schemas), fires `BeforeModelCall`, streams the model call (at most two attempts when a rail requests a retry after `OnModelException`), fires `AfterModelCall`, and observes the assistant message.
- No tool calls: `model_continue` keeps iterating, empty text retries once after a notice (`_openjiuwen_message_source_kind = "empty_response"`) and then yields `ResultType::Error` with `模型未返回有效内容（空响应），请重试或检查上下文。`, and non-empty text yields the answer with finish reason, reasoning flag, iteration count, and observations.
- Tool calls: `BeforeToolCall` fires for every call in model order and each call's requests are drained immediately (force-finish, interrupt requests, and `skip_tool` are honoured); a `skip_tool` result is substituted without executing, so `BeforeToolCall` is the only hook that can skip a call.
- When any call of the turn recorded a request, no tool of that turn runs and the run pauses once with all requests collected so far; otherwise the queued calls run in one `execute_calls` batch, then `AfterToolCall` fires per result in model order, each tool message is observed, and observations are recorded in model order.
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

- Unit tests beside the modules: `agent_loop::tests` (tool call → answer, empty-response retry, iteration limit, `FinalAnswer` policy, confirm pause/resume, ask-user answer, `request_for`, `reset_ctx`, `with_progress`), `model_step::tests` (non-streaming reply, provider error passthrough, streaming content/reasoning/usage frames and their index sequence), `mod.rs` (input/model errors, invoke answer, streaming frames), `resume.rs`, `prompt.rs`, and `config.rs`, all on `ScriptedModel`.
- `../../tests/react_agent.rs` covers the loop end to end: no-tool answer, single/parallel/failing tools, the three max-iteration policies, empty-response retry, prompt rendering, and input/config errors.
- `../../tests/streaming.rs` covers frame assembly plus session mirroring, and `../../tests/interrupt.rs` covers the pause/resume flows end to end.

## Routing Table

- `./config.rs` → configuration type and iteration policy
- `./agent_loop.rs` → runtime snapshot and control flow
- `./model_step.rs` → model invocation and stream accumulation
- `./resume.rs` → interrupt resume replay
- `./prompt.rs` → system prompt assembly
