# react/

## Intent

The ReAct agent: configuration, the model → tools → observation loop, and the streaming model step.
Ports Python `core/single_agent/agents/react_agent.py` (`ReActAgent`, `ReActAgentConfig`, `_inner_invoke`, `_railed_model_call`, `_finish_max_iterations`) without the legacy controller stack.

## API Surface

- `mod.rs` — `ReActAgent` (card + interior-mutable config/model/abilities/rails/prompt builder/context engine/callback framework/steering queue/exposure policy/MCP allowlist/client tools/resilience/final check), `ReActAgent::configure/set_model/model/add_tool/add_tools/remove_tool/tools/register_rail/unregister_rail/rails/prompt_builder/context_engine`, the wiring setters `set_callback_framework/callbacks/register_callback`, `steering/steer`, `set_tool_exposure_policy/tool_exposure_policy`, `set_mcp_tool_allowlist/mcp_tool_allowlist`, `set_client_tools/client_tools`, `set_tool_resilience/tool_resilience`, `set_final_request_check/final_request_check`, and `impl Agent`; defaults are empty callbacks/steering, allow-all exposure, no MCP restriction, an empty client registry, a default (single-attempt) resilience policy, and no final check, so a caller that configures nothing gets the pre-seam loop.
- `Agent::invoke` parses the payload, resolves the session (`run::resolve_session`), runs the loop, and finishes an owned session; `Agent::stream` snapshots the agent, spawns the loop, and forwards frames through an unbounded channel (one terminal `answer` frame, or one terminal `Err` for invalid input, an unconfigured model, or cancellation).
- `config.rs` — `ReActAgentConfig` with Python field names/defaults plus `MaxIterationsPolicy` (`Error | FinalAnswer | ReturnObservations`, wire `error|final_answer|return_observations`).
- `agent_loop.rs` — `AgentRuntime::snapshot(&ReActAgent)`, `run(agent, ..)`, `run_with(&AgentRuntime, ..)`, `run_inner(..)`, and the pause helper `pause_output(..)`; the snapshot also carries the callback framework, steering queue, policies, and the precomputed model-facing projection (`visible_tool_info`, `visible_cards`).
- `model_step.rs` — a single model call: `call_model` (non-streaming, used only by the tool-free `FinalAnswer` finalization) and `call_model_streaming` driving `Model::stream` through a `StreamAccumulator`, emitting `llm_reasoning`/`llm_output`/`llm_usage` frames through `DeltaSink`, and building `ModelReply { message, finish_reason, usage, reasoning_present }`.
- `resume.rs` — `replay_interrupted(state, interactive, tools, options, resilience, client_tools)` replays interrupted tool calls: a `ClientTool` request (or a registered client-tool name) is completed through `ClientToolRegistry::invoke` (handler error → failed result), approvals re-execute through `execute_calls_with` with the resilience policy, rejections carry `reject_feedback`, ask-user answers become the result text, and unmatched requests are returned for re-pausing.
- `hooks.rs` — callback firing (`invoke_start/step_start/step_end/tool_call_start/tool_call_end/invoke_end`), `drain_steering`, and `check_final_request` with `FinalCheckOutcome { Allow, Reject { reason, inject }, Finish }`.
- `frames.rs` — `context_usage(publisher, session, iteration, &PreparedContext)` and `llm_route(publisher, session, &ModelRequestConfig, route, reason)`, both emitted through `DeltaSink` so the stream and the session mirror see them.
- `exposure.rs` — `visible_tools(tool_info, cards, &ToolExposurePolicy, &McpToolAllowlist) -> (Vec<Value>, Vec<ToolCard>)` (one aligned keep-mask) and `is_client_tool(registry, cards, name)`.
- `wiring/` — `#[cfg(test)]` behaviour suites for the loop wiring (`callbacks`, `steering`, `policy`, `client_tools`, `final_check`).
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
- Every iteration starts by draining the steering queue into the window (`ContextEngine::observe` per message, rendered by `SteeringMessage::to_message`), so a message pushed before the run reaches the first request and one pushed mid-run reaches the next iteration's request; the drain is rail-free (no `BeforeSteeringDrain` event exists because `openjiuwen-harness` matches `RailEvent` exhaustively and adding the variant would break it).
- Callbacks fire from `run_inner`: `InvokeStart` once after the effective limit is resolved (original query, resolved limit), `StepStart` before each model call (the messages actually sent, the model-visible cards), `StepEnd` after the reply (text, tool-call count, usage, finish reason), `ToolCallStart` per call that will produce a result (queued calls in model order before the batch, rail-skipped calls at substitution), `ToolCallEnd` in the final per-result loop with the final result (success, failure, or rail-skipped), and `InvokeEnd` unconditionally in `run_with` for both successful and failed runs. A callback error is logged with `tracing::warn!` and never aborts the run; the payload is not built when no callback subscribed.
- The model sees only the projected tools (`ToolExposurePolicy` then `McpToolAllowlist`, precomputed in the snapshot); execution always uses the full tool map, so a hidden tool is still callable through other paths but never advertised. MCP scoping reads `ToolCard.metadata` (`server_name`/`server_id`/`mcp_server`, as `load_mcp_tools` writes) and strips an optional `mcp_<server>_` prefix from the tool name.
- A call to a client tool (registered in `ClientToolRegistry` or carried by a card with `execution == ToolExecution::Client`) is never executed locally: the turn pauses once with an `InterruptKind::ClientTool` request (message "Please run this tool and return its result"), and a resume answer is routed through `ClientToolRegistry::invoke(&call, &answer)` — the handler value (string verbatim, else JSON text) becomes the tool result and a handler error becomes a failed result.
- Tool execution applies the per-tool resilience budget (`ToolResiliencePolicy::for_tool`): the per-call timeout overrides `ToolExecutionOptions::timeout`, tool/timeout failures retry up to `attempts()` with `backoff_for(attempt)`, and `Cancelled`/`Validation` failures are never retried; resume replay uses the same helper.
- A configured `FinalRequestCheck` runs before returning a no-tool-call answer: `Allow` keeps today's behavior, `Reject { inject }` observes `inject` (else the reason) as a user message and iterates while the budget remains (a rejection at the limit falls through), and `Finish(output)` returns that output with progress fields filled; no check configured leaves the empty-response retry untouched.
- Frame emission: `context.usage` after every successful `prepare` with `{iteration, message_count, tool_count, token_count, overflow, window_size, result_type}`; `llm_route` with `{route: "backup", model: <request model name>, reason: "model_override", result_type}` before a model call whose resolved model is not the primary one (rail `model_override`, e.g. `ModelBackupRail`, including a retry after `OnModelException`); the primary model emits no route frame.

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
