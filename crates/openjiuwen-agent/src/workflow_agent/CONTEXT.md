# workflow_agent/

## Intent

`WorkflowAgent`: run one of a set of `openjiuwen_workflow::Workflow` values, with optional LLM intent detection and HITL resume.
Ports Python `core/application/workflow_agent/` (`WorkflowAgent`, `WorkflowAgentConfig`) without the legacy `WorkflowController`/`Runner.resource_mgr` machinery.

## API Surface

- `mod.rs` — `WorkflowAgent { card, config: RwLock<WorkflowAgentConfig>, workflows: RwLock<Vec<Arc<Workflow>>>, model: RwLock<Option<Arc<dyn Model>>> }` implementing `Agent`: `new(WorkflowAgentConfig)`, `add_workflows` (dedupes by card id), `remove_workflow`, `workflows`, `config`, `set_model`; `invoke` and `stream` run the implemented dispatch below.
- `config.rs` — `WorkflowAgentConfig { id, name, description, version, constrain, default_response, global_params }`, `ConstrainConfig { reserved_max_chat_rounds: 10, max_iteration: 5 }`, `DefaultResponse { text }`.
- `select.rs` — `Selection { Workflow(usize), Default(String) }` and `select(workflows, query, model, request, default_text)`.

## Invoke

`Agent::invoke` parses `AgentInput`, resolves the agent session with `run::resolve_session`, runs the body in a private helper, and always calls `AgentRun::finish` before returning the helper's result.
The body errors with `Error::workflow("No workflows configured for single_agent")` when no workflow is registered.
It clones the config and model out of their locks before any `.await`, then builds `ModelRequestConfig` from the model name (or `Default`).
`select` returns `Selection::Workflow(index)` (runs that workflow) or `Selection::Default(text)` (returns `AgentOutput::answer(text)` with `data = {"answer", "status": "default_response"}` and `extra["status"] = "default_response"`).
The workflow runs in a child `WorkflowSession` built with `session_id = "<agent>::<workflow>"`, `workflow_id`, and `parent = Some(agent session)`.
A child with a parent shares the parent's stream hub, config, checkpointer, and trace id.
Dispatch: an `InteractiveInput` query calls `Workflow::resume` (a missing checkpointer propagates `Error::workflow`), otherwise `Workflow::invoke` receives the start payload.
The payload starts from `config.global_params`, then adds caller extras filtered to the workflow card's `input_params.properties` when that object exists, then overwrites `"query"` with the query text.
Mapping: `Completed` → `AgentOutput::answer(JSON text of result)` with `data = result` and `workflow_output = output`; `InputRequired` → `AgentOutput::interrupt(frames, ids)` from the `__interaction__` wires in `result` (id from `payload["id"]`) with `workflow_output = output`; `Error` → `AgentOutput::error` with `termination_reason = "workflow_error"` and `workflow_output = output`.
An interrupted output also fills `component_ids` from the interaction payload ids (frame order, non-string ids skipped, duplicates dropped), while `interrupt_ids` keeps the raw per-frame ids.

## Stream

`Agent::stream` parses input synchronously (malformed input → one `Err` item), snapshots the card identity, config, workflows, and model, and drives a spawned task over an unbounded `Result<AgentChunk>` channel returned as an `unfold` stream.
The task resolves the session, runs `select`, and either sends one terminal default-response `answer` chunk or calls `workflow.stream_with(payload, child_session, WorkflowRunOptions { interactive, resume: interactive.is_some() })`.
Every `Ok` workflow frame (component frames, `__interaction__`, `workflow_final`) is forwarded verbatim as `AgentChunk::new(item_type, index, payload)` with no synthesized duplicates.
Collected `__interaction__` frames become agent `OutputFrame`s (index from the payload, default 0) and ids come from `payload["id"]`.
Exactly one terminal `answer` chunk follows: `AgentOutput::interrupt` (with `component_ids` filled from the interaction payload ids) with `workflow_output = {result: <wire frames>, state: InputRequired}` when interactions arrived and no `workflow_final` did, otherwise `AgentOutput::answer(JSON text of the final payload)` with `data` and `workflow_output = {result, state: Completed}`.
The agent mirrors its own synthesized chunks into the supplied agent session (`session.write_stream(chunk.to_stream_item()).await`): the forwarded `workflow_final` chunk, the default-response `answer` chunk, and the terminal `answer` chunk; engine-originated frames reach that hub implicitly through the child session's shared writer and are never re-written.
A mirroring failure is logged at debug level and never changes the caller-visible stream.
Any non-cancellation setup/stream error becomes one terminal `answer` chunk carrying `AgentOutput::error`; `ErrorKind::Cancelled` surfaces as one `Err` item.
The session is finished before the sender drops.

## Select

`select` is async on every path: an empty slice is `Error::workflow("No workflows configured for single_agent")`, exactly one workflow returns `Workflow(0)` without touching the model, and several workflows consult the model when one is configured.
With several workflows and a model it sends the system prompt `You route user requests to one workflow. Reply with JSON {"result": <n>}.` and a user prompt listing `N. <description or name>` followed by a blank line and `Query: <query>`.
The reply is parsed by taking the first run of digits as a 1-based index; numbers outside `1..=len` and unparsable replies fall back.
A model `Err`, a missing model, or a failed parse all fall back to `Default(text)` when `default_text` is set, otherwise `Workflow(0)`; selection never errors except on an empty slice.

## Constraints

- Inputs are dispatched as `{"query": <text>}` merged over `global_params` and the caller's filtered `AgentInput.extra` keys.
- `InteractiveInput` in the query selects the resume path; interruption results preserve `WorkflowExecutionState::InputRequired`.
- Completed runs return `result_type = Answer`, `output` = JSON text of `workflow_output.result`, with the structured `WorkflowOutput` in `workflow_output`/`data`.
- The workflow engine already writes its frames (including `__interaction__`) to the child session's shared hub, so the agent must never re-write them; only its synthesized `workflow_final` and `answer` chunks are mirrored explicitly.
- An interrupted output reports the interaction payload ids in both `interrupt_ids` and `component_ids`.
- The agent never mutates a `Workflow`; workflows are shared as `Arc<Workflow>` and may run concurrently.
- No global registry lookups; model use is limited to intent detection.
- Library code is panic-free: index lookups use `get`/`ok_or_else`, never direct indexing.

## Test Strategy

`../../tests/workflow_agent.rs` covers the public contract: single- and multi-workflow invoke/stream, default-response fallback, interrupt → `result_type = interrupt` with `__interaction__` frames, and resume after an `InteractiveInput` answer.
Inline `select` unit tests cover the empty-slice error, the single-workflow fast path with zero model calls, `{"result": 2}`/bare `2`/quoted `"3"` parsing, the description-or-name prompt, garbage and out-of-range replies, missing default text, and model errors.
Inline `mod.rs` tests cover card/registration behavior, `build_payload` merging and extra filtering, the no-workflow error, a `StartComponent`→`EndComponent` invoke, interrupt mapping (`component_ids` from interaction ids, order/dedup of the helper), stream forwarding plus its terminal/error frames, and mirroring of the synthesized `workflow_final`/`answer` frames into a supplied session's hub (500 ms bounded drain).

## Routing Table

- `./mod.rs` → agent implementation, session bridging, payload/mapping helpers, and the stream task
- `./config.rs` → configuration types
- `./select.rs` → selection fast paths, prompt, parsing, and fallbacks
