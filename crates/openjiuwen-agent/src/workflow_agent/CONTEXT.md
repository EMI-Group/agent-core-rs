# workflow_agent/

## Intent

`WorkflowAgent`: run one of a set of `openjiuwen_workflow::Workflow` values, with optional LLM intent detection and HITL resume.
Ports Python `core/application/workflow_agent/` (`WorkflowAgent`, `WorkflowAgentConfig`) without the legacy `WorkflowController`/`Runner.resource_mgr` machinery.

## API Surface

- `mod.rs` — `WorkflowAgent { card, config, workflows, model for intent detection }` implementing `Agent`: `new(WorkflowAgentConfig)`, `add_workflows`, `remove_workflow`, `workflows`, `set_model`; `invoke` resolves the agent session, creates a child `WorkflowSession` (`parent_session_id` = agent session id), selects a workflow, and runs `Workflow::invoke`/`Workflow::resume`; `stream` forwards `StreamItem` frames as `AgentChunk`s (`workflow_final`, `__interaction__`, component frames) and mirrors them into the session.
- `config.rs` — `WorkflowAgentConfig { id, name, description, version, constrain, default_response, global_params }`, `ConstrainConfig { reserved_max_chat_rounds: 10, max_iteration: 5 }`, `DefaultResponse { text }`.
- `select.rs` — workflow selection: single workflow wins; with a model, a numbered intent-detection prompt over `description or name` parsed from `{"result": N}` / a bare integer; fallback to the first workflow; `default_response` when detection fails and text is set.

## Constraints

- Inputs are dispatched as `{"query": <text>}` merged with the caller's `AgentInput.extra` keys, filtered to the workflow's `input_params` properties when that schema declares them.
- `InteractiveInput` in the query (or a stored interrupted state) selects the resume path; interruption results preserve `WorkflowExecutionState::InputRequired`.
- Completed runs return `result_type = Answer`, `output` = JSON text of `workflow_output.result`, with the structured `WorkflowOutput` in `workflow_output`/`data`.
- The agent never mutates a `Workflow`; workflows are shared as `Arc<Workflow>` and may run concurrently.
- No global registry lookups; model use is limited to intent detection.

## Test Strategy

`../../tests/workflow_agent.rs`: single-workflow invoke/stream (with `StartComponent`→`EndComponent`), multi-workflow selection driven by a scripted model, default response fallback, interruption → `result_type = interrupt` with `__interaction__` frame, resume after an `InteractiveInput` answer.

## Routing Table

- `./mod.rs` → agent implementation and session bridging
- `./config.rs` → configuration types
- `./select.rs` → workflow selection/intent detection
