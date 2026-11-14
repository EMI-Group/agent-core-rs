# openjiuwen-agent controller reasoner

## Intent

The task-operation state machine: one intent becomes `TaskOp`s, a clarification is recorded, and the ops are applied to the shared `TaskManager`.
Ports the reasoning path of Python's `EventHandlerWithIntentRecognition` (`openjiuwen/core/controller/modules/intent_recognizer.py`), split into pure planning and an applying step with an explicit pause/cancel/resume seam.
It owns no per-invocation state beyond the controller config and an optional `TaskControl`; the `TaskManager` and `Planner` are passed in, so one reasoner serves every session.

## API Surface

- `Reasoner` (`Clone + Default`) — `new(ControllerConfig, Option<Arc<dyn TaskControl>>)`, `config`, `set_config`, `control`, `set_control`, `plan_ops(&Intent) -> Vec<TaskOp>`, `async reason(&Intent, Option<&Arc<dyn Session>>) -> Result<Vec<TaskOp>>`, `async apply_task_ops(&[TaskOp], &TaskManager, &mut Planner, session_id, Option<&Arc<dyn Session>>) -> Vec<TaskOpOutcome>`.
- `TaskControl` (async trait) — `pause`, `cancel`, `resume`, each `(&self, task_id: &str) -> bool`; `BaseController` implements it, and a reasoner without one reports control ops as not applied.
- `TaskOp` (enum, no serde) — `Create { intent, task_type }`, `Pause { task_id }`, `Resume { task_id }`, `Cancel { task_id }`, `Continue { depend_task_id, event, metadata, description }`, `Supplement { task_id, info }`, `Modify { task_id, details }`, `Unknown { prompt }`; `kind()` and `task_id()`.
- `TaskOpOutcome` — `kind: &'static str`, `task_id: Option<String>`, `status: Option<TaskStatus>`, `applied: bool`, `message: String`, `clarification: Option<String>`.
- Constants — `CLARIFICATION_STATE_KEY = "__controller_clarification__"`, `DEFAULT_CLARIFICATION = "Please clarify your request."`, `SUPPLEMENT_MARKER = "\n\n任务补充信息:\n"`, `MODIFY_MARKER = "\n\n修改详情:\n"`.

## Planning (`plan_ops`)

- The confidence gate runs before any kind dispatch: a `NaN` or a confidence strictly below `ControllerConfig::intent_confidence_threshold` (default `0.7`) yields exactly one `TaskOp::Unknown`, so equality with the threshold passes and a gated `SwitchTask` produces one unknown op.
- `CreateTask` yields one `Create` with `task_type = intent.task_type` or `config.default_task_type`; the intent is not validated here, so a blank description fails later in the planner.
- `PauseTask`/`ResumeTask`/`CancelTask` yield the matching op with the trimmed `target_task_id`, or `Unknown` when it is missing or blank.
- `ContinueTask` yields `Continue` with the trimmed non-empty `depend_task_id`, the intent's event and metadata, and the trimmed description, or `Unknown` when no dependency remains.
- `SupplementTask`/`ModifyTask` require both a target id and a non-blank payload and otherwise degrade to `Unknown`.
- `SwitchTask` yields `Cancel` for the current task when it is targeted, followed by a `Create` built from a clone retyped to `CreateTask` whose description falls back to the event text and whose `name` is preserved.
- `UnknownTask` yields `Unknown` with the trimmed `clarification_prompt` or `DEFAULT_CLARIFICATION`.
- The function is pure: it clones the intent and event and mutates no state.

## Clarification (`reason`)

- It plans the ops and, only when the first op is `Unknown` and a session exists, merges `{"prompt", "event_id", "confidence"}` under `CLARIFICATION_STATE_KEY` and writes one `task_interaction` text frame (index 0) through `session.write_stream`.
- Both session calls are best-effort: failures are warned and swallowed, so the function returns `Ok` and the `Result` is kept as a seam.
- The frame is written to the session stream rather than returned, and applying an `Unknown` op afterwards changes no state.
- Stale clarification state is only overwritten by a later unknown intent; it is never cleared by a recognized one.
- The base turn does not call this: it keeps its own clarification constant and output, so `reason` is a host seam.

## Applying (`apply_task_ops`)

- One outcome per op in input order; a duplicate id, a missing task, or a refused control call becomes a failed outcome, and the batch never aborts and never returns `Err`.
- Scope is the trimmed `session_id`, else the session's id, else the empty string; ids are minted by the injected `Planner` as `{scope}_task_{seq}`, so creates never collide across turns.
- `Create` plans the intent with the op's task type, then sets that type, `priority = config.default_task_priority` (default `1`), `status = Submitted`, `context_id = "{scope}_{task_id}"`, `inputs = [event]`, and `metadata = intent.metadata` (replacing the planner's intent keys).
- A create whose intent the planner drops (for example a blank description) fails with `"no task planned for intent {label}"`; a duplicate id fails with the manager's add error.
- `Pause`/`Cancel` check existence, then the seam, then report `applied` with `Paused`/`Canceled` or `"task {id} pause|cancel refused"` / `"no task control installed"` / `"task {id} not found"`.
- `Resume` first requires the stored task to be `Paused` (otherwise `"task {id} is not paused"` with the current status and the control untouched), then applies through the seam with `Submitted`.
- `Continue` requires every dependency to exist (the first missing one is reported as the outcome's task id), inherits the dependencies' inputs in order and appends the event, uses the primary dependency's `context_id` (else `"{scope}_{id}"`) and task type, and sets `parent_task_id` to the primary.
- `Continue` copies no context-engine messages and injects no `depends_on`; a host that wants scheduler gating supplies it through intent metadata.
- `Supplement` pauses the task through the seam (a refusal fails the op), appends `SUPPLEMENT_MARKER` plus the info to `description`, and re-queues it as `Submitted` without creating a task.
- `Modify` cancels the original through the seam, then creates a replacement whose description appends `MODIFY_MARKER` plus the details, whose inputs, metadata, and context id are copied from the original, and whose `parent_task_id` is the original; the original stays stored as the audit trail.
- `Unknown` returns an outcome with the prompt and `clarification: Some(prompt)` and touches neither the manager nor the session.

## Constraints

- Planning and applying are deterministic: no clock, no randomness, and no model call; task ids come only from the planner.
- Only `intent_confidence_threshold`, `default_task_type`, and `default_task_priority` are read from `ControllerConfig`; `intent_type_list`/`accepts` are not consulted here.
- The config is snapshotted once per call, so a mid-batch `set_config` does not affect a running `apply_task_ops`.
- `TaskOp` and `TaskOpOutcome` are internal Rust types with no serde form; `Debug`/`Clone`/`PartialEq` are the only derives.
- No panics, `unwrap`, or `expect` outside tests.

## Divergences from Python

- The confidence gate lives in planning rather than in recognition, and it applies to every intent kind.
- Intents are applied sequentially in input order instead of `asyncio.gather`.
- Create ids come from the planner instead of `target_task_id` or a UUID, and `inputs` always carries the event.
- Pause/cancel/resume report an outcome instead of discarding the scheduler's result.
- Continue cannot read context-engine messages and therefore inherits recorded inputs and the primary context id.
- Modify creates a linked replacement instead of mutating the original in place.
- Supplement requires the pause to succeed, where Python ignores it.
- `SwitchTask` is first-class (cancel + create), where Python raises through its unknown-intent path.
- Unknown intents persist `{prompt, event_id, confidence}` and emit a frame where Python writes only a clarification dict.
- Blank targets and payloads degrade to `Unknown` or a failed outcome instead of raising.
- No `ReasonerConfig`/`PlannerConfig` family and no LLM planning inside the reasoner.

## Test Strategy

- 19 offline tests in `tests.rs` (6 synchronous, 13 async), driven by a real `TaskManager`, a real `Planner`, an `AgentSession`, and a recording `StubControl`; no model is used.
- `plan_ops` tests cover every intent kind, task-type resolution, trimming, `SwitchTask` cancel+create (targeted and untargeted), blank-target/payload degradation, and the confidence gate at 0.69/0.70/0.95 and `NaN`.
- `reason` tests assert the persisted `{prompt, event_id, confidence}` state and the single `task_interaction` frame, plus the no-session path.
- Applying tests cover create (id shape, context id, priority, metadata, repeated applies), default priority and scope fallback, pause/cancel forwarding order, the missing-seam/refused/missing-task outcomes, resume's paused precondition, continue inheritance, supplement append + re-queue, modify replacement with the original retained, unknown mutation-free behaviour, duplicate-id failure, batch continuation after a failure, and `kind()`/`task_id()` helpers.
- The only timing margin is a 500 ms bound on reading the session stream; everything else is direct state assertion.

## Routing Table

- `./mod.rs` → `Reasoner`, `TaskControl`, `TaskOp`, `TaskOpOutcome`, the clarification/marker constants, planning and applying
- `./tests.rs` → planning, clarification, applying, and batch-continuation suite
