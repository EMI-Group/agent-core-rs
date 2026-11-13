# schema

## Intent

Pure data vocabulary of the harness: configuration, persisted state, tasks/todos, stop conditions, interaction/round types, and loop events.
Ports `openjiuwen/harness/schema/` (`config.py`, `state.py`, `task.py`, `stop_condition.py`, `interaction.py`, `loop_event.py`, `agent_mode.py`).
No behavior beyond pure helpers, validation and JSON round-trips; every other harness module depends on this one.
No IO, no live handles, no harness-module imports beyond the three leaf references `crate::workspace::Workspace`, `crate::permission::PermissionsSection`, and `crate::prompts::PromptMode` (defined here, re-exported there).
Every public item is re-exported from `crate::schema`; the child modules themselves are private.

## API Surface

### `mode.rs` (re-exported `AgentMode`, `PromptMode`)

- `AgentMode { Normal, Plan }` — serde wire `normal` / `plan`; `Default = Normal`; `as_str() -> &'static str`, `parse(&str) -> Option<Self>` (exact match).
- `PromptMode { Full, Minimal, None }` — wire `full` / `minimal` / `none`; `Default = Full`; `as_str()`, `parse(&str) -> Option<Self>`, `resolve(Option<&str>) -> Self` (unknown/`None` → `Full`, Python `prompts.resolve_mode`).

### `config.rs` (re-exported `DeepAgentConfig`, `OnMaxIterations`, `SubAgentConfig`, `TaskLoopNoProgressGuardConfig`, `resolve_inner_react_max_iterations`)

- `OnMaxIterations` = re-export alias of `openjiuwen_agent::MaxIterationsPolicy` (wire `error` / `final_answer` / `return_observations`; `Default = Error`).
- `TaskLoopNoProgressGuardConfig { enabled: bool = true, max_consecutive_empty_answers: usize = 3, min_answer_chars: usize = 20 }`.
- `SubAgentConfig` (serde data, no validation, `Default`):
  `agent_card: AgentCard`, `system_prompt: Option<String>`, `tools: Vec<String>` (tool names), `skills: Vec<String>`,
  `max_iterations: Option<u32>`, `enable_task_loop: bool = false`, `language: Option<String>`, `prompt_mode: Option<PromptMode>`,
  `factory_name: Option<String>`, `factory_kwargs: BTreeMap<String, Value>`, `enable_plan_mode: bool = false`,
  `parallel_tool_calls: bool = true`, `restrict_to_work_dir: bool = true`, `display_name: Option<String>`, `role: Option<String>`,
  `subagent_type: Option<String>`.
  Constructors/builders: `new(AgentCard)` plus consuming `#[must_use] with_*` per field.
- `DeepAgentConfig` (serde data only, `Default`, `#[serde(default)]`, unknown fields tolerated):
  model settings `model_name: Option<String>`, `model_provider: Option<String>`, `api_key: Option<SecretString>`,
  `api_base: Option<String>`, `custom_headers: BTreeMap<String, String>`, `model_config: Option<ModelRequestConfig>`,
  `model_client_config: Option<ModelClientConfig>`;
  `card: AgentCard`, `system_prompt: Option<String>`, `max_iterations: Option<u32>`, `max_iterations_hard_limit: Option<u32>`,
  `on_max_iterations: OnMaxIterations`, `enable_task_loop: bool = false`, `enable_async_subagent: bool = false`,
  `enable_subagent_runtime: bool = false`, `add_general_purpose_agent: bool = false`,
  `completion_timeout_secs: f64 = 600.0` (serde alias `completion_timeout`), `workspace: Workspace`, `cwd: Option<PathBuf>`,
  `project_root: Option<PathBuf>`, `skills: Vec<String>` (deserializer also accepts a single string), `enable_skill_discovery: bool = false`,
  `language: String = "cn"`, `prompt_mode: PromptMode`, `subagents: Vec<SubAgentConfig>`, `default_mode: AgentMode`,
  `enable_plan_mode: bool = false`, `parallel_tool_calls: bool = true`, `restrict_to_work_dir: bool = true`,
  `enable_tool_resilience_rail: bool = true`, `enable_task_planning: bool = false`, `enable_security_rail: bool = true`,
  `permissions: Option<PermissionsSection>`, `task_loop_no_progress_guard: TaskLoopNoProgressGuardConfig`,
  `skill_budget_max_skills: Option<usize>`, `skill_budget_max_total_chars: Option<usize>`, `invocation_log_dir: Option<PathBuf>`.
  Consuming `#[must_use] with_*` builders for every field.
  `validate(&self) -> Result<()>` using `Error::config` and the Python `__post_init__` messages:
  a hard limit requires `enable_task_loop == false`; the ceiling must be `>= 1`; `max_iterations` must be present, `>= 1` and `<= ceiling`; a standalone `max_iterations == Some(0)` is rejected as "max_iterations must be a positive integer".
  `Debug` is derived; `SecretString` and `ModelClientConfig` redact credentials.
- `resolve_inner_react_max_iterations(Option<u32>) -> usize` — `usize::MAX` when `None` (Python `sys.maxsize`).

### `state.rs` (re-exported `DeepAgentState`, `PlanModeState`, `SESSION_STATE_KEY`, `SESSION_RUNTIME_ATTR`)

- `SESSION_STATE_KEY = "deepagent"`, `SESSION_RUNTIME_ATTR = "_deepagent_runtime_state"`.
- `PlanModeState { mode: String = "normal", pre_plan_mode: Option<String>, plan_slug: Option<String>, prompt_context: Option<String> }`; `Default` sets `pre_plan_mode = Some("normal")`; a missing/null `pre_plan_mode` in JSON decodes to `None` (Python `from_dict`); `to_session_dict() -> Value`, `from_session_dict(Option<&Value>) -> Self`.
- `DeepAgentState { iteration: usize, task_plan: Option<TaskPlan>, stop_condition_state: Option<Value>, pending_follow_ups: Vec<String>, plan_mode: PlanModeState }`; `to_session_dict() -> Value` writes exactly `iteration`, `task_plan`, `stop_condition_state`, `pending_follow_ups`, `plan_mode`; `from_session_dict(Option<&Value>) -> Self` is lenient (non-object/non-dict `task_plan` → `None`, unknown keys dropped, numeric-string `iteration` accepted).

### `task.rs` (re-exported `TodoStatus`, `TodoItem`, `TaskPlan`, `ModelUsageRecord`, `status_icon`)

- `TodoStatus { Pending, InProgress, Completed, Cancelled }` (`pending`/`in_progress`/`completed`/`cancelled`), `Default = Pending`, `as_str()`, `parse(&str) -> Option<Self>`.
- `status_icon(TodoStatus) -> &'static str` — `"[ ]"`, `"[→]"`, `"[√]"`, `"[×]"`.
- `TodoItem { id (uuid v4 string default), content: String, active_form: String (serde name `activeForm`), description: String, status: TodoStatus, depends_on: Vec<String>, result_summary: Option<String>, meta_data: Option<BTreeMap<String, Value>>, selected_model_id: Option<String> }`; `to_dict() -> Value` with the Python key order, `from_dict(&Value) -> Result<Self>` (`Error::validation`).
- `TaskPlan { goal: String, tasks: Vec<TodoItem>, current_task_id: Option<String> }`; `get_task`, `next_task` (Python `get_next_task`), `add_task`, `mark_in_progress`, `mark_completed(task_id, summary)`, `mark_cancelled(task_id, reason)`, `progress_summary`, `to_markdown`, `to_dict`, `from_dict(&Value) -> Result<Self>` (null/`{}` → default).
  `next_task` semantics: scan `tasks` in list order; candidates are `Pending` only; "done" = `Completed` or `Cancelled`; every `depends_on` id must be done; an unknown dependency id blocks forever; empty `depends_on` is immediately eligible.
- `ModelUsageRecord { model_id: String, input_tokens: u64, output_tokens: u64 }`; `add(input, output)`, `Display`.

### `stop_condition.rs` (re-exported `StopEvaluationContext`, `StopVerdict`, `StopConditionEvaluator`, `MaxRounds`, `TokenBudget`, `Timeout`, `NoProgressAnswer`, `CompletionPromise`, `CustomPredicate`, `StopConditionState`)

- `StopEvaluationContext { iteration: usize, token_usage: u64, elapsed_seconds: f64, last_result: Option<Value>, extra: BTreeMap<String, Value> }` (`Default`, builders).
- `StopVerdict { reason: String, detail: Option<String> }`; `new(reason)`, `with_detail(detail)`. `reason` is the evaluator name so `stop_reason` strings stay Python-compatible.
- `#[async_trait] pub trait StopConditionEvaluator: Send + Sync { fn name(&self) -> &str; async fn evaluate(&mut self, ctx: &StopEvaluationContext) -> Option<StopVerdict>; fn reset(&mut self) {} fn get_state(&self) -> Option<Value> { None } fn load_state(&mut self, state: &Value) {} }` (object-safe).
- Built-ins (exact Python class names from `name()`): `MaxRounds { max_rounds }` (`iteration >= max_rounds`), `TokenBudget { max_tokens }` (`token_usage >= max_tokens`), `Timeout { timeout_seconds }` (`elapsed_seconds >= timeout_seconds`), `NoProgressAnswer { max_consecutive_empty_answers = 3, min_answer_chars = 20 }` (counts consecutive `result_type == "answer"` rounds whose stripped `output` is shorter than the threshold; counter resets otherwise), `CompletionPromise { promise, required_confirmations = 1 }` (externally driven by `notify_fulfilled` / `notify_absent`; consecutive confirmations), `CustomPredicate` (closure `Box<dyn Fn(&StopEvaluationContext) -> bool + Send + Sync>` + name; manual `Debug`).
  Every built-in has a `new(...)` constructor, `Default` where sensible, and implements the state hooks it uses.
- `StopConditionState { iteration: usize, token_usage: u64, stop_reason: Option<String>, evaluator_states: BTreeMap<String, Value> }`; `Default`, serde, `get_state() -> Value`, `restore(&Value) -> Self` (tolerant).

### `interaction.rs` (re-exported `InteractionPhase`, `InputDispatchMode`, `SendInputRequest`, `RoundWorkKind`, `RoundWorkItem`, `RoundOutcome`, `ActiveInteractionRound`, `InteractionEvent`, `InteractionEventType`)

- `InteractionPhase { Idle, Running, Terminated }` (`idle`/`running`/`terminated`), `Default = Idle`.
- `InputDispatchMode { FollowUp, Steer }` (`follow_up`/`steer`).
- `InteractionEventType { GoalUpdated, ExecutionError }` (`goal.updated`/`execution.error`).
- `SendInputRequest { request_id: String, inputs: BTreeMap<String, Value>, mode: Option<InputDispatchMode> }`.
- `RoundWorkKind { User, Goal }` (`user`/`goal`), `Default = User`.
- `RoundWorkItem { kind: RoundWorkKind, request_id: Option<String>, inputs: BTreeMap<String, Value>, context: BTreeMap<String, Value>, is_follow_up: bool }`; `user(request_id, inputs, is_follow_up, reset_loop)`, `goal(inputs, goal_id, revision, session_id)` (context `{goal_id, revision, session_id, reset_loop: true}`), `query() -> Option<&Value>`, `reset_loop() -> bool` (defaults `true`).
- `RoundOutcome { next_work: Option<RoundWorkItem>, error_code: Option<String>, error_message: Option<String> }`; `answer(next_work)`, `timeout()` (code `completion_timeout`), `abort()` (code `aborted`), `failed(code, message)`, `is_error()`.
- `ActiveInteractionRound { work: RoundWorkItem, task_id: Option<String> }`; `run_kind()`, `run_context()`, `goal_id()`, `revision()`.
- `InteractionEvent { event_type: InteractionEventType (serde name `type`), payload: BTreeMap<String, Value> }`; `goal_updated(Option<Value>)`, `execution_error(code, message, goal: Option<Value>)`, `to_output_value() -> Value` (`{"type", "index": 0, "payload"}`).

### `loop_event.rs` (re-exported `DeepLoopEventType`, `DeepLoopEvent`, `create_loop_event`, `default_event_priority`)

- `DeepLoopEventType { Followup, Steer, Abort }` (`followup`/`steer`/`abort`), `Default = Followup`.
- `default_event_priority(DeepLoopEventType) -> i32` — abort `0` < steer `1` < followup `10` (unknown/custom defaults to `10`).
- `DeepLoopEvent { priority: i32, seq: u64, created_at: f64 (monotonic seconds), event_id: String (uuid v4 default), event_type: DeepLoopEventType, content: String, task_id: Option<String>, metadata: BTreeMap<String, Value> }`; `Ord`/`Eq` compare only `(priority, seq)` (Python `dataclass(order=True)`).
- `DeepLoopEvent` fields are private with no accessors: construct via `Default`/`create_loop_event` and read them back through serde (`serde_json::to_value`).
- `create_loop_event(seq, event_type, content, task_id, metadata, priority: Option<i32>) -> DeepLoopEvent` — `priority = Some(0)` is honoured, `None` uses the type default.

## Constraints

- Serde snake_case, tolerant of unknown fields, defaults matching Python; no live handles (`Arc<dyn Model>`, rails, `SysOperation`) in these types.
- `SecretString` for API keys; `Debug` output never contains the key.
- `DeepAgentConfig`, `SubAgentConfig` and the persisted state types derive `JsonSchema`; schemars overrides cover `SecretString`, `ModelRequestConfig`, `ModelClientConfig`, `MaxIterationsPolicy` and `PathBuf` fields.
- Validation returns `openjiuwen_common::error::{Error, Result}` with Python message text; no panics outside tests.
- This module must not import any other harness module except `crate::workspace::Workspace`, `crate::permission::PermissionsSection`, and `crate::prompts::PromptMode` (re-export of `mode::PromptMode`, leaf reference only).
- Files ≤1000 lines; `config` is split into `config/{mod,agent,subagent,tests}.rs` (the facade re-exports keep the API unchanged).
- Unit tests live in each file (`#[cfg(test)] mod tests`) or a sibling `*_tests.rs`; no integration-test changes (that directory is owned elsewhere).

## Design decisions

- Python `DeepAgentConfig` mixes live objects (`Model`, rails, `SysOperation`, `permission_host`, `stream_task_lifecycle`) with data; the Rust type is serde data only. Live handles are supplied through `DeepAgentBuilder`. The live `model` field is replaced by `model_name`/`model_provider`/`api_key`/`api_base`/`custom_headers`/`model_client_config`/`model_config`.
- Deferred Python config fields (tool/-MCP object lists, `backend`, `context_engine_config`, `kv_cache_affinity_config`, `model_selection`, `permission_host`, `moa`, vision/audio model configs, `auto_create_workspace`, `tool_owner_id`) are intentionally absent; their owning subsystems carry that state.
- `progressive_tool_enabled`/`tool_search_limit` are not configuration here: progressive discovery is a tool-registry concern (`ToolRegistry::with_deferred_tools`, `tool_search`, `reveal`).
- `OnMaxIterations` is the kernel's `MaxIterationsPolicy` re-exported under the Python-facing name, so no conversion is needed when mapping onto `ReActAgentConfig`.
- `StopConditionEvaluator` is async and carries mutation (`&mut self`) instead of Python's sync `should_stop(&ctx) -> bool`; the verdict's `reason` is the Python evaluator class name, preserving persisted/logged `stop_reason` values (`MaxRoundsEvaluator`, `NoProgressAnswerEvaluator`, …). The Python `reset`/`get_state`/`load_state` hooks are kept.
- `RoundOutcome` keeps Python's fields and adds the `answer`/`timeout`/`abort`/`failed` constructors used by `task_loop`; timeout/abort codes are `completion_timeout`/`aborted`.
- `TaskPlan::next_task` is Python's `get_next_task`; completion order is list order, done = completed **or** cancelled, unknown dependency ids block permanently.
- `TodoItem`/`TaskPlan` derive serde with `activeForm` renamed; their `from_dict` is strict (`Error::validation`), while `DeepAgentState::from_session_dict` degrades malformed subtrees to defaults for persistence tolerance.
- `PlanModeState::pre_plan_mode` is `Option<String>`: `Default`/dataclass default is `Some("normal")`, but restored `null`/missing is `None` (Python `from_dict` asymmetry).
- Interaction streaming guards (`OutputLease`, `OutputLeaseManager`, `InteractionOutputStream`) are runtime concurrency, not schema data; they belong to the agent streaming layer and are not ported here.
- `NoProgressAnswer` defaults `min_answer_chars` to 20 (the production guard-config value); Python's standalone evaluator default of 80 only applies when constructed directly and is not used by `DeepAgent`.
- The second, flat session layout written by `ContextProcessorRail` (`task_state`, `iteration`, `pending_follow_ups`, `plan_mode`) is a consumer concern; this module only owns the `"deepagent"` payload.
- `StopEvaluationContext` also exposes read accessors (`iteration`, `token_usage`, `elapsed_seconds`, `last_result`, `extra`) because `CustomPredicate` closures receive `&StopEvaluationContext` and cannot move fields out.
- `TaskPlan::to_markdown` joins its lines with `\n` (Python `"\n".join`): a plan with tasks ends on the last task line with no trailing newline, while a task-less plan still ends with one because of its blank second line.
- `resolve_inner_react_max_iterations` is a `const fn`, so callers may use it in const context.

## Routing Table

- `./mode.rs` → `AgentMode`, `PromptMode`
- `./config/` → `DeepAgentConfig`, `SubAgentConfig`, `OnMaxIterations`, `TaskLoopNoProgressGuardConfig`, iteration helper (`mod.rs` facade, `agent.rs`, `subagent.rs`, `tests.rs`; public paths stay `crate::schema::<item>`)
- `./state.rs` → `DeepAgentState`, `PlanModeState`, session JSON round-trip, session keys
- `./task.rs` → `TaskPlan`, `TodoItem`, `TodoStatus`, `ModelUsageRecord`, status icons
- `./stop_condition.rs` → evaluation context/verdict, evaluator trait, built-in evaluators, persisted state
- `./interaction.rs` → phase/dispatch/event/round types, work items, outcomes
- `./loop_event.rs` → `DeepLoopEvent`, event type, priorities, factory
- `./mod.rs` → crate-facing re-exports (all public items)

## Test Strategy

- `mode`: defaults, wire names, `parse`/`resolve` fallbacks.
- `config`: default values; JSON round-trip; unknown-field tolerance; `skills` scalar-or-list; `validate` for hard-limit conflicts, non-positive/inconsistent `max_iterations`, and legal defaults; redacted `Debug`; builder coverage; `SubAgentConfig`/guard defaults.
- `state`: `to_session_dict` key set, round-trip through `from_session_dict` including pending follow-ups and plan mode, null/empty/unknown-key tolerance, non-dict `task_plan` degradation, `pre_plan_mode` asymmetry.
- `task`: `activeForm` wire name, JSON round-trip, dependency-ordered `next_task` (list order, completed/cancelled done set, blocked unknown dependency, empty dependency), mark/progress/markdown helpers, `ModelUsageRecord::add`.
- `stop_condition`: each evaluator's boundary (`>=`, inclusive), no-progress counter reset/trigger and state round-trip, completion-promise consecutive confirmations and OR-merge on restore, verdict reason = class name, custom predicate.
- `interaction`: enum wire values, work-item factories/context, `reset_loop` default, outcome constructors/codes, event payload shape and `index: 0`.
- `loop_event`: priority map, `(priority, seq)` ordering in a sorted collection, custom priority including `0`, metadata defaults.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` all clean.
