# schema

## Intent

Pure data vocabulary of the harness: configuration, persisted state, tasks/todos, stop conditions, interaction/round types, and loop events.
Ports `openjiuwen/harness/schema/` (`config.py`, `state.py`, `task.py`, `stop_condition.py`, `interaction.py`, `loop_event.py`, `agent_mode.py`).
No behavior beyond pure helpers, validation and JSON round-trips; every other harness module depends on this one.

## API Surface

- `AgentMode` (`mode.rs`): `Normal` (default) | `Plan`, wire `normal`/`plan`.
- `config.rs`: `DeepAgentConfig` (serde data only: model settings `model_name`/`model_provider`/`api_key`/`api_base`/`custom_headers`, `model_config`/`model_client_config`, `card: AgentCard`, `system_prompt`, `max_iterations`, `max_iterations_hard_limit`, `on_max_iterations`, `enable_task_loop`, `enable_async_subagent`, `enable_subagent_runtime`, `add_general_purpose_agent`, `completion_timeout_secs` (600.0), `workspace: Workspace`, `cwd`/`project_root`, `skills: Vec<String>`, `enable_skill_discovery`, `language` (`"cn"`), `prompt_mode: PromptMode`, `subagents: Vec<SubAgentConfig>`, `default_mode`, `enable_plan_mode`, `parallel_tool_calls` (true), `restrict_to_work_dir` (true), `enable_tool_resilience_rail` (true), `enable_task_planning` (false), `permissions: Option<PermissionsSection>`, `task_loop_no_progress_guard`, `skill_budget_max_skills`, `skill_budget_max_total_chars`, `invocation_log_dir`, `enable_security_rail` (true)); `validate()` reproduces Python `__post_init__`.
- `OnMaxIterations` (`error` | `final_answer` | `return_observations`), `TaskLoopNoProgressGuardConfig` (`enabled=true`, `max_consecutive_empty_answers=3`, `min_answer_chars=20`).
- `SubAgentConfig`: `agent_card: AgentCard`, `system_prompt: Option<String>`, `tools: Vec<String>` (tool names), `skills`, `max_iterations`, `enable_task_loop`, `language`, `prompt_mode`, `factory_name`, `factory_kwargs: BTreeMap<String, Value>`, `enable_plan_mode`, `parallel_tool_calls`, `restrict_to_work_dir`, `display_name`, `role`, `subagent_type`.
- `state.rs`: `DeepAgentState { iteration, task_plan: Option<TaskPlan>, stop_condition_state: Option<Value>, pending_follow_ups: Vec<String>, plan_mode: PlanModeState }` with `to_session_dict`/`from_session_dict` (Python session key `deepagent`), plus `PlanModeState { mode, pre_plan_mode, plan_slug, prompt_context }`; `SESSION_STATE_KEY`.
- `task.rs`: `TodoItem` (`id`, `content`, `activeForm` camelCase wire name, `description`, `status`, `depends_on`, `result_summary`, `meta_data`, `selected_model_id`), `TodoStatus` (`pending`/`in_progress`/`completed`/`cancelled`), `TaskPlan { goal, tasks, current_task_id }` with `next_task()` honouring `depends_on`.
- `stop_condition.rs`: `StopEvaluationContext`, `StopConditionEvaluator` trait (`name`, `evaluate(&ctx) -> Option<StopVerdict>`), built-ins `MaxRounds`, `TokenBudget`, `Timeout`, `NoProgressAnswer`, `CompletionPromise`, `CustomPredicate`; `StopConditionState` (serde, Python `get_state`/`restore`).
- `interaction.rs`: `InteractionPhase`, `InputDispatchMode` (`follow_up`/`steer`), `SendInputRequest`, `RoundWorkItem`, `RoundOutcome`, `ActiveInteractionRound`, `InteractionEvent`.
- `loop_event.rs`: `DeepLoopEventType` (`followup`/`steer`/`abort`), priority map (abort 0 < steer 1 < followup 10), `DeepLoopEvent { priority, seq, created_at, event_id, event_type, content, task_id, metadata }`, `create_loop_event`.

## Constraints

- Serde snake_case, tolerant of unknown fields, defaults matching Python; no live handles (`Arc<dyn Model>`) in these types.
- `SecretString` for API keys; `Debug` redacts `api_key`.
- Validation returns `openjiuwen_common::error::Error::{config,validation}` with Python message text; no panics.
- This module must not import any other harness module except `crate::workspace::Workspace`, `crate::permission::PermissionsSection`, and `crate::prompts::PromptMode` (leaf references only).
- Files ≤1000 lines; split by concern (`config`, `state`, `task`, `stop_condition`, `interaction`, `loop_event`, `mode`).

## Routing Table

- `./mode.rs` → `AgentMode`
- `./config.rs` → `DeepAgentConfig`, `SubAgentConfig`, `OnMaxIterations`, `TaskLoopNoProgressGuardConfig`
- `./state.rs` → `DeepAgentState`, `PlanModeState`, session JSON round-trip
- `./task.rs` → `TaskPlan`, `TodoItem`, `TodoStatus`
- `./stop_condition.rs` → stop-condition evaluators and state
- `./interaction.rs` → interaction phases, round work items/outcomes, dispatch modes
- `./loop_event.rs` → `DeepLoopEvent` and priorities

## Test Strategy

- Unit tests in each file: config defaults + validation errors, state `to_session_dict`/`from_session_dict` round-trip (including pending follow-ups and plan mode), `TodoItem` camelCase `activeForm` wire name, `TaskPlan::next_task` dependency order, stop evaluators (max rounds, token budget, timeout, no-progress, completion promise), loop-event ordering (priority then seq).
- Behavior parity references: `tests/unit_tests/harness/test_deep_agent_session_state.py`, `test_loop_event_schema.py`, `test_loop_coordinator.py`.
