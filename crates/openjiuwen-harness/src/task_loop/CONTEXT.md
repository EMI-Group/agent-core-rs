# task_loop

## Intent

The outer task loop that wraps single ReAct turns: round coordination, iteration/token/time budgets, follow-up and steering queues, pause/resume/abort, timeout handling, and event-handler hooks with state persistence after each round.
Ports `openjiuwen/harness/task_loop/{loop_coordinator,loop_queues,event_manager,task_loop_controller,task_loop_event_handler,task_loop_event_executor}.py` and the `_run_task_loop` driver in `deep_agent.py`.

## API Surface

- `LoopQueues`: `follow_up: VecDeque<QueuedInput>`, `steering: VecDeque<QueuedInput>`; `push_follow_up`, `push_steer`, `drain_follow_up`, `drain_steering`, `is_empty`.
- `LoopCoordinator`: `new(evaluators)`, `with_max_outer_rounds(50)`, `should_continue(&StopEvaluationContext) -> bool`, `evaluate(...) -> Option<StopVerdict>`, `stop_reason()`, `request_abort()`, `is_aborted()`, `observe_round(iteration, usage, elapsed, last_result)`, `state()/restore(state)`, `add_token_usage`.
- `EventManager`: bounded priority queue of `DeepLoopEvent`; `publish`, `pop` (lowest priority value first, then seq), `drain`, `len`.
- `TaskLoopController`: `new(handler, coordinator, queues, config)`, `submit_round(inputs, is_follow_up, task_id) -> RoundId`, `run_round(RoundWorkItem) -> RoundOutcome`, `run_until_idle()`, `wait_round_completion(timeout)`, `pause()/resume()`, `cancel()`, `is_running`.
- `TaskLoopEventHandler` (trait + `DefaultTaskLoopEventHandler`): `prepare_round`, `handle_input`, `handle_interaction`, `handle_task_completion`, `handle_task_failed`, `on_abort`, `after_round`. Hooks fire exactly once per round.
- `TaskLoopConfig { completion_timeout: Duration (600 s), max_outer_rounds: usize (50), resume_continuation: bool, persist_state: bool }`.
- `RoundExecutor` trait: the single-round work (`run_round(&RoundWorkItem) -> RoundOutcome`), implemented by `DeepAgent`; the controller owns sequencing and budgets.

## Constraints

- The controller never owns the agent: execution goes through the `RoundExecutor` trait (object-safe, `async_trait`).
- Cancellation via `CancellationToken`; a timeout cancels the active round and yields `RoundOutcome::timeout()` with `stop_reason = "completion_timeout"`; abort yields `stop_reason = "aborted"`.
- Follow-ups drain at round boundaries (one per round) and are persisted in `DeepAgentState::pending_follow_ups`; steering is handed to the inner loop and drained before each model call.
- State is persisted after every round through a `StateSink` callback supplied by the agent; the controller itself performs no IO.
- Budgets: any evaluator returning a verdict stops the loop; `max_outer_rounds` is a hard cap yielding `stop_reason = "MaxOuterRounds"`.
- No panics; all failures become `RoundOutcome::failed`.

## Routing Table

- `./coordinator.rs` → `LoopCoordinator`, budget/stop evaluation, abort flag
- `./queues.rs` → `LoopQueues`, `QueuedInput`
- `./events.rs` → `EventManager`, priority queue over `DeepLoopEvent`
- `./controller.rs` → `TaskLoopController`, `TaskLoopConfig`, `RoundExecutor`, round lifecycle and timeouts
- `./handler.rs` → `TaskLoopEventHandler` trait + default implementation

## Test Strategy

- Unit tests: coordinator OR-evaluation and first-stop-wins, max outer rounds, abort wins immediately, queue FIFO/priority, event-manager ordering (abort < steer < follow-up, then seq), round timeout, follow-up drained per round, handler hooks fire once, state sink called after each round.
- Parity references: `tests/unit_tests/harness/test_loop_coordinator.py`, `test_loop_queues.py`, `test_deep_agent_event_handler.py`, `test_deep_agent_event_executor.py`, `test_deep_agent_interaction.py`.
