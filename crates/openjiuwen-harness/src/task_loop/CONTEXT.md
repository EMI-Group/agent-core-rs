# task_loop

## Intent

The outer task loop that wraps single ReAct turns: round coordination, iteration/token/time budgets, follow-up and steering queues, pause/resume/abort, completion timeouts, and event-handler hooks with state persistence after each round.
Ports `openjiuwen/harness/task_loop/{loop_coordinator,loop_queues,event_manager,task_loop_controller,task_loop_event_handler,task_loop_event_executor}.py` and the `_run_task_loop` driver in `deep_agent.py`.
The controller never owns the agent: one round of work is delegated to a `RoundExecutor` implementation supplied by `crate::agent`.
Design rule: the queue/coordinator/handler are pure loop machinery; the agent owns model, stream and session.

## API Surface

All public items are re-exported flat from `crate::task_loop` (submodules are private).

### `queues.rs`

- `QueuedInput { content: String, task_id: Option<String>, metadata: BTreeMap<String, Value> }` — serde, `PartialEq`; `new(content)`, `with_task_id(Option<String>)`, `with_metadata(BTreeMap<String, Value>)`, accessors `content()`, `task_id()`, `metadata()`, `From<String>`/`From<&str>`.
- `LoopQueues` — FIFO follow-up and steering queues behind `Arc<parking_lot::Mutex<VecDeque<QueuedInput>>>`; `Clone` shares the queues (`Default` = empty, unrelated queues).
  Methods (`&self`): `push_follow_up(input)`, `push_steer(input)`, `pop_follow_up() -> Option<QueuedInput>`, `pop_steer()`, `drain_follow_up() -> Vec<QueuedInput>`, `drain_steering() -> Vec<QueuedInput>`, `has_follow_up()`, `has_steering()`, `is_empty()`, `follow_up_len()`, `steering_len()`, `clear()`.

### `coordinator.rs`

- `DEFAULT_MAX_OUTER_ROUNDS: usize = 50`.
- `default_evaluators(guard: &TaskLoopNoProgressGuardConfig) -> Vec<Box<dyn StopConditionEvaluator>>` — the task-loop default chain: exactly the `NoProgressAnswer` guard when `guard.enabled`, otherwise empty.
- `LoopCoordinator` (manual `Debug` printing evaluator names):
  `new(Vec<Box<dyn StopConditionEvaluator>>)`, `with_defaults(&TaskLoopNoProgressGuardConfig)`, `with_max_outer_rounds(usize)`, `set_max_outer_rounds(usize)`, `max_outer_rounds()`.
  `async should_continue(&mut self, &mut StopEvaluationContext) -> bool`, `async evaluate(&mut self, &mut StopEvaluationContext) -> Option<StopVerdict>`.
  `stop_reason() -> Option<&str>`, `request_abort(&self)`, `clear_abort(&self)`, `is_aborted() -> bool`.
  `iteration()`, `token_usage()`, `elapsed_seconds()`, `last_result()`, `increment_iteration()`, `add_token_usage(u64)`, `set_last_result(Option<Value>)`, `observe_round(iteration, token_usage_delta, elapsed_seconds, last_result)`.
  `reset()`, `state() -> StopConditionState`, `restore(&StopConditionState)`, `evaluator_names() -> Vec<&str>`.

### `events.rs`

- `DEFAULT_EVENT_CAPACITY: usize = 0` (0 = unbounded, Python parity).
- `EventManager` — bounded priority queue of `DeepLoopEvent` ordered by `(priority, seq)` (lowest priority value first, then seq) in a sorted `Vec` behind `parking_lot::Mutex`; `Clone` shares the queue.
  `new(capacity)`, `with_capacity(capacity)`, `capacity()`, `publish(event) -> Option<DeepLoopEvent>`, `publish_event(event_type, content, task_id, metadata, priority) -> Option<DeepLoopEvent>`, `pop()`, `peek()`, `drain() -> Vec<DeepLoopEvent>`, `len()`, `is_empty()`, `clear()`, `next_seq() -> u64`.
  `publish` returns the dropped event, if any: a full queue evicts its least urgent queued event (largest `(priority, seq)`); if the incoming event is itself the least urgent it is rejected and returned.

### `handler.rs`

- `#[async_trait] TaskLoopEventHandler: Send + Sync` — every hook has a default `Ok(())` body:
  `prepare_round(round_id, round: &ActiveInteractionRound) -> Result<()>`,
  `handle_input(round_id, round: &ActiveInteractionRound) -> Result<()>`,
  `handle_interaction(round_id, input: QueuedInput) -> Result<()>`,
  `handle_task_completion(round_id, round: &ActiveInteractionRound, outcome: &RoundOutcome) -> Result<()>`,
  `handle_task_failed(round_id, round: &ActiveInteractionRound, outcome: &RoundOutcome) -> Result<()>`,
  `on_abort(round_id, round: &ActiveInteractionRound) -> Result<()>`,
  `after_round(round_id, round: &ActiveInteractionRound, outcome: &RoundOutcome) -> Result<()>`.
- `DefaultTaskLoopEventHandler` — `new()`, `with_queues(LoopQueues)`, `queues()`, `last_outcome()`, `last_error()`; `handle_interaction` pushes the input onto its steering queue when one is configured, terminal hooks record the outcome.

### `controller.rs`

- `type RoundId = u64` (1-based, monotonic per controller).
- `type StateSink = Arc<dyn Fn(&DeepAgentState) -> Result<()> + Send + Sync>`.
- `#[async_trait] RoundExecutor: Send + Sync`:
  `async fn run_round(&self, work: RoundWorkItem) -> Result<RoundOutcome>` (required);
  `fn last_round_result(&self) -> Option<Value> { None }` (the Python `{output, result_type, ...}` dict fed to stop-condition evaluators);
  `fn token_usage(&self) -> u64 { 0 }` (cumulative tokens; the controller feeds the delta to `observe_round`).
- `TaskLoopConfig { completion_timeout: Duration (600 s), max_outer_rounds: usize (50), resume_continuation: bool (false), persist_state: bool (true) }` — `Default`, consuming `with_*` builders; `Duration::ZERO` disables the completion timeout.
- `LoopReport { rounds: usize, stop_reason: Option<String>, last_outcome: Option<RoundOutcome>, aborted: bool, paused: bool }` — `Default`, `is_error()`.
- `TaskLoopController`:
  `new(handler: Box<dyn TaskLoopEventHandler>, coordinator: LoopCoordinator, queues: LoopQueues, config: TaskLoopConfig)` (applies `config.max_outer_rounds` to the coordinator),
  `with_state(state)`, `with_state_sink(StateSink)`, `set_state_sink(StateSink)`, `set_state(DeepAgentState)`, `state() -> DeepAgentState`, `persist() -> Result<()>`,
  `submit_round(inputs: BTreeMap<String, Value>, is_follow_up: bool, task_id: Option<String>) -> RoundId`, `submit_work(work, task_id) -> RoundId`, `pending_rounds() -> usize`,
  `async run_round(&mut self, &dyn RoundExecutor, work: RoundWorkItem) -> RoundOutcome`, `async run_until_idle(&mut self, &dyn RoundExecutor) -> LoopReport`,
  `async wait_round_completion(&self, timeout: Option<Duration>) -> RoundOutcome`,
  `push_follow_up(&self, impl Into<QueuedInput>)`, `push_steer(&self, impl Into<QueuedInput>)`, `drain_follow_up() -> Vec<QueuedInput>`, `has_follow_up() -> bool`, `queues() -> &LoopQueues`,
  `pause() -> Result<()>`, `resume() -> Result<()>`, `is_paused()`, `cancel()`, `is_cancelled()`, `request_abort()`, `is_aborted()`, `stop_reason() -> Option<&str>`, `iteration()`, `phase() -> InteractionPhase`, `is_running()`, `config()`, `config_mut() -> &mut TaskLoopConfig`.

## Constraints

- Depends only on `crate::schema` (plus std, tokio, tokio-util, async-trait, parking_lot, serde_json, tracing, `openjiuwen_common::error`); never on `crate::agent`, `crate::rails` or any live agent handle.
- `RoundExecutor` is object-safe and `Send + Sync`; the controller stores no agent, model or session.
- Cancellation via `tokio_util::sync::CancellationToken`; `parking_lot` guards are never held across `.await` (only brief read/insert/counter sections).
- Hooks fire exactly once per round in this order: `prepare_round` → `handle_input` → `handle_interaction` (0..n drained steering) → one of `handle_task_completion`/`handle_task_failed`/`on_abort` → `after_round`.
- A hook returning `Err` never panics and never escapes: the round becomes `RoundOutcome::failed("handler_error", message)`; an `after_round` failure is logged only.
- Executor errors become `RoundOutcome::failed("round_execution_error", message)`; the timeout code is `completion_timeout`, the abort code is `aborted` (`RoundOutcome::timeout()`/`abort()`).
- State is persisted through `StateSink` after every round when `config.persist_state` (sink errors are logged, never fatal); the controller performs no session IO itself.
- Tracing only; no stdout; no panics on user input; files ≤1000 lines.

## Design decisions

- The Python loop is a single `_run_task_loop` generator in `deep_agent.py`; the Rust port splits it into an explicit `TaskLoopController` driver plus the agent-supplied `RoundExecutor`, keeping the controller free of agent dependencies.
- Python's `TaskLoopController` has no parameters; `completion_timeout` (600 s) comes from `DeepAgentConfig` and the 50-round cap is a local in `_run_task_loop`. Both live in `TaskLoopConfig`, and `new` pushes `max_outer_rounds` into the coordinator so `stop_reason = "MaxOuterRounds"` is produced by the single budget authority.
- `LoopQueues` uses shared interior mutability (`Arc<Mutex<VecDeque>>`) instead of public `VecDeque` fields, because the handler/agent pushes steering while a round is running (Python shares one `asyncio.Queue` object by identity).
- `LoopCoordinator::should_continue` refreshes the caller's `StopEvaluationContext` from its own counters (iteration, token usage, elapsed, last result) before evaluating, so `extra` supplied by the caller survives while loop bookkeeping stays authoritative. `evaluate` is the evaluator-chain half of the same pass; call one per pass, never both (evaluators are stateful).
- The abort flag is a shared `AtomicBool` so a supervisor task can `request_abort()` through `&self`; abort is observed at the top of `should_continue` (stop reason `Aborted`) and applied after the active round.
- `EventManager` is a priority queue over `DeepLoopEvent` per the schema's dormant Python design (`schema/loop_event.py` has no Python consumer). Python's `harness/interaction/event_manager.py` (user/goal work queues with goal dedup) is a different component and is deliberately not ported here; work-item scheduling lives in the interaction supervisor, not in this module.
- `RoundExecutor::last_round_result`/`token_usage` are optional default methods: `RoundOutcome` carries no model payload, so the agent exposes the Python result dict (`result_type`/`output`) and cumulative token count through them; defaults keep the no-progress guard inert rather than misclassifying every successful answer as short.
- `run_until_idle` runs only while work is available (pending submissions or follow-ups), so stop evaluators are never evaluated twice for one round (which would double-count the no-progress streak).
- A completion timeout latches the coordinator abort (Python parity) and ends the loop with `stop_reason = "completion_timeout"`; a pending cancellation ends it with `aborted`.
- `pause`/`resume` are controller-level (absent from Python): `run_until_idle` stops cleanly between rounds with `LoopReport::paused = true`; both return `Error::validation` once the loop is `Terminated` (after `cancel`/abort).
- `resume_continuation` stamps `context["resume_continuation"] = true` on every round after the first in one `run_until_idle` call, mirroring Python's continuation metadata for post-interrupt rounds.
- Steering has two planes: `queues.steering` is the data plane the executor drains before each model call (identity-shared, never drained by this module), and `push_steer` also logs the input so the handler's `handle_interaction` hook fires exactly once per steering input.
- `stop_condition_state` in `DeepAgentState` is written after each round from `coordinator.state().get_state()`; restoring it remains the caller's choice (Python's task loop resets rather than resumes it).

## Routing Table

- `./queues.rs` → `LoopQueues`, `QueuedInput`
- `./coordinator.rs` → `LoopCoordinator`, `default_evaluators`, `DEFAULT_MAX_OUTER_ROUNDS`
- `./events.rs` → `EventManager`, `DEFAULT_EVENT_CAPACITY`
- `./controller.rs` → `TaskLoopController`, `TaskLoopConfig`, `LoopReport`, `RoundExecutor`, `RoundId`, `StateSink`
- `./handler.rs` → `TaskLoopEventHandler` trait, `DefaultTaskLoopEventHandler`
- `./mod.rs` → module docs and flat re-exports
- `../agent/` → owner of the `RoundExecutor` implementation and the only caller of `run_until_idle` (sibling — read-only, escalate writes to parent)

## Notes for `crate::agent`

- Build: `TaskLoopController::new(Box::new(handler), LoopCoordinator::with_defaults(&config.task_loop_no_progress_guard), private_queues.clone(), TaskLoopConfig::default().with_completion_timeout(...))` plus `.with_state(initial_state)` and `.with_state_sink(sink)` where the sink writes `SESSION_STATE_KEY`.
- Hand the agent its `LoopQueues` clone so steering can be drained before each model call and follow-ups can be injected.
- Implement `RoundExecutor` on the agent (or a thin adapter): `run_round` executes one ReAct turn for `work.query()` and returns `RoundOutcome::answer(outcome.next_work)`; also return `last_round_result()` (the Python result dict) and cumulative `token_usage()`.
- Drive one user turn with `submit_round(inputs, is_follow_up, task_id)` then `run_until_idle(&self)`; the report's `stop_reason`/`last_outcome` become the invoke result and `pending_follow_ups` persist through the state sink.
- `deepagent.event_handler`-style callbacks map onto `TaskLoopEventHandler`; `abort()` should call `controller.request_abort()` and/or `cancel()`.

## Test Strategy

- Unit tests beside each module with a fake `RoundExecutor` (records calls, can sleep, can fail, can return follow-ups) and a recording handler.
- `queues`: FIFO push/pop/drain, shared clone visibility, per-queue lengths.
- `coordinator`: OR evaluation and first-stop-wins, max outer rounds stop reason, abort wins immediately without evaluating, `observe_round` refresh of the context, `reset`, state round-trip including `NoProgressAnswer` streak.
- `events`: abort < steer < follow-up then seq, custom priorities, unbounded default, bounded eviction/rejection return values, `publish_event` monotonic seq, drain order.
- `controller`: hook order and exactly-once firing; executor error → failed; timeout → `completion_timeout` and loop stop; abort → `aborted` with `on_abort`; follow-up drained one per round; steering dispatch via `handle_interaction` while the queue keeps its copy; max-round stop; state sink called after every round with pending follow-ups and stop-condition state; pause/resume including terminated errors; `submit_round` round ids reaching hooks; `wait_round_completion` idle and timeout paths.
- Parity references (Python): `tests/unit_tests/harness/test_loop_coordinator.py`, `test_loop_queues.py`, `test_deep_agent_event_handler.py`, `test_deep_agent_event_executor.py`, `test_deep_agent_interaction.py`.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
