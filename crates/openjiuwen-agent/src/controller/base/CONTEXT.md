# openjiuwen-agent controller base

## Intent

The controller turn loop: a bounded event queue, intent recognition, planning, sequential task dispatch, output aggregation, and session-persisted task state.
Ports the driver half of Python `openjiuwen/core/controller/base.py` (`invoke`, `stream`, `bind_session`, `add_task_executor`, `set_event_handler`, `_ensure_started`) onto a single-turn in-process loop.

## API Surface

- `Controller` — `card`, `config`, `async invoke(inputs, session) -> Result<ControllerOutput>`, `stream(inputs, session) -> BoxStream<'static, Result<ControllerOutputChunk>>`.
- `BaseController` — `Arc<Inner>` handle (`Clone`): `new`, `with_recognizer`, queue accessors, recognizer/handler/router/planner setters, executor registry accessors, `registry`/`scheduler`, `start`/`stop`/`is_running`, `bind_session`/`unbind_session`/`bound_sessions`, `save_state`/`load_state`/`tasks`.
- `EventHandler` / `EventHandlerOutcome` — per-event override seam (`Intents`, `Output`, `Ignored`).
- `TASKS_STATE_KEY` — `"__controller_tasks__"`; state is `{"tasks": [<Task>...], "next_seq": <u64>}`.

## Constraints

- `invoke` always resolves a session via `crate::run::resolve_session` (caller session borrowed, otherwise a fresh `AgentSession` is created and finished), loads state, runs one turn, saves state, unbinds the session.
- Nothing in the turn returns `Err` for a missing executor, a failing executor, a timeout, or an unknown task target: those become failed tasks/outputs.
- `stream` must emit a first `processing` chunk before running the turn, and exactly one terminal `last_chunk` chunk (unless `suppress_completion_signal`).
- `Inner`/`TurnState` fields are `pub(super)` only so `state.rs`/`turn.rs` can share them; nothing leaks outside this module tree.

## Notes for Agents

- `base/turn.rs` owns all turn sequencing; `base/state.rs` owns persistence; `base/mod.rs` owns the public types and the accessors. Put new loop behaviour in `turn.rs`, new persisted fields in `state.rs`.
- Aggregation precedence is `InputRequired` > `Completed` > `Failed`; completed texts join with `\n\n`; `default_response.text` fills an empty answer; `data.tasks` lists per-task status/text.
- The router is consulted once per turn and only when at least two executors are registered.
- Restored tasks carry `created_seq = 0` (it is not serialized); the planner sequence comes from the persisted `next_seq`.
- `base/fixtures.rs` holds the shared test doubles (stub executors/handler/recognizer, session helper); `base/tests.rs` holds the routing/aggregation/streaming tests.

## Routing Table

- `./mod.rs` → `Controller`, `BaseController`, `EventHandler`, accessors, lifecycle, queue tests
- `./turn.rs` → `impl Controller`, event processing, intent routing, dispatch, aggregation, streaming
- `./state.rs` → `save_state`/`load_state`/`tasks`, the `__controller_tasks__` codec
- `./tests.rs` → routing, persistence, streaming, handler, signal-event tests
- `./fixtures.rs` → shared test doubles
