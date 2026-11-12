# tests/ — runner integration suites

## Intent

Behaviour-level tests for the crate's public seams: the registries, the callback framework, the guardrail binding, the in-process queue, the runner lifecycle, and the three execution paths (agent, workflow, team) with their streaming variants.
Every suite is offline and deterministic and drives only the public API, exactly as `openjiuwen-harness` and `openjiuwen-cli` will.

## API Surface

Suites (one file per area):
- `resources.rs` — register/get/remove/list/clear for tools/agents/workflows/teams; duplicate policies (`Fail`/`Keep`/`Replace`); card and name lookups; provider-per-get semantics; revision counters; `release`; concurrent registration; card JSON round-trip.
- `callbacks.rs` — event-name mapping over `CallbackEvent::ALL`; priority ordering (descending, insertion tiebreak); sync and async handlers; error isolation (`Err` recorded, later handlers still run); `Abort` and `Stop` verdicts; `once`; enable/disable; unregister by id/event/namespace/tag; no-subscriber fast path; per-event isolation.
- `security_binding.rs` — `bind_guardrails` registering four stages at priority 100 in namespace `guardrail`; a blocking prompt-injection finding on `LlmInvokeInput` aborting `trigger` with `Error::Security`; a non-blocking finding left in the report; `scan_stage_for`/`event_for_stage` round-trip; `Runner::scan` returning the pipeline outcome and its "no pipeline" error.
- `message_queue.rs` — publish/request happy paths; per-topic FIFO ordering; a request on an unsubscribed topic (`NotFound`) and one with no handler; handler errors returned to the requester; request timeout (`Timeout`) with the handler still running; publish before `start` (`Config`); start/stop idempotence; `unsubscribe` joining the worker; `create_message_queue` unsupported backends.
- `runner_lifecycle.rs` — `new`/`with_config`/accessors; `start`/`stop` idempotence; start builds the checkpointer and starts the queue; stop releases the registries; `release(session_id)`; `Runner::global()` identity plus `reset_global()` isolation; execution before `start` still works locally (Python parity: runs do not require `start`).
- `run_agent.rs` — local `ScriptedAgent` through `AgentRef::Id` and `AgentRef::of`, plus one real `ReActAgent`+`ScriptedModel` run; owned session id from `conversation_id`, from `RunOptions::session_id`, and from `default_session_id`; caller session untouched (no commit); missing id (`NotFound`); callback order and priority; `Abort` from an input handler; timeout and cancellation (pre-cancelled and mid-run); streaming frame order, one `AgentStreamOutput` per frame with 0-based `index`, terminal-error behaviour, owned-session checkpoint written via `post_run`.
- `run_workflow.rs` — start→echo→end workflow through `WorkflowRef::Id` and instance with `${echo.query}` passthrough; `WorkflowInvokeInput`/`Output`; created vs caller session (the injected checkpointer proves the created backend and id); a failing component returning the error and firing `WorkflowError`; streaming frames plus `WorkflowStreamInput`/`Output`.
- `run_team.rs` — a real `BaseTeam` + handoff strategy over two closure members: `invoke_team` returns the final answer and tears the owned lease down (runtime unbinding asserted); streaming forwards strategy frames verbatim and tears down on completion; caller-provided session reused and left bound; missing id (`NotFound`).
- `api_surface.rs` — compile-time pinning of the crate-root surface: every re-export, trait-object coercion (`AgentRef`/`WorkflowRef`/`TeamRef`/`MessageHandler`/`CallbackHandler`/`AgentSpawner`), `RunOptions` builder reachability, `CallbackEvent::ALL` length/round-trip, and serde round-trips of `RunnerConfig`/`QueueMessage`/`SpawnSpec`/`CallbackInfo`.

## Constraints

- Only the public API of `openjiuwen-runner` (plus `openjiuwen-test-support`, `openjiuwen-agent`, `openjiuwen-session`, `openjiuwen-workflow`, `openjiuwen-multiagent`, `openjiuwen-security` for fixtures).
- No network, no sleeps for ordering, no wall-clock assertions; timeouts are short and prove a deadline only.
- The execution suites drive the `openjiuwen_runner::run` free functions over a `RunServices` fixture; only lifecycle suites construct `Runner`, and never `Runner::global()` without `reset_global()`.
- Behavior ports cite the Python test file they mirror in a comment.
- `common/mod.rs` holds shared fixtures: `services()`, `services_with_checkpointer`, `services_with_default_session`, `services_with_checkpointer_and_default_session`, a local `ScriptedAgent` (delay/gate/failure/stream chunks/call counters) plus `scripted_agent(id, answer)` and `answer_frames()`, `EchoComponent`/`FailingComponent`, `passthrough_workflow(id)`/`failing_workflow(id)`, a real two-member handoff `BaseTeam` with a `TeamCalls` log (`handoff_team_fixture`, `team_calls`, `provider`), `RecordingSession`, `RecordingCheckpointer`, and the recording callback helpers (`recorder`, `record_at`, `record_all`, `recorded*`, `agent_events`, `workflow_events`).
- `common/mod.rs` is included per test binary with `mod common;`; it is private to this crate's test targets, so another crate (e.g. `openjiuwen-e2e`) cannot import it and must inline its own fakes or add its own `tests/common/mod.rs`.
- Only `runner_lifecycle.rs` exercises the `Runner` facade for execution (`run_agent_through_the_facade_without_start`); every execution-path suite drives the `openjiuwen_runner::run` free functions over `RunServices`, and there is no facade `run_workflow` example (workflow coverage is `invoke_workflow`/`stream_workflow` only).
- No suite reads or mutates the process environment: there is no `TempHomeGuard`, `EnvGuard`, `#[serial]`, or env mutex here, and none is needed because the runner never reads `OPENJIUWEN_*`.
- Tokio tests are plain `#[tokio::test]` (current-thread) except the two concurrency cases in `resources.rs`, which use `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]`; waits are `Notify` handshakes or short `RunOptions::with_timeout` deadlines, never sleeps for ordering.

## Routing Table

- `./tests/common/mod.rs` → shared fixtures (scripted agents, workflow builder, handoff team, recording handler)
- `./tests/*.rs` → the suites listed above
- `../src/` → the implementation under test
