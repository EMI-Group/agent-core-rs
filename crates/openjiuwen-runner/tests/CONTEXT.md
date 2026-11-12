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
- `run_agent.rs` — `ScriptedModel`-driven `ReActAgent` through `AgentRef::Id` and `AgentRef::Agent`; owned session id from `conversation_id` and from `default_session_id`; caller session untouched (no commit); missing id (`NotFound`); callback firing order (`AgentInvokeInput` before `AgentInvokeOutput`); blocking input guardrail aborting the run; timeout and cancellation; streaming frame order and `AgentStreamOutput` per frame; owned-session checkpoint written via `post_run`.
- `run_workflow.rs` — start→LLM→end workflow through `WorkflowRef::Id` and instance; `WorkflowInvokeInput`/`Output`; streaming frames plus `WorkflowStreamInput`/`Output`; error mapping from a failing component.
- `run_team.rs` — a handoff team of two `ScriptedModel` agents: `run_agent_team` returns the final answer and tears down the team session (unbound, stream ended, committed); streaming forwards strategy frames and tears down on completion; caller-provided team session left bound; missing id (`NotFound`).
- `api_surface.rs` — compile-time pinning of the crate-root surface: every re-export, trait-object coercion (`AgentRef`/`WorkflowRef`/`TeamRef`/`MessageHandler`/`CallbackHandler`/`AgentSpawner`), `RunOptions` builder reachability, `CallbackEvent::ALL` length/round-trip, and serde round-trips of `RunnerConfig`/`QueueMessage`/`SpawnSpec`/`CallbackInfo`.

## Constraints

- Only the public API of `openjiuwen-runner` (plus `openjiuwen-test-support`, `openjiuwen-agent`, `openjiuwen-session`, `openjiuwen-workflow`, `openjiuwen-multiagent`, `openjiuwen-security` for fixtures).
- No network, no sleeps for ordering, no wall-clock assertions; timeouts are short and prove a deadline only.
- Each test builds its own `Runner::new()` — never `Runner::global()` except in the singleton suite, which calls `reset_global()`.
- Behavior ports cite the Python test file they mirror in a comment.
- `common/mod.rs` holds shared helpers (scripted ReAct agent, echo tool, linear workflow, two-agent handoff team, recording callback).

## Routing Table

- `./tests/common/mod.rs` → shared fixtures (scripted agents, workflow builder, handoff team, recording handler)
- `./tests/*.rs` → the suites listed above
- `../src/` → the implementation under test
