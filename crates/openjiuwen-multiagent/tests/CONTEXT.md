# openjiuwen-multiagent :: tests

## Intent

Crate-level integration suites for `openjiuwen-multiagent`.
Each file is its own test crate and drives the public crate-root API only, so a suite fails when a re-export or a wire contract changes.
Everything is offline and deterministic: members are in-process closures, sessions are local, and the only timers are explicit short deadlines.

## API Surface

`common/mod.rs` holds the shared fakes (`#![allow(dead_code)]`, since each suite compiles it separately):

- `FakeAgent` — sync-closure `Agent`; `FakeAgent::new(card, handler)` and `FakeAgent::echoing(name)`.
- `AsyncFakeAgent` — `BoxFuture`-closure `Agent` for delays (`AsyncFakeAgent::delayed(name, delay, text)`).
- `handoff_output(text, target, message, reason)` — `AgentOutput` carrying a handoff directive as an `Observation` (a plain `AgentOutput::answer` has no signal key).
- `provider(agent)` / `host_provider(agent, host)` — `AgentProvider` builders (`TeamAgent::new` / `TeamAgent::with_tool_host`).
- `RecordingToolHost` — `ToolHost` that records mounted tool-card names in mount order (`mounts()`, `card_count()`).
- `agent_card(name)` — card with `id == name == name`.
- `RecordingHandler` — `MessageHandler` recording P2P/pub-sub deliveries with scripted replies.

## Constraints

- Tests may `unwrap`/`expect`/`panic`; library code never does.
- No network, no sleeps outside the timeout cases, no reliance on wall-clock ordering.
- Every suite uses distinct team names and `conversation_id`s: session ids and bus topics are process-global per runtime.
- Helpers are shared only through `common/mod.rs`; never duplicate a fake in a suite.

## Routing Table

- `./team_card.rs` → `TeamCard`/`EventDrivenTeamCard`/`TeamConfig` wire shape and defaults.
- `./message_bus.rs` → bus FIFO ordering, fan-out, timeouts, cleanup, shutdown.
- `./runtime.rs` → runtime registration, provider binding, P2P session construction, empty-response error.
- `./team.rs` → `BaseTeam` member management, validation, custom strategies, start/stop idempotence.
- `./streaming.rs` → `SessionLease`/`team_stream` frame order and teardown rules.
- `./agent_tool.rs` → `AgentTool` card shape, dispatch, failures, concurrency limiter.
- `./handoff.rs` → handoff strategy: route graph, coordinator semantics, save/restore, signal locators, `HandoffTool`, request wire form, end-to-end chains, streaming, resume.
- `./api_surface.rs` → compile-time pinning of every crate-root re-export and trait-object coercion.

## Test Strategy

- `tests/handoff.rs` (30 tests) is grouped by section comments: route graph (3), coordinator semantics (7), save/restore (2), signal extraction (5), `HandoffTool` (3), request round-trip (2), then end-to-end chains (8) covering direct handoff, hop cap, unknown target, chain timeout, interrupt, tool injection, streaming, and resume.
- A caller-provided `Arc<AgentTeamSession>` must be bound with `runtime.bind_team_session(..)` before `Team::invoke`/`Team::stream`; a caller session is only end-streamed, never unbound.
- Chains run one member at a time; each hop writes one `{"type":"message","payload":<AgentOutput::to_value()>}` frame to the team stream.
- Timeout cases use `TeamConfig::configure_timeout(0.02)` against a 200 ms `AsyncFakeAgent`, so the deadline wins by an order of magnitude and the test never depends on scheduler timing.
- Resume cases seed `__handoff_coordinator__`/`__handoff_history__` on the caller session; the interrupted entry (`result_type == "interrupt"`) is dropped and the valid entry is forwarded.
