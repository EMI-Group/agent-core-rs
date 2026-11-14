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
- `RecordingToolHost` — `ToolHost` that records mounted tools in mount order (`mounts()`, `card_count()`, and the handles through `tools()` / `tool(name)`).
- `StreamAgent` — `Agent` with a scripted `stream` (`StreamFrame::Chunk` / `StreamFrame::Fail`) that records the inputs of every stream call (`seen()`); `invoke` answers `""`.
- `counting_provider(member, calls)` — `AgentProvider` that hands out the same `TeamAgent` and counts resolutions.
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
- `./hierarchical.rs` → hierarchical strategies: msgbus supervision (readiness, single hop, deadline, one-frame stream), agents-as-tools wiring and pinning (`RecordingToolHost` mounts, mounted-tool dispatch, root stream shaping, error frame), and the `SupervisorAgent` factory (validation, card, end-to-end dispatch, model-visible tool cards, parallel budget).
- `./api_surface.rs` → compile-time pinning of every crate-root re-export and trait-object coercion, including the hierarchical configs, strategies, team builders, and supervisor factory, their serde round-trip and `schemars` property names, the two strategy `kind()` strings, and (`new_*_are_re_exported_at_the_root`) the extended subsystems published at the root: events, autosubscribe, pyrepr, resources, member checkpoints, transports, and the P2P ability manager — plus its Python-parity `hierarchical::P2PAbilityManager` path.
- `./events.rs` → typed team-event bus: wire names and Python parity constants, filtered delivery, unsubscribe, bounded history, serde/schema contracts, concurrent emission.
- `./autosubscribe.rs` → card-driven subscription derivation (broadcast, directed tags, overrides) and the three reconciliation modes against the runtime registry.
- `./pyrepr.rs` → `parse_python_literal`/`literal_or_json` parsing and `render_python_value` rendering, including the depth cap.
- `./member_checkpoint.rs` → member-scoped keys/ids and per-member save/restore over one shared `MemoryCheckpointer`.
- `./resources.rs` → `LocalResourceMgr` inherent surface and `TeamResourceMgr` trait-object defaults, duplicates, listings, revision and `clear`.
- `./transport.rs` → `TeamTransport` in-process delegation (ordering, fan-out, subscribers, deadlines, introspection) and backend selection (`Pulsar` `Unsupported`, Redis-without-settings `Config`).
- `./transport_redis.rs` → the Redis Streams backend over a loopback fake RESP server with a stream + group + SET registry state machine: cross-instance round-trip, one delivery per consumer group, reconnect/claim, poison-entry ack, pub/sub fan-out across instances, registry refresh convergence, write-failure resilience, `stop` cleanup, `Cluster`/`Sentinel` `Unsupported`, and the "no list commands" assertion.
- `./routing.rs` → `GroupRegistry`, `broadcast_targets`, `broadcast_to_group` (metadata stamp, skipped members, isolated failures, one group event), `RoutingTable` resolution and serde.
- `./inbox.rs` → `InboxFilter` rules, `InboxFilterRegistry` default-accept, and runtime P2P/pub-sub enforcement.
- `./communicable.rs` → `CommunicableAgent` binding lifecycle and an in-process messaging round trip.
- `./ability.rs` → `P2PAbilityManager` batch dispatch: P2P vs local routing, input-order results, per-call failures, parallel cap.

## Test Strategy

- `tests/handoff.rs` (30 tests) is grouped by section comments: route graph (3), coordinator semantics (7), save/restore (2), signal extraction (5), `HandoffTool` (3), request round-trip (2), then end-to-end chains (8) covering direct handoff, hop cap, unknown target, chain timeout, interrupt, tool injection, streaming, and resume.
- A caller-provided `Arc<AgentTeamSession>` must be bound with `runtime.bind_team_session(..)` before `Team::invoke`/`Team::stream`; a caller session is only end-streamed, never unbound.
- Chains run one member at a time; each hop writes one `{"type":"message","payload":<AgentOutput::to_value()>}` frame to the team stream.
- Timeout cases use `TeamConfig::configure_timeout(0.02)` against a 200 ms `AsyncFakeAgent`, so the deadline wins by an order of magnitude and the test never depends on scheduler timing.
- Resume cases seed `__handoff_coordinator__`/`__handoff_history__` on the caller session; the interrupted entry (`result_type == "interrupt"`) is dropped and the valid entry is forwarded.
- `tests/hierarchical.rs` msgbus group pins the readiness errors (`ErrorKind::Config`, exact texts), the verbatim input of the single P2P hop, the child session's `parent_id()` (`conversation_id`), the deadline adoption on supervisor registration, the exactly-one `{"type":"message","index":0,"payload":{"output":..}}` stream frame, and the per-invoke `ErrorKind::Timeout`.
- `tests/hierarchical.rs` agents-as-tools group pins the mounts on `RecordingToolHost` in member order, the single provider resolution plus pinning across two invocations, mounted-tool dispatch through the runtime, the shaped root stream input (`conversation_id`/`sender` injected, scalars wrapped under `query`), the root readiness errors, and the single `result_type: "error"` frame after an agent failure.
- `tests/hierarchical.rs` supervisor group pins the empty-agent validation, the returned card round-trip, one end-to-end tool-call run over the bus, the model-visible tool cards (`function.name`/`description`/`parameters`, empty-object fallback), and that `max_parallel_sub_agents` `0` and `1` both dispatch.
