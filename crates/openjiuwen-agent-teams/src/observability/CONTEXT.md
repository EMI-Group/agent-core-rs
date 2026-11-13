# observability

## Intent

In-process OTel-style span and hook seam for the team runtime — the Rust port of Python `openjiuwen.agent_teams.observability` (v0.1.20).
Python owns an OpenTelemetry `TracerProvider`, a process-global session-keyed root registry, an async monitor handler and a paired agent-tier rail; this port keeps the observable contract and drops the SDK.
One team run owns one `TeamSpan` trace, rooted at `team.{team_name}`; the runtime derives `member.{name}`, `round.{member}.{round}` and `task.{id}` children and emits typed events to registered hooks.
Everything is synchronous, infallible and free of global state: no network, no `Result`, no `println!`.

## API Surface

### `span.rs`

- [`SpanContext`](span.rs) — `{ trace_id, span_id }`, serde; `new`, `child` (keeps the trace, fresh span id).
- [`TeamSpan`](span.rs) — `{ name, context, parent: Option<SpanContext>, attributes: BTreeMap<String, Value>, started_at: u64 }`, serde; `root`, `root_for`, `child`, `with_attribute`, `attribute`, `trace_id`, `span_id`.
- `TeamSpan::root(name)` generates a fresh 32-hex trace id and 16-hex span id and records `team.name = name`; `root_for(team_name)` names the span `team.{team_name}` and adds `team.id` + `oj.agent.mode = "team"` (Python `span_context.get_or_create_team_span` parity).
- `TeamSpan::child(name)` is creation-time inheritance: it clones the parent's attribute map, sets `parent = Some(parent.context)`, keeps `trace_id` and takes a fresh span id.
- Semantic-convention constants: `AT_TEAM_NAME = "team.name"`, `AT_TEAM_ID = "team.id"`, `AT_MEMBER_NAME = "member.name"`, `AT_MEMBER_ROLE = "member.role"`, `AT_TASK_ID = "task.id"`, `GEN_AI_CONVERSATION_ID = "gen_ai.conversation_id"`, `OJ_AGENT_MODE = "oj.agent.mode"`.

### `hooks.rs`

- Event payloads (serde, snake_case, unknown fields tolerated): `MemberObservableEvent { team_name, member_name, role, status }`, `RoundObservableEvent { team_name, member_name, round, outcome }`, `TaskTransitionEvent { team_name, task_id, from, to, actor }`, `DelegationEvent { team_name, from_member, to_member, task_id, reason }`, `TurnMetrics { team_name, member_name, turn, input_tokens, output_tokens, tool_calls, duration_ms }`.
- `ObservabilityHooks: Send + Sync` — seven callbacks, every one with a no-op default body: `on_member_spawned`, `on_member_stopped`, `on_round_started`, `on_round_finished`, `on_task_transition`, `on_delegation`, `on_turn_metrics`.
- `NoopHooks` — unit struct with an empty impl (the "observability disabled" hook).
- `ObservableEvent` — internally tagged (`kind`: `member | round | task | delegation | turn`) wrapper over the five payloads; `ObservableEvent::kind()` is the payload-level tag.
- `RecordedEvent { kind: &'static str, span: TeamSpan, event: ObservableEvent }` — diagnostic snapshot; serializes but does not deserialize.
- `RecordingHooks` — `parking_lot::Mutex<Vec<RecordedEvent>>`; `new`/`Default`, `events`, `kinds`, `len`, `is_empty`, `clear`; records exactly one entry per callback in call order with callback-specific `kind` (`member_spawned`, `member_stopped`, `round_started`, `round_finished`, `task_transition`, `delegation`, `turn_metrics`).
- `TeamObservability` — the facade the runtime touches: `new`, `with_hooks`, `add_hook`, `hooks`, `team_name`, `start_team`, `root_span`, `finish_team`, `member_span`, `round_span`, `task_span`, and the seven `on_*` fan-out methods.
- `TeamObservability::start_team(Some(session))` records `gen_ai.conversation_id`; `finish_team()` drops the stored root and is a no-op when none was started.

### `otel.rs` (feature `otel` only)

- `SpanRecord { name, trace_id, span_id, parent_span_id, attributes, timestamp }` — flattened snapshot cached by the sink; `timestamp` copies the span's `started_at`.
- `OtelSpanSink` — bounded `VecDeque<SpanRecord>` behind a mutex: `new(capacity)`, `record`, `records`, `len`, `is_empty`, `clear`, `evicted`; positive capacity is a ring, capacity `0` retains nothing and counts every record as evicted, `clear` leaves the cumulative counter intact.
- `OtelHooks` — `new(Arc<OtelSpanSink>)`, `sink`; every callback emits one structured `tracing::info!` (trace/span identity plus team, member, task/status/round/metrics fields, `outcome` as the Python wire string) and appends one `SpanRecord` named `member.spawned`, `member.stopped`, `round.started`, `round.finished`, `task.transition`, `delegation` or `turn.metrics`.

### `mod.rs`

- Re-exports every public item above; `#[cfg(feature = "otel")]` additionally exports the sink and hooks.
- `pub const OTLP_EXPORT_AVAILABLE: bool = false;` in both configurations.

## Attribute conventions

| Attribute | Set by | Value |
| --- | --- | --- |
| `team.name` | `root`, `root_for`, standalone span fallback | team name |
| `team.id` | `root_for` | team name (Python sets the same value) |
| `oj.agent.mode` | `root_for` | `"team"` |
| `gen_ai.conversation_id` | `TeamObservability::start_team` | session id, only when `Some` and non-empty |
| `member.name` | `member_span`, `round_span` | member name |
| `member.role` | `member_span` | `TeamRole::as_str()` |
| `task.id` | `task_span` | task id |

- A child inherits the parent's attributes at creation; its own `with_attribute` calls win on the child only and never mutate the parent.
- There is no lazy parent-chain walk: a span handed to a hook is self-contained, `Send + Sync` through `Arc`/clone, and cheap.
- Keep the constants stable — dashboards, trajectory consumers and tests pin these exact strings.

## Runtime wiring (emission points)

| Call | When the runtime should call it |
| --- | --- |
| `TeamObservability::start_team(Some(session_id))` | once per team run, after the team session id is known and before the leader/teammate loop starts |
| `finish_team()` | in the run's `finally`/`stop_team` path, after every member span has been emitted |
| `member_span(name, role)` + `on_member_spawned` | after the member row reaches `starting`/`ready` and its runtime is live |
| `on_member_stopped` | after the member row reaches a terminal or stopped status |
| `round_span(name, round)` + `on_round_started` | when `MemberObserver::on_round(Started)` fires |
| `on_round_finished` | on `Finished`, `Aborted` or `Failed` |
| `task_span(task_id)` + `on_task_transition` | after a successful `TeamStore::transition_task`/`update_task`, with the previous and new status |
| `on_delegation` | after dispatch has written the task assignment |
| `on_turn_metrics` | when the member runtime reports the end of one turn |

- Callbacks are observational only: the trait is infallible, so a hook cannot fail a run; keep hook bodies short because fan-out is synchronous.
- A runtime that cannot flow a facade into a teammate task should share the same `Arc<TeamObservability>` — there is no ambient or process-global lookup.

## Constraints

- No OpenTelemetry SDK, no exporter, no network: only serde/serde_json, `parking_lot`, `tracing`, `uuid`, `chrono` from the crate's existing dependencies.
- The module is infallible — it deliberately does not use `openjiuwen_common::error`; do not introduce `Result` for API shape.
- No `unsafe`, no `println!`/`eprintln!`; diagnostics go through `tracing`.
- No undocumented global mutable state: the root span lives on the facade, never in a process-wide registry.
- `--no-default-features` compiles only the span/hook seam; `otel` adds `otel.rs` and nothing else changes in the baseline.
- Files stay under the workspace ~1000-line baseline; every public item carries a `///` doc and pure getters are `#[must_use]`.

## Design Decisions

- **Session-scoped root, no global registry.** Python keys the team root by session id in a module-level dict so in-process teammates can find it; here the root belongs to one `TeamObservability` value that the runtime passes to its members. This removes process-global state and makes two teams in one process trivially independent; the cost is that cross-task discovery needs an explicit `Arc`, documented in Runtime wiring.
- **Creation-time attribute inheritance.** Copying the map at `child()` time keeps spans plain data (`Send`, `Sync`, no lifetimes, no lock on read) and makes "the child overrode a key" local and testable; Python relies on live `Span` handles instead.
- **Fan-out ordering.** Each `on_*` clones the `Arc` hook list, then calls hooks synchronously in registration order without holding the root lock, so a hook may re-enter the facade and registration order is the only ordering rule.
- **Feature split without an SDK dependency.** `otel` compiles only structured `tracing` events plus a bounded in-memory sink; real OTLP export would need the `opentelemetry` stack, and `openjiuwen-extensions` (which owns it) is a same-layer crate this crate may not depend on. `OTLP_EXPORT_AVAILABLE` therefore stays `false` in every build.
- **Ids.** `uuid` v4 supplies a 32-hex trace id and the first eight bytes give a 16-hex span id, matching the OTel id shapes Python produces; `SpanContext::new` trusts caller-supplied ids while generated ids are always the right shape.
- **Root attributes.** `root_for` also stamps `team.id = team_name`, mirroring Python's `get_or_create_team_span`, even though `team.name` alone is enough to identify the team.
- **Round outcome wire strings.** `RoundOutcome` has no `as_str` in `crate::schema` (read-only), so `otel.rs` carries a private `round_outcome_str` helper and a test pins it to the serde wire values.
- **Owned span data.** Hooks receive `&TeamSpan`, never a live span handle; recording hooks clone it. This is what makes the seam testable offline and independent of any tracer implementation.

## Test Strategy

34 inline `#[cfg(test)]` tests, all deterministic and offline; no `#[ignore]`, no sleeps, no wall-clock assertions (33 run under `--features otel` because one `mod.rs` test is `cfg(not(feature = "otel"))`).

- `span.rs` (10): `root_generates_32_and_16_lowercase_hex_ids`, `root_for_sets_python_parity_attributes`, `root_name_is_the_team_name_attribute`, `child_keeps_trace_and_gets_fresh_span_id`, `child_inherits_attributes_at_creation_time`, `child_override_does_not_touch_parent_attributes`, `with_attribute_replaces_and_absent_key_is_none`, `span_and_context_round_trip_through_json`, `span_tolerates_unknown_and_missing_fields`, `context_child_keeps_trace_and_regenerates_span`.
- `hooks.rs` (13): `recording_hooks_preserve_call_order_and_counts`, `observable_event_kinds_are_payload_level`, `two_recording_hooks_see_the_same_sequence`, `noop_hooks_are_silent`, `facade_without_hooks_never_panics`, `start_team_stores_root_and_records_session_id`, `finish_team_clears_the_root`, `member_span_is_a_child_of_a_live_root`, `member_span_without_root_is_standalone_with_team_name`, `round_and_task_spans_carry_names_and_attributes`, `add_hook_appends_in_registration_order`, `payloads_round_trip_through_json`, `observable_event_round_trips_with_internal_tag`.
- `mod.rs` (2): `otlp_export_is_never_available` (both configs), `facade_records_events_without_the_otel_feature` (`cfg(not(feature = "otel"))`).
- `otel.rs` (9): `sink_retains_oldest_first_up_to_capacity`, `capacity_zero_retains_nothing_and_counts_every_record`, `record_copies_identity_attributes_and_start_time`, `clear_drops_records_but_keeps_the_evicted_counter`, `span_record_round_trips_and_tolerates_unknown_fields`, `span_record_omits_absent_parent_and_keeps_a_present_one`, `round_outcome_str_matches_the_python_wire_values`, `hooks_record_one_span_per_callback`, `hooks_wired_through_the_facade_record_spans`.
- Definition of done (all green): `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --offline --lib` (with and without `--no-default-features` and `--features otel`), and `cargo clippy -p openjiuwen-agent-teams --all-targets --offline [--features otel] -- -D warnings`.

## Routing Table

- `./span.rs` → span identity, attribute constants, creation-time attribute inheritance.
- `./hooks.rs` → event payloads, `ObservabilityHooks`, `NoopHooks`/`RecordingHooks`, `TeamObservability` facade and fan-out.
- `./otel.rs` → feature-gated structured `tracing` events and the bounded in-memory `SpanRecord` sink.
- `./mod.rs` → module doc, re-exports and `OTLP_EXPORT_AVAILABLE`.

## Deferred

- Real OTLP export, the `opentelemetry` SDK, `TracerProvider`/span processors and exporter configuration — offline dependency policy plus the same-layer `openjiuwen-extensions` ownership; the feature provides the testable seam instead.
- Python `monitor_handler.py`'s long-lived `task.{id}` spans, `msg.*` and `event.*` spans, reflection fallback over `TeamEvent` types and `_task_spans` recovery — the runtime reports task transitions through `on_task_transition` instead.
- Python `rail.py`'s agent-tier decoration and the `agent.{member}.task_iteration.N` / `llm.call` / `tool.*` span stack — that belongs to the harness agent rail; this port exposes `TurnMetrics` and member spans.
- Python `setup.py`'s process-global tracer demand bookkeeping (`init/acquire/release/shutdown`), the session-keyed root registry, `attach_to_team_agent` and the ambient `set_ambient_team_span` fallback — replaced by a facade-local root.
- Span status/end-time/forced-close APIs, span events, sampling and redaction of `openjiuwen.span.output` — no live span object exists here.
- `rail.py`'s reliability hooks (retry/health remediation) — a separate concern from observation and not ported.

## Notes for Agents

- Do not add a process-global registry or session-keyed root map; pass an `Arc<TeamObservability>` to every member that must emit.
- To add a new observation: add the payload struct, a default trait method, the facade fan-out method, the `RecordingHooks` kind string, the `OtelHooks` `tracing::info!` + `sink.record` call, and one test per layer.
- Renaming an `AT_*` constant or span-name format is a breaking change for consumers that pin exact strings.
- `RecordedEvent` intentionally does not implement `Deserialize` (its `kind` is `&'static str`); record through `RecordingHooks` instead.
