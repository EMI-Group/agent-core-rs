# observability

## Intent

In-process OTel-style span and hook seam for the team runtime — the Rust port of Python `openjiuwen.agent_teams.observability` (v0.1.20).
Python owns an OpenTelemetry `TracerProvider`, a process-global session-keyed root registry, an async monitor handler and a paired agent-tier rail; this port keeps the observable contract and drops the SDK.
One team run owns one `TeamSpan` trace, rooted at `team.{team_name}`; the runtime derives `member.{name}`, `round.{member}.{round}`, `task.{id}` and `memory.{operation}` children and emits typed events to registered hooks.
The `memory.*` branch comes from `MemorySpanObserver`, the bridge onto the team-memory observer seam (`openjiuwen_memory::team::TeamMemoryObserver`), so team memory is visible inside the same trace as the members and tasks.
The opt-in `otlp` feature adds a second bridge, `OtlpMemoryObserver`, which forwards the same events into a real `OpenTelemetry` pipeline owned by `openjiuwen-extensions`.
Everything except that OTLP bridge is synchronous, infallible and free of global state: no network, no `Result`, no `println!`.

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

### `memory.rs`
- `MemorySpanObserver` — the team-memory bridge: `Debug + Clone`, `Send + Sync`, implements `openjiuwen_memory::team::TeamMemoryObserver` (`fn observe(&self, &TeamMemoryEvent)`).
- `MemorySpanObserver::new(team_name)` builds the root with `TeamSpan::root_for(team_name)`, so each observer mints its own trace; `root()` and `trace_id()` expose it, and `memory_observer(team_name)` is the free-function shortcut.
- `observe` derives one child span named `memory.{operation}` per event via `TeamSpan::child` (same trace, root as parent, root attributes inherited at creation) and stamps the event identity on top; the child span is built by the private `span_for`.
- `#[cfg(feature = "otel")] MemorySpanObserver::with_sink(Arc<OtelSpanSink>)` makes every observed event also append one `SpanRecord` named `memory.{operation}` to the sink; the baseline build has no sink and no `with_sink`, so it records the `tracing` event only.
- Every event emits one structured `tracing` event at target `openjiuwen_agent_teams::memory`: `debug!` on `TeamMemoryOutcome::Ok`, `warn!` on `Failed`, both carrying trace/span id plus `team`, `operation`, `outcome`, `count`, `member_id`, `kind` and (on failure) `error` — the same fields the span carries.
- Attribute constants (`"team_id"`, `"operation"`, `"outcome"`, `"count"`, `"member_id"`, `"kind"`, `"error"`): `AT_MEMORY_TEAM_ID`, `AT_MEMORY_OPERATION`, `AT_MEMORY_OUTCOME`, `AT_MEMORY_COUNT`, `AT_MEMORY_MEMBER_ID`, `AT_MEMORY_KIND`, `AT_MEMORY_ERROR`.
- `observe` is infallible and non-blocking: no `.await`, and the only lock is the short one inside the sink.

### `otlp.rs` (feature `otlp` only)

- `OtlpMemoryObserver` — the OTLP bridge: `Debug + Clone`, `Send + Sync`, implements `openjiuwen_memory::team::TeamMemoryObserver`; it holds `Arc<openjiuwen_extensions::otel::OtelTraceSink>` plus an optional session id.
- Constructors: `OtlpMemoryObserver::new(Arc<OtelTraceSink>)` (no session id, so the sink's own fallback applies), `OtlpMemoryObserver::with_sink(Arc<OtelTraceSink>)` (an alias of `new`, named after `MemorySpanObserver::with_sink`), and `otlp_memory_observer(session_id) -> openjiuwen_common::error::Result<OtlpMemoryObserver>` (delegates to `OtelTraceSink::with_session`, i.e. a private in-memory provider that never reaches the network). Getters: `sink()`, `session_id()`.
- `observe` maps each `TeamMemoryEvent` to one **component** frame — `TraceFrame::workflow(TraceSpan { kind: TraceSpanKind::Component, .. })` — and calls `TraceSink::emit` on the sink; a sink whose `is_recording()` is `false` (shut-down handle, or extensions built without its `otel` feature) is a silent no-op apart from one `debug!` at the shared `openjiuwen_agent_teams::memory` target.
- Frame shape (private `span_for`): `name = memory.{operation}`; `invoke_id = memory.{operation}.{16 hex}` from an FNV-1a digest of `team_id` + `operation` + `member_id` (a retry reuses the id, a different operation/member/team does not); `trace_id` = 32 lowercase hex from `team_id`; `session_id` = the observer's session id or empty; `status` = `Finish` for `Ok` / `Error` for `Failed`, with the failure text in `error`; `end_time = start_time` and `elapsed_time = Some(0)` (the facade reports no timing); `component_id` = the invoke id, `component_name` = `memory.{operation}`, `component_type = "MemoryOperation"`; `parent_invoke_id` stays `None`.
- `meta_data` is exactly the `memory.rs` attribute bag, reusing `AT_MEMORY_TEAM_ID` / `AT_MEMORY_OPERATION` / `AT_MEMORY_OUTCOME` / `AT_MEMORY_COUNT` plus `AT_MEMORY_MEMBER_ID` / `AT_MEMORY_KIND` / `AT_MEMORY_ERROR` only when present and non-empty; the extensions sink serializes it as the `openjiuwen.meta_data` attribute.
- Routing: `OtelTraceSink::emit` dispatches on the span kind, so a component frame takes the workflow handler's generic component branch — component identity, trace/session identity, metadata blob, plain finish/error status, no `gen_ai.*` attribute. Exported OTel span name is that handler's `component.{invoke_id}`; the frame's own name travels as `openjiuwen.workflow.component.name`.
- Each frame has no parent span context (the in-process `TeamSpan` tree is plain data), so the sink starts it as its own OTel trace root; the derived team trace id is carried as `openjiuwen.trace.id` rather than installed as the OTel trace id.

### `mod.rs`

- Re-exports every public item above; `#[cfg(feature = "otel")]` additionally exports the sink and hooks, and `#[cfg(feature = "otlp")]` exports `OtlpMemoryObserver` + `otlp_memory_observer`.
- `pub const OTLP_EXPORT_AVAILABLE: bool = false;` in every configuration: it states the crate-wide baseline (no exporter), and the `otlp` feature upgrades only the memory bridge.

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
| `team_id` | `MemorySpanObserver::span_for` | the memory event's `team_id` (its own scope id, distinct from `team.id`) |
| `operation` | `span_for` | `TeamMemoryOperation::as_str()` (`search`, `write`, `forget`, `clear`, `prune`, `extract`) |
| `outcome` | `span_for` | `TeamMemoryOutcome::as_str()` (`ok`, `failed`) |
| `count` | `span_for` | records touched, as a JSON number |
| `member_id` | `span_for` | the event's `member_id`, only when `Some` and non-empty |
| `kind` | `span_for` | `TeamMemoryKind::as_str()`, only when `Some` and non-empty |
| `error` | `span_for` | the failed event's error text, only when `Some` and non-empty |

- A child inherits the parent's attributes at creation; its own `with_attribute` calls win on the child only and never mutate the parent.
- There is no lazy parent-chain walk: a span handed to a hook is self-contained, `Send + Sync` through `Arc`/clone, and cheap.
- `otlp.rs` reuses the same `AT_MEMORY_*` keys verbatim (in the frame's `meta_data`) and adds no attribute of its own; the extensions sink owns every `openjiuwen.*`/`gen_ai.*` key the exported span carries.
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
| `TeamMemory::in_memory().with_observer(Arc::new(memory_observer(team_name)))` (plus `.with_sink(sink)` under `otel`) | once per team run, when the team-memory facade is built — the memory wiring in `../memory/wiring.rs` is the natural build site |
| `TeamMemory::in_memory().with_observer(Arc::new(OtlpMemoryObserver::new(sink)))` under `otlp` | the same build site, when the host wants the memory events in its own `OTel` pipeline instead of (or in addition to) the in-process spans; `sink` comes from the host's `TracerHandle` (`OtelTraceSink::new(&handle)`) or from `otlp_memory_observer(session_id)` for a private in-memory provider |

- Callbacks are observational only: the trait is infallible, so a hook cannot fail a run; keep hook bodies short because fan-out is synchronous.
- A runtime that cannot flow a facade into a teammate task should share the same `Arc<TeamObservability>` — there is no ambient or process-global lookup.

## Constraints

- No OpenTelemetry SDK, no exporter, no network in the baseline build: only serde/serde_json, `parking_lot`, `tracing`, `uuid`, `chrono` from the crate's existing dependencies.
- The `otlp` feature is the one exception: it adds the optional `openjiuwen-extensions` dependency (`features = ["otel"]`), which owns the SDK and the exporters, and `otlp.rs` reaches it only through that crate's public `otel` surface (`OtelTraceSink`; the tests also use `init_otel_tracer`/`TracerHandle`/`CaptureExporter` to assert the exported spans offline).
- The span/hook seam is infallible — it deliberately does not use `openjiuwen_common::error`; only the fallible OTLP constructor (`otlp_memory_observer`) returns `Result`, and it never introduces an error path into `observe`.
- No `unsafe`, no `println!`/`eprintln!`; diagnostics go through `tracing`.
- No undocumented global mutable state: the root span lives on the facade, never in a process-wide registry, and the extensions sink owns a private provider (no global tracer is installed).
- `--no-default-features` compiles the span/hook seam plus the memory bridge; `otel` adds `otel.rs` and `MemorySpanObserver::with_sink`; `otlp` adds `otlp.rs` and its exports, and neither feature changes the other's build.
- Files stay under the workspace ~1000-line baseline; every public item carries a `///` doc and pure getters are `#[must_use]`.

## Design Decisions

- **Session-scoped root, no global registry.** Python keys the team root by session id in a module-level dict so in-process teammates can find it; here the root belongs to one `TeamObservability` value that the runtime passes to its members. This removes process-global state and makes two teams in one process trivially independent; the cost is that cross-task discovery needs an explicit `Arc`, documented in Runtime wiring.
- **Creation-time attribute inheritance.** Copying the map at `child()` time keeps spans plain data (`Send`, `Sync`, no lifetimes, no lock on read) and makes "the child overrode a key" local and testable; Python relies on live `Span` handles instead.
- **Fan-out ordering.** Each `on_*` clones the `Arc` hook list, then calls hooks synchronously in registration order without holding the root lock, so a hook may re-enter the facade and registration order is the only ordering rule.
- **Feature split without a baseline SDK dependency.** `otel` compiles only structured `tracing` events plus a bounded in-memory sink. The real `opentelemetry` stack belongs to `openjiuwen-extensions`, which is layer 5 while this crate is layer 6 — so depending on it is legal (the edge points downward and extensions never depends on this crate), and the opt-in `otlp` feature does exactly that for the memory bridge alone. `OTLP_EXPORT_AVAILABLE` therefore still reports `false`: it describes the crate-wide baseline, where the member/round/task spans and the default memory bridge stay on the in-memory/`tracing` seam.
- **OTLP bridge as a second `TeamMemoryObserver`.** Instead of teaching `MemorySpanObserver` about exporters, `otlp.rs` implements the engine's trait again and forwards frames into `OtelTraceSink`. The two bridges therefore compose (a host may attach either) and the baseline keeps zero new dependencies; the cost is that the frame shape lives in both modules, pinned by tests on each side.
- **Component/workflow frame, not an agent frame.** `OtelTraceSink::emit` dispatches on the span kind; a memory operation is a leaf unit of work with no LLM/tool/retrieval semantics, which is the workflow handler's generic component branch. An agent frame would have to claim an invoke type (`prompt`, `chain`, …) the operation is not, and the exported span would gain `gen_ai.*` attributes. `component_type = "MemoryOperation"` matches none of the handler's classifiers, so no such attribute is invented.
- **Deterministic ids instead of a registry.** `invoke_id` and `trace_id` are FNV-1a digests (`std::hash`'s `RandomState` is explicitly not stable across processes), so a retried operation reuses its span id and one team scope keeps one derived trace id with no lookup table. Two identical events in one run still export two spans: each frame is terminal and self-contained, so the sink opens and closes it independently.
- **No parent span context.** The in-process `TeamSpan` tree is plain data, not an OTel span, so there is nothing to parent onto; the sink starts each memory frame as its own OTel trace root and the derived team trace id travels as `openjiuwen.trace.id`. That is the honest limit of this bridge and the reason the crate-wide OTLP promise stays `false`.
- **Ids.** `uuid` v4 supplies a 32-hex trace id and the first eight bytes give a 16-hex span id, matching the OTel id shapes Python produces; `SpanContext::new` trusts caller-supplied ids while generated ids are always the right shape.
- **Root attributes.** `root_for` also stamps `team.id = team_name`, mirroring Python's `get_or_create_team_span`, even though `team.name` alone is enough to identify the team.
- **Round outcome wire strings.** `RoundOutcome` has no `as_str` in `crate::schema` (read-only), so `otel.rs` carries a private `round_outcome_str` helper and a test pins it to the serde wire values.
- **Owned span data.** Hooks receive `&TeamSpan`, never a live span handle; recording hooks clone it. This is what makes the seam testable offline and independent of any tracer implementation.
- **Memory as spans, not a second log stream.** `MemorySpanObserver` bridges the engine's event seam (`openjiuwen_memory::team::TeamMemoryObserver`) instead of the engine learning about spans, so `openjiuwen-memory` stays observability-free and this crate owns the mapping; one observer per team run keeps the memory spans inside the team's trace.
- **The baseline memory bridge needs no new dependency.** `openjiuwen-memory` is already a dependency of this crate, so the bridge implements the engine's trait directly; the sink half reuses `otel.rs`. The OTLP bridge is a separate, feature-gated module precisely so the baseline stays dependency-free.

## Test Strategy

60 inline `#[cfg(test)]` tests, all deterministic and offline; no `#[ignore]`, no sleeps, no wall-clock assertions (32 run in the default build, 43 under `--features otel`, 48 under `--features otlp` and 59 under `--all-features`: `mod.rs` and three `memory.rs` tests are feature-gated, and the 15 `otlp.rs` tests plus one `mod.rs` test need `otlp`).

- `span.rs` (10): `root_generates_32_and_16_lowercase_hex_ids`, `root_for_sets_python_parity_attributes`, `root_name_is_the_team_name_attribute`, `child_keeps_trace_and_gets_fresh_span_id`, `child_inherits_attributes_at_creation_time`, `child_override_does_not_touch_parent_attributes`, `with_attribute_replaces_and_absent_key_is_none`, `span_and_context_round_trip_through_json`, `span_tolerates_unknown_and_missing_fields`, `context_child_keeps_trace_and_regenerates_span`.
- `hooks.rs` (13): `recording_hooks_preserve_call_order_and_counts`, `observable_event_kinds_are_payload_level`, `two_recording_hooks_see_the_same_sequence`, `noop_hooks_are_silent`, `facade_without_hooks_never_panics`, `start_team_stores_root_and_records_session_id`, `finish_team_clears_the_root`, `member_span_is_a_child_of_a_live_root`, `member_span_without_root_is_standalone_with_team_name`, `round_and_task_spans_carry_names_and_attributes`, `add_hook_appends_in_registration_order`, `payloads_round_trip_through_json`, `observable_event_round_trips_with_internal_tag`.
- `mod.rs` (2, plus one `otlp`-only): `otlp_export_is_never_available` (both configs), `facade_records_events_without_the_otel_feature` (`cfg(not(feature = "otel"))`), and `the_otlp_feature_upgrades_only_the_memory_bridge` (`cfg(feature = "otlp")`: the constant stays `false` while `otlp_memory_observer("s-1")` builds a recording sink).
- `otel.rs` (9): `sink_retains_oldest_first_up_to_capacity`, `capacity_zero_retains_nothing_and_counts_every_record`, `record_copies_identity_attributes_and_start_time`, `clear_drops_records_but_keeps_the_evicted_counter`, `span_record_round_trips_and_tolerates_unknown_fields`, `span_record_omits_absent_parent_and_keeps_a_present_one`, `round_outcome_str_matches_the_python_wire_values`, `hooks_record_one_span_per_callback`, `hooks_wired_through_the_facade_record_spans`.
- `memory.rs` (10): `an_ok_event_builds_one_child_span_with_the_documented_attributes`, `span_names_follow_memory_dot_operation`, `a_failed_event_carries_the_failed_outcome_and_the_error`, `optional_identity_attributes_are_present_only_when_set_and_non_empty`, `observing_two_teams_keeps_two_distinct_traces`, `the_observer_is_send_and_sync_and_usable_as_a_trait_object`, `real_facade_events_reach_the_bridge` (drives `TeamMemory::in_memory().with_observer(...)` with a counting delegate), plus the `otel`-only `the_sink_holds_one_record_per_event_with_the_shared_trace`, `a_capacity_zero_sink_stores_nothing` and `the_sink_records_the_failed_outcome_and_the_error`.
- Without the `otel` feature the bridge's spans are asserted through the private `span_for` builder (there is no sink to inspect) and the facade test proves real events reach `observe`.
- `otlp.rs` (15, compiled with the module under `feature = "otlp"`, tests under `#[cfg(test)]`): `an_ok_event_builds_a_finished_component_frame_with_the_documented_attributes`, `span_names_follow_memory_dot_operation`, `a_failed_event_carries_the_error_text_and_the_failed_status`, `optional_identity_attributes_are_present_only_when_set_and_non_empty`, `the_invoke_id_is_stable_across_a_retry_and_differs_per_operation_member_and_team`, `the_trace_id_is_32_lowercase_hex_and_shared_by_one_team`, `a_frame_built_without_a_session_leaves_the_sink_fallback_in_charge`, `otlp_memory_observer_builds_over_a_private_memory_provider`, `with_sink_is_the_same_constructor_as_new`, `observing_on_a_not_recording_sink_is_a_silent_no_op`, `the_observer_is_send_and_sync_and_usable_as_a_trait_object`, plus four end-to-end tests that run the real `openjiuwen-extensions` sink over its in-memory exporter (no collector, no network): `the_extensions_sink_exports_one_finished_component_span_per_event` (`component.memory.write.{id}` name, `openjiuwen.status = finish`, component type/name, the serialized metadata blob), `a_failed_event_exports_one_error_status_span` (`openjiuwen.status = error` plus the failure text in the metadata blob), `two_events_of_one_team_carry_one_derived_trace_id` (`openjiuwen.trace.id` on both spans; the OTel trace ids differ because each parentless frame is its own root), and `a_real_facade_event_reaches_the_exporter` (a real `TeamMemory::in_memory()` `remember` + `search` pair lands as two `component.memory.*` spans).
- The OTLP suite builds its own `TracerHandle` over `ExporterType::Memory` and reads `CaptureExporter`, so nothing needs a collector; the `not recording` case shuts the handle down first.
- Definition of done (all green): `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --offline --lib` (with and without `--no-default-features`, `--features otel` and `--features otlp`), and `cargo clippy -p openjiuwen-agent-teams --all-targets --offline [--features otel|otlp] -- -D warnings`.

## Routing Table

- `./span.rs` → span identity, attribute constants, creation-time attribute inheritance.
- `./hooks.rs` → event payloads, `ObservabilityHooks`, `NoopHooks`/`RecordingHooks`, `TeamObservability` facade and fan-out.
- `./otel.rs` → feature-gated structured `tracing` events and the bounded in-memory `SpanRecord` sink.
- `./memory.rs` → `MemorySpanObserver` (team-memory events → `memory.{operation}` spans), the `AT_MEMORY_*` attribute keys and `memory_observer`.
- `./otlp.rs` → `OtlpMemoryObserver` + `otlp_memory_observer`: team-memory events → one workflow/component `TraceFrame` each, emitted into `openjiuwen_extensions::otel::OtelTraceSink`.
- `./mod.rs` → module doc, re-exports and `OTLP_EXPORT_AVAILABLE`.

## Deferred

- Real OTLP export for the rest of the crate: with `otlp` the *team-memory* bridge drives the `openjiuwen-extensions` exporter, but the runtime's member/round/task spans, the hook fan-out and `MemorySpanObserver` stay on the in-memory/`tracing` seam, so `OTLP_EXPORT_AVAILABLE` remains `false`. Linking the memory spans into the team trace (a live parent span), exporter configuration in this crate (`TracerProvider`/span processors/sampling) and any exporter setup of its own stay out: the host owns the `TracerHandle` and passes its sink to `OtlpMemoryObserver::new`.
- OTLP/gRPC and protobuf transport selection — an `openjiuwen-extensions` concern (its `OtlpProtocol::Grpc` answers `Error::Unsupported`); this crate never names a protocol.
- Python `monitor_handler.py`'s long-lived `task.{id}` spans, `msg.*` and `event.*` spans, reflection fallback over `TeamEvent` types and `_task_spans` recovery — the runtime reports task transitions through `on_task_transition` instead.
- Python `rail.py`'s agent-tier decoration and the `agent.{member}.task_iteration.N` / `llm.call` / `tool.*` span stack — that belongs to the harness agent rail; this port exposes `TurnMetrics` and member spans.
- Python `setup.py`'s process-global tracer demand bookkeeping (`init/acquire/release/shutdown`), the session-keyed root registry, `attach_to_team_agent` and the ambient `set_ambient_team_span` fallback — replaced by a facade-local root (and, for OTLP, a host-supplied sink over a private provider).
- Span status/end-time/forced-close APIs, span events, sampling and redaction of `openjiuwen.span.output` — no live span object exists here; redaction belongs to the extensions sink configuration.
- `rail.py`'s reliability hooks (retry/health remediation) — a separate concern from observation and not ported.
- Team-memory operation *metrics*, aggregation and sampling — the bridge reports one span and one log line per event; counting belongs to whatever consumes them.

## Notes for Agents

- Do not add a process-global registry or session-keyed root map; pass an `Arc<TeamObservability>` to every member that must emit.
- There is no `MemorySpanObserver::from_root(TeamSpan)` constructor: `src/memory/` (`trace.rs`, `wiring.rs`, `mod.rs`) names it as the missing seam for joining the `memory.*` spans to the runtime's `team.{name}` root, so today the two traces correlate only through the `team.name`/`team.id` attributes both roots carry.
- The module docs of `memory.rs` and `otel.rs` still say `openjiuwen-extensions` "sits in the same dependency layer as this crate (so it cannot be depended on from here)"; the `otlp` feature does depend on it, so read those sentences as stale prose, not as a constraint.
- The `otlp` feature does not imply `otel`: `--features otlp` builds `otlp.rs` over `openjiuwen-extensions` while `observability::otel` stays absent (`memory.rs` then has no sink slot), and only `--features otel` or `--all-features` compiles both bridges' sink halves.
- Two memory bridges exist and neither replaces the other: `memory_observer(team_name)` for the in-process span tree (+ `tracing`), and — under `otlp` — `OtlpMemoryObserver::new(sink)` (or `otlp_memory_observer(session_id)` for a private in-memory provider) for the real `OTel` pipeline. Both implement `TeamMemoryObserver`, both attach at the same `TeamMemory::with_observer(Arc::new(..))` site, and the facade keeps only one observer per instance.
- A host wiring OTLP builds its own `TracerHandle` (`ExporterType::Otlp` over OTLP/HTTP) and passes `OtelTraceSink::new(&handle)`; the observer never installs a global tracer and never constructs an `opentelemetry` span itself.
- To add a new observation: add the payload struct, a default trait method, the facade fan-out method, the `RecordingHooks` kind string, the `OtelHooks` `tracing::info!` + `sink.record` call, and one test per layer; a memory-facing change touches `memory.rs` and `otlp.rs` together, because both own a `span_for` over the same event.
- Renaming an `AT_*` constant or span-name format is a breaking change for consumers that pin exact strings.
- `RecordedEvent` intentionally does not implement `Deserialize` (its `kind` is `&'static str`); record through `RecordingHooks` instead.
- Attach the memory bridge where the team's `TeamMemory` is built: `TeamMemory::in_memory().with_observer(Arc::new(memory_observer(team_name)))`, plus `.with_sink(sink)` on the observer under `otel`; `TeamMemory::with_observer` consumes the facade, so it must run before the facade is shared.
- Keep the bridge's `tracing` target `openjiuwen_agent_teams::memory` aligned with `../memory/wiring.rs` — both report the team's memory under one target, and `otlp.rs` reuses it for its `not recording` line.
- `MemorySpanObserver` and `OtlpMemoryObserver` never join the `ObservabilityHooks` fan-out: both implement the engine's `TeamMemoryObserver`, and the seams stay separate because the memory facade calls back synchronously on its own error path.
