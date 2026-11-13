# trajectory

## Intent
Recording, grouping, windowing and digesting an agent run so evaluation, experience extraction and TTSE share one transcript type.
Ports the portable part of Python `openjiuwen.agent_evolving.trajectory`: `model.py`, `schema.py`, `spans.py`, `windows.py` and the offline builder.

## API Surface
- `Trajectory` — `trajectory_id`, `session_id`, `case_id`, `steps`, `outcome`, `score`, `elapsed_ms`, `metadata`; `new` rejects a blank id, and `push`/`extend` assign step indices, with `digest`, `is_successful`, `error_count`, `tool_names` over the recorded steps.
- `TrajectoryStep` — `index`, `kind`, `role`, `content`, `tool_name`, `tool_call_id`, `status`, `metadata`.
- `StepKind` (`message`/`tool_call`/`tool_result`/`model`/`system`/`other`), `StepStatus` (`ok`/`error`), `TrajectoryOutcome` (`success`/`failure`/`unknown`).
- `TrajectoryBuilder` / `from_messages` — incremental construction from steps, tool calls and tool results (`push_step`, `push_tool_call`, `push_tool_result`, `finish`), plus a projection of `BaseMessage`s.
- `TrajectoryStore` — in-memory archive with `add`/`get`/`remove`, `by_session`/`by_case`/`by_outcome`, `group_by_session`/`group_by_case`, `summary` and `group_summaries`.
- `spans`, `span_by_name`, `longest_span`, `failed_spans`, `TrajectorySpan` — tool-call/result pairing with open-span markers.
- `windows`, `window_around`, `last_n`, `first_error`, `render_digest`, `TrajectoryWindow` — bounded views for prompt injection, plus `TrajectoryDigest`.

## Constraints
- No clock is ever read: `elapsed_ms` is caller-supplied, so a recording is deterministic and serializable.
- Span and window ranges are positions in `Trajectory::steps` (`start` inclusive, `end` exclusive); an unanswered tool call is an open span with `end == start`, `open == true`, and the call's own status.
- Tool calls carried on a `BaseMessage` are not expanded automatically; `TrajectoryBuilder::push_tool_call`/`push_tool_result` keep the call ids pairable.
- Messages are copied, never borrowed, so a `Trajectory` is `'static` and can be stored, serialized or sent to another task.
- `TrajectoryStore::add` replaces an existing trajectory in place (same id) and keeps insertion order inside each group; group maps are `BTreeMap`s so keys iterate sorted.

## Design Decisions
- Typed steps replace Python's canonical OTLP payload: the `snake_case` field names are kept, which is enough for offline evaluation and prompt injection and avoids an OTLP dependency.
- `Trajectory::digest` and the `render_digest` helpers bound their output in characters, so a long transcript cannot silently consume a prompt budget.
- Span pairing uses the provider call id rather than step adjacency, and a call with an absent or empty id can never pair, which keeps malformed recordings honest.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./model.rs` → `Trajectory`, `TrajectoryStep`, vocabularies, `TrajectoryDigest`
- `./builder.rs` → `TrajectoryBuilder`, `from_messages`
- `./spans.rs` → span projection and query helpers
- `./windows.rs` → windows, `render_digest`, `first_error`, `last_n`
- `./store.rs` → `TrajectoryStore`, `TrajectoryStats`, `TrajectoryGroupSummary`
