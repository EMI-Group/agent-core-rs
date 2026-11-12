# openjiuwen-agent-teams/tests — public-API integration suites

## Intent

Crate-level integration suites for `openjiuwen-agent-teams`: the contract *between* modules, driven only through the crate's public API (`openjiuwen_agent_teams::…`, never `crate::`).
The ~660 unit tests beside the modules already pin each module's internals; these suites pin what the modules promise each other and what a caller outside the crate can observe.
Every suite is offline and deterministic: scripted harnesses, an in-memory store, a tempdir workspace provider, in-process transport, bounded waits.

## API Surface — suites

| File | Tests | Behaviour pinned |
| --- | --- | --- |
| `spec_build.rs` | 12 | Full `TeamAgentSpec` JSON parse (defaults, unknown fields at top level/`leader`/`workspace`, `predefined_members`, `transport`, `storage`); `validate()` errors (missing leader, reserved names, framework bootstrap without scheduled dispatch) by `ErrorKind` + message; `build_with` card `<team>_<leader>`, roster and row statuses, leader-only harness creation, workspace root handed to the factory; language resolution from deps vs spec and the no-mutation guarantee; `activate` admission rows through `TeamRuntimeManager` (create, same-session attach, cross-session rejection, persistent resume). |
| `team_run.rs` | 9 | `activate` → `invoke` returns the scripted leader answer with one recorded query; null input settles with no leader turn; `stream` ends with exactly one `team.completed` marker last and the answer in an earlier chunk; plain and nested inputs reach the leader verbatim; an open task keeps the run open until the test completes it through `backend().tasks()`; `interact` on a live team queues (`ok`), unknown team is `not_active`, a closed gate is `gate_closed`, after `stop_team` is `not_active`; `snapshot()` roster sorted; stop/live lifecycle. |
| `member_and_mail.rs` | 14 | `spawn_member` builds exactly one harness and readies the row (idempotent); factory failure leaves the row `unstarted`; `shutdown_member` lifetime/force matrix (Dormant, ShutDown, validation from unstarted, unknown → `NotFound`) in both the handle and the store; `auto_start_all` gives every member a distinct on-disk workspace under the provider root; direct/broadcast/multicast mail rows, events, sender stamping and target de-duplication; `mark_read` clears only named ids and never advances the broadcast watermark; leader-only task creation and the teammate role label; roster accessors agree and are sorted. |
| `task_flow.rs` | 32 | `TaskBoard` + `InMemoryTeamStore` only: dependency edges (`blocked` → `pending` on upstream completion), cycle/duplicate/blank-id rejection messages, claim/assign/reassign rules, no-gate completion, autonomous review gate (verify pass/fail, revision notes, round increment), scheduled vote tally (`ReviewTally`, threshold, round bound), subtask depth limits, cancel/reset and `NotFound` rows. |
| `hitl.rs` | 33 | `parse_interact_str` grammar over a roster (mentions, `@all`/`@*`, `#`/`##`, `$` channels, unknown-recipient fold-back); `resolve_targets` multicast ordering, de-duplication and the role placeholders; `parse_mention` and reserved names; `InteractGate` open/close/admit/drain with `ok`/`not_active`/`gate_closed`; `ToolApprovalRequest::to_interaction` + `parse_confirm` + `approval_result_event`; `InteractPayload`/`DeliverResult` wire round-trips. |
| `protocol_adapter.rs` | 13 | `HarnessMemberRuntime` command forwarding (start/send steer vs follow-up/abort force vs graceful/pause/resume/stop), observer state+round+chunk callback order, chunk `{"type","payload"}` shape; the `state_to_member_status`/`state_to_execution_status` tables; `state_from_event`/`round_from_event`/`is_terminal_event`/`chunk_from_event` projections; `NativeCheckpoint` JSON round-trip, snake_case keys, unknown-field tolerance, identity validation; `TeamAgent::state()` mirrors the leader harness. |
| `cli.rs` | 36 | `SkillCli::try_parse_from` for every operator and member command (and the scope split in both directions); descriptor flags before/after the subcommand, `load_descriptor` precedence and error text, `parse_descriptor_args`; `SkillCli::run` against `StoreTeamClient` over `InMemoryTeamStore` + `TaskBoard` + in-process bus: pinned inbox/board/roster/membership lines, consume-on-read, `null` for an unknown task, acknowledgements, refusals; exit codes 0/1/2. |
| `mcp.rs` | 21 | `TeamMcpServer::list_tools` per scope (names, order, required properties), `instructions()` per scope, `call_tool` exact text for all ten operator tools (inbox, send, create/update/claim/complete task, list/claimable/get, list members), `get_task` missing → `"null"`, `Unknown tool: <name>` for unknown and member-scope `read_inbox`, argument errors as text with the server still usable. |
| `expert_catalog.rs` | 12 | `load_agent_group` over a tempdir package (manifest, roster, personas, skills, language aliases, unknown keys), load-failure kinds/messages, `bind_group` ordering and content, `ExpertRef` wire shape, `compose_template_member` precedence (persona prompt prefix, skill union, empty base), `ExpertCatalog` singleton accessors, and a `build_with` using a template-composed `agents["teammate"]`. |

182 integration tests in total; each file is its own test binary.

## API Surface — `support/`

Shared fakes, public API only (`openjiuwen_agent_teams::…`), modelled on the in-crate `runtime::team_agent::test_support`.

```rust
pub const BOUND: Duration;                                 // 5 s upper bound for every wait
pub async fn wait_until(what: &str, predicate: impl FnMut() -> bool);
pub async fn collect(BoxStream<'static, Result<AgentChunk>>) -> Vec<Result<AgentChunk>>;
pub fn marker_count(&[Result<AgentChunk>]) -> usize;

pub enum Call { Start, Stop, Pause, Resume, Send(String), Abort(AbortMode) }
pub struct ScriptedHarness;      // impl HarnessProtocol
pub struct FakeHarnessFactory;   // impl MemberHarnessFactory; name() == "fake"
pub struct FakeWorkspaceProvider; // impl WorkspaceProvider; owns a TempDir
pub struct Fixture { manager, factory, workspace, store }   // one TeamRuntimeManager over fresh fakes

pub fn build_deps(&Arc<FakeHarnessFactory>, &Arc<FakeWorkspaceProvider>, Option<Arc<InMemoryTeamStore>>) -> TeamBuildDeps;
pub fn sample_spec() -> TeamAgentSpec;            // "alpha", leader "team_leader", predefined "dev-1", agents["teammate"]
pub fn sample_spec_verified() -> TeamAgentSpec;   // enable_task_verification = true

pub type Recorder = Arc<Mutex<Vec<EventMessage>>>;
pub fn subscribe_observer(&Arc<dyn Messager>, &[TeamTopic]) -> Recorder;
pub fn recorded(&Recorder) -> Vec<EventMessage>;   pub fn event_types(&Recorder) -> Vec<String>;
```

`ScriptedHarness::new(answer)` answers a `send` with `turn_lifecycle started` → one answer `output` → `turn_lifecycle finished` and returns to `Idle`; `hanging` leaves it `Running` and emits nothing.
`emit`/`emit_turn`/`emit_answer` script extra observations; `calls()`/`sends()`/`abort_modes()`/`state_now()`/`set_state()` observe it.
`FakeHarnessFactory` records one `MemberHarnessContext` per `create` in creation order, can fail (`Error::config`) or hang per member and scripts a per-member answer.
`FakeWorkspaceProvider::allocate` returns `<tempdir>/<team>/<session>/<member>` (created on disk) and records requests, allocations and releases.

## Constraints

- Only files under `tests/` are edited; `src/`, `Cargo.toml` and sibling crates are out of scope — a missing seam is reported, never widened.
- Public API only: no `crate::` paths, no `#[path]` tricks into `src/`.
- `support/mod.rs` carries `#![allow(dead_code, unused_imports, missing_docs)]` because each test binary uses a different subset of it.
- Offline and deterministic: scripted harnesses, `InMemoryTeamStore`, tempdirs, in-process transport; no network, no model, no clock dependence.
- Every wait is bounded by `support::BOUND` (`tokio::time::timeout` or `wait_until`, 5 ms poll); no bare sleeps beyond a single short observation window (≤200 ms) where an ordering must be proven.
- No `#[ignore]`, no skipping: a behaviour the crate does not implement is pinned as the closest reachable assertion and reported.
- Tests assert on public values (statuses, rows, events, exact strings), never on log output; `unwrap`/`expect`/`panic` are allowed in test code.
- Each suite creates its own fixtures; only `expert_catalog.rs` touches the process-global catalog, and it clears it through a `Drop` guard.
- No suite calls `run_skill_cli` or otherwise depends on the process-global team registry.

## Routing Table

- `./support/` → shared public-API fakes, fixture, bounded wait/drain helpers (`support/mod.rs`)
- `./spec_build.rs` → spec JSON/validation/build and runtime admission
- `./team_run.rs` → end-to-end run, stream marker, open task, interact, snapshot, stop
- `./member_and_mail.rs` → member spawn/shutdown, workspace isolation, mail routing, leader-only tasks
- `./task_flow.rs` → task board graph, claims, completion, review gate, votes, subtasks
- `./hitl.rs` → interaction grammar, gate, tool-approval bridge, payload wire shapes
- `./protocol_adapter.rs` → harness protocol adapter, mapping tables, checkpoint, leader state
- `./cli.rs` → `team-member` clap tree and `SkillCli::run` over `StoreTeamClient`
- `./mcp.rs` → `openjiuwen-team-mcp` tool tables, instructions and `call_tool` text
- `./expert_catalog.rs` → agent-group packages, persona composition, catalog seam

## Test Strategy

- One file per behaviour area; a file may add private helpers but edits `support/mod.rs` never (sibling suites share it and are written in parallel).
- Reuse over re-implementation: drive the same seams the crate's own `src/runtime/tests/` crossed-seam suite drives, but from outside the crate.
- Exact expectations (messages, statuses, JSON strings) are mined from the unit tests beside each module; the integration suite proves they survive across the module boundary.
- Run `cargo test -p openjiuwen-agent-teams --tests` (182 integration + 662 unit) in both feature configurations; `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` is clean.

## Known Issues

- `cargo test -p openjiuwen-agent-teams` fails its **doctests** (`src/lib.rs` line 27 and `src/constants.rs` line 38) because the crate root does not re-export `DEFAULT_LEADER_MEMBER_NAME`, `is_reserved_name` or `parse_mention` flat, although the crate `CONTEXT.md` says it does; `src/` is out of scope for this node.
- `cargo fmt -p openjiuwen-agent-teams -- --check` reports one pre-existing line in `src/constants.rs` (the multi-line `parse_mention` assertion); all files under `tests/` are rustfmt-clean.
- The teammate `create_task` rejection (`ensure_leader_role`, `src/runtime/team_agent.rs`) is unreachable from outside the crate: `build_with` only produces a leader handle and `TeamAgent`'s constructor is crate-private, so `member_and_mail.rs` pins the leader path plus the ungated board seam instead.
- `SkillCli`'s `descriptor`/`command` fields and its command enums are `pub(crate)` with no accessor, so `cli.rs` pins parsed shapes through `Debug` rendering and end-to-end behaviour.
- `HarnessMemberRuntime::send` discards the `SendReceipt` and `support::ScriptedHarness` records only the text, so delivery/abort modes are observed through a file-local `ModeRecorder` decorator in `protocol_adapter.rs`.
- `RoundOutcome` lives in `schema`, not `member`; `native checkpoint` incompatibility messages name only the checkpoint's own session/card id.
- Board behaviours worth knowing when asserting: an unknown `depends_on` parks the task `blocked` (not an error), promotion is reported through `TaskMutation::unblocked` rather than an event, `detail(unknown)` is `Ok(None)`, `review_tally` reports round 0 before the first vote, and `max_review_rounds` is stored but not enforced.
- MCP/Store client asymmetry: `claim_task` on an unknown id surfaces as `Internal error: not_found: …` while `complete_task` renders `{"ok":false,"reason":"not_found: …"}`.
- `load_agent_group` reports a missing package directory as `Validation`, not `NotFound`; `NotFound` is reachable through `load_member_persona`.
