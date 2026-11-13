# openjiuwen-agent-teams/tests — public-API integration suites

## Intent

Crate-level integration suites for `openjiuwen-agent-teams`: the contract *between* modules, driven only through the crate's public API (`openjiuwen_agent_teams::…`, never `crate::`).
The 1411 unit tests beside the modules (default build) pin each module's internals; these suites pin what the modules promise each other and what a caller outside the crate can observe.
Every suite is offline and deterministic: scripted harnesses or a loopback mock model, an in-memory or SQLite store, tempdir workspaces, in-process transport, real Unix sockets or binaries only where the suite says so, bounded waits.

## Suite table

| File | Tests | Behaviour pinned |
| --- | --- | --- |
| `spec_build.rs` | 12 | Full `TeamAgentSpec` JSON parse (defaults, unknown fields at top level/`leader`/`workspace`, `predefined_members`, `transport`, `storage`); `validate()` errors (missing leader, reserved names, framework bootstrap without scheduled dispatch) by `ErrorKind` + message; `build_with` card `<team>_<leader>`, roster and row statuses, leader-only harness creation, workspace root handed to the factory; language resolution from deps vs spec and the no-mutation guarantee; `activate` admission (create, same-session attach, cross-session rejection, persistent resume). |
| `team_run.rs` | 9 | `activate` → `invoke` returns the scripted leader answer with one recorded query; null input settles with no leader turn; `stream` ends with exactly one `team.completed` marker last and the answer in an earlier chunk; plain and nested inputs reach the leader verbatim; an open task keeps the run open until the test completes it through `backend().tasks()`; `interact` queues while a run is open and reports `not_active`/`gate_closed`; `snapshot()` roster sorted; stop/live lifecycle. |
| `member_and_mail.rs` | 14 | `spawn_member` builds exactly one harness and readies the row (idempotent); factory failure leaves the row `unstarted`; each member runtime runs under its derived session id `MemberHarnessContext::member_session_id(agent.session_id(), member)` while `TeamAgent::session_id()` stays the team session; `shutdown_member` lifetime/force matrix (Dormant, ShutDown, validation from unstarted, unknown → `NotFound`) in both the handle and the store; `auto_start_all` gives every member a distinct on-disk workspace under the provider root; direct/broadcast/multicast mail rows, events, sender stamping and target de-duplication; `mark_read` clears only named ids and never advances the broadcast watermark; leader-only task creation and the teammate role label; roster accessors agree and are sorted. |
| `task_flow.rs` | 32 | `TaskBoard` + `InMemoryTeamStore` only: dependency edges (`blocked` → `pending` on upstream completion, reported through `TaskMutation::unblocked`), cycle/duplicate/blank-id rejection messages, claim/assign/reassign rules, no-gate completion, autonomous review gate (verify pass/fail, revision notes, round increment), scheduled vote tally (`ReviewTally`, threshold, round bound), subtask depth limits, cancel/reset and `NotFound` rows. |
| `hitl.rs` | 33 | `parse_interact_str` grammar over a roster (mentions, `@all`/`@*`, `#`/`##`, `$` channels, unknown-recipient fold-back); `resolve_targets` multicast ordering, de-duplication and the role placeholders; `parse_mention` and reserved names; `InteractGate` open/close/admit/drain with `ok`/`not_active`/`gate_closed`; `ToolApprovalRequest::to_interaction` + `parse_confirm` + `approval_result_event`; `InteractPayload`/`DeliverResult` wire round-trips. |
| `protocol_adapter.rs` | 13 | `HarnessMemberRuntime` command forwarding (start/send steer vs follow-up/abort force vs graceful/pause/resume/stop), observer state+round+chunk callback order, chunk `{"type","payload"}` shape; the `state_to_member_status`/`state_to_execution_status` tables; `state_from_event`/`round_from_event`/`is_terminal_event`/`chunk_from_event` projections; `NativeCheckpoint` JSON round-trip, snake_case keys, unknown-field tolerance, identity validation; `TeamAgent::state()` mirrors the leader harness. |
| `cli.rs` | 39 | `SkillCli::try_parse_from` for every operator and member command (and the scope split in both directions), the scope-independent `serve` stage; descriptor flags before/after the subcommand, `load_descriptor` precedence and error text, `parse_descriptor_args`; publicly inspectable parsed shapes (`command()`, `descriptor_args()`); `SkillCli::run` against `StoreTeamClient` over `InMemoryTeamStore` + `TaskBoard` + in-process bus: pinned inbox/board/roster/membership lines, consume-on-read, `null` for an unknown task, acknowledgements, refusals; exit codes 0/1/2. |
| `mcp.rs` | 21 | `TeamMcpServer::list_tools` per scope (names, order, required properties), `instructions()` per scope, `call_tool` exact text for all ten operator tools (inbox, send, create/update/claim/complete task, list/claimable/get, list members), `get_task` missing → `"null"`, `Unknown tool: <name>` for unknown and member-scope `read_inbox`, argument errors as text with the server still usable. |
| `expert_catalog.rs` | 12 | `load_agent_group` over a tempdir package (manifest, roster, personas, skills, language aliases, unknown keys), load-failure kinds/messages, `bind_group` ordering and content, `ExpertRef` wire shape, `compose_template_member` precedence (persona prompt prefix, skill union, empty base), `ExpertCatalog` singleton accessors behind a `Drop` guard, and a `build_with` using a template-composed `agents["teammate"]`. |
| `bins.rs` | 6 | The two binaries as real child processes through `CARGO_BIN_EXE_*`: `--help` and `serve --help` on **stdout** with exit `0` and no descriptor, no arguments reporting the missing descriptor, an unknown connection flag as a CLI-side error on **stderr** with exit `2`, an unhosted team exiting `1`, and `openjiuwen-team-mcp` without a descriptor exiting `1` with a stderr diagnostic while stdout stays the wire. |
| `full_team_sqlite.rs` | 4 | `#![cfg(feature = "sqlite")]`: a full run (build → start → invoke → board → mail → spawn → stop) persists the team row and the roster, task rows/dependencies/filters survive a reopen, mailbox rows and the broadcast watermark survive a reopen, and reopening twice keeps every row with the database still writable — each check opened through a **second** `TeamStore` over the same file. |
| `member_process_kind.rs` | 5 | The child-process member path against the real `team-member` binary: `serve --help` on stdout with exit `0` and `serve --definitely-not-a-flag` on stderr with exit `2` (neither needs a descriptor), a garbage request line answered with an `error` frame while the loop keeps serving until `stop`, a blueprint selecting the `process` spawn kind really launching a child and completing the `ready`/`state` handshake (credentials from the loopback `MockLlmServer`), and a refusing child failing the spawn with a typed error; a process member stays on the team session id (only in-process runtimes get the derived per-member one). |
| `attach_e2e.rs` | 6 | `#![cfg(unix)]`: the socket exists only after `start()` for every documented `attach` metadata shape, `SocketTeamClient` lists the roster, reads the leader's inbox, sends mail that lands in the recipient's stored mailbox and snapshots the team, creates/edits/claims/completes tasks on the hosting board, an op-level fault (`not_found`) stays in band and keeps the connection while a frame-level fault ends that connection, and `stop_team` answers `{"stopping": true}`, cancels the run, and the host's `stop()` removes the socket. |
| `org_flow.rs` | 9 | `metadata["organization"]` installed at build as the `OrgBinding`-shaped facade: a broken chart (bad id, dangling reporting line) fails the build with `Validation`; an allowed hop (reporting line, depth, budget) is applied to the org task and the task runs create→complete in the reserved namespace; refused delegations (self, non-member target, cross-team sibling) leave the pool untouched; the router enforces the depth cap, per-team budget and role pairs; org mail is acknowledged only after the leader flush; a team without the block keeps the flat board. |
| `workspace_isolation.rs` | 4 | `metadata["workspace"] = {"isolation": "worktree"}` against a **real** temporary git repository and a real `LocalWorkspaceProvider`: every member (leader and teammates) gets its own registered worktree inside the provider layout carrying the `.git` file marker and used as that harness's workspace root; the bare `"worktree"` string isolates the leader too; `stop_team` releases the worktrees and unregisters them from `git worktree list`; a team without the key stays on plain local directories. |

219 integration tests in total across the 15 files (215 in the default build, since `full_team_sqlite.rs` is feature-gated); each file is its own test binary.

## API Surface — `support/`

Shared fakes and helpers, public API only (`openjiuwen_agent_teams::…`), modelled on the in-crate `runtime::team_agent::test_support`.

```rust
pub const BOUND: Duration;                                 // 5 s upper bound for every wait
pub async fn wait_until(what: &str, predicate: impl FnMut() -> bool);
pub async fn collect(stream: BoxStream<'static, Result<AgentChunk>>) -> Vec<Result<AgentChunk>>;
pub fn marker_count(frames: &[Result<AgentChunk>]) -> usize;

pub enum Call { Start, Stop, Pause, Resume, Send(String), Abort(AbortMode) }
pub struct ScriptedHarness;      // impl HarnessProtocol
impl ScriptedHarness {
    pub fn new(answer: impl Into<String>) -> Arc<Self>;    // answers a send, returns to Idle
    pub fn hanging(answer: impl Into<String>) -> Arc<Self>; // stays Running, emits nothing
    pub fn emit(&self, event: HarnessEvent);  pub fn emit_turn(&self, kind: TurnEventKind);
    pub fn emit_answer(&self, text: &str);
    pub fn calls(&self) -> Vec<Call>;         pub fn sends(&self) -> Vec<String>;
    pub fn abort_modes(&self) -> Vec<AbortMode>;
    pub fn state_now(&self) -> HarnessState;  pub fn set_state(&self, state: HarnessState);
}
pub struct FakeHarnessFactory;   // impl MemberHarnessFactory; name() == "fake"
impl FakeHarnessFactory {
    pub fn new() -> Arc<Self>;                            // default answer `done: <member>`
    pub fn failing_for(&self, member: &str);              // create -> Error::config
    pub fn hanging_for(&self, member: &str);
    pub fn answers_with(&self, member: &str, answer: &str);
    pub fn calls(&self) -> Vec<MemberHarnessContext>;     pub fn created_members(&self) -> Vec<String>;
    pub fn workspace_root(&self, member: &str) -> Option<PathBuf>;
    pub fn harness(&self, member: &str) -> Option<Arc<ScriptedHarness>>;
}
pub struct FakeWorkspaceProvider; // impl WorkspaceProvider; owns a TempDir
impl FakeWorkspaceProvider {
    pub fn new() -> Arc<Self>;
    pub fn requests(&self) -> Vec<WorkspaceRequest>;  pub fn allocations(&self) -> Vec<String>;
    pub fn releases(&self) -> Vec<String>;            pub fn root(&self) -> &Path;
    pub fn root_for(&self, member: &str) -> Option<PathBuf>;
}
pub struct Fixture { pub manager, factory, workspace, store }  // one TeamRuntimeManager over fresh fakes
impl Fixture { pub fn new() -> Self; pub async fn activate(&self, spec, session) -> Arc<TeamAgent>;
               pub fn harness(&self, member: &str) -> Arc<ScriptedHarness>; pub fn sends(&self, member) -> Vec<String>; }

pub fn build_deps(&Arc<FakeHarnessFactory>, &Arc<FakeWorkspaceProvider>, Option<Arc<InMemoryTeamStore>>) -> TeamBuildDeps;
pub fn sample_spec() -> TeamAgentSpec;            // "alpha", leader "team_leader", predefined "dev-1", agents["teammate"]
pub fn sample_spec_verified() -> TeamAgentSpec;   // enable_task_verification = true

pub type Recorder = Arc<Mutex<Vec<EventMessage>>>;
pub fn subscribe_observer(messager: &Arc<dyn Messager>, topics: &[TeamTopic]) -> Recorder;
pub fn recorded(recorder: &Recorder) -> Vec<EventMessage>;  pub fn event_types(recorder: &Recorder) -> Vec<String>;
```

`ScriptedHarness::new(answer)` answers a `send` with `turn_lifecycle started` → one answer `output` → `turn_lifecycle finished` and returns to `Idle`; `hanging` leaves it `Running` and emits nothing; `emit`/`emit_turn`/`emit_answer` script extra observations.
`FakeHarnessFactory` records one `MemberHarnessContext` per `create` in creation order, can fail (`Error::config`) or hang per member and scripts a per-member answer.
`FakeWorkspaceProvider::allocate` returns a distinct directory under its tempdir root (created on disk) and records requests, allocations and releases.

## Constraints

- Only files under `tests/` are edited; `src/`, `Cargo.toml` and sibling crates are out of scope — a missing seam is reported, never widened.
- Public API only: no `crate::` paths, no `#[path]` tricks into `src/`.
- `support/mod.rs` carries `#![allow(dead_code, unused_imports, missing_docs)]` because each test binary uses a different subset of it, and no suite edits that file.
- Offline and deterministic: scripted harnesses, `InMemoryTeamStore`/`SqliteTeamStore`, tempdirs, in-process transport, loopback sockets; no external service, no vendor CLI.
- Every wait is bounded by `support::BOUND` (5 s; `tokio::time::timeout` or `wait_until`, 5 ms poll); no bare sleeps beyond a single short observation window (≤200 ms) where an ordering must be proven.
- No `#[ignore]`, no skipping: a behaviour the crate does not implement is pinned as the closest reachable assertion and reported.
- Tests assert on public values (statuses, rows, events, exact strings), never on log output; `unwrap`/`expect`/`panic` are allowed in test code.
- Each suite creates its own fixtures; only `expert_catalog.rs` touches the process-global catalog, and it clears it through a `Drop` guard.
- No suite calls `run_skill_cli` or mutates the process-global runtime registry, except where a suite says so (`member_process_kind.rs` drives the binary as a child process instead).
- Environment needs of the newer suites: `--features sqlite` for `full_team_sqlite.rs`; `git` on `PATH` for `workspace_isolation.rs` (a missing binary is a clear panic, never a silent skip); the `team-member`/`openjiuwen-team-mcp` binaries via `CARGO_BIN_EXE_*` for `bins.rs` and `member_process_kind.rs`; a loopback `MockLlmServer` only inside `member_process_kind.rs`, and real Unix sockets only in `attach_e2e.rs` and the Unix-gated unit suites.

## Test Strategy

- One file per behaviour area; a file may add private helpers but never edits `support/mod.rs` (sibling suites share it).
- Reuse over re-implementation: drive the same seams the crate's own cross-seam suites drive, but from outside the crate.
- Exact expectations (messages, statuses, JSON strings) are mined from the unit tests beside each module; the integration suite proves they survive across the module boundary.
- Run `cargo test -p openjiuwen-agent-teams` (1411 unit + 215 integration + 37 doctests and one `ignore` block in the default build; 1430 unit and 219 integration with `--features sqlite`; 1439 unit with `--all-features`; 1409 unit with `--no-default-features`) in every feature configuration: default, `--no-default-features`, `--features sqlite`, `--features otel`, `--all-features`; `cargo clippy -p openjiuwen-agent-teams --all-targets --all-features --offline -- -D warnings` is clean.
- `cargo test -p openjiuwen-agent-teams --doc` is green: the crate-level doctest in `src/lib.rs` exercises the flat root re-exports (`DEFAULT_LEADER_MEMBER_NAME`, `is_reserved_name`, `parse_mention`).

## Known Issues

- The attach handler answers `send_member`, `create_task` and `complete_task` with the persisted record, not with an outcome envelope, so `SendOutcome::ok`, `CreateOutcome::ok` and `OpOutcome::ok` decode to `false` even on success; the suites assert the record/ids instead of the flag.
- `snapshot` and `member_status` exist on `AttachClient` only, not on the `TeamClient` trait, so the CLI/MCP surface cannot reach them; `attach_e2e.rs` calls the client directly.
- Attach `stop_team` only cancels the team's run token and answers `{"stopping": true}`; the host must call `TeamAgent::stop()` (or `TeamRuntimeManager::stop_team`) for the socket to be removed.
- `TeamRuntimeManager::finalize_member` stops the runtime, shuts the row down and drops it from the roster without releasing its workspace; the workspace is released by `stop()`.
- `create_org_task` and `delegate_org_task` on a team built without `metadata["organization"]` return `ErrorKind::Config` (a refused hop inside a configured organisation is `Validation`).
- The process-member suite pins the child launch, the `serve` CLI contract and the `ready`/`state` handshake, not a full scripted turn: the child builds its own member from `DeepAgentSpec::default()`, so leader-side per-member configuration does not cross the boundary.
- A worktree team whose layout root sits outside the repository needs a `GitWorktreeBackend` pinned with `with_repo_root`; discovery from the root only works when an ancestor of the layout is the repository.
- The teammate `create_task` rejection (`ensure_leader_role`) is unreachable from outside the crate: `build_with` only produces a leader handle and `TeamAgent`'s constructor is crate-private, so `member_and_mail.rs` pins the leader path plus the ungated board seam instead.
- `HarnessMemberRuntime::send` discards the `SendReceipt` and `support::ScriptedHarness` records only the text, so delivery/abort modes are observed through a file-local `ModeRecorder` decorator in `protocol_adapter.rs`.
- `RoundOutcome` lives in `schema`, not `member`; native-checkpoint incompatibility messages name only the checkpoint's own session/card id.
- Board behaviours worth knowing when asserting: an unknown `depends_on` parks the task `blocked` (not an error), promotion is reported through `TaskMutation::unblocked` rather than an event, `detail(unknown)` is `Ok(None)`, `review_tally` reports round 0 before the first vote, and `max_review_rounds` is recorded on the task but not enforced as a round ceiling.
- MCP/Store client asymmetry: `claim_task` on an unknown id surfaces as `Internal error: not_found: …` while `complete_task` renders `{"ok":false,"reason":"not_found: …"}`.
- `load_agent_group` reports a missing package directory as `Validation`, not `NotFound`; `NotFound` is reachable through `load_member_persona`.

## Routing Table

- `./support/` → shared public-API fakes, fixture, bounded wait/drain helpers (`support/mod.rs`)
- `./spec_build.rs` → spec JSON/validation/build and runtime admission
- `./team_run.rs` → end-to-end run, stream marker, open task, interact, snapshot, stop
- `./member_and_mail.rs` → member spawn/shutdown, workspace isolation, mail routing, leader-only tasks
- `./task_flow.rs` → task board graph, claims, completion, review gate, votes, subtasks
- `./hitl.rs` → interaction grammar, gate, tool-approval bridge, payload wire shapes
- `./protocol_adapter.rs` → harness protocol adapter, mapping tables, checkpoint, leader state
- `./cli.rs` → `team-member` clap tree (including `serve`) and `SkillCli::run` over `StoreTeamClient`
- `./mcp.rs` → `openjiuwen-team-mcp` tool tables, instructions and `call_tool` text
- `./expert_catalog.rs` → agent-group packages, persona composition, catalog seam
- `./bins.rs` → the two real binaries: help/exit codes/stream contract
- `./full_team_sqlite.rs` → persisted team run over `SqliteTeamStore` (`sqlite` feature)
- `./member_process_kind.rs` → child-process members against the real `team-member serve`
- `./attach_e2e.rs` → Unix-socket attach host driven through `SocketTeamClient`
- `./org_flow.rs` → organisation chart, delegation router, org task pool and mailbox
- `./workspace_isolation.rs` → git-worktree allocation and teardown against a real repository
