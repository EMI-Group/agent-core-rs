# openjiuwen-agent-teams/src/member

## Intent

The member seam of the crate: the runtime a member's "brain" implements, the status machine over one roster row, the leader-side activity projection, and the spawn seams (in-process task handle plus three child-process spawners).
Python parity sources: `openjiuwen/agent_teams/agent/member.py` (`TeamMember.update_status` / `update_execution_status`), `agent/member_runtime.py` (`MemberRuntime` Protocol), `agent/member_activity.py` (`MemberActivityRegistry`, `IdleSignal`), `spawn/inprocess_spawn.py` + `spawn/inprocess_handle.py`, `spawn/external_cli_spawn.py`, `external/cli_agent/spawn.py` and `schema/ssh_transport.py`.
This node owns *member-level lifecycle and member launch*: it writes the roster row through the `TeamStore` CAS seam, reports transitions, and builds a runtime that can live in-process or in a child process.
Task assignment, mailbox routing, dispatch, event publication (topic + payload) and the team-idle marker belong to `../runtime/`; building an in-process harness belongs to `../harness/`.

## API Surface

```rust
// runtime.rs — the brain seam
pub trait MemberObserver: Send + Sync {
    fn on_state(&self, state: HarnessState);
    fn on_round(&self, outcome: RoundOutcome);
    fn on_chunk(&self, chunk: Value);
}
#[async_trait] pub trait MemberRuntime: Send + Sync {
    fn session_id(&self) -> &str;
    fn state(&self) -> HarnessState;
    async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()>;
    async fn stop(&self) -> Result<()>;
    async fn send(&self, content: String, immediate: bool) -> Result<()>;
    async fn abort(&self, immediate: bool) -> Result<()>;
    async fn pause(&self) -> Result<()>;
    async fn resume(&self, query: Option<String>) -> Result<()>;
    fn subscribe(&self, observer: Arc<dyn MemberObserver>);
}
pub enum ObserverEvent { State(HarnessState), Round(RoundOutcome), Chunk(Value) }
pub struct RecordingObserver;   // ::new(), take() drains, len(), is_empty()
pub struct ObserverSet;         // ::new(), add(), state()/round()/chunk(), len(), is_empty()

// lifecycle.rs — the status machine
pub struct TeamMember;
impl TeamMember {
    pub fn new(store: Arc<dyn TeamStore>, record: MemberRecord) -> Self;
    pub fn record(&self) -> MemberRecord;
    pub fn team_name(&self) -> &str;   pub fn member_name(&self) -> &str;
    pub fn role(&self) -> TeamRole;    pub fn status(&self) -> MemberStatus;      // cached projection
    pub fn execution(&self) -> ExecutionStatus;
    pub async fn stored_status(&self) -> Result<MemberStatus>;                  // store read + cache refresh
    pub fn runtime(&self) -> Option<Arc<dyn MemberRuntime>>;
    pub fn attach_runtime(&self, runtime: Arc<dyn MemberRuntime>);
    pub fn observe(&self, observer: Arc<dyn MemberObserver>);
    pub async fn refresh(&self) -> Result<MemberRecord>;
    pub async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()>;
    pub async fn mark(&self, to: MemberStatus) -> Result<bool>;
    pub async fn mark_execution(&self, to: ExecutionStatus) -> Result<()>;
    pub async fn note_round(&self, outcome: RoundOutcome) -> Result<()>;
    pub async fn shutdown(&self, lifetime: MemberLifetime, force: bool) -> Result<MemberStatus>;
    pub async fn error(&self, message: &str) -> Result<()>;
}

// activity.rs — leader-side projection and the pure debounce predicate
pub const DEFAULT_IDLE_DEBOUNCE: Duration;   // 2 s, Python _TEAM_IDLE_DEBOUNCE_SECONDS
pub struct MemberActivityRegistry;           // register/mark_busy/mark_idle/is_moving/idle_since/
                                             // last_activity/settled_members/snapshot/remove/len/is_empty
pub struct IdleDebounce { pub window: Duration }
impl IdleDebounce { pub const fn new(window: Duration) -> Self;
                    pub fn should_emit_idle(&self, now: Instant, last_activity: Option<Instant>) -> bool; }

// spawn.rs — the spawner seam and the in-process task handle
pub struct MemberSpawnRequest { pub team_name, pub member_name, pub session_id: String,
                                pub workspace_root: PathBuf, pub options: MemberSpawnOptions }
impl MemberSpawnRequest {
    pub fn new(team_name, member_name, session_id, workspace_root) -> Self;  // options default (in-process)
    pub fn with_options(self, options: MemberSpawnOptions) -> Self;
    pub fn identity_env(&self) -> BTreeMap<String, String>;
    pub fn apply_identity_env(&self, command: &mut StdioCommand);
}
pub const TEAM_NAME_ENV: &str = "OPENJIUWEN_TEAM_NAME";
pub const MEMBER_NAME_ENV: &str = "OPENJIUWEN_MEMBER_NAME";
pub const SESSION_ID_ENV: &str = "OPENJIUWEN_SESSION_ID";
pub const MEMBER_WORKSPACE_ENV: &str = "OPENJIUWEN_MEMBER_WORKSPACE";
#[async_trait] pub trait MemberSpawner: Send + Sync {
    async fn spawn(&self, request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>>;
    async fn despawn(&self, member_name: &str) -> Result<()>;
}
pub struct InProcessSpawnHandle;   // new(), with_cancellation(), member_name(), cancellation_token(),
                                   // is_finished(), cancel(), abort(), async join() -> Result<()>
pub struct SpawnRegistry;          // insert -> Option<handle>, contains, remove, take,
                                   // cancel_all() -> Vec<handle>, len, is_empty, names (sorted)

// spawn_config.rs — spawn mode plus the per-kind launch configuration
pub enum MemberSpawnKind { InProcess, Process, ExternalCli, Ssh }   // wire: inprocess|process|external_cli|ssh
pub struct StdioCommand { pub program: String, pub args: Vec<String>,
                          pub env: BTreeMap<String, String>, pub cwd: Option<PathBuf> }
impl StdioCommand { new/with_arg/with_args/with_env/with_cwd, program/args/env/cwd,
                    resolve_program() -> Result<PathBuf>, to_argv() -> Vec<String>,
                    to_command() -> Result<tokio::process::Command> }
pub struct ProcessMemberConfig { pub command: StdioCommand }        // default: team-member serve
pub struct ExternalCliRuntimeConfig { pub command: StdioCommand,
                                      pub cli_agent/provider/model: Option<String> }
pub struct SshTransportSpec { pub program /*ssh*/, host, port /*22*/, username, key_file, password,
                               agent, known_hosts, disable_host_key_check, connect_timeout_s /*15.0*/,
                               use_exec /*true*/, options: BTreeMap<String, String>, remote_command }
impl SshTransportSpec { remote_target(), quoted_remote_command(), to_argv(), validate(), to_command() }
pub struct MemberSpawnOptions { pub kind: MemberSpawnKind, pub process: Option<ProcessMemberConfig>,
                                pub external_cli: Option<ExternalCliRuntimeConfig>,
                                pub ssh: Option<SshTransportSpec> }
impl MemberSpawnOptions { from_kind, with_process, with_external_cli, with_ssh, is_in_process }

// protocol.rs — the child-process member protocol
pub enum MemberRequest { Start { session_id }, Send { content, immediate }, Abort { immediate },
                         Pause, Resume { query }, Stop }          // wire tag `type`
pub enum MemberResponse { Ready { session_id }, State { state: HarnessState },
                          Round { outcome: RoundOutcome }, Chunk { chunk: Value }, Error { message } }
impl MemberRequest/MemberResponse { frame_type(), to_json(), to_line(), parse_line() }
impl MemberResponse { parse_noisy_line() -> Result<Option<Self>>, state_change(), round_outcome(),
                      error_message(), is_ready(), ready_session_id() }

// stdio.rs — one runtime for any child process
pub const DEFAULT_IO_TIMEOUT: Duration;   // 10 s
pub struct StdioRuntimeOptions { pub io_timeout: Duration, pub cancellation: Option<CancellationToken> }
pub struct StdioMemberRuntime;
impl StdioMemberRuntime {
    pub async fn spawn(command: &StdioCommand, session_id: impl Into<String>) -> Result<Self>;
    pub async fn spawn_with(command: &StdioCommand, session_id: impl Into<String>,
                            options: StdioRuntimeOptions) -> Result<Self>;
    pub fn session_id_value(&self) -> &str;  pub fn child_pid(&self) -> Option<u32>;
    pub fn is_running(&self) -> bool;        pub fn exit_code(&self) -> Option<i32>;
    pub fn stderr_tail(&self) -> String;     pub fn cancellation_token(&self) -> CancellationToken;
    pub fn options(&self) -> &StdioRuntimeOptions;
}
impl MemberRuntime for StdioMemberRuntime; impl Drop for StdioMemberRuntime;

// process_spawner.rs / external_cli_spawner.rs / ssh_spawner.rs
pub struct ProcessMemberSpawner;   // new(ProcessMemberConfig), config, is_running, live_members, child_pid
pub struct ExternalCliMemberSpawner;  // new(ExternalCliRuntimeConfig), config, is_running, live_members, child_pid
pub struct SshMemberSpawner;       // new(SshTransportSpec), spec, command_for(&request), argv_for(&request),
                                   // remote_command_for(&request), is_running, live_members, child_pid
// each: #[async_trait] impl MemberSpawner
```

`mod.rs` re-exports the surface flat, mirroring `agent_teams/__init__.py`'s member exports.
It also re-exports the schema types its own signatures use — `RoundOutcome`, `MemberStatus`, `ExecutionStatus`, `TeamRole`, `MemberLifetime` — so callers can drive the member seam through this module alone; `../schema/` remains their owner and this node never redefines them.
`openjiuwen_harness_protocol::HarnessState` is re-exported by `../schema/`; this node imports it and never defines a second state enum.

## Lifecycle rules

| Operation | From | To | Notes |
| --- | --- | --- | --- |
| `start` | `unstarted` | `starting` → `ready` | CAS; non-`unstarted` is `Error::validation("member '<name>' is not in unstarted state")` |
| `start` (runtime error) | `starting` | `unstarted` | rollback via CAS; a failed rollback is logged, the original error is returned unchanged |
| `mark` (same status) | any | same | idempotent `Ok(true)`, no store write |
| `mark` (illegal edge) | any | rejected | `Ok(false)`, store untouched |
| `mark` (lost CAS) | any | rejected | `Ok(false)` after refreshing the cached row |
| `note_round(Started)` | `idle` | `starting → running` | execution walks the `round_transitions` table in order |
| `note_round(Finished)` | `running` | `completing → completed → idle` | |
| `note_round(Aborted)` | `running` | `cancel_requested → cancelling → cancelled → idle` | |
| `note_round(Failed)` | `running` | `failed → idle` | |
| `note_round` (member mirror) | walk ends `running` | `busy` | applied only when the member table allows the edge |
| `note_round` (member mirror) | walk ends `idle` | `ready` | same "where legal" guard; a retiring member keeps `shutdown_requested` |
| `shutdown(lifetime, force)` | live status | `shutdown_requested` → `retire_target` | direct to target when `shutdown_requested` is unreachable (e.g. `unstarted → shut_down`) |
| `shutdown` (dormant, unstarted) | `unstarted` | rejected | `Error::validation("member '<name>' cannot shut down from status 'unstarted'")` |
| `shutdown` (already at target) | target | target | idempotent `Ok(target)` |
| `error(message)` | recoverable | `error` | best effort; logs the message; a departed member keeps its status and the call still returns `Ok(())` |

`retire_target(MemberLifetime::Shutdown | Dormant, force)`: `dormant` only for `Dormant` with `force == false`, `shut_down` otherwise.
The status machine never stops the runtime: `shutdown` only settles the row, and the runtime manager owns teardown order.

## Member process protocol

A child-process member is driven by one JSON object per line: leader→member request frames on the child's stdin, member→leader response frames on the child's stdout, stderr purely diagnostic.
Request frames: `{"type":"start","session_id":…}`, `{"type":"send","content":…,"immediate":false}`, `{"type":"abort","immediate":false}`, `{"type":"pause"}`, `{"type":"resume","query":…}`, `{"type":"stop"}`.
Response frames: `{"type":"ready","session_id":…}`, `{"type":"state","state":"idle|running|pausing|paused|terminated"}`, `{"type":"round","outcome":"started|finished|aborted|failed"}`, `{"type":"chunk","chunk":{…}}`, `{"type":"error","message":"…"}`.
The child must flush after every line; blank lines are skipped; unknown JSON fields are ignored; a garbage stdout line is logged and skipped, and a child that produces 100 consecutive garbage lines is treated as terminated.
Frame mapping: `ready` completes the pending `start` handshake only, `state` → `MemberObserver::on_state` (and the stored `HarnessState`), `round` → `on_round`, `chunk` → `on_chunk`, `error` fails the pending `start` and is otherwise logged without touching the lifecycle state.
Child exit stores `HarnessState::Terminated`, notifies observers exactly once and turns every later runtime call into a typed `member process … exited` error.
The full framing rules, transcript and mapping table are the `//!` docs of [`protocol`], which is the single source of truth for this wire format.

## Spawners

`MemberSpawnKind::InProcess` (the default) keeps the existing path: the harness factory builds an in-process runtime and `MemberSpawnOptions` is ignored.
The three child-process spawners (`ProcessMemberSpawner`, `ExternalCliMemberSpawner`, `SshMemberSpawner`) all launch a child through `StdioMemberRuntime` and differ only in the command they build.
Each spawner keeps its launched runtimes in its own `parking_lot::Mutex<HashMap<String, Arc<StdioMemberRuntime>>>`; re-spawn replaces the entry (the replaced runtime is stopped and dropped, killing its child), `despawn` removes first and then `stop`s, and an unknown member is `Ok(())`.
Per-request precedence is uniform: a request whose `options.kind` matches the spawner's kind and carries the per-kind config overrides the spawner's own config for that spawn only; otherwise the spawner's config applies (a request of another kind never picks up that kind's config).
`ProcessMemberSpawner` defaults to `team-member serve` and exports the identity through `MemberSpawnRequest::apply_identity_env` (`OPENJIUWEN_TEAM_NAME`, `OPENJIUWEN_MEMBER_NAME`, `OPENJIUWEN_SESSION_ID`, `OPENJIUWEN_MEMBER_WORKSPACE`); a spec that pre-sets one of those variables wins.
`ExternalCliMemberSpawner` launches a configured third-party CLI with the same env contract; `cli_agent`/`provider`/`model` are diagnostic labels only (logged, never on argv, never turned into env variables), and credentials come from the inherited environment plus `StdioCommand::env`.
`SshMemberSpawner` builds `ssh -p <port> [-i <key_file>] [-o ConnectTimeout=<s>] [-o UserKnownHostsFile=<p>] [-o StrictHostKeyChecking=no (only when disable_host_key_check)] [-o K=V …sorted] -- <user@host> '<remote command>'` and speaks the protocol over the ssh pipe.
SSH identity crosses as a remote `export …;` prologue inside the single quoted remote-command word (sshd drops channel env; the spawner never injects `-o SendEnv`, `BatchMode` or host-key flags of its own; `password` never reaches the argv).
A missing `ssh`, `team-member` or CLI binary is `ErrorKind::Config` from `StdioCommand::resolve_program`, naming the program and the fix.

## Constraints

- Only files under `src/member/` are edited; `../lib.rs`, `../schema/`, `../storage/`, `../runtime/` and sibling modules belong to other nodes.
- `openjiuwen_common::error::{Error, Result}` only; no panic on user input; no `unwrap`/`expect` outside `#[cfg(test)]`.
- Async-first: `#[async_trait]` for the object-safe seams; `parking_lot` locks are held for one map read/write only and never across `.await`; cancellation is `tokio_util::sync::CancellationToken`; every child-process write and handshake wait is bounded by `StdioRuntimeOptions::io_timeout`.
- Config errors for missing build dependencies and missing programs, validation errors for bad names/states, not-found for missing rows, unsupported for a capability this node does not implement.
- `tracing` only; no `println!`/`eprintln!`; secrets (an ssh `password`) are never logged and never placed on an argv.
- No `unsafe`; no global mutable state — every registry is a local instance owned by the caller.
- Files ≤1000 lines; `stdio.rs` (952 lines) is the single largest file because it owns one coherent lifecycle (launch, reader, writer, exit, cancellation) that must stay in one place; the status-machine suite lives in `lifecycle_tests.rs` and the child-process suites in their own `*_tests.rs`.
- `--no-default-features` must build and test: this node has no feature gates.

## Routing Table

- `./mod.rs` → module docs, flat re-exports, `#[cfg(test)]` module declarations
- `./runtime.rs` → `MemberRuntime`, `MemberObserver`, `RecordingObserver`, `ObserverEvent`, `ObserverSet`
- `./lifecycle.rs` → `TeamMember`: cached row, `mark`/`mark_execution`, `note_round`, `start`+rollback, `shutdown`, `error`
- `./lifecycle_tests.rs` → status-machine suite with `FakeMemberRuntime` (`#[cfg(test)]`)
- `./activity.rs` → `MemberActivityRegistry`, `IdleDebounce`, `DEFAULT_IDLE_DEBOUNCE`
- `./spawn.rs` → `MemberSpawner`, `MemberSpawnRequest` + `identity_env`/`apply_identity_env`, identity env constants, `InProcessSpawnHandle`, `SpawnRegistry`
- `./spawn_config.rs` → `MemberSpawnKind`, `MemberSpawnOptions`, `StdioCommand`, `ProcessMemberConfig`, `ExternalCliRuntimeConfig`, `SshTransportSpec`
- `./spawn_config_tests.rs` → serde/shape, `resolve_program`, ssh argv construction, options precedence
- `./protocol.rs` → `MemberRequest`, `MemberResponse`, frame codec and response accessors
- `./protocol_tests.rs` → pure encode/decode suite for every frame
- `./stdio.rs` → `StdioMemberRuntime`, `StdioRuntimeOptions`, `DEFAULT_IO_TIMEOUT`
- `./stdio_tests.rs` → handshake, fan-out, exit, stop/kill, drop, cancellation, timeouts
- `./process_spawner.rs` → `ProcessMemberSpawner` (`team-member` children)
- `./process_spawner_tests.rs` → launch/precedence/identity-env/despawn/registry suite
- `./external_cli_spawner.rs` → `ExternalCliMemberSpawner` (third-party CLI children)
- `./external_cli_spawner_tests.rs` → config labels, launch, identity env, despawn suite
- `./ssh_spawner.rs` → `SshMemberSpawner`, the remote `export` prologue and `argv_for`/`command_for`
- `./ssh_spawner_tests.rs` → literal argv assertions plus protocol-over-fake-ssh process cases
- `./test_child.rs` → `FakeChild` (`serve`, `silent`, `failing_start`, `with_body`, `path`, `program`, `sh_command`) plus `spawn_with_retry`, `SPAWN_ATTEMPTS`/`SPAWN_RETRY_BACKOFF`, `is_text_file_busy`: POSIX `sh` children that speak the protocol (`#[cfg(test)]`)

## Test Strategy

Unit tests live beside each file under `#[cfg(test)]`; `cargo test -p openjiuwen-agent-teams --lib` runs them offline and deterministically (1245 crate tests, **196 in this node**).
`lifecycle_tests.rs` (28) covers the status machine; `runtime.rs` (3) recording order and fan-out; `activity.rs` (14) movement, settled members, idempotent markers, sorted snapshots and the `IdleDebounce` boundary; `spawn.rs` (17) handle cancel/join/abort, registry bookkeeping, request serde and the identity-env contract; `spawn_config_tests.rs` (30) serde wire values, `resolve_program` failures, literal ssh argv shapes, quoting/escaping and options builders; `protocol_tests.rs` (31) every frame's exact JSON, defaults, unknown fields, error kinds and accessors; `stdio_tests.rs` (21) handshake, ordered frame fan-out, pause/resume/abort, refused start, immediate exit, garbage-line tolerance, stderr tail, idempotent stop of a silent child, termination notified exactly once, drop-kills-child (`kill -0` poll), cancellation, missing program, short-IO-timeout, fail-fast after exit and concurrent-write serialization; `process_spawner_tests.rs` (17), `external_cli_spawner_tests.rs` (14) and `ssh_spawner_tests.rs` (16) cover launch, per-request precedence, identity env (written to a temp file by the child and polled), missing binaries, despawn idempotence, re-spawn replacement, sorted `live_members` and object safety.
`FakeChild` writes a `#!/bin/sh` script with a unique name per fixture into a `tempfile::TempDir`, flushes it, `sync_all`s it and closes the descriptor before any test can spawn it.
The stdio, process and external-CLI suites launch it through `FakeChild::sh_command` (`sh <path>`): `sh` itself is never written, so those execs cannot be refused with `ETXTBSY` while parallel tests fork.
The SSH suite must exec the script directly (it stands in for the `ssh` binary and the spawner renders the whole argv), so its spawns go through `spawn_with_retry` — three attempts, 20 ms apart, retried only on a `Transport`-level `Text file busy` — and take the forgiving `SPAWN_BOUND` (30 s) while protocol awaits keep `BOUND` (10 s).
`test_child.rs` (5) proves the fixture's file shape and uniqueness, the `sh_command` shape, the `ETXTBSY` predicate, retry-then-succeed, no retry for a config error, and the bounded attempt count.
Every await in the child-process suites is bounded (`timeout` with 20 ms polling), there is no network, no `#[ignore]` and no fixed long sleep.
Every file has a doctest; all pass.
Definition of done for this node: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --offline` (also `--no-default-features`), `cargo clippy -p openjiuwen-agent-teams --all-targets --offline -- -D warnings`, and `cargo doc -p openjiuwen-agent-teams --no-deps` warning-free.

## Notes for Agents

- Python keeps the `TeamMember` handle, the member table and the event stream in sync; the Rust port keeps the row authoritative behind `TeamStore::transition_member` (CAS) and mirrors it into a cached projection for the synchronous getters. Call `refresh()` when a peer may have written.
- Event publication is deliberately absent: `TeamMember` has no messager. The runtime (`TeamBackend`) owns the topic and publishes `MemberStatusChangedEvent` / `MemberExecutionChangedEvent` after a successful `mark`.
- `mark` short-circuits an equal status to `Ok(true)` (Python's no-op path), so recovery code can assert `ready` unconditionally without polluting a status stream.
- `mark_execution` validates locally because the store's `update_member` patch path is verbatim; it is an error (not `Ok(false)`) to ask for an illegal execution edge, because only `note_round` is expected to tolerate that.
- `note_round` skips execution steps the current status cannot reach, which makes late or duplicated observer callbacks harmless; the final step is what the row ends on.
- A round walk that changed nothing mirrors no lifecycle status unless the member is still `busy`/`starting`, so a duplicate terminal note cannot revive a `paused` or `stopped` member.
- `shutdown` mirrors the member-side settle (`TeamAgent.shutdown_self`) rather than the leader-side request (`shutdown_member` writes `shutdown_requested` and stops there): it may write the terminal target directly when `shutdown_requested` is unreachable, so a never-started member still retires to `shut_down`.
- The activity registry is a deliberate Rust reshape of Python's `IdleSignal`-returning registry: it exposes the questions (`is_moving`, `settled_members`, `idle_since`, `last_activity`) and leaves the marker timer to `TeamBackend`; `IdleDebounce` is the only timing concept and it is a pure predicate.
- Dropping a `tokio::task::JoinHandle` detaches the task, so every handle that leaves `SpawnRegistry` must be joined or aborted: `take`/`cancel_all` cancel the token and hand the handles back, and `insert` returns the replaced handle. `TeamBackend` is the documented owner.
- `MemberSpawner` builds but never starts a runtime: `MemberRuntime::start` is called by `TeamMember::start` so a failed start can roll the row back. `StdioMemberRuntime::spawn` therefore only launches the child; `start` performs the `ready` handshake.
- `MemberSpawnRequest::options` is the seam the runtime dispatches on: `MemberSpawnKind::InProcess` (default) keeps the harness path, and the three child-process kinds are opt-in per member. Nothing in this node selects a spawner automatically.
- The child-process identity contract is env-only because the protocol's `start` frame carries just the session id; `apply_identity_env` skips empty request fields and never overrides an explicitly configured value.
- `StdioMemberRuntime::start` ignores the `Option<Arc<dyn Session>>` argument: only the session id can cross a process boundary, and harness-state persistence belongs to the child. `session_id()` is fixed at construction, so a `ready` frame naming a different id only logs a warning.
- A child-process runtime's `state()` starts at `Idle`, follows the child's `state` frames, and becomes `Terminated` on exit; `start` itself changes no state.
- `StdioMemberRuntime::stop` is idempotent and `despawn` relies on that: best-effort `stop` frame, kill, bounded wait, then `Terminated` + one observer notification.
- `SshMemberSpawner` deliberately does not call `apply_identity_env` (the local environment must stay untouched) and does not inject any ssh flag beyond what `SshTransportSpec` produces; the remote command is one quoted argv element because `ssh` re-parses it with the remote login shell.
- `StdioCommand::env` is additive over the inherited environment and never clears it; that is how credentials reach a child without appearing in an argv.
- `TeamMember` implements `Debug` non-exhaustively; `ObserverSet` prints its observer count. `status()` stays the synchronous cached projection, so the store-reading accessor is `stored_status()`.

## Known gaps and deferred

- `MemberRuntime::send` returns `Result<()>` and therefore drops any transport-level `SendReceipt`: harness-level receipts are not surfaced through the member seam, and a consumer that needs one must wrap the runtime in its own `MemberRuntime`.
- The `team-member` binary's `serve` subcommand — the child end of the member protocol — is not wired yet; `ProcessMemberConfig::default()` already launches `team-member serve`, so process mode needs that subcommand at the crate root (`../skill/`, outside this node).
- External-CLI protocol translation is deferred: Python's `external/cli_agent/*` adapters (`format_input`/`summarize_output_line`, `StdinPipeInjector` plain-text or `INPUT_CODEX_PROTO`, `task_complete`/`turn.completed` markers) and `model_compat.py` provider/model negotiation are not ported; a third-party CLI must speak the member protocol itself or be wrapped by a script.
- Python implements the SSH transport with `asyncssh` and never runs the `ssh` binary; the Rust port shells out to `ssh` instead. SSH attach to a team hosted in another process (shared store or remote runtime) stays deferred, as do remote team stores and distributed transports.
- Vendor CLI SDKs (`claude`, `codex`, `dsh` harnesses) remain deferred in `../harness/` and `openjiuwen-harness-providers`; this node only carries a generic stdio child.
- Python's rail/memory/background-controller helpers on `MemberRuntime` (`find_rails`, `register_member_tools`, `inject_member_memory`, `set_background_task_controller`, `workspace`, `sys_operation`) are not part of the Rust seam; the harness adapter reaches its own agent directly.
- Process-mode extras from Python remain out of scope: health-check ping/pong messages (`MessageType.HEALTH_CHECK`), the `SpawnConfig` health/shutdown timeouts, and fork/compact context injection (`inprocess_spawn.py`).
