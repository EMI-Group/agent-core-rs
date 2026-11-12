# openjiuwen-agent-teams/src/runtime/tests
## Intent
The crossed-seam integration suite of the team runtime.
The unit tests beside `team_agent`, `leader`, `manager`, `backend`, `spawner` and `dispatch` pin each file in isolation; this node drives the chains between them — manager → adapters → leader loop → backend → store/bus → spawner → harness factory → workspace provider.
Every case is offline and deterministic: scripted harnesses, a recording workspace provider and an `InMemoryTeamStore`, no model, socket or clock.
## API Surface
```rust
pub(crate) struct Fixture {              // one manager over fresh fakes and its own store
    pub manager: Arc<TeamRuntimeManager>, pub factory: Arc<ScriptedFactory>,
    pub workspace: Arc<RecordingWorkspaceProvider>, pub store: Arc<InMemoryTeamStore>,
}
impl Fixture {
    pub fn new() -> Self;
    pub async fn activate(&self, spec: &TeamAgentSpec, session: &str) -> Arc<TeamAgent>;
    pub fn harness(&self, member: &str) -> Arc<ScriptedHarness>;
    pub fn sends(&self, member: &str) -> Vec<String>;      // texts the member's harness received
}
pub const BOUND: Duration;                       // 2 s: the upper bound of every wait
pub const WINDOW: Duration;                      // 150 ms: the open-task observation window
pub type Recorder = Arc<Mutex<Vec<EventMessage>>>;
pub fn subscribe_observer(&Arc<dyn Messager>, &[TeamTopic]) -> Recorder;   // member "observer"
pub fn recorded(&Recorder) -> Vec<EventMessage>;  pub fn event_types(&Recorder) -> Vec<String>;
pub async fn drain(BoxStream<'static, Result<AgentChunk>>) -> Vec<Result<AgentChunk>>;
pub fn marker_count(&[Result<AgentChunk>]) -> usize;
pub async fn wait_until(what: &str, impl FnMut() -> bool);
pub use team_agent::test_support::{blueprint, RecordingWorkspaceProvider};
```
## Constraints
- Only files under `src/runtime/tests/` are edited; `src/`, `Cargo.toml` and sibling modules are read-only.
- Fakes are re-used from `crate::runtime::team_agent::test_support` (`ScriptedHarness`, `ScriptedFactory`, `RecordingWorkspaceProvider`, `blueprint`, `deps`); never re-implemented here.
- All waits are bounded: `tokio::time::timeout(BOUND, ..)` or `wait_until(..)`; the only fixed delay is the single 150 ms (`WINDOW`) observation, never a bare long sleep.
- No asserts on log output, on internal chunk counts, or on timings tighter than 150 ms; no `println!`, no `unsafe`.
- Every test is independent: a fresh `Fixture` per test, and the global registry suite clears the process-global slot with a `Drop` guard even on failure.
- Harness names must be unique per team inside one `ScriptedFactory` (it keys harnesses by member name), so the two-session suite uses distinct leader names.
## Test Strategy
- `end_to_end.rs` — manager activate → `live`/`live_state`; `Agent::invoke` and `Team::invoke` return the scripted leader answer; the stream carries exactly one `team.completed` marker last; a `$human @member body` delivery never reaches the leader while a plain query does; an open task keeps the run alive until the test completes it; two teams in two sessions stay isolated; the global registry installs, activates, runs and clears.
- `members.rs` — `auto_start_all` readies every teammate with a distinct workspace root from the recording provider; a second `spawn_member` is idempotent (no second harness); `shutdown_member` settles on `dormant`/`shut_down` in both the handle and the store; an unknown member is `NotFound`; `create_task` persists the row and publishes `task_created`.
- `mail.rs` — a direct message is one row in the target's mailbox plus one `message` event; a broadcast is one row plus one `broadcast` event; a repeated target is de-duplicated; a live run turns unread mail and a posted `task_status_changed` into leader steering deliveries, with the board event stamped with the team as sender.
- `support.rs` — the fixture, the recorder, the bounded drain and the bounded poll helper; no tests.
- `mod.rs` — module docs and the module list.
## Notes for Agents
- `ScriptedHarness` exposes `sends()`, not a `deliveries()` alias: the leader deliveries asserted here are its recorded `Send` texts.
- Team runs are one-per-team (the bus receiver is single-consumer), so a suite that needs two live runs activates two teams.
- The open-task suite creates its task immediately after `stream()` and before the first poll; the run needs two 25 ms settle ticks to finish, so the window is what keeps the ordering deterministic.
