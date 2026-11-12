# src/taskboard — task board, dependency graph, review gate

## Intent

The team's task board: the only writer of task rows, dependency edges and review votes.
It turns the task-board operations of Python `TeamTaskManager` (`openjiuwen/agent_teams/tools/task_manager.py`) and the atomic mutations of `tools/database/task_dao.py` into one domain object over [`crate::storage::TeamStore`].
The board is policy, the store is state: the store validates keys/transitions, the board decides *when* a task is claimable, when it unblocks, and whether a review passes.
It takes no LLM-facing arguments and renders no prose — the runtime's tool wrappers do that — and it never touches the messager: every mutation returns the events for the caller to publish.

## API Surface

```rust
// mod.rs
pub struct TaskBoardLimits { pub default_max_review_rounds: u32, pub max_subtask_depth: usize }  // 3, 0
impl TaskBoardLimits { pub const fn new(); with_default_max_review_rounds(u32); with_max_subtask_depth(usize); }

pub struct TaskMutation { pub task: TaskRecord, pub events: Vec<EventPayload>, pub unblocked: Vec<String> }
impl TaskMutation { pub fn new(task, events); pub fn task_id(&self) -> &str; }

pub struct TaskOpResult { pub ok: bool, pub reason: Option<String> }
impl TaskOpResult { pub const fn ok(); pub fn failed(reason: impl Into<String>); pub const fn is_ok(&self); pub fn reason(&self) -> Option<&str>; }
impl From<Result<()>> for TaskOpResult;

pub const REASON_STALE_CONTENT: &str;                  // board.rs

// board.rs
pub struct TaskBoard;   // Debug + Clone
impl TaskBoard {
    pub fn new(store: Arc<dyn TeamStore>, team_name: impl Into<String>, limits: TaskBoardLimits) -> Self;
    pub fn with_defaults(store, team_name) -> Self;
    pub fn store(&self) -> &Arc<dyn TeamStore>;  pub fn team_name(&self) -> &str;  pub const fn limits(&self) -> TaskBoardLimits;

    // creation
    pub async fn create_task(&self, spec: TaskGraphSpec) -> Result<TaskMutation>;
    pub async fn create_subtask(&self, parent_id: &str, spec: TaskGraphSpec) -> Result<TaskMutation>;
    // ownership
    pub async fn claim(&self, task_id: &str, member: &str) -> Result<TaskOpResult>;
    pub async fn assign(&self, task_id: &str, assignee: &str, reason: Option<&str>) -> Result<TaskOpResult>;
    pub async fn reassign(&self, task_id: &str, from: &str, to: &str) -> Result<TaskOpResult>;
    pub async fn set_plan(&self, task_id: &str, member: &str, plan: &str) -> Result<TaskOpResult>;
    // completion / review / cancel / reset
    pub async fn complete(&self, task_id: &str, member: &str, summary: &str) -> Result<TaskMutation>;
    pub async fn verify(&self, task_id: &str, reviewer: &str, verdict: Verdict, feedback: &str) -> Result<TaskMutation>;
    pub async fn record_vote(&self, task_id: &str, reviewer: &str, verdict: Verdict) -> Result<TaskMutation>;
    pub async fn review_tally(&self, task_id: &str) -> Result<ReviewTally>;
    pub async fn settle_review(&self, task_id: &str) -> Result<TaskMutation>;
    pub async fn cancel(&self, task_id: &str) -> Result<TaskMutation>;
    pub async fn reset(&self, task_id: &str) -> Result<TaskMutation>;
    // reads
    pub async fn view(&self, filter: &TaskFilter) -> Result<Vec<TaskSummary>>;
    pub async fn detail(&self, task_id: &str) -> Result<Option<TaskDetail>>;
    pub async fn claimable(&self) -> Result<Vec<TaskSummary>>;
    pub async fn subtasks(&self, parent_id: &str) -> Result<Vec<TaskSummary>>;
    pub async fn is_settled(&self) -> Result<bool>;
    pub async fn is_completed(&self) -> Result<bool>;
    pub async fn stale_tasks(&self, now_ms: u64, claim_idle_ms: u64, pending_idle_ms: u64) -> Result<Vec<TaskSummary>>;
}

// graph.rs
pub fn detect_cycle(edges: &BTreeMap<String, BTreeSet<String>>) -> Option<Vec<String>>;  // ["a","b","a"]
pub fn depth_of(parent_chain: &[String]) -> usize;                                       // ancestors = depth

// review.rs
pub struct ReviewTally { pub round: u32, pub pass_count: u32, pub fail_count: u32, pub reviewer_count: u32, pub passed: bool }
impl ReviewTally { pub const fn voted(&self) -> u32; pub const fn decided(&self) -> bool; pub const fn is_empty(&self) -> bool; }
pub fn tally(votes: &[ReviewVote]) -> ReviewTally;
pub fn tally_with(votes: &[ReviewVote], reviewers: &[String]) -> ReviewTally;
```

`snake_case` serde on every serializable type (`TaskBoardLimits`, `TaskOpResult`, `ReviewTally`), unknown fields tolerated, missing fields default.

## Lifecycle rules

- **Creation**: `task_id` is `spec.task_id` (trimmed) or `task_<32 lowercase hex>`; a duplicate id is `Error::Validation`.
- **Blocking**: a new task is `blocked` when any `depends_on` id is missing from the board or non-terminal, else `pending`. A `depended_by` id must exist (`Error::Validation`).
- **Edges**: `depends_on` → `add_dependency(new, dep)`; `depended_by` → `add_dependency(dependent, new)`; subtask → the parent depends on the child. All edges land before the row.
- **Cycles**: the post-mutation edge set is checked with `detect_cycle` before any write; a cycle returns `Error::Validation("Circular dependency: a -> b -> a")` with the path rotated to start at its smallest id, and nothing is written.
- **Claim**: only `pending`; an assignee must be the caller. A re-claim by the owner is a no-op success that refreshes `observed_revision` (claiming counts as reading the revision).
- **Assign**: leader path. Keeps the status except `pending`, which enters the owner's entry gate — `build_mode` → `in_progress`, `plan_mode` → `planning`; a `blocked` task keeps `blocked`. Appends a `assigned to <name>: <reason>` change note when a reason is given. Refuses a different current owner (use `reassign` / `reset`).
- **Reassign**: assignee swap only (`pending|blocked|in_progress`, see `TaskStatus::is_reassignable`); the status never bounces through `pending`, so no spurious release wake-up.
- **Plan**: `set_plan` requires the caller to be the assignee and `pending|in_progress`, moves to `planning`, stores the plan in `note`.
- **Complete**: sources `in_progress` and `planning` (a plan-mode task resolves `planning -> in_progress -> completed` in one row write because the transition table has no direct edge). Refused when the caller is not the assignee, when `revision > observed_revision`, or from any other status. Reviewers empty → `completed`; otherwise → `in_review` with `review_round += 1` (rounds are monotonic per task, so the previous round's votes are simply never read again — that is the "reset votes").
- **Verify (autonomous)**: requires `in_review` and reviewer membership. `pass` → `completed` + unblock; `fail` → `in_progress`, the assignee is cleared only when the reviewer is not the author, the feedback becomes a change note and the note.
- **Vote (scheduled)**: `record_vote` persists the vote for the current round and changes nothing else; `review_tally` reports the round; `settle_review` applies it — pass → `completed` + unblock, fail → `in_progress` with the author still assigned, and the failed round recorded in `note`. A round settles only when every reviewer voted: `passed = voted >= reviewer_count && pass > fail`; a tie never passes; a vote from a former reviewer is ignored; a repeated vote keeps the latest.
- **Cancel**: any non-terminal → `cancelled`, assignee kept on the row, dependents unblocked like a completion. Subtask cascade is the runtime's depth-first job, not the board's.
- **Reset**: `completed|cancelled|in_review` → `pending`, clearing assignee/summary/note. `review_round` is deliberately not rewound (votes are kept forever and a rewound round would re-read the previous cycle's verdicts).
- **Unblocking**: after a task becomes terminal, every direct dependent still `blocked` whose dependencies are all terminal becomes `pending` and is listed in `TaskMutation::unblocked` (sorted). The pass is idempotent.
- **Settled vs completed**: `is_settled` = no open task (an empty board is settled); `is_completed` = settled **and** at least one task exists, so an untouched board never reads as a finished team.
- **Stale sweep**: `planning|in_progress` stale at `age >= claim_idle_ms`; unassigned `pending` stale at `age >= pending_idle_ms` (inclusive, `age = now_ms - updated_at`); `blocked`, `in_review` and terminal rows are never stale. Rows are sorted by `task_id`.

### Stable reason strings (shown to the model — treat as API)

| Situation | String |
| --- | --- |
| complete/reassign not the assignee | `task is assigned to '<name>'` (`<unassigned>` when free) |
| stale content | `content changed; re-read the task before completing` |
| complete from a wrong status | `task '<id>' cannot be completed from status '<status>'` |
| claim held by somebody else | `task '<id>' is already claimed by '<owner>'` |
| claim from a wrong status | `task '<id>' cannot be claimed from status '<status>' (only pending tasks are claimable)` |
| assign refused | `task '<id>' cannot be assigned from status '<status>'` / `… is already claimed by '<owner>'; reset the task before reassigning to '<assignee>'` |
| unknown member | `member '<name>' not found in team '<team>'` |
| reassign refused | `task '<id>' is not assigned to '<from>'` / `task '<id>' cannot be reassigned from status '<status>'` / `task '<id>' has no current assignee to reassign from` |
| plan refused | `task '<id>' cannot enter planning from status '<status>'` |
| verify refused | `task '<id>' is not under review (status '<status>'); nothing to verify` / `'<reviewer>' is not a reviewer of task '<id>'; cannot verify it` |
| vote refused | `task '<id>' is not under review (status '<status>'); nothing to vote on` / `'<reviewer>' is not a reviewer of task '<id>'; cannot vote on it` |
| settle refused | `task '<id>' is not under review (status '<status>'); nothing to settle` / `review tally of task '<id>' is not decided yet (<voted> of <n> votes)` |
| cancel refused | `task '<id>' cannot be cancelled from status '<status>' (already terminal)` |
| reset refused | `task '<id>' cannot be reset from status '<status>'; only completed / cancelled / in-review tasks can be reset` |
| subtask too deep | `subtask depth <n> exceeds max_subtask_depth <m>` |
| cycle | `Circular dependency: <a -> b -> a>` |
| missing task | `Error::NotFound("task '<id>' not found")` |

`TaskOpResult`-returning calls report these as `ok = false` + `reason`; `TaskMutation`-returning calls report them as `Err(Error::Validation(reason))`.

### Events

Sources: create → `task_created` then `task_status_changed`; complete/cancel/reset/assign-side writes → `task_status_changed`; `verify` → `task_review_vote`, `task_status_changed`, `task_verified` (exactly one `task_verified`); `record_vote` → `task_review_vote` only (no status change); `settle_review` → `task_status_changed`, `task_verified`, `task_review_vote` (round-tagged, `reviewer` empty); `create_subtask` → the child's create pair plus the parent's `task_status_changed` when it was parked.
Unblocked tasks are reported through `TaskMutation::unblocked`, not as events; the runtime decides whether to wake members by message or by event.

## Constraints

- Only `src/taskboard/**` is edited; `src/lib.rs`, `Cargo.toml`, `src/schema/`, `src/storage/` and sibling modules are read-only reference.
- Reads only through [`crate::storage::TeamStore`]; no direct row maps, no process-global state, no clock of its own (`now_ms` is a parameter of `stale_tasks`).
- Every mutation writes exactly one task row (`insert_task` with `updated_at = 0` so the store stamps the write) plus the edges/votes it needs; `transition_task` is not used because a whole-row write keeps status, assignee, notes and revision consistent in one step.
- Status legality is checked against `TaskStatus::can_transition_to` / `is_reassignable` before the write; the two documented exceptions write the row directly (`reset`, parent parking).
- Errors: `openjiuwen_common::error::{Error, Result}`; `Error::NotFound` for missing tasks, `Error::Validation` for caller-supplied bad input. No panics, no `unwrap`/`expect` outside tests.
- No I/O beyond the store; no `tracing` calls (the runtime logs the mutation at its own level); no `unsafe`.
- Files ≤1000 lines (`board_tests.rs` is the sanctioned long file: one focused test per rule; `board.rs` is sanctioned at ~1200 lines with a top-of-file note — the board is one object and its lifecycle rules only read as a whole, while the pure decisions already live in `graph.rs`/`review.rs`).

## Routing Table

- `./mod.rs` → node contract: `TaskBoardLimits`, `TaskMutation`, `TaskOpResult`, re-exports, `missing_task`
- `./board.rs` → `TaskBoard` and every board operation, `Snapshot` projections, `REASON_STALE_CONTENT`
- `./graph.rs` → `detect_cycle`, `depth_of` (pure)
- `./review.rs` → `ReviewTally`, `tally`, `tally_with` (pure)
- `./board_tests.rs` → the crate-internal board suite (`#[cfg(test)]`, in-memory store only)
- Records and DTOs used here (`TaskRecord`, `TaskGraphSpec`, `TaskSummary`, `TaskDetail`, `TaskFilter`, `ReviewVote`, `Verdict`, `TaskPatch`) live in `../schema/task.rs`; statuses and `TASK_TRANSITIONS` in `../schema/status.rs`. There is no `priority` field and no `next_claimable`/`list`/`summaries` alias — `claimable()` and `view()` are the only read entry points.

## Test Strategy

`board_tests.rs`, all async and deterministic against `InMemoryTeamStore` (`#[tokio::test]`, unwrap/expect allowed inside the module): creation with dependencies and minted ids, cycle rejection leaving no rows or edges, unblocking on complete and on cancel, claim ownership and reason strings, completion with and without reviewers, verify pass/fail, the scheduled tally (1 of 3 → not decided, then pass/fail majority), the revision gate plus re-read recovery, assign/reassign/set_plan/reset/cancel, the subtask depth gate and parent parking, `is_settled`/`is_completed`, the inclusive stale boundaries, and the `view`/`claimable`/`detail`/`subtasks` shapes.
`graph.rs` and `review.rs` carry their own pure unit tests (cycle determinism, depth, tally majorities/ties/duplicates/former reviewers).
Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --lib`, `cargo test -p openjiuwen-agent-teams --no-default-features --lib`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` green for this node.

## Notes for Agents

- Python `TeamTaskManager` is both domain and tool surface; here the tool wrappers belong to `src/runtime/` and the board stays argument-free. Do not add prompt strings or language handling here.
- Deliberate deviations from Python, all observable: `claim`/`assign`/`reassign`/`set_plan` return `TaskOpResult` without events (Python publishes `TASK_CLAIMED` from the same call), so the runtime publishes from its own dispatch loop; `record_vote` carries no tally counts (the crate's `TaskReviewVoteEvent` has none) — read `review_tally`; `reset` keeps `review_round` monotonic; `is_completed` requires a non-empty board; subtask cancellation is not cascaded.
- Not ported: plan-file storage and `submit_plan`/`approve_plan`, structured reviewers (`verifier`/`challenger`/`inspector`, scored votes), `TASK_LIST_DRAINED`/`TASK_UNBLOCKED` publication, one-active-task-per-member enforcement, leader-ownership validation (`_validate_task_owner`), and batch `add_graph` (call `create_task` per spec; the board rejects a cycle on each step).
- The board re-reads the whole board for every query (`Snapshot`: tasks + edges + roster statuses) because the store seam has no aggregate query; at team scale (tens of tasks) that is cheaper than caching and cannot go stale.
- `TaskSummary::stale_against` is rendered as `r<parent revision now> vs r<revision the subtask was derived from>` and is only set while the parent advanced and the subtask is not `completed`.
