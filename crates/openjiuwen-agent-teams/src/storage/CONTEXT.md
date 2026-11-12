# openjiuwen-agent-teams/src/storage

## Intent

The persistence seam of the crate: one object-safe `TeamStore` trait over the team, roster, task board, dependency-edge, review-vote and mailbox tables, plus the deterministic in-process backend every test and every single-process runtime uses.
Python parity source: `openjiuwen/agent_teams/tools/database/` (SQLAlchemy DAOs `team_dao`, `member_dao`, `task_dao`, `message_dao` and the SQLite `TeamDatabase` they run on) and the `team_backend` persistence calls that drive them.
This node is a *state* seam, not a policy seam: it validates row keys and status transitions, and answers queries in a fixed order.
Dependency unblocking, review gating, dispatch, member spawning and message routing belong to the sibling nodes (`../taskboard/`, `../member/`, `../runtime/`), which code against this trait.

## API Surface

```rust
pub const MEMORY_STORAGE_KIND: &str = "memory";
pub const PLAIN_MESSAGE_PROTOCOL: &str = "plain";

pub struct TeamInfoRecord {          // Python TeamInfo
    pub team_name: String, pub display_name: String,
    pub leader_member_name: String,  // default DEFAULT_LEADER_MEMBER_NAME
    pub desc: String, pub prompt: String,
    pub dispatch_mode: DispatchMode, pub enable_task_verification: bool,
    pub created: u64, pub updated_at: u64,
}
impl TeamInfoRecord {
    pub fn new(team_name: impl Into<String>) -> Self;
    pub fn display_name_or_default(&self) -> &str;      // display_name, else team_name
}

pub struct MemberRecord {            // Python TeamMember
    pub team_name: String, pub member_name: String, pub display_name: String, pub desc: String,
    pub role: TeamRole, pub status: MemberStatus, pub execution: ExecutionStatus,
    pub mode: MemberMode, pub prompt: String,
    pub model_ref: Option<String>, pub worktree: Option<String>, pub updated_at: u64,
}
impl MemberRecord {
    pub fn new(team_name: impl Into<String>, member_name: impl Into<String>, role: TeamRole) -> Self;
    pub fn display_name_or_default(&self) -> &str;
}

pub struct MemberPatch {             // every field Option; `None` = leave unchanged
    pub display_name: Option<String>, pub desc: Option<String>, pub role: Option<TeamRole>,
    pub status: Option<MemberStatus>, pub execution: Option<ExecutionStatus>,
    pub mode: Option<MemberMode>, pub prompt: Option<String>,
    pub model_ref: Option<String>, pub clear_model: bool,
    pub worktree: Option<String>, pub clear_worktree: bool,
}
impl MemberPatch {
    pub fn new() -> Self;              pub fn is_empty(&self) -> bool;
    pub fn with_display_name(..) -> Self; pub fn with_desc(..) -> Self;
    pub fn with_role(..) -> Self;      pub fn with_status(..) -> Self;
    pub fn with_execution(..) -> Self; pub fn with_mode(..) -> Self;
    pub fn with_prompt(..) -> Self;    pub fn with_model_ref(..) -> Self;
    pub fn clearing_model(..) -> Self; pub fn with_worktree(..) -> Self;
    pub fn clearing_worktree(..) -> Self;
    pub fn apply(&self, record: &mut MemberRecord) -> bool;   // clear_* wins over the setter
}

pub struct MessageRecord {           // Python TeamMessage
    pub team_name: String, pub message_id: String, pub from_member_name: String,
    pub to_member_name: Option<String>, pub content: String, pub timestamp: u64,
    pub broadcast: bool, pub protocol: String /* "plain" */, pub is_read: bool,
}
impl MessageRecord {
    pub fn direct(team, message_id, from, to, content, timestamp) -> Self;
    pub fn broadcast(team, message_id, from, content, timestamp) -> Self;
    pub fn is_broadcast(&self) -> bool;
}

#[async_trait] pub trait TeamStore: Send + Sync {
    fn name(&self) -> &'static str;                                   // "memory"
    async fn upsert_team(&self, rec: TeamInfoRecord) -> Result<()>;
    async fn team(&self, team: &str) -> Result<Option<TeamInfoRecord>>;
    async fn insert_member(&self, rec: MemberRecord) -> Result<()>;
    async fn member(&self, team: &str, name: &str) -> Result<Option<MemberRecord>>;
    async fn list_members(&self, team: &str) -> Result<Vec<MemberRecord>>;
    async fn update_member(&self, team: &str, name: &str, patch: &MemberPatch) -> Result<bool>;
    async fn transition_member(&self, team: &str, name: &str, from: MemberStatus, to: MemberStatus) -> Result<bool>;
    async fn insert_task(&self, rec: TaskRecord) -> Result<()>;
    async fn task(&self, team: &str, task_id: &str) -> Result<Option<TaskRecord>>;
    async fn list_tasks(&self, team: &str, filter: &TaskFilter) -> Result<Vec<TaskRecord>>;
    async fn update_task(&self, team: &str, task_id: &str, patch: &TaskPatch) -> Result<bool>;
    async fn transition_task(&self, team: &str, task_id: &str, from: TaskStatus, to: TaskStatus) -> Result<bool>;
    async fn add_dependency(&self, team: &str, task_id: &str, depends_on: &str) -> Result<()>;
    async fn dependencies(&self, team: &str, task_id: &str) -> Result<Vec<String>>;
    async fn dependents(&self, team: &str, task_id: &str) -> Result<Vec<String>>;
    async fn insert_vote(&self, rec: ReviewVote) -> Result<()>;
    async fn votes(&self, team: &str, task_id: &str, round: u32) -> Result<Vec<ReviewVote>>;
    async fn insert_message(&self, rec: MessageRecord) -> Result<()>;
    async fn unread_messages(&self, team: &str, member: &str) -> Result<Vec<MessageRecord>>;
    async fn mark_read(&self, team: &str, member: &str, ids: &[String]) -> Result<()>;
    async fn read_watermark(&self, team: &str, member: &str) -> Result<Option<u64>>;
    async fn set_read_watermark(&self, team: &str, member: &str, ts: u64) -> Result<()>;
    async fn message_count(&self, team: &str, member: &str) -> Result<usize>;
}

pub struct InMemoryTeamStore;                    // ::new(), Default, Arc<dyn TeamStore>
pub fn open_store(spec: &StorageSpec) -> Result<Arc<dyn TeamStore>>;
```

Task/status types are *not* redefined here: `TaskRecord`, `TaskPatch`, `TaskFilter`, `ReviewVote`, `Verdict`, `TaskRevision` come from `../schema/task.rs` and `MemberStatus`, `ExecutionStatus`, `TaskStatus`, `TeamRole`, `MemberMode`, `DispatchMode` from `../schema/status.rs` and `../schema/team.rs`.

## Constraints

- Only this directory is edited; `../lib.rs`, `../constants.rs`, `../schema/` and the sibling modules belong to other nodes.
- `openjiuwen_common::error::{Error, Result}` only; no panic on user input; no `unwrap`/`expect` outside `#[cfg(test)]`.
- No `unsafe`. No process-global store: `open_store` returns an independent instance per call, and the crate shares one by holding the `Arc`.
- Async-first tokio with `#[async_trait]`; `InMemoryTeamStore` holds a `parking_lot::RwLock` for the duration of one map operation only and never awaits while holding it.
- `tracing` only; this module currently logs nothing (there is no recoverable anomaly a caller cannot see from a `Result`).
- Every public item carries a doc comment (workspace `missing_docs`), and every fallible method documents its errors (clippy `missing_errors_doc`).
- Files ≤1000 lines; the test suite lives in `store_tests.rs` rather than inline so the implementation stays readable.

## Behaviour (wire-visible and caller-visible)

- Row keys: team ∈ `team_name`; roster ∈ `(team_name, member_name)`; task ∈ `(team_name, task_id)`; message ∈ `(team_name, message_id)`. An empty key is `Error::validation("<field> must not be empty")`.
- `insert_*` / `upsert_team` are keyed upserts, not append-only inserts: re-inserting the same key replaces the row (a re-inserted member therefore carries whatever lifecycle state the incoming record has — use `transition_member`/`update_member` for lifecycle changes).
- Stamps: a caller-supplied non-zero `updated_at`/`created` is preserved; a `0` is stamped with `max(unix_millis, previous + 1)` so ordering survives two writes in the same millisecond. `upsert_team` keeps the existing `created` when the incoming record leaves it `0`.
- `transition_member` / `transition_task` are compare-and-set: `true` only when the stored status equals `from` **and** `schema::is_valid_member_transition` / `TaskStatus::is_valid_transition` accepts the edge; a `false` is a normal lost race, not an error. Both refresh `updated_at`.
- `update_member` / `update_task` apply the patch verbatim (no transition validation); an empty patch on a known row returns `true` and writes nothing; an unknown row returns `false`.
- `update_task` bumps `revision` and appends a rendered `TaskRevision` note (`r{n}: content updated` / `r{n}: title updated` / `r{n}: title and content updated`) only when the title or content actually changes and the patch does not set `revision` explicitly. `clear_assignee` wins over `assignee`.
- `list_members` sorts by `member_name`; `list_tasks` sorts by `updated_at` descending then `task_id` ascending and truncates to `TaskFilter::limit`; `dependencies`/`dependents` sort lexicographically; `votes` sorts by `reviewer`; `unread_messages` sorts by `timestamp` then `message_id`.
- `add_dependency` is idempotent, rejects a self-dependency (`Error::validation("task '<id>' cannot depend on itself")`) and does not require either task row to exist yet.
- `insert_vote` replaces a reviewer's vote in the same `(team, task, round)` instead of appending; rounds are isolated from each other.
- Mailbox: a direct row is unread while `is_read == false` and `to_member_name == member`; a broadcast is unread while its `timestamp` is strictly greater than the member's watermark and the member is not its sender. No watermark means every broadcast is unread.
- `mark_read` flags direct rows only, for the addressed member; broadcast ids are skipped silently (read state for broadcasts is the watermark). Advancing a broadcast read state is `set_read_watermark`, which never regresses.
- `message_count` is the unread count (`unread_messages(..).len()` without materializing rows) — the mailbox-depth signal for the runtime's idle sweep.
- `open_store`: `"memory"` (or an empty/whitespace kind) → fresh `InMemoryTeamStore`; `"sqlite" | "postgresql" | "mysql"` → `Error::unsupported("storage backend '<kind>' is not implemented; use 'memory'")`; any other kind → `Error::unsupported("unknown storage backend '<kind>'; use 'memory'")`. `StorageSpec::params` is accepted and ignored.

## Routing Table

- `./mod.rs` → `TeamStore` trait, `MEMORY_STORAGE_KIND`, `open_store`, flat re-exports
- `./records.rs` → `TeamInfoRecord`, `MemberRecord`, `MemberPatch`, `MessageRecord`, `PLAIN_MESSAGE_PROTOCOL` (+ wire-shape tests)
- `./memory.rs` → `InMemoryTeamStore`, stamping helpers, the unread predicate
- `./store_tests.rs` → the store suite on the `dyn TeamStore` seam (`#[cfg(test)]`)

## Test Strategy

`cargo test -p openjiuwen-agent-teams --lib` covers 27 storage tests beside the code; all run offline and deterministically, with no sleeps (caller-supplied stamps, or assertions that a stamp strictly grew).

- `records.rs`: serde wire names and defaults for all four records (including `"autonomous"`, `"build_mode"`, `"unstarted"`, `"idle"`, `protocol: "plain"`, `clear_model: false`), unknown-field tolerance and full-default decoding, `MemberPatch` merge/clear/no-op/`is_empty`, message constructors.
- `store_tests.rs`: team upsert (created preservation, replacement, store stamping), empty-key validation for every row kind, member CAS (first `unstarted -> starting` true, replay false, illegal `ready -> dormant` false, stale `from` false, unknown member false), member patch merge/clear/stamp, member listing order, task CAS + `updated_at` refresh + terminal status, `update_task` revision bump and change notes (content / title+content / identical content / note-only / explicit revision / `clear_assignee`), `list_tasks` filters (status, assignee, parent, limit) and ordering, dependency idempotence + sorting + team scoping + self-dependency rejection, per-round vote isolation with reviewer replacement, mailbox ordering, `mark_read` semantics, watermark coverage/regression, `message_count`, keyed message replacement, keyed member replacement, `open_store` selection and unsupported-backend messages, per-call store independence.

Definition of done for this node: `rustfmt --check` clean on the four files, `cargo test -p openjiuwen-agent-teams --lib` and `--no-default-features --lib` green, no clippy diagnostic pointing at `src/storage/`.
Crate-wide `cargo clippy -- -D warnings` still reports `missing_docs` on the ten still-empty sibling modules and two crate-root doctests (`lib.rs`, `constants.rs`) that reference root re-exports not yet declared; both belong to the crate root node.

## Notes for Agents

- `TeamStore` is the only persistence seam. Do not add a second one, and do not read the clock in a caller: pass the stamp, or let the store stamp it.
- The in-process backend is not a SQL port: there is no schema DDL, no migration and no transaction API. Multi-row atomicity comes from the single `RwLock` — if you need several writes to be atomic, add one store method instead of composing calls.
- `transition_*` returning `false` is expected under concurrency (the member/task board sweeps and the leader both write). Treat it as "re-read the row", never as a hard failure.
- Dependency edges are stored independently of task rows, so a `TaskGraphSpec` may be materialized in any order (`depends_on` before the referenced task exists). `dependents()` is an O(edges) scan — fine at team scale.
- `unread_messages` deliberately excludes a member's own broadcasts (Python `get_broadcast_messages`). A member that must see its own announcement reads it back through `mark_read`-free paths such as the team transcript, not through the mailbox.
- Broadcast read state is a single per-member watermark, never a row flag. `mark_read` on a broadcast id is a silent no-op; use `set_read_watermark` (Python's `mark_message_read` does both in one call — this port splits them).
- `MemberPatch::apply` is the single place patch precedence lives (`clear_model`/`clear_worktree` beat the setters); `update_member` only adds the stamp.
- `update_task` distinguishes an explicit `revision` (author-controlled, no change note) from an implicit bump (title/content edit, note appended). Keep that split: the task board renders `change_notes` as history.
- Deferred backends stay behind `open_store`: `sqlite`/`postgresql`/`mysql` return `Error::Unsupported` with the Python capability name, and the trait is the only thing a new backend has to implement.
