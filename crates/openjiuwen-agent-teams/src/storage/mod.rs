//! The persistence seam: [`TeamStore`] and the in-memory implementation every other module starts
//! against.
//!
//! Python parity source: `openjiuwen/agent_teams/tools/database/` (`SQLAlchemy` DAOs for the team,
//! member, task and message tables) plus the `team_backend` persistence calls that drive them.
//! This crate defines one object-safe trait over those tables instead of porting the SQL schema:
//! the runtime, the task board, the mailbox and the CLI/MCP surface all read and write through
//! [`TeamStore`], so a real backend can be added behind the same seam later.
//!
//! Module map:
//!
//! | File | Contents |
//! | --- | --- |
//! | [`records`] | [`TeamInfoRecord`], [`MemberRecord`], [`MemberPatch`], [`MessageRecord`] |
//! | [`memory`] | [`InMemoryTeamStore`], the deterministic in-process backend |
//! | `stamps` | row stamping and key validation shared by both backends |
//! | `sqlite` | `SqliteTeamStore`, the persistent backend (feature `sqlite`) |
//! | `store_tests` | the crate-internal store suite (`#[cfg(test)]`) |
//!
//! The store is a *state* seam, not a policy seam: it validates keys and status transitions only.
//! Dependency unblocking, review gating, dispatch and message routing belong to
//! [`crate::taskboard`], [`crate::member`] and [`crate::runtime`].
//!
//! ```
//! use std::sync::Arc;
//! use openjiuwen_agent_teams::schema::{
//!     ExecutionStatus, MemberMode, MemberStatus, StorageSpec, TeamRole,
//! };
//! use openjiuwen_agent_teams::storage::{open_store, MemberRecord, TeamStore};
//!
//! # async fn demo() -> openjiuwen_common::error::Result<()> {
//! let store = open_store(&StorageSpec::in_memory())?;
//! assert_eq!(store.name(), "memory");
//! store
//!     .insert_member(MemberRecord::new("alpha", "dev-1", TeamRole::Teammate))
//!     .await?;
//! assert!(store
//!     .transition_member("alpha", "dev-1", MemberStatus::Unstarted, MemberStatus::Starting)
//!     .await?);
//! let member = store.member("alpha", "dev-1").await?.expect("member row");
//! assert_eq!(member.execution, ExecutionStatus::Idle);
//! assert_eq!(member.mode, MemberMode::BuildMode);
//! # let _: Arc<dyn TeamStore> = store;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

use crate::schema::{
    MemberStatus, ReviewVote, StorageSpec, TaskFilter, TaskPatch, TaskRecord, TaskStatus,
};

pub mod memory;
pub mod records;
mod stamps;

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(test)]
mod store_tests;

pub use memory::InMemoryTeamStore;
pub use records::{
    MemberPatch, MemberRecord, MessageRecord, TeamInfoRecord, PLAIN_MESSAGE_PROTOCOL,
};

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteTeamStore;

/// Storage kind of the in-process backend; the `StorageSpec` default and `open_store` fallback.
pub const MEMORY_STORAGE_KIND: &str = "memory";

/// The persisted state of one team: team row, roster, task board, dependency edges, review votes
/// and mailbox.
///
/// All methods are idempotent where a caller may reasonably retry: `insert_*` replaces a row with
/// the same key, `add_dependency` ignores an edge it already has, `mark_read` re-flags a row that is
/// already read, and `set_read_watermark` never moves the cursor backwards.
///
/// Two write paths differ deliberately:
///
/// * [`TeamStore::transition_member`] / [`TeamStore::transition_task`] are compare-and-set: they
///   succeed only when the stored status equals `from` *and* the transition is listed in the
///   [`crate::schema`] transition table, and they refresh `updated_at`.
/// * [`TeamStore::update_member`] / [`TeamStore::update_task`] apply a patch verbatim and do not
///   validate status legality; the caller reaches them only after a lifecycle decision.
///
/// Any `*_at` stamp the caller supplies is preserved; a record written with `0` gets a store stamp
/// (`max(now, previous + 1)`), so timestamps stay monotonic even when two writes land in the same
/// millisecond.
#[async_trait]
pub trait TeamStore: Send + Sync {
    /// Returns the backend kind, matching the `StorageSpec` value (`"memory"` for the in-process
    /// store).
    fn name(&self) -> &'static str;

    /// Inserts or replaces the team row.
    ///
    /// The stored [`TeamInfoRecord::created`] is kept from the existing row when the incoming
    /// record leaves it at `0`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when
    /// `team_name` is empty, and whatever the backend fails with on an I/O error.
    async fn upsert_team(&self, record: TeamInfoRecord) -> Result<()>;

    /// Returns the team row, or `None` when the team was never persisted.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn team(&self, team_name: &str) -> Result<Option<TeamInfoRecord>>;

    /// Inserts or replaces a roster row, keyed by `(team_name, member_name)`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when a key
    /// is empty.
    async fn insert_member(&self, record: MemberRecord) -> Result<()>;

    /// Returns one roster row, or `None` when the member is unknown.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn member(&self, team_name: &str, member_name: &str) -> Result<Option<MemberRecord>>;

    /// Returns the roster of `team_name` sorted by `member_name`.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn list_members(&self, team_name: &str) -> Result<Vec<MemberRecord>>;

    /// Applies `patch` to one roster row.
    ///
    /// Returns `false` when the member is unknown; an empty patch on a known member returns `true`
    /// and writes nothing.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn update_member(
        &self,
        team_name: &str,
        member_name: &str,
        patch: &MemberPatch,
    ) -> Result<bool>;

    /// Moves a member from `from` to `to` when the stored status still equals `from` and the
    /// transition is legal per [`crate::schema::is_valid_member_transition`].
    ///
    /// Returns `false` for an unknown member, a stale `from`, or an illegal edge; a `false` result is
    /// the normal "somebody else got there first" answer, not an error.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn transition_member(
        &self,
        team_name: &str,
        member_name: &str,
        from: MemberStatus,
        to: MemberStatus,
    ) -> Result<bool>;

    /// Inserts or replaces a task row, keyed by `(team_name, task_id)`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when a key
    /// is empty.
    async fn insert_task(&self, record: TaskRecord) -> Result<()>;

    /// Returns one task row, or `None` when the task is unknown.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn task(&self, team_name: &str, task_id: &str) -> Result<Option<TaskRecord>>;

    /// Returns the tasks of `team_name` matching `filter`, ordered by `updated_at` descending then
    /// `task_id` ascending, truncated to [`TaskFilter::limit`].
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn list_tasks(&self, team_name: &str, filter: &TaskFilter) -> Result<Vec<TaskRecord>>;

    /// Applies `patch` to one task row.
    ///
    /// A title or content change bumps [`TaskRecord::revision`] (unless the patch sets it
    /// explicitly) and appends the rendered [`crate::schema::TaskRevision`] note to
    /// [`TaskRecord::change_notes`]. Returns `false` when the task is unknown.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn update_task(&self, team_name: &str, task_id: &str, patch: &TaskPatch) -> Result<bool>;

    /// Moves a task from `from` to `to` when the stored status still equals `from` and the edge is
    /// listed in [`crate::schema::TASK_TRANSITIONS`].
    ///
    /// The transition refreshes [`TaskRecord::updated_at`]. Unblocking dependents is the task
    /// board's job, so a `blocked` task's dependents are untouched here.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn transition_task(
        &self,
        team_name: &str,
        task_id: &str,
        from: TaskStatus,
        to: TaskStatus,
    ) -> Result<bool>;

    /// Records that `task_id` is blocked by `depends_on`.
    ///
    /// Idempotent. Neither task has to exist yet — the task board may create the edge while
    /// materializing a graph.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) for an
    /// empty id or a self-dependency, and whatever the backend fails with.
    async fn add_dependency(&self, team_name: &str, task_id: &str, depends_on: &str) -> Result<()>;

    /// Returns the ids `task_id` directly depends on, sorted.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn dependencies(&self, team_name: &str, task_id: &str) -> Result<Vec<String>>;

    /// Returns the ids that directly depend on `task_id`, sorted.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn dependents(&self, team_name: &str, task_id: &str) -> Result<Vec<String>>;

    /// Records a review verdict; a reviewer voting again in the same round replaces their vote.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn insert_vote(&self, record: ReviewVote) -> Result<()>;

    /// Returns the votes cast on `task_id` in `round`, sorted by reviewer.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn votes(&self, team_name: &str, task_id: &str, round: u32) -> Result<Vec<ReviewVote>>;

    /// Inserts or replaces a mailbox row, keyed by `(team_name, message_id)`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when a key
    /// is empty.
    async fn insert_message(&self, record: MessageRecord) -> Result<()>;

    /// Returns the messages `member_name` has not read yet, ordered by `timestamp` then
    /// `message_id`.
    ///
    /// Direct messages count while `is_read` is `false` and they address the member. Broadcasts
    /// count while their `timestamp` exceeds the member's watermark (and are never counted for
    /// their own sender, matching Python's `get_broadcast_messages`).
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn unread_messages(
        &self,
        team_name: &str,
        member_name: &str,
    ) -> Result<Vec<MessageRecord>>;

    /// Flags the given direct messages as read by `member_name`.
    ///
    /// Ids that are unknown, already read, addressed to somebody else, or broadcast rows are
    /// skipped silently: broadcast read state is the per-member watermark, advanced by
    /// [`TeamStore::set_read_watermark`] and never by a row flag.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn mark_read(
        &self,
        team_name: &str,
        member_name: &str,
        message_ids: &[String],
    ) -> Result<()>;

    /// Returns the member's broadcast watermark, or `None` when the member never read a broadcast.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn read_watermark(&self, team_name: &str, member_name: &str) -> Result<Option<u64>>;

    /// Advances the member's broadcast watermark; an older timestamp never moves it backwards.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn set_read_watermark(
        &self,
        team_name: &str,
        member_name: &str,
        timestamp: u64,
    ) -> Result<()>;

    /// Returns how many messages are waiting for `member_name`, i.e. the size of
    /// [`TeamStore::unread_messages`] without materializing the rows.
    ///
    /// The runtime uses it as the mailbox-depth signal for its idle sweep.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend fails with.
    async fn message_count(&self, team_name: &str, member_name: &str) -> Result<usize>;
}

/// Opens the store selected by `spec`.
///
/// `"memory"` (and an empty kind) yields a fresh [`InMemoryTeamStore`]: the store is deliberately
/// *not* a process singleton, so each call is an independent, empty backend — build one per runtime
/// and share the `Arc`.
///
/// `"sqlite"` yields a `SqliteTeamStore` (feature `sqlite`), configured from `spec.params`:
///
/// * `path` (preferred), `url` or `database`, each a string — `data/team.db`,
///   `sqlite://data/team.db`, `sqlite:///abs/team.db` or `:memory:`. Without one of them the call
///   fails with [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config).
/// * The database is connected lazily, on the store's first call, because this function is
///   synchronous; use `SqliteTeamStore::open` when construction errors must surface immediately.
///
/// Without the feature, `"sqlite"` reports that the feature is missing instead of degrading.
///
/// # Errors
///
/// Returns [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported) for the
/// deferred `postgresql` / `mysql` backends, for `sqlite` when the `sqlite` feature is off, and for an
/// unknown kind; returns whatever the SQLite backend reports for an unusable parameter set.
pub fn open_store(spec: &StorageSpec) -> Result<Arc<dyn TeamStore>> {
    let kind = spec.kind.trim();
    if kind.is_empty() || kind == MEMORY_STORAGE_KIND {
        return Ok(Arc::new(InMemoryTeamStore::new()));
    }
    match kind {
        "sqlite" => open_sqlite(spec),
        "postgresql" | "mysql" => Err(Error::unsupported(format!(
            "storage backend '{kind}' is not implemented; the {kind} driver is not available offline"
        ))),
        other => Err(Error::unsupported(format!(
            "unknown storage backend '{other}'; use '{MEMORY_STORAGE_KIND}'"
        ))),
    }
}

/// Builds the lazily connected SQLite backend selected by `spec`.
#[cfg(feature = "sqlite")]
fn open_sqlite(spec: &StorageSpec) -> Result<Arc<dyn TeamStore>> {
    Ok(Arc::new(SqliteTeamStore::lazy_from_params(&spec.params)?))
}

/// Reports the missing `sqlite` feature of a `sqlite` storage spec.
#[cfg(not(feature = "sqlite"))]
fn open_sqlite(_spec: &StorageSpec) -> Result<Arc<dyn TeamStore>> {
    Err(Error::unsupported(
        "storage backend 'sqlite' requires the 'sqlite' feature; \
         enable 'openjiuwen-agent-teams/sqlite'",
    ))
}
