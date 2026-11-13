//! Schema DDL and the migration step of the SQLite team store.
//!
//! Every table is created with `IF NOT EXISTS` and the whole DDL is applied on each connect, so a v1
//! database written by an older build keeps working. `PRAGMA user_version` records which migration
//! version is on disk: `0` (a fresh, or a pre-versioning, database) is stamped with the current
//! version by the DDL above, the current version is a no-op, and a newer version is rejected instead
//! of being written behind a newer build's back. A future version appends its `ALTER TABLE` steps
//! keyed on the version read here, before the stamp.
//!
//! No foreign keys are declared: [`TeamStore`](crate::storage::TeamStore) deliberately allows a
//! member, task, dependency or message row to exist before its team row (the task board materializes
//! a graph in any order and the leader may address a member it has not spawned yet), so referential
//! integrity is the caller's job. `foreign_keys` is still enabled on every connection, so a
//! constraint added by a future migration is enforced.

use sqlx::SqlitePool;

use openjiuwen_common::error::{Error, Result};

use super::rows::sqlite_error;

/// Schema version this build writes and understands.
pub(super) const SCHEMA_VERSION: i64 = 1;

/// The team row: identity, the leader-facing description/prompt and the dispatch knobs.
const TEAM_INFO: &str = "\
CREATE TABLE IF NOT EXISTS team_info (
    team_name TEXT PRIMARY KEY NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    leader_member_name TEXT NOT NULL DEFAULT 'team_leader',
    description TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    dispatch_mode TEXT NOT NULL DEFAULT 'autonomous',
    enable_task_verification INTEGER NOT NULL DEFAULT 0,
    created INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL DEFAULT 0
)";

/// The roster: lifecycle status, execution status and the allocated model/workspace per member.
const TEAM_MEMBER: &str = "\
CREATE TABLE IF NOT EXISTS team_member (
    team_name TEXT NOT NULL,
    member_name TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'teammate',
    status TEXT NOT NULL DEFAULT 'unstarted',
    execution TEXT NOT NULL DEFAULT 'idle',
    mode TEXT NOT NULL DEFAULT 'build_mode',
    prompt TEXT NOT NULL DEFAULT '',
    model_ref TEXT NULL,
    worktree TEXT NULL,
    updated_at INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (team_name, member_name)
)";

/// The task board row; `reviewers` and `change_notes` are JSON arrays.
const TEAM_TASK: &str = "\
CREATE TABLE IF NOT EXISTS team_task (
    team_name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    assignee TEXT NULL,
    reviewers TEXT NOT NULL DEFAULT '[]',
    review_round INTEGER NOT NULL DEFAULT 0,
    max_review_rounds INTEGER NOT NULL DEFAULT 3,
    summary TEXT NULL,
    note TEXT NULL,
    parent_task_id TEXT NULL,
    parent_revision INTEGER NULL,
    created_by TEXT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    observed_revision INTEGER NOT NULL DEFAULT 0,
    change_notes TEXT NOT NULL DEFAULT '[]',
    updated_at INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (team_name, task_id)
)";

/// Dependency edges, stored independently of the task rows they reference.
const TEAM_TASK_DEPENDENCY: &str = "\
CREATE TABLE IF NOT EXISTS team_task_dependency (
    team_name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    depends_on TEXT NOT NULL,
    PRIMARY KEY (team_name, task_id, depends_on)
)";

/// The review gate: one verdict per reviewer per round.
const TEAM_REVIEW_VOTE: &str = "\
CREATE TABLE IF NOT EXISTS team_review_vote (
    team_name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    review_round INTEGER NOT NULL,
    reviewer TEXT NOT NULL,
    verdict TEXT NOT NULL DEFAULT 'pass',
    PRIMARY KEY (team_name, task_id, review_round, reviewer)
)";

/// The mailbox: direct rows carry `is_read`, broadcast rows carry the per-member watermark instead.
const TEAM_MESSAGE: &str = "\
CREATE TABLE IF NOT EXISTS team_message (
    team_name TEXT NOT NULL,
    message_id TEXT NOT NULL,
    from_member_name TEXT NOT NULL DEFAULT '',
    to_member_name TEXT NULL,
    content TEXT NOT NULL DEFAULT '',
    timestamp INTEGER NOT NULL DEFAULT 0,
    broadcast INTEGER NOT NULL DEFAULT 0,
    protocol TEXT NOT NULL DEFAULT 'plain',
    is_read INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (team_name, message_id)
)";

/// One broadcast read cursor per member.
const TEAM_READ_WATERMARK: &str = "\
CREATE TABLE IF NOT EXISTS team_read_watermark (
    team_name TEXT NOT NULL,
    member_name TEXT NOT NULL,
    watermark INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (team_name, member_name)
)";

/// Index backing the task-board status filter and the leader's claimable scan.
const TASK_STATUS_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_team_task_status
    ON team_task (team_name, status)";

/// Index backing the "my tasks" view and the stale-claim sweep.
const TASK_ASSIGNEE_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_team_task_assignee
    ON team_task (team_name, assignee)";

/// Index backing the subtask listing of one parent.
const TASK_PARENT_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_team_task_parent
    ON team_task (team_name, parent_task_id)";

/// Index backing [`TeamStore::list_tasks`](crate::storage::TeamStore::list_tasks) ordering.
const TASK_UPDATED_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_team_task_updated
    ON team_task (team_name, updated_at DESC, task_id)";

/// Index backing `dependents` (the reverse edge scan).
const TASK_DEPENDENCY_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_team_task_dependency_depends_on
    ON team_task_dependency (team_name, depends_on)";

/// Index backing the unread direct-message scan.
const MESSAGE_RECIPIENT_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_team_message_recipient
    ON team_message (team_name, to_member_name, is_read)";

/// Index backing the unread broadcast scan ordered by the read cursor.
const MESSAGE_BROADCAST_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_team_message_broadcast
    ON team_message (team_name, broadcast, timestamp)";

/// Every statement applied on connect, in one idempotent batch.
const DDL: [&str; 14] = [
    TEAM_INFO,
    TEAM_MEMBER,
    TEAM_TASK,
    TEAM_TASK_DEPENDENCY,
    TEAM_REVIEW_VOTE,
    TEAM_MESSAGE,
    TEAM_READ_WATERMARK,
    TASK_STATUS_INDEX,
    TASK_ASSIGNEE_INDEX,
    TASK_PARENT_INDEX,
    TASK_UPDATED_INDEX,
    TASK_DEPENDENCY_INDEX,
    MESSAGE_RECIPIENT_INDEX,
    MESSAGE_BROADCAST_INDEX,
];

/// Applies the DDL and the migration step to a freshly opened pool.
///
/// # Errors
///
/// Returns whatever the driver reports, and
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported) when the database
/// carries a schema version newer than this build understands.
pub(super) async fn ensure_schema(pool: &SqlitePool) -> Result<()> {
    let mut tx = pool.begin().await.map_err(sqlite_error)?;
    for statement in DDL {
        sqlx::query(statement)
            .execute(&mut *tx)
            .await
            .map_err(sqlite_error)?;
    }
    let found: i64 = sqlx::query_scalar::<_, i64>("PRAGMA user_version")
        .fetch_one(&mut *tx)
        .await
        .map_err(sqlite_error)?;
    if found > SCHEMA_VERSION {
        return Err(Error::unsupported(format!(
            "sqlite team store schema version {found} is newer than this build (supports {SCHEMA_VERSION})"
        )));
    }
    if found < SCHEMA_VERSION {
        // A `0` database is v1 after the DDL above, so recording the version is the whole 0 → 1
        // step; later versions insert their `ALTER TABLE` statements before this stamp.
        let statement = format!("PRAGMA user_version = {SCHEMA_VERSION}");
        sqlx::query(&statement)
            .execute(&mut *tx)
            .await
            .map_err(sqlite_error)?;
    }
    tx.commit().await.map_err(sqlite_error)
}
