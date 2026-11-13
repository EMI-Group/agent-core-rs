//! The SQL statements and column lists the SQLite backend runs.
//!
//! Every read selects one of the `*_COLUMNS` lists and every write is one named statement, so the
//! column order lives in one place: a new column is added to the list, the codec in `rows` and the
//! statement here, and nothing else has to change.

/// Columns every team-row read selects; [`rows::decode_team`] reads them by name.
pub(super) const TEAM_COLUMNS: &str = "\
team_name, display_name, leader_member_name, description, prompt, dispatch_mode, \
enable_task_verification, created, updated_at";

/// Columns every roster read selects.
pub(super) const MEMBER_COLUMNS: &str = "\
team_name, member_name, display_name, description, role, status, execution, mode, prompt, \
model_ref, worktree, updated_at";

/// Columns every task read selects.
pub(super) const TASK_COLUMNS: &str = "\
team_name, task_id, title, content, status, assignee, reviewers, review_round, \
max_review_rounds, summary, note, parent_task_id, parent_revision, created_by, revision, \
observed_revision, change_notes, updated_at";

/// Columns every review-vote read selects.
pub(super) const VOTE_COLUMNS: &str = "team_name, task_id, review_round, reviewer, verdict";

/// Columns every mailbox read selects.
pub(super) const MESSAGE_COLUMNS: &str = "\
team_name, message_id, from_member_name, to_member_name, content, timestamp, broadcast, \
protocol, is_read";

/// The team-row upsert: every column is replaced, nothing is merged.
pub(super) const UPSERT_TEAM: &str = "\
INSERT INTO team_info (
    team_name, display_name, leader_member_name, description, prompt, dispatch_mode,
    enable_task_verification, created, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(team_name) DO UPDATE SET
    display_name = excluded.display_name,
    leader_member_name = excluded.leader_member_name,
    description = excluded.description,
    prompt = excluded.prompt,
    dispatch_mode = excluded.dispatch_mode,
    enable_task_verification = excluded.enable_task_verification,
    created = excluded.created,
    updated_at = excluded.updated_at";

/// The roster-row upsert.
pub(super) const UPSERT_MEMBER: &str = "\
INSERT INTO team_member (
    team_name, member_name, display_name, description, role, status, execution, mode, prompt,
    model_ref, worktree, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(team_name, member_name) DO UPDATE SET
    display_name = excluded.display_name,
    description = excluded.description,
    role = excluded.role,
    status = excluded.status,
    execution = excluded.execution,
    mode = excluded.mode,
    prompt = excluded.prompt,
    model_ref = excluded.model_ref,
    worktree = excluded.worktree,
    updated_at = excluded.updated_at";

/// Rewrites every field of one roster row after a patch was applied in Rust.
pub(super) const UPDATE_MEMBER_ROW: &str = "\
UPDATE team_member SET
    display_name = ?, description = ?, role = ?, status = ?, execution = ?, mode = ?, prompt = ?,
    model_ref = ?, worktree = ?, updated_at = ?
WHERE team_name = ? AND member_name = ?";

/// The task-row upsert.
pub(super) const UPSERT_TASK: &str = "\
INSERT INTO team_task (
    team_name, task_id, title, content, status, assignee, reviewers, review_round,
    max_review_rounds, summary, note, parent_task_id, parent_revision, created_by, revision,
    observed_revision, change_notes, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(team_name, task_id) DO UPDATE SET
    title = excluded.title,
    content = excluded.content,
    status = excluded.status,
    assignee = excluded.assignee,
    reviewers = excluded.reviewers,
    review_round = excluded.review_round,
    max_review_rounds = excluded.max_review_rounds,
    summary = excluded.summary,
    note = excluded.note,
    parent_task_id = excluded.parent_task_id,
    parent_revision = excluded.parent_revision,
    created_by = excluded.created_by,
    revision = excluded.revision,
    observed_revision = excluded.observed_revision,
    change_notes = excluded.change_notes,
    updated_at = excluded.updated_at";

/// Rewrites every field of one task row after a patch was applied in Rust.
pub(super) const UPDATE_TASK_ROW: &str = "\
UPDATE team_task SET
    title = ?, content = ?, status = ?, assignee = ?, reviewers = ?, review_round = ?,
    max_review_rounds = ?, summary = ?, note = ?, parent_task_id = ?, parent_revision = ?,
    created_by = ?, revision = ?, observed_revision = ?, change_notes = ?, updated_at = ?
WHERE team_name = ? AND task_id = ?";

/// The review-vote upsert: a reviewer voting again in the same round replaces the verdict.
pub(super) const UPSERT_VOTE: &str = "\
INSERT INTO team_review_vote (team_name, task_id, review_round, reviewer, verdict)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(team_name, task_id, review_round, reviewer) DO UPDATE SET verdict = excluded.verdict";

/// The mailbox upsert.
pub(super) const UPSERT_MESSAGE: &str = "\
INSERT INTO team_message (
    team_name, message_id, from_member_name, to_member_name, content, timestamp, broadcast,
    protocol, is_read
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(team_name, message_id) DO UPDATE SET
    from_member_name = excluded.from_member_name,
    to_member_name = excluded.to_member_name,
    content = excluded.content,
    timestamp = excluded.timestamp,
    broadcast = excluded.broadcast,
    protocol = excluded.protocol,
    is_read = excluded.is_read";

/// Flags one direct message as read; a broadcast/foreign/unknown id matches no row.
pub(super) const MARK_MESSAGE_READ: &str = "\
UPDATE team_message SET is_read = 1
WHERE team_name = ? AND message_id = ? AND broadcast = 0 AND to_member_name = ?";

/// Advances a member's broadcast watermark without ever moving it backwards.
pub(super) const ADVANCE_WATERMARK: &str = "\
INSERT INTO team_read_watermark (team_name, member_name, watermark) VALUES (?, ?, ?)
ON CONFLICT(team_name, member_name) DO UPDATE SET
    watermark = MAX(team_read_watermark.watermark, excluded.watermark)";

/// Predicate of "still waiting in `member`'s mailbox"; it references the member twice and the
/// watermark twice, in that order.
pub(super) const UNREAD_PREDICATE: &str = "\
(broadcast = 0 AND is_read = 0 AND to_member_name = ?)
OR (broadcast = 1 AND from_member_name <> ? AND (? IS NULL OR timestamp > ?))";
