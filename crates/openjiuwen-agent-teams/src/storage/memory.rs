//! The deterministic in-process [`TeamStore`]: [`InMemoryTeamStore`].
//!
//! This is the backend `StorageSpec::in_memory()` (the default) selects, and the one every test and
//! every single-process runtime uses. It holds the whole team state in `BTreeMap`s behind one
//! [`parking_lot::RwLock`]; there is no I/O, no clock-driven scheduling and no background task, so
//! two identical call sequences always produce the same state.
//!
//! Python parity source: `openjiuwen/agent_teams/tools/database/` (the SQLite `TeamDatabase`) and
//! the `_Bus` mailbox it backs. The Rust backend is deliberately *not* a SQL port: it is a keyed
//! row store with the same observable answers (row shapes, ordering, compare-and-set transitions,
//! broadcast watermark).
//!
//! Every method takes the write lock only for the duration of one map operation and never awaits
//! while holding it.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use parking_lot::RwLock;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::unix_millis;

use crate::schema::{
    is_valid_member_transition, MemberStatus, ReviewVote, TaskFilter, TaskPatch, TaskRecord,
    TaskRevision, TaskStatus,
};

use super::records::{MemberPatch, MemberRecord, MessageRecord, TeamInfoRecord};
use super::{TeamStore, MEMORY_STORAGE_KIND};

/// Key used by every row map and by the watermark map: `(team_name, row_id)`.
type RowKey = (String, String);

/// One monotonically increasing write stamp.
///
/// A real backend stamps `now()`; the in-process backend must also keep ordering stable when two
/// writes land inside the same millisecond, so it returns `max(now, previous + 1)`.
fn refresh(previous: u64) -> u64 {
    let now = u64::try_from(unix_millis()).unwrap_or(0);
    now.max(previous.saturating_add(1))
}

/// Returns the caller-supplied stamp when it is set, else a fresh write stamp after `previous`.
fn stamp(supplied: u64, previous: u64) -> u64 {
    if supplied == 0 {
        refresh(previous)
    } else {
        supplied
    }
}

/// Rejects an empty row key.
fn require_key(value: &str, label: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::validation(format!("{label} must not be empty")));
    }
    Ok(())
}

/// Returns `true` when `message` is still waiting in `member`'s mailbox.
///
/// Direct rows rely on the row flag; broadcast rows rely on the member watermark, and a broadcast is
/// never unread for its own sender (Python `get_broadcast_messages` behaves the same).
fn is_unread_for(message: &MessageRecord, member: &str, watermark: Option<u64>) -> bool {
    if message.broadcast {
        if message.from_member_name == member {
            return false;
        }
        return watermark.is_none_or(|read_at| message.timestamp > read_at);
    }
    !message.is_read && message.to_member_name.as_deref() == Some(member)
}

/// The persisted state of one or more teams, keyed exactly as a SQL backend would key it.
#[derive(Debug, Default)]
struct Inner {
    teams: BTreeMap<String, TeamInfoRecord>,
    members: BTreeMap<RowKey, MemberRecord>,
    tasks: BTreeMap<RowKey, TaskRecord>,
    /// `(team, task)` → set of task ids it depends on.
    depends_on: BTreeMap<RowKey, BTreeSet<String>>,
    votes: Vec<ReviewVote>,
    messages: Vec<MessageRecord>,
    watermarks: BTreeMap<RowKey, u64>,
}

/// The in-process [`TeamStore`].
///
/// Cheap to construct, `Send + Sync`, and shareable as `Arc<dyn TeamStore>`:
///
/// ```
/// use std::sync::Arc;
/// use openjiuwen_agent_teams::storage::{InMemoryTeamStore, TeamStore};
///
/// let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
/// assert_eq!(store.name(), "memory");
/// ```
#[derive(Debug, Default)]
pub struct InMemoryTeamStore {
    inner: RwLock<Inner>,
}

impl InMemoryTeamStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl TeamStore for InMemoryTeamStore {
    fn name(&self) -> &'static str {
        MEMORY_STORAGE_KIND
    }

    async fn upsert_team(&self, record: TeamInfoRecord) -> Result<()> {
        require_key(&record.team_name, "team_name")?;
        let mut inner = self.inner.write();
        let previous = inner.teams.get(&record.team_name).cloned();
        let mut record = record;
        if let Some(previous) = previous {
            if record.created == 0 {
                record.created = previous.created;
            }
            record.updated_at = stamp(record.updated_at, previous.updated_at);
        } else {
            record.created = stamp(record.created, 0);
            record.updated_at = stamp(record.updated_at, record.created);
        }
        inner.teams.insert(record.team_name.clone(), record);
        Ok(())
    }

    async fn team(&self, team_name: &str) -> Result<Option<TeamInfoRecord>> {
        Ok(self.inner.read().teams.get(team_name).cloned())
    }

    async fn insert_member(&self, record: MemberRecord) -> Result<()> {
        require_key(&record.team_name, "team_name")?;
        require_key(&record.member_name, "member_name")?;
        let mut inner = self.inner.write();
        let key = (record.team_name.clone(), record.member_name.clone());
        let previous = inner.members.get(&key).map(|row| row.updated_at);
        let mut record = record;
        record.updated_at = stamp(record.updated_at, previous.unwrap_or(0));
        inner.members.insert(key, record);
        Ok(())
    }

    async fn member(&self, team_name: &str, member_name: &str) -> Result<Option<MemberRecord>> {
        let key = (team_name.to_owned(), member_name.to_owned());
        Ok(self.inner.read().members.get(&key).cloned())
    }

    async fn list_members(&self, team_name: &str) -> Result<Vec<MemberRecord>> {
        let inner = self.inner.read();
        let mut rows: Vec<MemberRecord> = inner
            .members
            .values()
            .filter(|row| row.team_name == team_name)
            .cloned()
            .collect();
        rows.sort_by(|left, right| left.member_name.cmp(&right.member_name));
        Ok(rows)
    }

    async fn update_member(
        &self,
        team_name: &str,
        member_name: &str,
        patch: &MemberPatch,
    ) -> Result<bool> {
        let key = (team_name.to_owned(), member_name.to_owned());
        let mut inner = self.inner.write();
        let Some(mut record) = inner.members.get(&key).cloned() else {
            return Ok(false);
        };
        if !patch.apply(&mut record) {
            return Ok(true);
        }
        record.updated_at = refresh(record.updated_at);
        inner.members.insert(key, record);
        Ok(true)
    }

    async fn transition_member(
        &self,
        team_name: &str,
        member_name: &str,
        from: MemberStatus,
        to: MemberStatus,
    ) -> Result<bool> {
        let key = (team_name.to_owned(), member_name.to_owned());
        let mut inner = self.inner.write();
        let Some(mut record) = inner.members.get(&key).cloned() else {
            return Ok(false);
        };
        if record.status != from || !is_valid_member_transition(from, to) {
            return Ok(false);
        }
        record.status = to;
        record.updated_at = refresh(record.updated_at);
        inner.members.insert(key, record);
        Ok(true)
    }

    async fn insert_task(&self, record: TaskRecord) -> Result<()> {
        require_key(&record.team_name, "team_name")?;
        require_key(&record.task_id, "task_id")?;
        let mut inner = self.inner.write();
        let key = (record.team_name.clone(), record.task_id.clone());
        let previous = inner.tasks.get(&key).map(|row| row.updated_at);
        let mut record = record;
        record.updated_at = stamp(record.updated_at, previous.unwrap_or(0));
        inner.tasks.insert(key, record);
        Ok(())
    }

    async fn task(&self, team_name: &str, task_id: &str) -> Result<Option<TaskRecord>> {
        let key = (team_name.to_owned(), task_id.to_owned());
        Ok(self.inner.read().tasks.get(&key).cloned())
    }

    async fn list_tasks(&self, team_name: &str, filter: &TaskFilter) -> Result<Vec<TaskRecord>> {
        let inner = self.inner.read();
        let mut rows: Vec<TaskRecord> = inner
            .tasks
            .values()
            .filter(|row| row.team_name == team_name)
            .filter(|row| filter.status.is_none_or(|status| row.status == status))
            .filter(|row| {
                filter
                    .assignee
                    .as_deref()
                    .is_none_or(|assignee| row.assignee.as_deref() == Some(assignee))
            })
            .filter(|row| {
                filter
                    .parent
                    .as_deref()
                    .is_none_or(|parent| row.parent_task_id.as_deref() == Some(parent))
            })
            .cloned()
            .collect();
        rows.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| left.task_id.cmp(&right.task_id))
        });
        if let Some(limit) = filter.limit {
            rows.truncate(limit);
        }
        Ok(rows)
    }

    async fn update_task(&self, team_name: &str, task_id: &str, patch: &TaskPatch) -> Result<bool> {
        let key = (team_name.to_owned(), task_id.to_owned());
        let mut inner = self.inner.write();
        let Some(mut record) = inner.tasks.get(&key).cloned() else {
            return Ok(false);
        };
        let before = record.clone();
        let mut edited: Vec<&'static str> = Vec::new();
        if let Some(title) = &patch.title {
            if record.title != *title {
                record.title.clone_from(title);
                edited.push("title");
            }
        }
        if let Some(content) = &patch.content {
            if record.content != *content {
                record.content.clone_from(content);
                edited.push("content");
            }
        }
        if let Some(status) = patch.status {
            record.status = status;
        }
        if patch.clear_assignee {
            record.assignee = None;
        } else if let Some(assignee) = &patch.assignee {
            record.assignee = Some(assignee.clone());
        }
        if let Some(reviewers) = &patch.reviewers {
            record.reviewers.clone_from(reviewers);
        }
        if let Some(round) = patch.review_round {
            record.review_round = round;
        }
        if let Some(rounds) = patch.max_review_rounds {
            record.max_review_rounds = rounds;
        }
        if let Some(summary) = &patch.summary {
            record.summary = Some(summary.clone());
        }
        if let Some(note) = &patch.note {
            record.note = Some(note.clone());
        }
        if let Some(parent) = &patch.parent_task_id {
            record.parent_task_id = Some(parent.clone());
        }
        if let Some(observed) = patch.observed_revision {
            record.observed_revision = observed;
        }
        let bumped = patch.revision.is_none() && !edited.is_empty();
        if let Some(revision) = patch.revision {
            record.revision = revision;
        } else if bumped {
            record.revision = record.revision.saturating_add(1);
            let note = match edited.as_slice() {
                ["title"] => "title updated",
                ["content"] => "content updated",
                _ => "title and content updated",
            };
            record.push_change_note(
                TaskRevision {
                    revision: record.revision,
                    note: note.to_owned(),
                    by: None,
                }
                .to_note(),
            );
        }
        if let Some(updated_at) = patch.updated_at {
            record.updated_at = updated_at;
        } else if record != before {
            record.updated_at = refresh(before.updated_at);
        }
        inner.tasks.insert(key, record);
        Ok(true)
    }

    async fn transition_task(
        &self,
        team_name: &str,
        task_id: &str,
        from: TaskStatus,
        to: TaskStatus,
    ) -> Result<bool> {
        let key = (team_name.to_owned(), task_id.to_owned());
        let mut inner = self.inner.write();
        let Some(mut record) = inner.tasks.get(&key).cloned() else {
            return Ok(false);
        };
        if record.status != from || !TaskStatus::is_valid_transition(from, to) {
            return Ok(false);
        }
        record.status = to;
        record.updated_at = refresh(record.updated_at);
        inner.tasks.insert(key, record);
        Ok(true)
    }

    async fn add_dependency(&self, team_name: &str, task_id: &str, depends_on: &str) -> Result<()> {
        require_key(team_name, "team_name")?;
        require_key(task_id, "task_id")?;
        require_key(depends_on, "depends_on")?;
        if task_id == depends_on {
            return Err(Error::validation(format!(
                "task '{task_id}' cannot depend on itself"
            )));
        }
        let key = (team_name.to_owned(), task_id.to_owned());
        self.inner
            .write()
            .depends_on
            .entry(key)
            .or_default()
            .insert(depends_on.to_owned());
        Ok(())
    }

    async fn dependencies(&self, team_name: &str, task_id: &str) -> Result<Vec<String>> {
        let key = (team_name.to_owned(), task_id.to_owned());
        let inner = self.inner.read();
        Ok(inner
            .depends_on
            .get(&key)
            .map(|ids| ids.iter().cloned().collect())
            .unwrap_or_default())
    }

    async fn dependents(&self, team_name: &str, task_id: &str) -> Result<Vec<String>> {
        let inner = self.inner.read();
        let mut rows: Vec<String> = inner
            .depends_on
            .iter()
            .filter(|((team, _), _)| team == team_name)
            .filter(|(_, ids)| ids.contains(task_id))
            .map(|((_, dependent), _)| dependent.clone())
            .collect();
        rows.sort();
        Ok(rows)
    }

    async fn insert_vote(&self, record: ReviewVote) -> Result<()> {
        let mut inner = self.inner.write();
        let existing = inner.votes.iter().position(|vote| {
            vote.team_name == record.team_name
                && vote.task_id == record.task_id
                && vote.round == record.round
                && vote.reviewer == record.reviewer
        });
        match existing {
            Some(index) => inner.votes[index] = record,
            None => inner.votes.push(record),
        }
        Ok(())
    }

    async fn votes(&self, team_name: &str, task_id: &str, round: u32) -> Result<Vec<ReviewVote>> {
        let inner = self.inner.read();
        let mut rows: Vec<ReviewVote> = inner
            .votes
            .iter()
            .filter(|vote| {
                vote.team_name == team_name && vote.task_id == task_id && vote.round == round
            })
            .cloned()
            .collect();
        rows.sort_by(|left, right| left.reviewer.cmp(&right.reviewer));
        Ok(rows)
    }

    async fn insert_message(&self, record: MessageRecord) -> Result<()> {
        require_key(&record.team_name, "team_name")?;
        require_key(&record.message_id, "message_id")?;
        let mut inner = self.inner.write();
        let existing = inner.messages.iter().position(|message| {
            message.team_name == record.team_name && message.message_id == record.message_id
        });
        match existing {
            Some(index) => inner.messages[index] = record,
            None => inner.messages.push(record),
        }
        Ok(())
    }

    async fn unread_messages(
        &self,
        team_name: &str,
        member_name: &str,
    ) -> Result<Vec<MessageRecord>> {
        let inner = self.inner.read();
        let watermark = inner
            .watermarks
            .get(&(team_name.to_owned(), member_name.to_owned()))
            .copied();
        let mut rows: Vec<MessageRecord> = inner
            .messages
            .iter()
            .filter(|message| message.team_name == team_name)
            .filter(|message| is_unread_for(message, member_name, watermark))
            .cloned()
            .collect();
        rows.sort_by(|left, right| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.message_id.cmp(&right.message_id))
        });
        Ok(rows)
    }

    async fn mark_read(
        &self,
        team_name: &str,
        member_name: &str,
        message_ids: &[String],
    ) -> Result<()> {
        let mut inner = self.inner.write();
        for message_id in message_ids {
            let Some(message) = inner.messages.iter_mut().find(|message| {
                message.team_name == team_name && message.message_id == *message_id
            }) else {
                continue;
            };
            // Broadcast read state is the member watermark; a row flag never represents it.
            if message.broadcast || message.to_member_name.as_deref() != Some(member_name) {
                continue;
            }
            message.is_read = true;
        }
        Ok(())
    }

    async fn read_watermark(&self, team_name: &str, member_name: &str) -> Result<Option<u64>> {
        let key = (team_name.to_owned(), member_name.to_owned());
        Ok(self.inner.read().watermarks.get(&key).copied())
    }

    async fn set_read_watermark(
        &self,
        team_name: &str,
        member_name: &str,
        timestamp: u64,
    ) -> Result<()> {
        require_key(team_name, "team_name")?;
        require_key(member_name, "member_name")?;
        let key = (team_name.to_owned(), member_name.to_owned());
        let mut inner = self.inner.write();
        match inner.watermarks.get(&key).copied() {
            Some(read_at) if read_at >= timestamp => {}
            _ => {
                inner.watermarks.insert(key, timestamp);
            }
        }
        Ok(())
    }

    async fn message_count(&self, team_name: &str, member_name: &str) -> Result<usize> {
        let inner = self.inner.read();
        let watermark = inner
            .watermarks
            .get(&(team_name.to_owned(), member_name.to_owned()))
            .copied();
        Ok(inner
            .messages
            .iter()
            .filter(|message| message.team_name == team_name)
            .filter(|message| is_unread_for(message, member_name, watermark))
            .count())
    }
}
