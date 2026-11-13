//! Organisation mailbox: leader-to-leader messages, per-team receipts and system notices.
//!
//! The mailbox is the cross-team counterpart of a team's own mailbox: one leader writes to
//! another team's leader, or the organisation itself notifies a team. It sits *above*
//! [`crate::runtime`] and only persists rows — it never spawns, wakes or prompts a member.
//!
//! Python parity source: `openjiuwen/agent_teams/organization/message_service.py`
//! (`OrgMessageService`: durable leader messages plus a receipt row per recipient, and the
//! `__organization__` system notifications). The Rust port folds the Python message row and its
//! receipt row into the [`crate::storage::TeamStore`] message table: a direct row's `is_read` flag
//! and a broadcast row's per-team watermark *are* the receipts.
//!
//! ## Storage model
//!
//! Rows live in the same store as the organisation task pool, in the *message* table, under
//! `crate::org::org_namespace(organization_id)` (for example `org::acme`); there is no second
//! database. A row is encoded as:
//!
//! | Column | Value |
//! | --- | --- |
//! | `team_name` | the namespace, `org::<organization_id>` |
//! | `message_id` | `msg_<32 lowercase hex>` (from [`openjiuwen_common::utils::id::uuid_hex`]) |
//! | `from_member_name` | the sending team name, or [`crate::org::ORG_SYSTEM_SENDER`] for a notification |
//! | `to_member_name` | `Some(team)` for a direct message, `None` for a broadcast |
//! | `broadcast` | `true` for broadcasts only |
//! | `protocol` | [`crate::org::ORG_MESSAGE_PROTOCOL`] (`"org"`); reads skip every other protocol |
//! | `content` | JSON object `{ "kind": …, "subject": …, "body": … }` |
//! | `timestamp` | the mailbox's monotonic send stamp |
//!
//! Receipts are exactly the store's read state: `mark_read` flags the direct rows of one team, and
//! the per-team watermark (`read_watermark` / `set_read_watermark`) is the broadcast cursor. A
//! broadcast is unread for every team except its own sender while its timestamp is newer than that
//! team's watermark, so a team never receives its own announcement and an acknowledgment can never
//! move a cursor backwards.
//!
//! ## Stamps
//!
//! `TeamStore::insert_message` takes the caller's timestamp verbatim and offers no per-row getter,
//! so the mailbox stamps each row itself: [`openjiuwen_common::utils::id::unix_millis`] plus a
//! per-mailbox monotonic guard (`max(now, previous + 1)`), which keeps two sends inside one
//! millisecond ordered. Tests never assert a wall-clock value — only that stamps grow and that the
//! row round-trips. Because the stamp is supplied, the returned [`OrgMessage`] *is* the persisted
//! row; there is nothing to re-read (and no trait method to read a single message back).
//!
//! ## Delivery contract
//!
//! A mailbox message is a **persisted notification, not a wake-up**. Nothing in this module
//! publishes a bus event, calls a model, touches the network, reads a process-global or starts a
//! task: `inbox` only returns the unread rows, and the caller decides what to do with them. The
//! runtime integration — draining [`OrgMailbox::inbox`](crate::org::mailbox::OrgMailbox::inbox) in the leader loop
//! (`crate::runtime::leader::run`) and turning each row into a `TeamTopic::Message` event or a
//! leader prompt fragment — is a later step; the within-team analogue it follows is
//! `TeamBackend::send_message` in [`crate::runtime::backend`], whose rows this mailbox deliberately
//! does not touch (different table rows: protocol `"org"` vs `"plain"`, namespace vs team).
//!
//! Corrupt or foreign rows are skipped, never fatal: a row whose `protocol` is not `"org"` or whose
//! `content` is not a valid payload is invisible to every read.
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::{unix_millis, uuid_hex};
use serde::{Deserialize, Serialize};

use super::{
    org_namespace, validate_org_id, MEMBER_QUALIFIER, ORG_MESSAGE_PROTOCOL, ORG_NAMESPACE_PREFIX,
    ORG_SYSTEM_SENDER,
};
use crate::constants::MAX_MESSAGE_CONTENT_CHARS;
use crate::storage::{MessageRecord, TeamStore};

/// Maximum length of an organisation team name, in characters.
///
/// The team name is both a store key and (in the wider organisation layer) a path segment, so it is
/// bounded the same way an organisation id is.
pub const MAX_MAILBOX_TEAM_NAME_CHARS: usize = 96;

/// Prefix of every organisation message id: `msg_` plus 32 lowercase hex characters.
pub const ORG_MESSAGE_ID_PREFIX: &str = "msg_";

/// Subject a system notification carries when the caller does not supply one.
pub const ORG_SYSTEM_DEFAULT_SUBJECT: &str = "organization";

/// What a mailbox row represents.
///
/// Wire values are `leader`, `system` and `broadcast`; the default is
/// [`OrgMessageKind::Leader`], matching the `serde` default of the persisted payload.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum OrgMessageKind {
    /// A leader-to-leader direct message ([`OrgMailbox::send`]).
    #[default]
    Leader,
    /// A notification from the organisation itself ([`OrgMailbox::notify`] /
    /// [`OrgMailbox::announce`]).
    System,
    /// A leader's announcement to every other team ([`OrgMailbox::broadcast`]).
    Broadcast,
}

impl OrgMessageKind {
    /// Returns the wire string of this kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Leader => "leader",
            Self::System => "system",
            Self::Broadcast => "broadcast",
        }
    }
}

/// One organisation mailbox message, decoded from its store row.
///
/// `to_team == None` means a broadcast. `read` mirrors the persisted direct-row flag; every row
/// returned by [`OrgMailbox::inbox`] is unread, so it is `false` there, and a freshly sent message
/// is always `false`. Unknown fields are tolerated and every field defaults, so a row written by an
/// older build still decodes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgMessage {
    /// Message identity, `msg_<32 lowercase hex>`.
    pub message_id: String,
    /// Organisation the message belongs to.
    pub organization_id: String,
    /// Sending team name, or [`ORG_SYSTEM_SENDER`] for a system notification.
    pub from_team: String,
    /// Receiving team name; `None` for a broadcast.
    pub to_team: Option<String>,
    /// What the message represents.
    pub kind: OrgMessageKind,
    /// Short human-facing summary.
    pub subject: String,
    /// Message body.
    pub body: String,
    /// Monotonic send stamp in milliseconds, used for ordering and as the broadcast read cursor.
    pub timestamp: u64,
    /// Read state of the persisted direct row; broadcasts never carry a row flag.
    pub read: bool,
}

impl OrgMessage {
    /// Returns `true` when this message addresses every other team.
    #[must_use]
    pub const fn is_broadcast(&self) -> bool {
        self.to_team.is_none()
    }

    /// Returns `true` when this message comes from the organisation itself.
    ///
    /// Either marker is enough: [`OrgMessageKind::System`], or [`ORG_SYSTEM_SENDER`] as the sender
    /// (a decoded row keeps both in step, but a hand-built value may set only one).
    #[must_use]
    pub fn is_system(&self) -> bool {
        self.kind == OrgMessageKind::System || self.from_team == ORG_SYSTEM_SENDER
    }

    /// Returns at most `max_chars` characters of the message, for a log line or a prompt excerpt.
    ///
    /// The body is used, falling back to the subject when the body is empty. Truncation is
    /// char-boundary safe and gives up the last character for an ellipsis (`…`), so the result never
    /// exceeds `max_chars` characters; `max_chars == 0` yields an empty string.
    #[must_use]
    pub fn preview(&self, max_chars: usize) -> String {
        let source = if self.body.is_empty() {
            &self.subject
        } else {
            &self.body
        };
        if source.chars().count() <= max_chars {
            return source.clone();
        }
        if max_chars == 0 {
            return String::new();
        }
        let mut truncated: String = source.chars().take(max_chars - 1).collect();
        truncated.push('…');
        truncated
    }
}

/// Leader-to-leader mailbox of one organisation.
///
/// The handle is cheap to clone and shares the store and the monotonic stamp guard with its
/// clones. It holds no global state: build one per organisation (several may share the same
/// [`TeamStore`] without seeing each other's rows, because each writes under its own namespace).
#[derive(Clone)]
pub struct OrgMailbox {
    store: Arc<dyn TeamStore>,
    organization_id: String,
    namespace: String,
    last_stamp: Arc<AtomicU64>,
}

impl fmt::Debug for OrgMailbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrgMailbox")
            .field("organization_id", &self.organization_id)
            .field("namespace", &self.namespace)
            .field("store", &self.store.name())
            .field("last_stamp", &self.last_stamp.load(Ordering::Relaxed))
            .finish()
    }
}

impl OrgMailbox {
    /// Opens the mailbox of `organization_id` over `store`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `organization_id` fails
    /// [`validate_org_id`].
    pub fn new(store: Arc<dyn TeamStore>, organization_id: impl AsRef<str>) -> Result<Self> {
        let organization_id = organization_id.as_ref();
        validate_org_id(organization_id)?;
        Ok(Self {
            store,
            organization_id: organization_id.to_owned(),
            namespace: org_namespace(organization_id),
            last_stamp: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Returns the organisation id this mailbox belongs to.
    #[must_use]
    pub fn organization_id(&self) -> &str {
        &self.organization_id
    }

    /// Returns the store namespace holding every row of this mailbox (`org::<id>`).
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns the backing store.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn TeamStore> {
        &self.store
    }

    /// Sends a direct message from one team's leader to another team's leader.
    ///
    /// # Errors
    ///
    /// * [`Error::validation`] when either team name is empty, contains `::`, or uses the `org::`
    ///   namespace prefix, or when `subject`/`body` exceeds [`MAX_MESSAGE_CONTENT_CHARS`]
    ///   characters.
    /// * [`Error::validation`] with `team '<name>' cannot message itself` when both names match.
    /// * Whatever the store fails with.
    pub async fn send(
        &self,
        from_team: &str,
        to_team: &str,
        subject: &str,
        body: &str,
    ) -> Result<OrgMessage> {
        validate_team_name(from_team)?;
        validate_team_name(to_team)?;
        if from_team == to_team {
            return Err(Error::validation(format!(
                "team '{from_team}' cannot message itself"
            )));
        }
        validate_content_length("subject", subject)?;
        validate_content_length("body", body)?;
        self.persist(
            from_team,
            Some(to_team),
            OrgMessageKind::Leader,
            subject,
            body,
        )
        .await
    }

    /// Announces `subject`/`body` to every team except the sender.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an invalid team name or oversized content, and whatever the
    /// store fails with.
    pub async fn broadcast(
        &self,
        from_team: &str,
        subject: &str,
        body: &str,
    ) -> Result<OrgMessage> {
        validate_team_name(from_team)?;
        validate_content_length("subject", subject)?;
        validate_content_length("body", body)?;
        self.persist(from_team, None, OrgMessageKind::Broadcast, subject, body)
            .await
    }

    /// Sends a system notification from the organisation to one team.
    ///
    /// The subject defaults to [`ORG_SYSTEM_DEFAULT_SUBJECT`] and the sender is
    /// [`ORG_SYSTEM_SENDER`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an invalid team name or oversized body, and whatever the
    /// store fails with.
    pub async fn notify(&self, to_team: &str, body: &str) -> Result<OrgMessage> {
        validate_team_name(to_team)?;
        validate_content_length("body", body)?;
        self.persist(
            ORG_SYSTEM_SENDER,
            Some(to_team),
            OrgMessageKind::System,
            ORG_SYSTEM_DEFAULT_SUBJECT,
            body,
        )
        .await
    }

    /// Sends a system notification from the organisation to every team.
    ///
    /// The subject defaults to [`ORG_SYSTEM_DEFAULT_SUBJECT`]; the row is a broadcast, so the
    /// organisation itself never sees it in its own inbox.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an oversized body, and whatever the store fails with.
    pub async fn announce(&self, body: &str) -> Result<OrgMessage> {
        validate_content_length("body", body)?;
        self.persist(
            ORG_SYSTEM_SENDER,
            None,
            OrgMessageKind::System,
            ORG_SYSTEM_DEFAULT_SUBJECT,
            body,
        )
        .await
    }

    /// Returns the unread messages of `team`, oldest first.
    ///
    /// Rows are ordered by the store (`timestamp`, then `message_id`). Rows of another protocol and
    /// rows whose payload does not decode as an organisation message are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an invalid team name, and whatever the store fails with.
    pub async fn inbox(&self, team: &str) -> Result<Vec<OrgMessage>> {
        validate_team_name(team)?;
        let rows = self.store.unread_messages(&self.namespace, team).await?;
        Ok(rows.iter().filter_map(|row| self.decode(row)).collect())
    }

    /// Marks the given messages as read for `team`.
    ///
    /// Direct ids are flagged through [`TeamStore::mark_read`]; the broadcast ids among them advance
    /// the team's watermark to the newest acknowledged broadcast (never backwards). Ids that are
    /// unknown, already read, addressed to another team, or not organisation rows are ignored, so a
    /// second call with the same ids returns `0`.
    ///
    /// Returns how many rows actually changed the team's read state: direct rows flagged plus the
    /// acknowledged broadcasts that moved the watermark.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an invalid team name, and whatever the store fails with.
    pub async fn acknowledge(&self, team: &str, message_ids: &[String]) -> Result<usize> {
        validate_team_name(team)?;
        if message_ids.is_empty() {
            return Ok(0);
        }
        let unread = self.store.unread_messages(&self.namespace, team).await?;
        let mut direct_ids = Vec::new();
        let mut broadcast_stamps = Vec::new();
        for row in &unread {
            // Only rows the mailbox can decode count: a foreign-protocol or corrupt row is not a
            // mailbox message, so acknowledging its id changes nothing.
            if self.decode(row).is_none() || !message_ids.iter().any(|id| id == &row.message_id) {
                continue;
            }
            if row.broadcast {
                broadcast_stamps.push(row.timestamp);
            } else {
                direct_ids.push(row.message_id.clone());
            }
        }
        if !direct_ids.is_empty() {
            self.store
                .mark_read(&self.namespace, team, &direct_ids)
                .await?;
        }
        let mut cleared_broadcasts = 0;
        if let Some(newest) = broadcast_stamps.iter().copied().max() {
            let watermark = self.store.read_watermark(&self.namespace, team).await?;
            if watermark.is_none_or(|read_at| newest > read_at) {
                self.store
                    .set_read_watermark(&self.namespace, team, newest)
                    .await?;
                cleared_broadcasts = broadcast_stamps
                    .iter()
                    .filter(|stamp| watermark.is_none_or(|read_at| **stamp > read_at))
                    .count();
            }
        }
        Ok(direct_ids.len() + cleared_broadcasts)
    }

    /// Marks every unread message of `team` as read and returns how many were cleared.
    ///
    /// Direct rows are flagged and, when any broadcast is outstanding, the watermark moves to the
    /// newest unread organisation message visible to the team. A team with nothing unread (or only
    /// rows of another protocol) clears nothing and returns `0`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an invalid team name, and whatever the store fails with.
    pub async fn acknowledge_all(&self, team: &str) -> Result<usize> {
        validate_team_name(team)?;
        let unread = self.store.unread_messages(&self.namespace, team).await?;
        let org_rows: Vec<&MessageRecord> = unread
            .iter()
            .filter(|row| self.decode(row).is_some())
            .collect();
        if org_rows.is_empty() {
            return Ok(0);
        }
        let direct_ids: Vec<String> = org_rows
            .iter()
            .filter(|row| !row.broadcast)
            .map(|row| row.message_id.clone())
            .collect();
        if !direct_ids.is_empty() {
            self.store
                .mark_read(&self.namespace, team, &direct_ids)
                .await?;
        }
        if org_rows.iter().any(|row| row.broadcast) {
            if let Some(newest) = org_rows.iter().map(|row| row.timestamp).max() {
                self.store
                    .set_read_watermark(&self.namespace, team, newest)
                    .await?;
            }
        }
        Ok(org_rows.len())
    }

    /// Returns the team's broadcast watermark, or `None` when it never acknowledged a broadcast.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an invalid team name, and whatever the store fails with.
    pub async fn watermark(&self, team: &str) -> Result<Option<u64>> {
        validate_team_name(team)?;
        self.store.read_watermark(&self.namespace, team).await
    }

    /// Returns how many unread organisation messages `team` has, i.e. the size of
    /// [`OrgMailbox::inbox`].
    ///
    /// It counts decodable organisation rows only, which is why it does not delegate to the store's
    /// [`TeamStore::message_count`] (that counts every protocol and every corrupt row).
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an invalid team name, and whatever the store fails with.
    pub async fn unread_count(&self, team: &str) -> Result<usize> {
        Ok(self.inbox(team).await?.len())
    }

    /// Persists one row and returns the [`OrgMessage`] it represents.
    async fn persist(
        &self,
        from_team: &str,
        to_team: Option<&str>,
        kind: OrgMessageKind,
        subject: &str,
        body: &str,
    ) -> Result<OrgMessage> {
        let message_id = format!("{ORG_MESSAGE_ID_PREFIX}{}", uuid_hex());
        let timestamp = self.next_timestamp();
        let payload = OrgPayload {
            kind,
            subject: subject.to_owned(),
            body: body.to_owned(),
        };
        let content = serde_json::to_string(&payload).map_err(Error::external)?;
        let mut row = match to_team {
            Some(team) => MessageRecord::direct(
                self.namespace.as_str(),
                message_id.as_str(),
                from_team,
                team,
                content,
                timestamp,
            ),
            None => MessageRecord::broadcast(
                self.namespace.as_str(),
                message_id.as_str(),
                from_team,
                content,
                timestamp,
            ),
        };
        row.protocol = String::from(ORG_MESSAGE_PROTOCOL);
        self.store.insert_message(row).await?;
        tracing::debug!(
            organization = %self.organization_id,
            message = %message_id,
            from = from_team,
            to = to_team.unwrap_or("*"),
            kind = kind.as_str(),
            "organisation message persisted"
        );
        Ok(OrgMessage {
            message_id,
            organization_id: self.organization_id.clone(),
            from_team: from_team.to_owned(),
            to_team: to_team.map(str::to_owned),
            kind,
            subject: subject.to_owned(),
            body: body.to_owned(),
            timestamp,
            read: false,
        })
    }

    /// Decodes a store row, returning `None` for a foreign protocol or a corrupt payload.
    fn decode(&self, row: &MessageRecord) -> Option<OrgMessage> {
        if row.protocol != ORG_MESSAGE_PROTOCOL {
            return None;
        }
        // A payload is a JSON object; serde would happily decode an empty array into the
        // all-default struct, so the shape is checked before the typed decode.
        let payload: serde_json::Value = serde_json::from_str(&row.content).ok()?;
        if !payload.is_object() {
            return None;
        }
        let payload: OrgPayload = serde_json::from_value(payload).ok()?;
        Some(OrgMessage {
            message_id: row.message_id.clone(),
            organization_id: self.organization_id.clone(),
            from_team: row.from_member_name.clone(),
            to_team: if row.broadcast {
                None
            } else {
                row.to_member_name.clone()
            },
            kind: payload.kind,
            subject: payload.subject,
            body: payload.body,
            timestamp: row.timestamp,
            read: row.is_read,
        })
    }

    /// Returns the next send stamp: wall-clock milliseconds, never at or below the previous stamp.
    fn next_timestamp(&self) -> u64 {
        let now = u64::try_from(unix_millis()).unwrap_or(0);
        let mut previous = self.last_stamp.load(Ordering::Relaxed);
        loop {
            let next = now.max(previous.saturating_add(1));
            match self.last_stamp.compare_exchange_weak(
                previous,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return next,
                Err(observed) => previous = observed,
            }
        }
    }
}

/// Body of [`MessageRecord::content`]: the organisation-specific half of a row.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct OrgPayload {
    kind: OrgMessageKind,
    subject: String,
    body: String,
}

/// Rejects a team name that cannot be a mailbox key.
fn validate_team_name(team: &str) -> Result<()> {
    if team.is_empty() {
        return Err(Error::validation(
            "organization team name must not be empty",
        ));
    }
    if team.starts_with(ORG_NAMESPACE_PREFIX) {
        return Err(Error::validation(format!(
            "organization team name '{team}' uses the reserved organization namespace prefix \
             '{ORG_NAMESPACE_PREFIX}'"
        )));
    }
    if team.contains(MEMBER_QUALIFIER) {
        return Err(Error::validation(format!(
            "organization team name '{team}' must not contain '{MEMBER_QUALIFIER}'"
        )));
    }
    let length = team.chars().count();
    if length > MAX_MAILBOX_TEAM_NAME_CHARS {
        return Err(Error::validation(format!(
            "organization team name exceeds {MAX_MAILBOX_TEAM_NAME_CHARS} characters (got {length})"
        )));
    }
    Ok(())
}

/// Rejects a subject or body longer than [`MAX_MESSAGE_CONTENT_CHARS`].
fn validate_content_length(field: &str, value: &str) -> Result<()> {
    let length = value.chars().count();
    if length > MAX_MESSAGE_CONTENT_CHARS {
        return Err(Error::validation(format!(
            "organization message {field} exceeds {MAX_MESSAGE_CONTENT_CHARS} characters (got \
             {length})"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "mailbox_tests.rs"]
mod mailbox_tests;
