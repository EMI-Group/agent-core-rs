//! Append-only session records written by the CLI.
//!
//! Each CLI session is a JSON document at `<home>/sessions/<session_id>.json`
//! with the Python shape:
//!
//! ```json
//! {"session_id": "cli-1a2b3c4d", "model": "gpt-4o", "created_at": "…",
//!  "messages": [{"role": "user", "content": "hi", "timestamp": "…"}]}
//! ```
//!
//! Writes go through a temporary file in the same directory followed by a
//! rename, so a crashed process never leaves a half-written record behind.

use std::path::{Path, PathBuf};

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// One persisted CLI session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionRecord {
    /// Session identifier, for example `cli-1a2b3c4d`.
    pub session_id: String,
    /// Model name the session ran with; empty when it is not known yet.
    pub model: String,
    /// Creation timestamp, RFC 3339 in UTC.
    pub created_at: String,
    /// Conversation transcript, oldest message first.
    pub messages: Vec<MessageRecord>,
}

/// One message of a persisted session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MessageRecord {
    /// Message role: `user` or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
    /// Creation timestamp, RFC 3339 in UTC.
    pub timestamp: String,
}

/// Summary of one persisted CLI session.
///
/// This is the row `/sessions` prints: the identifying fields of a
/// [`SessionRecord`] plus the transcript length, without loading the messages.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionSummary {
    /// Session identifier, for example `cli-1a2b3c4d`.
    pub session_id: String,
    /// Model name the session ran with; empty when it was not recorded.
    pub model: String,
    /// Creation timestamp, RFC 3339 in UTC.
    pub created_at: String,
    /// Number of messages in the persisted transcript.
    pub messages: usize,
}

/// Reads and writes the session records of one [`OpenJiuwenHome`].
#[derive(Debug, Clone)]
pub struct SessionStore {
    home: OpenJiuwenHome,
}

impl SessionStore {
    /// Builds a store rooted at `home`.
    #[must_use]
    pub fn new(home: &OpenJiuwenHome) -> Self {
        Self { home: home.clone() }
    }

    /// Returns the record path: `<home>/sessions/<session_id>.json`.
    #[must_use]
    pub fn path_for(&self, session_id: &str) -> PathBuf {
        self.home.sessions_dir().join(format!("{session_id}.json"))
    }

    /// Creates (or overwrites) the record for `session_id` with an empty transcript.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the sessions directory cannot be created
    /// or the record cannot be written.
    pub fn new_session(&self, session_id: &str, model: &str) -> Result<SessionRecord> {
        let record = SessionRecord {
            session_id: session_id.to_owned(),
            model: model.to_owned(),
            created_at: timestamp(),
            messages: Vec::new(),
        };
        write_atomic(&self.path_for(session_id), &record)?;
        Ok(record)
    }

    /// Appends one message, creating the record when it does not exist yet.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the existing record cannot be read or the
    /// updated record cannot be written.
    pub fn append_message(&self, session_id: &str, role: &str, content: &str) -> Result<()> {
        let mut record = self.load(session_id)?.unwrap_or_else(|| SessionRecord {
            session_id: session_id.to_owned(),
            ..SessionRecord::default()
        });
        record.messages.push(MessageRecord {
            role: role.to_owned(),
            content: content.to_owned(),
            timestamp: timestamp(),
        });
        write_atomic(&self.path_for(session_id), &record)
    }

    /// Loads the record for `session_id`, or `Ok(None)` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] naming the path when the file cannot be read
    /// or does not contain a valid record.
    pub fn load(&self, session_id: &str) -> Result<Option<SessionRecord>> {
        let path = self.path_for(session_id);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(Error::session(format!(
                    "failed to read session file {}: {error}",
                    path.display()
                )))
            }
        };
        serde_json::from_str(&text).map(Some).map_err(|error| {
            Error::session(format!(
                "failed to parse session file {}: {error}",
                path.display()
            ))
        })
    }

    /// Lists every readable session record, ordered by session id.
    ///
    /// A missing sessions directory is not an error: it yields an empty list. The
    /// entries are the `*.json` files of the sessions directory, sorted by file
    /// name, which is the session-id order Python's `/sessions` uses.
    ///
    /// Listing is best effort per file, matching Python: a file that cannot be
    /// read or parsed is logged with `tracing::warn!` and skipped, so one corrupt
    /// record never hides the healthy ones. Only the identifying fields and the
    /// transcript length are returned; see `SessionSummary`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] naming the directory when it exists but cannot
    /// be read at all.
    pub fn list(&self) -> Result<Vec<SessionSummary>> {
        let dir = self.home.sessions_dir();
        let read_dir = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(Error::session(format!(
                    "failed to read sessions directory {}: {error}",
                    dir.display()
                )))
            }
        };

        let mut paths = Vec::new();
        for entry in read_dir {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    tracing::warn!(
                        dir = %dir.display(),
                        %error,
                        "skipping unreadable sessions directory entry"
                    );
                    continue;
                }
            };
            let path = entry.path();
            let is_json = path.extension().and_then(std::ffi::OsStr::to_str) == Some("json");
            if !is_json || !matches!(entry.file_type(), Ok(kind) if kind.is_file()) {
                continue;
            }
            paths.push(path);
        }
        paths.sort_by(|left, right| left.file_name().cmp(&right.file_name()));

        let mut summaries = Vec::with_capacity(paths.len());
        for path in paths {
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        %error,
                        "skipping unreadable session file"
                    );
                    continue;
                }
            };
            let record: SessionRecord = match serde_json::from_str(&text) {
                Ok(record) => record,
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        %error,
                        "skipping malformed session file"
                    );
                    continue;
                }
            };
            let stem = path
                .file_stem()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or_default();
            summaries.push(SessionSummary {
                session_id: if record.session_id.is_empty() {
                    stem.to_owned()
                } else {
                    record.session_id
                },
                model: record.model,
                created_at: record.created_at,
                messages: record.messages.len(),
            });
        }
        Ok(summaries)
    }

    /// Replaces the transcript of `session_id` (used by `/compact`).
    ///
    /// The record is created when it does not exist yet; its `model` and
    /// `created_at` are preserved from the existing file. Every other session
    /// record is left untouched.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the existing record cannot be read or the
    /// replaced record cannot be written.
    pub fn replace_messages(&self, session_id: &str, messages: Vec<MessageRecord>) -> Result<()> {
        let mut record = self.load(session_id)?.unwrap_or_else(|| SessionRecord {
            session_id: session_id.to_owned(),
            ..SessionRecord::default()
        });
        record.messages = messages;
        write_atomic(&self.path_for(session_id), &record)
    }
}

/// Returns the current time as an RFC 3339 UTC timestamp.
fn timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Serializes `record` to `<path>.tmp` and renames it over `path`.
fn write_atomic(path: &Path, record: &SessionRecord) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|error| {
            Error::session(format!(
                "failed to create session directory {}: {error}",
                dir.display()
            ))
        })?;
    }
    let json = serde_json::to_string_pretty(record).map_err(|error| {
        Error::session(format!(
            "failed to encode session file {}: {error}",
            path.display()
        ))
    })?;
    let temp = temp_path(path);
    std::fs::write(&temp, json).map_err(|error| {
        Error::session(format!(
            "failed to write session file {}: {error}",
            temp.display()
        ))
    })?;
    std::fs::rename(&temp, path).map_err(|error| {
        Error::session(format!(
            "failed to replace session file {}: {error}",
            path.display()
        ))
    })
}

/// Returns the sibling temporary path used for the atomic rewrite.
fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use tempfile::TempDir;

    use super::*;

    /// Builds a store over a fresh temporary home.
    fn store() -> (TempDir, SessionStore) {
        let temp = TempDir::new().unwrap();
        let home = OpenJiuwenHome::from_root(temp.path());
        (temp, SessionStore::new(&home))
    }

    #[test]
    fn list_orders_by_session_id_and_counts_messages() {
        let (_temp, store) = store();
        for (id, model, transcript) in [
            ("cli-bbb", "gpt-4o", vec!["user", "assistant"]),
            ("cli-aaa", "gpt-4o-mini", vec!["user"]),
            ("cli-ccc", "gpt-4o", vec!["user", "assistant", "user"]),
        ] {
            store.new_session(id, model).unwrap();
            for role in transcript {
                store.append_message(id, role, "text").unwrap();
            }
        }

        let listed = store.list().unwrap();
        assert_eq!(
            listed
                .iter()
                .map(|summary| summary.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["cli-aaa", "cli-bbb", "cli-ccc"]
        );
        assert_eq!(listed[0].model, "gpt-4o-mini");
        assert_eq!(listed[0].messages, 1);
        assert_eq!(listed[1].messages, 2);
        assert_eq!(listed[2].messages, 3);
        assert_ne!(listed[0].created_at, "");
    }

    #[test]
    fn list_skips_unreadable_and_malformed_records() {
        let (temp, store) = store();
        store.new_session("cli-good", "gpt-4o").unwrap();
        store.append_message("cli-good", "user", "hi").unwrap();

        let sessions = temp.path().join("sessions");
        std::fs::write(sessions.join("cli-broken.json"), "{not json").unwrap();
        // Neither a stray non-JSON file nor a leftover temp file is a session.
        std::fs::write(sessions.join("notes.txt"), "ignore me").unwrap();
        std::fs::write(sessions.join("cli-temp.json.tmp"), "{}").unwrap();

        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].session_id, "cli-good");
        assert_eq!(listed[0].messages, 1);
    }

    #[test]
    fn list_fills_a_missing_session_id_from_the_file_stem() {
        let (temp, _store) = store();
        let sessions = temp.path().join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let record = SessionRecord {
            model: "gpt-4o".to_owned(),
            ..SessionRecord::default()
        };
        std::fs::write(
            sessions.join("cli-from-file.json"),
            serde_json::to_string(&record).unwrap(),
        )
        .unwrap();

        let listed = SessionStore::new(&OpenJiuwenHome::from_root(temp.path()))
            .list()
            .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].session_id, "cli-from-file");
        assert_eq!(listed[0].model, "gpt-4o");
        assert_eq!(listed[0].messages, 0);
    }

    #[test]
    fn list_of_a_missing_sessions_directory_is_empty() {
        let (_temp, store) = store();
        assert_eq!(store.list().unwrap(), Vec::new());
    }

    #[test]
    fn replace_messages_round_trips_and_leaves_other_records_alone() {
        let (_temp, store) = store();
        let original = store.new_session("cli-keep", "gpt-4o").unwrap();
        store.append_message("cli-keep", "user", "keep").unwrap();
        store.new_session("cli-other", "gpt-4o").unwrap();
        store
            .append_message("cli-other", "user", "untouched")
            .unwrap();

        let compacted = vec![
            MessageRecord {
                role: "assistant".to_owned(),
                content: "summary".to_owned(),
                timestamp: "2026-01-01T00:00:00+00:00".to_owned(),
            },
            MessageRecord {
                role: "user".to_owned(),
                content: "next".to_owned(),
                timestamp: "2026-01-01T00:00:01+00:00".to_owned(),
            },
        ];
        store
            .replace_messages("cli-keep", compacted.clone())
            .unwrap();

        let loaded = store.load("cli-keep").unwrap().unwrap();
        assert_eq!(loaded.messages, compacted);
        assert_eq!(loaded.session_id, "cli-keep");
        assert_eq!(loaded.model, original.model);
        assert_eq!(loaded.created_at, original.created_at);

        let other = store.load("cli-other").unwrap().unwrap();
        assert_eq!(other.messages.len(), 1);
        assert_eq!(other.messages[0].content, "untouched");
    }

    #[test]
    fn replace_messages_creates_a_missing_record() {
        let (_temp, store) = store();
        let messages = vec![MessageRecord {
            role: "user".to_owned(),
            content: "fresh".to_owned(),
            timestamp: "2026-01-01T00:00:00+00:00".to_owned(),
        }];
        store
            .replace_messages("cli-fresh", messages.clone())
            .unwrap();

        let loaded = store.load("cli-fresh").unwrap().unwrap();
        assert_eq!(loaded.session_id, "cli-fresh");
        assert_eq!(loaded.messages, messages);
        assert_eq!(store.list().unwrap().len(), 1);
    }
}
