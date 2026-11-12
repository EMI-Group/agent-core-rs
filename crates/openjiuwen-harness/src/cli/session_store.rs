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
