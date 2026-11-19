//! Append-only session records written by the CLI.
//!
//! Each CLI session is one JSONL file at `<home>/sessions/<session_id>.json` — the Python
//! shape, stored as a journal instead of a document so a turn costs one append:
//!
//! ```text
//! {"type":"header","session_id":"cli-1a2b3c4d","model":"gpt-4o","created_at":"…"}
//! {"type":"message","role":"user","content":"hi","timestamp":"…"}
//! {"type":"message","role":"assistant","content":"hello","timestamp":"…"}
//! ```
//!
//! `SessionStore::load` rebuilds the [`SessionRecord`] the old document format carried, so
//! every reader (`sessions list`, `sessions show`, `--resume`/`--continue`) sees the same
//! transcript. A legacy single-document record (pretty-printed or one line, without a
//! `"type"` tag) is still read, so records written before the journal format keep working.
//!
//! # Cost
//!
//! Appending a turn is *one* `O_APPEND` write of the turn's lines: nothing is re-read,
//! re-parsed or re-printed, so a session of N turns costs O(N) in total instead of O(N²).
//! IO runs on [`tokio::task::spawn_blocking`] through [`SessionStore::append_turn`], so the
//! async runtime never blocks on the disk; `SessionStore::load`/`list` stay synchronous
//! because they run once per command, not once per turn.
//!
//! # Bounded retention
//!
//! A record never grows without limit: once the file passes
//! [`SessionStore::max_bytes`] (default [`MAX_RECORD_BYTES`]) the store compacts it down to
//! half the cap, keeping the **newest** messages (the newest is always kept, even when it
//! alone exceeds the cap) and writing an explicit
//! `{"type":"truncated","dropped":N,"timestamp":"…"}` marker line, so a reader can tell a
//! short transcript from a truncated one. Compaction only runs when the cap is crossed and
//! halves the file, so it stays amortised O(1) per append.
//!
//! # Concurrency
//!
//! Writers take a process-wide lock keyed by the record path (a private lock table, the only
//! process-global state in this module; it holds no user data), so two tasks appending to one
//! session can never interleave halves of a line. Across processes each append is a single
//! `O_APPEND` write, which the kernel serialises per file.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default cap on the retained size of one session record, in bytes.
pub const MAX_RECORD_BYTES: u64 = 4 * 1024 * 1024;

/// Bytes of a record head inspected when deciding whether it needs the legacy migration.
const MIGRATION_SNIFF_BYTES: usize = 4096;

/// Smallest cap [`SessionStore::with_max_bytes`] accepts, so a caller cannot force a store
/// into rewriting the record on every append.
pub const MIN_RECORD_BYTES: u64 = 1024;

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

impl MessageRecord {
    /// Builds a message stamped with the current time.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp: timestamp(),
        }
    }
}

/// Summary of one persisted CLI session.
///
/// This is the row `/sessions` prints: the identifying fields of a
/// [`SessionRecord`] plus the transcript length, without exposing the messages.
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

/// One line of a session record file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RecordLine {
    /// First line of a record: the session identity.
    Header {
        #[serde(default)]
        session_id: String,
        #[serde(default)]
        model: String,
        #[serde(default)]
        created_at: String,
    },
    /// One transcript message.
    Message {
        #[serde(default)]
        role: String,
        #[serde(default)]
        content: String,
        #[serde(default)]
        timestamp: String,
    },
    /// Marker written when compaction dropped the oldest messages.
    Truncated {
        #[serde(default)]
        dropped: usize,
        #[serde(default)]
        timestamp: String,
    },
}

/// Reads and writes the session records of one [`OpenJiuwenHome`].
#[derive(Debug, Clone)]
pub struct SessionStore {
    home: OpenJiuwenHome,
    max_bytes: u64,
}

impl SessionStore {
    /// Builds a store rooted at `home`, keeping up to [`MAX_RECORD_BYTES`] per record.
    #[must_use]
    pub fn new(home: &OpenJiuwenHome) -> Self {
        Self {
            home: home.clone(),
            max_bytes: MAX_RECORD_BYTES,
        }
    }

    /// Overrides the retained-size cap of one record.
    ///
    /// Values below [`MIN_RECORD_BYTES`] are clamped, so a record is never rewritten on every
    /// append. The cap is a byte budget for the whole file, header and marker included.
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes.max(MIN_RECORD_BYTES);
        self
    }

    /// The retained-size cap this store enforces.
    #[must_use]
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
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
        self.write_document(&record)?;
        Ok(record)
    }

    /// Appends one message, creating the record when it does not exist yet.
    ///
    /// This is the single-line, synchronous form kept for callers outside the turn loop; the
    /// async [`SessionStore::append_turn`] is what the CLI uses to persist a turn.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the record cannot be opened or written.
    pub fn append_message(&self, session_id: &str, role: &str, content: &str) -> Result<()> {
        self.append_blocking(
            session_id,
            &[MessageRecord {
                role: role.to_owned(),
                content: content.to_owned(),
                timestamp: timestamp(),
            }],
        )
    }

    /// Appends a whole turn's messages in one write, off the async runtime.
    ///
    /// One `O_APPEND` write per call means a turn costs O(1): nothing is re-read or
    /// re-encoded. The record is created (with its header line) when it does not exist, and
    /// the retained-size cap is enforced afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the record cannot be opened, written or compacted.
    pub async fn append_turn(&self, session_id: &str, messages: Vec<MessageRecord>) -> Result<()> {
        let store = self.clone();
        let session_id = session_id.to_owned();
        tokio::task::spawn_blocking(move || store.append_blocking(&session_id, &messages))
            .await
            .map_err(|error| Error::session(format!("session append task failed: {error}")))?
    }

    /// Appends `messages` synchronously; the blocking core of both append methods.
    fn append_blocking(&self, session_id: &str, messages: &[MessageRecord]) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        let path = self.path_for(session_id);
        let guard = record_lock(&path);
        let _held = guard.lock();
        ensure_parent(&path)?;
        migrate_legacy(&path)?;
        let mut buffer = String::new();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| io_error("open", &path, &error))?;
        let is_new = file
            .metadata()
            .map_err(|error| io_error("inspect", &path, &error))?
            .len()
            == 0;
        if is_new {
            buffer.push_str(&encode_line(&RecordLine::Header {
                session_id: session_id.to_owned(),
                model: String::new(),
                created_at: timestamp(),
            })?);
        }
        for message in messages {
            buffer.push_str(&encode_line(&RecordLine::Message {
                role: message.role.clone(),
                content: message.content.clone(),
                timestamp: message.timestamp.clone(),
            })?);
        }
        file.write_all(buffer.as_bytes())
            .map_err(|error| io_error("append to", &path, &error))?;
        drop(file);
        enforce_capacity(&path, self.max_bytes)
    }

    /// Loads the record for `session_id`, or `Ok(None)` when it does not exist.
    ///
    /// Both formats are read: the append-only journal this store writes and the legacy single
    /// JSON document earlier versions wrote.
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
        parse_record(&text, &path).map(Some)
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
            let record = match parse_record(&text, &path) {
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
        self.write_document(&record)
    }

    /// Replaces the transcript off the async runtime (the `/compact` path).
    ///
    /// # Errors
    ///
    /// Returns whatever [`SessionStore::replace_messages`] reports.
    pub async fn replace_messages_async(
        &self,
        session_id: &str,
        messages: Vec<MessageRecord>,
    ) -> Result<()> {
        let store = self.clone();
        let session_id = session_id.to_owned();
        tokio::task::spawn_blocking(move || store.replace_messages(&session_id, messages))
            .await
            .map_err(|error| Error::session(format!("session rewrite task failed: {error}")))?
    }

    /// Rewrites `record` as a fresh journal (header plus its messages), atomically.
    fn write_document(&self, record: &SessionRecord) -> Result<()> {
        let path = self.path_for(&record.session_id);
        let guard = record_lock(&path);
        let _held = guard.lock();
        write_journal(&path, record)
    }
}

/// Rewrites `path` as a journal holding `record`; the caller holds the path lock.
fn write_journal(path: &Path, record: &SessionRecord) -> Result<()> {
    let mut text = String::new();
    text.push_str(&encode_line(&RecordLine::Header {
        session_id: record.session_id.clone(),
        model: record.model.clone(),
        created_at: record.created_at.clone(),
    })?);
    for message in &record.messages {
        text.push_str(&encode_line(&RecordLine::Message {
            role: message.role.clone(),
            content: message.content.clone(),
            timestamp: message.timestamp.clone(),
        })?);
    }
    write_atomic(path, text.as_bytes())
}

/// Converts a legacy single-document record to the journal format before the first append.
///
/// The caller holds the path lock. The check reads only the first
/// [`MIGRATION_SNIFF_BYTES`] bytes of the record, so an append stays O(1): the whole file is
/// read exactly once, when a record written in the old document format is migrated.
fn migrate_legacy(path: &Path) -> Result<()> {
    let Some(prefix) = read_prefix(path, MIGRATION_SNIFF_BYTES)? else {
        return Ok(());
    };
    let first_line = prefix.lines().find(|line| !line.trim().is_empty());
    let Some(first_line) = first_line else {
        return Ok(());
    };
    let journal_line = serde_json::from_str::<RecordLine>(first_line).is_ok()
        || matches!(
            serde_json::from_str::<Value>(first_line),
            Ok(Value::Object(map)) if map.contains_key("type")
        );
    if journal_line {
        return Ok(());
    }
    let text = std::fs::read_to_string(path).map_err(|error| io_error("read", path, &error))?;
    let record = parse_record(&text, path)?;
    tracing::debug!(
        path = %path.display(),
        "migrating a legacy session record to the append-only journal format"
    );
    write_journal(path, &record)
}

/// Reads at most `limit` bytes from the head of `path`, or `None` when the file is empty.
fn read_prefix(path: &Path, limit: usize) -> Result<Option<String>> {
    use std::io::Read as _;
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_error("read", path, &error)),
    };
    let mut buffer = vec![0_u8; limit];
    let read = file
        .read(&mut buffer)
        .map_err(|error| io_error("read", path, &error))?;
    if read == 0 {
        return Ok(None);
    }
    buffer.truncate(read);
    Ok(Some(String::from_utf8_lossy(&buffer).into_owned()))
}

/// Returns the current time as an RFC 3339 UTC timestamp.
fn timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// The process-wide write-lock table, one lock per record path.
static RECORD_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();

/// Returns the write lock of `path`, creating it on first use.
///
/// The table is the module's only process-global state: it holds no user data, only one lock
/// per record path this process has written, so two [`SessionStore`] values (the CLI builds a
/// fresh one per call) still serialise their appends.
fn record_lock(path: &Path) -> Arc<Mutex<()>> {
    let table = RECORD_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut table = table.lock();
    Arc::clone(table.entry(path.to_path_buf()).or_default())
}

/// Encodes one record line, newline included.
fn encode_line(line: &RecordLine) -> Result<String> {
    let mut encoded = serde_json::to_string(line)
        .map_err(|error| Error::session(format!("failed to encode session line: {error}")))?;
    encoded.push('\n');
    Ok(encoded)
}

/// Creates the parent directory of `path` when it is missing.
fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|error| io_error("create the directory of", dir, &error))?;
    }
    Ok(())
}

/// Wraps one IO failure as a session error naming the path and the operation.
fn io_error(operation: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::session(format!(
        "failed to {operation} session file {}: {error}",
        path.display()
    ))
}

/// Classifies `text` as the journal format, by looking for a `"type"` tag on its first line.
fn is_journal(text: &str) -> bool {
    let Some(first) = text.lines().find(|line| !line.trim().is_empty()) else {
        return false;
    };
    matches!(
        serde_json::from_str::<Value>(first),
        Ok(Value::Object(map)) if map.contains_key("type")
    )
}

/// Parses a record: the journal format this store writes, or a legacy single document.
fn parse_record(text: &str, path: &Path) -> Result<SessionRecord> {
    if is_journal(text) {
        return Ok(parse_journal(text, path));
    }
    serde_json::from_str(text).map_err(|error| {
        Error::session(format!(
            "failed to parse session file {}: {error}",
            path.display()
        ))
    })
}

/// Rebuilds a [`SessionRecord`] from journal lines, skipping and reporting malformed ones.
fn parse_journal(text: &str, path: &Path) -> SessionRecord {
    let mut record = SessionRecord::default();
    let mut dropped = 0_usize;
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<RecordLine>(line) {
            Ok(RecordLine::Header {
                session_id,
                model,
                created_at,
            }) => {
                record.session_id = session_id;
                record.model = model;
                record.created_at = created_at;
            }
            Ok(RecordLine::Message {
                role,
                content,
                timestamp,
            }) => record.messages.push(MessageRecord {
                role,
                content,
                timestamp,
            }),
            Ok(RecordLine::Truncated {
                dropped: count,
                timestamp,
            }) => {
                dropped = dropped.saturating_add(count);
                tracing::debug!(
                    path = %path.display(),
                    dropped = count,
                    timestamp,
                    "session record was truncated by the retention cap"
                );
            }
            Err(error) => tracing::warn!(
                path = %path.display(),
                line = index + 1,
                %error,
                "skipping a malformed session record line"
            ),
        }
    }
    if dropped > 0 {
        tracing::debug!(
            path = %path.display(),
            dropped,
            retained = record.messages.len(),
            "session record history is shorter than the conversation"
        );
    }
    record
}

/// Compacts `path` when it grew past `max_bytes`; called with the path lock held.
fn enforce_capacity(path: &Path, max_bytes: u64) -> Result<()> {
    let len = match std::fs::metadata(path) {
        Ok(meta) => meta.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(io_error("inspect", path, &error)),
    };
    if len <= max_bytes {
        return Ok(());
    }
    compact(path, max_bytes)
}

/// Drops the oldest messages until the record fits in half of `max_bytes`.
///
/// Halving (rather than trimming to exactly the cap) is what keeps compaction amortised: the
/// next one is only needed after another half-cap of appends. The newest message is never
/// dropped, and a [`RecordLine::Truncated`] marker records how many went.
fn compact(path: &Path, max_bytes: u64) -> Result<()> {
    let text = std::fs::read_to_string(path).map_err(|error| io_error("read", path, &error))?;
    let mut record = parse_record(&text, path)?;
    let target = max_bytes / 2;
    let mut kept: Vec<MessageRecord> = Vec::new();
    let mut bytes = 0_u64;
    for message in record.messages.iter().rev() {
        let line = encode_line(&RecordLine::Message {
            role: message.role.clone(),
            content: message.content.clone(),
            timestamp: message.timestamp.clone(),
        })?;
        let size = line.len() as u64;
        if !kept.is_empty() && bytes.saturating_add(size) > target {
            break;
        }
        bytes = bytes.saturating_add(size);
        kept.push(message.clone());
    }
    kept.reverse();
    let dropped = record.messages.len().saturating_sub(kept.len());
    record.messages = kept;
    let mut out = String::new();
    out.push_str(&encode_line(&RecordLine::Header {
        session_id: record.session_id.clone(),
        model: record.model.clone(),
        created_at: record.created_at.clone(),
    })?);
    if dropped > 0 {
        out.push_str(&encode_line(&RecordLine::Truncated {
            dropped,
            timestamp: timestamp(),
        })?);
    }
    for message in &record.messages {
        out.push_str(&encode_line(&RecordLine::Message {
            role: message.role.clone(),
            content: message.content.clone(),
            timestamp: message.timestamp.clone(),
        })?);
    }
    tracing::warn!(
        path = %path.display(),
        dropped,
        retained = record.messages.len(),
        max_bytes,
        "session record reached its retention cap; the oldest messages were dropped"
    );
    write_atomic(path, out.as_bytes())
}

/// Replaces `path` with `bytes` through a sibling temporary file.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure_parent(path)?;
    let temp = temp_path(path);
    std::fs::write(&temp, bytes).map_err(|error| io_error("write", &temp, &error))?;
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

    /// Encodes one message the way the store does, for byte-exact expectations.
    fn message_line(role: &str, content: &str, timestamp: &str) -> String {
        encode_line(&RecordLine::Message {
            role: role.to_owned(),
            content: content.to_owned(),
            timestamp: timestamp.to_owned(),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn appending_turns_costs_one_append_per_turn_and_never_rewrites() {
        let (temp, store) = store();
        let path = store.path_for("cli-append");
        let mut expected = 0_u64;
        for turn in 0..25 {
            let messages = vec![
                MessageRecord::new("user", format!("question {turn}")),
                MessageRecord::new("assistant", format!("answer {turn}")),
            ];
            store
                .append_turn("cli-append", messages.clone())
                .await
                .unwrap();
            for message in &messages {
                expected +=
                    message_line(&message.role, &message.content, &message.timestamp).len() as u64;
            }
            let len = std::fs::metadata(&path).unwrap().len();
            if turn == 0 {
                expected += encode_line(&RecordLine::Header {
                    session_id: "cli-append".to_owned(),
                    model: String::new(),
                    created_at: String::new(),
                })
                .unwrap()
                .len() as u64;
                // The header carries the real creation timestamp.
                let header = std::fs::read_to_string(&path).unwrap();
                let header = header.lines().next().unwrap().to_owned();
                expected = u64::try_from(header.len() + 1).unwrap() + expected
                    - encode_line(&RecordLine::Header {
                        session_id: "cli-append".to_owned(),
                        model: String::new(),
                        created_at: String::new(),
                    })
                    .unwrap()
                    .len() as u64;
            }
            assert_eq!(
                len, expected,
                "turn {turn} must append exactly its own lines: a rewritten record would \
                 change the length and cost O(n)"
            );
        }
        // No temporary file survives an append-only record.
        assert!(!temp.path().join("sessions/cli-append.json.tmp").exists());
        let loaded = store.load("cli-append").unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 50);
        assert_eq!(loaded.messages[0].role, "user");
        assert_eq!(loaded.messages[49].content, "answer 24");
    }

    #[tokio::test]
    async fn a_turn_round_trips_through_the_journal() {
        let (temp, store) = store();
        store.new_session("cli-turn", "gpt-4o").unwrap();
        let messages = vec![
            MessageRecord::new("user", "hi"),
            MessageRecord::new("assistant", "hello"),
        ];
        store
            .append_turn("cli-turn", messages.clone())
            .await
            .unwrap();

        let loaded = store.load("cli-turn").unwrap().unwrap();
        assert_eq!(loaded.session_id, "cli-turn");
        assert_eq!(loaded.model, "gpt-4o");
        assert_eq!(loaded.messages, messages);

        let raw = std::fs::read_to_string(temp.path().join("sessions/cli-turn.json")).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 3, "one header plus one line per message");
        assert!(lines[0].contains("\"type\":\"header\""), "{}", lines[0]);
        assert!(lines[1].contains("\"type\":\"message\""), "{}", lines[1]);
        assert_eq!(store.list().unwrap()[0].messages, 2);
    }

    #[tokio::test]
    async fn a_resumed_session_sees_the_full_history() {
        let (_temp, store) = store();
        store.new_session("cli-long", "gpt-4o-mini").unwrap();
        for turn in 0..50 {
            store
                .append_turn(
                    "cli-long",
                    vec![
                        MessageRecord::new("user", format!("q{turn}")),
                        MessageRecord::new("assistant", format!("a{turn}")),
                    ],
                )
                .await
                .unwrap();
        }
        let loaded = store.load("cli-long").unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 100);
        for (index, message) in loaded.messages.iter().enumerate() {
            let turn = index / 2;
            if index % 2 == 0 {
                assert_eq!(message.role, "user");
                assert_eq!(message.content, format!("q{turn}"));
            } else {
                assert_eq!(message.role, "assistant");
                assert_eq!(message.content, format!("a{turn}"));
            }
        }
    }

    #[tokio::test]
    async fn compaction_keeps_the_newest_messages_and_records_the_drop() {
        let (temp, _store) = store();
        let store = SessionStore::new(&OpenJiuwenHome::from_root(temp.path())).with_max_bytes(1024);
        assert_eq!(store.max_bytes(), 1024);
        store.new_session("cli-capped", "gpt-4o").unwrap();
        let padding = "x".repeat(80);
        for turn in 0..40 {
            store
                .append_turn(
                    "cli-capped",
                    vec![MessageRecord::new("user", format!("q{turn}{padding}"))],
                )
                .await
                .unwrap();
        }
        let path = store.path_for("cli-capped");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("\"type\":\"truncated\""),
            "a truncated record must say so"
        );
        let loaded = store.load("cli-capped").unwrap().unwrap();
        assert!(
            loaded.messages.len() < 40,
            "the cap must have dropped the oldest messages"
        );
        assert!(
            !loaded.messages.is_empty(),
            "the newest message is never dropped"
        );
        assert_eq!(
            loaded.messages.last().unwrap().content,
            format!("q39{padding}")
        );
        assert_eq!(loaded.model, "gpt-4o");
        assert!(
            std::fs::metadata(&path).unwrap().len() <= 1024,
            "compaction must bring the record back under the cap"
        );
    }

    #[tokio::test]
    async fn with_max_bytes_clamps_a_pathological_cap() {
        let (temp, _store) = store();
        let store = SessionStore::new(&OpenJiuwenHome::from_root(temp.path())).with_max_bytes(1);
        assert_eq!(store.max_bytes(), MIN_RECORD_BYTES);
    }

    #[tokio::test]
    async fn concurrent_appends_do_not_interleave_lines() {
        let (temp, _store) = store();
        let home = OpenJiuwenHome::from_root(temp.path());
        let first = SessionStore::new(&home);
        let second = SessionStore::new(&home);
        first.new_session("cli-race", "gpt-4o").unwrap();

        let left = tokio::spawn(async move {
            for turn in 0..100 {
                first
                    .append_turn(
                        "cli-race",
                        vec![MessageRecord::new("user", format!("left-{turn}"))],
                    )
                    .await
                    .unwrap();
            }
        });
        let right = tokio::spawn(async move {
            for turn in 0..100 {
                second
                    .append_turn(
                        "cli-race",
                        vec![MessageRecord::new("assistant", format!("right-{turn}"))],
                    )
                    .await
                    .unwrap();
            }
        });
        left.await.unwrap();
        right.await.unwrap();

        let path = SessionStore::new(&home).path_for("cli-race");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = 0_usize;
        for line in text.lines() {
            assert!(
                serde_json::from_str::<RecordLine>(line).is_ok(),
                "every line must stay a whole record line: {line}"
            );
            lines += 1;
        }
        assert_eq!(lines, 201, "one header plus both writers' 200 messages");
        let loaded = SessionStore::new(&home).load("cli-race").unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 200);
        let left_count = loaded
            .messages
            .iter()
            .filter(|message| message.content.starts_with("left-"))
            .count();
        assert_eq!(left_count, 100);
    }

    #[tokio::test]
    async fn a_legacy_document_record_still_loads_and_keeps_accepting_turns() {
        let (temp, store) = store();
        let sessions = temp.path().join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let legacy = SessionRecord {
            session_id: "cli-legacy".to_owned(),
            model: "gpt-4o".to_owned(),
            created_at: "2026-01-01T00:00:00+00:00".to_owned(),
            messages: vec![MessageRecord {
                role: "user".to_owned(),
                content: "earlier".to_owned(),
                timestamp: "2026-01-01T00:00:01+00:00".to_owned(),
            }],
        };
        std::fs::write(
            sessions.join("cli-legacy.json"),
            serde_json::to_string_pretty(&legacy).unwrap(),
        )
        .unwrap();

        assert_eq!(store.load("cli-legacy").unwrap().unwrap(), legacy);
        store
            .append_turn("cli-legacy", vec![MessageRecord::new("assistant", "later")])
            .await
            .unwrap();
        let loaded = store.load("cli-legacy").unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].content, "earlier");
        assert_eq!(loaded.messages[1].content, "later");
        assert_eq!(loaded.created_at, legacy.created_at);
        assert_eq!(store.list().unwrap()[0].messages, 2);
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
