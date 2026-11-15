//! JSONL session logs under `sessions/` and their recall index.
//!
//! Ports the deferred session indexing of Python `core/memory/lite/`: each session is one
//! append-only JSONL file under `sessions/`, and a derived index makes those records
//! searchable alongside coding facts.
//!
//! # Layout
//!
//! ```text
//! <root>/sessions/YYYY-MM-DD/<session-id>.jsonl
//! ```
//!
//! The day comes from the caller (the day the turns belong to) and the session id is
//! sanitized, so the file name is self-describing and sortable and a hostile id cannot leave
//! [`SESSIONS_DIR`](crate::lite::session::SESSIONS_DIR). A portable id (one the sanitizer
//! leaves unchanged) names its log verbatim; an id that has to be folded carries an
//! eight-hex-digit discriminator, so `café` and `caf-` never share one file. The file holds
//! one JSON object per line — the text, a role or event kind, and a timestamp — so appending
//! a turn never rewrites earlier ones and the file stays readable with ordinary
//! line-oriented tools; a log whose last byte is not a newline (a truncated write) gains one
//! before the next append, so the tail is preserved and the new record starts on its own
//! line.
//! [`SessionStore`](crate::lite::session::SessionStore) discovers `sessions/**/*.jsonl`
//! recursively, as Python does, and exposes every record with its path relative to the store
//! root (`sessions/2026-11-14/abc.jsonl`) and its 1-based line number, so a hit can be cited
//! as `sessions/2026-11-14/abc.jsonl#L7`.
//!
//! # Tolerance
//!
//! Decoding is line-by-line and tolerant:
//! [`SessionRecord::parse_line`](crate::lite::session::SessionRecord::parse_line) decodes a
//! JSON object with a string content field, and a blank line is skipped. Every other line —
//! plain text, truncated JSON, a non-object payload, or a JSON object without a string
//! content — is kept as a record whose content is the raw line, so one bad append never makes
//! a session unreadable and no line is lost. A malformed line is tolerated input rather than
//! an error and is reported through the crate's diagnostics target; the line numbers of the
//! surviving records are preserved either way.
//!
//! # Recall
//!
//! [`rank_chunks`](crate::lite::session::rank_chunks) scores a record by keyword coverage of
//! the query (the same `\w+` tokenizer the crate uses elsewhere, lowercased, matching tokens
//! as substrings), normalized to `0.0..=1.0`, and orders hits by score descending then path
//! and line ascending. Scoring is keyword-only and needs no embedding model, so the index
//! always works offline; a [`SessionHit`](crate::lite::session::SessionHit) reports
//! `similarity = 0.0` and `keyword = score`.
//!
//! # Constraints
//!
//! Session logs are append-only and never rewritten or compacted by this module; the index is
//! derived and may be dropped and rebuilt from the JSONL files at any time. Writes stay inside
//! `<root>/sessions/`, filesystem failures are typed [`openjiuwen_common::error::Error`]s
//! carrying the path, and indexing runs only when the caller asks for it — there is no watcher
//! and no background task here.
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

use chrono::NaiveDate;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use openjiuwen_common::error::{Error, Result};

mod index;
mod record;

pub use index::{keyword_score, rank_chunks, SessionChunk, SessionHit};
pub use record::SessionRecord;

/// The directory (relative to the store root) holding the JSONL session logs.
pub const SESSIONS_DIR: &str = "sessions";

/// The tracing target of session-log diagnostics.
const TARGET: &str = "openjiuwen_memory::lite::session";

/// The recursive size stamp of the session directory.
///
/// It is the cheap staleness probe of a derived index: the file count, the summed length, and
/// the newest modification time in milliseconds since the Unix epoch (`0` when the platform
/// reports none). Two equal stamps mean no visible change, so a cache does not need to
/// re-read the logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirStamp {
    /// Number of `*.jsonl` files under [`SESSIONS_DIR`].
    pub files: usize,
    /// Total length of those files in bytes.
    pub total_len: u64,
    /// Newest modification time, in milliseconds since the Unix epoch.
    pub newest_ms: u128,
}

/// Reads and writes the JSONL session logs of one memory root.
///
/// The store is a plain path handle: cloning is free, no state is cached, and every method
/// touches the filesystem fresh. It never reaches outside `<root>/sessions/` for writes.
#[derive(Debug, Clone)]
pub struct SessionStore {
    /// The memory root; session logs live under `<root>/sessions/`.
    root: PathBuf,
}

impl SessionStore {
    /// Builds a store rooted at `root` (the memory root, not the sessions directory).
    #[must_use]
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// The memory root this store reads and writes under.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The sessions directory: `<root>/sessions`.
    #[must_use]
    pub fn dir(&self) -> PathBuf {
        self.root.join(SESSIONS_DIR)
    }

    /// Returns the log path of `session_id` on `date`:
    /// `<root>/sessions/YYYY-MM-DD/<sanitized id>.jsonl`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the sanitized id is empty (see
    /// [`sanitize_session_id`]). The `<id>.jsonl` file name means even an id of `.` or `..`
    /// stays a file inside the day directory, so the date directory is the only component
    /// that can vary.
    pub fn path_for(&self, session_id: &str, date: NaiveDate) -> Result<PathBuf> {
        let id = sanitize_session_id(session_id)?;
        let day = date.format("%Y-%m-%d").to_string();
        Ok(self.dir().join(day).join(format!("{id}.jsonl")))
    }

    /// Appends every non-empty record of `records` as one JSON line to the session log,
    /// creating the day directory and the file when they do not exist.
    ///
    /// Returns the log path. Empty (whitespace-only) records are skipped, so a caller may
    /// pass a whole turn list without filtering. Appending is additive: a second call adds
    /// its lines after the earlier ones and leaves them untouched.
    ///
    /// A log whose last byte is not a newline (a truncated write, or a file edited by hand)
    /// is repaired by writing one newline first, so the new records start on their own line
    /// instead of being glued to the truncated tail; the repair is reported.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `records` holds no non-empty record or the session
    /// id sanitizes to nothing, and [`Error::External`] carrying the path when the directory
    /// cannot be created or the file cannot be opened, written, or flushed.
    pub async fn append(
        &self,
        session_id: &str,
        date: NaiveDate,
        records: &[SessionRecord],
    ) -> Result<PathBuf> {
        let path = self.path_for(session_id, date)?;
        let mut buffer = String::new();
        let mut written = 0_usize;
        for record in records {
            if record.is_empty() {
                continue;
            }
            buffer.push_str(&record.to_json_line());
            buffer.push('\n');
            written += 1;
        }
        if written == 0 {
            return Err(Error::validation(format!(
                "session `{session_id}` append carries no non-empty record"
            )));
        }
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::external(error).with_context(format!(
                    "creating the session directory {}",
                    parent.display()
                ))
            })?;
        }
        if ends_mid_line(&path).await.map_err(|error| {
            Error::external(error)
                .with_context(format!("inspecting the session log {}", path.display()))
        })? {
            tracing::warn!(
                target: TARGET,
                path = %path.display(),
                "the session log ends mid-line; appending a newline before the new records"
            );
            buffer.insert(0, '\n');
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|error| {
                Error::external(error)
                    .with_context(format!("opening the session log {}", path.display()))
            })?;
        file.write_all(buffer.as_bytes()).await.map_err(|error| {
            Error::external(error)
                .with_context(format!("appending to the session log {}", path.display()))
        })?;
        file.flush().await.map_err(|error| {
            Error::external(error)
                .with_context(format!("flushing the session log {}", path.display()))
        })?;
        Ok(path)
    }

    /// Lists every `*.jsonl` file under [`SessionStore::dir`] recursively, sorted by path.
    ///
    /// A missing sessions directory is an empty list, not an error. The sort is by path
    /// components, so discovery order never leaks into a caller's output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] carrying the directory path when an existing directory or
    /// entry cannot be read.
    pub async fn files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        let mut pending = vec![self.dir()];
        while let Some(directory) = pending.pop() {
            let mut entries = match tokio::fs::read_dir(&directory).await {
                Ok(entries) => entries,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(Error::external(error).with_context(format!(
                        "listing the session directory {}",
                        directory.display()
                    )));
                }
            };
            while let Some(entry) = entries.next_entry().await.map_err(|error| {
                Error::external(error).with_context(format!(
                    "reading a session directory entry under {}",
                    directory.display()
                ))
            })? {
                let path = entry.path();
                let file_type = entry.file_type().await.map_err(|error| {
                    Error::external(error)
                        .with_context(format!("inspecting the session path {}", path.display()))
                })?;
                if file_type.is_dir() {
                    pending.push(path);
                } else if is_session_log(&path) {
                    files.push(path);
                }
            }
        }
        files.sort();
        Ok(files)
    }

    /// Loads the records of one session log, tolerantly.
    ///
    /// Blank lines are skipped and a malformed line becomes its own plain-text record, so the
    /// result carries one entry per non-blank line of the file. `path` is used as given; the
    /// paths returned by [`SessionStore::files`] and [`SessionStore::path_for`] are the usual
    /// inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] carrying the path when the file cannot be read as UTF-8
    /// text.
    pub async fn load(&self, path: &Path) -> Result<Vec<SessionRecord>> {
        let numbered = self.read_numbered(path).await?;
        Ok(numbered.into_iter().map(|(_, record)| record).collect())
    }

    /// Decodes every log under [`SessionStore::dir`] into chunks, in `(path, line)` order.
    ///
    /// The path of a chunk is relative to [`SessionStore::root`] and separated by `/`.
    ///
    /// # Errors
    ///
    /// Returns the [`SessionStore::files`] error or the [`SessionStore::load`] error of the
    /// first unreadable file.
    pub async fn chunks(&self) -> Result<Vec<SessionChunk>> {
        let mut chunks = Vec::new();
        for path in self.files().await? {
            let relative = self.relative(&path);
            for (line, record) in self.read_numbered(&path).await? {
                chunks.push(SessionChunk {
                    path: relative.clone(),
                    line,
                    record,
                });
            }
        }
        Ok(chunks)
    }

    /// Stamps the session directory: its file count, total length, and newest mtime.
    ///
    /// A missing directory stamps as zero files, zero bytes, and mtime `0`.
    ///
    /// # Errors
    ///
    /// Returns the [`SessionStore::files`] error and [`Error::External`] carrying the path
    /// when a listed file cannot be inspected.
    pub async fn stamp(&self) -> Result<DirStamp> {
        let mut stamp = DirStamp {
            files: 0,
            total_len: 0,
            newest_ms: 0,
        };
        for path in self.files().await? {
            let metadata = tokio::fs::metadata(&path).await.map_err(|error| {
                Error::external(error)
                    .with_context(format!("stamping the session log {}", path.display()))
            })?;
            stamp.files += 1;
            stamp.total_len = stamp.total_len.saturating_add(metadata.len());
            let newest = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |elapsed| elapsed.as_millis());
            stamp.newest_ms = stamp.newest_ms.max(newest);
        }
        Ok(stamp)
    }

    /// Reads `path` and returns its records with 1-based line numbers.
    async fn read_numbered(&self, path: &Path) -> Result<Vec<(usize, SessionRecord)>> {
        let text = tokio::fs::read_to_string(path).await.map_err(|error| {
            Error::external(error)
                .with_context(format!("reading the session log {}", path.display()))
        })?;
        Ok(decode_lines(&text))
    }

    /// Renders `path` relative to the store root with `/` separators.
    fn relative(&self, path: &Path) -> String {
        let relative = path.strip_prefix(&self.root).unwrap_or(path);
        relative
            .components()
            .filter_map(|component| match component {
                Component::CurDir | Component::RootDir | Component::Prefix(_) => None,
                Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
                Component::ParentDir => Some("..".to_owned()),
            })
            .collect::<Vec<_>>()
            .join("/")
    }
}

/// Sanitizes a session id into a safe single path component.
///
/// The id is trimmed; ASCII alphanumerics, `-`, `_`, and `.` are kept, and every other
/// character (including `/`, `\`, whitespace, and any non-ASCII scalar) folds to `-`.
/// Traversal is therefore impossible by construction: no separator survives, and callers
/// append `.jsonl` when building the file name, so the result can never be a `.` or `..`
/// path component.
///
/// # Identity
///
/// Folding is not injective: `café`, `caf-`, and `caf_` would all name the same log, which
/// would mix their turns and make every [`SessionHit`] cite the wrong session. A *portable*
/// id — one the fold leaves unchanged — is therefore used verbatim (so an existing log keeps
/// its file name), and any other id carries a `-<8 hex digit>` suffix of a hash over the
/// exact id. Distinct ids name distinct logs, and the suffix is stable across processes and
/// runs.
///
/// # Errors
///
/// Returns [`Error::Validation`] when nothing is left (empty or whitespace-only id).
pub fn sanitize_session_id(session_id: &str) -> Result<String> {
    let trimmed = session_id.trim();
    let mut sanitized = String::with_capacity(trimmed.len());
    for character in trimmed.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
            sanitized.push(character);
        } else {
            sanitized.push('-');
        }
    }
    if sanitized.is_empty() {
        return Err(Error::validation(format!(
            "session id `{session_id}` is empty after sanitizing"
        )));
    }
    if sanitized != trimmed {
        // The folded text cannot be told apart from another id folding to the same text, so
        // the exact id contributes a discriminator.
        sanitized.push('-');
        sanitized.push_str(&discriminator(trimmed));
    }
    Ok(sanitized)
}

/// Eight hex digits of an FNV-1a hash over `id`, used to keep folded ids distinct.
fn discriminator(id: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:08x}", (hash ^ (hash >> 32)) as u32)
}

/// Returns `true` when `path` exists, holds at least one byte, and its last byte is not a
/// newline.
///
/// A missing (or empty) file does not end mid-line: the first append starts a fresh line.
async fn ends_mid_line(path: &Path) -> std::io::Result<bool> {
    let mut file = match tokio::fs::File::open(path).await {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    let length = file.metadata().await?.len();
    if length == 0 {
        return Ok(false);
    }
    file.seek(std::io::SeekFrom::Start(length - 1)).await?;
    let mut last = [0_u8; 1];
    file.read_exact(&mut last).await?;
    Ok(last[0] != b'\n')
}

/// Returns `true` when `path` carries the session-log extension (`jsonl`).
fn is_session_log(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == OsStr::new("jsonl"))
}

/// Decodes `text` into records with 1-based line numbers, skipping blank lines.
fn decode_lines(text: &str) -> Vec<(usize, SessionRecord)> {
    text.lines()
        .enumerate()
        .filter_map(|(index, line)| {
            SessionRecord::parse_line(line).map(|record| (index + 1, record))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use tempfile::TempDir;

    use openjiuwen_common::error::Error as MemoryError;

    use super::*;

    /// The test day used by the layout and append tests.
    fn day() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, 14).unwrap()
    }

    /// Builds a store rooted at a fresh temporary directory.
    fn store() -> (TempDir, SessionStore) {
        let temp = TempDir::new().unwrap();
        let store = SessionStore::new(temp.path());
        (temp, store)
    }

    #[test]
    fn portable_ids_are_kept_and_folded_ids_carry_a_discriminator() {
        assert_eq!(sanitize_session_id("abc-1_2.3").unwrap(), "abc-1_2.3");
        assert_eq!(sanitize_session_id("  abc  ").unwrap(), "abc");
        assert_eq!(sanitize_session_id("..").unwrap(), "..");
        for id in ["a/b", "../../etc/passwd", "a\\b", "sess 1", "café", "日本"] {
            let sanitized = sanitize_session_id(id).unwrap();
            let (stem, suffix) = sanitized.rsplit_once('-').expect("a discriminator");
            assert!(
                stem.chars()
                    .all(|character| character.is_ascii_alphanumeric()
                        || matches!(character, '-' | '_' | '.')),
                "{id:?} folded to {sanitized:?}"
            );
            assert_eq!(suffix.len(), 8, "{id:?} folded to {sanitized:?}");
            assert!(
                suffix
                    .chars()
                    .all(|character| character.is_ascii_hexdigit()),
                "{id:?} folded to {sanitized:?}"
            );
            assert_ne!(sanitized, id, "a folded id must not be used verbatim");
        }
        assert_eq!(sanitize_session_id("a/b").unwrap().len(), "a-b-".len() + 8);
        // The same id always folds to the same name, so a log stays addressable.
        assert_eq!(
            sanitize_session_id("café").unwrap(),
            sanitize_session_id("café").unwrap()
        );
    }

    #[test]
    fn ids_that_fold_alike_still_name_different_logs() {
        let mut seen = std::collections::BTreeSet::new();
        for id in ["café", "caf-", "caf_", "caf ", "cafe", "日本", "--"] {
            let sanitized = sanitize_session_id(id).unwrap();
            assert!(
                seen.insert(sanitized.clone()),
                "{id:?} collided: {sanitized}"
            );
        }
        assert_eq!(sanitize_session_id("caf-").unwrap(), "caf-");
    }

    #[test]
    fn rejects_an_id_that_sanitizes_to_nothing() {
        for id in ["", "   ", "\t\n"] {
            let error = sanitize_session_id(id).unwrap_err();
            assert!(
                matches!(error, MemoryError::Validation(_)),
                "{id:?}: {error:?}"
            );
        }
    }

    #[test]
    fn sanitized_ids_cannot_escape_the_sessions_directory() {
        let (temp, store) = store();
        for id in ["../../etc/passwd", "..", ".", "a/b/../../c"] {
            let path = store.path_for(id, day()).unwrap();
            assert!(
                path.starts_with(store.dir()),
                "{id:?} escaped to {}",
                path.display()
            );
            assert!(path.starts_with(temp.path()), "{id:?}");
            assert_eq!(
                path.parent(),
                Some(store.dir().join("2026-11-14").as_path())
            );
        }
    }

    #[test]
    fn path_for_uses_the_day_and_the_sanitized_id() {
        let (_temp, store) = store();
        assert_eq!(store.dir(), store.root().join(SESSIONS_DIR));
        let path = store.path_for("abc", day()).unwrap();
        assert_eq!(path, store.root().join("sessions/2026-11-14/abc.jsonl"));
        let sanitized = store.path_for("a/b", day()).unwrap();
        let id = sanitize_session_id("a/b").unwrap();
        assert_eq!(
            sanitized,
            store.root().join(format!("sessions/2026-11-14/{id}.jsonl"))
        );
        assert_eq!(
            sanitized.parent(),
            store.path_for("abc", day()).unwrap().parent()
        );
    }

    #[tokio::test]
    async fn append_writes_one_line_per_record_and_accumulates() {
        let (_temp, store) = store();
        let path = store
            .append(
                "abc",
                day(),
                &[
                    SessionRecord::new("first"),
                    SessionRecord::new("second").with_role("user"),
                ],
            )
            .await
            .unwrap();
        assert_eq!(path, store.root().join("sessions/2026-11-14/abc.jsonl"));
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(text.lines().count(), 2);
        assert_eq!(
            store.load(&path).await.unwrap(),
            vec![
                SessionRecord::new("first"),
                SessionRecord::new("second").with_role("user"),
            ]
        );

        let again = store
            .append("abc", day(), &[SessionRecord::new("third")])
            .await
            .unwrap();
        assert_eq!(again, path);
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(text.lines().count(), 3);
        let records = store.load(&path).await.unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].content, "first");
        assert_eq!(records[1].content, "second");
        assert_eq!(records[2].content, "third");
    }

    #[tokio::test]
    async fn append_skips_empty_records_but_rejects_an_all_empty_set() {
        let (_temp, store) = store();
        let path = store
            .append(
                "abc",
                day(),
                &[SessionRecord::new("kept"), SessionRecord::new("   ")],
            )
            .await
            .unwrap();
        assert_eq!(
            store.load(&path).await.unwrap(),
            vec![SessionRecord::new("kept")]
        );

        let error = store
            .append(
                "abc",
                day(),
                &[SessionRecord::new(""), SessionRecord::new("\n\t")],
            )
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::Validation(_)), "{error:?}");
        assert_eq!(store.load(&path).await.unwrap().len(), 1);

        let error = store
            .append("", day(), &[SessionRecord::new("x")])
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn append_round_trips_timestamps_and_escapes() {
        let (_temp, store) = store();
        let record = SessionRecord::new("line one\nline \"two\"")
            .with_role("assistant")
            .with_timestamp(Utc.with_ymd_and_hms(2026, 11, 14, 9, 5, 0).unwrap());
        let path = store
            .append("abc", day(), std::slice::from_ref(&record))
            .await
            .unwrap();
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(text.lines().count(), 1, "escaped newline stays one line");
        assert_eq!(store.load(&path).await.unwrap(), vec![record]);
    }

    #[tokio::test]
    async fn ids_that_fold_alike_keep_themselves_apart_on_disk() {
        let (_temp, store) = store();
        let first = store
            .append("café", day(), &[SessionRecord::new("from cafe")])
            .await
            .unwrap();
        let second = store
            .append("caf-", day(), &[SessionRecord::new("from caf ")])
            .await
            .unwrap();
        assert_ne!(first, second, "two ids must not share one log");
        assert_eq!(store.load(&first).await.unwrap()[0].content, "from cafe");
        assert_eq!(store.load(&second).await.unwrap()[0].content, "from caf ");

        let chunks = store.chunks().await.unwrap();
        assert_eq!(chunks.len(), 2);
        let cafe = chunks
            .iter()
            .find(|chunk| chunk.record.content == "from cafe")
            .expect("the café turn is indexed");
        assert_eq!(
            cafe.path,
            format!(
                "sessions/2026-11-14/{}.jsonl",
                sanitize_session_id("café").unwrap()
            ),
            "the hit cites the log it came from"
        );
        let plain = chunks
            .iter()
            .find(|chunk| chunk.record.content == "from caf ")
            .expect("the portable turn is indexed");
        assert_eq!(plain.path, "sessions/2026-11-14/caf-.jsonl");
    }

    #[tokio::test]
    async fn append_repairs_a_log_that_ends_mid_line() {
        let (_temp, store) = store();
        let path = store.path_for("abc", day()).unwrap();
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        // A truncated write leaves a partial JSON line with no trailing newline.
        tokio::fs::write(&path, "{\"content\":\"trunc")
            .await
            .unwrap();

        store
            .append("abc", day(), &[SessionRecord::new("after the truncation")])
            .await
            .unwrap();
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(
            text.starts_with("{\"content\":\"trunc\n"),
            "the truncated tail is preserved and terminated: {text}"
        );
        assert_eq!(text.lines().count(), 2, "{text}");
        let records = store.load(&path).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].content, "{\"content\":\"trunc");
        assert_eq!(records[1].content, "after the truncation");

        // A log that already ends with a newline gains no blank line.
        store
            .append("abc", day(), &[SessionRecord::new("and one more")])
            .await
            .unwrap();
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(text.lines().count(), 3, "{text}");
        assert!(!text.contains("\n\n"), "{text}");
    }

    #[tokio::test]
    async fn files_are_recursive_and_sorted() {
        let (_temp, store) = store();
        assert!(store.files().await.unwrap().is_empty(), "absent directory");

        let nested = store.dir().join("2026-11-14").join("nested");
        tokio::fs::create_dir_all(&nested).await.unwrap();
        let day_dir = store.dir().join("2026-11-13");
        tokio::fs::create_dir_all(&day_dir).await.unwrap();
        for path in [
            store.dir().join("2026-11-14").join("b.jsonl"),
            nested.join("c.jsonl"),
            day_dir.join("a.jsonl"),
            store.dir().join("notes.txt"),
        ] {
            tokio::fs::write(&path, "{\"content\":\"x\"}\n")
                .await
                .unwrap();
        }

        let files = store.files().await.unwrap();
        assert_eq!(
            files,
            vec![
                store.dir().join("2026-11-13/a.jsonl"),
                store.dir().join("2026-11-14/b.jsonl"),
                nested.join("c.jsonl"),
            ]
        );
        let mut sorted = files.clone();
        sorted.sort();
        assert_eq!(files, sorted);
    }

    #[tokio::test]
    async fn load_mixes_json_malformed_lines_and_blanks() {
        let (_temp, store) = store();
        let path = store.path_for("mixed", day()).unwrap();
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        let text = concat!(
            "{\"content\":\"hello\",\"role\":\"user\"}\n",
            "\n",
            "{\"content\":\"truncated\"\n",
            "plain text turn\n",
            "   \n",
            "{\"content\":42}\n",
            "42\n",
        );
        tokio::fs::write(&path, text).await.unwrap();

        let records = store.load(&path).await.unwrap();
        assert_eq!(records.len(), 5);
        assert_eq!(records[0], SessionRecord::new("hello").with_role("user"));
        assert_eq!(records[1].content, "{\"content\":\"truncated\"");
        assert_eq!(records[2].content, "plain text turn");
        assert_eq!(records[3].content, "{\"content\":42}");
        assert_eq!(records[4].content, "42");
    }

    #[tokio::test]
    async fn load_of_a_missing_file_is_external_with_the_path() {
        let (_temp, store) = store();
        let path = store.dir().join("ghost.jsonl");
        let error = store.load(&path).await.unwrap_err();
        assert!(matches!(error, MemoryError::External(_)), "{error:?}");
        assert!(error.to_string().contains("ghost.jsonl"), "{error}");
    }

    #[tokio::test]
    async fn chunks_number_lines_and_name_paths_relative_to_the_root() {
        let (_temp, store) = store();
        let first = store.path_for("one", day()).unwrap();
        tokio::fs::create_dir_all(first.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(
            &first,
            "{\"content\":\"alpha\"}\n\n{\"content\":\"beta\",\"role\":\"user\"}\n",
        )
        .await
        .unwrap();
        let nested = store.dir().join("2026-11-13").join("deep");
        tokio::fs::create_dir_all(&nested).await.unwrap();
        tokio::fs::write(nested.join("two.jsonl"), "gamma\n")
            .await
            .unwrap();

        let chunks = store.chunks().await.unwrap();
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| (
                    chunk.path.as_str(),
                    chunk.line,
                    chunk.record.content.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("sessions/2026-11-13/deep/two.jsonl", 1, "gamma"),
                ("sessions/2026-11-14/one.jsonl", 1, "alpha"),
                ("sessions/2026-11-14/one.jsonl", 3, "beta"),
            ]
        );
        let hits = rank_chunks(&chunks, "beta", 0.5, 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 3);
        assert_eq!(hits[0].path, "sessions/2026-11-14/one.jsonl");
    }

    #[tokio::test]
    async fn stamp_counts_files_lengths_and_newest_mtime() {
        let (_temp, store) = store();
        let empty = store.stamp().await.unwrap();
        assert_eq!(empty.files, 0);
        assert_eq!(empty.total_len, 0);
        assert_eq!(empty.newest_ms, 0);

        let first = store
            .append("one", day(), &[SessionRecord::new("alpha")])
            .await
            .unwrap();
        let second = store
            .append(
                "two",
                day(),
                &[SessionRecord::new("beta"), SessionRecord::new("gamma")],
            )
            .await
            .unwrap();
        let expected = tokio::fs::metadata(&first).await.unwrap().len()
            + tokio::fs::metadata(&second).await.unwrap().len();
        let stamp = store.stamp().await.unwrap();
        assert_eq!(stamp.files, 2);
        assert_eq!(stamp.total_len, expected);
        assert!(stamp.newest_ms > 0, "{stamp:?}");
    }

    #[tokio::test]
    async fn appends_remain_visible_across_handles() {
        let (_temp, store) = store();
        let _ = store
            .append("abc", day(), &[SessionRecord::new("first")])
            .await
            .unwrap();
        let other = SessionStore::new(store.root());
        let _ = other
            .append("abc", day(), &[SessionRecord::new("second")])
            .await
            .unwrap();
        let records = store
            .load(&store.path_for("abc", day()).unwrap())
            .await
            .unwrap();
        assert_eq!(records.len(), 2);
    }
}
