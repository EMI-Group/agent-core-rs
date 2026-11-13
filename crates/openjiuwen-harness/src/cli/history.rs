//! Persistent REPL input history stored under the CLI home directory.
//!
//! The history file is plain UTF-8 with one submitted input line per entry and
//! no metadata, which is the shape the Python CLI writes to
//! `~/.openjiuwen_history`. The file is capped at [`MAX_HISTORY_ENTRIES`]
//! entries (oldest dropped first) by [`History`], so a long-lived CLI never
//! grows an unbounded state file.
//!
//! The module is deliberately forgiving: a missing, unreadable or corrupt
//! history file never blocks the CLI. [`History::load`] degrades to an empty
//! history with a `tracing` warning, invalid UTF-8 is decoded lossily and junk
//! lines are skipped, while write failures are reported to the caller because a
//! lost write is worth telling the user about.
//!
//! [`History::new`] uses the [`MAX_HISTORY_ENTRIES`] default; tests and embedders
//! can pick a smaller window with [`History::with_capacity`].

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use super::paths::MAX_HISTORY_ENTRIES;

/// Persistent REPL input history with a retained-entry cap.
///
/// Entries are held newest-last, exactly as they appear in the file on disk.
///
/// # File format
///
/// One trimmed entry per line, separated by `\n`, UTF-8 encoded. Blank lines
/// carry no entry and are skipped when the file is read.
#[derive(Debug, Clone)]
pub struct History {
    /// Location of the history file.
    path: PathBuf,
    /// Retained entries, oldest first.
    entries: Vec<String>,
    /// Maximum number of retained entries.
    capacity: usize,
}

impl History {
    /// Creates an empty in-memory history for `path` capped at
    /// [`MAX_HISTORY_ENTRIES`].
    ///
    /// Nothing is read here: call [`History::load`] to populate the entries.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with_capacity(path, MAX_HISTORY_ENTRIES)
    }

    /// Creates an empty in-memory history for `path` capped at `capacity`.
    #[must_use]
    pub fn with_capacity(path: impl Into<PathBuf>, capacity: usize) -> Self {
        Self {
            path: path.into(),
            entries: Vec::new(),
            capacity,
        }
    }

    /// Returns the history file location.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the maximum number of retained entries.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the retained entries, oldest first.
    #[must_use]
    pub fn entries(&self) -> &[String] {
        &self.entries
    }

    /// Returns the number of retained entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether no entry is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Loads the file into memory; returns the number of retained entries.
    ///
    /// A missing file yields zero entries and `Ok(0)`. The bytes are decoded
    /// lossily (invalid UTF-8 becomes `U+FFFD`), each line is trimmed, and blank
    /// lines are skipped, so a hand-edited or binary-corrupted file still loads.
    ///
    /// Only the last [`History::capacity`] entries are retained: a longer file is
    /// trimmed in memory and left untouched on disk until the next
    /// [`History::record`] that is retained, at which point the file is rewritten
    /// with exactly the retained window. Any other read error is logged with
    /// `tracing::warn!` and treated as an empty history, because a corrupt or
    /// unreadable history file must never block the CLI.
    ///
    /// # Errors
    ///
    /// Currently never fails; the `Result` keeps the door open for future
    /// validation and matches the write side.
    pub fn load(&mut self) -> Result<usize> {
        self.entries.clear();
        let bytes = match std::fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => {
                tracing::warn!(
                    path = %self.path.display(),
                    %error,
                    "failed to read the CLI history file; starting with an empty history"
                );
                return Ok(0);
            }
        };
        let text = String::from_utf8_lossy(&bytes);
        let mut loaded: Vec<String> = text
            .split('\n')
            .map(|line| line.trim().to_owned())
            .filter(|line| !line.is_empty())
            .collect();
        if loaded.len() > self.capacity {
            let excess = loaded.len() - self.capacity;
            loaded.drain(..excess);
        }
        self.entries = loaded;
        Ok(self.entries.len())
    }

    /// Records one submitted line; returns whether it was retained.
    ///
    /// The line is trimmed first. A blank line (nothing to remember) and a line
    /// identical to the most recent retained entry (no repeated `Up` entry) are
    /// both skipped and write nothing, returning `Ok(false)`.
    ///
    /// Otherwise the entry is pushed and persisted: while the retained window is
    /// below the cap the single line is appended to the file (creating the parent
    /// directory and the file when needed); once the window is at the cap the file
    /// is rewritten atomically (`<path>.tmp` + rename) with exactly the newest
    /// [`History::capacity`] entries. The rewrite is what trims a file that
    /// [`History::load`] had to shorten.
    ///
    /// A history with zero capacity retains nothing and never touches the disk,
    /// returning `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] naming the path when the parent directory cannot
    /// be created or the file cannot be written or replaced.
    pub fn record(&mut self, line: &str) -> Result<bool> {
        let entry = line.trim();
        if entry.is_empty() || self.capacity == 0 {
            return Ok(false);
        }
        if self.entries.last().is_some_and(|last| last == entry) {
            return Ok(false);
        }
        self.entries.push(entry.to_owned());
        if self.entries.len() > self.capacity {
            let excess = self.entries.len() - self.capacity;
            self.entries.drain(..excess);
        }
        if self.entries.len() < self.capacity {
            self.append_line(entry)?;
        } else {
            self.rewrite()?;
        }
        Ok(true)
    }

    /// Clears the retained entries and rewrites an empty file.
    ///
    /// The file itself is kept (created when missing) so a later
    /// [`History::record`] can append to it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] naming the path when the parent directory cannot
    /// be created or the empty file cannot be written.
    pub fn clear(&mut self) -> Result<()> {
        self.entries.clear();
        self.rewrite()
    }

    /// Appends one entry to the file, adding a separator when it lacks a final
    /// newline.
    fn append_line(&self, entry: &str) -> Result<()> {
        self.ensure_parent()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&self.path)
            .map_err(|error| self.write_error("open for append", &error))?;
        let length = file
            .metadata()
            .map_err(|error| self.write_error("inspect", &error))?
            .len();
        let mut payload = String::new();
        if length > 0 && !ends_with_newline(&mut file, length)? {
            payload.push('\n');
        }
        payload.push_str(entry);
        payload.push('\n');
        file.write_all(payload.as_bytes())
            .map_err(|error| self.write_error("append to", &error))
    }

    /// Rewrites the whole file atomically with exactly the retained entries.
    fn rewrite(&self) -> Result<()> {
        self.ensure_parent()?;
        let mut body = String::new();
        for entry in &self.entries {
            body.push_str(entry);
            body.push('\n');
        }
        let temp = temp_path(&self.path);
        std::fs::write(&temp, body).map_err(|error| self.write_error("write", &error))?;
        std::fs::rename(&temp, &self.path).map_err(|error| self.write_error("replace", &error))
    }

    /// Creates the history file's parent directory when it is missing.
    fn ensure_parent(&self) -> Result<()> {
        let Some(dir) = self.path.parent() else {
            return Ok(());
        };
        if dir.as_os_str().is_empty() {
            return Ok(());
        }
        std::fs::create_dir_all(dir).map_err(|error| {
            Error::sys_op(format!(
                "failed to create history directory {}: {error}",
                dir.display()
            ))
        })
    }

    /// Builds the error reported when a file operation on the history fails.
    fn write_error(&self, action: &str, error: &std::io::Error) -> Error {
        Error::sys_op(format!(
            "failed to {action} history file {}: {error}",
            self.path.display()
        ))
    }
}

/// Returns whether the history file currently ends with a newline.
///
/// `file` is opened for reading and appending; a zero length file is handled by
/// the caller, so `length` is always at least one here.
fn ends_with_newline(file: &mut std::fs::File, length: u64) -> Result<bool> {
    file.seek(SeekFrom::Start(length - 1))
        .map_err(|error| Error::sys_op(format!("failed to seek in history file: {error}")))?;
    let mut last = [0_u8; 1];
    file.read_exact(&mut last)
        .map_err(|error| Error::sys_op(format!("failed to read history file: {error}")))?;
    Ok(last[0] == b'\n')
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

    /// Returns the history file of a fresh temporary directory.
    fn history_file(temp: &TempDir) -> PathBuf {
        temp.path().join("history.txt")
    }

    #[test]
    fn record_then_load_round_trips_entries() {
        let temp = TempDir::new().unwrap();
        let path = history_file(&temp);
        let mut history = History::new(&path);
        assert_eq!(history.path(), path.as_path());
        assert_eq!(history.capacity(), MAX_HISTORY_ENTRIES);
        assert!(history.is_empty());

        assert!(history.record("  first  ").unwrap());
        assert!(history.record("second").unwrap());
        assert_eq!(history.len(), 2);
        assert_eq!(history.entries(), ["first".to_owned(), "second".to_owned()]);

        let mut reloaded = History::new(&path);
        assert_eq!(reloaded.load().unwrap(), 2);
        assert_eq!(reloaded.entries(), history.entries());
    }

    #[test]
    fn capacity_keeps_only_the_newest_entries_in_memory_and_on_disk() {
        let temp = TempDir::new().unwrap();
        let path = history_file(&temp);
        let mut history = History::with_capacity(&path, 3);
        assert_eq!(history.capacity(), 3);

        for line in ["one", "two", "three", "four", "five", "six"] {
            assert!(history.record(line).unwrap(), "{line} should be retained");
        }

        assert_eq!(history.len(), 3);
        assert_eq!(
            history.entries(),
            ["four".to_owned(), "five".to_owned(), "six".to_owned()]
        );
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk, "four\nfive\nsix\n");

        let mut reloaded = History::with_capacity(&path, 3);
        assert_eq!(reloaded.load().unwrap(), 3);
        assert_eq!(reloaded.entries(), history.entries());
    }

    #[test]
    fn blank_and_duplicate_lines_are_not_retained() {
        let temp = TempDir::new().unwrap();
        let path = history_file(&temp);
        let mut history = History::new(&path);

        assert!(!history.record("   ").unwrap());
        assert!(!history.record("").unwrap());
        assert!(history.is_empty());
        assert!(!path.exists());

        assert!(history.record("repeat").unwrap());
        assert!(!history.record("repeat").unwrap());
        assert!(!history.record("  repeat  ").unwrap());
        assert!(history.record("other").unwrap());
        assert!(history.record("repeat").unwrap());

        assert_eq!(
            history.entries(),
            ["repeat".to_owned(), "other".to_owned(), "repeat".to_owned()]
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 3);
    }

    #[test]
    fn zero_capacity_retains_nothing() {
        let temp = TempDir::new().unwrap();
        let path = history_file(&temp);
        let mut history = History::with_capacity(&path, 0);
        assert!(!history.record("ignored").unwrap());
        assert!(history.is_empty());
        assert!(!path.exists());
    }

    #[test]
    fn load_tolerates_invalid_utf8_and_junk_lines() {
        let temp = TempDir::new().unwrap();
        let path = history_file(&temp);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"good line\r\n");
        bytes.extend_from_slice(b"\n");
        bytes.extend_from_slice(b"   \n");
        bytes.extend_from_slice(&[0xff, 0xfe, b'\n']);
        bytes.extend_from_slice(b"  padded  \n");
        std::fs::write(&path, bytes).unwrap();

        let mut history = History::new(&path);
        assert_eq!(history.load().unwrap(), 3);
        assert_eq!(history.entries()[0], "good line");
        assert!(history.entries()[1].contains('\u{fffd}'));
        assert_eq!(history.entries()[2], "padded");
    }

    #[test]
    fn load_trims_a_longer_file_and_rewrites_it_on_the_next_record() {
        let temp = TempDir::new().unwrap();
        let path = history_file(&temp);
        std::fs::write(&path, "a\nb\nc\nd\n").unwrap();

        let mut history = History::with_capacity(&path, 2);
        assert_eq!(history.load().unwrap(), 2);
        assert_eq!(history.entries(), ["c".to_owned(), "d".to_owned()]);
        // Loading alone leaves the oversized file untouched.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nb\nc\nd\n");

        assert!(history.record("e").unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "d\ne\n");
        assert!(!temp.path().join("history.txt.tmp").exists());
    }

    #[test]
    fn load_of_a_missing_file_yields_no_entries() {
        let temp = TempDir::new().unwrap();
        let mut history = History::new(history_file(&temp));
        assert_eq!(history.load().unwrap(), 0);
        assert!(history.is_empty());
    }

    #[test]
    fn clear_empties_memory_and_file() {
        let temp = TempDir::new().unwrap();
        let path = history_file(&temp);
        let mut history = History::new(&path);
        assert!(history.record("keep me").unwrap());
        assert_ne!(std::fs::read_to_string(&path).unwrap(), "");

        history.clear().unwrap();
        assert!(history.is_empty());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");

        let mut reloaded = History::new(&path);
        assert_eq!(reloaded.load().unwrap(), 0);
    }

    #[test]
    fn append_creates_the_parent_directory_and_separates_a_missing_newline() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("nested/deeper/history.txt");
        let mut history = History::new(&path);
        assert!(history.record("one").unwrap());
        assert!(path.exists());

        // A hand-edited file without a trailing newline must not merge entries.
        std::fs::write(&path, "one").unwrap();
        assert_eq!(history.load().unwrap(), 1);
        assert!(history.record("two").unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "one\ntwo\n");
    }
}
