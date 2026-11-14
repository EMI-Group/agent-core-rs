//! Append-only execution journal: JSONL segments, bounded retention and replay.
//!
//! One journal is one directory. `meta.json` describes it (schema version, encoding, the execution
//! it belongs to and the next archive segment number) and the entries themselves live in
//! `entries.jsonl`. Rotating renames the active file to `entries.<generation>.jsonl` and prunes the
//! oldest segments, so sequence numbers keep growing across segments while a segment is immutable
//! once it has been written.
//!
//! Reads tolerate a crash: an unparseable or empty line is skipped and counted, exact duplicates
//! are collapsed, and two entries that share a sequence number but disagree on content drop the
//! whole group and are reported as conflicting. The journal is therefore the durability boundary
//! of a persistent execution — a resume replays it instead of trusting an in-memory transcript.
//!
//! One `tokio::sync::Mutex` guards the whole write state, and reads take the same mutex, so a
//! rotation can never race an append or a read. Every timestamp comes from the injected
//! [`Clock`], which keeps the module deterministic and offline in tests.
// Size note: the implementation is ~875 lines; the file exceeds the crate's 1000-line guideline
// only because the inline `#[cfg(test)]` module below covers rotation, retention and crash
// recovery against real temporary directories.
use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use super::entry::{EntryKind, EntryPayload, ExecutionEntry, ToolStatus, UsageStatus};
use super::identity::ExecutionIdentity;
use super::util::{read_json, write_json_atomic};
use super::{Clock, SystemClock};
use crate::workspace::Workspace;

/// Name of the active segment of a journal directory.
pub const LOG_ENTRY_FILE: &str = "entries.jsonl";

/// Name of the journal metadata file.
pub const LOG_META_FILE: &str = "meta.json";

/// Encoding identifier written into [`LogMeta::encoding`] and expected on reopen.
pub const LOG_ENCODING: &str = "pa-jsonl/1";

/// Schema version of the journal metadata.
const LOG_META_VERSION: u32 = 1;

/// Prefix of an archived segment file name (`entries.<generation>.jsonl`).
const SEGMENT_PREFIX: &str = "entries.";

/// Suffix of an archived segment file name (`entries.<generation>.jsonl`).
const SEGMENT_SUFFIX: &str = ".jsonl";

/// Workspace node the journal directory lives under.
const PERSISTENT_AGENT_NODE: &str = "persistent_agent";

/// Subdirectory of the workspace node that holds the per-execution journals.
const LOG_SUBDIR: &str = "logs";

/// Default number of entries kept by a journal.
const DEFAULT_MAX_ENTRIES: usize = 4096;

/// Default number of bytes kept in the active segment.
const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// Default number of archived segments kept by rotation.
const DEFAULT_KEEP_SEGMENTS: usize = 4;

/// Retention limits of one journal.
///
/// A limit of zero is rejected by [`LogCapacity::validate`], so an unbounded journal cannot be
/// constructed by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LogCapacity {
    /// Maximum number of entries a read retains (the newest ones win).
    pub max_entries: usize,
    /// Number of bytes after which the active segment is rotated before the next append.
    pub max_bytes: u64,
    /// Number of archived segments kept on disk after a rotation.
    pub keep_segments: usize,
}

impl Default for LogCapacity {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
            keep_segments: DEFAULT_KEEP_SEGMENTS,
        }
    }
}

impl LogCapacity {
    /// Rejects a zero limit.
    ///
    /// # Errors
    /// `Error::Validation` naming the first zero field.
    pub fn validate(&self) -> Result<()> {
        if self.max_entries == 0 {
            return Err(Error::validation("max_entries must be a positive integer"));
        }
        if self.max_bytes == 0 {
            return Err(Error::validation("max_bytes must be a positive integer"));
        }
        if self.keep_segments == 0 {
            return Err(Error::validation(
                "keep_segments must be a positive integer",
            ));
        }
        Ok(())
    }
}

/// The persisted description of one journal directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogMeta {
    /// Schema version of the metadata record.
    pub schema_version: u32,
    /// Encoding of the entry lines (see [`LOG_ENCODING`]).
    pub encoding: String,
    /// Execution the journal belongs to.
    pub identity: ExecutionIdentity,
    /// RFC 3339 creation timestamp of the journal.
    pub created_at: String,
    /// Next archive segment number; starts at `1`.
    pub generation: u64,
}

/// Position of a journal: the next sequence number to assign and the next segment number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogCursor {
    /// Sequence number the next append will use.
    pub next_seq: u64,
    /// Sequence number of the newest entry (`0` for an empty journal).
    pub last_seq: u64,
    /// Next archive segment number.
    pub generation: u64,
}

/// Counts and totals of a sequence of entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogDigest {
    /// Number of entries counted.
    pub entries: usize,
    /// Sequence number of the first entry.
    pub first_seq: Option<u64>,
    /// Sequence number of the last entry.
    pub last_seq: Option<u64>,
    /// Entry count per kind, ordered by the [`EntryKind`] declaration order.
    ///
    /// The map is keyed by the kind's snake-case spelling because [`ToolStatus`] and its siblings
    /// are not `Ord`; a `String` key keeps the digest serde-serializable and deterministic.
    pub counts: BTreeMap<EntryKind, usize>,
    /// Completed-tool counts per outcome, keyed by the status's snake-case spelling.
    pub tool_outcomes: BTreeMap<String, usize>,
    /// Sum of the input-token counters of the model turns.
    pub input_tokens: u64,
    /// Sum of the output-token counters of the model turns.
    pub output_tokens: u64,
    /// Number of model turns whose usage is not fully known.
    pub unknown_usage: usize,
    /// Sequence ranges no usable entry covers, in ascending order.
    pub missing_ranges: Vec<(u64, u64)>,
    /// Lines that were empty or unparseable.
    pub skipped_invalid: usize,
    /// Exact duplicates that were collapsed.
    pub duplicates_removed: usize,
}

impl LogDigest {
    /// Renders the digest as one deterministic line.
    ///
    /// The shape is fixed (`entries`, `seqs`, `kinds`, `tools`, `tokens`) and the optional
    /// trailing sections appear only when they are non-zero, so two digests of the same entries
    /// always render identically.
    #[must_use]
    pub fn summary(&self) -> String {
        let seqs = match (self.first_seq, self.last_seq) {
            (Some(first), Some(last)) => format!("{first}..{last}"),
            _ => "none".to_string(),
        };
        let kinds = render_counts(
            self.counts
                .iter()
                .map(|(kind, count)| (kind.as_str(), *count)),
        );
        let tools = render_counts(
            self.tool_outcomes
                .iter()
                .map(|(status, count)| (status.as_str(), *count)),
        );
        let mut summary = format!(
            "entries={} seqs={seqs} kinds=[{kinds}] tools=[{tools}] tokens=in:{},out:{},unknown:{}",
            self.entries, self.input_tokens, self.output_tokens, self.unknown_usage
        );
        if !self.missing_ranges.is_empty() {
            let ranges = self
                .missing_ranges
                .iter()
                .map(|(start, end)| format!("{start}..{end}"))
                .collect::<Vec<_>>()
                .join(",");
            let _ = write!(summary, " missing=[{ranges}]");
        }
        if self.skipped_invalid > 0 {
            let _ = write!(summary, " skipped_invalid={}", self.skipped_invalid);
        }
        if self.duplicates_removed > 0 {
            let _ = write!(summary, " duplicates_removed={}", self.duplicates_removed);
        }
        summary
    }
}

/// Renders `(name, count)` pairs as `name=count` joined by commas.
fn render_counts<'a>(counts: impl Iterator<Item = (&'a str, usize)>) -> String {
    counts
        .map(|(name, count)| format!("{name}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// One read of one journal: the usable entries in sequence order and what was left out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogSnapshot {
    /// Execution the journal belongs to.
    pub identity: ExecutionIdentity,
    /// Usable entries, sorted by sequence number and trimmed to the newest `max_entries`.
    pub entries: Vec<ExecutionEntry>,
    /// Lines that were empty or unparseable.
    pub skipped_invalid: usize,
    /// Exact duplicates that were collapsed.
    pub duplicates_removed: usize,
    /// Sequence numbers whose entries disagreed with each other; those groups were dropped.
    pub conflicting_seqs: Vec<u64>,
    /// Sequence ranges no usable entry covers, computed before retention trimming.
    pub missing_ranges: Vec<(u64, u64)>,
    /// Number of usable entries dropped by retention.
    pub retained_out: usize,
    /// Why nothing can be used (`empty`, `no_usable_entry`), or `None` once one entry is usable.
    pub unusable: Option<String>,
}

impl LogSnapshot {
    /// Returns whether at least one entry survived the read.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.unusable.is_none() && !self.entries.is_empty()
    }

    /// Returns the digest of this snapshot's entries and counters.
    #[must_use]
    pub fn digest(&self) -> LogDigest {
        let mut counts: BTreeMap<EntryKind, usize> = BTreeMap::new();
        let mut tool_outcomes: BTreeMap<String, usize> = BTreeMap::new();
        let mut input_tokens = 0_u64;
        let mut output_tokens = 0_u64;
        let mut unknown_usage = 0_usize;
        for entry in &self.entries {
            *counts.entry(entry.kind()).or_insert(0) += 1;
            match &entry.payload {
                EntryPayload::ModelTurn { usage, .. } => {
                    input_tokens = input_tokens.saturating_add(usage.input_tokens.unwrap_or(0));
                    output_tokens = output_tokens.saturating_add(usage.output_tokens.unwrap_or(0));
                    let fully_known =
                        usage.status == UsageStatus::Known && usage.total_tokens().is_some();
                    if !fully_known {
                        unknown_usage += 1;
                    }
                }
                EntryPayload::ToolCompleted { status, .. } => {
                    *tool_outcomes
                        .entry(tool_status_key(*status).to_string())
                        .or_insert(0) += 1;
                }
                _ => {}
            }
        }
        LogDigest {
            entries: self.entries.len(),
            first_seq: self.entries.first().map(|entry| entry.seq),
            last_seq: self.entries.last().map(|entry| entry.seq),
            counts,
            tool_outcomes,
            input_tokens,
            output_tokens,
            unknown_usage,
            missing_ranges: self.missing_ranges.clone(),
            skipped_invalid: self.skipped_invalid,
            duplicates_removed: self.duplicates_removed,
        }
    }

    /// Returns whether anything was left out of this snapshot.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        self.retained_out > 0 || self.skipped_invalid > 0 || !self.conflicting_seqs.is_empty()
    }

    /// Wraps this snapshot as a [`Replay`].
    ///
    /// The cursor continues after the newest usable entry. Its `generation` is `0` because a
    /// snapshot does not carry the journal's segment counter; use [`ExecutionLog::cursor`] for the
    /// live value.
    #[must_use]
    pub fn replay(&self) -> Replay {
        let last_seq = self.entries.last().map_or(0, |entry| entry.seq);
        Replay {
            cursor: LogCursor {
                next_seq: last_seq.saturating_add(1),
                last_seq,
                generation: 0,
            },
            entries: self.entries.clone(),
            digest: self.digest(),
            truncated: self.is_truncated(),
        }
    }
}

/// A snapshot plus where to continue reading it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    /// Where the next append or read continues.
    pub cursor: LogCursor,
    /// Usable entries in sequence order.
    pub entries: Vec<ExecutionEntry>,
    /// Counts and totals of [`Replay::entries`].
    pub digest: LogDigest,
    /// Whether anything was left out of the replay.
    pub truncated: bool,
}

/// The append-only execution journal of one execution.
///
/// The directory holds `meta.json`, the active `entries.jsonl` and zero or more immutable
/// `entries.<generation>.jsonl` archives. Sequence numbers are journal-wide and keep growing
/// across rotations.
#[derive(Debug)]
pub struct ExecutionLog {
    /// Directory holding the metadata and the segments.
    dir: PathBuf,
    /// Execution the journal belongs to.
    identity: ExecutionIdentity,
    /// Retention limits.
    capacity: LogCapacity,
    /// Source of the entry timestamps.
    clock: Arc<dyn Clock>,
    /// Write state, shared with reads so a rotation cannot race either.
    write_state: Mutex<WriteState>,
    /// Journal creation timestamp, preserved across reopens.
    created_at: String,
}

/// Mutable part of a journal, guarded by [`ExecutionLog::write_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WriteState {
    /// Sequence number the next append will use.
    next_seq: u64,
    /// Number of lines in the active segment.
    lines: usize,
    /// Number of bytes in the active segment.
    bytes: u64,
    /// Next archive segment number.
    generation: u64,
}

impl ExecutionLog {
    /// Opens (and creates if needed) the journal in `dir`.
    ///
    /// # Errors
    /// `Error::Validation` when `capacity` is invalid or the directory belongs to a different
    /// execution; `Error::Session` when the directory or the metadata cannot be read or written.
    pub async fn open(
        dir: impl AsRef<Path>,
        identity: ExecutionIdentity,
        capacity: LogCapacity,
    ) -> Result<Self> {
        Self::open_with_clock(dir, identity, capacity, Arc::new(SystemClock::new())).await
    }

    /// Opens (and creates if needed) the journal in `dir`, stamping entries through `clock`.
    ///
    /// An existing `meta.json` must describe the same execution, otherwise the journal is refused
    /// rather than reused. The metadata is rewritten with the recomputed generation and segment
    /// counters, so reopening is idempotent.
    ///
    /// # Errors
    /// `Error::Validation("execution log belongs to a different execution")` when the directory
    /// belongs to another execution, `Error::Validation` when `capacity` is invalid or the
    /// metadata has an unsupported encoding, and `Error::Session` on IO failure.
    pub async fn open_with_clock(
        dir: impl AsRef<Path>,
        identity: ExecutionIdentity,
        capacity: LogCapacity,
        clock: Arc<dyn Clock>,
    ) -> Result<Self> {
        capacity.validate()?;
        let dir = dir.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|error| io_error(&dir, "create", &error))?;
        let (created_at, mut generation) =
            match read_json::<LogMeta>(&dir.join(LOG_META_FILE)).await? {
                Some(meta) => {
                    if meta.identity != identity {
                        return Err(Error::validation(
                            "execution log belongs to a different execution",
                        ));
                    }
                    if meta.encoding != LOG_ENCODING {
                        return Err(Error::validation(format!(
                            "unsupported execution log encoding `{}`; expected `{LOG_ENCODING}`",
                            meta.encoding
                        )));
                    }
                    (meta.created_at, meta.generation.max(1))
                }
                None => (clock.now_rfc3339(), 1),
            };
        let mut next_seq = 1_u64;
        let mut lines = 0_usize;
        let mut bytes = 0_u64;
        let segments = segment_paths(&dir).await?;
        let mut highest_segment = 0_u64;
        for (segment_generation, path) in &segments {
            highest_segment = highest_segment.max(*segment_generation);
            next_seq = next_seq.max(highest_seq(path, next_seq).await?);
        }
        let active = dir.join(LOG_ENTRY_FILE);
        if let Some(raw) = read_optional_bytes(&active).await? {
            let scan = scan_lines(&String::from_utf8_lossy(&raw));
            lines = scan.lines;
            bytes = u64::try_from(raw.len()).unwrap_or(u64::MAX);
            next_seq = next_seq.max(highest_scanned_seq(&scan, next_seq));
        }
        generation = generation.max(highest_segment.saturating_add(1));
        let log = Self {
            dir,
            identity,
            capacity,
            clock,
            write_state: Mutex::new(WriteState {
                next_seq,
                lines,
                bytes,
                generation,
            }),
            created_at,
        };
        log.write_meta(generation).await?;
        Ok(log)
    }

    /// Opens the journal of `identity` under the workspace's `persistent_agent` node.
    ///
    /// The path is `<persistent_agent node>/logs/<directory name of the identity>`; a workspace
    /// without that node falls back to `<root_path>/persistent_agent/logs/…`.
    ///
    /// # Errors
    /// Every error [`ExecutionLog::open`] reports.
    pub async fn for_workspace(
        workspace: &Workspace,
        identity: ExecutionIdentity,
        capacity: LogCapacity,
    ) -> Result<Self> {
        let base = workspace
            .get_node_path(PERSISTENT_AGENT_NODE)
            .unwrap_or_else(|| workspace.root_path.join(PERSISTENT_AGENT_NODE));
        let dir = base.join(LOG_SUBDIR).join(identity.directory_name());
        Self::open(dir, identity, capacity).await
    }

    /// Directory holding the metadata and the segments.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Execution the journal belongs to.
    #[must_use]
    pub fn identity(&self) -> &ExecutionIdentity {
        &self.identity
    }

    /// Retention limits of this journal.
    #[must_use]
    pub fn capacity(&self) -> LogCapacity {
        self.capacity
    }

    /// Sequence number the next append will use.
    #[must_use]
    pub async fn next_seq(&self) -> u64 {
        self.write_state.lock().await.next_seq
    }

    /// Live position of the journal.
    #[must_use]
    pub async fn cursor(&self) -> LogCursor {
        let state = self.write_state.lock().await;
        LogCursor {
            next_seq: state.next_seq,
            last_seq: state.next_seq.saturating_sub(1),
            generation: state.generation,
        }
    }

    /// Appends one payload and returns the persisted entry.
    ///
    /// The active segment is rotated first when it already reached `max_entries` lines or
    /// `max_bytes` bytes. The counters advance only after the line was written.
    ///
    /// # Errors
    /// `Error::Validation` when the entry cannot be encoded, and `Error::Session` when the segment
    /// or the metadata cannot be written.
    pub async fn append(&self, payload: EntryPayload) -> Result<ExecutionEntry> {
        let mut state = self.write_state.lock().await;
        if state.lines >= self.capacity.max_entries || state.bytes >= self.capacity.max_bytes {
            self.rotate_locked(&mut state).await?;
        }
        let seq = state.next_seq;
        let entry = ExecutionEntry::new(seq, self.clock.now_rfc3339(), payload);
        let mut line = serde_json::to_string(&entry)
            .map_err(|error| Error::validation(format!("failed to encode entry: {error}")))?;
        line.push('\n');
        let path = self.entry_path();
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|error| io_error(&path, "open", &error))?;
        file.write_all(line.as_bytes())
            .await
            .map_err(|error| io_error(&path, "append to", &error))?;
        state.next_seq = seq.saturating_add(1);
        state.lines = state.lines.saturating_add(1);
        state.bytes = state
            .bytes
            .saturating_add(u64::try_from(line.len()).unwrap_or(u64::MAX));
        Ok(entry)
    }

    /// Appends each payload in order and returns the persisted entries.
    ///
    /// # Errors
    /// Every error [`ExecutionLog::append`] reports; the entries appended before the failure stay
    /// persisted.
    pub async fn append_many(&self, payloads: &[EntryPayload]) -> Result<Vec<ExecutionEntry>> {
        let mut entries = Vec::with_capacity(payloads.len());
        for payload in payloads {
            entries.push(self.append(payload.clone()).await?);
        }
        Ok(entries)
    }

    /// Rotates the active segment now, regardless of its size.
    ///
    /// The active file becomes `entries.<generation>.jsonl`, the generation advances, the metadata
    /// is rewritten atomically and the oldest archives beyond `keep_segments` are deleted. An
    /// existing segment is never rewritten: the generation skips ahead if its name is taken.
    ///
    /// # Errors
    /// `Error::Session` when the rename, the metadata write or a prune fails.
    pub async fn rotate_now(&self) -> Result<()> {
        let mut state = self.write_state.lock().await;
        self.rotate_locked(&mut state).await
    }

    /// Reads every segment and returns the usable entries with the read report.
    ///
    /// # Errors
    /// `Error::Session` when a segment or the directory cannot be read.
    pub async fn read(&self) -> Result<LogSnapshot> {
        let state = self.write_state.lock().await;
        Ok(self.read_locked(&state).await?.0)
    }

    /// Reads the journal and returns the entries with the cursor to continue from.
    ///
    /// # Errors
    /// Every error [`ExecutionLog::read`] reports.
    pub async fn replay(&self) -> Result<Replay> {
        let state = self.write_state.lock().await;
        let (snapshot, cursor) = self.read_locked(&state).await?;
        let truncated = snapshot.is_truncated();
        Ok(Replay {
            cursor,
            digest: snapshot.digest(),
            entries: snapshot.entries,
            truncated,
        })
    }

    /// Returns the digest of the current read.
    ///
    /// # Errors
    /// Every error [`ExecutionLog::read`] reports.
    pub async fn digest(&self) -> Result<LogDigest> {
        Ok(self.read().await?.digest())
    }

    /// Returns the path of the active segment.
    fn entry_path(&self) -> PathBuf {
        self.dir.join(LOG_ENTRY_FILE)
    }

    /// Returns the path of the archive segment with sequence number `generation`.
    fn segment_path(&self, generation: u64) -> PathBuf {
        self.dir
            .join(format!("{SEGMENT_PREFIX}{generation}{SEGMENT_SUFFIX}"))
    }

    /// Writes the metadata for `generation`.
    async fn write_meta(&self, generation: u64) -> Result<()> {
        let meta = LogMeta {
            schema_version: LOG_META_VERSION,
            encoding: LOG_ENCODING.to_string(),
            identity: self.identity.clone(),
            created_at: self.created_at.clone(),
            generation,
        };
        write_json_atomic(&self.dir.join(LOG_META_FILE), &meta).await
    }

    /// Rotates the active segment; the caller holds the write state.
    async fn rotate_locked(&self, state: &mut WriteState) -> Result<()> {
        let mut generation = state.generation.max(1);
        loop {
            let candidate = self.segment_path(generation);
            match tokio::fs::try_exists(&candidate).await {
                Ok(false) => break,
                Ok(true) => generation = generation.saturating_add(1),
                Err(error) => return Err(io_error(&candidate, "inspect", &error)),
            }
        }
        let active = self.entry_path();
        match tokio::fs::rename(&active, self.segment_path(generation)).await {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(io_error(&active, "rename", &error)),
        }
        state.generation = generation.saturating_add(1);
        state.lines = 0;
        state.bytes = 0;
        self.write_meta(state.generation).await?;
        self.prune_segments().await
    }

    /// Deletes the oldest archives beyond `keep_segments`.
    async fn prune_segments(&self) -> Result<()> {
        let segments = segment_paths(&self.dir).await?;
        let excess = segments.len().saturating_sub(self.capacity.keep_segments);
        for (_, path) in segments.into_iter().take(excess) {
            match tokio::fs::remove_file(&path).await {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(io_error(&path, "remove", &error)),
            }
        }
        Ok(())
    }

    /// Reads the archives in generation order then the active segment; the caller holds the state.
    async fn read_locked(&self, state: &WriteState) -> Result<(LogSnapshot, LogCursor)> {
        let mut sources: Vec<Vec<u8>> = Vec::new();
        for (_, path) in segment_paths(&self.dir).await? {
            if let Some(raw) = read_optional_bytes(&path).await? {
                sources.push(raw);
            }
        }
        if let Some(raw) = read_optional_bytes(&self.entry_path()).await? {
            sources.push(raw);
        }
        let snapshot = self.snapshot_from(&sources);
        let cursor = LogCursor {
            next_seq: state.next_seq,
            last_seq: state.next_seq.saturating_sub(1),
            generation: state.generation,
        };
        Ok((snapshot, cursor))
    }

    /// Builds the snapshot of `sources`, which are the segment bytes in read order.
    fn snapshot_from(&self, sources: &[Vec<u8>]) -> LogSnapshot {
        let mut by_seq: BTreeMap<u64, Vec<ExecutionEntry>> = BTreeMap::new();
        let mut total_lines = 0_usize;
        let mut skipped_invalid = 0_usize;
        for raw in sources {
            let scan = scan_lines(&String::from_utf8_lossy(raw));
            total_lines = total_lines.saturating_add(scan.lines);
            skipped_invalid = skipped_invalid.saturating_add(scan.skipped);
            for entry in scan.entries {
                by_seq.entry(entry.seq).or_default().push(entry);
            }
        }
        let mut entries = Vec::with_capacity(by_seq.len());
        let mut duplicates_removed = 0_usize;
        let mut conflicting_seqs = Vec::new();
        for (seq, group) in by_seq {
            let mut unique: Vec<ExecutionEntry> = Vec::with_capacity(group.len());
            for entry in group {
                let digest = entry.digest();
                if unique
                    .iter()
                    .any(|existing: &ExecutionEntry| existing.digest() == digest)
                {
                    duplicates_removed = duplicates_removed.saturating_add(1);
                } else {
                    unique.push(entry);
                }
            }
            if unique.len() > 1 {
                conflicting_seqs.push(seq);
            } else if let Some(entry) = unique.pop() {
                entries.push(entry);
            }
        }
        let missing_ranges = missing_ranges(entries.iter().map(|entry| entry.seq));
        let retained_out = entries.len().saturating_sub(self.capacity.max_entries);
        if retained_out > 0 {
            entries.drain(..retained_out);
        }
        let unusable = if total_lines == 0 {
            Some("empty".to_string())
        } else if entries.is_empty() {
            Some("no_usable_entry".to_string())
        } else {
            None
        };
        LogSnapshot {
            identity: self.identity.clone(),
            entries,
            skipped_invalid,
            duplicates_removed,
            conflicting_seqs,
            missing_ranges,
            retained_out,
            unusable,
        }
    }
}

/// Returns the highest sequence number stored in `path`, plus one, or `fallback`.
async fn highest_seq(path: &Path, fallback: u64) -> Result<u64> {
    match read_optional_bytes(path).await? {
        Some(raw) => Ok(highest_scanned_seq(
            &scan_lines(&String::from_utf8_lossy(&raw)),
            fallback,
        )),
        None => Ok(fallback),
    }
}

/// Returns the highest sequence number of `scan`, plus one, or `fallback`.
fn highest_scanned_seq(scan: &LineScan, fallback: u64) -> u64 {
    scan.entries
        .iter()
        .map(|entry| entry.seq.saturating_add(1))
        .max()
        .unwrap_or(fallback)
        .max(fallback)
}

/// Returns the archived segments of `dir`, sorted by generation.
async fn segment_paths(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut reader = match tokio::fs::read_dir(dir).await {
        Ok(reader) => reader,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(io_error(dir, "read", &error)),
    };
    let mut segments = Vec::new();
    while let Some(entry) = reader
        .next_entry()
        .await
        .map_err(|error| io_error(dir, "read", &error))?
    {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(generation) = parse_segment_name(name) {
            segments.push((generation, entry.path()));
        }
    }
    segments.sort_by_key(|(generation, _)| *generation);
    Ok(segments)
}

/// Parses `entries.<generation>.jsonl` into its generation; other names yield `None`.
fn parse_segment_name(name: &str) -> Option<u64> {
    let digits = name
        .strip_prefix(SEGMENT_PREFIX)?
        .strip_suffix(SEGMENT_SUFFIX)?;
    if digits.is_empty() || !digits.chars().all(|character| character.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// Reads `path`, treating a missing file as `Ok(None)`.
async fn read_optional_bytes(path: &Path) -> Result<Option<Vec<u8>>> {
    match tokio::fs::read(path).await {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io_error(path, "read", &error)),
    }
}

/// Outcome of scanning one segment's text.
#[derive(Debug, Default)]
struct LineScan {
    /// Entries that parsed and validated.
    entries: Vec<ExecutionEntry>,
    /// Number of lines seen, including the skipped ones.
    lines: usize,
    /// Number of empty or unparseable (including invalid) lines.
    skipped: usize,
}

/// Scans `text` line by line, skipping empty, unparseable and invalid lines.
///
/// A line that parses but does not [`validate`](ExecutionEntry::validate) (zero sequence number or
/// a non-RFC 3339 timestamp) is skipped as well, so a damaged line can never break the replay.
fn scan_lines(text: &str) -> LineScan {
    let mut scan = LineScan::default();
    for line in text.lines() {
        scan.lines = scan.lines.saturating_add(1);
        match serde_json::from_str::<ExecutionEntry>(line) {
            Ok(entry) if entry.validate().is_ok() => scan.entries.push(entry),
            Ok(_) | Err(_) => scan.skipped = scan.skipped.saturating_add(1),
        }
    }
    scan
}

/// Returns the sequence ranges no entry of `seqs` covers, starting at `1`.
fn missing_ranges(seqs: impl Iterator<Item = u64>) -> Vec<(u64, u64)> {
    let mut ranges = Vec::new();
    let mut expected = 1_u64;
    for seq in seqs {
        if seq > expected {
            ranges.push((expected, seq.saturating_sub(1)));
        }
        expected = seq.saturating_add(1);
    }
    ranges
}

/// Returns the persisted snake-case spelling of `status`.
///
/// [`ToolStatus`] is serialized as a bare string, so this mirrors
/// [`serde_json`]'s rendering and keeps the digest keys stable.
const fn tool_status_key(status: ToolStatus) -> &'static str {
    match status {
        ToolStatus::Succeeded => "succeeded",
        ToolStatus::Failed => "failed",
        ToolStatus::Unknown => "unknown",
    }
}

/// Builds the IO error of `action` on `path`.
fn io_error(path: &Path, action: &str, error: &io::Error) -> Error {
    Error::session(format!("failed to {action} {}: {error}", path.display()))
}

impl fmt::Display for LogCursor {
    /// Renders the cursor as `seqs <next>/<last>, generation <generation>`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "seqs {}/{}, generation {}",
            self.next_seq, self.last_seq, self.generation
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::{ExecutionLog, LogCapacity, LogMeta, LOG_ENCODING, LOG_ENTRY_FILE, LOG_META_FILE};
    use crate::persistent_agent::entry::{EntryPayload, ExecutionEntry, ModelUsage, ToolStatus};
    use crate::persistent_agent::identity::ExecutionIdentity;
    use crate::persistent_agent::ManualClock;
    use crate::workspace::Workspace;

    const STAMP: i64 = 1_700_000_000_000;
    const STAMP_TEXT: &str = "2023-11-14T22:13:20.000Z";

    fn identity() -> ExecutionIdentity {
        ExecutionIdentity::new("ns", "agent", "session", "request", "execution", "source").unwrap()
    }

    fn other_identity() -> ExecutionIdentity {
        ExecutionIdentity::new("ns", "agent", "session", "request", "exec2", "source").unwrap()
    }

    fn clock() -> Arc<ManualClock> {
        Arc::new(ManualClock::new(STAMP))
    }

    async fn open(dir: &tempfile::TempDir, capacity: LogCapacity) -> ExecutionLog {
        ExecutionLog::open_with_clock(dir.path(), identity(), capacity, clock())
            .await
            .unwrap()
    }

    fn line(value: &serde_json::Value) -> String {
        format!("{}\n", serde_json::to_string(&value).unwrap())
    }

    #[tokio::test]
    async fn append_assigns_sequence_numbers_and_stamps_the_clock() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open(&dir, LogCapacity::default()).await;
        let first = log.append(EntryPayload::note("a")).await.unwrap();
        let second = log.append(EntryPayload::note("b")).await.unwrap();
        let third = log.append(EntryPayload::note("c")).await.unwrap();
        assert_eq!(
            [first.seq, second.seq, third.seq],
            [1, 2, 3],
            "sequence numbers are consecutive"
        );
        assert_eq!(first.occurred_at, STAMP_TEXT);
        assert_eq!(log.next_seq().await, 4);
        let cursor = log.cursor().await;
        assert_eq!(cursor.next_seq, 4);
        assert_eq!(cursor.last_seq, 3);
        assert_eq!(cursor.generation, 1);
        assert_eq!(cursor.to_string(), "seqs 4/3, generation 1");

        let snapshot = log.read().await.unwrap();
        assert!(snapshot.is_usable());
        assert_eq!(snapshot.identity, identity());
        assert_eq!(
            snapshot
                .entries
                .iter()
                .map(|entry| entry.seq)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(snapshot.skipped_invalid, 0);
        assert_eq!(snapshot.unusable, None);
        let text = tokio::fs::read_to_string(dir.path().join(LOG_ENTRY_FILE))
            .await
            .unwrap();
        assert_eq!(text.lines().count(), 3);
        assert!(text.ends_with('\n'));
        assert!(text.contains(r#""kind":"note""#));
    }

    #[tokio::test]
    async fn reopen_resumes_at_the_next_sequence() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let log = open(&dir, LogCapacity::default()).await;
            log.append(EntryPayload::note("a")).await.unwrap();
            log.append(EntryPayload::note("b")).await.unwrap();
            log.append(EntryPayload::note("c")).await.unwrap();
        }
        let log = open(&dir, LogCapacity::default()).await;
        assert_eq!(log.next_seq().await, 4, "a crash loses no sequence number");
        let resumed = log.append(EntryPayload::note("d")).await.unwrap();
        assert_eq!(resumed.seq, 4);
        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.entries.len(), 4);
        assert_eq!(snapshot.skipped_invalid, 0);
        assert_eq!(snapshot.missing_ranges, Vec::<(u64, u64)>::new());
        let meta: LogMeta = serde_json::from_str(
            &tokio::fs::read_to_string(dir.path().join(LOG_META_FILE))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(meta.schema_version, 1);
        assert_eq!(meta.encoding, LOG_ENCODING);
        assert_eq!(meta.identity, identity());
        assert_eq!(meta.created_at, STAMP_TEXT);
        assert_eq!(meta.generation, 1);
    }

    #[tokio::test]
    async fn partial_trailing_line_is_skipped_on_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let log = open(&dir, LogCapacity::default()).await;
            log.append(EntryPayload::note("a")).await.unwrap();
            log.append(EntryPayload::note("b")).await.unwrap();
        }
        let path = dir.path().join(LOG_ENTRY_FILE);
        let torn = r#"{"seq":3,"occurred_at":"2023-11-14T22:13:20.000Z","kind":"no"#;
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .await
            .unwrap();
        tokio::io::AsyncWriteExt::write_all(&mut file, torn.as_bytes())
            .await
            .unwrap();
        drop(file);

        let log = open(&dir, LogCapacity::default()).await;
        assert_eq!(log.next_seq().await, 3);
        let snapshot = log.read().await.unwrap();
        assert!(snapshot.is_usable());
        assert_eq!(snapshot.skipped_invalid, 1);
        assert_eq!(snapshot.entries.len(), 2);
        assert_eq!(snapshot.unusable, None);
        let replay = log.replay().await.unwrap();
        assert!(replay.truncated);
        assert_eq!(replay.cursor.next_seq, 3);
        assert_eq!(replay.cursor.generation, 1);
    }

    #[tokio::test]
    async fn rotation_keeps_the_newest_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let capacity = LogCapacity {
            max_entries: 2,
            ..LogCapacity::default()
        };
        let log = open(&dir, capacity).await;
        log.append(EntryPayload::note("a")).await.unwrap();
        log.append(EntryPayload::note("b")).await.unwrap();
        log.append(EntryPayload::note("c")).await.unwrap();
        assert_eq!(
            log.cursor().await.generation,
            2,
            "the first two entries were archived"
        );
        assert!(dir.path().join("entries.1.jsonl").exists());

        let snapshot = log.read().await.unwrap();
        assert_eq!(
            snapshot
                .entries
                .iter()
                .map(|entry| entry.seq)
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(snapshot.retained_out, 1);
        assert!(snapshot.is_truncated());
        assert_eq!(snapshot.missing_ranges, Vec::<(u64, u64)>::new());
        let replay = log.replay().await.unwrap();
        assert_eq!(replay.digest.entries, 2);
        assert_eq!(replay.digest.last_seq, Some(3));
    }

    #[tokio::test]
    async fn keep_segments_prunes_the_oldest_archives() {
        let dir = tempfile::TempDir::new().unwrap();
        let capacity = LogCapacity {
            max_entries: 1,
            keep_segments: 2,
            ..LogCapacity::default()
        };
        let log = open(&dir, capacity).await;
        for index in 1..=5 {
            log.append(EntryPayload::note(format!("n{index}")))
                .await
                .unwrap();
        }
        assert_eq!(log.cursor().await.generation, 5);
        assert!(!dir.path().join("entries.1.jsonl").exists());
        assert!(!dir.path().join("entries.2.jsonl").exists());
        assert!(dir.path().join("entries.3.jsonl").exists());
        assert!(dir.path().join("entries.4.jsonl").exists());

        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].seq, 5);
        assert_eq!(snapshot.retained_out, 2);
        assert_eq!(snapshot.missing_ranges, vec![(1, 2)]);
        assert_eq!(snapshot.unusable, None);
    }

    #[tokio::test]
    async fn duplicate_sequence_numbers_are_collapsed() {
        let dir = tempfile::TempDir::new().unwrap();
        let entry = ExecutionEntry::new(1, STAMP_TEXT, EntryPayload::note("a"));
        let second = ExecutionEntry::new(2, STAMP_TEXT, EntryPayload::note("b"));
        let mut text = String::new();
        text.push_str(&line(&serde_json::to_value(&entry).unwrap()));
        text.push_str(&line(&serde_json::to_value(&entry).unwrap()));
        text.push_str(&line(&serde_json::to_value(&second).unwrap()));
        tokio::fs::write(dir.path().join(LOG_ENTRY_FILE), text)
            .await
            .unwrap();
        let log = open(&dir, LogCapacity::default()).await;
        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.entries.len(), 2);
        assert_eq!(snapshot.duplicates_removed, 1);
        assert_eq!(snapshot.conflicting_seqs, Vec::<u64>::new());
        assert_eq!(snapshot.skipped_invalid, 0);
    }

    #[tokio::test]
    async fn conflicting_sequence_numbers_drop_the_group() {
        let dir = tempfile::TempDir::new().unwrap();
        let first = ExecutionEntry::new(1, STAMP_TEXT, EntryPayload::note("a"));
        let left = ExecutionEntry::new(2, STAMP_TEXT, EntryPayload::note("left"));
        let right = ExecutionEntry::new(2, STAMP_TEXT, EntryPayload::note("right"));
        let third = ExecutionEntry::new(3, STAMP_TEXT, EntryPayload::note("c"));
        let mut text = String::new();
        for entry in [&first, &left, &right, &third] {
            text.push_str(&line(&serde_json::to_value(entry).unwrap()));
        }
        tokio::fs::write(dir.path().join(LOG_ENTRY_FILE), text)
            .await
            .unwrap();
        let log = open(&dir, LogCapacity::default()).await;
        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.conflicting_seqs, vec![2]);
        assert_eq!(
            snapshot
                .entries
                .iter()
                .map(|entry| entry.seq)
                .collect::<Vec<_>>(),
            [1, 3]
        );
        assert_eq!(snapshot.missing_ranges, vec![(2, 2)]);
        assert!(snapshot.is_truncated());
        assert!(snapshot.is_usable());
    }

    #[tokio::test]
    async fn missing_ranges_report_gaps_from_one() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut text = String::new();
        for seq in [1, 4] {
            let entry = ExecutionEntry::new(seq, STAMP_TEXT, EntryPayload::note("x"));
            text.push_str(&line(&serde_json::to_value(&entry).unwrap()));
        }
        tokio::fs::write(dir.path().join(LOG_ENTRY_FILE), text)
            .await
            .unwrap();
        let log = open(&dir, LogCapacity::default()).await;
        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.missing_ranges, vec![(2, 3)]);
        assert_eq!(snapshot.entries.len(), 2);

        let mut only_later = String::new();
        let entry = ExecutionEntry::new(5, STAMP_TEXT, EntryPayload::note("x"));
        only_later.push_str(&line(&serde_json::to_value(&entry).unwrap()));
        tokio::fs::write(dir.path().join(LOG_ENTRY_FILE), only_later)
            .await
            .unwrap();
        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.missing_ranges, vec![(1, 4)]);
    }

    #[tokio::test]
    async fn empty_and_unusable_reads_are_reported() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open(&dir, LogCapacity::default()).await;
        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.unusable.as_deref(), Some("empty"));
        assert!(!snapshot.is_usable());
        assert_eq!(
            snapshot.digest().summary(),
            "entries=0 seqs=none kinds=[] tools=[] tokens=in:0,out:0,unknown:0"
        );

        tokio::fs::write(dir.path().join(LOG_ENTRY_FILE), b"{not json}\n\n")
            .await
            .unwrap();
        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.unusable.as_deref(), Some("no_usable_entry"));
        assert_eq!(snapshot.skipped_invalid, 2);
        assert!(!snapshot.is_usable());
    }

    #[tokio::test]
    async fn digest_reports_counts_tokens_and_summary() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open(&dir, LogCapacity::default()).await;
        log.append(EntryPayload::execution_started("ship it"))
            .await
            .unwrap();
        log.append(EntryPayload::model_turn(
            "mc_1",
            "hello",
            ModelUsage::known(10, 5),
        ))
        .await
        .unwrap();
        log.append(EntryPayload::tool_completed(
            "call_1",
            "bash",
            ToolStatus::Succeeded,
            "{}",
            "ok",
        ))
        .await
        .unwrap();
        log.append(EntryPayload::model_turn(
            "mc_2",
            "again",
            ModelUsage::unknown("provider omitted usage"),
        ))
        .await
        .unwrap();
        let digest = log.digest().await.unwrap();
        assert_eq!(digest.entries, 4);
        assert_eq!(digest.first_seq, Some(1));
        assert_eq!(digest.last_seq, Some(4));
        assert_eq!(digest.input_tokens, 10);
        assert_eq!(digest.output_tokens, 5);
        assert_eq!(digest.unknown_usage, 1);
        assert_eq!(digest.tool_outcomes.get("succeeded"), Some(&1));
        assert_eq!(
            digest
                .counts
                .get(&crate::persistent_agent::entry::EntryKind::ModelTurn),
            Some(&2)
        );
        let summary = digest.summary();
        assert!(summary.starts_with("entries=4 seqs=1..4 kinds=[execution_started=1,model_turn=2"));
        assert!(summary.contains("tools=[succeeded=1]"));
        assert!(summary.contains("tokens=in:10,out:5,unknown:1"));
        assert!(!summary.contains("missing="));
        assert_eq!(summary, digest.summary(), "the summary is deterministic");

        let decoded: super::LogDigest =
            serde_json::from_str(&serde_json::to_string(&digest).unwrap()).unwrap();
        assert_eq!(decoded, digest);
        let snapshot = log.read().await.unwrap();
        assert_eq!(snapshot.digest(), digest);
        let replay = snapshot.replay();
        assert!(!replay.truncated);
        assert_eq!(replay.cursor.next_seq, 5);
        assert_eq!(replay.cursor.generation, 0);
        assert_eq!(replay.entries.len(), 4);
        let decoded: super::Replay =
            serde_json::from_str(&serde_json::to_string(&replay).unwrap()).unwrap();
        assert_eq!(decoded, replay);
    }

    #[tokio::test]
    async fn identity_mismatch_is_rejected_on_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let log = open(&dir, LogCapacity::default()).await;
            log.append(EntryPayload::note("a")).await.unwrap();
        }
        let error = ExecutionLog::open(dir.path(), other_identity(), LogCapacity::default())
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(error
            .to_string()
            .contains("execution log belongs to a different execution"));
    }

    #[test]
    fn capacity_validation_rejects_zero_fields() {
        assert!(LogCapacity::default().validate().is_ok());
        let fields: [(LogCapacity, &str); 3] = [
            (
                LogCapacity {
                    max_entries: 0,
                    ..LogCapacity::default()
                },
                "max_entries",
            ),
            (
                LogCapacity {
                    max_bytes: 0,
                    ..LogCapacity::default()
                },
                "max_bytes",
            ),
            (
                LogCapacity {
                    keep_segments: 0,
                    ..LogCapacity::default()
                },
                "keep_segments",
            ),
        ];
        for (capacity, field) in fields {
            let error = capacity.validate().unwrap_err();
            assert!(matches!(
                error,
                openjiuwen_common::error::Error::Validation(_)
            ));
            assert!(error.to_string().contains(field));
        }
        let defaults = LogCapacity::default();
        assert_eq!(defaults.max_entries, 4096);
        assert_eq!(defaults.max_bytes, 4 * 1024 * 1024);
        assert_eq!(defaults.keep_segments, 4);
        let partial: LogCapacity = serde_json::from_str(r#"{"max_entries": 8}"#).unwrap();
        assert_eq!(partial.max_entries, 8);
        assert_eq!(partial.max_bytes, defaults.max_bytes);
    }

    #[tokio::test]
    async fn for_workspace_uses_the_persistent_agent_node() {
        let dir = tempfile::TempDir::new().unwrap();
        let workspace = Workspace::for_language(dir.path(), "en");
        let log = ExecutionLog::for_workspace(&workspace, identity(), LogCapacity::default())
            .await
            .unwrap();
        assert_eq!(
            log.dir(),
            dir.path()
                .join("persistent_agent")
                .join("logs")
                .join("execution")
        );
        log.append(EntryPayload::note("a")).await.unwrap();
        assert_eq!(
            ExecutionLog::for_workspace(&workspace, other_identity(), LogCapacity::default())
                .await
                .unwrap()
                .dir(),
            dir.path()
                .join("persistent_agent")
                .join("logs")
                .join("exec2")
        );
    }

    #[tokio::test]
    async fn rotate_now_archives_even_an_empty_segment_and_lands_in_meta() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open(&dir, LogCapacity::default()).await;
        log.rotate_now().await.unwrap();
        assert!(!dir.path().join("entries.1.jsonl").exists());
        assert_eq!(log.cursor().await.generation, 2);
        log.append(EntryPayload::note("a")).await.unwrap();
        log.rotate_now().await.unwrap();
        assert!(dir.path().join("entries.2.jsonl").exists());
        let meta = serde_json::from_str::<serde_json::Value>(
            &tokio::fs::read_to_string(dir.path().join(LOG_META_FILE))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(meta["generation"], json!(3));
        assert_eq!(meta["encoding"], json!(LOG_ENCODING));
    }

    #[tokio::test]
    async fn append_many_returns_the_persisted_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open(&dir, LogCapacity::default()).await;
        let entries = log
            .append_many(&[
                EntryPayload::execution_started("go"),
                EntryPayload::reply_delta("hi"),
            ])
            .await
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 1);
        assert_eq!(entries[1].seq, 2);
        assert_eq!(log.read().await.unwrap().entries, entries);
    }
}
