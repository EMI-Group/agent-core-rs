//! Read-before-write state: file fingerprints, read coverage and the per-path write/edit gates.
//!
//! The `read_file` tool records which byte ranges of a file the model has seen; `write_file`
//! refuses to clobber content that was never read, and `edit_file` refuses to edit a file that
//! changed on disk since it was last observed.
//!
//! Deviation from Python: the Python tool layer keeps a module-global `_FILE_READ_REGISTRY`.
//! Here the registry is an ordinary value owned by the tool registry, so tests are isolated and
//! there is no undocumented global mutable state.
//!
//! Line numbers in coverage ranges are 1-based and inclusive; byte offsets reported back to the
//! model are 0-based (the `offset` argument of `read_file`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// A cheap identity for one version of a file on disk.
///
/// Two fingerprints compare equal only when both the modification time and the length match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FileFingerprint {
    /// Last modification time in nanoseconds since the UNIX epoch, or `0` when unavailable.
    pub mtime_ns: u128,
    /// File length in bytes.
    pub size_bytes: u64,
}

/// Build a [`FileFingerprint`] from filesystem metadata.
///
/// Platforms that do not expose a modification time (or report one before the epoch) yield an
/// `mtime_ns` of `0`, which still compares consistently across two reads of the same file.
#[must_use]
pub fn file_fingerprint(metadata: &std::fs::Metadata) -> FileFingerprint {
    let mtime_ns = metadata
        .modified()
        .ok()
        .and_then(|mtime| mtime.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    FileFingerprint {
        mtime_ns,
        size_bytes: metadata.len(),
    }
}

/// One 1-based inclusive line range observed by the model.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReadCoverage {
    /// First observed line (1-based).
    pub start: usize,
    /// Last observed line (1-based, inclusive).
    pub end: usize,
}

/// Everything the registry remembers about one path.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FileReadState {
    /// Fingerprint of the file version that was observed.
    pub fingerprint: FileFingerprint,
    /// Total line count of the observed file (`0` when only a partial read happened and the
    /// total is not known).
    pub total_lines: usize,
    /// Merged, sorted, non-overlapping 1-based inclusive line ranges the model has read.
    ///
    /// `None` means nothing has been read yet.
    pub read_ranges: Option<Vec<(usize, usize)>>,
    /// Full content of the observed file, when a read or write captured it.
    pub content: Option<String>,
}

/// Verdict for a `write_file` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteGate {
    /// The whole file was read (or written) and has not changed: writing is safe.
    Ok,
    /// The file was never read in this session.
    NotRead,
    /// Only part of the file was read; re-read from `resume_offset` before writing.
    Partial {
        /// Number of lines already read across all recorded ranges.
        lines_read: usize,
        /// Total number of lines in the file.
        total_lines: usize,
        /// 0-based line offset at which reading should resume.
        resume_offset: usize,
    },
    /// The file changed on disk since it was read.
    Stale,
}

/// Verdict for an `edit_file` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditGate {
    /// The file was read and has not changed (partial reads are acceptable for edits).
    Ok,
    /// The file was never read in this session.
    NotRead,
    /// The file changed on disk since it was read.
    Stale,
}

/// Per-registry registry of read state plus per-path async locks.
///
/// The inner [`Mutex`] is only ever held for short, synchronous map operations; it is never held
/// across an `.await`.
#[derive(Debug, Default)]
pub struct ReadStateRegistry {
    states: Mutex<HashMap<PathBuf, FileReadState>>,
    locks: Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>,
}

impl ReadStateRegistry {
    /// Create an empty registry behind an [`Arc`] for sharing between tools.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record that the 1-based inclusive `coverage` range of `path` was read.
    ///
    /// The fingerprint is always refreshed. `content` is replaced only when supplied (a partial
    /// read carries no content and therefore leaves any previously recorded content intact).
    pub fn record_read(
        &self,
        path: &Path,
        fingerprint: FileFingerprint,
        coverage: Option<ReadCoverage>,
        content: Option<String>,
    ) {
        let mut states = self.states.lock();
        let state = states.entry(path.to_path_buf()).or_default();
        state.fingerprint = fingerprint;
        if let Some(coverage) = coverage {
            let ranges = state.read_ranges.get_or_insert_with(Vec::new);
            merge_range(ranges, coverage.start, coverage.end);
            if coverage.start <= 1 {
                // The read began at the start of the file, so `end` is the line count.
                state.total_lines = coverage.end;
            }
        }
        if let Some(content) = content {
            state.content = Some(content);
        }
    }

    /// Record that `path` was just written with `content`.
    ///
    /// A successful write means the whole file content is known, so the entry is replaced with
    /// full coverage of the written content.
    pub fn record_written(&self, path: &Path, fingerprint: FileFingerprint, content: String) {
        let total_lines = content.lines().count().max(1);
        let state = FileReadState {
            fingerprint,
            total_lines,
            read_ranges: Some(vec![(1, total_lines)]),
            content: Some(content),
        };
        self.states.lock().insert(path.to_path_buf(), state);
    }

    /// Decide whether a `write_file` may proceed for the current on-disk content.
    #[must_use]
    pub fn write_gate(
        &self,
        path: &Path,
        fingerprint: FileFingerprint,
        current: &str,
    ) -> WriteGate {
        let states = self.states.lock();
        let Some(state) = states.get(path) else {
            return WriteGate::NotRead;
        };
        let content_matches = state.content.as_deref() == Some(current);
        if state.fingerprint != fingerprint && !content_matches {
            return WriteGate::Stale;
        }
        let fully_read = state.total_lines > 0
            && state
                .read_ranges
                .as_deref()
                .is_some_and(|ranges| ranges_cover(ranges, 1, state.total_lines));
        if fully_read || content_matches {
            return WriteGate::Ok;
        }
        WriteGate::Partial {
            lines_read: lines_read(state.read_ranges.as_deref()),
            total_lines: state.total_lines,
            resume_offset: first_unread_offset(state.read_ranges.as_deref()),
        }
    }

    /// Decide whether a `write_file` may proceed using only the recorded fingerprint.
    ///
    /// The counterpart of [`Self::write_gate`] for a target too large to load: the fingerprint
    /// alone settles "unchanged" (the observed version) and "changed" (`Stale`); the content
    /// comparison is unavailable, so a partial read stays partial. No allocation depends on the
    /// file size.
    #[must_use]
    pub fn write_gate_metadata(&self, path: &Path, fingerprint: FileFingerprint) -> WriteGate {
        let states = self.states.lock();
        let Some(state) = states.get(path) else {
            return WriteGate::NotRead;
        };
        if state.fingerprint != fingerprint {
            return WriteGate::Stale;
        }
        let fully_read = state.total_lines > 0
            && state
                .read_ranges
                .as_deref()
                .is_some_and(|ranges| ranges_cover(ranges, 1, state.total_lines));
        if fully_read {
            return WriteGate::Ok;
        }
        WriteGate::Partial {
            lines_read: lines_read(state.read_ranges.as_deref()),
            total_lines: state.total_lines,
            resume_offset: first_unread_offset(state.read_ranges.as_deref()),
        }
    }

    /// Decide whether an `edit_file` may proceed for the current on-disk content.
    #[must_use]
    pub fn edit_gate(&self, path: &Path, fingerprint: FileFingerprint, current: &str) -> EditGate {
        let states = self.states.lock();
        let Some(state) = states.get(path) else {
            return EditGate::NotRead;
        };
        if state.fingerprint != fingerprint && state.content.as_deref() != Some(current) {
            return EditGate::Stale;
        }
        EditGate::Ok
    }

    /// Forget everything recorded for `path`.
    ///
    /// The per-path lock is intentionally kept so in-flight lock holders stay serialized with
    /// later callers; locks are cheap and bounded by the number of distinct paths touched.
    pub fn invalidate(&self, path: &Path) {
        self.states.lock().remove(path);
    }

    /// Forget every recorded path.
    pub fn clear(&self) {
        self.states.lock().clear();
    }

    /// Acquire the per-path exclusive lock, creating it on demand.
    ///
    /// The returned guard is owned, so it may be held across `.await` inside the caller.
    pub async fn lock(&self, path: &Path) -> OwnedMutexGuard<()> {
        let mutex = {
            let mut locks = self.locks.lock();
            Arc::clone(
                locks
                    .entry(path.to_path_buf())
                    .or_insert_with(|| Arc::new(AsyncMutex::new(()))),
            )
        };
        mutex.lock_owned().await
    }

    /// Number of paths currently tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.lock().len()
    }

    /// Whether no paths are currently tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.lock().is_empty()
    }
}

/// Merge a 1-based inclusive range into `ranges`, keeping the vector sorted and non-overlapping.
///
/// Inverted ranges (`end < start`) are ignored. Overlapping, contained and adjacent ranges
/// (`end + 1 == next.start`) are folded together.
pub fn merge_range(ranges: &mut Vec<(usize, usize)>, start: usize, end: usize) {
    if end < start {
        return;
    }
    let position = ranges.partition_point(|&(existing_start, _)| existing_start < start);
    ranges.insert(position, (start, end));

    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(range_start, range_end) in ranges.iter() {
        match merged.last_mut() {
            Some(last) if range_start <= last.1.saturating_add(1) => {
                if range_end > last.1 {
                    last.1 = range_end;
                }
            }
            _ => merged.push((range_start, range_end)),
        }
    }
    *ranges = merged;
}

/// Whether the union of `ranges` covers the 1-based inclusive span `start..=end` contiguously.
#[must_use]
pub fn ranges_cover(ranges: &[(usize, usize)], start: usize, end: usize) -> bool {
    if end < start {
        return false;
    }
    let mut cursor = start;
    for &(range_start, range_end) in ranges {
        if range_end < cursor {
            continue;
        }
        if range_start > cursor {
            return false;
        }
        if range_end >= end {
            return true;
        }
        cursor = range_end.saturating_add(1);
    }
    false
}

/// The 0-based line offset at which reading should resume.
///
/// This is the end of the contiguous covered prefix starting at line 1, or `0` when nothing (or
/// no range starting at line 1) has been read.
#[must_use]
pub fn first_unread_offset(ranges: Option<&[(usize, usize)]>) -> usize {
    let Some(ranges) = ranges else {
        return 0;
    };
    let mut cursor = 1usize;
    for &(range_start, range_end) in ranges {
        if range_end < cursor {
            continue;
        }
        if range_start > cursor {
            return cursor - 1;
        }
        cursor = range_end.saturating_add(1);
    }
    cursor - 1
}

/// Total number of lines covered by `ranges`.
fn lines_read(ranges: Option<&[(usize, usize)]>) -> usize {
    ranges.map_or(0, |ranges| {
        ranges
            .iter()
            .map(|&(start, end)| end.saturating_sub(start).saturating_add(1))
            .sum()
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    fn fingerprint(mtime_ns: u128, size_bytes: u64) -> FileFingerprint {
        FileFingerprint {
            mtime_ns,
            size_bytes,
        }
    }

    fn coverage(start: usize, end: usize) -> ReadCoverage {
        ReadCoverage { start, end }
    }

    #[test]
    fn merge_range_merges_overlapping_and_adjacent() {
        let mut ranges = vec![(1, 3), (10, 12)];
        merge_range(&mut ranges, 3, 5);
        assert_eq!(ranges, vec![(1, 5), (10, 12)]);

        merge_range(&mut ranges, 6, 9);
        assert_eq!(ranges, vec![(1, 12)]);
    }

    #[test]
    fn merge_range_keeps_sorted() {
        let mut ranges = Vec::new();
        merge_range(&mut ranges, 20, 25);
        merge_range(&mut ranges, 1, 4);
        merge_range(&mut ranges, 10, 12);
        assert_eq!(ranges, vec![(1, 4), (10, 12), (20, 25)]);
    }

    #[test]
    fn merge_range_ignores_inverted_and_empty() {
        let mut ranges = vec![(1, 3)];
        merge_range(&mut ranges, 6, 4);
        assert_eq!(ranges, vec![(1, 3)]);
    }

    #[test]
    fn merge_range_bridges_two_ranges() {
        let mut ranges = vec![(1, 2), (6, 8)];
        merge_range(&mut ranges, 3, 5);
        assert_eq!(ranges, vec![(1, 8)]);
    }

    #[test]
    fn merge_range_contained_is_noop() {
        let mut ranges = vec![(1, 10)];
        merge_range(&mut ranges, 4, 6);
        assert_eq!(ranges, vec![(1, 10)]);
    }

    #[test]
    fn ranges_cover_contiguous_and_gaps() {
        assert!(ranges_cover(&[(1, 3), (4, 8)], 1, 8));
        assert!(!ranges_cover(&[(1, 3), (5, 8)], 1, 8));
        assert!(!ranges_cover(&[], 1, 1));
        assert!(ranges_cover(&[(1, 5)], 1, 5));
        assert!(ranges_cover(&[(1, 5)], 2, 5));
        assert!(!ranges_cover(&[(1, 5)], 1, 6));
        assert!(!ranges_cover(&[(2, 5)], 1, 5));
    }

    #[test]
    fn first_unread_offset_cases() {
        assert_eq!(first_unread_offset(None), 0);
        assert_eq!(first_unread_offset(Some(&[])), 0);
        assert_eq!(first_unread_offset(Some(&[(1, 20)])), 20);
        assert_eq!(first_unread_offset(Some(&[(1, 5), (10, 12)])), 5);
        assert_eq!(first_unread_offset(Some(&[(3, 9)])), 0);
    }

    #[test]
    fn gates_report_not_read_for_unknown_path() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("missing.txt");
        assert_eq!(
            registry.write_gate(path, fingerprint(1, 1), "x"),
            WriteGate::NotRead
        );
        assert_eq!(
            registry.edit_gate(path, fingerprint(1, 1), "x"),
            EditGate::NotRead
        );
    }

    #[test]
    fn full_read_allows_write() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("full.txt");
        let print = fingerprint(1, 10);
        registry.record_read(path, print, Some(coverage(1, 3)), Some("a\nb\nc".into()));
        assert_eq!(registry.write_gate(path, print, "a\nb\nc"), WriteGate::Ok);
        assert_eq!(registry.edit_gate(path, print, "a\nb\nc"), EditGate::Ok);
    }

    #[test]
    fn partial_read_reports_partial_gate() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("partial.txt");
        let print = fingerprint(2, 20);
        // Only windows that never start at line 1 were read, so the total line count is unknown
        // and `write_file` must ask for a read from the beginning of the file.
        registry.record_read(path, print, Some(coverage(5, 8)), None);
        registry.record_read(path, print, Some(coverage(12, 13)), None);
        assert_eq!(
            registry.write_gate(path, print, "unseen content"),
            WriteGate::Partial {
                lines_read: 6,
                total_lines: 0,
                resume_offset: 0,
            }
        );
    }

    #[test]
    fn fingerprint_change_without_content_is_stale() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("changed.txt");
        registry.record_read(path, fingerprint(1, 5), Some(coverage(1, 2)), None);
        let newer = fingerprint(2, 5);
        assert_eq!(registry.write_gate(path, newer, "a\nb"), WriteGate::Stale);
        assert_eq!(registry.edit_gate(path, newer, "a\nb"), EditGate::Stale);
    }

    #[test]
    fn fingerprint_change_with_matching_content_is_ok() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("touched.txt");
        registry.record_read(
            path,
            fingerprint(1, 3),
            Some(coverage(1, 2)),
            Some("a\nb".into()),
        );
        let newer = fingerprint(9, 3);
        assert_eq!(registry.write_gate(path, newer, "a\nb"), WriteGate::Ok);
        assert_eq!(registry.edit_gate(path, newer, "a\nb"), EditGate::Ok);
    }

    #[test]
    fn record_read_keeps_previous_content_when_none() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("kept.txt");
        let print = fingerprint(1, 3);
        registry.record_read(path, print, Some(coverage(1, 2)), Some("a\nb".into()));
        registry.record_read(path, print, Some(coverage(1, 1)), None);
        // Content survived, so an unrelated fingerprint still matches content.
        assert_eq!(
            registry.write_gate(path, fingerprint(2, 3), "a\nb"),
            WriteGate::Ok
        );
    }

    #[test]
    fn record_written_allows_immediate_rewrite() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("written.txt");
        let print = fingerprint(3, 4);
        registry.record_written(path, print, "a\nb".to_string());
        assert_eq!(registry.write_gate(path, print, "a\nb"), WriteGate::Ok);
        assert_eq!(registry.edit_gate(path, print, "a\nb"), EditGate::Ok);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn record_written_empty_content_covers_one_line() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("empty.txt");
        let print = fingerprint(3, 0);
        registry.record_written(path, print, String::new());
        assert_eq!(registry.write_gate(path, print, ""), WriteGate::Ok);
    }

    #[test]
    fn edit_gate_accepts_partial_read() {
        let registry = ReadStateRegistry::new();
        let path = Path::new("partial-edit.txt");
        let print = fingerprint(1, 100);
        registry.record_read(path, print, Some(coverage(1, 1)), None);
        assert_eq!(registry.edit_gate(path, print, "anything"), EditGate::Ok);
    }

    #[test]
    fn invalidate_and_clear_reset_tracking() {
        let registry = ReadStateRegistry::new();
        assert!(registry.is_empty());
        let path = Path::new("a.txt");
        registry.record_written(path, fingerprint(1, 1), "x".to_string());
        registry.record_written(Path::new("b.txt"), fingerprint(1, 1), "y".to_string());
        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());

        registry.invalidate(path);
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry.write_gate(path, fingerprint(1, 1), "x"),
            WriteGate::NotRead
        );

        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn file_fingerprint_matches_real_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fp.txt");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"hello").unwrap();
        file.flush().unwrap();

        let first = file_fingerprint(&std::fs::metadata(&path).unwrap());
        let second = file_fingerprint(&std::fs::metadata(&path).unwrap());
        assert_eq!(first.size_bytes, 5);
        assert_eq!(first, second);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lock_serializes_same_path() {
        let registry = ReadStateRegistry::new();
        let path = PathBuf::from("shared.txt");
        let counter = Arc::new(parking_lot::Mutex::new(0usize));

        let mut handles = Vec::new();
        for _ in 0..2 {
            let registry = Arc::clone(&registry);
            let path = path.clone();
            let counter = Arc::clone(&counter);
            handles.push(tokio::spawn(async move {
                let guard = registry.lock(&path).await;
                *counter.lock() += 1;
                tokio::task::yield_now().await;
                drop(guard);
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }
        assert_eq!(*counter.lock(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lock_is_independent_per_path() {
        let registry = ReadStateRegistry::new();
        let first = registry.lock(Path::new("one.txt")).await;
        let second = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            registry.lock(Path::new("two.txt")),
        )
        .await
        .expect("locking a different path must not block");
        drop(second);
        drop(first);
    }
}
