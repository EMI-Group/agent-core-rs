//! Per-path trailing debounce that turns raw notifications into [`WatchEvent`]s.
//!
//! Raw OS notifications arrive in bursts: one save can produce a create, several
//! modifies and a metadata change for the same path. [`Debouncer`] keeps at most
//! one pending entry per path, merges every follow-up through
//! [`ChangeKind::merge`], and moves that path's deadline to `now + window`, so
//! only the last quiet moment in a burst is reported.
//!
//! The state machine never reads the clock: the caller passes `now` as a
//! [`std::time::Instant`], which makes every rule below unit-testable without
//! timers or sleeps.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::watcher::event::{ChangeKind, WatchEvent, WatchTargetKind};

/// Default cap on the number of distinct paths pending at once.
///
/// Reached only by a genuine event storm; the input that would exceed it is
/// dropped (and counted) instead of growing the queue without bound.
pub const DEFAULT_MAX_PENDING: usize = 4096;

/// A producer-side change before sequence assignment.
///
/// [`Debouncer::push`] stores it; a pending entry that matches the same path
/// absorbs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebounceInput {
    /// The kind of harness root the path belongs to.
    pub kind: WatchTargetKind,
    /// The path that changed.
    pub path: PathBuf,
    /// How the path changed.
    pub change: ChangeKind,
    /// The original path of a rename, when the backend reported one.
    pub previous_path: Option<PathBuf>,
    /// Stable label of the target the path matched.
    pub label: Option<String>,
}

/// One path waiting for its debounce window to elapse.
#[derive(Debug, Clone)]
struct Pending {
    kind: WatchTargetKind,
    change: ChangeKind,
    previous_path: Option<PathBuf>,
    label: Option<String>,
    deadline: Instant,
}

impl Pending {
    /// Folds `input` into this entry, keeping the deadline handling to the caller.
    fn absorb(&mut self, input: DebounceInput) {
        self.change = self.change.merge(input.change);
        self.kind = input.kind;
        if let Some(previous) = input.previous_path {
            self.previous_path = Some(previous);
        }
        if self.change != ChangeKind::Renamed {
            self.previous_path = None;
        }
        if input.label.is_some() {
            self.label = input.label;
        }
    }
}

/// Coalescing state machine behind the watcher.
///
/// Entries are kept in a [`BTreeMap`] keyed by path, so a flush emits events in
/// deterministic path order and sequence numbers increase monotonically from
/// `1`, independent of the order the OS delivered the notifications in.
#[derive(Debug)]
pub struct Debouncer {
    window: Duration,
    max_pending: usize,
    sequence: u64,
    dropped: u64,
    pending: BTreeMap<PathBuf, Pending>,
}

impl Debouncer {
    /// Creates a debouncer with `window` and the default pending cap.
    #[must_use]
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            max_pending: DEFAULT_MAX_PENDING,
            sequence: 0,
            dropped: 0,
            pending: BTreeMap::new(),
        }
    }

    /// Replaces the cap on simultaneously pending paths.
    #[must_use]
    pub fn with_max_pending(mut self, max_pending: usize) -> Self {
        self.max_pending = max_pending;
        self
    }

    /// Returns the debounce window.
    #[must_use]
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Returns how many distinct paths are currently pending.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Returns the earliest deadline among the pending paths.
    ///
    /// `None` when nothing is pending; the caller uses it to sleep until the
    /// next possible flush.
    #[must_use]
    pub fn next_deadline(&self) -> Option<Instant> {
        self.pending.values().map(|pending| pending.deadline).min()
    }

    /// Returns how many inputs were dropped because the pending cap was hit.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Queues `input`, coalescing it into the entry for the same path.
    ///
    /// The path's deadline always moves to `now + window` (trailing debounce).
    /// A new path that would exceed the pending cap (see
    /// [`Self::with_max_pending`]) is dropped: the counter advances and a
    /// warning is logged. Paths already pending keep coalescing, so a full
    /// queue still reports changes it already knows about.
    pub fn push(&mut self, input: DebounceInput, now: Instant) {
        let deadline = now + self.window;
        if let Some(pending) = self.pending.get_mut(&input.path) {
            pending.absorb(input);
            pending.deadline = deadline;
            return;
        }
        if self.pending.len() >= self.max_pending {
            self.dropped = self.dropped.saturating_add(1);
            tracing::warn!(
                path = %input.path.display(),
                max_pending = self.max_pending,
                "watcher debounce queue is full; dropping the event for a new path"
            );
            return;
        }
        self.pending.insert(
            input.path,
            Pending {
                kind: input.kind,
                change: input.change,
                previous_path: input.previous_path,
                label: input.label,
                deadline,
            },
        );
    }

    /// Drains every path whose deadline is at or before `now`.
    ///
    /// Events are returned in path order with increasing sequence numbers.
    pub fn flush(&mut self, now: Instant) -> Vec<WatchEvent> {
        let due: Vec<PathBuf> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.deadline <= now)
            .map(|(path, _)| path.clone())
            .collect();
        let mut events = Vec::with_capacity(due.len());
        for path in due {
            if let Some(pending) = self.pending.remove(&path) {
                events.push(self.next_event(path, pending));
            }
        }
        events
    }

    /// Drains everything regardless of deadline, for a shutdown drain.
    ///
    /// Events are returned in path order with increasing sequence numbers.
    pub fn take_all(&mut self) -> Vec<WatchEvent> {
        let pending = std::mem::take(&mut self.pending);
        let mut events = Vec::with_capacity(pending.len());
        for (path, entry) in pending {
            events.push(self.next_event(path, entry));
        }
        events
    }

    /// Builds the next event, advancing the sequence counter.
    fn next_event(&mut self, path: PathBuf, pending: Pending) -> WatchEvent {
        self.sequence = self.sequence.saturating_add(1);
        WatchEvent {
            sequence: self.sequence,
            kind: pending.kind,
            path,
            change: pending.change,
            previous_path: pending.previous_path,
            label: pending.label,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const WINDOW: Duration = Duration::from_millis(250);

    fn input(path: &str, change: ChangeKind) -> DebounceInput {
        DebounceInput {
            kind: WatchTargetKind::Skills,
            path: PathBuf::from(path),
            change,
            previous_path: None,
            label: Some("skills:workspace".to_owned()),
        }
    }

    #[test]
    fn accessors_expose_the_configuration() {
        let debouncer = Debouncer::new(WINDOW);
        assert_eq!(debouncer.window(), WINDOW);
        assert_eq!(debouncer.pending_len(), 0);
        assert_eq!(debouncer.next_deadline(), None);
        assert_eq!(debouncer.dropped(), 0);
        assert_eq!(DEFAULT_MAX_PENDING, 4096);
    }

    #[test]
    fn two_pushes_for_one_path_coalesce_into_one_event() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        debouncer.push(input("/w/skills/A.md", ChangeKind::Created), now);
        debouncer.push(
            input("/w/skills/A.md", ChangeKind::Modified),
            now + Duration::from_millis(10),
        );
        assert_eq!(debouncer.pending_len(), 1);

        let events = debouncer.flush(now + Duration::from_secs(1));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[0].change, ChangeKind::Created);
        assert_eq!(events[0].path, PathBuf::from("/w/skills/A.md"));
        assert_eq!(events[0].kind, WatchTargetKind::Skills);
        assert_eq!(events[0].label.as_deref(), Some("skills:workspace"));
        assert!(debouncer.next_deadline().is_none());
    }

    #[test]
    fn removed_then_created_nets_to_modified() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        debouncer.push(input("/w/skills/A.md", ChangeKind::Removed), now);
        debouncer.push(input("/w/skills/A.md", ChangeKind::Created), now);
        let events = debouncer.take_all();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].change, ChangeKind::Modified);
    }

    #[test]
    fn each_push_extends_the_deadline() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        debouncer.push(input("/w/skills/A.md", ChangeKind::Created), now);
        assert_eq!(debouncer.next_deadline(), Some(now + WINDOW));

        debouncer.push(input("/w/skills/A.md", ChangeKind::Modified), now + WINDOW);
        assert_eq!(debouncer.next_deadline(), Some(now + WINDOW + WINDOW));

        // The original deadline has passed, but the path was extended.
        assert_eq!(debouncer.flush(now + WINDOW).len(), 0);
        let events = debouncer.flush(now + WINDOW + WINDOW);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].sequence, 1);
    }

    #[test]
    fn flush_emits_due_entries_in_path_order_with_increasing_sequences() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        debouncer.push(input("/w/skills/b.md", ChangeKind::Created), now);
        debouncer.push(input("/w/skills/a.md", ChangeKind::Created), now);
        debouncer.push(
            input("/w/skills/late.md", ChangeKind::Created),
            now + WINDOW,
        );

        let events = debouncer.flush(now + WINDOW);
        let paths: Vec<&str> = events
            .iter()
            .map(|event| event.path.to_str().unwrap())
            .collect();
        assert_eq!(paths, vec!["/w/skills/a.md", "/w/skills/b.md"]);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[1].sequence, 2);

        // `late.md` was pushed at the flush instant, so it is due only one
        // window later and continues the sequence.
        let later = debouncer.flush(now + WINDOW + WINDOW);
        assert_eq!(later.len(), 1);
        assert_eq!(later[0].path, PathBuf::from("/w/skills/late.md"));
        assert_eq!(later[0].sequence, 3);
    }

    #[test]
    fn zero_window_emits_on_the_same_instant() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(Duration::ZERO);
        debouncer.push(input("/w/skills/A.md", ChangeKind::Created), now);
        assert_eq!(debouncer.next_deadline(), Some(now));

        let events = debouncer.flush(now);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(debouncer.flush(now).len(), 0);
    }

    #[test]
    fn pending_cap_drops_only_new_paths() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW).with_max_pending(1);
        debouncer.push(input("/w/skills/a.md", ChangeKind::Created), now);
        // An existing path still coalesces while the queue is full.
        debouncer.push(input("/w/skills/a.md", ChangeKind::Modified), now);
        assert_eq!(debouncer.dropped(), 0);

        debouncer.push(input("/w/skills/b.md", ChangeKind::Created), now);
        assert_eq!(debouncer.dropped(), 1);
        assert_eq!(debouncer.pending_len(), 1);

        let events = debouncer.flush(now + WINDOW);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].path, PathBuf::from("/w/skills/a.md"));
        // The dropped path stays counted after the flush.
        assert_eq!(debouncer.dropped(), 1);
    }

    #[test]
    fn zero_capacity_drops_every_new_path() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW).with_max_pending(0);
        debouncer.push(input("/w/skills/a.md", ChangeKind::Created), now);
        debouncer.push(input("/w/skills/b.md", ChangeKind::Created), now);
        assert_eq!(debouncer.dropped(), 2);
        assert_eq!(debouncer.take_all().len(), 0);
    }

    #[test]
    fn take_all_drains_regardless_of_deadline() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        debouncer.push(input("/w/skills/a.md", ChangeKind::Created), now);
        debouncer.push(input("/w/skills/b.md", ChangeKind::Created), now);
        assert_eq!(debouncer.flush(now).len(), 0);

        let events = debouncer.take_all();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[1].sequence, 2);
        assert_eq!(debouncer.pending_len(), 0);
        assert_eq!(debouncer.take_all().len(), 0);
        // Sequence numbers keep increasing across drains.
        debouncer.push(input("/w/skills/c.md", ChangeKind::Created), now);
        let events = debouncer.take_all();
        assert_eq!(events[0].sequence, 3);
    }

    #[test]
    fn rename_keeps_the_previous_path_only_while_renamed() {
        let now = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        let mut renamed = input("/w/skills/new.md", ChangeKind::Renamed);
        renamed.previous_path = Some(PathBuf::from("/w/skills/old.md"));
        debouncer.push(renamed, now);
        let events = debouncer.take_all();
        assert_eq!(events[0].change, ChangeKind::Renamed);
        assert_eq!(
            events[0].previous_path.as_deref(),
            Some(std::path::Path::new("/w/skills/old.md"))
        );

        let mut renamed = input("/w/skills/new.md", ChangeKind::Renamed);
        renamed.previous_path = Some(PathBuf::from("/w/skills/old.md"));
        debouncer.push(renamed, now);
        debouncer.push(input("/w/skills/new.md", ChangeKind::Removed), now);
        let events = debouncer.take_all();
        assert_eq!(events[0].change, ChangeKind::Removed);
        assert!(events[0].previous_path.is_none());
    }
}
