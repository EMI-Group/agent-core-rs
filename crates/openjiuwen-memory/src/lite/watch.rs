//! Debounced filesystem watching that feeds the lite reindex pipeline.
//!
//! Python's lite memory watches its Markdown sources and rebuilds the derived index on a
//! debounce timer. This module is the Rust equivalent of that seam: [`MemoryWatcher`]
//! watches files or directories, keeps only Markdown memory paths, coalesces a burst of
//! changes into a single [`WatchEvent`], and hands it to the caller over a channel. It
//! never reads or writes memory content — the reindex pipeline (`lite::local`) decides
//! what to do with an event.
//!
//! # Design choices
//!
//! - **An unbounded `tokio::sync::mpsc` channel instead of a boxed async callback.**
//!   `notify` invokes its handler on a thread it owns, and that thread must neither block
//!   nor await. The handler therefore forwards every event into an internal unbounded
//!   channel (a non-blocking, runtime-free `send`); all debouncing and delivery happen on
//!   a normal tokio task. The outward channel is unbounded for the same reason: the
//!   consumer must never make the watcher lose a change or stall a notify thread.
//! - **Only Markdown paths, coalesced within `debounce`.** Events are filtered to paths
//!   with a case-insensitive `md` extension that are not directories and not inside a
//!   directory component named `.memory`. Everything collected until the internal channel
//!   has been quiet for `debounce` becomes exactly one [`WatchEvent`] with deduplicated,
//!   sorted paths and deduplicated kinds in [`WatchKind`] declaration order. An empty
//!   burst is never emitted, and a burst without a single Markdown path is not an event.
//! - **Watcher errors never abort the watch loop.** `notify` errors (kernel or filesystem
//!   failures, possibly per path) are logged with `tracing::warn!` and dropped: they can
//!   be transient and the watch stays registered. [`MemoryWatcher::start`] returns
//!   [`Error::external`] only when the watcher cannot be created or a requested path
//!   cannot be watched, with that path in the message.
//! - **Shutdown is explicit or by drop.** [`MemoryWatcher::stop`] cancels the debounce
//!   task, awaits it, and drops the `notify` watcher (which stops its thread); dropping
//!   the watcher performs the same cancellation and watcher drop without awaiting, so
//!   `stop()` is the clean, awaitable shutdown.
//! - **Non-recursive watches.** Each path is watched with
//!   [`RecursiveMode::NonRecursive`](notify::RecursiveMode::NonRecursive); a reindex
//!   pipeline that wants subdirectories passes them explicitly, which keeps the watched
//!   set explicit and `.memory` subtrees out of it.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use notify::event::ModifyKind;
use notify::{EventKind, RecursiveMode, Watcher};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

/// The tracing target of the filesystem-watcher diagnostics.
const TARGET: &str = "openjiuwen_memory::lite::watch";

/// The directory name whose contents are never reported as memory changes.
const EXCLUDED_DIR: &str = ".memory";

/// The kind of change coalesced into one [`WatchEvent`].
///
/// The names are the `notify` event classes reduced to what a reindex cares about; a
/// rename is reported as [`WatchKind::Renamed`] rather than a remove plus a create,
/// because both paths of the rename are carried by the same event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WatchKind {
    /// A path was created.
    Created,
    /// A path's content, metadata, or name-in-place changed.
    Modified,
    /// A path was removed (or moved out of the watched tree).
    Removed,
    /// A path was renamed or moved.
    Renamed,
    /// An access event, an unsupported backend event, or anything not classified above.
    Other,
}

impl WatchKind {
    /// The canonical declaration order used to sort the kinds of one [`WatchEvent`].
    const fn ordinal(self) -> u8 {
        match self {
            Self::Created => 0,
            Self::Modified => 1,
            Self::Removed => 2,
            Self::Renamed => 3,
            Self::Other => 4,
        }
    }
}

/// A coalesced burst of filesystem changes for `.md` paths.
///
/// `paths` is deduplicated and sorted, and `kinds` is deduplicated and in
/// [`WatchKind`] declaration order (`created`, `modified`, `removed`, `renamed`, `other`),
/// so two runs over the same burst produce identical values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchEvent {
    /// The deduplicated, sorted Markdown paths touched by the burst.
    pub paths: Vec<PathBuf>,
    /// The deduplicated kinds observed in the burst, in declaration order.
    pub kinds: Vec<WatchKind>,
}

/// A running watcher: dropping it (or calling [`MemoryWatcher::stop`]) shuts the watcher
/// thread and the debounce task down.
///
/// [`MemoryWatcher::start`] must be called from within a tokio runtime; the debounce task
/// lives on that runtime for the lifetime of the watcher.
pub struct MemoryWatcher {
    /// The registered `notify` watcher; dropping it stops its thread.
    watcher: Option<notify::RecommendedWatcher>,
    /// Cancels the debounce task on [`MemoryWatcher::stop`] and on drop.
    cancel: CancellationToken,
    /// The debounce/delivery task; `None` once it has been joined or dropped.
    task: Option<JoinHandle<()>>,
    /// The quiet window a burst must reach before it is emitted.
    debounce: Duration,
}

impl fmt::Debug for MemoryWatcher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryWatcher")
            .field("running", &self.is_running())
            .field("debounce", &self.debounce)
            .finish_non_exhaustive()
    }
}

impl MemoryWatcher {
    /// Starts watching `paths` (files or directories, non-recursive) and delivers
    /// debounced events on `sender`.
    ///
    /// Must be called from within a tokio runtime (it spawns the debounce task on it).
    ///
    /// A burst of changes is emitted once the watcher has been quiet for `debounce`;
    /// a zero debounce therefore emits one event per received `notify` event.
    ///
    /// # Errors
    ///
    /// Returns [`Error::external`] with the offending path in the message when a watched
    /// path cannot be registered, and [`Error::external`] with a `creating a filesystem
    /// watcher` context when no `notify` watcher can be created for this platform.
    pub fn start(
        paths: impl IntoIterator<Item = PathBuf>,
        debounce: Duration,
        sender: UnboundedSender<WatchEvent>,
    ) -> Result<Self> {
        let (raw_sender, raw_receiver) = mpsc::unbounded_channel::<notify::Event>();
        // The handler runs on a notify-owned thread: it only forwards (non-blocking) and
        // never panics, so a watcher error or a dropped receiver cannot abort the watch.
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| match event {
                Ok(event) => {
                    if raw_sender.send(event).is_err() {
                        debug!(
                            target: TARGET,
                            "debounce task is gone; dropping a filesystem event"
                        );
                    }
                }
                Err(error) => {
                    warn!(target: TARGET, %error, "filesystem watcher reported an error");
                }
            })
            .map_err(|error| {
                Error::external(error).with_context("creating a filesystem watcher")
            })?;

        for path in paths {
            watcher
                .watch(&path, RecursiveMode::NonRecursive)
                .map_err(|error| {
                    Error::external(error).with_context(format!("watching {}", path.display()))
                })?;
        }

        let cancel = CancellationToken::new();
        let task = tokio::spawn(debounce_loop(
            raw_receiver,
            sender,
            debounce,
            cancel.clone(),
        ));
        Ok(Self {
            watcher: Some(watcher),
            cancel,
            task: Some(task),
            debounce,
        })
    }

    /// Stops watching and awaits the debounce task.
    ///
    /// Any burst still being collected is dropped rather than delivered. This is the
    /// clean shutdown; dropping the watcher cancels the task too, but cannot await it.
    pub async fn stop(&mut self) {
        self.cancel.cancel();
        self.watcher.take();
        if let Some(task) = self.task.take() {
            if let Err(error) = task.await {
                warn!(target: TARGET, %error, "debounce task did not shut down cleanly");
            }
        }
    }

    /// Whether the watcher is still running.
    ///
    /// Returns `false` after [`MemoryWatcher::stop`], and also when the debounce task
    /// ended on its own because the `notify` watcher was dropped.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.task.as_ref().is_some_and(|task| !task.is_finished())
    }
}

impl Drop for MemoryWatcher {
    fn drop(&mut self) {
        // Cannot await here: cancel so the debounce task exits at its next suspend point,
        // and drop the notify watcher so its thread stops sending.
        self.cancel.cancel();
        self.watcher.take();
        self.task.take();
    }
}

/// Whether `path` is a memory Markdown path: case-insensitive `md` extension and no
/// `.memory` directory component.
fn is_candidate_path(path: &Path) -> bool {
    let is_markdown = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("md"));
    if !is_markdown {
        return false;
    }
    !path
        .components()
        .any(|component| component.as_os_str() == EXCLUDED_DIR)
}

/// Whether `path` should be reported: a candidate Markdown path that is not a directory.
///
/// The metadata lookup is async and a path that no longer exists (removed or renamed
/// away) counts as a file, so removals still reach the reindex pipeline.
async fn is_watchable(path: &Path) -> bool {
    if !is_candidate_path(path) {
        return false;
    }
    match tokio::fs::metadata(path).await {
        Ok(metadata) => !metadata.is_dir(),
        // The path vanished between the event and this check (removed, renamed away, or
        // a transient editor save): report it so a removal still reaches the pipeline.
        Err(_) => true,
    }
}

/// Maps a `notify` event kind to the reduced [`WatchKind`].
fn watch_kind(kind: EventKind) -> WatchKind {
    match kind {
        EventKind::Create(_) => WatchKind::Created,
        EventKind::Modify(ModifyKind::Name(_)) => WatchKind::Renamed,
        EventKind::Modify(_) => WatchKind::Modified,
        EventKind::Remove(_) => WatchKind::Removed,
        EventKind::Access(_) | EventKind::Any | EventKind::Other => WatchKind::Other,
    }
}

/// The accumulating state of one debounce burst.
#[derive(Default)]
struct Burst {
    /// Deduplicated, sorted Markdown paths touched by the burst.
    paths: BTreeSet<PathBuf>,
    /// Kinds observed for those paths, deduplicated (kept in a small `Vec`).
    kinds: Vec<WatchKind>,
}

impl Burst {
    /// Folds one `notify` event into the burst, ignoring paths that are not watchable.
    async fn push(&mut self, event: &notify::Event) {
        let kind = watch_kind(event.kind);
        let mut matched = false;
        for path in &event.paths {
            if is_watchable(path).await {
                self.paths.insert(path.clone());
                matched = true;
            }
        }
        if matched && !self.kinds.contains(&kind) {
            self.kinds.push(kind);
        }
    }

    /// Emits the burst as one [`WatchEvent`], or does nothing when no path matched.
    ///
    /// The receiver may be gone (the reindex pipeline stopped): that is logged at
    /// `debug` and the burst is discarded, never panicked on.
    fn emit(&mut self, sender: &UnboundedSender<WatchEvent>) {
        if self.paths.is_empty() {
            return;
        }
        let mut kinds = std::mem::take(&mut self.kinds);
        kinds.sort_by_key(|kind| kind.ordinal());
        let event = WatchEvent {
            paths: std::mem::take(&mut self.paths).into_iter().collect(),
            kinds,
        };
        if sender.send(event).is_err() {
            debug!(
                target: TARGET,
                "watch event receiver is gone; dropping the coalesced burst"
            );
        }
    }
}

/// Collects raw `notify` events into bursts and delivers one [`WatchEvent`] per quiet
/// `debounce` window.
///
/// Exits when cancelled, when the internal channel closes (the `notify` watcher was
/// dropped), or when the outward receiver is gone.
async fn debounce_loop(
    mut raw: UnboundedReceiver<notify::Event>,
    sender: UnboundedSender<WatchEvent>,
    debounce: Duration,
    cancel: CancellationToken,
) {
    let mut burst = Burst::default();
    'watch: loop {
        let first = tokio::select! {
            () = cancel.cancelled() => break,
            received = raw.recv() => received,
        };
        let Some(event) = first else {
            debug!(target: TARGET, "filesystem event source closed; stopping the debounce task");
            break;
        };
        burst.push(&event).await;
        // Keep extending the burst while events keep arriving; emit once the channel has
        // been quiet for `debounce`.
        loop {
            let next = tokio::select! {
                () = cancel.cancelled() => break 'watch,
                received = tokio::time::timeout(debounce, raw.recv()) => received,
            };
            match next {
                Ok(Some(event)) => burst.push(&event).await,
                Ok(None) => {
                    burst.emit(&sender);
                    debug!(
                        target: TARGET,
                        "filesystem event source closed; stopping the debounce task"
                    );
                    break 'watch;
                }
                Err(_elapsed) => break,
            }
        }
        burst.emit(&sender);
        if sender.is_closed() {
            debug!(target: TARGET, "watch event receiver is gone; stopping the debounce task");
            break;
        }
    }
    debug!(target: TARGET, "filesystem watch debounce task stopped");
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    const BOUND: Duration = Duration::from_secs(5);
    const DEBOUNCE: Duration = Duration::from_millis(100);
    const QUIET: Duration = Duration::from_millis(400);

    /// Starts a watcher over `dir` and returns it with the receiving end.
    fn start_watcher(dir: &Path) -> (MemoryWatcher, UnboundedReceiver<WatchEvent>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let watcher =
            MemoryWatcher::start([dir.to_path_buf()], DEBOUNCE, sender).expect("watch temp dir");
        (watcher, receiver)
    }

    /// Whether the event carries `file` (by file name, so temp-path canonicalization is
    /// irrelevant).
    fn covers(event: &WatchEvent, file: &str) -> bool {
        event
            .paths
            .iter()
            .any(|path| path.file_name() == Some(OsStr::new(file)))
    }

    #[tokio::test]
    async fn emits_markdown_change() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_watcher, mut receiver) = start_watcher(dir.path());
        tokio::fs::write(dir.path().join("a.md"), "# a\n")
            .await
            .expect("write a.md");

        let deadline = tokio::time::Instant::now() + BOUND;
        let mut found: Option<WatchEvent> = None;
        while found.is_none() {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, receiver.recv()).await {
                Ok(Some(event)) if covers(&event, "a.md") => found = Some(event),
                // Another Markdown path's event: keep waiting for a.md.
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => break,
            }
        }
        let event = found.expect("an a.md watch event within the bound");
        assert!(
            event
                .kinds
                .iter()
                .any(|kind| matches!(kind, WatchKind::Created | WatchKind::Modified)),
            "unexpected kinds: {:?}",
            event.kinds
        );
        assert!(
            event.paths.iter().all(|path| path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))),
            "non-Markdown paths leaked through: {:?}",
            event.paths
        );
    }

    #[tokio::test]
    async fn ignores_non_markdown_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_watcher, mut receiver) = start_watcher(dir.path());
        tokio::fs::write(dir.path().join("a.txt"), "not memory")
            .await
            .expect("write a.txt");

        match tokio::time::timeout(QUIET, receiver.recv()).await {
            Err(_elapsed) => {}
            Ok(None) => {}
            Ok(Some(event)) => panic!("a non-Markdown write produced {event:?}"),
        }
    }

    #[tokio::test]
    async fn stop_ends_delivery() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (mut watcher, mut receiver) = start_watcher(dir.path());
        tokio::fs::write(dir.path().join("a.md"), "one")
            .await
            .expect("write a.md");
        tokio::time::timeout(BOUND, receiver.recv())
            .await
            .expect("an event before stop")
            .expect("the channel stays open");

        watcher.stop().await;
        assert!(
            !watcher.is_running(),
            "the watcher reports running after stop"
        );

        tokio::fs::write(dir.path().join("a.md"), "two")
            .await
            .expect("rewrite a.md");
        match tokio::time::timeout(QUIET, receiver.recv()).await {
            Err(_elapsed) => {}
            Ok(None) => {}
            Ok(Some(event)) => panic!("an event arrived after stop: {event:?}"),
        }
    }

    #[tokio::test]
    async fn coalesces_a_burst() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (_watcher, mut receiver) = start_watcher(dir.path());
        let names = ["a.md", "b.md", "c.md"];
        for name in names {
            tokio::fs::write(dir.path().join(name), name)
                .await
                .expect("write a memory file");
        }

        let deadline = tokio::time::Instant::now() + BOUND;
        let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
        while seen.len() < names.len() {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, receiver.recv()).await {
                Ok(Some(event)) => {
                    seen.extend(event.paths);
                }
                Ok(None) | Err(_) => break,
            }
        }
        for name in names {
            assert!(
                seen.iter()
                    .any(|path| path.file_name() == Some(OsStr::new(name))),
                "missing {name} in {seen:?}"
            );
        }
    }

    #[tokio::test]
    async fn ignores_paths_under_the_memory_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        let nested = dir.path().join(EXCLUDED_DIR);
        std::fs::create_dir_all(&nested).expect("create .memory dir");
        let (_watcher, mut receiver) = start_watcher(dir.path());

        // The new directory itself also produces an event, but it is not a Markdown path.
        tokio::fs::write(nested.join("a.md"), "# a\n")
            .await
            .expect("write a.md inside .memory");

        match tokio::time::timeout(QUIET, receiver.recv()).await {
            Err(_elapsed) => {}
            Ok(None) => {}
            Ok(Some(event)) => panic!("a .memory path produced {event:?}"),
        }
    }

    #[test]
    fn watch_kind_maps_notify_kinds() {
        assert_eq!(
            watch_kind(EventKind::Create(notify::event::CreateKind::File)),
            WatchKind::Created
        );
        assert_eq!(
            watch_kind(EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Content
            ))),
            WatchKind::Modified
        );
        assert_eq!(
            watch_kind(EventKind::Modify(notify::event::ModifyKind::Name(
                notify::event::RenameMode::Both
            ))),
            WatchKind::Renamed
        );
        assert_eq!(
            watch_kind(EventKind::Remove(notify::event::RemoveKind::File)),
            WatchKind::Removed
        );
        assert_eq!(watch_kind(EventKind::Any), WatchKind::Other);
        assert_eq!(
            watch_kind(EventKind::Access(notify::event::AccessKind::Any)),
            WatchKind::Other
        );
        assert_eq!(watch_kind(EventKind::Other), WatchKind::Other);
    }

    #[test]
    fn candidate_paths_are_markdown_only() {
        assert!(is_candidate_path(Path::new("/tmp/memory/USER.md")));
        assert!(is_candidate_path(Path::new("/tmp/memory/notes.MD")));
        assert!(!is_candidate_path(Path::new("/tmp/memory/notes.txt")));
        assert!(!is_candidate_path(Path::new("/tmp/memory/notes")));
        assert!(!is_candidate_path(Path::new("/tmp/memory/.memory/USER.md")));
    }

    #[test]
    fn watch_event_round_trips_through_json() {
        let event = WatchEvent {
            paths: vec![PathBuf::from("b.md"), PathBuf::from("a.md")],
            kinds: vec![WatchKind::Created, WatchKind::Renamed],
        };
        let json = serde_json::to_string(&event).expect("serialize a watch event");
        assert_eq!(
            json,
            r#"{"paths":["b.md","a.md"],"kinds":["created","renamed"]}"#
        );
        let decoded: WatchEvent = serde_json::from_str(&json).expect("deserialize a watch event");
        assert_eq!(decoded, event);
    }
}
