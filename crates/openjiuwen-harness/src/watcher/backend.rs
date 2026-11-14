//! The OS-notification backend of the workspace watcher.
//!
//! [`start`] is the only entry point, called by
//! [`WorkspaceWatcher::start`](crate::watcher::WorkspaceWatcher::start). It
//! validates the plan, registers every distinct watch path on
//! `notify::recommended_watcher`, spawns the intake task that owns that watcher
//! plus the reload dispatcher task, and returns the [`WatcherHandle`] the caller
//! keeps.
//!
//! The intake task is a plain loop, not an actor: it blocks in a
//! `tokio::select!` over three sources — cancellation, raw `notify` events, and
//! the next debounce (or drop-notice) deadline. Raw events are converted to the
//! internal `RawChange` vocabulary, resolved against the plan's targets, and
//! pushed into the [`Debouncer`](crate::watcher::Debouncer); when a deadline
//! elapses the due events are published as
//! [`WatchUpdate::Event`](crate::watcher::event::WatchUpdate::Event) and queued
//! for the reload dispatcher, which runs the handlers on its own task.
//!
//! Four policies keep the task bounded and quiet under stress:
//!
//! - the raw intake queue is bounded (`WatcherConfig::intake_capacity`), so a
//!   notification burst cannot grow memory behind a busy loop; a full queue drops
//!   the newest notification and the drop is counted, logged and announced;
//! - reload handlers run on a dispatcher task of their own through a second
//!   bounded queue, so a slow handler stalls neither debouncing nor publication;
//!   the overflow of that queue is counted and announced the same way;
//! - an error whose every path is already gone is logged at `debug` and never
//!   published, so a removed subtree cannot surface as a backend failure;
//! - consecutive identical backend failures are published once, and every repeat
//!   is logged at `debug`, so an error storm cannot flood subscribers or the log.
//!
//! The `notify` dependency is optional. Without the feature the whole module
//! still compiles and [`start`] answers
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported).
use openjiuwen_common::error::{Error, Result};

use crate::watcher::handle::WatcherHandle;
use crate::watcher::WorkspaceWatcher;

/// Starts the OS watcher described by `builder`.
///
/// See [`WorkspaceWatcher::start`](crate::watcher::WorkspaceWatcher::start) for
/// the errors and `mod.rs` in this directory for the module contract.
//
// The `async` signature is shared with the `notify`-enabled body (and with
// `WorkspaceWatcher::start`), which is why the feature-off build keeps it.
#[cfg(not(feature = "notify"))]
#[allow(clippy::unused_async)]
pub(crate) async fn start(_builder: WorkspaceWatcher) -> Result<WatcherHandle> {
    Err(Error::unsupported(
        "workspace watcher requires the `notify` feature: rebuild openjiuwen-harness with \
         `--features notify`",
    ))
}

/// Starts the OS watcher described by `builder`.
///
/// See [`WorkspaceWatcher::start`](crate::watcher::WorkspaceWatcher::start) for
/// the errors and `mod.rs` in this directory for the module contract.
//
// `async` is part of the shared signature; everything that waits happens inside
// the spawned task, so this body itself never awaits.
#[cfg(feature = "notify")]
#[allow(clippy::unused_async)]
pub(crate) async fn start(builder: WorkspaceWatcher) -> Result<WatcherHandle> {
    use std::path::Path;

    use notify::{RecursiveMode, Watcher};
    use tokio::sync::{broadcast, mpsc};
    use tokio_util::sync::CancellationToken;

    use crate::watcher::event::{WatchEvent, WatchUpdate};
    use crate::watcher::intake::{push_raw, DropCounters};

    let WorkspaceWatcher { config, plan, hub } = builder;
    if plan.is_empty() {
        return Err(Error::config(
            "workspace watcher requires at least one watch target",
        ));
    }
    plan.validate()?;

    let registrations = registrations(plan.targets());
    let target_count = plan.len();
    let targets = plan.targets().to_vec();

    let counters = DropCounters::new();
    let (raw_tx, raw_rx) =
        mpsc::channel::<notify::Result<notify::Event>>(config.intake_capacity.max(1));
    // The callback runs on the backend's own thread, so it can neither await nor
    // block: the bounded queue is filled with `try_send`, the newest notification
    // is dropped when it is full, and the drop is counted (never silent).
    let callback_counters = counters.clone();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = push_raw(&raw_tx, &callback_counters, event);
    })
    .map_err(|error| map_notify_error(&error, Path::new(""), None))?;

    for (path, registration) in &registrations {
        let mode = if registration.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        watcher
            .watch(path, mode)
            .map_err(|error| map_notify_error(&error, path, registration.label.as_deref()))?;
    }
    tracing::debug!(
        paths = registrations.len(),
        targets = target_count,
        "workspace watcher registered its watch paths"
    );

    // `broadcast` rejects a zero capacity, so an opt-in `0` means "keep one update".
    let (events, _) = broadcast::channel::<WatchUpdate>(config.event_capacity.max(1));
    let publisher = events.clone();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let window = config.debounce;
    let notice_interval = window.max(MIN_DROP_NOTICE_INTERVAL);

    // Reload handlers run on their own task, so a slow body delays neither the
    // debounce loop nor publication: it can only fill this bounded queue, whose
    // overflow is counted and announced instead of growing memory. Handlers of
    // one event still run one after another, so two bodies never race.
    let (dispatch_tx, dispatch_rx) = mpsc::channel::<WatchEvent>(config.dispatch_capacity.max(1));
    let dispatch_task = spawn_dispatcher(hub, dispatch_rx, events.clone());

    let intake = IntakeLoop {
        watcher,
        targets,
        channels: IntakeChannels {
            publisher,
            dispatches: dispatch_tx,
            counters: counters.clone(),
        },
        cancel: task_cancel,
        window,
        notice_interval,
        raw_rx,
    };
    let task = tokio::spawn(intake.run());

    Ok(WatcherHandle::new(
        events,
        cancel,
        vec![task, dispatch_task],
        counters,
        target_count,
    ))
}

/// The sender side the intake loop publishes and dispatches through.
#[cfg(feature = "notify")]
struct IntakeChannels {
    /// Broadcast sender of the handle's subscription ring buffer.
    publisher: tokio::sync::broadcast::Sender<crate::watcher::event::WatchUpdate>,
    /// Bounded queue to the reload dispatcher.
    dispatches: tokio::sync::mpsc::Sender<crate::watcher::event::WatchEvent>,
    /// Shared drop counters of both queues.
    counters: crate::watcher::intake::DropCounters,
}

/// Spawns the task that runs the reload handlers.
///
/// It ends when the intake task drops the sender side of its queue, so it drains
/// the events already queued and then returns; a panic in a handler surfaces
/// through the task's join handle, which [`WatcherHandle::shutdown`] reports as
/// an `Error::sys_op`.
#[cfg(feature = "notify")]
fn spawn_dispatcher(
    hub: crate::watcher::hub::ReloadHub,
    mut dispatches: tokio::sync::mpsc::Receiver<crate::watcher::event::WatchEvent>,
    publisher: tokio::sync::broadcast::Sender<crate::watcher::event::WatchUpdate>,
) -> tokio::task::JoinHandle<()> {
    use crate::watcher::event::WatchUpdate;

    tokio::spawn(async move {
        while let Some(event) = dispatches.recv().await {
            for failure in hub.dispatch(&event).await {
                let _ = publisher.send(WatchUpdate::ReloadFailed(failure));
            }
        }
        tracing::debug!("workspace watcher reload dispatcher stopped");
    })
}

/// Everything one running intake task owns.
///
/// The task is a plain loop over three sources — cancellation, raw `notify`
/// events and the next deadline — and it never awaits a reload handler, so
/// debouncing and publication stay independent of handler latency.
#[cfg(feature = "notify")]
struct IntakeLoop {
    /// The OS backend; dropping it releases every watch.
    watcher: notify::RecommendedWatcher,
    /// The plan's targets, used to classify a raw change.
    targets: Vec<crate::watcher::WatchTarget>,
    /// Publication and dispatch plumbing.
    channels: IntakeChannels,
    /// Cancelled by [`WatcherHandle::shutdown`] and by dropping the handle.
    cancel: tokio_util::sync::CancellationToken,
    /// Trailing debounce window.
    window: std::time::Duration,
    /// Throttle of the drop notices.
    notice_interval: std::time::Duration,
    /// Bounded queue the OS callback fills.
    raw_rx: tokio::sync::mpsc::Receiver<notify::Result<notify::Event>>,
}

#[cfg(feature = "notify")]
impl IntakeLoop {
    /// Runs until cancellation or until the backend closes its channel.
    async fn run(self) {
        use std::time::Instant;

        use crate::watcher::debounce::Debouncer;
        use crate::watcher::intake::DropReporter;

        let Self {
            watcher,
            targets,
            channels,
            cancel,
            window,
            notice_interval,
            mut raw_rx,
        } = self;
        let IntakeChannels {
            publisher,
            dispatches,
            counters,
        } = channels;
        // The backend lives exactly as long as this task: dropping it here (on
        // cancellation or when the channel closes) releases every OS watch.
        let _watcher = watcher;
        let mut debouncer = Debouncer::new(window);
        let mut last_failure: Option<String> = None;
        let mut reporter = DropReporter::new(notice_interval);
        loop {
            // Announce what the bounded queues dropped since the previous notice.
            // The reporter throttles itself, so a sustained overflow gives one
            // notice per interval instead of an error storm.
            reporter.report(&publisher, &counters, Instant::now());
            // Both timers share one `select!` branch: whichever falls due first
            // (a debounce window or a throttled drop notice) wakes the loop.
            let deadline = earliest(
                debouncer.next_deadline(),
                reporter.next_deadline(&counters, Instant::now()),
            );
            tokio::select! {
                () = cancel.cancelled() => {
                    // Publish what the window already saw before leaving, and
                    // report the drops of the final window.
                    let drained = debouncer.take_all();
                    publish_events(&publisher, &dispatches, &counters, drained);
                    reporter.flush(&publisher, &counters, Instant::now());
                    tracing::debug!("workspace watcher stopped");
                    break;
                }
                incoming = raw_rx.recv() => match incoming {
                    Some(Ok(event)) => {
                        if let Some(raw) = raw_change(event) {
                            let now = Instant::now();
                            for input in crate::watcher::targets::classify(&targets, &raw) {
                                debouncer.push(input, now);
                            }
                        }
                    }
                    Some(Err(error)) => announce_failure(&publisher, &error, &mut last_failure),
                    None => {
                        // The backend dropped its sender, so no further event can
                        // arrive: flush and finish rather than waiting forever.
                        let drained = debouncer.take_all();
                        publish_events(&publisher, &dispatches, &counters, drained);
                        reporter.flush(&publisher, &counters, Instant::now());
                        tracing::debug!("workspace watcher backend closed its channel");
                        break;
                    }
                },
                () = wait_until(deadline) => {
                    let due = debouncer.flush(Instant::now());
                    publish_events(&publisher, &dispatches, &counters, due);
                }
            }
        }
        // Dropping `dispatches` here closes the dispatch queue, so the dispatcher
        // runs the events already queued and then finishes.
    }
}

/// Shortest interval between two drop notices.
///
/// A watcher configured with a tiny debounce window would otherwise announce a
/// sustained queue overflow almost per notification; the floor keeps the notice
/// rate bounded without hiding the loss (the counters are exact either way).
#[cfg(feature = "notify")]
const MIN_DROP_NOTICE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

/// Returns the earlier of two optional deadlines.
///
/// `None` means "no deadline"; when both are `None` the caller waits for an
/// event instead of a timer.
#[cfg(feature = "notify")]
fn earliest(
    first: Option<std::time::Instant>,
    second: Option<std::time::Instant>,
) -> Option<std::time::Instant> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (first, None) => first,
        (None, second) => second,
    }
}

/// One distinct path the backend registers, with the strongest mode and a label.
#[cfg(feature = "notify")]
#[derive(Debug, Clone)]
struct Registration {
    /// `true` when at least one target covering the path is a directory target.
    recursive: bool,
    /// Label of the first labelled target on the path, for error messages.
    label: Option<String>,
}

/// Collapses the plan's targets into one registration per watch path.
///
/// Several targets share a path — `IDENTITY.md` and `OPENJIUWEN.md` both watch
/// their workspace directory, for example. Registering a path twice is at best
/// redundant and at worst a backend error, so the paths are deduplicated; a
/// recursive registration wins over a non-recursive one for the same path.
#[cfg(feature = "notify")]
fn registrations(
    targets: &[crate::watcher::WatchTarget],
) -> std::collections::BTreeMap<std::path::PathBuf, Registration> {
    use std::collections::btree_map::Entry;
    use std::collections::BTreeMap;

    let mut registrations: BTreeMap<std::path::PathBuf, Registration> = BTreeMap::new();
    for target in targets {
        let path = target.watch_path().to_path_buf();
        match registrations.entry(path) {
            Entry::Occupied(mut entry) => {
                let registration = entry.get_mut();
                registration.recursive |= target.is_recursive();
                if registration.label.is_none() {
                    registration.label = target.label().map(str::to_owned);
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(Registration {
                    recursive: target.is_recursive(),
                    label: target.label().map(str::to_owned),
                });
            }
        }
    }
    registrations
}

/// Converts one `notify` event into the internal raw-change vocabulary.
///
/// `Access` events are reads, not changes, and produce `None`; an event without
/// a path cannot be attributed to a target and produces `None` as well. Every
/// other [`notify::EventKind`] maps as documented in `targets::classify`:
/// creation to `Created`, removal to `Removed`, any rename to `Renamed`, and
/// every remaining modification, `Any` and `Other` to `Modified`.
#[cfg(feature = "notify")]
fn raw_change(event: notify::Event) -> Option<crate::watcher::targets::RawChange> {
    use notify::event::ModifyKind;
    use notify::EventKind;

    use crate::watcher::targets::{RawChange, RawChangeKind};

    let notify::Event { kind, paths, .. } = event;
    let kind = match kind {
        EventKind::Access(_) => return None,
        EventKind::Create(_) => RawChangeKind::Created,
        EventKind::Remove(_) => RawChangeKind::Removed,
        EventKind::Modify(ModifyKind::Name(_)) => RawChangeKind::Renamed,
        EventKind::Modify(_) | EventKind::Any | EventKind::Other => RawChangeKind::Modified,
    };
    if paths.is_empty() {
        return None;
    }
    Some(RawChange { kind, paths })
}

/// Waits until `deadline`, or forever when nothing is pending.
///
/// A pending-free debouncer reports no deadline; the branch then has to stay
/// asleep until a raw event wakes the loop instead of spinning on `None`.
#[cfg(feature = "notify")]
async fn wait_until(deadline: Option<std::time::Instant>) {
    match deadline {
        Some(deadline) => {
            tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
        }
        None => std::future::pending::<()>().await,
    }
}

/// Publishes `events` and queues each of them for the reload dispatcher.
///
/// Publishing never blocks and fails only when nobody is subscribed, which is
/// not an error. Handing an event to the dispatcher is a `try_send` into a
/// bounded queue: a full queue (the handlers are slower than the events arrive)
/// or a closed one (the dispatcher task is gone) drops that event, counts it in
/// `counters` and is announced through a
/// [`WatchUpdate::Dropped`](crate::watcher::event::WatchUpdate::Dropped) notice
/// by the drop reporter. The function is synchronous on purpose: publication and
/// the debounce loop must never wait for a reload handler.
#[cfg(feature = "notify")]
fn publish_events(
    events: &tokio::sync::broadcast::Sender<crate::watcher::event::WatchUpdate>,
    dispatches: &tokio::sync::mpsc::Sender<crate::watcher::event::WatchEvent>,
    counters: &crate::watcher::intake::DropCounters,
    drained: Vec<crate::watcher::event::WatchEvent>,
) {
    use tokio::sync::mpsc::error::TrySendError;

    use crate::watcher::event::WatchUpdate;

    for event in drained {
        let _ = events.send(WatchUpdate::Event(event.clone()));
        if let Err(error) = dispatches.try_send(event) {
            counters.record_dispatch();
            let reason = match error {
                TrySendError::Full(_) => "the reload dispatch queue is full",
                TrySendError::Closed(_) => "the reload dispatcher is gone",
            };
            tracing::debug!(reason, "workspace watcher could not queue a reload");
        }
    }
}

/// Reports a runtime backend error under the documented quieting policy.
///
/// A failure whose paths all disappeared is the normal consequence of a removed
/// subtree: it is logged at `debug` and not published. Otherwise the failure is
/// published as
/// [`WatchUpdate::WatcherFailed`](crate::watcher::event::WatchUpdate::WatcherFailed)
/// unless it repeats the previous message, in which case only a `debug` line is
/// written.
#[cfg(feature = "notify")]
fn announce_failure(
    events: &tokio::sync::broadcast::Sender<crate::watcher::event::WatchUpdate>,
    error: &notify::Error,
    last: &mut Option<String>,
) {
    use crate::watcher::event::{WatchFailure, WatchUpdate};

    if !error.paths.is_empty() && error.paths.iter().all(|path| !path.exists()) {
        tracing::debug!(
            paths = ?error.paths,
            error = %error,
            "workspace watcher reported a path that no longer exists"
        );
        return;
    }
    let path = error.paths.first().cloned().unwrap_or_default();
    let failure = WatchFailure {
        error_kind: map_notify_error(error, &path, None)
            .kind()
            .as_str()
            .to_owned(),
        message: describe_notify_error(error, &path, None),
    };
    if last.as_deref() == Some(failure.message.as_str()) {
        tracing::debug!(
            error = %failure.message,
            "workspace watcher repeated the previous runtime failure"
        );
        return;
    }
    tracing::warn!(
        kind = %failure.error_kind,
        error = %failure.message,
        "workspace watcher backend failed"
    );
    *last = Some(failure.message.clone());
    let _ = events.send(WatchUpdate::WatcherFailed(failure));
}

/// Maps a `notify` error onto the harness error vocabulary.
///
/// | `notify::ErrorKind` | harness error |
/// | --- | --- |
/// | `PathNotFound` | [`Error::NotFound`] |
/// | `Io(PermissionDenied)` | [`Error::SysOp`] |
/// | other `Io`, `MaxFilesWatch`, `WatchNotFound` | [`Error::SysOp`] |
/// | `InvalidConfig`, `Generic`, backend construction | [`Error::Config`] |
///
/// The message always names the path and, when one is known, the target label.
#[cfg(feature = "notify")]
fn map_notify_error(error: &notify::Error, path: &std::path::Path, label: Option<&str>) -> Error {
    let message = describe_notify_error(error, path, label);
    match &error.kind {
        notify::ErrorKind::PathNotFound => Error::not_found(message),
        notify::ErrorKind::Io(io) if io.kind() == std::io::ErrorKind::PermissionDenied => {
            Error::sys_op(message)
        }
        // Any other IO failure, an exhausted inotify watch limit and an unwatch of
        // a watch that no longer exists are all state problems, not configuration
        // ones: the plan stays valid.
        notify::ErrorKind::Io(_)
        | notify::ErrorKind::MaxFilesWatch
        | notify::ErrorKind::WatchNotFound => Error::sys_op(message),
        notify::ErrorKind::InvalidConfig(_) | notify::ErrorKind::Generic(_) => {
            Error::config(message)
        }
    }
}

/// Builds the human-readable message of a `notify` error.
///
/// An empty `path` means the error has no single path (backend construction);
/// the message then names the watcher roots instead.
#[cfg(feature = "notify")]
fn describe_notify_error(
    error: &notify::Error,
    path: &std::path::Path,
    label: Option<&str>,
) -> String {
    let subject = if path.as_os_str().is_empty() {
        "the workspace watcher roots".to_owned()
    } else {
        path.display().to_string()
    };
    let label = label.unwrap_or("<unlabelled>");
    format!("watch target `{label}` cannot watch {subject}: {error}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::watcher::{WatchTarget, WatchTargetKind};
    use openjiuwen_common::error::ErrorKind;

    #[cfg(feature = "notify")]
    use crate::watcher::WatchPlan;

    #[tokio::test]
    async fn an_empty_plan_is_rejected() {
        let error = WorkspaceWatcher::new().start().await.unwrap_err();
        if cfg!(feature = "notify") {
            assert_eq!(error.kind(), ErrorKind::Config);
            assert!(
                error.to_string().contains("at least one watch target"),
                "{error}"
            );
        } else {
            assert_eq!(error.kind(), ErrorKind::Unsupported);
        }
    }

    /// A watcher with one target that resolves to `temp/<name>`.
    fn watcher_for(path: std::path::PathBuf, recursive: bool) -> WorkspaceWatcher {
        let kind = WatchTargetKind::Skills;
        let target = if recursive {
            WatchTarget::directory(kind, path)
        } else {
            WatchTarget::file(kind, path)
        };
        WorkspaceWatcher::new().watch_target(target)
    }

    #[cfg(not(feature = "notify"))]
    #[tokio::test]
    async fn the_feature_off_body_answers_unsupported() {
        let dir = tempfile::tempdir().unwrap();
        let error = watcher_for(dir.path().to_path_buf(), true)
            .start()
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("notify"), "{error}");
        assert!(error.to_string().contains("--features notify"), "{error}");
    }

    #[cfg(feature = "notify")]
    #[tokio::test]
    async fn a_missing_watch_path_is_not_found() {
        let error = watcher_for(std::path::PathBuf::from("/nonexistent/oj-watcher"), true)
            .start()
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(
            error.to_string().contains("/nonexistent/oj-watcher"),
            "{error}"
        );
    }

    #[cfg(feature = "notify")]
    #[tokio::test]
    async fn a_file_target_on_a_directory_is_a_config_error() {
        let dir = tempfile::tempdir().unwrap();
        let error = watcher_for(dir.path().to_path_buf(), false)
            .start()
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("found a directory"), "{error}");
    }

    #[cfg(feature = "notify")]
    #[test]
    fn registrations_deduplicate_paths_and_prefer_recursive() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let plan = WatchPlan::new()
            .with_target(WatchTarget::file(
                WatchTargetKind::InstructionFile,
                root.join("OPENJIUWEN.md"),
            ))
            .with_target(
                WatchTarget::file(WatchTargetKind::McpConfig, root.join("mcp.json"))
                    .with_label("mcp:workspace"),
            )
            .with_target(
                WatchTarget::directory(WatchTargetKind::Skills, root.join("skills"))
                    .with_label("skills:workspace"),
            );

        let registered = registrations(plan.targets());
        // Both file targets watch the same parent directory; the skills root is
        // its own registration.
        assert_eq!(registered.len(), 2);
        let root_registration = &registered[&root];
        assert!(!root_registration.recursive);
        assert_eq!(root_registration.label.as_deref(), Some("mcp:workspace"));
        let skills = &registered[&root.join("skills")];
        assert!(skills.recursive);
        assert_eq!(skills.label.as_deref(), Some("skills:workspace"));

        // A directory target on the same path upgrades the registration.
        let upgraded = WatchPlan::new()
            .with_target(WatchTarget::file(
                WatchTargetKind::McpConfig,
                root.join("mcp.json"),
            ))
            .with_target(WatchTarget::directory(
                WatchTargetKind::Skills,
                root.clone(),
            ));
        let registered = registrations(upgraded.targets());
        assert_eq!(registered.len(), 1);
        assert!(registered[&root].recursive);
    }

    #[cfg(feature = "notify")]
    #[test]
    fn notify_errors_map_onto_the_documented_kinds() {
        use notify::Config;

        let path = std::path::Path::new("/w/skills");
        let cases: Vec<(notify::Error, ErrorKind, &str)> = vec![
            (
                notify::Error::path_not_found(),
                ErrorKind::NotFound,
                "No path",
            ),
            (
                notify::Error::io(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "denied",
                )),
                ErrorKind::SysOp,
                "denied",
            ),
            (
                notify::Error::io(std::io::Error::other("broken")),
                ErrorKind::SysOp,
                "broken",
            ),
            (
                notify::Error::new(notify::ErrorKind::MaxFilesWatch),
                ErrorKind::SysOp,
                "watch limit",
            ),
            (
                notify::Error::watch_not_found(),
                ErrorKind::SysOp,
                "watch was found",
            ),
            (
                notify::Error::new(notify::ErrorKind::InvalidConfig(Config::default())),
                ErrorKind::Config,
                "Invalid configuration",
            ),
            (
                notify::Error::generic("backend exploded"),
                ErrorKind::Config,
                "backend exploded",
            ),
        ];
        for (notify_error, expected, fragment) in cases {
            let mapped = map_notify_error(&notify_error, path, Some("skills:workspace"));
            assert_eq!(mapped.kind(), expected, "{notify_error}");
            let message = mapped.to_string();
            assert!(message.contains(fragment), "{message}");
            assert!(message.contains("/w/skills"), "{message}");
            assert!(message.contains("skills:workspace"), "{message}");
        }
    }

    #[cfg(feature = "notify")]
    #[test]
    fn construction_errors_name_the_roots_instead_of_a_path() {
        let error = notify::Error::generic("no backend");
        let mapped = map_notify_error(&error, std::path::Path::new(""), None);
        assert_eq!(mapped.kind(), ErrorKind::Config);
        let message = mapped.to_string();
        assert!(message.contains("the workspace watcher roots"), "{message}");
        assert!(message.contains("<unlabelled>"), "{message}");
    }

    #[cfg(feature = "notify")]
    #[test]
    fn raw_change_maps_every_event_kind() {
        use notify::event::{CreateKind, DataChange, ModifyKind, RemoveKind, RenameMode};
        use notify::{Event, EventKind};

        use crate::watcher::targets::RawChangeKind;

        let path = std::path::PathBuf::from("/w/skills/a.md");
        let cases: Vec<(EventKind, Option<RawChangeKind>)> = vec![
            (EventKind::Access(notify::event::AccessKind::Any), None),
            (
                EventKind::Create(CreateKind::File),
                Some(RawChangeKind::Created),
            ),
            (
                EventKind::Create(CreateKind::Any),
                Some(RawChangeKind::Created),
            ),
            (
                EventKind::Remove(RemoveKind::File),
                Some(RawChangeKind::Removed),
            ),
            (
                EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
                Some(RawChangeKind::Renamed),
            ),
            (
                EventKind::Modify(ModifyKind::Name(RenameMode::From)),
                Some(RawChangeKind::Renamed),
            ),
            (
                EventKind::Modify(ModifyKind::Data(DataChange::Any)),
                Some(RawChangeKind::Modified),
            ),
            (
                EventKind::Modify(ModifyKind::Any),
                Some(RawChangeKind::Modified),
            ),
            (
                EventKind::Modify(ModifyKind::Other),
                Some(RawChangeKind::Modified),
            ),
            (EventKind::Any, Some(RawChangeKind::Modified)),
            (EventKind::Other, Some(RawChangeKind::Modified)),
        ];
        for (event_kind, expected) in cases {
            let event = Event::new(event_kind).add_path(path.clone());
            let raw = raw_change(event);
            assert_eq!(raw.as_ref().map(|raw| raw.kind), expected, "{event_kind:?}");
            if let Some(raw) = raw {
                assert_eq!(raw.paths, vec![path.clone()]);
            }
        }
    }

    #[cfg(feature = "notify")]
    #[test]
    fn a_rename_keeps_both_endpoints() {
        use notify::event::{ModifyKind, RenameMode};
        use notify::{Event, EventKind};

        let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
            .add_path(std::path::PathBuf::from("/w/skills/old.md"))
            .add_path(std::path::PathBuf::from("/w/skills/new.md"));
        let raw = raw_change(event).unwrap();
        assert_eq!(
            raw.paths,
            vec![
                std::path::PathBuf::from("/w/skills/old.md"),
                std::path::PathBuf::from("/w/skills/new.md"),
            ]
        );
    }

    #[cfg(feature = "notify")]
    #[test]
    fn a_pathless_event_is_dropped() {
        use notify::{Event, EventKind};

        assert!(raw_change(Event::new(EventKind::Any)).is_none());
    }

    #[cfg(feature = "notify")]
    #[test]
    fn a_backend_failure_is_published_once_and_quieted_afterwards() {
        use crate::watcher::event::WatchUpdate;

        let (events, mut receiver) = tokio::sync::broadcast::channel::<WatchUpdate>(8);
        let error = notify::Error::new(notify::ErrorKind::MaxFilesWatch);
        let mut last = None;

        announce_failure(&events, &error, &mut last);
        let first = receiver.try_recv().unwrap();
        let WatchUpdate::WatcherFailed(failure) = first else {
            panic!("expected a watcher failure, got {first:?}");
        };
        assert_eq!(failure.error_kind, "sys_op");
        assert!(
            failure.message.contains("watch limit"),
            "{}",
            failure.message
        );

        // The identical message repeats: nothing new is published.
        announce_failure(&events, &error, &mut last);
        assert!(receiver.try_recv().is_err());
    }

    #[cfg(feature = "notify")]
    #[test]
    fn a_failure_for_a_removed_subtree_is_not_published() {
        use crate::watcher::event::WatchUpdate;

        let (events, mut receiver) = tokio::sync::broadcast::channel::<WatchUpdate>(8);
        let mut last = None;
        let error = notify::Error::path_not_found()
            .add_path(std::path::PathBuf::from("/nonexistent/oj-watcher/subtree"));
        announce_failure(&events, &error, &mut last);
        assert!(receiver.try_recv().is_err());
        assert!(last.is_none());
    }
}
