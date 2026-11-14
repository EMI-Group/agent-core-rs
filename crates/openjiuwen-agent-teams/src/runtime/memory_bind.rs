//! Team-memory wiring: how the runtime resolves, mounts and drives a team's memory.
//!
//! Python parity source: `openjiuwen/agent_teams/memory/manager.py` (`TeamMemoryManager`) as the
//! team agent drives it. The engine and its runtime handle live in [`crate::memory`]; this module
//! owns the three decisions the *runtime* makes:
//!
//! 1. **Resolution** — [`resolve_memory`] decides whether a team runs with memory at all and opens
//!    the one [`TeamMemoryWiring`] the whole team shares. Precedence: an injected
//!    [`TeamBuildDeps::memory`](super::deps::TeamBuildDeps::memory) wins (a configured
//!    `metadata["memory"]` block is then logged and ignored, like an injected store), else a
//!    configured `metadata["memory"]` block opens one, else the team runs without memory. A block
//!    that is configured but says `{"enabled": false}` allocates **no** wiring, so nothing is
//!    written and no hook runs. An injected
//!    [`TeamBuildDeps::memory_observer`](super::deps::TeamBuildDeps::memory_observer) is copied
//!    onto the parsed configuration's
//!    [`observer`](crate::memory::MemoryConfig::observer) before the wiring is opened; with an
//!    injected wiring it is ignored.
//! 2. **Lifecycle hooks** — the `warn!`-and-continue pass-throughs ([`session_started`],
//!    [`member_joined`], [`member_left`], [`finish`], [`export_document`]). Memory is advisory for
//!    a run: a hook failure is logged with the team (and member) and never fails the lifecycle
//!    operation that triggered it, so a broken memory store can never block a start, a spawn, a
//!    shutdown or a stop.
//! 3. **Who is live** — [`is_live`] defines the roster rows a `stop` still has to hand a
//!    `handoff` note for.
//!
//! Mounting is *not* done here: the build path and the spawner put
//! [`TeamMemoryWiring::toolkit`] on a member's
//! [`MemberHarnessContext`](crate::harness::MemberHarnessContext) through
//! `with_memory_toolkit`, and the native harness factory turns it into the `team_memory_*` tools.
//!
//! ```text
//! resolve: injected wiring -> metadata["memory"] -> no wiring (disabled or unconfigured)
//! start   -> on_session_start   (warn! on failure)
//! spawn   -> on_member_join     (warn! on failure)
//! shutdown/stop -> on_member_leave for every live row, then flush() + export() (warn! on failure)
//! ```
use std::path::Path;
use std::sync::Arc;

use tracing::{debug, warn};

use super::deps::TeamBuildDeps;
use crate::memory::{memory_config, TeamMemoryWiring, MEMORY_METADATA_KEY};
use crate::schema::{MemberStatus, TeamAgentSpec};

/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_agent_teams::runtime::memory";

/// Resolves the team's memory wiring once per built team.
///
/// `workspace_root` is the leader's workspace, so the default `TEAM_MEMORY.md` document lives at
/// `<leader workspace>/TEAM_MEMORY.md` (a `store.dir` configuration overrides it).
///
/// A [`TeamBuildDeps::memory_observer`] injection is copied onto
/// [`MemoryConfig::observer`](crate::memory::MemoryConfig::observer) of the metadata-resolved
/// configuration, so the host's observer replaces the crate's built-in bridge; an injected wiring
/// is used verbatim and the observer is then ignored (one `debug!`).
///
/// # Errors
///
/// [`Error::Validation`](openjiuwen_common::error::Error::Validation) when a configured team id or
/// configuration does not validate, and the store's own error: memory is a *configured capability*,
/// so a wiring that cannot be opened fails the build instead of degrading silently.
pub(crate) async fn resolve_memory(
    spec: &TeamAgentSpec,
    deps: &TeamBuildDeps,
    team_name: &str,
    workspace_root: &Path,
) -> openjiuwen_common::error::Result<Option<Arc<TeamMemoryWiring>>> {
    let configured = spec
        .metadata
        .get(MEMORY_METADATA_KEY)
        .is_some_and(|value| !value.is_null());
    if let Some(injected) = deps.memory.as_ref() {
        if configured {
            warn!(
                target: LOG_TARGET,
                team = team_name,
                "the injected team memory wins over `metadata[\"memory\"]`; the metadata block is \
                 ignored"
            );
        }
        if deps.has_memory_observer() {
            debug!(
                target: LOG_TARGET,
                team = team_name,
                "the injected team memory wiring is used verbatim; the injected team-memory \
                 observer is ignored (an existing facade cannot be re-observed)"
            );
        }
        return Ok(Some(Arc::clone(injected)));
    }
    if !configured {
        return Ok(None);
    }
    let mut config = memory_config(spec);
    if !config.enabled {
        debug!(
            target: LOG_TARGET,
            team = team_name,
            "team memory is disabled; no wiring allocated"
        );
        return Ok(None);
    }
    config.observer = deps.memory_observer.clone();
    let wiring = TeamMemoryWiring::open(team_name, config, workspace_root).await?;
    debug!(
        target: LOG_TARGET,
        team = team_name,
        document = %wiring.document_path().display(),
        "opened the team's memory wiring"
    );
    Ok(Some(wiring))
}

/// Whether a roster row is a member a `stop` still has to hand a handoff note for.
///
/// A row that never started and a row that already departed are not live, so the join/leave pair
/// stays symmetric: a member only "leaves" when it "joined".
pub(crate) fn is_live(status: MemberStatus) -> bool {
    status != MemberStatus::Unstarted && !status.is_departed()
}

/// Installs the coordination session, logging (never failing) a memory error.
///
/// Called by `TeamAgent::start` before the leader runtime starts: team memory is advisory for a
/// run, so a rejected session leaves the team running without it.
pub(crate) async fn session_started(
    memory: Option<&Arc<TeamMemoryWiring>>,
    team: &str,
    session_id: &str,
) {
    let Some(wiring) = memory else {
        return;
    };
    if let Err(error) = wiring.on_session_start(session_id).await {
        warn!(
            target: LOG_TARGET,
            team,
            session = session_id,
            %error,
            "failed to install the team-memory session; continuing without it"
        );
    }
}

/// Records one member join, logging (never failing) a memory error.
pub(crate) async fn member_joined(
    memory: Option<&Arc<TeamMemoryWiring>>,
    team: &str,
    member: &str,
) {
    let Some(wiring) = memory else {
        return;
    };
    if let Err(error) = wiring.on_member_join(member).await {
        warn!(
            target: LOG_TARGET,
            team,
            member,
            %error,
            "failed to record the member join in team memory"
        );
    }
}

/// Records one member leave (a handoff note), logging (never failing) a memory error.
pub(crate) async fn member_left(memory: Option<&Arc<TeamMemoryWiring>>, team: &str, member: &str) {
    let Some(wiring) = memory else {
        return;
    };
    if let Err(error) = wiring.on_member_leave(member).await {
        warn!(
            target: LOG_TARGET,
            team,
            member,
            %error,
            "failed to record the member leave in team memory"
        );
    }
}

/// Flushes the buffered turns and, when configured, exports `TEAM_MEMORY.md`.
///
/// Both steps are best-effort and independent: a failed flush still exports, and a failed export
/// never undoes a flush. A caller may run this on `stop` and again on `snapshot`; a flush with no
/// buffered turn and an export over an unchanged store are both idempotent.
pub(crate) async fn finish(memory: Option<&Arc<TeamMemoryWiring>>, team: &str) {
    let Some(wiring) = memory else {
        return;
    };
    match wiring.flush().await {
        Ok(flush) => debug!(
            target: LOG_TARGET,
            team,
            records = flush.records.len(),
            turns = flush.turns,
            "flushed the team-memory turns"
        ),
        Err(error) => warn!(
            target: LOG_TARGET,
            team,
            %error,
            "failed to flush the team-memory turns; continuing the teardown"
        ),
    }
    export_document(Some(wiring), team).await;
}

/// Exports the team document when the configuration exports on stop, logging a failure.
///
/// The export itself is gated by [`crate::memory::MemoryConfig::export_on_stop`], so a team that
/// only wants the in-process store never writes a file.
pub(crate) async fn export_document(memory: Option<&Arc<TeamMemoryWiring>>, team: &str) {
    let Some(wiring) = memory else {
        return;
    };
    if !wiring.is_enabled() || !wiring.config().export_on_stop {
        return;
    }
    match wiring.export().await {
        Ok(Some(_)) => debug!(
            target: LOG_TARGET,
            team,
            document = %wiring.document_path().display(),
            "exported the team memory document"
        ),
        Ok(None) => debug!(
            target: LOG_TARGET,
            team,
            "no team memory document to export"
        ),
        Err(error) => warn!(
            target: LOG_TARGET,
            team,
            document = %wiring.document_path().display(),
            %error,
            "failed to export the team memory document"
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memory::{MemoryConfig, MemoryStoreConfig};

    /// A blueprint whose metadata carries the given `memory` block.
    fn spec_with(metadata: serde_json::Value) -> TeamAgentSpec {
        let mut spec = TeamAgentSpec {
            team_name: "mem-team".to_owned(),
            ..TeamAgentSpec::default()
        };
        spec.metadata
            .insert(MEMORY_METADATA_KEY.to_owned(), metadata);
        spec
    }

    /// The default dependencies: no injected wiring.
    fn deps() -> TeamBuildDeps {
        TeamBuildDeps::default()
    }

    #[tokio::test]
    async fn an_unconfigured_team_allocates_no_wiring() {
        let dir = tempfile::tempdir().unwrap();
        let spec = TeamAgentSpec {
            team_name: "mem-team".to_owned(),
            ..TeamAgentSpec::default()
        };
        assert!(resolve_memory(&spec, &deps(), "mem-team", dir.path())
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn a_configured_block_opens_one_wiring_at_the_workspace_root() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = resolve_memory(
            &spec_with(serde_json::json!({})),
            &deps(),
            "mem-team",
            dir.path(),
        )
        .await
        .unwrap()
        .expect("a configured team has memory");
        assert!(wiring.is_enabled());
        assert_eq!(
            wiring.document_path(),
            dir.path().join(crate::memory::TEAM_MEMORY_FILE)
        );
    }

    #[tokio::test]
    async fn a_disabled_block_allocates_no_wiring() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = resolve_memory(
            &spec_with(serde_json::json!({"enabled": false})),
            &deps(),
            "mem-team",
            dir.path(),
        )
        .await
        .unwrap();
        assert!(wiring.is_none());
    }

    #[tokio::test]
    async fn a_null_block_is_not_a_configuration() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_memory(
            &spec_with(serde_json::Value::Null),
            &deps(),
            "mem-team",
            dir.path()
        )
        .await
        .unwrap()
        .is_none());
    }

    #[tokio::test]
    async fn the_injected_wiring_wins_over_the_metadata_block() {
        let dir = tempfile::tempdir().unwrap();
        let injected = TeamMemoryWiring::open(
            "injected-team",
            MemoryConfig {
                store: MemoryStoreConfig::File {
                    dir: Some(dir.path().to_path_buf()),
                },
                ..MemoryConfig::default()
            },
            dir.path(),
        )
        .await
        .unwrap();
        let deps = deps().with_memory(Arc::clone(&injected));
        let wiring = resolve_memory(
            &spec_with(serde_json::json!({})),
            &deps,
            "mem-team",
            dir.path(),
        )
        .await
        .unwrap()
        .expect("the injected wiring");
        assert!(Arc::ptr_eq(&injected, &wiring));
        assert_eq!(wiring.team_id(), "injected-team");
    }

    /// A test observer that records the operations it sees, in order.
    #[derive(Default)]
    struct RecordingObserver {
        seen: parking_lot::Mutex<Vec<String>>,
    }

    impl RecordingObserver {
        fn operations(&self) -> Vec<String> {
            self.seen.lock().clone()
        }
    }

    impl openjiuwen_memory::team::TeamMemoryObserver for RecordingObserver {
        fn observe(&self, event: &openjiuwen_memory::team::TeamMemoryEvent) {
            self.seen.lock().push(event.operation.as_str().to_owned());
        }
    }

    #[tokio::test]
    async fn an_injected_observer_lands_on_the_metadata_resolved_configuration() {
        let dir = tempfile::tempdir().unwrap();
        let recorder = Arc::new(RecordingObserver::default());
        let deps = deps().with_memory_observer(
            Arc::clone(&recorder) as Arc<dyn openjiuwen_memory::team::TeamMemoryObserver>
        );
        assert!(deps.has_memory_observer());
        let wiring = resolve_memory(
            &spec_with(serde_json::json!({})),
            &deps,
            "mem-team",
            dir.path(),
        )
        .await
        .unwrap()
        .expect("a configured team has memory");
        let observer = wiring.observer().expect("the injected observer");
        assert!(Arc::ptr_eq(
            observer,
            &(Arc::clone(&recorder) as Arc<dyn openjiuwen_memory::team::TeamMemoryObserver>)
        ));
        wiring.on_member_join("dev-1").await.unwrap();
        assert_eq!(recorder.operations(), vec!["write".to_owned()]);
    }

    #[tokio::test]
    async fn an_injected_wiring_ignores_the_injected_observer() {
        let dir = tempfile::tempdir().unwrap();
        let injected = TeamMemoryWiring::open("injected-team", MemoryConfig::default(), dir.path())
            .await
            .unwrap();
        let recorder = Arc::new(RecordingObserver::default());
        let deps = deps()
            .with_memory(Arc::clone(&injected))
            .with_memory_observer(
                Arc::clone(&recorder) as Arc<dyn openjiuwen_memory::team::TeamMemoryObserver>
            );
        let wiring = resolve_memory(
            &spec_with(serde_json::json!({})),
            &deps,
            "mem-team",
            dir.path(),
        )
        .await
        .unwrap()
        .expect("the injected wiring");
        assert!(Arc::ptr_eq(&injected, &wiring));
        let observer = wiring.observer().expect("the wiring's own observer");
        assert!(
            !Arc::ptr_eq(
                observer,
                &(Arc::clone(&recorder) as Arc<dyn openjiuwen_memory::team::TeamMemoryObserver>)
            ),
            "the injected wiring keeps the observer it was opened with"
        );
        wiring.on_member_join("dev-1").await.unwrap();
        assert!(recorder.operations().is_empty());
    }

    #[test]
    fn liveness_excludes_unstarted_and_departed_rows() {
        assert!(!is_live(MemberStatus::Unstarted));
        assert!(!is_live(MemberStatus::ShutdownRequested));
        assert!(!is_live(MemberStatus::ShutDown));
        assert!(is_live(MemberStatus::Starting));
        assert!(is_live(MemberStatus::Ready));
        assert!(is_live(MemberStatus::Busy));
        assert!(is_live(MemberStatus::Paused));
        assert!(is_live(MemberStatus::Stopped));
        assert!(is_live(MemberStatus::Dormant));
    }
}
