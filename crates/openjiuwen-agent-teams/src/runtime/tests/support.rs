//! Shared helpers for the runtime integration suite.
//!
//! The fakes are *re-used*, not re-implemented: [`ScriptedFactory`],
//! [`RecordingWorkspaceProvider`], [`ScriptedHarness`], [`blueprint`] and [`deps`] come from
//! `crate::runtime::team_agent::test_support`, so the crossed-seam suites drive the same offline
//! harnesses the unit tests do.
//!
//! What lives here is only the wiring and the observation helpers a suite needs on top of them:
//! a [`Fixture`] (one manager over fresh fakes and its own in-memory store), a bus
//! [`Recorder`](subscribe_observer), a bounded stream [`drain`] and a bounded [`wait_until`].

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use futures::stream::{BoxStream, StreamExt};
use openjiuwen_agent::AgentChunk;
use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use tokio::time::{timeout, Instant};

use crate::messager::{EventSink, Messager};
use crate::runtime::deps::TeamBuildDeps;
use crate::runtime::manager::TeamRuntimeManager;
use crate::runtime::team_agent::test_support::{deps, ScriptedFactory, ScriptedHarness};
use crate::runtime::team_agent::TeamAgent;
use crate::schema::{is_team_event_marker, EventMessage, TeamAgentSpec, TeamTopic};
use crate::storage::{InMemoryTeamStore, TeamStore};

pub(crate) use crate::runtime::team_agent::test_support::{blueprint, RecordingWorkspaceProvider};

/// Upper bound for any single wait in these suites.
pub(crate) const BOUND: Duration = Duration::from_secs(2);

/// How long the one observation window of the open-task suite lasts.
pub(crate) const WINDOW: Duration = Duration::from_millis(150);

/// Poll interval of [`wait_until`].
const POLL: Duration = Duration::from_millis(5);

/// One runtime manager over fresh scripted fakes, its in-memory store and the two fakes.
///
/// The manager owns the dependencies; the fakes and the store are kept here as well because every
/// suite asserts against what the harnesses received, which workspace roots were handed out and
/// which rows the store holds.
pub(crate) struct Fixture {
    /// The manager under test, shared so the global-registry suite can install it.
    pub(crate) manager: Arc<TeamRuntimeManager>,
    /// Harness factory: records one scripted harness per member.
    pub(crate) factory: Arc<ScriptedFactory>,
    /// Workspace provider: records one allocation per member.
    pub(crate) workspace: Arc<RecordingWorkspaceProvider>,
    /// The store injected through [`TeamBuildDeps::with_store`].
    pub(crate) store: Arc<InMemoryTeamStore>,
}

impl Fixture {
    /// Builds a fixture: one scripted factory, one recording workspace, one in-memory store.
    pub(crate) fn new() -> Self {
        let factory = ScriptedFactory::new();
        let workspace = RecordingWorkspaceProvider::new();
        let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
        let deps: TeamBuildDeps =
            deps(&factory, &workspace).with_store(Arc::clone(&store) as Arc<dyn TeamStore>);
        Self {
            manager: Arc::new(TeamRuntimeManager::new(deps)),
            factory,
            workspace,
            store,
        }
    }

    /// Activates `spec` for `session` through the manager and returns the live team handle.
    pub(crate) async fn activate(&self, spec: &TeamAgentSpec, session: &str) -> Arc<TeamAgent> {
        self.manager
            .activate(spec, session, &serde_json::Value::Null)
            .await
            .expect("the team did not activate")
    }

    /// Returns the harness the factory built for `member`.
    pub(crate) fn harness(&self, member: &str) -> Arc<ScriptedHarness> {
        self.factory
            .harness(member)
            .unwrap_or_else(|| panic!("no harness was built for '{member}'"))
    }

    /// Returns the texts `member`'s harness received, in order.
    pub(crate) fn sends(&self, member: &str) -> Vec<String> {
        self.harness(member).sends()
    }
}

/// Events recorded by one observing bus sink.
pub(crate) type Recorder = Arc<Mutex<Vec<EventMessage>>>;

/// Subscribes one recording observer to `topics` on the team's own transport.
pub(crate) fn subscribe_observer(messager: &Arc<dyn Messager>, topics: &[TeamTopic]) -> Recorder {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let sink: EventSink = {
        let recorder = Arc::clone(&recorder);
        Arc::new(move |event: EventMessage| recorder.lock().push(event))
    };
    messager
        .subscribe("observer", topics, sink)
        .expect("the observer could not subscribe");
    recorder
}

/// Returns a snapshot of the recorded events.
pub(crate) fn recorded(recorder: &Recorder) -> Vec<EventMessage> {
    recorder.lock().clone()
}

/// Returns the recorded event types, in order.
pub(crate) fn event_types(recorder: &Recorder) -> Vec<String> {
    recorded(recorder)
        .iter()
        .map(|event| event.event_type.clone())
        .collect()
}

/// Drains `stream` to its end, failing when it does not finish within [`BOUND`].
pub(crate) async fn drain(
    mut stream: BoxStream<'static, Result<AgentChunk>>,
) -> Vec<Result<AgentChunk>> {
    timeout(BOUND, async move {
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        items
    })
    .await
    .expect("the run did not end within the bound")
}

/// Returns how many frames of `frames` are team event markers.
pub(crate) fn marker_count(frames: &[Result<AgentChunk>]) -> usize {
    frames
        .iter()
        .filter(|item| {
            item.as_ref()
                .is_ok_and(|chunk| is_team_event_marker(&chunk.payload))
        })
        .count()
}

/// Polls `predicate` until it holds, failing after [`BOUND`].
pub(crate) async fn wait_until(what: &str, mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + BOUND;
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
    assert!(predicate(), "timed out waiting for {what}");
}
