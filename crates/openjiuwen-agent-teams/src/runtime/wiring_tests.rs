//! The wiring suite: one group per metadata-driven seam of the runtime.
//!
//! Each group drives a real build (or a real spawner) over the offline fakes of
//! [`crate::runtime::team_agent::test_support`] and asserts the seam the build path is supposed to
//! close:
//!
//! | group | wiring point |
//! | --- | --- |
//! | [`spawn_modes`] | `metadata["spawn"]` → process / external-CLI / SSH spawner, in-process default |
//! | [`routing`] | `TeamModelRouter` → the `DeepAgentSpec.model` the harness factory receives |
//! | [`kv_affinity`] | KV-cache register → env → release across a member's lifecycle, forks included |
//! | [`signals`] | observability hooks + monitor events for every lifecycle fact |
//! | [`organization`] | `metadata["organization"]` → chart, org task events, mail ack, delegation |
//! | [`workspace`] | `metadata["workspace"]` → `IsolationMode`, `None` by default |
//! | [`attach`] | `metadata["attach"]` / env → a bound Unix socket served by the backend handler |
//! | [`checkpoints`] | `metadata["checkpoint"]` → team store, member identity, turn/stop capture |
//! | [`memory`] | `metadata["memory"]` → wiring, lifecycle hooks, member toolkits, turn seam, export |
//!
//! Everything here is offline, deterministic and bounded: scripted harnesses, recording
//! workspace providers, in-memory stores, temp directories and [`BOUND`]-limited waits.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::time::Instant;

use crate::monitor::{MonitorEvent, MonitorReceiver};
use crate::runtime::deps::TeamBuildDeps;
use crate::runtime::team_agent::{SpecBuild, TeamAgent};
use crate::schema::TeamAgentSpec;

pub(crate) use crate::runtime::team_agent::test_support::{
    blueprint, deps, RecordingWorkspaceProvider, ScriptedFactory,
};

mod attach;
mod checkpoints;
mod kv_affinity;
mod memory;
mod organization;
mod routing;
mod signals;
mod spawn_modes;
mod workspace;

/// Upper bound for every single wait in this suite.
pub(crate) const BOUND: Duration = Duration::from_secs(2);

/// Poll interval of [`wait_until`].
const POLL: Duration = Duration::from_millis(5);

/// One built team plus the fakes its dependencies were built over.
pub(crate) struct Wiring {
    /// The leader handle the build produced.
    pub(crate) agent: Arc<TeamAgent>,
    /// Harness factory: records one context per built member.
    pub(crate) factory: Arc<ScriptedFactory>,
    /// Workspace provider: records one allocation per member.
    pub(crate) workspace: Arc<RecordingWorkspaceProvider>,
}

/// Builds `spec` over fresh scripted fakes, letting `configure` adjust the dependencies.
pub(crate) async fn wire(
    spec: &TeamAgentSpec,
    configure: impl FnOnce(TeamBuildDeps) -> TeamBuildDeps,
) -> Wiring {
    let factory = ScriptedFactory::new();
    let workspace = RecordingWorkspaceProvider::new();
    let deps = configure(deps(&factory, &workspace));
    let agent = spec.build_with(deps).await.expect("the team did not build");
    Wiring {
        agent,
        factory,
        workspace,
    }
}

/// Returns `spec` with one metadata entry inserted.
pub(crate) fn with_metadata(mut spec: TeamAgentSpec, key: &str, value: Value) -> TeamAgentSpec {
    spec.metadata.insert(key.to_owned(), value);
    spec
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

/// Collects everything already queued on `receiver`.
pub(crate) fn take_monitor(receiver: &mut MonitorReceiver) -> Vec<MonitorEvent> {
    let mut events = Vec::new();
    while let Some(stamped) = receiver.try_recv() {
        events.push(stamped.event);
    }
    events
}

/// Returns the wire kind of every event, in order.
pub(crate) fn monitor_kinds(events: &[MonitorEvent]) -> Vec<&'static str> {
    events.iter().map(MonitorEvent::kind).collect()
}

/// Returns the position of the first event whose kind is `kind`.
pub(crate) fn position_of(events: &[MonitorEvent], kind: &str) -> Option<usize> {
    events.iter().position(|event| event.kind() == kind)
}
