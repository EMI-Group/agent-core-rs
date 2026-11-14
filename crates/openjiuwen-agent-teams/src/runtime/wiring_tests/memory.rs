//! Team-memory wiring: resolution, the lifecycle hooks, the toolkit and the turn seam.
//!
//! The seam under test is `TeamAgentSpec::metadata["memory"]` plus the wiring the build path
//! resolves once per team:
//!
//! | fact | effect |
//! | --- | --- |
//! | `TeamBuildDeps::with_memory` | the injected wiring wins and `has_injected_memory()` reports it |
//! | no injection, no metadata | the team runs without memory: no wiring, no toolkit, no document |
//! | `metadata["memory"]` | one wiring, its `TEAM_MEMORY.md` at the leader's workspace root |
//! | `{"enabled": false}` | no wiring at all, so nothing is written |
//! | `start` | the coordination session is installed on the hooks |
//! | `spawn_member` / `shutdown_member` | one join record, then one handoff note |
//! | a delivered message | exactly one buffered turn (the single turn seam) |
//! | `stop` / `snapshot` | flush + export of `TEAM_MEMORY.md`, best-effort and idempotent |
//! | a failing memory hook | logged, never fails `start`, a spawn, a shutdown or `stop` |
//!
//! Every test is offline and deterministic: scripted harnesses, the recording workspace provider,
//! temp directories and [`BOUND`]-limited waits.
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::record::CallOptions;
use openjiuwen_memory::team::{
    TeamExtraction, TeamMemoryExtractor, TeamMemoryFilter, TeamMemoryKind, TeamTranscript,
};
use serde_json::json;

use crate::memory::{MemoryConfig, TeamMemoryWiring, MEMORY_METADATA_KEY, TEAM_MEMORY_FILE};
use crate::runtime::deps::TeamBuildDeps;
use crate::schema::{MemberLifetime, TeamAgentSpec};

use super::{blueprint, wire, with_metadata};

/// Returns `spec` with a default `metadata["memory"]` block.
fn memory_enabled(mut spec: TeamAgentSpec) -> TeamAgentSpec {
    spec.metadata
        .insert(MEMORY_METADATA_KEY.to_owned(), json!({}));
    spec
}

/// Opens a team-memory wiring over `dir`, detached from the runtime.
async fn opened(dir: &std::path::Path, team: &str) -> Arc<TeamMemoryWiring> {
    TeamMemoryWiring::open(team, MemoryConfig::default(), dir)
        .await
        .expect("the wiring did not open")
}

/// Extractors that always fail, so the flush path can be exercised without a broker.
struct FailingExtractor;

#[async_trait]
impl TeamMemoryExtractor for FailingExtractor {
    fn name(&self) -> &'static str {
        "failing"
    }

    async fn extract(
        &self,
        _transcript: &TeamTranscript,
        _options: &CallOptions,
    ) -> Result<TeamExtraction> {
        Err(Error::config("scripted memory failure"))
    }
}

/// Returns the recorded records of `wiring`, ordered by id.
async fn records(wiring: &Arc<TeamMemoryWiring>) -> Vec<openjiuwen_memory::team::TeamMemoryRecord> {
    wiring
        .memory()
        .list(&TeamMemoryFilter::for_team(wiring.team_id()))
        .await
        .expect("the team memory store answered")
}

#[test]
fn an_injected_wiring_is_reported_by_the_dependencies() {
    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime");
    let dir = tempfile::tempdir().expect("a temp dir");
    let wiring = runtime.block_on(opened(dir.path(), "alpha"));
    let deps = TeamBuildDeps::default().with_memory(Arc::clone(&wiring));
    assert!(deps.has_injected_memory());
    assert!(!TeamBuildDeps::default().has_injected_memory());
    assert!(deps
        .memory
        .as_ref()
        .is_some_and(|held| Arc::ptr_eq(held, &wiring)));
}

#[tokio::test]
async fn an_unconfigured_team_runs_without_memory() {
    let built = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| deps).await;
    assert!(
        built.agent.memory().is_none(),
        "an unconfigured team allocates no wiring"
    );
    assert!(built.agent.spawner().memory().is_none());
    built.agent.start().await.unwrap();
    built.agent.spawn_member("dev-1").await.unwrap();
    assert!(
        built
            .factory
            .calls()
            .iter()
            .all(|ctx| ctx.memory_toolkit().is_none()),
        "no member mounts a memory toolkit without a wiring"
    );
    built.agent.stop().await.unwrap();
}

#[tokio::test]
async fn a_configured_team_opens_one_wiring_at_the_leader_workspace() {
    let built = wire(&memory_enabled(blueprint("alpha", "boss", &[])), |deps| {
        deps
    })
    .await;
    let wiring = built.agent.memory().expect("a configured team has memory");
    assert!(wiring.is_enabled());
    assert_eq!(wiring.team_id(), "alpha");
    let expected = std::env::temp_dir()
        .join("openjiuwen-agent-teams-tests")
        .join("alpha")
        .join(built.agent.session_id())
        .join("boss")
        .join(TEAM_MEMORY_FILE);
    assert_eq!(
        wiring.document_path(),
        expected,
        "the document lives at the leader's workspace root"
    );
    assert!(
        built.agent.spawner().memory().is_some(),
        "the spawner carries the resolved wiring for its teammates"
    );
}

#[tokio::test]
async fn a_disabled_block_opens_no_wiring() {
    let spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        MEMORY_METADATA_KEY,
        json!({"enabled": false}),
    );
    let built = wire(&spec, |deps| deps).await;
    assert!(built.agent.memory().is_none());
    built.agent.start().await.unwrap();
    built.agent.spawn_member("dev-1").await.unwrap();
    built.agent.stop().await.unwrap();
    assert!(
        built
            .factory
            .calls()
            .iter()
            .all(|ctx| ctx.memory_toolkit().is_none()),
        "a disabled block mounts no toolkit"
    );
}

#[tokio::test]
async fn the_injected_wiring_wins_over_a_configured_block() {
    let dir = tempfile::tempdir().unwrap();
    let injected = opened(dir.path(), "injected").await;
    let built = wire(
        &memory_enabled(blueprint("alpha", "boss", &[])),
        move |deps| deps.with_memory(Arc::clone(&injected)),
    )
    .await;
    let wiring = built.agent.memory().expect("the injected wiring");
    assert_eq!(wiring.team_id(), "injected");
    assert_eq!(
        wiring.document_path(),
        dir.path().join(TEAM_MEMORY_FILE),
        "an injected wiring keeps the document root it was opened with"
    );
}

#[tokio::test]
async fn start_installs_the_coordination_session() {
    let built = wire(&memory_enabled(blueprint("alpha", "boss", &[])), |deps| {
        deps
    })
    .await;
    let wiring = built.agent.memory().expect("a wiring");
    assert_eq!(wiring.hooks().session_id(), None);
    built.agent.start().await.unwrap();
    assert_eq!(
        wiring.hooks().session_id().as_deref(),
        Some(built.agent.session_id()),
        "start installs the team session"
    );
}

#[tokio::test]
async fn spawning_and_shutting_down_a_member_records_its_presence() {
    let built = wire(
        &memory_enabled(blueprint("alpha", "boss", &["dev-1"])),
        |deps| deps,
    )
    .await;
    let wiring = built.agent.memory().expect("a wiring");
    built.agent.start().await.unwrap();
    built.agent.spawn_member("dev-1").await.unwrap();
    assert!(wiring.hooks().members().contains(&"dev-1".to_owned()));
    assert!(
        records(&wiring).await.iter().any(|record| {
            record.kind == TeamMemoryKind::MemberProfile
                && record.member_id.as_deref() == Some("dev-1")
        }),
        "the join wrote one member profile"
    );
    built
        .agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, false)
        .await
        .unwrap();
    assert!(!wiring.hooks().members().contains(&"dev-1".to_owned()));
    assert!(
        records(&wiring).await.iter().any(|record| {
            record.kind == TeamMemoryKind::HandoffNote
                && record.member_id.as_deref() == Some("dev-1")
        }),
        "the shutdown wrote one handoff note"
    );
}

#[tokio::test]
async fn a_failed_spawn_records_no_join() {
    let built = wire(
        &memory_enabled(blueprint("alpha", "boss", &["dev-1"])),
        |deps| deps,
    )
    .await;
    let wiring = built.agent.memory().expect("a wiring");
    built.factory.failing_for("dev-1");
    built.agent.start().await.unwrap();
    assert!(built.agent.spawn_member("dev-1").await.is_err());
    assert!(
        !wiring.hooks().members().contains(&"dev-1".to_owned()),
        "a member that never became live never joined"
    );
}

#[tokio::test]
async fn a_delivered_message_reaches_the_turn_buffer_exactly_once() {
    let built = wire(
        &memory_enabled(blueprint("alpha", "boss", &["dev-1"])),
        |deps| deps,
    )
    .await;
    let wiring = built.agent.memory().expect("a wiring");
    built.agent.start().await.unwrap();
    built.agent.spawn_member("dev-1").await.unwrap();
    assert_eq!(wiring.hooks().pending_turns(), 0);
    built
        .agent
        .send_message(Some("dev-1"), "please take the release checklist")
        .await
        .unwrap();
    assert_eq!(
        wiring.hooks().pending_turns(),
        1,
        "one direct message buffers one turn"
    );
    built
        .agent
        .send_message(None, "stand-up in five minutes")
        .await
        .unwrap();
    assert_eq!(
        wiring.hooks().pending_turns(),
        2,
        "a broadcast buffers one turn too"
    );
}

#[tokio::test]
async fn stop_flushes_the_turns_and_writes_the_document() {
    let built = wire(
        &memory_enabled(blueprint("alpha", "boss", &["dev-1"])),
        |deps| deps,
    )
    .await;
    let wiring = built.agent.memory().expect("a wiring");
    built.agent.start().await.unwrap();
    built.agent.spawn_member("dev-1").await.unwrap();
    built
        .agent
        .send_message(Some("dev-1"), "please take the release checklist")
        .await
        .unwrap();
    assert_eq!(wiring.hooks().pending_turns(), 1);
    built.agent.stop().await.unwrap();
    assert_eq!(wiring.hooks().pending_turns(), 0, "stop flushed the turns");
    let document = wiring.document_path();
    assert!(document.exists(), "stop wrote {document:?}");
    let markdown = tokio::fs::read_to_string(document).await.unwrap();
    assert!(markdown.contains("TEAM_MEMORY") || markdown.contains("alpha"));
    // A second teardown is idempotent: it flushes nothing new and stores no new record (the
    // document is re-rendered, so its `generated_at` header moves while its entries stay put).
    let before: Vec<String> = records(&built.agent.memory().expect("a wiring"))
        .await
        .iter()
        .map(|record| record.id.clone())
        .collect();
    built.agent.stop().await.unwrap();
    assert_eq!(wiring.hooks().pending_turns(), 0);
    let after: Vec<String> = records(&wiring)
        .await
        .iter()
        .map(|record| record.id.clone())
        .collect();
    assert_eq!(before, after, "a repeated stop adds no record");
    assert!(document.exists(), "the document survives a repeated stop");
    if let Some(team_dir) = document.parent().and_then(|root| root.parent()) {
        let _ = tokio::fs::remove_dir_all(team_dir).await;
    }
}

#[tokio::test]
async fn a_snapshot_exports_the_document_without_failing() {
    let built = wire(
        &memory_enabled(blueprint("alpha", "boss", &["dev-1"])),
        |deps| deps,
    )
    .await;
    let wiring = built.agent.memory().expect("a wiring");
    built.agent.spawn_member("dev-1").await.unwrap();
    let snapshot = built.agent.snapshot().await;
    assert_eq!(snapshot.team_name, "alpha");
    assert_eq!(snapshot.members.len(), 2, "leader plus dev-1");
    assert!(
        wiring.document_path().exists(),
        "the snapshot exported the team document"
    );
    if let Some(team_dir) = wiring
        .document_path()
        .parent()
        .and_then(|root| root.parent())
    {
        let _ = tokio::fs::remove_dir_all(team_dir).await;
    }
}

#[tokio::test]
async fn a_failing_memory_hook_cannot_fail_stop() {
    let dir = tempfile::tempdir().unwrap();
    let failing = opened(dir.path(), "alpha")
        .await
        .with_extractor(Arc::new(FailingExtractor));
    let built = wire(
        &memory_enabled(blueprint("alpha", "boss", &["dev-1"])),
        move |deps| deps.with_memory(failing),
    )
    .await;
    let wiring = built.agent.memory().expect("the injected wiring");
    built.agent.start().await.unwrap();
    built.agent.spawn_member("dev-1").await.unwrap();
    built
        .agent
        .send_message(Some("dev-1"), "please take the release checklist")
        .await
        .unwrap();
    built.agent.stop().await.unwrap();
    assert!(
        !built.agent.is_started(),
        "the team stopped despite the failure"
    );
    assert_eq!(
        wiring.hooks().pending_turns(),
        1,
        "a failed flush keeps its turns for a retry"
    );
}

#[tokio::test]
async fn the_member_contexts_carry_the_team_toolkit() {
    let built = wire(
        &memory_enabled(blueprint("alpha", "boss", &["dev-1"])),
        |deps| deps,
    )
    .await;
    let wiring = built.agent.memory().expect("a wiring");
    built.agent.spawn_member("dev-1").await.unwrap();
    let calls = built.factory.calls();
    let leader = calls
        .iter()
        .find(|ctx| ctx.member_name == "boss")
        .expect("the leader harness was built");
    let teammate = calls
        .iter()
        .find(|ctx| ctx.member_name == "dev-1")
        .expect("the teammate harness was built");
    for (ctx, name) in [(leader, "boss"), (teammate, "dev-1")] {
        let mounted = ctx
            .memory_toolkit()
            .unwrap_or_else(|| panic!("{name} mounts the team toolkit"));
        let expected = wiring.toolkit(name, None).expect("a toolkit");
        assert!(
            Arc::ptr_eq(mounted, &expected),
            "{name} mounts the team's own toolkit"
        );
    }
}
