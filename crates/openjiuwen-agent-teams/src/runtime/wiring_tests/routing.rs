//! `TeamModelRouter` → the model identity the harness factory receives for each member.
//!
//! The router is built once at team build from `model_pool`/`model_pool_strategy` and shared with
//! the spawner, so the pool entry a member resolves to rewrites that member's `DeepAgentSpec.model`
//! before the harness factory builds its client. These tests pin that rewrite at the factory seam
//! (a local recording factory), the roster reference it leaves behind, and the router's own
//! selection rules.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_harness_protocol::HarnessProtocol;

use crate::harness::{MemberHarnessContext, MemberHarnessFactory};
use crate::member::{MemberSpawnRequest, MemberSpawner};
use crate::runtime::routing::{ModelSelection, TeamModelRouter};
use crate::runtime::spawner::TeamMemberSpawner;
use crate::runtime::team_agent::{SpecBuild, TeamAgent};
use crate::schema::{
    DeepAgentSpec, ModelPoolEntry, ModelPoolStrategy, ModelRef, TeamAgentSpec, TeamRole,
};

use super::{blueprint, deps, RecordingWorkspaceProvider, ScriptedFactory};

/// Two endpoints: `m0` on `openai`, `m1` on `anthropic`.
fn entries() -> Vec<ModelPoolEntry> {
    vec![
        ModelPoolEntry::new("m0", "openai", "https://m0.example", "key-0"),
        ModelPoolEntry::new("m1", "anthropic", "https://m1.example", "key-1"),
    ]
}

/// Returns a blueprint that declares the two-entry pool under `strategy`.
fn pooled(strategy: ModelPoolStrategy) -> TeamAgentSpec {
    let mut spec = blueprint("alpha", "boss", &["dev-1", "dev-2"]);
    spec.model_pool = entries();
    spec.model_pool_strategy = strategy;
    spec
}

/// Harness factory that records the spec every member's brain is built from.
struct RecordingFactory {
    /// The scripted factory every call is forwarded to.
    inner: Arc<ScriptedFactory>,
    /// `(spec, context)` of every `create` call, in order.
    calls: Mutex<Vec<(DeepAgentSpec, MemberHarnessContext)>>,
}

impl RecordingFactory {
    /// Wraps `inner`, recording every build.
    fn new(inner: Arc<ScriptedFactory>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            calls: Mutex::new(Vec::new()),
        })
    }

    /// Returns every spec `member` was built with, in build order.
    fn specs(&self, member: &str) -> Vec<DeepAgentSpec> {
        self.calls
            .lock()
            .iter()
            .filter(|(_, ctx)| ctx.member_name == member)
            .map(|(spec, _)| spec.clone())
            .collect()
    }

    /// Returns the spec `member` was built with.
    fn spec(&self, member: &str) -> Option<DeepAgentSpec> {
        self.specs(member).into_iter().next()
    }
}

#[async_trait]
impl MemberHarnessFactory for RecordingFactory {
    async fn create(
        &self,
        spec: &DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>> {
        self.calls.lock().push((spec.clone(), ctx.clone()));
        self.inner.create(spec, ctx).await
    }

    fn name(&self) -> &'static str {
        "recording"
    }
}

/// Builds `spec` over a recording factory and returns it with its scripted inner factory.
async fn build_recording(
    spec: &TeamAgentSpec,
) -> (Arc<TeamAgent>, Arc<RecordingFactory>, Arc<ScriptedFactory>) {
    let inner = ScriptedFactory::new();
    let factory = RecordingFactory::new(Arc::clone(&inner));
    let workspace = RecordingWorkspaceProvider::new();
    let build_deps = deps(&inner, &workspace)
        .with_harness_factory(Arc::clone(&factory) as Arc<dyn MemberHarnessFactory>);
    let agent = spec
        .build_with(build_deps)
        .await
        .expect("the team did not build");
    (agent, factory, inner)
}

/// Returns the model name of `member`'s recorded spec.
fn model_name(factory: &RecordingFactory, member: &str) -> Option<String> {
    factory
        .spec(member)
        .and_then(|spec| spec.model)
        .and_then(|model| model.model_name)
}

#[tokio::test]
async fn a_pool_entry_rewrites_the_member_spec_and_the_roster_row() {
    let (agent, factory, _inner) = build_recording(&pooled(ModelPoolStrategy::RoundRobin)).await;
    agent
        .spawn_member("dev-1")
        .await
        .expect("the pooled member did not spawn");

    let leader = factory
        .spec("boss")
        .and_then(|spec| spec.model)
        .expect("the leader model");
    assert_eq!(leader.model_name.as_deref(), Some("m0"));
    assert_eq!(leader.provider.as_deref(), Some("openai"));

    let teammate = factory
        .spec("dev-1")
        .and_then(|spec| spec.model)
        .expect("the teammate model");
    // The allocator hands out one entry per member: the leader took `m0`, so the first teammate in
    // roster order rotates onto `m1`.
    assert_eq!(teammate.model_name.as_deref(), Some("m1"));
    assert_eq!(teammate.provider.as_deref(), Some("anthropic"));

    // The roster row and the harness spec come from the same allocation, for every member.
    for (member, expected) in [("boss", "m0"), ("dev-1", "m1")] {
        assert_eq!(
            model_name(&factory, member).as_deref(),
            Some(expected),
            "the harness spec of {member}"
        );
        assert_eq!(
            agent
                .backend()
                .member(member)
                .and_then(|row| row.record().model_ref.clone())
                .as_deref(),
            Some(expected),
            "the roster row of {member} disagrees with its harness"
        );
    }
}

#[tokio::test]
async fn an_empty_pool_leaves_the_member_spec_untouched() {
    let spec = blueprint("alpha", "boss", &["dev-1"]);
    let (agent, factory, _inner) = build_recording(&spec).await;

    assert_eq!(factory.spec("boss").and_then(|spec| spec.model), None);
    assert_eq!(
        agent
            .backend()
            .member("boss")
            .map(|row| row.record().model_ref),
        Some(None)
    );

    agent
        .spawn_member("dev-1")
        .await
        .expect("an unpolled member must still spawn");
    assert_eq!(factory.spec("dev-1").and_then(|spec| spec.model), None);
    assert_eq!(
        agent
            .backend()
            .member("dev-1")
            .map(|row| row.record().model_ref),
        Some(None)
    );
}

#[tokio::test]
async fn by_name_routes_the_requested_entry_and_falls_back_for_unknown_names() {
    let mut spec = pooled(ModelPoolStrategy::ByModelName);
    let requested = |name: &str| ModelRef {
        model_name: Some(name.to_owned()),
        ..ModelRef::default()
    };
    spec.agents
        .get_mut("dev-1")
        .expect("the dev-1 blueprint")
        .model = Some(requested("m1"));
    spec.agents
        .get_mut("dev-2")
        .expect("the dev-2 blueprint")
        .model = Some(requested("ghost"));
    let (agent, factory, _inner) = build_recording(&spec).await;
    agent
        .spawn_member("dev-1")
        .await
        .expect("dev-1 did not spawn");
    agent
        .spawn_member("dev-2")
        .await
        .expect("dev-2 did not spawn");

    assert_eq!(model_name(&factory, "dev-1").as_deref(), Some("m1"));
    let teammate = factory
        .spec("dev-1")
        .and_then(|spec| spec.model)
        .expect("dev-1 model");
    assert_eq!(teammate.provider.as_deref(), Some("anthropic"));
    // An unknown requested name is not an error: the router serves the first entry.
    assert_eq!(model_name(&factory, "dev-2").as_deref(), Some("m0"));
    // The leader requests nothing, so the by-name strategy also serves the first entry.
    assert_eq!(model_name(&factory, "boss").as_deref(), Some("m0"));
    assert_eq!(
        agent
            .backend()
            .member("dev-1")
            .map(|row| row.record().model_ref),
        Some(Some("m1".to_owned()))
    );
}

#[test]
fn select_honours_the_requested_name_under_by_name() {
    let router = TeamModelRouter::new(&entries(), ModelPoolStrategy::ByModelName).expect("router");
    let selection = router
        .select(Some("m1"), TeamRole::Teammate)
        .expect("the requested entry");
    assert_eq!(selection.model_name(), "m1");
    assert_eq!(selection.allocation.index, 1);
    assert_eq!(
        selection.entry.map(|entry| entry.api_provider).as_deref(),
        Some("anthropic")
    );

    let unknown = router
        .select(Some("ghost"), TeamRole::Teammate)
        .expect("an unknown name falls back");
    assert_eq!(unknown.model_name(), "m0");
}

#[test]
fn select_rotates_the_round_robin_entry_on_every_call() {
    let router = TeamModelRouter::new(&entries(), ModelPoolStrategy::RoundRobin).expect("router");
    let selections: Vec<_> = (0..3)
        .map(|_| {
            router
                .select(None, TeamRole::Teammate)
                .expect("a pooled selection")
        })
        .collect();
    let indices: Vec<usize> = selections.iter().map(|sel| sel.allocation.index).collect();
    assert_eq!(indices, vec![0, 1, 0], "round-robin rotates per allocation");
    let names: Vec<&str> = selections.iter().map(ModelSelection::model_name).collect();
    assert_eq!(names, vec!["m0", "m1", "m0"]);
    // Round-robin ignores a requested name by contract: the rotation decides, here `m1`.
    let requested = router
        .select(Some("m0"), TeamRole::Teammate)
        .expect("a pooled selection");
    assert_eq!(requested.allocation.index, 1);
    assert_eq!(requested.model_name(), "m1");
    // By-name honours the same request (the strategy difference is documented on the allocator).
    let by_name = TeamModelRouter::new(&entries(), ModelPoolStrategy::ByModelName).expect("router");
    assert_eq!(
        by_name
            .select(Some("m1"), TeamRole::Teammate)
            .expect("a pooled selection")
            .model_name(),
        "m1"
    );
}

#[test]
fn select_for_allocates_one_endpoint_per_member() {
    let router = TeamModelRouter::new(&entries(), ModelPoolStrategy::RoundRobin).expect("router");
    let first = router
        .select_for("dev-1", None, TeamRole::Teammate)
        .expect("a pooled selection");
    let second = router
        .select_for("dev-1", None, TeamRole::Teammate)
        .expect("the memoized selection");
    assert_eq!(first, second, "a member is allocated exactly once");
    assert_eq!(first.model_name(), "m0");

    let other = router
        .select_for("dev-2", None, TeamRole::Teammate)
        .expect("the next member");
    assert_eq!(
        other.model_name(),
        "m1",
        "the cursor moved on to the next member"
    );

    // An empty pool without a requested name is the unrouted selection, cached like any other.
    let empty = TeamModelRouter::empty();
    let unrouted = empty
        .select_for("boss", None, TeamRole::Leader)
        .expect("the unrouted selection");
    assert_eq!(unrouted.allocation.model_name, "");
    assert_eq!(
        empty
            .select_for("boss", None, TeamRole::Leader)
            .expect("the memoized selection"),
        unrouted
    );
}

#[tokio::test]
async fn each_member_keeps_its_endpoint_across_respawns() {
    let (agent, factory, _inner) = build_recording(&pooled(ModelPoolStrategy::RoundRobin)).await;
    agent
        .spawn_member("dev-1")
        .await
        .expect("dev-1 did not spawn");
    agent
        .spawn_member("dev-2")
        .await
        .expect("dev-2 did not spawn");

    // Two entries over three members: the documented rotation, in roster order (the leader first,
    // then the teammates sorted by name).
    for (member, expected) in [("boss", "m0"), ("dev-1", "m1"), ("dev-2", "m0")] {
        assert_eq!(
            model_name(&factory, member).as_deref(),
            Some(expected),
            "the harness spec of {member}"
        );
        assert_eq!(
            agent
                .backend()
                .member(member)
                .and_then(|row| row.record().model_ref.clone())
                .as_deref(),
            Some(expected),
            "the roster row of {member} disagrees with its harness"
        );
    }

    let first = factory.spec("dev-1").expect("dev-1's first harness");
    // Materialize dev-1's harness a second time through the same spawner: the memo must hand back
    // the endpoint the member was allocated at build time, not the next round-robin slot.
    agent
        .spawner()
        .spawn(MemberSpawnRequest::new(
            agent.team_name(),
            "dev-1",
            agent.session_id(),
            std::env::temp_dir(),
        ))
        .await
        .expect("dev-1 did not respawn");

    let specs = factory.specs("dev-1");
    assert_eq!(specs.len(), 2, "one harness per spawn: {specs:?}");
    assert_eq!(
        specs[0], specs[1],
        "a respawn rebuilt the member for a different model"
    );
    assert_eq!(specs[1], first);
    assert_eq!(
        specs[1]
            .model
            .as_ref()
            .and_then(|model| model.model_name.as_deref()),
        Some("m1")
    );
    assert_eq!(
        agent
            .backend()
            .member("dev-1")
            .and_then(|row| row.record().model_ref)
            .as_deref(),
        Some("m1")
    );
}

#[test]
fn effective_spec_on_an_empty_pool_echoes_the_requested_name() {
    let router = TeamModelRouter::empty();
    let spec = DeepAgentSpec::default();

    let (effective, selection) = router
        .effective_spec(&spec, Some("gpt-x"), TeamRole::Leader)
        .expect("an echo selection");
    assert_eq!(selection.allocation.model_name, "gpt-x");
    assert!(selection.entry.is_none());
    assert_eq!(
        effective.model, spec.model,
        "an empty pool must not rewrite the spec"
    );

    let (effective, selection) = router
        .effective_spec(&spec, None, TeamRole::Leader)
        .expect("the unrouted selection");
    assert_eq!(selection.allocation.model_name, "");
    assert_eq!(effective.model, spec.model);
}

#[test]
fn effective_spec_preserves_the_spec_params() {
    let router = TeamModelRouter::new(&entries(), ModelPoolStrategy::RoundRobin).expect("router");
    let mut params = std::collections::BTreeMap::new();
    params.insert("temperature".to_owned(), serde_json::json!(0.2));
    let spec = DeepAgentSpec {
        model: Some(ModelRef {
            model_name: Some("declared".to_owned()),
            provider: Some("byo".to_owned()),
            params,
            ..ModelRef::default()
        }),
        ..DeepAgentSpec::default()
    };

    let (effective, selection) = router
        .effective_spec(&spec, None, TeamRole::Teammate)
        .expect("a pooled selection");
    let model = effective.model.expect("the rewritten model");
    assert_eq!(model.model_name.as_deref(), Some("m0"));
    assert_eq!(model.provider.as_deref(), Some("openai"));
    assert_eq!(
        model.params.get("temperature"),
        Some(&serde_json::json!(0.2)),
        "the pool entry replaces the identity, not the request parameters"
    );
    assert!(selection.entry.is_some());
}

#[tokio::test]
async fn a_directly_built_spawner_uses_the_injected_router() {
    let inner = ScriptedFactory::new();
    let factory = RecordingFactory::new(Arc::clone(&inner));
    let workspace = RecordingWorkspaceProvider::new();
    let routing = Arc::new(
        TeamModelRouter::new(
            &[ModelPoolEntry::new(
                "m9",
                "local",
                "https://m9.example",
                "key-9",
            )],
            ModelPoolStrategy::RoundRobin,
        )
        .expect("the injected router"),
    );
    let build_deps = deps(&inner, &workspace)
        .with_harness_factory(Arc::clone(&factory) as Arc<dyn MemberHarnessFactory>)
        .with_routing(Arc::clone(&routing));
    let spawner = TeamMemberSpawner::new(
        blueprint("alpha", "boss", &["dev-1"]),
        build_deps,
        "session-1",
    );
    assert!(
        Arc::ptr_eq(spawner.routing(), &routing),
        "the spawner must keep the injected router"
    );

    spawner
        .spawn(MemberSpawnRequest::new(
            "alpha",
            "dev-1",
            "session-1",
            std::env::temp_dir(),
        ))
        .await
        .expect("the member did not spawn");
    assert_eq!(model_name(&factory, "dev-1").as_deref(), Some("m9"));
}

#[tokio::test]
async fn build_with_replaces_an_injected_router_with_the_spec_pool() {
    let inner = ScriptedFactory::new();
    let factory = RecordingFactory::new(Arc::clone(&inner));
    let workspace = RecordingWorkspaceProvider::new();
    let routing = Arc::new(
        TeamModelRouter::new(
            &[ModelPoolEntry::new(
                "m9",
                "local",
                "https://m9.example",
                "key-9",
            )],
            ModelPoolStrategy::RoundRobin,
        )
        .expect("the injected router"),
    );
    let build_deps = deps(&inner, &workspace)
        .with_harness_factory(Arc::clone(&factory) as Arc<dyn MemberHarnessFactory>)
        .with_routing(Arc::clone(&routing));

    let agent = pooled(ModelPoolStrategy::RoundRobin)
        .build_with(build_deps)
        .await
        .expect("the team did not build");
    assert_eq!(
        model_name(&factory, "boss").as_deref(),
        Some("m0"),
        "build_with derives its router from the spec pool, not from the injected one"
    );
    assert!(!Arc::ptr_eq(agent.spawner().routing(), &routing));
}

#[tokio::test]
async fn an_intelli_router_spec_deserializes_and_the_build_refuses_it() {
    // A Python-format spec carrying the `intelli_router` strategy must parse, and the build path
    // must then refuse it with a typed error instead of silently rotating the pool.
    let wire = serde_json::to_value(pooled(ModelPoolStrategy::IntelliRouter)).expect("wire form");
    assert_eq!(
        wire["model_pool_strategy"],
        serde_json::json!("intelli_router")
    );
    assert_eq!(
        wire["model_pool"].as_array().map(Vec::len),
        Some(2),
        "the pool is declared, so the refusal cannot come from an empty pool"
    );
    let spec: TeamAgentSpec = serde_json::from_value(wire).expect("the spec must deserialize");
    assert_eq!(spec.model_pool_strategy, ModelPoolStrategy::IntelliRouter);

    let inner = ScriptedFactory::new();
    let factory = RecordingFactory::new(Arc::clone(&inner));
    let workspace = RecordingWorkspaceProvider::new();
    let build_deps = deps(&inner, &workspace)
        .with_harness_factory(Arc::clone(&factory) as Arc<dyn MemberHarnessFactory>);

    let error = spec
        .build_with(build_deps)
        .await
        .expect_err("an intelli_router team must not build");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(
        message.contains("intelli_router") || message.contains("IntelliRouter"),
        "{message}"
    );
    assert!(
        factory.spec("boss").is_none(),
        "the refusal fires before any harness is built"
    );
}
