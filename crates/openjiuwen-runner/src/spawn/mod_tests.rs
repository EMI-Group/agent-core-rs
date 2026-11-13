#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Unit tests of the spawn seam itself: the wire names and ordering of
//! [`SpawnBackend`], the [`SpawnHealth`] snapshot shape, the
//! [`SpawnSpec::backend`] routing hint, and the defaults every [`SpawnedAgent`]
//! and [`AgentSpawner`] implementation inherits.
use super::*;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;

#[test]
fn spec_defaults_match_python() {
    let spec = SpawnSpec::default();
    assert_eq!(spec.agent_kind, SpawnAgentKind::ClassAgent);
    assert_eq!(spec.agent_module, None);
    assert_eq!(spec.agent_class, None);
    assert_eq!(spec.init_kwargs, json!({}));
    assert_eq!(spec.payload, Value::Null);
    assert_eq!(spec.session_id, None);
    assert!((spec.health_check_interval_secs - 5.0).abs() < f64::EPSILON);
    assert!((spec.shutdown_timeout_secs - 10.0).abs() < f64::EPSILON);
    assert!(spec.envs.is_empty());
}

#[test]
fn kind_wire_names_are_snake_case() {
    assert_eq!(
        serde_json::to_value(SpawnAgentKind::ClassAgent).unwrap(),
        json!("class_agent")
    );
    assert_eq!(
        serde_json::to_value(SpawnAgentKind::TeamAgent).unwrap(),
        json!("team_agent")
    );
    let parsed: SpawnAgentKind = serde_json::from_value(json!("team_agent")).unwrap();
    assert_eq!(parsed, SpawnAgentKind::TeamAgent);
    assert_eq!(SpawnAgentKind::default(), SpawnAgentKind::ClassAgent);
}

#[test]
fn class_agent_builder_fills_module_and_class() {
    let spec = SpawnSpec::class_agent("my_pkg.agents", "HelperAgent");
    assert_eq!(spec.agent_kind, SpawnAgentKind::ClassAgent);
    assert_eq!(spec.agent_module.as_deref(), Some("my_pkg.agents"));
    assert_eq!(spec.agent_class.as_deref(), Some("HelperAgent"));
    assert_eq!(spec.init_kwargs, json!({}));
}

#[test]
fn team_agent_builder_carries_the_payload() {
    let spec = SpawnSpec::team_agent(json!({"name": "crew"}));
    assert_eq!(spec.agent_kind, SpawnAgentKind::TeamAgent);
    assert_eq!(spec.payload, json!({"name": "crew"}));
    assert_eq!(spec.agent_module, None);
}

#[test]
fn builders_chain() {
    let spec = SpawnSpec::class_agent("mod", "Cls")
        .with_session_id("session-1")
        .with_init_kwargs(json!({"a": 1}))
        .with_env("OPENJIUWEN_LOG_LEVEL", "debug")
        .with_env("API_KEY", "secret");
    assert_eq!(spec.session_id.as_deref(), Some("session-1"));
    assert_eq!(spec.init_kwargs, json!({"a": 1}));
    assert_eq!(spec.envs.len(), 2);
    assert_eq!(spec.envs.get("API_KEY").map(String::as_str), Some("secret"));
    assert_eq!(
        spec.envs.get("OPENJIUWEN_LOG_LEVEL").map(String::as_str),
        Some("debug")
    );
}

#[test]
fn spec_round_trips_through_json() {
    let spec = SpawnSpec::class_agent("mod", "Cls")
        .with_session_id("s")
        .with_init_kwargs(json!({"n": 2}))
        .with_env("K", "V");
    let value = serde_json::to_value(&spec).unwrap();
    assert_eq!(value["agent_kind"], json!("class_agent"));
    assert_eq!(value["health_check_interval_secs"], json!(5.0));
    assert_eq!(value["shutdown_timeout_secs"], json!(10.0));
    let restored: SpawnSpec = serde_json::from_value(value).unwrap();
    assert_eq!(restored, spec);
}

#[test]
fn spec_tolerates_unknown_fields() {
    let spec: SpawnSpec = serde_json::from_value(json!({
        "agent_kind": "team_agent",
        "payload": {"name": "crew"},
        "extra_field": "ignored",
    }))
    .unwrap();
    assert_eq!(spec.agent_kind, SpawnAgentKind::TeamAgent);
    assert_eq!(spec.payload, json!({"name": "crew"}));
    assert!((spec.shutdown_timeout_secs - 10.0).abs() < f64::EPSILON);
}

#[test]
fn spec_accepts_partial_json() {
    let spec: SpawnSpec = serde_json::from_value(json!({
        "agent_module": "mod",
        "agent_class": "Cls",
        "session_id": "s-1",
    }))
    .unwrap();
    assert_eq!(spec.agent_kind, SpawnAgentKind::ClassAgent);
    assert_eq!(spec.agent_module.as_deref(), Some("mod"));
    assert_eq!(spec.agent_class.as_deref(), Some("Cls"));
    assert_eq!(spec.session_id.as_deref(), Some("s-1"));
    assert_eq!(spec.init_kwargs, json!({}));
    assert!(spec.envs.is_empty());
}

#[tokio::test]
async fn unsupported_spawner_reports_the_seam() {
    let error = UnsupportedSpawner
        .spawn(SpawnSpec::class_agent("mod", "Cls"))
        .await
        .err()
        .expect("the built-in spawner never spawns");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("subprocess agent spawning"),
        "unexpected message: {error}"
    );
    assert!(
        error.to_string().contains("Runner::set_spawner"),
        "unexpected message: {error}"
    );
}

// --- SpawnBackend -----------------------------------------------------------

#[test]
fn backend_wire_names_are_snake_case() {
    let cases = [
        (SpawnBackend::Unknown, "unknown"),
        (SpawnBackend::Subprocess, "subprocess"),
        (SpawnBackend::ExternalCommand, "external_command"),
        (SpawnBackend::InProcess, "in_process"),
        (SpawnBackend::RemoteHttp, "remote_http"),
        (SpawnBackend::RemoteMq, "remote_mq"),
        (SpawnBackend::RemoteA2a, "remote_a2a"),
    ];
    for (backend, name) in cases {
        assert_eq!(backend.as_str(), name);
        assert_eq!(backend.to_string(), name);
        assert_eq!(serde_json::to_value(backend).unwrap(), json!(name));
        let parsed: SpawnBackend = serde_json::from_value(json!(name)).unwrap();
        assert_eq!(parsed, backend);
    }
    assert_eq!(SpawnBackend::default(), SpawnBackend::Unknown);
}

#[test]
fn only_the_four_landed_backends_are_implemented() {
    for backend in [
        SpawnBackend::Subprocess,
        SpawnBackend::ExternalCommand,
        SpawnBackend::InProcess,
        SpawnBackend::RemoteHttp,
    ] {
        assert!(backend.is_implemented(), "{backend} should be implemented");
    }
    for backend in [
        SpawnBackend::Unknown,
        SpawnBackend::RemoteMq,
        SpawnBackend::RemoteA2a,
    ] {
        assert!(!backend.is_implemented(), "{backend} is deferred");
    }
}

#[test]
fn backends_order_by_declaration_for_sorted_listings() {
    let mut backends = vec![
        SpawnBackend::RemoteA2a,
        SpawnBackend::Subprocess,
        SpawnBackend::Unknown,
    ];
    backends.sort();
    assert_eq!(
        backends,
        vec![
            SpawnBackend::Unknown,
            SpawnBackend::Subprocess,
            SpawnBackend::RemoteA2a,
        ]
    );
}

// --- SpawnHealth ------------------------------------------------------------

#[test]
fn health_default_is_unknown_and_down() {
    let health = SpawnHealth::default();
    assert_eq!(health.backend, SpawnBackend::Unknown);
    assert!(!health.alive);
    assert!(!health.healthy);
    assert_eq!(health.pid, None);
    assert_eq!(health.last_error, None);
}

#[test]
fn health_round_trips_and_tolerates_unknown_fields() {
    let health = SpawnHealth {
        backend: SpawnBackend::Subprocess,
        alive: true,
        healthy: false,
        pid: Some(4242),
        last_error: Some("probe timed out".to_owned()),
    };
    let value = serde_json::to_value(&health).unwrap();
    assert_eq!(
        value,
        json!({
            "backend": "subprocess",
            "alive": true,
            "healthy": false,
            "pid": 4242,
            "last_error": "probe timed out",
        })
    );
    assert_eq!(
        serde_json::from_value::<SpawnHealth>(value).unwrap(),
        health
    );

    let partial: SpawnHealth =
        serde_json::from_value(json!({"backend": "in_process", "extra": 1})).unwrap();
    assert_eq!(partial.backend, SpawnBackend::InProcess);
    assert!(!partial.alive);
    assert_eq!(partial.pid, None);
}

#[test]
fn health_healthy_helper_reports_up_and_healthy() {
    let health = SpawnHealth::healthy(SpawnBackend::InProcess, None);
    assert_eq!(health.backend, SpawnBackend::InProcess);
    assert!(health.alive);
    assert!(health.healthy);
    assert_eq!(health.pid, None);
    assert_eq!(health.last_error, None);
}

// --- SpawnSpec::backend -----------------------------------------------------

#[test]
fn spec_backend_defaults_to_none_and_round_trips() {
    let spec = SpawnSpec::default();
    assert_eq!(spec.backend, None);
    let value = serde_json::to_value(&spec).unwrap();
    assert!(
        value.get("backend").is_none(),
        "an unset hint is omitted from the wire object: {value}"
    );

    let routed = SpawnSpec::class_agent("mod", "Cls").with_backend(SpawnBackend::InProcess);
    assert_eq!(routed.backend, Some(SpawnBackend::InProcess));
    let value = serde_json::to_value(&routed).unwrap();
    assert_eq!(value["backend"], json!("in_process"));
    assert_eq!(serde_json::from_value::<SpawnSpec>(value).unwrap(), routed);
}

#[test]
fn spec_backend_is_additive_metadata_for_a_child() {
    // A Python child ignores the hint: unknown/present fields are tolerated and
    // the frames themselves are unchanged.
    let spec: SpawnSpec = serde_json::from_value(json!({
        "agent_kind": "class_agent",
        "backend": "external_command",
        "unknown_backend_field": true,
    }))
    .unwrap();
    assert_eq!(spec.backend, Some(SpawnBackend::ExternalCommand));
    assert_eq!(spec.agent_kind, SpawnAgentKind::ClassAgent);
}

// --- defaulted accessors ----------------------------------------------------

/// A handle that implements only the required methods, i.e. the smallest
/// implementation the sealed-in defaults must support.
struct MinimalAgent {
    alive: bool,
}

#[async_trait]
impl SpawnedAgent for MinimalAgent {
    fn id(&self) -> &'static str {
        "minimal"
    }

    fn is_alive(&self) -> bool {
        self.alive
    }

    fn pid(&self) -> Option<u32> {
        Some(7)
    }

    async fn send(&self, _inputs: Value) -> Result<AgentOutput> {
        Ok(AgentOutput::answer("ok"))
    }

    fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }

    async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
        Ok(true)
    }

    async fn force_kill(&self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn a_minimal_handle_inherits_the_seam_defaults() {
    let agent = MinimalAgent { alive: true };
    assert_eq!(agent.backend(), SpawnBackend::Unknown);
    assert_eq!(agent.is_healthy(), agent.is_alive());

    let snapshot = agent.health_snapshot();
    assert_eq!(snapshot.backend, SpawnBackend::Unknown);
    assert!(snapshot.alive);
    assert_eq!(snapshot.healthy, agent.is_healthy());
    assert_eq!(snapshot.pid, Some(7));
    assert_eq!(snapshot.last_error, None);

    let dead = MinimalAgent { alive: false };
    assert!(!dead.is_healthy());
    assert!(!dead.health_snapshot().alive);
}

#[tokio::test]
async fn a_minimal_handle_reports_the_probe_and_cancel_seams() {
    let agent = MinimalAgent { alive: true };
    let error = agent.health().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("agent health probe"), "{error}");
    assert!(error.to_string().contains("unknown"), "{error}");

    let error = agent.cancel_task("task-1").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("spawned task cancellation"),
        "{error}"
    );
    assert!(error.to_string().contains("unknown"), "{error}");
}

#[test]
fn the_unsupported_spawner_names_no_backend() {
    assert_eq!(UnsupportedSpawner.backend(), SpawnBackend::Unknown);
}
#[test]
fn the_default_and_the_built_in_default_describe_nothing() {
    assert_eq!(UnsupportedSpawner.describe(), Value::Null);
    assert_eq!(
        UnsupportedSpawner.backend(),
        SpawnBackend::Unknown,
        "the default spawner names no backend"
    );
}
#[test]
fn a_deferred_spawner_describes_its_backend_and_reason() {
    let described = DeferredSpawner::for_backend(SpawnBackend::RemoteMq).describe();
    assert_eq!(described["backend"], json!("remote_mq"));
    assert_eq!(described["deferred"], json!(true));
    let reason = described["reason"].as_str().expect("a reason string");
    let hint = described["hint"].as_str().expect("a hint string");
    assert!(reason.contains("Pulsar"), "{reason}");
    assert!(hint.contains("reply-topic"), "{hint}");

    // Calling it through the object-safe seam gives the same summary.
    let spawner: Arc<dyn AgentSpawner> = Arc::new(DeferredSpawner::new(
        SpawnBackend::RemoteA2a,
        "a2a spawn transport",
        "use the A2A spawner",
    ));
    let dynamic = spawner.describe();
    assert_eq!(dynamic["backend"], json!("remote_a2a"));
    assert_eq!(dynamic["reason"], json!("a2a spawn transport"));
    assert_eq!(dynamic["hint"], json!("use the A2A spawner"));
}
