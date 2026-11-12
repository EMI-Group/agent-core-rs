//! Crate-boundary integration suite for the extensions adapters.
//!
//! Every assertion goes through the public `openjiuwen_extensions` API, so a
//! regression in a re-export, a factory signature, or a deferral contract fails
//! here first. In-module unit tests cover the same modules at a lower level.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_extensions::checkpointer::{
    build_checkpointer, Checkpoint, CheckpointMeta, CheckpointNamespace, CheckpointerConfig,
    CheckpointerKind,
};
use openjiuwen_extensions::context_evolver::{
    normalize_algo_name, AddMemoryRequest, Algo, RetrieveRequest, SummarizeRequest,
    TaskMemoryService, UnsupportedTaskMemoryService,
};
use openjiuwen_extensions::message_queue::{
    create_queue, JsonMessageHandler, QueueBridge, QueueConfig,
};
use openjiuwen_extensions::store::{build_kv_store, KvValue, StoreBackend};
use openjiuwen_extensions::sys_operation::{
    is_deferred, register_deferred, registered_types, SandboxEndpoint, SandboxGatewayConfig,
    DEFERRED_SANDBOX_TYPES,
};
use serde_json::{json, Value};

/// Unwraps a `Result`'s error; the deferred backends return trait objects that
/// are not `Debug`, so `unwrap_err` is unavailable at this boundary.
fn expect_error<T>(result: Result<T>) -> Error {
    match result {
        Ok(_) => panic!("expected a deferred backend to fail"),
        Err(error) => error,
    }
}

// Port of tests/unit_tests/extensions/checkpointer/test_agent_storage.py (save/recover).
#[tokio::test]
async fn memory_checkpointer_round_trips_a_checkpoint() {
    let config = CheckpointerConfig {
        kind: CheckpointerKind::InMemory,
        ..CheckpointerConfig::default()
    };
    let checkpointer = build_checkpointer(&config).await.unwrap();
    assert!(!checkpointer.session_exists("it-session").await.unwrap());

    let checkpoint = Checkpoint::new("it-session", CheckpointNamespace::Workflow, "flow-1")
        .with_global_state(json!({"step": 1}))
        .with_component_state(json!({"planner": {"done": false}}));
    let checkpoint_id = checkpointer.save(checkpoint).await.unwrap();
    assert_ne!(checkpoint_id, "", "the backend must mint an id");

    let loaded = checkpointer
        .load("it-session", &checkpoint_id)
        .await
        .unwrap()
        .expect("the checkpoint was just saved");
    assert_eq!(loaded.session_id, "it-session");
    assert_eq!(loaded.namespace, CheckpointNamespace::Workflow);
    assert_eq!(loaded.entity_id, "flow-1");
    assert_eq!(loaded.global_state, json!({"step": 1}));
    assert_eq!(loaded.component_state, json!({"planner": {"done": false}}));

    let metas: Vec<CheckpointMeta> = checkpointer.list("it-session").await.unwrap();
    assert_eq!(metas.len(), 1);
    assert_eq!(metas[0].checkpoint_id, checkpoint_id);
    assert!(checkpointer.session_exists("it-session").await.unwrap());
}

// Port of tests/unit_tests/extensions/store/test_es_vector_store.py (backend selection).
#[tokio::test]
async fn memory_store_backend_selects_a_working_kv_store() {
    let backend = StoreBackend::from_json(&json!({"kind": "memory"})).unwrap();
    assert_eq!(backend, StoreBackend::Memory);

    let store = build_kv_store(&backend).await.unwrap();
    assert!(store.get("it:key").await.unwrap().is_none());
    store.set("it:key", KvValue::from("value-1")).await.unwrap();
    assert!(store.exists("it:key").await.unwrap());
    let stored = store.get("it:key").await.unwrap().unwrap();
    assert_eq!(stored.as_str(), Some("value-1"));
}

/// Doubles an integer payload; the JSON handler used by the bridge round-trip.
struct Doubler;

#[async_trait]
impl JsonMessageHandler for Doubler {
    async fn handle_json(&self, payload: Value) -> Result<Value> {
        let value = payload
            .as_i64()
            .ok_or_else(|| Error::validation("Doubler expects an integer payload"))?;
        Ok(json!(value * 2))
    }
}

// Port of tests/unit_tests/extensions/message_queue/test_drain.py (publish/request).
#[tokio::test]
async fn in_memory_queue_bridge_round_trips_json() {
    let queue = create_queue(&QueueConfig::in_memory()).await.unwrap();
    assert!(!queue.is_started(), "the factory must not start the queue");
    queue.start().await.unwrap();

    let bridge = QueueBridge::new(Arc::clone(&queue));
    bridge
        .subscribe_json("math", Arc::new(Doubler))
        .await
        .unwrap();
    bridge.publish_json("math", json!(4)).await.unwrap();
    let reply = bridge
        .request_json("math", json!(21), Some(Duration::from_secs(5)))
        .await
        .unwrap();
    assert_eq!(reply, json!(42));

    bridge.queue().stop().await.unwrap();
    assert!(!queue.is_started());
}

// Port of tests/unit_tests/extensions/sys_operation/sandbox/test_aio_agent.py
// (the Python provider names register; the protocols stay deferred).
#[tokio::test]
async fn deferred_sandbox_providers_register_and_fail_creation() {
    // The registry seam is re-exported unchanged from the kernel umbrella.
    let registry =
        openjiuwen_core::sys_operation::sandbox::provider::SandboxProviderRegistry::new();
    register_deferred(&registry).await.unwrap();

    let registered = registered_types(&registry);
    assert_eq!(registered.len(), DEFERRED_SANDBOX_TYPES.len());
    for sandbox_type in DEFERRED_SANDBOX_TYPES {
        assert!(
            registered.iter().any(|name| name == sandbox_type),
            "{sandbox_type} is missing from {registered:?}"
        );
        assert!(is_deferred(sandbox_type));
        assert!(registry.has_fs(sandbox_type));
    }

    let endpoint = SandboxEndpoint::default();
    let gateway = SandboxGatewayConfig::default();
    let error = expect_error(registry.create_fs("aio", &endpoint, &gateway));
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("aio"), "unexpected: {error}");
    let error = expect_error(registry.create_shell("jiuwenbox", &endpoint, &gateway));
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let error = expect_error(registry.create_code("yuanrong", &endpoint, &gateway));
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

// Port of tests/unit_tests/extensions/context_evolver/test_retrieve_flow.py
// (`TestAlgorithmNormalization`; spot checks at the crate boundary).
#[test]
fn normalize_algo_name_matches_the_python_table() {
    let cases = [
        ("ace", Algo::Ace),
        ("RB", Algo::ReasoningBank),
        ("reasoning_bank", Algo::ReasoningBank),
        ("reme", Algo::Reme),
        ("Ref_Con", Algo::Refcon),
        ("ref-cOn", Algo::Refcon),
        ("cognition", Algo::Cognition),
    ];
    for (name, expected) in cases {
        assert_eq!(normalize_algo_name(name).unwrap(), expected, "name: {name}");
    }

    let error = normalize_algo_name("not-an-algorithm").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("not-an-algorithm"));
}

// Port of tests/unit_tests/extensions/context_evolver/test_summary_flow.py
// (the deferred service seam; Python skips the live LLM flows).
#[tokio::test]
async fn unsupported_task_memory_service_reports_every_flow() {
    let service = UnsupportedTaskMemoryService::new(Algo::Reme);
    assert_eq!(service.algo(), Algo::Reme);

    let retrieve = RetrieveRequest {
        user_id: "user-1".to_string(),
        query: "what did we decide?".to_string(),
    };
    let error = service.retrieve(retrieve).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("reme"), "unexpected: {error}");

    let summarize = SummarizeRequest {
        user_id: "user-1".to_string(),
        trajectories: json!([{"step": 1}]),
    };
    let error = service.summarize(summarize).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);

    let request = AddMemoryRequest {
        content: "prefer small commits".to_string(),
        ..AddMemoryRequest::default()
    };
    let error = service.add_memory("user-1", request).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);

    let error = service.load_memories("user-1").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

// Port of tests/unit_tests/extensions/checkpointer/test_redis_checkpointer_provider.py
// (the provider is registered but its backend is deferred).
#[cfg(feature = "redis")]
#[tokio::test]
async fn redis_backends_are_deferred() {
    use openjiuwen_extensions::checkpointer::redis::{
        RedisCheckpointer, RedisCheckpointerConfig, RedisConnectionConfig,
    };
    use openjiuwen_extensions::store::redis::{RedisStore, RedisStoreConfig};

    let config =
        RedisCheckpointerConfig::new(RedisConnectionConfig::new("redis://localhost:6379/0"));
    let error = expect_error(RedisCheckpointer::connect(config).await);
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("redis"), "unexpected: {error}");

    let store_config = RedisStoreConfig {
        url: Some("redis://localhost:6379/0".to_string()),
        ..RedisStoreConfig::default()
    };
    let error = expect_error(RedisStore::connect(store_config).await);
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("redis"), "unexpected: {error}");
}

// Port of tests/unit_tests/extensions/message_queue/test_start_with_retry.py
// (the transport is deferred; the retry budget is never consumed).
#[cfg(feature = "pulsar")]
#[tokio::test]
async fn pulsar_queue_is_deferred() {
    use openjiuwen_extensions::message_queue::pulsar::{self, PulsarConfig};

    let config = PulsarConfig::new("pulsar://localhost:6650");
    let error = expect_error(pulsar::connect(config.clone()).await);
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("pulsar"), "unexpected: {error}");

    let error = expect_error(pulsar::start_with_retry(config, 3, Duration::from_millis(1)).await);
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

// Port of tests/unit_tests/extensions/store/test_es_vector_store.py
// (no feature gate: the vector backend is always API-shaped and deferred).
#[tokio::test]
async fn elasticsearch_vector_store_is_deferred() {
    use openjiuwen_extensions::store::elasticsearch::{
        ElasticsearchVectorStore, ElasticsearchVectorStoreConfig,
    };

    let config = ElasticsearchVectorStoreConfig {
        url: "http://localhost:9200".to_string(),
        ..ElasticsearchVectorStoreConfig::default()
    };
    let error = expect_error(ElasticsearchVectorStore::connect(config).await);
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.message().contains("elasticsearch"),
        "unexpected: {error}"
    );
}
