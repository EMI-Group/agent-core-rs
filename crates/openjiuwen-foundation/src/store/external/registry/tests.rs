//! Offline tests for the name-driven backend registry.
//!
//! Every backend is either constructed in memory, rooted in a `TempDir`, or pointed
//! at a loopback address: no test needs a Chroma/Milvus/Elasticsearch/Redis/S3
//! service, credentials, or the network.

use super::*;
use crate::store::kv::KvValue;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use tempfile::TempDir;

/// A temporary directory rendered as a configuration `root`.
fn root(dir: &TempDir) -> String {
    dir.path().to_string_lossy().into_owned()
}

/// A KV factory returning a fresh in-memory store, used by the plugin tests.
fn memory_kv_factory() -> KvStoreFactory {
    Arc::new(|_config: &Value| Ok(Arc::new(InMemoryKvStore::new()) as Arc<dyn KvStore>))
}

#[test]
fn parse_accepts_names_and_aliases() {
    for (name, expected) in [
        ("memory", BackendKind::Memory),
        ("Memory", BackendKind::Memory),
        (" FILE ", BackendKind::File),
        ("local", BackendKind::File),
        ("sqlite", BackendKind::Sqlite),
        ("Redis", BackendKind::Redis),
        ("chroma", BackendKind::Chroma),
        ("milvus", BackendKind::Milvus),
        ("elasticsearch", BackendKind::Elasticsearch),
        ("ES", BackendKind::Elasticsearch),
        ("s3", BackendKind::S3),
        ("pgvector", BackendKind::Pgvector),
        ("postgres", BackendKind::Postgres),
        ("gauss", BackendKind::GaussVector),
        ("gaussvector", BackendKind::GaussVector),
    ] {
        assert_eq!(BackendKind::parse(name), Some(expected), "{name}");
    }
    assert_eq!(BackendKind::parse("nope"), None);
    assert_eq!(BackendKind::parse(""), None);
}

#[test]
fn names_round_trip_and_display() {
    for kind in [
        BackendKind::Memory,
        BackendKind::File,
        BackendKind::Sqlite,
        BackendKind::Redis,
        BackendKind::Chroma,
        BackendKind::Milvus,
        BackendKind::Elasticsearch,
        BackendKind::S3,
        BackendKind::Pgvector,
        BackendKind::GaussVector,
        BackendKind::Postgres,
    ] {
        assert_eq!(BackendKind::parse(kind.as_str()), Some(kind));
        assert_eq!(format!("{kind}"), kind.as_str());
    }
}

#[test]
fn deferred_kinds_are_flagged() {
    for kind in [
        BackendKind::Pgvector,
        BackendKind::GaussVector,
        BackendKind::Postgres,
    ] {
        assert!(kind.is_deferred(), "{kind}");
    }
    for kind in [
        BackendKind::Memory,
        BackendKind::File,
        BackendKind::Sqlite,
        BackendKind::Redis,
        BackendKind::Chroma,
        BackendKind::Milvus,
        BackendKind::Elasticsearch,
        BackendKind::S3,
    ] {
        assert!(!kind.is_deferred(), "{kind}");
    }
}

#[test]
fn spec_parses_every_accepted_shape() {
    let bare = StoreSpec::parse(&json!("memory")).expect("a bare string is accepted");
    assert_eq!(bare.backend, BackendKind::Memory);
    assert_eq!(bare.name, "memory");
    assert_eq!(bare.config, json!({}));

    let typed = StoreSpec::parse(&json!({"type": "local", "root": "/tmp/oj"}))
        .expect("the type key is accepted");
    assert_eq!(typed.backend, BackendKind::File);
    assert_eq!(typed.name, "local");
    assert_eq!(typed.config, json!({"root": "/tmp/oj"}));

    let aliased = StoreSpec::parse(&json!({"backend": "es", "endpoint": "http://host:9200"}))
        .expect("an alias is accepted");
    assert_eq!(aliased.backend, BackendKind::Elasticsearch);
    assert_eq!(aliased.name, "es");
    assert_eq!(aliased.config, json!({"endpoint": "http://host:9200"}));

    let map: BTreeMap<String, Value> = [
        ("backend".to_owned(), json!("memory")),
        ("limit".to_owned(), json!(7)),
    ]
    .into_iter()
    .collect();
    let from_map = StoreSpec::from_map(&map).expect("a map is accepted");
    assert_eq!(from_map.backend, BackendKind::Memory);
    assert_eq!(from_map.config, json!({"limit": 7}));
}

#[test]
fn spec_rejects_malformed_and_unknown_names() {
    for spec in [
        json!({}),
        json!({"backend": 3}),
        json!({"backend": "  "}),
        json!({"type": null}),
        json!([1, 2]),
    ] {
        let error = StoreSpec::parse(&spec).expect_err("a malformed spec is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "{spec}");
        let message = error.to_string();
        assert!(message.contains("backend"), "{message}");
        assert!(message.contains("type"), "{message}");
    }

    let error = StoreSpec::parse(&json!({"backend": "nope"})).expect_err("unknown backend");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    for name in ["chroma", "memory", "redis"] {
        assert!(message.contains(name), "{message} misses {name}");
    }
}

#[test]
fn available_names_are_sorted_and_cover_each_family() {
    for (family, expected) in [
        (StoreFamily::Kv, "redis"),
        (StoreFamily::Document, "elasticsearch"),
        (StoreFamily::Vector, "chroma"),
        (StoreFamily::Object, "s3"),
        (StoreFamily::Graph, "memory"),
    ] {
        let names = available_backend_names(family);
        assert!(names.contains(&expected), "{family} misses {expected}");
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "{family} is not sorted");
    }
    assert_eq!(StoreFamily::Vector.to_string(), "vector");
}

#[tokio::test]
async fn kv_backends_construct() {
    let registry = StoreRegistry::new();
    let memory = registry
        .create_kv_store(&json!({"backend": "memory"}))
        .await
        .expect("memory kv");
    memory.set("k", KvValue::text("v")).await.expect("set");
    assert_eq!(
        memory.get("k").await.expect("get"),
        Some(KvValue::text("v"))
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let file = registry
        .create_kv_store(&json!({"backend": "file", "root": root(&dir)}))
        .await
        .expect("file kv");
    file.set("k", KvValue::text("v")).await.expect("set");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    registry
        .create_kv_store(&json!({"backend": "redis", "addr": addr}))
        .await
        .expect("redis kv");
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_kv_backend_constructs_with_the_feature() {
    let registry = StoreRegistry::new();
    let store = registry
        .create_kv_store(&json!({"backend": "sqlite", "url": "sqlite::memory:"}))
        .await
        .expect("sqlite kv");
    store.set("k", KvValue::text("v")).await.expect("set");
    assert_eq!(store.get("k").await.expect("get"), Some(KvValue::text("v")));

    let documents = registry
        .create_document_store(&json!({"backend": "sqlite", "url": "sqlite::memory:"}))
        .await
        .expect("sqlite documents");
    assert_eq!(
        documents.list_collections().await.expect("collections"),
        Vec::<String>::new()
    );
}

#[cfg(not(feature = "sqlite"))]
#[tokio::test]
async fn sqlite_backend_requires_the_feature() {
    let registry = StoreRegistry::new();
    for family in [StoreFamily::Kv, StoreFamily::Document] {
        let error = match family {
            StoreFamily::Kv => registry
                .create_kv_store(&json!({"backend": "sqlite", "url": "sqlite::memory:"}))
                .await
                .err(),
            _ => registry
                .create_document_store(&json!({"backend": "sqlite", "url": "sqlite::memory:"}))
                .await
                .err(),
        }
        .expect("sqlite is gated");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        assert!(
            error
                .to_string()
                .contains("sqlite backend requires the sqlite feature"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn document_backends_construct() {
    let registry = StoreRegistry::new();
    let memory = registry
        .create_document_store(&json!({"backend": "memory"}))
        .await
        .expect("memory documents");
    assert_eq!(
        memory.list_collections().await.expect("list"),
        Vec::<String>::new()
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let file = registry
        .create_document_store(&json!({"backend": "local", "path": root(&dir)}))
        .await
        .expect("file documents");
    assert_eq!(
        file.list_collections().await.expect("list"),
        Vec::<String>::new()
    );

    for backend in ["elasticsearch", "es"] {
        registry
            .create_document_store(&json!({
                "backend": backend,
                "endpoint": "http://127.0.0.1:1",
            }))
            .await
            .unwrap_or_else(|error| panic!("{backend} documents: {error}"));
    }
}

#[tokio::test]
async fn vector_backends_construct() {
    let registry = StoreRegistry::new();
    let memory = registry
        .create_vector_store(&json!({"backend": "memory"}))
        .await
        .expect("memory vectors");
    assert_eq!(
        memory.list_collection_names().await.expect("names"),
        Vec::<String>::new()
    );

    let dir = tempfile::tempdir().expect("tempdir");
    registry
        .create_vector_store(&json!({"backend": "file", "root": root(&dir)}))
        .await
        .expect("file vectors");

    for backend in ["chroma", "milvus", "elasticsearch", "es"] {
        registry
            .create_vector_store(&json!({
                "backend": backend,
                "endpoint": "http://127.0.0.1:1",
            }))
            .await
            .unwrap_or_else(|error| panic!("{backend} vectors: {error}"));
    }
}

#[tokio::test]
async fn object_backends_construct() {
    let registry = StoreRegistry::new();
    let memory = registry
        .create_object_store(&json!({"backend": "memory"}))
        .await
        .expect("memory objects");
    assert_eq!(
        memory.list_buckets().await.expect("buckets"),
        Vec::<String>::new()
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let file = registry
        .create_object_store(&json!({"backend": "file", "root": root(&dir)}))
        .await
        .expect("file objects");
    file.create_bucket("bucket").await.expect("bucket");
    assert_eq!(
        file.list_buckets().await.expect("buckets"),
        vec!["bucket".to_owned()]
    );

    registry
        .create_object_store(&json!({
            "backend": "s3",
            "endpoint": "http://127.0.0.1:9000",
            "region": "us-east-1",
            "credentials": {"access_key_id": "ak", "secret_access_key": "sk"},
        }))
        .await
        .expect("s3 objects");
}

#[tokio::test]
async fn graph_backends_construct_and_external_ones_are_unsupported() {
    let registry = StoreRegistry::new();
    let memory = registry
        .create_graph_store(&json!({"backend": "memory"}))
        .await
        .expect("memory graph");
    assert_eq!(memory.len().await.expect("len"), (0, 0));

    let dir = tempfile::tempdir().expect("tempdir");
    let file = registry
        .create_graph_store(&json!({"backend": "file", "root": root(&dir)}))
        .await
        .expect("file graph");
    assert_eq!(file.len().await.expect("len"), (0, 0));

    for backend in [
        "redis",
        "chroma",
        "milvus",
        "elasticsearch",
        "s3",
        "pgvector",
    ] {
        let error = registry
            .create_graph_store(&json!({"backend": backend}))
            .await
            .err()
            .expect("no external graph backend exists");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        assert!(
            error
                .to_string()
                .contains("external graph backends are not implemented"),
            "{error}"
        );
    }

    let error = registry
        .create_graph_store(&json!({"backend": "sqlite"}))
        .await
        .err()
        .expect("sqlite is not a graph backend");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn deferred_vector_backends_report_the_missing_driver() {
    let registry = StoreRegistry::new();
    for backend in ["pgvector", "postgres", "gauss", "gaussvector"] {
        let error = registry
            .create_vector_store(&json!({"backend": backend}))
            .await
            .err()
            .expect("deferred backend");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let message = error.to_string();
        let expected = if backend.starts_with("gauss") {
            "GaussDB driver that is not available offline"
        } else {
            "PostgreSQL driver that is not available offline"
        };
        assert!(message.contains(expected), "{backend}: {message}");
    }
}

#[tokio::test]
async fn wrong_family_backend_is_a_validation_error_listing_names() {
    let registry = StoreRegistry::new();
    let error = registry
        .create_kv_store(&json!({"backend": "chroma"}))
        .await
        .err()
        .expect("chroma is a vector backend");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    for name in ["file", "memory", "redis", "sqlite"] {
        assert!(message.contains(name), "{message} misses {name}");
    }

    let error = registry
        .create_object_store(&json!({"backend": "redis"}))
        .await
        .err()
        .expect("redis is a kv backend");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("s3"), "{error}");
}

#[tokio::test]
async fn file_backend_requires_a_root() {
    let registry = StoreRegistry::new();
    let spec = json!({"backend": "file"});
    assert_eq!(
        registry
            .create_kv_store(&spec)
            .await
            .err()
            .expect("kv root")
            .kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        registry
            .create_document_store(&spec)
            .await
            .err()
            .expect("document root")
            .kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        registry
            .create_vector_store(&spec)
            .await
            .err()
            .expect("vector root")
            .kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        registry
            .create_object_store(&spec)
            .await
            .err()
            .expect("object root")
            .kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        registry
            .create_graph_store(&spec)
            .await
            .err()
            .expect("graph root")
            .kind(),
        ErrorKind::Validation
    );
}

#[tokio::test]
async fn unknown_name_through_a_registry_lists_that_family() {
    let registry = StoreRegistry::new();
    let error = registry
        .create_vector_store(&json!({"backend": "nope"}))
        .await
        .err()
        .expect("unknown backend");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("chroma"), "{message}");
    assert!(message.contains("milvus"), "{message}");
}

#[tokio::test]
async fn plugin_is_local_to_its_registry() {
    let registry = StoreRegistry::new();
    assert!(registry
        .register_kv_backend("acme", memory_kv_factory())
        .is_none());
    assert!(registry
        .available_names(StoreFamily::Kv)
        .contains(&"acme".to_owned()));

    let store = registry
        .create_kv_store(&json!({"backend": "acme"}))
        .await
        .expect("the plugin builds a store");
    store.set("k", KvValue::text("v")).await.expect("set");
    assert_eq!(store.get("k").await.expect("get"), Some(KvValue::text("v")));

    let isolated = StoreRegistry::new();
    let error = isolated
        .create_kv_store(&json!({"backend": "acme"}))
        .await
        .err()
        .expect("plugins do not leak between instances");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(!isolated
        .available_names(StoreFamily::Kv)
        .contains(&"acme".to_owned()));
}

#[tokio::test]
async fn global_plugin_is_visible_through_the_free_function() {
    let name = "global-registry-test-kv";
    assert!(register_kv_backend(name, memory_kv_factory()).is_none());
    assert!(global_registry()
        .available_names(StoreFamily::Kv)
        .contains(&name.to_owned()));

    let store = create_kv_store(&json!({"backend": name}))
        .await
        .expect("the global plugin builds a store");
    store.set("k", KvValue::text("v")).await.expect("set");
    assert_eq!(store.get("k").await.expect("get"), Some(KvValue::text("v")));

    assert!(register_kv_backend(name, memory_kv_factory()).is_some());
}

#[tokio::test]
async fn plugin_overrides_a_builtin_name() {
    let registry = StoreRegistry::new();
    let factory: KvStoreFactory = Arc::new(|_config: &Value| Err(Error::config("the plugin ran")));
    assert!(registry.register_kv_backend("memory", factory).is_none());
    let error = registry
        .create_kv_store(&json!({"backend": "memory"}))
        .await
        .err()
        .expect("the plugin replaced the built-in");
    assert!(error.to_string().contains("the plugin ran"), "{error}");
}

#[test]
fn debug_lists_registered_names() {
    let registry = StoreRegistry::new();
    assert_eq!(
        format!("{registry:?}"),
        "StoreRegistry { kv: [], document: [], vector: [], object: [], graph: [] }"
    );
    assert!(registry
        .register_graph_backend(
            "acme",
            Arc::new(|_config: &Value| {
                Ok(Arc::new(InMemoryGraphStore::new()) as Arc<dyn GraphStore>)
            })
        )
        .is_none());
    assert!(format!("{registry:?}").contains("acme"));
}
