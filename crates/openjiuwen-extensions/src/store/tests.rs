// Python counterparts: `tests/unit_tests/extensions/store/test_es_vector_store.py`
// (Elasticsearch vector store, now live) and `test_gauss_db.py` (Gauss document
// store, not ported because the kernel document contract replaces the
// `SQLAlchemy` engine).
#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::SecretString;
use openjiuwen_core::foundation::store::{FieldSchema, VectorDataType};
use serde_json::json;
#[test]
fn from_json_reads_every_kind() {
    let memory = StoreBackend::from_json(&json!({"kind": "memory"})).unwrap();
    assert_eq!(memory.kind(), StoreKind::Memory);
    let memory_via_type = StoreBackend::from_json(&json!({"type": "MEMORY"})).unwrap();
    assert_eq!(memory_via_type, StoreBackend::Memory);
    let file = StoreBackend::from_json(&json!({"kind": "file", "root": "/tmp/store"})).unwrap();
    assert_eq!(
        file,
        StoreBackend::File {
            root: PathBuf::from("/tmp/store")
        }
    );
    assert_eq!(file.kind(), StoreKind::File);
    let sqlite =
        StoreBackend::from_json(&json!({"kind": "SQLite", "url": "sqlite::memory:"})).unwrap();
    assert_eq!(
        sqlite,
        StoreBackend::Sqlite {
            url: "sqlite::memory:".to_string()
        }
    );
    assert_eq!(sqlite.kind(), StoreKind::Sqlite);
    let redis_without_url = StoreBackend::from_json(&json!({"kind": "redis"})).unwrap();
    assert_eq!(redis_without_url, StoreBackend::Redis { url: None });
    assert_eq!(redis_without_url.kind(), StoreKind::Redis);
    let redis_with_url =
        StoreBackend::from_json(&json!({"kind": "redis", "url": "redis://localhost:6379"}))
            .unwrap();
    assert_eq!(
        redis_with_url,
        StoreBackend::Redis {
            url: Some("redis://localhost:6379".to_string())
        }
    );
    let es_default =
        StoreBackend::from_json(&json!({"kind": "elasticsearch", "url": "http://localhost:9200"}))
            .unwrap();
    assert_eq!(
        es_default,
        StoreBackend::Elasticsearch {
            url: "http://localhost:9200".to_string(),
            index_prefix: "agent_vector".to_string(),
            username: None,
            password: None,
            api_key: None,
        }
    );
    assert_eq!(es_default.kind(), StoreKind::Elasticsearch);
    let es_explicit = StoreBackend::from_json(
        &json!({"kind": "elasticsearch", "url": "http://localhost:9200", "index_prefix": "custom"}),
    )
    .unwrap();
    assert_eq!(
        es_explicit,
        StoreBackend::Elasticsearch {
            url: "http://localhost:9200".to_string(),
            index_prefix: "custom".to_string(),
            username: None,
            password: None,
            api_key: None,
        }
    );
    let es_credentials = StoreBackend::from_json(&json!({
        "kind": "elasticsearch",
        "url": "http://localhost:9200",
        "username": " elastic ",
        "password": "s3cret",
        "api_key": "  api-key-123  ",
    }))
    .unwrap();
    assert_eq!(
        es_credentials,
        StoreBackend::Elasticsearch {
            url: "http://localhost:9200".to_string(),
            index_prefix: "agent_vector".to_string(),
            username: Some("elastic".to_string()),
            password: Some("s3cret".to_string()),
            api_key: Some("api-key-123".to_string()),
        }
    );
    assert_eq!(es_credentials.kind(), StoreKind::Elasticsearch);
}
#[test]
fn from_json_rejects_invalid_configs() {
    for value in [
        json!({"kind": "unknown"}),
        json!({}),
        json!({"kind": 5}),
        json!([]),
    ] {
        let error = StoreBackend::from_json(&value).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "config: {value}");
    }
    let error = StoreBackend::from_json(&json!({"kind": "unknown"})).unwrap_err();
    let message = error.message().to_ascii_lowercase();
    for kind in ["memory", "file", "sqlite", "redis", "elasticsearch"] {
        assert!(
            message.contains(kind),
            "message must list `{kind}`: {message}"
        );
    }
    let error = StoreBackend::from_json(&json!({"kind": "file"})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("root"));
    let error = StoreBackend::from_json(&json!({"kind": "file", "root": ""})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = StoreBackend::from_json(&json!({"kind": "sqlite"})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("url"));
    let error = StoreBackend::from_json(&json!({"kind": "redis", "url": 42})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = StoreBackend::from_json(
        &json!({"kind": "elasticsearch", "url": "http://x", "index_prefix": 7}),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    for field in ["username", "password", "api_key"] {
        let error = StoreBackend::from_json(&json!({
            "kind": "elasticsearch",
            "url": "http://x",
            field: 7,
        }))
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "field: {field}");
        assert!(error.message().contains(field), "{}", error.message());
    }
}
#[test]
fn store_backend_serde_round_trips() {
    let backends = [
        StoreBackend::Memory,
        StoreBackend::File {
            root: PathBuf::from("/tmp/store"),
        },
        StoreBackend::Sqlite {
            url: "sqlite::memory:".to_string(),
        },
        StoreBackend::Redis { url: None },
        StoreBackend::Redis {
            url: Some("redis://localhost:6379".to_string()),
        },
        StoreBackend::Elasticsearch {
            url: "http://localhost:9200".to_string(),
            index_prefix: "custom".to_string(),
            username: Some("elastic".to_string()),
            password: Some("s3cret".to_string()),
            api_key: Some("api-key-123".to_string()),
        },
    ];
    for backend in &backends {
        let encoded = serde_json::to_value(backend).unwrap();
        assert_eq!(&StoreBackend::from_json(&encoded).unwrap(), backend);
    }
    let decoded: StoreBackend =
        serde_json::from_value(json!({"kind": "elasticsearch", "url": "http://localhost:9200"}))
            .unwrap();
    assert_eq!(
        decoded,
        StoreBackend::Elasticsearch {
            url: "http://localhost:9200".to_string(),
            index_prefix: "agent_vector".to_string(),
            username: None,
            password: None,
            api_key: None,
        },
        "a credential-less config still decodes with the credential fields empty"
    );
    // `index_prefix` and the credential fields all default, so the minimal
    // JSON shape keeps working.
    let minimal: StoreBackend =
        serde_json::from_value(json!({"kind": "elasticsearch", "url": "http://localhost:9200"}))
            .unwrap();
    assert_eq!(minimal, decoded);
}
#[tokio::test]
async fn memory_kv_store_round_trips() {
    let store = build_kv_store(&StoreBackend::Memory).await.unwrap();
    store.set("alpha", KvValue::text("one")).await.unwrap();
    assert_eq!(
        store.get("alpha").await.unwrap(),
        Some(KvValue::text("one"))
    );
    assert!(store.exists("alpha").await.unwrap());
    store.set("pre/a", KvValue::text("a")).await.unwrap();
    store.set("pre/b", KvValue::text("b")).await.unwrap();
    let prefixed = store.get_by_prefix("pre/").await.unwrap();
    assert_eq!(prefixed.len(), 2);
    assert_eq!(prefixed.get("pre/a"), Some(&KvValue::text("a")));
    store.delete("alpha").await.unwrap();
    assert!(!store.exists("alpha").await.unwrap());
    assert_eq!(store.get("alpha").await.unwrap(), None);
}
#[tokio::test]
async fn file_kv_store_round_trips_and_persists() {
    let dir = tempfile::TempDir::new().unwrap();
    let backend = StoreBackend::File {
        root: dir.path().to_path_buf(),
    };
    let store = build_kv_store(&backend).await.unwrap();
    store
        .set("persisted", KvValue::text("value"))
        .await
        .unwrap();
    assert_eq!(
        store.get("persisted").await.unwrap(),
        Some(KvValue::text("value"))
    );
    drop(store);
    let reopened = build_kv_store(&backend).await.unwrap();
    assert_eq!(
        reopened.get("persisted").await.unwrap(),
        Some(KvValue::text("value"))
    );
}
#[tokio::test]
async fn memory_document_store_factory_round_trips() {
    let store = build_document_store(&StoreBackend::Memory).await.unwrap();
    store.create_collection("docs").await.unwrap();
    assert_eq!(
        store.list_collections().await.unwrap(),
        vec!["docs".to_string()]
    );
}
#[tokio::test]
async fn memory_vector_store_factory_creates_collections() {
    let store = build_vector_store(&StoreBackend::Memory).await.unwrap();
    let schema = CollectionSchema::new(vec![
        FieldSchema {
            name: "id".to_string(),
            dtype: VectorDataType::Varchar,
            is_primary: true,
            ..FieldSchema::default()
        },
        FieldSchema {
            name: "embedding".to_string(),
            dtype: VectorDataType::FloatVector,
            dim: Some(3),
            ..FieldSchema::default()
        },
    ]);
    store.create_collection("vectors", schema).await.unwrap();
    assert!(store.collection_exists("vectors").await.unwrap());
}
/// Python counterpart: `tests/unit_tests/extensions/store/test_es_vector_store.py`.
///
/// Elasticsearch serves documents and vectors but not key values: the
/// key-value factory reports it and names a backend that does implement
/// that contract, while the document arm builds the kernel's live REST
/// document store and the vector arm builds this crate's live REST vector
/// store. Neither constructor performs I/O, so the assertions stay offline.
#[tokio::test]
async fn elasticsearch_factories_split_key_value_document_and_vector() {
    let backend = StoreBackend::Elasticsearch {
        url: "http://localhost:9200".to_string(),
        index_prefix: "agent_vector".to_string(),
        username: None,
        password: None,
        api_key: None,
    };
    let error = build_kv_store(&backend)
        .await
        .err()
        .expect("elasticsearch is not a key-value store");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.message().to_ascii_lowercase();
    assert!(message.contains("elasticsearch"), "{message}");
    assert!(message.contains("in-memory"), "{message}");
    // The document factory builds a live REST store without any I/O, so a
    // loopback endpoint that is never contacted is enough.
    let documents: Arc<dyn DocumentStore> = build_document_store(&backend)
        .await
        .expect("elasticsearch is a live document store backend");
    drop(documents);
    build_vector_store(&backend)
        .await
        .expect("elasticsearch is a live vector store backend");
    // The factory passes `url` and `index_prefix` straight through, so the
    // concrete store built from them names indices the documented way.
    let config = elasticsearch::ElasticsearchVectorStoreConfig {
        url: "http://localhost:9200".to_string(),
        index_prefix: "agent_vector".to_string(),
        ..elasticsearch::ElasticsearchVectorStoreConfig::default()
    };
    let store = elasticsearch::ElasticsearchVectorStore::new(config)
        .expect("a valid elasticsearch config must build without I/O");
    assert_eq!(store.index_name("docs"), "agent_vector__docs");
}
/// The document helper forwards the endpoint and the index prefix, and
/// strips everything a kernel endpoint cannot carry.
#[test]
fn elasticsearch_document_config_trims_the_endpoint_and_forwards_the_prefix() {
    let config = elasticsearch_document_config(
        "  http://localhost:9200/base/?pretty=1#top  ",
        "custom",
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(config.endpoint, "http://localhost:9200/base");
    assert_eq!(config.index_prefix, "custom");
    assert_eq!(config.username, None);
    assert_eq!(config.password, None);
    assert_eq!(config.api_key, None);
}
/// Explicit credential fields win over the URL userinfo, field by field.
#[test]
fn elasticsearch_document_config_prefers_explicit_credentials() {
    // Userinfo only: both halves come from the URL.
    let config = elasticsearch_document_config(
        "http://alice:wonderland@127.0.0.1:9200",
        "agent_vector",
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(config.endpoint, "http://127.0.0.1:9200");
    assert_eq!(config.username.as_deref(), Some("alice"));
    assert_eq!(
        config.password.as_ref().map(SecretString::expose_secret),
        Some("wonderland")
    );
    // Explicit fields replace the userinfo halves.
    let config = elasticsearch_document_config(
        "http://alice:wonderland@127.0.0.1:9200",
        "agent_vector",
        Some("bob"),
        Some("builder"),
        Some("api-key-123"),
    )
    .unwrap();
    assert_eq!(config.endpoint, "http://127.0.0.1:9200");
    assert_eq!(config.username.as_deref(), Some("bob"));
    assert_eq!(
        config.password.as_ref().map(SecretString::expose_secret),
        Some("builder")
    );
    assert_eq!(
        config.api_key.as_ref().map(SecretString::expose_secret),
        Some("api-key-123")
    );
    // A user name without a password sends an empty password, like the
    // vector backend's URL userinfo handling.
    let config = elasticsearch_document_config(
        "http://alice@127.0.0.1:9200",
        "agent_vector",
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(config.username.as_deref(), Some("alice"));
    assert_eq!(
        config.password.as_ref().map(SecretString::expose_secret),
        Some("")
    );
    // A password without a user name is not sent: basic auth needs both.
    let config = elasticsearch_document_config(
        "http://127.0.0.1:9200",
        "agent_vector",
        None,
        Some("orphan"),
        None,
    )
    .unwrap();
    assert_eq!(config.username, None);
    assert_eq!(
        config.password.as_ref().map(SecretString::expose_secret),
        Some("orphan")
    );
    // The store exposes exactly the configuration the helper produced.
    let store = ElasticsearchDocumentStore::new(config)
        .expect("a host-only endpoint is a valid kernel configuration");
    assert_eq!(store.config().endpoint, "http://127.0.0.1:9200");
}
/// An unusable backend URL is rejected before any request is sent.
#[test]
fn elasticsearch_document_config_validates_the_url() {
    for url in [
        "",
        "   ",
        "localhost:9200",
        "ftp://localhost:9200",
        "http://",
    ] {
        let error =
            elasticsearch_document_config(url, "agent_vector", None, None, None).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "url: {url:?}");
        assert!(error.message().contains("url"), "{}", error.message());
    }
    let error = elasticsearch_document_config("localhost:9200", "agent_vector", None, None, None)
        .unwrap_err();
    assert!(
        error.message().contains("http://"),
        "the message names the scheme requirement: {}",
        error.message()
    );
}
/// The factory forwards the credentials of a configured backend.
#[tokio::test]
async fn elasticsearch_document_factory_forwards_the_credentials() {
    let store = elasticsearch_document_store(
        "http://alice:wonderland@127.0.0.1:9200",
        "agent_vector",
        Some("bob"),
        None,
        Some("api-key-123"),
    )
    .expect("the store builds without I/O");
    assert_eq!(store.config().endpoint, "http://127.0.0.1:9200");
    assert_eq!(store.config().username.as_deref(), Some("bob"));
    assert_eq!(
        store
            .config()
            .password
            .as_ref()
            .map(SecretString::expose_secret),
        Some("wonderland"),
        "the explicit user name keeps the userinfo password"
    );
    assert_eq!(
        store
            .config()
            .api_key
            .as_ref()
            .map(SecretString::expose_secret),
        Some("api-key-123")
    );
}
#[cfg(feature = "redis")]
#[tokio::test]
async fn redis_kv_store_round_trips_against_the_fake_server() {
    let fake = crate::redis::testsupport::FakeRedis::start().await;
    let backend = StoreBackend::Redis {
        url: Some(fake.url()),
    };
    let store = build_kv_store(&backend).await.unwrap();
    store.set("alpha", KvValue::text("one")).await.unwrap();
    assert_eq!(
        store.get("alpha").await.unwrap(),
        Some(KvValue::text("one"))
    );
    assert!(store.exists("alpha").await.unwrap());
    store.set("pre/a", KvValue::text("a")).await.unwrap();
    store
        .set("pre/b", KvValue::bytes(vec![0, 1, 2, 255]))
        .await
        .unwrap();
    let prefixed = store.get_by_prefix("pre/").await.unwrap();
    assert_eq!(prefixed.len(), 2);
    assert_eq!(prefixed.get("pre/a"), Some(&KvValue::text("a")));
    assert_eq!(
        prefixed.get("pre/b"),
        Some(&KvValue::bytes(vec![0, 1, 2, 255]))
    );
    let keys = vec!["alpha".to_string(), "missing".to_string()];
    assert_eq!(
        store.mget(&keys).await.unwrap(),
        vec![Some(KvValue::text("one")), None]
    );
    let mut pipeline = store.pipeline();
    pipeline.set("pipe", KvValue::text("value"));
    pipeline.get("pipe");
    pipeline.exists("pipe");
    assert_eq!(
        pipeline.execute().await.unwrap(),
        vec![
            None,
            Some(KvValue::text("value")),
            Some(KvValue::text("true")),
        ]
    );
    assert_eq!(store.delete_by_prefix("pre/").await.unwrap(), 2);
    store.delete("alpha").await.unwrap();
    assert!(!store.exists("alpha").await.unwrap());
    fake.shutdown().await;
}
#[cfg(feature = "redis")]
#[tokio::test]
async fn redis_factories_require_a_url() {
    let backend = StoreBackend::Redis { url: None };
    for (name, error) in [
        ("key-value", build_kv_store(&backend).await.err()),
        ("document", build_document_store(&backend).await.err()),
        ("vector", build_vector_store(&backend).await.err()),
    ] {
        let error = error.unwrap_or_else(|| panic!("{name}: missing url must fail"));
        assert_eq!(error.kind(), ErrorKind::Validation, "{name}: {error}");
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("url"), "{name}: {message}");
        assert!(message.contains("redis"), "{name}: {message}");
    }
}
#[cfg(feature = "redis")]
#[tokio::test]
async fn redis_factories_round_trip_documents_and_vectors_against_the_fake_server() {
    let fake = crate::redis::testsupport::FakeRedis::start().await;
    let backend = StoreBackend::Redis {
        url: Some(fake.url()),
    };
    let documents = build_document_store(&backend).await.unwrap();
    documents.create_collection("docs").await.unwrap();
    let written = vec![
        Document::new("a").with_field("age", 36),
        Document::new("b").with_field("age", 20),
    ];
    assert_eq!(documents.upsert("docs", written).await.unwrap(), 2);
    assert_eq!(documents.list_collections().await.unwrap(), vec!["docs"]);
    let read = documents
        .query(
            "docs",
            &Query {
                filter: serde_json::json!({"age": {"$gte": 30}}),
                ..Query::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].id, "a");
    let vectors = build_vector_store(&backend).await.unwrap();
    let schema = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(2),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
    ]);
    vectors
        .create_collection("embeddings", schema.clone())
        .await
        .unwrap();
    assert_eq!(
        vectors.get_schema("embeddings").await.unwrap(),
        Some(schema)
    );
    let batch = vec![
        serde_json::json!({"id": "near", "embedding": [1.0, 0.0]}),
        serde_json::json!({"id": "far", "embedding": [0.0, 1.0]}),
    ]
    .into_iter()
    .filter_map(|doc| doc.as_object().cloned())
    .collect();
    assert_eq!(vectors.add_docs("embeddings", batch).await.unwrap(), 2);
    let hits = vectors
        .search("embeddings", &VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hits[0].fields["id"], "near");
    assert_eq!(hits.len(), 2);
    fake.shutdown().await;
}
#[cfg(not(feature = "redis"))]
#[tokio::test]
async fn redis_stores_require_the_feature() {
    let backend = StoreBackend::Redis {
        url: Some("redis://localhost:6379".to_string()),
    };
    for (name, error) in [
        ("key-value", build_kv_store(&backend).await.err()),
        ("document", build_document_store(&backend).await.err()),
        ("vector", build_vector_store(&backend).await.err()),
    ] {
        let error = error.unwrap_or_else(|| panic!("{name}: the feature-off backend must fail"));
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{name}: {error}");
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("redis"), "{name}: {message}");
        assert!(message.contains("`redis` feature"), "{name}: {message}");
    }
}
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_factories_use_an_in_memory_database() {
    let backend = StoreBackend::Sqlite {
        url: "sqlite::memory:".to_string(),
    };
    let kv = build_kv_store(&backend).await.unwrap();
    kv.set("key", KvValue::text("value")).await.unwrap();
    assert_eq!(kv.get("key").await.unwrap(), Some(KvValue::text("value")));
    let documents = build_document_store(&backend).await.unwrap();
    documents.create_collection("docs").await.unwrap();
    assert_eq!(
        documents.list_collections().await.unwrap(),
        vec!["docs".to_string()]
    );
    let vectors = build_vector_store(&backend).await.unwrap();
    assert_eq!(
        vectors.list_collection_names().await.unwrap(),
        Vec::<String>::new()
    );
}
#[cfg(not(feature = "sqlite"))]
#[tokio::test]
async fn sqlite_kv_store_requires_the_feature() {
    let backend = StoreBackend::Sqlite {
        url: "sqlite::memory:".to_string(),
    };
    let error = build_kv_store(&backend)
        .await
        .err()
        .expect("sqlite backend without the feature must fail");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().to_ascii_lowercase().contains("sqlite"));
}
// The vector backend is the one SQLite surface that only becomes available with
// the feature, so a default build must keep reporting it as deferred.
#[cfg(not(feature = "sqlite"))]
#[tokio::test]
async fn sqlite_vector_store_requires_the_feature() {
    let backend = StoreBackend::Sqlite {
        url: "sqlite::memory:".to_string(),
    };
    let error = build_vector_store(&backend)
        .await
        .err()
        .expect("the sqlite vector backend needs the feature");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().to_ascii_lowercase().contains("sqlite"));
    assert!(error.message().contains("vector"));
}
