//! Unit tests for the live `Redis` vector backend.
//!
//! Python counterpart: none — `openjiuwen/extensions/store` has only the
//! Elasticsearch vector store, so these suites verify the Rust-side extension
//! against the kernel `InMemoryVectorStore` reference.

use super::*;
use crate::redis::testsupport::{FakeOptions, FakeRedis};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::store::{
    CollectionSchema, FieldSchema, InMemoryVectorStore, Metric, VectorDataType, VectorQuery,
    VectorSearchResult, VectorStore,
};
use serde_json::{json, Map, Value};

/// A field without flags, so the fixtures stay readable.
fn field(name: &str, dtype: VectorDataType) -> FieldSchema {
    FieldSchema {
        name: name.to_string(),
        dtype,
        ..FieldSchema::default()
    }
}

/// The three-field schema every suite but the `auto_id` one uses.
fn schema() -> CollectionSchema {
    CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..field("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(3),
            ..field("embedding", VectorDataType::FloatVector)
        },
        field("kind", VectorDataType::Varchar),
    ])
}

/// The JSON object of `value`, for the document fixtures.
fn obj(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

/// A document with a three-component embedding.
fn doc(id: &str, embedding: [f32; 3]) -> Map<String, Value> {
    obj(json!({
        "id": id,
        "embedding": embedding.to_vec(),
        "kind": "note",
    }))
}

/// A configuration carrying only `url`.
fn url_config(url: &str) -> RedisStoreConfig {
    RedisStoreConfig {
        url: Some(url.to_string()),
        ..RedisStoreConfig::default()
    }
}

/// A vector store bound to a running fake server (the client connects lazily).
fn store_for(fake: &FakeRedis) -> RedisVectorStore {
    RedisVectorStore::new(url_config(&fake.url())).unwrap()
}

/// The ids of `hits`, in order.
fn ids(hits: &[VectorSearchResult]) -> Vec<String> {
    hits.iter()
        .map(|hit| hit.fields["id"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// Creates [`schema`] and seeds it with the four ordering fixtures.
async fn seeded(fake: &FakeRedis) -> RedisVectorStore {
    let store = store_for(fake);
    store.create_collection("docs", schema()).await.unwrap();
    store
        .add_docs(
            "docs",
            vec![
                doc("a", [1.0, 0.0, 0.0]),
                doc("b", [0.0, 1.0, 0.0]),
                doc("c", [-1.0, 0.0, 0.0]),
                doc("d", [1.0, 1.0, 0.0]),
            ],
        )
        .await
        .unwrap();
    store
}

#[tokio::test]
async fn vector_store_round_trips_the_collection_schema() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let schema = CollectionSchema {
        fields: vec![
            FieldSchema {
                is_primary: true,
                max_length: Some(64),
                ..field("id", VectorDataType::Varchar)
            },
            FieldSchema {
                dim: Some(3),
                ..field("embedding", VectorDataType::FloatVector)
            },
            FieldSchema {
                default_value: Some(json!("unknown")),
                ..field("kind", VectorDataType::Varchar)
            },
        ],
        description: Some("notes".to_string()),
        enable_dynamic_field: true,
    };
    store
        .create_collection("docs", schema.clone())
        .await
        .unwrap();

    let stored = store.get_schema("docs").await.unwrap().expect("schema");
    assert_eq!(stored, schema);
    assert_eq!(stored.fields[0].max_length, Some(64));
    assert_eq!(stored.fields[1].dim, Some(3));
    assert_eq!(stored.fields[2].default_value, Some(json!("unknown")));
    assert!(stored.description.is_some());
    assert!(stored.enable_dynamic_field);

    assert!(store.collection_exists("docs").await.unwrap());
    assert_eq!(store.get_schema("absent").await.unwrap(), None);
    assert!(!store.collection_exists("absent").await.unwrap());
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        vec!["docs".to_string()]
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_rejects_a_duplicate_or_invalid_collection() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    store.create_collection("docs", schema()).await.unwrap();
    let error = store
        .create_collection("docs", schema())
        .await
        .expect_err("duplicate collection");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("\"docs\""), "unexpected: {error}");

    // An invalid schema never creates a collection.
    let invalid = CollectionSchema::new(vec![FieldSchema {
        dim: Some(3),
        ..field("embedding", VectorDataType::FloatVector)
    }]);
    let error = store
        .create_collection("broken", invalid)
        .await
        .expect_err("invalid schema");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(!store.collection_exists("broken").await.unwrap());
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        vec!["docs".to_string()]
    );

    for name in ["", "a/b", "a\\b", "..", ".", "bad\0name"] {
        let error = store
            .create_collection(name, schema())
            .await
            .expect_err("invalid name");
        assert_eq!(error.kind(), ErrorKind::Validation, "name {name:?}");
        assert!(
            error.message().contains("invalid collection name"),
            "unexpected: {error}"
        );
        assert_eq!(
            store
                .search(name, &VectorQuery::new(vec![0.0, 0.0, 0.0]))
                .await
                .expect_err("invalid name")
                .kind(),
            ErrorKind::Validation,
            "name {name:?}"
        );
    }
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_lists_deletes_and_reports_existence() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    store.create_collection("beta", schema()).await.unwrap();
    store.create_collection("alpha", schema()).await.unwrap();
    store
        .add_docs("beta", vec![doc("a", [1.0, 0.0, 0.0])])
        .await
        .unwrap();
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );

    // Deleting a missing collection is a no-op.
    store.delete_collection("absent").await.unwrap();
    store.delete_collection("beta").await.unwrap();
    assert!(!store.collection_exists("beta").await.unwrap());
    assert_eq!(store.get_schema("beta").await.unwrap(), None);
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        vec!["alpha".to_string()]
    );
    assert!(
        fake.state().keys().iter().all(|key| !key.contains("beta")),
        "the dropped collection keeps keys: {:?}",
        fake.state().keys()
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_add_fills_defaults_and_mints_auto_ids() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let auto = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            auto_id: true,
            ..field("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(1),
            ..field("embedding", VectorDataType::FloatVector)
        },
        FieldSchema {
            default_value: Some(json!("unknown")),
            ..field("kind", VectorDataType::Varchar)
        },
    ]);
    store.create_collection("auto", auto).await.unwrap();
    assert_eq!(
        store
            .add_docs("auto", vec![obj(json!({"embedding": [1.0]}))])
            .await
            .unwrap(),
        1
    );

    let hits = store
        .search("auto", &VectorQuery::new(vec![1.0]))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].fields["kind"], "unknown", "default applied");
    assert_eq!(hits[0].fields["embedding"], json!([1.0]));
    let generated = hits[0].fields["id"]
        .as_str()
        .expect("generated id")
        .to_string();
    assert_ne!(generated, "");

    // The prepared document was persisted, so a fresh store sees it.
    let fresh = store_for(&fake);
    let persisted = fresh
        .search("auto", &VectorQuery::new(vec![1.0]))
        .await
        .unwrap();
    assert_eq!(ids(&persisted), vec![generated.clone()]);
    assert_eq!(persisted[0].fields["kind"], "unknown");

    assert_eq!(
        store
            .delete_docs_by_ids("auto", &[generated])
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store
            .search("auto", &VectorQuery::new(vec![1.0]))
            .await
            .unwrap(),
        Vec::<VectorSearchResult>::new()
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_rejects_duplicate_primary_keys_without_persisting() {
    let fake = FakeRedis::start().await;
    let store = seeded(&fake).await;
    let error = store
        .add_docs("docs", vec![doc("a", [0.0, 0.0, 1.0])])
        .await
        .expect_err("duplicate primary key");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("already exists"), "{error}");

    // A batch that repeats a key and a batch with a mistyped field both fail
    // whole, so the stored view is unchanged.
    let error = store
        .add_docs(
            "docs",
            vec![doc("x", [0.0, 0.0, 1.0]), doc("x", [1.0, 1.0, 1.0])],
        )
        .await
        .expect_err("duplicate key inside the batch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = store
        .add_docs(
            "docs",
            vec![obj(json!({"id": "y", "embedding": [1.0], "kind": "note"}))],
        )
        .await
        .expect_err("wrong dimension");
    assert_eq!(error.kind(), ErrorKind::Validation);

    let hits = store
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(10),
        )
        .await
        .unwrap();
    let mut stored = ids(&hits);
    stored.sort();
    assert_eq!(stored, vec!["a", "b", "c", "d"]);
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_search_orders_by_score_then_primary_key_and_truncates() {
    let fake = FakeRedis::start().await;
    let store = seeded(&fake).await;
    // Cosine similarity to [1, 0, 0]: a = 1.0, d ≈ 0.8536, b = 0.5, c = 0.0.
    let hits = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["a", "d", "b", "c"]);
    assert!(
        (hits[0].score - 1.0).abs() < 1e-6,
        "identical vector is 1.0"
    );
    assert!(hits.windows(2).all(|pair| pair[0].score >= pair[1].score));
    assert!(hits.iter().all(|hit| hit.score.is_finite()));
    assert!(hits.iter().all(|hit| (0.0..=1.0).contains(&hit.score)));

    let hits = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(2))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["a", "d"], "top_k truncates");

    // Equal scores are broken by primary key, ascending.
    store
        .add_docs(
            "docs",
            ["z", "m"]
                .into_iter()
                .map(|id| doc(id, [1.0, 0.0, 0.0]))
                .collect(),
        )
        .await
        .unwrap();
    let hits = store
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0, 0.0])
                .with_top_k(2)
                .with_filters(json!({"id": {"$in": ["m", "z"]}})),
        )
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["m", "z"]);
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_search_matches_the_reference_results() {
    let fake = FakeRedis::start().await;
    let store = seeded(&fake).await;
    let reference = InMemoryVectorStore::new();
    reference.create_collection("docs", schema()).await.unwrap();
    reference
        .add_docs(
            "docs",
            vec![
                doc("a", [1.0, 0.0, 0.0]),
                doc("b", [0.0, 1.0, 0.0]),
                doc("c", [-1.0, 0.0, 0.0]),
                doc("d", [1.0, 1.0, 0.0]),
            ],
        )
        .await
        .unwrap();

    for query in [
        VectorQuery::new(vec![1.0, 0.0, 0.0]),
        VectorQuery::new(vec![0.0, 1.0, 0.0]).with_top_k(3),
        VectorQuery::new(vec![1.0, 1.0, 1.0])
            .with_filters(json!({"kind": "note"}))
            .with_top_k(2),
        VectorQuery::new(vec![1.0, 0.0, 0.0])
            .with_filters(json!({"kind": "missing"}))
            .with_top_k(2),
        VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(0),
    ] {
        let expected = reference.search("docs", &query).await.unwrap();
        let actual = store.search("docs", &query).await.unwrap();
        assert_eq!(actual, expected, "query {query:?}");
    }

    // The vector field can be named explicitly.
    let named = VectorQuery::new(vec![0.0, 0.0, 1.0]).with_vector_field("embedding");
    assert_eq!(
        store.search("docs", &named).await.unwrap(),
        reference.search("docs", &named).await.unwrap()
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_search_applies_filters() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let schema = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..field("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(2),
            ..field("embedding", VectorDataType::FloatVector)
        },
        field("kind", VectorDataType::Varchar),
        field("rank", VectorDataType::Int64),
    ]);
    store.create_collection("docs", schema).await.unwrap();
    store
        .add_docs(
            "docs",
            vec![
                obj(json!({"id": "a", "embedding": [1.0, 0.0], "kind": "note", "rank": 1})),
                obj(json!({"id": "b", "embedding": [1.0, 0.0], "kind": "task", "rank": 2})),
                obj(json!({"id": "c", "embedding": [1.0, 0.0], "kind": "note", "rank": 3})),
            ],
        )
        .await
        .unwrap();

    let notes = store
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0]).with_filters(json!({"kind": "note"})),
        )
        .await
        .unwrap();
    assert_eq!(ids(&notes), ["a", "c"]);

    let ranked = store
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0])
                .with_filters(json!({"rank": {"$gte": 2}}))
                .with_top_k(1),
        )
        .await
        .unwrap();
    assert_eq!(ids(&ranked), ["b"]);
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_metric_defaults_to_cosine_and_can_be_changed() {
    let fake = FakeRedis::start().await;
    let store = seeded(&fake).await;
    assert_eq!(store.metric(), Metric::Cosine);
    let query = VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(4);
    let cosine = store.search("docs", &query).await.unwrap();
    assert!((cosine[0].score - 1.0).abs() < 1e-6);

    let euclidean = store_for(&fake).with_metric(Metric::Euclidean);
    assert_eq!(euclidean.metric(), Metric::Euclidean);
    let hits = euclidean.search("docs", &query).await.unwrap();
    assert_eq!(
        ids(&hits),
        ids(&cosine),
        "the ranking is metric-independent"
    );
    for hit in &hits {
        let embedding: Vec<f32> = serde_json::from_value(hit.fields["embedding"].clone())
            .expect("stored embedding is a float array");
        let expected = Metric::Euclidean.score(&[1.0, 0.0, 0.0], &embedding);
        assert!(
            (hit.score - expected).abs() < 1e-6,
            "{}: expected {expected}, got {}",
            hit.fields["id"],
            hit.score
        );
    }
    // The cosine score of the orthogonal document is 0.5; Euclidean differs.
    assert!((cosine[2].score - 0.5).abs() < 1e-6);
    assert!((hits[2].score - 0.5).abs() > 1e-6);
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_deletes_by_ids_and_filters() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let schema = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..field("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(2),
            ..field("embedding", VectorDataType::FloatVector)
        },
        field("kind", VectorDataType::Varchar),
    ]);
    store.create_collection("docs", schema).await.unwrap();
    store
        .add_docs(
            "docs",
            vec![
                obj(json!({"id": "a", "embedding": [1.0, 0.0], "kind": "keep"})),
                obj(json!({"id": "b", "embedding": [1.0, 0.0], "kind": "drop"})),
                obj(json!({"id": "c", "embedding": [1.0, 0.0], "kind": "drop"})),
                obj(json!({"id": "d", "embedding": [1.0, 0.0], "kind": "keep"})),
            ],
        )
        .await
        .unwrap();

    // Unknown ids are ignored.
    assert_eq!(
        store
            .delete_docs_by_ids("docs", &["a".to_string(), "missing".to_string()])
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store
            .delete_docs_by_filters("docs", &json!({"kind": "drop"}))
            .await
            .unwrap(),
        2
    );
    let hits = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
        .await
        .unwrap();
    assert_eq!(ids(&hits), ["d"]);

    // The removals are persisted: a fresh store sees only the survivor.
    let fresh = store_for(&fake);
    assert_eq!(
        ids(&fresh
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(10))
            .await
            .unwrap()),
        ["d"]
    );
    assert_eq!(
        store
            .delete_docs_by_ids("docs", &["d".to_string()])
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]))
            .await
            .unwrap(),
        Vec::<VectorSearchResult>::new()
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_unknown_collections_are_not_found() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let query = VectorQuery::new(vec![1.0, 0.0, 0.0]);
    for error in [
        store.search("absent", &query).await.expect_err("search"),
        store
            .add_docs("absent", vec![doc("a", [1.0, 0.0, 0.0])])
            .await
            .expect_err("add"),
        store
            .delete_docs_by_ids("absent", &["a".to_string()])
            .await
            .expect_err("delete ids"),
        store
            .delete_docs_by_filters("absent", &json!({"id": "a"}))
            .await
            .expect_err("delete filters"),
    ] {
        assert_eq!(error.kind(), ErrorKind::NotFound, "unexpected: {error}");
        assert!(error.message().contains("absent"), "unexpected: {error}");
    }
    assert_eq!(store.get_schema("absent").await.unwrap(), None);
    assert!(!store.collection_exists("absent").await.unwrap());
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        Vec::<String>::new()
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn vector_store_connect_validates_pings_and_reports_unreachable_hosts() {
    let fake = FakeRedis::start().await;
    let store = RedisVectorStore::connect(url_config(&fake.url()))
        .await
        .expect("connect to the fake server");
    store.create_collection("live", schema()).await.unwrap();
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        vec!["live".to_string()]
    );
    fake.shutdown().await;

    let error = RedisVectorStore::new(url_config("http://localhost:6379")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("redis://"), "unexpected: {error}");

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let error = RedisVectorStore::connect(url_config(&format!("redis://{addr}/0")))
        .await
        .err()
        .expect("a closed port must fail the eager connect");
    assert_eq!(error.kind(), ErrorKind::Transport);

    let strict = FakeRedis::start_with(FakeOptions::auth_required(None, "secret")).await;
    let error = RedisVectorStore::connect(url_config(&strict.url()))
        .await
        .err()
        .expect("an unauthenticated PING must fail");
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.message().to_ascii_uppercase().contains("NOAUTH"),
        "unexpected: {error}"
    );
    strict.shutdown().await;

    let error = RedisVectorStore::connect(url_config("rediss://localhost:6379"))
        .await
        .err()
        .expect("TLS is not supported");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}
