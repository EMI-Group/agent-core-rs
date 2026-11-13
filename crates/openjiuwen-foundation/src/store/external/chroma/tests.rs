//! Offline `wiremock` tests for the Chroma vector backend.
//!
//! Every test talks to a loopback mock server, so the suite needs no Chroma
//! installation, no network, and no credentials. The filter compiler and the
//! score normalization are covered by pure unit tests.

use super::*;

use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A schema with a primary key, one embedding and a document + metadata field.
fn schema() -> CollectionSchema {
    CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(2),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
        FieldSchema::new("content", VectorDataType::Varchar),
        FieldSchema::new("score", VectorDataType::Int64),
    ])
}

/// The schema as it is embedded in a collection's metadata.
fn schema_json(schema: &CollectionSchema) -> String {
    serde_json::to_string(schema).expect("serialize schema")
}

/// Mounts the `GET /collections/{name}` response that returns `schema`.
async fn mount_collection(server: &MockServer, collection: &str, schema: &CollectionSchema) {
    Mock::given(method("GET"))
        .and(path(format!("/api/v1/collections/{collection}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": collection,
            "metadata": {"openjiuwen:schema": schema_json(schema)},
        })))
        .mount(server)
        .await;
}

/// Builds a document for the [`schema`] above.
fn doc(id: &str, embedding: Vec<f64>, score: i64) -> Map<String, Value> {
    json!({
        "id": id,
        "content": format!("content-{id}"),
        "embedding": embedding,
        "score": score,
    })
    .as_object()
    .cloned()
    .expect("object literal")
}

/// Builds a store pointing at `server`.
fn store(server: &MockServer) -> ChromaVectorStore {
    ChromaVectorStore::new(ChromaConfig::new(server.uri())).expect("build store")
}

#[tokio::test]
async fn create_collection_sends_the_schema_and_the_configured_headers() {
    let server = MockServer::start().await;
    let schema = schema();
    Mock::given(method("POST"))
        .and(path("/api/v1/collections"))
        .and(header("x-chroma-tenant", "tenant-a"))
        .and(header("x-chroma-database", "db-a"))
        .and(header("authorization", "Bearer key-123"))
        .and(body_json(json!({
            "name": "docs",
            "get_or_create": false,
            "metadata": {
                "hnsw:space": "ip",
                "openjiuwen:schema": schema_json(&schema),
            },
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"id": "c1", "name": "docs"})))
        .expect(1)
        .mount(&server)
        .await;

    let store = ChromaVectorStore::new(
        ChromaConfig::new(format!("{}/", server.uri()))
            .with_tenant("tenant-a")
            .with_database("db-a")
            .with_api_key("key-123")
            .with_metric(Metric::Dot),
    )
    .expect("build store");
    store
        .create_collection("docs", schema)
        .await
        .expect("create collection");
}

#[tokio::test]
async fn create_collection_maps_a_conflict_to_validation() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections"))
        .respond_with(
            ResponseTemplate::new(409)
                .set_body_json(json!({"error": "Collection docs already exists"})),
        )
        .mount(&server)
        .await;

    let error = store(&server)
        .create_collection("docs", schema())
        .await
        .expect_err("conflict");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("already exists"), "{error}");
}

#[tokio::test]
async fn create_collection_rejects_unrepresentable_schemas_without_any_request() {
    let server = MockServer::start().await;
    let store = store(&server);

    let array_field = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(2),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
        FieldSchema::new("tags", VectorDataType::Array),
    ]);
    let error = store
        .create_collection("docs", array_field)
        .await
        .expect_err("array field");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("tags"), "{error}");

    let two_vectors = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(2),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
        FieldSchema {
            dim: Some(2),
            ..FieldSchema::new("other", VectorDataType::FloatVector)
        },
    ]);
    let error = store
        .create_collection("docs", two_vectors)
        .await
        .expect_err("two vector fields");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("FLOAT_VECTOR"), "{error}");

    assert!(
        server
            .received_requests()
            .await
            .expect("received requests")
            .is_empty(),
        "schema validation must happen before any HTTP call"
    );
}

#[tokio::test]
async fn list_collection_names_sorts_and_accepts_both_response_shapes() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/collections"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"name": "beta"},
            {"name": "alpha"},
            "gamma",
            {"id": "no-name"},
            7,
        ])))
        .mount(&server)
        .await;

    let names = store(&server)
        .list_collection_names()
        .await
        .expect("list collections");
    assert_eq!(names, ["alpha", "beta", "gamma"]);
}

#[tokio::test]
async fn collection_exists_and_delete_handle_missing_collections() {
    let present = MockServer::start().await;
    mount_collection(&present, "docs", &schema()).await;
    assert!(store(&present)
        .collection_exists("docs")
        .await
        .expect("existing collection"));

    let missing = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/collections/docs"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&missing)
        .await;
    assert!(!store(&missing)
        .collection_exists("docs")
        .await
        .expect("absent collection"));
    store(&missing)
        .delete_collection("docs")
        .await
        .expect("deleting a missing collection is a no-op");

    let deletable = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/api/v1/collections/docs"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&deletable)
        .await;
    store(&deletable)
        .delete_collection("docs")
        .await
        .expect("delete collection");
}

#[tokio::test]
async fn get_schema_round_trips_and_reports_absent_or_foreign_collections() {
    let server = MockServer::start().await;
    let schema = schema();
    mount_collection(&server, "docs", &schema).await;
    assert_eq!(
        store(&server).get_schema("docs").await.expect("schema"),
        Some(schema)
    );

    let missing = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/collections/docs"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&missing)
        .await;
    assert_eq!(
        store(&missing).get_schema("docs").await.expect("absent"),
        None
    );

    let foreign = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/collections/docs"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"name": "docs", "metadata": {"hnsw:space": "cosine"}})),
        )
        .mount(&foreign)
        .await;
    let error = store(&foreign)
        .get_schema("docs")
        .await
        .expect_err("foreign collection");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("not created through"), "{error}");
}

#[tokio::test]
async fn add_docs_sends_ids_embeddings_documents_and_metadatas() {
    let server = MockServer::start().await;
    mount_collection(&server, "docs", &schema()).await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/add"))
        .and(body_json(json!({
            "ids": ["a", "b"],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "documents": ["content-a", "content-b"],
            "metadatas": [{"id": "a", "score": 1}, {"id": "b", "score": 2}],
        })))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;

    let inserted = store(&server)
        .add_docs(
            "docs",
            vec![doc("a", vec![1.0, 0.0], 1), doc("b", vec![0.0, 1.0], 2)],
        )
        .await
        .expect("add documents");
    assert_eq!(inserted, 2);
}

#[tokio::test]
async fn add_docs_generates_auto_ids_and_omits_absent_parts() {
    let server = MockServer::start().await;
    let schema = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            auto_id: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(2),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
        FieldSchema::new("note", VectorDataType::Varchar),
    ]);
    let collection = CollectionSchema::new(schema.fields.clone());
    mount_collection(&server, "docs", &collection).await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/add"))
        .respond_with(ResponseTemplate::new(201))
        .mount(&server)
        .await;

    let inserted = store(&server)
        .add_docs(
            "docs",
            vec![json!({"embedding": [0.5, 0.5], "note": "hello"})
                .as_object()
                .cloned()
                .expect("object literal")],
        )
        .await
        .expect("add documents");
    assert_eq!(inserted, 1);

    let requests = server.received_requests().await.expect("received requests");
    let body: Value =
        serde_json::from_slice(&requests[requests.len() - 1].body).expect("json body");
    let generated = body["ids"][0].as_str().expect("generated id");
    assert_eq!(generated.len(), 36, "{generated}");
    assert_eq!(body["documents"], json!(["hello"]));
    assert_eq!(body["metadatas"], json!([{"id": generated}]));
}

#[tokio::test]
async fn add_docs_rejects_invalid_documents_before_writing() {
    let server = MockServer::start().await;
    mount_collection(&server, "docs", &schema()).await;
    let store = store(&server);

    let duplicate = store
        .add_docs(
            "docs",
            vec![doc("a", vec![1.0, 0.0], 1), doc("a", vec![0.0, 1.0], 2)],
        )
        .await
        .expect_err("duplicate primary key");
    assert_eq!(duplicate.kind(), ErrorKind::Validation);
    assert!(duplicate.to_string().contains("duplicate primary key"));

    let dimension = store
        .add_docs("docs", vec![doc("a", vec![1.0, 0.0, 0.0], 1)])
        .await
        .expect_err("wrong dimension");
    assert_eq!(dimension.kind(), ErrorKind::Validation);
    assert!(dimension.to_string().contains("embedding"), "{dimension}");

    let mut unknown = doc("a", vec![1.0, 0.0], 1);
    unknown.insert("extra".to_string(), json!("nope"));
    let unknown = store
        .add_docs("docs", vec![unknown])
        .await
        .expect_err("unknown field");
    assert_eq!(unknown.kind(), ErrorKind::Validation);
    assert!(unknown.to_string().contains("enable_dynamic_field"));

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(requests.len(), 3, "each call only reads the schema");
    assert!(requests
        .iter()
        .all(|request| request.method.as_str() == "GET"));
}

#[tokio::test]
async fn add_docs_maps_a_service_rejection_to_validation() {
    let server = MockServer::start().await;
    mount_collection(&server, "docs", &schema()).await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/add"))
        .respond_with(ResponseTemplate::new(409).set_body_string("ID already exists: a"))
        .mount(&server)
        .await;

    let error = store(&server)
        .add_docs("docs", vec![doc("a", vec![1.0, 0.0], 1)])
        .await
        .expect_err("duplicate id");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("already exists"), "{error}");
}

#[tokio::test]
async fn search_sends_the_query_and_normalizes_hits() {
    let server = MockServer::start().await;
    mount_collection(&server, "docs", &schema()).await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/query"))
        .and(body_json(json!({
            "query_embeddings": [[0.5, 0.5]],
            "n_results": 3,
            "include": ["metadatas", "documents", "distances"],
            "where": {"$and": [{"score": {"$gte": 2}}, {"id": {"$ne": "skip"}}]},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ids": [["b", "a", "c"]],
            "distances": [[1.0, 0.0, 4.0]],
            "metadatas": [[{"score": 2}, {"score": 4}, {"score": 0}]],
            "documents": [["second", "first", "third"]],
        })))
        .mount(&server)
        .await;

    let query = VectorQuery::new(vec![0.5, 0.5])
        .with_top_k(3)
        .with_filters(json!({"score": {"$gte": 2}, "id": {"$ne": "skip"}}));
    let hits = store(&server).search("docs", &query).await.expect("search");

    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].fields["id"], "a");
    assert_eq!(hits[0].fields["content"], "first");
    assert_eq!(hits[0].fields["score"], 4);
    assert!((hits[0].score - 1.0).abs() < 1e-6, "{}", hits[0].score);
    assert_eq!(hits[1].fields["id"], "b");
    assert!((hits[1].score - 0.5).abs() < 1e-6, "{}", hits[1].score);
    assert_eq!(hits[2].fields["id"], "c");
    assert!((hits[2].score - 0.0).abs() < 1e-6, "{}", hits[2].score);
    assert!(hits
        .iter()
        .all(|hit| (0.0..=1.0).contains(&hit.score) && hit.score.is_finite()));
}

#[tokio::test]
async fn search_validates_the_vector_field_and_skips_requests_for_zero_top_k() {
    let server = MockServer::start().await;
    mount_collection(&server, "docs", &schema()).await;
    let store = store(&server);

    let empty = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(0))
        .await
        .expect("zero top_k");
    assert_eq!(empty, Vec::new());

    let field = store
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0]).with_vector_field("content"),
        )
        .await
        .expect_err("not a vector field");
    assert_eq!(field.kind(), ErrorKind::Validation);

    let dimension = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 3.0]))
        .await
        .expect_err("wrong query dimension");
    assert_eq!(dimension.kind(), ErrorKind::Validation);

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(requests.len(), 2, "only the two schema lookups happened");
    assert!(requests
        .iter()
        .all(|request| request.method.as_str() == "GET"));
}

#[tokio::test]
async fn delete_docs_by_ids_counts_only_existing_ids() {
    let server = MockServer::start().await;
    mount_collection(&server, "docs", &schema()).await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/get"))
        .and(body_json(json!({"ids": ["a", "ghost"], "include": []})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ids": [["a"]]})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/delete"))
        .and(body_json(json!({"ids": ["a"]})))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let deleted = store(&server)
        .delete_docs_by_ids("docs", &["a".to_string(), "ghost".to_string()])
        .await
        .expect("delete by ids");
    assert_eq!(deleted, 1);
}

#[tokio::test]
async fn delete_docs_by_filters_collects_ids_then_deletes_them() {
    let server = MockServer::start().await;
    mount_collection(&server, "docs", &schema()).await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/get"))
        .and(body_json(json!({
            "where": {"score": {"$gte": 2}},
            "include": [],
            "limit": 500,
            "offset": 0,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ids": [["a", "b"]]})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/delete"))
        .and(body_json(json!({"ids": ["a", "b"]})))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let deleted = store(&server)
        .delete_docs_by_filters("docs", &json!({"score": {"$gte": 2}}))
        .await
        .expect("delete by filters");
    assert_eq!(deleted, 2);
}

#[tokio::test]
async fn delete_docs_by_filters_without_a_filter_deletes_everything_listed() {
    let server = MockServer::start().await;
    mount_collection(&server, "docs", &schema()).await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/get"))
        .and(body_json(json!({"include": [], "limit": 500, "offset": 0})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ids": [["only"]]})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/v1/collections/docs/delete"))
        .and(body_json(json!({"ids": ["only"]})))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let deleted = store(&server)
        .delete_docs_by_filters("docs", &Value::Null)
        .await
        .expect("delete everything");
    assert_eq!(deleted, 1);
}

#[tokio::test]
async fn a_server_error_names_the_status_the_url_and_a_body_excerpt() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/collections/docs"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom: the index is corrupt"))
        .mount(&server)
        .await;

    let error = store(&server)
        .collection_exists("docs")
        .await
        .expect_err("server error");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(message.contains("500"), "{message}");
    assert!(
        message.contains(&format!("{}/api/v1/collections/docs", server.uri())),
        "{message}"
    );
    assert!(message.contains("boom"), "{message}");
}

#[tokio::test]
async fn a_missing_collection_fails_document_operations_with_not_found() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/collections/docs"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    let store = store(&server);

    for error in [
        store
            .add_docs("docs", vec![doc("a", vec![1.0, 0.0], 1)])
            .await
            .expect_err("add"),
        store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]))
            .await
            .expect_err("search"),
        store
            .delete_docs_by_ids("docs", &["a".to_string()])
            .await
            .expect_err("delete by ids"),
        store
            .delete_docs_by_filters("docs", &Value::Null)
            .await
            .expect_err("delete by filters"),
    ] {
        assert_eq!(error.kind(), ErrorKind::NotFound, "{error}");
    }
}

#[test]
fn compile_where_translates_the_supported_operators() {
    assert_eq!(compile_where(&Value::Null).expect("null"), None);
    assert_eq!(compile_where(&json!({})).expect("empty"), None);
    assert_eq!(
        compile_where(&json!({"$and": []})).expect("tautology"),
        None
    );
    assert_eq!(
        compile_where(&json!({"score": 3})).expect("equality"),
        Some(json!({"score": 3}))
    );
    assert_eq!(
        compile_where(&json!({"score": 3, "id": "a"})).expect("implicit and"),
        Some(json!({"$and": [{"score": 3}, {"id": "a"}]}))
    );
    assert_eq!(
        compile_where(&json!({"id": {"$eq": "a"}})).expect("$eq"),
        Some(json!({"id": "a"}))
    );
    assert_eq!(
        compile_where(&json!({"id": {"$ne": "a"}})).expect("$ne"),
        Some(json!({"id": {"$ne": "a"}}))
    );
    assert_eq!(
        compile_where(&json!({"score": {"$gt": 1, "$lte": 9}})).expect("range"),
        Some(json!({"score": {"$gt": 1, "$lte": 9}}))
    );
    assert_eq!(
        compile_where(&json!({"id": {"$in": ["a", "b"]}})).expect("$in"),
        Some(json!({"id": {"$in": ["a", "b"]}}))
    );
    assert_eq!(
        compile_where(&json!({"$or": [{"score": {"$lt": 2}}, {"id": "a"}]})).expect("$or"),
        Some(json!({"$or": [{"score": {"$lt": 2}}, {"id": "a"}]}))
    );
    assert_eq!(
        compile_where(&json!({"$and": [{"score": 1}, {"$or": [{"id": "a"}, {"id": "b"}]}]}))
            .expect("nested"),
        Some(json!({"$and": [{"score": 1}, {"$or": [{"id": "a"}, {"id": "b"}]}]}))
    );
}

#[test]
fn compile_where_rejects_filters_chroma_cannot_express() {
    for filter in [
        json!(5),
        json!({"meta.owner": "a"}),
        json!({"tags": {"$contains": "rust"}}),
        json!({"$not": {"score": 1}}),
        json!({"score": {"$exists": true}}),
        json!({"score": {"$eq": null}}),
        json!({"score": {"gte": 1}}),
        json!({"score": [1, 2]}),
        json!({"score": {"$in": 1}}),
        json!({"score": {"nested": {"a": 1}}}),
        json!({"$or": []}),
        json!({"$and": 1}),
    ] {
        let error = compile_where(&filter).expect_err("inexpressible filter");
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{filter}: {error}");
    }
}

#[test]
fn normalize_distance_stays_inside_the_unit_interval() {
    let close = |actual: f32, expected: f32| {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    };

    close(normalize_distance(Metric::Cosine, 0.0), 1.0);
    close(normalize_distance(Metric::Cosine, 1.0), 0.5);
    close(normalize_distance(Metric::Cosine, 4.0), 0.0);
    close(normalize_distance(Metric::Dot, 0.5), 0.75);
    close(normalize_distance(Metric::Euclidean, 0.0), 1.0);
    close(normalize_distance(Metric::Euclidean, 1.0), 0.5);
    close(normalize_distance(Metric::Euclidean, -3.0), 0.0);
    close(normalize_distance(Metric::Cosine, f64::INFINITY), 0.0);
    close(normalize_distance(Metric::Cosine, f64::NAN), 0.0);
    close(normalize_distance(Metric::Euclidean, f64::NAN), 0.0);
}

#[test]
fn encode_segment_escapes_path_characters() {
    assert_eq!(encode_segment("docs"), "docs");
    assert_eq!(encode_segment("a/b c"), "a%2Fb%20c");
}

#[test]
fn config_round_trips_and_rejects_an_empty_endpoint() {
    let config = ChromaConfig::new("http://localhost:8000")
        .with_metric(Metric::Euclidean)
        .with_schema_metadata_key("custom:schema");
    let value = serde_json::to_value(&config).expect("serialize config");
    assert_eq!(value["endpoint"], "http://localhost:8000");
    assert_eq!(value["metric"], "euclidean");
    assert_eq!(value["schema_metadata_key"], "custom:schema");

    let restored: ChromaConfig =
        serde_json::from_value(json!({"endpoint": "http://localhost:8000"}))
            .expect("deserialize config");
    assert_eq!(restored.metric, Metric::Cosine);
    assert_eq!(restored.timeout, Duration::from_secs(30));
    assert_eq!(restored.tenant, None);

    let error = ChromaVectorStore::new(ChromaConfig::new("   ")).expect_err("empty endpoint");
    assert_eq!(error.kind(), ErrorKind::Config);
}
