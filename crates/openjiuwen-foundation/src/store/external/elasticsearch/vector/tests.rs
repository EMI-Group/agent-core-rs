//! Offline `wiremock` tests for the Elasticsearch vector backend.
//!
//! Every test talks to a loopback mock server, so the suite needs no
//! Elasticsearch installation, no network, and no credentials. The document
//! validation, the score normalization and the local filter fallback are covered
//! through the same mock server so the whole request path is exercised.

use super::*;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::http::HttpClientConfig;
use serde_json::json;
use wiremock::matchers::{body_json, body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::store::vector::Metric;

/// The index name of the `docs` collection below.
const INDEX: &str = "openjiuwen__docs";

/// A schema with a primary key, one embedding and a string field.
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
        FieldSchema::new("kind", VectorDataType::Varchar),
    ])
}

/// The [`schema`] with `enable_dynamic_field` switched on.
fn dynamic_schema() -> CollectionSchema {
    CollectionSchema {
        enable_dynamic_field: true,
        ..schema()
    }
}

/// The schema serialized the way the index metadata stores it.
fn schema_json(schema: &CollectionSchema) -> String {
    serde_json::to_string(schema).expect("serialize schema")
}

/// The `properties` mapping of [`schema`] with the cosine similarity.
fn properties() -> Value {
    json!({
        "id": {"type": "keyword"},
        "embedding": {"type": "dense_vector", "dims": 2, "index": true, "similarity": "cosine"},
        "kind": {"type": "keyword"},
    })
}

/// The `GET /{index}` payload of an index this store created for `schema`.
fn index_payload(schema: &CollectionSchema) -> Value {
    json!({
        "openjiuwen__docs": {
            "mappings": {
                "_meta": {"openjiuwen_schema": schema_json(schema)},
                "properties": properties(),
            },
        },
    })
}

/// Mounts the `GET /{index}` response that reports `schema`.
async fn mount_schema(server: &MockServer, schema: &CollectionSchema) {
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(index_payload(schema)))
        .mount(server)
        .await;
}

/// Mounts the `HEAD /{index}` response of a collection that exists or not.
async fn mount_exists(server: &MockServer, exists: bool) {
    let status = if exists { 200 } else { 404 };
    Mock::given(method("HEAD"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(status))
        .mount(server)
        .await;
}

/// Builds a document for the [`schema`] above.
fn doc(id: &str, embedding: Vec<f64>, kind: &str) -> Map<String, Value> {
    json!({"id": id, "embedding": embedding, "kind": kind})
        .as_object()
        .cloned()
        .expect("object literal")
}

/// Builds a store pointing at `server` with every default.
fn store(server: &MockServer) -> ElasticsearchVectorStore {
    ElasticsearchVectorStore::new(ElasticsearchConfig::new(server.uri())).expect("build store")
}

/// The body of the single request whose path ends with `suffix`.
async fn recorded_body(server: &MockServer, suffix: &str) -> String {
    let requests = server.received_requests().await.expect("received requests");
    let request = requests
        .iter()
        .find(|request| request.url.path().ends_with(suffix))
        .unwrap_or_else(|| panic!("no request to {suffix}: {requests:?}"));
    String::from_utf8(request.body.clone()).expect("utf-8 body")
}

#[test]
fn configuration_is_exposed_and_validated() {
    let config = ElasticsearchConfig::new("http://es:9200").with_index_prefix("app");
    let http = HttpClient::new(HttpClientConfig::default()).expect("http client");
    let store = ElasticsearchVectorStore::with_client(http, config).expect("build store");
    assert_eq!(store.config().index_prefix, "app");

    let error = ElasticsearchVectorStore::new(ElasticsearchConfig::default())
        .expect_err("an empty endpoint is rejected");
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[tokio::test]
async fn create_collection_sends_the_mapping_and_the_schema_marker() {
    let server = MockServer::start().await;
    let schema = schema();
    Mock::given(method("PUT"))
        .and(path("/openjiuwen__docs"))
        .and(body_json(json!({
            "mappings": {
                "_meta": {"openjiuwen_schema": schema_json(&schema)},
                "properties": properties(),
            },
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "acknowledged": true,
            "index": INDEX,
        })))
        .expect(1)
        .mount(&server)
        .await;

    store(&server)
        .create_collection("docs", schema)
        .await
        .expect("create collection");
}

#[tokio::test]
async fn create_collection_uses_basic_auth_and_the_configured_metric() {
    let server = MockServer::start().await;
    let schema = schema();
    Mock::given(method("PUT"))
        .and(path("/openjiuwen__docs"))
        .and(header("authorization", "Basic ZWxhc3RpYzpzZWNyZXQ="))
        .and(body_json(json!({
            "mappings": {
                "_meta": {"openjiuwen_schema": schema_json(&schema)},
                "properties": {
                    "id": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 2,
                        "index": true,
                        "similarity": "dot_product",
                    },
                    "kind": {"type": "keyword"},
                },
            },
        })))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    ElasticsearchVectorStore::new(
        ElasticsearchConfig::new(server.uri())
            .with_username("elastic")
            .with_password("secret")
            .with_metric(Metric::Dot),
    )
    .expect("build store")
    .create_collection("docs", schema)
    .await
    .expect("create collection");
}

#[tokio::test]
async fn create_collection_uses_the_api_key_and_the_configured_prefix() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/app__docs"))
        .and(header("authorization", "ApiKey key-123"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    ElasticsearchVectorStore::new(
        ElasticsearchConfig::new(server.uri())
            .with_index_prefix("app")
            .with_api_key("key-123"),
    )
    .expect("build store")
    .create_collection("docs", schema())
    .await
    .expect("create collection");
}

#[tokio::test]
async fn create_collection_maps_an_existing_index_to_validation() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "type": "resource_already_exists_exception",
                "reason": "index [openjiuwen__docs] already exists",
            },
            "status": 400,
        })))
        .mount(&server)
        .await;

    let error = store(&server)
        .create_collection("docs", schema())
        .await
        .expect_err("duplicate index");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("already exists"), "{error}");
}

#[tokio::test]
async fn create_collection_rejects_an_array_field_without_http() {
    let server = MockServer::start().await;
    let schema = CollectionSchema::new(vec![
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

    let error = store(&server)
        .create_collection("docs", schema)
        .await
        .expect_err("array field");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("tags"), "{error}");
    let requests = server.received_requests().await.expect("received requests");
    assert!(requests.is_empty(), "no request must be sent: {requests:?}");
}

#[tokio::test]
async fn collection_exists_and_delete_collection_handle_missing_indexes() {
    let present = MockServer::start().await;
    mount_exists(&present, true).await;
    assert!(store(&present)
        .collection_exists("docs")
        .await
        .expect("existing collection"));

    let missing = MockServer::start().await;
    mount_exists(&missing, false).await;
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
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&deletable)
        .await;
    store(&deletable)
        .delete_collection("docs")
        .await
        .expect("delete collection");
}

#[tokio::test]
async fn get_schema_round_trips_the_written_schema() {
    let server = MockServer::start().await;
    let schema = schema();
    mount_schema(&server, &schema).await;

    assert_eq!(
        store(&server).get_schema("docs").await.expect("schema"),
        Some(schema)
    );
}

#[tokio::test]
async fn get_schema_reconstructs_an_index_without_the_schema_marker() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "openjiuwen__docs": {"mappings": {"properties": properties()}},
        })))
        .mount(&server)
        .await;

    let schema = store(&server)
        .get_schema("docs")
        .await
        .expect("schema")
        .expect("existing index");
    assert_eq!(
        schema.primary_key().map(|field| field.name.as_str()),
        Some("id")
    );
    assert_eq!(
        schema.default_vector_field().and_then(|field| field.dim),
        Some(2)
    );
    assert_eq!(schema.fields.len(), 3);
    assert!(!schema.enable_dynamic_field);
}

#[tokio::test]
async fn get_schema_reports_a_missing_index() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"type": "index_not_found_exception"},
        })))
        .mount(&server)
        .await;

    assert_eq!(
        store(&server).get_schema("docs").await.expect("absent"),
        None
    );
}

#[tokio::test]
async fn add_docs_sends_two_ndjson_lines_per_document() {
    let server = MockServer::start().await;
    mount_schema(&server, &schema()).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .and(header("content-type", "application/x-ndjson"))
        .and(body_string_contains(r#"{"create":{"_id":"a"}}"#))
        .and(body_string_contains(
            r#"{"id":"b","embedding":[0.0,1.0],"kind":"b"}"#,
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [
                {"create": {"_id": "a", "status": 201, "result": "created"}},
                {"create": {"_id": "b", "status": 201, "result": "created"}},
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let inserted = store(&server)
        .add_docs(
            "docs",
            vec![doc("a", vec![1.0, 0.0], "a"), doc("b", vec![0.0, 1.0], "b")],
        )
        .await
        .expect("add documents");
    assert_eq!(inserted, 2);

    let body = recorded_body(&server, "/_bulk").await;
    assert_eq!(
        body,
        concat!(
            "{\"create\":{\"_id\":\"a\"}}\n",
            "{\"id\":\"a\",\"embedding\":[1.0,0.0],\"kind\":\"a\"}\n",
            "{\"create\":{\"_id\":\"b\"}}\n",
            "{\"id\":\"b\",\"embedding\":[0.0,1.0],\"kind\":\"b\"}\n",
        )
    );
}

#[tokio::test]
async fn add_docs_reports_a_duplicate_primary_key_as_validation() {
    for error_type in [
        "version_conflict_engine_exception",
        "document_already_exists",
    ] {
        let server = MockServer::start().await;
        mount_schema(&server, &schema()).await;
        Mock::given(method("POST"))
            .and(path("/openjiuwen__docs/_bulk"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "errors": true,
                "items": [{
                    "create": {
                        "_id": "a",
                        "status": 409,
                        "error": {"type": error_type, "reason": "document already exists"},
                    },
                }],
            })))
            .mount(&server)
            .await;

        let error = store(&server)
            .add_docs("docs", vec![doc("a", vec![1.0, 0.0], "a")])
            .await
            .expect_err("duplicate primary key");
        assert_eq!(error.kind(), ErrorKind::Validation, "type: {error_type}");
        assert!(error.to_string().contains("\"a\""), "{error}");
    }
}

#[tokio::test]
async fn add_docs_surfaces_another_bulk_failure_as_a_transport_error() {
    let server = MockServer::start().await;
    mount_schema(&server, &schema()).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": true,
            "items": [{
                "create": {
                    "_id": "a",
                    "status": 400,
                    "error": {"type": "mapper_parsing_exception", "reason": "failed to parse"},
                },
            }],
        })))
        .mount(&server)
        .await;

    let error = store(&server)
        .add_docs("docs", vec![doc("a", vec![1.0, 0.0], "a")])
        .await
        .expect_err("bulk failure");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(message.contains("mapper_parsing_exception"), "{message}");
    assert!(message.contains("failed to parse"), "{message}");
}

#[tokio::test]
async fn add_docs_validates_the_batch_before_sending_anything() {
    let server = MockServer::start().await;
    mount_schema(&server, &schema()).await;
    let store = store(&server);

    let duplicate = vec![doc("a", vec![1.0, 0.0], "a"), doc("a", vec![0.0, 1.0], "b")];
    let error = store
        .add_docs("docs", duplicate)
        .await
        .expect_err("duplicate in the batch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("duplicate primary key"),
        "{error}"
    );

    let short_vector = vec![doc("a", vec![1.0], "a")];
    let error = store
        .add_docs("docs", short_vector)
        .await
        .expect_err("wrong dimension");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("expects 2 components"),
        "{error}"
    );

    let unknown = json!({"id": "a", "embedding": [1.0, 0.0], "extra": 1})
        .as_object()
        .cloned()
        .expect("object literal");
    let error = store
        .add_docs("docs", vec![unknown])
        .await
        .expect_err("undeclared field");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("enable_dynamic_field"),
        "{error}"
    );

    let requests = server.received_requests().await.expect("received requests");
    assert!(
        !requests
            .iter()
            .any(|request| request.url.path().ends_with("/_bulk")),
        "a rejected batch must not reach the service: {requests:?}"
    );
}

#[tokio::test]
async fn add_docs_accepts_unknown_fields_of_a_dynamic_schema() {
    let server = MockServer::start().await;
    mount_schema(&server, &dynamic_schema()).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .and(body_string_contains(
            r#"{"id":"a","embedding":[1.0,0.0],"kind":"a","extra":1}"#,
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"errors": false})))
        .expect(1)
        .mount(&server)
        .await;

    let document = json!({"id": "a", "embedding": [1.0, 0.0], "kind": "a", "extra": 1})
        .as_object()
        .cloned()
        .expect("object literal");
    let inserted = store(&server)
        .add_docs("docs", vec![document])
        .await
        .expect("add document");
    assert_eq!(inserted, 1);
}

#[tokio::test]
async fn add_docs_generates_an_auto_id_and_applies_defaults() {
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
        FieldSchema {
            default_value: Some(json!("misc")),
            ..FieldSchema::new("kind", VectorDataType::Varchar)
        },
    ]);
    mount_schema(&server, &schema).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .and(body_string_contains(r#""kind":"misc""#))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"errors": false})))
        .expect(1)
        .mount(&server)
        .await;

    let document = json!({"embedding": [1.0, 0.0]})
        .as_object()
        .cloned()
        .expect("object literal");
    assert_eq!(
        store(&server)
            .add_docs("docs", vec![document])
            .await
            .expect("add document"),
        1
    );
    let body = recorded_body(&server, "/_bulk").await;
    let action: Value =
        serde_json::from_str(body.lines().next().expect("action line")).expect("parse action");
    let generated = action["create"]["_id"].as_str().expect("generated id");
    assert_eq!(generated.len(), 36, "expected a uuid, got {generated}");
}

#[tokio::test]
async fn add_docs_reports_an_unknown_collection() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let error = store(&server)
        .add_docs("docs", vec![doc("a", vec![1.0, 0.0], "a")])
        .await
        .expect_err("missing collection");
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[tokio::test]
async fn search_sends_the_knn_clause_and_ranks_the_hits() {
    let server = MockServer::start().await;
    mount_schema(&server, &schema()).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_search"))
        .and(body_json(json!({
            "knn": {
                "field": "embedding",
                "query_vector": [1.0, 0.0],
                "k": 2,
                "num_candidates": 8,
                "filter": {"term": {"kind": "a"}},
            },
            "size": 2,
            "_source": true,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hits": {
                "hits": [
                    {
                        "_id": "a",
                        "_score": 0.9,
                        "_source": {"id": "a", "embedding": [1.0, 0.0], "kind": "a"},
                    },
                    {
                        "_id": "c",
                        "_score": 0.9,
                        "_source": {"kind": "a"},
                    },
                    {
                        "_id": "b",
                        "_score": 0.4,
                        "_source": {"id": "b", "embedding": [0.0, 1.0], "kind": "a"},
                    },
                ],
            },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let hits = store(&server)
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0])
                .with_top_k(2)
                .with_filters(json!({"kind": "a"})),
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 2);
    // Equal scores are broken by primary key, and `_id` fills a missing key.
    assert_eq!(hits[0].fields["id"], json!("a"));
    assert!(
        (hits[0].score - 0.9).abs() < 1e-6,
        "score: {}",
        hits[0].score
    );
    assert_eq!(hits[1].fields["id"], json!("c"));
    assert!(
        (hits[1].score - 0.9).abs() < 1e-6,
        "score: {}",
        hits[1].score
    );
}

#[tokio::test]
async fn search_clamps_scores_into_the_unit_interval() {
    let server = MockServer::start().await;
    mount_schema(&server, &schema()).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hits": {
                "hits": [
                    {"_id": "high", "_score": 4.5, "_source": {"id": "high"}},
                    {"_id": "negative", "_score": -2.0, "_source": {"id": "negative"}},
                    {"_id": "absent", "_source": {"id": "absent"}},
                    {"_id": "text", "_score": "0.5", "_source": {"id": "text"}},
                ],
            },
        })))
        .mount(&server)
        .await;

    let hits = store(&server)
        .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(4))
        .await
        .expect("search");
    let scores: Vec<f32> = hits.iter().map(|hit| hit.score).collect();
    assert_eq!(scores, vec![1.0, 0.0, 0.0, 0.0]);
    // Ties keep the primary-key order, so the order is deterministic.
    let ids: Vec<&Value> = hits.iter().map(|hit| &hit.fields["id"]).collect();
    assert_eq!(
        ids,
        vec![
            &json!("high"),
            &json!("absent"),
            &json!("negative"),
            &json!("text"),
        ]
    );
}

#[tokio::test]
async fn search_widens_the_window_and_filters_locally_without_a_translation() {
    let server = MockServer::start().await;
    mount_schema(&server, &schema()).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_search"))
        .and(body_json(json!({
            "knn": {
                "field": "embedding",
                "query_vector": [1.0, 0.0],
                "k": 4,
                "num_candidates": 4,
            },
            "size": 4,
            "_source": true,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hits": {
                "hits": [
                    {"_id": "a", "_score": 0.5, "_source": {"id": "a", "meta": {"owner": "ada"}}},
                    {"_id": "b", "_score": 0.9, "_source": {"id": "b", "meta": {"owner": "bob"}}},
                    {"_id": "c", "_score": 0.7, "_source": {"id": "c", "meta": {"owner": "ada"}}},
                ],
            },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let hits = store(&server)
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0])
                .with_top_k(1)
                .with_filters(json!({"meta.owner": "ada"})),
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].fields["id"], json!("c"));
}

#[tokio::test]
async fn search_rejects_a_non_vector_field_and_a_wrong_dimension() {
    let server = MockServer::start().await;
    mount_schema(&server, &schema()).await;
    let store = store(&server);

    let error = store
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0]).with_vector_field("kind"),
        )
        .await
        .expect_err("scalar field");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("FLOAT_VECTOR"), "{error}");

    let error = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .expect_err("wrong dimension");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains('3') && message.contains('2'), "{message}");

    let requests = server.received_requests().await.expect("received requests");
    assert!(
        !requests
            .iter()
            .any(|request| request.url.path().ends_with("/_search")),
        "an invalid query must not reach the service: {requests:?}"
    );
}

#[tokio::test]
async fn search_with_top_k_zero_sends_nothing() {
    let server = MockServer::start().await;
    let hits = store(&server)
        .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(0))
        .await
        .expect("empty search");
    assert_eq!(hits, Vec::<VectorSearchResult>::new());
    let requests = server.received_requests().await.expect("received requests");
    assert!(requests.is_empty(), "no request must be sent: {requests:?}");
}

#[tokio::test]
async fn delete_docs_by_ids_counts_only_deletions() {
    let server = MockServer::start().await;
    mount_exists(&server, true).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .and(body_string_contains(r#"{"delete":{"_id":"a"}}"#))
        .and(body_string_contains(r#"{"delete":{"_id":"b"}}"#))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": true,
            "items": [
                {"delete": {"_id": "a", "status": 200, "result": "deleted"}},
                {"delete": {"_id": "b", "status": 404, "result": "not_found"}},
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store = store(&server);
    let deleted = store
        .delete_docs_by_ids("docs", &["a".to_string(), "b".to_string()])
        .await
        .expect("delete documents");
    assert_eq!(deleted, 1);
    assert_eq!(
        store
            .delete_docs_by_ids("docs", &[])
            .await
            .expect("empty id list"),
        0
    );
    assert_eq!(
        recorded_body(&server, "/_bulk").await,
        "{\"delete\":{\"_id\":\"a\"}}\n{\"delete\":{\"_id\":\"b\"}}\n"
    );
}

#[tokio::test]
async fn delete_docs_by_ids_reports_a_missing_collection() {
    let server = MockServer::start().await;
    mount_exists(&server, false).await;

    let error = store(&server)
        .delete_docs_by_ids("docs", &["a".to_string()])
        .await
        .expect_err("missing collection");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains("\"docs\""), "{error}");
}

#[tokio::test]
async fn delete_docs_by_filters_uses_delete_by_query() {
    let server = MockServer::start().await;
    mount_exists(&server, true).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_delete_by_query"))
        .and(body_json(json!({"query": {"term": {"kind": "a"}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "deleted": 3,
            "total": 3,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let deleted = store(&server)
        .delete_docs_by_filters("docs", &json!({"kind": "a"}))
        .await
        .expect("delete documents");
    assert_eq!(deleted, 3);
}

#[tokio::test]
async fn delete_docs_by_filters_pages_and_filters_locally_without_a_translation() {
    let server = MockServer::start().await;
    mount_exists(&server, true).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_search"))
        .and(body_json(json!({
            "query": {"match_all": {}},
            "_source": true,
            "size": FILTER_DELETE_PAGE,
            "from": 0,
            "sort": ["_doc"],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hits": {
                "hits": [
                    {"_id": "a", "_source": {"id": "a", "meta": {"owner": "ada"}}},
                    {"_id": "b", "_source": {"id": "b", "meta": {"owner": "bob"}}},
                    {"_id": "c", "_source": {"meta": {"owner": "ada"}}},
                ],
            },
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .and(body_string_contains(r#"{"delete":{"_id":"a"}}"#))
        .and(body_string_contains(r#"{"delete":{"_id":"c"}}"#))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [
                {"delete": {"_id": "a", "status": 200, "result": "deleted"}},
                {"delete": {"_id": "c", "status": 200, "result": "deleted"}},
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let deleted = store(&server)
        .delete_docs_by_filters("docs", &json!({"meta.owner": "ada"}))
        .await
        .expect("delete documents");
    assert_eq!(deleted, 2);
    assert_eq!(
        recorded_body(&server, "/_bulk").await,
        "{\"delete\":{\"_id\":\"a\"}}\n{\"delete\":{\"_id\":\"c\"}}\n"
    );
}

#[tokio::test]
async fn delete_docs_by_filters_reports_a_missing_collection() {
    let server = MockServer::start().await;
    mount_exists(&server, false).await;

    let error = store(&server)
        .delete_docs_by_filters("docs", &json!({"kind": "a"}))
        .await
        .expect_err("missing collection");
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[tokio::test]
async fn list_collection_names_strips_the_prefix_and_sorts() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "openjiuwen__beta": {"mappings": {}},
            "openjiuwen__alpha": {"mappings": {}},
            "openjiuwen__alpha2": {"mappings": {}},
            "other__gamma": {"mappings": {}},
            "openjiuwen__": {"mappings": {}},
        })))
        .mount(&server)
        .await;

    assert_eq!(
        store(&server)
            .list_collection_names()
            .await
            .expect("list collections"),
        ["alpha", "alpha2", "beta"]
    );

    let missing = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__*"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&missing)
        .await;
    let names = store(&missing)
        .list_collection_names()
        .await
        .expect("no collections");
    assert_eq!(names, Vec::<String>::new());
}

#[tokio::test]
async fn a_service_failure_names_the_status_the_index_and_the_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(500).set_body_string("cluster is on fire"))
        .mount(&server)
        .await;

    let error = store(&server)
        .get_schema("docs")
        .await
        .expect_err("service failure");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(message.contains("500"), "message: {message}");
    assert!(message.contains("openjiuwen__docs"), "message: {message}");
    assert!(message.contains("cluster is on fire"), "message: {message}");
}
