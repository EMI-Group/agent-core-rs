//! Crate-boundary integration suite for the live Elasticsearch vector store.
//!
//! The peer is a `wiremock` server, so every case asserts both directions: the
//! REST request the store sent (path, query string, body) and the kernel result
//! it produced from the mocked response. The in-module unit tests cover the pure
//! builders (`src/store/elasticsearch/tests.rs`).
//!
//! Python counterpart: `tests/unit_tests/extensions/store/test_es_vector_store.py`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::store::{FieldSchema, VectorDataType};
use openjiuwen_extensions::store::elasticsearch::{
    ElasticsearchVectorStore, ElasticsearchVectorStoreConfig,
};
use openjiuwen_extensions::store::{CollectionSchema, VectorQuery, VectorStore};
use serde_json::{json, Value};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The index under test: prefix `agent_vector`, collection `docs`.
const INDEX: &str = "/agent_vector__docs";

/// A schema with a keyword primary key, one 3-dimensional vector field, and a
/// keyword payload field.
fn schema() -> CollectionSchema {
    CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(3),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
        FieldSchema::new("title", VectorDataType::Varchar),
    ])
}

/// The `mappings` object the store must send for [`schema`].
fn mappings() -> Value {
    json!({
        "dynamic": "strict",
        "properties": {
            "id": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 3, "index": true, "similarity": "cosine"},
            "title": {"type": "keyword"},
        },
        "_meta": {"openjiuwen_schema": serde_json::to_value(schema()).unwrap()},
    })
}

/// A store connected to `server` (no request is sent by `connect`).
async fn connect(server: &MockServer) -> Arc<dyn VectorStore> {
    ElasticsearchVectorStore::connect(ElasticsearchVectorStoreConfig {
        url: server.uri(),
        index_prefix: "agent_vector".to_string(),
        dimension: 3,
    })
    .await
    .expect("connecting performs no I/O and accepts the mock endpoint")
}

/// Mounts the `GET /{index}/_mapping` mock that every document operation needs,
/// expecting `times` requests.
async fn mount_schema(server: &MockServer, times: u64) {
    Mock::given(method("GET"))
        .and(path(format!("{INDEX}/_mapping")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "agent_vector__docs": {"mappings": mappings()},
        })))
        .expect(times)
        .mount(server)
        .await;
}

/// The single request `server` recorded for `method` + `path`.
async fn request_body(server: &MockServer, method_name: &str, request_path: &str) -> Vec<u8> {
    let requests = server
        .received_requests()
        .await
        .expect("the mock records requests");
    let request = requests
        .iter()
        .find(|request| {
            request.method.as_str() == method_name && request.url.path() == request_path
        })
        .unwrap_or_else(|| panic!("no {method_name} {request_path} request was recorded"));
    request.body.clone()
}

#[tokio::test]
async fn create_read_and_delete_a_collection() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path(INDEX))
        .and(body_json(json!({"mappings": mappings()})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"acknowledged": true})))
        .expect(1)
        .mount(&server)
        .await;
    // `collection_exists` + `delete_collection`'s pre-flight `HEAD`.
    Mock::given(method("HEAD"))
        .and(path(INDEX))
        .respond_with(ResponseTemplate::new(200))
        .expect(2)
        .mount(&server)
        .await;
    mount_schema(&server, 1).await;
    Mock::given(method("DELETE"))
        .and(path(INDEX))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"acknowledged": true})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/_cat/indices/agent_vector__*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"index": "agent_vector__docs"},
            {"index": "unrelated_index"},
        ])))
        .expect(1)
        .mount(&server)
        .await;

    let store = connect(&server).await;
    store.create_collection("docs", schema()).await.unwrap();
    assert!(store.collection_exists("docs").await.unwrap());
    assert_eq!(store.get_schema("docs").await.unwrap(), Some(schema()));
    store.delete_collection("docs").await.unwrap();
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        vec!["docs".to_string()]
    );
}

#[tokio::test]
async fn creating_an_existing_index_is_a_validation_error() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path(INDEX))
        .respond_with(ResponseTemplate::new(409).set_body_json(json!({
            "error": {
                "type": "resource_already_exists_exception",
                "reason": "index [agent_vector__docs/abc] already exists",
            },
            "status": 409,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let error = connect(&server)
        .await
        .create_collection("docs", schema())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("docs"),
        "the error must name the collection: {}",
        error.message()
    );
}

#[tokio::test]
async fn add_docs_sends_one_ndjson_bulk_request() {
    let server = MockServer::start().await;
    mount_schema(&server, 1).await;
    Mock::given(method("POST"))
        .and(path("/_bulk"))
        .and(header("content-type", "application/x-ndjson"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "took": 2,
            "errors": false,
            "items": [
                {"index": {"_index": "agent_vector__docs", "_id": "a", "status": 201}},
                {"index": {"_index": "agent_vector__docs", "_id": "b", "status": 201}},
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let added = connect(&server)
        .await
        .add_docs(
            "docs",
            vec![
                json!({"id": "a", "embedding": [1.0, 0.0, 0.0], "title": "A"})
                    .as_object()
                    .cloned()
                    .unwrap(),
                json!({"id": "b", "embedding": [0.0, 1.0, 0.0], "title": "B"})
                    .as_object()
                    .cloned()
                    .unwrap(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(added, 2);

    let body = request_body(&server, "POST", "/_bulk").await;
    let body = String::from_utf8(body).expect("the bulk body is UTF-8");
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 4, "two documents are two NDJSON line pairs");
    assert_eq!(
        serde_json::from_str::<Value>(lines[0]).unwrap(),
        json!({"index": {"_index": "agent_vector__docs", "_id": "a"}})
    );
    assert_eq!(
        serde_json::from_str::<Value>(lines[1]).unwrap(),
        json!({"id": "a", "embedding": [1.0, 0.0, 0.0], "title": "A"})
    );
    assert_eq!(
        serde_json::from_str::<Value>(lines[2]).unwrap(),
        json!({"index": {"_index": "agent_vector__docs", "_id": "b"}})
    );
    assert_eq!(
        serde_json::from_str::<Value>(lines[3]).unwrap(),
        json!({"id": "b", "embedding": [0.0, 1.0, 0.0], "title": "B"})
    );
}

#[tokio::test]
async fn add_docs_fills_defaults_and_mints_auto_ids() {
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
            default_value: Some(json!("none")),
            ..FieldSchema::new("tag", VectorDataType::Varchar)
        },
    ]);
    Mock::given(method("GET"))
        .and(path(format!("{INDEX}/_mapping")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "agent_vector__docs": {"mappings": json!({
                "dynamic": "strict",
                "properties": {"id": {"type": "keyword"}},
                "_meta": {"openjiuwen_schema": serde_json::to_value(&schema).unwrap()},
            })},
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/_bulk"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [{"index": {"_id": "generated", "status": 201}}],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let added = connect(&server)
        .await
        .add_docs(
            "docs",
            vec![json!({"embedding": [1.0, 0.0]})
                .as_object()
                .cloned()
                .unwrap()],
        )
        .await
        .unwrap();
    assert_eq!(added, 1);

    let body = request_body(&server, "POST", "/_bulk").await;
    let body = String::from_utf8(body).expect("the bulk body is UTF-8");
    let lines: Vec<&str> = body.lines().collect();
    let action: Value = serde_json::from_str(lines[0]).unwrap();
    let id = action["index"]["_id"].as_str().expect("the id was minted");
    assert_eq!(id.len(), 36, "an `auto_id` primary key is a UUID");
    let source: Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(
        source["id"],
        json!(id),
        "the generated id is in the document"
    );
    assert_eq!(source["tag"], json!("none"), "the default was filled in");
}

#[tokio::test]
async fn a_bulk_item_error_names_the_document() {
    let server = MockServer::start().await;
    mount_schema(&server, 1).await;
    Mock::given(method("POST"))
        .and(path("/_bulk"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": true,
            "items": [{"index": {
                "_id": "bad",
                "status": 400,
                "error": {
                    "type": "mapper_parsing_exception",
                    "reason": "failed to parse field [embedding]",
                },
            }}],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let error = connect(&server)
        .await
        .add_docs(
            "docs",
            vec![json!({"id": "bad", "embedding": [1.0, 0.0, 0.0]})
                .as_object()
                .cloned()
                .unwrap()],
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("\"bad\""), "{}", error.message());
    assert!(
        error
            .message()
            .contains("failed to parse field [embedding]"),
        "the Elasticsearch reason is kept: {}",
        error.message()
    );
}

#[tokio::test]
async fn search_sends_a_knn_query_and_orders_the_hits() {
    let server = MockServer::start().await;
    mount_schema(&server, 1).await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_search")))
        .and(body_json(json!({
            "knn": {
                "field": "embedding",
                "query_vector": [1.0, 0.0, 0.0],
                "k": 3,
                "num_candidates": 100,
                "filter": {"range": {"year": {"gte": 2020}}},
            },
            "size": 3,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hits": {"hits": [
                {"_id": "c", "_score": 0.5, "_source": {"title": "C"}},
                {"_id": "b", "_score": 0.9, "_source": {"id": "b", "title": "B"}},
                {"_id": "a", "_score": 0.9, "_source": {
                    "id": "a", "title": "A", "_meta": {"internal": true},
                }},
                {"_id": "d", "_score": 2.0, "_source": {"id": "d", "title": "D"}},
            ]},
        })))
        .expect(1)
        .mount(&server)
        .await;

    let hits = connect(&server)
        .await
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0, 0.0])
                .with_top_k(3)
                .with_filters(json!({"year": {"$gte": 2020}})),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 3, "top_k truncates the longer hit list");
    assert_eq!(hits[0].fields["id"], json!("d"));
    assert_eq!(hits[1].fields["id"], json!("a"));
    assert_eq!(hits[2].fields["id"], json!("b"));
    assert_eq!(json!(hits[0].score), json!(1.0_f32), "the score is clamped");
    assert_eq!(json!(hits[1].score), json!(0.9_f32));
    assert!(
        !hits[1].fields.contains_key("_meta"),
        "`_meta` is stripped from the stored source"
    );
    assert_eq!(
        hits[0].fields["title"],
        json!("D"),
        "every stored field is returned"
    );
}

#[tokio::test]
async fn a_missing_index_is_reported_as_not_found() {
    let server = MockServer::start().await;
    mount_schema(&server, 1).await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_search")))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {
                "type": "index_not_found_exception",
                "reason": "no such index [agent_vector__docs]",
            },
            "status": 404,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let error = connect(&server)
        .await
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error.message().contains("no such index"),
        "the Elasticsearch reason is kept: {}",
        error.message()
    );
}

#[tokio::test]
async fn a_server_failure_is_a_transport_error_with_the_reason() {
    let server = MockServer::start().await;
    mount_schema(&server, 1).await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_search")))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({
            "error": {"reason": "cluster is unavailable"},
        })))
        .expect(1)
        .mount(&server)
        .await;

    let error = connect(&server)
        .await
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.message().contains("cluster is unavailable"),
        "the Elasticsearch reason is kept: {}",
        error.message()
    );
    assert!(error.message().contains("503"), "{}", error.message());
}

#[tokio::test]
async fn a_rejected_bulk_request_is_a_validation_error() {
    let server = MockServer::start().await;
    mount_schema(&server, 1).await;
    Mock::given(method("POST"))
        .and(path("/_bulk"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {"type": "illegal_argument_exception", "reason": "malformed bulk body"},
            "status": 400,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let error = connect(&server)
        .await
        .add_docs(
            "docs",
            vec![json!({"id": "a", "embedding": [1.0, 0.0, 0.0]})
                .as_object()
                .cloned()
                .unwrap()],
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("malformed bulk body"),
        "{}",
        error.message()
    );
}

#[tokio::test]
async fn invalid_dimensions_never_reach_elasticsearch() {
    let server = MockServer::start().await;
    mount_schema(&server, 2).await;
    // `.expect(0)`: a request would fail the mock at teardown.
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_search")))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/_bulk"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    let store = connect(&server).await;
    let query = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(query.kind(), ErrorKind::Validation);
    assert!(query.message().contains("dimension"), "{}", query.message());

    let document = store
        .add_docs(
            "docs",
            vec![json!({"id": "a", "embedding": [1.0, 0.0]})
                .as_object()
                .cloned()
                .unwrap()],
        )
        .await
        .unwrap_err();
    assert_eq!(document.kind(), ErrorKind::Validation);
    assert!(
        document.message().contains("expects 3 components"),
        "{}",
        document.message()
    );
}

#[tokio::test]
async fn duplicate_primary_keys_in_one_batch_are_rejected_locally() {
    let server = MockServer::start().await;
    mount_schema(&server, 1).await;
    Mock::given(method("POST"))
        .and(path("/_bulk"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    let error = connect(&server)
        .await
        .add_docs(
            "docs",
            vec![
                json!({"id": "a", "embedding": [1.0, 0.0, 0.0]})
                    .as_object()
                    .cloned()
                    .unwrap(),
                json!({"id": "a", "embedding": [0.0, 1.0, 0.0]})
                    .as_object()
                    .cloned()
                    .unwrap(),
            ],
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("duplicate primary key"),
        "{}",
        error.message()
    );
}

#[tokio::test]
async fn deleting_documents_uses_delete_by_query() {
    let server = MockServer::start().await;
    mount_schema(&server, 3).await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_delete_by_query")))
        .and(body_json(json!({"query": {"terms": {"id": ["a", "b"]}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"deleted": 2})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_delete_by_query")))
        .and(body_json(json!({"query": {"term": {"title": "A"}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"deleted": 1})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_delete_by_query")))
        .and(body_json(json!({"query": {"match_all": {}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"deleted": 3})))
        .expect(1)
        .mount(&server)
        .await;

    let store = connect(&server).await;
    assert_eq!(
        store
            .delete_docs_by_ids("docs", &["a".to_string(), "b".to_string()])
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        store
            .delete_docs_by_filters("docs", &json!({"title": "A"}))
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store
            .delete_docs_by_filters("docs", &Value::Null)
            .await
            .unwrap(),
        3,
        "a null filter deletes every document"
    );
}

#[tokio::test]
async fn deleting_a_missing_collection_is_a_no_op() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path(INDEX))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path(INDEX))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    connect(&server)
        .await
        .delete_collection("docs")
        .await
        .expect("a missing collection is a no-op");
}
