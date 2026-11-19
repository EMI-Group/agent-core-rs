//! Crate-boundary integration suite for the live Elasticsearch backends.
//!
//! The vector-store section covers `ElasticsearchVectorStore` and the
//! document-store section covers the kernel's `ElasticsearchDocumentStore` as
//! wired by [`build_document_store`]. The peer is a `wiremock` server, so every
//! case asserts both directions: the REST request the store sent (method, path,
//! query string, body, headers) and the kernel result it produced from the
//! mocked response. The in-module unit tests cover the pure builders
//! (`src/store/elasticsearch/tests.rs`).
//!
//! Python counterparts: `tests/unit_tests/extensions/store/test_es_vector_store.py`
//! (vector store) and `tests/unit_tests/foundation/store/test_es_document_store.py`
//! (document store).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use base64::Engine as _;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::store::{FieldSchema, VectorDataType};
use openjiuwen_extensions::store::elasticsearch::{
    ElasticsearchVectorStore, ElasticsearchVectorStoreConfig,
};
use openjiuwen_extensions::store::{
    build_document_store, build_vector_store, CollectionSchema, Document, DocumentStore,
    ElasticsearchConfig, ElasticsearchDocumentStore, Query, SortOrder, StoreBackend, VectorQuery,
    VectorStore,
};
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
        ..ElasticsearchVectorStoreConfig::default()
    })
    .await
    .expect("connecting performs no I/O and accepts the mock endpoint")
}

/// The `GET /{index}/_mapping` + `POST /{index}/_search` pair a search sends,
/// both carrying `authorization` (when `authorization` is `Some`).
async fn mount_authenticated_search(server: &MockServer, authorization: Option<&str>) {
    let mapping = Mock::given(method("GET")).and(path(format!("{INDEX}/_mapping")));
    let search = Mock::given(method("POST")).and(path(format!("{INDEX}/_search")));
    let (mapping, search) = match authorization {
        Some(expected) => (
            mapping.and(header("authorization", expected)),
            search.and(header("authorization", expected)),
        ),
        None => (mapping, search),
    };
    mapping
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "agent_vector__docs": {"mappings": mappings()},
        })))
        .expect(1)
        .mount(server)
        .await;
    search
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hits": {"hits": [
                {"_id": "a", "_score": 0.9, "_source": {"id": "a", "title": "A"}},
            ]},
        })))
        .expect(1)
        .mount(server)
        .await;
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

// ---------------------------------------------------------------------------
// Vector-store credentials
//
// Both Elasticsearch adapters resolve credentials through one shared helper, so
// a secured cluster authenticates the vector store exactly as it authenticates
// the document store: the URL userinfo supplies HTTP basic credentials, the
// explicit `username`/`password` fields win over it field by field, and an
// `api_key` is sent as `Authorization: ApiKey <key>` when no basic-auth pair
// applies.
// ---------------------------------------------------------------------------

/// Runs one real k-NN search against `server`, which mounts the search pair.
async fn search_once(store: &Arc<dyn VectorStore>) -> usize {
    store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(1))
        .await
        .expect("search")
        .len()
}

#[tokio::test]
async fn vector_basic_auth_comes_from_the_url_userinfo() {
    let server = MockServer::start().await;
    let expected = basic("elastic", "s3cret");
    mount_authenticated_search(&server, Some(&expected)).await;

    let store = ElasticsearchVectorStore::connect(ElasticsearchVectorStoreConfig {
        url: uri_with_userinfo(&server, "elastic", "s3cret"),
        index_prefix: "agent_vector".to_string(),
        dimension: 3,
        ..ElasticsearchVectorStoreConfig::default()
    })
    .await
    .expect("the userinfo endpoint is accepted");
    assert_eq!(search_once(&store).await, 1);
    assert_eq!(
        authorization_headers(&server).await,
        vec![expected.clone(), expected],
        "the mapping read and the search both carry the userinfo credentials"
    );
}

#[tokio::test]
async fn vector_explicit_credentials_win_over_the_url_userinfo() {
    let server = MockServer::start().await;
    let expected = basic("alice", "secret");
    mount_authenticated_search(&server, Some(&expected)).await;

    let store = ElasticsearchVectorStore::connect(ElasticsearchVectorStoreConfig {
        url: uri_with_userinfo(&server, "bob", "wrong"),
        index_prefix: "agent_vector".to_string(),
        dimension: 3,
        username: Some("alice".to_string()),
        password: Some("secret".to_string()),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .await
    .expect("the explicit credentials are accepted");
    assert_eq!(search_once(&store).await, 1);
    assert_eq!(
        authorization_headers(&server).await,
        vec![expected.clone(), expected],
        "the explicit fields replace the userinfo pair"
    );
}

#[tokio::test]
async fn vector_api_key_header_comes_from_the_backend() {
    let server = MockServer::start().await;
    let expected = "ApiKey vector-key-123";
    mount_authenticated_search(&server, Some(expected)).await;

    let store = ElasticsearchVectorStore::connect(ElasticsearchVectorStoreConfig {
        url: server.uri(),
        index_prefix: "agent_vector".to_string(),
        dimension: 3,
        api_key: Some("vector-key-123".to_string()),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .await
    .expect("the api key configuration is accepted");
    assert_eq!(search_once(&store).await, 1);
    assert_eq!(
        authorization_headers(&server).await,
        vec![expected.to_string(), expected.to_string()],
        "every vector request carries the API key"
    );
}

#[tokio::test]
async fn vector_store_factory_forwards_the_backend_credentials() {
    let server = MockServer::start().await;
    let expected = basic("alice", "secret");
    mount_authenticated_search(&server, Some(&expected)).await;

    let backend = StoreBackend::from_json(&json!({
        "kind": "elasticsearch",
        "url": uri_with_userinfo(&server, "bob", "wrong"),
        "index_prefix": "agent_vector",
        "username": "alice",
        "password": "secret",
    }))
    .expect("the backend configuration decodes");
    let store = build_vector_store(&backend)
        .await
        .expect("the factory builds a live REST vector store");
    assert_eq!(search_once(&store).await, 1);
    assert_eq!(
        authorization_headers(&server).await,
        vec![expected.clone(), expected],
        "the factory forwards the decoded credentials to the vector store"
    );
}

// ---------------------------------------------------------------------------
// Document store (`ElasticsearchDocumentStore`, wired by `build_document_store`)
//
// One collection is one index named `"{index_prefix}__{collection}"`, so the
// cases below reuse [`INDEX`] (`/agent_vector__docs`) with the `agent_vector`
// prefix that `StoreBackend::Elasticsearch` defaults to.
// ---------------------------------------------------------------------------

/// The `PUT /{index}` body the document store sends to create an index: only
/// the keyword primary key is mapped, every other field is dynamic.
fn document_index_body() -> Value {
    json!({"mappings": {"properties": {"id": {"type": "keyword"}}}})
}

/// A document store on `server`, built without any I/O.
fn document_store(server: &MockServer) -> ElasticsearchDocumentStore {
    ElasticsearchDocumentStore::new(
        ElasticsearchConfig::new(server.uri()).with_index_prefix("agent_vector"),
    )
    .expect("the document store builds without I/O")
}

/// The backend value the document-store factory takes for `server`.
fn document_backend(server: &MockServer) -> StoreBackend {
    StoreBackend::Elasticsearch {
        url: server.uri(),
        index_prefix: "agent_vector".to_string(),
        username: None,
        password: None,
        api_key: None,
    }
}

/// Mounts the `HEAD /{index}` existence probe with `status`.
async fn mount_document_head(server: &MockServer, status: u16) {
    Mock::given(method("HEAD"))
        .and(path(INDEX))
        .respond_with(ResponseTemplate::new(status))
        .expect(1)
        .mount(server)
        .await;
}

/// Mounts the `PUT /{index}` index creation of an absent index.
async fn mount_document_index_creation(server: &MockServer) {
    Mock::given(method("PUT"))
        .and(path(INDEX))
        .and(body_json(document_index_body()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"acknowledged": true})))
        .expect(1)
        .mount(server)
        .await;
}

/// Mounts the `POST /{index}/_bulk` write that accepts every item.
async fn mount_document_bulk(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_bulk")))
        .and(header("content-type", "application/x-ndjson"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "took": 1,
            "errors": false,
            "items": [
                {"index": {"_index": "agent_vector__docs", "_id": "a", "status": 201}},
                {"index": {"_index": "agent_vector__docs", "_id": "b", "status": 201}},
            ],
        })))
        .expect(1)
        .mount(server)
        .await;
}

/// Every request path `server` recorded, in order.
async fn request_paths(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .expect("the mock records requests")
        .iter()
        .map(|request| request.url.path().to_string())
        .collect()
}

/// Every `authorization` header `server` recorded, in order.
async fn authorization_headers(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .expect("the mock records requests")
        .iter()
        .map(|request| {
            request
                .headers
                .get("authorization")
                .map(|value| value.to_str().expect("ASCII header").to_string())
                .unwrap_or_default()
        })
        .collect()
}

/// `Basic <base64(user:pass)>`, the header HTTP basic authentication sends.
fn basic(user: &str, pass: &str) -> String {
    format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("{user}:{pass}"))
    )
}

/// `server.uri()` with `user:pass` userinfo spliced in.
fn uri_with_userinfo(server: &MockServer, user: &str, pass: &str) -> String {
    server
        .uri()
        .replacen("http://", &format!("http://{user}:{pass}@"), 1)
}

#[tokio::test]
async fn document_upsert_creates_the_index_and_bulks_ndjson() {
    let server = MockServer::start().await;
    mount_document_head(&server, 404).await;
    mount_document_index_creation(&server).await;
    mount_document_bulk(&server).await;

    let written = document_store(&server)
        .upsert(
            "docs",
            vec![
                Document::new("a")
                    .with_field("name", "Ada")
                    .with_field("age", 36),
                Document::new("b").with_field("name", "Bob"),
            ],
        )
        .await
        .expect("upsert");
    assert_eq!(written, 2, "upsert reports every document it wrote");

    // One NDJSON request: metadata line, source line, per document.
    let body = request_body(&server, "POST", &format!("{INDEX}/_bulk")).await;
    let body = String::from_utf8(body).expect("the bulk body is UTF-8");
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 4, "two documents are two NDJSON line pairs");
    assert_eq!(
        serde_json::from_str::<Value>(lines[0]).unwrap(),
        json!({"index": {"_id": "a"}})
    );
    assert_eq!(
        serde_json::from_str::<Value>(lines[1]).unwrap(),
        json!({"id": "a", "name": "Ada", "age": 36})
    );
    assert_eq!(
        serde_json::from_str::<Value>(lines[2]).unwrap(),
        json!({"index": {"_id": "b"}})
    );
    assert_eq!(
        serde_json::from_str::<Value>(lines[3]).unwrap(),
        json!({"id": "b", "name": "Bob"})
    );
    assert_eq!(
        request_paths(&server).await,
        vec![
            INDEX.to_string(),
            INDEX.to_string(),
            format!("{INDEX}/_bulk")
        ],
        "a missing index is probed, created, then written"
    );
}

#[tokio::test]
async fn document_upsert_reuses_an_existing_index() {
    let server = MockServer::start().await;
    mount_document_head(&server, 200).await;
    mount_document_bulk(&server).await;

    let written = document_store(&server)
        .upsert("docs", vec![Document::new("a").with_field("name", "Ada")])
        .await
        .expect("upsert");
    assert_eq!(written, 1, "upsert reports the documents it sent");

    // No `PUT` follows the successful probe.
    assert_eq!(
        request_paths(&server).await,
        vec![INDEX.to_string(), format!("{INDEX}/_bulk")]
    );
}

#[tokio::test]
async fn document_upsert_surfaces_item_level_errors() {
    let server = MockServer::start().await;
    mount_document_head(&server, 200).await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_bulk")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": true,
            "items": [{
                "index": {
                    "_id": "a",
                    "status": 409,
                    "error": {
                        "type": "version_conflict_engine_exception",
                        "reason": "version conflict, document already exists",
                    },
                },
            }],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let error = document_store(&server)
        .upsert("docs", vec![Document::new("a")])
        .await
        .expect_err("an item-level failure fails the write");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.message();
    assert!(message.contains("409"), "{message}");
    assert!(message.contains("agent_vector__docs"), "{message}");
    assert!(
        message.contains("version_conflict_engine_exception"),
        "{message}"
    );
    assert!(message.contains("document already exists"), "{message}");
}

#[tokio::test]
async fn document_get_reads_a_hit_and_a_miss() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(format!("{INDEX}/_doc/a")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "_index": "agent_vector__docs",
            "_id": "a",
            "_source": {"id": "a", "name": "Ada", "age": 36},
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("{INDEX}/_doc/missing")))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "_index": "agent_vector__docs",
            "_id": "missing",
            "found": false,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store = document_store(&server);
    assert_eq!(
        store.get("docs", "a").await.expect("get"),
        Some(
            Document::new("a")
                .with_field("name", "Ada")
                .with_field("age", 36)
        )
    );
    assert_eq!(
        store.get("docs", "missing").await.expect("get"),
        None,
        "a missing document reads as absent, not as an error"
    );
}

#[tokio::test]
async fn document_query_sends_the_filter_order_and_paging() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_search")))
        .and(body_json(json!({
            "query": {"range": {"age": {"gte": 18}}},
            "from": 1,
            "size": 2,
            "track_total_hits": true,
            "sort": [{"age": {
                "order": "desc",
                "missing": "_last",
                "unmapped_type": "keyword",
            }}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hits": {
                "total": {"value": 2, "relation": "eq"},
                "hits": [
                    {"_id": "b", "_source": {"id": "b", "name": "Bob", "age": 24}},
                    {"_id": "a", "_source": {"name": "Ada", "age": 20}},
                ],
            },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let query = Query {
        filter: json!({"age": {"$gte": 18}}),
        limit: Some(2),
        offset: 1,
        order_by: vec![("age".to_string(), SortOrder::Desc)],
    };
    let documents = document_store(&server)
        .query("docs", &query)
        .await
        .expect("query");
    assert_eq!(documents.len(), 2);
    assert_eq!(documents[0].id, "b");
    assert_eq!(
        documents[0].fields.get("name"),
        Some(&json!("Bob")),
        "the hit order of the server is kept"
    );
    assert_eq!(
        documents[1].id, "a",
        "a `_source` without `id` is completed from `_id`"
    );
    assert_eq!(documents[1].fields.get("age"), Some(&json!(20)));
}

#[tokio::test]
async fn document_count_sends_the_translated_filter() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_count")))
        .and(body_json(json!({"query": {"term": {"name": "Ada"}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count": 3,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_count")))
        .and(body_json(json!({"query": {"match_all": {}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"count": 7})))
        .expect(1)
        .mount(&server)
        .await;

    let store = document_store(&server);
    let query = Query {
        filter: json!({"name": "Ada"}),
        ..Query::default()
    };
    assert_eq!(store.count("docs", &query).await.expect("count"), 3);
    assert_eq!(
        store.count("docs", &Query::default()).await.expect("count"),
        7,
        "a null filter becomes a `match_all` count"
    );
}

#[tokio::test]
async fn document_delete_uses_delete_by_query() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_delete_by_query")))
        .and(body_json(json!({"query": {"range": {"age": {"lt": 30}}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "deleted": 2,
            "total": 2,
            "failures": [],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let query = Query {
        filter: json!({"age": {"$lt": 30}}),
        ..Query::default()
    };
    assert_eq!(
        document_store(&server)
            .delete("docs", &query)
            .await
            .expect("delete"),
        2
    );
}

#[tokio::test]
async fn document_list_collections_strips_the_prefix() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/agent_vector__*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "agent_vector__notes": {"mappings": {}},
            "agent_vector__docs": {"mappings": {}},
            "unrelated__hidden": {"mappings": {}},
        })))
        .expect(1)
        .mount(&server)
        .await;

    assert_eq!(
        document_store(&server)
            .list_collections()
            .await
            .expect("list"),
        vec!["docs".to_string(), "notes".to_string()],
        "foreign indices are ignored and the names are sorted"
    );
}

#[tokio::test]
async fn document_drop_collection_ignores_a_missing_index() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path(INDEX))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"type": "index_not_found_exception"},
            "status": 404,
        })))
        .expect(1)
        .mount(&server)
        .await;

    document_store(&server)
        .drop_collection("docs")
        .await
        .expect("dropping a missing collection is a no-op");
}

#[tokio::test]
async fn document_missing_index_reads_as_absent() {
    let server = MockServer::start().await;
    for suffix in ["/_search", "/_count", "/_delete_by_query"] {
        Mock::given(method("POST"))
            .and(path(format!("{INDEX}{suffix}")))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                "error": {"type": "index_not_found_exception"},
                "status": 404,
            })))
            .expect(1)
            .mount(&server)
            .await;
    }

    let store = document_store(&server);
    assert_eq!(
        store.query("docs", &Query::default()).await.expect("query"),
        Vec::<Document>::new()
    );
    assert_eq!(
        store.count("docs", &Query::default()).await.expect("count"),
        0
    );
    assert_eq!(
        store
            .delete("docs", &Query::default())
            .await
            .expect("delete"),
        0
    );
}

#[tokio::test]
async fn document_server_failure_is_a_transport_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(format!("{INDEX}/_doc/a")))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom: shard unavailable"))
        .expect(1)
        .mount(&server)
        .await;

    let error = document_store(&server)
        .get("docs", "a")
        .await
        .expect_err("500 must fail");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.message();
    assert!(message.contains("500"), "{message}");
    assert!(message.contains("agent_vector__docs"), "{message}");
    assert!(message.contains("shard unavailable"), "{message}");
}

#[tokio::test]
async fn document_store_factory_round_trips_a_document() {
    let server = MockServer::start().await;
    mount_document_head(&server, 404).await;
    mount_document_index_creation(&server).await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_bulk")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [
                {"index": {"_index": "agent_vector__docs", "_id": "a", "status": 201}},
                {"index": {"_index": "agent_vector__docs", "_id": "b", "status": 201}},
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("{INDEX}/_doc/a")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "_index": "agent_vector__docs",
            "_id": "a",
            "_source": {"id": "a", "name": "Ada"},
        })))
        .expect(1)
        .mount(&server)
        .await;

    // The factory is the wiring under test: it must reach the kernel document
    // store with the backend's index prefix and endpoint.
    let store = build_document_store(&document_backend(&server))
        .await
        .expect("the factory builds a live REST document store");
    assert_eq!(
        store
            .upsert("docs", vec![Document::new("a").with_field("name", "Ada")])
            .await
            .expect("upsert"),
        1
    );
    assert_eq!(
        store.get("docs", "a").await.expect("get"),
        Some(Document::new("a").with_field("name", "Ada"))
    );
    for requested in request_paths(&server).await {
        assert!(
            requested.starts_with("/agent_vector__docs"),
            "the backend prefix reaches the wire: {requested}"
        );
    }
}

#[tokio::test]
async fn document_basic_auth_comes_from_the_url_userinfo() {
    let server = MockServer::start().await;
    let expected = basic("elastic", "s3cret");
    Mock::given(method("HEAD"))
        .and(path(INDEX))
        .and(header("authorization", expected.as_str()))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_bulk")))
        .and(header("authorization", expected.as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [{"index": {"_id": "a", "status": 201}}],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let backend = StoreBackend::Elasticsearch {
        url: uri_with_userinfo(&server, "elastic", "s3cret"),
        index_prefix: "agent_vector".to_string(),
        username: None,
        password: None,
        api_key: None,
    };
    let store = build_document_store(&backend)
        .await
        .expect("the userinfo endpoint is accepted");
    store
        .upsert("docs", vec![Document::new("a")])
        .await
        .expect("upsert");
    assert_eq!(
        authorization_headers(&server).await,
        vec![expected.clone(), expected],
        "every request carries the userinfo credentials"
    );
}

#[tokio::test]
async fn document_explicit_credentials_win_over_the_url_userinfo() {
    let server = MockServer::start().await;
    let expected = basic("alice", "secret");
    Mock::given(method("HEAD"))
        .and(path(INDEX))
        .and(header("authorization", expected.as_str()))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_bulk")))
        .and(header("authorization", expected.as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [{"index": {"_id": "a", "status": 201}}],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let backend = StoreBackend::Elasticsearch {
        url: uri_with_userinfo(&server, "bob", "wrong"),
        index_prefix: "agent_vector".to_string(),
        username: Some("alice".to_string()),
        password: Some("secret".to_string()),
        api_key: None,
    };
    let store = build_document_store(&backend)
        .await
        .expect("the explicit credentials are accepted");
    store
        .upsert("docs", vec![Document::new("a")])
        .await
        .expect("upsert");
    assert_eq!(
        authorization_headers(&server).await,
        vec![expected.clone(), expected],
        "the explicit fields replace the userinfo pair"
    );
}

#[tokio::test]
async fn document_api_key_header_comes_from_the_backend() {
    let server = MockServer::start().await;
    let expected = "ApiKey api-key-123";
    Mock::given(method("HEAD"))
        .and(path(INDEX))
        .and(header("authorization", expected))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{INDEX}/_bulk")))
        .and(header("authorization", expected))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [{"index": {"_id": "a", "status": 201}}],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let backend = StoreBackend::Elasticsearch {
        url: server.uri(),
        index_prefix: "agent_vector".to_string(),
        username: None,
        password: None,
        api_key: Some("api-key-123".to_string()),
    };
    let store = build_document_store(&backend)
        .await
        .expect("the api key backend is accepted");
    store
        .upsert("docs", vec![Document::new("a")])
        .await
        .expect("upsert");
    assert_eq!(
        authorization_headers(&server).await,
        vec![expected.to_string(), expected.to_string()]
    );
}

/// A collection name is one URL path segment, so a name that could address a
/// different index (`x/../docs` normalises to `DELETE /docs`) is a validation
/// error on every entry point, before any request is built.
#[tokio::test]
async fn hostile_collection_names_never_reach_the_endpoint() {
    let server = MockServer::start().await;
    let store = connect(&server).await;
    for name in [
        "",
        "x/../docs",
        "/docs",
        "docs/",
        "a\\b",
        "..",
        "a..b",
        "nul\0byte",
        "docs?pretty",
        "docs#frag",
    ] {
        let outcomes = [
            store.create_collection(name, schema()).await.err(),
            store.delete_collection(name).await.err(),
            store.collection_exists(name).await.err(),
            store.get_schema(name).await.err(),
            store
                .search(name, &VectorQuery::new(vec![1.0, 0.0, 0.0]))
                .await
                .err(),
            store
                .add_docs(
                    name,
                    vec![
                        json!({"id": "a", "embedding": [1.0, 0.0, 0.0], "title": "A"})
                            .as_object()
                            .cloned()
                            .unwrap(),
                    ],
                )
                .await
                .err(),
        ];
        for error in outcomes {
            let error = error.unwrap_or_else(|| panic!("collection {name:?} was accepted"));
            assert_eq!(error.kind(), ErrorKind::Validation, "name {name:?}");
            assert!(
                error.message().contains("invalid vector collection name"),
                "name {name:?}: {error}"
            );
        }
    }
    assert!(
        server
            .received_requests()
            .await
            .expect("the mock records requests")
            .is_empty(),
        "a rejected collection name must not send anything"
    );
}

/// A benign name (underscores, dots, non-ASCII) stays one encoded path segment.
#[tokio::test]
async fn a_benign_collection_name_is_encoded_into_one_path_segment() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"acknowledged": true})))
        .mount(&server)
        .await;
    let store = connect(&server).await;
    store.create_collection("docs_v2", schema()).await.unwrap();
    store.create_collection("notes_ü", schema()).await.unwrap();

    let paths: Vec<String> = server
        .received_requests()
        .await
        .expect("the mock records requests")
        .iter()
        .map(|request| request.url.path().to_string())
        .collect();
    assert_eq!(
        paths,
        vec![
            "/agent_vector__docs_v2".to_string(),
            // `ü` is `%C3%BC`; the underscore and the dot stay literal.
            "/agent_vector__notes_%C3%BC".to_string(),
        ]
    );
}

/// A node that accepts the connection and never answers must not pin the task:
/// the configured request timeout maps onto `Error::Timeout`.
#[tokio::test]
async fn a_black_holed_node_yields_a_typed_timeout() {
    let bound = std::time::Duration::from_secs(10);
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path(INDEX))
        .respond_with(ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(30)))
        .mount(&server)
        .await;
    let store = ElasticsearchVectorStore::connect(ElasticsearchVectorStoreConfig {
        url: server.uri(),
        index_prefix: "agent_vector".to_string(),
        dimension: 3,
        timeout: std::time::Duration::from_millis(100),
        connect_timeout: std::time::Duration::from_secs(1),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .await
    .expect("connecting performs no I/O");

    let started = std::time::Instant::now();
    let error = tokio::time::timeout(bound, store.collection_exists("docs"))
        .await
        .expect("the request hung past the test bound")
        .expect_err("a stalled node must not answer");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.message().contains("timed out"), "message {error}");
    assert!(
        started.elapsed() < bound,
        "the configured request timeout was not applied"
    );
}
