//! Offline `wiremock` tests for the Elasticsearch document backend.
//!
//! Every test talks to a loopback mock server, so the suite needs no
//! Elasticsearch installation, no network, and no credentials. Each test mounts
//! exactly the requests the call is allowed to make and asserts their paths,
//! methods and bodies; an unexpected request is answered with 404 and therefore
//! fails the call under test.

use super::*;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use wiremock::matchers::{body_json, body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Builds a store pointing at `server` with the default index prefix.
fn store(server: &MockServer) -> ElasticsearchDocumentStore {
    ElasticsearchDocumentStore::new(ElasticsearchConfig::new(server.uri())).expect("build store")
}

/// Two documents of the `docs` collection.
fn documents() -> Vec<Document> {
    vec![
        Document::new("a")
            .with_field("name", "Ada")
            .with_field("age", 36),
        Document::new("b")
            .with_field("name", "Bob")
            .with_field("age", 24),
    ]
}

/// The `_source` view of [`documents`]`[0]`.
fn ada_source() -> Value {
    json!({"id": "a", "name": "Ada", "age": 36})
}

/// Mounts the `HEAD /{index}` existence probe with `status`.
async fn mount_head(server: &MockServer, index: &str, status: u16) {
    Mock::given(method("HEAD"))
        .and(path(format!("/{index}")))
        .respond_with(ResponseTemplate::new(status))
        .mount(server)
        .await;
}

#[tokio::test]
async fn create_collection_puts_the_keyword_mapping() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/openjiuwen__docs"))
        .and(header("content-type", "application/json"))
        .and(body_json(index_body()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"acknowledged": true})))
        .expect(1)
        .mount(&server)
        .await;

    store(&server)
        .create_collection("docs")
        .await
        .expect("create");
}

#[tokio::test]
async fn create_collection_tolerates_an_existing_index() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "type": "resource_already_exists_exception",
                "reason": "index [openjiuwen__docs/abc123] already exists",
            },
            "status": 400,
        })))
        .expect(1)
        .mount(&server)
        .await;

    store(&server)
        .create_collection("docs")
        .await
        .expect("creating an existing collection is a no-op");
}

#[tokio::test]
async fn create_collection_surfaces_other_rejections() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/openjiuwen__docs"))
        .respond_with(
            ResponseTemplate::new(400).set_body_string("mapper_parsing_exception: bad mapping"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let error = store(&server)
        .create_collection("docs")
        .await
        .expect_err("400 must fail");
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.to_string().contains("mapper_parsing_exception"));
}

#[tokio::test]
async fn invalid_collection_names_are_rejected_before_any_request() {
    let server = MockServer::start().await;
    let error = store(&server)
        .create_collection("a/b")
        .await
        .expect_err("invalid collection");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(server
        .received_requests()
        .await
        .expect("requests")
        .is_empty());
}

#[tokio::test]
async fn drop_collection_ignores_a_missing_index() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/openjiuwen__docs"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"type": "index_not_found_exception"},
            "status": 404,
        })))
        .expect(1)
        .mount(&server)
        .await;

    store(&server)
        .drop_collection("docs")
        .await
        .expect("dropping a missing collection is a no-op");
}

#[tokio::test]
async fn list_collections_strips_the_prefix_and_sorts() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "openjiuwen__zeta": {"mappings": {}},
            "openjiuwen__alpha": {"mappings": {}},
            "openjiuwen__": {"mappings": {}},
            "openjiuwen__alpha-beta": {"mappings": {}},
            "other__hidden": {"mappings": {}},
        })))
        .expect(1)
        .mount(&server)
        .await;

    assert_eq!(
        store(&server).list_collections().await.expect("list"),
        vec![
            "alpha".to_string(),
            "alpha-beta".to_string(),
            "zeta".to_string()
        ]
    );
}

#[tokio::test]
async fn list_collections_reads_a_missing_prefix_as_empty() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__*"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;

    assert_eq!(
        store(&server).list_collections().await.expect("list"),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn a_custom_prefix_moves_every_path() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/acme__docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"acknowledged": true})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/acme__*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "acme__docs": {"mappings": {}},
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store = ElasticsearchDocumentStore::new(
        ElasticsearchConfig::new(server.uri()).with_index_prefix("acme"),
    )
    .expect("build store");
    store.create_collection("docs").await.expect("create");
    assert_eq!(
        store.list_collections().await.expect("list"),
        vec!["docs".to_string()]
    );
}

#[tokio::test]
async fn upsert_creates_the_index_and_bulks_the_flat_documents() {
    let server = MockServer::start().await;
    mount_head(&server, "openjiuwen__docs", 404).await;
    Mock::given(method("PUT"))
        .and(path("/openjiuwen__docs"))
        .and(body_json(index_body()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"acknowledged": true})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .and(header("content-type", "application/x-ndjson"))
        .and(body_string_contains(r#"{"index":{"_id":"a"}}"#))
        .and(body_string_contains(r#"{"index":{"_id":"b"}}"#))
        .and(body_string_contains(r#""name":"Ada""#))
        .and(body_string_contains(r#""age":24"#))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [
                {"index": {"_id": "a", "status": 201, "result": "created"}},
                {"index": {"_id": "b", "status": 201, "result": "created"}},
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let written = store(&server)
        .upsert("docs", documents())
        .await
        .expect("upsert");
    assert_eq!(written, 2);
}

#[tokio::test]
async fn upsert_reuses_an_existing_index() {
    let server = MockServer::start().await;
    mount_head(&server, "openjiuwen__docs", 200).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .and(body_string_contains(r#""id":"a""#))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [{"index": {"_id": "a", "status": 200, "result": "updated"}}],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let written = store(&server)
        .upsert("docs", vec![Document::new("a").with_field("name", "Ada")])
        .await
        .expect("upsert");
    assert_eq!(written, 1);

    // No `PUT` was sent: an unmatched `PUT` would have failed the call.
    let methods: Vec<String> = server
        .received_requests()
        .await
        .expect("requests")
        .iter()
        .map(|request| request.method.as_str().to_string())
        .collect();
    assert_eq!(methods, vec!["HEAD".to_string(), "POST".to_string()]);
}

#[tokio::test]
async fn upsert_without_documents_sends_nothing() {
    let server = MockServer::start().await;
    assert_eq!(
        store(&server)
            .upsert("docs", Vec::new())
            .await
            .expect("upsert"),
        0
    );
    assert!(server
        .received_requests()
        .await
        .expect("requests")
        .is_empty());
}

#[tokio::test]
async fn upsert_surfaces_item_errors() {
    let server = MockServer::start().await;
    mount_head(&server, "openjiuwen__docs", 200).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
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

    let error = store(&server)
        .upsert("docs", vec![Document::new("a")])
        .await
        .expect_err("item error");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(message.contains("409"), "message: {message}");
    assert!(message.contains("openjiuwen__docs"), "message: {message}");
    assert!(
        message.contains("version_conflict_engine_exception"),
        "message: {message}"
    );
    assert!(
        message.contains("document already exists"),
        "message: {message}"
    );
}

#[tokio::test]
async fn get_parses_the_source_and_injects_a_missing_id() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs/_doc/a"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "_index": "openjiuwen__docs",
            "_id": "a",
            "_source": ada_source(),
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs/_doc/b"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "_index": "openjiuwen__docs",
            "_id": "b",
            "_source": {"name": "Bob", "age": 24},
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store = store(&server);
    assert_eq!(
        store.get("docs", "a").await.expect("get"),
        Some(
            Document::new("a")
                .with_field("name", "Ada")
                .with_field("age", 36)
        )
    );
    // `_source` without `id` is completed from `_id`.
    assert_eq!(
        store.get("docs", "b").await.expect("get"),
        Some(
            Document::new("b")
                .with_field("name", "Bob")
                .with_field("age", 24)
        )
    );
}

#[tokio::test]
async fn get_reads_a_missing_document_as_none() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs/_doc/missing"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "_index": "openjiuwen__docs",
            "_id": "missing",
            "found": false,
        })))
        .expect(1)
        .mount(&server)
        .await;

    assert_eq!(
        store(&server).get("docs", "missing").await.expect("get"),
        None
    );
}

#[tokio::test]
async fn query_sends_the_translated_filter_order_and_paging() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_search"))
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
    let documents = store(&server).query("docs", &query).await.expect("query");
    // Hits keep the server's order; a `_source` without `id` is completed.
    assert_eq!(documents.len(), 2);
    assert_eq!(documents[0].id, "b");
    assert_eq!(documents[1].id, "a");
    assert_eq!(documents[1].fields.get("age"), Some(&json!(20)));
}

#[tokio::test]
async fn query_sorts_ascending_with_missing_values_first() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_search"))
        .and(body_json(json!({
            "query": {"match_all": {}},
            "from": 0,
            "size": 10_000,
            "track_total_hits": true,
            "sort": [
                {"age": {"order": "asc", "missing": "_first", "unmapped_type": "keyword"}},
                {"id": {"order": "desc", "missing": "_last", "unmapped_type": "keyword"}},
            ],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"hits": {"hits": []}})))
        .expect(1)
        .mount(&server)
        .await;

    let query = Query {
        order_by: vec![
            ("age".to_string(), SortOrder::Asc),
            ("id".to_string(), SortOrder::Desc),
        ],
        ..Query::default()
    };
    assert_eq!(
        store(&server).query("docs", &query).await.expect("query"),
        Vec::<Document>::new()
    );
}

#[tokio::test]
async fn query_reads_a_missing_index_as_empty() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_search"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"type": "index_not_found_exception"},
            "status": 404,
        })))
        .expect(1)
        .mount(&server)
        .await;

    assert_eq!(
        store(&server)
            .query("docs", &Query::default())
            .await
            .expect("query"),
        Vec::<Document>::new()
    );
}

/// Mounts the fallback scan of `POST /openjiuwen__docs/_search` used by the
/// local re-check, returning three documents with nested metadata.
async fn mount_scan(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_search"))
        .and(body_json(json!({
            "query": {"match_all": {}},
            "_source": true,
            "from": 0,
            "size": 500,
            "sort": ["_doc"],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hits": {
                "hits": [
                    {
                        "_id": "a",
                        "_source": {"id": "a", "meta": {"owner": "ada"}, "age": 30},
                    },
                    {
                        "_id": "b",
                        "_source": {"id": "b", "meta": {"owner": "bob"}, "age": 20},
                    },
                    {
                        "_id": "c",
                        "_source": {"id": "c", "meta": {"owner": "ada"}, "age": 40},
                    },
                ],
            },
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn query_falls_back_to_a_bounded_scan_for_untranslatable_filters() {
    let server = MockServer::start().await;
    mount_scan(&server).await;
    let query = Query {
        filter: json!({"meta.owner": "ada"}),
        limit: Some(1),
        offset: 1,
        order_by: vec![("age".to_string(), SortOrder::Desc)],
    };
    let documents = store(&server).query("docs", &query).await.expect("query");
    // Matches `a` (30) and `c` (40); descending gives `c`, `a`; offset 1 keeps `a`.
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].id, "a");
}

#[tokio::test]
async fn count_sends_the_translated_filter() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_count"))
        .and(body_json(json!({"query": {"term": {"name": "Ada"}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count": 3,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        })))
        .expect(1)
        .mount(&server)
        .await;

    let query = Query {
        filter: json!({"name": "Ada"}),
        ..Query::default()
    };
    assert_eq!(
        store(&server).count("docs", &query).await.expect("count"),
        3
    );
}

#[tokio::test]
async fn count_reads_a_missing_index_as_zero() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_count"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;

    assert_eq!(
        store(&server)
            .count("docs", &Query::default())
            .await
            .expect("count"),
        0
    );
}

#[tokio::test]
async fn count_falls_back_to_the_bounded_scan() {
    let server = MockServer::start().await;
    mount_scan(&server).await;
    let query = Query {
        filter: json!({"meta.owner": "ada"}),
        ..Query::default()
    };
    assert_eq!(
        store(&server).count("docs", &query).await.expect("count"),
        2
    );
}

#[tokio::test]
async fn delete_sends_a_delete_by_query_with_the_translated_filter() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_delete_by_query"))
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
        store(&server).delete("docs", &query).await.expect("delete"),
        2
    );
}

#[tokio::test]
async fn delete_reads_a_missing_index_as_zero() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_delete_by_query"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;

    assert_eq!(
        store(&server)
            .delete("docs", &Query::default())
            .await
            .expect("delete"),
        0
    );
}

#[tokio::test]
async fn delete_falls_back_to_a_scan_and_a_bulk_delete() {
    let server = MockServer::start().await;
    mount_scan(&server).await;
    Mock::given(method("POST"))
        .and(path("/openjiuwen__docs/_bulk"))
        .and(header("content-type", "application/x-ndjson"))
        .and(body_string_contains(r#"{"delete":{"_id":"a"}}"#))
        .and(body_string_contains(r#"{"delete":{"_id":"c"}}"#))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "errors": false,
            "items": [
                {"delete": {"_id": "a", "status": 200, "result": "deleted"}},
                {"delete": {"_id": "c", "status": 404, "result": "not_found"}},
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let query = Query {
        filter: json!({"meta.owner": "ada"}),
        ..Query::default()
    };
    // Only the items Elasticsearch reports as `deleted` are counted.
    assert_eq!(
        store(&server).delete("docs", &query).await.expect("delete"),
        1
    );
}

#[tokio::test]
async fn service_failures_name_the_status_the_index_and_the_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/openjiuwen__docs/_doc/a"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom: shard unavailable"))
        .expect(1)
        .mount(&server)
        .await;

    let error = store(&server)
        .get("docs", "a")
        .await
        .expect_err("500 must fail");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(message.contains("500"), "message: {message}");
    assert!(message.contains("openjiuwen__docs"), "message: {message}");
    assert!(message.contains("shard unavailable"), "message: {message}");
}

#[test]
fn index_body_declares_the_keyword_primary_key() {
    assert_eq!(
        index_body(),
        json!({"mappings": {"properties": {"id": {"type": "keyword"}}}})
    );
}

#[test]
fn already_exists_recognises_the_error_type_and_the_phrase() {
    assert!(is_already_exists(
        r#"{"error":{"type":"resource_already_exists_exception"}}"#
    ));
    assert!(is_already_exists("Index Already Exists"));
    assert!(!is_already_exists("mapper_parsing_exception"));
}
