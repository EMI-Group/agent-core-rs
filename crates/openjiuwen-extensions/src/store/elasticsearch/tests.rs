//! Unit tests for the Elasticsearch vector backend: configuration, index
//! naming, the mapping builder and its schema round-trip, the filter
//! translation vectors, score normalization, and result ordering.
//!
//! Python counterpart: `tests/unit_tests/extensions/store/test_es_vector_store.py`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::store::{
    CollectionSchema, FieldSchema, Metric, VectorDataType, VectorQuery, VectorSearchResult,
};
use serde_json::{json, Map, Value};

use super::ElasticsearchVectorStoreConfig;
use super::{client, order_results, resolve_vector_field, ElasticsearchVectorStore};

/// A schema with a keyword primary key and one 3-dimensional vector field.
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
    ])
}

/// A full schema covering every Elasticsearch property mapping.
fn wide_schema() -> CollectionSchema {
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
        FieldSchema::new("year", VectorDataType::Int64),
        FieldSchema::new("rank", VectorDataType::Int32),
        FieldSchema::new("ratio", VectorDataType::Float),
        FieldSchema::new("precise", VectorDataType::Double),
        FieldSchema::new("flag", VectorDataType::Bool),
        FieldSchema::new("meta", VectorDataType::Json),
        FieldSchema::new("items", VectorDataType::Array),
    ])
}

/// A store configured for `http://localhost:9200` and the default prefix.
fn store() -> ElasticsearchVectorStore {
    ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
        url: "http://localhost:9200".to_string(),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .expect("the configuration is valid")
}

/// One result with `score` and primary key `id`.
fn result(score: f32, id: &str) -> VectorSearchResult {
    let mut fields = Map::new();
    fields.insert("id".to_string(), Value::String(id.to_string()));
    VectorSearchResult::new(score, fields)
}

#[test]
fn default_config_matches_python_defaults() {
    let config = ElasticsearchVectorStoreConfig::default();
    assert_eq!(config.url, "");
    assert_eq!(config.index_prefix, "agent_vector");
    assert_eq!(config.dimension, 768);
    assert_eq!(config.username, None);
    assert_eq!(config.password, None);
    assert_eq!(config.api_key, None);
}

/// The vector store and the document store share one credential-resolution
/// rule set, so this pins the precedence the shared helper implements.
#[test]
fn credentials_resolve_the_documented_precedence() {
    use super::credentials::ElasticsearchCredentials;
    use url::Url;

    // Userinfo only: both halves come from the URL.
    let userinfo = Url::parse("http://alice:wonderland@127.0.0.1:9200").unwrap();
    let resolved = ElasticsearchCredentials::resolve(&userinfo, None, None, None);
    assert_eq!(resolved.username.as_deref(), Some("alice"));
    assert_eq!(resolved.password.as_deref(), Some("wonderland"));
    assert_eq!(resolved.basic_auth(), Some(("alice", "wonderland")));
    assert_eq!(resolved.api_key, None);

    // Explicit fields win over the userinfo field by field: the explicit user
    // name keeps the userinfo password.
    let resolved =
        ElasticsearchCredentials::resolve(&userinfo, Some("bob"), None, Some("api-key-123"));
    assert_eq!(resolved.username.as_deref(), Some("bob"));
    assert_eq!(resolved.password.as_deref(), Some("wonderland"));
    assert_eq!(resolved.basic_auth(), Some(("bob", "wonderland")));
    assert_eq!(resolved.api_key.as_deref(), Some("api-key-123"));

    // A user name without a password sends an empty password.
    let bare = Url::parse("http://alice@127.0.0.1:9200").unwrap();
    let resolved = ElasticsearchCredentials::resolve(&bare, None, None, None);
    assert_eq!(resolved.username.as_deref(), Some("alice"));
    assert_eq!(resolved.password.as_deref(), Some(""));
    assert_eq!(resolved.basic_auth(), Some(("alice", "")));

    // A password without a user name is kept but never sent, so the API key
    // applies instead.
    let plain = Url::parse("http://127.0.0.1:9200").unwrap();
    let resolved = ElasticsearchCredentials::resolve(&plain, None, Some("orphan"), Some("api-key"));
    assert_eq!(resolved.username, None);
    assert_eq!(resolved.password.as_deref(), Some("orphan"));
    assert_eq!(resolved.basic_auth(), None);
    assert_eq!(resolved.api_key.as_deref(), Some("api-key"));
}

#[test]
fn valid_http_url_builds_a_store() {
    let store = store();
    assert_eq!(store.config().url, "http://localhost:9200");
    assert_eq!(store.index_name("docs"), "agent_vector__docs");
}

#[test]
fn trailing_slashes_are_trimmed_from_the_endpoint() {
    let store = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
        url: "https://localhost:9200/es/".to_string(),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .expect("the configuration is valid");
    assert_eq!(store.config().url, "https://localhost:9200/es");
}

#[test]
fn invalid_configs_are_rejected() {
    let empty =
        ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig::default()).unwrap_err();
    assert_eq!(empty.kind(), ErrorKind::Validation);
    assert!(empty.message().contains("url"));

    let wrong_scheme = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
        url: "redis://localhost:6379".to_string(),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .unwrap_err();
    assert_eq!(wrong_scheme.kind(), ErrorKind::Validation);

    let unparseable = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
        url: "http://exa mple:9200".to_string(),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .unwrap_err();
    assert_eq!(unparseable.kind(), ErrorKind::Validation);
    assert!(unparseable.message().contains("url"));

    let hostless = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
        url: "http://".to_string(),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .unwrap_err();
    assert_eq!(hostless.kind(), ErrorKind::Validation);

    let zero_dimension = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
        url: "http://localhost:9200".to_string(),
        dimension: 0,
        ..ElasticsearchVectorStoreConfig::default()
    })
    .unwrap_err();
    assert_eq!(zero_dimension.kind(), ErrorKind::Validation);
    assert!(zero_dimension.message().contains("dimension"));
}

#[test]
fn index_name_uses_the_prefix_separator() {
    let store = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
        url: "https://localhost:9200".to_string(),
        index_prefix: "custom".to_string(),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .unwrap();
    assert_eq!(store.index_name("docs"), "custom__docs");
}

#[tokio::test]
async fn connect_returns_a_live_store_without_io() {
    // `connect` only validates the configuration; a port nothing listens on is
    // still fine, so no network is touched here.
    ElasticsearchVectorStore::connect(ElasticsearchVectorStoreConfig {
        url: "http://127.0.0.1:9".to_string(),
        ..ElasticsearchVectorStoreConfig::default()
    })
    .await
    .expect("connecting performs no I/O");

    let error = ElasticsearchVectorStore::connect(ElasticsearchVectorStoreConfig::default())
        .await
        .err()
        .expect("an empty url is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn mapping_maps_every_kernel_type() {
    let body = client::mapping_body(&wide_schema(), 768).expect("the mapping builds");
    let mappings = &body["mappings"];
    assert_eq!(mappings["dynamic"], json!("strict"));
    assert_eq!(
        mappings["properties"],
        json!({
            "id": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 3, "index": true, "similarity": "cosine"},
            "title": {"type": "keyword"},
            "year": {"type": "long"},
            "rank": {"type": "integer"},
            "ratio": {"type": "float"},
            "precise": {"type": "double"},
            "flag": {"type": "boolean"},
            "meta": {"type": "object", "enabled": true},
            "items": {"type": "object", "enabled": true},
        })
    );
    assert_eq!(
        mappings["_meta"]["openjiuwen_schema"],
        serde_json::to_value(wide_schema()).unwrap(),
        "the full schema is persisted for get_schema"
    );
}

#[test]
fn mapping_honours_dynamic_fields_and_the_default_dimension() {
    let mut schema = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema::new("embedding", VectorDataType::FloatVector),
    ]);
    schema.enable_dynamic_field = true;
    let body = client::mapping_body(&schema, 64).expect("the mapping builds");
    assert_eq!(body["mappings"]["dynamic"], json!("true"));
    assert_eq!(
        body["mappings"]["properties"]["embedding"]["dims"],
        json!(64),
        "a vector field without dim falls back to the store dimension"
    );
}

#[test]
fn mapping_round_trips_through_schema_from_mappings() {
    let body = client::mapping_body(&wide_schema(), 768).expect("the mapping builds");
    let restored =
        client::schema_from_mappings(&body["mappings"]).expect("the schema is read back");
    assert_eq!(restored, wide_schema());
}

#[test]
fn schema_from_mappings_rebuilds_an_foreign_index() {
    let mappings = json!({
        "dynamic": "true",
        "properties": {
            "id": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 2},
            "count": {"type": "long"},
            "unknown": {"type": "geo_point"},
        },
    });
    let schema = client::schema_from_mappings(&mappings).expect("the properties describe a schema");
    assert!(schema.enable_dynamic_field);
    assert_eq!(
        schema.primary_key().map(|field| field.name.as_str()),
        Some("id")
    );
    assert_eq!(
        schema.default_vector_field().and_then(|field| field.dim),
        Some(2)
    );
    assert_eq!(schema.fields.len(), 3, "the geo_point property is ignored");
    assert!(schema.fields.iter().any(|field| field.name == "count"));
}

#[test]
fn similarity_maps_every_metric() {
    // The store itself always uses cosine: `CollectionSchema` has no metric field.
    assert_eq!(client::similarity_for(Metric::Cosine), "cosine");
    assert_eq!(client::similarity_for(Metric::Dot), "dot_product");
    assert_eq!(client::similarity_for(Metric::Euclidean), "l2_norm");
}

#[test]
fn filters_translate_to_the_elasticsearch_query_dsl() {
    let cases = [
        (json!(null), json!({"match_all": {}})),
        (json!({}), json!({"match_all": {}})),
        (json!("scalar"), json!({"match_none": {}})),
        (json!({"name": "Ada"}), json!({"term": {"name": "Ada"}})),
        (json!({"year": 2020}), json!({"term": {"year": 2020}})),
        (json!({"flag": true}), json!({"term": {"flag": true}})),
        (
            json!({"tags": ["rust", "storage"]}),
            json!({"terms": {"tags": ["rust", "storage"]}}),
        ),
        (
            json!({"age": {"$gte": 18, "$lt": 65}}),
            json!({"bool": {"must": [
                {"range": {"age": {"gte": 18}}},
                {"range": {"age": {"lt": 65}}},
            ]}}),
        ),
        (
            json!({"name": {"$gt": "Ada"}}),
            json!({"range": {"name": {"gt": "Ada"}}}),
        ),
        (
            json!({"$and": [{"a": 1}, {"b": "x"}]}),
            json!({"bool": {"must": [{"term": {"a": 1}}, {"term": {"b": "x"}}]}}),
        ),
        (
            json!({"$or": [{"a": 1}, {"b": "x"}]}),
            json!({"bool": {
                "should": [{"term": {"a": 1}}, {"term": {"b": "x"}}],
                "minimum_should_match": 1,
            }}),
        ),
        (
            json!({"$not": {"a": 1}}),
            json!({"bool": {"must": [{"match_all": {}}], "must_not": [{"term": {"a": 1}}]}}),
        ),
        (json!({"$and": []}), json!({"match_all": {}})),
        (json!({"$or": []}), json!({"match_none": {}})),
        (json!({"$and": "not-an-array"}), json!({"match_none": {}})),
        (json!({"$or": "not-an-array"}), json!({"match_none": {}})),
        (json!({"$not": "not-an-object"}), json!({"match_none": {}})),
        (
            json!({"a": {"$in": ["x", "y"]}}),
            json!({"terms": {"a": ["x", "y"]}}),
        ),
        (json!({"a": {"$in": []}}), json!({"match_none": {}})),
        (json!({"a": {"$in": "x"}}), json!({"match_none": {}})),
        (
            json!({"a": {"$ne": 1}}),
            json!({"bool": {"must": [{"match_all": {}}], "must_not": [{"term": {"a": 1}}]}}),
        ),
        (json!({"a": {"$eq": "x"}}), json!({"term": {"a": "x"}})),
        (json!({"a": {"$gt": null}}), json!({"match_none": {}})),
        (json!({"a": {"$bogus": 1}}), json!({"match_none": {}})),
        (
            json!({"a": {"$gt": 1, "unexpected": 1}}),
            json!({"match_none": {}}),
        ),
        (
            json!({"$and": [{"a": 1}], "b": 2}),
            json!({"bool": {"must": [{"term": {"a": 1}}, {"term": {"b": 2}}]}}),
        ),
    ];
    for (filter, expected) in cases {
        assert_eq!(
            client::translate_filter(&filter).unwrap(),
            expected,
            "filter: {filter}"
        );
    }
}

#[test]
fn unsupported_filters_are_precise_validation_errors() {
    for filter in [
        json!({"a": {"$contains": "x"}}),
        json!({"a": {"$contains": ["x"]}}),
        json!({"a": {"owner": "ada"}}),
        json!({"a": null}),
        json!({"a": {"$in": [{"nested": true}]}}),
        json!({"a": [null]}),
    ] {
        let error = client::translate_filter(&filter).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "filter: {filter}");
        assert!(
            error.message().contains("in-memory vector store"),
            "the error must point at the supported backend: {}",
            error.message()
        );
        assert!(
            error.message().contains("field \"a\""),
            "the error must name the field path: {}",
            error.message()
        );
    }
    let contains = client::translate_filter(&json!({"a": {"$contains": "x"}})).unwrap_err();
    assert!(
        contains.message().contains("$contains"),
        "the error must name the operator: {}",
        contains.message()
    );
}

#[test]
fn scores_are_clamped_into_the_unit_range() {
    assert_eq!(json!(super::normalize_score(0.25)), json!(0.25));
    assert_eq!(json!(super::normalize_score(2.0)), json!(1.0));
    assert_eq!(json!(super::normalize_score(-0.5)), json!(0.0));
    assert_eq!(json!(super::normalize_score(f64::NAN)), json!(0.0));
}

#[test]
fn results_are_ordered_and_truncated() {
    let mut results = vec![result(0.5, "c"), result(0.9, "b"), result(0.9, "a")];
    order_results(&mut results, Some("id"), 2);
    assert_eq!(results.len(), 2, "top_k truncates");
    assert_eq!(results[0].fields["id"], json!("a"));
    assert_eq!(results[1].fields["id"], json!("b"));

    let mut without_primary = vec![result(0.4, "z"), result(0.4, "y")];
    order_results(&mut without_primary, Some("id"), 5);
    assert_eq!(without_primary[0].fields["id"], json!("y"));
}

#[test]
fn vector_field_resolution_matches_the_memory_backend() {
    let schema = schema();
    let default = resolve_vector_field(&schema, "docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .expect("the first vector field is the default");
    assert_eq!(default.name, "embedding");

    let named = resolve_vector_field(
        &schema,
        "docs",
        &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_vector_field("embedding"),
    )
    .expect("the named vector field exists");
    assert_eq!(named.name, "embedding");

    let unknown = resolve_vector_field(
        &schema,
        "docs",
        &VectorQuery::new(vec![1.0]).with_vector_field("nope"),
    )
    .unwrap_err();
    assert_eq!(unknown.kind(), ErrorKind::Validation);
    assert!(unknown.message().contains("nope"));
}

#[test]
fn empty_filters_are_not_sent_to_elasticsearch() {
    assert!(super::filter_is_empty(&Value::Null));
    assert!(super::filter_is_empty(&json!({})));
    assert!(!super::filter_is_empty(&json!({"a": 1})));
    assert!(!super::filter_is_empty(&json!({"$and": []})));
}
