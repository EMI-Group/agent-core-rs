// The Python `extensions/store/` package has no SQLite vector backend, so these
// tests pin the Rust-side contract: reference parity against the kernel's
// `InMemoryVectorStore`, the brute-force blob layout, and the error taxonomy.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::store::{FieldSchema, VectorDataType};
use serde_json::json;
/// A file-backed database URL inside `dir`, so reopening can be exercised.
fn database_url(dir: &tempfile::TempDir) -> String {
    format!("sqlite://{}", dir.path().join("vectors.db").display())
}
/// A store on a throwaway in-memory database with `metric` configured.
async fn memory_store(metric: Metric) -> SqliteVectorStore {
    SqliteVectorStore::connect("sqlite::memory:")
        .await
        .unwrap()
        .with_metric(metric)
}
/// A schema of `id` (primary key), `embedding` (dim `dim`), and `kind`.
fn schema(dim: usize) -> CollectionSchema {
    CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(dim),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
        FieldSchema::new("kind", VectorDataType::Varchar),
    ])
}
/// The JSON object of `value`, for the document helpers below.
fn obj(value: &Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}
/// One document with the three-dimensional embedding the sample data uses.
fn doc(id: &str, embedding: [f32; 3], kind: &str) -> Map<String, Value> {
    obj(&json!({
        "id": id,
        "embedding": [embedding[0], embedding[1], embedding[2]],
        "kind": kind,
    }))
}
/// Four documents spanning both `kind` values and all score quadrants.
fn sample_docs() -> Vec<Map<String, Value>> {
    vec![
        doc("a", [1.0, 0.0, 0.0], "x"),
        doc("b", [0.0, 1.0, 0.0], "y"),
        doc("c", [0.9, 0.1, 0.0], "x"),
        doc("d", [-1.0, 0.0, 0.0], "y"),
    ]
}
/// The primary keys of `hits`, in order.
fn ids(hits: &[VectorSearchResult]) -> Vec<String> {
    hits.iter()
        .map(|hit| hit.fields["id"].as_str().unwrap_or_default().to_string())
        .collect()
}
/// A store holding [`sample_docs`] in collection `docs`.
async fn seeded_store(metric: Metric) -> SqliteVectorStore {
    let store = memory_store(metric).await;
    store.create_collection("docs", schema(3)).await.unwrap();
    store.add_docs("docs", sample_docs()).await.unwrap();
    store
}
#[tokio::test]
async fn create_add_and_search_round_trip_over_a_database_file() {
    let dir = tempfile::tempdir().unwrap();
    let url = database_url(&dir);
    let store = SqliteVectorStore::connect(&url).await.unwrap();
    store.create_collection("docs", schema(3)).await.unwrap();
    assert!(store.collection_exists("docs").await.unwrap());
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        vec!["docs".to_string()]
    );
    assert_eq!(store.get_schema("docs").await.unwrap(), Some(schema(3)));
    assert_eq!(store.add_docs("docs", sample_docs()).await.unwrap(), 4);
    assert_eq!(store.count("docs").await.unwrap(), 4);

    let hits = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(ids(&hits), vec!["a", "c", "b", "d"]);
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    assert_eq!(hits[0].fields["kind"], "x");
    // The stored embedding comes back as the reference's array view.
    assert_eq!(hits[0].fields["embedding"], json!([1.0, 0.0, 0.0]));

    // The database file, not the process, owns the data.
    drop(store);
    let reopened = SqliteVectorStore::connect(&url).await.unwrap();
    assert_eq!(reopened.count("docs").await.unwrap(), 4);
    assert_eq!(reopened.get_schema("docs").await.unwrap(), Some(schema(3)));
    assert_eq!(
        ids(&reopened
            .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
            .await
            .unwrap()),
        vec!["a", "c", "b", "d"]
    );
}
#[tokio::test]
async fn ranking_and_scores_match_the_reference_for_every_metric() {
    let queries = [
        (
            VectorQuery::new(vec![1.0, 0.0, 0.0]),
            Value::Null,
            "unfiltered",
        ),
        (
            VectorQuery::new(vec![0.0, 1.0, 0.0]),
            json!({"kind": "x"}),
            "filtered",
        ),
        (
            VectorQuery::new(vec![0.5, 0.5, 0.0]).with_top_k(2),
            json!({"$or": [{"kind": "x"}, {"kind": "y"}]}),
            "truncated",
        ),
    ];
    for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
        let store = seeded_store(metric).await;
        let reference = InMemoryVectorStore::new();
        reference
            .create_collection("docs", schema(3))
            .await
            .unwrap();
        reference.add_docs("docs", sample_docs()).await.unwrap();
        for (mut query, filters, label) in queries.clone() {
            query.filters = filters;
            let ours = store.search("docs", &query).await.unwrap();
            let theirs = reference
                .search_with_metric("docs", &query, metric)
                .unwrap();
            assert_eq!(ids(&ours), ids(&theirs), "{metric:?} {label}");
            for (left, right) in ours.iter().zip(&theirs) {
                assert!(
                    (left.score - right.score).abs() < 1e-6,
                    "{metric:?} {label}: {} vs {}",
                    left.score,
                    right.score
                );
                assert_eq!(left.fields, right.fields, "{metric:?} {label}");
            }
        }
    }
}
#[tokio::test]
async fn top_k_truncates_and_ties_break_on_primary_key() {
    let store = memory_store(Metric::Cosine).await;
    store.create_collection("docs", schema(3)).await.unwrap();
    store
        .add_docs(
            "docs",
            vec![
                doc("b", [1.0, 0.0, 0.0], "x"),
                doc("a", [1.0, 0.0, 0.0], "x"),
                doc("c", [0.0, 1.0, 0.0], "y"),
            ],
        )
        .await
        .unwrap();

    let truncated = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(2))
        .await
        .unwrap();
    assert_eq!(ids(&truncated), vec!["a", "b"]);
    assert!((truncated[0].score - truncated[1].score).abs() < f32::EPSILON);

    let none = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(0))
        .await
        .unwrap();
    assert_eq!(none, Vec::new());
    let all = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(ids(&all), vec!["a", "b", "c"]);
}
#[tokio::test]
async fn malformed_filters_are_validation_errors() {
    let store = seeded_store(Metric::Cosine).await;
    for filters in [
        json!(["kind"]),
        json!("kind"),
        json!(5),
        json!({"$xor": {"kind": "x"}}),
        json!({"$and": 5}),
        json!({"$not": ["kind"]}),
        json!({"kind": {"$mod": 2}}),
    ] {
        let query = VectorQuery::new(vec![1.0, 0.0, 0.0]).with_filters(filters.clone());
        let error = store.search("docs", &query).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "filters: {filters}");
        let error = store
            .delete_docs_by_filters("docs", &filters)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "filters: {filters}");
    }
    // Nothing was deleted by the rejected filters.
    assert_eq!(store.count("docs").await.unwrap(), 4);
    let error = store
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_filters(json!(["kind"])),
        )
        .await
        .unwrap_err();
    assert!(error.message().contains("object"), "unexpected: {error}");
}
#[tokio::test]
async fn delete_count_list_and_drop_round_trip() {
    let store = seeded_store(Metric::Cosine).await;
    let doomed = vec!["a".to_string(), "missing".to_string()];
    assert_eq!(store.delete_docs_by_ids("docs", &doomed).await.unwrap(), 1);
    assert_eq!(store.delete_docs_by_ids("docs", &[]).await.unwrap(), 0);
    assert_eq!(store.count("docs").await.unwrap(), 3);
    // `b` and `d` are the `y` documents.
    assert_eq!(
        store
            .delete_docs_by_filters("docs", &json!({"kind": "y"}))
            .await
            .unwrap(),
        2
    );
    assert_eq!(store.count("docs").await.unwrap(), 1);
    assert!(store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap()
        .iter()
        .all(|hit| hit.fields["id"] == "c"));

    // A null filter matches every remaining document.
    assert_eq!(
        store
            .delete_docs_by_filters("docs", &Value::Null)
            .await
            .unwrap(),
        1
    );
    assert_eq!(store.count("docs").await.unwrap(), 0);
    assert!(store.collection_exists("docs").await.unwrap());
    let remaining = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(remaining, Vec::new());

    store.delete_collection("docs").await.unwrap();
    assert!(!store.collection_exists("docs").await.unwrap());
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(store.get_schema("docs").await.unwrap(), None);
    // Deleting a missing collection is a no-op.
    store.delete_collection("docs").await.unwrap();
}
#[tokio::test]
async fn dimension_mismatches_are_validation_errors() {
    let store = memory_store(Metric::Cosine).await;
    let error = store
        .create_collection("zero", schema(0))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    store.create_collection("docs", schema(3)).await.unwrap();

    let error = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("dimension"), "unexpected: {error}");

    let error = store
        .search(
            "docs",
            &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_vector_field("missing"),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let narrow = obj(&json!({"id": "a", "embedding": [1.0, 0.0], "kind": "x"}));
    let error = store.add_docs("docs", vec![narrow]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(store.count("docs").await.unwrap(), 0);
}
#[tokio::test]
async fn unknown_collections_are_not_found() {
    let store = memory_store(Metric::Cosine).await;
    let query = VectorQuery::new(vec![1.0, 0.0, 0.0]);
    for (label, error) in [
        ("search", store.search("missing", &query).await.err()),
        (
            "add",
            store
                .add_docs("missing", vec![doc("a", [1.0, 0.0, 0.0], "x")])
                .await
                .err(),
        ),
        (
            "delete by ids",
            store
                .delete_docs_by_ids("missing", &["a".to_string()])
                .await
                .err(),
        ),
        (
            "delete by filters",
            store
                .delete_docs_by_filters("missing", &Value::Null)
                .await
                .err(),
        ),
        ("count", store.count("missing").await.err()),
    ] {
        let error = error.unwrap_or_else(|| panic!("{label}: expected a missing collection"));
        assert_eq!(error.kind(), ErrorKind::NotFound, "{label}: {error}");
        assert!(error.message().contains("missing"), "{label}: {error}");
    }
}
#[tokio::test]
async fn a_collection_metric_must_match_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let url = database_url(&dir);
    let store = SqliteVectorStore::connect(&url).await.unwrap();
    assert_eq!(store.metric(), Metric::Cosine);
    store.create_collection("docs", schema(3)).await.unwrap();
    store.add_docs("docs", sample_docs()).await.unwrap();
    let hits = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap();
    assert!((hits[0].score - 1.0).abs() < 1e-6);

    let mismatched = SqliteVectorStore::connect(&url)
        .await
        .unwrap()
        .with_metric(Metric::Dot);
    let error = mismatched
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(message.contains("cosine"), "unexpected: {message}");
    assert!(message.contains("dot"), "unexpected: {message}");
    // Writes are refused just as loudly, so a collection cannot be extended
    // through a store that would score it differently.
    let error = mismatched
        .add_docs("docs", vec![doc("e", [0.0, 0.0, 1.0], "x")])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    // A metric name this backend cannot score is a validation error too.
    assert_eq!(decode_metric("docs", "cosine").unwrap(), Metric::Cosine);
    assert_eq!(decode_metric("docs", "dot").unwrap(), Metric::Dot);
    assert_eq!(
        decode_metric("docs", "euclidean").unwrap(),
        Metric::Euclidean
    );
    let error = decode_metric("docs", "manhattan").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("manhattan"), "unexpected: {error}");
}
#[tokio::test]
async fn unusable_collection_names_are_validation_errors() {
    let store = memory_store(Metric::Cosine).await;
    for name in ["", "a/b", "a\\b", ".", "..", "nul\0byte"] {
        let error = store.create_collection(name, schema(3)).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "name: {name:?}");
        let error = store
            .search(name, &VectorQuery::new(vec![1.0, 0.0, 0.0]))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "name: {name:?}");
    }
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        Vec::<String>::new()
    );
}
#[tokio::test]
async fn duplicate_primary_keys_are_rejected_and_insert_nothing() {
    let store = memory_store(Metric::Cosine).await;
    store.create_collection("docs", schema(3)).await.unwrap();
    store
        .add_docs("docs", vec![doc("a", [1.0, 0.0, 0.0], "x")])
        .await
        .unwrap();
    let error = store
        .add_docs("docs", vec![doc("a", [0.0, 1.0, 0.0], "y")])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains('a'), "unexpected: {error}");

    let batch = vec![
        doc("b", [0.0, 1.0, 0.0], "y"),
        doc("b", [0.0, 0.0, 1.0], "y"),
        doc("c", [0.0, 0.0, 1.0], "y"),
    ];
    let error = store.add_docs("docs", batch).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    // Neither the duplicate nor the valid sibling of the batch was written.
    assert_eq!(store.count("docs").await.unwrap(), 1);
    assert_eq!(store.add_docs("docs", Vec::new()).await.unwrap(), 0);
}
#[tokio::test]
async fn defaults_and_auto_ids_come_from_the_reference() {
    let store = memory_store(Metric::Cosine).await;
    let auto = CollectionSchema::new(vec![
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
            default_value: Some(json!("unknown")),
            ..FieldSchema::new("kind", VectorDataType::Varchar)
        },
    ]);
    store.create_collection("auto", auto).await.unwrap();
    store
        .add_docs("auto", vec![obj(&json!({"embedding": [1.0, 0.0]}))])
        .await
        .unwrap();
    let hits = store
        .search("auto", &VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    let id = hits[0].fields["id"].as_str().unwrap_or_default();
    assert!(!id.is_empty(), "the reference mints a primary key");
    assert_eq!(hits[0].fields["kind"], "unknown");
    assert_eq!(store.count("auto").await.unwrap(), 1);
}
#[tokio::test]
async fn every_vector_field_of_a_document_is_persisted() {
    let store = memory_store(Metric::Cosine).await;
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
            dim: Some(3),
            ..FieldSchema::new("aux", VectorDataType::FloatVector)
        },
    ]);
    store.create_collection("multi", two_vectors).await.unwrap();
    store
        .add_docs(
            "multi",
            vec![obj(&json!({
                "id": "a",
                "embedding": [1.0, 0.0],
                "aux": [0.0, 1.0, 0.0],
            }))],
        )
        .await
        .unwrap();
    let hits = store
        .search("multi", &VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hits[0].fields["aux"], json!([0.0, 1.0, 0.0]));
    // The named field is scored, not the first one.
    let aux = store
        .search(
            "multi",
            &VectorQuery::new(vec![0.0, 1.0, 0.0]).with_vector_field("aux"),
        )
        .await
        .unwrap();
    assert!((aux[0].score - 1.0).abs() < 1e-6);
    assert_eq!(
        store
            .get_schema("multi")
            .await
            .unwrap()
            .map(|s| s.fields.len()),
        Some(3)
    );
}
#[tokio::test]
async fn a_corrupt_embedding_blob_is_an_external_error() {
    let store = seeded_store(Metric::Cosine).await;
    sqlx::query("UPDATE openjiuwen_vector_embeddings SET embedding = ?1 WHERE id = ?2")
        .bind(vec![0_u8; 3])
        .bind("a")
        .execute(store.pool())
        .await
        .unwrap();
    let error = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::External);
    assert!(error.message().contains("embedding"), "unexpected: {error}");

    sqlx::query("UPDATE openjiuwen_vector_documents SET doc_json = ?1 WHERE id = ?2")
        .bind("{not json")
        .bind("b")
        .execute(store.pool())
        .await
        .unwrap();
    let error = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::External);
}
#[test]
fn vectors_round_trip_through_little_endian_bytes() {
    let vector = [1.0_f32, -2.5, 0.0, 3.25];
    let bytes = encode_vector(&vector);
    assert_eq!(bytes.len(), vector.len() * 4);
    assert_eq!(&bytes[0..4], &1.0_f32.to_le_bytes());
    assert_eq!(
        decode_vector("docs", "a", "embedding", &bytes, vector.len()).unwrap(),
        vector
    );
    let error = decode_vector("docs", "a", "embedding", &bytes, 3).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::External);
}
