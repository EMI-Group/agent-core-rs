//! Document storage.
//!
//! Collections of JSON documents with a small filter language (equality plus
//! `$eq/$ne/$gt/$gte/$lt/$lte/$in/$contains`), ordering, pagination, and counts.
//! The Python `BaseDbStore` returns a `SQLAlchemy` engine; that is not portable, so
//! this crate exposes a document-shaped contract instead.
//!
//! Every Python consumer of the db store (message store, checkpoints, memory)
//! reads and writes JSON rows addressed by an `id` inside a named collection, so
//! [`Document`] is a [`serde_json::Map`] plus that id. Four backends implement
//! [`DocumentStore`]:
//!
//! - [`InMemoryDocumentStore`] — the behavioral reference (a `BTreeMap` behind a lock).
//! - [`LocalFileDocumentStore`] — one JSON snapshot per collection below a root.
//! - `SqliteDocumentStore` — `SQLite` behind the `sqlite` feature.
//! - `ElasticsearchDocumentStore` — an Elasticsearch index, in
//!   [`external::elasticsearch`](super::external::elasticsearch).
//!
//! # Shared semantics
//!
//! - A document's id is part of its JSON view ([`Document::to_value`]), so
//!   filters and `order_by` can address the field `id` like any other field.
//! - Reading or counting inside a missing collection yields `None` / an empty
//!   vector / `0`; creating and dropping are idempotent; `upsert` creates the
//!   collection it writes into.
//! - `upsert` replaces documents by id and reports how many documents it wrote.
//! - `query` filters, orders, then applies `offset` and `limit`, in that order;
//!   `count` and `delete` use only the filter.
//! - All ordering, filtering, and paging is implemented once in `run_query`,
//!   which every backend calls, so rankings cannot drift between backends.

use std::cmp::Ordering;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::json::get_path;

use super::filter;

pub mod file;
pub mod memory;
#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use file::LocalFileDocumentStore;
pub use memory::InMemoryDocumentStore;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteDocumentStore;

/// The field name reserved for a document's primary key.
///
/// [`Document::to_value`] always writes this key from [`Document::id`], so a
/// field with the same name is shadowed in the JSON view (and therefore in
/// filters and ordering).
pub const ID_FIELD: &str = "id";

/// One stored document: an id plus free-form JSON fields.
///
/// The struct form is `{"id": ..., "fields": {...}}`; [`Document::to_value`]
/// produces the flat `{"id": ..., <fields...>}` view the filter language and
/// `order_by` operate on, and [`Document::from_value`] converts back.
/// Deserialization is tolerant: unknown members are rejected only where serde
/// requires it, missing members fall back to their defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct Document {
    /// The primary key, unique within a collection.
    pub id: String,
    /// The document payload; every JSON value (including nested objects and
    /// arrays) is preserved verbatim.
    pub fields: Map<String, Value>,
}

impl Document {
    /// Builds an empty document with the given id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: Map::new(),
        }
    }

    /// Sets `name` to `value`, returning the updated document.
    ///
    /// A field named [`ID_FIELD`] is accepted but shadowed by the document id in
    /// [`Document::to_value`]; use [`Document::new`] to change the id.
    #[must_use]
    pub fn with_field(self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        let mut fields = self.fields;
        fields.insert(name.into(), value.into());
        Self {
            id: self.id,
            fields,
        }
    }

    /// Reads one top-level field, or `None` when it is absent.
    ///
    /// This is a plain map lookup; nested values are read with
    /// [`openjiuwen_common::utils::json::get_path`] on [`Document::to_value`].
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    /// The flat JSON view used by filters and ordering: `{"id": <id>, <fields...>}`.
    pub fn to_value(&self) -> Value {
        let mut map = self.fields.clone();
        map.insert(ID_FIELD.to_string(), Value::String(self.id.clone()));
        Value::Object(map)
    }

    /// Rebuilds a document from the flat JSON view produced by
    /// [`Document::to_value`].
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when `value`
    /// is not an object, when its `id` member is missing, or when `id` is not a
    /// string.
    pub fn from_value(value: Value) -> Result<Self> {
        let Value::Object(mut map) = value else {
            return Err(Error::validation(
                "document must be a JSON object to convert into a Document",
            ));
        };
        let id = match map.remove(ID_FIELD) {
            Some(Value::String(id)) => id,
            Some(_) => {
                return Err(Error::validation(
                    "document id must be a JSON string".to_string(),
                ))
            }
            None => return Err(Error::validation("document is missing its id field")),
        };
        Ok(Self { id, fields: map })
    }
}

/// Sort direction of one `order_by` criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    /// Ascending: smallest first, `null`/missing values first.
    #[default]
    Asc,
    /// Descending: largest first, `null`/missing values last.
    Desc,
}

/// A filtered, ordered, paginated read.
///
/// `filter` uses the [`super::filter`] language against
/// [`Document::to_value`], so `id` and dotted paths such as `meta.owner` are
/// addressable. `order_by` is a priority list: the first criterion is primary
/// and later criteria break its ties. `limit` is unbounded when `None` and
/// `offset` skips matching documents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Query {
    /// The filter; `null` matches every document.
    pub filter: Value,
    /// The maximum number of documents to return; `None` returns all that match.
    pub limit: Option<usize>,
    /// How many matching documents to skip before collecting results.
    pub offset: usize,
    /// Ordering criteria, most significant first.
    pub order_by: Vec<(String, SortOrder)>,
}

impl Default for Query {
    fn default() -> Self {
        Self {
            filter: Value::Null,
            limit: None,
            offset: 0,
            order_by: Vec::new(),
        }
    }
}

/// The contract every document backend implements.
///
/// Implementations are expected to be cheap to share: every method takes
/// `&self`, so a store can be used through an `Arc<dyn DocumentStore>`.
///
/// # Semantics
///
/// - `create_collection` and `drop_collection` are idempotent; dropping a
///   missing collection is a no-op.
/// - `list_collections` returns names in ascending order.
/// - `upsert` creates the collection when needed, replaces documents by id, and
///   returns the number of documents written (the length of the input).
/// - `get`, `query`, and `count` treat a missing collection as empty.
/// - `query` applies the filter, then `order_by`, then `offset`, then `limit`.
/// - `delete` and `count` use only `filter`; `limit`, `offset`, and `order_by`
///   are ignored. `delete` returns the number of documents removed.
///
/// # Errors
///
/// Every method returns [`openjiuwen_common::error::ErrorKind::Validation`] for
/// an invalid collection name (empty, a path separator, a NUL byte, or a `.` /
/// `..` segment) — collection names become file names in some backends.
/// Backends may also surface storage errors as
/// [`openjiuwen_common::error::ErrorKind::External`].
#[async_trait]
pub trait DocumentStore: Send + Sync {
    /// Creates an empty collection; creating an existing one is a no-op.
    async fn create_collection(&self, collection: &str) -> Result<()>;

    /// Lists every collection name in ascending order.
    async fn list_collections(&self) -> Result<Vec<String>>;

    /// Drops a collection and its documents; a missing collection is a no-op.
    async fn drop_collection(&self, collection: &str) -> Result<()>;

    /// Writes `documents` into `collection`, replacing documents with the same id.
    ///
    /// The collection is created when it does not exist. Returns the number of
    /// documents written.
    async fn upsert(&self, collection: &str, documents: Vec<Document>) -> Result<usize>;

    /// Reads one document by id, or `None` when it or its collection is absent.
    async fn get(&self, collection: &str, id: &str) -> Result<Option<Document>>;

    /// Reads the documents matching `query`, in the order the query requests.
    async fn query(&self, collection: &str, query: &Query) -> Result<Vec<Document>>;

    /// Removes the documents matching `query`'s filter, returning how many were
    /// removed.
    async fn delete(&self, collection: &str, query: &Query) -> Result<usize>;

    /// Counts the documents matching `query`'s filter.
    async fn count(&self, collection: &str, query: &Query) -> Result<usize>;
}

/// Rejects collection names that could escape or corrupt a backend's namespace.
///
/// A name must be non-empty, must not contain `/`, `\`, or NUL, and must not be
/// `.` or `..`.
pub(crate) fn validate_collection(collection: &str) -> Result<()> {
    let invalid = if collection.is_empty() {
        Some("must not be empty")
    } else if collection.contains('/') || collection.contains('\\') {
        Some("must not contain a path separator")
    } else if collection.contains('\0') {
        Some("must not contain a NUL byte")
    } else if collection == "." || collection == ".." {
        Some("must not be a dot segment")
    } else {
        None
    };
    match invalid {
        Some(reason) => Err(Error::validation(format!(
            "invalid document collection name `{collection}`: {reason}"
        ))),
        None => Ok(()),
    }
}

/// The documents of `documents` whose [`Document::to_value`] satisfies `filter`.
pub(crate) fn matching<'a>(
    documents: impl IntoIterator<Item = &'a Document>,
    filter: &Value,
) -> Vec<&'a Document> {
    documents
        .into_iter()
        .filter(|document| filter::matches(filter, &document.to_value()))
        .collect()
}

/// Filters, orders, and pages `documents` according to `query`.
///
/// This is the single implementation of query semantics shared by every backend:
/// [`Document::to_value`] feeds the filter, `order_by` breaks ties criterion by
/// criterion (stable, so equal documents keep the backend's iteration order),
/// then `offset` and `limit` are applied.
pub(crate) fn run_query<'a>(
    documents: impl IntoIterator<Item = &'a Document>,
    query: &Query,
) -> Vec<&'a Document> {
    let mut matched: Vec<(&'a Document, Value)> = documents
        .into_iter()
        .map(|document| (document, document.to_value()))
        .filter(|(_, value)| filter::matches(&query.filter, value))
        .collect();

    if !query.order_by.is_empty() {
        matched.sort_by(|(_, left), (_, right)| compare_rows(left, right, &query.order_by));
    }

    let limit = query.limit.unwrap_or(usize::MAX);
    matched
        .into_iter()
        .skip(query.offset)
        .take(limit)
        .map(|(document, _)| document)
        .collect()
}

/// Compares two document views by the `order_by` criteria, primary first.
fn compare_rows(left: &Value, right: &Value, order_by: &[(String, SortOrder)]) -> Ordering {
    for (field, order) in order_by {
        let ordering = compare_values(get_path(left, field), get_path(right, field));
        let ordering = match order {
            SortOrder::Asc => ordering,
            SortOrder::Desc => ordering.reverse(),
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

/// Ascending order of two optional field values.
///
/// `null` and missing values sort before every other value. Numbers compare
/// numerically, strings lexicographically, booleans with `false < true`; values
/// of different types compare by JSON type (number, string, boolean, array,
/// object), and two arrays or two objects compare by their serialized JSON so
/// the order is total and deterministic.
fn compare_values(left: Option<&Value>, right: Option<&Value>) -> Ordering {
    let left = left.filter(|value| !value.is_null());
    let right = right.filter(|value| !value.is_null());
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => match (left, right) {
            (Value::Number(left), Value::Number(right)) => match (left.as_f64(), right.as_f64()) {
                (Some(left), Some(right)) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
                _ => left.to_string().cmp(&right.to_string()),
            },
            (Value::String(left), Value::String(right)) => left.cmp(right),
            (Value::Bool(left), Value::Bool(right)) => left.cmp(right),
            _ => type_rank(left)
                .cmp(&type_rank(right))
                .then_with(|| left.to_string().cmp(&right.to_string())),
        },
    }
}

/// The position of a JSON type in the ordering of unlike values.
fn type_rank(value: &Value) -> u8 {
    match value {
        Value::Null => 0,
        Value::Number(_) => 1,
        Value::String(_) => 2,
        Value::Bool(_) => 3,
        Value::Array(_) => 4,
        Value::Object(_) => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> Vec<Document> {
        vec![
            Document::new("a")
                .with_field("name", "Ada")
                .with_field("age", 36),
            Document::new("b")
                .with_field("name", "Bob")
                .with_field("age", 24),
            Document::new("c")
                .with_field("name", "Cid")
                .with_field("age", 36),
        ]
    }

    fn ids<'a>(documents: &[&'a Document]) -> Vec<&'a str> {
        documents
            .iter()
            .map(|document| document.id.as_str())
            .collect()
    }

    #[test]
    fn document_json_view_includes_id_and_fields() {
        let document = Document::new("a")
            .with_field("age", 36)
            .with_field("meta", json!({"owner": "ada"}));
        assert_eq!(
            document.to_value(),
            json!({"id": "a", "age": 36, "meta": {"owner": "ada"}})
        );
        assert_eq!(document.field("age"), Some(&json!(36)));
        assert_eq!(document.field("missing"), None);
    }

    #[test]
    fn to_value_shadows_a_field_named_id() {
        let document = Document::new("real").with_field("id", "shadowed");
        assert_eq!(document.id, "real");
        assert_eq!(document.to_value(), json!({"id": "real"}));
    }

    #[test]
    fn from_value_round_trips_the_flat_view() {
        let document = Document::new("a")
            .with_field("nested", json!({"list": [1, 2, 3]}))
            .with_field("nil", Value::Null);
        let parsed = Document::from_value(document.to_value()).expect("parse");
        assert_eq!(parsed, document);
    }

    #[test]
    fn from_value_rejects_malformed_documents() {
        for value in [json!([]), json!("text"), json!(null), json!({"age": 1})] {
            let error = Document::from_value(value).expect_err("malformed");
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
        }
        let error = Document::from_value(json!({"id": 7})).expect_err("non-string id");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn query_default_matches_everything() {
        let query = Query::default();
        assert_eq!(query.filter, Value::Null);
        assert_eq!(query.limit, None);
        assert_eq!(query.offset, 0);
        assert_eq!(query.order_by, Vec::<(String, SortOrder)>::new());

        let parsed: Query = serde_json::from_value(json!({"unexpected": true})).expect("parse");
        assert_eq!(parsed, query);
    }

    #[test]
    fn query_serde_uses_snake_case_pairs() {
        let query = Query {
            filter: json!({"age": {"$gte": 18}}),
            limit: Some(2),
            offset: 1,
            order_by: vec![("age".to_string(), SortOrder::Desc)],
        };
        let encoded = serde_json::to_value(&query).expect("encode");
        assert_eq!(
            encoded,
            json!({
                "filter": {"age": {"$gte": 18}},
                "limit": 2,
                "offset": 1,
                "order_by": [["age", "desc"]],
            })
        );
        assert_eq!(
            serde_json::from_value::<Query>(encoded).expect("decode"),
            query
        );
    }

    #[test]
    fn sort_order_defaults_to_ascending() {
        assert_eq!(SortOrder::default(), SortOrder::Asc);
        assert_eq!(
            serde_json::from_value::<SortOrder>(json!("desc")).expect("parse"),
            SortOrder::Desc
        );
    }

    #[test]
    fn run_query_applies_filter_order_offset_and_limit() {
        let documents = sample();
        let query = Query {
            filter: json!({"age": {"$gte": 30}}),
            limit: Some(1),
            offset: 1,
            order_by: vec![("age".to_string(), SortOrder::Asc)],
        };
        // Both matches have age 36, so their input order survives; offset skips
        // `a` and the limit keeps `c`.
        assert_eq!(ids(&run_query(&documents, &query)), vec!["c"]);
    }

    #[test]
    fn ordering_compares_numbers_and_strings() {
        let documents = vec![
            Document::new("n").with_field("value", 10),
            Document::new("s").with_field("value", "9"),
            Document::new("n2").with_field("value", 9),
        ];
        let ascending = Query {
            order_by: vec![("value".to_string(), SortOrder::Asc)],
            ..Query::default()
        };
        assert_eq!(
            ids(&run_query(&documents, &ascending)),
            vec!["n2", "n", "s"]
        );
    }

    #[test]
    fn ordering_puts_null_and_missing_first_ascending_and_last_descending() {
        let documents = vec![
            Document::new("a").with_field("rank", 5),
            Document::new("b").with_field("rank", Value::Null),
            Document::new("c"),
            Document::new("d").with_field("rank", 2),
        ];
        let ascending = Query {
            order_by: vec![
                ("rank".to_string(), SortOrder::Asc),
                (ID_FIELD.to_string(), SortOrder::Asc),
            ],
            ..Query::default()
        };
        assert_eq!(
            ids(&run_query(&documents, &ascending)),
            vec!["b", "c", "d", "a"]
        );

        let descending = Query {
            order_by: vec![
                ("rank".to_string(), SortOrder::Desc),
                (ID_FIELD.to_string(), SortOrder::Asc),
            ],
            ..Query::default()
        };
        assert_eq!(
            ids(&run_query(&documents, &descending)),
            vec!["a", "d", "b", "c"]
        );
    }

    #[test]
    fn ordering_uses_every_criterion_in_priority_order() {
        let documents = sample();
        let query = Query {
            order_by: vec![
                ("age".to_string(), SortOrder::Desc),
                ("name".to_string(), SortOrder::Desc),
            ],
            ..Query::default()
        };
        assert_eq!(ids(&run_query(&documents, &query)), vec!["c", "a", "b"]);
    }

    #[test]
    fn collection_names_reject_escaping_shapes() {
        for name in ["", "a/b", "a\\b", "..", ".", "a\0b"] {
            let error = validate_collection(name).expect_err("invalid");
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
        }
        for name in ["docs", "a.b", ".hidden", "with space"] {
            validate_collection(name).expect("valid");
        }
    }
}

/// The trait contract every backend runs against itself.
///
/// Each backend's tests call [`run`] so filtering, ordering, paging, counting,
/// and missing-collection behavior are verified once and cannot drift.
#[cfg(test)]
pub(crate) mod contract {
    use super::{Document, DocumentStore, Query, SortOrder};
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    /// The fixture written by [`seed`]: ages 36, 24, 36 with an owner per document.
    fn sample() -> Vec<Document> {
        vec![
            Document::new("a")
                .with_field("name", "Ada")
                .with_field("age", 36)
                .with_field("meta", json!({"owner": "ada"}))
                .with_field("tags", json!(["rust", "store"])),
            Document::new("b")
                .with_field("name", "Bob")
                .with_field("age", 24)
                .with_field("meta", json!({"owner": "bob"}))
                .with_field("tags", json!(["rust"])),
            Document::new("c")
                .with_field("name", "Cid")
                .with_field("age", 36)
                .with_field("meta", json!({"owner": "ada"}))
                .with_field("tags", json!([])),
        ]
    }

    /// Writes [`sample`] into `collection` (which must be empty) and returns it.
    async fn seed(store: &dyn DocumentStore, collection: &str) -> Vec<Document> {
        let documents = sample();
        let written = store
            .upsert(collection, documents.clone())
            .await
            .expect("seed upsert");
        assert_eq!(written, 3);
        documents
    }

    /// Query helper with only a filter set.
    fn filtered(filter: serde_json::Value) -> Query {
        Query {
            filter,
            ..Query::default()
        }
    }

    /// The ids of `documents`, in order.
    fn ids(documents: &[Document]) -> Vec<String> {
        documents
            .iter()
            .map(|document| document.id.clone())
            .collect()
    }

    /// Verifies the whole [`DocumentStore`] contract against one open store.
    pub(crate) async fn run(store: &dyn DocumentStore) {
        missing_collections(store).await;
        collection_lifecycle(store).await;
        upsert_get_and_replace(store).await;
        filters(store).await;
        ordering_and_paging(store).await;
        null_and_type_ordering(store).await;
        counting(store).await;
        deleting(store).await;
        invalid_collection_names(store).await;
    }

    async fn missing_collections(store: &dyn DocumentStore) {
        let query = Query::default();
        assert!(store.get("absent", "a").await.expect("get").is_none());
        assert_eq!(
            store.query("absent", &query).await.expect("query"),
            Vec::<Document>::new()
        );
        assert_eq!(store.count("absent", &query).await.expect("count"), 0);
        assert_eq!(store.delete("absent", &query).await.expect("delete"), 0);
        assert!(!store
            .list_collections()
            .await
            .expect("list")
            .contains(&"absent".to_string()));
    }

    async fn collection_lifecycle(store: &dyn DocumentStore) {
        store.create_collection("lifecycle").await.expect("create");
        store
            .create_collection("lifecycle")
            .await
            .expect("create twice");
        let names = store.list_collections().await.expect("list");
        assert!(names.contains(&"lifecycle".to_string()));
        assert!(names.windows(2).all(|pair| pair[0] < pair[1]), "sorted");

        store.drop_collection("lifecycle").await.expect("drop");
        store
            .drop_collection("lifecycle")
            .await
            .expect("drop twice");
        assert!(!store
            .list_collections()
            .await
            .expect("list")
            .contains(&"lifecycle".to_string()));
    }

    async fn upsert_get_and_replace(store: &dyn DocumentStore) {
        seed(store, "upsert").await;

        let mut names = store.list_collections().await.expect("list");
        names.retain(|name| name == "upsert");
        assert_eq!(names, vec!["upsert".to_string()], "upsert auto-creates");

        let bob = store.get("upsert", "b").await.expect("get").expect("b");
        assert_eq!(bob.field("name"), Some(&json!("Bob")));
        assert_eq!(bob.field("meta"), Some(&json!({"owner": "bob"})));

        // A second upsert with the same id replaces the whole document.
        assert_eq!(
            store
                .upsert("upsert", vec![Document::new("b").with_field("age", 25)],)
                .await
                .expect("replace"),
            1
        );
        let replaced = store.get("upsert", "b").await.expect("get").expect("b");
        assert_eq!(replaced.field("age"), Some(&json!(25)));
        assert_eq!(replaced.field("name"), None);

        assert!(store.get("upsert", "missing").await.expect("get").is_none());
        assert_eq!(
            store
                .upsert("upsert", Vec::new())
                .await
                .expect("empty upsert"),
            0
        );
    }

    async fn filters(store: &dyn DocumentStore) {
        seed(store, "filters").await;

        let by_id = store
            .query("filters", &filtered(json!({"id": "b"})))
            .await
            .expect("query");
        assert_eq!(ids(&by_id), vec!["b"]);

        let implicit = store
            .query("filters", &filtered(json!({"name": "Ada"})))
            .await
            .expect("query");
        assert_eq!(ids(&implicit), vec!["a"]);

        let mut older = store
            .query("filters", &filtered(json!({"age": {"$gt": 30}})))
            .await
            .expect("query");
        older.sort_by(|left, right| left.id.cmp(&right.id));
        assert_eq!(ids(&older), vec!["a", "c"]);

        let mut rust = store
            .query("filters", &filtered(json!({"tags": {"$in": ["rust"]}})))
            .await
            .expect("query");
        rust.sort_by(|left, right| left.id.cmp(&right.id));
        assert_eq!(ids(&rust), vec!["a", "b"]);

        let mut owners = store
            .query("filters", &filtered(json!({"meta.owner": "ada"})))
            .await
            .expect("query");
        owners.sort_by(|left, right| left.id.cmp(&right.id));
        assert_eq!(ids(&owners), vec!["a", "c"]);

        let none = store
            .query("filters", &filtered(json!({"meta.owner": "nobody"})))
            .await
            .expect("query");
        assert_eq!(none, Vec::<Document>::new());
    }

    async fn ordering_and_paging(store: &dyn DocumentStore) {
        let documents = sample();
        store.upsert("ordering", documents).await.expect("seed");

        let ascending = Query {
            order_by: vec![
                ("age".to_string(), SortOrder::Asc),
                ("name".to_string(), SortOrder::Desc),
            ],
            ..Query::default()
        };
        assert_eq!(
            ids(&store.query("ordering", &ascending).await.expect("query")),
            vec!["b", "c", "a"]
        );

        let descending = Query {
            order_by: vec![
                ("age".to_string(), SortOrder::Desc),
                ("name".to_string(), SortOrder::Asc),
            ],
            ..Query::default()
        };
        assert_eq!(
            ids(&store.query("ordering", &descending).await.expect("query")),
            vec!["a", "c", "b"]
        );

        let ordered_by_id = Query {
            order_by: vec![("id".to_string(), SortOrder::Asc)],
            ..Query::default()
        };
        let page = Query {
            offset: 1,
            limit: Some(1),
            ..ordered_by_id.clone()
        };
        assert_eq!(
            ids(&store.query("ordering", &page).await.expect("query")),
            vec!["b"]
        );

        let beyond = Query {
            offset: 10,
            ..ordered_by_id.clone()
        };
        assert_eq!(
            store.query("ordering", &beyond).await.expect("query"),
            Vec::<Document>::new()
        );

        let unlimited = Query {
            offset: 1,
            ..ordered_by_id.clone()
        };
        assert_eq!(
            ids(&store.query("ordering", &unlimited).await.expect("query")),
            vec!["b", "c"]
        );

        let unordered = store
            .query("ordering", &Query::default())
            .await
            .expect("query");
        let mut unordered_ids = ids(&unordered);
        unordered_ids.sort();
        assert_eq!(unordered_ids, vec!["a", "b", "c"]);
    }

    async fn null_and_type_ordering(store: &dyn DocumentStore) {
        let documents = vec![
            Document::new("n1").with_field("rank", 5),
            Document::new("n2").with_field("rank", serde_json::Value::Null),
            Document::new("n3"),
            Document::new("n4").with_field("rank", 2),
        ];
        store.upsert("nulls", documents).await.expect("seed");

        let ascending = Query {
            order_by: vec![
                ("rank".to_string(), SortOrder::Asc),
                ("id".to_string(), SortOrder::Asc),
            ],
            ..Query::default()
        };
        assert_eq!(
            ids(&store.query("nulls", &ascending).await.expect("query")),
            vec!["n2", "n3", "n4", "n1"]
        );

        let descending = Query {
            order_by: vec![
                ("rank".to_string(), SortOrder::Desc),
                ("id".to_string(), SortOrder::Asc),
            ],
            ..Query::default()
        };
        assert_eq!(
            ids(&store.query("nulls", &descending).await.expect("query")),
            vec!["n1", "n4", "n2", "n3"]
        );
    }

    async fn counting(store: &dyn DocumentStore) {
        seed(store, "counting").await;
        assert_eq!(
            store
                .count("counting", &Query::default())
                .await
                .expect("count"),
            3
        );
        assert_eq!(
            store
                .count("counting", &filtered(json!({"age": 36})))
                .await
                .expect("count"),
            2
        );
        // `count` ignores paging and ordering.
        let paged = Query {
            filter: json!({"age": {"$gte": 30}}),
            limit: Some(1),
            offset: 5,
            order_by: vec![("age".to_string(), SortOrder::Desc)],
        };
        assert_eq!(store.count("counting", &paged).await.expect("count"), 2);
    }

    async fn deleting(store: &dyn DocumentStore) {
        seed(store, "deleting").await;

        // Paging and ordering are ignored by `delete`: every match is removed.
        let query = Query {
            filter: json!({"age": {"$gte": 30}}),
            limit: Some(1),
            offset: 1,
            order_by: vec![("age".to_string(), SortOrder::Desc)],
        };
        assert_eq!(store.delete("deleting", &query).await.expect("delete"), 2);
        assert_eq!(store.delete("deleting", &query).await.expect("delete"), 0);
        assert_eq!(
            store
                .count("deleting", &Query::default())
                .await
                .expect("count"),
            1
        );
        assert!(store.get("deleting", "a").await.expect("get").is_none());
        assert!(store.get("deleting", "b").await.expect("get").is_some());

        assert_eq!(
            store
                .delete("deleting", &Query::default())
                .await
                .expect("delete all"),
            1
        );
    }

    async fn invalid_collection_names(store: &dyn DocumentStore) {
        for name in ["", "a/b", "a\\b", "..", ".", "bad\0name"] {
            assert_eq!(
                store
                    .create_collection(name)
                    .await
                    .expect_err("invalid")
                    .kind(),
                ErrorKind::Validation,
                "create_collection({name:?})"
            );
            assert_eq!(
                store
                    .upsert(name, Vec::new())
                    .await
                    .expect_err("invalid")
                    .kind(),
                ErrorKind::Validation,
                "upsert({name:?})"
            );
            assert_eq!(
                store.get(name, "a").await.expect_err("invalid").kind(),
                ErrorKind::Validation,
                "get({name:?})"
            );
            assert_eq!(
                store
                    .query(name, &Query::default())
                    .await
                    .expect_err("invalid")
                    .kind(),
                ErrorKind::Validation,
                "query({name:?})"
            );
            assert_eq!(
                store
                    .count(name, &Query::default())
                    .await
                    .expect_err("invalid")
                    .kind(),
                ErrorKind::Validation,
                "count({name:?})"
            );
            assert_eq!(
                store
                    .delete(name, &Query::default())
                    .await
                    .expect_err("invalid")
                    .kind(),
                ErrorKind::Validation,
                "delete({name:?})"
            );
            assert_eq!(
                store
                    .drop_collection(name)
                    .await
                    .expect_err("invalid")
                    .kind(),
                ErrorKind::Validation,
                "drop_collection({name:?})"
            );
        }
    }
}
