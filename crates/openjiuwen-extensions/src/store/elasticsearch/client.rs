//! Elasticsearch REST transport, index-mapping builder, and filter translation.
//!
//! Everything here is private to the parent module: [`RestClient`] speaks the handful of
//! Elasticsearch REST endpoints the vector store needs, [`mapping_body`] turns a
//! [`CollectionSchema`] into an index mapping, [`schema_from_mappings`] reads it
//! back, and [`translate_filter`] compiles the kernel filter language
//! (`openjiuwen_foundation::store::filter`) into an Elasticsearch `query` DSL.
//!
//! # Filter subset
//!
//! The kernel filter language and the Elasticsearch query DSL overlap only
//! partially; the translation is faithful where Elasticsearch has an exact
//! equivalent and rejected otherwise (never silently approximated):
//!
//! - `null` and `{}` → `match_all`; any other non-object filter → `match_none`.
//! - A scalar condition (implicit equality or `$eq`) → `term`.
//! - An array condition → `terms` (Elasticsearch indexes an array field as one
//!   value per element, so whole-array equality is not expressible; that is a
//!   documented approximation of the in-memory backend).
//! - `$and`/`$or`/`$not` → `bool.must`/`bool.should` (with
//!   `minimum_should_match: 1`)/`bool.must_not`, combined with sibling field
//!   conditions using `must`. `$and: []` is `match_all`, `$or: []` is
//!   `match_none`, and a malformed combinator is `match_none`, matching the
//!   kernel truth table.
//! - `$ne` → `bool { must: [match_all], must_not: [term] }`.
//! - `$in` of scalars → `terms`.
//! - `$gt`/`$gte`/`$lt`/`$lte` of a number or string → `range`; any other
//!   condition value is `match_none` (the kernel never orders booleans, null,
//!   arrays, or mixed types).
//! - An unknown `$operator` is always false → `match_none`.
//! - `$contains`, equality to an object, equality to `null`, and non-scalar
//!   `terms` candidates have no faithful Elasticsearch equivalent, so they
//!   return [`Error::validation`] naming the operator and field path and
//!   pointing at the in-memory backend.
//!
//! Elasticsearch rejects a top-level empty query, so every translation returns
//! an explicit `match_all`/`match_none` instead of an empty `bool`.
//!
//! Dotted paths (`meta.owner`) are Elasticsearch field paths; an array-index
//! path (`items[1].kind`) cannot be addressed by Elasticsearch and is passed
//! through unchanged (documented deviation).

use std::fmt;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::{CollectionSchema, FieldSchema, Metric, VectorDataType};
use serde_json::{json, Map, Value};

/// The similarity metric of every index this backend creates.
///
/// [`CollectionSchema`] carries no metric field and the backend keeps no
/// per-collection metric registry, so every index uses cosine similarity — the
/// kernel's default [`Metric`].
const STORE_METRIC: Metric = Metric::Cosine;

/// The `dense_vector.similarity` Elasticsearch expects for a kernel [`Metric`].
pub(crate) fn similarity_for(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::Dot => "dot_product",
        Metric::Euclidean => "l2_norm",
    }
}

/// One raw search hit: the document id, Elasticsearch's native score, and the
/// stored source.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Hit {
    /// The Elasticsearch `_id`, which is the document's primary key.
    pub(crate) id: String,
    /// Elasticsearch's native `_score` (see the module docs for its per-metric
    /// normalization).
    pub(crate) score: f64,
    /// The stored document.
    pub(crate) source: Map<String, Value>,
}

/// A minimal Elasticsearch REST client over the configured endpoint.
///
/// The client is single-node: it never sniffs the cluster, follows cloud ids, or
/// retries behind the caller's back. Credentials from the URL userinfo become
/// an HTTP Basic `Authorization` header and never appear in the base URL (so a
/// `reqwest` error message can never leak them).
#[derive(Clone)]
pub(crate) struct RestClient {
    /// Shared HTTP connection pool.
    http: reqwest::Client,
    /// Endpoint without userinfo, query, fragment, or trailing slash.
    base_url: String,
    /// Basic-auth credentials parsed from the URL userinfo.
    auth: Option<(String, String)>,
}

impl fmt::Debug for RestClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RestClient")
            .field("http", &self.http)
            .field("base_url", &self.base_url)
            .field("basic_auth", &self.auth.is_some())
            .finish()
    }
}

impl RestClient {
    /// Builds a client for `base_url` (already stripped of userinfo and
    /// trailing slash).
    pub(crate) fn new(base_url: String, auth: Option<(String, String)>) -> Result<Self> {
        let http = reqwest::Client::builder().build().map_err(|error| {
            Error::config(format!(
                "elasticsearch HTTP client could not be built: {error}"
            ))
        })?;
        Ok(Self {
            http,
            base_url,
            auth,
        })
    }

    /// Starts a request to `path` (which must begin with `/`).
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let request = self
            .http
            .request(method, format!("{}{path}", self.base_url));
        match &self.auth {
            Some((username, password)) => request.basic_auth(username, Some(password)),
            None => request,
        }
    }

    /// `HEAD /{index}`: `true` for 200, `false` for 404.
    pub(crate) async fn index_exists(&self, index: &str) -> Result<bool> {
        let operation = format!("check vector index {index:?}");
        let response = self
            .request(reqwest::Method::HEAD, &format!("/{index}"))
            .send()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        if response.status().is_success() {
            return Ok(true);
        }
        if response.status().as_u16() == 404 {
            return Ok(false);
        }
        Err(response_error(response, &operation).await)
    }

    /// `PUT /{index}` with an index mapping; an existing index is a validation
    /// error, matching the in-memory contract.
    pub(crate) async fn create_index(
        &self,
        collection: &str,
        index: &str,
        body: &Value,
    ) -> Result<()> {
        let operation = format!("create vector collection {collection:?}");
        let response = self
            .request(reqwest::Method::PUT, &format!("/{index}"))
            .json(body)
            .send()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        if response.status().is_success() {
            return Ok(());
        }
        if response.status() == reqwest::StatusCode::CONFLICT {
            return Err(Error::validation(format!(
                "vector collection {collection:?} already exists"
            )));
        }
        Err(response_error(response, &operation).await)
    }

    /// Deletes an index; a missing index is a no-op.
    pub(crate) async fn delete_index(&self, collection: &str, index: &str) -> Result<()> {
        if !self.index_exists(index).await? {
            return Ok(());
        }
        let operation = format!("delete vector collection {collection:?}");
        let response = self
            .request(reqwest::Method::DELETE, &format!("/{index}"))
            .send()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        if response.status().is_success() || response.status().as_u16() == 404 {
            return Ok(());
        }
        Err(response_error(response, &operation).await)
    }

    /// `GET /{index}/_mapping`, returning the `mappings` object; `None` when the
    /// index does not exist.
    pub(crate) async fn mapping(&self, index: &str) -> Result<Option<Value>> {
        let operation = format!("read the mapping of vector index {index:?}");
        let response = self
            .request(reqwest::Method::GET, &format!("/{index}/_mapping"))
            .send()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        if response.status().as_u16() == 404 {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(response_error(response, &operation).await);
        }
        let body: Value = response
            .json()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        // Elasticsearch answers `{"<index>": {"mappings": {...}}}`; a bare
        // `{"mappings": {...}}` is tolerated for proxies that unwrap it.
        let mappings = body
            .get("mappings")
            .cloned()
            .or_else(|| {
                body.as_object()
                    .and_then(|entries| entries.values().next())
                    .and_then(|entry| entry.get("mappings").cloned())
            })
            .ok_or_else(|| {
                Error::transport(format!(
                    "elasticsearch {operation} response has no `mappings` object"
                ))
            })?;
        Ok(Some(mappings))
    }

    /// `POST /_bulk?refresh=false`: indexes one document per `(id, source)`
    /// pair, returning how many Elasticsearch accepted.
    pub(crate) async fn bulk_index(
        &self,
        index: &str,
        documents: &[(String, Map<String, Value>)],
    ) -> Result<usize> {
        let mut body = String::new();
        for (id, document) in documents {
            body.push_str(&json!({"index": {"_index": index, "_id": id}}).to_string());
            body.push('\n');
            body.push_str(&Value::Object(document.clone()).to_string());
            body.push('\n');
        }
        let operation = format!("bulk index into vector index {index:?}");
        let response = self
            .request(reqwest::Method::POST, "/_bulk?refresh=false")
            .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
            .body(body)
            .send()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        if !response.status().is_success() {
            return Err(response_error(response, &operation).await);
        }
        let answer: Value = response
            .json()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        let items = answer
            .get("items")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                Error::transport(format!(
                    "elasticsearch {operation} response has no `items` array"
                ))
            })?;
        let mut accepted = 0;
        for item in items {
            let result = item
                .as_object()
                .and_then(|operations| operations.values().next())
                .ok_or_else(|| {
                    Error::transport(format!(
                        "elasticsearch {operation} returned a malformed bulk item: {item}"
                    ))
                })?;
            match result.get("error") {
                Some(error) => {
                    let id = result
                        .get("_id")
                        .and_then(Value::as_str)
                        .unwrap_or("<unknown>");
                    let reason = error
                        .get("reason")
                        .and_then(Value::as_str)
                        .or_else(|| error.as_str())
                        .unwrap_or("unknown Elasticsearch error");
                    let error_type = error
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let status = result.get("status").and_then(Value::as_u64).unwrap_or(0);
                    return Err(bulk_item_error(id, reason, error_type, status));
                }
                None => accepted += 1,
            }
        }
        Ok(accepted)
    }

    /// `POST /{index}/_search` with a k-NN body.
    pub(crate) async fn knn_search(&self, index: &str, body: &Value) -> Result<Vec<Hit>> {
        let operation = format!("search vector index {index:?}");
        let response = self
            .request(reqwest::Method::POST, &format!("/{index}/_search"))
            .json(body)
            .send()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        if !response.status().is_success() {
            return Err(response_error(response, &operation).await);
        }
        let answer: Value = response
            .json()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        let hits = answer
            .pointer("/hits/hits")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                Error::transport(format!(
                    "elasticsearch {operation} response has no `hits.hits` array"
                ))
            })?;
        let mut parsed = Vec::with_capacity(hits.len());
        for hit in hits {
            let object = hit.as_object().ok_or_else(|| {
                Error::transport(format!(
                    "elasticsearch {operation} returned a malformed hit: {hit}"
                ))
            })?;
            let id = object
                .get("_id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    Error::transport(format!(
                        "elasticsearch {operation} returned a hit without `_id`: {hit}"
                    ))
                })?
                .to_string();
            let score = object.get("_score").and_then(Value::as_f64).unwrap_or(0.0);
            let source = object
                .get("_source")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            parsed.push(Hit { id, score, source });
        }
        Ok(parsed)
    }

    /// `POST /{index}/_delete_by_query?refresh=true`, returning how many
    /// documents Elasticsearch deleted.
    pub(crate) async fn delete_by_query(&self, index: &str, query: &Value) -> Result<usize> {
        let operation = format!("delete documents from vector index {index:?}");
        let response = self
            .request(
                reqwest::Method::POST,
                &format!("/{index}/_delete_by_query?refresh=true"),
            )
            .json(&json!({ "query": query }))
            .send()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        if !response.status().is_success() {
            return Err(response_error(response, &operation).await);
        }
        let answer: Value = response
            .json()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        let deleted = answer
            .get("deleted")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                Error::transport(format!(
                    "elasticsearch {operation} response has no `deleted` count"
                ))
            })?;
        usize::try_from(deleted).map_err(|error| {
            Error::transport(format!(
                "elasticsearch {operation} reported an unrepresentable delete count: {error}"
            ))
        })
    }

    /// `GET /_cat/indices/{prefix}__*?format=json&h=index`, returning the
    /// collection names (prefix stripped) in ascending order.
    pub(crate) async fn index_names(&self, prefix: &str) -> Result<Vec<String>> {
        let operation = format!("list vector collections with prefix {prefix:?}");
        let pattern = format!("{prefix}__*");
        let response = self
            .request(
                reqwest::Method::GET,
                &format!("/_cat/indices/{pattern}?format=json&h=index"),
            )
            .send()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        if !response.status().is_success() {
            return Err(response_error(response, &operation).await);
        }
        let answer: Value = response
            .json()
            .await
            .map_err(|error| transport_error(&error, &operation))?;
        let entries = answer.as_array().ok_or_else(|| {
            Error::transport(format!(
                "elasticsearch {operation} response is not a JSON array"
            ))
        })?;
        let marker = format!("{prefix}__");
        let mut names = Vec::new();
        for entry in entries {
            let Some(name) = entry.get("index").and_then(Value::as_str) else {
                continue;
            };
            if let Some(collection) = name.strip_prefix(&marker) {
                names.push(collection.to_string());
            }
        }
        names.sort();
        names.dedup();
        Ok(names)
    }
}

/// Builds the `PUT /{index}` body for `schema`.
///
/// The primary key is always a `keyword` (document ids are strings), a
/// `FLOAT_VECTOR` field is a `dense_vector` with the field's `dim` (or the
/// store's default dimension) and cosine similarity, scalar types map to the
/// matching Elasticsearch number/boolean type, and `JSON`/`ARRAY` fields are
/// enabled objects. The whole [`CollectionSchema`] is stored in
/// `_meta.openjiuwen_schema` so the store's `get_schema`
/// round-trips it (Python's `_collection_metadata__` document is not needed).
pub(crate) fn mapping_body(schema: &CollectionSchema, dimension: usize) -> Result<Value> {
    let mut properties = Map::new();
    for field in &schema.fields {
        properties.insert(field.name.clone(), property(field, dimension));
    }
    let stored = serde_json::to_value(schema).map_err(Error::external)?;
    let dynamic = if schema.enable_dynamic_field {
        "true"
    } else {
        "strict"
    };
    Ok(json!({
        "mappings": {
            "dynamic": dynamic,
            "properties": Value::Object(properties),
            "_meta": { "openjiuwen_schema": stored },
        },
    }))
}

/// The Elasticsearch property of one schema field.
fn property(field: &FieldSchema, dimension: usize) -> Value {
    if field.is_primary {
        return json!({ "type": "keyword" });
    }
    match field.dtype {
        VectorDataType::Varchar => json!({ "type": "keyword" }),
        VectorDataType::FloatVector => json!({
            "type": "dense_vector",
            "dims": field.dim.unwrap_or(dimension),
            "index": true,
            "similarity": similarity_for(STORE_METRIC),
        }),
        VectorDataType::Int64 => json!({ "type": "long" }),
        VectorDataType::Int32 => json!({ "type": "integer" }),
        VectorDataType::Float => json!({ "type": "float" }),
        VectorDataType::Double => json!({ "type": "double" }),
        VectorDataType::Bool => json!({ "type": "boolean" }),
        VectorDataType::Json | VectorDataType::Array => {
            json!({ "type": "object", "enabled": true })
        }
    }
}

/// Rebuilds a [`CollectionSchema`] from an index mapping.
///
/// The schema stored in `_meta.openjiuwen_schema` wins; when the index was not
/// created by this backend the schema is reconstructed from `properties`, the
/// property named `id` (or else the first non-vector property) becomes the
/// primary key, and `dynamic != "strict"` becomes `enable_dynamic_field`.
/// Returns `None` only when the mapping declares no `properties` object.
pub(crate) fn schema_from_mappings(mappings: &Value) -> Option<CollectionSchema> {
    if let Some(stored) = mappings.pointer("/_meta/openjiuwen_schema") {
        if let Ok(schema) = serde_json::from_value::<CollectionSchema>(stored.clone()) {
            return Some(schema);
        }
    }
    let properties = mappings.get("properties")?.as_object()?;
    let mut fields = Vec::with_capacity(properties.len());
    for (name, definition) in properties {
        let Some(dtype) = dtype_from_property(definition) else {
            continue;
        };
        let mut field = FieldSchema::new(name.clone(), dtype);
        if dtype == VectorDataType::FloatVector {
            field.dim = definition
                .get("dims")
                .and_then(Value::as_u64)
                .and_then(|dims| usize::try_from(dims).ok());
        }
        fields.push(field);
    }
    if let Some(primary) = fields.iter_mut().find(|field| field.name == "id") {
        primary.is_primary = true;
    } else if let Some(primary) = fields
        .iter_mut()
        .find(|field| field.dtype != VectorDataType::FloatVector)
    {
        primary.is_primary = true;
    }
    Some(CollectionSchema {
        fields,
        description: None,
        enable_dynamic_field: mappings.get("dynamic").and_then(Value::as_str) != Some("strict"),
    })
}

/// The kernel field type of one Elasticsearch property, if it has an analogue.
fn dtype_from_property(definition: &Value) -> Option<VectorDataType> {
    match definition.get("type").and_then(Value::as_str)? {
        "keyword" | "text" | "constant_keyword" => Some(VectorDataType::Varchar),
        "dense_vector" => Some(VectorDataType::FloatVector),
        "long" => Some(VectorDataType::Int64),
        "integer" | "short" | "byte" => Some(VectorDataType::Int32),
        "float" | "half_float" => Some(VectorDataType::Float),
        "double" => Some(VectorDataType::Double),
        "boolean" => Some(VectorDataType::Bool),
        "object" | "nested" => Some(VectorDataType::Json),
        _ => None,
    }
}

/// A `match_all` query.
pub(crate) fn match_all() -> Value {
    json!({ "match_all": {} })
}

/// A `match_none` query (the translation of a condition that is never true).
pub(crate) fn match_none() -> Value {
    json!({ "match_none": {} })
}

/// Compiles a kernel filter into an Elasticsearch query clause.
///
/// See the [module docs](self) for the supported subset; anything without a
/// faithful Elasticsearch equivalent returns a validation error instead of an
/// approximation.
pub(crate) fn translate_filter(filters: &Value) -> Result<Value> {
    match filters {
        Value::Null => Ok(match_all()),
        Value::Object(entries) => translate_object(entries),
        _ => Ok(match_none()),
    }
}

/// Compiles a filter object: combinators plus implicit/operator field
/// conditions, all combined with AND.
fn translate_object(entries: &Map<String, Value>) -> Result<Value> {
    if entries.is_empty() {
        return Ok(match_all());
    }
    let mut must: Vec<Value> = Vec::new();
    let mut should: Vec<Value> = Vec::new();
    let mut must_not: Vec<Value> = Vec::new();
    for (key, condition) in entries {
        match key.as_str() {
            "$and" => {
                let Some(filters) = condition.as_array() else {
                    return Ok(match_none());
                };
                for filter in filters {
                    must.push(translate_filter(filter)?);
                }
            }
            "$or" => {
                let Some(filters) = condition.as_array() else {
                    return Ok(match_none());
                };
                if filters.is_empty() {
                    return Ok(match_none());
                }
                for filter in filters {
                    should.push(translate_filter(filter)?);
                }
            }
            "$not" => {
                let Value::Object(sub_filter) = condition else {
                    return Ok(match_none());
                };
                must_not.push(translate_object(sub_filter)?);
            }
            field => must.push(translate_field(field, condition)?),
        }
    }
    Ok(combine(must, should, must_not))
}

/// Combines translated clauses into one `bool` query (or `match_all` for an
/// empty conjunction).
fn combine(must: Vec<Value>, should: Vec<Value>, must_not: Vec<Value>) -> Value {
    if must.is_empty() && should.is_empty() && must_not.is_empty() {
        return match_all();
    }
    if must.len() == 1 && should.is_empty() && must_not.is_empty() {
        return must.into_iter().next().unwrap_or_else(match_all);
    }
    let mut query = Map::new();
    if must.is_empty() && !must_not.is_empty() {
        query.insert("must".to_string(), Value::Array(vec![match_all()]));
    }
    if !must.is_empty() {
        query.insert("must".to_string(), Value::Array(must));
    }
    if !should.is_empty() {
        query.insert("should".to_string(), Value::Array(should));
        query.insert("minimum_should_match".to_string(), json!(1));
    }
    if !must_not.is_empty() {
        query.insert("must_not".to_string(), Value::Array(must_not));
    }
    json!({ "bool": query })
}

/// Compiles one `field: condition` entry.
fn translate_field(field: &str, condition: &Value) -> Result<Value> {
    if let Value::Object(operators) = condition {
        if operators.keys().any(|key| key.starts_with('$')) {
            return translate_operators(field, operators);
        }
    }
    equality(field, condition)
}

/// Compiles implicit equality (and `$eq`) on `field`.
fn equality(field: &str, value: &Value) -> Result<Value> {
    match value {
        Value::String(_) | Value::Bool(_) | Value::Number(_) => Ok(term_query(field, value)),
        Value::Array(items) => {
            if items.is_empty() {
                return Ok(match_none());
            }
            for item in items {
                ensure_term_value(field, item)?;
            }
            Ok(terms_query(field, items))
        }
        Value::Object(_) => Err(Error::validation(format!(
            "the elasticsearch vector store cannot translate the filter on field {field:?}: \
             Elasticsearch has no object-equality query; use the in-memory vector store for this filter"
        ))),
        Value::Null => Err(Error::validation(format!(
            "the elasticsearch vector store cannot translate equality to `null` on field {field:?}: \
             Elasticsearch indexes `null` as a missing field; use the in-memory vector store for this filter"
        ))),
    }
}

/// Rejects `term`/`terms` values Elasticsearch cannot index (`null`, objects,
/// and nested arrays), naming the field and the operator that produced them.
fn ensure_term_value(field: &str, value: &Value) -> Result<()> {
    match value {
        Value::String(_) | Value::Bool(_) | Value::Number(_) => Ok(()),
        _ => Err(Error::validation(format!(
            "the elasticsearch vector store cannot translate the filter on field {field:?}: \
             Elasticsearch term queries accept only strings, numbers, and booleans, got {value}; \
             use the in-memory vector store for this filter"
        ))),
    }
}

/// Compiles an operator object: every operator holds (AND).
fn translate_operators(field: &str, operators: &Map<String, Value>) -> Result<Value> {
    let mut clauses: Vec<Value> = Vec::with_capacity(operators.len());
    for (operator, expected) in operators {
        let clause = match operator.as_str() {
            "$eq" => equality(field, expected)?,
            "$ne" => {
                let inner = equality(field, expected)?;
                json!({ "bool": { "must": [match_all()], "must_not": [inner] } })
            }
            "$gt" | "$gte" | "$lt" | "$lte" => {
                let bound = &operator[1..];
                match expected {
                    Value::Number(_) | Value::String(_) => range_query(field, bound, expected),
                    // The kernel never orders booleans, null, arrays, or mixed types.
                    _ => return Ok(match_none()),
                }
            }
            "$in" => {
                let Some(candidates) = expected.as_array() else {
                    return Ok(match_none());
                };
                if candidates.is_empty() {
                    return Ok(match_none());
                }
                for candidate in candidates {
                    ensure_term_value(field, candidate)?;
                }
                terms_query(field, candidates)
            }
            "$contains" => {
                return Err(Error::validation(format!(
                    "the elasticsearch vector store does not support the `$contains` operator on \
                     field {field:?}: Elasticsearch has no faithful containment query; use the \
                     in-memory vector store for this filter"
                )));
            }
            // Unknown operators make their condition false.
            _ => return Ok(match_none()),
        };
        clauses.push(clause);
    }
    Ok(combine(clauses, Vec::new(), Vec::new()))
}

/// A `term` query on `field`.
fn term_query(field: &str, value: &Value) -> Value {
    json!({ "term": { field: value } })
}

/// A `terms` query on `field`.
fn terms_query(field: &str, values: &[Value]) -> Value {
    json!({ "terms": { field: values } })
}

/// A `range` query on `field` with one bound.
fn range_query(field: &str, bound: &str, value: &Value) -> Value {
    json!({ "range": { field: { bound: value } } })
}

/// Maps a non-2xx response onto the kernel error taxonomy.
async fn response_error(response: reqwest::Response, operation: &str) -> Error {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    status_error(status.as_u16(), operation, &body)
}

/// Maps one HTTP status plus response body onto the kernel error taxonomy.
///
/// The message always carries the Elasticsearch `error.reason` (or the raw body)
/// so no server-side explanation is lost, and never carries credentials.
fn status_error(status: u16, operation: &str, body: &str) -> Error {
    let reason = error_reason(body);
    match status {
        404 => Error::not_found(format!(
            "elasticsearch {operation} failed: {reason} (status 404)"
        )),
        400 | 409 => Error::validation(format!(
            "elasticsearch {operation} was rejected: {reason} (status {status})"
        )),
        401 | 403 => Error::validation(format!(
            "elasticsearch {operation} was not authorised (status {status}): check the configured \
             credentials"
        )),
        _ => Error::transport(format!(
            "elasticsearch {operation} failed: {reason} (status {status})"
        )),
    }
}

/// Extracts a human-readable reason from an Elasticsearch error body.
fn error_reason(body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    if let Some(reason) = parsed
        .as_ref()
        .and_then(|value| value.pointer("/error/reason"))
        .and_then(Value::as_str)
    {
        return reason.to_string();
    }
    if let Some(error) = parsed
        .as_ref()
        .and_then(|value| value.get("error"))
        .and_then(Value::as_str)
    {
        return error.to_string();
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        "no response body".to_string()
    } else {
        trimmed.chars().take(512).collect()
    }
}

/// Maps a `reqwest` transport failure; timeouts become [`Error::timeout`].
fn transport_error(error: &reqwest::Error, operation: &str) -> Error {
    if error.is_timeout() {
        Error::timeout(format!("elasticsearch {operation} timed out: {error}"))
    } else {
        Error::transport(format!("elasticsearch {operation} failed: {error}"))
    }
}

/// Maps a per-item `_bulk` error: document/mapping problems are validation
/// errors, everything else (shard failures, cluster state) is a transport
/// error.
fn bulk_item_error(id: &str, reason: &str, error_type: &str, status: u64) -> Error {
    let message = format!(
        "elasticsearch rejected document {id:?}: {reason} (status {status}, type {error_type})"
    );
    match (status, error_type) {
        (400 | 409, _)
        | (
            _,
            "mapper_parsing_exception"
            | "document_parsing_exception"
            | "strict_dynamic_mapping_exception"
            | "version_conflict_engine_exception",
        ) => Error::validation(message),
        _ => Error::transport(message),
    }
}
