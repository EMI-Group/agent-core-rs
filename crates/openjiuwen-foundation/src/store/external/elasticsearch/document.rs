//! `ElasticsearchDocumentStore`: the
//! [`DocumentStore`] contract on top of plain Elasticsearch index documents.
//!
//! One store-level collection is one Elasticsearch index named
//! `{index_prefix}__{collection}` ([`ElasticsearchConfig::index_prefix`]
//! defaults to `"openjiuwen"`), so both Elasticsearch adapters share one naming
//! rule and one identity space. A [`Document`] maps onto one index document: the
//! Elasticsearch `_id` is the document id and `_source` is the flat
//! [`Document::to_value`] view. That view also carries `id`, so the id is both
//! the document key and a field; a `_source` that lacks it is completed from
//! `_id` on read.
//!
//! # Wire contract
//!
//! Every request goes through the shared `ElasticsearchClient`, i.e. through
//! [`openjiuwen_common::http::HttpClient`]; a non-2xx response becomes an
//! [`Error::Transport`](openjiuwen_common::error::Error) naming the status code,
//! the URL and an excerpt of the body (`failure`).
//!
//! - `upsert` writes one `_bulk` request with `{"index":{"_id":…}}` metadata
//!   lines followed by the flat document lines, and reports the number of
//!   documents written. An item-level error surfaces as a transport error naming
//!   the status, the index, the error type and its reason.
//! - `query`, `count` and `delete` compile the filter with
//!   [`super::query::translate`]. When Elasticsearch can express it, the request
//!   is `_search`, `_count` or `_delete_by_query` respectively.
//!
//! # The local fallback
//!
//! A filter the translator cannot express (a dotted path, `$contains`, a
//! malformed operator, …) is never dropped: the store pages a bounded
//! `_search` result window with `{"match_all":{}}` and re-checks every document
//! with [`crate::store::filter`], reusing
//! `run_query` so ordering and paging match the in-memory reference exactly.
//! The fallback therefore sees only that bounded window: up to 10 000 documents
//! (`DEFAULT_QUERY_LIMIT`, Elasticsearch's
//! default `index.max_result_window`). Because the client must first read those
//! documents, the fallback costs one request per page of
//! `FILTER_DELETE_PAGE` (500) documents.
//!
//! # Collection names
//!
//! Collection names are validated twice: as a collection name
//! ([`validate_collection`](crate::store::document)) and as the derived
//! Elasticsearch index name, so an unusable name fails with
//! [`Error::Validation`](openjiuwen_common::error::Error) before any request is
//! sent.

use async_trait::async_trait;
use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};

use super::query;
use super::{
    encode_segment, failure, index_names_from, ElasticsearchClient, ElasticsearchConfig,
    DEFAULT_QUERY_LIMIT, FILTER_DELETE_PAGE,
};
use crate::store::document::{
    matching, run_query, Document, DocumentStore, Query, SortOrder, ID_FIELD,
};

#[cfg(test)]
mod tests;

/// The body that creates an index for this adapter.
///
/// Documents are dynamic, so only the primary key is mapped; every other field
/// is indexed by Elasticsearch's dynamic mapping. The mapping is also what
/// [`ElasticsearchDocumentStore::upsert`] sends when it creates a missing index.
fn index_body() -> Value {
    json!({"mappings": {"properties": {ID_FIELD: {"type": "keyword"}}}})
}

/// `true` when a response body reports that the index already exists.
///
/// Elasticsearch answers a repeated `PUT /{index}` with HTTP 400 and error type
/// `resource_already_exists_exception`; older proxies only echo the phrase.
fn is_already_exists(body: &str) -> bool {
    let lowered = body.to_lowercase();
    lowered.contains("resource_already_exists_exception") || lowered.contains("already exists")
}

/// The `sort` clause of a `_search` request, or `None` without ordering.
///
/// Every criterion keeps its direction and places documents missing the field
/// first for [`SortOrder::Asc`] and last for [`SortOrder::Desc`], which is the
/// order [`run_query`](crate::store::document::run_query) produces. No base
/// order is emitted, so Elasticsearch's own score/relevance order applies when
/// the query has no criteria.
fn sort_clause(order_by: &[(String, SortOrder)]) -> Option<Value> {
    if order_by.is_empty() {
        return None;
    }
    let criteria: Vec<Value> = order_by
        .iter()
        .map(|(field, order)| {
            let (order, missing) = match order {
                SortOrder::Asc => ("asc", "_first"),
                SortOrder::Desc => ("desc", "_last"),
            };
            let mut criterion = Map::new();
            criterion.insert(
                field.clone(),
                json!({"order": order, "missing": missing, "unmapped_type": "keyword"}),
            );
            Value::Object(criterion)
        })
        .collect();
    Some(Value::Array(criteria))
}

/// Writes one compact JSON line of an NDJSON bulk body.
fn push_line(buffer: &mut String, value: &Value) -> Result<()> {
    let line = serde_json::to_string(value).map_err(|err| {
        Error::validation(format!("serializing an Elasticsearch bulk line: {err}"))
    })?;
    buffer.push_str(&line);
    buffer.push('\n');
    Ok(())
}

/// Renders the `_bulk` body that indexes every document: metadata line, source
/// line, per document.
fn bulk_write_body(documents: &[Document]) -> Result<String> {
    let mut body = String::new();
    for document in documents {
        push_line(&mut body, &json!({"index": {"_id": document.id}}))?;
        push_line(&mut body, &document.to_value())?;
    }
    Ok(body)
}

/// Renders the `_bulk` body that deletes `ids`.
fn bulk_delete_body(ids: &[String]) -> Result<String> {
    let mut body = String::new();
    for id in ids {
        push_line(&mut body, &json!({"delete": {"_id": id}}))?;
    }
    Ok(body)
}

/// The first item-level error of a `_bulk` response, as a transport error.
///
/// Elasticsearch answers a bulk request with HTTP 200 even when single items
/// fail, so `errors: true` must be inspected explicitly; the message names the
/// index, the failed item's status, the error type and its reason.
fn bulk_failure(payload: &Value, index: &str) -> Option<Error> {
    if !payload
        .get("errors")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }
    let failure = payload
        .get("items")
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                let action = item.as_object()?.values().next()?;
                (action.get("error").is_some()).then_some(action)
            })
        });
    let Some(action) = failure else {
        return Some(Error::transport(format!(
            "Elasticsearch bulk write to index `{index}` reported errors: {}",
            super::excerpt(&payload.to_string())
        )));
    };
    let status = action.get("status").and_then(Value::as_u64).map_or_else(
        || "<unknown status>".to_string(),
        |status| status.to_string(),
    );
    let error = action.get("error").cloned().unwrap_or(Value::Null);
    let kind = error
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("<unknown type>");
    let reason = error
        .get("reason")
        .and_then(Value::as_str)
        .map_or_else(|| error.to_string(), str::to_string);
    Some(Error::transport(format!(
        "Elasticsearch bulk write to index `{index}` failed with {status}: {kind}: {reason}"
    )))
}

/// Rebuilds a [`Document`] from one document source, defaulting the id.
///
/// `_source` is the flat [`Document::to_value`] view; when it has no `id`
/// member, `fallback_id` (the response's `_id`) supplies it.
fn document_from_source(source: &Value, fallback_id: &str) -> Result<Document> {
    let Some(mut fields) = source.as_object().cloned() else {
        return Err(Error::validation(format!(
            "Elasticsearch returned a non-object document source: {}",
            super::excerpt(&source.to_string())
        )));
    };
    fields
        .entry(ID_FIELD.to_string())
        .or_insert_with(|| Value::String(fallback_id.to_string()));
    Document::from_value(Value::Object(fields))
}

/// Rebuilds a [`Document`] from one search hit or from a `GET /_doc/{id}` body.
fn document_from_body(payload: &Value, fallback_id: &str) -> Result<Document> {
    let id = payload
        .get("_id")
        .and_then(Value::as_str)
        .unwrap_or(fallback_id);
    let source = payload.get("_source").ok_or_else(|| {
        Error::validation(format!(
            "Elasticsearch document response has no `_source`: {}",
            super::excerpt(&payload.to_string())
        ))
    })?;
    document_from_source(source, id)
}

/// The documents of a `_search` response, in the order the server returned them.
fn search_documents(payload: &Value) -> Result<Vec<Document>> {
    let Some(hits) = payload
        .get("hits")
        .and_then(|hits| hits.get("hits"))
        .and_then(Value::as_array)
    else {
        return Ok(Vec::new());
    };
    hits.iter()
        .map(|hit| {
            let id = hit.get("_id").and_then(Value::as_str).ok_or_else(|| {
                Error::validation(format!(
                    "Elasticsearch search hit is missing its `_id`: {}",
                    super::excerpt(&hit.to_string())
                ))
            })?;
            document_from_body(hit, id)
        })
        .collect()
}

/// The `usize` value of a numeric member of a response, `0` when absent.
fn count_of(payload: &Value, member: &str) -> usize {
    payload.get(member).and_then(Value::as_u64).unwrap_or(0) as usize
}

/// An Elasticsearch-backed [`DocumentStore`].
///
/// The store is cheap to clone (both the HTTP client and the configuration are
/// shared) and every method takes `&self`, so one instance can serve an
/// `Arc<dyn DocumentStore>`.
#[derive(Debug, Clone)]
pub struct ElasticsearchDocumentStore {
    client: ElasticsearchClient,
}

impl ElasticsearchDocumentStore {
    /// Builds a store from `config` with a default-tuned HTTP client.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint or index prefix, and propagates the HTTP client's configuration
    /// errors.
    pub fn new(config: ElasticsearchConfig) -> Result<Self> {
        Ok(Self {
            client: ElasticsearchClient::new(config)?,
        })
    }

    /// Builds a store that reuses `http`, for callers with their own tuning.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint or index prefix.
    pub fn with_client(http: HttpClient, config: ElasticsearchConfig) -> Result<Self> {
        Ok(Self {
            client: ElasticsearchClient::with_client(http, config)?,
        })
    }

    /// The configuration this store was built from.
    #[must_use]
    pub fn config(&self) -> &ElasticsearchConfig {
        self.client.config()
    }

    /// Creates `index` unless it already exists; a 400 "already exists" is fine.
    async fn ensure_index(&self, index: &str) -> Result<()> {
        let path = self.client.path(index, "");
        let response = self
            .client
            .send_raw(HttpRequest::head(self.client.url(&path)))
            .await?;
        match response.status {
            404 => {}
            200..=299 => return Ok(()),
            _ => return Err(failure(&response, "checking index")),
        }
        self.create_index(&path).await
    }

    /// Sends `PUT {index}` with the permissive mapping of [`index_body`].
    async fn create_index(&self, path: &str) -> Result<()> {
        let response = self
            .client
            .send_raw(HttpRequest::put(self.client.url(path)).json(&index_body())?)
            .await?;
        if response.is_success() || (response.status == 400 && is_already_exists(&response.text()))
        {
            return Ok(());
        }
        Err(failure(&response, "creating index"))
    }

    /// Sends one `POST {index}/_bulk` request with an NDJSON `body`.
    async fn send_bulk(&self, index: &str, body: String, context: &str) -> Result<Value> {
        let path = self.client.path(index, "/_bulk");
        let request = HttpRequest::post(self.client.url(&path))
            .text(body)
            .header("content-type", "application/x-ndjson");
        let response = self.client.send(request, context).await?;
        response.json()
    }

    /// Reads the bounded result window the local fallbacks operate on.
    ///
    /// Pages `{"match_all":{}}` with `{"sort":["_doc"]}` until a short page, a
    /// page without fresh ids, or [`DEFAULT_QUERY_LIMIT`] documents. A missing
    /// index (404) means "no documents".
    async fn bounded_documents(&self, index: &str) -> Result<Vec<Document>> {
        let path = self.client.path(index, "/_search");
        let mut collected: Vec<Document> = Vec::new();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut offset = 0usize;
        while collected.len() < DEFAULT_QUERY_LIMIT {
            let size = FILTER_DELETE_PAGE.min(DEFAULT_QUERY_LIMIT - collected.len());
            let body = json!({
                "query": {"match_all": {}},
                "_source": true,
                "from": offset,
                "size": size,
                "sort": ["_doc"],
            });
            let response = self
                .client
                .send_raw(HttpRequest::post(self.client.url(&path)).json(&body)?)
                .await?;
            match response.status {
                404 => return Ok(Vec::new()),
                200..=299 => {}
                _ => return Err(failure(&response, "scanning documents")),
            }
            let page = search_documents(&response.json()?)?;
            let page_len = page.len();
            let mut fresh = 0usize;
            for document in page {
                if seen.insert(document.id.clone()) {
                    collected.push(document);
                    fresh += 1;
                }
            }
            if page_len < size || fresh == 0 {
                break;
            }
            offset += page_len;
        }
        collected.truncate(DEFAULT_QUERY_LIMIT);
        Ok(collected)
    }

    /// Reads a bounded window and keeps the documents whose flat view matches
    /// `filter`; the shared id-collection step of the local fallbacks.
    async fn matching_documents(&self, index: &str, filter: &Value) -> Result<Vec<Document>> {
        let documents = self.bounded_documents(index).await?;
        Ok(matching(&documents, filter).into_iter().cloned().collect())
    }
}

#[async_trait]
impl DocumentStore for ElasticsearchDocumentStore {
    async fn create_collection(&self, collection: &str) -> Result<()> {
        let index = self.client.index_name(collection)?;
        let path = self.client.path(&index, "");
        let response = self
            .client
            .send_raw(HttpRequest::put(self.client.url(&path)).json(&index_body())?)
            .await?;
        if response.is_success() || (response.status == 400 && is_already_exists(&response.text()))
        {
            return Ok(());
        }
        Err(failure(&response, "creating index"))
    }

    async fn list_collections(&self) -> Result<Vec<String>> {
        let response = self
            .client
            .send_raw(HttpRequest::get(
                self.client.url(&self.client.indexes_path()),
            ))
            .await?;
        match response.status {
            404 => return Ok(Vec::new()),
            200..=299 => {}
            _ => return Err(failure(&response, "listing indexes")),
        }
        let mut names: Vec<String> = index_names_from(&response.json()?)
            .iter()
            .filter_map(|index| self.client.collection_from_index(index))
            .collect();
        names.sort();
        names.dedup();
        Ok(names)
    }

    async fn drop_collection(&self, collection: &str) -> Result<()> {
        let index = self.client.index_name(collection)?;
        let path = self.client.path(&index, "");
        let response = self
            .client
            .send_raw(HttpRequest::delete(self.client.url(&path)))
            .await?;
        if response.is_success() || response.status == 404 {
            return Ok(());
        }
        Err(failure(&response, "deleting index"))
    }

    async fn upsert(&self, collection: &str, documents: Vec<Document>) -> Result<usize> {
        let index = self.client.index_name(collection)?;
        if documents.is_empty() {
            return Ok(0);
        }
        self.ensure_index(&index).await?;
        let payload = self
            .send_bulk(&index, bulk_write_body(&documents)?, "writing documents")
            .await?;
        if let Some(error) = bulk_failure(&payload, &index) {
            return Err(error);
        }
        Ok(documents.len())
    }

    async fn get(&self, collection: &str, id: &str) -> Result<Option<Document>> {
        let index = self.client.index_name(collection)?;
        let path = self
            .client
            .path(&index, &format!("/_doc/{}", encode_segment(id)));
        let response = self
            .client
            .send_raw(HttpRequest::get(self.client.url(&path)))
            .await?;
        match response.status {
            404 => Ok(None),
            200..=299 => {
                let payload: Value = response.json()?;
                document_from_body(&payload, id).map(Some)
            }
            _ => Err(failure(&response, "reading document")),
        }
    }

    async fn query(&self, collection: &str, query_spec: &Query) -> Result<Vec<Document>> {
        let index = self.client.index_name(collection)?;
        let Some(translated) = query::translate(&query_spec.filter) else {
            let documents = self.bounded_documents(&index).await?;
            return Ok(run_query(&documents, query_spec)
                .into_iter()
                .cloned()
                .collect());
        };
        let mut body = json!({
            "query": translated,
            "from": query_spec.offset,
            "size": query_spec.limit.unwrap_or(DEFAULT_QUERY_LIMIT),
            "track_total_hits": true,
        });
        if let Some(sort) = sort_clause(&query_spec.order_by) {
            if let Some(entries) = body.as_object_mut() {
                entries.insert("sort".to_string(), sort);
            }
        }
        let path = self.client.path(&index, "/_search");
        let response = self
            .client
            .send_raw(HttpRequest::post(self.client.url(&path)).json(&body)?)
            .await?;
        match response.status {
            404 => Ok(Vec::new()),
            200..=299 => search_documents(&response.json()?),
            _ => Err(failure(&response, "querying documents")),
        }
    }

    async fn count(&self, collection: &str, query_spec: &Query) -> Result<usize> {
        let index = self.client.index_name(collection)?;
        let Some(translated) = query::translate(&query_spec.filter) else {
            return Ok(self
                .matching_documents(&index, &query_spec.filter)
                .await?
                .len());
        };
        let path = self.client.path(&index, "/_count");
        let body = json!({"query": translated});
        let response = self
            .client
            .send_raw(HttpRequest::post(self.client.url(&path)).json(&body)?)
            .await?;
        match response.status {
            404 => Ok(0),
            200..=299 => Ok(count_of(&response.json()?, "count")),
            _ => Err(failure(&response, "counting documents")),
        }
    }

    async fn delete(&self, collection: &str, query_spec: &Query) -> Result<usize> {
        let index = self.client.index_name(collection)?;
        let Some(translated) = query::translate(&query_spec.filter) else {
            let matched = self.matching_documents(&index, &query_spec.filter).await?;
            if matched.is_empty() {
                return Ok(0);
            }
            let ids: Vec<String> = matched.into_iter().map(|document| document.id).collect();
            let payload = self
                .send_bulk(&index, bulk_delete_body(&ids)?, "deleting documents")
                .await?;
            let deleted = payload
                .get("items")
                .and_then(Value::as_array)
                .map_or(0, |items| {
                    items
                        .iter()
                        .filter(|item| {
                            item.get("delete")
                                .and_then(|action| action.get("result"))
                                .and_then(Value::as_str)
                                == Some("deleted")
                        })
                        .count()
                });
            return Ok(deleted);
        };
        let path = self.client.path(&index, "/_delete_by_query");
        let body = json!({"query": translated});
        let response = self
            .client
            .send_raw(HttpRequest::post(self.client.url(&path)).json(&body)?)
            .await?;
        match response.status {
            404 => Ok(0),
            200..=299 => Ok(count_of(&response.json()?, "deleted")),
            _ => Err(failure(&response, "deleting documents")),
        }
    }
}
