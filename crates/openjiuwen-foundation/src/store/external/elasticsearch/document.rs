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
//! # Pagination
//!
//! `query` reads with `search_after`: every request asks for at most
//! `SEARCH_PAGE_SIZE` (500) documents, sorted by the caller's `order_by`
//! criteria plus the keyword primary key as the stable tiebreaker, and
//! continues from the `sort` values of the previous page. `offset` and `limit`
//! are then applied to the assembled documents, so a deep page costs one
//! request per page instead of Elasticsearch's `from`/`size` deep-paging.
//!
//! A read that needs more than `MAX_SCAN_DOCUMENTS` (10 000) documents fails
//! with [`Error::Unsupported`](openjiuwen_common::error::Error) after a
//! `tracing::warn!` naming the cap and the filter — it never returns a silently
//! truncated list. Pass an explicit `limit` to read a small window of a large
//! collection.
//!
//! # The local fallback
//!
//! A filter the translator cannot express (a dotted path, `$contains`, a
//! malformed operator, …) is never dropped: the store logs one
//! `tracing::warn!` and then scans the index with `{"match_all":{}}` and the
//! same `search_after` paging, re-checks every document with
//! [`crate::store::filter`], and reuses `run_query` so ordering and paging match
//! the in-memory reference exactly. The fallback shares the `MAX_SCAN_DOCUMENTS`
//! cap: a collection larger than the cap fails with the same typed error
//! instead of reporting the matches of a truncated window. Because the client
//! must first read those documents, the fallback costs one request per page of
//! `SEARCH_PAGE_SIZE` (500) documents.
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
    encode_segment, failure, index_names_from, stable_sort, ElasticsearchClient,
    ElasticsearchConfig, SearchScan, MAX_SCAN_DOCUMENTS,
};
use crate::store::document::{matching, run_query, Document, DocumentStore, Query, ID_FIELD};

#[cfg(test)]
mod tests;

/// `tracing` target of this adapter's warnings.
const LOG_TARGET: &str = "openjiuwen_foundation::store::external::elasticsearch::document";

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

/// Rebuilds a [`Document`] from one `_source` object, defaulting the id.
///
/// `_source` is the flat [`Document::to_value`] view; when it has no `id`
/// member, `fallback_id` (the response's `_id`) supplies it.
fn document_from_fields(mut fields: Map<String, Value>, fallback_id: &str) -> Result<Document> {
    fields
        .entry(ID_FIELD.to_string())
        .or_insert_with(|| Value::String(fallback_id.to_string()));
    Document::from_value(Value::Object(fields))
}

/// Rebuilds a [`Document`] from one document source, defaulting the id.
fn document_from_source(source: &Value, fallback_id: &str) -> Result<Document> {
    let Some(fields) = source.as_object().cloned() else {
        return Err(Error::validation(format!(
            "Elasticsearch returned a non-object document source: {}",
            super::excerpt(&source.to_string())
        )));
    };
    document_from_fields(fields, fallback_id)
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

    /// Reads the documents of `query` with `search_after` pagination.
    ///
    /// Every request asks for at most [`SEARCH_PAGE_SIZE`](super::SEARCH_PAGE_SIZE)
    /// documents sorted by `sort` and continues from the `sort` values of the
    /// previous page (the shared [`SearchScan`]), so a deep page costs one
    /// request per page instead of deep `from`/`size` paging. `needed` stops the
    /// read once that many documents were assembled (`offset + limit` of a
    /// query); `None` reads until the index is exhausted. A missing index (404)
    /// reads as no documents.
    ///
    /// # Errors
    ///
    /// Propagates the paging errors of [`ElasticsearchClient::scan`] — the
    /// non-progress guards and the
    /// [`MAX_SCAN_DOCUMENTS`](super::MAX_SCAN_DOCUMENTS) cap — and the document
    /// parsing errors of a malformed hit.
    async fn paged_documents(
        &self,
        index: &str,
        query: &Value,
        sort: &Value,
        needed: Option<usize>,
        filter: &Value,
    ) -> Result<Vec<Document>> {
        let mut documents: Vec<Document> = Vec::new();
        self.client
            .scan(
                SearchScan {
                    index,
                    query: query.clone(),
                    sort: sort.clone(),
                    needed,
                    filter,
                },
                |page| {
                    for hit in page {
                        documents.push(document_from_fields(hit.source, &hit.id)?);
                    }
                    Ok(())
                },
            )
            .await?;
        Ok(documents)
    }

    /// Reads every document of `index` whose flat view matches `filter`; the
    /// shared step of the local fallbacks.
    ///
    /// The scan pages the whole index (see [`Self::paged_documents`]), so it
    /// shares the [`MAX_SCAN_DOCUMENTS`] cap and fails instead of reporting the
    /// matches of a truncated window.
    ///
    /// # Errors
    ///
    /// Propagates the paging errors of [`Self::paged_documents`].
    async fn matching_documents(&self, index: &str, filter: &Value) -> Result<Vec<Document>> {
        tracing::warn!(
            target: LOG_TARGET,
            index = %index,
            filter = %filter,
            cap = MAX_SCAN_DOCUMENTS,
            "a filter Elasticsearch cannot express falls back to a local scan",
        );
        // Scanning in primary-key order keeps tie-breaking deterministic when
        // `run_query` re-sorts the matches.
        let documents = self
            .paged_documents(
                index,
                &json!({"match_all": {}}),
                &stable_sort(&[], ID_FIELD),
                None,
                filter,
            )
            .await?;
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
            let documents = self.matching_documents(&index, &query_spec.filter).await?;
            return Ok(run_query(&documents, query_spec)
                .into_iter()
                .cloned()
                .collect());
        };
        let documents = self
            .paged_documents(
                &index,
                &translated,
                &stable_sort(&query_spec.order_by, ID_FIELD),
                query_spec
                    .limit
                    .map(|limit| query_spec.offset.saturating_add(limit)),
                &query_spec.filter,
            )
            .await?;
        Ok(documents
            .into_iter()
            .skip(query_spec.offset)
            .take(query_spec.limit.unwrap_or(usize::MAX))
            .collect())
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
