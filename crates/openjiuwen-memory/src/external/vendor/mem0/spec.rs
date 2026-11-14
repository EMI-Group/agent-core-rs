//! The mem0 wire shape: paths, request bodies, and response decoding.
//!
//! [`Mem0Spec`] is a [`WireSpec`], so it never sees a base URL, a credential, or a retry
//! schedule — it only builds a [`RestCall`] relative to the endpoint in use and decodes a
//! `2xx` body. The transport in [`super::super::super::rest`] owns everything else.
use chrono::Utc;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpResponse;
use serde_json::{json, Map, Value};

use super::{DISPLAY, ID, MISSING};
use crate::external::provider::ProviderCapabilities;
use crate::external::registry::ProviderConfig;
use crate::external::rest::{
    body_value, field_f32, field_str, results_array, vendor_hit, vendor_record, AuthScheme,
    RestCall, SpecContext, WireSpec,
};
use crate::record::{
    new_memory_id, AddOptions, AddOutcome, MemoryFilter, MemoryRecord, MemoryScope, MemoryType,
    MemoryUpdate, NewMemory, RecallHit, SearchOptions, DEFAULT_ID,
};

/// Path of the memory collection (the trailing slash is part of the platform API).
const MEMORIES_PATH: &str = "/v1/memories/";
/// Path of the semantic search endpoint.
const SEARCH_PATH: &str = "/v1/memories/search/";
/// Path of the platform health probe.
const PING_PATH: &str = "/v1/ping/";
/// Content keys of one mem0 memory row.
const ITEM_CONTENT_KEYS: &[&str] = &["memory", "data"];
/// List page size used when `options["page_size"]` names none (Python's default).
const DEFAULT_PAGE_SIZE: u64 = 100;
/// Largest page mem0 accepts for one listing request.
const MAX_PAGE_SIZE: u64 = 200;
/// Uppercase hex digits used by [`encode_segment`].
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// The static HTTP shape of the mem0 memory service.
pub(super) struct Mem0Spec;

impl WireSpec for Mem0Spec {
    fn id(&self) -> &'static str {
        ID
    }

    fn display(&self) -> &'static str {
        DISPLAY
    }

    fn missing(&self) -> &'static str {
        MISSING
    }

    /// mem0 recalls semantically only; it has no keyword mode, graph, variables, or
    /// message history, and its listing is not page-addressable.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            ..ProviderCapabilities::none()
        }
    }

    /// The mem0 platform authenticates with `Authorization: Token <api_key>`.
    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Token
    }

    /// mem0 has no static vendor headers beyond the transport's JSON ones.
    fn headers(&self, _config: &ProviderConfig) -> Vec<(String, String)> {
        Vec::new()
    }

    fn default_scope(&self, config: &ProviderConfig) -> MemoryScope {
        super::super::scope_from_options(config)
    }

    /// Stores one memory verbatim: one `user` message, `infer: false`.
    ///
    /// The transport calls this once per [`NewMemory`], so the slice always carries one
    /// entry; an empty slice is a caller bug and is reported as validation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when no memory is supplied.
    fn add_call(
        &self,
        cx: &SpecContext<'_>,
        memories: &[NewMemory],
        _options: &AddOptions,
    ) -> Result<RestCall> {
        let memory = memories.first().ok_or_else(|| {
            Error::validation("the mem0 memory provider writes exactly one memory per request")
        })?;
        let mut body = Map::new();
        body.insert(
            "messages".to_owned(),
            json!([{ "role": "user", "content": memory.content.as_str() }]),
        );
        body.insert("user_id".to_owned(), Value::String(cx.user_id().to_owned()));
        if let Some(agent_id) = agent_param(cx.scope) {
            body.insert("agent_id".to_owned(), Value::String(agent_id.to_owned()));
        }
        body.insert("infer".to_owned(), Value::Bool(false));
        Ok(RestCall::post(MEMORIES_PATH).with_body(Value::Object(body)))
    }

    /// Decodes the stored memory, keeping the vendor id when it answers with one.
    ///
    /// A `2xx` with an empty `results` array is mem0's "already stored, nothing added"
    /// answer and is reported as a skip. A `2xx` with no usable body still stored the
    /// memory (the request carried one message verbatim), so it is reported as added.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the body is not JSON.
    fn add_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        memories: &[NewMemory],
    ) -> Result<AddOutcome> {
        let value = body_value(response)?;
        let rows = add_rows(&value);
        let mut outcome = AddOutcome::default();
        if rows.is_empty() {
            if value
                .get("results")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                outcome
                    .skipped
                    .push(new_memory_id(cx.user_id(), Utc::now()));
            } else if let Some(memory) = memories.first() {
                outcome.added.push(stored_record(memory, None, cx.scope));
            }
            return Ok(outcome);
        }
        for row in rows {
            if let Some(memory) = memories.first() {
                outcome
                    .added
                    .push(stored_record(memory, Some(row), cx.scope));
            }
        }
        Ok(outcome)
    }

    /// Searches memories by meaning, scoped to the user (and agent when bound).
    ///
    /// # Errors
    ///
    /// Returns a typed error when the request body cannot be encoded.
    fn search_call(
        &self,
        cx: &SpecContext<'_>,
        query: &str,
        options: &SearchOptions,
    ) -> Result<RestCall> {
        let mut body = Map::new();
        body.insert("query".to_owned(), Value::String(query.to_owned()));
        body.insert("user_id".to_owned(), Value::String(cx.user_id().to_owned()));
        if let Some(agent_id) = agent_param(cx.scope) {
            body.insert("agent_id".to_owned(), Value::String(agent_id.to_owned()));
        }
        body.insert("limit".to_owned(), Value::from(options.top_k));
        if options.threshold > 0.0 {
            body.insert(
                "threshold".to_owned(),
                Value::from(f64::from(options.threshold)),
            );
        }
        Ok(RestCall::post(SEARCH_PATH).with_body(Value::Object(body)))
    }

    /// Decodes scored hits, dropping the ones below the caller's threshold.
    ///
    /// Hits are sorted by the vendor score descending and truncated to `top_k`; the score
    /// is mem0's similarity and is kept as the hit's blended score.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the body is not JSON.
    fn search_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        let value = body_value(response)?;
        let mut hits: Vec<RecallHit> = results_array(&value)
            .iter()
            .map(|row| {
                let record = vendor_record(row, cx.scope, MemoryType::SemanticMemory);
                vendor_hit(record, field_f32(row, &["score"]).unwrap_or(0.0))
            })
            .filter(|hit| options.threshold <= 0.0 || hit.score >= options.threshold)
            .collect();
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(options.top_k);
        Ok(hits)
    }

    /// Reads one memory by id; the id is encoded as a single path segment.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn get_call(&self, _cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
        Ok(RestCall::get(memory_path(id)))
    }

    /// Decodes one memory; an empty body means the record vanished.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the body is not JSON.
    fn get_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _id: &str,
    ) -> Result<Option<MemoryRecord>> {
        let value = body_value(response)?;
        if value.is_null() {
            return Ok(None);
        }
        if let Some(row) = results_array(&value).first() {
            return Ok(Some(vendor_record(
                row,
                cx.scope,
                MemoryType::SemanticMemory,
            )));
        }
        let looks_like_record = value.get("id").is_some_and(|id| !id.is_null())
            || field_str(&value, ITEM_CONTENT_KEYS).is_some();
        if value.is_object() && looks_like_record {
            Ok(Some(vendor_record(
                &value,
                cx.scope,
                MemoryType::SemanticMemory,
            )))
        } else {
            Ok(None)
        }
    }

    /// Patches the memory text and/or its metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the patch carries neither `content` nor
    /// `metadata`: mem0 has no wire field for importance, source, or kind.
    fn update_call(
        &self,
        _cx: &SpecContext<'_>,
        id: &str,
        update: &MemoryUpdate,
    ) -> Result<RestCall> {
        let mut body = Map::new();
        if let Some(content) = &update.content {
            body.insert("text".to_owned(), Value::String(content.clone()));
        }
        if !update.metadata.is_empty() {
            body.insert(
                "metadata".to_owned(),
                Value::Object(update.metadata.clone().into_iter().collect()),
            );
        }
        if body.is_empty() {
            return Err(Error::unsupported(
                "the mem0 memory provider can only update `content` or `metadata`",
            ));
        }
        Ok(RestCall::put(memory_path(id)).with_body(Value::Object(body)))
    }

    /// Decodes the updated record; an empty body yields the patch target itself.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the body is not JSON.
    fn update_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        id: &str,
    ) -> Result<MemoryRecord> {
        let value = body_value(response)?;
        let mut record = if value.is_null() {
            let mut record =
                MemoryRecord::new(cx.scope.clone(), String::new(), MemoryType::SemanticMemory);
            record.updated_at = Utc::now();
            record
        } else {
            vendor_record(&value, cx.scope, MemoryType::SemanticMemory)
        };
        id.clone_into(&mut record.id);
        Ok(record)
    }

    /// Removes one memory by id.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn delete_call(&self, _cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
        Ok(RestCall::delete(memory_path(id)))
    }

    /// Any `2xx` delete answer means the memory is gone; `404` is handled by the transport.
    ///
    /// # Errors
    ///
    /// This implementation never fails.
    fn delete_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _id: &str,
    ) -> Result<bool> {
        Ok(true)
    }

    /// Lists the user's memories one page at a time.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn list_call(&self, cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
        let page_size = cx
            .option("page_size")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_PAGE_SIZE)
            .clamp(1, MAX_PAGE_SIZE);
        let mut call = RestCall::get(MEMORIES_PATH)
            .query("user_id", cx.user_id())
            .query("page", "1")
            .query("page_size", page_size.to_string());
        if let Some(agent_id) = agent_param(cx.scope) {
            call = call.query("agent_id", agent_id);
        }
        Ok(call)
    }

    /// Decodes a listing; an empty body is an empty list.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the body is not JSON.
    fn list_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _filter: &MemoryFilter,
    ) -> Result<Vec<MemoryRecord>> {
        let value = body_value(response)?;
        Ok(results_array(&value)
            .iter()
            .map(|row| vendor_record(row, cx.scope, MemoryType::SemanticMemory))
            .collect())
    }

    /// Probes mem0's documented `GET /v1/ping/` health endpoint.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn health_call(&self, _cx: &SpecContext<'_>) -> Result<RestCall> {
        Ok(RestCall::get(PING_PATH))
    }
}

/// The agent id sent to mem0: present, non-blank, and not the placeholder default.
fn agent_param(scope: &MemoryScope) -> Option<&str> {
    scope
        .agent_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty() && *id != DEFAULT_ID)
}

/// The path of one memory: `/v1/memories/{id}/`, with `id` percent-encoded.
fn memory_path(id: &str) -> String {
    format!("/v1/memories/{}/", encode_segment(id.trim()))
}

/// Percent-encodes `segment` so it can be used as one URL path segment.
///
/// Letters, digits, and the unreserved punctuation of RFC 3986 are kept; every other byte
/// becomes `%XX` with uppercase hex, so an id containing `/`, `?`, or `#` cannot change
/// the shape of the request.
pub(super) fn encode_segment(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        let character = char::from(byte);
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '.' | '_' | '~') {
            encoded.push(character);
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
        }
    }
    encoded
}

/// The rows of a write answer: `results`/a bare array, or a bare object with an id.
fn add_rows(value: &Value) -> Vec<&Value> {
    let rows = results_array(value);
    if !rows.is_empty() {
        return rows.iter().collect();
    }
    if value.get("id").is_some_and(|id| !id.is_null()) {
        return vec![value];
    }
    Vec::new()
}

/// Maps one vendor row (or the input, when there is no row) onto the stored record.
///
/// The vendor id and timestamps win when present; the input content is the fallback so a
/// terse answer never yields an empty record.
fn stored_record(memory: &NewMemory, row: Option<&Value>, scope: &MemoryScope) -> MemoryRecord {
    let mut record = match row {
        Some(row) => vendor_record(row, scope, memory.kind),
        None => MemoryRecord::new(scope.clone(), memory.content.clone(), memory.kind),
    };
    if record.content.trim().is_empty() {
        record.content.clone_from(&memory.content);
    }
    record
}
