//! The `OpenViking` wire shape: paths, request bodies, and response decoding.
//!
//! [`OpenVikingSpec`] is a [`WireSpec`], so it never sees a base URL, a credential, or a
//! retry schedule — it only builds a [`RestCall`] relative to the endpoint in use and
//! decodes a `2xx` body. The transport in [`super::super::super::rest`] owns validation,
//! authentication, retries, failover, and status mapping.
//!
//! # Documented endpoints
//!
//! | Operation | Request | Answer |
//! | --- | --- | --- |
//! | search | `POST /api/v1/search/find` with `query` and `top_k` | `{"result": {"memories": [], "resources": [], "skills": [], "total": n}}` |
//! | add | `POST /api/v1/sessions/{session_id}/messages` with one `[Remember — <category>] <content>` text part | ignored |
//! | get | `GET /api/v1/content/read?uri=<uri>` | `{"result": "<text>"}` or `{"result": {"content": "<text>"}}` |
//! | list | `GET /api/v1/fs/ls?uri=<uri>` | `{"result": [{"rel_path", "name", "uri", "isDir", "abstract"}]}` |
//! | health | `GET /health` | any `2xx` |
//!
//! `OpenViking` documents no memory-delete, memory-update, or bulk-clear endpoint, so
//! those three operations stay typed [`Error::Unsupported`] (see [`no_operation`]); no
//! endpoint is invented for them.
//!
//! # Two decoding rules
//!
//! - A memory written through a session carries no memory id at write time; `OpenViking`
//!   assigns uris when the session is committed, so [`WireSpec::add_result`] reports a
//!   client-generated id (see [`crate::record::new_memory_id`]).
//! - `OpenViking` has no memory kinds; every record decoded from a vendor answer is
//!   [`MemoryType::SemanticMemory`].
use chrono::Utc;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpResponse;
use serde_json::{json, Map, Value};

use super::{DISPLAY, ID, MISSING};
use crate::external::provider::ProviderCapabilities;
use crate::external::registry::ProviderConfig;
use crate::external::rest::{
    body_value, field_f32, field_str, vendor_hit, vendor_record, AuthScheme, RestCall, SpecContext,
    WireSpec,
};
use crate::record::{
    new_memory_id, AddOptions, AddOutcome, MemoryFilter, MemoryRecord, MemoryScope, MemoryType,
    MemoryUpdate, NewMemory, RecallHit, SearchOptions,
};

/// Path of the cross-list search endpoint.
pub(super) const FIND_PATH: &str = "/api/v1/search/find";
/// Path of the raw-content read endpoint.
pub(super) const READ_PATH: &str = "/api/v1/content/read";
/// Path of the abstract-only content endpoint.
pub(super) const ABSTRACT_PATH: &str = "/api/v1/content/abstract";
/// Path of the overview content endpoint.
pub(super) const OVERVIEW_PATH: &str = "/api/v1/content/overview";
/// Path of the filesystem listing endpoint.
pub(super) const LS_PATH: &str = "/api/v1/fs/ls";
/// Path of the recursive filesystem listing endpoint.
pub(super) const TREE_PATH: &str = "/api/v1/fs/tree";
/// Path of the single-entry metadata endpoint.
pub(super) const STAT_PATH: &str = "/api/v1/fs/stat";
/// Path of the resource-indexing endpoint.
pub(super) const RESOURCES_PATH: &str = "/api/v1/resources";
/// Prefix of the per-session endpoints.
const SESSIONS_PATH: &str = "/api/v1/sessions";
/// Path of the health probe.
const HEALTH_PATH: &str = "/health";
/// Root URI listed when neither `options["root_uri"]` nor `options["uri"]` names one.
pub(super) const DEFAULT_ROOT_URI: &str = "viking://";
/// Memory category used when `metadata["category"]` names none.
const DEFAULT_CATEGORY: &str = "preference";
/// Longest content a read returns, mirroring the Python tool's ceiling.
const MAX_CONTENT_CHARS: usize = 8000;
/// Note appended to content shortened to [`MAX_CONTENT_CHARS`].
const TRUNCATED_NOTE: &str = "\n\n[... truncated, use a more specific URI or abstract level]";
/// The context lists of a find answer, with the singular type the rail reports.
///
/// The order is the order the Python adapter reads them in: memories first, then
/// resources, then skills.
pub(super) const FIND_LISTS: [(&str, &str); 3] = [
    ("memories", "memory"),
    ("resources", "resource"),
    ("skills", "skill"),
];

/// The static HTTP shape of the `OpenViking` context database.
pub(super) struct OpenVikingSpec;

impl WireSpec for OpenVikingSpec {
    fn id(&self) -> &'static str {
        ID
    }

    fn display(&self) -> &'static str {
        DISPLAY
    }

    fn missing(&self) -> &'static str {
        MISSING
    }

    /// `OpenViking` performs semantic recall only; it has no keyword mode, graph, variables,
    /// message history, or page-addressable listing.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            ..ProviderCapabilities::none()
        }
    }

    /// `OpenViking` authenticates with a raw API key in `X-API-Key`.
    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Header {
            name: "X-API-Key".to_owned(),
        }
    }

    /// `OpenViking` routes a request by account, user, and agent headers.
    ///
    /// The account falls back to `default`, the user to the configured
    /// `options["user_id"]` (or the default scope's user), and the agent to `hermes` when
    /// `options["agent_id"]` names none.
    fn headers(&self, config: &ProviderConfig) -> Vec<(String, String)> {
        let account = super::super::option_text(config, "account")
            .unwrap_or_else(|| super::DEFAULT_ACCOUNT.to_owned());
        let user = super::super::option_text(config, "user_id")
            .unwrap_or_else(|| super::super::scope_from_options(config).user_id);
        let agent = super::super::option_text(config, "agent_id")
            .unwrap_or_else(|| super::DEFAULT_AGENT.to_owned());
        vec![
            ("X-OpenViking-Account".to_owned(), account),
            ("X-OpenViking-User".to_owned(), user),
            ("X-OpenViking-Agent".to_owned(), agent),
        ]
    }

    fn default_scope(&self, config: &ProviderConfig) -> MemoryScope {
        super::super::scope_from_options(config)
    }

    /// Recalls context through the documented find endpoint.
    ///
    /// One `POST` returns every context list at once; [`WireSpec::search_result`] merges
    /// them client-side.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn search_call(
        &self,
        _cx: &SpecContext<'_>,
        query: &str,
        options: &SearchOptions,
    ) -> Result<RestCall> {
        Ok(RestCall::post(FIND_PATH).with_body(json!({
            "query": query,
            "top_k": options.top_k,
        })))
    }

    /// Decodes every context list into scored hits, best first.
    ///
    /// Hits are sorted by the vendor score descending, dropped below `options.threshold`
    /// when it is positive, and truncated to `options.top_k`. A `null` `result`, a
    /// non-object `result`, or a missing list decodes as "no hits" instead of failing, so
    /// an empty knowledge base is not an error.
    ///
    /// # Errors
    ///
    /// Returns a typed transport error when the body is not JSON.
    fn search_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        let value = body_value(response)?;
        let mut hits: Vec<RecallHit> = find_items(&value)
            .into_iter()
            .map(|(_kind, item)| vendor_hit(item_record(item, cx.scope), item_score(item)))
            .filter(|hit| options.threshold <= 0.0 || hit.score >= options.threshold)
            .collect();
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(options.top_k);
        Ok(hits)
    }

    /// Writes one memory through a session message.
    ///
    /// The text is `[Remember — <category>] <content>`, where the category comes from
    /// `metadata["category"]` and falls back to `preference`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when no memory is supplied or when neither
    /// `MemoryScope::session_id` nor `options["session_id"]` names the session the message
    /// belongs to.
    fn add_call(
        &self,
        cx: &SpecContext<'_>,
        memories: &[NewMemory],
        _options: &AddOptions,
    ) -> Result<RestCall> {
        let memory = memories.first().ok_or_else(|| {
            Error::validation(
                "the OpenViking memory provider writes exactly one memory per request",
            )
        })?;
        let session_id = add_session(cx).ok_or_else(|| session_required("store a memory"))?;
        let text = format!("[Remember — {}] {}", category(memory), memory.content);
        Ok(RestCall::post(messages_path(&session_id)).with_body(json!({
            "role": "user",
            "parts": [{ "type": "text", "text": text }],
        })))
    }

    /// Reports the accepted message as an added memory.
    ///
    /// `OpenViking` assigns uris (and extracts memories) when the session is committed, so
    /// the answer carries no memory id: the record uses a client-generated id and the
    /// submitted content, kind, importance, metadata, and source.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when no memory is supplied.
    fn add_result(
        &self,
        cx: &SpecContext<'_>,
        _response: &HttpResponse,
        memories: &[NewMemory],
    ) -> Result<AddOutcome> {
        let memory = memories.first().ok_or_else(|| {
            Error::validation(
                "the OpenViking memory provider writes exactly one memory per request",
            )
        })?;
        let mut record = MemoryRecord::new(cx.scope.clone(), memory.content.clone(), memory.kind)
            .with_id(new_memory_id(cx.user_id(), Utc::now()));
        if let Some(importance) = memory.importance {
            record = record.with_importance(importance);
        }
        record.metadata.clone_from(&memory.metadata);
        if let Some(source) = &memory.source {
            record.source = Some(source.clone());
        }
        Ok(AddOutcome {
            added: vec![record],
            ..AddOutcome::default()
        })
    }

    /// Reads the content at one `viking://` uri.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn get_call(&self, _cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
        Ok(RestCall::get(READ_PATH).query("uri", id.trim()))
    }

    /// Decodes the read answer into a record whose id is the requested uri.
    ///
    /// Content longer than 8000 characters is truncated — the Python tool does the same —
    /// so one read cannot flood the caller's context window. A blank `result` is reported
    /// as "no record".
    ///
    /// # Errors
    ///
    /// Returns a typed transport error when the body is not JSON.
    fn get_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        id: &str,
    ) -> Result<Option<MemoryRecord>> {
        let value = body_value(response)?;
        let Some(text) = read_text(&value) else {
            return Ok(None);
        };
        let record = MemoryRecord::new(
            cx.scope.clone(),
            truncate_content(text),
            MemoryType::SemanticMemory,
        )
        .with_id(id.trim().to_owned());
        Ok(Some(record))
    }

    /// `OpenViking` exposes no memory-update endpoint.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] naming the missing endpoint.
    fn update_call(
        &self,
        _cx: &SpecContext<'_>,
        _id: &str,
        _update: &MemoryUpdate,
    ) -> Result<RestCall> {
        Err(no_operation("update", NO_UPDATE))
    }

    /// `OpenViking` exposes no memory-delete endpoint.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] naming the missing endpoint.
    fn delete_call(&self, _cx: &SpecContext<'_>, _id: &str) -> Result<RestCall> {
        Err(no_operation("delete", NO_DELETE))
    }

    /// `OpenViking` exposes no bulk-clear endpoint.
    ///
    /// The transport reacts to this `Unsupported` by listing the filter and deleting each
    /// id individually, and every delete is itself `Unsupported`, so a clear removes
    /// nothing and reports the typed delete error.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] naming the missing endpoint.
    fn clear_call(&self, _cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
        Err(no_operation("clear", NO_CLEAR))
    }

    /// Lists one directory of the knowledge tree.
    ///
    /// The listed uri is `options["root_uri"]`, then `options["uri"]`, then the `viking://`
    /// root.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn list_call(&self, cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
        Ok(RestCall::get(LS_PATH).query("uri", list_root(cx)))
    }

    /// Decodes the directory listing, skipping every directory entry.
    ///
    /// Each non-directory entry becomes a record whose id is its `uri` and whose content is
    /// its `abstract`, `rel_path`, or `name` (in that order). A non-list `result` or an
    /// absent `result` decodes as an empty listing.
    ///
    /// # Errors
    ///
    /// Returns a typed transport error when the body is not JSON.
    fn list_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _filter: &MemoryFilter,
    ) -> Result<Vec<MemoryRecord>> {
        let value = body_value(response)?;
        Ok(list_entries(&value)
            .iter()
            .filter(|entry| !is_directory(entry))
            .map(|entry| listing_record(entry, cx.scope))
            .collect())
    }

    /// Probes the documented `GET /health` endpoint.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn health_call(&self, _cx: &SpecContext<'_>) -> Result<RestCall> {
        Ok(RestCall::get(HEALTH_PATH))
    }
}

/// The dependency name and absence message of an unconfigured handle.
///
/// The rail sends its own requests (see [`super::send_rail`]), so it needs the same typed
/// error the transport reports for a handle without an `endpoint`.
pub(super) fn unconfigured() -> Error {
    Error::unsupported(format!(
        "the {DISPLAY} memory provider requires {MISSING}, which this build does not provide: no `endpoint` is configured"
    ))
}

/// Why an update cannot be sent.
const NO_UPDATE: &str = "the OpenViking API exposes no memory-update endpoint (a memory is rewritten by sending a new message to POST /api/v1/sessions/{session_id}/messages and committing the session)";
/// Why a delete cannot be sent.
const NO_DELETE: &str = "the OpenViking API exposes no memory-delete endpoint (writes are committed through POST /api/v1/sessions/{session_id}/commit)";
/// Why a bulk clear cannot be sent.
const NO_CLEAR: &str = "the OpenViking API exposes no bulk-clear endpoint (writes are committed through POST /api/v1/sessions/{session_id}/commit)";

/// The typed [`Error::Unsupported`] of an operation `OpenViking` does not document.
pub(super) fn no_operation(operation: &str, detail: &str) -> Error {
    Error::unsupported(format!(
        "the {DISPLAY} memory provider has no {operation} operation: {detail}"
    ))
}

/// The typed [`Error::Validation`] of an operation that needs a session id but has none.
pub(super) fn session_required(purpose: &str) -> Error {
    Error::validation(format!(
        "the {DISPLAY} memory provider requires a session id (MemoryScope::session_id or options[\"session_id\"]) to {purpose}"
    ))
}

/// The message path of one session.
///
/// The id is interpolated verbatim, exactly as the Python adapter builds it.
pub(super) fn messages_path(session_id: &str) -> String {
    format!("{SESSIONS_PATH}/{session_id}/messages")
}

/// The commit path of one session.
pub(super) fn commit_path(session_id: &str) -> String {
    format!("{SESSIONS_PATH}/{session_id}/commit")
}

/// The session `add` writes into: the scope's session, else `options["session_id"]`.
fn add_session(cx: &SpecContext<'_>) -> Option<String> {
    let scope_session = cx
        .session_id()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned);
    scope_session.or_else(|| {
        cx.option("session_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(ToOwned::to_owned)
    })
}

/// The category of one write: `metadata["category"]`, else `preference`.
fn category(memory: &NewMemory) -> &str {
    memory
        .metadata
        .get("category")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|category| !category.is_empty())
        .unwrap_or(DEFAULT_CATEGORY)
}

/// The uri a listing reads: `options["root_uri"]`, `options["uri"]`, else `viking://`.
pub(super) fn list_root(cx: &SpecContext<'_>) -> String {
    ["root_uri", "uri"]
        .into_iter()
        .find_map(|key| {
            cx.option(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|uri| !uri.is_empty())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| DEFAULT_ROOT_URI.to_owned())
}

/// The `result` value of an answer.
pub(super) fn result_ref(value: &Value) -> Option<&Value> {
    value.get("result").filter(|result| !result.is_null())
}

/// One context list of a find answer, or an empty slice when it is absent.
pub(super) fn list_items<'a>(value: &'a Value, list: &str) -> &'a [Value] {
    result_ref(value)
        .and_then(|result| result.get(list))
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// Every context item of a find answer, with the singular type of its list.
pub(super) fn find_items(value: &Value) -> Vec<(&'static str, &Value)> {
    let mut items = Vec::new();
    for (list, kind) in FIND_LISTS {
        for item in list_items(value, list) {
            items.push((kind, item));
        }
    }
    items
}

/// The entry list of a listing answer, or an empty slice when it is absent.
pub(super) fn list_entries(value: &Value) -> &[Value] {
    result_ref(value)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// The reported `total` of a find answer.
pub(super) fn answer_total(value: &Value) -> Option<u64> {
    result_ref(value)
        .and_then(|result| result.get("total"))
        .and_then(Value::as_u64)
}

/// The score of one context item (`0.0` when absent or non-finite).
pub(super) fn item_score(item: &Value) -> f32 {
    field_f32(item, &["score"]).unwrap_or(0.0)
}

/// Maps one find item onto a record: id from `uri`, content from `abstract`.
pub(super) fn item_record(item: &Value, scope: &MemoryScope) -> MemoryRecord {
    vendor_record(
        &normalized_row(item, &["uri"], &["abstract"]),
        scope,
        MemoryType::SemanticMemory,
    )
}

/// Maps one listing entry onto a record.
///
/// The id is the entry's `uri`; the content is its `abstract`, else its `rel_path`, else
/// its `name`, so a directory entry that carries no abstract still lists readably (it is
/// skipped by [`WireSpec::list_result`] anyway).
pub(super) fn listing_record(entry: &Value, scope: &MemoryScope) -> MemoryRecord {
    vendor_record(
        &normalized_row(entry, &["uri"], &["abstract", "rel_path", "name"]),
        scope,
        MemoryType::SemanticMemory,
    )
}

/// Whether one listing entry is a directory (`isDir`, or its `snake_case` alias).
pub(super) fn is_directory(entry: &Value) -> bool {
    ["isDir", "is_dir"]
        .into_iter()
        .find_map(|key| entry.get(key).and_then(Value::as_bool))
        .unwrap_or(false)
}

/// Rewrites a vendor item as a row [`vendor_record`] understands.
///
/// `vendor_record` reads ids from `id`/`memory_id`/… and content from
/// `memory`/`content`/`text`/`data`, while `OpenViking` names them `uri` and `abstract`,
/// so the aliases are resolved here instead of duplicating its decoding.
fn normalized_row(item: &Value, id_keys: &[&str], content_keys: &[&str]) -> Value {
    let mut row = Map::new();
    if let Some(id) = field_str(item, id_keys) {
        row.insert("id".to_owned(), Value::String(id.to_owned()));
    }
    if let Some(content) = field_str(item, content_keys) {
        row.insert("content".to_owned(), Value::String(content.to_owned()));
    }
    Value::Object(row)
}

/// The text of a read answer: `{"result": "<text>"}` or `{"result": {"content": "<text>"}}`.
///
/// A blank or absent body reads as "no record", so a caller sees `None` rather than an
/// empty record.
fn read_text(value: &Value) -> Option<&str> {
    let result = result_ref(value)?;
    let text = match result {
        Value::String(text) => text.as_str(),
        Value::Object(object) => object.get("content")?.as_str()?,
        _ => return None,
    };
    (!text.trim().is_empty()).then_some(text)
}

/// Truncates `text` to `limit` characters, keeping the short form unchanged.
pub(super) fn truncate_chars(text: &str, limit: usize) -> &str {
    match text.char_indices().nth(limit) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

/// Truncates read content to [`MAX_CONTENT_CHARS`] and notes the shortening.
pub(super) fn truncate_content(text: &str) -> String {
    if text.chars().count() <= MAX_CONTENT_CHARS {
        return text.to_owned();
    }
    let mut shortened = truncate_chars(text, MAX_CONTENT_CHARS).to_owned();
    shortened.push_str(TRUNCATED_NOTE);
    shortened
}
