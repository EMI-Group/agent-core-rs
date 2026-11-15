//! The `LakeBase` wire shape: paths, request bodies, and response decoding.
//!
//! [`LakeBaseSpec`] is a [`WireSpec`], so it never sees a base URL, a credential, or a
//! retry schedule — it only builds a [`RestCall`] relative to the endpoint in use and
//! decodes a `2xx` body. The transport in [`crate::external::rest`] owns everything else.
//!
//! # Paths
//!
//! Python's `base_url` default is `http://localhost:8080/api/v1` and every path below is
//! appended to it, so [`ProviderConfig::endpoint`] is that same base:
//!
//! | Operation | Request |
//! | --- | --- |
//! | add | `POST /memory/bases/{base_id}/ingest` |
//! | search | `POST /memory/bases/{base_id}/recall` |
//! | list | `GET /memory/bases/{base_id}/memories?offset=&limit=&memory_type=` |
//! | get | `GET /memory/bases/{base_id}/memories/{memory_id}` |
//! | delete | `DELETE /memory/bases/{base_id}/memories/{memory_id}` |
//! | health | `GET /memory/bases/{base_id}/stats` |
//!
//! `base_id` is `options["base_id"]` (default [`super::DEFAULT_BASE_ID`]).
//! `options["database_id"]` (Python's default `db_agent_memory`) is only used by the
//! branch/version APIs, which this record-oriented adapter does not expose.
//!
//! # Ids
//!
//! `LakeBase` identifies memories by integer; this crate's records carry strings. The
//! boundary converts both ways: a vendor `memory_id` is stringified into
//! [`MemoryRecord::id`], and a record id is parsed back to an integer before it is
//! placed in a path — a non-numeric id is [`Error::Validation`], never a guessed request.
//!
//! # Memory types
//!
//! `LakeBase`'s six vendor types are `fact`, `episode`, `procedural`, `decision`,
//! `rejection`, and `convention`; this crate's [`MemoryType`] maps onto them as follows:
//!
//! | [`MemoryType`] | Vendor type |
//! | --- | --- |
//! | `EpisodicMemory` | `episode` |
//! | `Coding` | `convention` |
//! | `UserProfile`, `SemanticMemory`, `Summary`, `Variable`, `Unknown` | `fact` |
//!
//! A `metadata["memory_type"]` string on a [`NewMemory`] wins over the kind mapping, so a
//! caller can name `procedural`, `decision`, or `rejection` explicitly. The inverse
//! mapping decodes a vendor row: `episode` → `EpisodicMemory`, `convention` → `Coding`,
//! anything else → `SemanticMemory` (the vendor's own label is kept in the record's
//! `metadata["memory_type"]`).
use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpResponse;
use serde_json::{Map, Value};

use super::{
    DEFAULT_IMPORTANCE, DEFAULT_LIST_LIMIT, DEFAULT_MEMORY_TYPE, DISPLAY, ID, MAX_LIST_LIMIT,
    MEMORY_TYPES, MEMORY_TYPE_KEY, MISSING,
};
use crate::external::provider::ProviderCapabilities;
use crate::external::registry::ProviderConfig;
use crate::external::rest::{
    body_value, field_f32, field_str, listing_record, rank_hits, results_array,
    results_array_checked, vendor_record, AuthScheme, Op, RestCall, SpecContext, WireSpec,
};
use crate::record::{
    AddOptions, AddOutcome, MemoryFilter, MemoryRecord, MemoryScope, MemoryType, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions,
};

/// Prefix every `LakeBase` memory request shares.
const BASES_PATH: &str = "/memory/bases";
/// Value of the `role` field of an ingested memory (Python's fixed `"user"`).
const ROLE_USER: &str = "user";
/// Response/body key carrying the vendor memory type.
const TYPE_FIELD: &str = "memory_type";
/// Row keys holding the vendor's integer memory id, in priority order.
const ID_KEYS: &[&str] = &["memory_id", "id"];
/// The message of the one operation `LakeBase` does not document.
const NO_UPDATE: &str = "the LakeBase memory provider has no update operation: \
PUT /memory/bases/{base_id}/memories/{id} is not part of the documented LakeBase REST API";

/// The static HTTP shape of the `LakeBase` memory service.
pub(super) struct LakeBaseSpec;

impl WireSpec for LakeBaseSpec {
    fn id(&self) -> &'static str {
        ID
    }

    fn display(&self) -> &'static str {
        DISPLAY
    }

    fn missing(&self) -> &'static str {
        MISSING
    }

    /// `LakeBase` recalls semantically and pages its listings; it has no keyword mode,
    /// graph, variables, or message history.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            pagination: true,
            ..ProviderCapabilities::none()
        }
    }

    /// The `LakeBase` API takes a bearer token.
    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Bearer
    }

    /// The transport supplies the JSON content type and `Accept` headers; `LakeBase` adds
    /// no static vendor header of its own.
    fn headers(&self, _config: &ProviderConfig) -> Vec<(String, String)> {
        Vec::new()
    }

    fn default_scope(&self, config: &ProviderConfig) -> MemoryScope {
        super::super::scope_from_options(config)
    }

    /// Every operation but `update`, which `LakeBase` does not document.
    fn supports(&self, op: Op) -> bool {
        op != Op::Update
    }

    /// Ingests one memory: `{content, role: "user", memory_type, importance, metadata}`.
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
        let memory = single(memories)?;
        let mut body = Map::new();
        body.insert("content".to_owned(), Value::String(memory.content.clone()));
        body.insert("role".to_owned(), Value::String(ROLE_USER.to_owned()));
        body.insert(
            TYPE_FIELD.to_owned(),
            Value::String(vendor_type_for(memory.kind, &memory.metadata)),
        );
        body.insert(
            "importance".to_owned(),
            Value::from(f64::from(memory.importance.unwrap_or(DEFAULT_IMPORTANCE))),
        );
        body.insert("metadata".to_owned(), metadata_object(&memory.metadata));
        Ok(RestCall::post(ingest_path(&base_id(cx.config))).with_body(Value::Object(body)))
    }

    /// Decodes `{memory_id, memory_type}` into the stored record.
    ///
    /// The returned id is stringified; the submitted content, kind, importance, and
    /// metadata are kept, and the vendor's type is recorded as `metadata["memory_type"]`
    /// (the submitted type when the answer omits one). `LakeBase` addresses a memory by its
    /// integer id, so an answer without one is logged: the record is still reported as
    /// added (the ingest was accepted) with the client-generated placeholder id, which
    /// cannot be read, updated, or deleted afterwards.
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
        let memory = single(memories)?;
        let value = body_value(response)?;
        let mut record = MemoryRecord::new(cx.scope.clone(), memory.content.clone(), memory.kind);
        if let Some(importance) = memory.importance {
            record.importance = importance.clamp(0.0, 1.0);
        }
        record.metadata = memory.metadata.clone();
        record.source.clone_from(&memory.source);
        match stringified_id(&value) {
            Some(id) => record.id = id,
            None => {
                tracing::warn!(
                    target: super::LOG_TARGET,
                    vendor = ID,
                    "the LakeBase ingest answer carried no `memory_id`; the stored memory is \
                     reported under a client-generated id and cannot be addressed"
                );
            }
        }
        let vendor_type = field_str(&value, &[TYPE_FIELD]).map_or_else(
            || vendor_type_for(memory.kind, &memory.metadata),
            ToOwned::to_owned,
        );
        record
            .metadata
            .insert(MEMORY_TYPE_KEY.to_owned(), Value::String(vendor_type));
        let mut outcome = AddOutcome::default();
        outcome.added.push(record);
        Ok(outcome)
    }

    /// Recalls memories by meaning, with an optional vendor-type filter.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn search_call(
        &self,
        cx: &SpecContext<'_>,
        query: &str,
        options: &SearchOptions,
    ) -> Result<RestCall> {
        let mut body = Map::new();
        body.insert("query".to_owned(), Value::String(query.to_owned()));
        body.insert("top_k".to_owned(), Value::from(options.top_k));
        body.insert(
            "memory_types".to_owned(),
            memory_types_param(&options.kinds),
        );
        Ok(RestCall::post(recall_path(&base_id(cx.config))).with_body(Value::Object(body)))
    }

    /// Decodes `{memories: [{content, memory_type, score}]}` into scored hits.
    ///
    /// Hits are dropped when they are under a positive `options.threshold`, then sorted by
    /// score descending and truncated to `options.top_k`. A row carries no id, so the
    /// record gets a generated one. A row the vendor answered without a `score` is a match
    /// the backend did not rank, so the threshold never drops it; a row that carries no
    /// content at all is skipped with a warning instead of being reported as an empty
    /// memory.
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
        let candidates = results_array(&value)
            .iter()
            .filter_map(|row| {
                let record = listing_row(row, cx.scope, "recall")?;
                Some((record, field_f32(row, &["score"])))
            })
            .collect();
        Ok(rank_hits(candidates, options.threshold, options.top_k))
    }

    /// Reads one memory; the record id must be the vendor's integer id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `id` is not an integer.
    fn get_call(&self, cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
        Ok(RestCall::get(memory_path(
            &base_id(cx.config),
            parse_id(id)?,
        )))
    }

    /// Decodes one memory; an empty or `null` body means there is no record.
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
        let row = match &value {
            Value::Object(_) => &value,
            other => match results_array(other).first() {
                Some(row) => row,
                None => return Ok(None),
            },
        };
        Ok(Some(decode_row(row, cx.scope)))
    }

    /// `LakeBase` documents no update endpoint.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] naming the undocumented `PUT`.
    fn update_call(
        &self,
        _cx: &SpecContext<'_>,
        _id: &str,
        _update: &MemoryUpdate,
    ) -> Result<RestCall> {
        Err(no_update())
    }

    /// `LakeBase` documents no update endpoint.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] naming the undocumented `PUT`.
    fn update_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _id: &str,
    ) -> Result<MemoryRecord> {
        Err(no_update())
    }

    /// Removes one memory; the record id must be the vendor's integer id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `id` is not an integer.
    fn delete_call(&self, cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
        Ok(RestCall::delete(memory_path(
            &base_id(cx.config),
            parse_id(id)?,
        )))
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

    /// Lists one page: `options["offset"]` (default `0`) and `options["limit"]` (default
    /// [`DEFAULT_LIST_LIMIT`], clamped to `1..=MAX_LIST_LIMIT`).
    ///
    /// A caller-side `filter.kinds` restriction is expressed as one `memory_type` query
    /// parameter; several kinds cannot be expressed because the API takes a single value,
    /// so a multi-kind filter is sent unfiltered.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn list_call(&self, cx: &SpecContext<'_>, filter: &MemoryFilter) -> Result<RestCall> {
        let offset = cx
            .option("offset")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let limit = cx
            .option("limit")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_LIST_LIMIT, |value| {
                usize::try_from(value).unwrap_or(usize::MAX)
            });
        let mut call = RestCall::get(memories_path(&base_id(cx.config)))
            .query("offset", offset.to_string())
            .query("limit", limit.clamp(1, MAX_LIST_LIMIT).to_string());
        if let [kind] = filter.kinds.as_slice() {
            call = call.query(TYPE_FIELD, vendor_type_for_kind(*kind));
        }
        Ok(call)
    }

    /// Decodes `{memories: [...], total: n}` into records.
    ///
    /// `total` is ignored: the record SPI has no page-count channel, and inventing one
    /// would be a silent truncation. A row that carries no memory content is skipped with
    /// a warning rather than reported as an empty record.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the body is not JSON, or when a non-empty body carries no
    /// recognised row array.
    fn list_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _filter: &MemoryFilter,
    ) -> Result<Vec<MemoryRecord>> {
        let value = body_value(response)?;
        Ok(results_array_checked(response, &value)?
            .iter()
            .filter_map(|row| listing_row(row, cx.scope, "list"))
            .collect())
    }

    /// Probes the memory base statistics endpoint.
    ///
    /// # Errors
    ///
    /// This implementation never fails while building the call.
    fn health_call(&self, cx: &SpecContext<'_>) -> Result<RestCall> {
        Ok(RestCall::get(stats_path(&base_id(cx.config))))
    }
}

/// The one memory of a per-call write.
fn single(memories: &[NewMemory]) -> Result<&NewMemory> {
    memories.first().ok_or_else(|| {
        Error::validation("the LakeBase memory provider ingests exactly one memory per request")
    })
}

/// The message of the undocumented update operation.
fn no_update() -> Error {
    Error::unsupported(NO_UPDATE)
}

/// The base id of a configuration: `options["base_id"]`, defaulting to `mem_default`.
pub(super) fn base_id(config: &ProviderConfig) -> String {
    super::super::option_text(config, "base_id")
        .unwrap_or_else(|| super::DEFAULT_BASE_ID.to_owned())
}

/// The path of one memory base: `/memory/bases/{base_id}`.
fn base_path(base: &str) -> String {
    format!("{BASES_PATH}/{base}")
}

/// The path of the ingest endpoint of one base.
fn ingest_path(base: &str) -> String {
    format!("{}/ingest", base_path(base))
}

/// The path of the recall endpoint of one base.
fn recall_path(base: &str) -> String {
    format!("{}/recall", base_path(base))
}

/// The path of the memory collection of one base.
fn memories_path(base: &str) -> String {
    format!("{}/memories", base_path(base))
}

/// The path of one memory of one base.
fn memory_path(base: &str, id: i64) -> String {
    format!("{}/{id}", memories_path(base))
}

/// The path of the statistics endpoint of one base.
fn stats_path(base: &str) -> String {
    format!("{}/stats", base_path(base))
}

/// The vendor type a [`NewMemory`] is ingested as.
///
/// A non-blank `metadata["memory_type"]` wins, so a caller can name `procedural`,
/// `decision`, or `rejection`; otherwise the kind mapping of [`vendor_type_for_kind`]
/// applies.
pub(super) fn vendor_type_for(kind: MemoryType, metadata: &BTreeMap<String, Value>) -> String {
    if let Some(named) = metadata
        .get(MEMORY_TYPE_KEY)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|named| !named.is_empty())
    {
        return named.to_owned();
    }
    vendor_type_for_kind(kind).to_owned()
}

/// The vendor type of a crate kind: `episode` for episodic memory, `convention` for coding
/// memory, and `fact` for everything else.
pub(super) fn vendor_type_for_kind(kind: MemoryType) -> &'static str {
    match kind {
        MemoryType::EpisodicMemory => "episode",
        MemoryType::Coding => "convention",
        MemoryType::UserProfile
        | MemoryType::SemanticMemory
        | MemoryType::Summary
        | MemoryType::Variable
        | MemoryType::Unknown => DEFAULT_MEMORY_TYPE,
    }
}

/// The crate kind of a vendor type: `episode` → episodic, `convention` → coding, anything
/// else → semantic memory.
pub(super) fn kind_for_vendor_type(raw: &str) -> MemoryType {
    match raw.trim().to_ascii_lowercase().as_str() {
        "episode" => MemoryType::EpisodicMemory,
        "convention" => MemoryType::Coding,
        _ => MemoryType::SemanticMemory,
    }
}

/// Whether `raw` is one of the six documented vendor types.
pub(super) fn is_vendor_type(raw: &str) -> bool {
    MEMORY_TYPES.contains(&raw)
}

/// The `memory_types` request field: `null` without a restriction, otherwise the deduped
/// vendor types of `kinds`.
fn memory_types_param(kinds: &[MemoryType]) -> Value {
    if kinds.is_empty() {
        return Value::Null;
    }
    let mut names: Vec<&str> = Vec::with_capacity(kinds.len());
    for kind in kinds {
        let name = vendor_type_for_kind(*kind);
        if !names.contains(&name) {
            names.push(name);
        }
    }
    Value::Array(
        names
            .into_iter()
            .map(ToOwned::to_owned)
            .map(Value::String)
            .collect(),
    )
}

/// Encodes a [`NewMemory`] metadata map as a JSON object (`{}` when empty).
fn metadata_object(metadata: &BTreeMap<String, Value>) -> Value {
    Value::Object(
        metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

/// Maps one vendor row onto a record, honouring the vendor's own memory type.
fn decode_row(row: &Value, scope: &MemoryScope) -> MemoryRecord {
    let vendor_type = field_str(row, &[TYPE_FIELD]);
    let kind = vendor_type.map_or(MemoryType::SemanticMemory, kind_for_vendor_type);
    let mut record = vendor_record(row, scope, kind);
    apply_row_identity(&mut record, row, vendor_type);
    record
}

/// Maps one *listing* row onto a record, skipping a row the vendor answered without content.
///
/// `LakeBase`'s listing and recall rows are decoded by [`decode_row`], but a row that
/// carries no `content` is not a memory the base stored: reporting it would put an empty
/// record with an invented id in the caller's listing. `operation` names the request the
/// row came from in the warning.
fn listing_row(row: &Value, scope: &MemoryScope, operation: &str) -> Option<MemoryRecord> {
    let vendor_type = field_str(row, &[TYPE_FIELD]);
    let kind = vendor_type.map_or(MemoryType::SemanticMemory, kind_for_vendor_type);
    let mut record = listing_record(row, scope, kind, operation)?;
    apply_row_identity(&mut record, row, vendor_type);
    Some(record)
}

/// Copies the vendor's integer id and own memory type onto a decoded record.
///
/// `vendor_record` only reads string ids, so `LakeBase`'s integer `memory_id` is
/// stringified here; the vendor's type label is kept as `metadata["memory_type"]` when the
/// row carried one.
fn apply_row_identity(record: &mut MemoryRecord, row: &Value, vendor_type: Option<&str>) {
    if let Some(id) = stringified_id(row) {
        record.id = id;
    }
    if let Some(vendor_type) = vendor_type {
        record.metadata.insert(
            MEMORY_TYPE_KEY.to_owned(),
            Value::String(vendor_type.to_owned()),
        );
    }
}

/// The vendor memory id of a response or row, stringified.
///
/// `LakeBase` answers integers, so a string-only accessor would silently drop the id.
fn stringified_id(value: &Value) -> Option<String> {
    ID_KEYS.iter().find_map(|key| match value.get(*key) {
        Some(Value::String(raw)) => {
            let raw = raw.trim();
            (!raw.is_empty()).then(|| raw.to_owned())
        }
        Some(Value::Number(number)) => Some(
            number
                .as_i64()
                .map_or_else(|| number.to_string(), |id| id.to_string()),
        ),
        _ => None,
    })
}

/// Parses a record id into the integer id `LakeBase` uses.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `id` when it is not an integer.
pub(super) fn parse_id(id: &str) -> Result<i64> {
    let trimmed = id.trim();
    trimmed.parse::<i64>().map_err(|_| {
        Error::validation(format!(
            "the LakeBase memory provider identifies memories by an integer id, but `{trimmed}` is not one"
        ))
    })
}
