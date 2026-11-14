//! The `Jiuwen` memory server/SDK adapter (`jiuwen_memory`).
//!
//! The Python adapter speaks the Jiuwen `Mem2` `/v1/*` HTTP API (`httpx.AsyncClient`)
//! or imports the in-process `jiuwen_memory` kernel; this Rust adapter speaks the same
//! record surface over the shared [`RestProvider`](super::super::rest::RestProvider).
//!
//! # The documented wire
//!
//! The `Mem2` server documents exactly three endpoints, and [`JiuwenSpec`] implements
//! exactly those:
//!
//! | Verb | Call | Result |
//! | --- | --- | --- |
//! | `search` | `POST /v1/search` with `query`, `context.scope`, `top_k`, `disclosure` | an object with `items` (`unit_id`, `content`, `score`) |
//! | `add` | `POST /v1/add` with `content`, `scope`, `tags`, `system_metadata` | a **top-level array** of units (`id`, `tier`, `segments`) |
//! | `health` | `GET /healthz` | any answer (degraded, never failing) |
//!
//! The five record verbs the server does not document — `get`, `update`, `delete`,
//! `list`, and `clear` — return a typed `Unsupported` naming the verb and stating that
//! `POST /v1/<verb>` is not part of the API, so no endpoint is ever invented.
//!
//! # Scope payload
//!
//! Every write and every recall carries the server's five-segment scope object built by
//! [`scope_payload`]: the authenticated identity (`org`, `user`) comes from the
//! configuration options, and the per-call `agent`/`session` axes come from
//! [`MemoryScope`], so tenants and users stay mutually invisible underneath one
//! authenticated identity.
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpResponse;
use serde_json::{json, Value};

use super::super::provider::{MemoryProvider, ProviderCapabilities};
use super::super::registry::{ProviderConfig, ProviderKind};
use super::super::rest::{
    body_value, field_f32, field_str, vendor_hit, vendor_record, AuthScheme, RestCall,
    RestProvider, SpecContext, WireSpec,
};
use crate::external::rail;
use crate::record::{
    AddOptions, AddOutcome, MemoryFilter, MemoryRecord, MemoryScope, MemoryType, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions, DEFAULT_ID,
};

/// Registry id of the adapter (the Python provider id).
const ID: &str = "jiuwen_memory";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "Jiuwen";
/// The dependency an unconfigured handle names.
const MISSING: &str = "the `jiuwen_memory` SDK and its HTTP server transport";
/// `tracing` target of every event this adapter emits.
const LOG_TARGET: &str = "openjiuwen_memory::external::vendor";
/// The recall endpoint (`MemoryAPI.search`).
const SEARCH_PATH: &str = "/v1/search";
/// The write endpoint (`MemoryAPI.add`).
const ADD_PATH: &str = "/v1/add";
/// The liveness probe.
const HEALTH_PATH: &str = "/healthz";
/// Disclosure level asking the server for the full memory content.
const DISCLOSURE: &str = "l2";
/// The recall size the server uses when a caller asks for zero hits.
const DEFAULT_TOP_K: usize = 10;
/// The largest recall size the server accepts.
const MAX_TOP_K: usize = 50;
/// `options["identity_org"]` fallback (the dev authenticator's fixed identity).
const ORG_FALLBACK: &str = "local";
/// `options["identity_user"]` fallback (the dev authenticator's fixed identity).
const USER_FALLBACK: &str = "developer";
/// `options["tenant_id"]` fallback (the `agent` scope segment).
const TENANT_FALLBACK: &str = "default";
/// Tags attached to every write that does not bring its own.
const DEFAULT_TAGS: [&str; 2] = ["conversation", "user"];
/// The recall tool the rail advertises.
const MEM2_SEARCH_TOOL: &str = "mem2_search";
/// The write tool the rail advertises.
const MEM2_ADD_TOOL: &str = "mem2_add";

/// The static HTTP shape of the Jiuwen memory server.
pub(super) struct JiuwenSpec;

impl JiuwenSpec {
    /// The error a record verb the `Mem2` HTTP surface does not document returns.
    ///
    /// The server exposes only `POST /v1/search`, `POST /v1/add`, and `GET /healthz`, so
    /// the adapter never invents `POST /v1/<verb>` endpoints for the remaining verbs.
    #[allow(clippy::unused_self)] // the spec method shape keeps `&self`, like every `WireSpec` hook
    fn no_verb(&self, verb: &str) -> Error {
        Error::unsupported(format!(
            "the {DISPLAY} memory server has no {verb} operation: POST /v1/{verb} is not \
             part of the documented Mem2 HTTP API (only POST /v1/search, POST /v1/add and \
             GET /healthz)"
        ))
    }
}

impl WireSpec for JiuwenSpec {
    fn id(&self) -> &'static str {
        ID
    }

    fn display(&self) -> &'static str {
        DISPLAY
    }

    fn missing(&self) -> &'static str {
        MISSING
    }

    /// The Jiuwen server keeps session message history alongside semantic recall; it has
    /// no keyword mode, graph, variables, or page-addressable listing.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            message_history: true,
            ..ProviderCapabilities::none()
        }
    }

    /// The Jiuwen server takes a bearer token.
    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Bearer
    }

    /// The transport supplies the JSON content type and `Accept` headers; Jiuwen adds no
    /// static vendor header of its own.
    fn headers(&self, _config: &ProviderConfig) -> Vec<(String, String)> {
        Vec::new()
    }

    fn default_scope(&self, config: &ProviderConfig) -> MemoryScope {
        super::scope_from_options(config)
    }

    /// Builds the recall request: the server rejects unknown fields with a `400`, so the
    /// body carries exactly `query`, `context`, `top_k`, and `disclosure`.
    fn search_call(
        &self,
        cx: &SpecContext<'_>,
        query: &str,
        options: &SearchOptions,
    ) -> Result<RestCall> {
        let body = json!({
            "query": query,
            "context": { "scope": scope_payload(cx) },
            "top_k": effective_top_k(options.top_k),
            "disclosure": DISCLOSURE,
        });
        Ok(RestCall::post(SEARCH_PATH).with_body(body))
    }

    /// Decodes the recall answer.
    ///
    /// The server answers an object with an `items` array of `{unit_id, content, score}`;
    /// a non-object body or a non-list `items` decodes as "no hits" instead of failing,
    /// mirroring the Python backend.
    fn search_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        let value = match body_value(response) {
            Ok(value) => value,
            Err(error) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    %error,
                    "the Jiuwen recall answer was not JSON; treating it as no hits"
                );
                return Ok(Vec::new());
            }
        };
        let Some(items) = value.get("items").and_then(Value::as_array) else {
            return Ok(Vec::new());
        };
        let mut scored: Vec<(MemoryRecord, f32)> = items
            .iter()
            .filter(|item| item.is_object())
            .map(|item| {
                let record = vendor_record(item, cx.scope, MemoryType::SemanticMemory);
                let score = field_f32(item, &["score"]).unwrap_or(0.0);
                (record, score)
            })
            .filter(|(_, score)| options.threshold <= 0.0 || *score >= options.threshold)
            .collect();
        scored.sort_by(|left, right| right.1.total_cmp(&left.1));
        scored.truncate(effective_top_k(options.top_k));
        Ok(scored
            .into_iter()
            .map(|(record, score)| vendor_hit(record, score))
            .collect())
    }

    /// Builds the write request: one memory per call, with the server's exact field
    /// names (`content`, `scope`, `tags`, `system_metadata`).
    fn add_call(
        &self,
        cx: &SpecContext<'_>,
        memories: &[NewMemory],
        _options: &AddOptions,
    ) -> Result<RestCall> {
        let memory = memories.first().ok_or_else(|| {
            Error::validation("the Jiuwen memory server accepts exactly one memory per write")
        })?;
        let system_metadata = if infer_enabled(memory) {
            json!({ "infer": "true" })
        } else {
            Value::Null
        };
        let body = json!({
            "content": memory.content,
            "scope": scope_payload(cx),
            "tags": tags_payload(memory),
            "system_metadata": system_metadata,
        });
        Ok(RestCall::post(ADD_PATH).with_body(body))
    }

    /// Decodes the write answer.
    ///
    /// The server answers a **top-level array** of serialized units; the first unit's
    /// `id` becomes the record id and its `segments` are joined with newlines into the
    /// content (a serialized unit carries no top-level content field). An empty array is
    /// the legal "everything deduplicated" answer, and any other shape is treated the
    /// same way: the input is reported as skipped, never as a failed write.
    fn add_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        memories: &[NewMemory],
    ) -> Result<AddOutcome> {
        let memory = memories.first().ok_or_else(|| {
            Error::validation("the Jiuwen memory server accepts exactly one memory per write")
        })?;
        let value = match body_value(response) {
            Ok(value) => value,
            Err(error) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    %error,
                    "the Jiuwen write answer was not JSON; treating the memory as deduplicated"
                );
                return Ok(skipped_outcome(cx, memory));
            }
        };
        let Value::Array(units) = value else {
            tracing::debug!(
                target: LOG_TARGET,
                "the Jiuwen write answer was not a top-level array; treating the memory as deduplicated"
            );
            return Ok(skipped_outcome(cx, memory));
        };
        let Some(unit) = units.first() else {
            return Ok(skipped_outcome(cx, memory));
        };
        let mut record = vendor_record(unit, cx.scope, MemoryType::SemanticMemory);
        record.content = segment_content(unit);
        if let Some(tier) = field_str(unit, &["tier"]) {
            record
                .metadata
                .insert("tier".to_owned(), Value::String(tier.to_owned()));
        }
        Ok(AddOutcome {
            added: vec![record],
            ..AddOutcome::default()
        })
    }

    /// The server documents no point read.
    fn get_call(&self, _cx: &SpecContext<'_>, _id: &str) -> Result<RestCall> {
        Err(self.no_verb("get"))
    }

    /// The server documents no in-place update.
    fn update_call(
        &self,
        _cx: &SpecContext<'_>,
        _id: &str,
        _update: &MemoryUpdate,
    ) -> Result<RestCall> {
        Err(self.no_verb("update"))
    }

    /// The server documents no point delete.
    fn delete_call(&self, _cx: &SpecContext<'_>, _id: &str) -> Result<RestCall> {
        Err(self.no_verb("delete"))
    }

    /// The server documents no listing.
    fn list_call(&self, _cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
        Err(self.no_verb("list"))
    }

    /// The server documents no bulk delete.
    fn clear_call(&self, _cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
        Err(self.no_verb("clear"))
    }

    /// The liveness probe every Python backend performs on setup.
    fn health_call(&self, _cx: &SpecContext<'_>) -> Result<RestCall> {
        Ok(RestCall::get(HEALTH_PATH))
    }
}

/// The one [`WireSpec`] instance every Jiuwen adapter shares.
pub(super) static SPEC: &dyn WireSpec = &JiuwenSpec;

/// Handle for the `Jiuwen` memory server or SDK (`jiuwen_memory`).
///
/// Construction never fails: without an `endpoint` the handle is inert and every
/// operation names `MISSING` verbatim.
pub struct JiuwenMemoryProvider {
    /// The REST backend driven by [`SPEC`].
    pub(super) inner: RestProvider,
}

impl JiuwenMemoryProvider {
    /// Builds the adapter handle for `config`.
    #[must_use]
    pub fn new(config: &ProviderConfig) -> Self {
        Self {
            inner: RestProvider::from_config(SPEC, config),
        }
    }

    /// The configuration this handle was built from.
    #[must_use]
    pub fn config(&self) -> &ProviderConfig {
        self.inner.config()
    }

    /// The kind this adapter implements.
    #[must_use]
    pub fn kind(&self) -> ProviderKind {
        ProviderKind::Jiuwen
    }

    /// The wire shape this adapter speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.inner.spec()
    }

    /// The guidance the rail splices into the agent's system prompt.
    fn mem2_system_prompt(&self) -> String {
        let user = super::option_text(self.config(), "user_id").unwrap_or_else(|| "?".to_owned());
        let tenant =
            super::option_text(self.config(), "tenant_id").unwrap_or_else(|| "?".to_owned());
        format!(
            "# {DISPLAY} Memory\nActive. user={user}, tenant={tenant}.\n\
             Use mem2_search to recall memories, mem2_add to store a fact. \
             Conversation turns are extracted into memories automatically."
        )
    }

    /// The recall size of a prefetch: `options["top_k"]` (`5` by default, `0` stays `5`),
    /// clamped to the server maximum.
    fn prefetch_top_k(&self) -> usize {
        let configured = self
            .config()
            .option("top_k")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .unwrap_or(rail::DEFAULT_PREFETCH_TOP_K);
        configured.min(MAX_TOP_K)
    }

    /// Recalls memories for the `mem2_search` tool and renders its JSON payload.
    async fn mem2_search(&self, args: &Value) -> Value {
        let map = rail::args_object(args);
        let query = rail::string_arg(map, "query")
            .map(str::trim)
            .filter(|query| !query.is_empty());
        let Some(query) = query else {
            return rail::tool_error("missing required parameter: query");
        };
        let top_k = rail::integer_arg(map, "top_k")
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .map_or(DEFAULT_TOP_K, |value| value.min(MAX_TOP_K));
        let options = SearchOptions::default().with_top_k(top_k);
        match self.search(&self.default_scope(), query, &options).await {
            Ok(hits) => {
                let results: Vec<Value> = hits
                    .iter()
                    .map(|hit| {
                        json!({
                            "content": hit.record.content,
                            "item_id": hit.record.id,
                            "score": hit.score,
                        })
                    })
                    .collect();
                json!({ "results": results, "count": hits.len() })
            }
            Err(error) => {
                log_rail_failure(MEM2_SEARCH_TOOL, &error);
                rail::tool_error(error.to_string())
            }
        }
    }

    /// Stores one memory for the `mem2_add` tool and renders its JSON payload.
    async fn mem2_add(&self, args: &Value) -> Value {
        let map = rail::args_object(args);
        let content = rail::string_arg(map, "content")
            .map(str::trim)
            .filter(|content| !content.is_empty());
        let Some(content) = content else {
            return rail::tool_error("missing required parameter: content");
        };
        let mut memory = NewMemory::semantic(content);
        memory
            .metadata
            .insert("infer".to_owned(), Value::Bool(arg_infer(map.get("infer"))));
        if let Some(Value::Array(tags)) = map.get("tags") {
            let tags: Vec<Value> = tags
                .iter()
                .filter_map(|tag| tag.as_str().map(|tag| Value::String(tag.to_owned())))
                .collect();
            memory
                .metadata
                .insert("tags".to_owned(), Value::Array(tags));
        }
        match self
            .add(&self.default_scope(), vec![memory], &AddOptions::default())
            .await
        {
            Ok(outcome) => match outcome.added.first().or_else(|| outcome.merged.first()) {
                Some(record) => json!({
                    "result": "stored",
                    "item_id": record.id,
                    "content": record.content,
                    "tier": record.metadata.get("tier").cloned().unwrap_or(Value::Null),
                }),
                None => json!({
                    "result": "deduped",
                    "item_id": Value::Null,
                    "content": "",
                    "tier": "",
                }),
            },
            Err(error) => {
                log_rail_failure(MEM2_ADD_TOOL, &error);
                rail::tool_error(error.to_string())
            }
        }
    }
}

impl fmt::Debug for JiuwenMemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JiuwenMemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config().endpoint)
            .finish_non_exhaustive()
    }
}

implement_vendor_provider!(JiuwenMemoryProvider, inner;
    fn get_tool_schemas(&self) -> Vec<Value> {
        mem2_tool_schemas()
    }

    fn system_prompt_block(&self) -> String {
        self.mem2_system_prompt()
    }

    async fn prefetch(&self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            return String::new();
        }
        let options = SearchOptions::default().with_top_k(self.prefetch_top_k());
        match self.search(&self.default_scope(), query, &options).await {
            Ok(hits) => rail::render_bullets(
                "Jiuwen Memory",
                hits.iter().map(|hit| hit.record.content.clone()),
            ),
            Err(error) => {
                log_rail_failure("prefetch", &error);
                String::new()
            }
        }
    }

    async fn handle_tool_call(&self, tool: &str, args: &Value) -> Value {
        match tool {
            MEM2_SEARCH_TOOL => self.mem2_search(args).await,
            MEM2_ADD_TOOL => self.mem2_add(args).await,
            other => rail::tool_error(format!("unknown tool: {other}")),
        }
    }

    fn default_scope(&self) -> MemoryScope {
        super::scope_from_options(self.config())
    }
);

/// The five-segment scope object every `Mem2` request carries.
///
/// `org`/`user` are the authenticated identity and therefore come from
/// `options["identity_org"]`/`options["identity_user"]` rather than per-call data; the
/// per-call axes ride underneath as `agent` (the tenant) and `session` (the user or
/// scope). `space` is currently always empty, as in the Python backend.
fn scope_payload(cx: &SpecContext<'_>) -> Value {
    let agent = cx
        .agent_id()
        .map(str::trim)
        .filter(|agent| !agent.is_empty() && *agent != DEFAULT_ID)
        .map(ToOwned::to_owned)
        .or_else(|| option_text(cx, "tenant_id"))
        .unwrap_or_else(|| TENANT_FALLBACK.to_owned());
    let session = cx
        .session_id()
        .map(str::trim)
        .filter(|session| !session.is_empty())
        .map_or_else(|| cx.user_id().to_owned(), ToOwned::to_owned);
    json!({
        "org": option_text(cx, "identity_org").unwrap_or_else(|| ORG_FALLBACK.to_owned()),
        "space": "",
        "user": option_text(cx, "identity_user").unwrap_or_else(|| USER_FALLBACK.to_owned()),
        "agent": agent,
        "session": session,
    })
}

/// Reads one trimmed, non-blank string option of the call context.
fn option_text(cx: &SpecContext<'_>, key: &str) -> Option<String> {
    cx.option(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

/// The recall size sent to the server: zero means the server default, otherwise the
/// caller's request clamped to the documented maximum.
fn effective_top_k(top_k: usize) -> usize {
    if top_k == 0 {
        DEFAULT_TOP_K
    } else {
        top_k.min(MAX_TOP_K)
    }
}

/// Whether the server should run extraction and dedupe over the memory.
///
/// Extraction is on unless the memory's metadata says `infer: false` (as a boolean or as
/// the string `"false"`); that flag rides in `system_metadata` as `{"infer": "true"}`.
fn infer_enabled(memory: &NewMemory) -> bool {
    match memory.metadata.get("infer") {
        Some(Value::Bool(value)) => *value,
        Some(Value::String(value)) => !value.trim().eq_ignore_ascii_case("false"),
        _ => true,
    }
}

/// The tags of a write: the memory's metadata `tags` array when present, otherwise the
/// conversation tags the Python backend attaches.
fn tags_payload(memory: &NewMemory) -> Value {
    match memory.metadata.get("tags").and_then(Value::as_array) {
        Some(tags) => Value::Array(
            tags.iter()
                .filter_map(|tag| tag.as_str().map(|tag| Value::String(tag.to_owned())))
                .collect(),
        ),
        None => Value::Array(
            DEFAULT_TAGS
                .iter()
                .map(|tag| Value::String((*tag).to_owned()))
                .collect(),
        ),
    }
}

/// Rebuilds a serialized unit's content from its `segments`, newline-joined.
fn segment_content(unit: &Value) -> String {
    unit.get("segments")
        .and_then(Value::as_array)
        .map(|segments| {
            segments
                .iter()
                .filter(|segment| segment.is_object())
                .map(|segment| {
                    segment
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// The outcome of a write the server deduplicated: nothing added, the input's generated
/// id listed in `skipped`.
fn skipped_outcome(cx: &SpecContext<'_>, memory: &NewMemory) -> AddOutcome {
    let record = MemoryRecord::new(cx.scope.clone(), memory.content.clone(), memory.kind);
    AddOutcome {
        skipped: vec![record.id],
        ..AddOutcome::default()
    }
}

/// Logs a failed rail call; the model sees the message, the cause stays on `debug`.
fn log_rail_failure(tool: &str, error: &Error) {
    tracing::debug!(
        target: LOG_TARGET,
        provider = ID,
        tool,
        %error,
        "the Jiuwen memory rail call failed"
    );
}

/// Reads the `infer` argument of `mem2_add`: a boolean, or the string `"false"`/`"true"`;
/// absent means no extraction (the tool's documented default).
fn arg_infer(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(value)) => *value,
        Some(Value::String(value)) => value.trim().eq_ignore_ascii_case("true"),
        _ => false,
    }
}

/// The two tool specs the Jiuwen rail advertises to a model.
fn mem2_tool_schemas() -> Vec<Value> {
    vec![
        rail::tool_schema(
            MEM2_SEARCH_TOOL,
            "Search long-term memories by meaning. Returns ranked hits across keyword \
             (BM25), vector, and graph recall channels. Use this to recall user facts, \
             preferences, or past context.",
            json!({
                "query": rail::string_property("What to search for."),
                "top_k": top_k_property(),
            }),
            &["query"],
        ),
        rail::tool_schema(
            MEM2_ADD_TOOL,
            "Store a durable memory. By default the content is stored verbatim and indexed \
             (no LLM extraction). Set infer=true to have the server extract and dedup \
             self-contained facts from the content instead.",
            json!({
                "content": rail::string_property("The text to remember."),
                "infer": rail::boolean_property(
                    "Run LLM extraction + dedup on the content (default: false).",
                ),
                "tags": string_array_property("Optional labels for later filtering."),
            }),
            &["content"],
        ),
    ]
}

/// The `top_k` parameter: an integer defaulting to [`DEFAULT_TOP_K`], capped at
/// [`MAX_TOP_K`].
fn top_k_property() -> Value {
    let mut property = rail::integer_property("Max results (default 10, max 50).");
    if let Value::Object(fields) = &mut property {
        fields.insert("default".to_owned(), json!(DEFAULT_TOP_K));
        fields.insert("maximum".to_owned(), json!(MAX_TOP_K));
    }
    property
}

/// A parameter holding an array of strings.
fn string_array_property(description: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "string" },
        "description": description,
    })
}

#[cfg(test)]
#[allow(clippy::assert_is_empty, clippy::default_trait_access)]
mod tests;
