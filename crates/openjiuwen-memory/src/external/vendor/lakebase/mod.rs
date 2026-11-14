//! The `LakeBase`/`DBay` memory-service adapter (`lakebase`).
//!
//! The Python adapter speaks the `LakeBase` REST API over `httpx.AsyncClient` with a
//! bearer token and a single 60 s timeout; this Rust adapter speaks the same record
//! surface through the shared [`RestProvider`] (Python's `base_url` is
//! [`ProviderConfig::endpoint`]). The wire shape lives in [`spec`], so the transport's
//! validation, authentication, retries, failover, status mapping, and cancellation apply
//! unchanged.
//!
//! # Wire contract
//!
//! | Operation | Request |
//! | --- | --- |
//! | add | `POST /memory/bases/{base_id}/ingest` with `content`, `role: "user"`, `memory_type`, `importance`, `metadata` |
//! | search | `POST /memory/bases/{base_id}/recall` with `query`, `top_k`, `memory_types` |
//! | list | `GET /memory/bases/{base_id}/memories?offset=&limit=` plus `memory_type=` for a one-kind filter |
//! | get | `GET /memory/bases/{base_id}/memories/{memory_id}` |
//! | delete | `DELETE /memory/bases/{base_id}/memories/{memory_id}` |
//! | health | `GET /memory/bases/{base_id}/stats` |
//! | update | not part of the documented API — a typed `Unsupported` |
//!
//! `base_id` is `options["base_id"]` (`mem_default` by default). `LakeBase` ids are
//! integers, so a record id is stringified on the way in and parsed back on the way out;
//! a non-numeric id is a validation error. The [`MemoryType`] ↔ vendor-type mapping
//! (`episode`, `convention`, `fact`, …) is documented on [`spec`].
//!
//! # Rail
//!
//! Python's `LakeBase` tools are kept with their names and payload shapes:
//! `lkb_memory_search`, `lkb_memory_add`, `lkb_memory_list`, `lkb_memory_get`,
//! `lkb_memory_delete`, and `lkb_memory_stats`. Python's `lkb_memory_digest`,
//! `lkb_memory_traits`, `lkb_memory_switch_base`, and the branch/version tools are not
//! ported: they have no record-SPI equivalent and are not advertised.
//!
//! `lkb_memory_stats` routes through [`MemoryProvider::health`], because the record SPI
//! has no statistics channel: the tool reports whether the base's `/stats` probe answered
//! instead of inventing counters. `lkb_memory_list` applies its `offset`/`limit`
//! arguments to the page fetched with the configured `options["offset"]`/`options["limit"]`,
//! since the record SPI has no per-call paging parameter.
use std::collections::BTreeMap;
use std::fmt;

use serde_json::{json, Map, Value};

use super::super::provider::MemoryProvider;
use super::super::rail;
use super::super::registry::{ProviderConfig, ProviderKind};
use super::super::rest::{RestProvider, WireSpec};
use crate::record::{
    AddOptions, MemoryFilter, MemoryRecord, MemoryScope, NewMemory, RecallHit, SearchOptions,
};

mod spec;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;

use spec::LakeBaseSpec;

/// Registry id of the adapter (the Python provider id).
const ID: &str = "lakebase";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "LakeBase";
/// The dependency an unconfigured handle names.
const MISSING: &str = "the LakeBase REST transport (`httpx.AsyncClient`)";
/// `tracing` target of this module's events.
const LOG_TARGET: &str = "openjiuwen_memory::external::vendor::lakebase";
/// Memory base used when `options["base_id"]` names none (Python's default).
const DEFAULT_BASE_ID: &str = "mem_default";
/// Importance of an ingested memory that names none (Python's `_ingest` default).
const DEFAULT_IMPORTANCE: f32 = 0.5;
/// Vendor type of every kind that has no dedicated `LakeBase` label.
const DEFAULT_MEMORY_TYPE: &str = "fact";
/// The six documented `LakeBase` memory types.
const MEMORY_TYPES: [&str; 6] = [
    "fact",
    "episode",
    "procedural",
    "decision",
    "rejection",
    "convention",
];
/// Key naming the vendor memory type in a record's metadata.
const MEMORY_TYPE_KEY: &str = "memory_type";
/// Key naming the memory base in a rail payload.
const BASE_ID_KEY: &str = "base_id";
/// Page size of a listing when neither the configuration nor a tool call names one.
const DEFAULT_LIST_LIMIT: usize = 20;
/// Largest page `LakeBase` accepts for one listing request.
const MAX_LIST_LIMIT: usize = 100;
/// Hits a rail prefetch renders (Python's `prefetch` default).
const PREFETCH_TOP_K: usize = 5;
/// Heading of a rendered prefetch.
const PREFETCH_HEADING: &str = "## Related Memories";
/// Importance of a conversation episode written by [`MemoryProvider::sync_turn`].
const SYNC_TURN_IMPORTANCE: f32 = 0.4;
/// `lkb_memory_search` result count when the tool call names none.
const DEFAULT_TOOL_TOP_K: usize = 10;
/// `lkb_memory_search` result ceiling.
const MAX_TOOL_TOP_K: usize = 50;
/// Rail tool that searches memories by similarity.
const SEARCH_TOOL: &str = "lkb_memory_search";
/// Rail tool that stores one memory.
const ADD_TOOL: &str = "lkb_memory_add";
/// Rail tool that lists memories with pagination.
const LIST_TOOL: &str = "lkb_memory_list";
/// Rail tool that reads one memory by id.
const GET_TOOL: &str = "lkb_memory_get";
/// Rail tool that removes one memory by id.
const DELETE_TOOL: &str = "lkb_memory_delete";
/// Rail tool that reports the memory base's probe result.
const STATS_TOOL: &str = "lkb_memory_stats";

/// The one [`WireSpec`] instance every `LakeBase` adapter shares.
pub(super) static SPEC: &dyn WireSpec = &LakeBaseSpec;

/// Handle for the `LakeBase`/`DBay` memory service (`lakebase`).
///
/// Construction never fails: without an `endpoint` the handle is inert and every
/// operation names `MISSING` verbatim.
pub struct LakeBaseMemoryProvider {
    /// The REST backend driven by [`SPEC`].
    pub(super) inner: RestProvider,
}

impl LakeBaseMemoryProvider {
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
        ProviderKind::LakeBase
    }

    /// The wire shape this adapter speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.inner.spec()
    }
}

impl fmt::Debug for LakeBaseMemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LakeBaseMemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config().endpoint)
            .finish_non_exhaustive()
    }
}

/// The `{"error": …}` payload of a failed rail tool call, logged at `debug`.
fn tool_failure(tool: &str, error: &openjiuwen_common::error::Error) -> Value {
    tracing::debug!(
        target: LOG_TARGET,
        tool,
        %error,
        "the LakeBase rail tool call failed; the model sees the error payload"
    );
    json!({ "error": error.to_string() })
}

/// A JSON-Schema object parameter.
fn object_property(description: &str) -> Value {
    json!({ "type": "object", "description": description })
}

/// A JSON-Schema string-array parameter restricted to the six vendor types.
fn memory_types_property(description: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "string" },
        "enum": MEMORY_TYPES,
        "description": description,
    })
}

/// The six `LakeBase` rail tool specs, mirroring the Python adapter.
fn tool_schemas() -> Vec<Value> {
    vec![
        rail::tool_schema(
            SEARCH_TOOL,
            "Search memories by semantic similarity. Can filter by memory type.",
            json!({
                "query": rail::string_property("Search query text"),
                "top_k": rail::integer_property("Maximum number of results (default: 10)."),
                "memory_types": memory_types_property("Filter by memory types (optional)."),
            }),
            &["query"],
        ),
        rail::tool_schema(
            ADD_TOOL,
            "Store a new memory. Choose the appropriate memory_type for better organization.",
            json!({
                "content": rail::string_property("Memory content to store."),
                "memory_type": rail::enum_property(
                    "Type of memory: fact (knowledge), episode (events), procedural (how-to), decision (choices), rejection (avoid), convention (rules). Default: fact.",
                    &MEMORY_TYPES,
                ),
                "importance": rail::number_property("Importance score (0-1), higher = more significant. Default: 0.5."),
                "metadata": object_property("Optional structured metadata."),
            }),
            &["content"],
        ),
        rail::tool_schema(
            LIST_TOOL,
            "List memories with pagination. Optionally filter by type.",
            json!({
                "memory_type": rail::enum_property("Filter by memory type (optional).", &MEMORY_TYPES),
                "offset": rail::integer_property("Pagination offset (default: 0)."),
                "limit": rail::integer_property("Page size (default: 20, max: 100)."),
            }),
            &[],
        ),
        rail::tool_schema(
            GET_TOOL,
            "Get a single memory by ID.",
            json!({ "memory_id": rail::integer_property("Memory ID to retrieve.") }),
            &["memory_id"],
        ),
        rail::tool_schema(
            DELETE_TOOL,
            "Delete a memory by ID.",
            json!({ "memory_id": rail::integer_property("Memory ID to delete.") }),
            &["memory_id"],
        ),
        rail::tool_schema(
            STATS_TOOL,
            "Get memory base statistics (count, types, etc.).",
            json!({}),
            &[],
        ),
    ]
}

/// The vendor type label of one record.
///
/// The vendor's own `memory_type` metadata wins; otherwise the inverse kind mapping is
/// used so a record decoded from a projection still renders a label.
fn vendor_type_of(record: &MemoryRecord) -> String {
    record
        .metadata
        .get(MEMORY_TYPE_KEY)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map_or_else(
            || spec::vendor_type_for_kind(record.kind).to_owned(),
            ToOwned::to_owned,
        )
}

/// The `lkb_memory_search` payload: `{"memories": [{content, memory_type, score}], "count"}`.
fn search_payload(hits: &[RecallHit]) -> Value {
    let memories: Vec<Value> = hits
        .iter()
        .map(|hit| {
            json!({
                "content": hit.record.content,
                "memory_type": vendor_type_of(&hit.record),
                "score": hit.score,
            })
        })
        .collect();
    json!({ "memories": memories, "count": memories.len() })
}

/// The `lkb_memory_list` payload: `{"memories": [{memory_id, content, memory_type}], "count", "base_id"}`.
fn list_payload(records: &[MemoryRecord], base: &str) -> Value {
    let memories: Vec<Value> = records
        .iter()
        .map(|record| {
            json!({
                "memory_id": record.id,
                "content": record.content,
                "memory_type": vendor_type_of(record),
            })
        })
        .collect();
    json!({ "memories": memories, "count": memories.len(), "base_id": base })
}

/// The `lkb_memory_get` payload: the record plus its vendor type and memory base.
fn get_payload(record: &MemoryRecord, base: &str) -> Value {
    let mut payload = rail::record_json(record);
    if let Value::Object(object) = &mut payload {
        object.insert(
            MEMORY_TYPE_KEY.to_owned(),
            Value::String(vendor_type_of(record)),
        );
        object.insert(BASE_ID_KEY.to_owned(), Value::String(base.to_owned()));
    }
    payload
}

/// A positive tool page-size argument, clamped to `1..=MAX_LIST_LIMIT`.
fn page_arg(arguments: &Map<String, Value>, key: &str, default: usize) -> usize {
    rail::integer_arg(arguments, key)
        .map_or(default, |value| usize::try_from(value).unwrap_or(default))
}

/// The `lkb_memory_search` result count, clamped to `1..=MAX_TOOL_TOP_K`.
fn tool_top_k(arguments: &Map<String, Value>) -> usize {
    rail::integer_arg(arguments, "top_k")
        .map_or(DEFAULT_TOOL_TOP_K, |value| {
            usize::try_from(value).unwrap_or(DEFAULT_TOOL_TOP_K)
        })
        .clamp(1, MAX_TOOL_TOP_K)
}

/// The `lkb_memory_search` `memory_types` filter, or `None` when it names none.
fn tool_memory_types(arguments: &Map<String, Value>) -> Option<Vec<String>> {
    let types: Vec<String> = arguments
        .get("memory_types")?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|vendor_type| !vendor_type.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    (!types.is_empty()).then_some(types)
}

/// The `lkb_memory_add` importance argument, clamped to `0.0..=1.0` and narrowed to the
/// record model's `f32`.
#[allow(clippy::cast_possible_truncation)]
fn importance_arg(arguments: &Map<String, Value>) -> f32 {
    let importance = rail::number_arg(arguments, "importance")
        .unwrap_or(f64::from(DEFAULT_IMPORTANCE))
        .clamp(0.0, 1.0);
    importance as f32
}

/// A validated vendor memory type argument, or the `{"error": …}` payload it produces.
fn memory_type_arg<'a>(
    arguments: &'a Map<String, Value>,
    key: &str,
    default: &'a str,
) -> Result<&'a str, Value> {
    let vendor_type = rail::string_arg(arguments, key)
        .map(str::trim)
        .filter(|vendor_type| !vendor_type.is_empty())
        .unwrap_or(default);
    if spec::is_vendor_type(vendor_type) {
        Ok(vendor_type)
    } else {
        Err(json!({ "error": format!("unknown memory type: {vendor_type}") }))
    }
}

/// The structured metadata of an `lkb_memory_add` call.
fn metadata_arg(arguments: &Map<String, Value>, vendor_type: &str) -> BTreeMap<String, Value> {
    let mut metadata = BTreeMap::new();
    if let Some(object) = arguments.get("metadata").and_then(Value::as_object) {
        for (key, value) in object {
            if key != MEMORY_TYPE_KEY {
                metadata.insert(key.clone(), value.clone());
            }
        }
    }
    metadata.insert(
        MEMORY_TYPE_KEY.to_owned(),
        Value::String(vendor_type.to_owned()),
    );
    metadata
}

impl LakeBaseMemoryProvider {
    /// The `lkb_memory_search` handler: one recall, optionally restricted to vendor types.
    async fn rail_search(&self, arguments: &Map<String, Value>) -> Value {
        let Some(query) = rail::string_arg(arguments, "query")
            .map(str::trim)
            .filter(|query| !query.is_empty())
        else {
            return json!({ "error": "Missing required parameter: query" });
        };
        let mut options = SearchOptions::default()
            .with_top_k(tool_top_k(arguments))
            .with_threshold(0.0);
        if let Some(types) = tool_memory_types(arguments) {
            options = options.with_kinds(
                types
                    .iter()
                    .map(|vendor_type| spec::kind_for_vendor_type(vendor_type))
                    .collect(),
            );
        }
        let scope = super::scope_from_options(self.config());
        match MemoryProvider::search(&self.inner, &scope, query, &options).await {
            Ok(hits) => search_payload(&hits),
            Err(error) => tool_failure(SEARCH_TOOL, &error),
        }
    }

    /// The `lkb_memory_add` handler: one ingest, answered with the vendor id and type.
    async fn rail_add(&self, arguments: &Map<String, Value>) -> Value {
        let Some(content) = rail::string_arg(arguments, "content")
            .map(str::trim)
            .filter(|content| !content.is_empty())
        else {
            return json!({ "error": "Missing required parameter: content" });
        };
        let vendor_type = match memory_type_arg(arguments, MEMORY_TYPE_KEY, DEFAULT_MEMORY_TYPE) {
            Ok(vendor_type) => vendor_type,
            Err(payload) => return payload,
        };
        let mut memory = NewMemory::new(content, spec::kind_for_vendor_type(vendor_type))
            .with_importance(importance_arg(arguments));
        memory.metadata = metadata_arg(arguments, vendor_type);
        let scope = super::scope_from_options(self.config());
        match MemoryProvider::add(&self.inner, &scope, vec![memory], &AddOptions::default()).await {
            Ok(outcome) => match outcome.added.first() {
                Some(record) => json!({
                    "memory_id": record.id,
                    "memory_type": vendor_type_of(record),
                    "base_id": spec::base_id(self.config()),
                }),
                None => json!({ "error": "the `LakeBase` ingest answered without a memory" }),
            },
            Err(error) => tool_failure(ADD_TOOL, &error),
        }
    }

    /// The `lkb_memory_list` handler.
    ///
    /// The record SPI has no per-call paging parameter, so the `offset`/`limit` arguments
    /// slice the page fetched with the configured `options["offset"]`/`options["limit"]`.
    async fn rail_list(&self, arguments: &Map<String, Value>) -> Value {
        let mut filter = MemoryFilter::for_scope(&super::scope_from_options(self.config()));
        if let Some(vendor_type) = rail::string_arg(arguments, MEMORY_TYPE_KEY)
            .map(str::trim)
            .filter(|vendor_type| !vendor_type.is_empty())
        {
            if !spec::is_vendor_type(vendor_type) {
                return json!({ "error": format!("unknown memory type: {vendor_type}") });
            }
            filter = filter.kind(spec::kind_for_vendor_type(vendor_type));
        }
        let offset = page_arg(arguments, "offset", 0);
        let limit = page_arg(arguments, "limit", DEFAULT_LIST_LIMIT).clamp(1, MAX_LIST_LIMIT);
        match MemoryProvider::list(&self.inner, &filter).await {
            Ok(records) => {
                let page: Vec<MemoryRecord> =
                    records.into_iter().skip(offset).take(limit).collect();
                list_payload(&page, &spec::base_id(self.config()))
            }
            Err(error) => tool_failure(LIST_TOOL, &error),
        }
    }

    /// The `lkb_memory_get` handler.
    async fn rail_get(&self, arguments: &Map<String, Value>) -> Value {
        let Some(memory_id) = rail::integer_arg(arguments, "memory_id") else {
            return json!({ "error": "memory_id required" });
        };
        let scope = super::scope_from_options(self.config());
        match MemoryProvider::get(&self.inner, &scope, &memory_id.to_string()).await {
            Ok(Some(record)) => get_payload(&record, &spec::base_id(self.config())),
            Ok(None) => json!({ "error": "not found" }),
            Err(error) => tool_failure(GET_TOOL, &error),
        }
    }

    /// The `lkb_memory_delete` handler.
    async fn rail_delete(&self, arguments: &Map<String, Value>) -> Value {
        let Some(memory_id) = rail::integer_arg(arguments, "memory_id") else {
            return json!({ "error": "memory_id required" });
        };
        let scope = super::scope_from_options(self.config());
        match MemoryProvider::delete(&self.inner, &scope, &memory_id.to_string()).await {
            Ok(deleted) => json!({
                "deleted": deleted,
                "memory_id": memory_id,
                "base_id": spec::base_id(self.config()),
            }),
            Err(error) => tool_failure(DELETE_TOOL, &error),
        }
    }

    /// The `lkb_memory_stats` handler: the base's health probe, because the record SPI has
    /// no statistics channel to read counters from.
    async fn rail_stats(&self) -> Value {
        match MemoryProvider::health(&self.inner).await {
            Ok(health) => json!({
                "healthy": health.healthy,
                "detail": health.detail,
                "base_id": spec::base_id(self.config()),
            }),
            Err(error) => tool_failure(STATS_TOOL, &error),
        }
    }
}

implement_vendor_provider!(
    LakeBaseMemoryProvider,
    inner;

    /// The six `LakeBase` tool specs.
    fn get_tool_schemas(&self) -> Vec<Value> {
        tool_schemas()
    }

    /// The scope `LakeBase` rail calls use: the configured `user_id`/`scope_id` fallback.
    fn default_scope(&self) -> MemoryScope {
        super::scope_from_options(self.config())
    }

    /// Python's `LakeBase` prompt block, naming the memory base and the six memory types.
    fn system_prompt_block(&self) -> String {
        let base = spec::base_id(self.config());
        format!(
            "# LakeBase Memory System\n\
You have access to LakeBase memory with semantic search and structured storage (base: {base}).\n\
## Memory Operations\n\
- `lkb_memory_search`: Search memories by semantic similarity\n\
- `lkb_memory_add`: Store new memories with appropriate type\n\
- `lkb_memory_list`: List memories with pagination\n\
- `lkb_memory_stats`: Get memory base statistics\n\
## Memory Types\n\
- `fact`: Static knowledge (preferences, system info)\n\
- `episode`: Events and interactions (session history)\n\
- `procedural`: How-to knowledge (workflows, procedures)\n\
- `decision`: Choices made (architecture decisions)\n\
- `rejection`: What to avoid (failed approaches)\n\
- `convention`: Project rules (coding standards)\n\
Search before storing to avoid duplicates."
        )
    }

    /// Renders one recall of at most five hits; blank queries and failures render nothing.
    async fn prefetch(&self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            return String::new();
        }
        let scope = self.default_scope();
        let options = SearchOptions::default()
            .with_top_k(PREFETCH_TOP_K)
            .with_threshold(0.0);
        match MemoryProvider::search(&self.inner, &scope, query, &options).await {
            Ok(hits) => {
                let lines: Vec<String> = hits
                    .iter()
                    .filter(|hit| !hit.record.content.trim().is_empty())
                    .map(|hit| {
                        format!(
                            "- [{}] {} (score: {:.3})",
                            vendor_type_of(&hit.record),
                            hit.record.content.trim(),
                            hit.score
                        )
                    })
                    .collect();
                if lines.is_empty() {
                    String::new()
                } else {
                    format!("{PREFETCH_HEADING}\n{}", lines.join("\n"))
                }
            }
            Err(error) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    tool = "prefetch",
                    %error,
                    "the LakeBase prefetch failed; the turn continues without recalled memories"
                );
                String::new()
            }
        }
    }

    /// Writes one conversation episode; a blank user message writes nothing.
    ///
    /// The content is `"User: <u>\nAssistant: <a>"` (the assistant line only when it is
    /// non-blank), the type is `episode`, the importance is `0.4`, and the metadata records
    /// `source = "conversation"` — Python's `sync_turn` exactly. A backend failure is
    /// propagated, because the caller owns what a failed turn write means.
    async fn sync_turn(&self, user_msg: &str, assistant_msg: &str) -> openjiuwen_common::error::Result<()> {
        let user = user_msg.trim();
        if user.is_empty() {
            return Ok(());
        }
        let mut content = format!("User: {user}");
        let assistant = assistant_msg.trim();
        if !assistant.is_empty() {
            content.push_str("\nAssistant: ");
            content.push_str(assistant);
        }
        let memory = NewMemory::episodic(content)
            .with_importance(SYNC_TURN_IMPORTANCE)
            .with_metadata("source", json!("conversation"));
        MemoryProvider::add(
            &self.inner,
            &self.default_scope(),
            vec![memory],
            &AddOptions::default(),
        )
        .await
        .map(|_outcome| ())
    }

    /// Dispatches the six `LakeBase` rail tools over the record SPI.
    ///
    /// A backend error is returned as `{"error": <message>}` instead of propagating,
    /// because the caller is a model.
    async fn handle_tool_call(&self, tool: &str, args: &Value) -> Value {
        let arguments = rail::args_object(args);
        match tool {
            SEARCH_TOOL => self.rail_search(arguments).await,
            ADD_TOOL => self.rail_add(arguments).await,
            LIST_TOOL => self.rail_list(arguments).await,
            GET_TOOL => self.rail_get(arguments).await,
            DELETE_TOOL => self.rail_delete(arguments).await,
            STATS_TOOL => self.rail_stats().await,
            other => json!({ "error": format!("unknown tool: {other}") }),
        }
    }
);
