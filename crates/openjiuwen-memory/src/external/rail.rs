//! The reusable pieces of the memory-provider tool rail.
//!
//! Python's `openjiuwen.core.memory.external.MemoryProvider` is tool-rail oriented:
//! it advertises JSON-Schema tool specs ([`MemoryProvider::get_tool_schemas`]), answers
//! tool calls ([`MemoryProvider::handle_tool_call`]), and injects recalled context
//! ([`MemoryProvider::prefetch`]) and provider guidance
//! ([`MemoryProvider::system_prompt_block`]) into a turn.
//!
//! This module holds only the *pure* half of that rail — schema construction, argument
//! parsing, result shaping, and Markdown rendering. Nothing here performs I/O, so a
//! vendor wire spec or a test can build and inspect the same values the trait's default
//! implementations produce.
//!
//! # Tool set
//!
//! [`default_tool_schemas`] is the vendor-neutral set every provider inherits: one write
//! tool ([`ADD_MEMORY_TOOL`]), one recall tool ([`SEARCH_MEMORY_TOOL`]), one listing tool
//! ([`LIST_MEMORIES_TOOL`]), and one delete tool ([`DELETE_MEMORY_TOOL`]). A vendor may
//! advertise its own names instead; it then overrides
//! [`MemoryProvider::get_tool_schemas`] and [`MemoryProvider::handle_tool_call`]
//! together, as Python's `mem0` provider does.
//!
//! # Error policy
//!
//! Argument parsing is the only fallible step and it reports
//! [`openjiuwen_common::error::Error::Validation`] naming the offending parameter.
//! A tool call never propagates that error to the model: [`MemoryProvider::handle_tool_call`]
//! turns it into a [`tool_error`] object and logs it, mirroring the Python rail, which
//! returns a JSON error instead of raising.
//!
//! [`MemoryProvider::get_tool_schemas`]: crate::external::MemoryProvider::get_tool_schemas
//! [`MemoryProvider::handle_tool_call`]: crate::external::MemoryProvider::handle_tool_call
//! [`MemoryProvider::prefetch`]: crate::external::MemoryProvider::prefetch
//! [`MemoryProvider::system_prompt_block`]: crate::external::MemoryProvider::system_prompt_block
//! [`MemoryProvider`]: crate::external::MemoryProvider

use std::sync::OnceLock;

use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};

use super::provider::ProviderCapabilities;
use crate::record::{MemoryRecord, MemoryType, NewMemory, RecallHit, SearchOptions};

/// The name of the tool that stores one memory.
pub const ADD_MEMORY_TOOL: &str = "add_memory";

/// The name of the tool that recalls memories relevant to a query.
pub const SEARCH_MEMORY_TOOL: &str = "search_memory";

/// The name of the tool that lists the memories of the provider's scope.
pub const LIST_MEMORIES_TOOL: &str = "list_memories";

/// The name of the tool that deletes one memory by id.
pub const DELETE_MEMORY_TOOL: &str = "delete_memory";

/// How many hits [`MemoryProvider::prefetch`](crate::external::MemoryProvider::prefetch)
/// requests by default.
pub const DEFAULT_PREFETCH_TOP_K: usize = 5;

/// The token budget [`MemoryProvider::prefetch`](crate::external::MemoryProvider::prefetch)
/// renders into the prompt.
pub const DEFAULT_PREFETCH_MAX_TOKENS: usize = 512;

/// The largest `top_k` a search tool call may request.
pub const MAX_SEARCH_TOP_K: usize = 50;

/// How many records the `list_memories` tool returns at most, whatever the call asks for.
pub const DEFAULT_LIST_LIMIT: usize = 50;

/// The hard ceiling the `list_memories` tool applies to a requested `limit`.
pub const MAX_LIST_LIMIT: usize = 500;

/// An empty argument map, handed out to non-object `args` values.
///
/// It is initialized once and never mutated, so it is immutable in effect.
static EMPTY_ARGS: OnceLock<Map<String, Value>> = OnceLock::new();

/// Builds one tool spec: `{name, description, parameters}`.
///
/// `properties` is the JSON-Schema `properties` object; `required` lists the mandatory
/// parameter names (an empty array when every parameter is optional, which is what the
/// Python schemas carry).
#[must_use]
pub fn tool_schema(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    })
}

/// A required or optional string parameter.
#[must_use]
pub fn string_property(description: &str) -> Value {
    json!({ "type": "string", "description": description })
}

/// An integer parameter.
#[must_use]
pub fn integer_property(description: &str) -> Value {
    json!({ "type": "integer", "description": description })
}

/// A boolean parameter.
#[must_use]
pub fn boolean_property(description: &str) -> Value {
    json!({ "type": "boolean", "description": description })
}

/// A string parameter restricted to `values`.
#[must_use]
pub fn enum_property(description: &str, values: &[&str]) -> Value {
    json!({ "type": "string", "description": description, "enum": values })
}

/// The four vendor-neutral tool specs, in declaration order.
///
/// The set mirrors the record SPI: one write, one recall, one listing, and one delete
/// tool. Each description is a single sentence, and only `add_memory`/`search_memory`/
/// `delete_memory` mark a parameter required — a listing needs nothing but its filters.
#[must_use]
pub fn default_tool_schemas() -> Vec<Value> {
    let kinds = kind_names();
    vec![
        tool_schema(
            ADD_MEMORY_TOOL,
            "Store one durable fact about the user in long-term memory.",
            json!({
                "content": string_property("The fact to store, written as one self-contained sentence."),
                "kind": enum_property("The memory kind, defaulting to semantic_memory.", &kinds),
                "importance": number_property("The importance of the fact in 0.0..=1.0."),
            }),
            &["content"],
        ),
        tool_schema(
            SEARCH_MEMORY_TOOL,
            "Recall the memories that are relevant to a query.",
            json!({
                "query": string_property("What to search for."),
                "top_k": integer_property("The maximum number of hits to return (1..=50, default 5)."),
                "threshold": number_property("The minimum score in 0.0..=1.0 a hit must reach."),
            }),
            &["query"],
        ),
        tool_schema(
            LIST_MEMORIES_TOOL,
            "List the memories of this provider, optionally narrowed to one kind.",
            json!({
                "kind": string_property("Return only memories of this kind."),
                "limit": integer_property("The maximum number of memories to return (default 50)."),
            }),
            &[],
        ),
        tool_schema(
            DELETE_MEMORY_TOOL,
            "Delete one memory by its id.",
            json!({
                "id": string_property("The id of the memory to delete."),
            }),
            &["id"],
        ),
    ]
}

/// A JSON number parameter.
#[must_use]
pub fn number_property(description: &str) -> Value {
    json!({ "type": "number", "description": description })
}

/// The `{"error": message}` object a rail tool call returns on failure.
#[must_use]
pub fn tool_error(message: impl Into<String>) -> Value {
    json!({ "error": message.into() })
}

/// The `{"result": value}` object a rail tool call returns on success.
#[must_use]
pub fn tool_result(value: Value) -> Value {
    json!({ "result": value })
}

/// The `{"result": text}` object a rail tool call returns for a plain-text answer.
#[must_use]
pub fn tool_ok_result(text: impl Into<String>) -> Value {
    tool_result(Value::String(text.into()))
}

/// One recall hit as `{id, content, score, type}`.
#[must_use]
pub fn hit_json(hit: &RecallHit) -> Value {
    json!({
        "id": hit.record.id,
        "content": hit.record.content,
        "score": hit.score,
        "type": hit.record.kind.as_str(),
    })
}

/// One record as `{id, content, type, importance}`.
#[must_use]
pub fn record_json(record: &MemoryRecord) -> Value {
    json!({
        "id": record.id,
        "content": record.content,
        "type": record.kind.as_str(),
        "importance": record.importance,
    })
}

/// The `search_memory` payload: `{"results": [...], "count": n}`.
#[must_use]
pub fn search_results(hits: &[RecallHit]) -> Value {
    let results: Vec<Value> = hits.iter().map(hit_json).collect();
    json!({ "results": results, "count": results.len() })
}

/// The `list_memories` payload: `{"memories": [...], "count": n}`.
#[must_use]
pub fn memory_listing(records: &[MemoryRecord]) -> Value {
    let memories: Vec<Value> = records.iter().map(record_json).collect();
    json!({ "memories": memories, "count": memories.len() })
}

/// The `add_memory` payload: `{"result": "stored", "id", "content"}`.
#[must_use]
pub fn stored_result(record: &MemoryRecord) -> Value {
    json!({ "result": "stored", "id": record.id, "content": record.content })
}

/// The `delete_memory` payload: `{"result": "deleted" | "not_found", "id"}`.
#[must_use]
pub fn deleted_result(id: &str, existed: bool) -> Value {
    let result = if existed { "deleted" } else { "not_found" };
    json!({ "result": result, "id": id })
}

/// Renders `lines` as a Markdown bullet list under `heading`.
///
/// The shape is `"## <heading>\n- a\n- b"`; blank lines are dropped, every other line is
/// kept verbatim, and no lines at all yields `""` (matching the Python prefetch format).
#[must_use]
pub fn render_bullets(heading: &str, lines: impl IntoIterator<Item = String>) -> String {
    let bullets: Vec<String> = lines
        .into_iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| format!("- {line}"))
        .collect();
    if bullets.is_empty() {
        return String::new();
    }
    format!("## {heading}\n{}", bullets.join("\n"))
}

/// Parses the arguments of a search tool call into a query and its options.
///
/// `query` is required and must hold a non-whitespace character. `top_k` is clamped to
/// `1..=MAX_SEARCH_TOP_K` (default [`DEFAULT_PREFETCH_TOP_K`]) and `threshold` to
/// `0.0..=1.0`, mirroring the Python rail's `min(top_k, 50)` clamp. Other keys — including
/// a Python-style `kinds` list — are ignored: the rail never widens or narrows the
/// provider's searched kinds.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Validation`] when `query` is missing, blank,
/// or not a string.
pub fn parse_search_args(args: &Value) -> Result<(String, SearchOptions)> {
    let map = args_object(args);
    let query = string_arg(map, "query")
        .map(str::trim)
        .filter(|query| !query.is_empty())
        .ok_or_else(|| Error::validation("missing required parameter: query"))?;

    let top_k = integer_arg(map, "top_k")
        .map_or(DEFAULT_PREFETCH_TOP_K, |top_k| clamp_top_k(top_k) as usize);
    let mut options = SearchOptions::default().with_top_k(top_k);
    if let Some(threshold) = number_arg(map, "threshold") {
        options = options.with_threshold(clamp_threshold(threshold));
    }
    Ok((query.to_owned(), options))
}

/// Parses the arguments of an add tool call into a [`NewMemory`].
///
/// `content` is required and must hold a non-whitespace character. `kind` accepts the
/// `snake_case` [`MemoryType`] names (`user_profile`, `semantic_memory`, …) and defaults to
/// `semantic_memory`; `importance` is clamped to `0.0..=1.0`; `metadata` must be a JSON
/// object and is copied field by field.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Validation`] when `content` is missing,
/// blank, or not a string, when `kind` is not a [`MemoryType`] name, when `importance` is
/// not a finite number, or when `metadata` is not an object.
pub fn parse_add_args(args: &Value) -> Result<NewMemory> {
    let map = args_object(args);
    let content = string_arg(map, "content")
        .map(str::trim)
        .filter(|content| !content.is_empty())
        .ok_or_else(|| Error::validation("missing required parameter: content"))?;

    let mut memory = NewMemory::new(content, MemoryType::SemanticMemory);
    if let Some(raw) = string_arg(map, "kind") {
        memory.kind = parse_kind(raw)?;
    }
    if let Some(importance) = number_arg(map, "importance") {
        if !importance.is_finite() {
            return Err(Error::validation(format!(
                "`importance` must be a finite number (got {importance})"
            )));
        }
        memory.importance = Some(importance.clamp(0.0, 1.0) as f32);
    }
    match map.get("metadata") {
        None | Some(Value::Null) => {}
        Some(Value::Object(fields)) => {
            memory.metadata = fields
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
        }
        Some(other) => {
            return Err(Error::validation(format!(
                "`metadata` must be a JSON object (got {other})"
            )));
        }
    }
    Ok(memory)
}

/// The arguments of a tool call as an object.
///
/// A non-object `args` (Python hands over a dict, but a model may emit anything) is
/// treated as an empty object, so every accessor below yields `None` instead of failing.
#[must_use]
pub fn args_object(args: &Value) -> &Map<String, Value> {
    match args {
        Value::Object(map) => map,
        _ => EMPTY_ARGS.get_or_init(Map::new),
    }
}

/// The string value of `key`, or `None` when absent or of another JSON type.
#[must_use]
pub fn string_arg<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    map.get(key).and_then(Value::as_str)
}

/// The integer value of `key`, or `None` when absent or of another JSON type.
#[must_use]
pub fn integer_arg(map: &Map<String, Value>, key: &str) -> Option<i64> {
    map.get(key).and_then(Value::as_i64)
}

/// The numeric value of `key` (integers included), or `None` when absent or not a number.
#[must_use]
pub fn number_arg(map: &Map<String, Value>, key: &str) -> Option<f64> {
    map.get(key).and_then(Value::as_f64)
}

/// Whether a provider with `capabilities` is configured and ready.
///
/// The check is local only — no network call — matching Python's
/// `is_available()` contract: a backend that declares no capability at all is not ready
/// for the rail.
#[must_use]
pub fn is_available(capabilities: &ProviderCapabilities) -> bool {
    *capabilities != ProviderCapabilities::none()
}

/// The `snake_case` names of every [`MemoryType`], in declaration order.
fn kind_names() -> Vec<&'static str> {
    MemoryType::ALL.iter().map(|kind| kind.as_str()).collect()
}

/// Maps a kind name to its [`MemoryType`], rejecting names the enum does not carry.
///
/// The comparison is case-insensitive and ignores surrounding whitespace, and only the
/// exact `snake_case` names of [`MemoryType::ALL`] are accepted — `"coding"` and `"Coding"`
/// map to [`MemoryType::Coding`], while `"code"` and `"nonsense"` are rejected. Short
/// Python aliases are deliberately not accepted here: a tool argument is model-generated,
/// so an ambiguous name is reported instead of guessed.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Validation`] naming the rejected value and
/// listing the accepted names.
pub fn parse_kind(raw: &str) -> Result<MemoryType> {
    let normalized = raw.trim().to_ascii_lowercase();
    MemoryType::ALL
        .iter()
        .copied()
        .find(|kind| kind.as_str() == normalized)
        .ok_or_else(|| {
            Error::validation(format!(
                "unknown memory kind: `{raw}` (expected one of {})",
                kind_names().join(", ")
            ))
        })
}

/// Clamps a requested `top_k` into `1..=MAX_SEARCH_TOP_K`.
fn clamp_top_k(top_k: i64) -> i64 {
    let ceiling = i64::try_from(MAX_SEARCH_TOP_K).unwrap_or(i64::MAX);
    top_k.clamp(1, ceiling)
}

/// Clamps a requested `threshold` into `0.0..=1.0` (non-finite values keep the default).
fn clamp_threshold(threshold: f64) -> f32 {
    if threshold.is_finite() {
        threshold.clamp(0.0, 1.0) as f32
    } else {
        // The engine rejects a non-finite threshold; keep its documented default instead.
        SearchOptions::default().threshold
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::record::MemoryScope;

    fn hit(kind: MemoryType, content: &str, score: f32) -> RecallHit {
        RecallHit {
            record: MemoryRecord::new(MemoryScope::new("alice", "ws"), content, kind)
                .with_id("m-1"),
            score,
            similarity: 0.0,
            keyword: score,
            importance: 0.5,
            recency: 1.0,
        }
    }

    #[test]
    fn tool_schema_carries_name_description_and_object_parameters() {
        let schema = tool_schema(
            "echo",
            "Echo one value.",
            json!({ "text": string_property("The text to echo.") }),
            &["text"],
        );
        assert_eq!(schema["name"], json!("echo"));
        assert_eq!(schema["description"], json!("Echo one value."));
        assert_eq!(schema["parameters"]["type"], json!("object"));
        assert_eq!(
            schema["parameters"]["properties"]["text"]["type"],
            json!("string")
        );
        assert_eq!(schema["parameters"]["required"], json!(["text"]));

        let optional = tool_schema("noop", "Do nothing.", json!({}), &[]);
        assert_eq!(optional["parameters"]["required"], json!([]));
    }

    #[test]
    fn property_helpers_emit_json_schema_fragments() {
        assert_eq!(string_property("a").as_object().unwrap().len(), 2);
        assert_eq!(integer_property("a")["type"], json!("integer"));
        assert_eq!(number_property("a")["type"], json!("number"));
        assert_eq!(boolean_property("a")["type"], json!("boolean"));
        let enumerated = enum_property("a", &["x", "y"]);
        assert_eq!(enumerated["type"], json!("string"));
        assert_eq!(enumerated["enum"], json!(["x", "y"]));
    }

    #[test]
    fn the_default_tool_set_has_unique_names_and_required_arrays() {
        let schemas = default_tool_schemas();
        assert_eq!(schemas.len(), 4);
        let names: Vec<&str> = schemas
            .iter()
            .map(|schema| schema["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                ADD_MEMORY_TOOL,
                SEARCH_MEMORY_TOOL,
                LIST_MEMORIES_TOOL,
                DELETE_MEMORY_TOOL
            ]
        );
        for schema in &schemas {
            assert_eq!(schema["parameters"]["type"], json!("object"));
            assert!(schema["parameters"]["properties"].is_object());
            assert!(schema["parameters"]["required"].is_array());
            assert_eq!(
                schema["description"].as_str().unwrap().matches('.').count(),
                1,
                "one sentence per description"
            );
        }
        let add = &schemas[0];
        assert_eq!(add["parameters"]["required"], json!(["content"]));
        assert_eq!(
            add["parameters"]["properties"]["kind"]["enum"]
                .as_array()
                .unwrap()
                .len(),
            7
        );
        let list = &schemas[2];
        assert_eq!(list["parameters"]["required"], json!([]));
    }

    #[test]
    fn tool_result_helpers_wrap_and_shape_payloads() {
        assert_eq!(tool_error("boom"), json!({ "error": "boom" }));
        assert_eq!(
            tool_result(json!({ "a": 1 })),
            json!({ "result": { "a": 1 } })
        );
        assert_eq!(tool_ok_result("done"), json!({ "result": "done" }));

        let recalled = hit(MemoryType::SemanticMemory, "dark mode", 0.5);
        let results = search_results(std::slice::from_ref(&recalled));
        assert_eq!(results["count"], json!(1));
        assert_eq!(results["results"][0]["id"], json!("m-1"));
        assert_eq!(results["results"][0]["content"], json!("dark mode"));
        assert_eq!(results["results"][0]["type"], json!("semantic_memory"));
        assert_eq!(results["results"][0]["score"], json!(0.5));

        let record = MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            "likes rust",
            MemoryType::Coding,
        )
        .with_id("m-2")
        .with_importance(0.25);
        let listing = memory_listing(std::slice::from_ref(&record));
        assert_eq!(listing["count"], json!(1));
        assert_eq!(listing["memories"][0]["type"], json!("coding"));
        assert_eq!(listing["memories"][0]["importance"], json!(0.25));

        assert_eq!(
            stored_result(&record),
            json!({ "result": "stored", "id": "m-2", "content": "likes rust" })
        );
        assert_eq!(
            deleted_result("m-2", true),
            json!({ "result": "deleted", "id": "m-2" })
        );
        assert_eq!(
            deleted_result("m-2", false),
            json!({ "result": "not_found", "id": "m-2" })
        );
    }

    #[test]
    fn render_bullets_matches_the_python_prefetch_shape() {
        let rendered = render_bullets("Mem0 Memory", vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(rendered, "## Mem0 Memory\n- a\n- b");
        assert_eq!(render_bullets("Heading", Vec::new()), "");
        assert_eq!(render_bullets("Heading", vec!["  ".to_owned()]), "");
        assert_eq!(
            render_bullets(
                "Heading",
                vec!["a".to_owned(), String::new(), "b".to_owned()]
            ),
            "## Heading\n- a\n- b"
        );
    }

    #[test]
    fn search_args_require_a_query_and_clamp_the_bounds() {
        let error = parse_search_args(&json!({})).expect_err("missing query");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert_eq!(
            error.to_string(),
            "validation: missing required parameter: query"
        );
        assert!(parse_search_args(&json!({ "query": "   " })).is_err());
        assert!(parse_search_args(&json!({ "query": 42 })).is_err());

        let (query, options) = parse_search_args(&json!({ "query": " dark mode " })).unwrap();
        assert_eq!(query, "dark mode");
        assert_eq!(options.top_k, DEFAULT_PREFETCH_TOP_K);
        assert!((options.threshold - 0.3).abs() < f32::EPSILON);

        let (_, options) = parse_search_args(&json!({ "query": "x", "top_k": 0 })).unwrap();
        assert_eq!(options.top_k, 1);
        let (_, options) = parse_search_args(&json!({ "query": "x", "top_k": 500 })).unwrap();
        assert_eq!(options.top_k, MAX_SEARCH_TOP_K);
        let (_, options) =
            parse_search_args(&json!({ "query": "x", "threshold": -1.0, "top_k": 3 })).unwrap();
        assert_eq!(options.top_k, 3);
        assert!((options.threshold - 0.0).abs() < f32::EPSILON);
        let (_, options) = parse_search_args(&json!({ "query": "x", "threshold": 4 })).unwrap();
        assert!((options.threshold - 1.0).abs() < f32::EPSILON);
        // A Python-style `kinds` list is not part of this tool.
        let (_, options) =
            parse_search_args(&json!({ "query": "x", "kinds": ["coding"] })).unwrap();
        assert_eq!(options.kinds, Vec::new());
    }

    #[test]
    fn add_args_validate_content_kind_importance_and_metadata() {
        let error = parse_add_args(&json!({})).expect_err("missing content");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(parse_add_args(&json!({ "content": "  " })).is_err());

        let memory = parse_add_args(&json!({ "content": " alice prefers dark mode " })).unwrap();
        assert_eq!(memory.content, "alice prefers dark mode");
        assert_eq!(memory.kind, MemoryType::SemanticMemory);
        assert!(memory.importance.is_none());
        memory.validate().expect("a parsed input is a valid write");

        let memory = parse_add_args(&json!({
            "content": "uses rust",
            "kind": "coding",
            "importance": 3.0,
            "metadata": { "repo": "openjiuwen" },
        }))
        .unwrap();
        assert_eq!(memory.kind, MemoryType::Coding);
        assert_eq!(memory.importance, Some(1.0));
        assert_eq!(memory.metadata["repo"], json!("openjiuwen"));

        let error = parse_add_args(&json!({ "content": "x", "kind": "nonsense" }))
            .expect_err("unknown kind");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("unknown memory kind"), "{error}");
        assert!(parse_add_args(&json!({ "content": "x", "importance": "high" })).is_ok());
        let error = parse_add_args(&json!({ "content": "x", "metadata": [] }))
            .expect_err("metadata must be an object");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[test]
    fn args_accessors_tolerate_anything_a_model_may_emit() {
        let args = json!({ "text": "hi", "count": 3, "ratio": 0.5, "flag": true });
        let map = args_object(&args);
        assert_eq!(string_arg(map, "text"), Some("hi"));
        assert_eq!(string_arg(map, "count"), None);
        assert_eq!(integer_arg(map, "count"), Some(3));
        assert_eq!(integer_arg(map, "ratio"), None);
        assert_eq!(number_arg(map, "ratio"), Some(0.5));
        assert_eq!(number_arg(map, "count"), Some(3.0));
        assert_eq!(number_arg(map, "flag"), None);
        assert_eq!(string_arg(map, "absent"), None);

        for args in [json!(null), json!("text"), json!([1, 2]), json!(7)] {
            let map = args_object(&args);
            assert_eq!(map.len(), 0);
            assert_eq!(string_arg(map, "query"), None);
        }
    }

    #[test]
    fn availability_follows_the_declared_capabilities() {
        assert!(!is_available(&ProviderCapabilities::none()));
        assert!(is_available(&ProviderCapabilities::local(false)));
        assert!(is_available(&ProviderCapabilities {
            pagination: true,
            ..ProviderCapabilities::none()
        }));
    }
}
