//! The `MemoryTool` implementation: cards, argument schemas, backend dispatch and rendering.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
};

use super::super::descriptions;
use super::super::support::{parse_args, schema_for, tool_card, ToolPayload};
use super::backend::{MemoryScope, MemorySearchRequest, MemoryToolBackend};
use super::kinds::MemoryToolKind;

/// Default number of search hits (`memory_search` `max_results`).
const DEFAULT_MAX_RESULTS: u64 = 10;

/// Default relevance floor (`memory_search` `min_score`).
const DEFAULT_MIN_SCORE: f64 = 0.3;

/// The failure reported when no memory backend is wired into the tool.
pub(super) const NO_BACKEND_MESSAGE: &str = "memory tool backend is not configured";

/// Arguments of `memory_search` (`get_memory_search_input_params`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MemorySearchArgs {
    /// Search query string
    pub query: String,
    /// Maximum number of results (optional)
    #[serde(default)]
    #[schemars(with = "u64")]
    pub max_results: Option<u64>,
    /// Minimum relevance score threshold (optional)
    #[serde(default)]
    #[schemars(with = "f64")]
    pub min_score: Option<f64>,
    /// Session key (optional, for scoping/filtering)
    #[serde(default)]
    #[schemars(with = "String")]
    pub session_key: Option<String>,
}

/// Arguments of `memory_get` (`get_memory_get_input_params`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MemoryGetArgs {
    /// Target path under memory/ (relative path)
    pub path: String,
    /// Starting line number (optional)
    #[serde(default)]
    #[schemars(with = "u64")]
    pub from_line: Option<u64>,
    /// Number of lines to read (optional)
    #[serde(default)]
    #[schemars(with = "u64")]
    pub lines: Option<u64>,
}

/// Arguments shared by `write_memory` and `coding_memory_write`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MemoryWriteArgs {
    /// Target path under memory/ (relative path)
    pub path: String,
    /// Content to write
    pub content: String,
    /// Append to file instead of overwrite (default false)
    #[serde(default)]
    #[schemars(with = "bool")]
    pub append: Option<bool>,
}

/// Arguments shared by `read_memory` and `coding_memory_read`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MemoryReadArgs {
    /// Target path under memory/ (relative path)
    pub path: String,
    /// Line offset to start reading from (optional)
    #[serde(default)]
    #[schemars(with = "u64")]
    pub offset: Option<u64>,
    /// Maximum number of lines to read (optional)
    #[serde(default)]
    #[schemars(with = "u64")]
    pub limit: Option<u64>,
}

/// Arguments shared by `edit_memory` and `coding_memory_edit`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MemoryEditArgs {
    /// Target path under memory/ (relative path)
    pub path: String,
    /// Original text to replace
    pub old_text: String,
    /// New replacement text
    pub new_text: String,
}

/// One of the eight memory tools: a card plus the backend it calls.
///
/// Python parity: the five `memory.py` classes and the three `coding_memory.py` classes,
/// collapsed into one implementation because they differ only in the card, the scope and the
/// payload/render shape selected by [`MemoryToolKind`].
pub struct MemoryTool {
    /// Which of the eight tools this instance implements.
    kind: MemoryToolKind,
    /// The memory backend, when one is wired in.
    backend: Option<Arc<dyn MemoryToolBackend>>,
    /// The model-facing card built from the description language.
    card: ToolCard,
}

impl MemoryTool {
    /// Builds the tool of `kind` over `backend` with the description language `language`.
    ///
    /// A `None` backend is legal: [`Tool::invoke`] then reports the "backend is not
    /// configured" failure instead of calling into an unwired memory implementation.
    #[must_use]
    pub fn new(
        kind: MemoryToolKind,
        backend: Option<Arc<dyn MemoryToolBackend>>,
        language: &str,
    ) -> Self {
        Self {
            kind,
            backend,
            card: Self::card_for(kind, language),
        }
    }

    /// Builds the model-facing card of `kind` in `language`.
    ///
    /// Unknown languages fall back to the Python default (`cn`) in
    /// [`descriptions::description`].
    #[must_use]
    pub fn card_for(kind: MemoryToolKind, language: &str) -> ToolCard {
        let mut input_params = args_schema(kind);
        install_defaults(kind, &mut input_params);
        let description = descriptions::description(kind.name(), language);
        tool_card(kind.name(), description, input_params, true)
    }

    /// Validates `inputs` and dispatches to the backend call of this tool's kind.
    ///
    /// Python checks the required arguments before touching the context, so a missing
    /// argument never surfaces as a backend error.
    async fn call(&self, inputs: &Value) -> ToolPayload {
        if let Some(payload) = required_failure(self.kind, inputs) {
            return payload;
        }
        let Some(backend) = self.backend.as_ref().map(Arc::as_ref) else {
            return ToolPayload::fail(NO_BACKEND_MESSAGE);
        };
        let scope = self.kind.scope().unwrap_or(MemoryScope::Memory);
        match self.kind {
            MemoryToolKind::Search => match parse_args::<MemorySearchArgs>(inputs) {
                Ok(args) => search_payload(backend, args).await,
                Err(payload) => payload,
            },
            MemoryToolKind::Get => match parse_args::<MemoryGetArgs>(inputs) {
                Ok(args) => get_payload(backend, args).await,
                Err(payload) => payload,
            },
            MemoryToolKind::Read | MemoryToolKind::CodingRead => {
                match parse_args::<MemoryReadArgs>(inputs) {
                    Ok(args) => read_payload(backend, scope, args).await,
                    Err(payload) => payload,
                }
            }
            MemoryToolKind::Write | MemoryToolKind::CodingWrite => {
                match parse_args::<MemoryWriteArgs>(inputs) {
                    Ok(args) => write_payload(backend, scope, args).await,
                    Err(payload) => payload,
                }
            }
            MemoryToolKind::Edit | MemoryToolKind::CodingEdit => {
                match parse_args::<MemoryEditArgs>(inputs) {
                    Ok(args) => edit_payload(backend, scope, args).await,
                    Err(payload) => payload,
                }
            }
        }
    }
}

#[async_trait]
impl Tool for MemoryTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.call(&inputs).await.to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        // Parse with `ToolOutput` (not `ToolPayload`) because `ToolOutput.success` has no serde
        // default: a plain object that merely looks like a payload must fall back to
        // `render_payload_text` instead of reading as a failure.
        let Ok(parsed) = serde_json::from_value::<ToolOutput>(output.clone()) else {
            return render_payload_text(output);
        };
        if !parsed.success {
            return render_tool_output(&parsed);
        }
        let data = parsed.data.unwrap_or(Value::Null);
        match self.kind {
            MemoryToolKind::Search => render_search(&data),
            MemoryToolKind::Get => data
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map_or_else(|| "No memory content found.".to_owned(), str::to_owned),
            // Python keeps the default rendering, which already picks `data.content`.
            MemoryToolKind::Read | MemoryToolKind::CodingRead => render_tool_output(&ToolOutput {
                success: true,
                data: Some(data),
                error: None,
            }),
            MemoryToolKind::Write | MemoryToolKind::CodingWrite => {
                let path = data.get("path").and_then(Value::as_str).unwrap_or_default();
                if data
                    .get("appended")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    format!("Appended to memory file {path}.")
                } else {
                    format!("Wrote memory file {path}.")
                }
            }
            MemoryToolKind::Edit | MemoryToolKind::CodingEdit => {
                let path = data.get("path").and_then(Value::as_str).unwrap_or_default();
                format!("Edited memory file {path}.")
            }
        }
    }
}

/// `memory_search`: forwards the request and mirrors the Python success/disabled rules.
async fn search_payload(backend: &dyn MemoryToolBackend, args: MemorySearchArgs) -> ToolPayload {
    let request = MemorySearchRequest {
        query: args.query,
        max_results: saturating_usize(args.max_results.unwrap_or(DEFAULT_MAX_RESULTS)),
        min_score: args.min_score.unwrap_or(DEFAULT_MIN_SCORE),
        session_key: args.session_key,
        scope: MemoryScope::Memory,
    };
    match backend.search(request).await {
        Ok(data) => {
            if data
                .get("disabled")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                let message = data
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                ToolPayload::fail_with_data(message, data)
            } else {
                ToolPayload::ok(data)
            }
        }
        Err(error) => backend_failure(&error),
    }
}

/// `memory_get`: reads a line slice and maps it onto the Python `memory_get` payload.
async fn get_payload(backend: &dyn MemoryToolBackend, args: MemoryGetArgs) -> ToolPayload {
    let offset = saturating_usize(args.from_line.unwrap_or(0));
    let limit = args.lines.map(saturating_usize);
    match backend
        .read(MemoryScope::Memory, &args.path, offset, limit)
        .await
    {
        Ok(data) => ToolPayload::ok(json!({
            "path": data.path,
            "text": data.text,
            "totalLines": data.total_lines,
            "fromLine": data.start_line,
            "toLine": data.end_line,
            "disabled": false,
        })),
        Err(error) => backend_failure(&error),
    }
}

/// `read_memory` / `coding_memory_read`: pages through a file of `scope`.
async fn read_payload(
    backend: &dyn MemoryToolBackend,
    scope: MemoryScope,
    args: MemoryReadArgs,
) -> ToolPayload {
    let offset = saturating_usize(args.offset.unwrap_or(0));
    let limit = args.limit.map(saturating_usize);
    match backend.read(scope, &args.path, offset, limit).await {
        Ok(data) => ToolPayload::ok(json!({
            "success": true,
            "path": data.path,
            "content": data.text,
            "totalLines": data.total_lines,
            "start_line": data.start_line,
            "end_line": data.end_line,
            "truncated": data.end_line < data.total_lines,
        })),
        Err(error) => backend_failure(&error),
    }
}

/// `write_memory` / `coding_memory_write`: overwrites or appends to a file of `scope`.
async fn write_payload(
    backend: &dyn MemoryToolBackend,
    scope: MemoryScope,
    args: MemoryWriteArgs,
) -> ToolPayload {
    let append = args.append.unwrap_or(false);
    match backend
        .write(scope, &args.path, &args.content, append)
        .await
    {
        Ok(data) => ToolPayload::ok(json!({
            "success": true,
            "path": data.path,
            "fullPath": data.full_path,
            "appended": data.appended,
            "fileExisted": data.file_existed,
        })),
        Err(error) => backend_failure(&error),
    }
}

/// `edit_memory` / `coding_memory_edit`: replaces `old_text` with `new_text` in `scope`.
async fn edit_payload(
    backend: &dyn MemoryToolBackend,
    scope: MemoryScope,
    args: MemoryEditArgs,
) -> ToolPayload {
    match backend
        .edit(scope, &args.path, &args.old_text, &args.new_text)
        .await
    {
        Ok(data) => ToolPayload::ok(json!({
            "success": true,
            "path": data.path,
            "replaced": data.replacements,
        })),
        Err(error) => backend_failure(&error),
    }
}

/// Returns the generated JSON Schema of the argument struct of `kind`.
fn args_schema(kind: MemoryToolKind) -> Value {
    match kind {
        MemoryToolKind::Search => schema_for::<MemorySearchArgs>(),
        MemoryToolKind::Get => schema_for::<MemoryGetArgs>(),
        MemoryToolKind::Write | MemoryToolKind::CodingWrite => schema_for::<MemoryWriteArgs>(),
        MemoryToolKind::Read | MemoryToolKind::CodingRead => schema_for::<MemoryReadArgs>(),
        MemoryToolKind::Edit | MemoryToolKind::CodingEdit => schema_for::<MemoryEditArgs>(),
    }
}

/// Injects the Python default values into the card schema of `kind`.
///
/// schemars renders `#[serde(default)]` on an `Option` field as `"default": null`, while the
/// Python cards either omit the key or carry the concrete default. The nulls are dropped and
/// the Python defaults (`max_results`, `min_score`, `append`) are inserted explicitly.
fn install_defaults(kind: MemoryToolKind, schema: &mut Value) {
    let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) else {
        return;
    };
    for property in properties.values_mut() {
        if let Some(object) = property.as_object_mut() {
            if object.get("default").is_some_and(Value::is_null) {
                object.remove("default");
            }
        }
    }
    let defaults = match kind {
        MemoryToolKind::Search => vec![
            ("max_results", json!(DEFAULT_MAX_RESULTS)),
            ("min_score", json!(DEFAULT_MIN_SCORE)),
        ],
        MemoryToolKind::Write | MemoryToolKind::CodingWrite => vec![("append", json!(false))],
        _ => Vec::new(),
    };
    for (field, value) in defaults {
        if let Some(object) = properties.get_mut(field).and_then(Value::as_object_mut) {
            object.insert("default".to_owned(), value);
        }
    }
}

/// Reports the Python "X is required" failure for a missing required argument.
///
/// The checks run on the raw arguments, before [`parse_args`], so a missing argument yields
/// the Python message rather than a serde error. `query`/`path` follow Python's truthiness
/// check (`if not inputs.get(...)`) while `content`/`old_text`/`new_text` follow its
/// `is None` check, meaning an empty string is a valid value there.
fn required_failure(kind: MemoryToolKind, inputs: &Value) -> Option<ToolPayload> {
    if !inputs.is_object() {
        return None;
    }
    if kind == MemoryToolKind::Search {
        return is_blank(inputs, "query").then(|| ToolPayload::fail("query is required"));
    }
    if is_blank(inputs, "path") {
        return Some(ToolPayload::fail("path is required"));
    }
    match kind {
        MemoryToolKind::Write | MemoryToolKind::CodingWrite if inputs.get("content").is_none() => {
            Some(ToolPayload::fail("content is required"))
        }
        MemoryToolKind::Edit | MemoryToolKind::CodingEdit => {
            if inputs.get("old_text").is_none() {
                return Some(ToolPayload::fail("old_text is required"));
            }
            inputs
                .get("new_text")
                .is_none()
                .then(|| ToolPayload::fail("new_text is required"))
        }
        _ => None,
    }
}

/// Whether the raw `field` is falsy, matching Python's `if not inputs.get(field)`.
fn is_blank(inputs: &Value, field: &str) -> bool {
    match inputs.get(field) {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.is_empty(),
        Some(Value::Bool(flag)) => !flag,
        Some(Value::Number(number)) => number.as_f64().is_none_or(|value| value == 0.0),
        Some(_) => false,
    }
}

/// Converts a model-supplied `u64` to `usize`, saturating instead of truncating.
fn saturating_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// Turns a backend failure into the payload the model sees.
///
/// Only the error message is forwarded (no `"tool: "` kind prefix), so the model reads the
/// same text Python's `build_error` produces.
fn backend_failure(error: &Error) -> ToolPayload {
    ToolPayload::fail(error.message().into_owned())
}

/// Renders the `memory_search` hits as citation/score blocks.
///
/// Python parity: `"{citation} (score {score:.2f})\n{snippet}"` blocks joined by a blank line.
fn render_search(data: &Value) -> String {
    let Some(results) = data
        .get("results")
        .and_then(Value::as_array)
        .filter(|results| !results.is_empty())
    else {
        return "No matching memories found.".to_owned();
    };
    results
        .iter()
        .map(|hit| {
            let citation = hit
                .get("citation")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let score = hit.get("score").and_then(Value::as_f64).unwrap_or(0.0);
            let snippet = hit
                .get("snippet")
                .and_then(Value::as_str)
                .unwrap_or_default();
            format!("{citation} (score {score:.2})\n{snippet}")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
