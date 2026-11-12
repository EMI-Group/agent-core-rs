//! Port of `openjiuwen/harness/tools/memory.py` and `coding_memory.py`: the eight memory tools.
//!
//! Python wires each tool to a `MemoryToolContext`/`CodingMemoryToolContext` and calls the
//! `memory_tool_ops`/`coding_memory_tool_ops` functions. Here the storage and retrieval
//! implementation sits behind [`MemoryToolBackend`] (the SQLite/embedding hybrid lives in
//! `openjiuwen_memory`), so this module owns only the model-facing contract: the cards, the
//! Python validation messages, the payload shapes and the per-kind [`Tool::render_for_llm`]
//! strings.
//!
//! The file is longer than the ~1000-line concern threshold on purpose: eight tool kinds share
//! one card/payload/validation shape and one backend seam, so splitting them into per-kind
//! modules would duplicate that shared scaffolding, and the offline test suite (about half the
//! file) has to live beside the kinds it covers.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
};

use super::descriptions;
use super::support::{parse_args, schema_for, tool_card, ToolPayload};

/// Default number of search hits (`memory_search` `max_results`).
const DEFAULT_MAX_RESULTS: u64 = 10;

/// Default relevance floor (`memory_search` `min_score`).
const DEFAULT_MIN_SCORE: f64 = 0.3;

/// The failure reported when no memory backend is wired into the tool.
const NO_BACKEND_MESSAGE: &str = "memory tool backend is not configured";

/// The memory tree a tool addresses.
///
/// Python parity: `memory` and `coding_memory` are served by separate contexts
/// (`MemoryToolContext`, `CodingMemoryToolContext`); the scope lets one backend serve both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// The long-term `memory` tree.
    Memory,
    /// The `coding_memory` tree with its front-matter rules.
    CodingMemory,
}

/// One `memory_search` call, as the backend sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemorySearchRequest {
    /// The keywords or question to look up.
    pub query: String,
    /// Maximum number of hits to return.
    pub max_results: usize,
    /// Minimum relevance score a hit must reach.
    pub min_score: f64,
    /// Optional session key used to scope or filter the hits.
    pub session_key: Option<String>,
    /// The memory tree to search.
    pub scope: MemoryScope,
}

/// The window of a memory file a read returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryReadData {
    /// The resolved path relative to the memory root.
    pub path: String,
    /// The text of the returned window.
    pub text: String,
    /// Total number of lines in the file.
    pub total_lines: usize,
    /// 1-based line number the window starts at.
    pub start_line: usize,
    /// 1-based line number the window ends at.
    pub end_line: usize,
}

/// The outcome of a memory write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryWriteData {
    /// The resolved path relative to the memory root.
    pub path: String,
    /// The absolute path of the written file.
    pub full_path: String,
    /// Whether the content was appended instead of replacing the file.
    pub appended: bool,
    /// Whether the file already existed before the write.
    pub file_existed: bool,
}

/// The outcome of a memory edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEditData {
    /// The resolved path relative to the memory root.
    pub path: String,
    /// Number of `old_text` occurrences that were replaced.
    pub replacements: usize,
}

/// The seam between the memory tools and the memory implementation.
///
/// Python parity: the `memory_tool_ops`/`coding_memory_tool_ops` functions, which own the
/// settings, the SQLite index and the embedding search. The harness only speaks this trait, so
/// `openjiuwen_memory` (or a test double) can provide the storage.
#[async_trait]
pub trait MemoryToolBackend: Send + Sync {
    /// Searches long-term memory and returns the Python `memory_search` result object, which
    /// the tool forwards to the model verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the search cannot run; the tool forwards the message.
    async fn search(&self, request: MemorySearchRequest) -> Result<Value>;

    /// Reads the window of `path` starting at `offset` (1-based; `0` means the first line).
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the file does not exist or cannot be read.
    async fn read(
        &self,
        scope: MemoryScope,
        path: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<MemoryReadData>;

    /// Writes `content` to `path`, appending when `append` is set.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the file cannot be written or violates the tree's rules.
    async fn write(
        &self,
        scope: MemoryScope,
        path: &str,
        content: &str,
        append: bool,
    ) -> Result<MemoryWriteData>;

    /// Replaces every `old_text` occurrence in `path` with `new_text`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the file cannot be read or written.
    async fn edit(
        &self,
        scope: MemoryScope,
        path: &str,
        old_text: &str,
        new_text: &str,
    ) -> Result<MemoryEditData>;
}

/// One of the eight model-facing memory tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryToolKind {
    /// `memory_search`: search long-term memory.
    Search,
    /// `memory_get`: read a line slice of a memory file.
    Get,
    /// `write_memory`: overwrite or append a memory file.
    Write,
    /// `read_memory`: page through a memory file by offset/limit.
    Read,
    /// `edit_memory`: exact string replacement in a memory file.
    Edit,
    /// `coding_memory_read`: page through a coding-memory file.
    CodingRead,
    /// `coding_memory_write`: write a coding-memory file.
    CodingWrite,
    /// `coding_memory_edit`: exact string replacement in a coding-memory file.
    CodingEdit,
}

impl MemoryToolKind {
    /// Every kind, in Python registry order (`create_memory_tools`, then the coding tools).
    pub const ALL: [Self; 8] = [
        Self::Search,
        Self::Get,
        Self::Write,
        Self::Read,
        Self::Edit,
        Self::CodingRead,
        Self::CodingWrite,
        Self::CodingEdit,
    ];

    /// The model-facing tool name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Search => "memory_search",
            Self::Get => "memory_get",
            Self::Write => "write_memory",
            Self::Read => "read_memory",
            Self::Edit => "edit_memory",
            Self::CodingRead => "coding_memory_read",
            Self::CodingWrite => "coding_memory_write",
            Self::CodingEdit => "coding_memory_edit",
        }
    }

    /// The memory tree the kind addresses.
    ///
    /// Every kind addresses exactly one tree today; the `Option` mirrors the Python split
    /// where a future kind (for example a cross-tree search) could address none, so the
    /// `Some`-only body is intentional and not narrowed.
    #[must_use]
    #[allow(clippy::unnecessary_wraps)]
    pub fn scope(self) -> Option<MemoryScope> {
        Some(match self {
            Self::Search | Self::Get | Self::Write | Self::Read | Self::Edit => MemoryScope::Memory,
            Self::CodingRead | Self::CodingWrite | Self::CodingEdit => MemoryScope::CodingMemory,
        })
    }

    /// Looks a kind up by model-facing name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    /// The canned backend reply of one test case.
    enum Reply {
        Search(Value),
        Read(MemoryReadData),
        Write(MemoryWriteData),
        Edit(MemoryEditData),
    }

    impl Reply {
        /// Wraps this reply in a fresh recording backend.
        fn backend(self) -> Arc<FakeBackend> {
            let mut backend = FakeBackend::default();
            match self {
                Self::Search(data) => backend.search = Some(data),
                Self::Read(data) => backend.read = Some(data),
                Self::Write(data) => backend.write = Some(data),
                Self::Edit(data) => backend.edit = Some(data),
            }
            Arc::new(backend)
        }
    }

    /// A [`MemoryToolBackend`] that records each call as a compact string and replays a reply.
    #[derive(Default)]
    struct FakeBackend {
        calls: Mutex<Vec<String>>,
        search: Option<Value>,
        read: Option<MemoryReadData>,
        write: Option<MemoryWriteData>,
        edit: Option<MemoryEditData>,
        failure: Option<&'static str>,
    }

    impl FakeBackend {
        /// Records `call` and fails the call when a failure is configured.
        fn record(&self, call: String) -> Result<()> {
            self.calls.lock().unwrap().push(call);
            match self.failure {
                Some(message) => Err(Error::tool(message)),
                None => Ok(()),
            }
        }

        /// The calls seen so far, in order.
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl MemoryToolBackend for FakeBackend {
        async fn search(&self, request: MemorySearchRequest) -> Result<Value> {
            self.record(format!(
                "search {} {} {} {:?} {:?}",
                request.query,
                request.max_results,
                request.min_score,
                request.session_key,
                request.scope
            ))?;
            self.search
                .clone()
                .ok_or_else(|| Error::tool("no search reply configured"))
        }

        async fn read(
            &self,
            scope: MemoryScope,
            path: &str,
            offset: usize,
            limit: Option<usize>,
        ) -> Result<MemoryReadData> {
            self.record(format!("read {scope:?} {path} {offset} {limit:?}"))?;
            self.read
                .clone()
                .ok_or_else(|| Error::tool("no read reply configured"))
        }

        async fn write(
            &self,
            scope: MemoryScope,
            path: &str,
            content: &str,
            append: bool,
        ) -> Result<MemoryWriteData> {
            self.record(format!("write {scope:?} {path} {content:?} {append}"))?;
            self.write
                .clone()
                .ok_or_else(|| Error::tool("no write reply configured"))
        }

        async fn edit(
            &self,
            scope: MemoryScope,
            path: &str,
            old_text: &str,
            new_text: &str,
        ) -> Result<MemoryEditData> {
            self.record(format!("edit {scope:?} {path} {old_text:?} {new_text:?}"))?;
            self.edit
                .clone()
                .ok_or_else(|| Error::tool("no edit reply configured"))
        }
    }

    /// Builds a tool of `kind` over `backend` in English.
    fn tool(kind: MemoryToolKind, backend: Option<Arc<dyn MemoryToolBackend>>) -> MemoryTool {
        MemoryTool::new(kind, backend, "en")
    }

    /// Invokes `tool` with a fresh, uncancelled context and unwraps the payload.
    async fn invoke(tool: &MemoryTool, inputs: Value) -> Value {
        Tool::invoke(tool, inputs, &ToolContext::new())
            .await
            .unwrap()
    }

    /// Parses `text` as JSON, so expected payloads stay on one line.
    fn json_text(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    /// Arguments that satisfy every required argument of `kind`.
    fn valid_args(kind: MemoryToolKind) -> Value {
        match kind {
            MemoryToolKind::Search => json!({"query": "q"}),
            MemoryToolKind::Get | MemoryToolKind::Read | MemoryToolKind::CodingRead => {
                json!({"path": "a.md"})
            }
            MemoryToolKind::Write | MemoryToolKind::CodingWrite => {
                json!({"path": "a.md", "content": "body"})
            }
            MemoryToolKind::Edit | MemoryToolKind::CodingEdit => {
                json!({"path": "a.md", "old_text": "a", "new_text": "b"})
            }
        }
    }

    /// A read result of `path` carrying `text` as lines `start..=end` of `total`.
    fn read_data(path: &str, text: &str, total: usize, start: usize, end: usize) -> MemoryReadData {
        MemoryReadData {
            path: path.to_owned(),
            text: text.to_owned(),
            total_lines: total,
            start_line: start,
            end_line: end,
        }
    }

    /// A write result of `path`.
    fn write_data(path: &str, appended: bool, existed: bool) -> MemoryWriteData {
        MemoryWriteData {
            path: path.to_owned(),
            full_path: format!("/mem/{path}"),
            appended,
            file_existed: existed,
        }
    }

    /// An edit result of `path`.
    fn edit_data(path: &str, replacements: usize) -> MemoryEditData {
        MemoryEditData {
            path: path.to_owned(),
            replacements,
        }
    }

    #[test]
    fn kinds_scope_and_lookup_follow_the_registry() {
        let names: Vec<&str> = MemoryToolKind::ALL.iter().map(|kind| kind.name()).collect();
        assert_eq!(
            names,
            [
                "memory_search",
                "memory_get",
                "write_memory",
                "read_memory",
                "edit_memory",
                "coding_memory_read",
                "coding_memory_write",
                "coding_memory_edit",
            ]
        );
        for kind in MemoryToolKind::ALL {
            assert_eq!(MemoryToolKind::from_name(kind.name()), Some(kind));
            let expected = match kind {
                MemoryToolKind::CodingRead
                | MemoryToolKind::CodingWrite
                | MemoryToolKind::CodingEdit => MemoryScope::CodingMemory,
                _ => MemoryScope::Memory,
            };
            assert_eq!(kind.scope(), Some(expected), "{}", kind.name());
        }
        assert_eq!(MemoryToolKind::from_name(""), None);
        assert_eq!(MemoryToolKind::from_name("Memory_Search"), None);
        assert_eq!(MemoryToolKind::from_name("read_file"), None);
    }

    #[test]
    fn cards_carry_the_python_arguments_and_defaults() {
        let required = [
            (MemoryToolKind::Search, &["query"][..]),
            (MemoryToolKind::Get, &["path"][..]),
            (MemoryToolKind::Write, &["path", "content"][..]),
            (MemoryToolKind::Read, &["path"][..]),
            (MemoryToolKind::Edit, &["path", "old_text", "new_text"][..]),
            (MemoryToolKind::CodingRead, &["path"][..]),
            (MemoryToolKind::CodingWrite, &["path", "content"][..]),
            (
                MemoryToolKind::CodingEdit,
                &["path", "old_text", "new_text"][..],
            ),
        ];
        let properties = [
            (
                MemoryToolKind::Search,
                &["max_results", "min_score", "query", "session_key"][..],
            ),
            (MemoryToolKind::Get, &["from_line", "lines", "path"][..]),
            (MemoryToolKind::Write, &["append", "content", "path"][..]),
            (MemoryToolKind::Read, &["limit", "offset", "path"][..]),
            (MemoryToolKind::Edit, &["new_text", "old_text", "path"][..]),
            (MemoryToolKind::CodingRead, &["limit", "offset", "path"][..]),
            (
                MemoryToolKind::CodingWrite,
                &["append", "content", "path"][..],
            ),
            (
                MemoryToolKind::CodingEdit,
                &["new_text", "old_text", "path"][..],
            ),
        ];

        for kind in MemoryToolKind::ALL {
            let card = MemoryTool::card_for(kind, "en");
            assert_eq!(card.name, kind.name());
            assert!(card.parallel_safe);
            assert_eq!(
                card.description,
                descriptions::description(kind.name(), "en")
            );
            assert_eq!(card.input_params["type"], json!("object"));

            let index = MemoryToolKind::ALL
                .iter()
                .position(|candidate| *candidate == kind)
                .unwrap();
            assert_eq!(card.input_params["required"], json!(required[index].1));
            let mut keys: Vec<&str> = card.input_params["properties"]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            assert_eq!(keys, properties[index].1, "{}", kind.name());

            let rendered = card.input_params.to_string();
            assert!(!rendered.contains("$ref"), "dangling reference: {rendered}");
            assert!(
                !rendered.contains("\"default\":null"),
                "null default: {rendered}"
            );
            // Unknown languages fall back to the Python default (`cn`).
            assert_eq!(
                MemoryTool::card_for(kind, "fr").description,
                descriptions::description(kind.name(), "cn")
            );
        }

        let search = MemoryTool::card_for(MemoryToolKind::Search, "en");
        let search = &search.input_params["properties"];
        assert_eq!(search["query"]["type"], json!("string"));
        assert_eq!(search["max_results"]["type"], json!("integer"));
        assert_eq!(search["max_results"]["default"], json!(10));
        assert_eq!(search["min_score"]["type"], json!("number"));
        assert_eq!(search["min_score"]["default"], json!(0.3));
        assert_eq!(search["session_key"]["type"], json!("string"));
        assert_eq!(search["session_key"].get("default"), None);

        for kind in [MemoryToolKind::Write, MemoryToolKind::CodingWrite] {
            let card = MemoryTool::card_for(kind, "en");
            let properties = &card.input_params["properties"];
            assert_eq!(properties["append"]["type"], json!("boolean"));
            assert_eq!(properties["append"]["default"], json!(false));
        }
        for kind in [MemoryToolKind::Read, MemoryToolKind::CodingRead] {
            let card = MemoryTool::card_for(kind, "en");
            let properties = &card.input_params["properties"];
            assert_eq!(properties["offset"]["type"], json!("integer"));
            assert_eq!(properties["limit"].get("default"), None);
        }
    }

    #[tokio::test]
    async fn a_missing_backend_and_missing_arguments_report_python_messages() {
        for kind in MemoryToolKind::ALL {
            let out = invoke(&tool(kind, None), valid_args(kind)).await;
            assert_eq!(out["success"], json!(false), "{}", kind.name());
            assert_eq!(out["error"], json!(NO_BACKEND_MESSAGE), "{}", kind.name());
            assert_eq!(out["data"], Value::Null);
            for key in [
                "success",
                "data",
                "error",
                "extracted_content",
                "include_extracted_content_only_once",
                "long_term_memory",
            ] {
                assert!(out.get(key).is_some(), "missing payload key {key}");
            }
        }

        let cases: [(MemoryToolKind, Value, &str); 11] = [
            (MemoryToolKind::Search, json!({}), "query is required"),
            (
                MemoryToolKind::Search,
                json!({"query": ""}),
                "query is required",
            ),
            (MemoryToolKind::Get, json!({}), "path is required"),
            (
                MemoryToolKind::Get,
                json!({"path": null}),
                "path is required",
            ),
            (MemoryToolKind::Read, json!({}), "path is required"),
            (MemoryToolKind::CodingRead, json!({}), "path is required"),
            (
                MemoryToolKind::Write,
                json!({"content": "body"}),
                "path is required",
            ),
            (
                MemoryToolKind::Write,
                json!({"path": "a.md"}),
                "content is required",
            ),
            (
                MemoryToolKind::CodingWrite,
                json!({"path": "a.md"}),
                "content is required",
            ),
            (
                MemoryToolKind::Edit,
                json!({"path": "a.md"}),
                "old_text is required",
            ),
            (
                MemoryToolKind::CodingEdit,
                json!({"path": "a.md", "old_text": "a"}),
                "new_text is required",
            ),
        ];
        for (kind, inputs, message) in cases {
            let out = invoke(&tool(kind, None), inputs).await;
            assert_eq!(out["success"], json!(false), "{}", kind.name());
            assert_eq!(out["error"], json!(message), "{}", kind.name());
        }
    }

    #[tokio::test]
    async fn every_kind_forwards_the_python_arguments() {
        let cases: [(MemoryToolKind, &str, Reply, &str); 9] = [
            (
                MemoryToolKind::Search,
                r#"{"query":"q"}"#,
                Reply::Search(json!({"results": []})),
                "search q 10 0.3 None Memory",
            ),
            (
                MemoryToolKind::Search,
                r#"{"query":"rust","max_results":3,"min_score":0.5,"session_key":"s1"}"#,
                Reply::Search(json!({"results": []})),
                r#"search rust 3 0.5 Some("s1") Memory"#,
            ),
            (
                MemoryToolKind::Get,
                r#"{"path":"a.md","from_line":2,"lines":3}"#,
                Reply::Read(read_data("a.md", "", 0, 0, 0)),
                "read Memory a.md 2 Some(3)",
            ),
            (
                MemoryToolKind::Read,
                r#"{"path":"a.md","offset":1}"#,
                Reply::Read(read_data("a.md", "", 0, 0, 0)),
                "read Memory a.md 1 None",
            ),
            (
                MemoryToolKind::CodingRead,
                r#"{"path":"s.md"}"#,
                Reply::Read(read_data("s.md", "", 0, 0, 0)),
                "read CodingMemory s.md 0 None",
            ),
            (
                MemoryToolKind::Write,
                r#"{"path":"a.md","content":"body"}"#,
                Reply::Write(write_data("a.md", false, false)),
                r#"write Memory a.md "body" false"#,
            ),
            (
                MemoryToolKind::CodingWrite,
                r#"{"path":"s.md","content":"","append":true}"#,
                Reply::Write(write_data("s.md", true, true)),
                r#"write CodingMemory s.md "" true"#,
            ),
            (
                MemoryToolKind::Edit,
                r#"{"path":"a.md","old_text":"a","new_text":"b"}"#,
                Reply::Edit(edit_data("a.md", 1)),
                r#"edit Memory a.md "a" "b""#,
            ),
            (
                MemoryToolKind::CodingEdit,
                r#"{"path":"s.md","old_text":"x","new_text":"y"}"#,
                Reply::Edit(edit_data("s.md", 1)),
                r#"edit CodingMemory s.md "x" "y""#,
            ),
        ];
        for (kind, inputs, reply, expected) in cases {
            let backend = reply.backend();
            invoke(&tool(kind, Some(backend.clone())), json_text(inputs)).await;
            assert_eq!(backend.calls(), [expected], "{}", kind.name());
        }
    }

    #[tokio::test]
    async fn every_kind_maps_the_python_payload_and_render() {
        let search_results = r#"{"results":[{"citation":"notes.md:3","score":0.83,"snippet":"prefers rust"},{"citation":"people.md:1","score":0.5,"snippet":"alice"}],"provider":"lite"}"#;
        let cases: [(MemoryToolKind, &str, Reply, &str, &str); 9] = [
            (
                MemoryToolKind::Search,
                r#"{"query":"rust","max_results":2}"#,
                Reply::Search(json_text(search_results)),
                search_results,
                "notes.md:3 (score 0.83)\nprefers rust\n\npeople.md:1 (score 0.50)\nalice",
            ),
            (
                MemoryToolKind::Get,
                r#"{"path":"notes.md","from_line":2,"lines":3}"#,
                Reply::Read(read_data("memory/notes.md", "# Notes", 9, 2, 4)),
                r##"{"path":"memory/notes.md","text":"# Notes","totalLines":9,"fromLine":2,"toLine":4,"disabled":false}"##,
                "# Notes",
            ),
            (
                MemoryToolKind::Read,
                r#"{"path":"notes.md","limit":2}"#,
                Reply::Read(read_data("notes.md", "one\ntwo", 5, 1, 2)),
                r#"{"success":true,"path":"notes.md","content":"one\ntwo","totalLines":5,"start_line":1,"end_line":2,"truncated":true}"#,
                "one\ntwo",
            ),
            (
                MemoryToolKind::CodingRead,
                r#"{"path":"style.md"}"#,
                Reply::Read(read_data("style.md", "---", 3, 1, 3)),
                r#"{"success":true,"path":"style.md","content":"---","totalLines":3,"start_line":1,"end_line":3,"truncated":false}"#,
                "---",
            ),
            (
                MemoryToolKind::Write,
                r#"{"path":"notes.md","content":"body"}"#,
                Reply::Write(write_data("notes.md", false, true)),
                r#"{"success":true,"path":"notes.md","fullPath":"/mem/notes.md","appended":false,"fileExisted":true}"#,
                "Wrote memory file notes.md.",
            ),
            (
                MemoryToolKind::CodingWrite,
                r#"{"path":"style.md","content":"","append":true}"#,
                Reply::Write(write_data("style.md", true, true)),
                r#"{"success":true,"path":"style.md","fullPath":"/mem/style.md","appended":true,"fileExisted":true}"#,
                "Appended to memory file style.md.",
            ),
            (
                MemoryToolKind::Edit,
                r#"{"path":"notes.md","old_text":"a","new_text":"b"}"#,
                Reply::Edit(edit_data("notes.md", 2)),
                r#"{"success":true,"path":"notes.md","replaced":2}"#,
                "Edited memory file notes.md.",
            ),
            (
                MemoryToolKind::CodingEdit,
                r#"{"path":"style.md","old_text":"x","new_text":"y"}"#,
                Reply::Edit(edit_data("style.md", 1)),
                r#"{"success":true,"path":"style.md","replaced":1}"#,
                "Edited memory file style.md.",
            ),
            (
                MemoryToolKind::Get,
                r#"{"path":"empty.md"}"#,
                Reply::Read(read_data("empty.md", "", 0, 0, 0)),
                r#"{"path":"empty.md","text":"","totalLines":0,"fromLine":0,"toLine":0,"disabled":false}"#,
                "No memory content found.",
            ),
        ];
        for (kind, inputs, reply, data, render) in cases {
            let memory = tool(kind, Some(reply.backend()));
            let out = invoke(&memory, json_text(inputs)).await;
            assert_eq!(out["success"], json!(true), "{}", kind.name());
            assert_eq!(out["data"], json_text(data), "{}", kind.name());
            assert_eq!(out["error"], Value::Null, "{}", kind.name());
            assert_eq!(memory.render_for_llm(&out), render, "{}", kind.name());
        }
    }

    #[tokio::test]
    async fn an_empty_search_renders_the_placeholder_and_disabled_fails() {
        let memory = tool(
            MemoryToolKind::Search,
            Some(Reply::Search(json!({})).backend()),
        );
        let out = invoke(&memory, json!({"query": "q"})).await;
        assert_eq!(out["success"], json!(true));
        assert_eq!(memory.render_for_llm(&out), "No matching memories found.");

        let data = json!({"disabled": true, "error": "memory is disabled", "results": []});
        let memory = tool(
            MemoryToolKind::Search,
            Some(Reply::Search(data.clone()).backend()),
        );
        let out = invoke(&memory, json!({"query": "q"})).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("memory is disabled"));
        assert_eq!(out["data"], data);
        assert_eq!(memory.render_for_llm(&out), "memory is disabled");
    }

    #[tokio::test]
    async fn backend_errors_become_failure_payloads() {
        let backend = Arc::new(FakeBackend {
            failure: Some("disk is full"),
            ..FakeBackend::default()
        });
        for kind in MemoryToolKind::ALL {
            let out = invoke(&tool(kind, Some(backend.clone())), valid_args(kind)).await;
            assert_eq!(out["success"], json!(false), "{}", kind.name());
            assert_eq!(out["error"], json!("disk is full"), "{}", kind.name());
            assert_eq!(out["data"], Value::Null, "{}", kind.name());
        }
    }

    #[test]
    fn render_falls_back_to_payload_text_for_non_envelopes() {
        let write = tool(MemoryToolKind::Write, None);
        assert_eq!(write.render_for_llm(&json!("plain")), "plain");
        assert_eq!(write.render_for_llm(&json!({"content": "raw"})), "raw");
        assert_eq!(
            write.render_for_llm(&json!({"success": false, "error": "boom"})),
            "boom"
        );

        let read = tool(MemoryToolKind::Read, None);
        assert_eq!(
            read.render_for_llm(&json!({"success": true, "data": {"content": "done"}})),
            "done"
        );
    }

    #[tokio::test]
    async fn a_cancelled_context_returns_the_cancelled_error() {
        let memory = tool(MemoryToolKind::Search, None);
        let ctx = ToolContext::new();
        ctx.cancellation.cancel();

        let error = Tool::invoke(&memory, json!({"query": "q"}), &ctx)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::Cancelled));
    }
}
