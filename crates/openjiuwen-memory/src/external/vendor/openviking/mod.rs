//! The `OpenViking` context-database adapter (`openviking`).
//!
//! The Python adapter drives the `OpenViking` REST API through a sync `httpx` client; this
//! Rust adapter speaks the same record surface through the shared
//! [`RestProvider`](super::super::rest::RestProvider), with the wire shape in [`spec`].
//! The transport owns validation, authentication, retries, failover, status mapping, and
//! cancellation, so this file is the vendor shape plus the session-based tool rail.
//!
//! # Wire contract
//!
//! | Operation | Request |
//! | --- | --- |
//! | search | `POST /api/v1/search/find` with `query`, `top_k` |
//! | add | `POST /api/v1/sessions/{session_id}/messages` with one `[Remember — <category>] <content>` text part |
//! | get | `GET /api/v1/content/read?uri=<uri>` |
//! | list | `GET /api/v1/fs/ls?uri=<uri>` (`options["root_uri"]`, `options["uri"]`, else `viking://`) |
//! | delete | no documented endpoint — typed `Unsupported` |
//! | update | no documented endpoint — typed `Unsupported` |
//! | clear | no documented endpoint — typed `Unsupported` |
//! | health | `GET /health` |
//!
//! Requests carry `Content-Type: application/json`, `X-OpenViking-Account`,
//! `X-OpenViking-User`, `X-OpenViking-Agent`, and — when a key is configured — the raw
//! key in `X-API-Key`, which matches the Python client. `OpenViking` is session based: a
//! write is a message on a session, and the memory is extracted and indexed when the
//! session is committed.
//!
//! # Addressed by uri
//!
//! `OpenViking` names everything with a `viking://` uri, so a record id is a uri: `get`
//! and `list` use it verbatim (`get` as the `uri` query parameter), and `add` reports a
//! client-generated id because the vendor assigns the uri only at commit time.
//!
//! # Rail
//!
//! The adapter keeps Python's five tools — `viking_search`, `viking_read`,
//! `viking_browse`, `viking_remember`, `viking_add_resource` — their payload shapes, and
//! its prompt block, dispatching them over the same transport so the wire paths are
//! exercised by the rail too. Four of them use endpoints the record SPI does not express
//! (a caller-chosen search mode, the abstract/overview content levels, resource indexing,
//! and the session commit), which [`send_rail`] sends directly.
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpRequest;
use serde_json::{json, Map, Value};

use super::super::rail;
use super::super::registry::{ProviderConfig, ProviderKind};
use super::super::rest::{body_value, field_str, Op, RestCall, RestProvider, WireSpec};
use crate::analyzer::ConversationTurn;
use crate::record::MemoryScope;

mod spec;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;

use spec::OpenVikingSpec;

/// Registry id of the adapter (the Python provider id).
const ID: &str = "openviking";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "OpenViking";
/// The dependency an unconfigured handle names.
const MISSING: &str = "the OpenViking API client";
/// Account sent when `options["account"]` names none.
const DEFAULT_ACCOUNT: &str = "default";
/// Agent sent when `options["agent_id"]` names none.
const DEFAULT_AGENT: &str = "hermes";
/// `tracing` target of this module's events.
const LOG_TARGET: &str = "openjiuwen_memory::external::vendor::openviking";
/// Top-k of a rail prefetch, matching the Python adapter.
const PREFETCH_TOP_K: usize = 5;
/// Items a rail prefetch renders from each of the first two context lists.
const PREFETCH_ITEMS_PER_LIST: usize = 3;
/// Title of a rendered rail prefetch.
const PREFETCH_TITLE: &str = "## OpenViking Context";
/// `viking_search` result ceiling.
const MAX_TOOL_TOP_K: i64 = 50;
/// Entries `viking_browse` renders.
const MAX_BROWSE_ENTRIES: usize = 50;
/// Longest turn text `sync_turn` posts (Python's `[:4000]`).
const MAX_TURN_CHARS: usize = 4000;
/// Message `viking_remember` reports after storing a fact.
const REMEMBER_MESSAGE: &str = "Memory recorded. Will be extracted and indexed on session commit.";
/// Python's prompt block for the five tools.
const SYSTEM_PROMPT_BLOCK: &str = "# OpenViking Memory\n\nUse `viking_search` to find knowledge (modes: auto/fast/deep).\nUse `viking_read` to read content at a viking:// URI (levels: abstract/overview/full).\nUse `viking_browse` to navigate the knowledge structure.\nUse `viking_remember` to explicitly store facts.\nUse `viking_add_resource` to index URLs/documents.";
/// Rail tool that searches the whole knowledge base.
const SEARCH_TOOL: &str = "viking_search";
/// Rail tool that reads a `viking://` uri.
const READ_TOOL: &str = "viking_read";
/// Rail tool that navigates the knowledge tree.
const BROWSE_TOOL: &str = "viking_browse";
/// Rail tool that stores one fact.
const REMEMBER_TOOL: &str = "viking_remember";
/// Rail tool that indexes a url or document.
const ADD_RESOURCE_TOOL: &str = "viking_add_resource";

/// The one [`WireSpec`] instance every `OpenViking` adapter shares.
pub(super) static SPEC: &dyn WireSpec = &OpenVikingSpec;

/// Handle for the `OpenViking` context database (`openviking`).
///
/// Construction never fails: without an `endpoint` the handle is inert and every
/// operation names `MISSING` verbatim.
pub struct OpenVikingMemoryProvider {
    /// The REST backend driven by [`SPEC`].
    pub(super) inner: RestProvider,
}

impl OpenVikingMemoryProvider {
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
        ProviderKind::OpenViking
    }

    /// The wire shape this adapter speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.inner.spec()
    }

    /// The session the configured handshake belongs to (`options["session_id"]`).
    fn session_id(&self) -> Option<String> {
        super::option_text(self.config(), "session_id")
    }

    /// The configured session id, or the typed validation error naming the requirement.
    fn require_session(&self, purpose: &str) -> Result<String> {
        self.session_id()
            .ok_or_else(|| spec::session_required(purpose))
    }

    /// Posts one text message on `session_id`.
    async fn post_message(&self, session_id: &str, body: Value) -> Result<Value> {
        let call = RestCall::post(spec::messages_path(session_id)).with_body(body);
        send_rail(&self.inner, Op::Add, &call).await
    }

    /// Runs the `viking_search` tool.
    async fn tool_search(&self, args: &Value) -> Value {
        let map = rail::args_object(args);
        let Some(query) = rail::string_arg(map, "query")
            .map(str::trim)
            .filter(|query| !query.is_empty())
        else {
            return rail::tool_error("missing required parameter: query");
        };
        let mut body = Map::new();
        body.insert("query".to_owned(), Value::String(query.to_owned()));
        if let Some(mode) = rail::string_arg(map, "mode")
            .map(str::trim)
            .filter(|mode| !mode.is_empty() && !mode.eq_ignore_ascii_case("auto"))
        {
            body.insert("mode".to_owned(), Value::String(mode.to_owned()));
        }
        if let Some(scope) = rail::string_arg(map, "scope")
            .map(str::trim)
            .filter(|scope| !scope.is_empty())
        {
            body.insert("target_uri".to_owned(), Value::String(scope.to_owned()));
        }
        let top_k = rail::integer_arg(map, "top_k")
            .or_else(|| rail::integer_arg(map, "limit"))
            .map(|top_k| top_k.clamp(1, MAX_TOOL_TOP_K));
        if let Some(top_k) = top_k {
            body.insert("top_k".to_owned(), Value::from(top_k));
        }
        let call = RestCall::post(spec::FIND_PATH).with_body(Value::Object(body));
        let answer = match send_rail(&self.inner, Op::Search, &call).await {
            Ok(answer) => answer,
            Err(error) => return rail::tool_failure(SEARCH_TOOL, &error),
        };
        let mut entries: Vec<(f32, Value)> = spec::find_items(&answer)
            .into_iter()
            .map(|(kind, item)| {
                let score = spec::item_score(item);
                let entry = json!({
                    "uri": field_str(item, &["uri"]).unwrap_or_default(),
                    "type": kind,
                    "score": round3(score),
                    "abstract": field_str(item, &["abstract"]).unwrap_or_default(),
                });
                (score, entry)
            })
            .collect();
        entries.sort_by(|left, right| right.0.total_cmp(&left.0));
        let results: Vec<Value> = entries.into_iter().map(|(_score, entry)| entry).collect();
        let total = spec::answer_total(&answer).unwrap_or(results.len() as u64);
        json!({ "results": results, "total": total })
    }

    /// Runs the `viking_read` tool.
    async fn tool_read(&self, args: &Value) -> Value {
        let map = rail::args_object(args);
        let Some(uri) = rail::string_arg(map, "uri")
            .map(str::trim)
            .filter(|uri| !uri.is_empty())
        else {
            return rail::tool_error("missing required parameter: uri");
        };
        let level = rail::string_arg(map, "level")
            .or_else(|| rail::string_arg(map, "detail"))
            .map(str::trim)
            .filter(|level| !level.is_empty())
            .unwrap_or("overview");
        let path = match level {
            "abstract" => spec::ABSTRACT_PATH,
            "full" => spec::READ_PATH,
            _ => spec::OVERVIEW_PATH,
        };
        let call = RestCall::get(path).query("uri", uri);
        let answer = match send_rail(&self.inner, Op::Get, &call).await {
            Ok(answer) => answer,
            Err(error) => return rail::tool_failure(READ_TOOL, &error),
        };
        let content = spec::result_ref(&answer).map_or_else(String::new, |result| {
            result
                .as_str()
                .or_else(|| result.get("content").and_then(Value::as_str))
                .unwrap_or_default()
                .to_owned()
        });
        json!({ "uri": uri, "level": level, "content": spec::truncate_content(&content) })
    }

    /// Runs the `viking_browse` tool.
    async fn tool_browse(&self, args: &Value) -> Value {
        let map = rail::args_object(args);
        let action = rail::string_arg(map, "action")
            .map(str::trim)
            .filter(|action| !action.is_empty())
            .unwrap_or("list");
        let path = rail::string_arg(map, "path")
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .unwrap_or(spec::DEFAULT_ROOT_URI);
        let endpoint = match action {
            "tree" => spec::TREE_PATH,
            "stat" => spec::STAT_PATH,
            _ => spec::LS_PATH,
        };
        let call = RestCall::get(endpoint).query("uri", path);
        let answer = match send_rail(&self.inner, Op::List, &call).await {
            Ok(answer) => answer,
            Err(error) => return rail::tool_failure(BROWSE_TOOL, &error),
        };
        let Some(result) = spec::result_ref(&answer) else {
            return json!({ "path": path });
        };
        if action == "stat" {
            return result.clone();
        }
        let entries: Vec<Value> = match result.as_array() {
            Some(entries) => entries.as_slice(),
            None => &[],
        }
        .iter()
        .take(MAX_BROWSE_ENTRIES)
        .map(|entry| {
            json!({
                "name": field_str(entry, &["rel_path", "name"]).unwrap_or_default(),
                "uri": field_str(entry, &["uri"]).unwrap_or_default(),
                "type": if spec::is_directory(entry) { "dir" } else { "file" },
                "abstract": field_str(entry, &["abstract"]).unwrap_or_default(),
            })
        })
        .collect();
        json!({ "path": path, "entries": entries })
    }

    /// Runs the `viking_remember` tool.
    async fn tool_remember(&self, args: &Value) -> Value {
        let map = rail::args_object(args);
        let Some(content) = rail::string_arg(map, "content")
            .map(str::trim)
            .filter(|content| !content.is_empty())
        else {
            return rail::tool_error("missing required parameter: content");
        };
        let session_id = match self.require_session("store a memory") {
            Ok(session_id) => session_id,
            Err(error) => return rail::tool_failure(REMEMBER_TOOL, &error),
        };
        let text = match rail::string_arg(map, "category")
            .map(str::trim)
            .filter(|category| !category.is_empty())
        {
            Some(category) => format!("[Remember — {category}] {content}"),
            None => format!("[Remember] {content}"),
        };
        let body = json!({ "role": "user", "parts": [{ "type": "text", "text": text }] });
        match self.post_message(&session_id, body).await {
            Ok(_answer) => json!({ "status": "stored", "message": REMEMBER_MESSAGE }),
            Err(error) => rail::tool_failure(REMEMBER_TOOL, &error),
        }
    }

    /// Runs the `viking_add_resource` tool.
    async fn tool_add_resource(&self, args: &Value) -> Value {
        let map = rail::args_object(args);
        let Some(url) = rail::string_arg(map, "url")
            .map(str::trim)
            .filter(|url| !url.is_empty())
        else {
            return rail::tool_error("missing required parameter: url");
        };
        let call = RestCall::post(spec::RESOURCES_PATH).with_body(json!({ "path": url }));
        let answer = match send_rail(&self.inner, Op::Add, &call).await {
            Ok(answer) => answer,
            Err(error) => return rail::tool_failure(ADD_RESOURCE_TOOL, &error),
        };
        let root_uri = spec::result_ref(&answer)
            .and_then(|result| result.get("root_uri"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        json!({ "status": "added", "root_uri": root_uri })
    }
}

impl fmt::Debug for OpenVikingMemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenVikingMemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config().endpoint)
            .finish_non_exhaustive()
    }
}

/// Sends one rail request through the transport the spec uses.
///
/// The rail drives endpoints the record SPI does not express, so this helper reuses the
/// provider's shared [`HttpClient`](openjiuwen_common::http::HttpClient), endpoint list,
/// vendor headers, and resolved authentication scheme instead of building a second
/// transport. A non-`2xx` answer is mapped with the provider's own status mapping, so the
/// rail sees the same error kinds as the record SPI; an empty body decodes as
/// [`Value::Null`].
async fn send_rail(provider: &RestProvider, op: Op, call: &RestCall) -> Result<Value> {
    let base = provider
        .endpoints()
        .first()
        .ok_or_else(spec::unconfigured)?;
    let client = provider.client().ok_or_else(|| {
        Error::transport(format!(
            "the {DISPLAY} memory provider could not build its HTTP transport"
        ))
    })?;
    let mut request = HttpRequest::new(
        call.method.clone(),
        RestCall::absolute_url(base, &call.path),
    )
    .queries(call.query.clone())
    .with_headers(SPEC.headers(provider.config()))
    .with_headers(call.headers.clone());
    let api_key = provider.config().api_key.as_deref().map_or("", str::trim);
    if let Some((name, value)) = provider.auth().header_for(api_key) {
        request = request.header(name, value);
    }
    if let Some((name, value)) = provider.auth().query_for(api_key) {
        request = request.query(name, value);
    }
    if let Some(body) = &call.body {
        request = request.json(body)?;
    }
    let response = client.execute(request).await?;
    if !response.is_success() {
        return Err(provider.status_error(op, response.status, &response.url, &response.text()));
    }
    body_value(&response)
}

/// Rounds a score to three decimals, as the Python tool does.
fn round3(score: f32) -> f64 {
    if score.is_finite() {
        (f64::from(score) * 1000.0).round() / 1000.0
    } else {
        0.0
    }
}

/// Renders a find answer as the prefetch prompt block.
///
/// At most [`PREFETCH_ITEMS_PER_LIST`] items with a non-empty abstract are rendered from
/// the memory list and then from the resource list, in the order the vendor returns them,
/// each as `- [<score>] <abstract> (<uri>)` with the score at three decimals. Nothing to
/// render — no items, no abstracts, or an empty `result` — yields an empty string.
fn render_context(answer: &Value) -> String {
    let mut lines = Vec::new();
    for list in ["memories", "resources"] {
        for item in spec::list_items(answer, list)
            .iter()
            .take(PREFETCH_ITEMS_PER_LIST)
        {
            let abstract_text = field_str(item, &["abstract"]).unwrap_or_default();
            if abstract_text.is_empty() {
                continue;
            }
            let uri = field_str(item, &["uri"]).unwrap_or_default();
            let score = spec::item_score(item);
            lines.push(format!("- [{score:.3}] {abstract_text} ({uri})"));
        }
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("{PREFETCH_TITLE}\n{}", lines.join("\n"))
    }
}

/// The five `viking_*` rail tool specs, mirroring the Python adapter.
fn tool_schemas() -> Vec<Value> {
    vec![
        rail::tool_schema(
            SEARCH_TOOL,
            "Search the whole OpenViking knowledge base.",
            json!({
                "query": rail::string_property("What to search for."),
                "mode": rail::enum_property("Search mode (default: auto).", &["auto", "fast", "deep"]),
                "top_k": rail::integer_property("Maximum number of results (default: 10)."),
            }),
            &["query"],
        ),
        rail::tool_schema(
            READ_TOOL,
            "Read the content at a viking:// URI.",
            json!({
                "uri": rail::string_property("The viking:// URI to read."),
                "detail": rail::enum_property(
                    "Detail level (default: overview).",
                    &["abstract", "overview", "full"],
                ),
            }),
            &["uri"],
        ),
        rail::tool_schema(
            BROWSE_TOOL,
            "Navigate the knowledge structure.",
            json!({
                "action": rail::enum_property("The browse action.", &["list", "tree", "stat"]),
                "path": rail::string_property("The path to browse (default: viking://)."),
            }),
            &["action"],
        ),
        rail::tool_schema(
            REMEMBER_TOOL,
            "Explicitly store one fact or preference.",
            json!({
                "content": rail::string_property("The fact to remember."),
                "category": rail::enum_property(
                    "The memory category.",
                    &["preference", "entity", "event", "case", "pattern"],
                ),
            }),
            &["content"],
        ),
        rail::tool_schema(
            ADD_RESOURCE_TOOL,
            "Index a URL or document so it can be searched later.",
            json!({
                "url": rail::string_property("The URL or file path to index."),
                "title": rail::string_property("Optional title."),
            }),
            &["url"],
        ),
    ]
}

implement_vendor_provider!(
    OpenVikingMemoryProvider,
    inner;

    /// The five `viking_*` tool specs (search, read, browse, remember, add resource).
    fn get_tool_schemas(&self) -> Vec<Value> {
        tool_schemas()
    }

    /// The scope rail calls use: the configured user/scope/session/agent fallback.
    fn default_scope(&self) -> MemoryScope {
        super::scope_from_options(self.config())
    }

    /// Python's five-line `OpenViking` prompt block, verbatim.
    fn system_prompt_block(&self) -> String {
        SYSTEM_PROMPT_BLOCK.to_owned()
    }

    /// Recalls context for the current turn and renders it as a prompt block.
    ///
    /// One `POST /api/v1/search/find` with `top_k = 5`; a blank query, an empty answer, or
    /// a failed call renders nothing (the failure is logged at `debug`).
    async fn prefetch(&self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            return String::new();
        }
        let call = RestCall::post(spec::FIND_PATH).with_body(json!({
            "query": query,
            "top_k": PREFETCH_TOP_K,
        }));
        match send_rail(&self.inner, Op::Search, &call).await {
            Ok(answer) => render_context(&answer),
            Err(error) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    tool = "prefetch",
                    %error,
                    "the OpenViking prefetch failed; the turn continues without recalled context"
                );
                String::new()
            }
        }
    }

    /// Dispatches the five `viking_*` rail tools over the transport.
    ///
    /// Python's payloads are kept verbatim; a backend error becomes
    /// `{"error": <message>}` instead of propagating, because the caller is a model.
    async fn handle_tool_call(&self, tool: &str, args: &Value) -> Value {
        match tool {
            SEARCH_TOOL => self.tool_search(args).await,
            READ_TOOL => self.tool_read(args).await,
            BROWSE_TOOL => self.tool_browse(args).await,
            REMEMBER_TOOL => self.tool_remember(args).await,
            ADD_RESOURCE_TOOL => self.tool_add_resource(args).await,
            other => json!({ "error": format!("unknown tool: {other}") }),
        }
    }

    /// Records one exchange as session messages.
    ///
    /// A blank user message is a no-op. The user message is posted first, then the
    /// assistant message when it holds text; each body is `{"role", "content"}` with the
    /// text truncated to 4000 characters, as the Python adapter sends it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when no session id is configured (the turn has
    /// nowhere to go) and the transport error when a post fails.
    async fn sync_turn(&self, user_msg: &str, assistant_msg: &str) -> Result<()> {
        if user_msg.trim().is_empty() {
            return Ok(());
        }
        let session_id = self.require_session("record a conversation turn")?;
        let user_body = json!({
            "role": "user",
            "content": spec::truncate_chars(user_msg, MAX_TURN_CHARS),
        });
        self.post_message(&session_id, user_body).await?;
        if !assistant_msg.trim().is_empty() {
            let assistant_body = json!({
                "role": "assistant",
                "content": spec::truncate_chars(assistant_msg, MAX_TURN_CHARS),
            });
            self.post_message(&session_id, assistant_body).await?;
        }
        Ok(())
    }

    /// Commits the session so its turn messages are extracted and indexed.
    ///
    /// Without a configured session id there is nothing to commit and the call is a no-op.
    ///
    /// # Errors
    ///
    /// Returns the transport error when the commit fails (after logging it at `debug`), so
    /// a caller can tell "the session is not persisted" from "there was nothing to commit".
    async fn on_session_end(&self, _messages: &[ConversationTurn]) -> Result<()> {
        let Some(session_id) = self.session_id() else {
            return Ok(());
        };
        let call = RestCall::post(spec::commit_path(&session_id)).with_body(json!({}));
        match send_rail(&self.inner, Op::Add, &call).await {
            Ok(_answer) => Ok(()),
            Err(error) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    session = %session_id,
                    %error,
                    "the OpenViking session commit failed"
                );
                Err(error)
            }
        }
    }
);
