//! The `skill_index` toolkit: the model-facing tool, its card, and its typed result.
//!
//! Ports Python `openjiuwen.symphony.discovery.directory_toolkit.InstalledSkillsDirectoryToolkit`
//! plus `_DirectoryFunction.invoke` and `_tool_card`, on top of the view, render, and
//! validation layers. One toolkit owns one live [`SkillFs`] catalog and one session's
//! state: the pagination cursors, the incremental-notice cursor, and the `META.md` paths
//! an earlier result observed.
//!
//! # Two entry points
//!
//! [`SkillIndexToolkit::invoke`] is the model-facing shape: exactly
//! `{"operation", "category", "query", "cursor"}`. An unknown key is rejected with
//! Python's message instead of being ignored, a blank `category` is dropped, a string
//! `query` becomes a one-element array, and an empty `query` on `list` is dropped. It
//! always uses Python's simple shape (details view for `list`, one page of 10 hits per
//! query for `search`).
//!
//! [`SkillIndexToolkit::execute`] takes a typed [`SkillIndexRequest`], so a host can drive
//! the advanced shape (paths, `match`, `result`, pipelines, `output_mode`, explicit
//! budgets). A request whose advanced fields are all at their defaults takes the same
//! simple path as [`SkillIndexToolkit::invoke`].
//!
//! # Execution order
//!
//! Python's `_execute`, in order: resolve the category against the view, compute the
//! output budget, dispatch the operation, apply the pipeline, page, fit the rows into the
//! remaining budget, add the empty-result message, add the `[next]` footer or the
//! `[truncated]` note, hand the model content to the incremental notice, and record the
//! observed paths.
//!
//! # Cursors
//!
//! A cursor token is deterministic: the URL-safe unpadded base64 of the first 16 bytes of
//! SHA-256 over the canonical JSON of the continuation state (operation, category,
//! queries, offset, page size, and the catalog fingerprint), which is 22 characters — well
//! inside [`CURSOR_MAX_CHARS`]. The state itself lives in a [`MAX_CURSORS`]-entry
//! least-recently-used map on the toolkit, so a token that was evicted is "unknown or
//! expired". Python instead mints a random `token_urlsafe(12)` per page; a deterministic
//! token keeps an identical request byte-identical.
//!
//! # No process-global state
//!
//! Python keeps its notice cursors in a module-global registry keyed by tool id. Every
//! cursor here (page cursors, the notice cursor, the observed paths) is a field on the
//! toolkit, so two toolkits never share state and a test never has to clear a global.
//!
//! Everything is offline and deterministic: no clock, no randomness, no hash-map
//! iteration order.

use std::collections::BTreeSet;

use base64::Engine as _;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::discovery::settings::DiscoverySettings;
use crate::discovery::toolkit::{
    self, IncrementalNoticeSession, SkillCard, SkillCards, DEFAULT_NOTICE_MAX_CHARS,
};
use crate::discovery::SkillFs;
use crate::error::{validation, Result};

use super::ops;
use super::render::{
    append_note, candidate_tokens, compact, empty_message, fit_rows, row_count, ListView,
    ResponseSummary, Row,
};
use super::validate::{
    apply_simple_defaults, output_budget, validate_category_paths, validate_paths, Operation,
    OutputMode, ReadMode, RequestIssue, SearchMatch, SkillIndexRequest, ValidatedRequest,
    MAX_OUTPUT_CHARS, MIN_OUTPUT_CHARS,
};
use super::view::SkillDirectoryView;

/// The tool name the model sees ([`crate::discovery::notice::SKILL_INDEX_TOOL_NAME`]).
pub const SKILL_INDEX_TOOL_NAME: &str = crate::discovery::notice::SKILL_INDEX_TOOL_NAME;

/// The maximum number of page cursors one toolkit keeps; the least recently used is
/// dropped past it (Python's `_MAX_CURSORS`).
pub const MAX_CURSORS: usize = 32;

/// The maximum characters a cursor token may have (Python's bound on the wire value).
pub const CURSOR_MAX_CHARS: usize = 128;

/// The characters reserved for the `[next]` footer while fitting rows.
pub const CURSOR_FOOTER_CHARS: usize = 96;

/// The schema version the cursor token hashes.
pub const CURSOR_STATE_VERSION: u64 = 1;

/// The bytes of the SHA-256 digest a cursor token keeps.
pub const CURSOR_TOKEN_BYTES: usize = 16;

/// The operations the model-facing tool accepts.
pub const MODEL_OPERATIONS: [Operation; 2] = [Operation::List, Operation::Search];

/// The tool description, verbatim from Python's `_tool_card`.
const TOOL_DESCRIPTION: &str = "Read-only Skill catalogue, not a filesystem tool. [category] is \
a virtual group, not a disk directory or Skill; counts include descendants. [skill] names a \
Skill; path is its real SKILL.md for file tools, not an input here. Calls do not change a \
working directory. Search returns ranked text matches, not verified capabilities. No query \
translation.";

/// The `operation` property description, verbatim from Python's `_tool_card`.
const OPERATION_DESCRIPTION: &str = "list: direct subcategories and Skills. search: names, \
aliases, descriptions and SKILL.md including descendants; not other package files.";

/// The `category` property description, verbatim from Python's `_tool_card`.
const CATEGORY_DESCRIPTION: &str = "Scope: returned category name or A > B chain, never a disk \
path. Omit for root/global on each call.";

/// The `query` property description, verbatim from Python's `_tool_card`.
const QUERY_DESCRIPTION: &str = "Search only. Array even for one query: [\"PDF OCR\"]. Keep \
task-specific terms; use the language of Skill descriptions (often English). Independent \
queries, interleaved and deduplicated. Terms need not all match; no Skill matches yields \
category hints.";

/// The `cursor` property description, verbatim from Python's `_tool_card`.
const CURSOR_DESCRIPTION: &str = "Copy [next] unchanged. To change query/category, omit cursor.";

/// The keys the model-facing tool accepts.
const ACCEPTED_KEYS: [&str; 4] = ["operation", "category", "query", "cursor"];

/// The `skillfs_layout` a result reports; the port has no pinned index snapshot.
const LAYOUT_FLAT: &str = "flat";

/// The `index_state` a result reports; the port builds no persistent index snapshot.
const INDEX_MISSING: &str = "missing";

/// One toolkit's configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillIndexConfig {
    /// The session scope the tool id and the notice cursor are keyed by; required.
    pub session_scope: String,
    /// The default output budget in characters, within
    /// [`MIN_OUTPUT_CHARS`]`..=`[`MAX_OUTPUT_CHARS`].
    pub max_output_chars: usize,
    /// The character budget of an incremental notice reminder.
    pub incremental_notice_max_chars: usize,
}

impl Default for SkillIndexConfig {
    fn default() -> Self {
        let settings = DiscoverySettings::default();
        Self {
            session_scope: String::new(),
            max_output_chars: settings.max_output_chars,
            incremental_notice_max_chars: DEFAULT_NOTICE_MAX_CHARS,
        }
    }
}

impl SkillIndexConfig {
    /// Builds a config for `session_scope` with the default budgets
    /// ([`DiscoverySettings::default`]'s `max_output_chars` and
    /// [`DEFAULT_NOTICE_MAX_CHARS`]).
    #[must_use]
    pub fn new(session_scope: impl Into<String>) -> Self {
        Self {
            session_scope: session_scope.into(),
            ..Self::default()
        }
    }
}

/// The card the host publishes for one `skill_index` tool instance.
///
/// `input_schema` is Python's pinned `_tool_card` `input_params` object: `operation` is the
/// only required property, `query` is an array of 1–8 strings of 1–512 characters,
/// `category` and `cursor` are non-empty strings, and `additionalProperties` is `false`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillIndexToolCard {
    /// The per-session tool id (`skill_index__<sha256 hex>`).
    pub id: String,
    /// The tool name ([`SKILL_INDEX_TOOL_NAME`]).
    pub name: String,
    /// The model-facing description of what the tool is and is not.
    pub description: String,
    /// The pinned JSON Schema of the accepted arguments.
    pub input_schema: Value,
    /// Whether the tool may run in parallel with other tools; always `false`.
    pub parallel_safe: bool,
    /// Whether the tool is stateless; always `false` (it keeps page and notice cursors).
    pub stateless: bool,
}

impl openjiuwen_common::card::Card for SkillIndexToolCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some(&self.description)
    }
}

/// One `skill_index` response: the delivered text plus the diagnostics a host can log.
///
/// `content` is the concise text the tool returns (the paged body plus any `[next]` or
/// `[truncated]` note and, when it fitted, the incremental notice); `model_content` is the
/// same text with the summary header. The counters mirror Python's `detailed_output`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillIndexResult {
    /// The concise result text.
    pub content: String,
    /// The model-facing text including the summary header.
    pub model_content: String,
    /// The operation that produced this result.
    pub operation: Operation,
    /// The fitted rows, in render order.
    pub rows: Vec<Row>,
    /// The Skill ids the result observed, in row order.
    pub observed_skill_ids: Vec<String>,
    /// Whether the output budget dropped a row or a row was paged away.
    pub truncated: bool,
    /// Why the result was shortened (`max_output_chars`), when it was.
    pub truncation_reason: Option<String>,
    /// How many entries the operation matched, before paging.
    pub total_count: usize,
    /// How many rows the fitted body shows (or the pipeline count).
    pub shown_count: usize,
    /// Whether the caller has seen everything this request could return.
    pub result_complete: bool,
    /// The token that asks for the next page, when one exists.
    pub next_cursor: Option<String>,
    /// The output budget the response was fitted into; `None` when truncation is off.
    pub effective_max_output_chars: Option<usize>,
    /// The catalog layout; always [`LAYOUT_FLAT`].
    pub skillfs_layout: String,
    /// The index state; always [`INDEX_MISSING`].
    pub index_state: String,
    /// How many Skills the catalog holds.
    pub candidate_count: usize,
    /// The estimated model tokens of the catalog's candidate lines.
    pub estimated_candidate_tokens: usize,
    /// The token budget for skill candidates ([`DiscoverySettings::default`]).
    pub candidate_budget_tokens: usize,
}

/// The model-facing `skill_index` tool over one live catalog.
#[derive(Debug)]
pub struct SkillIndexToolkit {
    /// The catalog every operation reads.
    catalog: SkillFs,
    /// The virtual category tree built from the catalog.
    view: SkillDirectoryView,
    /// The budgets and the session scope.
    config: SkillIndexConfig,
    /// The per-session tool id.
    tool_id: String,
    /// The incremental notice cursor for this session.
    notice: IncrementalNoticeSession,
    /// The page cursors, least recently used first.
    cursors: CursorStore,
    /// The `META.md` paths an earlier result observed.
    observed_meta_paths: BTreeSet<String>,
    /// The Skill ids an earlier result observed.
    observed_skills: BTreeSet<String>,
}

impl SkillIndexToolkit {
    /// Opens a toolkit over `catalog` for one session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `session_scope` when it is blank, naming
    /// `max_output_chars` when it is outside
    /// [`MIN_OUTPUT_CHARS`]`..=`[`MAX_OUTPUT_CHARS`], and naming
    /// `incremental_notice_max_chars` when it is zero.
    pub fn new(catalog: SkillFs, config: SkillIndexConfig) -> Result<Self> {
        let scope = config.session_scope.trim();
        if scope.is_empty() {
            return Err(validation("session_scope", "must be non-empty"));
        }
        if !(MIN_OUTPUT_CHARS..=MAX_OUTPUT_CHARS).contains(&config.max_output_chars) {
            return Err(validation(
                "max_output_chars",
                format!("must be between {MIN_OUTPUT_CHARS} and {MAX_OUTPUT_CHARS}"),
            ));
        }
        if config.incremental_notice_max_chars == 0 {
            return Err(validation(
                "incremental_notice_max_chars",
                "must be positive",
            ));
        }
        let tool_id = toolkit::tool_id(scope, SKILL_INDEX_TOOL_NAME)?;
        let cards = catalog_cards(&catalog);
        let notice =
            IncrementalNoticeSession::new(scope, &cards, config.incremental_notice_max_chars)?;
        let view = SkillDirectoryView::from_catalog(&catalog);
        Ok(Self {
            catalog,
            view,
            config,
            tool_id,
            notice,
            cursors: CursorStore::default(),
            observed_meta_paths: BTreeSet::new(),
            observed_skills: BTreeSet::new(),
        })
    }

    /// The per-session tool id.
    #[must_use]
    pub fn tool_id(&self) -> &str {
        &self.tool_id
    }

    /// The catalog every operation reads.
    #[must_use]
    pub fn catalog(&self) -> &SkillFs {
        &self.catalog
    }

    /// The virtual category tree built from the catalog.
    #[must_use]
    pub fn view(&self) -> &SkillDirectoryView {
        &self.view
    }

    /// The card the host publishes for this tool.
    #[must_use]
    pub fn card(&self) -> SkillIndexToolCard {
        SkillIndexToolCard {
            id: self.tool_id.clone(),
            name: SKILL_INDEX_TOOL_NAME.to_string(),
            description: TOOL_DESCRIPTION.to_string(),
            input_schema: tool_input_schema(),
            parallel_safe: false,
            stateless: false,
        }
    }

    /// Runs one model-facing call.
    ///
    /// Accepts exactly `operation`, `category`, `query`, and `cursor`; `cursor` resumes an
    /// earlier page (and rejects an operation, category, or query that differs from the
    /// original request, or a catalog that changed since the page was minted).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown argument key, a value of the wrong
    /// JSON type, an invalid request, or a cursor that cannot be resumed.
    pub fn invoke(&mut self, arguments: &Value) -> Result<SkillIndexResult> {
        let Value::Object(map) = arguments else {
            return Err(validation(
                "arguments",
                "must be a JSON object with operation, category, query, or cursor",
            ));
        };
        let mut unexpected: Vec<&str> = map
            .keys()
            .map(String::as_str)
            .filter(|key| !ACCEPTED_KEYS.contains(key))
            .collect();
        if !unexpected.is_empty() {
            unexpected.sort_unstable();
            return Err(validation(
                "arguments",
                format!(
                    "Unknown skill_index arguments: {}. Only operation, category, query, cursor \
are accepted. Use {{\"operation\":\"list\",\"category\":\"A > B\"}} or \
{{\"operation\":\"search\",\"query\":[\"keywords\"],\"category\":\"A > B\"}}; omit category \
for root/global. Disk paths and globs belong to file tools. For the next page, copy [next] \
unchanged.",
                    unexpected.join(", ")
                ),
            ));
        }
        let operation = map
            .get("operation")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let category = match map.get("category") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => {
                let trimmed = value.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            }
            Some(_) => return Err(validation("category", "must be a non-empty string")),
        };
        let mut query = match map.get("query") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(vec![value.clone()]),
            Some(Value::Array(items)) => {
                let mut values: Vec<String> = Vec::with_capacity(items.len());
                for item in items {
                    let Value::String(value) = item else {
                        return Err(validation("query", "must be an array of strings"));
                    };
                    values.push(value.clone());
                }
                Some(values)
            }
            Some(_) => return Err(validation("query", "must be an array of strings")),
        };
        if operation == Operation::List.as_str()
            && query.as_deref().is_some_and(<[String]>::is_empty)
        {
            query = None;
        }
        if let Some(Value::String(token)) = map.get("cursor") {
            return self.resume(&operation, token, category.as_deref(), query.as_deref());
        }
        if let Some(value) = map.get("cursor") {
            if !value.is_null() {
                return Err(validation("cursor", "must be a non-empty string"));
            }
        }
        let request = SkillIndexRequest {
            operation,
            category,
            query,
            ..SkillIndexRequest::default()
        };
        self.execute(&request)
    }

    /// Runs one typed request.
    ///
    /// A request whose advanced fields are all at their defaults takes Python's simple
    /// shape ([`apply_simple_defaults`]); every other request is validated with
    /// [`ops::validate_request`] and keeps its own paging, pipeline, and output mode.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the first rejected field, or the operation's
    /// own error (an unknown category or Skill, an invalid pipeline).
    pub fn execute(&mut self, request: &SkillIndexRequest) -> Result<SkillIndexResult> {
        let validated = if is_simple(request) {
            apply_simple_defaults(request).map_err(|issues| first_issue(&issues))?
        } else {
            if request.limit.is_some() {
                return Err(validation(
                    "limit",
                    "cannot be combined with advanced arguments",
                ));
            }
            ops::validate_request(request).map_err(|issues| first_issue(&issues))?
        };
        self.run(&validated)
    }

    /// Drops every page cursor; the notice cursor and the observed paths stay.
    pub fn close(&mut self) {
        self.cursors.clear();
    }

    /// Re-scans the catalog root and keeps this session's cursors and observed paths.
    ///
    /// Python's toolkit reads whatever `SkillFs` the host hands it, so a host that
    /// re-scans in place makes the tool see new Skills immediately. Rust's
    /// [`SkillFs`] is an immutable snapshot, so an explicit re-scan is the seam: call
    /// this after the host adds, edits, or removes a `SKILL.md`, and the next call sees
    /// the new catalog, reports the added Skills in its notice reminder, and rejects any
    /// page cursor minted against the old fingerprint.
    ///
    /// # Errors
    ///
    /// Propagates [`SkillFs::open_with`]'s failure (an unusable root). The previous
    /// catalog stays in place when the re-scan fails.
    pub async fn reload(&mut self) -> Result<usize> {
        let config = self.catalog.config().clone();
        let root = self.catalog.root().to_path_buf();
        let catalog = SkillFs::open_with(config, &root).await?;
        let count = catalog.records().len();
        self.view = SkillDirectoryView::from_catalog(&catalog);
        self.catalog = catalog;
        Ok(count)
    }

    /// How many page cursors the toolkit holds.
    #[must_use]
    pub fn cursor_count(&self) -> usize {
        self.cursors.len()
    }

    /// Resumes an earlier page from its cursor token.
    fn resume(
        &mut self,
        operation: &str,
        token: &str,
        category: Option<&str>,
        query: Option<&[String]>,
    ) -> Result<SkillIndexResult> {
        let Some(parsed) =
            Operation::parse(operation).filter(|parsed| MODEL_OPERATIONS.contains(parsed))
        else {
            return Err(validation(
                "operation",
                format!(
                    "must be one of: {}",
                    MODEL_OPERATIONS.map(Operation::as_str).join(", ")
                ),
            ));
        };
        let token = token.trim();
        if token.is_empty() {
            return Err(validation("cursor", "must be a non-empty string"));
        }
        if token.chars().count() > CURSOR_MAX_CHARS {
            return Err(validation(
                "cursor",
                format!("must not exceed {CURSOR_MAX_CHARS} characters"),
            ));
        }
        let Some(state) = self.cursors.get(token).cloned() else {
            return Err(validation(
                "cursor",
                "Unknown or expired skill_index cursor",
            ));
        };
        if parsed != state.operation {
            return Err(validation(
                "cursor",
                format!("cursor belongs to operation={}", state.operation.as_str()),
            ));
        }
        if let Some(category) = category {
            if state.category.as_deref() != Some(category) {
                return Err(validation(
                    "cursor",
                    "cursor category differs from the original request; omit cursor to change category",
                ));
            }
        }
        if let Some(query) = query {
            if query != state.queries.as_slice() {
                return Err(validation(
                    "cursor",
                    "cursor query differs from the original request; omit cursor to start a new search",
                ));
            }
        }
        if state.fingerprint != self.catalog.inventory().fingerprint {
            return Err(validation(
                "cursor",
                "Skill index changed; start a new list or search request",
            ));
        }
        self.cursors.remove(token);
        let mut request = SkillIndexRequest {
            operation: state.operation.as_str().to_string(),
            category: state.category.clone(),
            ..SkillIndexRequest::default()
        };
        if state.operation == Operation::Search {
            request.query = Some(state.queries.clone());
        }
        if state.operation == Operation::List {
            request.view = ListView::Details.as_str().to_string();
        }
        let mut validated =
            ops::validate_request(&request).map_err(|issues| first_issue(&issues))?;
        validated.page_size = Some(state.page_size.max(1));
        validated.page_offset = state.offset;
        self.run(&validated)
    }

    /// Runs Python's `_execute` over an already validated request.
    #[allow(clippy::too_many_lines)]
    fn run(&mut self, request: &ValidatedRequest) -> Result<SkillIndexResult> {
        let operation = request.operation;
        let category = request.category.clone();
        if operation == Operation::Read && category.is_some() {
            return Err(validation("category", "is valid only for list and search"));
        }
        let paths = if operation == Operation::Read {
            Vec::new()
        } else {
            let resolved = validate_category_paths(
                &self.view,
                category.as_deref(),
                request.request.paths.as_deref(),
            )?;
            validate_paths(&Some(resolved), false)?
        };
        let budget = output_budget(
            request.max_output_chars,
            request.disable_output_truncation,
            self.config.max_output_chars,
        )?;
        let execution = match operation {
            Operation::List => ops::execute_list(&self.view, &paths, request)?,
            Operation::Search => ops::execute_search(&self.catalog, &self.view, &paths, request)?,
            Operation::Read => {
                let skills =
                    ops::resolve_read_skills(&self.view, request, &self.observed_meta_paths)?;
                ops::execute_read(&self.view, &skills, request)?
            }
        };
        let outcome = ops::apply_pipeline(&execution.rows, &request.pipeline, request.output_mode)?;
        let mut complete = execution.complete && outcome.complete;
        let all_rows: Vec<Row> = outcome.rows;
        let mut page_size = request.page_size;
        if page_size == Some(0) {
            // Show the branch choices together; Skill candidate pages stay bounded.
            page_size = Some(if all_rows.iter().any(|row| !row.worker_id.is_empty()) {
                10
            } else {
                all_rows.len().max(1)
            });
        }
        let page_offset = request.page_offset;
        let rows: Vec<Row> = match page_size {
            Some(size) => {
                let window: Vec<Row> = all_rows
                    .iter()
                    .skip(page_offset)
                    .take(size)
                    .cloned()
                    .collect();
                complete = complete && page_offset + window.len() >= all_rows.len();
                window
            }
            None => all_rows.clone(),
        };
        let result_entry_count = match outcome.count_value {
            Some(count) => count,
            None => match outcome.transformed_count {
                Some(count) => count,
                None => {
                    if request.pipeline.is_empty() || !outcome.complete {
                        execution.total_count
                    } else {
                        row_count(&rows)
                    }
                }
            },
        };
        let mut returned_skills: Vec<&str> = Vec::new();
        let mut returned_categories = 0usize;
        for row in &rows {
            if row.worker_id.is_empty() {
                continue;
            }
            if !returned_skills.contains(&row.worker_id.as_str()) {
                returned_skills.push(row.worker_id.as_str());
            }
        }
        for row in &rows {
            if row.is_category_row() {
                returned_categories += 1;
            }
        }
        let previously_shown = operation == Operation::Search
            && page_size.is_some()
            && returned_skills
                .iter()
                .all(|worker_id| self.observed_skills.contains(*worker_id));
        let summary = ResponseSummary {
            operation: operation.as_str().to_string(),
            returned_skill_count: returned_skills.len(),
            returned_category_count: returned_categories,
            previously_shown,
        };
        let mut reserve = 0usize;
        if page_size.is_some() && !rows.is_empty() {
            reserve += CURSOR_FOOTER_CHARS;
        }
        let fit_budget =
            budget.map(|budget| (MIN_OUTPUT_CHARS / 2).max(budget.saturating_sub(reserve)));
        let fitted = fit_rows(
            &rows,
            &summary,
            fit_budget,
            outcome.count_value,
            page_size.is_none(),
        );
        let mut content = fitted.body.clone();
        let mut model_content = fitted.model_content.clone();
        if content.is_empty() && request.pipeline.is_empty() {
            let message = empty_message(operation.as_str(), category.as_deref());
            match budget {
                None => {
                    content.clone_from(&message);
                    model_content = format!("{model_content}\n\n{message}");
                }
                Some(budget) => {
                    let remaining = budget
                        .saturating_sub(model_content.chars().count())
                        .saturating_sub(2);
                    if remaining > 3 {
                        let rendered = compact(&message, remaining);
                        content.clone_from(&rendered);
                        model_content = format!("{model_content}\n\n{rendered}");
                    }
                }
            }
        }
        let mut next_cursor: Option<String> = None;
        let has_more = page_size.is_some() && page_offset + fitted.shown_count < all_rows.len();
        if has_more {
            let state = CursorState {
                operation,
                category: category.clone(),
                queries: if operation == Operation::Search {
                    request.queries.clone()
                } else {
                    Vec::new()
                },
                offset: page_offset + fitted.shown_count,
                page_size: page_size.unwrap_or(1),
                fingerprint: self.catalog.inventory().fingerprint.clone(),
            };
            let token = state.token();
            self.cursors.push(token.clone(), state);
            let footer = format!(
                "[next] {{\"operation\":\"{}\",\"cursor\":\"{token}\"}}",
                operation.as_str()
            );
            content = append_note(&content, &footer, budget);
            model_content = append_note(&model_content, &footer, budget);
            next_cursor = Some(token);
        } else if page_size.is_some() && fitted.shortened {
            let note = "[truncated] Candidate details shortened.";
            content = append_note(&content, note, budget);
            model_content = append_note(&model_content, note, budget);
        }
        if page_size.is_some() {
            content.clone_from(&model_content);
        }
        let cards = catalog_cards(&self.catalog);
        let (delivered, reminder) = self.notice.append(&model_content, &cards, budget);
        if let Some(reminder) = reminder {
            content = if content.is_empty() {
                reminder
            } else {
                format!("{content}\n\n{reminder}")
            };
            model_content = delivered;
        }
        for worker_id in &fitted.observed {
            if let Ok(path) = self.view.metadata_path(worker_id) {
                if let Ok(normalized) = SkillDirectoryView::normalize_path(&path) {
                    self.observed_meta_paths.insert(normalized);
                }
            }
            self.observed_skills.insert(worker_id.clone());
        }
        let result_rows: Vec<Row> = if page_size.is_some() {
            rows.clone()
        } else if let Some(count) = outcome.count_value {
            vec![Row::text_only(count.to_string())]
        } else {
            rows.iter().take(fitted.shown_count).cloned().collect()
        };
        let records = self.catalog.records();
        Ok(SkillIndexResult {
            content,
            model_content,
            operation,
            rows: result_rows,
            observed_skill_ids: fitted.observed.clone(),
            truncated: fitted.shortened,
            truncation_reason: fitted.shortened.then(|| "max_output_chars".to_string()),
            total_count: result_entry_count,
            shown_count: fitted.shown_count,
            result_complete: !has_more && complete && !fitted.shortened,
            next_cursor,
            effective_max_output_chars: budget,
            skillfs_layout: LAYOUT_FLAT.to_string(),
            index_state: INDEX_MISSING.to_string(),
            candidate_count: records.len(),
            estimated_candidate_tokens: candidate_tokens(records),
            candidate_budget_tokens: DiscoverySettings::default().candidate_budget_tokens(),
        })
    }
}

/// One page cursor's continuation state.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CursorState {
    /// The operation the page belongs to.
    operation: Operation,
    /// The category scope, when one was named.
    category: Option<String>,
    /// The validated search queries; empty for a list.
    queries: Vec<String>,
    /// The offset the next page starts at.
    offset: usize,
    /// The page size the original request used.
    page_size: usize,
    /// The catalog fingerprint the page was minted against.
    fingerprint: String,
}

impl CursorState {
    /// The deterministic token for this state.
    ///
    /// The URL-safe unpadded base64 of the first [`CURSOR_TOKEN_BYTES`] of SHA-256 over
    /// the canonical JSON of `{v, operation, category, queries, offset, page_size,
    /// fingerprint}`. The field order is fixed and the document holds no maps, so the
    /// bytes — and therefore the token — are identical for identical states.
    fn token(&self) -> String {
        let document = json!({
            "v": CURSOR_STATE_VERSION,
            "operation": self.operation.as_str(),
            "category": self.category,
            "queries": self.queries,
            "offset": self.offset,
            "page_size": self.page_size,
            "fingerprint": self.fingerprint,
        });
        let canonical = document.to_string();
        let digest = Sha256::digest(canonical.as_bytes());
        let head = digest
            .get(..CURSOR_TOKEN_BYTES)
            .unwrap_or(digest.as_slice());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(head)
    }
}

/// The page cursors, ordered least recently used first.
#[derive(Debug, Default, Clone)]
struct CursorStore {
    /// `(token, state)` pairs, the most recently used last.
    entries: Vec<(String, CursorState)>,
}

impl CursorStore {
    /// The state behind `token`, when it is still held.
    fn get(&self, token: &str) -> Option<&CursorState> {
        self.entries
            .iter()
            .find(|(held, _)| held == token)
            .map(|(_, state)| state)
    }

    /// Stores `state` under `token`, evicting the least recently used past
    /// [`MAX_CURSORS`].
    fn push(&mut self, token: String, state: CursorState) {
        self.entries.retain(|(held, _)| held != &token);
        self.entries.push((token, state));
        while self.entries.len() > MAX_CURSORS {
            self.entries.remove(0);
        }
    }

    /// Drops `token` from the store.
    fn remove(&mut self, token: &str) {
        self.entries.retain(|(held, _)| held != token);
    }

    /// How many cursors the store holds.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Drops every cursor.
    fn clear(&mut self) {
        self.entries.clear();
    }
}

/// The pinned argument schema Python's `_tool_card` publishes.
#[must_use]
pub fn tool_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": MODEL_OPERATIONS.map(Operation::as_str),
                "description": OPERATION_DESCRIPTION,
            },
            "category": {
                "type": "string",
                "minLength": 1,
                "description": CATEGORY_DESCRIPTION,
            },
            "query": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 512},
                "minItems": 1,
                "maxItems": 8,
                "description": QUERY_DESCRIPTION,
            },
            "cursor": {
                "type": "string",
                "minLength": 1,
                "maxLength": CURSOR_MAX_CHARS,
                "description": CURSOR_DESCRIPTION,
            },
        },
        "required": ["operation"],
        "additionalProperties": false,
    })
}

/// The notice cards of one catalog: the worker id as the name, the description (or the
/// display name when it is empty) as the text.
///
/// Python's `{item.worker_id: {"name": item.worker_id, "description":
/// item.description or item.name}}`, so a reminder row reads `- <worker_id>: <text>`.
fn catalog_cards(catalog: &SkillFs) -> SkillCards {
    catalog
        .records()
        .iter()
        .map(|record| {
            let description = if record.description.is_empty() {
                record.name.clone()
            } else {
                record.description.clone()
            };
            (
                record.worker_id.clone(),
                SkillCard {
                    name: Some(record.worker_id.clone()),
                    description: Some(description),
                },
            )
        })
        .collect()
}

/// Whether a request uses Python's model-facing simple shape.
///
/// A simple request names no `skills`/`paths` and leaves every advanced field at its
/// default; `operation`, `category`, `query`, and `limit` do not make a request advanced,
/// exactly like Python's `simple_request` test over its keyword arguments.
fn is_simple(request: &SkillIndexRequest) -> bool {
    request.skills.is_none()
        && request.paths.is_none()
        && request.view == ListView::Names.as_str()
        && !request.recursive
        && !request.directory_entry
        && !request.directories_only
        && request.per_query_limit.is_none()
        && request.search_match == SearchMatch::Content.as_str()
        && request.result == super::validate::SearchResult::Files.as_str()
        && request.case_insensitive
        && !request.fixed_strings
        && request.max_depth.is_none()
        && request.read_mode == ReadMode::Full.as_str()
        && request.line_count.is_none()
        && request.start_line.is_none()
        && request.end_line.is_none()
        && request.pipeline.is_empty()
        && request.output_mode == OutputMode::Entries.as_str()
        && request.max_output_chars.is_none()
        && !request.disable_output_truncation
}

/// Converts the first validation issue into the crate's typed error.
fn first_issue(issues: &[RequestIssue]) -> crate::error::Error {
    match issues.first() {
        Some(issue) => crate::error::Error::validation(issue.to_string()),
        None => crate::error::Error::validation("request is invalid"),
    }
}
