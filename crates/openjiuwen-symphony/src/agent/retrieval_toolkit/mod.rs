//! The agent-facing skill-retrieval toolkit: four read-only operations over one index.
//!
//! Ports the browsing and search surface of Python
//! `openjiuwen.symphony.agent.retrieval_toolkit` (v0.1.20,
//! `AgenticSkillRetrievalToolkit`) onto the query-time runtime in
//! [`crate::retrieval::search::service::Retriever`]. Python exposes three model tools
//! (`skill_index_build`, `skill_branch_explore`, `skill_branch_peek`) plus a blocking
//! build surface; this port exposes exactly one model-facing tool,
//! [`RetrievalToolkit`], whose [`RetrievalOperation`] selects between `list`, `browse`,
//! `search` and `read`. Index construction stays in [`crate::retrieval::build`].
//!
//! # The four operations
//!
//! - `list` — the direct sub-branches of a cid (the root when no cid is given), each with
//!   its item and descendant-branch counts. Mirrors Python's `branch_peek` outline.
//! - `browse` — one node: its label, description, direct sub-branches and its *own*
//!   items (payload cid, worker id, first-line description). Mirrors Python's
//!   `branch_explore` disclosure.
//! - `search` — ranked hits for a task phrase from
//!   [`crate::retrieval::search::service::Retriever::search_details_with`], in the
//!   retriever's rank order.
//! - `read` — one item's full routing description, resolved by leaf cid *or* worker id
//!   through the loaded catalog and item-path index.
//!
//! # Abstaining
//!
//! `search` never fabricates a hit and never errors on an unusable query: a blank or
//! missing query and a zero-hit search return a normal [`RetrievalResponse`] with
//! [`RetrievalResponse::abstained`] set, a pinned [`RetrievalResponse::abstain_reason`]
//! ([`ABSTAIN_BLANK_QUERY`] or [`ABSTAIN_NO_RESULTS`]) and a suggestion. Every other
//! failure is a typed [`crate::error::Error`].
//!
//! # Determinism
//!
//! Everything is offline and deterministic: branches sort by `node_id`, items by
//! `item_id`, search hits keep the retriever's rank order, page cursors are a hash of the
//! continuation state plus a structural index fingerprint, and no output depends on the
//! wall clock, randomness or hash-map iteration order. Identical requests produce
//! byte-identical `content` and `model_content`.
//!
//! # Validation and errors
//!
//! [`RetrievalToolkit::new`] rejects an unusable [`RetrievalToolkitConfig`]; the request
//! path rejects a blank or missing cid where one is required, a field used by the wrong
//! operation, an unknown argument key, a wrong JSON type, and a cursor that is unknown,
//! expired or belongs to a different request. An unknown *scope* cid and an unknown
//! *item* are [`Error::NotFound`](crate::error::Error::NotFound); every other rejection is
//! [`Error::Validation`](crate::error::Error::Validation) naming the offending field.
//! Request `page_size` and `top_k` are clamped to the configured maxima (`0` selects the
//! default) rather than rejected; the response reports the effective values.
mod ops;
mod render;
mod toolkit;
mod validate;

#[cfg(test)]
mod tests;

pub use render::{render_retrieval_prompt, RetrievalPromptSnapshot};
pub use toolkit::{RetrievalToolkit, BLANK_QUERY_SUGGESTION, NO_RESULTS_SUGGESTION};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;
use std::str::FromStr;

use crate::error::{validation, Result};
use crate::retrieval::types::ROOT_NODE_ID;

/// The model-facing tool name (Python's three default tool names collapse into one
/// operation-dispatch tool here).
pub const SKILL_RETRIEVAL_TOOL_NAME: &str = "skill_retrieval";
/// The maximum number of page cursors one toolkit keeps; the least recently used is
/// dropped past it.
pub const MAX_CURSORS: usize = 32;
/// The maximum characters a cursor token may have on the wire.
pub const CURSOR_MAX_CHARS: usize = 128;
/// The characters reserved for the `[next]` footer while fitting a page.
pub const CURSOR_FOOTER_CHARS: usize = 96;
/// The version the cursor token hashes.
pub const CURSOR_STATE_VERSION: u64 = 1;
/// The bytes of the SHA-256 digest a cursor token keeps.
pub const CURSOR_TOKEN_BYTES: usize = 16;
/// The smallest output budget [`RetrievalToolkitConfig::max_output_chars`] accepts.
pub const MIN_OUTPUT_CHARS: usize = 512;
/// The largest output budget [`RetrievalToolkitConfig::max_output_chars`] accepts.
pub const MAX_OUTPUT_CHARS: usize = 48_000;
/// The default output budget.
pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 12_000;
/// The default page size.
pub const DEFAULT_PAGE_SIZE: usize = 10;
/// The largest page size one request may use.
pub const MAX_PAGE_SIZE: usize = 100;
/// The default number of ranked search hits one request pages over.
pub const DEFAULT_TOP_K: usize = 10;
/// The default cap on a search request's `top_k`.
pub const DEFAULT_MAX_TOP_K: usize = 20;
/// The largest `top_k` one search request may use.
pub const MAX_TOP_K: usize = 100;
/// The abstain reason of a blank or missing search query.
pub const ABSTAIN_BLANK_QUERY: &str = "blank_query";
/// The abstain reason of a search that matched no indexed item.
pub const ABSTAIN_NO_RESULTS: &str = "no_results";

impl Default for RetrievalToolkitConfig {
    fn default() -> Self {
        Self {
            session_scope: String::new(),
            max_output_chars: DEFAULT_MAX_OUTPUT_CHARS,
            default_page_size: DEFAULT_PAGE_SIZE,
            max_page_size: MAX_PAGE_SIZE,
            max_top_k: DEFAULT_MAX_TOP_K,
        }
    }
}

/// The operations the model-facing tool accepts.
pub const MODEL_OPERATIONS: [RetrievalOperation; 4] = [
    RetrievalOperation::List,
    RetrievalOperation::Browse,
    RetrievalOperation::Search,
    RetrievalOperation::Read,
];

/// One read-only operation over the capability tree.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalOperation {
    /// List the direct sub-branches of a cid (the root when no cid is given).
    #[default]
    List,
    /// Show one node's description, direct sub-branches and its own items.
    Browse,
    /// Rank indexed items for a task phrase.
    Search,
    /// Read one item's full routing description.
    Read,
}

impl RetrievalOperation {
    /// The wire name of this operation (`"list"`, `"browse"`, `"search"`, `"read"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::List => "list",
            Self::Browse => "browse",
            Self::Search => "search",
            Self::Read => "read",
        }
    }
}

impl fmt::Display for RetrievalOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for RetrievalOperation {
    type Err = crate::error::Error;

    /// Parses one wire name, trimming surrounding whitespace.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error::Validation) naming `operation` for an
    /// unknown name.
    fn from_str(value: &str) -> Result<Self> {
        match value.trim() {
            "list" => Ok(Self::List),
            "browse" => Ok(Self::Browse),
            "search" => Ok(Self::Search),
            "read" => Ok(Self::Read),
            other => Err(validation(
                "operation",
                format!(
                    "must be one of: {} (got `{other}`)",
                    MODEL_OPERATIONS.map(Self::as_str).join(", ")
                ),
            )),
        }
    }
}

/// One typed retrieval request.
///
/// Every field has a `serde` default, so `RetrievalRequest::default()` asks for the root
/// listing. Field bounds:
///
/// - `cid` names a *branch* (`RetrieverNode::node_id`) for `list`/`browse` and an item
///   leaf cid or worker id for `read`; it is omitted for the root in `list`/`browse`,
///   required for `read`, and rejected for `search`.
/// - `query` is required by `search` (a blank query abstains) and rejected elsewhere.
/// - `page_size` applies to `list`, `browse` and `search`; `0` selects
///   [`RetrievalToolkitConfig::default_page_size`] and a larger value is clamped to
///   [`RetrievalToolkitConfig::max_page_size`].
/// - `top_k` applies to `search` only; `0` selects the default and a larger value is
///   clamped to [`RetrievalToolkitConfig::max_top_k`].
/// - `cursor` resumes the page it was minted for; the other fields may be repeated only
///   with their original values.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievalRequest {
    /// The operation to run.
    pub operation: RetrievalOperation,
    /// The branch cid (list/browse) or item cid/worker id (read).
    pub cid: Option<String>,
    /// The search phrase (search only).
    pub query: Option<String>,
    /// The `[next]` cursor of an earlier page.
    pub cursor: Option<String>,
    /// The requested page size; `0` selects the configured default.
    pub page_size: Option<usize>,
    /// The requested number of ranked search hits; `0` selects the default.
    pub top_k: Option<usize>,
}

/// One typed row of a retrieval response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievalEntry {
    /// The row kind: `branch`, `item` or `search_hit`.
    pub kind: String,
    /// The branch `node_id`, the item's leaf cid, or the search hit's leaf cid.
    pub cid: String,
    /// The item's public worker id; empty for branch rows.
    pub worker_id: String,
    /// The display label (the branch label, item name, or search-hit name).
    pub label: String,
    /// The full description; the renderers shorten it per row.
    pub description: String,
    /// The ancestor cid chain, root excluded.
    pub path: Vec<String>,
    /// The catalog category (the ancestor segments dot-joined).
    pub category: String,
    /// The one-based search rank; `0` for branch and item rows.
    pub rank: usize,
    /// How many items the branch's subtree carries; `0` for item rows.
    pub item_count: usize,
    /// How many descendant branches the branch carries; `0` for item rows.
    pub branch_count: usize,
}

/// One retrieval response: the delivered text, the typed rows and the paging state.
///
/// `content` is the concise body (a title, the rows, and any `[next]` footer);
/// `model_content` is the same body behind a one-line `[skill_retrieval] …` summary.
/// `abstained` is set only for [`ABSTAIN_BLANK_QUERY`] and [`ABSTAIN_NO_RESULTS`] search
/// answers; every other request either returns rows or a typed error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievalResponse {
    /// The operation that produced this response.
    pub operation: RetrievalOperation,
    /// The concise result text.
    pub content: String,
    /// The model-facing text including the summary header.
    pub model_content: String,
    /// The typed rows of this page, in render order.
    pub entries: Vec<RetrievalEntry>,
    /// The resolved branch cid, the read item's leaf cid, or empty for search.
    pub cid: String,
    /// The resolved cid's ancestor chain, root excluded.
    pub path: Vec<String>,
    /// How many entries the operation matched, before paging.
    pub total_count: usize,
    /// How many entries this page shows.
    pub shown_count: usize,
    /// Whether this response dropped or paged away an entry.
    pub truncated: bool,
    /// Why the response was shortened: `max_output_chars` or `page`.
    pub truncation_reason: Option<String>,
    /// The token that asks for the next page, when one exists.
    pub next_cursor: Option<String>,
    /// Whether the caller has seen everything this request could return.
    pub result_complete: bool,
    /// Whether the operation abstained instead of returning rows.
    pub abstained: bool,
    /// The pinned abstain reason ([`ABSTAIN_BLANK_QUERY`] or [`ABSTAIN_NO_RESULTS`]).
    pub abstain_reason: Option<String>,
    /// What to try next after an abstain.
    pub suggestion: Option<String>,
    /// The backend that answered (`progressive` or `lexical_fallback`).
    pub method: String,
    /// The effective `top_k` of a search request; `0` for the other operations.
    pub effective_top_k: usize,
    /// The effective page size of this request; `0` for `read`.
    pub page_size: usize,
}

/// One toolkit's configuration.
///
/// The bounds are validated once by [`RetrievalToolkit::new`]; a caller may keep the
/// struct in configuration and change it only by building a new toolkit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievalToolkitConfig {
    /// The session scope the tool id is keyed by; required and non-blank.
    pub session_scope: String,
    /// The output budget in characters, within [`MIN_OUTPUT_CHARS`]`..=`[`MAX_OUTPUT_CHARS`].
    pub max_output_chars: usize,
    /// The page size a request gets when it asks for `0` or names none;
    /// `1..=max_page_size`.
    pub default_page_size: usize,
    /// The largest page size one request may use; `1..=`[`MAX_PAGE_SIZE`].
    pub max_page_size: usize,
    /// The largest `top_k` one search may use; `1..=`[`MAX_TOP_K`].
    pub max_top_k: usize,
}

impl RetrievalToolkitConfig {
    /// Builds a config for `session_scope` with the default budgets.
    #[must_use]
    pub fn new(session_scope: impl Into<String>) -> Self {
        Self {
            session_scope: session_scope.into(),
            ..Self::default()
        }
    }
}

/// The card the host publishes for one retrieval tool instance.
///
/// `input_schema` is [`tool_input_schema`]: `operation` is the only required property,
/// the other five are optional, and `additionalProperties` is `false`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievalToolCard {
    /// The per-session tool id (`skill_retrieval__<sha256 hex>`).
    pub id: String,
    /// The tool name ([`SKILL_RETRIEVAL_TOOL_NAME`]).
    pub name: String,
    /// The model-facing description of what the tool is and is not.
    pub description: String,
    /// The pinned JSON Schema of the accepted arguments.
    pub input_schema: Value,
    /// Whether the tool may run in parallel with other tools; always `false`.
    pub parallel_safe: bool,
    /// Whether the tool is stateless; always `false` (it keeps page cursors).
    pub stateless: bool,
}

impl openjiuwen_common::card::Card for RetrievalToolCard {
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

/// The tool description, verbatim, pinned in [`RetrievalToolCard::description`].
pub(crate) const TOOL_DESCRIPTION: &str = "Read-only skill-tree retrieval over the indexed \
capabilities. list: a branch's direct sub-branches with skill and descendant counts. browse: one \
branch's description, sub-branches and its own skills. search: skill entries ranked for a task \
phrase. read: one item's routing description. Cids are backticked tree ids such as `Skills.Data`, \
never display numbers or filesystem paths. Search never invents a skill: a blank query or a \
zero-hit search abstains. Calls only move a page cursor.";

/// The `operation` property description.
pub(crate) const OPERATION_DESCRIPTION: &str = "list: direct sub-branches with counts. browse: \
one branch's sub-branches and own skills. search: ranked skill entries for a task phrase. read: one \
item's routing description.";

/// The `cid` property description.
pub(crate) const CID_DESCRIPTION: &str = "Branch cid for list/browse, item cid or worker id for \
read; omit for the root in list/browse. Never a display number or a filesystem path.";

/// The `query` property description.
pub(crate) const QUERY_DESCRIPTION: &str = "Search only. A short task phrase; an empty or \
unmatched query abstains instead of inventing a skill.";

/// The `cursor` property description.
pub(crate) const CURSOR_DESCRIPTION: &str = "Copy the [next] cursor unchanged. To change the \
operation, cid or query, omit the cursor.";

/// The `page_size` property description.
pub(crate) const PAGE_SIZE_DESCRIPTION: &str = "Entries per page; 0 selects the configured \
default and a larger value is clamped to the configured maximum.";

/// The `top_k` property description.
pub(crate) const TOP_K_DESCRIPTION: &str = "Search only. How many ranked hits to page over; 0 \
selects the default and a larger value is clamped to the configured maximum.";

/// The pinned argument schema [`RetrievalToolCard::input_schema`] publishes.
#[must_use]
pub fn tool_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": MODEL_OPERATIONS.map(RetrievalOperation::as_str),
                "description": OPERATION_DESCRIPTION,
            },
            "cid": {
                "type": "string",
                "minLength": 1,
                "description": CID_DESCRIPTION,
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 512,
                "description": QUERY_DESCRIPTION,
            },
            "cursor": {
                "type": "string",
                "minLength": 1,
                "maxLength": CURSOR_MAX_CHARS,
                "description": CURSOR_DESCRIPTION,
            },
            "page_size": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_PAGE_SIZE,
                "description": PAGE_SIZE_DESCRIPTION,
            },
            "top_k": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_TOP_K,
                "description": TOP_K_DESCRIPTION,
            },
        },
        "required": ["operation"],
        "additionalProperties": false,
    })
}

/// The root cid a `list`/`browse` request without a cid resolves to.
#[must_use]
pub fn root_cid() -> &'static str {
    ROOT_NODE_ID
}
