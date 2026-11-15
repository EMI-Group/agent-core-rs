//! Operation execution for the `skill_index` tool: list, search, read, and pipelines.
//!
//! Ports the execution half of Python `openjiuwen.symphony.discovery.directory_toolkit`
//! (`_list`, `_search`, `_search_categories`, `_search_content_one`, `_search_entry_one`,
//! `_matched_snippets`, `_read_skills`, `_read`, and `_apply_pipeline`). Pagination
//! cursors, output budgeting and the incremental notice live in the toolkit layer, which
//! drives these functions.
//!
//! # Search assembly
//!
//! A search runs one query at a time and keeps, per query, the ranked matches bounded by
//! the per-query limit. A batch (more than one query) on the advanced shape interleaves
//! the per-query groups by rank so no single query floods the result; the simple shape
//! concatenates them and lets the response page over the union. Identities are
//! de-duplicated, and the query indexes that matched an identity are recorded so a row
//! can render `matches: 1, 3`.
//!
//! When a content query on the simple shape finds nothing, the category tree answers
//! instead ([`search_categories`]), so the model gets "try this group" hints rather than
//! an empty answer.
//!
//! # Deviations from Python
//!
//! - **No regular expressions.** Python compiles the query with
//!   `retrieval.search.runtime.lexical.compile_matcher`, which is a bounded regex unless
//!   `fixed_strings` is set. This crate ships no regex engine, so a query matches either
//!   as a case-(in)sensitive substring or, when it carries `*`, `?`, or `[`, as a glob
//!   through [`glob_match`]; `fixed_strings` forces the substring reading. The evidence a
//!   `matches` search reports is therefore line-oriented rather than pattern-oriented.
//! - **Content ranking is BM25 plus a literal pass.** Python's lexical index scores
//!   weighted fields, phrase hits, and identity tiers over the whole document. The port
//!   ranks with the catalog's BM25 index ([`SkillFs::search`](crate::discovery::SkillFs::search)) and then appends the
//!   scoped records whose text contains the query verbatim, which keeps the ordering
//!   deterministic and needs no model.
//! - **A missing record is skipped, not raised.** Python indexes the view and raises
//!   `KeyError`; a view built from the same catalog cannot be missing the record, so the
//!   port degrades instead of panicking.
//!
//! Everything here is synchronous, offline, and deterministic: ties break on the path (or
//! worker id), never on hash-map iteration order.
//!
//! [`glob_match`]: crate::discovery::directory_toolkit::matcher::glob_match

use crate::discovery::directory_toolkit::render::Row;

pub mod list;
pub mod pipeline;
pub mod read;
pub mod request;
pub mod search;

pub use list::execute_list;
pub use pipeline::apply_pipeline;
pub use read::{execute_read, resolve_read_skills};
pub use request::validate_request;
pub use search::{content_hits, entry_hits, execute_search, matched_snippets, search_categories};

/// The characters a `matches` search snippet is compacted to.
pub const SNIPPET_CHARS: usize = 180;

/// The characters of a category description a category hint shows.
pub const CATEGORY_HINT_CHARS: usize = 300;

/// The per-query hit limit a multi-query advanced search falls back to.
pub const DEFAULT_MULTI_QUERY_LIMIT: usize = 5;

/// The lines a `head` read returns when the request names none.
pub const DEFAULT_READ_LINES: usize = 10;

/// The capability type the category-hint index uses for its documents.
pub const CATEGORY_CAPABILITY_TYPE: &str = "category";

/// One operation's rows plus the counters a caller needs to page and summarize them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Execution {
    /// The rendered rows, before any pipeline or paging.
    pub rows: Vec<Row>,
    /// How many entries the operation matched, before any pipeline or paging.
    pub total_count: usize,
    /// Whether the operation returned every match (false when a per-query limit bit).
    pub complete: bool,
}

/// The result of applying a pipeline (and the output mode) to rendered rows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineOutcome {
    /// The rows left after the stages.
    pub rows: Vec<Row>,
    /// Whether the stages removed nothing.
    pub complete: bool,
    /// The count a `count` stage (or `output_mode = count`) produced.
    pub count_value: Option<usize>,
    /// The row count after a stage that followed a `count` stage.
    pub transformed_count: Option<usize>,
}
