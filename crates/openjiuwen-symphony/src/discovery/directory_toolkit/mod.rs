//! The model-facing `skill_index` tool: a read-only structured view of one
//! `SkillFs` catalog.
//!
//! Ports Python `openjiuwen.symphony.discovery.directory_toolkit`: the virtual category
//! tree, the pinned row renderers with their output budgeting, the full
//! argument-validation pass, the operation engine, and the top-level toolkit.
//!
//! | module | Python source | role |
//! | --- | --- | --- |
//! | [`view`] | `skillfs.py` `DirectoryEntry`, `SkillDirectoryView` | the category tree: paths, children, skill counts, scoped entries, metadata paths, category names |
//! | [`render`] | `directory_toolkit.py` row/summary/budget helpers | pinned row text, summary headers, `compact`/`append_note`, the metadata card, `fit_rows` |
//! | [`validate`](validate::validate) | `directory_toolkit.py` argument layer | the typed request, the six enums, every per-field rule, and Python's simple-shape defaults |
//! | [`matcher`] | Python `str.casefold`/`fnmatch`/`==` | case folding, glob matching, literal match ranking |
//! | [`ops`] | `directory_toolkit.py` `_list`/`_search`/`_read`/`_apply_pipeline` | operation dispatch, multi-query search assembly, snippets, pipelines |
//! | [`toolkit`](toolkit::SkillIndexToolkit) | `directory_toolkit.py` `InstalledSkillsDirectoryToolkit` | the model-facing tool: `invoke`/`execute`, the card, pagination cursors, the notice cursor |
//!
//! # Design
//!
//! **Categories are the front-matter `category`.** Python's retriever-node taxonomy does
//! not exist in the Rust crate, so [`SkillDirectoryView`] builds the same virtual tree
//! from each skill's `CapabilityDescriptor::classification`,
//! which is the value the prompt snapshot already uses as a category label. A
//! classification splits on `>` (or on `/` when it has none), each segment is trimmed and
//! blank segments are dropped, so `"Data > Export"` and `"Data/Export"` both mean the
//! category chain `Data` then `Export`, and a blank classification puts the skill
//! directly under [`ROOT_PATH`] (label [`ROOT_LABEL`], description
//! [`ROOT_DESCRIPTION`]).
//!
//! **Sibling collisions are disambiguated deterministically.** Two chains can name the
//! same segment with different case, and a category label can collide with a skill worker
//! id. Python's retriever nodes carry their own identity; here a colliding segment gets
//! [`DISAMBIGUATION_MARKER`]-suffixed names (`~2`, `~3`, …) assigned in
//! `(segment.to_lowercase(), segment)` order, and a skill whitelists its own worker id
//! first. The tree is therefore a pure function of the catalog.
//!
//! **Case folding is `to_lowercase`, not `casefold`.** Python's `str.casefold()` folds
//! `ß` to `ss` and `İ` to `i̇`; Rust's `str::to_lowercase()` does not. Every
//! case-insensitive comparison in this tree (category lookup, glob matching, literal
//! ranking, the notice cursor elsewhere in the crate) uses the same fold, so the
//! behavior is self-consistent, and the difference only shows up for those exotic
//! letters.
//!
//! **No panics.** Every renderer is total over a well-formed view, and the paths the
//! renderers need that a caller could get wrong (`skill_count`, `skill_category`) have
//! `*_or_zero`/`*_or_empty` variants so a row degrades instead of raising.
//!
//! Everything here is synchronous, offline, and deterministic: no I/O, no clock, no
//! model.

pub mod matcher;
pub mod ops;
pub mod render;
pub mod toolkit;
pub mod validate;
pub mod view;

pub use matcher::{case_fold, glob_match, has_wildcard, literal_match_rank};
pub use ops::{
    apply_pipeline, content_hits, entry_hits, execute_list, execute_read, execute_search,
    matched_snippets, resolve_read_skills, search_categories, validate_request, Execution,
    PipelineOutcome,
};
pub use render::{
    append_note, bounded_summary_line, candidate_tokens, compact, directory_description,
    empty_message, fit_rows, inline_text, list_row, metadata_card, row_count, safe_match_snippet,
    sanitize_model_text, search_directory_row, search_row, skill_path, summary_line, FittedRows,
    ListView, ResponseSummary, Row, CATEGORY_DESCRIPTION_CHARS, CATEGORY_ROUTING_CHARS,
    CATEGORY_ROW_PREFIX, SEARCH_CATEGORY_DESCRIPTION_CHARS, SELECT_WHEN_CHARS, SHORTENED_MARKER,
    SKILL_DESCRIPTION_CHARS,
};
pub use toolkit::{
    tool_input_schema, SkillIndexConfig, SkillIndexResult, SkillIndexToolCard, SkillIndexToolkit,
    CURSOR_MAX_CHARS, MAX_CURSORS, SKILL_INDEX_TOOL_NAME,
};
pub use validate::{
    apply_simple_defaults, output_budget, validate, validate_category_paths, validate_identifiers,
    validate_paths, validate_pipeline, validate_queries, Operation, OutputMode, PipelineOperation,
    PipelineStage, ReadMode, RequestIssue, SearchMatch, SearchResult, SkillIndexRequest,
    ValidatedRequest, DEFAULT_RESULT_LIMIT, DEFAULT_SEARCH_PAGE_SIZE, MAX_IDENTIFIER_CHARS,
    MAX_LINES, MAX_OUTPUT_CHARS, MAX_PATHS, MAX_PATH_CHARS, MAX_PER_QUERY_LIMIT, MAX_QUERIES,
    MAX_QUERY_CHARS, MAX_RESULT_LIMIT, MAX_SEARCH_PAGE_SIZE, MIN_OUTPUT_CHARS, ROOT_PATH,
};
pub use view::{
    category_segments, has_category, DirectoryEntry, DirectoryNode, EntryKind, SkillDirectoryView,
    CATEGORY_LABEL_SEPARATOR, CATEGORY_PATH_SEPARATOR, CATEGORY_SEPARATOR, DISAMBIGUATION_MARKER,
    MAX_CATEGORY_DEPTH, METADATA_FILENAME, ROOT_DESCRIPTION, ROOT_LABEL,
};
