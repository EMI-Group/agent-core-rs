//! The typed `skill_index` request and its full validation pass.
//!
//! Ports the argument layer of Python `openjiuwen.symphony.discovery.directory_toolkit`:
//! `_validate_request`, `_queries`, `_category_paths`, `_apply_simple_defaults`,
//! `_output_budget`, `_enum`, `_positive`, `_boolean`, `_nonempty`, `_bounded_query`,
//! `_paths`, and `_identifiers`.
//!
//! # The request type is the wire shape
//!
//! Python's request is an untyped `dict[str, Any]` filled from
//! `_SKILL_INDEX_DEFAULTS`, so an unrecognized enum value reaches `_enum` and becomes a
//! field error. [`SkillIndexRequest`] keeps that property: the nine enum-valued slots are
//! [`String`]s ([`Operation`], [`ListView`], [`SearchMatch`], [`SearchResult`],
//! [`ReadMode`], [`OutputMode`], and [`PipelineOperation`] are the parsed forms), and the
//! numeric slots are `Option<i64>`, so a zero or a negative number is representable and
//! reportable instead of failing deserialization. [`validate`] parses all of them into a
//! [`ValidatedRequest`], which carries the typed enums, the normalized strings and
//! numbers, and the normalized request.
//!
//! # Accumulation
//!
//! Python raises on the first problem. The Rust pass collects every issue into a
//! `Vec<RequestIssue>` and returns them together, each naming its field (`view`,
//! `pipeline[0].lines`, …). That is a deliberate improvement: a model that gets one
//! field wrong usually gets several wrong, and one round trip should surface them all.
//!
//! # Deviations from Python
//!
//! - Issues accumulate instead of raising, and each names exactly one field.
//! - Python's `query` (list) and `queries` (list of 2+) are one field here: [`SkillIndexRequest::query`]
//!   accepts 1–8 values, and the batch rules (a single `limit` pipeline stage,
//!   `output_mode=entries`, `per_query_limit <= 10`) apply when it holds more than one.
//! - A blank `category` becomes "no category", mirroring `_DirectoryFunction`, instead of
//!   raising `_nonempty`'s error.
//! - `limit` is validated here but "`limit` cannot be combined with advanced arguments"
//!   stays with the toolkit layer, which is the only layer that knows whether the call
//!   used the simple or the advanced shape.
//! - The read-requires-`skills` and `search`-requires-a-`query` checks run in this pass;
//!   Python raises them later, in `_read_skills` and `_search`.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::discovery::directory_toolkit::render::ListView;
use crate::discovery::directory_toolkit::view::SkillDirectoryView;
use crate::error::{validation, Error, Result};

/// The smallest accepted output budget, in characters.
pub const MIN_OUTPUT_CHARS: usize = 512;
/// The largest accepted output budget, in characters.
pub const MAX_OUTPUT_CHARS: usize = 48_000;
/// The largest number of lines one `limit`/`slice` stage may address.
pub const MAX_LINES: usize = 5_000;
/// The largest number of paths one request may scope.
pub const MAX_PATHS: usize = 32;
/// The largest number of characters one search query may hold.
pub const MAX_QUERY_CHARS: usize = 512;
/// The largest number of characters one path may hold.
pub const MAX_PATH_CHARS: usize = 4_096;
/// The largest number of characters one identifier may hold.
pub const MAX_IDENTIFIER_CHARS: usize = 512;
/// The largest number of search queries one request may hold.
pub const MAX_QUERIES: usize = 8;
/// The default result limit of the model-facing simple shape.
pub const DEFAULT_RESULT_LIMIT: usize = 10;
/// The largest result limit the model-facing simple shape accepts.
pub const MAX_RESULT_LIMIT: usize = 10;
/// The default page size of the model-facing search shape.
pub const DEFAULT_SEARCH_PAGE_SIZE: usize = 10;
/// The largest page size the model-facing search shape ever returns.
pub const MAX_SEARCH_PAGE_SIZE: usize = 20;
/// The largest `per_query_limit` one search may request.
pub const MAX_PER_QUERY_LIMIT: usize = 10;
/// The root path a request without `paths` scopes.
pub const ROOT_PATH: &str = "/";

/// The `skill_index` operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// List the categories and Skills in a scope.
    #[default]
    List,
    /// Search names, descriptions, or content.
    Search,
    /// Read one Skill's metadata card.
    Read,
}

/// Which `list` view renders the rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SearchMatch {
    /// Match the Skill's own text (body, name, aliases, description, category, path).
    #[default]
    Content,
    /// Match the entry name.
    Name,
    /// Match the entry path.
    Path,
}

/// What a `search` returns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SearchResult {
    /// One row per matching Skill.
    #[default]
    Files,
    /// One row per matching line.
    Matches,
}

/// How much of a Skill's metadata card a `read` returns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReadMode {
    /// The whole card.
    #[default]
    Full,
    /// The first `line_count` lines.
    Head,
    /// The `start_line`..=`end_line` slice.
    Range,
}

/// Whether a response renders rows or a count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    /// Render the rows.
    #[default]
    Entries,
    /// Render how many rows there would be.
    Count,
}

/// One `pipeline` stage operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PipelineOperation {
    /// Keep the first `lines` rows.
    #[default]
    Limit,
    /// Keep the `start_line`..=`end_line` rows.
    Slice,
    /// Keep the rows whose text matches `query`.
    Filter,
    /// Replace the rows with their count.
    Count,
}

/// Implements the shared "parse a wire name" behaviour of the request enums.
trait ModelEnum: Copy + Default {
    /// Parses a wire name, returning `None` for an unknown value.
    fn parse(value: &str) -> Option<Self>;

    /// The names of every variant, in declaration order.
    fn names() -> Vec<&'static str>;
}

/// Generates the [`ModelEnum`] implementation shared by every request enum.
macro_rules! model_enum {
    ($name:ident { $($variant:ident => $wire:literal),+ $(,)? }) => {
        impl $name {
            /// Every variant, in the order the error message lists them.
            pub const ALL: [Self; [$($wire),+].len()] = [$(Self::$variant),+];

            /// The wire name of this variant.
            #[must_use]
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $wire),+
                }
            }

            /// Parses a wire name, returning `None` for an unknown value.
            #[must_use]
            pub fn parse(value: &str) -> Option<Self> {
                match value {
                    $($wire => Some(Self::$variant),)+
                    _ => None,
                }
            }

            /// The names of every variant, in [`Self::ALL`] order.
            #[must_use]
            pub fn names() -> Vec<&'static str> {
                Self::ALL.into_iter().map(Self::as_str).collect()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl ModelEnum for $name {
            fn parse(value: &str) -> Option<Self> {
                Self::parse(value)
            }

            fn names() -> Vec<&'static str> {
                vec![$($wire),+]
            }
        }
    };
}

model_enum!(Operation { List => "list", Search => "search", Read => "read" });
model_enum!(SearchMatch { Content => "content", Name => "name", Path => "path" });
model_enum!(SearchResult { Files => "files", Matches => "matches" });
model_enum!(ReadMode { Full => "full", Head => "head", Range => "range" });
model_enum!(OutputMode { Entries => "entries", Count => "count" });
model_enum!(PipelineOperation {
    Limit => "limit",
    Slice => "slice",
    Filter => "filter",
    Count => "count"
});

/// One `pipeline` stage as it arrives on the wire.
///
/// `operation` is a [`String`] for the same reason the request's enum slots are: an
/// unknown value must be reportable as `pipeline[<i>].operation`. Every field a stage's
/// operation does not take defaults to empty/`false` and is rejected when set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PipelineStage {
    /// The stage operation (`limit`, `slice`, `filter`, or `count`).
    pub operation: String,
    /// The `filter` pattern.
    pub query: Option<String>,
    /// How many rows a `limit` keeps.
    pub lines: Option<i64>,
    /// The first line a `slice` keeps (1-based).
    pub start_line: Option<i64>,
    /// The last line a `slice` keeps (1-based, inclusive).
    pub end_line: Option<i64>,
    /// Whether a `filter` ignores case.
    pub case_insensitive: bool,
    /// Whether a `filter` keeps the rows that do *not* match.
    pub invert: bool,
    /// Whether a `filter` treats the pattern as a literal string.
    pub fixed_strings: bool,
}

impl PipelineStage {
    /// Builds a `limit` stage.
    #[must_use]
    pub fn limit(lines: usize) -> Self {
        Self {
            operation: PipelineOperation::Limit.as_str().to_string(),
            lines: i64::try_from(lines).ok(),
            ..Self::default()
        }
    }

    /// Builds a `slice` stage from 1-based, inclusive bounds.
    #[must_use]
    pub fn slice(start_line: usize, end_line: usize) -> Self {
        Self {
            operation: PipelineOperation::Slice.as_str().to_string(),
            start_line: i64::try_from(start_line).ok(),
            end_line: i64::try_from(end_line).ok(),
            ..Self::default()
        }
    }

    /// Builds a `filter` stage.
    #[must_use]
    pub fn filter(query: impl Into<String>) -> Self {
        Self {
            operation: PipelineOperation::Filter.as_str().to_string(),
            query: Some(query.into()),
            case_insensitive: true,
            ..Self::default()
        }
    }

    /// Builds a `count` stage.
    #[must_use]
    pub fn count() -> Self {
        Self {
            operation: PipelineOperation::Count.as_str().to_string(),
            ..Self::default()
        }
    }
}

/// The model-facing `skill_index` request.
///
/// Every field defaults to Python's `_SKILL_INDEX_DEFAULTS`, unknown keys are ignored,
/// and the enum slots stay strings so an unrecognized value is reported by [`validate`]
/// rather than by the deserializer.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillIndexRequest {
    /// The operation: `list`, `search`, or `read`.
    pub operation: String,
    /// The category scope, as a returned name or an `A > B` chain.
    pub category: Option<String>,
    /// The Skill ids a `read` returns (mutually exclusive with `paths`).
    pub skills: Option<Vec<String>>,
    /// The paths a request scopes (the legacy `category` form).
    pub paths: Option<Vec<String>>,
    /// The search queries, 1–8 values.
    pub query: Option<Vec<String>>,
    /// The `list` view: `names`, `details`, or `tree`.
    pub view: String,
    /// Whether a `list` recurses into subcategories.
    pub recursive: bool,
    /// Whether a `list` also returns the requested category itself.
    pub directory_entry: bool,
    /// Whether a `tree` list returns only categories.
    pub directories_only: bool,
    /// The `search` match target: `content`, `name`, or `path`.
    #[serde(rename = "match")]
    pub search_match: String,
    /// What a content `search` returns: `files` or `matches`.
    pub result: String,
    /// Whether a content `search` ignores case.
    pub case_insensitive: bool,
    /// Whether a content `search` treats the query as a literal string.
    pub fixed_strings: bool,
    /// How deep a `tree` list descends below each scope root.
    pub max_depth: Option<i64>,
    /// How much of a card a `read` returns: `full`, `head`, or `range`.
    pub read_mode: String,
    /// How many lines a `head` read returns.
    pub line_count: Option<i64>,
    /// The first line a `range` read returns (1-based).
    pub start_line: Option<i64>,
    /// The last line a `range` read returns (1-based, inclusive).
    pub end_line: Option<i64>,
    /// How many hits each search query contributes.
    pub per_query_limit: Option<i64>,
    /// The post-processing stages applied to the rendered rows.
    pub pipeline: Vec<PipelineStage>,
    /// Whether the response renders rows or a count.
    pub output_mode: String,
    /// The output budget, in characters.
    pub max_output_chars: Option<i64>,
    /// Whether the output budget is ignored.
    pub disable_output_truncation: bool,
    /// How many rows the model-facing simple shape returns.
    pub limit: Option<i64>,
}

impl Default for SkillIndexRequest {
    fn default() -> Self {
        Self {
            operation: Operation::List.as_str().to_string(),
            category: None,
            skills: None,
            paths: None,
            query: None,
            view: ListView::Names.as_str().to_string(),
            recursive: false,
            directory_entry: false,
            directories_only: false,
            search_match: SearchMatch::Content.as_str().to_string(),
            result: SearchResult::Files.as_str().to_string(),
            case_insensitive: true,
            fixed_strings: false,
            max_depth: None,
            read_mode: ReadMode::Full.as_str().to_string(),
            line_count: None,
            start_line: None,
            end_line: None,
            per_query_limit: None,
            pipeline: Vec::new(),
            output_mode: OutputMode::Entries.as_str().to_string(),
            max_output_chars: None,
            disable_output_truncation: false,
            limit: None,
        }
    }
}

impl SkillIndexRequest {
    /// Builds an empty request (Python's `_SKILL_INDEX_DEFAULTS`).
    #[must_use]
    pub fn new(operation: Operation) -> Self {
        Self {
            operation: operation.as_str().to_string(),
            ..Self::default()
        }
    }

    /// Sets the category scope.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Sets the search queries.
    #[must_use]
    pub fn with_query(mut self, query: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.query = Some(query.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the Skill ids a `read` returns.
    #[must_use]
    pub fn with_skills(mut self, skills: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.skills = Some(skills.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the paths a request scopes.
    #[must_use]
    pub fn with_paths(mut self, paths: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.paths = Some(paths.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the `list` view.
    #[must_use]
    pub fn with_view(mut self, view: ListView) -> Self {
        self.view = view.as_str().to_string();
        self
    }

    /// Sets the pipeline.
    #[must_use]
    pub fn with_pipeline(mut self, pipeline: Vec<PipelineStage>) -> Self {
        self.pipeline = pipeline;
        self
    }

    /// Sets the simple-shape result limit.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = i64::try_from(limit).ok();
        self
    }
}

/// One rejected request field.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RequestIssue {
    /// The offending field, including a pipeline index (`pipeline[0].lines`).
    pub field: String,
    /// What is wrong with it, without the field prefix.
    pub message: String,
}

impl RequestIssue {
    /// Builds an issue naming `field`.
    #[must_use]
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RequestIssue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.field, self.message)
    }
}

/// A request that passed every validation rule.
///
/// The typed fields are the validated projection of [`Self::request`]; the toolkit layer
/// resolves `category`/`paths` against a [`SkillDirectoryView`] afterwards, with
/// [`validate_category_paths`].
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ValidatedRequest {
    /// The normalized request the typed fields were derived from.
    pub request: SkillIndexRequest,
    /// The parsed operation.
    pub operation: Operation,
    /// The parsed `list` view.
    pub view: ListView,
    /// The parsed `search` match target.
    #[serde(rename = "match")]
    pub search_match: SearchMatch,
    /// The parsed `search` result shape.
    pub search_result: SearchResult,
    /// The parsed `read` mode.
    pub read_mode: ReadMode,
    /// The parsed output mode.
    pub output_mode: OutputMode,
    /// Whether a `list` recurses.
    pub recursive: bool,
    /// Whether a `list` returns the requested category itself.
    pub directory_entry: bool,
    /// Whether a `tree` list returns only categories.
    pub directories_only: bool,
    /// Whether a content `search` ignores case.
    pub case_insensitive: bool,
    /// Whether a content `search` is literal.
    pub fixed_strings: bool,
    /// The validated `tree` depth.
    pub max_depth: Option<usize>,
    /// The validated `head` line count.
    pub line_count: Option<usize>,
    /// The validated `range` start line.
    pub start_line: Option<usize>,
    /// The validated `range` end line.
    pub end_line: Option<usize>,
    /// The per-query hit limit, including a resolved `limit` pipeline stage.
    pub per_query_limit: Option<usize>,
    /// The simple-shape result limit, when the caller supplied one.
    pub limit: Option<usize>,
    /// The output budget in characters; `None` when truncation is disabled.
    pub max_output_chars: Option<usize>,
    /// Whether the output budget is ignored.
    pub disable_output_truncation: bool,
    /// The trimmed category scope, `None` when absent or blank.
    pub category: Option<String>,
    /// The normalized Skill ids.
    pub skills: Vec<String>,
    /// The normalized paths (`["/"]` when the request names none).
    pub paths: Vec<String>,
    /// The normalized, de-duplicated search queries.
    pub queries: Vec<String>,
    /// The normalized pipeline stages.
    pub pipeline: Vec<PipelineStage>,
    /// Python's `_page_size`: `None` for the advanced shape, `Some(0)` for a simple
    /// `list` without an explicit limit.
    pub page_size: Option<usize>,
    /// Python's `_page_offset`.
    pub page_offset: usize,
}

/// Validates a request without resolving its category or paths against a view.
///
/// Every issue is collected in one pass, so a caller can report them together. The
/// returned request is normalized (trimmed and de-duplicated strings, validated
/// numbers, a normalized pipeline) and its typed projection is ready for the toolkit
/// layer to resolve.
///
/// # Errors
///
/// Returns every [`RequestIssue`] found, or an empty vector never: a request with no
/// issue yields `Ok`.
#[allow(clippy::too_many_lines)]
pub fn validate(
    request: &SkillIndexRequest,
) -> std::result::Result<ValidatedRequest, Vec<RequestIssue>> {
    let mut issues: Vec<RequestIssue> = Vec::new();
    let operation = choose::<Operation>(&request.operation, "operation", &mut issues);
    let view = choose::<ListView>(&request.view, "view", &mut issues);
    let search_match = choose::<SearchMatch>(&request.search_match, "match", &mut issues);
    let search_result = choose::<SearchResult>(&request.result, "result", &mut issues);
    let read_mode = choose::<ReadMode>(&request.read_mode, "read_mode", &mut issues);
    let output_mode = choose::<OutputMode>(&request.output_mode, "output_mode", &mut issues);
    let max_depth = positive_integer(request.max_depth, "max_depth", &mut issues);
    let line_count = positive_integer(request.line_count, "line_count", &mut issues);
    let start_line = positive_integer(request.start_line, "start_line", &mut issues);
    let end_line = positive_integer(request.end_line, "end_line", &mut issues);
    let per_query_limit = positive_integer(request.per_query_limit, "per_query_limit", &mut issues);
    let limit = positive_integer(request.limit, "limit", &mut issues);
    let max_output_chars = output_chars(
        request.max_output_chars,
        request.disable_output_truncation,
        &mut issues,
    );
    let category = normalized_category(request.category.as_deref());

    check_only_for_operation(
        request,
        operation,
        view,
        search_match,
        search_result,
        read_mode,
        &mut issues,
    );
    check_scope(
        category.is_some(),
        request.paths.is_some(),
        operation,
        &mut issues,
    );
    if operation == Operation::List {
        check_list(request, view, max_depth, &mut issues);
    } else if operation == Operation::Search {
        check_search(search_match, search_result, &mut issues);
    } else {
        check_read(
            request,
            read_mode,
            line_count,
            start_line,
            end_line,
            max_depth,
            category.is_some(),
            &mut issues,
        );
    }
    check_output(operation, output_mode, request, &mut issues);

    let queries = if request.query.is_some() || operation == Operation::Search {
        match normalize_queries(request.query.as_deref()) {
            Ok(values) => values,
            Err(mut problems) => {
                issues.append(&mut problems);
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    let skills = collect(
        normalize_identifiers(request.skills.as_deref(), "skills", false),
        &mut issues,
    );
    let paths = collect(
        normalize_paths(request.paths.as_deref(), false),
        &mut issues,
    );
    let pipeline = match normalize_pipeline(&request.pipeline) {
        Ok(stages) => stages,
        Err(mut problems) => {
            issues.append(&mut problems);
            Vec::new()
        }
    };
    let per_query_limit =
        resolve_per_query_limit(per_query_limit, &queries, &pipeline, &mut issues);

    finish(
        request,
        issues,
        Projection {
            operation,
            view,
            search_match,
            search_result,
            read_mode,
            output_mode,
            max_depth,
            line_count,
            start_line,
            end_line,
            per_query_limit,
            limit,
            max_output_chars,
            category,
            skills,
            paths,
            queries,
            pipeline,
        },
        None,
    )
}

/// Validates a request that used the model-facing simple shape.
///
/// Applies Python's `_apply_simple_defaults` before the full pass: `list` always uses the
/// details view, `search` pages at 10 hits per query (capped at 20 overall), `limit`
/// may not exceed [`MAX_RESULT_LIMIT`], `query` is rejected on `list`/`read`, and
/// `skills` is rejected outside `read`. The returned [`ValidatedRequest::page_size`] is
/// Python's `_page_size` (`Some(0)` for a `list` without a limit).
///
/// # Errors
///
/// Returns every [`RequestIssue`] found, including those of the full pass run over the
/// defaulted request.
pub fn apply_simple_defaults(
    request: &SkillIndexRequest,
) -> std::result::Result<ValidatedRequest, Vec<RequestIssue>> {
    let mut issues: Vec<RequestIssue> = Vec::new();
    let operation = choose::<Operation>(&request.operation, "operation", &mut issues);
    let limit = positive_integer(request.limit, "limit", &mut issues);
    if let Some(limit) = limit {
        if limit > MAX_RESULT_LIMIT {
            issues.push(RequestIssue::new(
                "limit",
                format!("must not exceed {MAX_RESULT_LIMIT} for {operation}"),
            ));
        }
    }
    let mut normalized = request.clone();
    let page_size = match operation {
        Operation::List => {
            if request.query.is_some() {
                issues.push(RequestIssue::new(
                    "query",
                    "is not accepted by list (only category and limit are)",
                ));
            }
            if request.skills.is_some() {
                issues.push(RequestIssue::new(
                    "skills",
                    "is not accepted by list (only category and limit are)",
                ));
            }
            normalized.view = ListView::Details.as_str().to_string();
            Some(limit.unwrap_or(0))
        }
        Operation::Search => {
            if request.skills.is_some() {
                issues.push(RequestIssue::new("skills", "are valid only for read"));
            }
            let queries = match normalize_queries(request.query.as_deref()) {
                Ok(values) => values,
                Err(mut problems) => {
                    issues.append(&mut problems);
                    Vec::new()
                }
            };
            let default = (DEFAULT_SEARCH_PAGE_SIZE * queries.len()).min(MAX_SEARCH_PAGE_SIZE);
            Some(limit.unwrap_or(default))
        }
        Operation::Read => {
            if request.query.is_some() {
                issues.push(RequestIssue::new("query", "is valid only for search"));
            }
            None
        }
    };
    if !issues.is_empty() {
        return Err(issues);
    }
    validate_with_page_size(&normalized, page_size)
}

/// Runs [`validate`] over `request` with `page_size` attached.
fn validate_with_page_size(
    request: &SkillIndexRequest,
    page_size: Option<usize>,
) -> std::result::Result<ValidatedRequest, Vec<RequestIssue>> {
    let validated = validate(request)?;
    Ok(ValidatedRequest {
        page_size,
        page_offset: 0,
        ..validated
    })
}

/// The validated projection [`finish`] attaches to a request.
struct Projection {
    operation: Operation,
    view: ListView,
    search_match: SearchMatch,
    search_result: SearchResult,
    read_mode: ReadMode,
    output_mode: OutputMode,
    max_depth: Option<usize>,
    line_count: Option<usize>,
    start_line: Option<usize>,
    end_line: Option<usize>,
    per_query_limit: Option<usize>,
    limit: Option<usize>,
    max_output_chars: Option<usize>,
    category: Option<String>,
    skills: Vec<String>,
    paths: Vec<String>,
    queries: Vec<String>,
    pipeline: Vec<PipelineStage>,
}

/// Builds the normalized request and returns it as a [`ValidatedRequest`].
#[allow(clippy::too_many_lines)]
fn finish(
    request: &SkillIndexRequest,
    issues: Vec<RequestIssue>,
    projection: Projection,
    page_size: Option<usize>,
) -> std::result::Result<ValidatedRequest, Vec<RequestIssue>> {
    if !issues.is_empty() {
        return Err(issues);
    }
    let mut normalized = request.clone();
    normalized.category.clone_from(&projection.category);
    normalized.skills = request.skills.as_ref().map(|_| projection.skills.clone());
    normalized.paths = request.paths.as_ref().map(|_| projection.paths.clone());
    normalized.query = request.query.as_ref().map(|_| projection.queries.clone());
    normalized.pipeline.clone_from(&projection.pipeline);
    if let Some(value) = projection.category.as_deref() {
        normalized.category = Some(value.to_string());
    }
    let per_query = projection
        .per_query_limit
        .and_then(|value| i64::try_from(value).ok());
    if per_query.is_some() {
        normalized.per_query_limit = per_query;
    }
    Ok(ValidatedRequest {
        request: normalized,
        operation: projection.operation,
        view: projection.view,
        search_match: projection.search_match,
        search_result: projection.search_result,
        read_mode: projection.read_mode,
        output_mode: projection.output_mode,
        recursive: request.recursive,
        directory_entry: request.directory_entry,
        directories_only: request.directories_only,
        case_insensitive: request.case_insensitive,
        fixed_strings: request.fixed_strings,
        max_depth: projection.max_depth,
        line_count: projection.line_count,
        start_line: projection.start_line,
        end_line: projection.end_line,
        per_query_limit: projection.per_query_limit,
        limit: projection.limit,
        max_output_chars: projection.max_output_chars,
        disable_output_truncation: request.disable_output_truncation,
        category: projection.category,
        skills: projection.skills,
        paths: projection.paths,
        queries: projection.queries,
        pipeline: projection.pipeline,
        page_size,
        page_offset: 0,
    })
}

/// Implements [`ModelEnum`] for the list view, which lives with its renderer.
impl ModelEnum for ListView {
    fn parse(value: &str) -> Option<Self> {
        Self::parse(value)
    }

    fn names() -> Vec<&'static str> {
        Self::ALL.into_iter().map(Self::as_str).collect()
    }
}

/// Parses one enum slot, recording an issue and falling back to the default variant.
fn choose<E: ModelEnum>(raw: &str, field: &str, issues: &mut Vec<RequestIssue>) -> E {
    if let Some(value) = E::parse(raw) {
        return value;
    }
    let allowed = E::names().join(", ");
    issues.push(RequestIssue::new(
        field,
        format!("must be one of: {allowed}"),
    ));
    E::default()
}

/// Parses a positive integer slot, recording an issue for zero or a negative value.
fn positive_integer(
    value: Option<i64>,
    field: &str,
    issues: &mut Vec<RequestIssue>,
) -> Option<usize> {
    let raw = value?;
    match usize::try_from(raw) {
        Ok(parsed) if parsed >= 1 => Some(parsed),
        _ => {
            issues.push(RequestIssue::new(field, "must be a positive integer"));
            None
        }
    }
}

/// Parses the output budget, recording an issue when it is out of range.
fn output_chars(
    value: Option<i64>,
    disable: bool,
    issues: &mut Vec<RequestIssue>,
) -> Option<usize> {
    if disable {
        return None;
    }
    let parsed = positive_integer(value, "max_output_chars", issues)?;
    if !(MIN_OUTPUT_CHARS..=MAX_OUTPUT_CHARS).contains(&parsed) {
        issues.push(RequestIssue::new(
            "max_output_chars",
            format!("must be between {MIN_OUTPUT_CHARS} and {MAX_OUTPUT_CHARS}"),
        ));
        return None;
    }
    Some(parsed)
}

/// Trims a category, treating blank as absent.
fn normalized_category(category: Option<&str>) -> Option<String> {
    let value = category?.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

/// Rejects a field that only one operation accepts.
fn check_only_for_operation(
    request: &SkillIndexRequest,
    operation: Operation,
    view: ListView,
    search_match: SearchMatch,
    search_result: SearchResult,
    read_mode: ReadMode,
    issues: &mut Vec<RequestIssue>,
) {
    check_list_only_fields(request, operation, view, issues);
    check_search_only_fields(request, operation, search_match, search_result, issues);
    check_read_only_fields(request, operation, read_mode, issues);
    check_skill_scope(request, operation, issues);
}
/// Rejects the `list`-only fields on any other operation.
fn check_list_only_fields(
    request: &SkillIndexRequest,
    operation: Operation,
    view: ListView,
    issues: &mut Vec<RequestIssue>,
) {
    if operation == Operation::List {
        return;
    }
    if view != ListView::Names {
        issues.push(RequestIssue::new(
            "view",
            "is only valid for operation=list",
        ));
    }
    if request.recursive {
        issues.push(RequestIssue::new(
            "recursive",
            "is only valid for operation=list",
        ));
    }
    if request.directory_entry {
        issues.push(RequestIssue::new(
            "directory_entry",
            "is only valid for operation=list",
        ));
    }
    if request.directories_only {
        issues.push(RequestIssue::new(
            "directories_only",
            "is only valid for operation=list",
        ));
    }
}
/// Rejects the `search`-only fields on any other operation.
fn check_search_only_fields(
    request: &SkillIndexRequest,
    operation: Operation,
    search_match: SearchMatch,
    search_result: SearchResult,
    issues: &mut Vec<RequestIssue>,
) {
    if operation == Operation::Search {
        return;
    }
    if request.query.is_some() {
        issues.push(RequestIssue::new(
            "query",
            "is only valid for operation=search",
        ));
    }
    if request.per_query_limit.is_some() {
        issues.push(RequestIssue::new(
            "per_query_limit",
            "is only valid for operation=search",
        ));
    }
    if search_match != SearchMatch::Content {
        issues.push(RequestIssue::new(
            "match",
            "is only valid for operation=search",
        ));
    }
    if search_result != SearchResult::Files {
        issues.push(RequestIssue::new(
            "result",
            "is only valid for operation=search",
        ));
    }
    if !request.case_insensitive {
        issues.push(RequestIssue::new(
            "case_insensitive",
            "is only valid for operation=search",
        ));
    }
    if request.fixed_strings {
        issues.push(RequestIssue::new(
            "fixed_strings",
            "is only valid for operation=search",
        ));
    }
}
/// Rejects the `read`-only fields on any other operation.
fn check_read_only_fields(
    request: &SkillIndexRequest,
    operation: Operation,
    read_mode: ReadMode,
    issues: &mut Vec<RequestIssue>,
) {
    if operation == Operation::Read {
        return;
    }
    if read_mode != ReadMode::Full {
        issues.push(RequestIssue::new(
            "read_mode",
            "is only valid for operation=read",
        ));
    }
    if request.line_count.is_some() {
        issues.push(RequestIssue::new(
            "line_count",
            "is only valid for operation=read",
        ));
    }
    if request.start_line.is_some() {
        issues.push(RequestIssue::new(
            "start_line",
            "is only valid for operation=read",
        ));
    }
    if request.end_line.is_some() {
        issues.push(RequestIssue::new(
            "end_line",
            "is only valid for operation=read",
        ));
    }
}
/// Applies the `skills`/`paths` scope rules: skills are `read`-only, exclusive with paths,
/// and one of the two must be supplied.
fn check_skill_scope(
    request: &SkillIndexRequest,
    operation: Operation,
    issues: &mut Vec<RequestIssue>,
) {
    if request.skills.is_some() && operation != Operation::Read {
        issues.push(RequestIssue::new("skills", "are valid only for read"));
    }
    if operation != Operation::Read {
        return;
    }
    if request.skills.is_some() && request.paths.is_some() {
        issues.push(RequestIssue::new(
            "paths",
            "is mutually exclusive with skills",
        ));
    }
    if request.skills.is_none() && request.paths.is_none() {
        issues.push(RequestIssue::new(
            "skills",
            "are required for read (or supply paths)",
        ));
    }
}
/// Rejects a scope that names both a category and paths, or a category on `read`.
fn check_scope(
    has_category: bool,
    has_paths: bool,
    operation: Operation,
    issues: &mut Vec<RequestIssue>,
) {
    if has_category && has_paths {
        issues.push(RequestIssue::new(
            "paths",
            "is mutually exclusive with category",
        ));
    }
    if has_category && operation == Operation::Read {
        issues.push(RequestIssue::new(
            "category",
            "is only valid for operation=list or operation=search",
        ));
    }
}

/// Applies the `list` view rules.
fn check_list(
    request: &SkillIndexRequest,
    view: ListView,
    max_depth: Option<usize>,
    issues: &mut Vec<RequestIssue>,
) {
    if view == ListView::Tree {
        if request.recursive {
            issues.push(RequestIssue::new(
                "recursive",
                "is not valid with tree view",
            ));
        }
        if request.directory_entry {
            issues.push(RequestIssue::new(
                "directory_entry",
                "is not valid with tree view",
            ));
        }
        return;
    }
    if max_depth.is_some() {
        issues.push(RequestIssue::new("max_depth", "requires tree view"));
    }
    if request.directories_only {
        issues.push(RequestIssue::new("directories_only", "requires tree view"));
    }
}

/// Applies the `search` match/result rules.
fn check_search(
    search_match: SearchMatch,
    search_result: SearchResult,
    issues: &mut Vec<RequestIssue>,
) {
    if search_match != SearchMatch::Content && search_result != SearchResult::Files {
        issues.push(RequestIssue::new(
            "result",
            "is only valid for content search",
        ));
    }
}

/// Applies the `read` mode and line-range rules.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn check_read(
    request: &SkillIndexRequest,
    read_mode: ReadMode,
    line_count: Option<usize>,
    start_line: Option<usize>,
    end_line: Option<usize>,
    max_depth: Option<usize>,
    has_category: bool,
    issues: &mut Vec<RequestIssue>,
) {
    if max_depth.is_some() {
        issues.push(RequestIssue::new(
            "max_depth",
            "is only valid for operation=list or operation=search",
        ));
    }
    if has_category {
        // Reported by `check_scope`; kept out of here to name the field once.
    }
    if read_mode == ReadMode::Full
        && (line_count.is_some() || start_line.is_some() || end_line.is_some())
    {
        issues.push(RequestIssue::new(
            "line_count",
            "and start_line/end_line require head or range read_mode",
        ));
    }
    if read_mode == ReadMode::Head && (start_line.is_some() || end_line.is_some()) {
        issues.push(RequestIssue::new(
            "start_line",
            "and end_line are valid only for range read_mode",
        ));
    }
    if read_mode == ReadMode::Range && line_count.is_some() {
        issues.push(RequestIssue::new(
            "line_count",
            "is valid only for head read_mode",
        ));
    }
    if read_mode == ReadMode::Range && (start_line.is_none() || end_line.is_none()) {
        issues.push(RequestIssue::new(
            "start_line",
            "and end_line are required for range read_mode",
        ));
    }
    if read_mode == ReadMode::Range {
        if let (Some(start), Some(end)) = (start_line, end_line) {
            if end < start {
                issues.push(RequestIssue::new(
                    "end_line",
                    "must be greater than or equal to start_line",
                ));
            }
        }
    }
    if request.skills.is_some() && request.paths.is_some() {
        // Reported by `check_only_for_operation`.
    }
}

/// Applies the output-mode rules.
fn check_output(
    operation: Operation,
    output_mode: OutputMode,
    request: &SkillIndexRequest,
    issues: &mut Vec<RequestIssue>,
) {
    if output_mode != OutputMode::Count {
        return;
    }
    if operation == Operation::Read {
        issues.push(RequestIssue::new(
            "output_mode",
            "count is only valid for operation=list or operation=search",
        ));
    }
    if !request.pipeline.is_empty() {
        issues.push(RequestIssue::new(
            "pipeline",
            "is not valid with output_mode=count",
        ));
    }
}

/// Resolves `per_query_limit` against the pipeline, following Python's tail rules.
///
/// A batch (more than one query) accepts exactly one `limit` stage and uses its `lines`
/// as the per-query limit, capped at [`MAX_PER_QUERY_LIMIT`]; a single query accepts a
/// lone `limit` stage only when it agrees with `per_query_limit`, which itself is capped
/// at [`MAX_PER_QUERY_LIMIT`].
fn resolve_per_query_limit(
    per_query_limit: Option<usize>,
    queries: &[String],
    pipeline: &[PipelineStage],
    issues: &mut Vec<RequestIssue>,
) -> Option<usize> {
    let single_limit = if pipeline.len() == 1 && stage_is(pipeline, PipelineOperation::Limit) {
        pipeline.first().and_then(|stage| stage.lines)
    } else {
        None
    };
    if queries.len() > 1 {
        let Some(lines) = single_limit else {
            if !pipeline.is_empty() {
                issues.push(RequestIssue::new(
                    "pipeline",
                    "queries support only one limit pipeline stage",
                ));
            }
            return per_query_limit;
        };
        let Some(lines) = usize::try_from(lines).ok().filter(|value| *value >= 1) else {
            return per_query_limit;
        };
        if lines > MAX_PER_QUERY_LIMIT {
            issues.push(RequestIssue::new(
                "pipeline[0].lines",
                format!("must not exceed {MAX_PER_QUERY_LIMIT} for queries"),
            ));
        }
        if per_query_limit.is_some() && per_query_limit != Some(lines) {
            issues.push(RequestIssue::new(
                "per_query_limit",
                "conflicts with pipeline limit",
            ));
            return per_query_limit;
        }
        return Some(lines);
    }
    let per_query_limit = per_query_limit?;
    if per_query_limit > MAX_PER_QUERY_LIMIT {
        issues.push(RequestIssue::new(
            "per_query_limit",
            format!("must not exceed {MAX_PER_QUERY_LIMIT}"),
        ));
    }
    if pipeline.is_empty() {
        return Some(per_query_limit);
    }
    let lines = single_limit.and_then(|value| usize::try_from(value).ok());
    if lines != Some(per_query_limit) {
        issues.push(RequestIssue::new(
            "per_query_limit",
            "conflicts with pipeline limit",
        ));
    }
    Some(per_query_limit)
}

/// Whether the single stage in `pipeline` is `operation`.
fn stage_is(pipeline: &[PipelineStage], operation: PipelineOperation) -> bool {
    pipeline
        .first()
        .and_then(|stage| PipelineOperation::parse(&stage.operation))
        == Some(operation)
}

/// Validates and normalizes a pipeline.
///
/// # Errors
///
/// Returns [`Error::Validation`] `field: message` for the first problem: an unknown
/// operation, a missing required field, a field the operation does not take, an
/// out-of-range line number, or `end_line < start_line`.
pub fn validate_pipeline(stages: &[PipelineStage]) -> Result<Vec<PipelineStage>> {
    normalize_pipeline(stages).map_err(|issues| first_issue(&issues))
}

/// Validates and normalizes a pipeline, collecting one issue per problem.
#[allow(clippy::too_many_lines)]
fn normalize_pipeline(
    stages: &[PipelineStage],
) -> std::result::Result<Vec<PipelineStage>, Vec<RequestIssue>> {
    let mut issues: Vec<RequestIssue> = Vec::new();
    let mut normalized: Vec<PipelineStage> = Vec::with_capacity(stages.len());
    for (index, stage) in stages.iter().enumerate() {
        let field = format!("pipeline[{index}]");
        let operation = choose::<PipelineOperation>(
            &stage.operation,
            &format!("{field}.operation"),
            &mut issues,
        );
        let query = match stage.query.as_deref() {
            Some(raw) => match bounded_query(raw, &format!("{field}.query")) {
                Ok(value) => Some(value),
                Err(issue) => {
                    issues.push(issue);
                    None
                }
            },
            None => None,
        };
        let lines = positive_integer(stage.lines, &format!("{field}.lines"), &mut issues);
        if lines.is_some_and(|lines| lines > MAX_LINES) {
            issues.push(RequestIssue::new(
                format!("{field}.lines"),
                format!("must not exceed {MAX_LINES}"),
            ));
        }
        let start = positive_integer(
            stage.start_line,
            &format!("{field}.start_line"),
            &mut issues,
        );
        let end = positive_integer(stage.end_line, &format!("{field}.end_line"), &mut issues);
        match operation {
            PipelineOperation::Limit => {
                if lines.is_none() {
                    issues.push(RequestIssue::new(
                        format!("{field}.lines"),
                        "is required for limit",
                    ));
                }
                if query.is_some()
                    || start.is_some()
                    || end.is_some()
                    || stage.case_insensitive
                    || stage.invert
                    || stage.fixed_strings
                {
                    issues.push(RequestIssue::new(&field, "has fields not valid for limit"));
                }
                normalized.push(PipelineStage::limit(
                    stage
                        .lines
                        .unwrap_or(1)
                        .max(1)
                        .unsigned_abs()
                        .try_into()
                        .unwrap_or(1),
                ));
            }
            PipelineOperation::Slice => {
                if start.is_none() || end.is_none() {
                    issues.push(RequestIssue::new(
                        &field,
                        "slice requires start_line and end_line",
                    ));
                }
                if let (Some(start), Some(end)) = (start, end) {
                    if end < start {
                        issues.push(RequestIssue::new(
                            format!("{field}.end_line"),
                            "must be >= start_line",
                        ));
                    }
                }
                if query.is_some()
                    || lines.is_some()
                    || stage.case_insensitive
                    || stage.invert
                    || stage.fixed_strings
                {
                    issues.push(RequestIssue::new(&field, "has fields not valid for slice"));
                }
                normalized.push(PipelineStage {
                    operation: PipelineOperation::Slice.as_str().to_string(),
                    start_line: stage.start_line,
                    end_line: stage.end_line,
                    ..PipelineStage::default()
                });
            }
            PipelineOperation::Filter => {
                if query.is_none() && stage.query.is_none() {
                    issues.push(RequestIssue::new(
                        format!("{field}.query"),
                        "is required for filter",
                    ));
                }
                if lines.is_some() || start.is_some() || end.is_some() {
                    issues.push(RequestIssue::new(&field, "has fields not valid for filter"));
                }
                normalized.push(PipelineStage {
                    operation: PipelineOperation::Filter.as_str().to_string(),
                    query: query.clone().or_else(|| stage.query.clone()),
                    case_insensitive: stage.case_insensitive,
                    invert: stage.invert,
                    fixed_strings: stage.fixed_strings,
                    ..PipelineStage::default()
                });
            }
            PipelineOperation::Count => {
                if query.is_some()
                    || lines.is_some()
                    || start.is_some()
                    || end.is_some()
                    || stage.case_insensitive
                    || stage.invert
                    || stage.fixed_strings
                {
                    issues.push(RequestIssue::new(&field, "count takes no other fields"));
                }
                normalized.push(PipelineStage::count());
            }
        }
    }
    if issues.is_empty() {
        Ok(normalized)
    } else {
        Err(issues)
    }
}

/// Resolves the output budget.
///
/// `None` when `disable` is set (the budget is ignored), `Some(default)` when no budget
/// was requested, and [`Error::Validation`] when the requested budget is zero or outside
/// [`MIN_OUTPUT_CHARS`]`..=`[`MAX_OUTPUT_CHARS`].
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `max_output_chars` for an out-of-range budget.
pub fn output_budget(value: Option<usize>, disable: bool, default: usize) -> Result<Option<usize>> {
    if disable {
        return Ok(None);
    }
    match value {
        None => Ok(Some(default)),
        Some(parsed) => {
            if parsed == 0 {
                return Err(validation("max_output_chars", "must be a positive integer"));
            }
            if !(MIN_OUTPUT_CHARS..=MAX_OUTPUT_CHARS).contains(&parsed) {
                return Err(validation(
                    "max_output_chars",
                    format!("must be between {MIN_OUTPUT_CHARS} and {MAX_OUTPUT_CHARS}"),
                ));
            }
            Ok(Some(parsed))
        }
    }
}

/// Validates and normalizes the search queries.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `query` when the field is missing, empty, longer
/// than [`MAX_QUERIES`], or holds a blank, NUL-containing, or over-long value.
#[allow(clippy::ref_option)] // the request-shaped signature is part of the tool contract
pub fn validate_queries(query: &Option<Vec<String>>) -> Result<Vec<String>> {
    normalize_queries(query.as_deref()).map_err(|issues| first_issue(&issues))
}

/// Validates and normalizes the queries, collecting one issue per problem.
fn normalize_queries(
    query: Option<&[String]>,
) -> std::result::Result<Vec<String>, Vec<RequestIssue>> {
    let Some(items) = query else {
        return Err(vec![RequestIssue::new(
            "query",
            "must be a non-empty array of 1-8 strings",
        )]);
    };
    let mut issues: Vec<RequestIssue> = Vec::new();
    if items.is_empty() || items.len() > MAX_QUERIES {
        issues.push(RequestIssue::new(
            "query",
            format!("must contain 1-{MAX_QUERIES} strings"),
        ));
    }
    let mut values: Vec<String> = Vec::new();
    for item in items {
        match bounded_query(item, "query") {
            Ok(value) => {
                if !values.contains(&value) {
                    values.push(value);
                }
            }
            Err(issue) => issues.push(issue),
        }
    }
    if issues.is_empty() {
        Ok(values)
    } else {
        Err(issues)
    }
}

/// Trims and bounds one query value.
fn bounded_query(value: &str, field: &str) -> std::result::Result<String, RequestIssue> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(RequestIssue::new(field, "must be a non-empty string"));
    }
    if trimmed.contains('\0') {
        return Err(RequestIssue::new(field, "must not contain a null byte"));
    }
    if trimmed.chars().count() > MAX_QUERY_CHARS {
        return Err(RequestIssue::new(
            field,
            format!("must not exceed {MAX_QUERY_CHARS} characters"),
        ));
    }
    Ok(trimmed.to_string())
}

/// Validates and normalizes the scoped paths.
///
/// `None` becomes `["/"]` unless `required`, and every path is trimmed and de-duplicated.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `paths` when a required path list is missing,
/// the list is empty or longer than [`MAX_PATHS`], or a path is blank, contains a NUL, or
/// exceeds [`MAX_PATH_CHARS`].
#[allow(clippy::ref_option)] // the request-shaped signature is part of the tool contract
pub fn validate_paths(paths: &Option<Vec<String>>, required: bool) -> Result<Vec<String>> {
    normalize_paths(paths.as_deref(), required).map_err(|issues| first_issue(&issues))
}

/// Validates and normalizes the scoped paths, collecting one issue per problem.
fn normalize_paths(
    paths: Option<&[String]>,
    required: bool,
) -> std::result::Result<Vec<String>, Vec<RequestIssue>> {
    let Some(items) = paths else {
        if required {
            return Err(vec![RequestIssue::new("paths", "are required for read")]);
        }
        return Ok(vec![ROOT_PATH.to_string()]);
    };
    let mut issues: Vec<RequestIssue> = Vec::new();
    if items.is_empty() {
        issues.push(RequestIssue::new("paths", "must be a non-empty array"));
    }
    if items.len() > MAX_PATHS {
        issues.push(RequestIssue::new(
            "paths",
            format!("must contain at most {MAX_PATHS} items"),
        ));
    }
    let mut values: Vec<String> = Vec::new();
    for item in items {
        match bounded_value(item, "paths", MAX_PATH_CHARS) {
            Ok(value) => {
                if !values.contains(&value) {
                    values.push(value);
                }
            }
            Err(issue) => issues.push(issue),
        }
    }
    if issues.is_empty() {
        Ok(values)
    } else {
        Err(issues)
    }
}

/// Validates and normalizes a list of identifiers (Skill ids).
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `name` when a required list is missing, the list
/// is empty or longer than [`MAX_PATHS`], or a value is blank, contains a NUL, or exceeds
/// [`MAX_IDENTIFIER_CHARS`].
#[allow(clippy::ref_option)] // the request-shaped signature is part of the tool contract
pub fn validate_identifiers(
    values: &Option<Vec<String>>,
    name: &str,
    required: bool,
) -> Result<Vec<String>> {
    normalize_identifiers(values.as_deref(), name, required).map_err(|issues| first_issue(&issues))
}

/// Validates and normalizes identifiers, collecting one issue per problem.
fn normalize_identifiers(
    values: Option<&[String]>,
    name: &str,
    required: bool,
) -> std::result::Result<Vec<String>, Vec<RequestIssue>> {
    let Some(items) = values else {
        if required {
            return Err(vec![RequestIssue::new(name, "are required")]);
        }
        return Ok(Vec::new());
    };
    let mut issues: Vec<RequestIssue> = Vec::new();
    if items.is_empty() {
        issues.push(RequestIssue::new(name, "must be a non-empty array"));
    }
    if items.len() > MAX_PATHS {
        issues.push(RequestIssue::new(
            name,
            format!("must contain at most {MAX_PATHS} items"),
        ));
    }
    let mut normalized: Vec<String> = Vec::new();
    for item in items {
        match bounded_value(item, name, MAX_IDENTIFIER_CHARS) {
            Ok(value) => {
                if !normalized.contains(&value) {
                    normalized.push(value);
                }
            }
            Err(issue) => issues.push(issue),
        }
    }
    if issues.is_empty() {
        Ok(normalized)
    } else {
        Err(issues)
    }
}

/// Trims and bounds one string value.
fn bounded_value(
    value: &str,
    field: &str,
    limit: usize,
) -> std::result::Result<String, RequestIssue> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(RequestIssue::new(field, "must be a non-empty string"));
    }
    if trimmed.contains('\0') {
        return Err(RequestIssue::new(field, "must not contain null bytes"));
    }
    if trimmed.chars().count() > limit {
        return Err(RequestIssue::new(
            field,
            format!("must not exceed {limit} characters"),
        ));
    }
    Ok(trimmed.to_string())
}

/// Resolves the category (or the legacy paths) against a view.
///
/// `category = None` validates the legacy `paths` (defaulting to `["/"]`); a category and
/// paths together are rejected; otherwise [`SkillDirectoryView::category_path`] resolves
/// the name, chain, or root alias.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a category/paths conflict, an unknown or ambiguous
/// category, or an invalid path list.
pub fn validate_category_paths(
    view: &SkillDirectoryView,
    category: Option<&str>,
    legacy_paths: Option<&[String]>,
) -> Result<Vec<String>> {
    let Some(category) = category else {
        return validate_paths(&legacy_paths.map(<[String]>::to_vec), false);
    };
    if legacy_paths.is_some() {
        return Err(validation(
            "paths",
            "category and paths are mutually exclusive",
        ));
    }
    Ok(vec![view.category_path(category)?])
}

/// Appends the issues of one validation result, returning its value or an empty vector.
fn collect(
    result: std::result::Result<Vec<String>, Vec<RequestIssue>>,
    issues: &mut Vec<RequestIssue>,
) -> Vec<String> {
    match result {
        Ok(values) => values,
        Err(mut problems) => {
            issues.append(&mut problems);
            Vec::new()
        }
    }
}

/// Converts an issue list into the crate's typed error.
fn first_issue(issues: &[RequestIssue]) -> Error {
    match issues.first().cloned() {
        Some(issue) => Error::validation(issue.to_string()),
        None => Error::validation("request is invalid"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn enums_parse_and_name_themselves() {
        assert_eq!(Operation::ALL.len(), 3);
        assert_eq!(Operation::parse("search"), Some(Operation::Search));
        assert_eq!(Operation::parse("SEARCH"), None);
        assert_eq!(Operation::Search.as_str(), "search");
        assert_eq!(Operation::Search.to_string(), "search");
        assert_eq!(SearchMatch::parse("path"), Some(SearchMatch::Path));
        assert_eq!(
            PipelineOperation::names(),
            ["limit", "slice", "filter", "count"]
        );
    }

    #[test]
    fn request_defaults_match_the_python_defaults() {
        let request = SkillIndexRequest::default();
        assert_eq!(request.operation, "list");
        assert_eq!(request.view, "names");
        assert_eq!(request.search_match, "content");
        assert_eq!(request.result, "files");
        assert!(request.case_insensitive);
        assert_eq!(request.read_mode, "full");
        assert_eq!(request.output_mode, "entries");
        let decoded: SkillIndexRequest =
            serde_json::from_str(r#"{"operation":"search","match":"name","future":1}"#).unwrap();
        assert_eq!(decoded.search_match, "name");
        assert!(decoded.case_insensitive);
    }

    #[test]
    fn a_valid_list_request_passes() {
        let request = SkillIndexRequest::new(Operation::List);
        let validated = validate(&request).unwrap();
        assert_eq!(validated.operation, Operation::List);
        assert_eq!(validated.view, ListView::Names);
        assert_eq!(validated.paths, ["/"]);
        assert_eq!(validated.page_offset, 0);
        assert!(validated.page_size.is_none());
    }

    #[test]
    fn issues_accumulate_and_name_their_fields() {
        let request = SkillIndexRequest {
            operation: "bogus".to_string(),
            view: "wide".to_string(),
            max_depth: Some(0),
            ..SkillIndexRequest::default()
        };
        let issues = validate(&request).unwrap_err();
        let fields: Vec<&str> = issues.iter().map(|issue| issue.field.as_str()).collect();
        assert!(fields.contains(&"operation"));
        assert!(fields.contains(&"view"));
        assert!(fields.contains(&"max_depth"));
        assert!(issues
            .iter()
            .any(|issue| issue.to_string() == "view: must be one of: names, details, tree"));
    }

    #[test]
    fn pipeline_rules_are_typed_errors_too() {
        assert_eq!(
            validate_pipeline(&[PipelineStage::limit(3)]).unwrap(),
            [PipelineStage::limit(3)]
        );
        let error = validate_pipeline(&[PipelineStage::limit(0)]).unwrap_err();
        assert_eq!(
            error.message(),
            "pipeline[0].lines: must be a positive integer"
        );
        let error = validate_pipeline(&[PipelineStage::slice(5, 4)]).unwrap_err();
        assert_eq!(
            error.message(),
            "pipeline[0].end_line: must be >= start_line"
        );
        let error = validate_pipeline(&[PipelineStage::limit(5_001)]).unwrap_err();
        assert_eq!(error.message(), "pipeline[0].lines: must not exceed 5000");
        let mut stage = PipelineStage::limit(2);
        stage.query = Some("pdf".to_string());
        let error = validate_pipeline(&[stage]).unwrap_err();
        assert_eq!(
            error.message(),
            "pipeline[0]: has fields not valid for limit"
        );
        let mut missing = PipelineStage::filter("x");
        missing.query = None;
        let error = validate_pipeline(&[missing]).unwrap_err();
        assert_eq!(error.message(), "pipeline[0].query: is required for filter");
        let mut bad = PipelineStage::count();
        bad.operation = "shuffle".to_string();
        let error = validate_pipeline(&[bad]).unwrap_err();
        assert!(error
            .message()
            .contains("pipeline[0].operation: must be one of"));
    }

    #[test]
    fn budgets_are_bounded() {
        assert_eq!(output_budget(None, false, 12_000).unwrap(), Some(12_000));
        assert_eq!(output_budget(Some(600), false, 12_000).unwrap(), Some(600));
        assert_eq!(output_budget(Some(600), true, 12_000).unwrap(), None);
        assert!(output_budget(Some(0), false, 12_000).is_err());
        assert!(output_budget(Some(511), false, 12_000).is_err());
        assert!(output_budget(Some(48_001), false, 12_000).is_err());
    }

    #[test]
    fn queries_paths_and_identifiers_are_normalized() {
        assert_eq!(
            validate_queries(&Some(vec![" pdf ".to_string(), "pdf".to_string()])).unwrap(),
            ["pdf"]
        );
        assert!(validate_queries(&None).is_err());
        assert!(validate_queries(&Some(vec!["  ".to_string()])).is_err());
        assert_eq!(validate_paths(&None, false).unwrap(), ["/"]);
        assert!(validate_paths(&None, true).is_err());
        assert_eq!(
            validate_paths(&Some(vec![" /a ".to_string(), "/a".to_string()]), false).unwrap(),
            ["/a"]
        );
        assert_eq!(
            validate_identifiers(&Some(vec![" alpha ".to_string()]), "skills", true).unwrap(),
            ["alpha"]
        );
        assert!(validate_identifiers(&None, "skills", true).is_err());
    }
}
