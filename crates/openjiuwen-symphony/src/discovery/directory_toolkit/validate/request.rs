//! The typed `skill_index` request and the parsed forms a validated request exposes.
//!
//! The enum-valued slots stay [`String`]s on [`SkillIndexRequest`] so an unknown value is
//! reportable instead of failing deserialization; [`Operation`], [`SearchMatch`],
//! [`SearchResult`], [`ReadMode`], [`OutputMode`], and [`PipelineOperation`] are the parsed
//! forms, and [`ValidatedRequest`] carries them alongside the normalized request.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::discovery::directory_toolkit::render::ListView;

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
pub(super) trait ModelEnum: Copy + Default {
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
/// and the enum slots stay strings so an unrecognized value is reported by [`validate`](crate::discovery::directory_toolkit::validate::validate)
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
/// resolves `category`/`paths` against a
/// [`SkillDirectoryView`](crate::discovery::directory_toolkit::view::SkillDirectoryView)
/// afterwards, with
/// [`validate_category_paths`](crate::discovery::directory_toolkit::validate::validate_category_paths).
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

/// Implements [`ModelEnum`] for the list view, which lives with its renderer.
impl ModelEnum for ListView {
    fn parse(value: &str) -> Option<Self> {
        Self::parse(value)
    }

    fn names() -> Vec<&'static str> {
        Self::ALL.into_iter().map(Self::as_str).collect()
    }
}
