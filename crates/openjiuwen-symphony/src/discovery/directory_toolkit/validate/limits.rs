//! The scalar limits and the bounded helpers that normalize one field at a time.

use crate::discovery::directory_toolkit::view::SkillDirectoryView;
use crate::error::{validation, Error, Result};

use super::request::RequestIssue;

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
pub(super) fn normalize_queries(
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
pub(super) fn bounded_query(value: &str, field: &str) -> std::result::Result<String, RequestIssue> {
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
pub(super) fn normalize_paths(
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
pub(super) fn normalize_identifiers(
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
pub(super) fn collect(
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
pub(super) fn first_issue(issues: &[RequestIssue]) -> Error {
    match issues.first().cloned() {
        Some(issue) => Error::validation(issue.to_string()),
        None => Error::validation("request is invalid"),
    }
}
