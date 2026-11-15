//! The full validation pass: one issue per rejected field, accumulated in a single call.

use crate::discovery::directory_toolkit::render::ListView;

use super::limits::{
    collect, normalize_identifiers, normalize_paths, normalize_queries, DEFAULT_SEARCH_PAGE_SIZE,
    MAX_OUTPUT_CHARS, MAX_PER_QUERY_LIMIT, MAX_RESULT_LIMIT, MAX_SEARCH_PAGE_SIZE,
    MIN_OUTPUT_CHARS,
};
use super::pipeline::{normalize_pipeline, stage_is};
use super::request::{
    ModelEnum, Operation, OutputMode, PipelineOperation, PipelineStage, ReadMode, RequestIssue,
    SearchMatch, SearchResult, SkillIndexRequest, ValidatedRequest,
};

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

/// Parses one enum slot, recording an issue and falling back to the default variant.
pub(super) fn choose<E: ModelEnum>(raw: &str, field: &str, issues: &mut Vec<RequestIssue>) -> E {
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
pub(super) fn positive_integer(
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
