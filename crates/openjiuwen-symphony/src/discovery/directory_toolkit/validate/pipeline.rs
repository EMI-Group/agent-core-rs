//! `pipeline` stage validation: the wire stages parsed, bounded, and normalized.

use crate::error::Result;

use super::limits::{bounded_query, first_issue, MAX_LINES};
use super::request::{PipelineOperation, PipelineStage, RequestIssue};
use super::rules::{choose, positive_integer};

/// Whether the single stage in `pipeline` is `operation`.
pub(super) fn stage_is(pipeline: &[PipelineStage], operation: PipelineOperation) -> bool {
    pipeline
        .first()
        .and_then(|stage| PipelineOperation::parse(&stage.operation))
        == Some(operation)
}

/// Validates and normalizes a pipeline.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) `field: message` for
/// the first problem: an unknown
/// operation, a missing required field, a field the operation does not take, an
/// out-of-range line number, or `end_line < start_line`.
pub fn validate_pipeline(stages: &[PipelineStage]) -> Result<Vec<PipelineStage>> {
    normalize_pipeline(stages).map_err(|issues| first_issue(&issues))
}

/// Validates and normalizes a pipeline, collecting one issue per problem.
#[allow(clippy::too_many_lines)]
pub(super) fn normalize_pipeline(
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
