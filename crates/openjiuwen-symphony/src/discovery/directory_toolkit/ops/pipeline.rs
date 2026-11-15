//! The post-processing pipeline: `limit`, `slice`, `filter`, and `count` over rendered rows.

use crate::discovery::directory_toolkit::render::{row_count, Row};
use crate::discovery::directory_toolkit::validate::{OutputMode, PipelineOperation, PipelineStage};
use crate::error::{validation, Result};

use super::request::positive;
use super::search::TextMatcher;
use super::PipelineOutcome;

/// Applies a pipeline and the output mode to rendered rows.
///
/// Every row is first flattened into one row per rendered line (Python does the same),
/// then the stages run in order. A `count` stage replaces the rows with a single count
/// row; the output mode `count` does the same after the stages.
///
/// `filter` matches a row's whole text with the same substitution the rest of the crate
/// uses for Python's matcher: a case-(in)sensitive substring, or a `*`/`?` glob when the
/// pattern carries a wildcard and `fixed_strings` is unset. `fixed_strings` is accepted
/// for compatibility and only disables that glob reading; there is no regex engine.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) for a malformed stage: a missing or non-positive line
/// count, a blank filter query, an inverted slice, or an unknown operation.
pub fn apply_pipeline(
    rows: &[Row],
    pipeline: &[PipelineStage],
    output_mode: OutputMode,
) -> Result<PipelineOutcome> {
    let mut current: Vec<Row> = rows.to_vec();
    let mut complete = true;
    let mut count_value: Option<usize> = None;
    let mut count_seen = false;
    if !pipeline.is_empty() {
        current = current
            .iter()
            .flat_map(|row| {
                row.text
                    .split('\n')
                    .map(|line| Row::new(line, row.worker_id.clone()))
                    .collect::<Vec<Row>>()
            })
            .collect();
    }
    for stage in pipeline {
        let Some(operation) = PipelineOperation::parse(&stage.operation) else {
            return Err(validation(
                "pipeline.operation",
                format!("must be one of: {}", PipelineOperation::names().join(", ")),
            ));
        };
        match operation {
            PipelineOperation::Limit => {
                count_value = None;
                let limit = positive(stage.lines, "lines")?;
                complete = complete && current.len() <= limit;
                current.truncate(limit);
            }
            PipelineOperation::Slice => {
                count_value = None;
                let start = positive(stage.start_line, "start_line")?;
                let end = positive(stage.end_line, "end_line")?;
                if end < start {
                    return Err(validation(
                        "pipeline.end_line",
                        "must be greater than or equal to start_line",
                    ));
                }
                complete = complete && start == 1 && end >= current.len();
                let from = (start - 1).min(current.len());
                let to = end.min(current.len());
                current = if from < to {
                    current[from..to].to_vec()
                } else {
                    Vec::new()
                };
            }
            PipelineOperation::Filter => {
                count_value = None;
                let query = stage.query.as_deref().unwrap_or_default();
                let matcher = TextMatcher::new(query, stage.case_insensitive, stage.fixed_strings)?;
                let invert = stage.invert;
                current.retain(|row| matcher.matches(&row.text) != invert);
            }
            PipelineOperation::Count => {
                let count = row_count(&current);
                count_value = Some(count);
                count_seen = true;
                current = vec![Row::text_only(count.to_string())];
            }
        }
    }
    if output_mode == OutputMode::Count {
        let count = row_count(&current);
        count_value = Some(count);
        count_seen = true;
        current = vec![Row::text_only(count.to_string())];
    }
    let transformed_count = if count_seen && count_value.is_none() {
        Some(row_count(&current))
    } else {
        None
    };
    Ok(PipelineOutcome {
        rows: current,
        complete,
        count_value,
        transformed_count,
    })
}
