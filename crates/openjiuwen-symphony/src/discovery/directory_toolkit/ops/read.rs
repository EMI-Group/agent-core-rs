//! The `read` operation: the metadata cards of the resolved Skills.

use std::collections::BTreeSet;

use crate::discovery::directory_toolkit::render::{metadata_card, Row};
use crate::discovery::directory_toolkit::validate::{validate_identifiers, validate_paths};
use crate::discovery::directory_toolkit::validate::{ReadMode, ValidatedRequest};
use crate::discovery::directory_toolkit::view::SkillDirectoryView;
use crate::error::{validation, Result};

use super::{Execution, DEFAULT_READ_LINES};

/// Runs the `read` operation over resolved Skill ids.
///
/// One row per Skill: `full` renders the whole
/// [`metadata_card`], `head`
/// its first [`DEFAULT_READ_LINES`] lines (or `line_count`), and `range` the 1-based
/// inclusive `start_line..=end_line` slice.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) `Unknown Skill: <id>` for an unknown id, or naming
/// `end_line` when a `range` read asks for an inverted range.
pub fn execute_read(
    view: &SkillDirectoryView,
    skills: &[String],
    request: &ValidatedRequest,
) -> Result<Execution> {
    let mut rows: Vec<Row> = Vec::with_capacity(skills.len());
    for skill in skills {
        let Some(record) = view.record(skill) else {
            return Err(validation("skills", format!("Unknown Skill: {skill}")));
        };
        let card = metadata_card(record);
        let mut lines: Vec<&str> = card.split('\n').collect();
        match request.read_mode {
            ReadMode::Full => {}
            ReadMode::Head => {
                lines.truncate(request.line_count.unwrap_or(DEFAULT_READ_LINES));
            }
            ReadMode::Range => {
                let start = request.start_line.unwrap_or(1);
                let end = request.end_line.unwrap_or(start);
                if end < start {
                    return Err(validation(
                        "end_line",
                        "must be greater than or equal to start_line",
                    ));
                }
                let from = start.saturating_sub(1).min(lines.len());
                let to = end.min(lines.len());
                lines = if from < to {
                    lines[from..to].to_vec()
                } else {
                    Vec::new()
                };
            }
        }
        rows.push(Row::new(lines.join("\n"), record.worker_id.clone()));
    }
    let total_count = rows.len();
    Ok(Execution {
        rows,
        total_count,
        complete: true,
    })
}

/// Resolves the Skill ids a `read` request names.
///
/// `skills` wins; the legacy `paths` form only resolves `META.md` paths that an earlier
/// result observed (`observed_meta_paths`), which is how Python's `_observed_meta_paths`
/// cross-call scope works.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) when both forms are supplied, when neither is, when a
/// path was never observed (`read accepts only Skills returned by an earlier skill_index
/// result`), or when a listed value is malformed.
pub fn resolve_read_skills(
    view: &SkillDirectoryView,
    request: &ValidatedRequest,
    observed_meta_paths: &BTreeSet<String>,
) -> Result<Vec<String>> {
    let skills = request.request.skills.as_ref();
    let legacy = request.request.paths.as_ref();
    if skills.is_some() && legacy.is_some() {
        return Err(validation(
            "skills",
            "skills and paths are mutually exclusive",
        ));
    }
    if let Some(paths) = legacy {
        let mut resolved: Vec<String> = Vec::new();
        for path in validate_paths(&Some(paths.clone()), true)? {
            let normalized = SkillDirectoryView::normalize_path(&path)?;
            if !observed_meta_paths.contains(&normalized) {
                return Err(validation(
                    "paths",
                    "read accepts only Skills returned by an earlier skill_index result",
                ));
            }
            let record = view.resolve_metadata_path(&normalized)?;
            if !resolved.contains(&record.worker_id) {
                resolved.push(record.worker_id.clone());
            }
        }
        return Ok(resolved);
    }
    validate_identifiers(&request.request.skills, "skills", true)
}
