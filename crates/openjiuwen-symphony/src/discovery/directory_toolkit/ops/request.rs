//! The argument pass the toolkit and the executor share.

use crate::discovery::directory_toolkit::validate::{
    validate, RequestIssue, SkillIndexRequest, ValidatedRequest,
};
use crate::error::{validation, Result};

/// Validates one model-facing request into its typed projection.
///
/// Python's `_validate_request` together with the `_enum`/`_positive`/`_boolean` parsers
/// it calls. In Rust the booleans and integers are already typed, so those failures are
/// impossible; every remaining failure — an unknown enum value, an out-of-range number, a
/// field the operation does not accept, a malformed pipeline — becomes a [`RequestIssue`]
/// naming its field instead of a panic.
///
/// # Errors
///
/// Returns every issue found; a valid request yields `Ok`.
pub fn validate_request(
    request: &SkillIndexRequest,
) -> std::result::Result<ValidatedRequest, Vec<RequestIssue>> {
    validate(request)
}

/// Reads one positive line number out of a pipeline stage.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) naming `pipeline.<field>` when the stage does not carry a
/// positive value.
pub(super) fn positive(value: Option<i64>, field: &str) -> Result<usize> {
    match value.and_then(|value| usize::try_from(value).ok()) {
        Some(value) if value >= 1 => Ok(value),
        _ => {
            let field = format!("pipeline.{field}");
            Err(validation(&field, "must be a positive integer"))
        }
    }
}
