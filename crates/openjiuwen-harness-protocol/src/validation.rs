//! Validation primitives shared by every model in this crate.
//!
//! Rust constructors cannot validate implicitly, so the invariants Python enforces in
//! `__post_init__` are expressed by the [`Validate`] trait. Decode paths (the wire codec,
//! [`from_value_validated`](crate::from_value_validated),
//! [`from_str_validated`](crate::from_str_validated)) call [`Validate::validate`]; code that
//! builds a model with a struct literal must call `validate()` itself.
//!
//! The helpers below emit the Python wording verbatim (minus Python quoting) because tests and
//! provider adapters match substrings: `"{field_name} must not be empty"`, etc.
//! Models whose Python message is a combined sentence (for example
//! `"content block id and kind must not be empty"`) build [`HarnessError::validation`] directly
//! instead of composing helpers.

use std::collections::BTreeSet;

use crate::errors::{HarnessError, HarnessResult};

/// Checks the invariants the Python model enforces in `__post_init__`.
///
/// Constructors do not validate implicitly; decode paths must call `validate()`.
pub trait Validate {
    /// Returns [`HarnessError::Validation`] when an invariant is violated.
    ///
    /// # Errors
    ///
    /// The exact Python message for the first violated invariant.
    fn validate(&self) -> HarnessResult<()>;
}

/// Requires `value` to be non-empty.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with `"{field_name} must not be empty"` when `value` is
/// empty.
pub fn require_non_empty(field_name: &str, value: &str) -> HarnessResult<()> {
    if value.is_empty() {
        return Err(HarnessError::validation(format!(
            "{field_name} must not be empty"
        )));
    }
    Ok(())
}

/// Requires `value`, when present, to be non-empty.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with `"{field_name} must not be empty"` when `value` is
/// `Some("")`.
pub fn require_optional_non_empty(field_name: &str, value: Option<&str>) -> HarnessResult<()> {
    match value {
        Some(text) => require_non_empty(field_name, text),
        None => Ok(()),
    }
}

/// Requires `value` to be a finite, non-negative Unix timestamp.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with `"{field_name} must be a finite Unix timestamp"`
/// when `value` is `NaN`, infinite, or negative.
pub fn require_finite_timestamp(field_name: &str, value: f64) -> HarnessResult<()> {
    if value.is_finite() && value >= 0.0 {
        return Ok(());
    }
    Err(HarnessError::validation(format!(
        "{field_name} must be a finite Unix timestamp"
    )))
}

/// Requires `value` to be a finite, non-negative number.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with
/// `"{field_name} must be a finite non-negative number"` when `value` is `NaN`, infinite, or
/// negative.
pub fn require_finite_non_negative(field_name: &str, value: f64) -> HarnessResult<()> {
    if value.is_finite() && value >= 0.0 {
        return Ok(());
    }
    Err(HarnessError::validation(format!(
        "{field_name} must be a finite non-negative number"
    )))
}

/// Requires `values` to contain neither empty strings nor duplicates.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with `"{field_name} must not contain empty values"` for
/// the first empty entry, or `"{field_name} must not contain duplicates"` for the first repeated
/// entry (after all entries were checked for emptiness).
pub fn require_unique_non_empty(field_name: &str, values: &[String]) -> HarnessResult<()> {
    if values.iter().any(String::is_empty) {
        return Err(HarnessError::validation(format!(
            "{field_name} must not contain empty values"
        )));
    }
    let mut seen = BTreeSet::new();
    if values.iter().any(|value| !seen.insert(value.as_str())) {
        return Err(HarnessError::validation(format!(
            "{field_name} must not contain duplicates"
        )));
    }
    Ok(())
}

/// Requires `value` to be one of `allowed`.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with
/// `"{field_name} must be one of: {allowed sorted, joined by ", "}"` when `value` is not listed.
pub fn require_subset(field_name: &str, value: &str, allowed: &[&str]) -> HarnessResult<()> {
    if allowed.contains(&value) {
        return Ok(());
    }
    Err(HarnessError::validation(format!(
        "{field_name} must be one of: {}",
        join_sorted(allowed)
    )))
}

/// Sorts `values` and joins them with `", "`.
///
/// Reproduces Python's `", ".join(sorted(...))` output used in validation messages.
#[must_use]
pub fn join_sorted<I, S>(values: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut items: Vec<S> = values.into_iter().collect();
    items.sort_unstable_by(|left, right| left.as_ref().cmp(right.as_ref()));
    items
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<&str>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::{
        join_sorted, require_finite_non_negative, require_finite_timestamp, require_non_empty,
        require_optional_non_empty, require_subset, require_unique_non_empty,
    };
    use crate::HarnessErrorKind;

    fn kinds(result: &Result<(), crate::HarnessError>) -> Option<HarnessErrorKind> {
        result.as_ref().err().map(crate::HarnessError::kind)
    }

    #[test]
    fn join_sorted_orders_lexicographically() {
        assert_eq!(
            join_sorted(["gamma", "alpha", "beta"]),
            "alpha, beta, gamma"
        );
        assert_eq!(join_sorted(Vec::<String>::new()), "");
        assert_eq!(join_sorted(["only"]), "only");
    }

    #[test]
    fn non_empty_and_optional_non_empty() {
        assert!(require_non_empty("name", "x").is_ok());
        let error = require_non_empty("name", "").expect_err("empty");
        assert_eq!(
            error.to_string(),
            "invalid harness value: name must not be empty"
        );
        assert!(require_optional_non_empty("name", None).is_ok());
        assert_eq!(
            kinds(&require_optional_non_empty("name", Some(""))),
            Some(HarnessErrorKind::Validation)
        );
    }

    #[test]
    fn timestamps_and_counts_reject_non_finite_and_negative() {
        assert!(require_finite_timestamp("started_at", 0.0).is_ok());
        assert!(require_finite_timestamp("started_at", 12.5).is_ok());
        for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = require_finite_timestamp("started_at", bad).expect_err("bad timestamp");
            assert_eq!(
                error.message(),
                "started_at must be a finite Unix timestamp"
            );
        }
        assert!(require_finite_non_negative("cost", 0.0).is_ok());
        for bad in [-0.5, f64::NAN, f64::INFINITY] {
            let error = require_finite_non_negative("cost", bad).expect_err("bad amount");
            assert_eq!(error.message(), "cost must be a finite non-negative number");
        }
    }

    #[test]
    fn unique_non_empty_reports_empties_before_duplicates() {
        let values = ["a".to_owned(), "a".to_owned(), String::new()];
        let error = require_unique_non_empty("ids", &values).expect_err("invalid");
        assert_eq!(error.message(), "ids must not contain empty values");

        let duplicates = ["a".to_owned(), "b".to_owned(), "a".to_owned()];
        let error = require_unique_non_empty("ids", &duplicates).expect_err("invalid");
        assert_eq!(error.message(), "ids must not contain duplicates");

        assert!(require_unique_non_empty("ids", &[]).is_ok());
        assert!(require_unique_non_empty("ids", &["b".to_owned(), "a".to_owned()]).is_ok());
    }

    #[test]
    fn subset_lists_allowed_values_sorted() {
        assert!(require_subset("effort", "high", &["low", "high"]).is_ok());
        let error =
            require_subset("effort", "max", &["medium", "high", "low"]).expect_err("not allowed");
        assert_eq!(error.message(), "effort must be one of: high, low, medium");
    }
}
