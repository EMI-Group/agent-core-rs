//! Error constructors and helpers over [`openjiuwen_common::error`].
//!
//! Every failure in this crate is an [`openjiuwen_common::error::Error`], so callers see the
//! same discriminants ([`ErrorKind`](crate::error::ErrorKind)) as the rest of the workspace.
//! The three helpers here cover the crate's recurring shapes: a message naming the offending
//! field ([`validation`](crate::error::validation)), the standard message used by every
//! deferred Python area ([`unsupported`](crate::error::unsupported)), and the discriminant
//! test used by callers that degrade gracefully
//! ([`is_unsupported`](crate::error::is_unsupported)).

use std::fmt;

pub use openjiuwen_common::error::{Error, ErrorKind, Result};

/// Builds an [`Error::Unsupported`] naming the deferred `feature`.
///
/// The message names the feature and points at the crate's deferred scope, which is
/// the single place where every stubbed Python area is justified.
///
/// ```
/// use openjiuwen_symphony::error::{is_unsupported, unsupported};
///
/// let error = unsupported("LLM tree retrieval");
/// assert!(is_unsupported(&error));
/// assert!(error.message().contains("LLM tree retrieval"));
/// ```
pub fn unsupported(feature: &str) -> Error {
    Error::unsupported(format!(
        "{feature} is not implemented by openjiuwen-symphony: it is part of the crate's \
         deferred scope (LLM/ML stack or external services required)"
    ))
}

/// Builds an [`Error::Validation`] rendered as `<field>: <reason>`.
///
/// Naming the offending field keeps validation failures actionable, both in logs and
/// in caller-facing messages.
pub fn validation(field: &str, reason: impl fmt::Display) -> Error {
    Error::validation(format!("{field}: {reason}"))
}

/// Reports whether `err` is an [`Error::Unsupported`].
///
/// Callers use this to degrade gracefully when an optional Python-parity feature is
/// deferred rather than failing the whole operation.
#[must_use]
pub fn is_unsupported(err: &Error) -> bool {
    err.kind() == ErrorKind::Unsupported
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_names_the_feature_and_is_recognized() {
        let error = unsupported("flow distillation");
        assert!(is_unsupported(&error));
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().contains("flow distillation"));
        assert!(error.message().contains("deferred"));
    }

    #[test]
    fn validation_names_the_field() {
        let error = validation("capability_id", "must not be blank");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "capability_id: must not be blank");
    }

    #[test]
    fn validation_renders_any_display_reason() {
        let error = validation("capability_type", 42);
        assert_eq!(error.message(), "capability_type: 42");
    }

    #[test]
    fn other_kinds_are_not_unsupported() {
        assert!(!is_unsupported(&Error::validation("x")));
        assert!(!is_unsupported(&Error::not_found("x")));
    }
}
