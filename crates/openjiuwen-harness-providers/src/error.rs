//! Translation from the harness SPI error type to the workspace error type.
//!
//! Every provider SPI call returns [`HarnessResult`](openjiuwen_harness_protocol::HarnessResult);
//! registry, factory and adapter functions return
//! [`openjiuwen_common::error::Result`]. [`providers_error_to_common`] is the single translation
//! point between the two, so the mapping stays identical across the crate.
//!
//! Python has one `ValueError` family at this boundary; Rust keeps the categories distinct
//! (documented deviation): a missing capability is [`Error::Unsupported`], an invalid provider
//! configuration is [`Error::Config`], an invalid field value is [`Error::Validation`], a
//! checkpoint conflict is [`Error::Session`], and an unrecognized future kind is wrapped as
//! [`Error::External`] so its message survives.

use openjiuwen_common::error::Error;
use openjiuwen_harness_protocol::{HarnessError, HarnessErrorKind};

/// Converts a harness SPI error into the workspace error, keeping Python's message payload.
///
/// The [`HarnessErrorKind`] selects the workspace variant:
///
/// | `HarnessErrorKind` | `openjiuwen_common::error::Error` |
/// |---|---|
/// | `UnsupportedCapability` | [`Error::Unsupported`] |
/// | `Validation` | [`Error::Validation`] |
/// | `State`, `Protocol` | [`Error::Config`] |
/// | `CheckpointConflict` | [`Error::Session`] |
/// | any future kind | [`Error::External`] wrapping an [`std::io::Error`] with the message |
///
/// The payload is [`HarnessError::message`], i.e. the Python wording without the `Display`
/// prefix.
// The harness SPI hands errors over by value; the translation borrows the payload only, so the
// constructor parameter stays owned to keep call sites identical to Python's `raise ... from error`.
#[allow(clippy::needless_pass_by_value)]
pub fn providers_error_to_common(error: HarnessError) -> Error {
    match error.kind() {
        HarnessErrorKind::UnsupportedCapability => Error::unsupported(error.message()),
        HarnessErrorKind::Validation => Error::validation(error.message()),
        HarnessErrorKind::State | HarnessErrorKind::Protocol => Error::config(error.message()),
        HarnessErrorKind::CheckpointConflict => Error::session(error.message()),
        // `HarnessErrorKind` is `#[non_exhaustive]`; a new kind must not be lost silently, so it
        // is preserved as an external error.
        _ => Error::external(std::io::Error::other(error.message().to_owned())),
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_common::error::Error;
    use openjiuwen_harness_protocol::HarnessError;

    use super::providers_error_to_common;

    #[test]
    fn unsupported_capability_maps_to_unsupported() {
        let error = providers_error_to_common(HarnessError::unsupported_capability(
            "graceful abort is not supported",
        ));
        assert!(matches!(error, Error::Unsupported(_)));
        assert_eq!(
            error.to_string(),
            "unsupported: graceful abort is not supported"
        );
    }

    #[test]
    fn validation_maps_to_validation() {
        let error = providers_error_to_common(HarnessError::validation("model must be a string"));
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(error.to_string(), "validation: model must be a string");
    }

    #[test]
    fn state_maps_to_config() {
        let error = providers_error_to_common(HarnessError::state("harness is not started"));
        assert!(matches!(error, Error::Config(_)));
        assert_eq!(error.to_string(), "config: harness is not started");
    }

    #[test]
    fn protocol_maps_to_config() {
        let error = providers_error_to_common(HarnessError::protocol("missing send receipt"));
        assert!(matches!(error, Error::Config(_)));
        assert_eq!(error.to_string(), "config: missing send receipt");
    }

    #[test]
    fn checkpoint_conflict_maps_to_session() {
        let error =
            providers_error_to_common(HarnessError::checkpoint_conflict("stale checkpoint"));
        assert!(matches!(error, Error::Session(_)));
        assert_eq!(error.to_string(), "session: stale checkpoint");
    }
}
