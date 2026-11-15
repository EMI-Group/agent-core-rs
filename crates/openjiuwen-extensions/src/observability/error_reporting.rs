//! Central normalization of error causes before they are written to a span.
//!
//! Port of the pure half of
//! `openjiuwen/extensions/observability/error_reporting.py` (v0.1.20).
//!
//! An exception often carries its real explanation outside `str(exc)`: a bare
//! timeout renders as an empty string, while the layer that raised it built a
//! detailed `error_message` (stage, timeouts, chunk counts).
//! Every error span written by the observability layer goes through one
//! priority chain so the recorded cause follows one policy:
//!
//! 1. the caller-supplied `error_message` when non-blank,
//! 2. a non-blank rendering of the exception,
//! 3. the exception type name (e.g. `TimeoutError`),
//! 4. the caller's generic fallback text.
//!
//! # What this module deliberately does not do
//!
//! Python's `record_span_error` is SDK-only: it records the exception event on a
//! live `opentelemetry` span, overrides `exception.message` /
//! `exception.stacktrace` with masked variants, sets `error.type` and the error
//! status, and ends the span.
//! That half lives in [`crate::otel`] — see `crate::otel::OtelAgentHandler`,
//! which owns the span lifecycle — and is not re-implemented here.
//! The masking the Python function applies (`redact_error_summary`,
//! `redact_error_stacktrace`) is likewise owned by the redaction module
//! (`crate::observability::redaction`); this function returns the *raw* reason,
//! exactly like Python's `error_reason`, and the caller masks it.
//!
//! Python's module defines no constant of its own: the `error.type` key it
//! writes is `crate::otel::semconv::ERROR_TYPE` (`error.type`), and
//! [`TOOL_REPORTED_FAILURE`](crate::observability::tool_outcome::TOOL_REPORTED_FAILURE)
//! is the value used for a call that returned a failure instead of raising.
/// Resolves the raw error reason through the shared priority chain.
///
/// The chain is Python's, parameter for parameter:
///
/// - `error_message` — trimmed and returned when non-blank;
/// - `exception_text` — the exception's own rendering (`str(exc)` in Python);
///   trimmed and returned when non-blank. `None` means "no exception";
/// - `exception_type` — Python's `type(exception).__name__` fallback, which it
///   reaches whenever an exception exists but renders blank. Pass the type name
///   here so the chain stays faithful; `None` falls through to `default`;
/// - `default` — the caller's generic fallback text.
///
/// Python's order is `error_message` first even when an exception was also
/// supplied, because a caller-supplied message carries the richer context.
///
/// ```
/// use openjiuwen_extensions::observability::error_reporting::error_reason;
///
/// // The detailed message wins over the exception's rendering.
/// assert_eq!(
///     error_reason(Some("TimeoutError()"), Some("stage embed timed out"), Some("TimeoutError"), "failed"),
///     "stage embed timed out"
/// );
/// // A blank rendering falls back to the type name, like Python.
/// assert_eq!(
///     error_reason(Some("   "), None, Some("TimeoutError"), "failed"),
///     "TimeoutError"
/// );
/// // With nothing to go on, the caller's fallback is used.
/// assert_eq!(error_reason(None, None, None, "failed"), "failed");
/// ```
#[must_use]
pub fn error_reason(
    exception_text: Option<&str>,
    error_message: Option<&str>,
    exception_type: Option<&str>,
    default: &str,
) -> String {
    if let Some(message) = error_message {
        let trimmed = message.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    if let Some(text) = exception_text {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    if let Some(name) = exception_type {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    default.to_owned()
}
#[cfg(test)]
mod tests {
    //! Port of the `error_reason` assertions of
    //! `tests/unit_tests/extensions/observability/test_callback_handler.py`.
    //!
    //! A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::error_reason;
    #[test]
    fn the_error_message_wins_over_the_exception() {
        // Python: `error_message` is consulted before `str(exception)` even when
        // both are present, because the caller-supplied text has more context.
        assert_eq!(
            error_reason(
                Some("boom"),
                Some("  stage llm failed  "),
                Some("RuntimeError"),
                "fallback"
            ),
            "stage llm failed"
        );
    }
    #[test]
    fn a_blank_error_message_falls_through_to_the_exception() {
        for message in [None, Some(""), Some("   \n\t ")] {
            assert_eq!(
                error_reason(Some("boom"), message, Some("RuntimeError"), "fallback"),
                "boom"
            );
        }
    }
    #[test]
    fn a_blank_exception_rendering_falls_back_to_the_type_name() {
        // Python: a bare `asyncio.TimeoutError()` renders as `""`, so the type
        // name is what the span records.
        assert_eq!(
            error_reason(Some(""), None, Some("TimeoutError"), "fallback"),
            "TimeoutError"
        );
        assert_eq!(
            error_reason(Some("  "), Some(" "), Some("TimeoutError"), "fallback"),
            "TimeoutError"
        );
    }
    #[test]
    fn no_exception_no_message_uses_the_default() {
        assert_eq!(error_reason(None, None, None, "fallback"), "fallback");
        // Python never reaches the type name without an exception, and neither
        // does the chain: an empty type name is treated as absent.
        assert_eq!(error_reason(None, None, Some("  "), "fallback"), "fallback");
        assert_eq!(error_reason(Some(""), None, None, "fallback"), "fallback");
    }
    #[test]
    fn the_exception_rendering_is_trimmed() {
        assert_eq!(
            error_reason(
                Some("  KeyError: 'path'\n"),
                None,
                Some("KeyError"),
                "fallback"
            ),
            "KeyError: 'path'"
        );
    }
    #[test]
    fn the_default_is_returned_verbatim() {
        assert_eq!(error_reason(None, None, None, ""), "");
        assert_eq!(
            error_reason(None, None, None, " tool failed "),
            " tool failed "
        );
    }
}
