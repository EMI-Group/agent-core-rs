//! Status values owned by the embedded `PersonalContext` package.
//!
//! Port of `openjiuwen/harness/personal_context/status_codes.py`: seven
//! `PersonalContext`-owned status values numbered `154000`–`154006`, each with
//! the Python status name, an `error_msg`-templated message, and a mapping onto
//! the kernel [`openjiuwen_common::error::Error`] variants.
//!
//! The Python module merges these seven values into a namespace that also holds
//! every framework-wide status code; the Rust port only carries the
//! `PersonalContext`-owned ones, because the framework statuses are already
//! modelled by [`openjiuwen_common::error::ErrorKind`].

use openjiuwen_common::error::Error;
use serde::{Deserialize, Serialize};

/// The seven status values owned by the `PersonalContext` package.
///
/// The serde representation is the `snake_case` variant name, so a persisted
/// status round-trips as `"config_invalid"`, `"runtime_timeout"`, and so on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonalContextStatus {
    /// `CONTEXT_PROACTIVE_CONFIG_INVALID` (154000): the proactive config is invalid.
    ConfigInvalid,
    /// `CONTEXT_PROACTIVE_STATE_INVALID` (154001): the proactive state is invalid.
    StateInvalid,
    /// `CONTEXT_PROACTIVE_FILE_EXECUTION_ERROR` (154002): a `proactive_file` stage failed.
    FileExecutionError,
    /// `CONTEXT_PROACTIVE_FETCH_EXECUTION_ERROR` (154003): a `proactive_fetch` stage failed.
    FetchExecutionError,
    /// `CONTEXT_PROACTIVE_PIPELINE_EXECUTION_ERROR` (154004): a `proactive_pipeline` stage failed.
    PipelineExecutionError,
    /// `CONTEXT_PROACTIVE_PUBLISH_EXECUTION_ERROR` (154005): a `proactive_publish` stage failed.
    PublishExecutionError,
    /// `CONTEXT_PROACTIVE_RUNTIME_TIMEOUT` (154006): the runtime exceeded its deadline.
    RuntimeTimeout,
}

impl PersonalContextStatus {
    /// Returns the numeric status code (`154000`–`154006`).
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::ConfigInvalid => 154_000,
            Self::StateInvalid => 154_001,
            Self::FileExecutionError => 154_002,
            Self::FetchExecutionError => 154_003,
            Self::PipelineExecutionError => 154_004,
            Self::PublishExecutionError => 154_005,
            Self::RuntimeTimeout => 154_006,
        }
    }

    /// Returns the Python status name, e.g. `CONTEXT_PROACTIVE_CONFIG_INVALID`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ConfigInvalid => "CONTEXT_PROACTIVE_CONFIG_INVALID",
            Self::StateInvalid => "CONTEXT_PROACTIVE_STATE_INVALID",
            Self::FileExecutionError => "CONTEXT_PROACTIVE_FILE_EXECUTION_ERROR",
            Self::FetchExecutionError => "CONTEXT_PROACTIVE_FETCH_EXECUTION_ERROR",
            Self::PipelineExecutionError => "CONTEXT_PROACTIVE_PIPELINE_EXECUTION_ERROR",
            Self::PublishExecutionError => "CONTEXT_PROACTIVE_PUBLISH_EXECUTION_ERROR",
            Self::RuntimeTimeout => "CONTEXT_PROACTIVE_RUNTIME_TIMEOUT",
        }
    }

    /// Returns the message template, with an `{error_msg}` placeholder for the reason.
    #[must_use]
    pub const fn message_template(self) -> &'static str {
        match self {
            Self::ConfigInvalid => "context proactive_config is invalid, reason: {error_msg}",
            Self::StateInvalid => "context proactive_state is invalid, reason: {error_msg}",
            Self::FileExecutionError => {
                "context proactive_file execution error, reason: {error_msg}"
            }
            Self::FetchExecutionError => {
                "context proactive_fetch execution error, reason: {error_msg}"
            }
            Self::PipelineExecutionError => {
                "context proactive_pipeline execution error, reason: {error_msg}"
            }
            Self::PublishExecutionError => {
                "context proactive_publish execution error, reason: {error_msg}"
            }
            Self::RuntimeTimeout => "context proactive_runtime timeout, reason: {error_msg}",
        }
    }

    /// Renders the status message, substituting `reason` for `{error_msg}`.
    #[must_use]
    pub fn message(self, reason: impl AsRef<str>) -> String {
        self.message_template()
            .replace("{error_msg}", reason.as_ref())
    }
}

/// Builds the kernel error for `status`, with `status.message(reason)` as its message.
///
/// The status family decides the [`openjiuwen_common::error::Error`] kind:
/// [`PersonalContextStatus::ConfigInvalid`] maps to [`Error::config`] and
/// [`PersonalContextStatus::StateInvalid`] to [`Error::validation`].
/// The four execution statuses — [`PersonalContextStatus::FileExecutionError`],
/// [`PersonalContextStatus::FetchExecutionError`],
/// [`PersonalContextStatus::PipelineExecutionError`] and
/// [`PersonalContextStatus::PublishExecutionError`] — map to [`Error::tool`].
/// [`PersonalContextStatus::RuntimeTimeout`] maps to [`Error::timeout`].
///
/// This function cannot fail: every status maps onto exactly one error kind.
pub fn build_error(status: PersonalContextStatus, reason: impl AsRef<str>) -> Error {
    let message = status.message(reason);
    match status {
        PersonalContextStatus::ConfigInvalid => Error::config(message),
        PersonalContextStatus::StateInvalid => Error::validation(message),
        PersonalContextStatus::FileExecutionError
        | PersonalContextStatus::FetchExecutionError
        | PersonalContextStatus::PipelineExecutionError
        | PersonalContextStatus::PublishExecutionError => Error::tool(message),
        PersonalContextStatus::RuntimeTimeout => Error::timeout(message),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// Every status with its exact code, Python name and message template.
    const CASES: [(PersonalContextStatus, u32, &str, &str); 7] = [
        (
            PersonalContextStatus::ConfigInvalid,
            154_000,
            "CONTEXT_PROACTIVE_CONFIG_INVALID",
            "context proactive_config is invalid, reason: {error_msg}",
        ),
        (
            PersonalContextStatus::StateInvalid,
            154_001,
            "CONTEXT_PROACTIVE_STATE_INVALID",
            "context proactive_state is invalid, reason: {error_msg}",
        ),
        (
            PersonalContextStatus::FileExecutionError,
            154_002,
            "CONTEXT_PROACTIVE_FILE_EXECUTION_ERROR",
            "context proactive_file execution error, reason: {error_msg}",
        ),
        (
            PersonalContextStatus::FetchExecutionError,
            154_003,
            "CONTEXT_PROACTIVE_FETCH_EXECUTION_ERROR",
            "context proactive_fetch execution error, reason: {error_msg}",
        ),
        (
            PersonalContextStatus::PipelineExecutionError,
            154_004,
            "CONTEXT_PROACTIVE_PIPELINE_EXECUTION_ERROR",
            "context proactive_pipeline execution error, reason: {error_msg}",
        ),
        (
            PersonalContextStatus::PublishExecutionError,
            154_005,
            "CONTEXT_PROACTIVE_PUBLISH_EXECUTION_ERROR",
            "context proactive_publish execution error, reason: {error_msg}",
        ),
        (
            PersonalContextStatus::RuntimeTimeout,
            154_006,
            "CONTEXT_PROACTIVE_RUNTIME_TIMEOUT",
            "context proactive_runtime timeout, reason: {error_msg}",
        ),
    ];

    #[test]
    fn codes_names_and_templates_are_exact() {
        for (status, code, name, template) in CASES {
            assert_eq!(status.code(), code, "code of {name}");
            assert_eq!(status.name(), name);
            assert_eq!(status.message_template(), template, "template of {name}");
        }
    }

    #[test]
    fn message_substitutes_the_reason_placeholder() {
        for (status, _, _, template) in CASES {
            let expected = template.replace("{error_msg}", "something broke");
            assert_eq!(status.message("something broke"), expected);
        }
        assert_eq!(
            PersonalContextStatus::RuntimeTimeout.message("late"),
            "context proactive_runtime timeout, reason: late"
        );
    }

    #[test]
    fn build_error_maps_each_status_family_to_an_error_kind() {
        let cases = [
            (PersonalContextStatus::ConfigInvalid, ErrorKind::Config),
            (PersonalContextStatus::StateInvalid, ErrorKind::Validation),
            (PersonalContextStatus::FileExecutionError, ErrorKind::Tool),
            (PersonalContextStatus::FetchExecutionError, ErrorKind::Tool),
            (
                PersonalContextStatus::PipelineExecutionError,
                ErrorKind::Tool,
            ),
            (
                PersonalContextStatus::PublishExecutionError,
                ErrorKind::Tool,
            ),
            (PersonalContextStatus::RuntimeTimeout, ErrorKind::Timeout),
        ];
        for (status, expected_kind) in cases {
            let error = build_error(status, "why");
            assert_eq!(error.kind(), expected_kind, "kind of {}", status.name());
            assert_eq!(
                error.to_string(),
                format!("{}: {}", expected_kind, status.message("why")),
                "message of {}",
                status.name()
            );
        }
    }

    #[test]
    fn build_error_keeps_the_message_verbatim() {
        let error = build_error(PersonalContextStatus::ConfigInvalid, "missing prompt");
        assert_eq!(
            error.to_string(),
            "config: context proactive_config is invalid, reason: missing prompt"
        );
        assert!(matches!(error, Error::Config(_)));
    }

    #[test]
    fn status_serde_uses_snake_case() {
        let encoded = serde_json::to_string(&PersonalContextStatus::PublishExecutionError).unwrap();
        assert_eq!(encoded, "\"publish_execution_error\"");
        let decoded: PersonalContextStatus = serde_json::from_str("\"runtime_timeout\"").unwrap();
        assert_eq!(decoded, PersonalContextStatus::RuntimeTimeout);
    }
}
