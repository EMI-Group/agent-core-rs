//! Failure classification of the Claude Code adapter.
//!
//! Ports Python `openjiuwen.harness_providers.claudecode.failure_classifier` (v0.1.20): pure
//! functions turn the CLI's structured failure fields — `assistant.message.error`,
//! `result.is_error` / `api_error_status` / `errors` / `subtype`, and a process-level CLI failure —
//! into a protocol [`TurnError`] carrying one of the shared external-runtime categories.
//!
//! Every classification reports the vendor's own evidence in `provider_data` (`sdk_error_type` for
//! an identifier or exception type, `http_status` for an API status), a `code` naming what the CLI
//! said, and `retryable = true` exactly for the three transient categories.
//!
//! # Python parity
//!
//! - [`FAILURE_CATEGORIES`] is the vocabulary a classification can produce. Python's tuple omits
//!   `request_rejected` although both of its maps emit it; the Rust list is the real vocabulary, so
//!   a consumer can match every category that can appear.
//! - [`classify_result_message`] takes the decoded `result` frame instead of an SDK message, and
//!   also derives a category from the frame's `subtype` when the API status does not name one
//!   (Python only reads `api_error_status`). The CLI's `result` text is used as a message source
//!   after the `errors` list, because the CLI reports API failures there.
//! - [`classify_cli_failure`] takes an error type name and phase in place of an exception object,
//!   because this adapter drives the CLI as a child process rather than through the vendor SDK.

use openjiuwen_harness_protocol::{JsonObject, JsonValue, TurnError};
use serde_json::json;

/// Shared failure vocabulary of the external-runtime reliability layer.
pub const FAILURE_CATEGORIES: [&str; 9] = [
    "auth_required",
    "quota_exceeded",
    "rate_limited",
    "server_unavailable",
    "network_timeout",
    "process_start_failed",
    "request_rejected",
    "sdk_error",
    "unknown",
];

/// Upper bound (characters) for assistant text folded into a failure message (`_MAX_FAILURE_DETAIL_CHARS`).
pub const MAX_FAILURE_DETAIL_CHARS: usize = 8000;

/// Suffix marking a detail string that hit [`MAX_FAILURE_DETAIL_CHARS`].
const TRUNCATION_SUFFIX: &str = "...[truncated]";

/// CLI error types that mean the process itself never got going (`CLINotFoundError` & co).
const PROCESS_START_TYPES: [&str; 3] = ["CLINotFoundError", "CLIConnectionError", "ProcessError"];

/// Phase in which a CLI/SDK failure was raised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailurePhase {
    /// Bringing the CLI up (spawn, handshake); a process-level failure is `process_start_failed`.
    Startup,
    /// Running a turn; a process-level failure is `sdk_error`, because the cycle had already begun.
    Turn,
}

/// Classifies a Claude `assistant.error` identifier.
///
/// `message` is the owning assistant message when available: its text blocks carry the
/// human-readable cause the identifier omits, and are folded into the error message (bounded by
/// [`MAX_FAILURE_DETAIL_CHARS`]).
#[must_use]
pub fn classify_assistant_error(
    error: &str,
    message: Option<&serde_json::Map<String, JsonValue>>,
) -> TurnError {
    let category = assistant_error_category(error);
    let detail = message.map(assistant_failure_detail).unwrap_or_default();
    let text = if detail.is_empty() {
        error.to_owned()
    } else {
        detail
    };
    classified(
        text,
        category,
        Some(error.to_owned()).filter(|code| !code.is_empty()),
        None,
        None,
    )
}

/// Classifies a Claude `result` frame failure by `api_error_status`, `errors` and `subtype`.
#[must_use]
pub fn classify_result_message(result: &serde_json::Map<String, JsonValue>) -> TurnError {
    let subtype = result
        .get("subtype")
        .and_then(JsonValue::as_str)
        .unwrap_or("error");
    let http_status = result.get("api_error_status").and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|status| u64::try_from(status).ok()))
    });
    // The CLI reuses the `assistant.error` identifiers as result subtypes, so one map serves both.
    let category = http_status
        .and_then(api_status_category)
        .unwrap_or_else(|| assistant_error_category(subtype));
    let message = errors_text(result)
        .or_else(|| {
            result
                .get("result")
                .and_then(JsonValue::as_str)
                .filter(|text| !text.trim().is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| match http_status {
            Some(status) => format!("Claude turn failed: HTTP {status}"),
            None => format!("Claude turn failed ({subtype})"),
        });
    classified(
        message,
        category,
        Some(subtype.to_owned()).filter(|code| !code.is_empty()),
        None,
        http_status,
    )
}

/// Folds an earlier assistant diagnostic into the terminal failure.
///
/// The pending category wins only where the terminal state is generic (no HTTP status, or
/// `sdk_error`), but the pending message is kept either way: it is usually the only text naming the
/// actual cause.
#[must_use]
pub fn merge_pending_error(pending: Option<&TurnError>, terminal: TurnError) -> TurnError {
    let Some(pending) = pending else {
        return terminal;
    };
    let mut category = terminal.category.clone();
    let mut retryable = terminal.retryable;
    let mut provider_data = terminal.provider_data.clone();
    let generic_terminal = !terminal.provider_data.contains_key("http_status")
        || terminal.category.as_deref() == Some("sdk_error");
    if generic_terminal
        && pending
            .category
            .as_deref()
            .is_some_and(|category| category != "sdk_error")
    {
        category.clone_from(&pending.category);
        retryable = pending.retryable;
        if provider_data.is_empty() {
            provider_data = pending.provider_data.clone();
        }
    }
    TurnError {
        message: merged_message(pending, &terminal),
        code: terminal.code.clone().or_else(|| pending.code.clone()),
        category,
        retryable,
        provider_data,
    }
}

/// Classifies a process-level CLI failure by its error type name and the phase it happened in.
///
/// `error_type` is the vendor exception/marker name (`CLINotFoundError`, `CLIConnectionError`,
/// `ProcessError`, `TimeoutError`, …); `message` is its text; `stderr_tail` is the CLI's captured
/// error output, appended when non-empty.
#[must_use]
pub fn classify_cli_failure(
    phase: FailurePhase,
    error_type: &str,
    message: &str,
    stderr_tail: Option<&str>,
) -> TurnError {
    let base = if message.is_empty() {
        error_type
    } else {
        message
    };
    let text = match stderr_tail.filter(|tail| !tail.trim().is_empty()) {
        Some(tail) if base.is_empty() => format!("Error output: {tail}"),
        Some(tail) => format!("{base}\nError output: {tail}"),
        None => base.to_owned(),
    };
    let category = if is_timeout(error_type) {
        "network_timeout"
    } else if phase == FailurePhase::Startup && PROCESS_START_TYPES.contains(&error_type) {
        "process_start_failed"
    } else {
        "sdk_error"
    };
    classified(text, category, None, Some(error_type), None)
}

/// Returns the category a Claude `assistant.error` identifier names.
fn assistant_error_category(error: &str) -> &'static str {
    match error {
        "authentication_failed" => "auth_required",
        "billing_error" => "quota_exceeded",
        "invalid_request" => "request_rejected",
        "rate_limit" => "rate_limited",
        "server_error" => "server_unavailable",
        _ => "sdk_error",
    }
}

/// Returns the category an API HTTP status names, when it names one.
fn api_status_category(status: u64) -> Option<&'static str> {
    match status {
        400 => Some("request_rejected"),
        401 | 403 => Some("auth_required"),
        429 => Some("rate_limited"),
        500 | 529 => Some("server_unavailable"),
        _ => None,
    }
}

/// Returns whether `error_type` names a timeout.
fn is_timeout(error_type: &str) -> bool {
    error_type == "TimeoutError"
        || error_type == "Elapsed"
        || error_type.to_ascii_lowercase().contains("timeout")
}

/// Joins the informative `errors` strings of a result frame, ignoring the CLI's `unknown` filler.
fn errors_text(result: &serde_json::Map<String, JsonValue>) -> Option<String> {
    let errors = result.get("errors").and_then(JsonValue::as_array)?;
    let informative: Vec<&str> = errors
        .iter()
        .filter_map(JsonValue::as_str)
        .filter(|item| !item.trim().is_empty() && !item.trim().eq_ignore_ascii_case("unknown"))
        .collect();
    if informative.is_empty() {
        None
    } else {
        Some(informative.join("\n"))
    }
}

/// Extracts the bounded text diagnostics of a failed assistant message.
fn assistant_failure_detail(message: &serde_json::Map<String, JsonValue>) -> String {
    let Some(JsonValue::Array(content)) = message.get("content") else {
        return String::new();
    };
    let parts: Vec<&str> = content
        .iter()
        .filter_map(JsonValue::as_object)
        .filter(|block| block.get("type").and_then(JsonValue::as_str) == Some("text"))
        .filter_map(|block| block.get("text").and_then(JsonValue::as_str))
        .filter(|text| !text.is_empty())
        .collect();
    let detail = parts.join("\n");
    if detail.chars().count() <= MAX_FAILURE_DETAIL_CHARS {
        return detail;
    }
    let mut truncated: String = detail.chars().take(MAX_FAILURE_DETAIL_CHARS).collect();
    truncated.push_str(TRUNCATION_SUFFIX);
    truncated
}

/// Combines a pending diagnostic with the terminal failure message.
fn merged_message(pending: &TurnError, terminal: &TurnError) -> String {
    let pending_message = pending.message.trim();
    let terminal_message = terminal.message.trim();
    if pending_message.is_empty() || Some(pending_message) == pending.code.as_deref() {
        return if terminal_message.is_empty() {
            pending_message.to_owned()
        } else {
            terminal_message.to_owned()
        };
    }
    if terminal_message.is_empty() {
        return pending_message.to_owned();
    }
    if terminal_message.contains(pending_message) {
        return terminal_message.to_owned();
    }
    format!("{terminal_message}\n{pending_message}")
}

/// Builds one classified [`TurnError`].
fn classified(
    message: String,
    category: &str,
    code: Option<String>,
    sdk_error_type: Option<&str>,
    http_status: Option<u64>,
) -> TurnError {
    let mut provider_data = JsonObject::new();
    if let Some(kind) = sdk_error_type.filter(|kind| !kind.is_empty()) {
        provider_data.insert("sdk_error_type".to_owned(), json!(kind));
    }
    if let Some(status) = http_status {
        provider_data.insert("http_status".to_owned(), json!(status));
    }
    TurnError {
        message: if message.is_empty() {
            category.to_owned()
        } else {
            message
        },
        code,
        category: Some(category.to_owned()),
        retryable: Some(matches!(
            category,
            "rate_limited" | "server_unavailable" | "network_timeout"
        )),
        provider_data,
    }
}
