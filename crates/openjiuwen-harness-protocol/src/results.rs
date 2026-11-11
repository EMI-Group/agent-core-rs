//! Turn outcome vocabulary for external harness turns.
//!
//! Mirrors the Python `openjiuwen.harness_protocol.results` module (v0.1.20): the terminal
//! [`TurnResult`] a harness emits for one external-input-driven turn, plus its normalized parts
//! (messages, content blocks, termination, usage, cost, and error).
//!
//! Wire shape: every type serializes with `snake_case` keys, all keys are always present
//! (`null` for [`None`], `[]` for empty sequences, `{}` for empty objects), enums serialize to
//! their Python `.value` strings, and unknown JSON keys are tolerated on decode. Nothing here is
//! `deny_unknown_fields`, and nothing uses `skip_serializing_if` — the JSON shape is stable for
//! provider adapters.
//!
//! Python's `freeze_json_value`/`freeze_json_object` have no Rust analogue: [`JsonValue`] is
//! already owned and JSON-compatible, so deep immutability is replaced by value semantics.
//!
//! Constructors do not validate implicitly; decode paths validate via [`Validate`].

use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::json::{JsonObject, JsonValue};
use crate::validation::{require_finite_timestamp, require_non_empty, Validate};

/// Terminal status of one external-input-driven turn.
///
/// Wire values are `completed`, `interrupted`, and `failed`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    /// The turn reached a normal completion.
    Completed,
    /// The turn was interrupted and carries a [`TurnTermination`].
    Interrupted,
    /// The turn failed and carries a [`TurnError`].
    Failed,
}

/// Portable role of a message retained in a terminal turn result.
///
/// Wire values are `user`, `assistant`, `system`, and `tool`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// A user (external input) message.
    User,
    /// An assistant (model output) message.
    Assistant,
    /// A system instruction message.
    System,
    /// A tool result message.
    Tool,
}

/// Reason a turn ended without reaching a normal completion.
///
/// Wire values are `user_abort`, `timeout`, `policy`, `provider`, and `harness_stop`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TurnTerminationKind {
    /// The user aborted the turn.
    UserAbort,
    /// The turn exceeded its deadline.
    Timeout,
    /// A policy (guardrail, permission, budget) stopped the turn.
    Policy,
    /// The harness provider stopped the turn.
    Provider,
    /// The harness reached its own stop condition.
    HarnessStop,
}

/// One ordered, provider-neutral message content block.
///
/// `kind` names the provider-neutral block type (for example `text`, `reasoning`, `tool_use`);
/// `content` holds the provider JSON for that block, and `data` carries provider extensions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ContentBlock {
    /// Stable identifier of this block within its message.
    pub block_id: String,
    /// Provider-neutral block type discriminator.
    pub kind: String,
    /// Provider JSON payload of the block.
    pub content: JsonValue,
    /// Provider-specific extension data; defaults to an empty object.
    #[serde(default)]
    pub data: JsonObject,
}

/// One normalized message retained by the terminal turn result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct TurnMessage {
    /// Stable identifier of the message.
    pub message_id: String,
    /// Portable role of the message author.
    pub role: MessageRole,
    /// Ordered content blocks of the message.
    pub content: Vec<ContentBlock>,
    /// Provider-specific extension data; defaults to an empty object.
    #[serde(default)]
    pub data: JsonObject,
}

/// Structured cause for an interrupted turn.
///
/// Python declares no `__post_init__` rules for this type, so [`Validate`] is `Ok(())`; it exists
/// only because [`TurnResult`] composes the nested value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct TurnTermination {
    /// Why the turn was interrupted.
    pub kind: TurnTerminationKind,
    /// Human-readable explanation, when the provider supplied one.
    #[serde(default)]
    pub message: Option<String>,
    /// Provider-specific machine-readable code, when available.
    #[serde(default)]
    pub code: Option<String>,
    /// Provider-specific extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

/// Exact monetary amount represented in one-millionth currency units.
///
/// `micros` is unsigned, which subsumes Python's `monetary micros must be non-negative` check: a
/// negative wire value is rejected by serde.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct MonetaryAmount {
    /// Amount in micro-units of `currency` (`1_000_000` micros = one currency unit).
    pub micros: u64,
    /// Three-letter ISO-4217-style currency code, uppercase; defaults to `"USD"`.
    #[serde(default = "default_currency")]
    pub currency: String,
}

/// Returns the default [`MonetaryAmount::currency`] (`"USD"`), matching Python's default.
fn default_currency() -> String {
    "USD".to_owned()
}

impl MonetaryAmount {
    /// Builds an amount, normalizing `currency` to uppercase.
    ///
    /// Python's `__post_init__` uppercases the currency in place, so `new(1_500_000, "eur")` and
    /// `new(1_500_000, "EUR")` are equal. Validation only rejects codes that are not three ASCII
    /// letters; call [`Validate::validate`] to enforce that.
    #[must_use]
    pub fn new(micros: u64, currency: impl Into<String>) -> Self {
        Self {
            micros,
            currency: currency.into().to_uppercase(),
        }
    }
}

/// Normalized token usage with room for provider-specific counters.
///
/// Every field is optional because providers report different subsets. All counters are unsigned,
/// which subsumes Python's `turn usage counters must be non-negative` check.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, rename_all = "snake_case")]
pub struct TurnUsage {
    /// Tokens billed for the model input.
    pub input_tokens: Option<u64>,
    /// Tokens generated by the model.
    pub output_tokens: Option<u64>,
    /// Input tokens served from a provider prompt cache.
    pub cached_input_tokens: Option<u64>,
    /// Output tokens spent on reasoning, when reported separately.
    pub reasoning_output_tokens: Option<u64>,
    /// Provider-reported total, which need not equal the sum of the other counters.
    pub total_tokens: Option<u64>,
    /// Provider-specific extension data; defaults to an empty object.
    pub provider_data: JsonObject,
}

/// Normalized failure information for a turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct TurnError {
    /// Human-readable failure message; must not be empty.
    pub message: String,
    /// Provider-specific machine-readable code, when available.
    #[serde(default)]
    pub code: Option<String>,
    /// Coarse failure category (for example `rate_limit` or `context_overflow`).
    #[serde(default)]
    pub category: Option<String>,
    /// Whether a retry may succeed, when the provider knows.
    #[serde(default)]
    pub retryable: Option<bool>,
    /// Provider-specific extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

/// Complete provider-neutral terminal result for one turn.
///
/// `messages` is the lossless normalized output; `final_output` and `structured_output` are
/// convenience projections for simple consumers (both default to JSON `null`).
///
/// [`TurnResult::duration_ms`] is unsigned, which subsumes Python's
/// `"turn duration_ms must be non-negative monotonic elapsed time"` check: a negative wire value
/// is rejected by serde.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct TurnResult {
    /// How the turn ended.
    pub status: TurnStatus,
    /// Lossless normalized messages produced by the turn.
    #[serde(default)]
    pub messages: Vec<TurnMessage>,
    /// Convenience projection of the last assistant output; defaults to JSON `null`.
    #[serde(default)]
    pub final_output: JsonValue,
    /// Convenience projection of a structured (typed) output; defaults to JSON `null`.
    #[serde(default)]
    pub structured_output: JsonValue,
    /// Provider stop reason, when reported (for example `stop` or `max_tokens`).
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// Present exactly when the status is [`TurnStatus::Interrupted`].
    #[serde(default)]
    pub termination: Option<TurnTermination>,
    /// Present exactly when the status is [`TurnStatus::Failed`].
    #[serde(default)]
    pub error: Option<TurnError>,
    /// Token usage for the turn, when reported.
    #[serde(default)]
    pub usage: Option<TurnUsage>,
    /// Monetary cost of the turn, when reported.
    #[serde(default)]
    pub cost: Option<MonetaryAmount>,
    /// Turn start time as Unix epoch seconds.
    #[serde(default)]
    pub started_at: Option<f64>,
    /// Turn completion time as Unix epoch seconds.
    #[serde(default)]
    pub completed_at: Option<f64>,
    /// Monotonic elapsed time in milliseconds, when measured.
    #[serde(default)]
    pub duration_ms: Option<u64>,
    /// Provider-specific extension data; defaults to an empty object.
    #[serde(default)]
    pub provider_data: JsonObject,
}

impl Validate for ContentBlock {
    /// Rejects an empty `block_id` or `kind` with the combined Python message.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with `"content block id and kind must not be empty"`.
    fn validate(&self) -> HarnessResult<()> {
        if self.block_id.is_empty() || self.kind.is_empty() {
            return Err(HarnessError::validation(
                "content block id and kind must not be empty",
            ));
        }
        Ok(())
    }
}

impl Validate for TurnMessage {
    /// Requires a non-empty `message_id` and validates every block in order.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with `"turn message_id must not be empty"`, or the first
    /// block's validation error.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("turn message_id", &self.message_id)?;
        for block in &self.content {
            block.validate()?;
        }
        Ok(())
    }
}

impl Validate for TurnTermination {
    /// Always succeeds.
    ///
    /// Python's `TurnTermination.__post_init__` only freezes `provider_data`; it has no validation
    /// rules to port, so this impl exists only so [`TurnResult`] can compose the nested value.
    ///
    /// # Errors
    ///
    /// Never returns an error.
    fn validate(&self) -> HarnessResult<()> {
        Ok(())
    }
}

impl Validate for MonetaryAmount {
    /// Requires the uppercased currency to be three ASCII letters.
    ///
    /// Lowercase input is accepted because Python uppercases the currency before checking it;
    /// [`MonetaryAmount::new`] performs the same normalization for structs built in Rust.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with `"monetary currency must be a three-letter code"`.
    fn validate(&self) -> HarnessResult<()> {
        let currency = self.currency.to_ascii_uppercase();
        if currency.len() != 3 || !currency.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(HarnessError::validation(
                "monetary currency must be a three-letter code",
            ));
        }
        Ok(())
    }
}

impl Validate for TurnUsage {
    /// Always succeeds.
    ///
    /// The unsigned counters subsume Python's `"turn usage counters must be non-negative"` check
    /// (a negative wire value is rejected by serde), so no runtime rule remains. The impl exists
    /// because [`TurnResult`] and the event payloads compose this type.
    ///
    /// # Errors
    ///
    /// Never returns an error.
    fn validate(&self) -> HarnessResult<()> {
        Ok(())
    }
}

impl Validate for TurnError {
    /// Requires a non-empty `message`.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with `"turn error message must not be empty"`.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("turn error message", &self.message)
    }
}

impl Validate for TurnResult {
    /// Checks the cross-field rules, then the timestamps, then the nested values.
    ///
    /// Order of checks (each failure reports the first violated rule):
    ///
    /// 1. `Failed` requires `error`; a non-`Failed` status forbids it.
    /// 2. `Interrupted` requires `termination`; a non-`Interrupted` status forbids it.
    /// 3. `started_at`/`completed_at`, when present, are finite non-negative Unix timestamps.
    /// 4. When both are present, `completed_at` must not precede `started_at`.
    /// 5. `messages` (in order), `error`, `termination`, `usage`, and `cost` are validated.
    ///
    /// `duration_ms` needs no rule: the unsigned field subsumes Python's
    /// `"turn duration_ms must be non-negative monotonic elapsed time"` check.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with the exact Python message for the first violated rule.
    fn validate(&self) -> HarnessResult<()> {
        if self.status == TurnStatus::Failed {
            if self.error.is_none() {
                return Err(HarnessError::validation(
                    "failed turn result requires error",
                ));
            }
        } else if self.error.is_some() {
            return Err(HarnessError::validation(
                "only failed turn result may contain error",
            ));
        }

        if self.status == TurnStatus::Interrupted {
            if self.termination.is_none() {
                return Err(HarnessError::validation(
                    "interrupted turn result requires termination",
                ));
            }
        } else if self.termination.is_some() {
            return Err(HarnessError::validation(
                "only interrupted turn result may contain termination",
            ));
        }

        if let Some(started_at) = self.started_at {
            require_finite_timestamp("turn started_at", started_at)?;
        }
        if let Some(completed_at) = self.completed_at {
            require_finite_timestamp("turn completed_at", completed_at)?;
        }
        if let (Some(started_at), Some(completed_at)) = (self.started_at, self.completed_at) {
            if completed_at < started_at {
                return Err(HarnessError::validation(
                    "turn completed_at must not precede started_at",
                ));
            }
        }

        for message in &self.messages {
            message.validate()?;
        }
        if let Some(error) = &self.error {
            error.validate()?;
        }
        if let Some(termination) = &self.termination {
            termination.validate()?;
        }
        if let Some(usage) = &self.usage {
            usage.validate()?;
        }
        if let Some(cost) = &self.cost {
            cost.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{
        ContentBlock, MessageRole, MonetaryAmount, TurnError, TurnMessage, TurnResult, TurnStatus,
        TurnTermination, TurnTerminationKind, TurnUsage,
    };
    use crate::{from_value_validated, HarnessErrorKind, JsonValue, Validate};

    fn block(block_id: &str, kind: &str) -> ContentBlock {
        ContentBlock {
            block_id: block_id.to_owned(),
            kind: kind.to_owned(),
            content: json!({"text": "hi"}),
            data: crate::empty_json_object(),
        }
    }

    fn message(message_id: &str, content: Vec<ContentBlock>) -> TurnMessage {
        TurnMessage {
            message_id: message_id.to_owned(),
            role: MessageRole::Assistant,
            content,
            data: crate::empty_json_object(),
        }
    }

    fn completed_result() -> TurnResult {
        TurnResult {
            status: TurnStatus::Completed,
            messages: vec![message("m1", vec![block("b1", "text")])],
            final_output: json!("hi"),
            structured_output: JsonValue::Null,
            stop_reason: Some("stop".to_owned()),
            termination: None,
            error: None,
            usage: Some(TurnUsage {
                input_tokens: Some(3),
                output_tokens: Some(2),
                total_tokens: Some(5),
                ..TurnUsage::default()
            }),
            cost: Some(MonetaryAmount::new(1_500_000, "usd")),
            started_at: Some(1_700_000_000.0),
            completed_at: Some(1_700_000_001.5),
            duration_ms: Some(1500),
            provider_data: crate::empty_json_object(),
        }
    }

    fn validation_message(result: crate::HarnessResult<()>) -> String {
        let error = result.expect_err("expected a validation error");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        error.message().to_owned()
    }

    #[test]
    fn content_block_requires_id_and_kind() {
        assert!(block("b1", "text").validate().is_ok());
        assert_eq!(
            validation_message(block("", "text").validate()),
            "content block id and kind must not be empty"
        );
        assert_eq!(
            validation_message(block("b1", "").validate()),
            "content block id and kind must not be empty"
        );
    }

    #[test]
    fn turn_message_requires_id_and_validates_blocks_in_order() {
        assert_eq!(
            validation_message(message("", Vec::new()).validate()),
            "turn message_id must not be empty"
        );
        assert_eq!(
            validation_message(message("m1", vec![block("", "text")]).validate()),
            "content block id and kind must not be empty"
        );
    }

    #[test]
    fn termination_validates_nothing() {
        let termination = TurnTermination {
            kind: TurnTerminationKind::UserAbort,
            message: Some(String::new()),
            code: None,
            provider_data: crate::empty_json_object(),
        };
        assert!(termination.validate().is_ok());
    }

    #[test]
    fn monetary_amount_defaults_normalises_and_checks_three_letters() {
        let defaulted: MonetaryAmount =
            from_value_validated(json!({"micros": 2500})).expect("default currency");
        assert_eq!(defaulted.currency, "USD");

        assert_eq!(MonetaryAmount::new(1, "eur").currency, "EUR");
        assert!(MonetaryAmount::new(1, "usd").validate().is_ok());
        assert!(
            from_value_validated::<MonetaryAmount>(json!({"micros": 1, "currency": "eur"})).is_ok()
        );

        for bad in ["EURO", "EU", "1AB", "eu r"] {
            let amount = MonetaryAmount {
                micros: 1,
                currency: bad.to_owned(),
            };
            assert_eq!(
                validation_message(amount.validate()),
                "monetary currency must be a three-letter code",
                "currency {bad:?}"
            );
        }

        assert_eq!(
            serde_json::to_value(MonetaryAmount::new(7, "jpy")).expect("serializes"),
            json!({"micros": 7, "currency": "JPY"})
        );
    }

    #[test]
    fn turn_usage_is_all_default_and_keeps_every_key() {
        let usage: TurnUsage = from_value_validated(json!({})).expect("all defaults");
        assert_eq!(usage, TurnUsage::default());
        assert!(usage.validate().is_ok());

        let encoded = serde_json::to_value(&usage).expect("serializes");
        let keys: Vec<&str> = encoded
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "reasoning_output_tokens",
                "total_tokens",
                "provider_data"
            ]
        );
        assert_eq!(encoded["input_tokens"], JsonValue::Null);
        assert_eq!(encoded["provider_data"], json!({}));

        let decoded: TurnUsage =
            from_value_validated(json!({"input_tokens": 4, "future_counter": 9}))
                .expect("unknown keys tolerated");
        assert_eq!(decoded.input_tokens, Some(4));
        assert_eq!(
            from_value_validated::<TurnUsage>(json!({"input_tokens": -1}))
                .expect_err("negative counter")
                .kind(),
            HarnessErrorKind::Protocol
        );
    }

    #[test]
    fn turn_error_requires_message() {
        let error = TurnError {
            message: String::new(),
            code: None,
            category: None,
            retryable: None,
            provider_data: crate::empty_json_object(),
        };
        assert_eq!(
            validation_message(error.validate()),
            "turn error message must not be empty"
        );

        let decoded: TurnError =
            from_value_validated(json!({"message": "boom", "retryable": true}))
                .expect("valid error");
        assert_eq!(decoded.code, None);
        assert_eq!(decoded.retryable, Some(true));
    }

    #[test]
    fn turn_result_requires_error_for_failed_and_forbids_it_otherwise() {
        let failed = TurnResult {
            status: TurnStatus::Failed,
            ..completed_result()
        };
        assert_eq!(
            validation_message(failed.validate()),
            "failed turn result requires error"
        );

        let completed_with_error = TurnResult {
            error: Some(TurnError {
                message: "boom".to_owned(),
                code: None,
                category: None,
                retryable: None,
                provider_data: crate::empty_json_object(),
            }),
            ..completed_result()
        };
        assert_eq!(
            validation_message(completed_with_error.validate()),
            "only failed turn result may contain error"
        );

        let failed_with_error = TurnResult {
            status: TurnStatus::Failed,
            error: completed_with_error.error.clone(),
            ..completed_result()
        };
        assert!(failed_with_error.validate().is_ok());
    }

    #[test]
    fn turn_result_requires_termination_for_interrupted_and_forbids_it_otherwise() {
        let interrupted = TurnResult {
            status: TurnStatus::Interrupted,
            ..completed_result()
        };
        assert_eq!(
            validation_message(interrupted.validate()),
            "interrupted turn result requires termination"
        );

        let termination = TurnTermination {
            kind: TurnTerminationKind::Timeout,
            message: None,
            code: Some("deadline".to_owned()),
            provider_data: crate::empty_json_object(),
        };
        let interrupted_with_termination = TurnResult {
            status: TurnStatus::Interrupted,
            termination: Some(termination.clone()),
            ..completed_result()
        };
        assert!(interrupted_with_termination.validate().is_ok());

        let completed_with_termination = TurnResult {
            termination: Some(termination),
            ..completed_result()
        };
        assert_eq!(
            validation_message(completed_with_termination.validate()),
            "only interrupted turn result may contain termination"
        );
    }

    #[test]
    fn turn_result_timestamps_are_finite_and_ordered() {
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let result = TurnResult {
                started_at: Some(bad),
                ..completed_result()
            };
            assert_eq!(
                validation_message(result.validate()),
                "turn started_at must be a finite Unix timestamp"
            );

            let result = TurnResult {
                started_at: None,
                completed_at: Some(bad),
                ..completed_result()
            };
            assert_eq!(
                validation_message(result.validate()),
                "turn completed_at must be a finite Unix timestamp"
            );
        }

        let reversed = TurnResult {
            started_at: Some(10.0),
            completed_at: Some(9.999),
            ..completed_result()
        };
        assert_eq!(
            validation_message(reversed.validate()),
            "turn completed_at must not precede started_at"
        );

        let no_timestamps = TurnResult {
            started_at: None,
            completed_at: None,
            duration_ms: Some(0),
            ..completed_result()
        };
        assert!(no_timestamps.validate().is_ok());
    }

    #[test]
    fn turn_result_validates_nested_values() {
        let bad_message = TurnResult {
            messages: vec![message("", vec![block("b1", "text")])],
            ..completed_result()
        };
        assert_eq!(
            validation_message(bad_message.validate()),
            "turn message_id must not be empty"
        );

        let bad_cost = TurnResult {
            cost: Some(MonetaryAmount {
                micros: 1,
                currency: "EU".to_owned(),
            }),
            ..completed_result()
        };
        assert_eq!(
            validation_message(bad_cost.validate()),
            "monetary currency must be a three-letter code"
        );

        let bad_error = TurnResult {
            status: TurnStatus::Failed,
            error: Some(TurnError {
                message: String::new(),
                code: None,
                category: None,
                retryable: None,
                provider_data: crate::empty_json_object(),
            }),
            ..completed_result()
        };
        assert_eq!(
            validation_message(bad_error.validate()),
            "turn error message must not be empty"
        );
    }

    #[test]
    fn turn_result_round_trips_with_every_key() {
        let result = completed_result();
        let encoded = serde_json::to_value(&result).expect("serializes");
        let keys: Vec<&str> = encoded
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "status",
                "messages",
                "final_output",
                "structured_output",
                "stop_reason",
                "termination",
                "error",
                "usage",
                "cost",
                "started_at",
                "completed_at",
                "duration_ms",
                "provider_data"
            ]
        );
        assert_eq!(encoded["status"], json!("completed"));
        assert_eq!(encoded["messages"][0]["role"], json!("assistant"));
        assert_eq!(
            encoded["cost"],
            json!({"micros": 1_500_000, "currency": "USD"})
        );
        assert_eq!(encoded["error"], JsonValue::Null);
        assert_eq!(encoded["termination"], JsonValue::Null);

        let decoded: TurnResult = from_value_validated(encoded).expect("round trip");
        assert_eq!(decoded, result);
    }

    #[test]
    fn enum_wire_values_match_python() {
        assert_eq!(
            serde_json::to_value(TurnStatus::Interrupted).expect("serializes"),
            json!("interrupted")
        );
        assert_eq!(
            serde_json::to_value(MessageRole::Tool).expect("serializes"),
            json!("tool")
        );
        assert_eq!(
            serde_json::to_value(TurnTerminationKind::HarnessStop).expect("serializes"),
            json!("harness_stop")
        );
    }
}
