//! Provider `usage_metadata` normalization.
//!
//! Ports `usage/provider_usage.py`: [`request_usage_from_metadata`] turns a
//! provider's usage record into the canonical [`RequestKVCacheUsage`] contract.
//!
//! Missing values stay `None`. An inconsistent provider report is marked with
//! [`RequestKVCacheUsage::invalid_reason`] and is deliberately never clamped
//! into a plausible miss: the session aggregator excludes such a call from
//! hit/miss totals while keeping its quality state.

use serde_json::Value;

use super::context_tokens::json_truthy;
use super::models::{
    RequestKVCacheUsage, NOT_REPORTED, SOURCE_PROVIDER_USAGE, STATUS_FULL_HIT, STATUS_INVALID,
    STATUS_MISS, STATUS_NOT_REPORTED, STATUS_PARTIAL_HIT,
};

/// The metadata key of the legacy cache-read counter.
pub const LEGACY_CACHE_TOKENS_KEY: &str = "cache_tokens";

/// The metadata key of the provider's cache status verdict.
pub const CACHE_STATUS_KEY: &str = "cache_status";

/// The metadata key of the provider's cache source label.
pub const CACHE_SOURCE_KEY: &str = "cache_source";

/// The metadata key of the provider's cache-authoritative flag.
pub const CACHE_AUTHORITATIVE_KEY: &str = "cache_authoritative";

/// Converts provider metadata to the canonical cache-usage contract.
///
/// Reads `input_tokens`, `cache_read_tokens`, `cache_miss_tokens` and
/// `cache_write_tokens`. `cache_tokens` is the legacy read field and is used
/// only when `cache_read_tokens` is absent and the legacy value is neither
/// `null` nor zero, because a model's zero default historically meant "not
/// reported".
///
/// `cache_mode` and `cache_scope` stay unset here: they are properties of the
/// emitted snapshot and of the session aggregation key, not of one provider
/// record.
#[must_use]
pub fn request_usage_from_metadata(usage: &Value) -> RequestKVCacheUsage {
    let Value::Object(usage) = usage else {
        return RequestKVCacheUsage::default();
    };

    let (input_tokens, input_error) = parse_nonnegative(usage.get("input_tokens"), "input_tokens");
    let raw_read = match usage.get("cache_read_tokens") {
        Some(value) if !value.is_null() => Some(value),
        _ => usage
            .get(LEGACY_CACHE_TOKENS_KEY)
            .filter(|value| is_reported_non_zero(value)),
    };
    let (read_tokens, read_error) = parse_nonnegative(raw_read, "cache_read_tokens");
    let (miss_tokens, miss_error) =
        parse_nonnegative(usage.get("cache_miss_tokens"), "cache_miss_tokens");
    let (write_tokens, write_error) =
        parse_nonnegative(usage.get("cache_write_tokens"), "cache_write_tokens");

    let mut invalid_reason = [input_error, read_error, miss_error, write_error]
        .into_iter()
        .flatten()
        .reduce(|current, reason| format!("{current}; {reason}"));
    if let (Some(input_tokens), Some(read_tokens)) = (input_tokens, read_tokens) {
        if read_tokens > input_tokens {
            invalid_reason = Some(append_reason(
                invalid_reason,
                "cache_read_tokens_exceeds_input_tokens",
            ));
        }
    }

    let mut miss_tokens = miss_tokens;
    let mut miss_tokens_derived = false;
    if invalid_reason.is_none() {
        if let (Some(input_tokens), Some(read_tokens)) = (input_tokens, read_tokens) {
            let expected_miss = input_tokens.saturating_sub(read_tokens);
            match miss_tokens {
                None => {
                    miss_tokens = Some(expected_miss);
                    miss_tokens_derived = true;
                }
                Some(reported) if reported != expected_miss => {
                    invalid_reason = Some(append_reason(
                        invalid_reason,
                        "cache_read_plus_miss_does_not_equal_input",
                    ));
                }
                Some(_) => {}
            }
            if let Some(write_tokens) = write_tokens {
                if write_tokens > expected_miss {
                    invalid_reason = Some(append_reason(
                        invalid_reason,
                        "cache_write_tokens_exceeds_cache_miss_tokens",
                    ));
                }
            }
        }
    }

    let authoritative = usage.get(CACHE_AUTHORITATIVE_KEY).is_some_and(json_truthy);
    let source = usage
        .get(CACHE_SOURCE_KEY)
        .and_then(Value::as_str)
        .filter(|label| !label.is_empty())
        .map_or_else(
            || {
                if read_tokens.is_some() || miss_tokens.is_some() || write_tokens.is_some() {
                    SOURCE_PROVIDER_USAGE.to_owned()
                } else {
                    NOT_REPORTED.to_owned()
                }
            },
            str::to_owned,
        );
    let status = cache_status(
        usage.get(CACHE_STATUS_KEY),
        input_tokens,
        read_tokens,
        miss_tokens,
        invalid_reason.as_deref(),
    );
    let hit_rate = match (read_tokens, input_tokens, invalid_reason.as_deref()) {
        (Some(read), Some(input), None) if input > 0 => Some(read as f64 / input as f64),
        _ => None,
    };

    RequestKVCacheUsage {
        input_tokens,
        cache_read_tokens: read_tokens,
        cache_miss_tokens: miss_tokens,
        cache_write_tokens: write_tokens,
        hit_rate,
        status,
        source,
        authoritative,
        miss_tokens_derived,
        cache_mode: None,
        cache_scope: None,
        invalid_reason,
    }
}

/// Parses one non-negative integer field, reporting a reason instead of
/// clamping.
///
/// Integers, integral finite floats and digit strings (one optional leading
/// `+`) are accepted; booleans, negatives, non-integers and composite values
/// are rejected. Non-ASCII digits are rejected as well, unlike Python's
/// `str.isdigit`.
fn parse_nonnegative(value: Option<&Value>, field_name: &str) -> (Option<u64>, Option<String>) {
    let Some(value) = value else {
        return (None, None);
    };
    match value {
        Value::Null => (None, None),
        Value::Number(number) => {
            if let Some(integer) = number.as_u64() {
                return (Some(integer), None);
            }
            if number.as_i64().is_some() {
                return (None, Some(format!("{field_name}_is_negative")));
            }
            match number.as_f64() {
                Some(float)
                    if float.is_finite() && (float - float.round()).abs() < f64::EPSILON =>
                {
                    if float >= 0.0 {
                        (Some(float as u64), None)
                    } else {
                        (None, Some(format!("{field_name}_is_negative")))
                    }
                }
                _ => (None, Some(format!("{field_name}_is_not_an_integer"))),
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return (None, Some(format!("{field_name}_is_not_an_integer")));
            }
            let digits = trimmed.strip_prefix('+').unwrap_or(trimmed);
            if digits.is_empty() || !digits.chars().all(|character| character.is_ascii_digit()) {
                return (None, Some(format!("{field_name}_is_not_an_integer")));
            }
            match digits.parse::<u64>() {
                Ok(integer) => (Some(integer), None),
                Err(_) => (None, Some(format!("{field_name}_is_not_an_integer"))),
            }
        }
        _ => (None, Some(format!("{field_name}_is_not_an_integer"))),
    }
}

/// Whether a value counts as reported by Python's `not in (None, 0)` test.
fn is_reported_non_zero(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number
            .as_f64()
            .is_none_or(|float| float.abs() >= f64::EPSILON),
        Value::String(_) | Value::Array(_) | Value::Object(_) => true,
    }
}

/// Appends a semantic-validation reason to the current one.
fn append_reason(current: Option<String>, reason: &str) -> String {
    match current {
        Some(current) => format!("{current}; {reason}"),
        None => reason.to_owned(),
    }
}

/// Derives the request status from the provider verdict and the local rules.
///
/// Provider verdicts in `miss`/`partial_hit`/`full_hit` are trusted as-is; the
/// provider's `observed` verdict is intentionally not trusted and recomputed,
/// because a call that observed a cache does not say whether it hit it.
fn cache_status(
    provider_status: Option<&Value>,
    input_tokens: Option<u64>,
    read_tokens: Option<u64>,
    miss_tokens: Option<u64>,
    invalid_reason: Option<&str>,
) -> String {
    if invalid_reason.is_some() {
        return STATUS_INVALID.to_owned();
    }
    let (Some(input_tokens), Some(read_tokens)) = (input_tokens, read_tokens) else {
        return STATUS_NOT_REPORTED.to_owned();
    };
    if let Some(Value::String(status)) = provider_status {
        if matches!(
            status.as_str(),
            STATUS_MISS | STATUS_PARTIAL_HIT | STATUS_FULL_HIT
        ) {
            return status.clone();
        }
    }
    if read_tokens == 0 {
        return STATUS_MISS.to_owned();
    }
    let full_hit = match miss_tokens {
        Some(miss) => miss == 0,
        None => read_tokens == input_tokens,
    };
    if full_hit {
        STATUS_FULL_HIT.to_owned()
    } else {
        STATUS_PARTIAL_HIT.to_owned()
    }
}
