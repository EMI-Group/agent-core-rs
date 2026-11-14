//! The measured parts of one window.
//!
//! A category is measured one or more times and merged into a single
//! [`PartMeasurement`]; attachments and declared fragments are split by carrier
//! text length, and every merge reports `mixed` when its inputs disagree.
//! These helpers are internal to the analyzer: they name no public seam.
use std::collections::BTreeSet;

use openjiuwen_foundation::prompt::BaseMessage;
use serde_json::Value;

use crate::tokenizers::TokenMeasurement;

use super::models::{
    source_name, ContextCategory, ContextPartUsage, CounterIdentity, MIXED, NOT_REPORTED,
};

/// The measured view of one category, before it becomes a part.
#[derive(Debug, Clone)]
pub(crate) struct PartMeasurement {
    /// Tokens attributed to the category.
    pub(crate) tokens: u64,
    /// How the tokens were produced.
    pub(crate) source: String,
    /// Whether the count is an estimate.
    pub(crate) estimated: bool,
    /// The tokenizer name, when one can be named.
    pub(crate) tokenizer: Option<String>,
    /// Why a fallback strategy was used, when one was.
    pub(crate) fallback_reason: Option<String>,
    /// The model whose vocabulary had to stand in, when one did.
    pub(crate) fallback_tokenizer_model: Option<String>,
}

impl PartMeasurement {
    /// The zero measurement of a category that was not measured.
    pub(crate) fn not_reported() -> Self {
        Self {
            tokens: 0,
            source: NOT_REPORTED.to_owned(),
            estimated: true,
            tokenizer: None,
            fallback_reason: None,
            fallback_tokenizer_model: None,
        }
    }

    /// Converts one tokenizer measurement into a part measurement.
    pub(crate) fn from_measurement(
        measurement: &TokenMeasurement,
        identity: &CounterIdentity,
    ) -> Self {
        Self {
            tokens: measurement.tokens as u64,
            source: source_name(measurement.source).to_owned(),
            estimated: measurement.estimated,
            tokenizer: measurement.tokenizer.clone(),
            fallback_reason: measurement.fallback_reason.clone(),
            fallback_tokenizer_model: identity.fallback_tokenizer_model.clone(),
        }
    }
}

/// Merges the measurements of one category.
pub(crate) fn merge_measurements(measurements: &[PartMeasurement]) -> PartMeasurement {
    let Some(first) = measurements.first() else {
        return PartMeasurement::not_reported();
    };
    let tokens = measurements
        .iter()
        .map(|measurement| measurement.tokens)
        .fold(0u64, u64::saturating_add);
    let sources: BTreeSet<&str> = measurements
        .iter()
        .map(|measurement| measurement.source.as_str())
        .collect();
    let tokenizers: BTreeSet<&str> = measurements
        .iter()
        .filter_map(|measurement| measurement.tokenizer.as_deref())
        .collect();
    let reasons: BTreeSet<&str> = measurements
        .iter()
        .filter_map(|measurement| measurement.fallback_reason.as_deref())
        .collect();
    let fallback_models: BTreeSet<&str> = measurements
        .iter()
        .filter_map(|measurement| measurement.fallback_tokenizer_model.as_deref())
        .collect();
    PartMeasurement {
        tokens,
        source: merge_names(sources).unwrap_or_else(|| first.source.clone()),
        estimated: measurements.iter().any(|measurement| measurement.estimated),
        tokenizer: merge_names(tokenizers),
        fallback_reason: merge_names(reasons),
        fallback_tokenizer_model: merge_names(fallback_models),
    }
}

/// Merges a set of names into one value, `mixed`, or nothing.
pub(crate) fn merge_names(names: BTreeSet<&str>) -> Option<String> {
    match names.len() {
        0 => None,
        1 => names.iter().next().map(|name| (*name).to_owned()),
        _ => Some(MIXED.to_owned()),
    }
}

/// Merges the carriers of one category.
pub(crate) fn merge_carriers(carriers: &[String]) -> String {
    let names: BTreeSet<&str> = carriers
        .iter()
        .map(String::as_str)
        .filter(|carrier| !carrier.is_empty())
        .collect();
    merge_names(names).unwrap_or_else(|| "system_message".to_owned())
}

/// Builds a part from a merged measurement and its carrier.
pub(crate) fn part_from_merge(measurement: &PartMeasurement, carrier: &str) -> ContextPartUsage {
    ContextPartUsage {
        tokens: measurement.tokens,
        estimated: measurement.estimated,
        tokenizer: measurement.tokenizer.clone(),
        carrier: Some(carrier.to_owned()),
        source: measurement.source.clone(),
        measurement_source: measurement.source.clone(),
        fallback_reason: measurement.fallback_reason.clone(),
        fallback_tokenizer_model: measurement.fallback_tokenizer_model.clone(),
        ..ContextPartUsage::default()
    }
}

/// Builds a part from a single tokenizer measurement and its carrier.
pub(crate) fn part_from_measurement(
    measurement: &TokenMeasurement,
    identity: &CounterIdentity,
    carrier: &str,
) -> ContextPartUsage {
    part_from_merge(
        &PartMeasurement::from_measurement(measurement, identity),
        carrier,
    )
}

/// Splits one attachment message's measurement across its logical fragments.
///
/// The model still receives one rendered message, so the carrier total is
/// distributed by fragment text length and the rounding remainder goes to the
/// final fragment. Only `system_prompt` and `skills` are accepted; anything else
/// folds into `system_prompt`.
pub(crate) fn split_fragment_measurement(
    measurement: &TokenMeasurement,
    fragments: &[Value],
    identity: &CounterIdentity,
) -> Vec<(ContextCategory, PartMeasurement, String)> {
    let valid: Vec<&Value> = fragments
        .iter()
        .filter(|fragment| fragment.is_object())
        .collect();
    if valid.is_empty() {
        return vec![(
            ContextCategory::SystemPrompt,
            PartMeasurement::from_measurement(measurement, identity),
            "context_message".to_owned(),
        )];
    }
    let weights: Vec<u64> = valid
        .iter()
        .map(|fragment| {
            let text = fragment.get("text").map(value_text).unwrap_or_default();
            u64::try_from(text.chars().count())
                .unwrap_or(u64::MAX)
                .max(1)
        })
        .collect();
    let total_weight = weights.iter().copied().fold(0u64, u64::saturating_add);
    let total_tokens = measurement.tokens as u64;
    let mut remaining = total_tokens;
    let mut result = Vec::with_capacity(valid.len());
    for (index, (fragment, weight)) in valid.iter().zip(weights.iter()).enumerate() {
        let tokens = if index + 1 == valid.len() {
            remaining
        } else {
            proportion(total_tokens, *weight, total_weight).min(remaining)
        };
        remaining = remaining.saturating_sub(tokens);
        let category = fragment
            .get("category")
            .and_then(Value::as_str)
            .and_then(ContextCategory::from_name)
            .filter(|category| category.accepts_attachment())
            .unwrap_or(ContextCategory::SystemPrompt);
        let carrier = fragment
            .get("carrier")
            .and_then(Value::as_str)
            .filter(|carrier| !carrier.is_empty())
            .unwrap_or("context_message")
            .to_owned();
        let part = PartMeasurement {
            tokens,
            ..PartMeasurement::from_measurement(measurement, identity)
        };
        result.push((category, part, carrier));
    }
    result
}

/// `total * weight / denominator`, computed without float accumulation.
pub(crate) fn proportion(total: u64, weight: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    let scaled = u128::from(total) * u128::from(weight) / u128::from(denominator);
    u64::try_from(scaled).unwrap_or(u64::MAX)
}

/// Reads a metadata category value, normalizing anything unknown.
pub(crate) fn attachment_category(value: &Value) -> ContextCategory {
    value
        .as_str()
        .and_then(ContextCategory::from_name)
        .unwrap_or(ContextCategory::SystemPrompt)
}

/// Renders a JSON value the way Python's `str()` renders a fragment text.
pub(crate) fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Serializes one message for the measurement-cache key.
pub(crate) fn message_value(message: &BaseMessage) -> Value {
    serde_json::to_value(message).unwrap_or_else(|error| {
        tracing::debug!(
            target: "openjiuwen_context::usage",
            %error,
            "message could not be serialized for the measurement cache key"
        );
        Value::Null
    })
}

/// Serializes a message list for the measurement-cache key.
pub(crate) fn messages_value(messages: &[BaseMessage]) -> Value {
    Value::Array(messages.iter().map(message_value).collect())
}
