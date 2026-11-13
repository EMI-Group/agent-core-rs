//! Discovery settings: result caps, output budget, and token accounting.
//!
//! Ports Python `openjiuwen.symphony.discovery`'s configuration surface
//! (`DiscoverySettings`): how many candidate skills a discovery round surfaces, how
//! much text a selection prompt may carry, and which slice of the model context window
//! is reserved for skill candidates.
//!
//! [`DiscoverySettings::from_value`] is deliberately lenient — it reads a JSON object,
//! coerces numbers, numeric strings, and booleans where it can, and falls back to the
//! default for every missing or unusable field, so a host configuration file never
//! fails discovery over one badly typed key.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The default number of candidate skills a discovery round returns.
pub const DEFAULT_MAX_RESULTS: usize = 10;
/// The default cap on the characters of skill text a prompt may carry.
pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 12_000;
/// The default share of the context window reserved for skill candidates.
pub const DEFAULT_CANDIDATE_BUDGET_RATIO: f64 = 0.01;
/// The default model context window, in tokens.
pub const DEFAULT_CONTEXT_WINDOW_TOKENS: usize = 200_000;

/// Configuration for one skill-discovery round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DiscoverySettings {
    /// The maximum number of candidate skills to surface.
    pub max_results: usize,
    /// The maximum characters of skill text to include in one selection prompt.
    pub max_output_chars: usize,
    /// The share of the context window reserved for skill candidates.
    pub candidate_budget_ratio: f64,
    /// The model context window, in tokens.
    pub context_window_tokens: usize,
    /// Whether an existing semantic index may be reused instead of rebuilt.
    pub use_existing_index: bool,
}

impl Default for DiscoverySettings {
    fn default() -> Self {
        Self {
            max_results: DEFAULT_MAX_RESULTS,
            max_output_chars: DEFAULT_MAX_OUTPUT_CHARS,
            candidate_budget_ratio: DEFAULT_CANDIDATE_BUDGET_RATIO,
            context_window_tokens: DEFAULT_CONTEXT_WINDOW_TOKENS,
            use_existing_index: true,
        }
    }
}

impl DiscoverySettings {
    /// The token budget for skill candidates: the context window times the ratio.
    ///
    /// The ratio is clamped to `[0, 1]` and a non-finite or negative ratio yields `0`,
    /// so the budget is always a finite, non-negative token count. The default is
    /// `200_000 * 0.01 = 2_000` tokens.
    ///
    /// ```
    /// use openjiuwen_symphony::discovery::DiscoverySettings;
    ///
    /// assert_eq!(DiscoverySettings::default().candidate_budget_tokens(), 2_000);
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // token counts fit f64 exactly enough for a ratio.
    #[allow(clippy::cast_sign_loss)] // clamped to a non-negative value first.
    #[allow(clippy::cast_possible_truncation)] // floor of a finite, positive product.
    pub fn candidate_budget_tokens(&self) -> usize {
        let ratio = if self.candidate_budget_ratio.is_finite() && self.candidate_budget_ratio > 0.0
        {
            self.candidate_budget_ratio.min(1.0)
        } else {
            0.0
        };
        (self.context_window_tokens as f64 * ratio) as usize
    }

    /// Reads settings from a JSON value, falling back to the default for every field
    /// that is missing or not coercible.
    ///
    /// Numeric fields accept JSON numbers and numeric strings; `use_existing_index`
    /// accepts booleans and the usual `"true"`/`"false"` strings. A non-object value
    /// yields the defaults.
    #[must_use]
    pub fn from_value(value: Value) -> Self {
        let defaults = Self::default();
        let Value::Object(mapping) = value else {
            return defaults;
        };
        Self {
            max_results: mapping
                .get("max_results")
                .and_then(usize_value)
                .unwrap_or(defaults.max_results),
            max_output_chars: mapping
                .get("max_output_chars")
                .and_then(usize_value)
                .unwrap_or(defaults.max_output_chars),
            candidate_budget_ratio: mapping
                .get("candidate_budget_ratio")
                .and_then(f64_value)
                .unwrap_or(defaults.candidate_budget_ratio),
            context_window_tokens: mapping
                .get("context_window_tokens")
                .and_then(usize_value)
                .unwrap_or(defaults.context_window_tokens),
            use_existing_index: mapping
                .get("use_existing_index")
                .and_then(bool_value)
                .unwrap_or(defaults.use_existing_index),
        }
    }
}

/// Reads a non-negative integer from a JSON number or numeric string.
fn usize_value(value: &Value) -> Option<usize> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .and_then(|number| usize::try_from(number).ok()),
        Value::String(text) => text.trim().parse::<usize>().ok(),
        _ => None,
    }
}

/// Reads a finite number from a JSON number or numeric string.
fn f64_value(value: &Value) -> Option<f64> {
    let parsed = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }?;
    parsed.is_finite().then_some(parsed)
}

/// Reads a boolean from a JSON boolean or the usual `"true"`/`"false"` strings.
fn bool_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => Some(true),
            "false" | "no" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_python_settings() {
        let settings = DiscoverySettings::default();
        assert_eq!(settings.max_results, 10);
        assert_eq!(settings.max_output_chars, 12_000);
        assert_eq!(settings.candidate_budget_ratio, 0.01);
        assert_eq!(settings.context_window_tokens, 200_000);
        assert!(settings.use_existing_index);
        assert_eq!(settings.candidate_budget_tokens(), 2_000);
    }

    #[test]
    fn budget_is_clamped_and_non_negative() {
        let clamped = DiscoverySettings {
            candidate_budget_ratio: 4.0,
            ..DiscoverySettings::default()
        };
        assert_eq!(clamped.candidate_budget_tokens(), 200_000);
        let negative = DiscoverySettings {
            candidate_budget_ratio: -1.0,
            ..DiscoverySettings::default()
        };
        assert_eq!(negative.candidate_budget_tokens(), 0);
        let not_a_number = DiscoverySettings {
            candidate_budget_ratio: f64::NAN,
            ..DiscoverySettings::default()
        };
        assert_eq!(not_a_number.candidate_budget_tokens(), 0);
    }

    #[test]
    fn reader_coerces_numbers_and_strings() {
        let settings = DiscoverySettings::from_value(serde_json::json!({
            "max_results": "3",
            "max_output_chars": 4000,
            "candidate_budget_ratio": 0.25,
            "context_window_tokens": 8000,
            "use_existing_index": "false"
        }));
        assert_eq!(settings.max_results, 3);
        assert_eq!(settings.max_output_chars, 4_000);
        assert_eq!(settings.candidate_budget_ratio, 0.25);
        assert_eq!(settings.context_window_tokens, 8_000);
        assert!(!settings.use_existing_index);
        assert_eq!(settings.candidate_budget_tokens(), 2_000);
    }

    #[test]
    fn reader_falls_back_on_unusable_fields() {
        let settings = DiscoverySettings::from_value(serde_json::json!({
            "max_results": -1,
            "max_output_chars": "many",
            "candidate_budget_ratio": "half",
            "context_window_tokens": null,
            "use_existing_index": 42
        }));
        assert_eq!(settings, DiscoverySettings::default());

        assert_eq!(
            DiscoverySettings::from_value(Value::Null),
            DiscoverySettings::default()
        );
        assert_eq!(
            DiscoverySettings::from_value(serde_json::json!([1, 2])),
            DiscoverySettings::default()
        );
    }
}
