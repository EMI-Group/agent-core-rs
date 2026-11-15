//! Prompt/completion redaction: truncation, hashing, and JSON walking.
//!
//! Port of `openjiuwen/extensions/tracer_otel/redaction.py` plus the
//! `_redacted_json` walker from `handler.py` (v0.1.20). The Python helpers take
//! `object` and stringify with `str(value)`; here the payload is already a
//! [`serde_json::Value`], so non-string values become compact JSON text instead
//! of Python's `repr`.
//!
//! This module compiles with and without the `otel` feature: it only needs the
//! configuration type and `sha2`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::config::OtelTracerConfig;

/// Prefix identifying a hashed attribute value: `"sha256:"`.
const REDACTED_PREFIX: &str = "sha256:";
/// Number of hexadecimal characters kept from the SHA-256 digest.
const HASH_PREFIX_LEN: usize = 16;
/// Object keys whose values carry prompt or completion text.
const REDACTED_KEYS: [&str; 3] = ["content", "arguments", "response"];
/// Lowercase hexadecimal digits used to encode the digest by hand.
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

/// Caps an `OTel` attribute and identifies the truncation layer.
///
/// A `max_length` of `0` disables truncation (Python parity: `max_length <= 0`
/// returns the value untouched). The value is measured in characters, so
/// multi-byte text is never split on a byte boundary.
///
/// ```
/// use openjiuwen_extensions::otel::redaction::truncate;
///
/// assert_eq!(truncate("abc", 10), "abc");
/// assert_eq!(
///     truncate("abcdefghij", 5),
///     "abcde...<OTel attribute truncated: 5 chars omitted>"
/// );
/// ```
#[must_use]
pub fn truncate(value: &str, max_length: usize) -> String {
    if max_length == 0 {
        return value.to_owned();
    }
    let total = value.chars().count();
    if total <= max_length {
        return value.to_owned();
    }
    let omitted = total - max_length;
    let kept: String = value.chars().take(max_length).collect();
    format!("{kept}...<OTel attribute truncated: {omitted} chars omitted>")
}

/// Replaces a value with a short content hash for correlation.
///
/// The result is `"sha256:"` followed by the first 16 lowercase hexadecimal
/// characters of the SHA-256 digest of the UTF-8 bytes (Python parity), so the
/// same input always maps to the same opaque identifier.
///
/// ```
/// use openjiuwen_extensions::otel::redaction::hash_value;
///
/// assert_eq!(hash_value(""), "sha256:e3b0c44298fc1c14");
/// ```
#[must_use]
pub fn hash_value(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut hashed = String::with_capacity(REDACTED_PREFIX.len() + HASH_PREFIX_LEN);
    hashed.push_str(REDACTED_PREFIX);
    for &byte in digest.iter().take(HASH_PREFIX_LEN / 2) {
        hashed.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
        hashed.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
    }
    hashed
}

/// Selects the per-field redaction override that applies to a payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionField {
    /// Prompt/input text; honours [`OtelTracerConfig::redact_prompts`].
    Prompts,
    /// Completion/output text; honours [`OtelTracerConfig::redact_completions`].
    Completions,
}

/// Applies an [`OtelTracerConfig`]'s redaction policy to attribute values.
///
/// Redaction is either a hash (when the policy applies) or a plain
/// truncation to [`OtelTracerConfig::max_attr_length`]; system instructions are
/// exempt from truncation so their identity stays complete.
#[derive(Debug, Clone, Copy)]
pub struct Redactor<'a> {
    /// The configuration that drives hashing, truncation, and field overrides.
    pub config: &'a OtelTracerConfig,
}

impl<'a> Redactor<'a> {
    /// Builds a redactor over `config`.
    #[must_use]
    pub fn new(config: &'a OtelTracerConfig) -> Self {
        Self { config }
    }

    /// Applies the policy to `value` and always returns a string.
    ///
    /// `field` selects the fine-grained prompt/completion override; `None` uses
    /// the legacy [`OtelTracerConfig::redaction_enabled`] flag. A `None` value
    /// redacts to the empty string (or its hash).
    #[must_use]
    pub fn redact(&self, value: Option<&Value>, field: Option<RedactionField>) -> String {
        let text = text_of(value);
        if should_redact(self.config, field) {
            return hash_value(&text);
        }
        truncate(&text, self.config.max_attr_length)
    }

    /// Applies the prompt policy to a system instruction.
    ///
    /// Prompts are hashed only when prompt redaction is enabled; otherwise the
    /// complete text is returned and the length cap never applies.
    #[must_use]
    pub fn redact_system_prompt(&self, value: Option<&Value>) -> String {
        let text = text_of(value);
        if should_redact(self.config, Some(RedactionField::Prompts)) {
            return hash_value(&text);
        }
        text
    }

    /// Redacts every prompt/completion-bearing field of a JSON payload.
    ///
    /// Object keys named exactly `content`, `arguments`, or `response` are
    /// replaced by a redacted string; every other object value and every array
    /// element is walked recursively; scalars pass through unchanged. `field`
    /// picks the override, and `system_prompt` switches the replacement to
    /// [`Self::redact_system_prompt`]. The result is compact JSON with non-ASCII
    /// characters preserved.
    #[must_use]
    pub fn redact_json(
        &self,
        value: &Value,
        field: Option<RedactionField>,
        system_prompt: bool,
    ) -> String {
        let redacted = self.walk(value, field, system_prompt);
        // `Value`'s `Display` is the same compact JSON `serde_json::to_string`
        // writes, and it is infallible: a `Value` contains neither non-string map
        // keys nor non-finite numbers, so this can never degrade to an empty
        // string.
        redacted.to_string()
    }

    /// Recursive worker behind [`Self::redact_json`].
    fn walk(self, value: &Value, field: Option<RedactionField>, system_prompt: bool) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(key, nested)| {
                        let replacement = if REDACTED_KEYS.contains(&key.as_str()) {
                            let text = if system_prompt {
                                self.redact_system_prompt(Some(nested))
                            } else {
                                self.redact(Some(nested), field)
                            };
                            Value::String(text)
                        } else {
                            self.walk(nested, field, system_prompt)
                        };
                        (key.clone(), replacement)
                    })
                    .collect(),
            ),
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .map(|item| self.walk(item, field, system_prompt))
                    .collect(),
            ),
            scalar => scalar.clone(),
        }
    }
}

/// Resolves whether a payload should be hashed rather than truncated.
///
/// `redact_prompts` / `redact_completions` override `redaction_enabled` when set
/// to `Some(true)` or `Some(false)`; `None` falls back to the legacy flag.
fn should_redact(config: &OtelTracerConfig, field: Option<RedactionField>) -> bool {
    let override_flag = match field {
        Some(RedactionField::Prompts) => config.redact_prompts,
        Some(RedactionField::Completions) => config.redact_completions,
        None => None,
    };
    override_flag.unwrap_or(config.redaction_enabled)
}

/// Extracts the text an attribute would carry for `value`.
///
/// Strings are used verbatim, `None` is the empty string, and any other JSON is
/// rendered as compact JSON (Python parity: `str(value)`).
fn text_of(value: Option<&Value>) -> String {
    match value {
        None => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    // port of tests/unit_tests/extensions/tracer_otel/test_redaction.py

    /// `sha256("")`, i.e. the digest of the empty payload.
    const SHA256_EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn short_value_is_not_truncated() {
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn long_value_is_truncated_with_marker() {
        assert_eq!(
            truncate("abcdefghij", 5),
            "abcde...<OTel attribute truncated: 5 chars omitted>"
        );
    }

    #[test]
    fn exact_length_is_not_truncated() {
        assert_eq!(truncate("abcde", 5), "abcde");
    }

    #[test]
    fn zero_max_length_passes_through() {
        assert_eq!(truncate("abc", 0), "abc");
    }

    #[test]
    fn multibyte_value_truncates_on_character_boundaries() {
        assert_eq!(
            truncate("日本語テキスト", 3),
            "日本語...<OTel attribute truncated: 4 chars omitted>"
        );
    }

    #[test]
    fn hash_uses_sha256_prefix() {
        // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        assert_eq!(hash_value("hello"), "sha256:2cf24dba5fb0a30e");
    }

    #[test]
    fn hash_of_empty_string() {
        assert_eq!(hash_value(""), format!("sha256:{}", &SHA256_EMPTY[..16]));
        assert_eq!(hash_value(""), "sha256:e3b0c44298fc1c14");
    }

    #[test]
    fn hash_is_deterministic_and_case_preserving() {
        assert_eq!(hash_value("hello"), hash_value("hello"));
        assert_ne!(hash_value("hello"), hash_value("Hello"));
        let hashed = hash_value("abc");
        assert_eq!(hashed.len(), "sha256:".len() + 16);
        assert!(hashed["sha256:".len()..]
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn redaction_enabled_returns_hash() {
        let config = OtelTracerConfig {
            redaction_enabled: true,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let result = redactor.redact(Some(&json!("hello world")), None);
        assert!(result.starts_with("sha256:"), "{result}");
        assert_eq!(result, hash_value("hello world"));
    }

    #[test]
    fn redaction_disabled_truncates() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            max_attr_length: 10,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        assert_eq!(
            redactor.redact(Some(&json!("hello world longer text")), None),
            "hello worl...<OTel attribute truncated: 13 chars omitted>"
        );
    }

    #[test]
    fn redact_none_returns_hash_of_empty_string() {
        let config = OtelTracerConfig::default();
        let redactor = Redactor::new(&config);
        assert_eq!(
            redactor.redact(None, None),
            format!("sha256:{}", &SHA256_EMPTY[..16])
        );
    }

    #[test]
    fn redact_none_with_redaction_disabled_returns_empty() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        assert_eq!(redactor.redact(None, None), "");
    }

    #[test]
    fn redact_non_string_value_is_hashed() {
        let config = OtelTracerConfig::default();
        let redactor = Redactor::new(&config);
        assert_eq!(redactor.redact(Some(&json!(123)), None), hash_value("123"));
        assert_eq!(
            redactor.redact(Some(&json!({"a": [1, 2]})), None),
            hash_value("{\"a\":[1,2]}")
        );
    }

    #[test]
    fn redact_short_value_without_hash_or_truncation() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            max_attr_length: 100,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        assert_eq!(redactor.redact(Some(&json!("short")), None), "short");
    }

    #[test]
    fn should_redact_field_none_follows_legacy_flag() {
        let enabled = OtelTracerConfig {
            redaction_enabled: true,
            ..OtelTracerConfig::default()
        };
        let disabled = OtelTracerConfig {
            redaction_enabled: false,
            ..OtelTracerConfig::default()
        };
        assert!(should_redact(&enabled, None));
        assert!(!should_redact(&disabled, None));
    }

    #[test]
    fn should_redact_prompts_override() {
        let overridden_on = OtelTracerConfig {
            redaction_enabled: false,
            redact_prompts: Some(true),
            ..OtelTracerConfig::default()
        };
        let overridden_off = OtelTracerConfig {
            redaction_enabled: true,
            redact_prompts: Some(false),
            ..OtelTracerConfig::default()
        };
        let fallback = OtelTracerConfig {
            redaction_enabled: true,
            redact_prompts: None,
            ..OtelTracerConfig::default()
        };
        assert!(should_redact(&overridden_on, Some(RedactionField::Prompts)));
        assert!(!should_redact(
            &overridden_off,
            Some(RedactionField::Prompts)
        ));
        assert!(should_redact(&fallback, Some(RedactionField::Prompts)));
    }

    #[test]
    fn should_redact_completions_override() {
        let overridden_on = OtelTracerConfig {
            redaction_enabled: false,
            redact_completions: Some(true),
            ..OtelTracerConfig::default()
        };
        let overridden_off = OtelTracerConfig {
            redaction_enabled: true,
            redact_completions: Some(false),
            ..OtelTracerConfig::default()
        };
        let fallback = OtelTracerConfig {
            redaction_enabled: false,
            redact_completions: None,
            ..OtelTracerConfig::default()
        };
        assert!(should_redact(
            &overridden_on,
            Some(RedactionField::Completions)
        ));
        assert!(!should_redact(
            &overridden_off,
            Some(RedactionField::Completions)
        ));
        assert!(!should_redact(&fallback, Some(RedactionField::Completions)));
    }

    #[test]
    fn should_redact_fields_are_independent() {
        let config = OtelTracerConfig {
            redaction_enabled: true,
            redact_prompts: Some(false),
            redact_completions: Some(true),
            ..OtelTracerConfig::default()
        };
        assert!(!should_redact(&config, Some(RedactionField::Prompts)));
        assert!(should_redact(&config, Some(RedactionField::Completions)));
    }

    #[test]
    fn redact_prompt_with_override_truncates() {
        let config = OtelTracerConfig {
            redaction_enabled: true,
            redact_prompts: Some(false),
            max_attr_length: 100,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let result = redactor.redact(Some(&json!("secret prompt")), Some(RedactionField::Prompts));
        assert!(!result.starts_with("sha256:"));
        assert_eq!(result, "secret prompt");
    }

    #[test]
    fn redact_completion_with_override_truncates() {
        let config = OtelTracerConfig {
            redaction_enabled: true,
            redact_completions: Some(false),
            max_attr_length: 100,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let result = redactor.redact(
            Some(&json!("secret response")),
            Some(RedactionField::Completions),
        );
        assert!(!result.starts_with("sha256:"));
        assert_eq!(result, "secret response");
    }

    #[test]
    fn redact_prompt_override_hashes() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            redact_prompts: Some(true),
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let result = redactor.redact(Some(&json!("test")), Some(RedactionField::Prompts));
        assert!(result.starts_with("sha256:"));
        assert_eq!(result, hash_value("test"));
    }

    #[test]
    fn redact_field_none_uses_legacy_flag() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            redact_prompts: Some(true),
            max_attr_length: 100,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let result = redactor.redact(Some(&json!("data")), None);
        assert!(!result.starts_with("sha256:"));
        assert_eq!(result, "data");
    }

    #[test]
    fn system_prompt_bypasses_length_cap() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            max_attr_length: 5,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let value = json!("complete-system-prompt");
        assert_eq!(redactor.redact_system_prompt(Some(&value)), value);
    }

    #[test]
    fn system_prompt_hashes_when_prompt_redaction_is_on() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            redact_prompts: Some(true),
            max_attr_length: 5,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let hashed = redactor.redact_system_prompt(Some(&json!("secret-system-prompt")));
        assert!(hashed.starts_with("sha256:"), "{hashed}");
        assert_eq!(hashed, hash_value("secret-system-prompt"));
    }

    #[test]
    fn system_prompt_none_is_empty_or_hashed() {
        let plain = OtelTracerConfig {
            redaction_enabled: false,
            ..OtelTracerConfig::default()
        };
        assert_eq!(Redactor::new(&plain).redact_system_prompt(None), "");
        let hashing = OtelTracerConfig::default();
        assert_eq!(
            Redactor::new(&hashing).redact_system_prompt(None),
            hash_value("")
        );
    }

    #[test]
    fn redact_json_redacts_nested_content_arguments_and_response() {
        let config = OtelTracerConfig::default();
        let redactor = Redactor::new(&config);
        let payload = json!({
            "role": "user",
            "content": "hello",
            "tool_calls": [
                {"arguments": {"q": "secret"}, "id": "call-1"},
            ],
            "response": "world",
        });
        let rendered = redactor.redact_json(&payload, Some(RedactionField::Prompts), false);
        let walked: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(walked["role"], json!("user"));
        assert_eq!(walked["id"], Value::Null);
        assert_eq!(walked["content"], json!(hash_value("hello")));
        assert_eq!(walked["response"], json!(hash_value("world")));
        assert_eq!(
            walked["tool_calls"][0]["arguments"],
            json!(hash_value("{\"q\":\"secret\"}"))
        );
        assert_eq!(walked["tool_calls"][0]["id"], json!("call-1"));
    }

    #[test]
    fn redact_json_leaves_untouched_keys_verbatim_when_redaction_is_off() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            max_attr_length: 100,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let payload = json!({"messages": [{"role": "user", "content": "héllo"}]});
        let rendered = redactor.redact_json(&payload, Some(RedactionField::Prompts), false);
        assert!(
            rendered.contains("héllo"),
            "non-ASCII must stay raw: {rendered}"
        );
        let walked: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(walked, payload);
    }

    #[test]
    fn redact_json_system_prompt_is_not_truncated() {
        let config = OtelTracerConfig {
            redaction_enabled: false,
            max_attr_length: 5,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let payload = json!({"content": "a-complete-system-prompt"});
        assert_eq!(
            redactor.redact_json(&payload, Some(RedactionField::Prompts), true),
            "{\"content\":\"a-complete-system-prompt\"}"
        );
    }

    #[test]
    fn redact_json_system_prompt_still_hashes_when_enabled() {
        let config = OtelTracerConfig::default();
        let redactor = Redactor::new(&config);
        let payload = json!({"content": "system prompt"});
        let rendered = redactor.redact_json(&payload, Some(RedactionField::Prompts), true);
        let walked: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(walked["content"], json!(hash_value("system prompt")));
    }

    #[test]
    fn redact_json_passes_scalars_and_arrays_through() {
        let config = OtelTracerConfig::default();
        let redactor = Redactor::new(&config);
        assert_eq!(redactor.redact_json(&json!(7), None, false), "7");
        assert_eq!(
            redactor.redact_json(&json!([1, "two", null, true]), None, false),
            "[1,\"two\",null,true]"
        );
        assert_eq!(redactor.redact_json(&json!({}), None, false), "{}");
    }

    #[test]
    fn redact_json_completion_field_uses_completion_override() {
        let config = OtelTracerConfig {
            redaction_enabled: true,
            redact_completions: Some(false),
            max_attr_length: 100,
            ..OtelTracerConfig::default()
        };
        let redactor = Redactor::new(&config);
        let payload = json!({"content": "open completion"});
        assert_eq!(
            redactor.redact_json(&payload, Some(RedactionField::Completions), false),
            "{\"content\":\"open completion\"}"
        );
        // The prompt field keeps hashing under the same configuration.
        let prompts = redactor.redact_json(&payload, Some(RedactionField::Prompts), false);
        let walked: Value = serde_json::from_str(&prompts).unwrap();
        assert_eq!(walked["content"], json!(hash_value("open completion")));
    }
}
