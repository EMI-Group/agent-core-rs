//! Serde shapes of a `HuggingFace` `tokenizer.json` document.
//!
//! Only the sections this crate decodes are typed; the polymorphic pipeline nodes stay [`serde_json::Value`] because each one carries a different set of fields.
//! Unknown fields are ignored, so a newer artifact keeps loading and everything not read here is dropped.

use serde::Deserialize;
use serde_json::Value;

/// The top-level `tokenizer.json` document.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TokenizerJson {
    /// Optional artifact name.
    #[serde(default)]
    pub(crate) name: Option<String>,
    /// Optional model id that some exports carry instead of a name.
    #[serde(default)]
    pub(crate) model_id: Option<String>,
    /// The decoding model, the only section a load cannot proceed without.
    #[serde(default)]
    pub(crate) model: Option<Value>,
    /// The normalization pipeline.
    #[serde(default)]
    pub(crate) normalizer: Option<Value>,
    /// The pre-tokenization pipeline.
    #[serde(default)]
    pub(crate) pre_tokenizer: Option<Value>,
    /// The decoding pipeline used by [`super::NativeTokenizer::decode`].
    #[serde(default)]
    pub(crate) decoder: Option<Value>,
    /// Added and special tokens, resolved before the model sees the text.
    #[serde(default)]
    pub(crate) added_tokens: Option<Vec<AddedTokenJson>>,
}

/// One entry of the `added_tokens` array.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AddedTokenJson {
    /// The id the token is emitted with.
    pub(crate) id: u32,
    /// The text that matches this token.
    pub(crate) content: String,
    /// Whether the token is a control token (`<|endoftext|>`, `[CLS]`, ...).
    #[serde(default)]
    pub(crate) special: bool,
    /// Whether the token may only match when surrounded by non-word characters.
    #[serde(default)]
    pub(crate) single_word: bool,
    /// Whether whitespace before the match is dropped.
    #[serde(default)]
    pub(crate) lstrip: bool,
    /// Whether whitespace after the match is dropped (`rsw` is `HuggingFace`'s name for right-strip).
    #[serde(default)]
    pub(crate) rsw: bool,
}

/// The `"type"` tag of a polymorphic pipeline node.
pub(crate) fn kind_of(value: &Value) -> Option<&str> {
    value.get("type").and_then(Value::as_str)
}

/// A boolean member, defaulting when absent or of the wrong type.
pub(crate) fn bool_of(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// A string member.
pub(crate) fn str_of<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

/// An unsigned integer member, defaulting when absent or of the wrong type.
pub(crate) fn usize_of(value: &Value, key: &str, default: usize) -> usize {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .unwrap_or(default)
}

/// The literal inside a `{"String": "..."}` pattern, when the pattern is not a regex.
///
/// This resolves the object shape only: a bare string pattern belongs to [`regex_pattern`], so the two resolvers never claim the same pattern and a caller always knows which kind it got.
pub(crate) fn literal_pattern(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)?
        .get("String")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// The source of a `{"Regex": "..."}` pattern, or of a bare string pattern.
///
/// A bare string is the legacy regex shape and is claimed only here, because [`literal_pattern`] resolves the `{"String": ...}` object alone.
pub(crate) fn regex_pattern<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    let pattern = value.get(key)?;
    if let Some(source) = pattern.get("Regex") {
        return source.as_str();
    }
    pattern.as_str()
}

/// Whether a pattern member is a regex.
pub(crate) fn pattern_is_regex(value: &Value, key: &str) -> bool {
    value
        .get(key)
        .is_some_and(|pattern| pattern.get("Regex").is_some())
}

/// A one-line description of a pattern member's JSON shape, for a warning that has to say what the artifact declared.
pub(crate) fn describe_pattern(value: &Value, key: &str) -> String {
    match value.get(key) {
        None => format!("no `{key}` member"),
        Some(Value::String(text)) => format!("the bare string `{text}`"),
        Some(Value::Object(object)) => match object.iter().next() {
            Some((variant, Value::String(text))) => format!("the {variant} pattern `{text}`"),
            Some((variant, _)) => format!("a {variant} pattern that is not a string"),
            None => "an empty pattern object".to_owned(),
        },
        Some(other) => format!("a pattern of JSON type {}", json_type(other)),
    }
}

/// The JSON type of a value, for a diagnostic.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_two_pattern_resolvers_are_exclusive() {
        let regex = json!({"pattern": {"Regex": "\\s+"}});
        assert_eq!(regex_pattern(&regex, "pattern"), Some("\\s+"));
        assert_eq!(literal_pattern(&regex, "pattern"), None);

        let literal = json!({"pattern": {"String": "\\s+"}});
        assert_eq!(regex_pattern(&literal, "pattern"), None);
        assert_eq!(
            literal_pattern(&literal, "pattern").as_deref(),
            Some("\\s+")
        );

        // A bare string is the legacy regex shape, so only `regex_pattern` claims it.
        let bare = json!({"pattern": "a+"});
        assert_eq!(regex_pattern(&bare, "pattern"), Some("a+"));
        assert_eq!(literal_pattern(&bare, "pattern"), None);

        let missing = json!({});
        assert_eq!(regex_pattern(&missing, "pattern"), None);
        assert_eq!(literal_pattern(&missing, "pattern"), None);
    }

    #[test]
    fn a_non_string_regex_member_resolves_to_nothing() {
        let value = json!({"pattern": {"Regex": 7}});
        assert_eq!(regex_pattern(&value, "pattern"), None);
        assert!(pattern_is_regex(&value, "pattern"));
        assert_eq!(
            describe_pattern(&value, "pattern"),
            "a Regex pattern that is not a string"
        );
    }

    #[test]
    fn describe_pattern_names_the_shape() {
        assert_eq!(
            describe_pattern(&json!({}), "pattern"),
            "no `pattern` member"
        );
        assert_eq!(
            describe_pattern(&json!({"pattern": "x"}), "pattern"),
            "the bare string `x`"
        );
        assert_eq!(
            describe_pattern(&json!({"pattern": {"String": "-"}}), "pattern"),
            "the String pattern `-`"
        );
        assert_eq!(
            describe_pattern(&json!({"pattern": {}}), "pattern"),
            "an empty pattern object"
        );
        assert_eq!(
            describe_pattern(&json!({"pattern": 3}), "pattern"),
            "a pattern of JSON type number"
        );
    }
}
