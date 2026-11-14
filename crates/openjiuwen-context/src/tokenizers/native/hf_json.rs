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

/// The literal inside a `{"String": "..."}` pattern (or a bare string), when the pattern is not a regex.
pub(crate) fn literal_pattern(value: &Value, key: &str) -> Option<String> {
    let pattern = value.get(key)?;
    if let Some(text) = pattern.get("String").and_then(Value::as_str) {
        return Some(text.to_owned());
    }
    pattern.as_str().map(str::to_owned)
}

/// Whether a pattern member is a regex.
pub(crate) fn pattern_is_regex(value: &Value, key: &str) -> bool {
    value
        .get(key)
        .is_some_and(|pattern| pattern.get("Regex").is_some())
}
