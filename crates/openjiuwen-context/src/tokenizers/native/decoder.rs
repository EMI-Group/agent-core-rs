//! Turning token ids back into text.
//!
//! The common `HuggingFace` decoders are reproduced: byte-level (the reverse byte mapping), metaspace, word-piece, `Fuse`, literal `Replace`, `Sequence` chains and the plain case, which joins tokens with spaces.
//! Post-processor templates are deliberately ignored: they add framing tokens and never change a token count.

use serde_json::Value;

use super::bytes;
use super::hf_json::{bool_of, kind_of, literal_pattern, str_of};

/// The default metaspace replacement character.
const DEFAULT_METASPACE_REPLACEMENT: char = '\u{2581}';

/// A decoding step.
#[derive(Debug, Clone, Default)]
pub(crate) enum Decoder {
    /// Join the tokens with spaces, as `HuggingFace` does when no decoder is declared.
    #[default]
    Plain,
    /// Concatenate the tokens without a separator.
    Fuse,
    /// Undo the byte mapping.
    ByteLevel,
    /// Undo the metaspace rewrite.
    Metaspace {
        /// The marker that stood for a space.
        replacement: char,
        /// Whether a leading space is dropped.
        prepend_space: bool,
    },
    /// Join word pieces, gluing continuation pieces back onto the previous word.
    WordPiece {
        /// The continuation prefix.
        prefix: String,
    },
    /// Replace a literal string in the joined text.
    Replace {
        /// The literal to find.
        pattern: String,
        /// The replacement.
        content: String,
    },
    /// Steps applied in order.
    Sequence(Vec<Decoder>),
}

impl Decoder {
    /// Decodes a decoder section.
    pub(crate) fn from_json(value: Option<&Value>, warnings: &mut Vec<String>) -> Self {
        let Some(value) = value else {
            return Self::Plain;
        };
        match value {
            Value::Null => Self::Plain,
            Value::Object(_) => Self::from_object(value, warnings),
            _ => {
                warnings.push("decoder is not an object; tokens are joined with spaces".to_owned());
                Self::Plain
            }
        }
    }

    /// Decodes one object node.
    fn from_object(value: &Value, warnings: &mut Vec<String>) -> Self {
        let Some(kind) = kind_of(value) else {
            warnings.push("decoder declares no type; tokens are joined with spaces".to_owned());
            return Self::Plain;
        };
        match kind {
            "ByteLevel" => Self::ByteLevel,
            "Fuse" => Self::Fuse,
            "Metaspace" => Self::Metaspace {
                replacement: str_of(value, "replacement")
                    .and_then(|text| text.chars().next())
                    .unwrap_or(DEFAULT_METASPACE_REPLACEMENT),
                prepend_space: match str_of(value, "prepend_scheme") {
                    Some(scheme) => !scheme.eq_ignore_ascii_case("never"),
                    None => bool_of(value, "add_prefix_space", true),
                },
            },
            "WordPiece" => Self::WordPiece {
                prefix: str_of(value, "prefix").unwrap_or("##").to_owned(),
            },
            "Replace" => {
                let Some(pattern) = literal_pattern(value, "pattern") else {
                    warnings.push(
                        "decoder Replace uses a regex pattern; tokens are joined without it"
                            .to_owned(),
                    );
                    return Self::Plain;
                };
                Self::Replace {
                    pattern,
                    content: str_of(value, "content").unwrap_or_default().to_owned(),
                }
            }
            "Sequence" => Self::Sequence(
                value
                    .get("decoders")
                    .and_then(Value::as_array)
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| Self::from_object(entry, warnings))
                            .collect()
                    })
                    .unwrap_or_default(),
            ),
            other => {
                warnings.push(format!(
                    "decoder `{other}` is not implemented; tokens are joined with spaces"
                ));
                Self::Plain
            }
        }
    }

    /// Decodes `tokens` into text.
    pub(crate) fn decode(&self, tokens: &[String]) -> String {
        match self {
            Self::Plain => tokens.join(" "),
            Self::Fuse => tokens.concat(),
            Self::ByteLevel => String::from_utf8_lossy(&byte_stream(tokens)).into_owned(),
            Self::Metaspace {
                replacement,
                prepend_space,
            } => {
                let mut text = tokens.concat().replace(*replacement, " ");
                if *prepend_space {
                    if let Some(stripped) = text.strip_prefix(' ') {
                        text = stripped.to_owned();
                    }
                }
                text
            }
            Self::WordPiece { prefix } => {
                let mut text = String::new();
                for token in tokens {
                    if let Some(rest) = token.strip_prefix(prefix.as_str()) {
                        text.push_str(rest);
                    } else {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(token);
                    }
                }
                text
            }
            Self::Replace { pattern, content } => {
                let joined = tokens.join(" ");
                if pattern.is_empty() {
                    joined
                } else {
                    joined.replace(pattern.as_str(), content)
                }
            }
            Self::Sequence(stages) => {
                let Some((last, head)) = stages.split_last() else {
                    return tokens.join(" ");
                };
                let mut current = tokens.to_vec();
                for stage in head {
                    current = stage.transform(current);
                }
                last.decode(&current)
            }
        }
    }

    /// Applies the step to a token list, for a step that is not the last of a sequence.
    fn transform(&self, tokens: Vec<String>) -> Vec<String> {
        match self {
            Self::Plain => vec![tokens.join(" ")],
            Self::Fuse => vec![tokens.concat()],
            Self::ByteLevel => tokens
                .into_iter()
                .map(|token| String::from_utf8_lossy(&bytes::token_bytes(&token)).into_owned())
                .collect(),
            Self::Metaspace { replacement, .. } => tokens
                .into_iter()
                .map(|token| token.replace(*replacement, " "))
                .collect(),
            Self::Replace { pattern, content } => {
                if pattern.is_empty() {
                    tokens
                } else {
                    tokens
                        .into_iter()
                        .map(|token| token.replace(pattern.as_str(), content))
                        .collect()
                }
            }
            Self::WordPiece { .. } | Self::Sequence(_) => tokens,
        }
    }
}

/// The byte stream a token list stands for after undoing the byte mapping.
fn byte_stream(tokens: &[String]) -> Vec<u8> {
    let mut stream = Vec::new();
    for token in tokens {
        stream.extend(bytes::token_bytes(token));
    }
    stream
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plain_join_inserts_spaces() {
        assert_eq!(
            Decoder::Plain.decode(&["a".to_owned(), "b".to_owned()]),
            "a b"
        );
    }

    #[test]
    fn word_piece_glues_continuations() {
        let mut warnings = Vec::new();
        let decoder = Decoder::from_json(
            Some(&json!({"type": "WordPiece", "prefix": "##"})),
            &mut warnings,
        );
        assert_eq!(
            decoder.decode(&["hello".to_owned(), "##ing".to_owned()]),
            "helloing"
        );
    }

    #[test]
    fn byte_level_decodes_multibyte_text() {
        let mut warnings = Vec::new();
        let decoder = Decoder::from_json(Some(&json!({"type": "ByteLevel"})), &mut warnings);
        let mapped = bytes::encode("héllo");
        let tokens: Vec<String> = mapped.chars().map(String::from).collect();
        assert_eq!(decoder.decode(&tokens), "héllo");
    }

    #[test]
    fn metaspace_turns_the_marker_back_into_a_space() {
        let mut warnings = Vec::new();
        let decoder = Decoder::from_json(
            Some(
                &json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always"}),
            ),
            &mut warnings,
        );
        assert_eq!(
            decoder.decode(&["\u{2581}hello".to_owned(), "\u{2581}world".to_owned()]),
            "hello world"
        );
    }

    #[test]
    fn sequences_apply_every_step() {
        let mut warnings = Vec::new();
        let decoder = Decoder::from_json(
            Some(
                &json!({"type": "Sequence", "decoders": [{"type": "ByteLevel"}, {"type": "Fuse"}]}),
            ),
            &mut warnings,
        );
        let mapped = bytes::encode("hi");
        let tokens: Vec<String> = mapped.chars().map(String::from).collect();
        assert_eq!(decoder.decode(&tokens), "hi");
    }
}
