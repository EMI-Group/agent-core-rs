//! Text normalization applied before pre-tokenization.
//!
//! Only the operations a `tokenizer.json` can express without a Unicode-normalization crate are reproduced: case folding, `BERT` whitespace/control handling, accent stripping, `Strip`, `Prepend` and literal `Replace`.
//! A normalizer this module cannot reproduce (the `NFC`/`NFD`/`NFKC`/`NFKD` families, `Nmt`, `Precompiled`) is recorded as a warning and skipped rather than failing the load, because an unnormalized token count is still useful.

use serde_json::Value;

use super::hf_json::{bool_of, kind_of, literal_pattern, str_of};

/// A normalization step.
#[derive(Debug, Clone, Default)]
pub(crate) enum Normalizer {
    /// The text is left as it is.
    #[default]
    Identity,
    /// Steps applied in order.
    Sequence(Vec<Normalizer>),
    /// Lowercase the text.
    LowerCase,
    /// `BERT`'s normalizer.
    Bert(BertConfig),
    /// Strip whitespace from either end.
    Strip {
        /// Strip the left end.
        left: bool,
        /// Strip the right end.
        right: bool,
    },
    /// Prefix the text.
    Prepend {
        /// The prefix.
        prepend: String,
    },
    /// Replace a literal string.
    Replace {
        /// The literal to find.
        pattern: String,
        /// The replacement.
        content: String,
    },
}

impl Normalizer {
    /// Decodes a normalizer section, recording anything it cannot reproduce.
    pub(crate) fn from_json(value: Option<&Value>, warnings: &mut Vec<String>) -> Self {
        let Some(value) = value else {
            return Self::Identity;
        };
        match value {
            Value::Null => Self::Identity,
            Value::Object(_) => Self::from_object(value, warnings),
            _ => {
                warnings
                    .push("normalizer is not an object; the text is left unnormalized".to_owned());
                Self::Identity
            }
        }
    }

    /// Decodes one object node.
    fn from_object(value: &Value, warnings: &mut Vec<String>) -> Self {
        let Some(kind) = kind_of(value) else {
            warnings.push("normalizer declares no type; the text is left unnormalized".to_owned());
            return Self::Identity;
        };
        match kind {
            "Sequence" => {
                let stages = value
                    .get("normalizers")
                    .and_then(Value::as_array)
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| Self::from_object(entry, warnings))
                            .collect()
                    })
                    .unwrap_or_default();
                Self::Sequence(stages)
            }
            "Lowercase" | "LowerCase" => Self::LowerCase,
            "BertNormalizer" => {
                let lowercase = bool_of(value, "lowercase", true);
                Self::Bert(BertConfig {
                    clean_text: bool_of(value, "clean_text", true),
                    handle_chinese_chars: bool_of(value, "handle_chinese_chars", true),
                    strip_accents: value
                        .get("strip_accents")
                        .and_then(Value::as_bool)
                        .unwrap_or(lowercase),
                    lowercase,
                })
            }
            "Strip" => Self::Strip {
                left: bool_of(value, "strip_left", true),
                right: bool_of(value, "strip_right", true),
            },
            "Prepend" => Self::Prepend {
                prepend: str_of(value, "prepend").unwrap_or_default().to_owned(),
            },
            "Replace" => {
                let Some(pattern) = literal_pattern(value, "pattern") else {
                    warnings.push(
                        "normalizer Replace uses a regex pattern; the text is left unnormalized"
                            .to_owned(),
                    );
                    return Self::Identity;
                };
                Self::Replace {
                    pattern,
                    content: str_of(value, "content").unwrap_or_default().to_owned(),
                }
            }
            other => {
                warnings.push(format!(
                    "normalizer `{other}` is not implemented; the text is left unnormalized"
                ));
                Self::Identity
            }
        }
    }

    /// Applies the step.
    pub(crate) fn normalize(&self, text: &str) -> String {
        match self {
            Self::Identity => text.to_owned(),
            Self::Sequence(stages) => stages
                .iter()
                .fold(text.to_owned(), |current, stage| stage.normalize(&current)),
            Self::LowerCase => text.to_lowercase(),
            Self::Bert(config) => config.normalize(text),
            Self::Strip { left, right } => {
                let mut current = text;
                if *left {
                    current = current.trim_start();
                }
                if *right {
                    current = current.trim_end();
                }
                current.to_owned()
            }
            Self::Prepend { prepend } => format!("{prepend}{text}"),
            Self::Replace { pattern, content } => {
                if pattern.is_empty() {
                    text.to_owned()
                } else {
                    text.replace(pattern.as_str(), content)
                }
            }
        }
    }
}

/// `BERT`'s normalizer flags.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BertConfig {
    /// Drop control characters and fold whitespace into single spaces.
    pub(crate) clean_text: bool,
    /// Surround CJK characters with spaces so they pre-tokenize separately.
    pub(crate) handle_chinese_chars: bool,
    /// Remove combining marks and decompose common accented letters.
    pub(crate) strip_accents: bool,
    /// Lowercase the text.
    pub(crate) lowercase: bool,
}

impl BertConfig {
    /// Applies the flags to `text`.
    fn normalize(self, text: &str) -> String {
        let stripped = self.strip_accents(self.clean(text));
        if self.lowercase {
            stripped.to_lowercase()
        } else {
            stripped
        }
    }

    /// Drops control characters and separate CJK characters with spaces.
    fn clean(self, text: &str) -> String {
        let mut cleaned = String::with_capacity(text.len());
        for ch in text.chars() {
            if self.clean_text {
                if ch == '\u{0}' || ch == '\u{fffd}' || ch.is_control() {
                    continue;
                }
                if ch.is_whitespace() {
                    cleaned.push(' ');
                    continue;
                }
            }
            if self.handle_chinese_chars && is_cjk(ch) {
                cleaned.push(' ');
                cleaned.push(ch);
                cleaned.push(' ');
            } else {
                cleaned.push(ch);
            }
        }
        cleaned
    }

    /// Removes combining marks and decomposes common accented letters, best effort without a normalization crate.
    fn strip_accents(self, text: String) -> String {
        if !self.strip_accents {
            return text;
        }
        let mut normalized = String::with_capacity(text.len());
        for ch in text.chars() {
            if let Some(base) = accent_base(ch) {
                normalized.push_str(base);
                continue;
            }
            if is_combining_mark(ch) {
                continue;
            }
            normalized.push(ch);
        }
        normalized
    }
}

/// Whether a character is a combining diacritical mark.
fn is_combining_mark(ch: char) -> bool {
    ('\u{0300}'..='\u{036f}').contains(&ch)
}

/// The unaccented spelling of a precomposed Latin letter.
fn accent_base(ch: char) -> Option<&'static str> {
    let base = match ch {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => "a",
        'ç' | 'ć' | 'č' => "c",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => "e",
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'į' => "i",
        'ñ' | 'ń' | 'ň' => "n",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ő' => "o",
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ů' | 'ű' => "u",
        'ý' | 'ÿ' => "y",
        'ś' | 'š' | 'ş' => "s",
        'ź' | 'ż' | 'ž' => "z",
        'ł' => "l",
        'đ' => "d",
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'Ā' => "A",
        'Ç' | 'Ć' | 'Č' => "C",
        'È' | 'É' | 'Ê' | 'Ë' | 'Ē' | 'Ė' | 'Ę' => "E",
        'Ì' | 'Í' | 'Î' | 'Ï' | 'Ī' => "I",
        'Ñ' | 'Ń' | 'Ň' => "N",
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' | 'Ō' => "O",
        'Ù' | 'Ú' | 'Û' | 'Ü' | 'Ū' | 'Ů' => "U",
        'Ý' => "Y",
        'Ś' | 'Š' | 'Ş' => "S",
        'Ź' | 'Ż' | 'Ž' => "Z",
        'Ł' => "L",
        'Đ' => "D",
        'ß' => "ss",
        _ => return None,
    };
    Some(base)
}

/// Whether a character belongs to a CJK block, which `BERT` separates with spaces.
fn is_cjk(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
        || ('\u{3400}'..='\u{4dbf}').contains(&ch)
        || ('\u{f900}'..='\u{faff}').contains(&ch)
        || ('\u{3040}'..='\u{30ff}').contains(&ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bert_normalizer_lowercases_and_strips_accents() {
        let mut warnings = Vec::new();
        let normalizer = Normalizer::from_json(
            Some(&json!({"type": "BertNormalizer", "lowercase": true})),
            &mut warnings,
        );
        assert_eq!(normalizer.normalize("HÉLLO"), "hello");
        assert_eq!(warnings.len(), 0);
    }

    #[test]
    fn bert_normalizer_separates_cjk_and_folds_whitespace() {
        let mut warnings = Vec::new();
        let normalizer =
            Normalizer::from_json(Some(&json!({"type": "BertNormalizer"})), &mut warnings);
        assert_eq!(normalizer.normalize("中\t文"), " \u{4e2d}  \u{6587} ");
    }

    #[test]
    fn sequences_apply_in_order() {
        let mut warnings = Vec::new();
        let normalizer = Normalizer::from_json(
            Some(&json!({"type": "Sequence", "normalizers": [
                {"type": "Lowercase"},
                {"type": "Prepend", "prepend": "x"}
            ]})),
            &mut warnings,
        );
        assert_eq!(normalizer.normalize("ABC"), "xabc");
    }

    #[test]
    fn unsupported_normalizers_only_warn() {
        let mut warnings = Vec::new();
        let normalizer = Normalizer::from_json(Some(&json!({"type": "NFKC"})), &mut warnings);
        assert_eq!(normalizer.normalize("a"), "a");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("NFKC"));
    }
}
