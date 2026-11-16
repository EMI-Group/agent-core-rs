//! Text normalization applied before pre-tokenization.
//!
//! The operations a `tokenizer.json` can express are reproduced here: a `Sequence` of stages, case folding, `BERT` whitespace/control handling, accent stripping, `Strip`, `Prepend`, literal and regex `Replace`, and the `NFC`/`NFD`/`NFKC`/`NFKD` Unicode normalizations (applied through `unicode-normalization`).
//! A regex `Replace` compiles through the `regex` crate, so Rust's replacement syntax (`$1`, `${name}`) applies; a pattern that does not compile is recorded as a warning naming the pattern source and the `regex` error, and that step leaves the text unchanged.
//! Only `Nmt` and `Precompiled` remain unimplemented: they need the `SentencePiece` compiled-character-map model, which has no offline implementation here, so they warn and are skipped rather than failing the load, because an unnormalized token count is still useful.

use regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use super::hf_json::{bool_of, describe_pattern, kind_of, literal_pattern, regex_pattern, str_of};

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
    /// Unicode canonical composition (`NFC`), the form a `tokenizer.json` writes as `{"type": "NFC"}`.
    Nfc,
    /// Unicode canonical decomposition (`NFD`).
    Nfd,
    /// Unicode compatibility composition (`NFKC`), which also folds full-width and ligature forms.
    Nfkc,
    /// Unicode compatibility decomposition (`NFKD`).
    Nfkd,
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
    /// Replace every match of a compiled regular expression.
    ReplacePattern {
        /// The compiled pattern.
        regex: Regex,
        /// The replacement, in Rust's `$1`/`${name}` syntax.
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
            // `HuggingFace` serialises the Unicode form as `{"type": "NFC", ...}`; the extra
            // SentencePiece fields some exports carry (`longest`) describe pre-tokenization and
            // are not part of the normalization itself.
            "NFC" => Self::Nfc,
            "NFD" => Self::Nfd,
            "NFKC" => Self::Nfkc,
            "NFKD" => Self::Nfkd,
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
            "Replace" => Self::replace(value, warnings),
            "Nmt" | "Precompiled" => {
                warnings.push(format!(
                    "normalizer `{kind}` needs the SentencePiece compiled character map, which is not implemented; the text is left unnormalized by that step"
                ));
                Self::Identity
            }
            other => {
                warnings.push(format!(
                    "normalizer `{other}` is not implemented; the text is left unnormalized"
                ));
                Self::Identity
            }
        }
    }

    /// Decodes a `Replace` node, which is literal for a `{"String": ...}` pattern and a regex otherwise.
    fn replace(value: &Value, warnings: &mut Vec<String>) -> Self {
        let content = str_of(value, "content").unwrap_or_default().to_owned();
        if let Some(pattern) = literal_pattern(value, "pattern") {
            return Self::Replace { pattern, content };
        }
        let Some(source) = regex_pattern(value, "pattern") else {
            warnings.push(format!(
                "normalizer Replace declares {}; the text is left unnormalized by that step",
                describe_pattern(value, "pattern")
            ));
            return Self::Identity;
        };
        match Regex::new(source) {
            Ok(regex) => Self::ReplacePattern { regex, content },
            Err(error) => {
                warnings.push(format!(
                    "normalizer Replace regex `{source}` does not compile ({error}); the text is left unnormalized by that step"
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
            Self::Nfc => text.nfc().collect(),
            Self::Nfd => text.nfd().collect(),
            Self::Nfkc => text.nfkc().collect(),
            Self::Nfkd => text.nfkd().collect(),
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
            Self::ReplacePattern { regex, content } => {
                regex.replace_all(text, content.as_str()).into_owned()
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

    /// Decodes one normalizer, asserting that it produced no warning.
    fn clean(node: Value) -> Normalizer {
        let mut warnings = Vec::new();
        let normalizer = Normalizer::from_json(Some(&node), &mut warnings);
        assert_eq!(warnings, Vec::<String>::new(), "{node}");
        normalizer
    }

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
    fn nfc_composes_and_nfd_decomposes() {
        let decomposed = "e\u{0301}";
        let composed = "\u{e9}";
        assert_eq!(
            clean(json!({"type": "NFC"})).normalize(decomposed),
            composed
        );
        assert_eq!(
            clean(json!({"type": "NFD"})).normalize(composed),
            decomposed
        );
        // Canonical forms agree on text that is already in one piece.
        assert_eq!(clean(json!({"type": "NFC"})).normalize("plain"), "plain");
        // `NFC` is not a compatibility fold: the full-width letter stays full-width.
        assert_eq!(
            clean(json!({"type": "NFC"})).normalize("\u{ff21}"),
            "\u{ff21}"
        );
        assert_eq!(
            clean(json!({"type": "NFD"})).normalize("\u{ff21}"),
            "\u{ff21}"
        );
    }

    #[test]
    fn nfkc_and_nfkd_apply_compatibility_folds() {
        // Full-width `A`, the `fi` ligature and the circled digit one all fold to ASCII.
        for kind in ["NFKC", "NFKD"] {
            let normalizer = clean(json!({"type": kind}));
            assert_eq!(normalizer.normalize("\u{ff21}"), "A", "{kind}");
            assert_eq!(normalizer.normalize("\u{fb01}"), "fi", "{kind}");
            assert_eq!(normalizer.normalize("\u{2460}"), "1", "{kind}");
            assert_eq!(normalizer.normalize("plain"), "plain", "{kind}");
        }
        // `NFKD` decomposes the composed accent that `NFC` would build.
        assert_eq!(
            clean(json!({"type": "NFKD"})).normalize("\u{e9}"),
            "e\u{0301}"
        );
    }

    #[test]
    fn strip_and_prepend_act_on_the_ends() {
        assert_eq!(
            clean(json!({"type": "Strip", "strip_left": true, "strip_right": false}))
                .normalize("  ab  "),
            "ab  "
        );
        assert_eq!(clean(json!({"type": "Strip"})).normalize("  ab  "), "ab");
        assert_eq!(
            clean(json!({"type": "Prepend", "prepend": "\u{2581}"})).normalize("ab"),
            "\u{2581}ab"
        );
    }

    #[test]
    fn a_literal_replace_only_rewrites_the_literal() {
        let normalizer =
            clean(json!({"type": "Replace", "pattern": {"String": "."}, "content": "_"}));
        // The dot is a literal here, so only the dots change.
        assert_eq!(normalizer.normalize("a.bc"), "a_bc");
        assert!(matches!(normalizer, Normalizer::Replace { .. }));
    }

    #[test]
    fn a_regex_replace_uses_rust_replacement_syntax() {
        let spaces =
            clean(json!({"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": "_"}));
        assert_eq!(spaces.normalize("a \t\n b"), "a_b");
        assert!(matches!(spaces, Normalizer::ReplacePattern { .. }));

        let groups = clean(json!({
            "type": "Replace",
            "pattern": {"Regex": "(\\w+)@(\\w+)"},
            "content": "$2:$1"
        }));
        assert_eq!(groups.normalize("user@host"), "host:user");

        // A named group is reachable through `${name}`.
        let named = clean(json!({
            "type": "Replace",
            "pattern": {"Regex": "(?P<word>[a-z]+)"},
            "content": "<${word}>"
        }));
        assert_eq!(named.normalize("ab 12"), "<ab> 12");
    }

    #[test]
    fn a_bare_string_pattern_is_a_regex() {
        let normalizer = clean(json!({"type": "Replace", "pattern": "a+", "content": "-"}));
        assert!(matches!(normalizer, Normalizer::ReplacePattern { .. }));
        assert_eq!(normalizer.normalize("caaat"), "c-t");
    }

    #[test]
    fn a_regex_that_does_not_compile_warns_and_leaves_the_text_unchanged() {
        let mut warnings = Vec::new();
        let normalizer = Normalizer::from_json(
            Some(&json!({"type": "Replace", "pattern": {"Regex": "(unclosed"}, "content": "x"})),
            &mut warnings,
        );
        assert_eq!(normalizer.normalize("(unclosed text"), "(unclosed text");
        assert_eq!(warnings.len(), 1);
        // The warning names the pattern source and the `regex` error.
        assert!(warnings[0].contains("(unclosed"), "{}", warnings[0]);
        assert!(warnings[0].contains("regex"), "{}", warnings[0]);
        assert!(warnings[0].contains("does not compile"), "{}", warnings[0]);
    }

    #[test]
    fn a_replace_without_a_usable_pattern_warns() {
        let mut warnings = Vec::new();
        let normalizer = Normalizer::from_json(Some(&json!({"type": "Replace"})), &mut warnings);
        assert_eq!(normalizer.normalize("abc"), "abc");
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("no `pattern` member"),
            "{}",
            warnings[0]
        );
    }

    #[test]
    fn an_unnormalizable_step_only_warns() {
        for kind in ["Nmt", "Precompiled"] {
            let mut warnings = Vec::new();
            let normalizer = Normalizer::from_json(Some(&json!({"type": kind})), &mut warnings);
            assert_eq!(normalizer.normalize("a"), "a");
            assert_eq!(warnings.len(), 1, "{kind}");
            assert!(warnings[0].contains(kind), "{kind}: {}", warnings[0]);
        }
        let mut warnings = Vec::new();
        let normalizer = Normalizer::from_json(Some(&json!({"type": "Fancy"})), &mut warnings);
        assert_eq!(normalizer.normalize("a"), "a");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Fancy"));
    }

    #[test]
    fn sequences_keep_the_order_of_a_compatibility_fold_and_a_regex() {
        // `NFKC` folds the ligature to `fi` first, so the regex then turns it into `FI`.
        let fold_first = clean(json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "NFKC"},
                {"type": "Replace", "pattern": {"Regex": "fi"}, "content": "FI"}
            ]
        }));
        assert_eq!(fold_first.normalize("\u{fb01}sh"), "FIsh");

        // Run the other way round the regex never sees `fi`, so the text only folds.
        let replace_first = clean(json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "Replace", "pattern": {"Regex": "fi"}, "content": "FI"},
                {"type": "NFKC"}
            ]
        }));
        assert_eq!(replace_first.normalize("\u{fb01}sh"), "fish");
    }
}
