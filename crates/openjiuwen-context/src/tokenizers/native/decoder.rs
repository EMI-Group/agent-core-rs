//! Turning token ids back into text.
//!
//! The common `HuggingFace` decoders are reproduced: byte-level (the reverse byte mapping), metaspace, word-piece, `BPE`, `CTC`, `ByteFallback`, `Strip`, `Fuse`, literal and regex `Replace`, the `NFC`/`NFD`/`NFKC`/`NFKD` normalizations (applied through `unicode-normalization`), `Sequence` chains and the plain case, which joins tokens with spaces.
//! A regex `Replace` compiles through the `regex` crate, so Rust's replacement syntax (`$1`, `${name}`) applies; a pattern that does not compile is recorded as a warning naming the pattern source and the `regex` error, and that step is left out.
//! A node this module cannot reproduce is also recorded as a warning and replaced by the plain join, because a decode stays useful without it; the plain join itself is the default only for an artifact that declares no decoder section at all.
//! Post-processor templates are deliberately ignored: they add framing tokens and never change a token count.

use regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use super::bytes;
use super::hf_json::{
    bool_of, describe_pattern, kind_of, literal_pattern, regex_pattern, str_of, usize_of,
};

/// The default metaspace replacement character.
const DEFAULT_METASPACE_REPLACEMENT: char = '\u{2581}';

/// The default word-end suffix of the original `BPE` decoder.
const DEFAULT_BPE_SUFFIX: &str = "</w>";

/// The default CTC pad token.
const DEFAULT_CTC_PAD: &str = "<pad>";

/// The default CTC word delimiter.
const DEFAULT_CTC_WORD_DELIMITER: &str = "|";

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
    /// Replace every match of a compiled regular expression.
    ReplacePattern {
        /// The compiled pattern.
        regex: Regex,
        /// The replacement, in Rust's `$1`/`${name}` syntax.
        content: String,
    },
    /// Drop up to `start` occurrences of a character from the head and up to `stop` from the tail of every token.
    ///
    /// `HuggingFace`'s `Strip` decoder strips one occurrence per budget unit and stops at the first mismatch, which is what this reproduces; a `content` that is not a single character is truncated to its first one.
    Strip {
        /// The character to strip.
        content: char,
        /// How many occurrences may be removed from the head.
        start: usize,
        /// How many occurrences may be removed from the tail.
        stop: usize,
    },
    /// Turn `<0xXX>` tokens back into the bytes they name.
    ///
    /// A run of byte tokens is decoded as UTF-8 with replacement characters for invalid sequences, so a byte-level vocabulary that was cut mid-codepoint decodes lossily rather than failing.
    /// Every other token ends the run and passes through unchanged, and as the last step of a chain the decoded tokens are joined with spaces, matching `HuggingFace`'s default decoder.
    ByteFallback,
    /// Undo the original `BPE` word-end suffix.
    ///
    /// Following `HuggingFace`, the suffix becomes a space inside every token except the last one, where it is removed, and the resulting chain is joined with spaces; a suffix that never appears therefore leaves the tokens space-joined.
    Bpe {
        /// The word-end suffix (`</w>` by default).
        suffix: String,
    },
    /// Collapse a CTC alignment: drop repeated and padded frames, then turn the word delimiter into a space.
    ///
    /// The repeated-frame and pad handling matches `HuggingFace` exactly; the cleanup pass is its punctuation/contraction fix-up and, like `HuggingFace`, the word delimiter is only rewritten when `cleanup` is on.
    Ctc {
        /// The frame that carries no symbol (`<pad>` by default).
        pad_token: String,
        /// The token that stands for a word boundary (`|` by default).
        word_delimiter_token: String,
        /// Whether to run the punctuation/contraction cleanup.
        cleanup: bool,
    },
    /// Unicode canonical composition (`NFC`) of every token.
    Nfc,
    /// Unicode canonical decomposition (`NFD`) of every token.
    Nfd,
    /// Unicode compatibility composition (`NFKC`) of every token.
    Nfkc,
    /// Unicode compatibility decomposition (`NFKD`) of every token.
    Nfkd,
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
            "Replace" => Self::replace(value, warnings),
            "Strip" => Self::Strip {
                content: str_of(value, "content")
                    .and_then(|text| text.chars().next())
                    .unwrap_or(' '),
                start: usize_of(value, "start", 0),
                stop: usize_of(value, "stop", 0),
            },
            "ByteFallback" => Self::ByteFallback,
            "BPEDecoder" => Self::Bpe {
                suffix: str_of(value, "suffix")
                    .unwrap_or(DEFAULT_BPE_SUFFIX)
                    .to_owned(),
            },
            "CTC" => Self::Ctc {
                pad_token: str_of(value, "pad_token")
                    .unwrap_or(DEFAULT_CTC_PAD)
                    .to_owned(),
                word_delimiter_token: str_of(value, "word_delimiter_token")
                    .unwrap_or(DEFAULT_CTC_WORD_DELIMITER)
                    .to_owned(),
                cleanup: bool_of(value, "cleanup", true),
            },
            "NFC" => Self::Nfc,
            "NFD" => Self::Nfd,
            "NFKC" => Self::Nfkc,
            "NFKD" => Self::Nfkd,
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

    /// Decodes a `Replace` node, which is literal for a `{"String": ...}` pattern and a regex otherwise.
    fn replace(value: &Value, warnings: &mut Vec<String>) -> Self {
        let content = str_of(value, "content").unwrap_or_default().to_owned();
        if let Some(pattern) = literal_pattern(value, "pattern") {
            return Self::Replace { pattern, content };
        }
        let Some(source) = regex_pattern(value, "pattern") else {
            warnings.push(format!(
                "decoder Replace declares {}; that step is left out and the tokens are joined without it",
                describe_pattern(value, "pattern")
            ));
            return Self::Plain;
        };
        match Regex::new(source) {
            Ok(regex) => Self::ReplacePattern { regex, content },
            Err(error) => {
                warnings.push(format!(
                    "decoder Replace regex `{source}` does not compile ({error}); that step is left out and the tokens are joined without it"
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
            Self::ReplacePattern { regex, content } => regex
                .replace_all(&tokens.join(" "), content.as_str())
                .into_owned(),
            Self::Strip { .. }
            | Self::ByteFallback
            | Self::Bpe { .. }
            | Self::Ctc { .. }
            | Self::Nfc
            | Self::Nfd
            | Self::Nfkc
            | Self::Nfkd => self.transform(tokens.to_vec()).join(" "),
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
            Self::ReplacePattern { regex, content } => tokens
                .into_iter()
                .map(|token| regex.replace_all(&token, content.as_str()).into_owned())
                .collect(),
            Self::Strip {
                content,
                start,
                stop,
            } => tokens
                .into_iter()
                .map(|token| strip_token(&token, *content, *start, *stop))
                .collect(),
            Self::ByteFallback => byte_fallback_chain(tokens),
            Self::Bpe { suffix } => bpe_chain(tokens, suffix),
            Self::Ctc {
                pad_token,
                word_delimiter_token,
                cleanup,
            } => ctc_chain(tokens, pad_token, word_delimiter_token, *cleanup),
            Self::Nfc => tokens
                .into_iter()
                .map(|token| token.nfc().collect())
                .collect(),
            Self::Nfd => tokens
                .into_iter()
                .map(|token| token.nfd().collect())
                .collect(),
            Self::Nfkc => tokens
                .into_iter()
                .map(|token| token.nfkc().collect())
                .collect(),
            Self::Nfkd => tokens
                .into_iter()
                .map(|token| token.nfkd().collect())
                .collect(),
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

/// Removes up to `start` leading and `stop` trailing occurrences of `content`.
fn strip_token(token: &str, content: char, start: usize, stop: usize) -> String {
    let mut current = token;
    for _ in 0..start {
        match current.strip_prefix(content) {
            Some(rest) => current = rest,
            None => break,
        }
    }
    for _ in 0..stop {
        match current.strip_suffix(content) {
            Some(rest) => current = rest,
            None => break,
        }
    }
    current.to_owned()
}

/// The byte a `<0xXX>` token names, when the token has that shape.
fn byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Replaces runs of byte tokens by the text they decode to, passing every other token through.
fn byte_fallback_chain(tokens: Vec<String>) -> Vec<String> {
    let mut chain = Vec::with_capacity(tokens.len());
    let mut pending: Vec<u8> = Vec::new();
    for token in tokens {
        if let Some(byte) = byte_token(&token) {
            pending.push(byte);
        } else {
            if !pending.is_empty() {
                chain.push(String::from_utf8_lossy(&pending).into_owned());
                pending.clear();
            }
            chain.push(token);
        }
    }
    if !pending.is_empty() {
        chain.push(String::from_utf8_lossy(&pending).into_owned());
    }
    chain
}

/// Replaces the word-end suffix with a space, removing it from the last token.
fn bpe_chain(tokens: Vec<String>, suffix: &str) -> Vec<String> {
    if suffix.is_empty() {
        return tokens;
    }
    let last = tokens.len().saturating_sub(1);
    tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| token.replace(suffix, if index == last { "" } else { " " }))
        .collect()
}

/// Collapses repeated and padded CTC frames and rewrites the word delimiter.
fn ctc_chain(
    tokens: Vec<String>,
    pad_token: &str,
    word_delimiter_token: &str,
    cleanup: bool,
) -> Vec<String> {
    let mut chain = Vec::with_capacity(tokens.len());
    let mut previous: Option<String> = None;
    for token in tokens {
        if previous.as_deref() == Some(token.as_str()) {
            continue;
        }
        previous = Some(token.clone());
        let mut replaced = token.replace(pad_token, "");
        if cleanup {
            replaced = wordpiece_cleanup(&replaced).replace(word_delimiter_token, " ");
        }
        if !replaced.is_empty() {
            chain.push(replaced);
        }
    }
    chain
}

/// `HuggingFace`'s punctuation and contraction cleanup, shared by the `CTC` decoder.
fn wordpiece_cleanup(text: &str) -> String {
    text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Decodes one decoder, asserting that it produced no warning.
    fn clean(node: Value) -> Decoder {
        let mut warnings = Vec::new();
        let decoder = Decoder::from_json(Some(&node), &mut warnings);
        assert_eq!(warnings, Vec::<String>::new(), "{node}");
        decoder
    }

    fn tokens(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|token| (*token).to_owned()).collect()
    }

    #[test]
    fn plain_join_inserts_spaces() {
        assert_eq!(Decoder::Plain.decode(&tokens(&["a", "b"])), "a b");
        // A missing decoder section is the only way to get the plain case by default.
        assert!(matches!(
            Decoder::from_json(None, &mut Vec::new()),
            Decoder::Plain
        ));
    }

    #[test]
    fn word_piece_glues_continuations() {
        let decoder = clean(json!({"type": "WordPiece", "prefix": "##"}));
        assert_eq!(decoder.decode(&tokens(&["hello", "##ing"])), "helloing");
    }

    #[test]
    fn byte_level_decodes_multibyte_text() {
        let decoder = clean(json!({"type": "ByteLevel"}));
        let mapped = bytes::encode("héllo");
        let entries: Vec<String> = mapped.chars().map(String::from).collect();
        assert_eq!(decoder.decode(&entries), "héllo");
    }

    #[test]
    fn metaspace_turns_the_marker_back_into_a_space() {
        let decoder = clean(
            json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always"}),
        );
        assert_eq!(
            decoder.decode(&tokens(&["\u{2581}hello", "\u{2581}world"])),
            "hello world"
        );
    }

    #[test]
    fn sequences_apply_every_step() {
        let decoder = clean(
            json!({"type": "Sequence", "decoders": [{"type": "ByteLevel"}, {"type": "Fuse"}]}),
        );
        let mapped = bytes::encode("hi");
        let entries: Vec<String> = mapped.chars().map(String::from).collect();
        assert_eq!(decoder.decode(&entries), "hi");
    }

    #[test]
    fn a_literal_replace_only_rewrites_the_literal() {
        let decoder = clean(json!({"type": "Replace", "pattern": {"String": "."}, "content": "_"}));
        assert!(matches!(decoder, Decoder::Replace { .. }));
        assert_eq!(decoder.decode(&tokens(&["a.bc"])), "a_bc");
        assert_eq!(decoder.decode(&tokens(&["a.", "bc"])), "a_ bc");
    }

    #[test]
    fn a_regex_replace_uses_rust_replacement_syntax() {
        let spaces =
            clean(json!({"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": "_"}));
        assert!(matches!(spaces, Decoder::ReplacePattern { .. }));
        assert_eq!(spaces.decode(&tokens(&["a", "b"])), "a_b");

        let groups = clean(json!({
            "type": "Replace",
            "pattern": {"Regex": "(\\w+) (\\w+)"},
            "content": "$2 $1"
        }));
        assert_eq!(groups.decode(&tokens(&["user", "host"])), "host user");

        // A bare string pattern is the legacy regex shape.
        let bare = clean(json!({"type": "Replace", "pattern": "a+", "content": "-"}));
        assert_eq!(bare.decode(&tokens(&["caaat"])), "c-t");
    }

    #[test]
    fn a_regex_that_does_not_compile_warns_and_is_left_out() {
        let mut warnings = Vec::new();
        let decoder = Decoder::from_json(
            Some(&json!({"type": "Replace", "pattern": {"Regex": "(unclosed"}, "content": "x"})),
            &mut warnings,
        );
        assert!(matches!(decoder, Decoder::Plain));
        assert_eq!(decoder.decode(&tokens(&["a", "b"])), "a b");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("(unclosed"), "{}", warnings[0]);
        assert!(warnings[0].contains("does not compile"), "{}", warnings[0]);
    }

    #[test]
    fn strip_removes_bounded_occurrences_from_each_token() {
        let decoder = clean(json!({"type": "Strip", "content": "\u{2581}", "start": 1, "stop": 0}));
        assert_eq!(
            decoder.transform(tokens(&["\u{2581}he", "\u{2581}llo", "\u{2581}\u{2581}x"])),
            tokens(&["he", "llo", "\u{2581}x"])
        );
        assert_eq!(
            decoder.decode(&tokens(&["\u{2581}he", "\u{2581}llo"])),
            "he llo"
        );

        let both = clean(json!({"type": "Strip", "content": "_", "start": 2, "stop": 1}));
        assert_eq!(both.transform(tokens(&["__ab_"])), tokens(&["ab"]));
        // A missing budget strips nothing.
        let none = clean(json!({"type": "Strip", "content": "_"}));
        assert_eq!(none.transform(tokens(&["_ab_"])), tokens(&["_ab_"]));
    }

    #[test]
    fn byte_fallback_rebuilds_bytes_through_the_alphabet() {
        let decoder = clean(json!({"type": "ByteFallback"}));
        let text = "héllo";
        let entries: Vec<String> = text.bytes().map(|byte| format!("<0x{byte:02x}>")).collect();
        assert_eq!(decoder.transform(entries.clone()), tokens(&[text]));
        // As the last step the decoded chain is joined with spaces.
        assert_eq!(decoder.decode(&entries), text);

        // A non-byte token ends the run and passes through.
        assert_eq!(
            decoder.transform(tokens(&["<0x68>", "<0x69>", "!"])),
            tokens(&["hi", "!"])
        );
        assert_eq!(decoder.decode(&tokens(&["<0x68>", "<0x69>", "!"])), "hi !");

        // Bytes that do not form UTF-8 decode lossily instead of failing.
        assert_eq!(
            decoder.transform(tokens(&["<0x61>", "<0xff>"])),
            tokens(&["a\u{fffd}"])
        );
        // A malformed byte token is ordinary text.
        assert_eq!(decoder.transform(tokens(&["<0xzz>"])), tokens(&["<0xzz>"]));
        assert_eq!(decoder.transform(tokens(&["<0x6>"])), tokens(&["<0x6>"]));
    }

    #[test]
    fn bpe_decoder_undoes_the_word_end_suffix() {
        let decoder = clean(json!({"type": "BPEDecoder"}));
        // `HuggingFace` turns the suffix into a space inside the chain and drops it from the last token,
        // so the chain is then joined with spaces.
        assert_eq!(
            decoder.transform(tokens(&["Hello</w>", "World</w>"])),
            tokens(&["Hello ", "World"])
        );
        assert_eq!(
            decoder.decode(&tokens(&["Hello</w>", "World</w>"])),
            "Hello  World"
        );
        // A lone token keeps its text: it is the last one, so the suffix is removed outright.
        assert_eq!(decoder.decode(&tokens(&["Hello</w>"])), "Hello");
        // A word split over two pieces decodes with the space the suffix stood for.
        assert_eq!(decoder.decode(&tokens(&["Hel", "lo</w>"])), "Hel lo");
        // A declared suffix is used instead of the default.
        let custom = clean(json!({"type": "BPEDecoder", "suffix": "_"}));
        assert_eq!(
            custom.transform(tokens(&["a_", "b_"])),
            tokens(&["a ", "b"])
        );
    }

    #[test]
    fn ctc_collapses_repeats_and_pads() {
        let decoder = clean(json!({"type": "CTC"}));
        let frames = tokens(&[
            "<pad>", "<pad>", "h", "e", "e", "l", "l", "<pad>", "l", "o", "o", "o", "<pad>",
        ]);
        assert_eq!(
            decoder.transform(frames.clone()),
            tokens(&["h", "e", "l", "l", "o"])
        );
        assert_eq!(decoder.decode(&frames), "h e l l o");

        // The delimiter becomes a space only under `cleanup`.
        let delimited = tokens(&["h", "|", "i"]);
        assert_eq!(
            decoder.transform(delimited.clone()),
            tokens(&["h", " ", "i"])
        );
        let raw = clean(json!({"type": "CTC", "cleanup": false}));
        assert_eq!(raw.transform(delimited), tokens(&["h", "|", "i"]));
    }

    #[test]
    fn unicode_decoders_rewrite_every_token() {
        assert_eq!(
            clean(json!({"type": "NFC"})).decode(&tokens(&["e", "\u{0301}"])),
            "e \u{0301}"
        );
        assert_eq!(
            clean(json!({"type": "NFD"})).decode(&tokens(&["\u{e9}"])),
            "e\u{0301}"
        );
        assert_eq!(
            clean(json!({"type": "NFKC"})).decode(&tokens(&["\u{ff21}"])),
            "A"
        );
        assert_eq!(
            clean(json!({"type": "NFKD"})).decode(&tokens(&["\u{fb01}"])),
            "fi"
        );
        // A composed token is rebuilt by `NFC` and split by `NFD`.
        assert_eq!(
            clean(json!({"type": "NFC"})).transform(tokens(&["e\u{0301}"])),
            tokens(&["\u{e9}"])
        );
    }

    #[test]
    fn an_unimplemented_decoder_only_warns() {
        let mut warnings = Vec::new();
        let decoder = Decoder::from_json(Some(&json!({"type": "Fancy"})), &mut warnings);
        assert!(matches!(decoder, Decoder::Plain));
        assert_eq!(decoder.decode(&tokens(&["a", "b"])), "a b");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Fancy"));
    }
}
