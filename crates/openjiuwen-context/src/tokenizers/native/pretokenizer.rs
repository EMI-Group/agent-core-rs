//! Pre-tokenization: how a normalized text is split before the model sees it.
//!
//! The `BERT`, whitespace, punctuation, digit, byte-level, metaspace, `Split` (literal or regex), `Replace`, `CharDelimiterSplit` and Unicode-wrapper nodes are reproduced, plus the `Sequence` node that chains them.
//! Byte-level pre-tokenization runs the GPT-2 split over the raw text and byte-maps each resulting piece afterwards, which is `HuggingFace`'s order: `hi there!` becomes `hi`, `Ġthere`, `!`.
//! A `Split` or `Replace` node's `{"Regex": ...}` pattern is compiled with `regex`; a pattern that does not compile is recorded as a warning naming the pattern and the error and degrades to whitespace splitting (or to no pre-tokenization), so a token count stays useful instead of the load failing.
//! A pipeline node this loader does not implement is recorded as a warning naming the node and leaves the text unsplit.

use std::sync::{Arc, OnceLock};

use regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use super::hf_json::{bool_of, kind_of, literal_pattern, pattern_is_regex, str_of};

mod gpt2;

/// The default metaspace replacement character.
const DEFAULT_METASPACE_REPLACEMENT: char = '\u{2581}';

/// When a metaspace pre-tokenizer prefixes the replacement character.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Prepend {
    /// Prefix every segment.
    Always,
    /// Prefix only the first segment of the input.
    First,
    /// Never prefix.
    Never,
}

/// What happens to a matched delimiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplitBehavior {
    /// The delimiter is dropped.
    Removed,
    /// The delimiter becomes its own piece.
    Isolated,
    /// The delimiter joins the piece before it.
    MergedWithPrevious,
    /// The delimiter joins the piece after it.
    MergedWithNext,
    /// Consecutive delimiters merge into one piece.
    Contiguous,
}

/// The Unicode normalization a wrapper pre-tokenizer applies before splitting on whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnicodeForm {
    /// Canonical composition (`NFC`).
    Nfc,
    /// Canonical decomposition (`NFD`).
    Nfd,
    /// Compatibility composition (`NFKC`).
    Nfkc,
    /// Compatibility decomposition (`NFKD`).
    Nfkd,
}

impl UnicodeForm {
    /// Rewrites `text` in this normalization form.
    fn apply(self, text: &str) -> String {
        match self {
            Self::Nfc => text.nfc().collect(),
            Self::Nfd => text.nfd().collect(),
            Self::Nfkc => text.nfkc().collect(),
            Self::Nfkd => text.nfkd().collect(),
        }
    }
}

/// A pre-tokenization step.
#[derive(Debug, Clone, Default)]
pub(crate) enum Pretokenizer {
    /// The text is a single piece.
    #[default]
    Identity,
    /// Steps applied in order.
    Sequence(Vec<Pretokenizer>),
    /// `BERT`: split on whitespace and punctuation.
    Bert,
    /// Split on whitespace.
    Whitespace,
    /// Split with the GPT-2 pattern and apply the byte mapping.
    ByteLevel {
        /// Whether the first piece is given a leading space.
        add_prefix_space: bool,
        /// Whether the GPT-2 split pattern is applied.
        use_regex: bool,
    },
    /// Split out punctuation characters.
    Punctuation,
    /// Split out digit runs.
    Digits {
        /// Whether every digit becomes its own piece.
        individual_digits: bool,
    },
    /// Replace spaces with a marker and split on it.
    Metaspace {
        /// The marker that replaces a space.
        replacement: char,
        /// When the marker is prefixed.
        prepend: Prepend,
        /// Whether pieces are cut at each marker.
        split: bool,
    },
    /// Split on a literal pattern.
    SplitLiteral {
        /// The literal delimiter.
        pattern: String,
        /// What happens to the delimiter.
        behavior: SplitBehavior,
        /// Whether the pattern marks the pieces instead of the delimiters.
        invert: bool,
    },
    /// Split on a regex pattern.
    SplitRegex {
        /// The compiled pattern.
        regex: Arc<Regex>,
        /// What happens to the delimiter.
        behavior: SplitBehavior,
        /// Whether the pattern marks the pieces instead of the delimiters.
        invert: bool,
    },
    /// Replace every regex match; the whole rewritten text stays one piece.
    Replace {
        /// The compiled pattern.
        regex: Arc<Regex>,
        /// What every match is replaced with.
        content: String,
    },
    /// Split on a delimiter character, dropping it.
    CharDelimiterSplit {
        /// The delimiter.
        delimiter: char,
    },
    /// Normalize to a Unicode form, then split on whitespace.
    UnicodeNormalize {
        /// The normalization form.
        form: UnicodeForm,
    },
}

impl Pretokenizer {
    /// Decodes a pre-tokenizer section, recording anything it cannot reproduce.
    pub(crate) fn from_json(value: Option<&Value>, warnings: &mut Vec<String>) -> Self {
        let Some(value) = value else {
            return Self::Identity;
        };
        match value {
            Value::Null => Self::Identity,
            Value::Object(_) => Self::from_object(value, warnings),
            _ => {
                warnings.push(
                    "pre_tokenizer is not an object; the text is not pre-tokenized".to_owned(),
                );
                Self::Identity
            }
        }
    }

    /// Decodes one object node.
    fn from_object(value: &Value, warnings: &mut Vec<String>) -> Self {
        let Some(kind) = kind_of(value) else {
            warnings
                .push("pre_tokenizer declares no type; the text is not pre-tokenized".to_owned());
            return Self::Identity;
        };
        match kind {
            "Sequence" => {
                let stages = value
                    .get("pretokenizers")
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
            "BertPreTokenizer" => Self::Bert,
            "Whitespace" | "WhitespaceSplit" => Self::Whitespace,
            "ByteLevel" => Self::ByteLevel {
                add_prefix_space: bool_of(value, "add_prefix_space", true),
                use_regex: bool_of(value, "use_regex", true),
            },
            "Punctuation" => Self::Punctuation,
            "Digits" => Self::Digits {
                individual_digits: bool_of(value, "individual_digits", false),
            },
            "Metaspace" => Self::Metaspace {
                replacement: str_of(value, "replacement")
                    .and_then(|text| text.chars().next())
                    .unwrap_or(DEFAULT_METASPACE_REPLACEMENT),
                prepend: match str_of(value, "prepend_scheme")
                    .unwrap_or("always")
                    .to_lowercase()
                    .as_str()
                {
                    "never" => Prepend::Never,
                    "first" => Prepend::First,
                    _ => Prepend::Always,
                },
                split: bool_of(value, "split", true),
            },
            "Split" => Self::split_node(value, warnings),
            "Replace" => Self::replace_node(value, warnings),
            "CharDelimiterSplit" => Self::char_delimiter_node(value, warnings),
            "NFC" => Self::UnicodeNormalize {
                form: UnicodeForm::Nfc,
            },
            "NFD" => Self::UnicodeNormalize {
                form: UnicodeForm::Nfd,
            },
            "NFKC" => Self::UnicodeNormalize {
                form: UnicodeForm::Nfkc,
            },
            "NFKD" => Self::UnicodeNormalize {
                form: UnicodeForm::Nfkd,
            },
            other => {
                warnings.push(format!(
                    "pre_tokenizer `{other}` is not implemented; the text is not pre-tokenized"
                ));
                Self::Identity
            }
        }
    }

    /// Decodes a `Split` node, which carries either a literal or a regex pattern.
    fn split_node(value: &Value, warnings: &mut Vec<String>) -> Self {
        let behavior = split_behavior(value);
        let invert = bool_of(value, "invert", false);
        if let Some(source) = regex_pattern(value, "pattern") {
            return match Regex::new(&source) {
                Ok(regex) => Self::SplitRegex {
                    regex: Arc::new(regex),
                    behavior,
                    invert,
                },
                Err(error) => {
                    warnings.push(format!(
                        "pre_tokenizer Split pattern `{source}` is not a usable regex ({error}); falling back to whitespace splitting"
                    ));
                    Self::Whitespace
                }
            };
        }
        let Some(pattern) = literal_pattern(value, "pattern") else {
            warnings.push(
                "pre_tokenizer Split declares neither a literal nor a regex pattern; the text is not pre-tokenized"
                    .to_owned(),
            );
            return Self::Identity;
        };
        Self::SplitLiteral {
            pattern,
            behavior,
            invert,
        }
    }

    /// Decodes a `Replace` node, whose pattern must be a regex.
    fn replace_node(value: &Value, warnings: &mut Vec<String>) -> Self {
        let Some(source) = regex_pattern(value, "pattern") else {
            warnings.push(
                "pre_tokenizer Replace declares no regex pattern; the text is not pre-tokenized"
                    .to_owned(),
            );
            return Self::Identity;
        };
        match Regex::new(&source) {
            Ok(regex) => Self::Replace {
                regex: Arc::new(regex),
                // An absent content deletes every match, which is a valid `Replace`.
                content: str_of(value, "content").unwrap_or("").to_owned(),
            },
            Err(error) => {
                warnings.push(format!(
                    "pre_tokenizer Replace pattern `{source}` is not a usable regex ({error}); the text is not pre-tokenized"
                ));
                Self::Identity
            }
        }
    }

    /// Decodes a `CharDelimiterSplit` node, whose `delimiter` is a single character.
    fn char_delimiter_node(value: &Value, warnings: &mut Vec<String>) -> Self {
        let Some(delimiter) = str_of(value, "delimiter") else {
            warnings.push(
                "pre_tokenizer CharDelimiterSplit declares no delimiter; the text is not pre-tokenized"
                    .to_owned(),
            );
            return Self::Identity;
        };
        let mut chars = delimiter.chars();
        let Some(first) = chars.next() else {
            warnings.push(
                "pre_tokenizer CharDelimiterSplit declares an empty delimiter; the text is not pre-tokenized"
                    .to_owned(),
            );
            return Self::Identity;
        };
        if chars.next().is_some() {
            warnings.push(format!(
                "pre_tokenizer CharDelimiterSplit delimiter `{delimiter}` is not a single character; splitting on `{first}`"
            ));
        }
        Self::CharDelimiterSplit { delimiter: first }
    }

    /// Splits `text` into the pieces the model will encode.
    pub(crate) fn apply(&self, text: &str) -> Vec<String> {
        self.split(text, true)
    }

    /// Splits one segment, remembering whether it is the first of the input.
    fn split(&self, segment: &str, first: bool) -> Vec<String> {
        if segment.is_empty() {
            return Vec::new();
        }
        match self {
            Self::Identity => vec![segment.to_owned()],
            Self::Sequence(stages) => {
                let mut current = vec![segment.to_owned()];
                for stage in stages {
                    let mut next = Vec::with_capacity(current.len());
                    for (index, piece) in current.iter().enumerate() {
                        next.extend(stage.split(piece, first && index == 0));
                    }
                    current = next;
                }
                current
            }
            Self::Bert => bert_split(segment),
            Self::Whitespace => segment.split_whitespace().map(str::to_owned).collect(),
            Self::ByteLevel {
                add_prefix_space,
                use_regex,
            } => gpt2::byte_level_split(segment, *add_prefix_space, *use_regex),
            Self::Punctuation => punctuation_split(segment),
            Self::Digits { individual_digits } => digits_split(segment, *individual_digits),
            Self::Metaspace {
                replacement,
                prepend,
                split,
            } => metaspace_split(segment, *replacement, *prepend, *split, first),
            Self::SplitLiteral {
                pattern,
                behavior,
                invert,
            } => split_literal(segment, pattern, *behavior, *invert),
            Self::SplitRegex {
                regex,
                behavior,
                invert,
            } => {
                let matches: Vec<(usize, usize)> = regex
                    .find_iter(segment)
                    .map(|found| (found.start(), found.end()))
                    .collect();
                assemble_split(segment, &matches, *behavior, *invert)
            }
            Self::Replace { regex, content } => {
                vec![regex.replace_all(segment, content.as_str()).into_owned()]
            }
            Self::CharDelimiterSplit { delimiter } => segment
                .split(*delimiter)
                .filter(|piece| !piece.is_empty())
                .map(str::to_owned)
                .collect(),
            Self::UnicodeNormalize { form } => form
                .apply(segment)
                .split_whitespace()
                .map(str::to_owned)
                .collect(),
        }
    }
}

/// The `behavior` member of a `Split` node, defaulting to `Isolated`.
fn split_behavior(value: &Value) -> SplitBehavior {
    match str_of(value, "behavior").unwrap_or("Isolated") {
        "Removed" => SplitBehavior::Removed,
        "MergedWithPrevious" => SplitBehavior::MergedWithPrevious,
        "MergedWithNext" => SplitBehavior::MergedWithNext,
        "Contiguous" => SplitBehavior::Contiguous,
        _ => SplitBehavior::Isolated,
    }
}

/// The regex source of a `{"Regex": "..."}` pattern.
fn regex_pattern(value: &Value, key: &str) -> Option<String> {
    if !pattern_is_regex(value, key) {
        return None;
    }
    value
        .get(key)
        .and_then(|pattern| pattern.get("Regex"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// Applies the metaspace rewrite and split.
fn metaspace_split(
    segment: &str,
    replacement: char,
    prepend: Prepend,
    split: bool,
    first: bool,
) -> Vec<String> {
    let mut rewritten: String = segment
        .chars()
        .map(|ch| if ch == ' ' { replacement } else { ch })
        .collect();
    let should_prepend = match prepend {
        Prepend::Always => true,
        Prepend::First => first,
        Prepend::Never => false,
    };
    if should_prepend && !rewritten.starts_with(replacement) {
        rewritten.insert(0, replacement);
    }
    if split {
        split_before_character(&rewritten, replacement)
    } else {
        vec![rewritten]
    }
}

/// Cuts `text` before every `delimiter`, keeping the delimiter with the piece that follows it.
fn split_before_character(text: &str, delimiter: char) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch == delimiter && !current.is_empty() {
            pieces.push(std::mem::take(&mut current));
        }
        current.push(ch);
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

/// `BERT` pre-tokenization: whitespace separates, punctuation is its own piece.
fn bert_split(segment: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for ch in segment.chars() {
        if ch.is_whitespace() {
            push_piece(&mut pieces, std::mem::take(&mut current));
        } else if is_punctuation(ch) {
            push_piece(&mut pieces, std::mem::take(&mut current));
            push_piece(&mut pieces, ch.to_string());
        } else {
            current.push(ch);
        }
    }
    push_piece(&mut pieces, current);
    pieces
}

/// Punctuation pre-tokenization: punctuation is its own piece, other characters (whitespace included) stay together.
fn punctuation_split(segment: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for ch in segment.chars() {
        if is_punctuation(ch) {
            push_piece(&mut pieces, std::mem::take(&mut current));
            push_piece(&mut pieces, ch.to_string());
        } else {
            current.push(ch);
        }
    }
    push_piece(&mut pieces, current);
    pieces
}

/// Digit pre-tokenization: digit runs are separated from the rest.
fn digits_split(segment: &str, individual_digits: bool) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut current_is_digit = false;
    for ch in segment.chars() {
        let digit = ch.is_numeric();
        if digit && individual_digits {
            push_piece(&mut pieces, std::mem::take(&mut current));
            push_piece(&mut pieces, ch.to_string());
            current_is_digit = false;
            continue;
        }
        if !current.is_empty() && digit != current_is_digit {
            push_piece(&mut pieces, std::mem::take(&mut current));
        }
        current.push(ch);
        current_is_digit = digit;
    }
    push_piece(&mut pieces, current);
    pieces
}

/// Literal `Split`: the delimiter is handled according to `behavior`.
fn split_literal(
    segment: &str,
    pattern: &str,
    behavior: SplitBehavior,
    invert: bool,
) -> Vec<String> {
    if pattern.is_empty() {
        return vec![segment.to_owned()];
    }
    let matches: Vec<(usize, usize)> = segment
        .match_indices(pattern)
        .map(|(start, found)| (start, start + found.len()))
        .collect();
    assemble_split(segment, &matches, behavior, invert)
}

/// Turns byte ranges into pieces according to `behavior`, shared by the literal and regex splits.
///
/// With `invert` set the matches are the pieces and everything between them is the delimiter.
fn assemble_split(
    text: &str,
    matches: &[(usize, usize)],
    behavior: SplitBehavior,
    invert: bool,
) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut cursor = 0;
    for &(start, end) in matches {
        if invert {
            push_piece(&mut pieces, text[start..end].to_owned());
            continue;
        }
        current.push_str(&text[cursor..start]);
        let delimiter = &text[start..end];
        match behavior {
            SplitBehavior::Removed => push_piece(&mut pieces, std::mem::take(&mut current)),
            SplitBehavior::Isolated => {
                push_piece(&mut pieces, std::mem::take(&mut current));
                push_piece(&mut pieces, delimiter.to_owned());
            }
            SplitBehavior::MergedWithPrevious => {
                current.push_str(delimiter);
                push_piece(&mut pieces, std::mem::take(&mut current));
            }
            SplitBehavior::MergedWithNext => {
                push_piece(&mut pieces, std::mem::take(&mut current));
                current.push_str(delimiter);
            }
            SplitBehavior::Contiguous => current.push_str(delimiter),
        }
        cursor = end;
    }
    if invert {
        return pieces;
    }
    current.push_str(&text[cursor..]);
    push_piece(&mut pieces, current);
    pieces
}

/// Pushes `piece` when it is not empty.
fn push_piece(pieces: &mut Vec<String>, piece: String) {
    if !piece.is_empty() {
        pieces.push(piece);
    }
}

/// Whether a character counts as punctuation.
///
/// ASCII punctuation is the printable, non-alphanumeric, non-space ASCII range; a non-ASCII character counts when it is in the Unicode `P` category, which `\p{P}` reports.
fn is_punctuation(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_punctuation();
    }
    match punctuation_pattern() {
        Some(pattern) => {
            let mut buffer = [0_u8; 4];
            pattern.is_match(ch.encode_utf8(&mut buffer))
        }
        None => !ch.is_alphanumeric() && !ch.is_whitespace() && !ch.is_control(),
    }
}

/// The Unicode punctuation category, compiled once.
fn punctuation_pattern() -> Option<&'static Regex> {
    static PUNCTUATION: OnceLock<Option<Regex>> = OnceLock::new();
    PUNCTUATION
        .get_or_init(|| Regex::new(r"\p{P}").ok())
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Decodes a pre-tokenizer node and its warnings.
    fn decode(value: Value) -> (Pretokenizer, Vec<String>) {
        let mut warnings = Vec::new();
        let pretokenizer = Pretokenizer::from_json(Some(&value), &mut warnings);
        (pretokenizer, warnings)
    }

    /// Decodes a `Split` node with a regex pattern.
    fn regex_split(pattern: &str, behavior: &str, invert: bool) -> Pretokenizer {
        let (pretokenizer, warnings) = decode(json!({
            "type": "Split",
            "pattern": {"Regex": pattern},
            "behavior": behavior,
            "invert": invert
        }));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        pretokenizer
    }

    #[test]
    fn bert_splits_punctuation_and_whitespace() {
        assert_eq!(
            bert_split("Hello, world!"),
            vec!["Hello", ",", "world", "!"]
        );
    }

    #[test]
    fn whitespace_split_drops_whitespace() {
        let (pretokenizer, warnings) = decode(json!({"type": "Whitespace"}));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(pretokenizer.apply(" a  b\tc "), vec!["a", "b", "c"]);
    }

    #[test]
    fn metaspace_prefixes_and_splits() {
        let (pretokenizer, warnings) = decode(json!({
            "type": "Metaspace",
            "replacement": "\u{2581}",
            "prepend_scheme": "always",
            "split": true
        }));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(
            pretokenizer.apply("hello world"),
            vec!["\u{2581}hello", "\u{2581}world"]
        );
    }

    #[test]
    fn byte_level_splits_before_it_maps() {
        let (pretokenizer, warnings) = decode(json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "use_regex": true
        }));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(
            pretokenizer.apply("hi there!"),
            vec!["hi", "\u{0120}there", "!"]
        );
        assert_eq!(
            pretokenizer.apply("Hello my friend, how is your day going?"),
            vec![
                "Hello",
                "\u{0120}my",
                "\u{0120}friend",
                ",",
                "\u{0120}how",
                "\u{0120}is",
                "\u{0120}your",
                "\u{0120}day",
                "\u{0120}going",
                "?"
            ]
        );
    }

    #[test]
    fn regex_splits_honour_every_behavior() {
        assert_eq!(
            regex_split("\\s+", "Removed", false).apply("a  b"),
            vec!["a", "b"]
        );
        assert_eq!(
            regex_split("\\s+", "Isolated", false).apply("a  b"),
            vec!["a", "  ", "b"]
        );
        assert_eq!(
            regex_split("\\s+", "MergedWithPrevious", false).apply("a  b"),
            vec!["a  ", "b"]
        );
        assert_eq!(
            regex_split("\\s+", "MergedWithNext", false).apply("a  b"),
            vec!["a", "  b"]
        );
        assert_eq!(
            regex_split("\\s+", "Contiguous", false).apply("a  b"),
            vec!["a  b"]
        );
    }

    #[test]
    fn regex_split_inverts_to_the_matches() {
        assert_eq!(
            regex_split("\\w+", "Removed", true).apply("a b"),
            vec!["a", "b"]
        );
        assert_eq!(
            regex_split("[0-9]+", "Isolated", true).apply("a12b34"),
            vec!["12", "34"]
        );
    }

    #[test]
    fn a_regex_that_does_not_compile_warns_with_the_pattern_and_the_error() {
        let (pretokenizer, warnings) = decode(json!({
            "type": "Split",
            "pattern": {"Regex": "(unclosed"},
            "behavior": "Isolated"
        }));
        assert!(matches!(pretokenizer, Pretokenizer::Whitespace));
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("(unclosed"),
            "the warning must name the pattern: {}",
            warnings[0]
        );
        assert!(
            warnings[0].to_lowercase().contains("regex"),
            "the warning must carry the regex error: {}",
            warnings[0]
        );
        assert_eq!(pretokenizer.apply("a  b"), vec!["a", "b"]);
    }

    #[test]
    fn replace_rewrites_every_match_into_one_piece() {
        let (pretokenizer, warnings) = decode(json!({
            "type": "Replace",
            "pattern": {"Regex": "\\s+"},
            "content": "_"
        }));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(pretokenizer.apply("a  b"), vec!["a_b"]);
    }

    #[test]
    fn replace_can_delete_matches_and_warns_on_a_bad_pattern() {
        let (pretokenizer, _warnings) = decode(json!({
            "type": "Replace",
            "pattern": {"Regex": "[aeiou]"}
        }));
        assert_eq!(pretokenizer.apply("hello"), vec!["hll"]);

        let (identity, warnings) = decode(json!({
            "type": "Replace",
            "pattern": {"Regex": "*"},
            "content": "_"
        }));
        assert!(matches!(identity, Pretokenizer::Identity));
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains('*'));
    }

    #[test]
    fn char_delimiter_split_removes_the_delimiter() {
        let (pretokenizer, warnings) = decode(json!({
            "type": "CharDelimiterSplit",
            "delimiter": "-"
        }));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(pretokenizer.apply("a-b--c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn char_delimiter_split_warns_on_a_missing_or_multi_character_delimiter() {
        let (identity, warnings) = decode(json!({"type": "CharDelimiterSplit"}));
        assert!(matches!(identity, Pretokenizer::Identity));
        assert_eq!(warnings.len(), 1);

        let (identity, warnings) = decode(json!({"type": "CharDelimiterSplit", "delimiter": ""}));
        assert!(matches!(identity, Pretokenizer::Identity));
        assert_eq!(warnings.len(), 1);

        let (pretokenizer, warnings) = decode(json!({
            "type": "CharDelimiterSplit",
            "delimiter": "--"
        }));
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("--"));
        assert_eq!(pretokenizer.apply("a-b-c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn unicode_wrapper_nodes_normalize_and_split_on_whitespace() {
        let (pretokenizer, warnings) = decode(json!({"type": "NFC"}));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(pretokenizer.apply("e\u{301} frank"), vec!["é", "frank"]);

        let (pretokenizer, _warnings) = decode(json!({"type": "NFD"}));
        assert_eq!(pretokenizer.apply("é frank"), vec!["e\u{301}", "frank"]);

        let (pretokenizer, _warnings) = decode(json!({"type": "NFKC"}));
        assert_eq!(pretokenizer.apply("ﬁ x"), vec!["fi", "x"]);

        let (pretokenizer, _warnings) = decode(json!({"type": "NFKD"}));
        assert_eq!(pretokenizer.apply("①"), vec!["1"]);
    }

    #[test]
    fn punctuation_follows_the_ascii_and_unicode_categories() {
        // ASCII symbols are punctuation.
        assert_eq!(
            punctuation_split("a$b+c=d"),
            vec!["a", "$", "b", "+", "c", "=", "d"]
        );
        // `©` is a symbol (So) rather than punctuation, while the em dash is `Pd`.
        assert_eq!(punctuation_split("a©b—c"), vec!["a©b", "—", "c"]);
        assert!(!is_punctuation('中'));
        assert!(is_punctuation('。'));
        assert!(is_punctuation('!'));
        assert!(!is_punctuation(' '));
        assert!(!is_punctuation('a'));
    }

    #[test]
    fn literal_splits_honour_the_behavior() {
        let (pretokenizer, warnings) = decode(json!({
            "type": "Split",
            "pattern": {"String": "-"},
            "behavior": "MergedWithPrevious"
        }));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(pretokenizer.apply("a-b-c"), vec!["a-", "b-", "c"]);
    }

    #[test]
    fn a_split_without_a_pattern_is_identity_with_a_warning() {
        let (pretokenizer, warnings) = decode(json!({"type": "Split", "behavior": "Isolated"}));
        assert!(matches!(pretokenizer, Pretokenizer::Identity));
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn unknown_nodes_warn_and_leave_the_text_unsplit() {
        let (pretokenizer, warnings) = decode(json!({"type": "Fancy"}));
        assert!(matches!(pretokenizer, Pretokenizer::Identity));
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Fancy"));
    }

    #[test]
    fn sequences_chain_splitters() {
        let (pretokenizer, warnings) = decode(json!({
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Whitespace"},
                {"type": "Digits"}
            ]
        }));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(pretokenizer.apply("ab12 cd"), vec!["ab", "12", "cd"]);
    }

    #[test]
    fn sequences_chain_a_replace_into_a_regex_split() {
        let (pretokenizer, warnings) = decode(json!({
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": "_"},
                {"type": "Split", "pattern": {"Regex": "_"}, "behavior": "Removed"}
            ]
        }));
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(pretokenizer.apply("ab cd"), vec!["ab", "cd"]);
    }
}
