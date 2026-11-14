//! Pre-tokenization: how a normalized text is split before the model sees it.
//!
//! The `BERT`, whitespace, punctuation, digit, byte-level, metaspace and literal-split pre-tokenizers are reproduced, plus the `Sequence` node that chains them.
//! Byte-level pre-tokenization applies the GPT-2/Llama byte mapping, so a byte-level vocabulary sees the printable scalars it was trained on.
//! A `Split` node with a regex pattern cannot be reproduced without a regex engine, so it is reported as a warning and degraded to whitespace splitting; a token count stays useful.

use serde_json::Value;

use super::bytes;
use super::hf_json::{bool_of, kind_of, literal_pattern, pattern_is_regex, str_of};

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
    Split {
        /// The literal delimiter.
        pattern: String,
        /// What happens to the delimiter.
        behavior: SplitBehavior,
        /// Whether the pattern marks the pieces instead of the delimiters.
        invert: bool,
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
            "Split" => {
                if pattern_is_regex(value, "pattern") {
                    warnings.push(
                        "pre_tokenizer Split uses a regex pattern, which this loader cannot apply; falling back to whitespace splitting"
                            .to_owned(),
                    );
                    return Self::Whitespace;
                }
                let Some(pattern) = literal_pattern(value, "pattern") else {
                    warnings.push(
                        "pre_tokenizer Split declares no literal pattern; the text is not pre-tokenized"
                            .to_owned(),
                    );
                    return Self::Identity;
                };
                Self::Split {
                    pattern,
                    behavior: match str_of(value, "behavior").unwrap_or("Isolated") {
                        "Removed" => SplitBehavior::Removed,
                        "MergedWithPrevious" => SplitBehavior::MergedWithPrevious,
                        "MergedWithNext" => SplitBehavior::MergedWithNext,
                        "Contiguous" => SplitBehavior::Contiguous,
                        _ => SplitBehavior::Isolated,
                    },
                    invert: bool_of(value, "invert", false),
                }
            }
            other => {
                warnings.push(format!(
                    "pre_tokenizer `{other}` is not implemented; the text is not pre-tokenized"
                ));
                Self::Identity
            }
        }
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
            } => {
                let prepared = if *add_prefix_space && !segment.starts_with(char::is_whitespace) {
                    format!(" {segment}")
                } else {
                    segment.to_owned()
                };
                let mapped = bytes::encode(&prepared);
                if *use_regex {
                    gpt2_split(&mapped)
                } else {
                    vec![mapped]
                }
            }
            Self::Punctuation => punctuation_split(segment),
            Self::Digits { individual_digits } => digits_split(segment, *individual_digits),
            Self::Metaspace {
                replacement,
                prepend,
                split,
            } => metaspace_split(segment, *replacement, *prepend, *split, first),
            Self::Split {
                pattern,
                behavior,
                invert,
            } => split_literal(segment, pattern, *behavior, *invert),
        }
    }
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
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut rest = segment;
    while let Some(index) = rest.find(pattern) {
        let before = &rest[..index];
        rest = &rest[index + pattern.len()..];
        if invert {
            // The complement is the delimiter, so every match is a piece on its own.
            push_piece(&mut pieces, pattern.to_owned());
            continue;
        }
        current.push_str(before);
        match behavior {
            SplitBehavior::Removed => push_piece(&mut pieces, std::mem::take(&mut current)),
            SplitBehavior::Isolated => {
                push_piece(&mut pieces, std::mem::take(&mut current));
                push_piece(&mut pieces, pattern.to_owned());
            }
            SplitBehavior::MergedWithPrevious => {
                current.push_str(pattern);
                push_piece(&mut pieces, std::mem::take(&mut current));
            }
            SplitBehavior::MergedWithNext => {
                push_piece(&mut pieces, std::mem::take(&mut current));
                current.push_str(pattern);
            }
            SplitBehavior::Contiguous => current.push_str(pattern),
        }
    }
    current.push_str(rest);
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
/// Without Unicode general categories this treats every non-alphanumeric, non-whitespace, non-control character as punctuation, which matches `BERT`'s ASCII set and covers the common Latin punctuation.
fn is_punctuation(ch: char) -> bool {
    !ch.is_alphanumeric() && !ch.is_whitespace() && !ch.is_control()
}

/// Splits `text` the way GPT-2's pre-tokenization pattern does.
fn gpt2_split(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut position = 0;
    while position < chars.len() {
        let length = gpt2_match(&chars, position).max(1);
        pieces.push(chars[position..position + length].iter().collect());
        position += length;
    }
    pieces
}

/// The length of the GPT-2 alternative that matches at `position`, or `0` when none does.
fn gpt2_match(chars: &[char], position: usize) -> usize {
    const CONTRACTIONS: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];
    if chars[position] == '\'' {
        for suffix in CONTRACTIONS {
            let suffix_chars: Vec<char> = suffix.chars().collect();
            let end = position + 1 + suffix_chars.len();
            if end <= chars.len() && chars[position + 1..end] == suffix_chars[..] {
                return end - position;
            }
        }
    }
    for class in [
        char::is_alphabetic as fn(char) -> bool,
        char::is_numeric,
        is_gpt2_other,
    ] {
        if let Some(length) = spaced_run(chars, position, class) {
            return length;
        }
    }
    if chars[position].is_whitespace() {
        let mut end = position;
        while end < chars.len() && chars[end].is_whitespace() {
            end += 1;
        }
        let run = end - position;
        if end == chars.len() || run == 1 {
            return run;
        }
        return run - 1;
    }
    0
}

/// A character class of the GPT-2 pattern outside letters, numbers and whitespace.
fn is_gpt2_other(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric()
}

/// The length of ` ?<class>+` at `position`.
fn spaced_run(chars: &[char], position: usize, class: fn(char) -> bool) -> Option<usize> {
    let start =
        if chars[position] == ' ' && position + 1 < chars.len() && class(chars[position + 1]) {
            position + 1
        } else {
            position
        };
    if !class(chars[start]) {
        return None;
    }
    let mut end = start;
    while end < chars.len() && class(chars[end]) {
        end += 1;
    }
    Some(end - position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bert_splits_punctuation_and_whitespace() {
        assert_eq!(
            bert_split("Hello, world!"),
            vec!["Hello", ",", "world", "!"]
        );
    }

    #[test]
    fn whitespace_split_drops_whitespace() {
        let mut warnings = Vec::new();
        let pretokenizer =
            Pretokenizer::from_json(Some(&json!({"type": "Whitespace"})), &mut warnings);
        assert_eq!(pretokenizer.apply(" a  b\tc "), vec!["a", "b", "c"]);
    }

    #[test]
    fn metaspace_prefixes_and_splits() {
        let mut warnings = Vec::new();
        let pretokenizer = Pretokenizer::from_json(
            Some(
                &json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true}),
            ),
            &mut warnings,
        );
        assert_eq!(
            pretokenizer.apply("hello world"),
            vec!["\u{2581}hello", "\u{2581}world"]
        );
    }

    #[test]
    fn byte_level_maps_bytes_and_splits_with_the_gpt2_pattern() {
        let mut warnings = Vec::new();
        let pretokenizer = Pretokenizer::from_json(
            Some(&json!({"type": "ByteLevel", "add_prefix_space": false, "use_regex": true})),
            &mut warnings,
        );
        let pieces = pretokenizer.apply("hi there!");
        assert_eq!(pieces, vec!["hi\u{0120}there", "!"]);
    }

    #[test]
    fn regex_splits_degrade_to_whitespace_with_a_warning() {
        let mut warnings = Vec::new();
        let pretokenizer = Pretokenizer::from_json(
            Some(&json!({"type": "Split", "pattern": {"Regex": "\\w+"}, "behavior": "Isolated"})),
            &mut warnings,
        );
        assert!(matches!(pretokenizer, Pretokenizer::Whitespace));
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn literal_splits_honour_the_behavior() {
        let mut warnings = Vec::new();
        let pretokenizer = Pretokenizer::from_json(
            Some(
                &json!({"type": "Split", "pattern": {"String": "-"}, "behavior": "MergedWithPrevious"}),
            ),
            &mut warnings,
        );
        assert_eq!(pretokenizer.apply("a-b-c"), vec!["a-", "b-", "c"]);
    }

    #[test]
    fn sequences_chain_splitters() {
        let mut warnings = Vec::new();
        let pretokenizer = Pretokenizer::from_json(
            Some(&json!({"type": "Sequence", "pretokenizers": [
                {"type": "Whitespace"},
                {"type": "Digits"}
            ]})),
            &mut warnings,
        );
        assert_eq!(pretokenizer.apply("ab12 cd"), vec!["ab", "12", "cd"]);
    }
}
