//! Text helpers for the rule-based triple extractor.
//!
//! Everything here is pure string processing with no I/O, no randomness, and no
//! model calls: sentence segmentation, word-bounded keyword search, entity
//! candidate detection, and entity cleaning. The helpers are crate-internal
//! (`pub(super)`) and are exercised by [`super::extract`].

/// Leading conjunctions/adverbs stripped from a candidate entity.
pub(super) const LEADING_CONNECTORS: &[&str] = &[
    "and ",
    "or ",
    "but ",
    "while ",
    "because ",
    "however ",
    "then ",
    "also ",
    "yet ",
    "so ",
    "although ",
];
/// Articles stripped from a candidate entity.
pub(super) const ARTICLES: &[&str] = &["the ", "a ", "an "];
/// Punctuation trimmed from the start of a candidate entity.
pub(super) const LEADING_PUNCTUATION: &str = "([{\"'“‘";
/// Punctuation trimmed from the end of a candidate entity.
pub(super) const TRAILING_PUNCTUATION: &str = ".,;:!?)]}\"'”’";
/// Pronouns/determiners that never form a usable entity.
pub(super) const STOPWORDS: &[&str] = &[
    "it", "them", "they", "this", "that", "these", "those", "he", "she", "we", "you", "i", "there",
    "here",
];
/// Sentence-leading prepositions removed together with their object.
const PREPOSITIONS: &[&str] = &[
    "in",
    "on",
    "at",
    "during",
    "after",
    "before",
    "since",
    "by",
    "for",
    "from",
    "as",
    "with",
    "under",
    "over",
    "between",
    "through",
    "according",
    "when",
    "while",
    "although",
];
/// Connectors that may join the words of one capitalised entity.
const PHRASE_CONNECTORS: &[&str] = &[
    "of", "the", "and", "for", "de", "van", "von", "&", "del", "di",
];
/// Markers that end an object: relative clauses and finite verbs.
///
/// Cutting here keeps a list item (`"Vue are popular"` → `"Vue"`) and a relative
/// clause (`"company that makes tools"` → `"company"`) out of the object.
const OBJECT_MARKERS: &[&str] = &[
    "which", "that", "who", "whose", "where", "is", "are", "was", "were", "can", "will", "may",
    "might", "must", "should",
];

/// Splits text into sentences on `.`, `!`, `?`, `;`, and newlines.
///
/// A `.` inside a number (`3.5`) or followed directly by an alphanumeric
/// (`e.g.` → the first dot) does not split.
pub(super) fn split_sentences(text: &str) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for (index, character) in characters.iter().enumerate() {
        let is_delimiter = matches!(character, '.' | '!' | '?' | ';' | '\n' | '\r');
        if !is_delimiter {
            current.push(*character);
            continue;
        }
        let next = characters.get(index + 1).copied();
        let inside_token = *character == '.'
            && (next.is_some_and(char::is_alphanumeric)
                || is_dotted_chain(&current)
                || (is_listed_abbreviation(&current)
                    && next_word_starts_lowercase(&characters, index)));
        if inside_token {
            current.push(*character);
            continue;
        }
        push_sentence(&mut sentences, &mut current);
    }
    push_sentence(&mut sentences, &mut current);
    sentences
}

/// Word abbreviations whose trailing `.` does not end a sentence when the next
/// word starts lowercase (`"Acme Inc. was founded"`).
const ABBREVIATIONS: &[&str] = &[
    "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.", "inc.", "ltd.", "co.", "corp.", "st.",
    "no.", "fig.", "al.",
];

/// The token that ends the current sentence buffer.
fn trailing_token(sentence: &str) -> &str {
    let token_start = sentence
        .rfind(char::is_whitespace)
        .map_or(0, |index| index + 1);
    &sentence[token_start..]
}

/// Whether the buffer ends with a dotted single-letter chain such as `e.g`/`U.S`.
///
/// These never end a sentence, so `"e.g. good"` stays one sentence.
fn is_dotted_chain(sentence: &str) -> bool {
    let token = trailing_token(sentence);
    let mut letters = 0;
    for part in token.split('.') {
        if part.len() == 1
            && part
                .chars()
                .all(|character| character.is_ascii_alphabetic())
        {
            letters += 1;
        } else {
            return false;
        }
    }
    letters >= 2
}

/// Whether the buffer ends with one of the [`ABBREVIATIONS`] word forms.
fn is_listed_abbreviation(sentence: &str) -> bool {
    let token = trailing_token(sentence);
    !token.is_empty() && ABBREVIATIONS.contains(&format!("{}.", token.to_lowercase()).as_str())
}

/// Whether the next non-whitespace character after `index` is lowercase.
fn next_word_starts_lowercase(characters: &[char], index: usize) -> bool {
    characters[index + 1..]
        .iter()
        .find(|character| !character.is_whitespace())
        .is_some_and(|character| character.is_lowercase())
}

/// Pushes the trimmed, non-empty `current` buffer and clears it.
fn push_sentence(sentences: &mut Vec<String>, current: &mut String) {
    let sentence = current.trim();
    if !sentence.is_empty() {
        sentences.push(sentence.to_owned());
    }
    current.clear();
}

/// Finds every word-bounded, ASCII-case-insensitive occurrence of `keyword`.
///
/// Returns byte ranges into `text`, in order.
pub(super) fn find_keyword(text: &str, keyword: &str) -> Vec<(usize, usize)> {
    let characters: Vec<(usize, char)> = text.char_indices().collect();
    let needle: Vec<char> = keyword.chars().collect();
    let mut hits = Vec::new();
    if needle.is_empty() || characters.len() < needle.len() {
        return hits;
    }
    let mut index = 0;
    while index + needle.len() <= characters.len() {
        let matched = needle
            .iter()
            .enumerate()
            .all(|(offset, expected)| characters[index + offset].1.eq_ignore_ascii_case(expected));
        let before_ok = index == 0 || !is_word_char(characters[index - 1].1);
        let after = characters.get(index + needle.len());
        let after_ok = after.is_none_or(|(_, character)| !is_word_char(*character));
        if matched && before_ok && after_ok {
            let start = characters[index].0;
            let end = after.map_or(text.len(), |(offset, _)| *offset);
            hits.push((start, end));
            index += needle.len();
            continue;
        }
        index += 1;
    }
    hits
}

/// Word characters for boundary checks: alphanumerics and `_`.
fn is_word_char(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// Cuts an object at the first trailing clause marker or finite verb.
pub(super) fn trim_object_tail(text: &str) -> &str {
    let mut cut = text.len();
    for marker in OBJECT_MARKERS {
        if let Some((start, _)) = find_keyword(text, marker).first() {
            cut = cut.min(*start);
        }
    }
    let trimmed = text[..cut].trim();
    if trimmed.is_empty() {
        text
    } else {
        trimmed
    }
}

/// Splits an exemplar list on `,`, `and`, `or`, and `&`.
pub(super) fn split_object_list(text: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    for comma_part in text.split(',') {
        let mut current = String::new();
        for token in comma_part.split_whitespace() {
            if matches!(token.to_lowercase().as_str(), "and" | "or" | "&") {
                if !current.is_empty() {
                    parts.push(std::mem::take(&mut current));
                }
                continue;
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(token);
        }
        if !current.is_empty() {
            parts.push(current);
        }
    }
    parts
}

/// Strips any of the ASCII `prefixes` repeatedly from the start of `text`.
pub(super) fn strip_leading<'a>(text: &'a str, prefixes: &[&str]) -> &'a str {
    let mut current = text;
    loop {
        let trimmed = current.trim_start();
        let lowered = trimmed.to_lowercase();
        let matched = prefixes
            .iter()
            .find(|prefix| lowered.starts_with(**prefix))
            .copied();
        let Some(prefix) = matched else {
            return current;
        };
        current = &trimmed[prefix.len()..];
    }
}

/// Drops a leading prepositional phrase (`"In 1990 Acme"` → `"Acme"`).
pub(super) fn strip_prepositional_prefix(text: &str) -> &str {
    let tokens = tokens(text);
    let Some((_, first)) = tokens.first() else {
        return text;
    };
    let first_word = first
        .trim_matches(|character: char| !character.is_alphanumeric())
        .to_lowercase();
    if !PREPOSITIONS.contains(&first_word.as_str()) {
        return text;
    }
    let offset = tokens
        .iter()
        .skip(1)
        .position(|(_, token)| has_single_entity_hint(token))
        .map(|position| position + 1);
    let Some(offset) = offset else {
        return text;
    };
    text.get(tokens[offset].0..).map_or(text, str::trim)
}

/// Whether `text` contains an uppercase or CJK character (entity evidence).
pub(super) fn has_entity_hint(text: &str) -> bool {
    text.chars()
        .any(|character| character.is_uppercase() || is_cjk(character))
}

/// Whether `text` starts with an uppercase character (CJK has its own detector).
fn starts_uppercase(text: &str) -> bool {
    text.trim_start_matches(|character: char| !character.is_alphanumeric())
        .chars()
        .next()
        .is_some_and(char::is_uppercase)
}

/// Whether `text` starts with an uppercase or CJK character.
fn has_single_entity_hint(text: &str) -> bool {
    text.trim_start_matches(|character: char| !character.is_alphanumeric())
        .chars()
        .next()
        .is_some_and(|character| character.is_uppercase() || is_cjk(character))
}

/// Whether `character` is a CJK ideograph or kana.
fn is_cjk(character: char) -> bool {
    matches!(character as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF)
}

/// Whitespace-separated tokens with their byte offsets.
fn tokens(text: &str) -> Vec<(usize, &str)> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (index, character) in text.char_indices() {
        if character.is_whitespace() {
            if let Some(begin) = start.take() {
                tokens.push((begin, &text[begin..index]));
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(begin) = start {
        tokens.push((begin, &text[begin..]));
    }
    tokens
}

/// Quoted spans (`"…"` and `“…”`) with their opening byte offsets.
pub(super) fn quoted_spans(text: &str) -> Vec<(usize, String)> {
    let characters: Vec<(usize, char)> = text.char_indices().collect();
    let mut spans = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        let (start, character) = characters[index];
        let closing = match character {
            '"' => Some('"'),
            '\u{201c}' => Some('\u{201d}'),
            _ => None,
        };
        let Some(closing) = closing else {
            index += 1;
            continue;
        };
        let mut content = String::new();
        let mut cursor = index + 1;
        let mut closed = false;
        while cursor < characters.len() {
            let (_, current) = characters[cursor];
            if current == closing {
                closed = true;
                break;
            }
            content.push(current);
            cursor += 1;
        }
        if closed {
            spans.push((start, content));
            index = cursor + 1;
        } else {
            index += 1;
        }
    }
    spans
}

/// Contiguous CJK runs with their byte offsets.
pub(super) fn cjk_runs(text: &str) -> Vec<(usize, String)> {
    let mut runs = Vec::new();
    let mut current = String::new();
    let mut start = 0;
    for (index, character) in text.char_indices() {
        if is_cjk(character) {
            if current.is_empty() {
                start = index;
            }
            current.push(character);
        } else if !current.is_empty() {
            runs.push((start, std::mem::take(&mut current)));
        }
    }
    if !current.is_empty() {
        runs.push((start, current));
    }
    runs
}

/// Capitalised multi-word phrases with their byte offsets.
pub(super) fn capitalised_phrases(text: &str) -> Vec<(usize, String)> {
    let tokens = tokens(text);
    let mut phrases = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        if !starts_uppercase(tokens[index].1) {
            index += 1;
            continue;
        }
        let start = tokens[index].0;
        let mut end = tokens[index].0 + tokens[index].1.len();
        let mut cursor = index + 1;
        while cursor < tokens.len() {
            if starts_uppercase(tokens[cursor].1) {
                end = tokens[cursor].0 + tokens[cursor].1.len();
                cursor += 1;
            } else if is_phrase_connector(tokens[cursor].1)
                && cursor + 1 < tokens.len()
                && starts_uppercase(tokens[cursor + 1].1)
            {
                end = tokens[cursor + 1].0 + tokens[cursor + 1].1.len();
                cursor += 2;
            } else {
                break;
            }
        }
        phrases.push((start, text[start..end].to_owned()));
        index = cursor;
    }
    phrases
}

/// Whether `token` may join two capitalised words into one entity.
fn is_phrase_connector(token: &str) -> bool {
    let lowered = token.to_lowercase();
    PHRASE_CONNECTORS.contains(&lowered.as_str())
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn sentences_split_on_terminators_and_keep_abbreviations() {
        let sentences =
            split_sentences("Acme is a company. It has 3.5 stars; e.g. good\nNext? Yes!");
        assert_eq!(
            sentences,
            vec![
                "Acme is a company",
                "It has 3.5 stars",
                "e.g. good",
                "Next",
                "Yes"
            ]
        );
    }

    #[test]
    fn listed_abbreviations_keep_a_sentence_only_before_lowercase() {
        assert_eq!(
            split_sentences("Acme Inc. was founded. Beta Corp. Gamma is new."),
            vec!["Acme Inc. was founded", "Beta Corp", "Gamma is new"]
        );
    }

    #[test]
    fn keyword_matching_respects_word_boundaries() {
        assert_eq!(find_keyword("This is a test", "is"), vec![(5, 7)]);
        assert_eq!(find_keyword("Acme has a plan", "has"), vec![(5, 8)]);
        assert_eq!(find_keyword("Acme has a plan", "as"), Vec::new());
        assert_eq!(find_keyword("A uses B", "uses"), vec![(2, 6)]);
        assert_eq!(find_keyword("A is an apple", "is a"), Vec::new());
    }

    #[test]
    fn object_tails_and_lists_are_trimmed() {
        assert_eq!(
            trim_object_tail("React and Vue are popular"),
            "React and Vue"
        );
        assert_eq!(trim_object_tail("a company that makes tools"), "a company");
        assert_eq!(
            split_object_list("Rust, Go and Python"),
            vec!["Rust", "Go", "Python"]
        );
    }
}
