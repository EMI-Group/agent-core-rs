//! The deterministic clause splitting and title helpers of the designer passes.
//!
//! Ports the text handling of Python's `workflow_designer`: a request becomes at
//! most `MAX_STEPS` task clauses (sentence terminators, sequencing phrases,
//! numbered list markers, stripped list markers, dropped empties and
//! duplicates), every clause becomes a step title, and the first clause names
//! the workflow.

use super::super::{
    CJK_TITLE_CHARS, MAX_STEPS, NAME_LIMIT, SEQUENCE_MARKERS, TERMINATORS, TITLE_WORDS,
};

/// Splits `query` into at most `MAX_STEPS` task clauses.
pub(super) fn task_clauses(query: &str) -> Vec<String> {
    let mut clauses: Vec<String> = Vec::new();
    for numbered in split_numbered(query) {
        for sentence in split_terminators(&numbered) {
            for sequenced in split_sequences(sentence) {
                let clause = strip_list_marker(&sequenced).trim();
                if clause.chars().all(|character| !character.is_alphabetic()) {
                    continue;
                }
                if clauses.iter().any(|seen| seen == clause) {
                    continue;
                }
                clauses.push(clause.to_string());
                if clauses.len() == MAX_STEPS {
                    return clauses;
                }
            }
        }
    }
    clauses
}

/// Splits numbered list items (`1.`, `2)`, `3、`) out of `text`.
fn split_numbered(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;

    while index < text.len() {
        let bytes = text.as_bytes();
        let at_boundary = index == 0 || bytes[index - 1].is_ascii_whitespace();
        if at_boundary && bytes[index].is_ascii_digit() {
            let mut digits_end = index;
            while digits_end < text.len() && bytes[digits_end].is_ascii_digit() {
                digits_end += 1;
            }
            let rest = &text[digits_end..];
            let marker_len = if rest.starts_with('、') {
                '、'.len_utf8()
            } else if rest.starts_with('.') || rest.starts_with(')') {
                let after = &rest[1..];
                usize::from(after.is_empty() || after.starts_with(char::is_whitespace))
            } else {
                0
            };
            if marker_len > 0 {
                if index > start {
                    parts.push(text[start..index].to_string());
                    start = index;
                }
                index = digits_end + marker_len;
                continue;
            }
        }
        index += 1;
    }

    parts.push(text[start..].to_string());
    parts
}

/// Splits `text` at every sentence terminator, dropping the terminator.
fn split_terminators(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    for (index, character) in text.char_indices() {
        if TERMINATORS.contains(&character) {
            parts.push(&text[start..index]);
            start = index + character.len_utf8();
        }
    }
    parts.push(&text[start..]);
    parts
}

/// Splits `text` at every sequencing phrase, dropping the phrase itself.
fn split_sequences(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut rest = text;
    while let Some((start, length)) = find_sequence_marker(rest) {
        parts.push(rest[..start].to_string());
        rest = &rest[start + length..];
    }
    parts.push(rest.to_string());
    parts
}

/// The earliest sequencing phrase in `text`, as `(offset, length)`.
///
/// Among phrases starting at the same offset the longest one wins, so
/// `", then "` is preferred over `" then "`.
fn find_sequence_marker(text: &str) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (index, _) in text.char_indices() {
        if best.is_some_and(|(position, _)| position < index) {
            break;
        }
        for marker in SEQUENCE_MARKERS {
            let Some(length) = match_prefix_ci(&text[index..], marker) else {
                continue;
            };
            best = match best {
                Some((position, current)) if position == index && current >= length => {
                    Some((position, current))
                }
                _ => Some((index, length)),
            };
        }
    }
    best
}

/// The byte length of `prefix` when `text` starts with it, ignoring ASCII case.
fn match_prefix_ci(text: &str, prefix: &str) -> Option<usize> {
    let mut characters = text.chars();
    for wanted in prefix.chars() {
        if !characters.next()?.eq_ignore_ascii_case(&wanted) {
            return None;
        }
    }
    Some(prefix.len())
}

/// Strips one leading list marker (`-`, `*`, `1.`, `1)`, `1、`) from `text`.
fn strip_list_marker(text: &str) -> &str {
    let trimmed = text.trim_start();
    for marker in ['-', '*'] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return rest.trim_start();
        }
    }

    let mut digits_end = 0usize;
    for (index, character) in trimmed.char_indices() {
        if !character.is_ascii_digit() {
            break;
        }
        digits_end = index + character.len_utf8();
    }
    if digits_end > 0 {
        let rest = &trimmed[digits_end..];
        for marker in ['.', ')', '、'] {
            if let Some(rest) = rest.strip_prefix(marker) {
                return rest.trim_start();
            }
        }
    }
    trimmed
}

/// A deterministic title for an LLM step.
///
/// CJK clauses keep their first `CJK_TITLE_CHARS` characters, other clauses
/// their first `TITLE_WORDS` words; trailing punctuation is dropped. A clause
/// with no usable title falls back to `Step <position>`.
pub(super) fn step_title(clause: &str, position: usize) -> String {
    let title = if clause.chars().any(is_cjk) {
        clause.chars().take(CJK_TITLE_CHARS).collect::<String>()
    } else {
        clause
            .split_whitespace()
            .take(TITLE_WORDS)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let title = title
        .trim()
        .trim_end_matches(is_trailing_punctuation)
        .trim();
    if title.is_empty() {
        format!("Step {position}")
    } else {
        title.to_string()
    }
}

/// Whether `character` ends a clause fragment and should not end a title.
fn is_trailing_punctuation(character: char) -> bool {
    matches!(
        character,
        '.' | ',' | ';' | ':' | '!' | '?' | '。' | '，' | '；' | '：' | '！' | '？' | '、'
    )
}

/// Whether `character` is a CJK ideograph or kana.
fn is_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{3040}'..='\u{30FF}'   // hiragana and katakana
        | '\u{3400}'..='\u{4DBF}' // CJK unified ideographs extension A
        | '\u{4E00}'..='\u{9FFF}' // CJK unified ideographs
        | '\u{F900}'..='\u{FAFF}' // CJK compatibility ideographs
    )
}

/// Slugifies `text` into a workflow name.
///
/// Alphanumerics (including CJK) are kept, every other run becomes one `-`, the
/// result is trimmed of `-` and capped at `NAME_LIMIT` characters. Text with
/// nothing to keep yields `"workflow"`.
pub(super) fn slugify(text: &str) -> String {
    let mut slug = String::new();
    let mut separator = false;
    for character in text.chars() {
        if character.is_alphanumeric() {
            if separator && !slug.is_empty() {
                slug.push('-');
            }
            separator = false;
            slug.extend(character.to_lowercase());
        } else {
            separator = true;
        }
    }

    let capped: String = slug.chars().take(NAME_LIMIT).collect();
    let trimmed = capped.trim_matches('-');
    if trimmed.is_empty() {
        "workflow".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Truncates `text` to at most `limit` characters.
pub(super) fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    text.chars().take(limit).collect()
}
