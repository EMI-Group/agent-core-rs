//! The pre-tokenization rules a `tiktoken` vocabulary can use.
//!
//! [`PretokenizePattern`] names the rule, its `match_len` runs it, and the matchers below implement the Kimi and GPT-2 alternations one alternative at a time.

use serde::{Deserialize, Serialize};

/// The pre-tokenization rule applied before the BPE merges.
///
/// The two rules differ in how they cut words, contractions, digits and whitespace; everything after the split shares the same merge algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PretokenizePattern {
    /// The Kimi rule (`KIMI_TIKTOKEN_PAT_STR`): Han runs, case-aware word runs with attached contractions, 1–3 digit runs, and the trailing-whitespace lookahead.
    #[default]
    Kimi,
    /// The GPT-2/`cl100k_base` rule: contractions, optional-space word/digit/symbol runs, and the trailing-whitespace lookahead.
    Gpt2,
}

impl PretokenizePattern {
    /// The absolute end index of the first match at `start`, or `None` when no alternative matches.
    pub(super) fn match_len(self, chars: &[char], start: usize) -> Option<usize> {
        match self {
            Self::Kimi => kimi_match(chars, start),
            Self::Gpt2 => gpt2_match(chars, start),
        }
    }
}

/// One alternative of a rule, returning the absolute end index.
type Matcher = fn(&[char], usize) -> Option<usize>;

/// Runs the alternatives in order, returning the first match.
fn first_match(chars: &[char], start: usize, alternatives: &[Matcher]) -> Option<usize> {
    alternatives
        .iter()
        .find_map(|alternative| alternative(chars, start))
}

/// The Kimi alternation, in the order of `KIMI_TIKTOKEN_PAT_STR`.
fn kimi_match(chars: &[char], start: usize) -> Option<usize> {
    first_match(
        chars,
        start,
        &[
            han_run,
            kimi_lowercase_word,
            kimi_uppercase_word,
            digit_run_kimi,
            symbol_run,
            whitespace_to_newline,
            trailing_whitespace,
            whitespace_run,
        ],
    )
}

/// The GPT-2/`cl100k_base` alternation.
fn gpt2_match(chars: &[char], start: usize) -> Option<usize> {
    first_match(
        chars,
        start,
        &[
            contraction,
            gpt2_letters,
            gpt2_numbers,
            gpt2_symbols,
            trailing_whitespace,
            whitespace_run,
        ],
    )
}

/// Whether `ch` belongs to the Han script, over the unified-ideograph and compatibility ranges.
fn is_han(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x2E80..=0x2EFF
            | 0x2F00..=0x2FDF
            | 0x3005
            | 0x3007
            | 0x3021..=0x3029
            | 0x3038..=0x303B
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2EBEF
            | 0x2F800..=0x2FA1F
            | 0x30000..=0x323AF
    )
}

/// Whether `ch` is a combining mark (`\p{M}`-like) in one of the common blocks.
fn is_mark(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x0591..=0x05BD
            | 0x05BF
            | 0x05C1..=0x05C2
            | 0x0610..=0x061A
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06DC
            | 0x0730..=0x074A
            | 0x07A6..=0x07B0
            | 0x0900..=0x0903
            | 0x093A..=0x094F
            | 0x0951..=0x0957
            | 0x0962..=0x0963
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20F0
            | 0x302A..=0x302F
            | 0x3099..=0x309A
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
    )
}

/// `\p{L}` without Han: alphabetic characters outside the Han script.
fn is_letter(ch: char) -> bool {
    ch.is_alphabetic() && !is_han(ch)
}

/// `\p{N}`: any numeric character.
fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`-like: a non-lowercase letter or a mark.
fn is_upper_like(ch: char) -> bool {
    is_mark(ch) || (is_letter(ch) && !ch.is_lowercase())
}

/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`-like: a lowercase letter or a mark.
fn is_lower_like(ch: char) -> bool {
    is_mark(ch) || (is_letter(ch) && ch.is_lowercase())
}

/// `[^\r\n\p{L}\p{N}]`: a character a word run may carry in front of itself.
fn is_word_lead(ch: char) -> bool {
    ch != '\r' && ch != '\n' && !ch.is_alphabetic() && !is_number(ch)
}

/// `[^\s\p{L}\p{N}]`: a character of a symbol run.
fn is_symbol(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphabetic() && !is_number(ch)
}

/// `[\r\n]`: a line break.
fn is_newline(ch: char) -> bool {
    ch == '\r' || ch == '\n'
}

/// `[\p{Han}]+`.
fn han_run(chars: &[char], start: usize) -> Option<usize> {
    if !is_han(chars[start]) {
        return None;
    }
    let mut end = start;
    while end < chars.len() && is_han(chars[end]) {
        end += 1;
    }
    Some(end)
}

/// `'s|'t|'re|'ve|'m|'ll|'d`, case-insensitively.
fn contraction(chars: &[char], start: usize) -> Option<usize> {
    if chars.get(start) != Some(&'\'') {
        return None;
    }
    ["s", "t", "re", "ve", "m", "ll", "d"]
        .iter()
        .find_map(|suffix| {
            let mut position = start + 1;
            for expected in suffix.chars() {
                if chars
                    .get(position)
                    .is_none_or(|ch| !ch.eq_ignore_ascii_case(&expected))
                {
                    return None;
                }
                position += 1;
            }
            Some(position)
        })
}

/// `[^\r\n\p{L}\p{N}]? [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]* [\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+ contraction?`.
fn kimi_lowercase_word(chars: &[char], start: usize) -> Option<usize> {
    for lead in [true, false] {
        let mut position = start;
        if lead {
            if !is_word_lead(chars[start]) {
                continue;
            }
            position += 1;
        }
        let upper_start = position;
        while position < chars.len() && is_upper_like(chars[position]) {
            position += 1;
        }
        let upper_count = position - upper_start;
        for available in (0..=upper_count).rev() {
            let lower_start = upper_start + available;
            if lower_start >= chars.len() || !is_lower_like(chars[lower_start]) {
                continue;
            }
            let mut end = lower_start;
            while end < chars.len() && is_lower_like(chars[end]) {
                end += 1;
            }
            return Some(contraction(chars, end).unwrap_or(end));
        }
    }
    None
}

/// `[^\r\n\p{L}\p{N}]? [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+ [\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]* contraction?`.
fn kimi_uppercase_word(chars: &[char], start: usize) -> Option<usize> {
    for lead in [true, false] {
        let mut position = start;
        if lead {
            if !is_word_lead(chars[start]) {
                continue;
            }
            position += 1;
        }
        let upper_start = position;
        while position < chars.len() && is_upper_like(chars[position]) {
            position += 1;
        }
        if position == upper_start {
            continue;
        }
        while position < chars.len() && is_lower_like(chars[position]) {
            position += 1;
        }
        return Some(contraction(chars, position).unwrap_or(position));
    }
    None
}

/// `\p{N}{1,3}`.
fn digit_run_kimi(chars: &[char], start: usize) -> Option<usize> {
    if !is_number(chars[start]) {
        return None;
    }
    let mut end = start;
    while end < chars.len() && end - start < 3 && is_number(chars[end]) {
        end += 1;
    }
    Some(end)
}

/// ` ?[^\s\p{L}\p{N}]+[\r\n]*`.
fn symbol_run(chars: &[char], start: usize) -> Option<usize> {
    for lead in [true, false] {
        let mut position = start;
        if lead {
            if chars[start] != ' ' {
                continue;
            }
            position += 1;
        }
        let symbol_start = position;
        while position < chars.len() && is_symbol(chars[position]) {
            position += 1;
        }
        if position == symbol_start {
            continue;
        }
        while position < chars.len() && is_newline(chars[position]) {
            position += 1;
        }
        return Some(position);
    }
    None
}

/// `\s*[\r\n]+`.
fn whitespace_to_newline(chars: &[char], start: usize) -> Option<usize> {
    let mut end = start;
    let mut last_newline = None;
    while end < chars.len() && chars[end].is_whitespace() {
        if is_newline(chars[end]) {
            last_newline = Some(end);
        }
        end += 1;
    }
    last_newline.map(|index| index + 1)
}

/// `\s+(?!\S)`.
fn trailing_whitespace(chars: &[char], start: usize) -> Option<usize> {
    let mut end = start;
    while end < chars.len() && chars[end].is_whitespace() {
        end += 1;
    }
    let run = end - start;
    if run == 0 {
        return None;
    }
    if end == chars.len() {
        return Some(end);
    }
    // `\s+` backtracks one character: the next one is then whitespace, so the
    // lookahead succeeds.
    (run > 1).then_some(end - 1)
}

/// `\s+`.
fn whitespace_run(chars: &[char], start: usize) -> Option<usize> {
    let mut end = start;
    while end < chars.len() && chars[end].is_whitespace() {
        end += 1;
    }
    (end > start).then_some(end)
}

/// ` ?\p{L}+`.
fn gpt2_letters(chars: &[char], start: usize) -> Option<usize> {
    optional_space_run(chars, start, char::is_alphabetic)
}

/// ` ?\p{N}+`.
fn gpt2_numbers(chars: &[char], start: usize) -> Option<usize> {
    optional_space_run(chars, start, is_number)
}

/// ` ?[^\s\p{L}\p{N}]+`.
fn gpt2_symbols(chars: &[char], start: usize) -> Option<usize> {
    optional_space_run(chars, start, is_symbol)
}

/// Matches ` ?` followed by one or more `class` characters, trying the space first.
fn optional_space_run(chars: &[char], start: usize, class: impl Fn(char) -> bool) -> Option<usize> {
    for lead in [true, false] {
        let mut position = start;
        if lead {
            if chars[start] != ' ' {
                continue;
            }
            position += 1;
        }
        let run_start = position;
        while position < chars.len() && class(chars[position]) {
            position += 1;
        }
        if position > run_start {
            return Some(position);
        }
    }
    None
}
