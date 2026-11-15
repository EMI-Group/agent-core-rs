//! The GPT-2/Llama split pattern, reproduced by hand instead of with a regex engine.
//!
//! The pattern is `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, matched left to right at every position.
//! The alternatives are tried in that order, so the optional space of ` ?<class>+` joins the piece that follows it before the whitespace alternatives can claim it.
//! `\s+(?!\S)` gives its last character away whenever the run is followed by a non-whitespace character, which is why every position after it can start with ` ?`.
//! Multi-byte text is split on `char`s rather than bytes, and `char::is_alphabetic` stands in for `\p{L}` while `char::is_numeric` stands in for `\p{N}`.
//! `ByteLevel` splits the raw text first and byte-maps each piece afterwards, which is `HuggingFace`'s order: a space stays a delimiter instead of becoming the letter-like `Ġ` that would merge the words on both sides of it.

use super::super::bytes;

/// Byte-level pre-tokenization: split the raw text with the GPT-2 pattern, then map every piece.
pub(super) fn byte_level_split(
    segment: &str,
    add_prefix_space: bool,
    use_regex: bool,
) -> Vec<String> {
    let prepared = if add_prefix_space && !segment.starts_with(char::is_whitespace) {
        format!(" {segment}")
    } else {
        segment.to_owned()
    };
    let pieces = if use_regex {
        gpt2_split(&prepared)
    } else {
        vec![prepared]
    };
    pieces
        .into_iter()
        .filter(|piece| !piece.is_empty())
        .map(|piece| bytes::encode(&piece))
        .collect()
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
        // `\s+(?!\S)` gives up its last character, which ` ?next` then picks up.
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

    #[test]
    fn gpt2_whitespace_runs_give_their_last_character_away() {
        assert_eq!(gpt2_split("a  b"), vec!["a", " ", " b"]);
        assert_eq!(gpt2_split("a   "), vec!["a", "   "]);
        // A contraction is one piece, and `'` without a known suffix stays with the symbols.
        assert_eq!(gpt2_split("it's"), vec!["it", "'s"]);
        assert_eq!(gpt2_split("a1"), vec!["a", "1"]);
        assert_eq!(gpt2_split("+-"), vec!["+-"]);
        // A space before a symbol joins the symbol, not the whitespace alternative.
        assert_eq!(gpt2_split("a !"), vec!["a", " !"]);
    }

    #[test]
    fn byte_level_maps_after_it_splits() {
        assert_eq!(
            byte_level_split("hi there!", false, true),
            vec!["hi", "\u{0120}there", "!"]
        );
        assert_eq!(
            byte_level_split("Hello my friend, how is your day going?", false, true),
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
    fn byte_level_without_the_pattern_maps_everything_as_one_piece() {
        assert_eq!(byte_level_split("a b", false, false), vec!["a\u{0120}b"]);
    }

    #[test]
    fn byte_level_prefix_space_is_added_to_the_whole_input() {
        assert_eq!(
            byte_level_split("hello", true, false),
            vec!["\u{0120}hello"]
        );
        // A leading space is not doubled.
        assert_eq!(
            byte_level_split(" hello", true, false),
            vec!["\u{0120}hello"]
        );
        assert_eq!(
            byte_level_split("hi there", true, true),
            vec!["\u{0120}hi", "\u{0120}there"]
        );
    }
}
