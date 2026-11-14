//! Added and special tokens, resolved before the model sees the text.
//!
//! The longest declared token wins at each position, `single_word` restricts a match to word boundaries, and `lstrip`/`rsw` drop the whitespace a token would otherwise be glued to.
//! Special tokens such as `<|endoftext|>` or `[CLS]` are ordinary added tokens in the artifact, so they always encode as exactly one id.

use std::collections::HashMap;

use super::hf_json::AddedTokenJson;

/// A slice of the input after added tokens have been cut out.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Chunk<'a> {
    /// A matched added token and the id it declares.
    Added {
        /// The declared id.
        id: u32,
    },
    /// Text the model still has to encode.
    Text(&'a str),
}

/// The added-token table of one tokenizer.
#[derive(Debug, Clone, Default)]
pub(crate) struct AddedTokens {
    /// The declared tokens, longest content first.
    entries: Vec<AddedTokenJson>,
    /// Id to content, for decoding.
    by_id: HashMap<u32, String>,
}

impl AddedTokens {
    /// Builds the table from the `added_tokens` array.
    pub(crate) fn new(tokens: Vec<AddedTokenJson>, warnings: &mut Vec<String>) -> Self {
        let mut entries = Vec::with_capacity(tokens.len());
        for token in tokens {
            if token.content.is_empty() {
                warnings.push(format!(
                    "added token {} declares empty content and was skipped",
                    token.id
                ));
                continue;
            }
            entries.push(token);
        }
        // Longest first, so the first match at a position is the longest one.
        entries.sort_by_key(|token| std::cmp::Reverse(token.content.chars().count()));
        let by_id = entries
            .iter()
            .map(|token| (token.id, token.content.clone()))
            .collect();
        Self { entries, by_id }
    }

    /// The number of declared tokens.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The content declared for `id`.
    pub(crate) fn token(&self, id: u32) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    /// The contents of the tokens declared special, in declaration order.
    pub(crate) fn special_tokens(&self) -> Vec<&str> {
        let mut contents: Vec<&str> = self
            .entries
            .iter()
            .filter(|token| token.special)
            .map(|token| token.content.as_str())
            .collect();
        contents.sort_unstable();
        contents
    }

    /// Cuts `text` into added tokens and the text between them.
    pub(crate) fn split<'a>(&self, text: &'a str) -> Vec<Chunk<'a>> {
        let mut chunks = Vec::new();
        if text.is_empty() {
            return chunks;
        }
        if self.entries.is_empty() {
            chunks.push(Chunk::Text(text));
            return chunks;
        }
        let mut plain_start = 0;
        let mut cursor = 0;
        while cursor < text.len() {
            let Some((entry, mut end)) = self.match_at(text, cursor) else {
                cursor += text[cursor..].chars().next().map_or(1, char::len_utf8);
                continue;
            };
            let mut plain_end = cursor;
            if entry.lstrip {
                plain_end = plain_start
                    + text[plain_start..cursor]
                        .trim_end_matches(char::is_whitespace)
                        .len();
            }
            if entry.rsw {
                while let Some(ch) = text[end..].chars().next() {
                    if !ch.is_whitespace() {
                        break;
                    }
                    end += ch.len_utf8();
                }
            }
            if plain_end > plain_start {
                chunks.push(Chunk::Text(&text[plain_start..plain_end]));
            }
            chunks.push(Chunk::Added { id: entry.id });
            plain_start = end;
            cursor = end;
        }
        if plain_start < text.len() {
            chunks.push(Chunk::Text(&text[plain_start..]));
        }
        chunks
    }

    /// The longest declared token that matches at `position`.
    fn match_at(&self, text: &str, position: usize) -> Option<(&AddedTokenJson, usize)> {
        let previous = text[..position].chars().next_back();
        for entry in &self.entries {
            if !text[position..].starts_with(&entry.content) {
                continue;
            }
            let end = position + entry.content.len();
            if entry.single_word {
                if previous.is_some_and(char::is_alphanumeric) {
                    continue;
                }
                if text[end..]
                    .chars()
                    .next()
                    .is_some_and(char::is_alphanumeric)
                {
                    continue;
                }
            }
            return Some((entry, end));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(id: u32, content: &str) -> AddedTokenJson {
        AddedTokenJson {
            id,
            content: content.to_owned(),
            special: content.starts_with('<') || content.starts_with('['),
            single_word: false,
            lstrip: false,
            rsw: false,
        }
    }

    #[test]
    fn longest_added_token_wins() {
        let mut warnings = Vec::new();
        let added = AddedTokens::new(
            vec![token(1, "<|end|>"), token(2, "<|endoftext|>")],
            &mut warnings,
        );
        assert_eq!(
            added.split("a<|endoftext|>b"),
            vec![Chunk::Text("a"), Chunk::Added { id: 2 }, Chunk::Text("b")]
        );
    }

    #[test]
    fn single_word_only_matches_at_word_boundaries() {
        let mut warnings = Vec::new();
        let mut entry = token(1, "cat");
        entry.single_word = true;
        let added = AddedTokens::new(vec![entry], &mut warnings);
        assert_eq!(
            added.split("a cat!"),
            vec![Chunk::Text("a "), Chunk::Added { id: 1 }, Chunk::Text("!")]
        );
        assert_eq!(added.split("category"), vec![Chunk::Text("category")]);
    }

    #[test]
    fn lstrip_and_rsw_drop_surrounding_whitespace() {
        let mut warnings = Vec::new();
        let mut entry = token(1, "[MASK]");
        entry.lstrip = true;
        entry.rsw = true;
        let added = AddedTokens::new(vec![entry], &mut warnings);
        assert_eq!(
            added.split("a  [MASK]  b"),
            vec![Chunk::Text("a"), Chunk::Added { id: 1 }, Chunk::Text("b")]
        );
    }

    #[test]
    fn special_tokens_are_listed() {
        let mut warnings = Vec::new();
        let added = AddedTokens::new(vec![token(0, "[CLS]"), token(1, "word")], &mut warnings);
        assert_eq!(added.special_tokens(), vec!["[CLS]"]);
        assert_eq!(added.token(0), Some("[CLS]"));
        assert_eq!(added.len(), 2);
    }
}
