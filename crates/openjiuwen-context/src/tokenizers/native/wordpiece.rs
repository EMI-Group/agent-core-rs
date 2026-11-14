//! `WordPiece`, the `BERT` subword model.
//!
//! Splits a piece greedily longest-match-first from the left; every piece after the first is looked up with the continuation prefix (`##` by default).
//! A word longer than `max_input_chars_per_word`, or one with no matching subword, becomes the unknown token.

use std::collections::HashMap;

use serde_json::Value;

/// A word-piece model.
#[derive(Debug, Clone)]
pub(crate) struct WordPieceModel {
    /// Token spelling to id, for encoding.
    vocab: HashMap<String, u32>,
    /// Id to token spelling, for decoding.
    tokens: Vec<Option<String>>,
    /// Id emitted for an unresolvable word.
    unk_id: Option<u32>,
    /// Prefix stored on every non-initial piece.
    continuing_subword_prefix: String,
    /// Longest word, in characters, the model will try to split.
    max_input_chars_per_word: usize,
}

impl WordPieceModel {
    /// Decodes a `{"type": "WordPiece", ...}` model section.
    pub(crate) fn from_json(
        value: &Value,
        _warnings: &mut Vec<String>,
    ) -> openjiuwen_common::error::Result<Self> {
        let vocab_json = value
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                openjiuwen_common::error::Error::Validation(
                    "WordPiece model declares no vocab".to_owned(),
                )
            })?;
        let (vocab, tokens) = super::model::decode_vocab(vocab_json)?;
        let unk_id =
            super::hf_json::str_of(value, "unk_token").and_then(|token| vocab.get(token).copied());
        Ok(Self {
            vocab,
            tokens,
            unk_id,
            continuing_subword_prefix: super::hf_json::str_of(value, "continuing_subword_prefix")
                .unwrap_or("##")
                .to_owned(),
            max_input_chars_per_word: value
                .get("max_input_chars_per_word")
                .and_then(Value::as_u64)
                .map_or(100, |limit| usize::try_from(limit).unwrap_or(usize::MAX)),
        })
    }

    /// Encodes one pre-tokenized piece into `out`.
    pub(crate) fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = piece.chars().collect();
        if chars.is_empty() {
            return;
        }
        if chars.len() > self.max_input_chars_per_word {
            self.push_unknown(out);
            return;
        }
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut matched = None;
            while start < end {
                let candidate: String = chars[start..end].iter().collect();
                let candidate = if start == 0 {
                    candidate
                } else {
                    format!("{}{candidate}", self.continuing_subword_prefix)
                };
                if let Some(id) = self.vocab.get(&candidate) {
                    matched = Some((end, *id));
                    break;
                }
                end -= 1;
            }
            let Some((next, id)) = matched else {
                self.push_unknown(out);
                return;
            };
            out.push(id);
            start = next;
        }
    }

    /// Emits the unknown token, when the vocabulary declares one.
    fn push_unknown(&self, out: &mut Vec<u32>) {
        if let Some(unk) = self.unk_id {
            out.push(unk);
        }
    }

    /// The number of vocabulary entries.
    pub(crate) fn vocab_size(&self) -> usize {
        self.tokens.iter().filter(|token| token.is_some()).count()
    }

    /// The spelling of `id`.
    pub(crate) fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).and_then(Option::as_deref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model() -> WordPieceModel {
        let value = json!({
            "type": "WordPiece",
            "vocab": {"[UNK]": 0, "hello": 1, "##ing": 2, "##s": 3, "he": 4, "##llo": 5},
            "unk_token": "[UNK]",
            "max_input_chars_per_word": 8
        });
        let mut warnings = Vec::new();
        WordPieceModel::from_json(&value, &mut warnings).expect("WordPiece fixture")
    }

    #[test]
    fn greedy_longest_match_prefers_the_longest_piece() {
        let mut ids = Vec::new();
        model().encode_piece("hello", &mut ids);
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn continuation_prefix_joins_subwords() {
        let mut ids = Vec::new();
        model().encode_piece("hellos", &mut ids);
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn unknown_words_and_overlong_words_become_unk() {
        let mut ids = Vec::new();
        model().encode_piece("zzz", &mut ids);
        assert_eq!(ids, vec![0]);
        let mut ids = Vec::new();
        model().encode_piece("hellosssss", &mut ids);
        assert_eq!(ids, vec![0]);
    }
}
