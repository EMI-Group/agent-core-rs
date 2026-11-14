//! A Unigram-lite sentence-piece model: best-scoring segmentation over the declared scores.
//!
//! A full `HuggingFace` Unigram model also knows a training-time pruning graph; for counting, what matters is the same forward dynamic program over `score(token)` with an unknown-character penalty, which this module implements in `O(n * max_token_len)`.
//! Pieces the vocabulary cannot cover become the unknown token, or `<0xXX>` byte tokens when the model declares `byte_fallback`.

use std::collections::HashMap;

use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

/// The score subtracted for a character no vocabulary entry covers.
const UNK_PENALTY: f64 = -100.0;

/// A Unigram model.
#[derive(Debug, Clone, Default)]
pub(crate) struct UnigramModel {
    /// Token spelling to id, for encoding.
    token_ids: HashMap<String, u32>,
    /// Id to token spelling, for decoding.
    tokens: Vec<Option<String>>,
    /// Log-probability of each token, indexed by id.
    scores: Vec<f64>,
    /// Id emitted for an uncovered character.
    unk_id: Option<u32>,
    /// Whether an uncovered character is spelled as `<0xXX>` byte tokens.
    byte_fallback: bool,
    /// Longest vocabulary token, in characters.
    max_token_chars: usize,
}

impl UnigramModel {
    /// Decodes a `{"type": "Unigram", "vocab": [[token, score], ...]}` model section.
    pub(crate) fn from_json(value: &Value, warnings: &mut Vec<String>) -> Result<Self> {
        let entries = value
            .get("vocab")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Validation("Unigram model declares no vocab".to_owned()))?;
        let mut token_ids = HashMap::with_capacity(entries.len());
        let mut tokens = Vec::with_capacity(entries.len());
        let mut scores = Vec::with_capacity(entries.len());
        let mut max_token_chars = 1;
        for (id, entry) in entries.iter().enumerate() {
            let id = u32::try_from(id)
                .map_err(|_| Error::Validation("Unigram vocab is too large".to_owned()))?;
            let Some(pair) = entry.as_array() else {
                warnings.push(format!(
                    "Unigram vocab entry {id} is not a [token, score] pair"
                ));
                continue;
            };
            let token = pair.first().and_then(Value::as_str).ok_or_else(|| {
                Error::Validation(format!("Unigram vocab entry {id} has no token"))
            })?;
            let score = pair.get(1).and_then(Value::as_f64).unwrap_or(0.0);
            max_token_chars = max_token_chars.max(token.chars().count());
            token_ids.insert(token.to_owned(), id);
            while tokens.len() < id as usize {
                tokens.push(None);
                scores.push(0.0);
            }
            tokens.push(Some(token.to_owned()));
            scores.push(score);
        }
        let unk_id = value
            .get("unk_id")
            .and_then(Value::as_u64)
            .map(|id| u32::try_from(id).unwrap_or(0))
            .or_else(|| token_ids.get("<unk>").copied());
        Ok(Self {
            token_ids,
            tokens,
            scores,
            unk_id,
            byte_fallback: super::hf_json::bool_of(value, "byte_fallback", false),
            max_token_chars,
        })
    }

    /// Encodes one pre-tokenized piece into `out`.
    pub(crate) fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = piece.chars().collect();
        if chars.is_empty() {
            return;
        }
        let mut best: Vec<Option<Lattice>> = vec![None; chars.len() + 1];
        best[0] = Some(Lattice {
            score: 0.0,
            previous: 0,
            id: None,
        });
        for start in 0..chars.len() {
            let Some(current) = best[start].as_ref().map(|lattice| lattice.score) else {
                continue;
            };
            consider(&mut best, start + 1, current + UNK_PENALTY, start, None);
            let limit = self.max_token_chars.min(chars.len() - start);
            for length in 1..=limit {
                let candidate: String = chars[start..start + length].iter().collect();
                let Some(id) = self.token_ids.get(&candidate) else {
                    continue;
                };
                let score = current + self.scores.get(*id as usize).copied().unwrap_or(0.0);
                consider(&mut best, start + length, score, start, Some(*id));
            }
        }
        let mut pieces: Vec<(usize, usize, Option<u32>)> = Vec::new();
        let mut position = chars.len();
        while position > 0 {
            let Some(lattice) = best[position].as_ref() else {
                position -= 1;
                continue;
            };
            pieces.push((lattice.previous, position, lattice.id));
            position = lattice.previous;
        }
        pieces.reverse();
        let mut unknown_run = false;
        for (start, end, id) in pieces {
            if let Some(id) = id {
                out.push(id);
                unknown_run = false;
                continue;
            }
            if self.byte_fallback && self.push_byte_tokens(&chars[start..end], out) {
                unknown_run = false;
                continue;
            }
            if let Some(unk) = self.unk_id {
                if !unknown_run {
                    out.push(unk);
                }
                unknown_run = true;
            }
        }
    }

    /// Emits `<0xXX>` tokens for every byte of an uncovered span.
    fn push_byte_tokens(&self, span: &[char], out: &mut Vec<u32>) -> bool {
        let text: String = span.iter().collect();
        let mut emitted = false;
        for byte in text.as_bytes() {
            let token = format!("<0x{byte:02X}>");
            if let Some(id) = self.token_ids.get(&token) {
                out.push(*id);
                emitted = true;
            }
        }
        emitted
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

/// One lattice edge: the score reaching a position, the position it came from, and the token used.
#[derive(Clone)]
struct Lattice {
    /// Accumulated log-probability.
    score: f64,
    /// The position the edge starts at.
    previous: usize,
    /// The token id, or `None` for an uncovered character.
    id: Option<u32>,
}

/// Keeps the best edge into `position`.
fn consider(
    best: &mut [Option<Lattice>],
    position: usize,
    score: f64,
    previous: usize,
    id: Option<u32>,
) {
    let Some(slot) = best.get_mut(position) else {
        return;
    };
    if slot.as_ref().is_none_or(|current| score > current.score) {
        *slot = Some(Lattice {
            score,
            previous,
            id,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_best_scoring_segmentation_wins() {
        let value = json!({
            "type": "Unigram",
            "vocab": [
                ["<unk>", 0.0],
                ["a", -2.0],
                ["b", -2.0],
                ["ab", -1.0],
                ["abc", -10.0],
                ["bc", -1.0]
            ],
            "unk_id": 0
        });
        let mut warnings = Vec::new();
        let model = UnigramModel::from_json(&value, &mut warnings).expect("Unigram fixture");
        let mut ids = Vec::new();
        model.encode_piece("abc", &mut ids);
        assert_eq!(ids, vec![1, 5]);
    }

    #[test]
    fn uncovered_characters_become_unknown() {
        let value = json!({"type": "Unigram", "vocab": [["<unk>", 0.0], ["a", -1.0]], "unk_id": 0});
        let mut warnings = Vec::new();
        let model = UnigramModel::from_json(&value, &mut warnings).expect("Unigram fixture");
        let mut ids = Vec::new();
        model.encode_piece("az", &mut ids);
        assert_eq!(ids, vec![1, 0]);
    }
}
