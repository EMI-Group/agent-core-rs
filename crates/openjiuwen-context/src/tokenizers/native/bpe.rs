//! Byte-pair encoding, including `HuggingFace`'s continuation, fallback and merge-list shapes.
//!
//! A merge list is applied highest-priority-first, exactly as the artifact declared it: the pair with the lowest rank merges first, all of its occurrences merge left to right, and the pass repeats until no ranked pair remains.
//! `continuing_subword_prefix` is applied to the initial symbols (the vocabulary stores it inside the piece), so merges concatenate the stored spellings.

use std::collections::HashMap;

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

/// A byte-pair-encoding model.
#[derive(Debug, Clone, Default)]
pub(crate) struct BpeModel {
    /// Token spelling to id, for encoding.
    vocab: HashMap<String, u32>,
    /// Id to token spelling, for decoding.
    tokens: Vec<Option<String>>,
    /// Merge ranks, keyed left piece then right piece.
    merges: HashMap<String, HashMap<String, usize>>,
    /// Id of the declared unknown token, when the vocabulary has one.
    unk_id: Option<u32>,
    /// Prefix stored on every non-initial piece, when the model uses one.
    continuing_subword_prefix: Option<String>,
    /// Suffix appended to the last piece before the vocabulary lookup.
    end_of_word_suffix: Option<String>,
    /// Whether an unresolvable piece falls back to `<0xXX>` byte tokens.
    byte_fallback: bool,
    /// Whether a run of unresolvable pieces collapses into one unknown token.
    fuse_unk: bool,
}

impl BpeModel {
    /// Decodes a `{"type": "BPE", ...}` model section.
    pub(crate) fn from_json(value: &Value, warnings: &mut Vec<String>) -> Result<Self> {
        let vocab_json = value
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Validation("BPE model declares no vocab".to_owned()))?;
        let (vocab, tokens) = super::model::decode_vocab(vocab_json)?;
        let unk_token = super::hf_json::str_of(value, "unk_token");
        let unk_id = unk_token.and_then(|token| vocab.get(token).copied());
        if let Some(dropout) = value.get("dropout").and_then(Value::as_f64) {
            if dropout > 0.0 {
                warnings.push(format!(
                    "BPE dropout {dropout} is ignored; merges are applied deterministically"
                ));
            }
        }
        let merges = decode_merges(value.get("merges"), warnings);
        Ok(Self {
            vocab,
            tokens,
            merges,
            unk_id,
            continuing_subword_prefix: super::hf_json::str_of(value, "continuing_subword_prefix")
                .map(str::to_owned),
            end_of_word_suffix: super::hf_json::str_of(value, "end_of_word_suffix")
                .map(str::to_owned),
            byte_fallback: super::hf_json::bool_of(value, "byte_fallback", false),
            fuse_unk: super::hf_json::bool_of(value, "fuse_unk", false),
        })
    }

    /// Encodes one pre-tokenized piece into `out`.
    pub(crate) fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let mut symbols: Vec<String> = piece.chars().map(String::from).collect();
        if let Some(prefix) = &self.continuing_subword_prefix {
            for symbol in symbols.iter_mut().skip(1) {
                symbol.insert_str(0, prefix);
            }
        }
        merge_symbols(&self.merges, &mut symbols);
        if let Some(suffix) = &self.end_of_word_suffix {
            if let Some(last) = symbols.last_mut() {
                last.push_str(suffix);
            }
        }
        let mut unknown_run = false;
        for symbol in &symbols {
            if let Some(id) = self.vocab.get(symbol) {
                out.push(*id);
                unknown_run = false;
                continue;
            }
            if self.byte_fallback && self.push_byte_tokens(symbol, out) {
                unknown_run = false;
                continue;
            }
            if let Some(unk) = self.unk_id {
                if !(self.fuse_unk && unknown_run) {
                    out.push(unk);
                }
                unknown_run = true;
            }
        }
    }

    /// Emits `<0xXX>` tokens for every byte of `symbol` the vocabulary knows.
    fn push_byte_tokens(&self, symbol: &str, out: &mut Vec<u32>) -> bool {
        let mut emitted = false;
        for byte in symbol.as_bytes() {
            let token = format!("<0x{byte:02X}>");
            if let Some(id) = self.vocab.get(&token) {
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

/// Runs the merge loop over `symbols` until no ranked pair remains.
fn merge_symbols(merges: &HashMap<String, HashMap<String, usize>>, symbols: &mut Vec<String>) {
    while symbols.len() > 1 {
        let Some((rank, index)) = best_pair(merges, symbols) else {
            return;
        };
        let _ = rank;
        let left = symbols[index].clone();
        let right = symbols[index + 1].clone();
        let combined = format!("{left}{right}");
        let mut cursor = 0;
        while cursor + 1 < symbols.len() {
            if symbols[cursor] == left && symbols[cursor + 1] == right {
                symbols[cursor].clone_from(&combined);
                symbols.remove(cursor + 1);
            }
            cursor += 1;
        }
    }
}

/// The lowest-ranked adjacent pair and its index.
fn best_pair(
    merges: &HashMap<String, HashMap<String, usize>>,
    symbols: &[String],
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for index in 0..symbols.len() - 1 {
        let rank = merges
            .get(&symbols[index])
            .and_then(|right| right.get(&symbols[index + 1]));
        if let Some(rank) = rank {
            if best.is_none_or(|(current, _)| *rank < current) {
                best = Some((*rank, index));
            }
        }
    }
    best
}

/// Decodes the merge list, accepting both `"a b"` strings and `["a", "b"]` pairs.
fn decode_merges(
    value: Option<&Value>,
    warnings: &mut Vec<String>,
) -> HashMap<String, HashMap<String, usize>> {
    let mut merges: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let Some(entries) = value.and_then(Value::as_array) else {
        return merges;
    };
    let mut skipped = 0_usize;
    for (rank, entry) in entries.iter().enumerate() {
        let pair = match entry {
            Value::String(text) => text
                .split_once(' ')
                .map(|(left, right)| (left.to_owned(), right.to_owned())),
            Value::Array(parts) if parts.len() == 2 => {
                match (parts[0].as_str(), parts[1].as_str()) {
                    (Some(left), Some(right)) => Some((left.to_owned(), right.to_owned())),
                    _ => None,
                }
            }
            _ => None,
        };
        match pair {
            Some((left, right)) => {
                merges.entry(left).or_default().insert(right, rank);
            }
            None => skipped += 1,
        }
    }
    if skipped > 0 {
        warnings.push(format!(
            "{skipped} BPE merge entries could not be decoded and were skipped"
        ));
    }
    merges
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model() -> BpeModel {
        let value = json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
            "merges": ["a b", "ab c"],
            "unk_token": "<unk>"
        });
        let mut warnings = Vec::new();
        BpeModel::from_json(&value, &mut warnings).expect("BPE fixture")
    }

    #[test]
    fn merges_apply_in_rank_order() {
        let mut ids = Vec::new();
        model().encode_piece("abc", &mut ids);
        assert_eq!(ids, vec![4]);
    }

    #[test]
    fn unknown_pieces_fall_back_to_the_unknown_token() {
        let mut ids = Vec::new();
        model().encode_piece("az", &mut ids);
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn byte_fallback_emits_hex_tokens() {
        let value = json!({
            "type": "BPE",
            "vocab": {"<0xC3>": 0, "<0xA9>": 1},
            "merges": [],
            "byte_fallback": true
        });
        let mut warnings = Vec::new();
        let model = BpeModel::from_json(&value, &mut warnings).expect("BPE fixture");
        let mut ids = Vec::new();
        model.encode_piece("é", &mut ids);
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn continuation_prefix_is_part_of_the_piece() {
        let value = json!({
            "type": "BPE",
            "vocab": {"a": 0, "##b": 1, "##c": 2, "a##b": 3},
            "merges": [["a", "##b"]],
            "continuing_subword_prefix": "##"
        });
        let mut warnings = Vec::new();
        let model = BpeModel::from_json(&value, &mut warnings).expect("BPE fixture");
        let mut ids = Vec::new();
        model.encode_piece("ab", &mut ids);
        assert_eq!(ids, vec![3]);
    }
}
