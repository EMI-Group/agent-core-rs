//! Byte-pair encoding, including `HuggingFace`'s continuation, fallback and merge-list shapes.
//!
//! A merge list is applied highest-priority-first, exactly as the artifact declared it: the pair with the lowest rank merges first, all of its occurrences merge left to right, and the pass repeats until no ranked pair remains.
//! `continuing_subword_prefix` is applied to the initial symbols (the vocabulary stores it inside the piece), so merges concatenate the stored spellings.
//! Decoding reverses the spelling rules: the continuation prefix and the end-of-word suffix are stripped again, and consecutive `<0xXX>` fallback tokens are reassembled into bytes, so an artifact that declares no decoder still round-trips.

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

    /// Decodes token spellings into the text they were cut from.
    ///
    /// Each piece loses the continuation prefix at its start and the end-of-word suffix at its end, so the merge-time spellings are undone.
    /// A `<0xXX>` token contributes its byte (only when the model declares `byte_fallback`); consecutive such tokens form one byte run that is decoded as UTF-8 once the run ends, so a character split across several fallback tokens is reassembled.
    /// With `fuse_unk`, a run of unknown-token spellings collapses into the first spelling of that run, mirroring how encoding fuses such a run into one id.
    pub(crate) fn decode(&self, tokens: &[String]) -> String {
        let unknown = self.unk_id.and_then(|id| self.id_to_token(id));
        let mut out = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        let mut previous_unknown = false;
        for token in tokens {
            let byte = if self.byte_fallback {
                super::model::hex_byte_token(token)
            } else {
                None
            };
            if let Some(byte) = byte {
                bytes.push(byte);
                previous_unknown = false;
                continue;
            }
            super::model::flush_byte_run(&mut bytes, &mut out);
            let is_unknown = unknown == Some(token.as_str());
            if is_unknown && self.fuse_unk && previous_unknown {
                continue;
            }
            out.push_str(self.strip_affixes(token));
            previous_unknown = is_unknown;
        }
        super::model::flush_byte_run(&mut bytes, &mut out);
        out
    }

    /// Removes the continuation prefix and the end-of-word suffix a piece carries.
    fn strip_affixes<'a>(&self, token: &'a str) -> &'a str {
        let mut piece = token;
        if let Some(prefix) = &self.continuing_subword_prefix {
            piece = piece.strip_prefix(prefix.as_str()).unwrap_or(piece);
        }
        if let Some(suffix) = &self.end_of_word_suffix {
            piece = piece.strip_suffix(suffix.as_str()).unwrap_or(piece);
        }
        piece
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

    fn decode_fixture(extra: Value) -> BpeModel {
        let mut value = extra;
        if let Some(object) = value.as_object_mut() {
            object.insert("type".to_owned(), Value::from("BPE"));
        }
        let mut warnings = Vec::new();
        BpeModel::from_json(&value, &mut warnings).expect("BPE fixture")
    }

    fn spellings(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|token| (*token).to_owned()).collect()
    }

    #[test]
    fn decode_strips_the_continuation_prefix() {
        let model = decode_fixture(json!({
            "vocab": {"a": 0, "##b": 1, "##c": 2},
            "merges": [],
            "continuing_subword_prefix": "##"
        }));
        assert_eq!(model.decode(&spellings(&["a", "##b", "##c"])), "abc");
        // The prefix is stripped wherever it appears: decoding cannot know which pieces the encoder prefixed.
        assert_eq!(model.decode(&spellings(&["##a", "##b"])), "ab");
    }

    #[test]
    fn decode_strips_the_end_of_word_suffix() {
        let model = decode_fixture(json!({
            "vocab": {"hel": 0, "lo</w>": 1, "world</w>": 2},
            "merges": [],
            "end_of_word_suffix": "</w>"
        }));
        assert_eq!(model.decode(&spellings(&["hel", "lo</w>"])), "hello");
        assert_eq!(model.decode(&spellings(&["world</w>"])), "world");
    }

    #[test]
    fn decode_reassembles_a_multi_byte_character_from_fallback_tokens() {
        let model = decode_fixture(json!({
            "vocab": {"<0xC3>": 0, "<0xA9>": 1, "<0xE4>": 2, "<0xB8>": 3, "<0xAD>": 4},
            "merges": [],
            "byte_fallback": true
        }));
        // `é` is `0xC3 0xA9`; `中` is the three bytes `0xE4 0xB8 0xAD`.
        assert_eq!(model.decode(&spellings(&["<0xC3>", "<0xA9>"])), "é");
        assert_eq!(
            model.decode(&spellings(&["<0xE4>", "<0xB8>", "<0xAD>"])),
            "中"
        );
        // The run ends at a real token, so text and bytes alternate without leaking into each other.
        assert_eq!(model.decode(&spellings(&["<0xC3>", "<0xA9>", "a"])), "éa");
    }

    #[test]
    fn decode_keeps_byte_tokens_literal_without_byte_fallback() {
        let model = decode_fixture(json!({
            "vocab": {"<0xC3>": 0, "<0xA9>": 1},
            "merges": []
        }));
        assert_eq!(
            model.decode(&spellings(&["<0xC3>", "<0xA9>"])),
            "<0xC3><0xA9>"
        );
    }

    #[test]
    fn decode_fuses_an_unknown_run_only_when_asked() {
        let fused = decode_fixture(json!({
            "vocab": {"<unk>": 0, "a": 1},
            "merges": [],
            "unk_token": "<unk>",
            "fuse_unk": true
        }));
        assert_eq!(
            fused.decode(&spellings(&["a", "<unk>", "<unk>", "a"])),
            "a<unk>a"
        );

        let unfused = decode_fixture(json!({
            "vocab": {"<unk>": 0, "a": 1},
            "merges": [],
            "unk_token": "<unk>"
        }));
        assert_eq!(
            unfused.decode(&spellings(&["a", "<unk>", "<unk>", "a"])),
            "a<unk><unk>a"
        );
    }

    #[test]
    fn byte_fallback_round_trips_through_encode() {
        let model = decode_fixture(json!({
            "vocab": {"<0xC3>": 0, "<0xA9>": 1, "<0xE4>": 2, "<0xB8>": 3, "<0xAD>": 4},
            "merges": [],
            "byte_fallback": true
        }));
        let text = "é中";
        let mut ids = Vec::new();
        for character in text.chars() {
            model.encode_piece(&character.to_string(), &mut ids);
        }
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
        let spellings: Vec<String> = ids
            .iter()
            .filter_map(|id| model.id_to_token(*id).map(str::to_owned))
            .collect();
        assert_eq!(model.decode(&spellings), text);
    }
}
