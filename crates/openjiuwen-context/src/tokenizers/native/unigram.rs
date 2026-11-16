//! A Unigram-lite sentence-piece model: best-scoring segmentation over the declared scores.
//!
//! A full `HuggingFace` Unigram model also knows a training-time pruning graph; for counting, what matters is the same forward dynamic program over `score(token)`, which this module implements in `O(n * max_token_chars)`.
//! Unknown pricing follows the reference: the unknown edge costs `min_score - unk_penalty` (`unk_penalty` defaults to `10.0`) and is inserted only where no single-character vocabulary token covers the position, so a declared multi-character token is not automatically worse than a run of unknown characters.
//! A vocabulary edge also wins a score tie against the unknown edge, and the first (shortest) of several equally scored vocabulary edges into a position keeps it.
//! Two deliberate approximations remain: the lattice walks characters instead of the reference's byte trie, so a token is matched by its scalar count rather than by a shared prefix walk, and the unknown run is emitted by scanning the chosen path instead of from the reference's node graph.
//! Pieces the vocabulary cannot cover become the unknown token (one per run with `fuse_unk`, which defaults to `true`, and one per character without it), or `<0xXX>` byte tokens when the model declares `byte_fallback`.
//! Decoding concatenates spellings and reassembles `<0xXX>` byte runs; metaspace (`▁`) handling is left to the decoder node an artifact declares, because the model itself never rewrites that marker.

use std::cmp::Ordering;
use std::collections::HashMap;

use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

/// The score subtracted from the vocabulary's minimum score for a character no vocabulary entry covers.
const DEFAULT_UNK_PENALTY: f64 = 10.0;

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
    /// Whether an uncovered run collapses into one unknown token.
    fuse_unk: bool,
    /// The distance below the vocabulary's minimum score an uncovered character costs.
    unk_penalty: f64,
    /// The minimum declared score, which anchors the unknown price.
    min_score: f64,
    /// Longest vocabulary token, in characters.
    max_token_chars: usize,
}

impl UnigramModel {
    /// Decodes a `{"type": "Unigram", "vocab": [[token, score], ...]}` model section.
    ///
    /// The unknown price is read from `unk_penalty`, or from `unigram_unk_penalty` when an export uses that name, and falls back to [`DEFAULT_UNK_PENALTY`].
    pub(crate) fn from_json(value: &Value, warnings: &mut Vec<String>) -> Result<Self> {
        let entries = value
            .get("vocab")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Validation("Unigram model declares no vocab".to_owned()))?;
        let mut token_ids = HashMap::with_capacity(entries.len());
        let mut tokens = Vec::with_capacity(entries.len());
        let mut scores = Vec::with_capacity(entries.len());
        let mut max_token_chars = 1;
        let mut min_score = f64::INFINITY;
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
            min_score = min_score.min(score);
            token_ids.insert(token.to_owned(), id);
            while tokens.len() < id as usize {
                tokens.push(None);
                scores.push(0.0);
            }
            tokens.push(Some(token.to_owned()));
            scores.push(score);
        }
        if !min_score.is_finite() {
            min_score = 0.0;
        }
        let unk_penalty = unk_penalty_of(value);
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
            fuse_unk: super::hf_json::bool_of(value, "fuse_unk", true),
            unk_penalty,
            min_score,
            max_token_chars,
        })
    }

    /// Encodes one pre-tokenized piece into `out`.
    ///
    /// The forward pass fills one best edge per position; the backward walk then turns the best path into ids, fusing unknown neighbours when `fuse_unk` is on.
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
            let limit = self.max_token_chars.min(chars.len() - start);
            let mut single_character = false;
            for length in 1..=limit {
                let candidate: String = chars[start..start + length].iter().collect();
                let Some(id) = self.token_ids.get(&candidate) else {
                    continue;
                };
                single_character |= length == 1;
                let score = current + self.scores.get(*id as usize).copied().unwrap_or(0.0);
                consider(&mut best, start + length, score, start, Some(*id));
            }
            // The reference inserts the unknown edge only where the vocabulary covers no single character.
            if !single_character {
                consider(
                    &mut best,
                    start + 1,
                    current + self.unknown_score(),
                    start,
                    None,
                );
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
                if !(self.fuse_unk && unknown_run) {
                    out.push(unk);
                }
                unknown_run = true;
            }
        }
    }

    /// The score an uncovered character costs: the vocabulary minimum minus the declared penalty.
    fn unknown_score(&self) -> f64 {
        self.min_score - self.unk_penalty
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

    /// Decodes token spellings into the text they were cut from.
    ///
    /// Spellings concatenate as they are; a `<0xXX>` token contributes its byte (only when the model declares `byte_fallback`) and consecutive such tokens form one byte run that is decoded as UTF-8 once the run ends.
    /// The metaspace marker `▁` is left untouched on purpose: undoing it is the declared decoder node's job, and a model that never rewrites it must not guess.
    pub(crate) fn decode(&self, tokens: &[String]) -> String {
        let mut out = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        for token in tokens {
            let byte = if self.byte_fallback {
                super::model::hex_byte_token(token)
            } else {
                None
            };
            if let Some(byte) = byte {
                bytes.push(byte);
                continue;
            }
            super::model::flush_byte_run(&mut bytes, &mut out);
            out.push_str(token);
        }
        super::model::flush_byte_run(&mut bytes, &mut out);
        out
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

/// The declared unknown penalty, accepting both key spellings.
fn unk_penalty_of(value: &Value) -> f64 {
    value
        .get("unk_penalty")
        .or_else(|| value.get("unigram_unk_penalty"))
        .and_then(Value::as_f64)
        .filter(|penalty| penalty.is_finite())
        .unwrap_or(DEFAULT_UNK_PENALTY)
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
///
/// A strictly better score always wins; on a tie a vocabulary edge beats the unknown edge, and between two vocabulary edges the earlier one (the shorter token, because lengths are tried in ascending order) keeps the position.
/// Scores are compared with [`f64::total_cmp`], so an artifact that stores a NaN score still segments deterministically.
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
    let wins = slot
        .as_ref()
        .is_none_or(|current| match score.total_cmp(&current.score) {
            Ordering::Greater => true,
            Ordering::Equal => id.is_some() && current.id.is_none(),
            Ordering::Less => false,
        });
    if wins {
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

    fn model(value: Value) -> UnigramModel {
        let mut warnings = Vec::new();
        UnigramModel::from_json(&value, &mut warnings).expect("Unigram fixture")
    }

    fn spellings(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|token| (*token).to_owned()).collect()
    }

    #[test]
    fn the_best_scoring_segmentation_wins() {
        let model = model(json!({
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
        }));
        let mut ids = Vec::new();
        model.encode_piece("abc", &mut ids);
        assert_eq!(ids, vec![1, 5]);
    }

    #[test]
    fn uncovered_characters_become_unknown() {
        let model =
            model(json!({"type": "Unigram", "vocab": [["<unk>", 0.0], ["a", -1.0]], "unk_id": 0}));
        let mut ids = Vec::new();
        model.encode_piece("az", &mut ids);
        assert_eq!(ids, vec![1, 0]);
    }

    #[test]
    fn the_unknown_price_is_min_score_minus_the_penalty() {
        // `-250.0` for `ab` against two unknown characters priced at `min_score - 10 = -260.0` each:
        // the reference picks `ab` (-250) over the run (-520), while a flat `-100.0` penalty would pick the run (-200).
        let model = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["ab", -250.0]],
            "unk_id": 0
        }));
        assert_eq!(model.unknown_score(), -260.0);
        let mut ids = Vec::new();
        model.encode_piece("ab", &mut ids);
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn the_declared_unknown_penalty_is_honoured() {
        // A negative penalty prices an uncovered character above the cheapest token, so `a` + unknown beats `az`.
        let permissive = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["a", -1.0], ["az", -30.0]],
            "unk_id": 0,
            "unk_penalty": -20.0
        }));
        assert_eq!(permissive.unknown_score(), -10.0);
        let mut ids = Vec::new();
        permissive.encode_piece("az", &mut ids);
        assert_eq!(ids, vec![1, 0]);

        // The default 10.0 penalty makes the same text keep the declared two-character token.
        let strict = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["a", -1.0], ["az", -30.0]],
            "unk_id": 0
        }));
        assert_eq!(strict.unknown_score(), -40.0);
        let mut ids = Vec::new();
        strict.encode_piece("az", &mut ids);
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn the_alternate_penalty_key_is_read() {
        let model = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["a", -4.0]],
            "unk_id": 0,
            "unigram_unk_penalty": 2.0
        }));
        assert_eq!(model.unknown_score(), -6.0);
    }

    #[test]
    fn a_covered_position_gets_no_unknown_edge() {
        // The negative penalty makes the unknown edge cheaper than the declared token, which is exactly what the
        // reference's `has_single_node` guard excludes: a position the vocabulary covers never takes that edge.
        let model = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["a", -30.0]],
            "unk_id": 0,
            "unk_penalty": -20.0
        }));
        assert_eq!(model.unknown_score(), -10.0);
        let mut ids = Vec::new();
        model.encode_piece("a", &mut ids);
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn unknown_runs_fuse_or_repeat_per_character() {
        let fused = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["a", -1.0]],
            "unk_id": 0
        }));
        let mut ids = Vec::new();
        fused.encode_piece("azz", &mut ids);
        assert_eq!(ids, vec![1, 0]);

        let unfused = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["a", -1.0]],
            "unk_id": 0,
            "fuse_unk": false
        }));
        let mut ids = Vec::new();
        unfused.encode_piece("azz", &mut ids);
        assert_eq!(ids, vec![1, 0, 0]);
    }

    #[test]
    fn a_vocabulary_edge_wins_a_tie() {
        // `ab` (-2) and `a` + `b` (-1 + -1) score the same; the multi-character token keeps the position.
        let model = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["ab", -2.0], ["a", -1.0], ["b", -1.0]],
            "unk_id": 0
        }));
        let mut ids = Vec::new();
        model.encode_piece("ab", &mut ids);
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn byte_fallback_spells_uncovered_characters_as_bytes() {
        let model = model(json!({
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["a", -1.0], ["<0xE4>", -2.0], ["<0xB8>", -2.0], ["<0xAD>", -2.0]],
            "unk_id": 0,
            "byte_fallback": true
        }));
        let mut ids = Vec::new();
        model.encode_piece("中", &mut ids);
        assert_eq!(ids, vec![2, 3, 4]);
        // The three byte tokens decode back into the character.
        let run: Vec<String> = ids
            .iter()
            .filter_map(|id| model.id_to_token(*id).map(str::to_owned))
            .collect();
        assert_eq!(model.decode(&run), "中");
    }

    #[test]
    fn decode_concatenates_and_leaves_metaspace_alone() {
        let model = model(json!({
            "type": "Unigram",
            "vocab": [["\u{2581}hello", -1.0], ["world", -1.0]]
        }));
        assert_eq!(
            model.decode(&spellings(&["\u{2581}hello", "world"])),
            "\u{2581}helloworld"
        );
        // Without `byte_fallback` a byte token is an ordinary spelling.
        assert_eq!(model.decode(&spellings(&["<0xC3>"])), "<0xC3>");
    }
}
