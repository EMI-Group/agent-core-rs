//! The four decoding models a `tokenizer.json` can declare, behind one dispatch.
//!
//! `BPE`, `WordPiece` and `Unigram` are the subword models; `WordLevel` is a whole-piece lookup.
//! Any other model type is refused with a typed [`Error::Unsupported`] that names the type, because guessing a vocabulary layout would silently bias every budget decision.

use std::collections::HashMap;

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};

use super::bpe::BpeModel;
use super::unigram::UnigramModel;
use super::wordpiece::WordPieceModel;

/// A spelling→id vocabulary lookup.
pub(crate) type Vocab = HashMap<String, u32>;

/// An id→spelling table, with `None` for ids the vocabulary leaves empty.
pub(crate) type TokenTable = Vec<Option<String>>;

/// A decoding model plus its vocabulary lookups.
#[derive(Debug, Clone)]
pub(crate) enum NativeModel {
    /// Byte-pair encoding.
    Bpe(BpeModel),
    /// `WordPiece`.
    WordPiece(WordPieceModel),
    /// Unigram-lite.
    Unigram(UnigramModel),
    /// Whole-piece lookup.
    WordLevel(WordLevelModel),
}

impl NativeModel {
    /// Decodes the `model` section of a `tokenizer.json`.
    pub(crate) fn from_json(value: &Value, warnings: &mut Vec<String>) -> Result<Self> {
        let kind = super::hf_json::kind_of(value)
            .ok_or_else(|| Error::Validation("tokenizer model declares no type".to_owned()))?;
        match kind {
            "BPE" => BpeModel::from_json(value, warnings).map(Self::Bpe),
            "WordPiece" => WordPieceModel::from_json(value, warnings).map(Self::WordPiece),
            "Unigram" => UnigramModel::from_json(value, warnings).map(Self::Unigram),
            "WordLevel" => WordLevelModel::from_json(value).map(Self::WordLevel),
            other => Err(Error::Unsupported(format!(
                "tokenizer model type `{other}` is not supported; this loader decodes BPE, WordPiece, Unigram and WordLevel"
            ))),
        }
    }

    /// Encodes one pre-tokenized piece into `out`.
    pub(crate) fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        match self {
            Self::Bpe(model) => model.encode_piece(piece, out),
            Self::WordPiece(model) => model.encode_piece(piece, out),
            Self::Unigram(model) => model.encode_piece(piece, out),
            Self::WordLevel(model) => model.encode_piece(piece, out),
        }
    }

    /// The number of vocabulary entries.
    pub(crate) fn vocab_size(&self) -> usize {
        match self {
            Self::Bpe(model) => model.vocab_size(),
            Self::WordPiece(model) => model.vocab_size(),
            Self::Unigram(model) => model.vocab_size(),
            Self::WordLevel(model) => model.vocab_size(),
        }
    }

    /// The spelling of `id`.
    pub(crate) fn id_to_token(&self, id: u32) -> Option<&str> {
        match self {
            Self::Bpe(model) => model.id_to_token(id),
            Self::WordPiece(model) => model.id_to_token(id),
            Self::Unigram(model) => model.id_to_token(id),
            Self::WordLevel(model) => model.id_to_token(id),
        }
    }
}

/// A whole-piece lookup: the pre-tokenized piece must be a vocabulary entry.
#[derive(Debug, Clone, Default)]
pub(crate) struct WordLevelModel {
    /// Token spelling to id, for encoding.
    vocab: HashMap<String, u32>,
    /// Id to token spelling, for decoding.
    tokens: Vec<Option<String>>,
    /// Id emitted for an unknown piece.
    unk_id: Option<u32>,
}

impl WordLevelModel {
    /// Decodes a `{"type": "WordLevel", ...}` model section.
    fn from_json(value: &Value) -> Result<Self> {
        let vocab_json = value
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Validation("WordLevel model declares no vocab".to_owned()))?;
        let (vocab, tokens) = decode_vocab(vocab_json)?;
        let unk_id =
            super::hf_json::str_of(value, "unk_token").and_then(|token| vocab.get(token).copied());
        Ok(Self {
            vocab,
            tokens,
            unk_id,
        })
    }

    /// Encodes one pre-tokenized piece into `out`.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        match self.vocab.get(piece) {
            Some(id) => out.push(*id),
            None => {
                if let Some(unk) = self.unk_id {
                    out.push(unk);
                }
            }
        }
    }

    /// The number of vocabulary entries.
    fn vocab_size(&self) -> usize {
        self.tokens.iter().filter(|token| token.is_some()).count()
    }

    /// The spelling of `id`.
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).and_then(Option::as_deref)
    }
}

/// Builds the spelling→id map plus the id→spelling table a decoder needs.
pub(crate) fn decode_vocab(vocab_json: &Map<String, Value>) -> Result<(Vocab, TokenTable)> {
    let mut vocab = HashMap::with_capacity(vocab_json.len());
    let mut highest = 0_usize;
    for (token, id) in vocab_json {
        let id = id.as_u64().ok_or_else(|| {
            Error::Validation(format!("vocabulary id of token `{token}` is not a number"))
        })?;
        let id = u32::try_from(id)
            .map_err(|_| Error::Validation("vocabulary id does not fit in 32 bits".to_owned()))?;
        highest = highest.max(id as usize);
        vocab.insert(token.clone(), id);
    }
    let mut tokens = vec![None; highest + 1];
    for (token, id) in &vocab {
        if let Some(slot) = tokens.get_mut(*id as usize) {
            *slot = Some(token.clone());
        }
    }
    Ok((vocab, tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unknown_model_types_are_unsupported() {
        let mut warnings = Vec::new();
        let error = NativeModel::from_json(&json!({"type": "UnigramV2"}), &mut warnings)
            .expect_err("unknown model type");
        assert!(matches!(error, Error::Unsupported(_)));
        assert!(error.to_string().contains("UnigramV2"));
    }

    #[test]
    fn word_level_looks_up_whole_pieces() {
        let mut warnings = Vec::new();
        let model = NativeModel::from_json(
            &json!({"type": "WordLevel", "vocab": {"a": 0, "b": 1}, "unk_token": "a"}),
            &mut warnings,
        )
        .expect("WordLevel fixture");
        let mut ids = Vec::new();
        model.encode_piece("b", &mut ids);
        model.encode_piece("c", &mut ids);
        assert_eq!(ids, vec![1, 0]);
        assert_eq!(model.id_to_token(0), Some("a"));
        assert_eq!(model.vocab_size(), 2);
    }
}
