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

    /// Decodes token spellings with the model's own rules, when the model declares any.
    ///
    /// `BPE`, `Unigram` and `WordLevel` spellings concatenate, and the two subword models reassemble `<0xXX>` byte runs when they declare `byte_fallback`, so an artifact that declares no decoder still round-trips.
    /// `WordPiece` answers `None`: its continuation prefix only becomes unambiguous through the declared decoder node, which owns the glue-and-space decision.
    pub(crate) fn decode(&self, tokens: &[String]) -> Option<String> {
        match self {
            Self::Bpe(model) => Some(model.decode(tokens)),
            Self::WordPiece(_) => None,
            Self::Unigram(model) => Some(model.decode(tokens)),
            Self::WordLevel(_) => Some(WordLevelModel::decode(tokens)),
        }
    }
}

/// The byte a `<0xXX>` fallback token stands for.
///
/// `HuggingFace` writes the byte as two upper-case hex digits, but artifacts produced by other tooling use lower case, so both spellings are accepted.
/// Anything else — a longer token, a non-hex digit, a real vocabulary entry that merely starts with `<0x` — is not a byte token.
pub(crate) fn hex_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Appends the bytes a run of `<0xXX>` tokens collected, clearing the run.
///
/// Decoding is lossy and happens only once a run ends, so a character whose UTF-8 bytes span several tokens is reassembled before the decode and an incomplete run becomes `U+FFFD` instead of failing.
pub(crate) fn flush_byte_run(bytes: &mut Vec<u8>, out: &mut String) {
    if bytes.is_empty() {
        return;
    }
    out.push_str(&String::from_utf8_lossy(bytes));
    bytes.clear();
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

    /// Decodes token spellings into the text they were cut from.
    ///
    /// A whole-piece vocabulary stores the piece verbatim, so decoding is plain concatenation; a decoder node that wants spaces in between has to be declared by the artifact.
    fn decode(tokens: &[String]) -> String {
        tokens.concat()
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

    #[test]
    fn word_level_decodes_by_concatenation() {
        let mut warnings = Vec::new();
        let model = NativeModel::from_json(
            &json!({"type": "WordLevel", "vocab": {"hello": 0, "world": 1}, "unk_token": "hello"}),
            &mut warnings,
        )
        .expect("WordLevel fixture");
        let tokens = vec!["hello".to_owned(), "world".to_owned()];
        assert_eq!(model.decode(&tokens).as_deref(), Some("helloworld"));
    }

    #[test]
    fn word_piece_has_no_model_level_decode() {
        let mut warnings = Vec::new();
        let model = NativeModel::from_json(
            &json!({"type": "WordPiece", "vocab": {"hello": 0}, "unk_token": "hello"}),
            &mut warnings,
        )
        .expect("WordPiece fixture");
        assert!(model.decode(&["hello".to_owned()]).is_none());
    }

    #[test]
    fn hex_byte_tokens_are_two_hex_digits() {
        assert_eq!(hex_byte_token("<0x41>"), Some(0x41));
        assert_eq!(hex_byte_token("<0xff>"), Some(0xFF));
        assert_eq!(hex_byte_token("<0x0A>"), Some(0x0A));
        assert_eq!(hex_byte_token("<0x4>"), None);
        assert_eq!(hex_byte_token("<0x4142>"), None);
        assert_eq!(hex_byte_token("<0xZZ>"), None);
        assert_eq!(hex_byte_token("<unk>"), None);
    }

    #[test]
    fn a_byte_run_reassembles_a_split_character() {
        let mut bytes = Vec::new();
        let mut out = String::new();
        // `é` is the two bytes `0xC3 0xA9`.
        bytes.push(0xC3);
        flush_byte_run(&mut bytes, &mut out);
        assert_eq!(out, "\u{FFFD}");
        bytes.push(0xC3);
        bytes.push(0xA9);
        flush_byte_run(&mut bytes, &mut out);
        assert!(out.ends_with('é'));
        assert_eq!(bytes, Vec::<u8>::new());
        // A trailing partial run is replaced instead of panicking.
        bytes.push(0xE4);
        flush_byte_run(&mut bytes, &mut out);
        assert!(out.ends_with('\u{FFFD}'));
    }
}
