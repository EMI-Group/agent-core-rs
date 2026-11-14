//! Pure-Rust decoding of native tokenizer artifacts.
//!
//! Two artifact kinds are supported: a `HuggingFace` `tokenizer.json` document and a `tiktoken` `.model` BPE vocabulary.
//! A `tokenizer.json` is read with `serde_json` and decoded into a model (`BPE`, `WordPiece`, a Unigram-lite branch, `WordLevel`) plus the normalizer, pre-tokenizer, added-token and decoder pipelines, so encoding needs neither the `tokenizers` runtime nor any network access.
//! A `.model` vocabulary is decoded with `base64` by [`tiktoken_model::TiktokenModel`] (see [`NativeTokenizer::from_tiktoken_model`]) and shares the same [`NativeTokenizer`] surface.
//! Loading happens once per context/model binding; every later [`NativeTokenizer::encode`] is pure computation over the in-memory tables.
//! Shapes this crate cannot reproduce exactly (regex splits, the Unicode-normalization families, unsupported pipeline nodes) are recorded in [`NativeTokenizer::warnings`] and degrade to a coarser but working behaviour instead of failing the load.
//! A model type this crate does not know is refused with [`Error::Unsupported`], because guessing a vocabulary layout would silently bias every budget decision.

mod added_tokens;
mod bpe;
mod bytes;
mod decoder;
mod hf_json;
mod model;
mod normalizer;
mod pretokenizer;
pub mod tiktoken_model;
mod unigram;
mod wordpiece;

#[cfg(test)]
mod modeltests;

use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::TokenCounter;

/// The 256-entry byte↔unicode alphabet of byte-level vocabularies.
///
/// Entry `byte` is the printable scalar a byte-level artifact stores for that byte; the mapping is the GPT-2/Llama one, shared with `tiktoken`.
/// Tooling that builds a byte-level vocabulary can use it to spell the 256 base tokens.
#[must_use]
pub fn byte_level_alphabet() -> [char; 256] {
    *bytes::table()
}

/// The decoding model an artifact declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeTokenizerKind {
    /// Byte-pair encoding (`"type": "BPE"`).
    Bpe,
    /// `WordPiece` (`"type": "WordPiece"`).
    WordPiece,
    /// Unigram-lite (`"type": "Unigram"`).
    Unigram,
    /// Whole-piece vocabulary lookup (`"type": "WordLevel"`).
    WordLevel,
    /// A `tiktoken` `.model` byte-pair vocabulary, which is not a `tokenizer.json` model type.
    Tiktoken,
}

impl NativeTokenizerKind {
    /// The `tokenizer.json` model type name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bpe => "BPE",
            Self::WordPiece => "WordPiece",
            Self::Unigram => "Unigram",
            Self::WordLevel => "WordLevel",
            Self::Tiktoken => "Tiktoken",
        }
    }
}

/// A tokenizer decoded from a native artifact.
///
/// The type is immutable, `Send` and `Sync`, so one instance can be shared by every context bound to the same model.
/// Encoding runs the artifact's own pipeline: normalize, cut out added tokens, pre-tokenize, then encode each piece with the model.
/// A `tiktoken` `.model` vocabulary bypasses that pipeline and uses its own pre-tokenization and merge loop.
pub struct NativeTokenizer {
    /// The decoding model and its vocabularies, for a `tokenizer.json` artifact.
    model: Option<model::NativeModel>,
    /// The normalization pipeline.
    normalizer: normalizer::Normalizer,
    /// The pre-tokenization pipeline.
    pretokenizer: pretokenizer::Pretokenizer,
    /// The decoding pipeline.
    decoder: decoder::Decoder,
    /// Added and special tokens.
    added: added_tokens::AddedTokens,
    /// The `tiktoken` vocabulary, for a `.model` artifact.
    tiktoken: Option<tiktoken_model::TiktokenModel>,
    /// The artifact name or model id, when the JSON declares one.
    tokenizer_id: Option<String>,
    /// Everything the load could not reproduce exactly.
    warnings: Vec<String>,
    /// The number of model vocabulary entries plus added tokens.
    vocab_size: usize,
}

impl NativeTokenizer {
    /// Decodes a `tokenizer.json` document held in memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document is not valid JSON or declares no model or vocabulary, and [`Error::Unsupported`] when the model type is not one this loader decodes.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let document: hf_json::TokenizerJson = serde_json::from_str(json)
            .map_err(|error| Error::Validation(format!("tokenizer.json is malformed: {error}")))?;
        Self::from_document(document)
    }

    /// Reads and decodes a `tokenizer.json` file.
    ///
    /// The file is read once, at construction; counting never touches the filesystem afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the file does not exist, [`Error::Validation`] when it cannot be read or decoded, and [`Error::Unsupported`] for an unknown model type.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::NotFound(format!(
                    "native tokenizer artifact not found: {}",
                    path.display()
                ))
            } else {
                Error::Validation(format!(
                    "native tokenizer artifact {} could not be read: {error}",
                    path.display()
                ))
            }
        })?;
        Self::from_json_str(&text)
    }

    /// Builds the tokenizer from a decoded document.
    fn from_document(document: hf_json::TokenizerJson) -> Result<Self> {
        let hf_json::TokenizerJson {
            name,
            model_id,
            model: model_json,
            normalizer: normalizer_json,
            pre_tokenizer,
            decoder: decoder_json,
            added_tokens,
        } = document;
        let mut warnings = Vec::new();
        let model_json = model_json
            .as_ref()
            .ok_or_else(|| Error::Validation("tokenizer.json declares no model".to_owned()))?;
        let model = model::NativeModel::from_json(model_json, &mut warnings)?;
        let normalizer = normalizer::Normalizer::from_json(normalizer_json.as_ref(), &mut warnings);
        let pretokenizer =
            pretokenizer::Pretokenizer::from_json(pre_tokenizer.as_ref(), &mut warnings);
        let decoder = decoder::Decoder::from_json(decoder_json.as_ref(), &mut warnings);
        let added = added_tokens::AddedTokens::new(added_tokens.unwrap_or_default(), &mut warnings);
        for warning in &warnings {
            tracing::warn!(
                target: "openjiuwen_context::tokenizers::native",
                %warning,
                "tokenizer.json feature degraded"
            );
        }
        Ok(Self {
            vocab_size: model.vocab_size() + added.len(),
            model: Some(model),
            normalizer,
            pretokenizer,
            decoder,
            added,
            tiktoken: None,
            tokenizer_id: name.or(model_id),
            warnings,
        })
    }

    /// Decodes a `tiktoken` `.model` document held in memory.
    ///
    /// The vocabulary uses the Kimi pre-tokenization rule and the built-in Kimi K2 special-token fallback, because a document held in memory carries no model name.
    /// Use [`NativeTokenizer::from_tiktoken_model_file`] for the sibling `tokenizer_config.json` and the K3 selection, or [`NativeTokenizer::from_tiktoken`] with [`tiktoken_model::TiktokenSpecials::for_model`] to choose explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the malformed line when the document is not a valid `.model` vocabulary.
    pub fn from_tiktoken_model(text: &str) -> Result<Self> {
        let model = tiktoken_model::TiktokenModel::from_text(text)?;
        let specials = tiktoken_model::TiktokenSpecials::for_model("", model.mergeable_count());
        Ok(Self::from_tiktoken(model.with_special_tokens(specials)))
    }

    /// Reads and decodes a `tiktoken` `.model` file.
    ///
    /// Special tokens come from the sibling `tokenizer_config.json` when it declares an `added_tokens_decoder`, else from the built-in Kimi fallback selected by the file stem.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the file does not exist and [`Error::Validation`] when it cannot be read or parsed.
    pub fn from_tiktoken_model_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_tiktoken(
            tiktoken_model::TiktokenModel::from_file(path)?,
        ))
    }

    /// Reads a `tiktoken` `.model` file and selects its special tokens by `model`.
    ///
    /// A model name containing `kimi-k3` uses the K3 fallback map, anything else K2; the sibling `tokenizer_config.json` still wins when it declares tokens.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the file does not exist and [`Error::Validation`] when it cannot be read or parsed.
    pub fn from_tiktoken_model_file_with_model(
        path: impl AsRef<Path>,
        model: &str,
    ) -> Result<Self> {
        Ok(Self::from_tiktoken(
            tiktoken_model::TiktokenModel::from_file_with_model(path, model)?,
        ))
    }

    /// Wraps an already-loaded `tiktoken` vocabulary.
    ///
    /// This is the seam for callers that attach their own [`tiktoken_model::TiktokenSpecials`] or [`tiktoken_model::PretokenizePattern`] before handing the vocabulary to the crate.
    #[must_use]
    pub fn from_tiktoken(model: tiktoken_model::TiktokenModel) -> Self {
        let mut warnings = model.warnings().to_vec();
        let mut pipeline_warnings = Vec::new();
        let normalizer = normalizer::Normalizer::from_json(None, &mut pipeline_warnings);
        let pretokenizer = pretokenizer::Pretokenizer::from_json(None, &mut pipeline_warnings);
        let decoder = decoder::Decoder::from_json(None, &mut pipeline_warnings);
        warnings.extend(pipeline_warnings);
        for warning in &warnings {
            tracing::warn!(
                target: "openjiuwen_context::tokenizers::native",
                %warning,
                "tiktoken model feature degraded"
            );
        }
        Self {
            vocab_size: model.vocab_size(),
            model: None,
            normalizer,
            pretokenizer,
            decoder,
            added: added_tokens::AddedTokens::default(),
            tiktoken: Some(model),
            tokenizer_id: None,
            warnings,
        }
    }

    /// Encodes `text` into token ids.
    ///
    /// Added tokens are matched first, so a special token always becomes exactly one id.
    /// A `tiktoken` vocabulary runs its own special-token split, pre-tokenization and merge loop.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(tiktoken) = &self.tiktoken {
            return tiktoken.encode(text);
        }
        let mut ids = Vec::new();
        if text.is_empty() {
            return ids;
        }
        let Some(model) = self.model.as_ref() else {
            return ids;
        };
        let normalized = self.normalizer.normalize(text);
        for chunk in self.added.split(&normalized) {
            match chunk {
                added_tokens::Chunk::Added { id } => ids.push(id),
                added_tokens::Chunk::Text(text) => {
                    for piece in self.pretokenizer.apply(text) {
                        model.encode_piece(&piece, &mut ids);
                    }
                }
            }
        }
        ids
    }

    /// Decodes token ids back into text.
    ///
    /// Ids the artifact does not declare are skipped; special tokens are included, unlike `HuggingFace`'s `skip_special_tokens = true` default.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(tiktoken) = &self.tiktoken {
            return tiktoken.decode(ids);
        }
        let Some(model) = self.model.as_ref() else {
            return String::new();
        };
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|id| {
                self.added
                    .token(*id)
                    .or_else(|| model.id_to_token(*id))
                    .map(str::to_owned)
            })
            .collect();
        self.decoder.decode(&tokens)
    }

    /// The number of tokens the artifact declares: model vocabulary plus added tokens.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The decoding model the artifact declares.
    #[must_use]
    pub const fn kind(&self) -> NativeTokenizerKind {
        let Some(model) = &self.model else {
            return NativeTokenizerKind::Tiktoken;
        };
        match model {
            model::NativeModel::Bpe(_) => NativeTokenizerKind::Bpe,
            model::NativeModel::WordPiece(_) => NativeTokenizerKind::WordPiece,
            model::NativeModel::Unigram(_) => NativeTokenizerKind::Unigram,
            model::NativeModel::WordLevel(_) => NativeTokenizerKind::WordLevel,
        }
    }

    /// Everything the load could not reproduce exactly, in the order it was found.
    ///
    /// A warning always means a coarser behaviour, never a failed load: a regex split became a whitespace split, or an unsupported normalizer was skipped.
    #[must_use]
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The artifact name or model id, when the JSON declares one.
    ///
    /// A `tiktoken` `.model` artifact carries no name, so this is `None` for it.
    #[must_use]
    pub fn tokenizer_id(&self) -> Option<&str> {
        self.tokenizer_id.as_deref()
    }

    /// The contents of the tokens the artifact declares special, sorted.
    #[must_use]
    pub fn special_tokens(&self) -> Vec<&str> {
        if let Some(tiktoken) = &self.tiktoken {
            return tiktoken.special_tokens().texts();
        }
        self.added.special_tokens()
    }
}

impl std::fmt::Debug for NativeTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeTokenizer")
            .field("kind", &self.kind())
            .field("vocab_size", &self.vocab_size)
            .field("tokenizer_id", &self.tokenizer_id)
            .field("warnings", &self.warnings)
            .finish_non_exhaustive()
    }
}

impl TokenCounter for NativeTokenizer {
    /// The number of ids [`NativeTokenizer::encode`] produces; the empty string counts as `0`.
    ///
    /// Message framing uses the [`TokenCounter`] default (a fixed four-token chat overhead plus the text) instead of Python's `<|start|>role` convention, so a native count stays comparable with every other counter in the workspace.
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        self.encode(text).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_names_match_the_artifact_types() {
        assert_eq!(NativeTokenizerKind::Bpe.as_str(), "BPE");
        assert_eq!(NativeTokenizerKind::WordPiece.as_str(), "WordPiece");
        assert_eq!(NativeTokenizerKind::Unigram.as_str(), "Unigram");
        assert_eq!(NativeTokenizerKind::WordLevel.as_str(), "WordLevel");
        assert_eq!(NativeTokenizerKind::Tiktoken.as_str(), "Tiktoken");
    }

    #[test]
    fn byte_level_alphabet_is_the_gpt2_table() {
        let alphabet = byte_level_alphabet();
        assert_eq!(alphabet[usize::from(b'a')], 'a');
        assert_eq!(alphabet[usize::from(b' ')], '\u{0120}');
        assert_eq!(alphabet.len(), 256);
    }

    #[test]
    fn documents_without_a_model_are_rejected() {
        let error = NativeTokenizer::from_json_str("{}").expect_err("no model");
        assert!(matches!(error, Error::Validation(_)));
    }

    #[test]
    fn malformed_json_is_a_validation_error() {
        let error = NativeTokenizer::from_json_str("not json").expect_err("malformed");
        assert!(matches!(error, Error::Validation(_)));
    }
}
