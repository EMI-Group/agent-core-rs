//! Pure-Rust decoding of native tokenizer artifacts.
//!
//! Two artifact kinds are supported: a `HuggingFace` `tokenizer.json` document and a `tiktoken` `.model` BPE vocabulary.
//! A `tokenizer.json` is read with `serde_json` and decoded into a model (`BPE`, `WordPiece`, a Unigram-lite branch, `WordLevel`) plus the normalizer, pre-tokenizer, added-token, post-processor and decoder pipelines, so encoding needs neither the `tokenizers` runtime nor any network access.
//! A `.model` vocabulary is decoded with `base64` by [`tiktoken_model::TiktokenModel`] (see [`NativeTokenizer::from_tiktoken_model`]) and shares the same [`NativeTokenizer`] surface.
//! Loading happens once per context/model binding; every later [`NativeTokenizer::encode`] is pure computation over the in-memory tables.
//! A declared post-processor adds its framing tokens on encode — `BERT`'s `[CLS] … [SEP]`, for instance — so a count includes the tokens the model really receives; with no post-processor declared, nothing is added.
//! Decoding runs the declared decoder node; an artifact that declares none (or only `Plain`) joins the token spellings with a single space, matching `HuggingFace`'s `tokenizers` and the Python reference.
//! [`NativeTokenizer::decode_model_level`] is the documented opt-in for the byte-aware model-level decoding that a byte-level artifact without a decoder node needs.
//! Shapes this crate cannot reproduce exactly (a node type it does not know, the `Nmt`/`Precompiled` normalization families, a template piece without declared ids) are recorded in [`NativeTokenizer::warnings`] and degrade to a coarser but working behaviour instead of failing the load.
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
use serde::Deserialize as _;
use serde_json::Value;

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
/// Encoding runs the artifact's own pipeline: normalize, cut out added tokens, pre-tokenize, encode each piece with the model, then apply the declared post-processor.
/// Decoding runs the other way: the declared decoder node, or the model's own concatenation when the artifact declares none.
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
    /// The post-processing pipeline that frames an encoding.
    post_processor: post_processing::PostProcessor,
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
        let value: Value = serde_json::from_str(json)
            .map_err(|error| Error::Validation(format!("tokenizer.json is malformed: {error}")))?;
        // `hf_json::TokenizerJson` types only the sections this loader reads, so the post-processor is taken from the raw value.
        let document = hf_json::TokenizerJson::deserialize(&value)
            .map_err(|error| Error::Validation(format!("tokenizer.json is malformed: {error}")))?;
        Self::from_document(document, value.get("post_processor"))
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
    fn from_document(
        document: hf_json::TokenizerJson,
        post_processor_json: Option<&Value>,
    ) -> Result<Self> {
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
        let post_processor =
            post_processing::PostProcessor::from_json(post_processor_json, &mut warnings);
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
            post_processor,
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
            post_processor: post_processing::PostProcessor::None,
            added: added_tokens::AddedTokens::default(),
            tiktoken: Some(model),
            tokenizer_id: None,
            warnings,
        }
    }

    /// Encodes `text` into token ids.
    ///
    /// Added tokens are matched first, so a special token always becomes exactly one id.
    /// The declared post-processor then frames the result, which is why `count_text` includes `[CLS]`/`[SEP]`-style tokens; an empty string produces no ids and therefore no framing either.
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
        self.post_processor.apply(&ids)
    }

    /// Decodes token ids back into text.
    ///
    /// Ids the artifact does not declare are skipped; special tokens are included, unlike `HuggingFace`'s `skip_special_tokens = true` default.
    /// The declared decoder node always wins, and an artifact that declares none (or only `Plain`) joins the token spellings with a single space, which is what `HuggingFace`'s `tokenizers` — and therefore the Python reference — does.
    /// [`NativeTokenizer::decode_model_level`] is the explicit opt-in for the model's own byte-aware reassembly, which a byte-level artifact without a decoder node needs to round-trip.
    /// A `tiktoken` vocabulary decodes with its own rules.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(tiktoken) = &self.tiktoken {
            return tiktoken.decode(ids);
        }
        if self.model.is_none() {
            return String::new();
        }
        self.decoder.decode(&self.token_spellings(ids))
    }

    /// Decodes token ids with the model's own rules, ignoring the declared decoder.
    ///
    /// This is an intentional deviation from `HuggingFace`'s `tokenizers`, whose `decode` only ever goes through the declared decoder node: a byte-level `BPE` or `Unigram` vocabulary that declares no decoder (or only `Plain`) encodes one id per byte, so only the model can strip continuation prefixes, reassemble `<0xXX>` byte runs and concatenate whole pieces back into the original text.
    /// A `WordPiece` model has no model-level rule — its continuation prefix is only unambiguous through the decoder node — so the declared decoder still answers, and with no decoder that is the same space join [`NativeTokenizer::decode`] returns.
    /// A `tiktoken` `.model` artifact carries no pipeline at all and decodes exactly like [`NativeTokenizer::decode`].
    #[must_use]
    pub fn decode_model_level(&self, ids: &[u32]) -> String {
        if let Some(tiktoken) = &self.tiktoken {
            return tiktoken.decode(ids);
        }
        let tokens = self.token_spellings(ids);
        self.model
            .as_ref()
            .and_then(|model| model.decode(&tokens))
            .unwrap_or_else(|| self.decoder.decode(&tokens))
    }

    /// The token spellings an id list resolves to, skipping ids the artifact does not declare.
    fn token_spellings(&self, ids: &[u32]) -> Vec<String> {
        let Some(model) = self.model.as_ref() else {
            return Vec::new();
        };
        ids.iter()
            .filter_map(|id| {
                self.added
                    .token(*id)
                    .or_else(|| model.id_to_token(*id))
                    .map(str::to_owned)
            })
            .collect()
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
    /// A warning always means a coarser behaviour, never a failed load: a node type this loader does not know is skipped, an `Nmt`/`Precompiled` normalizer leaves the text unnormalized, a `Split`/`Replace` regex that does not compile falls back to whitespace, or a template piece whose special token declares no ids is left out.
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

/// The framing tokens an artifact adds around an encoded sequence.
///
/// Only the single-sequence path is reproduced, because an encoding here is one flat id list: the pair templates and the per-token type ids have nowhere to live.
/// `TemplateProcessing` contributes its declared `single` template, `BertProcessing`/`RobertaProcessing` contribute their `[CLS] … [SEP]` pair, `ByteLevel` contributes nothing (it only trims offsets in the reference), and `Sequence` chains stages in order.
/// A shape this module does not reproduce, and a template piece whose special token declares no ids, is reported as a warning and left out, so a count stays comparable instead of silently gaining framing.
/// Nothing is declared means nothing is added, which keeps the artifact in charge of its own framing.
mod post_processing {
    use serde_json::Value;

    use super::hf_json::{bool_of, kind_of};

    /// A post-processing pipeline.
    #[derive(Debug, Clone, Default)]
    pub(crate) enum PostProcessor {
        /// Add nothing.
        #[default]
        None,
        /// Pieces assembled around the encoded sequence, in declaration order.
        Template(Vec<Piece>),
        /// Stages applied in order.
        Sequence(Vec<Self>),
    }

    /// One piece of a post-processing template.
    #[derive(Debug, Clone)]
    pub(crate) enum Piece {
        /// The encoded ids, in place.
        Sequence,
        /// Literal framing ids, such as the id of `[CLS]`.
        Ids(Vec<u32>),
    }

    impl PostProcessor {
        /// Decodes a `post_processor` section.
        pub(crate) fn from_json(value: Option<&Value>, warnings: &mut Vec<String>) -> Self {
            let Some(value) = value.filter(|value| !value.is_null()) else {
                return Self::None;
            };
            let Some(kind) = kind_of(value) else {
                warnings.push(
                    "post-processor declares no type; no framing tokens are added".to_owned(),
                );
                return Self::None;
            };
            match kind {
                "TemplateProcessing" => Self::template(value, warnings),
                "BertProcessing" | "RobertaProcessing" => Self::pair(value, kind, warnings),
                "ByteLevel" => Self::None,
                "Sequence" => {
                    let Some(entries) = value.get("processors").and_then(Value::as_array) else {
                        warnings.push(
                            "post-processor Sequence declares no `processors`; no framing tokens are added"
                                .to_owned(),
                        );
                        return Self::None;
                    };
                    Self::Sequence(
                        entries
                            .iter()
                            .map(|entry| Self::from_json(Some(entry), warnings))
                            .collect(),
                    )
                }
                other => {
                    warnings.push(format!(
                        "post-processor `{other}` is not implemented; no framing tokens are added"
                    ));
                    Self::None
                }
            }
        }

        /// Applies the framing to `ids`.
        pub(crate) fn apply(&self, ids: &[u32]) -> Vec<u32> {
            match self {
                Self::None => ids.to_vec(),
                Self::Template(pieces) => {
                    let mut out = Vec::with_capacity(ids.len() + pieces.len());
                    for piece in pieces {
                        match piece {
                            Piece::Sequence => out.extend_from_slice(ids),
                            Piece::Ids(framing) => out.extend_from_slice(framing),
                        }
                    }
                    out
                }
                Self::Sequence(stages) => stages
                    .iter()
                    .fold(ids.to_vec(), |current, stage| stage.apply(&current)),
            }
        }

        /// Decodes a `TemplateProcessing` node's single-sequence template.
        fn template(value: &Value, warnings: &mut Vec<String>) -> Self {
            if !bool_of(value, "add_special_tokens", true) {
                return Self::None;
            }
            let Some(entries) = value.get("single").and_then(Value::as_array) else {
                warnings.push(
                    "TemplateProcessing declares no single-sequence template; no framing tokens are added"
                        .to_owned(),
                );
                return Self::None;
            };
            let specials = value.get("special_tokens");
            let mut pieces = Vec::new();
            for entry in entries {
                if entry.get("Sequence").is_some() {
                    pieces.push(Piece::Sequence);
                    continue;
                }
                let Some(name) = entry
                    .get("SpecialToken")
                    .and_then(|special| special.get("id"))
                    .and_then(Value::as_str)
                else {
                    warnings.push(
                        "TemplateProcessing declares a template piece that is neither a `Sequence` nor a `SpecialToken`; it is skipped"
                            .to_owned(),
                    );
                    continue;
                };
                match special_ids(specials, name) {
                    Some(ids) => pieces.push(Piece::Ids(ids)),
                    None => warnings.push(format!(
                        "TemplateProcessing special token `{name}` declares no ids; it is skipped"
                    )),
                }
            }
            if pieces.is_empty() {
                Self::None
            } else {
                Self::Template(pieces)
            }
        }

        /// Decodes a `BertProcessing`/`RobertaProcessing` node, whose single-sequence form wraps the sequence in the declared cls/sep pair.
        fn pair(value: &Value, kind: &str, warnings: &mut Vec<String>) -> Self {
            // The offsets these nodes also carry (`trim_offsets`, `add_prefix_space`) never change an id.
            let (Some(cls), Some(sep)) = (framing_id(value, "cls"), framing_id(value, "sep"))
            else {
                warnings.push(format!(
                    "post-processor `{kind}` does not declare both `cls` and `sep`; no framing tokens are added"
                ));
                return Self::None;
            };
            Self::Template(vec![
                Piece::Ids(vec![cls]),
                Piece::Sequence,
                Piece::Ids(vec![sep]),
            ])
        }
    }

    /// The id of a `[token, id]` framing pair.
    fn framing_id(value: &Value, key: &str) -> Option<u32> {
        value
            .get(key)
            .and_then(Value::as_array)
            .and_then(|pair| pair.get(1))
            .and_then(Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
    }

    /// The ids a `TemplateProcessing` special token declares.
    fn special_ids(specials: Option<&Value>, name: &str) -> Option<Vec<u32>> {
        let ids: Vec<u32> = specials?
            .get(name)?
            .get("ids")?
            .as_array()?
            .iter()
            .filter_map(Value::as_u64)
            .filter_map(|id| u32::try_from(id).ok())
            .collect();
        (!ids.is_empty()).then_some(ids)
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

    fn parse(json: serde_json::Value) -> NativeTokenizer {
        NativeTokenizer::from_json_str(&json.to_string()).expect("fixture loads")
    }

    #[test]
    fn a_nameless_decoder_joins_spellings_and_the_model_level_opt_in_concatenates() {
        let tokenizer = parse(serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "##b": 2},
                "merges": [],
                "continuing_subword_prefix": "##"
            },
            "pre_tokenizer": {"type": "Whitespace"}
        }));
        // `HuggingFace`'s `tokenizers` joins the spellings with one space when no decoder is declared.
        assert_eq!(tokenizer.decode(&[0, 2]), "a ##b");
        assert_eq!(tokenizer.decode(&tokenizer.encode("a b")), "a b");
        // The byte-aware model-level decoding stays available as the explicit opt-in.
        assert_eq!(tokenizer.decode_model_level(&[0, 2]), "ab");
        assert_eq!(tokenizer.decode_model_level(&tokenizer.encode("a b")), "ab");
    }

    #[test]
    fn a_declared_decoder_wins_over_the_model() {
        let tokenizer = parse(serde_json::json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
            "decoder": {"type": "Replace", "pattern": {"String": "b"}, "content": "B"}
        }));
        assert_eq!(tokenizer.decode(&[0, 1]), "a B");
    }

    #[test]
    fn word_level_joins_with_spaces_and_concatenates_at_model_level() {
        let tokenizer = parse(serde_json::json!({
            "model": {"type": "WordLevel", "vocab": {"hello": 0, "world": 1}, "unk_token": "hello"},
            "pre_tokenizer": {"type": "Whitespace"}
        }));
        assert_eq!(tokenizer.decode(&[0, 1]), "hello world");
        assert_eq!(tokenizer.decode_model_level(&[0, 1]), "helloworld");
    }

    #[test]
    fn word_piece_keeps_the_plain_join_without_a_decoder() {
        let tokenizer = parse(serde_json::json!({
            "model": {"type": "WordPiece", "vocab": {"h": 0, "##i": 1}, "unk_token": "h"},
            "pre_tokenizer": {"type": "Whitespace"}
        }));
        // The continuation prefix is only unambiguous through the decoder node, so the model contributes nothing either way.
        assert_eq!(tokenizer.decode(&[0, 1]), "h ##i");
        assert_eq!(tokenizer.decode_model_level(&[0, 1]), "h ##i");
    }

    #[test]
    fn a_template_post_processor_frames_the_encoding() {
        let tokenizer = parse(serde_json::json!({
            "model": {
                "type": "WordLevel",
                "vocab": {"[CLS]": 0, "[SEP]": 1, "hello": 2, "world": 3},
                "unk_token": "[CLS]"
            },
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
                ],
                "pair": [],
                "special_tokens": {
                    "[CLS]": {"id": "[CLS]", "ids": [0], "tokens": ["[CLS]"]},
                    "[SEP]": {"id": "[SEP]", "ids": [1], "tokens": ["[SEP]"]}
                }
            }
        }));
        assert_eq!(tokenizer.encode("hello world"), vec![0, 2, 3, 1]);
        // The count includes the framing, because the model would receive those ids.
        assert_eq!(tokenizer.count_text("hello world"), 4);
        // An empty string produces no ids, so no framing is added either.
        assert_eq!(tokenizer.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn a_template_can_decline_to_add_special_tokens() {
        let tokenizer = parse(serde_json::json!({
            "model": {"type": "WordLevel", "vocab": {"hello": 0}, "unk_token": "hello"},
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {
                "type": "TemplateProcessing",
                "add_special_tokens": false,
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [9], "tokens": ["[CLS]"]}}
            }
        }));
        assert_eq!(tokenizer.encode("hello"), vec![0]);
        assert_eq!(tokenizer.warnings().len(), 0);
    }

    #[test]
    fn a_template_special_token_without_ids_is_skipped_with_a_warning() {
        let tokenizer = parse(serde_json::json!({
            "model": {"type": "WordLevel", "vocab": {"hello": 0}, "unk_token": "hello"},
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "special_tokens": {}
            }
        }));
        assert_eq!(tokenizer.encode("hello"), vec![0]);
        assert!(
            tokenizer
                .warnings()
                .iter()
                .any(|warning| warning.contains("[CLS]")),
            "the warning should name the token: {:?}",
            tokenizer.warnings()
        );
    }

    #[test]
    fn bert_and_roberta_post_processors_wrap_the_sequence() {
        let bert = parse(serde_json::json!({
            "model": {"type": "WordPiece", "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3}, "unk_token": "[UNK]"},
            "normalizer": {"type": "Lowercase"},
            "pre_tokenizer": {"type": "Whitespace"},
            "decoder": {"type": "WordPiece", "prefix": "##"},
            "post_processor": {"type": "BertProcessing", "cls": ["[CLS]", 1], "sep": ["[SEP]", 2]}
        }));
        assert_eq!(bert.encode("hello"), vec![1, 3, 2]);
        assert_eq!(bert.decode(&bert.encode("hello")), "[CLS] hello [SEP]");

        let roberta = parse(serde_json::json!({
            "model": {"type": "WordLevel", "vocab": {"<s>": 0, "</s>": 1, "hi": 2}, "unk_token": "<s>"},
            "pre_tokenizer": {"type": "Whitespace"},
            "decoder": {"type": "Fuse"},
            "post_processor": {
                "type": "RobertaProcessing",
                "cls": ["<s>", 0],
                "sep": ["</s>", 1],
                "trim_offsets": true,
                "add_prefix_space": true
            }
        }));
        assert_eq!(roberta.encode("hi"), vec![0, 2, 1]);
        assert_eq!(roberta.decode(&roberta.encode("hi")), "<s>hi</s>");
    }

    #[test]
    fn an_unsupported_post_processor_warns_instead_of_framing() {
        let tokenizer = parse(serde_json::json!({
            "model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"},
            "post_processor": {"type": "DialogFraming"}
        }));
        assert_eq!(tokenizer.encode("a"), vec![0]);
        assert!(
            tokenizer
                .warnings()
                .iter()
                .any(|warning| warning.contains("DialogFraming")),
            "the warning should name the node: {:?}",
            tokenizer.warnings()
        );
    }

    #[test]
    fn a_sequence_post_processor_without_stages_warns_instead_of_framing() {
        let tokenizer = parse(serde_json::json!({
            "model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"},
            "post_processor": {"type": "Sequence"}
        }));
        assert_eq!(tokenizer.encode("a"), vec![0]);
        assert!(
            tokenizer
                .warnings()
                .iter()
                .any(|warning| warning.contains("Sequence") && warning.contains("processors")),
            "the warning should name the missing stage list: {:?}",
            tokenizer.warnings()
        );
    }
}
