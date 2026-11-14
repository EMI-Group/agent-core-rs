//! Model-to-vocabulary resolution and the token-counting implementations.
//!
//! [`ModelTokenCounter`] counts with the BPE vocabulary a model name resolves
//! to, and [`SelectedTokenizer`] is the counter the selection ladder hands to
//! the context engine (it can also be backed by a native artifact). Both report
//! provenance through [`TokenMeasurement`] and [`TokenizerSource`], and both
//! degrade to a four-characters-per-token estimate when an embedded vocabulary
//! cannot be built, so counting never fails and never needs the network.
//!
//! The module is private: its public types are re-exported from [`super`].
use std::sync::{Arc, OnceLock};

use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, TokenCounter};
use serde::{Deserialize, Serialize};
use tiktoken_rs::CoreBPE;

use super::native::NativeTokenizer;
use super::spec::ModelEncoding;

/// Characters per token used by the counters when a BPE is unavailable.
const BPE_FALLBACK_CHARS_PER_TOKEN: usize = 4;

/// How a token measurement was produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerSource {
    /// A native `tokenizer.json` artifact was loaded and used.
    NativeTokenizer,
    /// A family-level tokenizer was used for a model variant.
    FamilyTokenizerFallback,
    /// A `tiktoken` BPE vocabulary was used.
    Tiktoken,
    /// A `tiktoken` BPE was assumed after the model could not be resolved.
    #[default]
    TiktokenFallback,
    /// A characters-per-token estimate was used.
    StringLengthFallback,
}

impl TokenizerSource {
    /// Parses the `snake_case` name used in `TokenizerSpec::source`.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "native_tokenizer" => Some(Self::NativeTokenizer),
            "family_tokenizer_fallback" => Some(Self::FamilyTokenizerFallback),
            "tiktoken" => Some(Self::Tiktoken),
            "tiktoken_fallback" => Some(Self::TiktokenFallback),
            "string_length_fallback" => Some(Self::StringLengthFallback),
            _ => None,
        }
    }
}

/// The result of measuring one text or one conversation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TokenMeasurement {
    /// The token count.
    pub tokens: usize,
    /// Which strategy produced the count.
    pub source: TokenizerSource,
    /// Whether the count is an estimate rather than a native tokenizer result.
    pub estimated: bool,
    /// The tokenizer used, when one can be named.
    pub tokenizer: Option<String>,
    /// The model the count was resolved for.
    pub model: Option<String>,
    /// Why a fallback strategy was used.
    pub fallback_reason: Option<String>,
}

impl TokenMeasurement {
    /// Builds a measurement with the estimation flag derived from the source.
    fn new(
        tokens: usize,
        source: TokenizerSource,
        tokenizer: Option<String>,
        model: Option<String>,
        fallback_reason: Option<String>,
    ) -> Self {
        Self {
            tokens,
            source,
            estimated: source != TokenizerSource::NativeTokenizer,
            tokenizer,
            model,
            fallback_reason,
        }
    }
}

/// The shared `cl100k_base` BPE, built on first use.
static CL100K_BASE: OnceLock<Option<CoreBPE>> = OnceLock::new();
/// The shared `o200k_base` BPE, built on first use.
static O200K_BASE: OnceLock<Option<CoreBPE>> = OnceLock::new();
/// The shared `p50k_base` BPE, built on first use.
static P50K_BASE: OnceLock<Option<CoreBPE>> = OnceLock::new();
/// The shared `r50k_base` BPE, built on first use.
static R50K_BASE: OnceLock<Option<CoreBPE>> = OnceLock::new();

/// Returns the cached BPE for `encoding`, or `None` when it cannot be built.
fn bpe_for(encoding: ModelEncoding) -> Option<&'static CoreBPE> {
    let cell = match encoding {
        ModelEncoding::Cl100kBase => &CL100K_BASE,
        ModelEncoding::O200kBase => &O200K_BASE,
        ModelEncoding::P50kBase => &P50K_BASE,
        ModelEncoding::R50kBase => &R50K_BASE,
    };
    cell.get_or_init(|| {
        let built = match encoding {
            ModelEncoding::Cl100kBase => tiktoken_rs::cl100k_base(),
            ModelEncoding::O200kBase => tiktoken_rs::o200k_base(),
            ModelEncoding::P50kBase => tiktoken_rs::p50k_base(),
            ModelEncoding::R50kBase => tiktoken_rs::r50k_base(),
        };
        built
            .map_err(|error| {
                tracing::warn!(
                    target: "openjiuwen_context::tokenizers",
                    %error,
                    encoding = encoding.name(),
                    "BPE vocabulary unavailable; falling back to an approximate token count"
                );
                error
            })
            .ok()
    })
    .as_ref()
}

/// A resolved `tiktoken` encoding plus the metadata of how it was chosen.
pub(crate) struct ResolvedEncoding {
    /// The vocabulary to use.
    pub(crate) encoding: ModelEncoding,
    /// Whether the model was known or assumed to use `cl100k_base`.
    pub(crate) source: TokenizerSource,
    /// The vocabulary name.
    pub(crate) tokenizer: &'static str,
    /// Why a fallback vocabulary was chosen.
    pub(crate) fallback_reason: Option<String>,
}

/// Maps a model to a `tiktoken` encoding, assuming `cl100k_base` when unknown.
pub(crate) fn resolve_encoding(model: &str) -> ResolvedEncoding {
    let model = model.trim();
    if let Some(encoding) = ModelEncoding::from_model(model) {
        return ResolvedEncoding {
            encoding,
            source: TokenizerSource::Tiktoken,
            tokenizer: encoding.name(),
            fallback_reason: None,
        };
    }
    if !model.is_empty() && tiktoken_rs::get_bpe_from_model(model).is_ok() {
        if let Some(encoding) =
            tiktoken_rs::tokenizer::get_tokenizer(model).map(encoding_from_tokenizer)
        {
            return ResolvedEncoding {
                encoding,
                source: TokenizerSource::Tiktoken,
                tokenizer: encoding.name(),
                fallback_reason: None,
            };
        }
    }
    ResolvedEncoding {
        encoding: ModelEncoding::Cl100kBase,
        source: TokenizerSource::TiktokenFallback,
        tokenizer: ModelEncoding::Cl100kBase.name(),
        fallback_reason: Some("model_not_registered".to_owned()),
    }
}

/// Maps a `tiktoken` tokenizer enum onto the four supported BPE families.
fn encoding_from_tokenizer(tokenizer: tiktoken_rs::tokenizer::Tokenizer) -> ModelEncoding {
    use tiktoken_rs::tokenizer::Tokenizer;
    match tokenizer {
        Tokenizer::Cl100kBase => ModelEncoding::Cl100kBase,
        Tokenizer::O200kBase | Tokenizer::O200kHarmony => ModelEncoding::O200kBase,
        Tokenizer::P50kBase | Tokenizer::P50kEdit => ModelEncoding::P50kBase,
        Tokenizer::R50kBase | Tokenizer::Gpt2 => ModelEncoding::R50kBase,
    }
}

/// A token counter whose vocabulary is chosen from a model name.
///
/// `gpt-4o`, `gpt-4.1`, and `o1`-family models use `o200k_base`; `gpt-4`,
/// `gpt-3.5-turbo`, and the embedding models use `cl100k_base`. Models that
/// `tiktoken-rs` recognizes outside that table (legacy `text-davinci` and
/// `gpt2` models) use their own vocabulary. Anything unknown is counted with
/// `cl100k_base` and reported as [`TokenizerSource::TiktokenFallback`].
pub struct ModelTokenCounter {
    /// The trimmed model name, when one was given.
    model: Option<String>,
    /// The resolved vocabulary.
    encoding: ModelEncoding,
    /// How the vocabulary was chosen.
    source: TokenizerSource,
    /// The vocabulary name.
    tokenizer: String,
    /// Why a fallback vocabulary was chosen.
    fallback_reason: Option<String>,
}

impl std::fmt::Debug for ModelTokenCounter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelTokenCounter")
            .field("model", &self.model)
            .field("encoding", &self.encoding)
            .field("source", &self.source)
            .field("tokenizer", &self.tokenizer)
            .field("fallback_reason", &self.fallback_reason)
            .finish()
    }
}

impl Clone for ModelTokenCounter {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            encoding: self.encoding,
            source: self.source,
            tokenizer: self.tokenizer.clone(),
            fallback_reason: self.fallback_reason.clone(),
        }
    }
}

impl ModelTokenCounter {
    /// Creates a counter for `model`.
    ///
    /// The constructor never fails: an unknown model falls back to
    /// `cl100k_base`, and an unavailable vocabulary falls back to the
    /// four-characters-per-token estimate at counting time.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        let resolved = resolve_encoding(&model);
        Self {
            model: non_empty(model),
            encoding: resolved.encoding,
            source: resolved.source,
            tokenizer: resolved.tokenizer.to_owned(),
            fallback_reason: resolved.fallback_reason,
        }
    }

    /// The model name this counter was built for.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The resolved vocabulary.
    #[must_use]
    pub fn encoding(&self) -> ModelEncoding {
        self.encoding
    }

    /// The vocabulary name, for example `o200k_base`.
    #[must_use]
    pub fn tokenizer(&self) -> &str {
        &self.tokenizer
    }

    /// How the count is produced right now.
    ///
    /// A vocabulary that fails to initialize downgrades the reported source to
    /// [`TokenizerSource::TiktokenFallback`].
    #[must_use]
    pub fn source(&self) -> TokenizerSource {
        if bpe_for(self.encoding).is_some() {
            self.source
        } else {
            TokenizerSource::TiktokenFallback
        }
    }

    /// Whether the produced count is an estimate.
    #[must_use]
    pub fn estimated(&self) -> bool {
        self.source() != TokenizerSource::NativeTokenizer
    }

    /// Why the count uses a fallback vocabulary, when it does.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&str> {
        if bpe_for(self.encoding).is_none() {
            Some("tiktoken_unavailable")
        } else {
            self.fallback_reason.as_deref()
        }
    }

    /// Measures one text fragment and reports how the count was produced.
    #[must_use]
    pub fn measure(&self, text: &str) -> TokenMeasurement {
        self.measurement(self.count_text(text))
    }

    /// Measures a conversation and reports how the count was produced.
    #[must_use]
    pub fn measure_messages(&self, messages: &[BaseMessage]) -> TokenMeasurement {
        self.measurement(self.count_messages(messages))
    }

    /// Builds a measurement for an already-computed count.
    fn measurement(&self, tokens: usize) -> TokenMeasurement {
        TokenMeasurement::new(
            tokens,
            self.source(),
            Some(self.tokenizer.clone()),
            self.model.clone(),
            self.fallback_reason().map(str::to_owned),
        )
    }
}

impl Default for ModelTokenCounter {
    /// A counter for an unspecified model: `cl100k_base` with a fallback note.
    fn default() -> Self {
        Self::new("")
    }
}

impl TokenCounter for ModelTokenCounter {
    /// The number of BPE tokens, or the four-characters-per-token estimate when
    /// the vocabulary is unavailable. The empty string counts as `0`.
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        match bpe_for(self.encoding) {
            Some(bpe) => bpe.encode_ordinary(text).len(),
            None => ApproxTokenCounter::with_chars_per_token(BPE_FALLBACK_CHARS_PER_TOKEN)
                .count_text(text),
        }
    }
}

/// The counting strategy behind a [`SelectedTokenizer`].
#[derive(Debug, Clone)]
enum SelectedBacking {
    /// A `tiktoken` BPE vocabulary.
    Bpe(ModelEncoding),
    /// A characters-per-token estimate.
    Approx(ApproxTokenCounter),
    /// A native `tokenizer.json` artifact.
    Native(Arc<NativeTokenizer>),
}

/// The token counter chosen by [`TokenizerSelector`](super::selector::TokenizerSelector).
///
/// This is the concrete [`TokenCounter`] handed to the context engine. Paths
/// that cannot be satisfied offline (native artifacts) degrade to a `tiktoken`
/// vocabulary or, last, to a three-characters-per-token estimate, and always
/// report that through [`SelectedTokenizer::source`] and
/// [`SelectedTokenizer::fallback_reason`].
#[derive(Debug, Clone)]
pub struct SelectedTokenizer {
    /// The chosen strategy.
    backing: SelectedBacking,
    /// How the count is produced.
    source: TokenizerSource,
    /// The requested model.
    model: Option<String>,
    /// The tokenizer name, when one can be named.
    tokenizer: Option<String>,
    /// Why a fallback strategy was chosen.
    fallback_reason: Option<String>,
}

impl SelectedTokenizer {
    /// Builds a BPE-backed selection.
    pub(crate) fn from_bpe(
        encoding: ModelEncoding,
        source: TokenizerSource,
        model: Option<String>,
        tokenizer: Option<String>,
        fallback_reason: Option<String>,
    ) -> Self {
        Self {
            backing: SelectedBacking::Bpe(encoding),
            source,
            model,
            tokenizer,
            fallback_reason,
        }
    }

    /// Builds an estimate-backed selection.
    pub(crate) fn from_approx(
        counter: ApproxTokenCounter,
        source: TokenizerSource,
        model: Option<String>,
        fallback_reason: Option<String>,
    ) -> Self {
        Self {
            backing: SelectedBacking::Approx(counter),
            source,
            model,
            tokenizer: None,
            fallback_reason,
        }
    }

    /// Builds a native-artifact-backed selection.
    ///
    /// The count is exact for the artifact the caller loaded, so
    /// [`SelectedTokenizer::source`] reports
    /// [`TokenizerSource::NativeTokenizer`], [`SelectedTokenizer::estimated`]
    /// is `false` and there is no fallback reason. `tokenizer` names the
    /// artifact (its id, else the path it was read from).
    #[must_use]
    pub fn from_native(
        native: Arc<NativeTokenizer>,
        model: Option<String>,
        tokenizer: Option<String>,
    ) -> Self {
        Self::from_native_with(
            native,
            TokenizerSource::NativeTokenizer,
            model,
            tokenizer,
            None,
        )
    }

    /// Builds a native-artifact-backed selection with explicit provenance.
    ///
    /// The count itself is exact for the loaded artifact, but
    /// [`TokenizerSource::NativeTokenizer`] is reserved for a tokenizer that is
    /// the model's own: a family or one-hop fallback passes
    /// [`TokenizerSource::FamilyTokenizerFallback`] here, and
    /// [`SelectedTokenizer::estimated`] then stays `true` so telemetry never
    /// mistakes a sibling vocabulary for the model's own count.
    #[must_use]
    pub fn from_native_with(
        native: Arc<NativeTokenizer>,
        source: TokenizerSource,
        model: Option<String>,
        tokenizer: Option<String>,
        fallback_reason: Option<String>,
    ) -> Self {
        Self {
            backing: SelectedBacking::Native(native),
            source,
            model,
            tokenizer,
            fallback_reason,
        }
    }

    /// The model the selection was made for.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The tokenizer used, when one can be named.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&str> {
        self.tokenizer.as_deref()
    }

    /// How the count is produced right now.
    #[must_use]
    pub fn source(&self) -> TokenizerSource {
        match &self.backing {
            SelectedBacking::Bpe(encoding) if bpe_for(*encoding).is_none() => {
                TokenizerSource::TiktokenFallback
            }
            _ => self.source,
        }
    }

    /// Whether the produced count is an estimate.
    #[must_use]
    pub fn estimated(&self) -> bool {
        self.source() != TokenizerSource::NativeTokenizer
    }

    /// Why a fallback strategy was used, when one was.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&str> {
        match &self.backing {
            SelectedBacking::Bpe(encoding) if bpe_for(*encoding).is_none() => {
                Some("tiktoken_unavailable")
            }
            _ => self.fallback_reason.as_deref(),
        }
    }

    /// Measures one text fragment and reports how the count was produced.
    #[must_use]
    pub fn measure(&self, text: &str) -> TokenMeasurement {
        self.measurement(self.count_text(text))
    }

    /// Measures a conversation and reports how the count was produced.
    #[must_use]
    pub fn measure_messages(&self, messages: &[BaseMessage]) -> TokenMeasurement {
        self.measurement(self.count_messages(messages))
    }

    /// Builds a measurement for an already-computed count.
    fn measurement(&self, tokens: usize) -> TokenMeasurement {
        TokenMeasurement::new(
            tokens,
            self.source(),
            self.tokenizer.clone(),
            self.model.clone(),
            self.fallback_reason().map(str::to_owned),
        )
    }
}

impl TokenCounter for SelectedTokenizer {
    /// The number of tokens according to the selected strategy. The empty
    /// string counts as `0`.
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        match &self.backing {
            SelectedBacking::Bpe(encoding) => match bpe_for(*encoding) {
                Some(bpe) => bpe.encode_ordinary(text).len(),
                None => ApproxTokenCounter::with_chars_per_token(BPE_FALLBACK_CHARS_PER_TOKEN)
                    .count_text(text),
            },
            SelectedBacking::Approx(counter) => counter.count_text(text),
            SelectedBacking::Native(native) => native.count_text(text),
        }
    }
}

/// Returns `Some` for a non-blank name, trimmed.
pub(crate) fn non_empty(value: String) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_owned())
    }
}
