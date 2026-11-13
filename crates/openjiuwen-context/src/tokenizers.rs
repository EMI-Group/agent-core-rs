//! Model-aware tokenizer registry and token accounting.
//!
//! Python keeps a registry of tokenizer artifacts in
//! `core.context_engine.token` so a context engine can pick the exact counter a
//! model needs. The Rust port keeps the same seam but resolves tokenizers
//! offline:
//!
//! - [`ModelEncoding`] maps well-known model names to one of the four `OpenAI`
//!   BPE vocabularies embedded in `tiktoken-rs`.
//! - [`TokenizerRegistry`] holds [`TokenizerSpec`] entries so deployments can
//!   declare a provider/family tokenizer without changing code.
//! - [`TokenizerSelector`] walks the selection ladder: explicit spec, registry
//!   match, `tiktoken` encoding, then a conservative string-length estimate.
//! - [`ModelTokenCounter`] and [`SelectedTokenizer`] implement the
//!   [`TokenCounter`] seam used by the rest of the workspace.
//!
//! Native `tokenizer.json` artifacts need the `HuggingFace` `tokenizers` runtime,
//! which is not available offline; [`load_native_tokenizer`] reports that as a
//! typed [`Error::Unsupported`] instead of silently degrading.

use std::sync::OnceLock;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, TokenCounter};
use serde::{Deserialize, Serialize};
use tiktoken_rs::CoreBPE;

/// Characters per token used by [`ModelTokenCounter`] when a BPE is unavailable.
const BPE_FALLBACK_CHARS_PER_TOKEN: usize = 4;

/// Characters per token used by the selector's last-resort estimate.
///
/// Three characters per token is deliberately pessimistic: callers that reach
/// this rung know nothing about the model, so over-counting keeps the context
/// window safe.
const STRING_LENGTH_CHARS_PER_TOKEN: usize = 3;

/// One of the BPE vocabulary families embedded in `tiktoken-rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelEncoding {
    /// The `cl100k_base` vocabulary (`gpt-4`, `gpt-3.5-turbo`, embeddings).
    Cl100kBase,
    /// The `o200k_base` vocabulary (`gpt-4o`, `gpt-4.1`, `o1`/`o3`).
    O200kBase,
    /// The `p50k_base` vocabulary (legacy code and `text-davinci` models).
    P50kBase,
    /// The `r50k_base` vocabulary (`gpt2` and the first text models).
    R50kBase,
}

impl ModelEncoding {
    /// The canonical `tiktoken-rs` vocabulary name, for example `cl100k_base`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cl100kBase => "cl100k_base",
            Self::O200kBase => "o200k_base",
            Self::P50kBase => "p50k_base",
            Self::R50kBase => "r50k_base",
        }
    }

    /// Resolves a model name to its encoding.
    ///
    /// Matching is case-insensitive and ignores surrounding whitespace. Exact
    /// names are checked first, then the deployment prefixes seen in `OpenAI`
    /// model ids (`gpt-4o-2024-08-06`, `ft:gpt-4-0613`, `o1-preview`). An
    /// unknown model returns `None` so the caller can fall back.
    #[must_use]
    pub fn from_model(model: &str) -> Option<Self> {
        let model = model.trim().to_lowercase();
        if model.is_empty() {
            return None;
        }
        match model.as_str() {
            "gpt-4"
            | "gpt-4-turbo"
            | "gpt-3.5-turbo"
            | "gpt-3.5"
            | "gpt-35-turbo"
            | "text-embedding-ada-002"
            | "text-embedding-3-small"
            | "text-embedding-3-large" => Some(Self::Cl100kBase),
            "gpt-4o" | "gpt-4o-mini" | "gpt-4.1" | "gpt-4.5" | "gpt-5" | "o1" | "o3"
            | "o4-mini" => Some(Self::O200kBase),
            _ => {
                if has_prefix(
                    &model,
                    &[
                        "o1-",
                        "o3-",
                        "o4-mini-",
                        "gpt-5-",
                        "gpt-4.5-",
                        "gpt-4.1-",
                        "chatgpt-4o-",
                        "gpt-4o-",
                        "ft:gpt-4o",
                    ],
                ) {
                    Some(Self::O200kBase)
                } else if has_prefix(
                    &model,
                    &[
                        "gpt-4-",
                        "gpt-3.5-turbo-",
                        "gpt-35-turbo-",
                        "ft:gpt-4",
                        "ft:gpt-3.5-turbo",
                    ],
                ) {
                    Some(Self::Cl100kBase)
                } else {
                    None
                }
            }
        }
    }
}

/// Whether `value` starts with any of `prefixes`.
fn has_prefix(value: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| value.starts_with(prefix))
}

/// Which tokenizer engine a [`TokenizerSpec`] asks for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerEngine {
    /// Use the best offline engine available (`tiktoken`, then the estimate).
    #[default]
    Auto,
    /// Use a `tiktoken` BPE vocabulary.
    Tiktoken,
    /// Use a native `tokenizer.json` artifact (deferred offline).
    Native,
}

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
    /// Parses the `snake_case` name used in [`TokenizerSpec::source`].
    fn from_name(name: &str) -> Option<Self> {
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

/// A declared tokenizer for one provider/model or one model family.
///
/// Specs are declarative metadata: the offline resolver can satisfy `tiktoken`
/// and `auto` engines, but a `native` engine needs an artifact runtime that is
/// deferred (see [`load_native_tokenizer`]). Every field is optional except the
/// provider/model pair, and unknown JSON fields are ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TokenizerSpec {
    /// Provider id, for example `openai` or `ollama`.
    pub provider: String,
    /// Exact model id, or the base of a model family.
    pub model: String,
    /// Optional family name matched against model variants.
    pub family: Option<String>,
    /// Engine requested by this entry.
    pub engine: TokenizerEngine,
    /// Tokenizer artifact id, carried through as measurement metadata.
    pub tokenizer_id: Option<String>,
    /// Declared measurement source (`tiktoken`, `native_tokenizer`, ...).
    pub source: Option<String>,
    /// Tokenizer artifact revision.
    pub revision: Option<String>,
    /// Local path of a native tokenizer artifact.
    pub artifact_path: Option<String>,
    /// Expected SHA-256 of a native tokenizer artifact.
    pub sha256: Option<String>,
}

impl TokenizerSpec {
    /// Creates a spec for one provider and model with the default engine.
    #[must_use]
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            ..Self::default()
        }
    }

    /// Validates that the entry can identify a tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when both `provider` and `model` are empty.
    pub fn validate(&self) -> Result<()> {
        if self.provider.trim().is_empty() && self.model.trim().is_empty() {
            return Err(Error::Config(
                "tokenizer spec requires a non-empty provider or model".to_owned(),
            ));
        }
        Ok(())
    }
}

/// How a registry entry matched a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerMatchKind {
    /// The provider and model matched exactly.
    Exact,
    /// The model is a variant of a registered model or family.
    Family,
}

/// A resolved registry entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TokenizerMatch {
    /// The entry that matched.
    pub spec: TokenizerSpec,
    /// Whether the match was exact or family-level.
    pub kind: TokenizerMatchKind,
}

impl Default for TokenizerMatch {
    fn default() -> Self {
        Self {
            spec: TokenizerSpec::default(),
            kind: TokenizerMatchKind::Exact,
        }
    }
}

/// A list of declared tokenizers resolved by provider and model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TokenizerRegistry {
    /// Registered specs, in registration order.
    pub specs: Vec<TokenizerSpec>,
}

impl TokenizerRegistry {
    /// Creates a registry from a list of specs.
    #[must_use]
    pub fn new(specs: Vec<TokenizerSpec>) -> Self {
        Self { specs }
    }

    /// Registers one spec; a later registration wins over an earlier one.
    pub fn register(&mut self, spec: TokenizerSpec) {
        self.specs.push(spec);
    }

    /// Whether the registry holds no specs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// The number of registered specs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Resolves a provider/model request.
    ///
    /// Provider and model are trimmed and lowercased. An exact entry always
    /// wins; otherwise the family step matches a requested model against a
    /// registered model or family, where the request must continue with one of
    /// `-`, `_`, `:`, or `.` (a registered `family` also matches when it equals
    /// the request). The longest matching base wins; if two distinct entries
    /// tie for that length the registry refuses to guess and returns `None`.
    #[must_use]
    pub fn resolve(&self, provider: &str, model: &str) -> Option<TokenizerMatch> {
        let provider = normalize(provider);
        let model = normalize(model);
        if model.is_empty() {
            return None;
        }

        for spec in self.specs.iter().rev() {
            if normalize(&spec.provider) == provider && normalize(&spec.model) == model {
                return Some(TokenizerMatch {
                    spec: spec.clone(),
                    kind: TokenizerMatchKind::Exact,
                });
            }
        }

        let mut best: Option<&TokenizerSpec> = None;
        let mut best_len = 0usize;
        let mut tied = false;
        let mut seen: Vec<(String, String)> = Vec::new();
        for spec in self.specs.iter().rev() {
            if normalize(&spec.provider) != provider {
                continue;
            }
            // A re-registration of the same base is the last one; never count
            // it twice, which would turn every duplicate into an ambiguity.
            let identity = (
                normalize(&spec.model),
                spec.family.as_deref().map(normalize).unwrap_or_default(),
            );
            if seen.contains(&identity) {
                continue;
            }
            seen.push(identity);

            let entry_len = family_base_len(&model, spec);
            if entry_len == 0 {
                continue;
            }
            if entry_len > best_len {
                best = Some(spec);
                best_len = entry_len;
                tied = false;
            } else if entry_len == best_len {
                tied = true;
            }
        }

        if tied {
            return None;
        }
        best.map(|spec| TokenizerMatch {
            spec: spec.clone(),
            kind: TokenizerMatchKind::Family,
        })
    }
}

/// Trims and lowercases a registry key component.
fn normalize(value: &str) -> String {
    value.trim().to_lowercase()
}

/// The length of the longest base through which `requested` descends from `spec`.
///
/// `0` means the spec does not describe a family ancestor of the request.
fn family_base_len(requested: &str, spec: &TokenizerSpec) -> usize {
    let mut base_len = 0usize;
    let model = normalize(&spec.model);
    if !model.is_empty() && requested.len() > model.len() && is_variant_of(requested, &model) {
        base_len = model.len();
    }
    if let Some(family) = spec.family.as_deref() {
        let family = normalize(family);
        if !family.is_empty() && is_variant_of(requested, &family) {
            base_len = base_len.max(family.len());
        }
    }
    base_len
}

/// Whether `requested` is `base` itself or a variant separated by punctuation.
fn is_variant_of(requested: &str, base: &str) -> bool {
    if requested == base {
        return true;
    }
    match requested.strip_prefix(base) {
        Some(rest) => rest
            .chars()
            .next()
            .is_some_and(|ch| matches!(ch, '-' | '_' | ':' | '.')),
        None => false,
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
struct ResolvedEncoding {
    /// The vocabulary to use.
    encoding: ModelEncoding,
    /// Whether the model was known or assumed to use `cl100k_base`.
    source: TokenizerSource,
    /// The vocabulary name.
    tokenizer: &'static str,
    /// Why a fallback vocabulary was chosen.
    fallback_reason: Option<String>,
}

/// Maps a model to a `tiktoken` encoding, assuming `cl100k_base` when unknown.
fn resolve_encoding(model: &str) -> ResolvedEncoding {
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
}

/// The token counter chosen by [`TokenizerSelector`].
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
    fn from_bpe(
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
    fn from_approx(
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
        }
    }
}

/// Chooses a [`SelectedTokenizer`] from an optional explicit spec and registry.
///
/// The ladder is:
/// 1. the explicit [`TokenizerSpec`], when one is set — the registry is never
///    consulted;
/// 2. a [`TokenizerRegistry`] hit (exact or family);
/// 3. the model's `tiktoken` vocabulary;
/// 4. a three-characters-per-token estimate.
///
/// A requested [`TokenizerEngine::Native`] cannot be satisfied offline, so it
/// falls through to steps 3 and 4 with the reason
/// `native_tokenizer_unavailable`.
#[derive(Debug, Clone, Default)]
pub struct TokenizerSelector {
    /// Provider used for registry lookup.
    provider: String,
    /// Model used for registry lookup and encoding resolution.
    model: String,
    /// Explicit spec that overrides the registry.
    spec: Option<TokenizerSpec>,
    /// Registry consulted when no explicit spec is set.
    registry: Option<TokenizerRegistry>,
}

impl TokenizerSelector {
    /// Creates a selector.
    ///
    /// `provider` and `model` may be empty; the selector still returns a working
    /// counter. `spec` wins over `registry` when both are present.
    #[must_use]
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        spec: Option<TokenizerSpec>,
        registry: Option<TokenizerRegistry>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            spec,
            registry,
        }
    }

    /// Sets the explicit spec, which overrides any registry entry.
    #[must_use]
    pub fn with_spec(mut self, spec: TokenizerSpec) -> Self {
        self.spec = Some(spec);
        self
    }

    /// Sets the registry consulted when no explicit spec is set.
    #[must_use]
    pub fn with_registry(mut self, registry: TokenizerRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// The provider used for registry lookup.
    #[must_use]
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The model used for lookup and encoding resolution.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Selects the tokenizer for this provider and model.
    #[must_use]
    pub fn select(&self) -> SelectedTokenizer {
        if let Some(spec) = &self.spec {
            if spec.engine == TokenizerEngine::Native {
                return self.native_selection(Some(spec));
            }
            return self.spec_selection(spec);
        }

        if let Some(hit) = self
            .registry
            .as_ref()
            .and_then(|registry| registry.resolve(&self.provider, &self.model))
        {
            match hit.kind {
                TokenizerMatchKind::Exact => {
                    if hit.spec.engine == TokenizerEngine::Native {
                        return self.native_selection(Some(&hit.spec));
                    }
                    return self.spec_selection(&hit.spec);
                }
                TokenizerMatchKind::Family => {
                    if let Some(selected) = self.tiktoken_selection(
                        TokenizerSource::FamilyTokenizerFallback,
                        Some("model_variant_tokenizer_family"),
                    ) {
                        return selected;
                    }
                    return self.string_length_selection(self.default_reason());
                }
            }
        }

        self.tiktoken_selection(TokenizerSource::Tiktoken, None)
            .unwrap_or_else(|| self.string_length_selection(self.default_reason()))
    }

    /// Builds the selection for an explicit (non-native) spec.
    fn spec_selection(&self, spec: &TokenizerSpec) -> SelectedTokenizer {
        let model = self.effective_model(spec);
        let resolved = resolve_encoding(&model);
        let source = spec
            .source
            .as_deref()
            .and_then(TokenizerSource::from_name)
            .unwrap_or(resolved.source);
        let tokenizer = spec
            .tokenizer_id
            .clone()
            .or_else(|| Some(resolved.tokenizer.to_owned()));
        SelectedTokenizer::from_bpe(
            resolved.encoding,
            source,
            non_empty(model),
            tokenizer,
            resolved.fallback_reason,
        )
    }

    /// Builds the selection when a native engine was requested but cannot be
    /// satisfied offline.
    fn native_selection(&self, spec: Option<&TokenizerSpec>) -> SelectedTokenizer {
        let model = spec.map_or_else(|| self.model.clone(), |spec| self.effective_model(spec));
        let reason = Some("native_tokenizer_unavailable".to_owned());
        let resolved = resolve_encoding(&model);
        if resolved.source == TokenizerSource::Tiktoken {
            SelectedTokenizer::from_bpe(
                resolved.encoding,
                TokenizerSource::TiktokenFallback,
                non_empty(model),
                Some(resolved.tokenizer.to_owned()),
                reason,
            )
        } else {
            SelectedTokenizer::from_approx(
                ApproxTokenCounter::with_chars_per_token(STRING_LENGTH_CHARS_PER_TOKEN),
                TokenizerSource::StringLengthFallback,
                non_empty(model),
                reason,
            )
        }
    }

    /// The model a spec resolves for: its own, else the selector's request.
    fn effective_model(&self, spec: &TokenizerSpec) -> String {
        if spec.model.trim().is_empty() {
            self.model.clone()
        } else {
            spec.model.clone()
        }
    }

    /// Step 3 of the ladder: a `tiktoken` vocabulary for the requested model.
    fn tiktoken_selection(
        &self,
        source: TokenizerSource,
        reason: Option<&str>,
    ) -> Option<SelectedTokenizer> {
        let resolved = resolve_encoding(&self.model);
        if resolved.source != TokenizerSource::Tiktoken {
            return None;
        }
        Some(SelectedTokenizer::from_bpe(
            resolved.encoding,
            source,
            non_empty(self.model.clone()),
            Some(resolved.tokenizer.to_owned()),
            reason.map(str::to_owned),
        ))
    }

    /// Step 4 of the ladder: the conservative characters-per-token estimate.
    fn string_length_selection(&self, reason: &str) -> SelectedTokenizer {
        SelectedTokenizer::from_approx(
            ApproxTokenCounter::with_chars_per_token(STRING_LENGTH_CHARS_PER_TOKEN),
            TokenizerSource::StringLengthFallback,
            non_empty(self.model.clone()),
            Some(reason.to_owned()),
        )
    }

    /// Why the estimate was used.
    fn default_reason(&self) -> &'static str {
        if self.model.trim().is_empty() {
            "tiktoken_disabled"
        } else {
            "model_not_registered"
        }
    }
}

/// Returns `Some` for a non-blank name, trimmed.
fn non_empty(value: String) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_owned())
    }
}

/// Loads a native `tokenizer.json` artifact.
///
/// # Errors
///
/// Always returns [`Error::Unsupported`]: loading a `HuggingFace` `tokenizers`
/// artifact needs a runtime that is not available offline. The path is never
/// read; use [`ModelTokenCounter`] or the selector's approximate fallback
/// instead.
pub fn load_native_tokenizer(path: impl AsRef<std::path::Path>) -> Result<SelectedTokenizer> {
    // Keep the path in the signature so callers can switch over without an API
    // change, but never touch the filesystem: this is a documented deferral.
    let _ = path.as_ref();
    Err(Error::Unsupported(
        "native tokenizer.json loading requires the HuggingFace tokenizers runtime; use the tiktoken or approximate counter"
            .to_owned(),
    ))
}
