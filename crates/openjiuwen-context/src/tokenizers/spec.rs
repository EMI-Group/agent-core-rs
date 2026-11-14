//! Declarative tokenizer specs, registry matching, and encoding catalogues.
//!
//! This module owns what a deployment declares: [`TokenizerSpec`] and its
//! one-hop [`CompatibleTokenizerSpec`] fallback, [`TokenizerRegistry`] and the
//! exact/family matching rules, plus the [`ModelEncoding`] and
//! [`TokenizerEngine`] catalogues the specs refer to.
//!
//! Nothing here counts tokens or resolves artifacts. The `counter` module turns
//! a resolved spec into a working counter and the `selector` module walks the
//! selection ladder; every public type here is re-exported from [`super`].
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

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
    /// Use the best offline engine available: a native artefact when one
    /// resolves, then `tiktoken`, then the estimate.
    #[default]
    Auto,
    /// Use a `tiktoken` BPE vocabulary, or a `tiktoken.model` artefact when a
    /// [`TokenizerArtifactManager`](super::artifacts::TokenizerArtifactManager)
    /// resolves one.
    Tiktoken,
    /// Use a native `tokenizer.json` artefact.
    ///
    /// A [`TokenizerSelector`](super::selector::TokenizerSelector) carrying a
    /// [`TokenizerArtifactManager`](super::artifacts::TokenizerArtifactManager)
    /// resolves it from the spec; without a manager the selection degrades
    /// through `tiktoken` to the estimate, and
    /// [`load_native_tokenizer`](super::selector::load_native_tokenizer) loads a
    /// file a caller already knows.
    Native,
}

/// A declared tokenizer for one provider/model or one model family.
///
/// Specs are declarative metadata. A
/// [`TokenizerSelector`](super::selector::TokenizerSelector) without a
/// [`TokenizerArtifactManager`](super::artifacts::TokenizerArtifactManager) can
/// only satisfy the `tiktoken` and `auto` engines; with a manager it resolves
/// [`TokenizerSpec::artifact_path`], [`TokenizerSpec::tokenizer_id`] through the
/// artefact cache, and the one-hop [`TokenizerSpec::compatible_fallbacks`].
/// Every field is optional except the provider/model pair, and unknown JSON
/// fields are ignored.
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
    /// Tokenizer artifact id, carried through as measurement metadata and used
    /// as the cache/hub id by a
    /// [`TokenizerArtifactManager`](super::artifacts::TokenizerArtifactManager).
    #[serde(alias = "id")]
    pub tokenizer_id: Option<String>,
    /// Declared measurement source (`tiktoken`, `native_tokenizer`, ...).
    pub source: Option<String>,
    /// Tokenizer artifact revision.
    pub revision: Option<String>,
    /// Local path of a native tokenizer artifact.
    pub artifact_path: Option<String>,
    /// Expected SHA-256 of a native tokenizer artifact.
    pub sha256: Option<String>,
    /// One-hop family fallbacks tried when this entry's artifact is missing.
    ///
    /// The selector reads at most the first entry, matching Python's
    /// `compatible_fallbacks[:1]`.
    pub compatible_fallbacks: Vec<CompatibleTokenizerSpec>,
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

/// A tokenizer from the same model family that may be used as a fallback.
///
/// The shape matches Python's `CompatibleTokenizerSpec`: the same fields as
/// [`TokenizerSpec`] minus `provider`, `family` and `compatible_fallbacks`,
/// because a fallback is chosen for a model of the same family and the chain
/// stays one hop deep.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CompatibleTokenizerSpec {
    /// The model the fallback tokenizer was published for.
    pub model: String,
    /// The artifact id the fallback resolves through.
    #[serde(alias = "id")]
    pub tokenizer_id: Option<String>,
    /// Artifact source (`huggingface`, `modelscope`, `provider_official`,
    /// `local`).
    pub source: Option<String>,
    /// Engine requested by the fallback.
    pub engine: TokenizerEngine,
    /// Artifact revision.
    pub revision: Option<String>,
    /// Local path of a native tokenizer artifact.
    pub artifact_path: Option<String>,
    /// Expected SHA-256 of a native tokenizer artifact.
    pub sha256: Option<String>,
}

impl From<CompatibleTokenizerSpec> for TokenizerSpec {
    /// Widens a fallback into a spec the artefact manager can resolve.
    fn from(fallback: CompatibleTokenizerSpec) -> Self {
        Self {
            provider: String::new(),
            model: fallback.model,
            family: None,
            engine: fallback.engine,
            tokenizer_id: fallback.tokenizer_id,
            source: fallback.source,
            revision: fallback.revision,
            artifact_path: fallback.artifact_path,
            sha256: fallback.sha256,
            compatible_fallbacks: Vec::new(),
        }
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
