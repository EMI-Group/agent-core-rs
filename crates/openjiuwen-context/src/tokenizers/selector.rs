//! The tokenizer selection ladder and native-artifact loading.
//!
//! [`TokenizerSelector`] resolves the counter a context engine should use: an
//! explicit spec, a registry match, a native artifact from an attached
//! [`artifacts::TokenizerArtifactManager`], a `tiktoken` vocabulary, then a
//! conservative string-length estimate. Every rung reports why it was taken,
//! and selection is synchronous and cache-only, so it never downloads.
//!
//! [`load_native_tokenizer`] loads an artifact a caller already knows; both
//! `tokenizer.json` documents and `tiktoken` `.model` vocabularies are decoded
//! in pure Rust by [`NativeTokenizer`].
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::ApproxTokenCounter;

use super::counter::{non_empty, resolve_encoding, SelectedTokenizer, TokenizerSource};
use super::native::NativeTokenizer;
use super::spec::{TokenizerEngine, TokenizerMatchKind, TokenizerRegistry, TokenizerSpec};
use super::{artifacts, native};

/// Characters per token used by the selector's last-resort estimate.
///
/// Three characters per token is deliberately pessimistic: callers that reach
/// this rung know nothing about the model, so over-counting keeps the context
/// window safe.
const STRING_LENGTH_CHARS_PER_TOKEN: usize = 3;

/// Chooses a [`SelectedTokenizer`] from an optional explicit spec and registry.
///
/// The ladder is:
/// 1. the explicit [`TokenizerSpec`], when one is set — the registry is never
///    consulted;
/// 2. a [`TokenizerRegistry`] hit (exact or family);
/// 3. the model's `tiktoken` vocabulary;
/// 4. a three-characters-per-token estimate.
///
/// With a [`artifacts::TokenizerArtifactManager`] attached, steps 1 and 2 first try the
/// spec's own native artifact, then the first
/// [`TokenizerSpec::compatible_fallbacks`] entry (one hop, reason
/// `target_tokenizer_unavailable`), and only then continue down the ladder. A
/// family registry match keeps its `family_tokenizer_fallback` source and
/// `model_variant_tokenizer_family` reason, and the selection reports
/// `estimated() == true` for it because the vocabulary is a sibling's.
///
/// A requested [`TokenizerEngine::Native`] without a resolvable artifact falls
/// through to steps 3 and 4 with the reason `native_tokenizer_unavailable`.
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
    /// Artifact resolver; absent means every native rung is unavailable.
    manager: Option<artifacts::TokenizerArtifactManager>,
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
            manager: None,
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

    /// Sets the artifact manager used to resolve native tokenizers.
    ///
    /// Selection is synchronous, so it only ever reads the local cache
    /// ([`artifacts::TokenizerArtifactManager::resolve_cached`]); a download
    /// stays an explicit [`artifacts::TokenizerArtifactManager::resolve`] or
    /// [`artifacts::TokenizerArtifactManager::download`] call.
    #[must_use]
    pub fn with_artifact_manager(mut self, manager: artifacts::TokenizerArtifactManager) -> Self {
        self.manager = Some(manager);
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

    /// The artifact manager, when one is attached.
    #[must_use]
    pub fn artifact_manager(&self) -> Option<&artifacts::TokenizerArtifactManager> {
        self.manager.as_ref()
    }

    /// Selects the tokenizer for this provider and model.
    #[must_use]
    pub fn select(&self) -> SelectedTokenizer {
        if let Some(spec) = &self.spec {
            if spec.engine == TokenizerEngine::Native {
                return self
                    .artifact_chain(spec, TokenizerSource::NativeTokenizer, None)
                    .unwrap_or_else(|| self.native_selection(Some(spec)));
            }
            if let Some(selected) =
                self.artifact_chain(spec, TokenizerSource::NativeTokenizer, None)
            {
                return selected;
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
                        return self
                            .artifact_chain(&hit.spec, TokenizerSource::NativeTokenizer, None)
                            .unwrap_or_else(|| self.native_selection(Some(&hit.spec)));
                    }
                    if let Some(selected) =
                        self.artifact_chain(&hit.spec, TokenizerSource::NativeTokenizer, None)
                    {
                        return selected;
                    }
                    return self.spec_selection(&hit.spec);
                }
                TokenizerMatchKind::Family => {
                    if let Some(selected) = self.artifact_chain(
                        &hit.spec,
                        TokenizerSource::FamilyTokenizerFallback,
                        Some("model_variant_tokenizer_family"),
                    ) {
                        return selected;
                    }
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

    /// Resolves a native artifact for `spec`, then for its one family fallback.
    ///
    /// Only the first [`TokenizerSpec::compatible_fallbacks`] entry is tried,
    /// matching Python's `compatible_fallbacks[:1]`: selection stays O(1) even
    /// with sloppy configuration.
    fn artifact_chain(
        &self,
        spec: &TokenizerSpec,
        source: TokenizerSource,
        reason: Option<&str>,
    ) -> Option<SelectedTokenizer> {
        if let Some(selected) = self.artifact_selection(spec, source, reason) {
            return Some(selected);
        }
        let fallback = spec.compatible_fallbacks.first()?.clone();
        self.artifact_selection(
            &TokenizerSpec::from(fallback),
            TokenizerSource::FamilyTokenizerFallback,
            Some("target_tokenizer_unavailable"),
        )
    }

    /// Builds a native selection for one spec, or `None` when its artifact is
    /// not available locally.
    fn artifact_selection(
        &self,
        spec: &TokenizerSpec,
        source: TokenizerSource,
        reason: Option<&str>,
    ) -> Option<SelectedTokenizer> {
        let manager = self.manager.as_ref()?;
        let path = manager.resolve_cached(spec)?;
        let artifact_display = path.display().to_string();
        let native = match load_native_artifact(&path, &artifact_display) {
            Ok(native) => Arc::new(native),
            Err(error) => {
                tracing::debug!(
                    target: "openjiuwen_context::tokenizers",
                    %error,
                    artifact = %artifact_display,
                    "native tokenizer artifact could not be decoded; continuing down the ladder"
                );
                return None;
            }
        };
        let tokenizer = spec
            .tokenizer_id
            .clone()
            .filter(|id| !id.trim().is_empty())
            .or_else(|| native.tokenizer_id().map(str::to_owned))
            .or(Some(artifact_display));
        Some(SelectedTokenizer::from_native_with(
            native,
            source,
            non_empty(self.model.clone()),
            tokenizer,
            reason.map(str::to_owned),
        ))
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

/// Loads a native tokenizer artifact.
///
/// Two artifact kinds are decoded in pure Rust: a `HuggingFace` `tokenizer.json`
/// document and a `tiktoken` `.model` BPE vocabulary. The file is read once,
/// here; the returned selection then counts entirely in memory, so this
/// constructor is a one-off per context/model binding and never a hot path.
///
/// The kind is chosen by file extension (`.model` or `.json`), then by content
/// sniffing: a document whose first meaningful line is `<base64-token> <rank>`
/// is a `.model` vocabulary. Anything else is tried as `tokenizer.json` first
/// and as a `.model` vocabulary second.
///
/// # Errors
///
/// - [`Error::NotFound`] when `path` does not exist;
/// - [`Error::Validation`] when neither loader can decode the file; the message
///   carries both attempts for an unrecognised extension;
/// - [`Error::Unsupported`] when the artifact declares a `tokenizer.json` model
///   type this crate does not decode.
pub fn load_native_tokenizer(path: impl AsRef<std::path::Path>) -> Result<SelectedTokenizer> {
    let path = path.as_ref();
    let display = path.display().to_string();
    if !path.exists() {
        return Err(Error::NotFound(format!(
            "native tokenizer artifact not found: {display}"
        )));
    }
    let native = load_native_artifact(path, &display)?;
    let tokenizer = Some(
        native
            .tokenizer_id()
            .map_or_else(|| display.clone(), str::to_owned),
    );
    Ok(SelectedTokenizer::from_native(
        Arc::new(native),
        None,
        tokenizer,
    ))
}

/// Decodes `path` as a `tokenizer.json`, a `tiktoken` `.model`, or both in turn.
fn load_native_artifact(path: &std::path::Path, display: &str) -> Result<NativeTokenizer> {
    if path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("model"))
    {
        return NativeTokenizer::from_tiktoken_model_file(path);
    }
    if path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
    {
        return NativeTokenizer::load(path);
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        // Let the `tokenizer.json` loader raise the typed read error.
        return NativeTokenizer::load(path);
    };
    if native::tiktoken_model::looks_like_tiktoken_model(&text) {
        return NativeTokenizer::from_tiktoken_model_file(path);
    }
    match NativeTokenizer::from_json_str(&text) {
        Ok(native) => Ok(native),
        Err(json_error) => match NativeTokenizer::from_tiktoken_model_file(path) {
            Ok(native) => Ok(native),
            Err(model_error) => Err(Error::Validation(format!(
                "{display} is neither a decodable tokenizer.json ({json_error}) nor a tiktoken `.model` vocabulary ({model_error})"
            ))),
        },
    }
}
