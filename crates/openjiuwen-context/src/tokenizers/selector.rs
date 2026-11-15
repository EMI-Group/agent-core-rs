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
//! in pure Rust by [`native::NativeTokenizer`].
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::ApproxTokenCounter;

use super::counter::{non_empty, resolve_encoding, SelectedTokenizer, TokenizerSource};
use super::spec::{TokenizerEngine, TokenizerMatchKind, TokenizerRegistry, TokenizerSpec};
use super::{artifacts, native};

/// Characters per token used by the selector's last-resort estimate.
///
/// Three characters per token is deliberately pessimistic: callers that reach
/// this rung know nothing about the model, so over-counting keeps the context
/// window safe.
const STRING_LENGTH_CHARS_PER_TOKEN: usize = 3;

/// The reason a selection carries when a resolved artifact cannot be decoded.
///
/// The artifact exists, so the degradation is not a cache miss: the model type
/// it declares (or its malformed body) is why counting falls back to `tiktoken`
/// or to the characters-per-token estimate, and this reason names that cause
/// for [`SelectedTokenizer::fallback_reason`] and the downstream `estimated`
/// telemetry.
pub const NATIVE_ARTIFACT_UNDECODABLE_REASON: &str = "native_artifact_undecodable";

/// The reason a requested native engine carries when nothing resolves.
const NATIVE_TOKENIZER_UNAVAILABLE_REASON: &str = "native_tokenizer_unavailable";

/// The reason a one-hop compatible fallback carries when nothing else explains it.
const TARGET_TOKENIZER_UNAVAILABLE_REASON: &str = "target_tokenizer_unavailable";

/// The reason a family registry match carries.
const MODEL_VARIANT_TOKENIZER_FAMILY_REASON: &str = "model_variant_tokenizer_family";

/// What one native-artifact rung produced.
///
/// The rung keeps its `Option`-shaped outcome — a caller still sees "selected or
/// not" — while the failure carries the reason the ladder must report, so a
/// refused artifact never degrades anonymously.
enum ArtifactAttempt {
    /// The rung produced a selection.
    Selected(SelectedTokenizer),
    /// The rung produced nothing; `reason` names a refusal when one happened.
    Failed(Option<&'static str>),
}

impl ArtifactAttempt {
    /// The reason to carry into the rung the ladder lands on.
    ///
    /// Only a refusal names one: a spec with no resolvable artifact leaves the
    /// reason to the rung that follows, which knows its own strategy.
    const fn reason(&self) -> Option<&'static str> {
        match self {
            Self::Selected(_) => None,
            Self::Failed(reason) => *reason,
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
///
/// An artifact that resolves but cannot be decoded is refused with [`NATIVE_ARTIFACT_UNDECODABLE_REASON`] instead: the rung it leaves behind — a `tiktoken` vocabulary, the estimate, or a one-hop sibling artifact — reports that reason, so the degradation always names its real cause.
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
    ///
    /// A selection that had to degrade below a native artifact emits exactly one
    /// `tracing::warn!` naming the model and the reason, so a degraded count is
    /// visible once per selection instead of once per count.
    #[must_use]
    pub fn select(&self) -> SelectedTokenizer {
        let selected = self.resolve();
        Self::warn_degradation(&selected);
        selected
    }

    /// Emits the one warning of a degraded selection.
    ///
    /// A resolved `tiktoken` vocabulary reports `estimated() == true` without a
    /// `fallback_reason`: nothing degraded, so nothing is logged.
    fn warn_degradation(selected: &SelectedTokenizer) {
        let Some(reason) = selected.fallback_reason() else {
            return;
        };
        tracing::warn!(
            target: "openjiuwen_context::tokenizers",
            model = selected.model().unwrap_or_default(),
            source = ?selected.source(),
            reason,
            note = selected.estimation_note().unwrap_or_default(),
            "token counting degraded to an estimate"
        );
    }

    /// Walks the selection ladder without logging.
    fn resolve(&self) -> SelectedTokenizer {
        if let Some(spec) = &self.spec {
            let attempt = self.artifact_chain(spec, TokenizerSource::NativeTokenizer, None);
            if let ArtifactAttempt::Selected(selected) = attempt {
                return selected;
            }
            if spec.engine == TokenizerEngine::Native {
                return self.native_selection(Some(spec), attempt.reason());
            }
            return self.spec_selection(spec, attempt.reason());
        }

        if let Some(hit) = self
            .registry
            .as_ref()
            .and_then(|registry| registry.resolve(&self.provider, &self.model))
        {
            match hit.kind {
                TokenizerMatchKind::Exact => {
                    let attempt =
                        self.artifact_chain(&hit.spec, TokenizerSource::NativeTokenizer, None);
                    if let ArtifactAttempt::Selected(selected) = attempt {
                        return selected;
                    }
                    if hit.spec.engine == TokenizerEngine::Native {
                        return self.native_selection(Some(&hit.spec), attempt.reason());
                    }
                    return self.spec_selection(&hit.spec, attempt.reason());
                }
                TokenizerMatchKind::Family => {
                    let attempt = self.artifact_chain(
                        &hit.spec,
                        TokenizerSource::FamilyTokenizerFallback,
                        Some(MODEL_VARIANT_TOKENIZER_FAMILY_REASON),
                    );
                    if let ArtifactAttempt::Selected(selected) = attempt {
                        return selected;
                    }
                    // A family artifact that exists but cannot be decoded is the
                    // real cause, so it replaces the family-match reason on the
                    // rung the ladder lands on.
                    if let Some(reason) = attempt.reason() {
                        return self
                            .tiktoken_selection(
                                TokenizerSource::FamilyTokenizerFallback,
                                Some(reason),
                            )
                            .unwrap_or_else(|| self.string_length_selection(reason));
                    }
                    if let Some(selected) = self.tiktoken_selection(
                        TokenizerSource::FamilyTokenizerFallback,
                        Some(MODEL_VARIANT_TOKENIZER_FAMILY_REASON),
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
    ///
    /// The reason the primary rung failed travels into the fallback rung, so a
    /// sibling vocabulary taken because the primary artifact could not be
    /// decoded reports [`NATIVE_ARTIFACT_UNDECODABLE_REASON`] rather than the
    /// generic `target_tokenizer_unavailable`.
    fn artifact_chain(
        &self,
        spec: &TokenizerSpec,
        source: TokenizerSource,
        reason: Option<&str>,
    ) -> ArtifactAttempt {
        let primary = self.artifact_selection(spec, source, reason);
        if let ArtifactAttempt::Selected(selected) = primary {
            return ArtifactAttempt::Selected(selected);
        }
        let failure = primary.reason();
        let Some(fallback) = spec.compatible_fallbacks.first().cloned() else {
            return ArtifactAttempt::Failed(failure);
        };
        self.artifact_selection(
            &TokenizerSpec::from(fallback),
            TokenizerSource::FamilyTokenizerFallback,
            Some(failure.unwrap_or(TARGET_TOKENIZER_UNAVAILABLE_REASON)),
        )
    }

    /// Builds a native selection for one spec.
    ///
    /// Nothing is selected when no artifact resolves, and also when the resolved
    /// artifact cannot be decoded: a refused artifact emits exactly one
    /// `tracing::warn!` naming it and the typed error, and its reason travels to
    /// the rung the ladder lands on.
    fn artifact_selection(
        &self,
        spec: &TokenizerSpec,
        source: TokenizerSource,
        reason: Option<&str>,
    ) -> ArtifactAttempt {
        let Some(manager) = self.manager.as_ref() else {
            return ArtifactAttempt::Failed(None);
        };
        let Some(path) = manager.resolve_cached(spec) else {
            return Self::refused_artifact(spec, manager.last_error());
        };
        let artifact_display = path.display().to_string();
        let native = match native::load_artifact_file(&path, &artifact_display) {
            Ok(native) => Arc::new(native),
            Err(error) => {
                return ArtifactAttempt::Failed(Some(Self::warn_undecodable(
                    &artifact_display,
                    &error.to_string(),
                )));
            }
        };
        let tokenizer = spec
            .tokenizer_id
            .clone()
            .filter(|id| !id.trim().is_empty())
            .or_else(|| native.tokenizer_id().map(str::to_owned))
            .or(Some(artifact_display));
        ArtifactAttempt::Selected(SelectedTokenizer::from_native_with(
            native,
            source,
            non_empty(self.model.clone()),
            tokenizer,
            reason.map(str::to_owned),
        ))
    }

    /// Reports a cached artifact the manager refused instead of missing.
    ///
    /// A manager that parsed an artifact's body and refused it records a typed
    /// reason; that refusal means the artifact exists but cannot be decoded or
    /// read, so it is named once here and carried into the next rung, where it
    /// replaces the generic reason a bare cache miss would report.
    fn refused_artifact(spec: &TokenizerSpec, refusal: Option<String>) -> ArtifactAttempt {
        let Some(message) = refusal else {
            return ArtifactAttempt::Failed(None);
        };
        if !is_artifact_refusal(&message) {
            return ArtifactAttempt::Failed(None);
        }
        let artifact = declared_artifact(spec);
        ArtifactAttempt::Failed(Some(Self::warn_undecodable(&artifact, &message)))
    }

    /// Emits the one warning of a refused artifact and names its reason.
    fn warn_undecodable(artifact: &str, error: &str) -> &'static str {
        tracing::warn!(
            target: "openjiuwen_context::tokenizers",
            artifact,
            error,
            reason = NATIVE_ARTIFACT_UNDECODABLE_REASON,
            "native tokenizer artifact could not be decoded; the selection continues down the ladder"
        );
        NATIVE_ARTIFACT_UNDECODABLE_REASON
    }

    /// Builds the selection for an explicit (non-native) spec.
    ///
    /// A spec that declares a fallback source keeps that source even when the
    /// model resolves cleanly, so the reason is filled from the declared source
    /// when encoding resolution found none: an estimate always names why.
    /// `artifact_reason` is the refusal of the spec's own artifact, when one was
    /// refused, so that failure is named when nothing more specific applies.
    fn spec_selection(
        &self,
        spec: &TokenizerSpec,
        artifact_reason: Option<&'static str>,
    ) -> SelectedTokenizer {
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
            resolved
                .fallback_reason
                .or_else(|| declared_fallback_reason(source))
                .or_else(|| artifact_reason.map(str::to_owned)),
        )
    }

    /// Builds the selection when a native engine was requested but cannot be
    /// satisfied offline.
    ///
    /// `artifact_reason` replaces the generic `native_tokenizer_unavailable` when
    /// an artifact resolved but was refused, because that refusal is the real
    /// cause of the degradation.
    fn native_selection(
        &self,
        spec: Option<&TokenizerSpec>,
        artifact_reason: Option<&'static str>,
    ) -> SelectedTokenizer {
        let model = spec.map_or_else(|| self.model.clone(), |spec| self.effective_model(spec));
        let reason = Some(
            artifact_reason
                .unwrap_or(NATIVE_TOKENIZER_UNAVAILABLE_REASON)
                .to_owned(),
        );
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

/// The reason a declared fallback source carries when nothing else explains it.
///
/// An exact `tiktoken` or native source needs none; the three fallback sources
/// do, so a measurement never reports an estimate without a named cause.
fn declared_fallback_reason(source: TokenizerSource) -> Option<String> {
    match source {
        TokenizerSource::TiktokenFallback => Some("tiktoken_unavailable".to_owned()),
        TokenizerSource::StringLengthFallback => Some("model_not_registered".to_owned()),
        TokenizerSource::NativeTokenizer
        | TokenizerSource::Tiktoken
        | TokenizerSource::FamilyTokenizerFallback => None,
    }
}

/// Whether a recorded manager failure refused a resolved artifact.
///
/// These are the codes the manager records for an artifact it found and then
/// refused — an undecoded model type, a malformed body, an unreadable file — as
/// opposed to a cache miss, which the following rung explains itself.
fn is_artifact_refusal(message: &str) -> bool {
    let head = message.split(':').next().unwrap_or_default().trim();
    head == artifacts::TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE
        || head == artifacts::TOKENIZER_DOWNLOAD_FAILED_CODE
}

/// The artifact a spec names, as a message about it describes it.
///
/// The declared `artifact_path` wins (that is the file a hand-placed artifact
/// lives at), then the tokenizer id, so a warning always identifies which
/// artifact was refused.
fn declared_artifact(spec: &TokenizerSpec) -> String {
    let path = spec
        .artifact_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(path) = path {
        return path.to_owned();
    }
    let id = spec.tokenizer_id.as_deref().map(str::trim);
    match id.filter(|value| !value.is_empty()) {
        Some(id) => id.to_owned(),
        None => "<unnamed tokenizer artifact>".to_owned(),
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
/// sniffing, exactly as [`artifacts::TokenizerArtifactManager`] validates a
/// resolved artifact — see [`native::load_artifact_file`].
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
    let native = native::load_artifact_file(path, &display)?;
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
