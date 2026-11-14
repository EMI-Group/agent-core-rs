//! Context-engine configuration: budgets, thresholds, counters, and processors.
//!
//! [`ContextConfig`] is the single source of truth for how a [`ContextEngine`]
//! sizes a window and when it compresses. Python's `ContextEngineConfig` and
//! `TokenCounterConfig` are merged into the same struct so serialized
//! configuration documents stay one object deep.
//!
//! ```
//! use openjiuwen_context::{ContextConfig, TokenCounterKind};
//!
//! let config = ContextConfig::default();
//! assert_eq!(config.compression_threshold, 0.85);
//! assert_eq!(config.token_counter.kind, TokenCounterKind::Approx);
//! config.validate().unwrap();
//! ```
//!
//! [`ContextEngine`]: crate::engine::ContextEngine
use std::path::PathBuf;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, TiktokenCounter, TokenCounter};
use serde::{Deserialize, Serialize};

use crate::policy::window::{ContextWindowPolicy, ContextWindowSource, ResolvedContextWindow};
use crate::processor::{ProcessorConfig, ToolResultConfig, TruncationConfig};
use crate::tokenizers::artifacts::TokenizerArtifactManager;
use crate::tokenizers::{TokenizerRegistry, TokenizerSelector, TokenizerSpec};

/// Context budget used when [`ContextConfig::max_context_tokens`] is unset.
///
/// Python's default `200_000` covers every modern provider's window; a caller
/// that knows the model's window sets the field explicitly.
pub const DEFAULT_CONTEXT_TOKENS: usize = 200_000;

/// Which token counter a context engine builds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenCounterKind {
    /// Foundation's dependency-free four-characters-per-token estimate.
    #[default]
    Approx,
    /// Python's `len(text) // chars_per_token` heuristic.
    Characters,
    /// `cl100k_base` BPE counting with foundation's heuristic fallback.
    Tiktoken,
    /// A counter selected from the model name through the tokenizer registry.
    Model,
}

/// Token-counter selection and its heuristic parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TokenCounterConfig {
    /// Which counter to build.
    pub kind: TokenCounterKind,
    /// Characters per token for [`TokenCounterKind::Characters`]; clamped to at
    /// least `1` when the counter is built.
    pub chars_per_token: usize,
}

impl Default for TokenCounterConfig {
    fn default() -> Self {
        Self {
            kind: TokenCounterKind::Approx,
            chars_per_token: 4,
        }
    }
}

/// Compression-recall settings.
///
/// Recall re-reads previously offloaded content in bounded chunks. The rust
/// port keeps the configuration surface (and validates it) even though the
/// rehydration prompt injection itself is deferred — see the crate `CONTEXT.md`
/// deferred list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CompressionRecallConfig {
    /// Whether recall is enabled.
    pub enabled: bool,
    /// Size of one recall chunk, in tokens.
    pub chunk_size_tokens: usize,
    /// Overlap between consecutive recall chunks, in tokens.
    pub chunk_overlap_tokens: usize,
}

impl Default for CompressionRecallConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            chunk_size_tokens: 3000,
            chunk_overlap_tokens: 300,
        }
    }
}

/// Everything the context engine needs to assemble and compress windows.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ContextConfig {
    /// Hard context-window limit reported by the provider; `None` means the
    /// window is resolved through [`ContextConfig::context_window_policy`].
    pub max_context_tokens: Option<usize>,
    /// Context-window resolution policy used when
    /// [`ContextConfig::max_context_tokens`] is unset.
    ///
    /// The precedence is: an explicit [`ContextConfig::max_context_tokens`]
    /// wins over everything; otherwise, when the policy carries any override,
    /// [`ContextWindowPolicy::resolve`] walks its ladder (global cap, selected
    /// model override, configured model map, built-in table, `OpenRouter`
    /// cache, then [`DEFAULT_CONTEXT_TOKENS`]); a completely default policy
    /// keeps [`DEFAULT_CONTEXT_TOKENS`], which is the pre-policy behaviour.
    pub context_window_policy: ContextWindowPolicy,
    /// Tokens held back for the model's answer.
    pub reserve_tokens: usize,
    /// Fraction of the effective maximum at which compression triggers
    /// (`0.85` by default).
    pub compression_threshold: f32,
    /// Newest messages summarisation never touches.
    pub recent_messages: usize,
    /// Rolling limit on stored messages, when set.
    pub max_context_messages: Option<usize>,
    /// Default message-count window for [`ContextEngine::prepare`].
    ///
    /// [`ContextEngine::prepare`]: crate::engine::ContextEngine::prepare
    pub default_window_messages: Option<usize>,
    /// Default dialogue-round window for [`ContextEngine::prepare`].
    ///
    /// [`ContextEngine::prepare`]: crate::engine::ContextEngine::prepare
    pub default_window_rounds: Option<usize>,
    /// Whether offloaded content may be rehydrated on demand.
    pub enable_reload: bool,
    /// Token counter selection.
    pub token_counter: TokenCounterConfig,
    /// Compression-recall settings.
    pub compression_recall: CompressionRecallConfig,
    /// Ordered processor pipeline; defaults to tool-result then truncation.
    pub processors: Vec<ProcessorConfig>,
    /// Model name used for summarisation requests and model-aware counting.
    pub model_name: Option<String>,
    /// Provider name used to resolve [`TokenCounterKind::Model`] through the
    /// tokenizer registry.
    pub tokenizer_provider: Option<String>,
    /// Declared tokenizers used by [`TokenCounterKind::Model`].
    pub tokenizer_registry: Vec<TokenizerSpec>,
    /// Directory holding cached tokenizer artifacts.
    ///
    /// `None` follows
    /// [`OPENJIUWEN_TOKENIZER_CACHE_DIR`](crate::tokenizers::artifacts::CACHE_DIR_ENV)
    /// and then `~/.cache/openjiuwen/tokenizers`.
    pub tokenizer_cache_dir: Option<PathBuf>,
    /// Whether a tokenizer artifact may be downloaded when it is not cached.
    ///
    /// Off by default: context creation and counting never touch the network.
    pub enable_tokenizer_download: bool,
    /// Whether artifact resolution is restricted to the local cache.
    pub tokenizer_offline: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: None,
            context_window_policy: ContextWindowPolicy::default(),
            reserve_tokens: 0,
            compression_threshold: 0.85,
            recent_messages: 6,
            max_context_messages: None,
            default_window_messages: None,
            default_window_rounds: None,
            enable_reload: false,
            token_counter: TokenCounterConfig::default(),
            compression_recall: CompressionRecallConfig::default(),
            processors: default_processors(),
            model_name: None,
            tokenizer_provider: None,
            tokenizer_registry: Vec::new(),
            tokenizer_cache_dir: None,
            enable_tokenizer_download: false,
            tokenizer_offline: false,
        }
    }
}

/// The default pipeline: trim tool results, then truncate whole rounds.
fn default_processors() -> Vec<ProcessorConfig> {
    vec![
        ProcessorConfig::ToolResult(ToolResultConfig::default()),
        ProcessorConfig::Truncation(TruncationConfig::default()),
    ]
}

impl ContextConfig {
    /// The usable context window and the rung that produced it.
    ///
    /// An explicit [`ContextConfig::max_context_tokens`] wins and reports
    /// [`ContextWindowSource::Global`]; otherwise a policy with any override
    /// resolves the window through
    /// [`ContextWindowPolicy::resolve`] for [`ContextConfig::model_name`] and
    /// [`ContextConfig::tokenizer_provider`]; a completely default policy keeps
    /// [`DEFAULT_CONTEXT_TOKENS`] with [`ContextWindowSource::Default`].
    #[must_use]
    pub fn resolved_window(&self) -> ResolvedContextWindow {
        if let Some(tokens) = self.max_context_tokens {
            return ResolvedContextWindow::new(tokens, ContextWindowSource::Global);
        }
        if policy_is_configured(&self.context_window_policy) {
            return self.context_window_policy.resolve(
                self.model_name.as_deref(),
                self.tokenizer_provider.as_deref(),
            );
        }
        ResolvedContextWindow::default()
    }

    /// The context budget after subtracting the reserve, saturating at zero.
    #[must_use]
    pub fn effective_max_tokens(&self) -> usize {
        self.resolved_window()
            .tokens
            .saturating_sub(self.reserve_tokens)
    }

    /// The token count at which compression triggers; always at least `1`.
    #[must_use]
    pub fn compression_trigger_tokens(&self) -> usize {
        let effective = self.effective_max_tokens() as f64;
        let trigger = (effective * f64::from(self.compression_threshold)).floor();
        // The cast is bounded by the finite, non-negative product above; keep a
        // floor of one token so a tiny budget cannot disable compression.
        (trigger as usize).max(1)
    }

    /// Builds the configured token counter.
    ///
    /// The returned handle is `Send + Sync` so processors can hold it across
    /// await points; every foundation counter satisfies both bounds.
    #[must_use]
    pub fn token_counter(&self) -> Arc<dyn TokenCounter + Send + Sync> {
        match self.token_counter.kind {
            TokenCounterKind::Approx => Arc::new(ApproxTokenCounter::new()),
            TokenCounterKind::Characters => Arc::new(ApproxTokenCounter::with_chars_per_token(
                self.token_counter.chars_per_token.max(1),
            )),
            TokenCounterKind::Tiktoken => Arc::new(TiktokenCounter::new()),
            TokenCounterKind::Model => {
                let registry = if self.tokenizer_registry.is_empty() {
                    None
                } else {
                    Some(TokenizerRegistry::new(self.tokenizer_registry.clone()))
                };
                let selector = TokenizerSelector::new(
                    self.tokenizer_provider.clone().unwrap_or_default(),
                    self.model_name.clone().unwrap_or_default(),
                    None,
                    registry,
                )
                .with_artifact_manager(self.tokenizer_artifact_manager());
                Arc::new(selector.select())
            }
        }
    }

    /// The artifact manager the model-aware counter selects with.
    ///
    /// Selection is synchronous, so the manager only reads the local cache even
    /// when `enable_tokenizer_download` is set; a download stays an explicit
    /// [`TokenizerArtifactManager::download`] call.
    #[must_use]
    pub fn tokenizer_artifact_manager(&self) -> TokenizerArtifactManager {
        let mut manager = TokenizerArtifactManager::new()
            .with_enable_download(self.enable_tokenizer_download)
            .with_offline(self.tokenizer_offline);
        if let Some(cache_dir) = &self.tokenizer_cache_dir {
            manager = manager.with_cache_dir(cache_dir.clone());
        }
        manager
    }

    /// Validates the configuration before it is used by an engine.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the compression threshold is outside
    /// `(0, 1]`, the effective maximum is zero, the `characters` counter has a
    /// zero ratio, a tokenizer registry entry names neither a provider nor a
    /// model, tokenizer downloads are enabled while the cache is offline, or
    /// compression recall is enabled with a chunk size of zero or an overlap
    /// that is not smaller than the chunk size.
    pub fn validate(&self) -> Result<()> {
        let threshold = self.compression_threshold;
        if !threshold.is_finite() || threshold <= 0.0 || threshold > 1.0 {
            return Err(Error::Config(format!(
                "compression_threshold must be in (0, 1], got {threshold}"
            )));
        }
        if self.effective_max_tokens() == 0 {
            return Err(Error::Config(
                "effective max context tokens is 0; raise max_context_tokens or lower reserve_tokens"
                    .to_owned(),
            ));
        }
        if self.token_counter.kind == TokenCounterKind::Characters
            && self.token_counter.chars_per_token == 0
        {
            return Err(Error::Config(
                "token_counter.chars_per_token must be > 0 for the characters counter".to_owned(),
            ));
        }
        for spec in &self.tokenizer_registry {
            spec.validate()?;
        }
        if self.enable_tokenizer_download && self.tokenizer_offline {
            return Err(Error::Config(
                "enable_tokenizer_download and tokenizer_offline are mutually exclusive".to_owned(),
            ));
        }
        let recall = &self.compression_recall;
        if recall.enabled {
            if recall.chunk_size_tokens == 0 {
                return Err(Error::Config(
                    "compression_recall.chunk_size_tokens must be > 0 when recall is enabled"
                        .to_owned(),
                ));
            }
            if recall.chunk_overlap_tokens >= recall.chunk_size_tokens {
                return Err(Error::Config(format!(
                    "compression_recall.chunk_overlap_tokens ({}) must be smaller than chunk_size_tokens ({})",
                    recall.chunk_overlap_tokens, recall.chunk_size_tokens
                )));
            }
        }
        Ok(())
    }
}

/// Whether `policy` carries any override beyond its defaults.
///
/// The `OpenRouter` cache handle is not part of the comparison by value, only
/// by presence, because the cache itself is an interior-mutable handle.
fn policy_is_configured(policy: &ContextWindowPolicy) -> bool {
    let default = ContextWindowPolicy::default();
    policy.context_window_tokens != default.context_window_tokens
        || policy.model_context_window_tokens_override
            != default.model_context_window_tokens_override
        || policy.model_context_window_tokens != default.model_context_window_tokens
        || policy.enable_openrouter_model_context_window_tokens
            != default.enable_openrouter_model_context_window_tokens
        || policy.openrouter_request_timeout != default.openrouter_request_timeout
        || policy.openrouter_endpoint != default.openrouter_endpoint
        || policy.model_name != default.model_name
        || policy.model_provider != default.model_provider
        || policy.openrouter_cache.is_some()
}

impl PartialEq for ContextConfig {
    fn eq(&self, other: &Self) -> bool {
        let policy_eq = self.context_window_policy.context_window_tokens
            == other.context_window_policy.context_window_tokens
            && self
                .context_window_policy
                .model_context_window_tokens_override
                == other
                    .context_window_policy
                    .model_context_window_tokens_override
            && self.context_window_policy.model_context_window_tokens
                == other.context_window_policy.model_context_window_tokens
            && self
                .context_window_policy
                .enable_openrouter_model_context_window_tokens
                == other
                    .context_window_policy
                    .enable_openrouter_model_context_window_tokens
            && self.context_window_policy.openrouter_request_timeout
                == other.context_window_policy.openrouter_request_timeout
            && self.context_window_policy.openrouter_endpoint
                == other.context_window_policy.openrouter_endpoint
            && self.context_window_policy.model_name == other.context_window_policy.model_name
            && self.context_window_policy.model_provider
                == other.context_window_policy.model_provider
            && self.context_window_policy.openrouter_cache.is_some()
                == other.context_window_policy.openrouter_cache.is_some();
        self.max_context_tokens == other.max_context_tokens
            && self.reserve_tokens == other.reserve_tokens
            && self.compression_threshold == other.compression_threshold
            && self.recent_messages == other.recent_messages
            && self.max_context_messages == other.max_context_messages
            && self.default_window_messages == other.default_window_messages
            && self.default_window_rounds == other.default_window_rounds
            && self.enable_reload == other.enable_reload
            && self.token_counter == other.token_counter
            && self.compression_recall == other.compression_recall
            && self.processors == other.processors
            && self.model_name == other.model_name
            && self.tokenizer_provider == other.tokenizer_provider
            && self.tokenizer_registry == other.tokenizer_registry
            && self.tokenizer_cache_dir == other.tokenizer_cache_dir
            && self.enable_tokenizer_download == other.enable_tokenizer_download
            && self.tokenizer_offline == other.tokenizer_offline
            && policy_eq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::SummarizationConfig;

    #[test]
    fn defaults_match_the_python_engine() {
        let config = ContextConfig::default();
        assert_eq!(config.max_context_tokens, None);
        assert_eq!(config.reserve_tokens, 0);
        assert_eq!(config.compression_threshold, 0.85);
        assert_eq!(config.recent_messages, 6);
        assert!(config
            .processors
            .contains(&ProcessorConfig::Truncation(TruncationConfig::default())));
        assert!(config
            .processors
            .contains(&ProcessorConfig::ToolResult(ToolResultConfig::default())));
        assert_eq!(config.token_counter.kind, TokenCounterKind::Approx);
        assert_eq!(config.token_counter.chars_per_token, 4);
        assert!(!config.compression_recall.enabled);
        assert!(!config.enable_reload);
    }

    #[test]
    fn budget_and_trigger_math() {
        let config = ContextConfig {
            max_context_tokens: Some(1000),
            reserve_tokens: 200,
            compression_threshold: 0.5,
            ..ContextConfig::default()
        };
        assert_eq!(config.effective_max_tokens(), 800);
        assert_eq!(config.compression_trigger_tokens(), 400);

        let reserved_away = ContextConfig {
            max_context_tokens: Some(10),
            reserve_tokens: 50,
            ..ContextConfig::default()
        };
        assert_eq!(reserved_away.effective_max_tokens(), 0);

        let tiny = ContextConfig {
            max_context_tokens: Some(1),
            compression_threshold: 0.1,
            ..ContextConfig::default()
        };
        assert_eq!(tiny.compression_trigger_tokens(), 1);
    }

    #[test]
    fn validation_accepts_the_default_and_rejects_bad_values() {
        ContextConfig::default().validate().unwrap();

        let bad_threshold = ContextConfig {
            compression_threshold: 1.5,
            ..ContextConfig::default()
        };
        assert!(matches!(bad_threshold.validate(), Err(Error::Config(_))));

        let nan_threshold = ContextConfig {
            compression_threshold: f32::NAN,
            ..ContextConfig::default()
        };
        assert!(matches!(nan_threshold.validate(), Err(Error::Config(_))));

        let zero_budget = ContextConfig {
            max_context_tokens: Some(10),
            reserve_tokens: 10,
            ..ContextConfig::default()
        };
        assert!(matches!(zero_budget.validate(), Err(Error::Config(_))));

        let zero_ratio = ContextConfig {
            token_counter: TokenCounterConfig {
                kind: TokenCounterKind::Characters,
                chars_per_token: 0,
            },
            ..ContextConfig::default()
        };
        assert!(matches!(zero_ratio.validate(), Err(Error::Config(_))));

        let bad_recall = ContextConfig {
            compression_recall: CompressionRecallConfig {
                enabled: true,
                chunk_size_tokens: 100,
                chunk_overlap_tokens: 100,
            },
            ..ContextConfig::default()
        };
        assert!(matches!(bad_recall.validate(), Err(Error::Config(_))));

        let zero_chunk = ContextConfig {
            compression_recall: CompressionRecallConfig {
                enabled: true,
                chunk_size_tokens: 0,
                chunk_overlap_tokens: 0,
            },
            ..ContextConfig::default()
        };
        assert!(matches!(zero_chunk.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn counters_are_built_from_the_config() {
        let approx = ContextConfig::default().token_counter();
        assert_eq!(approx.count_text("abcd"), 1);

        let characters = ContextConfig {
            token_counter: TokenCounterConfig {
                kind: TokenCounterKind::Characters,
                chars_per_token: 0,
            },
            ..ContextConfig::default()
        }
        .token_counter();
        assert_eq!(characters.count_text("abc"), 3);

        let characters_three = ContextConfig {
            token_counter: TokenCounterConfig {
                kind: TokenCounterKind::Characters,
                chars_per_token: 3,
            },
            ..ContextConfig::default()
        }
        .token_counter();
        assert_eq!(characters_three.count_text("abcdef"), 2);
        assert_eq!(characters_three.count_text("abcdefg"), 3);

        let tiktoken = ContextConfig {
            token_counter: TokenCounterConfig {
                kind: TokenCounterKind::Tiktoken,
                chars_per_token: 4,
            },
            ..ContextConfig::default()
        }
        .token_counter();
        assert_eq!(tiktoken.count_text(""), 0);
        assert!(tiktoken.count_text("hello world") > 0);
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = ContextConfig {
            max_context_tokens: Some(32_000),
            reserve_tokens: 512,
            default_window_rounds: Some(3),
            processors: vec![
                ProcessorConfig::Summarization(SummarizationConfig::default()),
                ProcessorConfig::Truncation(TruncationConfig::default()),
            ],
            model_name: Some("gpt-4o-mini".to_owned()),
            ..ContextConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: ContextConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, restored);

        // Unknown fields are tolerated and missing ones fall back to defaults.
        let partial: ContextConfig =
            serde_json::from_str(r#"{"compression_threshold": 0.5, "surprise": true}"#).unwrap();
        assert_eq!(partial.compression_threshold, 0.5);
        assert_eq!(partial.recent_messages, 6);
    }

    #[test]
    fn the_policy_resolves_the_window_only_when_it_is_configured() {
        // An explicit budget always wins.
        let explicit = ContextConfig {
            max_context_tokens: Some(1000),
            context_window_policy: ContextWindowPolicy::new()
                .with_global_tokens(500)
                .with_model_override(300),
            ..ContextConfig::default()
        };
        assert_eq!(explicit.resolved_window().tokens, 1000);
        assert_eq!(
            explicit.resolved_window().source,
            ContextWindowSource::Global
        );
        assert_eq!(explicit.effective_max_tokens(), 1000);

        // A completely default policy keeps today's 200000-token behaviour even
        // when a model name is configured.
        let defaulted = ContextConfig {
            model_name: Some("gpt-4o".to_owned()),
            ..ContextConfig::default()
        };
        assert_eq!(defaulted.resolved_window().tokens, DEFAULT_CONTEXT_TOKENS);
        assert_eq!(
            defaulted.resolved_window().source,
            ContextWindowSource::Default
        );

        let override_window = ContextConfig {
            context_window_policy: ContextWindowPolicy::new().with_model_override(111),
            ..ContextConfig::default()
        };
        assert_eq!(override_window.resolved_window().tokens, 111);
        assert_eq!(
            override_window.resolved_window().source,
            ContextWindowSource::Override
        );

        let model_map = ContextConfig {
            model_name: Some("acme-model".to_owned()),
            context_window_policy: ContextWindowPolicy::new().with_model_window("acme-model", 999),
            ..ContextConfig::default()
        };
        assert_eq!(model_map.resolved_window().tokens, 999);
        assert_eq!(
            model_map.resolved_window().source,
            ContextWindowSource::ConfiguredModel
        );

        let builtin = ContextConfig {
            context_window_policy: ContextWindowPolicy::new().with_model("gpt-4o"),
            ..ContextConfig::default()
        };
        assert_eq!(builtin.resolved_window().tokens, 128_000);
        assert_eq!(
            builtin.resolved_window().source,
            ContextWindowSource::BuiltIn
        );

        let unknown = ContextConfig {
            context_window_policy: ContextWindowPolicy::new().with_model("acme/unknown-model"),
            ..ContextConfig::default()
        };
        assert_eq!(unknown.resolved_window().tokens, DEFAULT_CONTEXT_TOKENS);
        assert_eq!(
            unknown.resolved_window().source,
            ContextWindowSource::Default
        );
    }

    #[test]
    fn the_policy_round_trips_and_compares_without_the_cache_handle() {
        let config = ContextConfig {
            context_window_policy: ContextWindowPolicy::new()
                .with_global_tokens(64_000)
                .with_model_window("acme", 32_000),
            ..ContextConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: ContextConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, restored);
        assert_eq!(
            restored.context_window_policy.context_window_tokens,
            Some(64_000)
        );

        // The cache handle is compared by presence only, so a default policy
        // equals a policy whose handle is absent.
        assert_eq!(ContextConfig::default(), ContextConfig::default());
    }
}
