//! Provider-specific context-window policy and KV-cache diffing.
//!
//! Ports `context/context_utils.py`'s window resolution (global cap, selected
//! model override, configured model map, built-in table, `OpenRouter` cache,
//! default) and the `ContextWindowChange` / `detect_context_window_change`
//! primitives from `base.py` and `context/context.py`, which describe whether a
//! prompt prefix can still be reused after a window is rebuilt.
//!
//! - [`window`] — [`ContextWindowPolicy`], [`ResolvedContextWindow`],
//!   [`ContextWindowSource`], and the built-in
//!   [`MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS`] table.
//! - [`change`] — [`ContextWindowSnapshot`], [`ContextWindowChange`],
//!   [`detect_context_window_change`], [`diff_positions`], and the
//!   [`PromptCacheImpact`] summary.
//! - [`openrouter`] — the process-wide [`OpenRouterWindowCache`], the catalogue
//!   parser, and the opt-in fetch/prefetch entry points.
//!
//! Everything except the explicit `fetch`/`prefetch` calls is offline and
//! synchronous, so resolution and diffing are safe on a context-creation path.
//!
//! ```
//! use openjiuwen_context::policy::{ContextWindowPolicy, ContextWindowSource};
//!
//! // No configuration: the built-in table answers for a known family and the
//! // 200000-token default answers for everything else.
//! let policy = ContextWindowPolicy::new();
//! let resolved = policy.resolve(Some("gpt-4o"), None);
//! assert_eq!(resolved.tokens, 128_000);
//! assert_eq!(resolved.source, ContextWindowSource::BuiltIn);
//!
//! let unknown = policy.resolve(Some("acme/unknown-model"), None);
//! assert_eq!(unknown.tokens, 200_000);
//! assert_eq!(unknown.source, ContextWindowSource::Default);
//! ```
pub mod change;
pub mod openrouter;
pub mod window;

pub use change::{
    detect_context_window_change, diff_positions, tools_signature, ContextWindowChange,
    ContextWindowSnapshot, MessageFingerprint, PositionChange, PromptCacheImpact,
};
pub use openrouter::{
    global_openrouter_window_cache, openrouter_disabled_error, openrouter_invalid_response_error,
    openrouter_unavailable_error, parse_openrouter_models, OpenRouterModelTable,
    OpenRouterWindowCache, DEFAULT_OPENROUTER_CACHE_TTL, DEFAULT_OPENROUTER_REQUEST_TIMEOUT,
    OPENROUTER_DISABLED_CODE, OPENROUTER_INVALID_RESPONSE_CODE, OPENROUTER_MODELS_URL,
    OPENROUTER_UNAVAILABLE_CODE,
};
pub use window::{
    lookup_model_default_window, ContextWindowPolicy, ContextWindowSource, ResolvedContextWindow,
    MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS,
};
