//! KV-cache affinity: runtime/config/metadata types and a request-affinity hook.
//!
//! Ports `openjiuwen/core/kv_cache/` plus `core/single_agent/kv_cache/`: an
//! in-process cache of request prefixes, the session-lineage metadata that makes
//! a request reusable, and the model-call hook that records and matches the
//! prefix identity.
//!
//! # Model
//!
//! - [`KVCacheIdentity`] names a reusable prefix: session lineage, the
//!   deterministic [`stable_prefix_hash`] of the messages and tool definitions,
//!   the model, and the covered message/tool counts.
//! - [`KVCacheEntry`] is the runtime's bookkeeping for one identity (covered
//!   message count, token estimate, `LRU` sequence, reuse count).
//! - [`InProcessKVCacheRuntime`] stores entries behind one mutex, evicts
//!   least-recently-used entries at capacity, expires idle entries by `TTL`, and
//!   can drop a whole session lineage.
//! - [`KVCacheModelCallHook`] looks up the longest reusable prefix before a model
//!   call, records the request that is about to be sent, accepts a token
//!   estimate afterwards, and renders child-session lineage kwargs.
//! - [`metadata`] carries the session env keys, lineage resolution, and the
//!   range/identity derivations used by context compressors and team members.
//! - [`KvCacheActionRuntime`] is the provider-action executor: it tracks
//!   admission and residency per cache scope, runs
//!   [`KvCacheAction::Prefetch`]/[`KvCacheAction::Offload`]/[`KvCacheAction::Evict`]
//!   through a [`KvCacheActionProvider`], and fails open so inference never waits
//!   on cache management.
//! - [`InProcessKvCacheActionProvider`] is the deterministic provider used by
//!   tests and hosts without a provider integration: every action succeeds and is
//!   recorded.
//!
//! Affinity is opt-in through [`KVCacheAffinityConfig::enabled`]; with the
//! default config the hook performs no runtime work and reports a miss, and the
//! action runtime is inert.
//!
//! # Deferred
//!
//! The `Redis`/`Distributed` prefix bookkeeping backends and provider HTTP
//! transports are not implemented: [`KVCacheBackend::runtime`] returns
//! `Error::Unsupported` with an explicit message for them instead of silently
//! falling back to the in-process runtime, and this crate ships only the
//! in-process action provider (a real provider implements
//! [`KvCacheActionProvider`] itself).
//!
//! ```
//! use openjiuwen_agent::kv_cache::{
//!     resolve_session_lineage, stable_prefix_hash, KVCacheBackend, KVCacheIdentity,
//!     KVCacheRuntimeConfig,
//! };
//!
//! let identity = KVCacheIdentity::new(
//!     "session-a",
//!     stable_prefix_hash(&[], &[], "gpt-test"),
//!     "gpt-test",
//! )
//! .with_parent("session-a");
//! assert!(identity.is_root());
//!
//! let config = KVCacheRuntimeConfig::default();
//! assert!(KVCacheBackend::InProcess.runtime(&config).is_ok());
//! assert!(KVCacheBackend::Redis.runtime(&config).is_err());
//!
//! let (session, parent) = resolve_session_lineage(Some("child"), Some("root"), |_| None);
//! assert_eq!(session.as_deref(), Some("child"));
//! assert_eq!(parent.as_deref(), Some("root"));
//! ```
pub mod action_runtime;
pub mod actions;
pub mod backend;
pub mod config;
pub mod hook;
pub mod metadata;
pub mod runtime;
pub mod types;

pub use action_runtime::{
    KvCacheActionRuntime, KvCacheAdmission, KvCacheBinding, KvCacheInferenceLease, KvCacheResidency,
};
pub use actions::{
    KvCacheAction, KvCacheActionProvider, KvCacheActionRecord, KvCacheActionRequest,
    KvCacheActionStats, KV_CACHE_ACTION_TARGET_SESSION,
};
pub use backend::InProcessKvCacheActionProvider;
pub use config::{
    resolve_kvc_action_timeout, KVCacheAffinityConfig, KVCacheRuntimeConfig,
    DEFAULT_KV_CACHE_CAPACITY, KVC_MANAGEMENT_MAX_ATTEMPTS, KVC_RANGE_ACTION_TIMEOUT_SECONDS,
    KVC_SESSION_EVICT_TIMEOUT_SECONDS, KVC_SESSION_OFFLOAD_PREFETCH_TIMEOUT_SECONDS,
    KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS,
};
pub use hook::{KVCacheCallCapabilities, KVCacheModelCallHook};
pub use metadata::{
    context_compressor_cache_identity, first_changed_index, message_range_kwargs,
    resolve_session_lineage, self_parent_kwargs, team_member_cache_id, team_member_cache_identity,
    tools_range_kwargs, CONTEXT_COMPRESSOR_CACHE_SUFFIX, KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV,
    KV_CACHE_AFFINITY_SESSION_ID_ENV,
};
pub use runtime::{stable_prefix_hash, InProcessKVCacheRuntime};
pub use types::{
    KVCacheBackend, KVCacheEntry, KVCacheIdentity, KVCacheLookup, KVCacheRuntimeProtocol,
    KVCacheStats,
};

impl KVCacheRuntimeConfig {
    /// Builds the runtime this configuration selects.
    ///
    /// # Errors
    ///
    /// Returns `Error::Unsupported` for a deferred backend (see
    /// [`KVCacheBackend::runtime`]).
    pub fn build_runtime(
        &self,
    ) -> openjiuwen_common::error::Result<std::sync::Arc<dyn KVCacheRuntimeProtocol>> {
        self.backend.runtime(self)
    }
}
