//! Team-member KV-cache affinity and terminal cleanup.
//!
//! Ports `openjiuwen/agent_teams/kv_cache/` (Python v0.1.20): the team runtime's
//! policy layer over the agent kernel's KV-cache primitives. It binds every team
//! member session to a stable product-level cache identity, derives forked child
//! identities, and runs a bounded, cancellation-safe release-then-dispose at
//! session teardown so provider-side cache cannot outlive a team.
//!
//! # Model
//!
//! - [`MemberCacheKey`] is the product identity of a member cache: team name,
//!   team session and member name.
//! - [`TeamKvCacheAffinity`] is the per-team registry. A root member gets
//!   [`team_member_cache_identity`](openjiuwen_agent::kv_cache::team_member_cache_identity)
//!   (persisted `team:{session}:team:{team}:member:{member}` cache id, parented
//!   at the team session, empty prefix hash, zero counts); a forked member gets
//!   `"{parent_cache_id}:fork:{child}"` as its cache id and the parent cache id
//!   as lineage.
//! - [`cancellation_safe_release_then_dispose`] evicts a member's cache and then
//!   disposes the owning runtime inside one spawned, timeout-bounded task, so a
//!   cancelled caller cannot leave cache behind.
//!
//! All cache I/O is delegated to
//! [`KVCacheRuntimeProtocol`](openjiuwen_agent::kv_cache::KVCacheRuntimeProtocol)
//! and the model-call hook is the agent crate's
//! [`KVCacheModelCallHook`](openjiuwen_agent::kv_cache::KVCacheModelCallHook);
//! no identity, runtime or hook type is redefined here.
//!
//! # Wiring
//!
//! A runtime registers each member once it knows the team session id and the
//! member's model, hands [`TeamKvCacheAffinity::env`] to the member session, and
//! releases the member when its work settles —
//! [`TeamKvCacheAffinity::release`] for a one-shot worker,
//! [`TeamKvCacheAffinity::release_team`] when the whole team stops. When the
//! session also owns a runtime, wrap the dispose in
//! [`cancellation_safe_release_then_dispose`] with
//! [`terminal_cleanup_timeout`] (or
//! [`KVCacheAffinityConfig::effective_terminal_cleanup_timeout`](openjiuwen_agent::kv_cache::KVCacheAffinityConfig::effective_terminal_cleanup_timeout)).
//!
//! # Disabled
//!
//! Affinity is opt-in through
//! [`KVCacheAffinityConfig`](openjiuwen_agent::kv_cache::KVCacheAffinityConfig).
//! With a disabled config the registry stores nothing, every accessor returns
//! `None`, `release` evicts nothing and no runtime is built.
//!
//! ```
//! use openjiuwen_agent::kv_cache::KVCacheAffinityConfig;
//! use openjiuwen_agent_teams::kv_cache::{MemberCacheKey, TeamKvCacheAffinity};
//!
//! let registry = TeamKvCacheAffinity::new(KVCacheAffinityConfig::enabled());
//! let key = MemberCacheKey::new("team-1", "team-session", "dev-1");
//! let identity = registry.register_member(key, "gpt-test");
//! assert_eq!(identity.session_id, "team:team-session:team:team-1:member:dev-1");
//! assert_eq!(registry.env("dev-1").unwrap().len(), 2);
//! ```
//!
//! # Deferred
//!
//! Provider-side prefetch/offload, `Redis`/distributed runtimes, the harness
//! `DeepAgent` session-lifecycle hooks, and cross-process member affinity are not
//! implemented; see this directory's `CONTEXT.md`.
pub mod affinity;
pub mod cleanup;
pub use affinity::{MemberCacheEntry, MemberCacheKey, TeamKvCacheAffinity, MEMBER_FORK_SEGMENT};
pub use cleanup::{
    cancellation_safe_release_then_dispose, terminal_cleanup_timeout, CleanupReport,
};
