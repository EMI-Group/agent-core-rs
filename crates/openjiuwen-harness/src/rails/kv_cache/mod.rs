//! KV-cache affinity policy for harness subagents.
//!
//! Ports `openjiuwen/harness/kv_cache/`: the single bridge between harness
//! subagent lifecycle events and the Session-level KV-cache API
//! (`Session.prepare_kvc/suspend_kvc/release_kvc` in Python). The Rust port keeps
//! the same three layers:
//!
//! - `policy` — pure decision helpers (is affinity on, is a type sticky, which
//!   cache identity a child delegation gets). No state, no IO.
//! - `provider` — the provider seam ([`KvCacheProviderRegistry`]) plus the
//!   typed [`UnsupportedKvCacheBackend`] for every backend that is deferred.
//! - `rail` — [`KvCacheSubagentLifecycle`], which tracks one
//!   [`KvCacheSubagentSession`] per delegation and drives the seam, and
//!   [`KvCacheLifecycleRail`], which observes the subagent tools and forwards
//!   their lifecycle events to the tracker.
//!
//! Lifecycle mapping (spec `S_16`): start + sticky → prepare; success + sticky →
//! suspend; success + one-shot → release; failure/cancel/any type → release;
//! explicit eviction → release.
//!
//! Affinity is an optional optimisation: when it is off nothing is stored and no
//! provider call is made, and every lifecycle error is logged and swallowed so
//! cache bookkeeping can never abort a run. Cache state is per tracker instance;
//! there are no module-level singletons.
// The items below are the public surface of `crate::rails`; the parent module
// re-exports them flat, but this subtree may not edit that file. Until that
// re-export lands nothing in-crate reaches these items, so keep the
// reachability, dead-code and unused-import lints quiet.
#![allow(unreachable_pub, dead_code, unused_imports)]
mod policy;
mod provider;
mod rail;
pub use policy::{
    affinity_enabled, is_sticky_subagent_type, resolve_sub_session_id,
    resolve_subagent_parent_cache_id, scope_sub_session_id, KvCacheAffinityConfig, KvCacheLineage,
    RuntimeSessionLineage, AFFINITY_PARENT_SESSION_ID_ENV, STICKY_SUBAGENT_TYPE,
};
pub use provider::{KvCacheProviderRegistry, UnsupportedKvCacheBackend};
pub use rail::{
    KvCacheAction, KvCacheLifecycleRail, KvCacheSubagent, KvCacheSubagentLifecycle,
    KvCacheSubagentSession, KvCacheSubagentState,
};
