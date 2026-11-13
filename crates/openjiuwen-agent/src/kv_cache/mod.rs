//! KV-cache affinity: runtime/config/metadata types and a request-affinity hook.
//!
//! Ports `openjiuwen/core/kv_cache/` plus `core/single_agent/kv_cache/`: an
//! in-process cache of request prefixes, the session-lineage metadata that makes
//! a request reusable, and the model-call hook that records and matches the
//! prefix identity.
