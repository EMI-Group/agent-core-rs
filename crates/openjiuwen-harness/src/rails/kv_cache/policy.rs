//! KV-cache affinity policy: the decision helpers of the harness-side lifecycle bridge.
//!
//! Ports `openjiuwen/harness/kv_cache/kv_cache_subagent_lifecycle.py`. This
//! module owns *policy only* — whether affinity is on, which subagent types are
//! sticky, and which cache identity a child delegation receives. Cache state,
//! provider calls and the action tail live in [`super::rail`] and
//! [`super::provider`], mirroring the Python split between this adapter and
//! `openjiuwen/core/kv_cache/`.
//!
//! Every helper is best-effort: lineage resolution degrades to the runtime
//! session id with a warning instead of surfacing an error into the delegation
//! path, exactly like the Python adapter.
use crate::rails::sha256::sha256_prefix;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
/// Environment variable carrying the parent cache identity to a child session.
///
/// Mirrors `openjiuwen.core.kv_cache.kv_cache_metadata.KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV`.
pub const AFFINITY_PARENT_SESSION_ID_ENV: &str = "kv_cache_affinity_parent_session_id";
/// The only subagent type whose model session is kept sticky across turns.
///
/// Browser process/profile reuse is owned by the browser registry, so
/// `browser_agent` is deliberately *not* sticky: keeping its model session
/// alive would leak the previous query's tools and page state into an unrelated
/// delegation.
pub const STICKY_SUBAGENT_TYPE: &str = "verification_agent";
/// Number of SHA-256 hex characters used to scope a child session id.
const SCOPE_DIGEST_CHARS: usize = 12;
/// The harness-side control for KV-cache affinity.
///
/// Python reads `deep_agent.deep_config.kv_cache_affinity_config`; the Rust port
/// receives the same data explicitly, so no live agent handle is consulted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KvCacheAffinityConfig {
    /// Whether subagent cache identities are prepared, suspended and released.
    pub enable_kv_cache_affinity: bool,
}
/// Reports whether the affinity optimisation is on.
///
/// A missing config and the [`Default`] config both mean "off", matching the
/// Python identity check against the literal `True`: an OFF deployment must not
/// even create child cache identities.
#[must_use]
pub fn affinity_enabled(config: Option<&KvCacheAffinityConfig>) -> bool {
    config.is_some_and(|config| config.enable_kv_cache_affinity)
}
/// Whether `subagent_type` keeps its model session sticky across turns.
///
/// The comparison trims surrounding whitespace and is otherwise exact, so
/// `" verification_agent "` is sticky while `"browser_agent"` is not.
#[must_use]
pub fn is_sticky_subagent_type(subagent_type: &str) -> bool {
    subagent_type.trim() == STICKY_SUBAGENT_TYPE
}
/// Resolves the provider-facing parent cache identity of a runtime session.
///
/// Implemented by upper layers that own the cache metadata store; the harness
/// only asks for the identity, it never touches provider-level state.
pub trait KvCacheLineage: Send + Sync {
    /// Returns the cache identity the provider uses for `runtime_session_id`.
    ///
    /// # Errors
    ///
    /// An identity/lineage lookup failure. [`resolve_subagent_parent_cache_id`]
    /// turns it into a warning plus the runtime session id, so callers in the
    /// delegation path stay infallible.
    fn resolve_parent_cache_id(&self, runtime_session_id: &str) -> Result<String>;
}
/// The default lineage: the runtime session id *is* the cache identity.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeSessionLineage;
impl KvCacheLineage for RuntimeSessionLineage {
    fn resolve_parent_cache_id(&self, runtime_session_id: &str) -> Result<String> {
        Ok(runtime_session_id.to_string())
    }
}
/// Resolves the parent cache id, degrading to the runtime id on any failure.
///
/// Python logs `[HarnessKVC] parent lineage resolution failed` and keeps going;
/// this port does the same, because losing lineage costs an optimisation, not
/// correctness. An empty lineage result also falls back to the runtime id.
#[must_use]
pub fn resolve_subagent_parent_cache_id(
    lineage: Option<&dyn KvCacheLineage>,
    runtime_session_id: &str,
) -> String {
    let fallback = runtime_session_id.trim().to_string();
    let Some(lineage) = lineage else {
        return fallback;
    };
    match lineage.resolve_parent_cache_id(runtime_session_id) {
        Ok(cache_id) => {
            let cache_id = cache_id.trim();
            if cache_id.is_empty() {
                fallback
            } else {
                cache_id.to_string()
            }
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                runtime_session_id,
                "harness kv_cache: parent lineage resolution failed; using runtime session"
            );
            fallback
        }
    }
}
/// Disambiguates a child session id when the parent cache identity differs.
///
/// A Team member child inherits its parent's runtime session id, so two
/// children of one parent would otherwise collide. Appending
/// `_scope_<sha256(parent_cache_id)[..12]>` keeps the runtime id path-safe while
/// separating their cache identities. When the ids match (the common case) the
/// id is returned unchanged.
#[must_use]
pub fn scope_sub_session_id(
    sub_session_id: &str,
    runtime_parent_session_id: &str,
    parent_cache_id: &str,
) -> String {
    if parent_cache_id.is_empty() || parent_cache_id == runtime_parent_session_id {
        return sub_session_id.to_string();
    }
    let digest = sha256_prefix(parent_cache_id, SCOPE_DIGEST_CHARS);
    format!("{sub_session_id}_scope_{digest}")
}
/// Resolves the child session id of a delegation.
///
/// A metadata-provided `sub_session_id` wins; otherwise the id is
/// `{parent_session_id}_sub_{task_id}`, with a blank task id replaced by
/// `"unknown"` so two anonymous delegations do not silently share an identity.
#[must_use]
pub fn resolve_sub_session_id(
    task_id: &str,
    parent_session_id: &str,
    metadata_sub_session_id: Option<&str>,
) -> String {
    if let Some(sub_session_id) = metadata_sub_session_id.filter(|id| !id.is_empty()) {
        return sub_session_id.to_string();
    }
    let safe_task_id = match task_id.trim() {
        "" => "unknown",
        trimmed => trimmed,
    };
    format!("{parent_session_id}_sub_{safe_task_id}")
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::Error;
    use serde_json::json;
    fn config(enabled: bool) -> KvCacheAffinityConfig {
        KvCacheAffinityConfig {
            enable_kv_cache_affinity: enabled,
        }
    }
    #[test]
    fn affinity_is_off_by_default_and_only_literal_true_counts() {
        assert!(!affinity_enabled(None));
        assert!(!affinity_enabled(Some(&KvCacheAffinityConfig::default())));
        assert!(!affinity_enabled(Some(&config(false))));
        assert!(affinity_enabled(Some(&config(true))));
    }
    #[test]
    fn stickiness_is_exactly_the_verification_agent() {
        assert!(is_sticky_subagent_type("verification_agent"));
        assert!(is_sticky_subagent_type(" verification_agent "));
        assert!(!is_sticky_subagent_type("verification_agent_2"));
        assert!(!is_sticky_subagent_type("browser_agent"));
        assert!(!is_sticky_subagent_type(""));
    }
    #[test]
    fn scoping_keeps_matching_ids_and_digests_differing_ones() {
        assert_eq!(scope_sub_session_id("child", "parent", "parent"), "child");
        assert_eq!(scope_sub_session_id("child", "parent", ""), "child");
        let scoped = scope_sub_session_id("child", "runtime-parent", "cache-parent");
        let digest = sha256_prefix("cache-parent", 12);
        assert_eq!(scoped, format!("child_scope_{digest}"));
        let suffix = scoped.strip_prefix("child_scope_").unwrap();
        assert_eq!(suffix.len(), 12);
        assert!(suffix
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));
        assert_eq!(
            scoped,
            scope_sub_session_id("child", "runtime-parent", "cache-parent")
        );
        assert_ne!(
            scoped,
            scope_sub_session_id("child", "runtime-parent", "other-parent")
        );
    }
    #[test]
    fn sub_session_id_prefers_metadata_and_defaults_to_unknown() {
        assert_eq!(
            resolve_sub_session_id("task-1", "parent", Some("explicit")),
            "explicit"
        );
        assert_eq!(
            resolve_sub_session_id("task-1", "parent", Some("")),
            "parent_sub_task-1"
        );
        assert_eq!(
            resolve_sub_session_id("task-1", "parent", None),
            "parent_sub_task-1"
        );
        assert_eq!(
            resolve_sub_session_id("", "parent", None),
            "parent_sub_unknown"
        );
        assert_eq!(
            resolve_sub_session_id("   ", "parent", None),
            "parent_sub_unknown"
        );
        assert_eq!(
            resolve_sub_session_id(" t ", "parent", None),
            "parent_sub_t"
        );
    }
    /// A lineage that always fails.
    struct FailingLineage;
    impl KvCacheLineage for FailingLineage {
        fn resolve_parent_cache_id(&self, _runtime_session_id: &str) -> Result<String> {
            Err(Error::config("lineage store unavailable"))
        }
    }
    /// A lineage that answers with a fixed identity.
    struct FixedLineage(&'static str);
    impl KvCacheLineage for FixedLineage {
        fn resolve_parent_cache_id(&self, _runtime_session_id: &str) -> Result<String> {
            Ok(self.0.to_string())
        }
    }
    #[test]
    fn parent_cache_id_uses_lineage_and_degrades_on_failure() {
        assert_eq!(
            resolve_subagent_parent_cache_id(Some(&FixedLineage("cache-9")), "runtime-1"),
            "cache-9"
        );
        assert_eq!(
            resolve_subagent_parent_cache_id(Some(&FixedLineage("  ")), "runtime-1"),
            "runtime-1"
        );
        assert_eq!(
            resolve_subagent_parent_cache_id(Some(&FailingLineage), " runtime-1 "),
            "runtime-1"
        );
        assert_eq!(
            resolve_subagent_parent_cache_id(None, "runtime-1"),
            "runtime-1"
        );
        let default_lineage = RuntimeSessionLineage;
        assert_eq!(
            resolve_subagent_parent_cache_id(Some(&default_lineage), "runtime-2"),
            "runtime-2"
        );
    }
    #[test]
    fn config_round_trips_and_tolerates_unknown_fields() {
        let encoded = serde_json::to_value(config(true)).unwrap();
        assert_eq!(encoded, json!({"enable_kv_cache_affinity": true}));
        let decoded: KvCacheAffinityConfig = serde_json::from_value(json!({
            "enable_kv_cache_affinity": true,
            "future_field": 7
        }))
        .unwrap();
        assert_eq!(decoded, config(true));
        let empty: KvCacheAffinityConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, KvCacheAffinityConfig::default());
    }
}
