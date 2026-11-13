//! Provider-facing identity, lineage, and reuse-range metadata.
//!
//! Ports `openjiuwen/core/kv_cache/kv_cache_metadata.py`: the session env keys
//! that carry KV-cache lineage across a spawn boundary, the env-override
//! resolution used for affinity hints, the range diff that decides what is still
//! reusable, and the derived identities for context compressors and team
//! members.
//!
//! The derived identities are *lineage-only*: their
//! [`KVCacheIdentity::prefix_hash`] is empty because the message payload is not
//! available at this layer. [`crate::kv_cache::hook::KVCacheModelCallHook`]
//! fills the concrete prefix hash in when it records a request.
use crate::kv_cache::types::KVCacheIdentity;
use serde_json::{json, Value};

/// Session env key that overrides the affinity session id.
///
/// This is the persisted wire contract shared by spawners, subagents, and child
/// sessions; renaming it silently loses cache reuse instead of failing.
pub const KV_CACHE_AFFINITY_SESSION_ID_ENV: &str = "kv_cache_affinity_session_id";

/// Session env key that carries the parent session id into a child session.
pub const KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV: &str = "kv_cache_affinity_parent_session_id";

/// Suffix appended to an owner session id to build a compressor cache id.
pub const CONTEXT_COMPRESSOR_CACHE_SUFFIX: &str = "compressor";

/// Trims `value` and drops empty strings; non-string values never reach here.
fn normalize(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Resolves the session id and parent session id used for affinity hints.
///
/// Precedence matches Python: the
/// [`KV_CACHE_AFFINITY_SESSION_ID_ENV`] env override wins over the runtime
/// `session_id`; the explicit `parent_session_id` wins over the
/// [`KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV`] override, and both fall back to
/// the resolved session id so a root session is its own parent.
///
/// `env` is the session env bag lookup (never the process environment); an empty
/// or whitespace-only value is ignored rather than coerced.
#[must_use]
pub fn resolve_session_lineage(
    session_id: Option<&str>,
    parent_session_id: Option<&str>,
    env: impl Fn(&str) -> Option<String>,
) -> (Option<String>, Option<String>) {
    let override_session_id = normalize(env(KV_CACHE_AFFINITY_SESSION_ID_ENV).as_deref());
    let override_parent_id = normalize(env(KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV).as_deref());
    let resolved_session_id = override_session_id.or_else(|| normalize(session_id));
    let resolved_parent_id = normalize(parent_session_id)
        .or(override_parent_id)
        .or_else(|| resolved_session_id.clone());
    (resolved_session_id, resolved_parent_id)
}

/// Returns how many leading items `previous` and `current` share.
///
/// That is the index of the first difference, `0` when they are unrelated (the
/// very first item already differs), and `min(len)` when one list is a prefix of
/// the other — so an append-only conversation reports the old length, which is
/// exactly the reusable prefix. An empty list shares nothing.
#[must_use]
pub fn first_changed_index<T: PartialEq>(previous: &[T], current: &[T]) -> usize {
    previous
        .iter()
        .zip(current)
        .position(|(previous, current)| previous != current)
        .unwrap_or_else(|| previous.len().min(current.len()))
}

/// Renders the Python-shaped half-open message range of the reusable prefix.
///
/// `from_index` is the first changed index from [`first_changed_index`], so the
/// reusable messages are `[0, from_index)` and the dict is
/// `{"msg_start": 0, "msg_end": from_index}`. Both bounds are always present;
/// Python never passes a missing index into a range dict.
#[must_use]
pub fn message_range_kwargs(from_index: usize) -> Value {
    json!({"msg_start": 0, "msg_end": from_index})
}

/// Renders the Python-shaped half-open tool range covering every supplied tool.
///
/// The reusable tool definitions are `[0, tool_names.len())`, rendered as
/// `{"tools_start": 0, "tools_end": len}`.
#[must_use]
pub fn tools_range_kwargs(tool_names: &[String]) -> Value {
    json!({"tools_start": 0, "tools_end": tool_names.len()})
}

/// Renders a self-parent lineage dict for a session that has no parent.
#[must_use]
pub fn self_parent_kwargs(cache_id: &str) -> Value {
    json!({"session_id": cache_id, "parent_session_id": cache_id})
}

/// Returns the stable cache id of one context compressor.
///
/// `session_id` is the owner cache id; the result is `"{owner}:compressor"` with
/// the owner as parent, matching the Python suffix contract. An empty owner
/// yields the bare `"compressor"` id with no parent.
#[must_use]
pub fn context_compressor_cache_identity(
    session_id: &str,
    parent_session_id: Option<&str>,
    message_count: usize,
    tool_count: usize,
    model: &str,
) -> KVCacheIdentity {
    let owner = session_id.trim();
    let cache_id = if owner.is_empty() {
        CONTEXT_COMPRESSOR_CACHE_SUFFIX.to_owned()
    } else {
        format!("{owner}:{CONTEXT_COMPRESSOR_CACHE_SUFFIX}")
    };
    let parent =
        normalize(parent_session_id).or_else(|| (!owner.is_empty()).then(|| owner.to_owned()));
    let mut identity = KVCacheIdentity::new(cache_id, String::new(), model.trim())
        .with_counts(message_count, tool_count);
    identity.parent_session_id = parent;
    identity
}

/// Returns the persisted cache id of one team member.
///
/// The format `"team:{team_session_id}:team:{team_id}:member:{member_id}"` is a
/// public/persisted contract; parts are interpolated verbatim, so callers pass
/// non-empty ids.
#[must_use]
pub fn team_member_cache_id(team_session_id: &str, team_id: &str, member_id: &str) -> String {
    format!("team:{team_session_id}:team:{team_id}:member:{member_id}")
}

/// Returns the lineage identity of one team member of `team_session_id`.
///
/// The member id is a child of the team session (`parent_session_id`), so two
/// members share the team's root while keeping separate prefixes in the same
/// control domain.
#[must_use]
pub fn team_member_cache_identity(
    team_session_id: &str,
    team_id: &str,
    member_id: &str,
    model: &str,
) -> KVCacheIdentity {
    let cache_id = team_member_cache_id(team_session_id, team_id, member_id);
    let mut identity = KVCacheIdentity::new(cache_id, String::new(), model.trim());
    identity.parent_session_id = normalize(Some(team_session_id));
    identity
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn env_from(entries: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: BTreeMap<String, String> = entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        move |key| map.get(key).cloned()
    }

    #[test]
    fn lineage_prefers_explicit_values_then_env() {
        let env = env_from(&[]);
        assert_eq!(
            resolve_session_lineage(Some("session-a"), Some("root"), env),
            (Some("session-a".to_owned()), Some("root".to_owned()))
        );

        let env = env_from(&[
            (KV_CACHE_AFFINITY_SESSION_ID_ENV, "cache-id"),
            (KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV, "cache-parent"),
        ]);
        assert_eq!(
            resolve_session_lineage(Some("session-a"), None, env),
            (Some("cache-id".to_owned()), Some("cache-parent".to_owned()))
        );
    }

    #[test]
    fn lineage_defaults_parent_to_the_resolved_session() {
        let env = env_from(&[]);
        assert_eq!(
            resolve_session_lineage(Some("session-a"), None, env),
            (Some("session-a".to_owned()), Some("session-a".to_owned()))
        );
        assert_eq!(
            resolve_session_lineage(None, None, env_from(&[])),
            (None, None)
        );
    }

    #[test]
    fn lineage_ignores_blank_overrides() {
        let env = env_from(&[
            (KV_CACHE_AFFINITY_SESSION_ID_ENV, "   "),
            (KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV, ""),
        ]);
        assert_eq!(
            resolve_session_lineage(Some(" session-a "), Some(" parent "), env),
            (Some("session-a".to_owned()), Some("parent".to_owned()))
        );
    }

    #[test]
    fn first_changed_index_matches_python_semantics() {
        let previous: Vec<String> = ["sys", "user-1"].iter().map(|v| (*v).to_owned()).collect();
        assert_eq!(first_changed_index(&previous, &previous), 2);

        let appended: Vec<String> = ["sys", "user-1", "assistant", "user-2"]
            .iter()
            .map(|v| (*v).to_owned())
            .collect();
        assert_eq!(first_changed_index(&previous, &appended), 2);
        assert_eq!(first_changed_index(&appended, &previous), 2);

        let unrelated: Vec<String> = ["other", "user-1"]
            .iter()
            .map(|v| (*v).to_owned())
            .collect();
        assert_eq!(first_changed_index(&previous, &unrelated), 0);

        assert_eq!(
            first_changed_index::<String>(&[], &appended),
            0,
            "an empty previous shares no prefix"
        );
    }

    #[test]
    fn range_kwargs_render_half_open_python_ranges() {
        assert_eq!(
            message_range_kwargs(3),
            json!({"msg_start": 0, "msg_end": 3})
        );
        let tools: Vec<String> = ["search", "read"]
            .iter()
            .map(|name| (*name).to_owned())
            .collect();
        assert_eq!(
            tools_range_kwargs(&tools),
            json!({"tools_start": 0, "tools_end": 2})
        );
        assert_eq!(
            tools_range_kwargs(&[]),
            json!({"tools_start": 0, "tools_end": 0})
        );
        assert_eq!(
            self_parent_kwargs("session-a"),
            json!({"session_id": "session-a", "parent_session_id": "session-a"})
        );
    }

    #[test]
    fn compressor_identity_is_a_child_of_its_owner() {
        let identity = context_compressor_cache_identity("session-a", None, 6, 2, "gpt-test");
        assert_eq!(identity.session_id, "session-a:compressor");
        assert_eq!(identity.parent(), Some("session-a"));
        assert_eq!(identity.model, "gpt-test");
        assert_eq!(identity.message_count, 6);
        assert_eq!(identity.tool_count, 2);
        assert_eq!(identity.prefix_hash, "");

        let empty = context_compressor_cache_identity("", None, 0, 0, "");
        assert_eq!(empty.session_id, CONTEXT_COMPRESSOR_CACHE_SUFFIX);
        assert!(empty.parent().is_none());
    }

    #[test]
    fn team_member_identity_keeps_the_persisted_format() {
        assert_eq!(
            team_member_cache_id("team-session", "team-1", "member-1"),
            "team:team-session:team:team-1:member:member-1"
        );
        let identity = team_member_cache_identity("team-session", "team-1", "member-1", "gpt-test");
        assert_eq!(
            identity.session_id,
            "team:team-session:team:team-1:member:member-1"
        );
        assert_eq!(identity.parent(), Some("team-session"));
        assert_eq!(identity.model, "gpt-test");
        assert_eq!(identity.prefix_hash, "");
    }
}
