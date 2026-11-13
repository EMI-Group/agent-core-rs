//! KV keys and vector collection names.
//!
//! The key layout is the Python `openjiuwen.core.memory` layout, so an operator can
//! inspect the same prefixes in any backend:
//!
//! - records: `UMD/{user_id}/{scope_id}/{record_id}`
//! - user variables: `user_var/{user_id}/{scope_id}/{name}`
//! - session variables: `session_var/{user_id}/{scope_id}/{session_id}/{name}`
//! - scope configuration: `memory_scope_config/{scope_id}`
//!
//! Vector collections are named `uid_{user}_gid_{scope}_mtype_{kind}`; every segment
//! passes through [`sanitize_segment`], because collection names become file names
//! in the local-filesystem backend.
//!
//! Record and variable keys use the raw scope ids: [`MemoryScope::validate`] already
//! rejects `/` in `scope_id`/`session_id`, and the user id is kept verbatim so the
//! Python-compatible layout stays inspectable.
//!
//! # Legacy key layout
//!
//! Python's `KvPrefixRegistry` (`core/memory/common/kv_prefix_registry.py`) tracks a
//! *current* and a *legacy* prefix per memory data family so a renamed keyspace stays
//! discoverable. Python v0.1.20 registers **no** legacy prefix (`UserMemStore.
//! LEGACY_PREFIXES` and `VariableManager.LEGACY_PREFIXES` are both `[]`), so the
//! compatibility seam below is declared here explicitly rather than ported:
//!
//! | data family | current prefix | legacy prefix |
//! | --- | --- | --- |
//! | records | [`RECORD_PREFIX`] (`UMD`) | [`LEGACY_RECORD_PREFIX`] (`user_mem`) |
//! | user variables | [`VARIABLE_PREFIX`] (`user_var`) | [`LEGACY_VARIABLE_PREFIX`] (`user_variable`) |
//! | session variables | [`SESSION_VARIABLE_PREFIX`] (`session_var`) | [`LEGACY_SESSION_VARIABLE_PREFIX`] (`session_variable`) |
//! | scope config | [`SCOPE_CONFIG_PREFIX`] (`memory_scope_config`) | [`LEGACY_SCOPE_CONFIG_PREFIX`] (`scope_config`) |
//!
//! The legacy layout has the same shape as the current one, only the first segment
//! differs (`user_mem/{user}/{scope}/{id}`, and so on). **Precedence rule: writes always
//! use the current layout; legacy keys are read-only compatibility.** A legacy key is
//! consulted only when the current key is absent, and deletes remove both variants.
//! [`legacy_alternatives`] derives the legacy candidates of a current key or prefix, and
//! [`current_equivalent`] maps back for diagnostics.
//!
//! [`MemoryScope::validate`]: crate::record::MemoryScope::validate

use crate::record::{MemoryScope, MemoryType};

/// KV namespace prefix of memory records (Python `UMD`).
pub const RECORD_PREFIX: &str = "UMD";
/// KV namespace prefix of user-scoped variables (Python `user_var`).
pub const VARIABLE_PREFIX: &str = "user_var";
/// KV namespace prefix of session-scoped variables (Python `session_var`).
pub const SESSION_VARIABLE_PREFIX: &str = "session_var";
/// KV namespace prefix of per-scope configuration (Python `memory_scope_config`).
pub const SCOPE_CONFIG_PREFIX: &str = "memory_scope_config";
/// Document collection holding conversation messages.
pub const MESSAGES_COLLECTION: &str = "messages";

/// Legacy KV namespace prefix of memory records (`UMD` superseded `user_mem`).
pub const LEGACY_RECORD_PREFIX: &str = "user_mem";
/// Legacy KV namespace prefix of user-scoped variables.
pub const LEGACY_VARIABLE_PREFIX: &str = "user_variable";
/// Legacy KV namespace prefix of session-scoped variables.
pub const LEGACY_SESSION_VARIABLE_PREFIX: &str = "session_variable";
/// Legacy KV namespace prefix of per-scope configuration.
pub const LEGACY_SCOPE_CONFIG_PREFIX: &str = "scope_config";

/// The single `(current, legacy)` prefix table every legacy helper reads.
///
/// Keeping one table makes the two mapping directions ([`legacy_prefix`] and
/// [`current_prefix`]) impossible to drift apart.
const LEGACY_PREFIX_PAIRS: &[(&str, &str)] = &[
    (RECORD_PREFIX, LEGACY_RECORD_PREFIX),
    (VARIABLE_PREFIX, LEGACY_VARIABLE_PREFIX),
    (SESSION_VARIABLE_PREFIX, LEGACY_SESSION_VARIABLE_PREFIX),
    (SCOPE_CONFIG_PREFIX, LEGACY_SCOPE_CONFIG_PREFIX),
];

/// The 32-bit FNV-1a offset basis.
const FNV1A_32_OFFSET_BASIS: u32 = 0x811c_9dc5;
/// The 32-bit FNV-1a prime.
const FNV1A_32_PRIME: u32 = 0x0100_0193;

/// Key of one record: `UMD/{user_id}/{scope_id}/{id}`.
#[must_use]
pub fn record_key(scope: &MemoryScope, id: &str) -> String {
    format!("{RECORD_PREFIX}/{}/{}/{id}", scope.user_id, scope.scope_id)
}

/// Key prefix of every record in `scope`: `UMD/{user_id}/{scope_id}/`.
#[must_use]
pub fn record_prefix(scope: &MemoryScope) -> String {
    format!("{RECORD_PREFIX}/{}/{}/", scope.user_id, scope.scope_id)
}

/// Key prefix of every record of `user_id`: `UMD/{user_id}/`.
#[must_use]
pub fn record_prefix_for(user_id: &str) -> String {
    format!("{RECORD_PREFIX}/{user_id}/")
}

/// Key prefix of every record with `scope_id`, regardless of user: `UMD/`.
///
/// The scope id is not part of the key ahead of the user id, so a scope-wide scan is
/// the whole record namespace and the caller filters by scope afterwards. The
/// `_scope_id` argument documents the intent and keeps the Python call shape.
#[must_use]
pub fn record_prefix_for_scope(_scope_id: &str) -> String {
    format!("{RECORD_PREFIX}/")
}

/// Key of one user-scoped variable: `user_var/{user_id}/{scope_id}/{name}`.
#[must_use]
pub fn variable_key(scope: &MemoryScope, name: &str) -> String {
    format!(
        "{VARIABLE_PREFIX}/{}/{}/{name}",
        scope.user_id, scope.scope_id
    )
}

/// Key prefix of every user-scoped variable in `scope`.
#[must_use]
pub fn variable_prefix(scope: &MemoryScope) -> String {
    format!("{VARIABLE_PREFIX}/{}/{}/", scope.user_id, scope.scope_id)
}

/// Key of one session-scoped variable:
/// `session_var/{user_id}/{scope_id}/{session_id}/{name}`.
#[must_use]
pub fn session_variable_key(scope: &MemoryScope, session_id: &str, name: &str) -> String {
    format!(
        "{SESSION_VARIABLE_PREFIX}/{}/{}/{session_id}/{name}",
        scope.user_id, scope.scope_id
    )
}

/// Key prefix of every session-scoped variable of `session_id`.
#[must_use]
pub fn session_variable_prefix(scope: &MemoryScope, session_id: &str) -> String {
    format!(
        "{SESSION_VARIABLE_PREFIX}/{}/{}/{session_id}/",
        scope.user_id, scope.scope_id
    )
}

/// Key of the configuration of `scope_id`: `memory_scope_config/{scope_id}`.
#[must_use]
pub fn scope_config_key(scope_id: &str) -> String {
    format!("{SCOPE_CONFIG_PREFIX}/{scope_id}")
}

/// Legacy key of one record: `user_mem/{user_id}/{scope_id}/{id}`.
#[must_use]
pub fn legacy_record_key(scope: &MemoryScope, id: &str) -> String {
    format!(
        "{LEGACY_RECORD_PREFIX}/{}/{}/{id}",
        scope.user_id, scope.scope_id
    )
}

/// Legacy key prefix of every record in `scope`: `user_mem/{user_id}/{scope_id}/`.
#[must_use]
pub fn legacy_record_prefix(scope: &MemoryScope) -> String {
    format!(
        "{LEGACY_RECORD_PREFIX}/{}/{}/",
        scope.user_id, scope.scope_id
    )
}

/// Legacy key prefix of every record of `user_id`: `user_mem/{user_id}/`.
#[must_use]
pub fn legacy_record_prefix_for(user_id: &str) -> String {
    format!("{LEGACY_RECORD_PREFIX}/{user_id}/")
}

/// Legacy record prefix of every record with `_scope_id`: `user_mem/`.
///
/// As in [`record_prefix_for_scope`], the scope id is not part of the key ahead of the
/// user id, so the whole legacy record namespace is scanned and filtered afterwards.
#[must_use]
pub fn legacy_record_prefix_for_scope(_scope_id: &str) -> String {
    format!("{LEGACY_RECORD_PREFIX}/")
}

/// Legacy key of one user-scoped variable:
/// `user_variable/{user_id}/{scope_id}/{name}`.
#[must_use]
pub fn legacy_variable_key(scope: &MemoryScope, name: &str) -> String {
    format!(
        "{LEGACY_VARIABLE_PREFIX}/{}/{}/{name}",
        scope.user_id, scope.scope_id
    )
}

/// Legacy key prefix of every user-scoped variable in `scope`.
#[must_use]
pub fn legacy_variable_prefix(scope: &MemoryScope) -> String {
    format!(
        "{LEGACY_VARIABLE_PREFIX}/{}/{}/",
        scope.user_id, scope.scope_id
    )
}

/// Legacy key of one session-scoped variable:
/// `session_variable/{user_id}/{scope_id}/{session_id}/{name}`.
#[must_use]
pub fn legacy_session_variable_key(scope: &MemoryScope, session_id: &str, name: &str) -> String {
    format!(
        "{LEGACY_SESSION_VARIABLE_PREFIX}/{}/{}/{session_id}/{name}",
        scope.user_id, scope.scope_id
    )
}

/// Legacy key prefix of every session-scoped variable of `session_id`.
#[must_use]
pub fn legacy_session_variable_prefix(scope: &MemoryScope, session_id: &str) -> String {
    format!(
        "{LEGACY_SESSION_VARIABLE_PREFIX}/{}/{}/{session_id}/",
        scope.user_id, scope.scope_id
    )
}

/// Legacy key of the configuration of `scope_id`: `scope_config/{scope_id}`.
#[must_use]
pub fn legacy_scope_config_key(scope_id: &str) -> String {
    format!("{LEGACY_SCOPE_CONFIG_PREFIX}/{scope_id}")
}

/// Legacy keys to try when a current-layout key or prefix misses.
///
/// The key is split at its first `/`; when the head is a current prefix the same key
/// with the legacy head is returned, otherwise the result is empty. Because a prefix
/// ends with `/` the identical first-segment rewrite works for point keys and for
/// prefixes. Returns one candidate per matching family.
#[must_use]
pub fn legacy_alternatives(key: &str) -> Vec<String> {
    let Some((head, rest)) = key.split_once('/') else {
        return Vec::new();
    };
    match legacy_prefix(head) {
        Some(legacy) => vec![format!("{legacy}/{rest}")],
        None => Vec::new(),
    }
}

/// The current-layout equivalent of a legacy key or prefix, for diagnostics.
///
/// The exact inverse of [`legacy_alternatives`]: the legacy head is replaced by its
/// current counterpart and the rest is preserved. `None` for an unknown or head-only
/// key.
#[must_use]
pub fn current_equivalent(key: &str) -> Option<String> {
    let (head, rest) = key.split_once('/')?;
    let current = current_prefix(head)?;
    Some(format!("{current}/{rest}"))
}

/// The legacy prefix registered for a current prefix, when one exists.
#[must_use]
pub fn legacy_prefix(current_prefix: &str) -> Option<&'static str> {
    LEGACY_PREFIX_PAIRS
        .iter()
        .find(|(current, _)| *current == current_prefix)
        .map(|(_, legacy)| *legacy)
}

/// The current prefix that supersedes a legacy prefix, when one exists.
#[must_use]
pub fn current_prefix(legacy_prefix: &str) -> Option<&'static str> {
    LEGACY_PREFIX_PAIRS
        .iter()
        .find(|(_, legacy)| *legacy == legacy_prefix)
        .map(|(current, _)| *current)
}

/// Name of the vector collection of one `(scope, kind)` pair.
///
/// Follows the Python `uid_{user}_gid_{scope}_mtype_{kind}` shape with every segment
/// sanitized for the local-filesystem vector backend.
#[must_use]
pub fn vector_collection(scope: &MemoryScope, kind: MemoryType) -> String {
    format!(
        "uid_{}_gid_{}_mtype_{}",
        sanitize_segment(&scope.user_id),
        sanitize_segment(&scope.scope_id),
        kind.as_str()
    )
}

/// Makes `value` safe as a single path/collection-name segment.
///
/// A value that already matches `[A-Za-z0-9._-]+` and is not `.`/`..` is returned
/// unchanged. Otherwise the value is filtered to those characters (an empty result
/// becomes `_`) and suffixed with `_` plus the 8-hex-digit FNV-1a hash of the
/// original bytes, so two unsafe values that filter to the same text stay distinct.
/// The result is never `.` or `..`.
#[must_use]
pub fn sanitize_segment(value: &str) -> String {
    if is_safe_segment(value) {
        return value.to_owned();
    }
    let filtered: String = value.chars().filter(|c| is_segment_char(*c)).collect();
    let base = if filtered.is_empty() {
        "_".to_owned()
    } else {
        filtered
    };
    let mut sanitized = format!("{base}_{:08x}", fnv1a_32(value.as_bytes()));
    if sanitized == "." || sanitized == ".." {
        sanitized.insert(0, '_');
    }
    sanitized
}

/// Whether `value` is a non-empty safe segment that needs no rewriting.
fn is_safe_segment(value: &str) -> bool {
    !value.is_empty() && value != "." && value != ".." && value.chars().all(is_segment_char)
}

/// Whether `c` belongs to the accepted segment alphabet `[A-Za-z0-9._-]`.
fn is_segment_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')
}

/// Computes the 32-bit FNV-1a hash of `bytes`.
fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut hash = FNV1A_32_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV1A_32_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    #[test]
    fn record_keys_follow_the_python_layout() {
        assert_eq!(
            record_key(&scope(), "abc123"),
            "UMD/alice/ws/abc123".to_owned()
        );
        assert_eq!(record_prefix(&scope()), "UMD/alice/ws/");
        assert_eq!(record_prefix_for("bob"), "UMD/bob/");
        assert_eq!(record_prefix_for_scope("ws"), "UMD/");
        assert_eq!(record_prefix_for_scope(""), "UMD/");
    }

    #[test]
    fn variable_keys_separate_user_and_session_namespaces() {
        assert_eq!(variable_key(&scope(), "theme"), "user_var/alice/ws/theme");
        assert_eq!(variable_prefix(&scope()), "user_var/alice/ws/");
        assert_eq!(
            session_variable_key(&scope(), "s1", "theme"),
            "session_var/alice/ws/s1/theme"
        );
        assert_eq!(
            session_variable_prefix(&scope(), "s1"),
            "session_var/alice/ws/s1/"
        );
    }

    #[test]
    fn scope_config_key_is_scope_wide() {
        assert_eq!(scope_config_key("ws"), "memory_scope_config/ws");
    }

    #[test]
    fn legacy_keys_mirror_the_current_shapes() {
        assert_eq!(
            legacy_record_key(&scope(), "abc123"),
            "user_mem/alice/ws/abc123"
        );
        assert_eq!(legacy_record_prefix(&scope()), "user_mem/alice/ws/");
        assert_eq!(legacy_record_prefix_for("bob"), "user_mem/bob/");
        assert_eq!(legacy_record_prefix_for_scope("ws"), "user_mem/");
        assert_eq!(
            legacy_variable_key(&scope(), "theme"),
            "user_variable/alice/ws/theme"
        );
        assert_eq!(legacy_variable_prefix(&scope()), "user_variable/alice/ws/");
        assert_eq!(
            legacy_session_variable_key(&scope(), "s1", "theme"),
            "session_variable/alice/ws/s1/theme"
        );
        assert_eq!(
            legacy_session_variable_prefix(&scope(), "s1"),
            "session_variable/alice/ws/s1/"
        );
        assert_eq!(legacy_scope_config_key("ws"), "scope_config/ws".to_owned());
    }

    #[test]
    fn legacy_alternatives_rewrite_only_the_first_segment() {
        assert_eq!(
            legacy_alternatives("UMD/alice/ws/abc"),
            ["user_mem/alice/ws/abc"]
        );
        assert_eq!(legacy_alternatives("UMD/alice/ws/"), ["user_mem/alice/ws/"]);
        assert_eq!(
            legacy_alternatives("UMD/bob/"),
            ["user_mem/bob/".to_owned()]
        );
        // A current head without a rest still maps, and an empty rest is preserved.
        assert_eq!(
            legacy_alternatives("user_var/alice/ws"),
            ["user_variable/alice/ws"]
        );
        assert_eq!(
            legacy_alternatives("session_var/alice/ws/s1/"),
            ["session_variable/alice/ws/s1/".to_owned()]
        );
        assert_eq!(
            legacy_alternatives("memory_scope_config/ws"),
            ["scope_config/ws"]
        );
        for unknown in ["other/x", "UMD", ""] {
            assert_eq!(legacy_alternatives(unknown), Vec::<String>::new());
        }
    }

    #[test]
    fn current_equivalent_inverts_legacy_alternatives() {
        for key in [
            "UMD/alice/ws/abc",
            "UMD/alice/ws/",
            "user_var/alice/ws/theme",
            "session_var/alice/ws/s1/theme",
            "memory_scope_config/ws",
        ] {
            let legacy = legacy_alternatives(key);
            assert_eq!(legacy.len(), 1, "{key}");
            assert_eq!(current_equivalent(&legacy[0]), Some(key.to_owned()));
        }
        assert_eq!(current_equivalent("unknown/x"), None);
        assert_eq!(current_equivalent("user_mem"), None);
        assert_eq!(current_equivalent(""), None);
    }

    #[test]
    fn prefix_tables_map_both_directions() {
        assert_eq!(legacy_prefix("UMD"), Some("user_mem"));
        assert_eq!(legacy_prefix("user_var"), Some("user_variable"));
        assert_eq!(current_prefix("scope_config"), Some("memory_scope_config"));
        assert_eq!(current_prefix("session_variable"), Some("session_var"));
        assert_eq!(legacy_prefix("user_mem"), None);
        assert_eq!(current_prefix("UMD"), None);
        assert_eq!(legacy_prefix(""), None);
        for (current, legacy) in LEGACY_PREFIX_PAIRS {
            assert_eq!(legacy_prefix(current), Some(*legacy));
            assert_eq!(current_prefix(legacy), Some(*current));
        }
    }

    #[test]
    fn vector_collection_keeps_safe_segments_readable() {
        assert_eq!(
            vector_collection(&scope(), MemoryType::SemanticMemory),
            "uid_alice_gid_ws_mtype_semantic_memory"
        );
        assert_eq!(
            vector_collection(&scope(), MemoryType::Coding),
            "uid_alice_gid_ws_mtype_coding"
        );
    }

    #[test]
    fn vector_collection_sanitizes_unsafe_segments() {
        let collection = vector_collection(
            &MemoryScope::new("alice/bob", "ws name"),
            MemoryType::UserProfile,
        );
        let parts: Vec<&str> = collection.split('_').collect();
        assert!(collection.starts_with("uid_alicebob_"));
        assert!(collection.contains("gid_wsname_"));
        assert!(collection.ends_with("mtype_user_profile"));
        assert!(parts.iter().all(|part| !part.is_empty()));
        assert!(!collection.contains('/'));
        assert!(!collection.contains(' '));
    }

    #[test]
    fn safe_segments_are_unchanged() {
        let long = "x".repeat(32);
        for value in ["alice", "a.b-c_1", "0", "__default__", long.as_str()] {
            assert_eq!(sanitize_segment(value), value);
        }
    }

    #[test]
    fn unsafe_segments_get_a_deterministic_hash_suffix() {
        let once = sanitize_segment("alice/bob");
        let twice = sanitize_segment("alice/bob");
        assert_eq!(once, twice);
        let (base, hash) = once.rsplit_once('_').expect("suffix separator");
        assert_eq!(base, "alicebob");
        assert_eq!(hash.len(), 8);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(once.chars().all(is_segment_char));
        assert_ne!(sanitize_segment("alice bob"), once);
    }

    #[test]
    fn dot_segments_never_escape() {
        for value in [".", "..", "./", "../", "", "???", " ", "a\0b"] {
            let sanitized = sanitize_segment(value);
            assert_ne!(sanitized, ".");
            assert_ne!(sanitized, "..");
            assert_ne!(sanitized, "");
            assert!(sanitized.chars().all(is_segment_char));
        }
        assert_eq!(sanitize_segment("").len(), 1 + 1 + 8);
        assert!(sanitize_segment(".").starts_with('.'));
        assert_ne!(sanitize_segment(".."), "");
    }

    #[test]
    fn hash_suffix_is_stable_and_collision_resistant() {
        assert_eq!(sanitize_segment("a/b"), "ab_3a8e75c1");
        assert_eq!(sanitize_segment("a b"), "ab_10a3f9f2");
        assert_ne!(sanitize_segment("a/b"), sanitize_segment("a b"));
        assert_ne!(sanitize_segment("a/b"), sanitize_segment("a:b"));
    }
}
