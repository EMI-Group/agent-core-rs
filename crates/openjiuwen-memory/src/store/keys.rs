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
