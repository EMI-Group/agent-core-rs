//! Provider session identity and resume resolution for the vendor CLI harnesses.
//!
//! A vendor CLI keeps its own conversation state, keyed by a session id the harness chooses (or by
//! the id of a checkpoint it is asked to resume). Three pieces are shared by every adapter:
//!
//! - [`derive_session_id`] derives a stable, UUID-shaped id from the host session and the agent, so
//!   restarting the same agent in the same host session attaches to the same CLI conversation;
//! - [`resolve_resume_id`] applies the host's [`ResumePolicy`] to the optional checkpoint;
//! - [`checkpoint_str`] reads one string field out of an opaque checkpoint payload.

use openjiuwen_common::utils::hash::generate_key;
use openjiuwen_harness_protocol::{
    HarnessCheckpoint, HarnessError, HarnessResult, JsonValue, ResumePolicy,
};

/// Length of the hex digest prefix formatted into the derived id (32 hex characters).
const DERIVED_HEX_CHARS: usize = 32;

/// Derives the provider session id of `agent_name` inside `host_session_id`.
///
/// The id is deterministic and stable across processes: the seed is the compact JSON array
/// `[host_session_id, agent_name]`, hashed with SHA-256 via
/// [`generate_key`], and its first 32 hex characters
/// are formatted as `8-4-4-4-12`.
///
/// Python derives a UUID5 over `NAMESPACE_URL` instead; Rust uses this stable UUID-shaped id, which
/// produces the same id on every run and platform without depending on UUID namespace semantics.
#[must_use]
pub fn derive_session_id(host_session_id: &str, agent_name: &str) -> String {
    let seed = match serde_json::to_string(&[host_session_id, agent_name]) {
        Ok(seed) => seed,
        Err(error) => {
            // A two-element string array always encodes; keep the derivation total anyway.
            tracing::warn!(%error, "failed to encode the session-id seed; hashing the identity pair");
            format!("{host_session_id:?}{agent_name:?}")
        }
    };
    // `generate_key` hashes the sorted concatenation of its three parts; two empty parts leave the
    // seed itself, i.e. exactly `sha256(seed)`.
    let digest = generate_key(&seed, "", "");
    let hex: Vec<char> = digest.chars().take(DERIVED_HEX_CHARS).collect();
    let segment =
        |start: usize, len: usize| -> String { hex.iter().skip(start).take(len).collect() };
    format!(
        "{}-{}-{}-{}-{}",
        segment(0, 8),
        segment(8, 4),
        segment(12, 4),
        segment(16, 4),
        segment(20, 12)
    )
}

/// Resolves the provider session id to resume, according to `policy`.
///
/// - [`ResumePolicy::New`] starts a fresh provider session, so no id is returned;
/// - [`ResumePolicy::ResumeIfAvailable`] resumes the checkpoint when one is supplied and otherwise
///   starts fresh;
/// - [`ResumePolicy::RequireResume`] resumes the checkpoint, else the already-`derived` id, and
///   fails with `missing_message` when neither is available.
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`] carrying `missing_message` when
/// [`ResumePolicy::RequireResume`] has nothing to resume.
pub fn resolve_resume_id(
    policy: ResumePolicy,
    checkpoint: Option<&HarnessCheckpoint>,
    derived: Option<String>,
    missing_message: &'static str,
) -> HarnessResult<Option<String>> {
    match policy {
        ResumePolicy::New => Ok(None),
        ResumePolicy::ResumeIfAvailable => Ok(checkpoint.map(checkpoint_id)),
        ResumePolicy::RequireResume => checkpoint
            .map(checkpoint_id)
            .or(derived)
            .map(Some)
            .ok_or_else(|| HarnessError::protocol(missing_message)),
    }
}

/// Reads one string field of an opaque [`HarnessCheckpoint`] payload.
///
/// Returns [`None`] when the key is absent or not a string; the checkpoint payload belongs to the
/// provider, so an unexpected shape is not an error here.
#[must_use]
pub fn checkpoint_str(checkpoint: &HarnessCheckpoint, key: &str) -> Option<String> {
    checkpoint
        .data
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::to_owned)
}

/// Returns the checkpoint's own identifier as the resume id.
fn checkpoint_id(checkpoint: &HarnessCheckpoint) -> String {
    checkpoint.checkpoint_id.clone()
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{
        HarnessCheckpoint, HarnessErrorKind, JsonObject, ResumePolicy,
    };
    use serde_json::json;

    use super::{checkpoint_str, derive_session_id, resolve_resume_id};

    /// Checkpoint with the given id and an empty payload.
    fn checkpoint(id: &str) -> HarnessCheckpoint {
        HarnessCheckpoint {
            provider: "claude-code".to_owned(),
            schema_version: "1".to_owned(),
            agent_id: "agent-a".to_owned(),
            host_session_id: "host-session".to_owned(),
            checkpoint_id: id.to_owned(),
            sequence: 1,
            data: JsonObject::new(),
            provider_session_id: None,
            revision: None,
        }
    }

    #[test]
    fn derived_session_id_is_uuid_shaped() {
        let id = derive_session_id("host-session", "agent-a");
        let segments: Vec<&str> = id.split('-').collect();
        assert_eq!(segments.len(), 5);
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.len())
                .collect::<Vec<_>>(),
            vec![8, 4, 4, 4, 12]
        );
        assert!(segments.iter().all(|segment| segment
            .chars()
            .all(|character| character.is_ascii_hexdigit())));
    }

    #[test]
    fn derived_session_id_is_stable_and_input_sensitive() {
        let first = derive_session_id("host-session", "agent-a");
        let second = derive_session_id("host-session", "agent-a");
        assert_eq!(first, second);
        assert_ne!(first, derive_session_id("host-session", "agent-b"));
        assert_ne!(first, derive_session_id("other-session", "agent-a"));
    }

    #[test]
    fn derived_session_id_matches_sha256_of_the_json_seed() {
        // The seed is the compact JSON array of the two identity parts.
        let seed = r#"["host-session","agent-a"]"#;
        let digest = openjiuwen_common::utils::hash::generate_key(seed, "", "");
        assert_eq!(
            derive_session_id("host-session", "agent-a"),
            format!(
                "{}-{}-{}-{}-{}",
                &digest[0..8],
                &digest[8..12],
                &digest[12..16],
                &digest[16..20],
                &digest[20..32]
            )
        );
    }

    #[test]
    fn new_policy_never_resumes() {
        let checkpoint = checkpoint("cp-1");
        assert_eq!(
            resolve_resume_id(
                ResumePolicy::New,
                Some(&checkpoint),
                Some("derived".to_owned()),
                "unused"
            )
            .expect("the new policy always succeeds"),
            None
        );
    }

    #[test]
    fn resume_if_available_uses_the_checkpoint_only() {
        let checkpoint = checkpoint("cp-1");
        assert_eq!(
            resolve_resume_id(
                ResumePolicy::ResumeIfAvailable,
                Some(&checkpoint),
                Some("derived".to_owned()),
                "unused"
            )
            .expect("a supplied checkpoint resumes"),
            Some("cp-1".to_owned())
        );
        assert_eq!(
            resolve_resume_id(
                ResumePolicy::ResumeIfAvailable,
                None,
                Some("derived".to_owned()),
                "unused"
            )
            .expect("a missing checkpoint starts fresh"),
            None
        );
    }

    #[test]
    fn require_resume_prefers_the_checkpoint_then_the_derived_id() {
        let checkpoint = checkpoint("cp-1");
        assert_eq!(
            resolve_resume_id(
                ResumePolicy::RequireResume,
                Some(&checkpoint),
                Some("derived".to_owned()),
                "unused"
            )
            .expect("a supplied checkpoint resumes"),
            Some("cp-1".to_owned())
        );
        assert_eq!(
            resolve_resume_id(
                ResumePolicy::RequireResume,
                None,
                Some("derived".to_owned()),
                "unused"
            )
            .expect("the derived id is the fallback"),
            Some("derived".to_owned())
        );
    }

    #[test]
    fn require_resume_without_a_source_reports_the_message() {
        let error = resolve_resume_id(
            ResumePolicy::RequireResume,
            None,
            None,
            "the Claude Code harness requires a checkpoint or a session id to resume",
        )
        .expect_err("nothing to resume");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert_eq!(
            error.message(),
            "the Claude Code harness requires a checkpoint or a session id to resume"
        );
    }

    #[test]
    fn checkpoint_str_reads_string_fields() {
        let mut payload = JsonObject::new();
        payload.insert("session_id".to_owned(), json!("provider-session"));
        payload.insert("count".to_owned(), json!(3));
        let checkpoint = HarnessCheckpoint {
            data: payload,
            ..checkpoint("cp-1")
        };
        assert_eq!(
            checkpoint_str(&checkpoint, "session_id").as_deref(),
            Some("provider-session")
        );
        assert_eq!(checkpoint_str(&checkpoint, "count"), None);
        assert_eq!(checkpoint_str(&checkpoint, "absent"), None);
    }
}
