//! Deterministic worktree naming and the git worktree backend seam.
//!
//! Python parity sources: `openjiuwen/agent_teams/worktree/naming.py` and
//! `openjiuwen/harness/tools/worktree/slug.py`.
//!
//! A teammate worktree is named
//! `agent-{slug(team)<=14}-{slug(member)<=16}-{digest[..10]}`; its branch is `worktree-<name>` and
//! its directory is `<session>/worktrees/<name>`. The digest is computed over the *raw*
//! (untruncated) inputs, so two teams whose names differ only past the 14-character truncation
//! still get distinct worktrees.
//!
//! The git backend itself is deferred: [`UnsupportedWorktreeBackend`] and
//! [`crate::workspace::LocalWorkspaceProvider`] without an injected backend return
//! [`Error::Unsupported`] naming the Python capability (`agent_teams.worktree.lifecycle`).

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

/// Characters of the team name kept in a worktree name (Python `_TEAM_PART_LENGTH`).
pub const TEAM_PART_CHARS: usize = 14;

/// Characters of the member name kept in a worktree name (Python `_MEMBER_PART_LENGTH`).
pub const MEMBER_PART_CHARS: usize = 16;

/// Characters of the digest kept in a worktree name.
pub const DIGEST_CHARS: usize = 10;

/// Digest used when the seed contains no alphanumeric characters at all.
const DIGEST_FALLBACK: &str = "0000000000";

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Prefix of every teammate worktree name and of its git branch.
const NAME_PREFIX: &str = "agent-";

/// Prefix of a worktree git branch.
const BRANCH_PREFIX: &str = "worktree-";

/// One worktree of a teammate: its slug, its directory and its git branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WorktreeInfo {
    /// Worktree slug, as produced by [`build_teammate_worktree_name`].
    pub name: String,
    /// Absolute directory of the worktree.
    pub path: PathBuf,
    /// Git branch checked out in the worktree, as produced by [`worktree_branch`].
    pub branch: String,
}

/// Reduces one name part to a slug of at most `max_chars` characters.
///
/// The value is lowercased, every character outside `[a-z0-9]` becomes `-`, runs of `-` collapse,
/// and leading/trailing `-` are stripped (before and after truncation). An empty result falls back
/// to `fallback`.
fn slug_part(value: &str, fallback: &str, max_chars: usize) -> String {
    let lowered = value.trim().to_lowercase();
    let mut slug = String::with_capacity(lowered.len());
    let mut previous_dash = false;
    for ch in lowered.chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() {
            slug.push(ch);
            previous_dash = false;
        } else if !previous_dash {
            slug.push('-');
            previous_dash = true;
        }
    }
    let trimmed = slug.trim_matches('-');
    let truncated: String = trimmed.chars().take(max_chars).collect();
    let part = truncated.trim_matches('-');
    if part.is_empty() {
        fallback.to_owned()
    } else {
        part.to_owned()
    }
}

/// Reduces a digest to its first [`DIGEST_CHARS`] lowercase alphanumerics.
///
/// Non-alphanumeric characters are dropped (a hex digest keeps all of its characters), the value is
/// lowercased, and a value with no alphanumerics falls back to `0000000000`.
fn digest_part(sha: &str) -> String {
    let filtered: String = sha
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|ch| ch.to_ascii_lowercase())
        .take(DIGEST_CHARS)
        .collect();
    if filtered.is_empty() {
        DIGEST_FALLBACK.to_owned()
    } else {
        filtered
    }
}

/// Builds the deterministic worktree name of a teammate.
///
/// Shape: `agent-{slug(team, 14)}-{slug(member, 16)}-{sha[..10]}`. `sha` is the digest the caller
/// computed over the raw inputs (`team:member:session` plus the project identity, see
/// [`worktree_digest`]), so names that truncate to the same slug still differ in the suffix.
///
/// # Examples
///
/// ```
/// use openjiuwen_agent_teams::workspace::build_teammate_worktree_name;
///
/// assert_eq!(
///     build_teammate_worktree_name("my-team", "dev-1", "abcdef1234567890"),
///     "agent-my-team-dev-1-abcdef1234",
/// );
/// ```
#[must_use]
pub fn build_teammate_worktree_name(team: &str, member: &str, sha: &str) -> String {
    let team = slug_part(team, "team", TEAM_PART_CHARS);
    let member = slug_part(member, "member", MEMBER_PART_CHARS);
    let digest = digest_part(sha);
    format!("{NAME_PREFIX}{team}-{member}-{digest}")
}

/// Returns the git branch of a worktree: `worktree-<name>`.
///
/// `/` is flattened to `+` (Python `_flatten_slug`) so a nested slug can never create a
/// directory/file conflict inside the git ref namespace.
///
/// # Examples
///
/// ```
/// use openjiuwen_agent_teams::workspace::worktree_branch;
///
/// assert_eq!(worktree_branch("agent-my-team-dev-1-abcdef1234"), "worktree-agent-my-team-dev-1-abcdef1234");
/// assert_eq!(worktree_branch("user/feature"), "worktree-user+feature");
/// ```
#[must_use]
pub fn worktree_branch(name: &str) -> String {
    format!("{BRANCH_PREFIX}{}", name.replace('/', "+"))
}

/// Returns a deterministic 10-hex-character digest of a worktree seed.
///
/// Python hashes `sha256(seed)[:10]`; this port uses 64-bit FNV-1a and keeps the first
/// [`DIGEST_CHARS`] hex characters, because `sha2` is not a dependency of this crate. The value is
/// stable across processes and runs (it is not a security primitive and must not be used as one).
///
/// # Examples
///
/// ```
/// use openjiuwen_agent_teams::workspace::worktree_digest;
///
/// let digest = worktree_digest("my-team:dev-1:session-1");
/// assert_eq!(digest.len(), 10);
/// assert_eq!(digest, worktree_digest("my-team:dev-1:session-1"));
/// ```
#[must_use]
pub fn worktree_digest(seed: &str) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in seed.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}").chars().take(DIGEST_CHARS).collect()
}

/// Everything a [`WorktreeBackend`] needs to create one teammate worktree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WorktreeCreateRequest {
    /// Team owning the worktree.
    pub team_name: String,
    /// Session owning the worktree.
    pub session_id: String,
    /// Member the worktree belongs to.
    pub member_name: String,
    /// Worktree slug, as produced by [`build_teammate_worktree_name`].
    pub name: String,
    /// Directory the worktree must be created at.
    pub path: PathBuf,
    /// Git ref the worktree branches off; `None` uses the backend default (usually `HEAD`).
    pub base_ref: Option<String>,
}

/// Creates and removes teammate worktrees.
///
/// The in-process core only ships [`UnsupportedWorktreeBackend`]; a host that owns a git
/// implementation injects its own backend into
/// [`LocalWorkspaceProvider`](crate::workspace::LocalWorkspaceProvider).
#[async_trait]
pub trait WorktreeBackend: Send + Sync {
    /// Returns the backend name used in logs and diagnostics.
    fn name(&self) -> &str;

    /// Creates the worktree described by `request`.
    ///
    /// # Errors
    ///
    /// Returns whatever the backend reports when the worktree cannot be created.
    async fn create(&self, request: &WorktreeCreateRequest) -> Result<WorktreeInfo>;

    /// Removes a worktree previously returned by [`Self::create`].
    ///
    /// # Errors
    ///
    /// Returns whatever the backend reports when the worktree cannot be removed.
    async fn remove(&self, worktree: &WorktreeInfo) -> Result<()>;
}

/// The [`WorktreeBackend`] of a runtime without git support.
///
/// Every operation fails closed with [`Error::Unsupported`] naming the Python capability, so a
/// caller that requests worktree isolation learns it is unavailable instead of silently running in
/// a shared directory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnsupportedWorktreeBackend;

#[async_trait]
impl WorktreeBackend for UnsupportedWorktreeBackend {
    fn name(&self) -> &'static str {
        "unsupported"
    }

    async fn create(&self, _request: &WorktreeCreateRequest) -> Result<WorktreeInfo> {
        Err(unsupported_worktree())
    }

    async fn remove(&self, _worktree: &WorktreeInfo) -> Result<()> {
        Err(unsupported_worktree())
    }
}

/// Builds the error returned by every unsupported worktree operation.
fn unsupported_worktree() -> Error {
    Error::unsupported(
        "git worktree backend is not implemented \
         (Python capability: agent_teams.worktree.lifecycle)",
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn name_matches_the_python_shape() {
        assert_eq!(
            build_teammate_worktree_name("my-team", "dev-1", "abcdef1234567890"),
            "agent-my-team-dev-1-abcdef1234"
        );
    }

    #[test]
    fn name_slugs_case_spaces_and_unicode() {
        assert_eq!(
            build_teammate_worktree_name("My Team", "Dev 1", "abc"),
            "agent-my-team-dev-1-abc"
        );
        // A fully non-ASCII name collapses to a single dash, which is stripped, so the fallback is
        // used.
        assert_eq!(
            build_teammate_worktree_name("团队甲", "成员乙", "deadbeef00"),
            "agent-team-member-deadbeef00"
        );
        assert_eq!(
            build_teammate_worktree_name("Team__A", "--dev--1--", "a-b-c"),
            "agent-team-a-dev-1-abc"
        );
    }

    #[test]
    fn name_truncates_but_keeps_long_names_distinguishable() {
        let team = "averyveryverylongteamname";
        let member = "averyveryverylongmembername";
        let first_sha = worktree_digest(&format!("{team}-1:{member}-1:session"));
        let second_sha = worktree_digest(&format!("{team}-2:{member}-2:session"));
        let first =
            build_teammate_worktree_name(&format!("{team}-1"), &format!("{member}-1"), &first_sha);
        let second =
            build_teammate_worktree_name(&format!("{team}-2"), &format!("{member}-2"), &second_sha);
        assert_eq!(
            first,
            format!(
                "agent-{}-{}-{}",
                &team[..14],
                &member[..16],
                &first_sha[..10]
            )
        );
        assert_eq!(
            second,
            format!(
                "agent-{}-{}-{}",
                &team[..14],
                &member[..16],
                &second_sha[..10]
            )
        );
        assert_ne!(first, second);
    }

    #[test]
    fn digest_is_deterministic_and_ten_hex_chars() {
        let digest = worktree_digest("my-team:dev-1:session-1");
        assert_eq!(digest.len(), DIGEST_CHARS);
        assert!(digest.chars().all(|ch| ch.is_ascii_hexdigit()));
        assert_eq!(digest, worktree_digest("my-team:dev-1:session-1"));
        assert_ne!(digest, worktree_digest("my-team:dev-1:session-2"));
    }

    #[test]
    fn branch_flattens_slashes() {
        assert_eq!(
            worktree_branch("agent-my-team-dev-1-abcdef1234"),
            "worktree-agent-my-team-dev-1-abcdef1234"
        );
        assert_eq!(worktree_branch("user/feature"), "worktree-user+feature");
    }

    #[tokio::test]
    async fn unsupported_backend_fails_closed() {
        let backend = UnsupportedWorktreeBackend;
        assert_eq!(backend.name(), "unsupported");
        let request = WorktreeCreateRequest {
            team_name: "my-team".to_owned(),
            session_id: "session-1".to_owned(),
            member_name: "dev-1".to_owned(),
            name: "agent-my-team-dev-1-abcdef1234".to_owned(),
            path: PathBuf::from("/tmp/worktrees/agent-my-team-dev-1-abcdef1234"),
            base_ref: None,
        };
        let err = backend.create(&request).await.expect_err("create fails");
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("agent_teams.worktree.lifecycle"));
        let info = WorktreeInfo {
            name: request.name.clone(),
            path: request.path.clone(),
            branch: worktree_branch(&request.name),
        };
        let err = backend.remove(&info).await.expect_err("remove fails");
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    }
}
