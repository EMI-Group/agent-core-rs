//! On-disk layout of a team and the sanitization of untrusted path segments.
//!
//! Python parity source: `openjiuwen/agent_teams/paths.py` and
//! `openjiuwen/agent_teams/workspace_layout.py`.
//!
//! Layout of one team under the agent-teams home (`<home>/.agent_teams`):
//!
//! ```text
//! <base>/<team>/
//!   sessions/
//!     <session>/
//!       workspaces/<member>_workspace/   member workspace (artifacts, memory)
//!       members/<member>/                per-member runtime state
//!       worktrees/<worktree-name>/       session-scoped git worktrees
//! ```
//!
//! The Rust port is session-scoped: the Python stable path
//! `{team_home}/workspaces/{member}_workspace` is replaced by
//! `{team_home}/sessions/{session_id}/workspaces/{member}_workspace`, so two concurrent sessions of
//! the same team never share mutable member state.
//!
//! Sanitization is fail-closed: Python joins team and member names into the path raw, this port
//! rejects anything that could escape the base directory instead of silently rewriting it.

use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

/// Maximum number of characters kept by [`sanitize_segment`] (Python `_safe_segment`'s `[:96]`).
pub const MAX_SEGMENT_CHARS: usize = 96;

/// Directory holding all per-session state of a team (relative to the team directory).
const SESSIONS_DIR: &str = "sessions";

/// Directory holding the member workspaces of a session.
const WORKSPACES_DIR: &str = "workspaces";

/// Directory holding the per-member runtime state of a session.
const MEMBERS_DIR: &str = "members";

/// Directory holding the session-owned worktrees.
const WORKTREES_DIR: &str = "worktrees";

/// Suffix appended to a sanitized member name to form its workspace directory.
const WORKSPACE_SUFFIX: &str = "_workspace";

/// Returns `true` for the characters [`sanitize_segment`] leaves untouched.
///
/// Python's `_safe_segment` keeps `[A-Za-z0-9_.-]`; everything else becomes `_`.
const fn is_segment_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-'
}

/// Returns `true` for the separator characters stripped from both ends of a segment.
const fn is_edge_char(ch: char) -> bool {
    ch == '.' || ch == '_' || ch == '-'
}

/// Builds the validation error for a rejected path segment.
fn invalid_segment(value: &str, reason: &str) -> Error {
    Error::validation(format!("invalid path segment {value:?}: {reason}"))
}

/// Sanitizes an untrusted string into exactly one safe path segment.
///
/// The rules (Python `_safe_segment`, hardened):
///
/// 1. Surrounding whitespace is trimmed.
/// 2. Fail closed with [`Error::Validation`] when the value is empty or whitespace-only, contains
///    `/`, `\` or NUL, is an absolute path, or is `.` or `..`.
/// 3. Every character outside `[A-Za-z0-9._-]` is replaced with `_`, per Unicode scalar value — a
///    multi-byte character therefore yields exactly one `_`.
/// 4. Leading and trailing `.`, `_` and `-` are stripped, the result is truncated to
///    [`MAX_SEGMENT_CHARS`] characters, and the strip is repeated.
/// 5. A result that is empty, `.` or `..` is rejected with [`Error::Validation`].
///
/// Because `/`, `\` and NUL are rejected before replacement and the edges are stripped afterwards,
/// the returned value is always a single relative path component: joining it to a base directory
/// can never produce a path outside that base.
///
/// # Errors
///
/// Returns [`Error::Validation`] with the offending value for every rejected input; see the rules
/// above.
///
/// # Examples
///
/// ```
/// use openjiuwen_agent_teams::workspace::sanitize_segment;
///
/// assert_eq!(sanitize_segment("dev-1").unwrap(), "dev-1");
/// assert!(sanitize_segment("../evil").is_err());
/// ```
pub fn sanitize_segment(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(invalid_segment(value, "empty or whitespace-only"));
    }
    if trimmed == "." || trimmed == ".." {
        return Err(invalid_segment(value, "is a relative path component"));
    }
    if trimmed
        .chars()
        .any(|ch| ch == '/' || ch == '\\' || ch == '\0')
    {
        return Err(invalid_segment(value, "contains a path separator or NUL"));
    }
    if Path::new(trimmed).is_absolute() {
        return Err(invalid_segment(value, "is an absolute path"));
    }
    let replaced: String = trimmed
        .chars()
        .map(|ch| if is_segment_char(ch) { ch } else { '_' })
        .collect();
    let stripped = replaced.trim_matches(is_edge_char);
    let truncated: String = stripped.chars().take(MAX_SEGMENT_CHARS).collect();
    let sanitized = truncated.trim_matches(is_edge_char);
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        return Err(invalid_segment(
            value,
            "sanitizes to an empty or relative path segment",
        ));
    }
    Ok(sanitized.to_owned())
}

/// Path layout of a team tree rooted at one base directory.
///
/// Every helper is fallible because every segment is sanitized by [`sanitize_segment`]; a hostile
/// team, session or member name is rejected instead of being written into a path.
///
/// # Examples
///
/// ```
/// use openjiuwen_agent_teams::workspace::MemberWorkspaceLayout;
///
/// let layout = MemberWorkspaceLayout::new("/srv/openjiuwen/.agent_teams");
/// let root = layout.member_workspace("my-team", "session-1", "dev-1").unwrap();
/// assert!(root.ends_with("my-team/sessions/session-1/workspaces/dev-1_workspace"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberWorkspaceLayout {
    /// Root directory of the agent-teams tree (Python `get_agent_teams_home()`).
    pub base: PathBuf,
}

impl MemberWorkspaceLayout {
    /// Creates a layout rooted at `base`.
    #[must_use]
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// Returns the root directory of the layout.
    #[must_use]
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// Returns `<base>/<team>/` (Python `team_home`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the team name is not a safe path segment.
    pub fn team_dir(&self, team: &str) -> Result<PathBuf> {
        Ok(self.base.join(sanitize_segment(team)?))
    }

    /// Returns `<base>/<team>/sessions/<session>/` (Python `team_session_dir`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the team name or session id is not a safe path segment.
    pub fn session_dir(&self, team: &str, session_id: &str) -> Result<PathBuf> {
        Ok(self
            .team_dir(team)?
            .join(SESSIONS_DIR)
            .join(sanitize_segment(session_id)?))
    }

    /// Returns `<session>/workspaces/<member>_workspace` (the member workspace).
    ///
    /// This is where a member keeps its artifacts and memory. The directory is only created when
    /// the team enables the shared workspace; otherwise callers still get the address so they can
    /// report it, matching Python's "no workspace manager" state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the team name, session id or member name is not a safe
    /// path segment.
    pub fn member_workspace(&self, team: &str, session_id: &str, member: &str) -> Result<PathBuf> {
        let segment = sanitize_segment(member)?;
        Ok(self
            .session_dir(team, session_id)?
            .join(WORKSPACES_DIR)
            .join(format!("{segment}{WORKSPACE_SUFFIX}")))
    }

    /// Returns `<session>/members/<member>/` (per-member runtime state).
    ///
    /// The directory sits outside the workspace tree on purpose: resetting or re-creating a member
    /// workspace must not touch the member's runtime state (skill visibility, cursors, caches).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the team name, session id or member name is not a safe
    /// path segment.
    pub fn member_dir(&self, team: &str, session_id: &str, member: &str) -> Result<PathBuf> {
        Ok(self
            .session_dir(team, session_id)?
            .join(MEMBERS_DIR)
            .join(sanitize_segment(member)?))
    }

    /// Returns `<session>/worktrees/` (Python `team_session_worktrees_dir`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the team name or session id is not a safe path segment.
    pub fn worktrees_dir(&self, team: &str, session_id: &str) -> Result<PathBuf> {
        Ok(self.session_dir(team, session_id)?.join(WORKTREES_DIR))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn layout() -> MemberWorkspaceLayout {
        MemberWorkspaceLayout::new("/srv/openjiuwen/.agent_teams")
    }

    #[test]
    fn layout_paths_match_the_documented_tree() {
        let base = PathBuf::from("/srv/openjiuwen/.agent_teams");
        let layout = layout();
        assert_eq!(layout.base(), base.as_path());
        assert_eq!(layout.team_dir("my-team").unwrap(), base.join("my-team"));
        assert_eq!(
            layout.session_dir("my-team", "session-1").unwrap(),
            base.join("my-team").join("sessions").join("session-1")
        );
        assert_eq!(
            layout
                .member_workspace("my-team", "session-1", "dev-1")
                .unwrap(),
            base.join("my-team")
                .join("sessions")
                .join("session-1")
                .join("workspaces")
                .join("dev-1_workspace")
        );
        assert_eq!(
            layout.member_dir("my-team", "session-1", "dev-1").unwrap(),
            base.join("my-team")
                .join("sessions")
                .join("session-1")
                .join("members")
                .join("dev-1")
        );
        assert_eq!(
            layout.worktrees_dir("my-team", "session-1").unwrap(),
            base.join("my-team")
                .join("sessions")
                .join("session-1")
                .join("worktrees")
        );
    }

    #[test]
    fn sanitize_keeps_normal_names_unchanged() {
        for name in ["dev-1", "team_leader", "a", "v1.2.3", "A9"] {
            assert_eq!(sanitize_segment(name).unwrap(), name);
        }
        assert_eq!(sanitize_segment("  dev-1  ").unwrap(), "dev-1");
    }

    #[test]
    fn sanitize_rejects_traversal_and_separators() {
        for bad in [
            "",
            "   ",
            ".",
            "..",
            "../evil",
            "..\\evil",
            "/etc/passwd",
            "Team A/2024",
            "a\\b",
            "a\0b",
            "...",
        ] {
            let err = sanitize_segment(bad).expect_err(bad);
            assert!(
                matches!(err, Error::Validation(_)),
                "{bad:?} produced {err:?}"
            );
        }
    }

    #[test]
    fn sanitize_replaces_every_non_segment_scalar_with_one_underscore() {
        // Each of `é`, `日` and `本` is one scalar value, so each becomes a single `_`; the
        // trailing run of `_`/`-` is then stripped (Python `strip("._-")` does the same).
        assert_eq!(sanitize_segment("café-日本").unwrap(), "caf");
        assert_eq!(sanitize_segment("a b").unwrap(), "a_b");
        // Unlike Python's `re.sub` with `+`, runs are not collapsed: one `_` per scalar.
        assert_eq!(sanitize_segment("a+++b").unwrap(), "a___b");
        assert_eq!(sanitize_segment("a:b").unwrap(), "a_b");
    }

    #[test]
    fn sanitize_truncates_to_the_character_limit() {
        let long = "a".repeat(200);
        assert_eq!(sanitize_segment(&long).unwrap().chars().count(), 96);
        // The truncation lands on a dash, which the repeated edge strip then removes.
        let mixed = format!("{}-b", "x".repeat(95));
        assert_eq!(sanitize_segment(&mixed).unwrap(), "x".repeat(95));
    }

    #[test]
    fn hostile_segments_never_escape_the_layout_base() {
        let layout = layout();
        let base = layout.base().to_path_buf();
        for hostile in [
            "..",
            ".",
            "../..",
            "/etc",
            "a/../../b",
            "a\\..\\b",
            "%2e%2e%2f",
            "...a...",
        ] {
            // Either the value is rejected outright, or the resulting path stays under the base.
            if let Ok(path) = layout.team_dir(hostile) {
                assert!(path.starts_with(&base), "{hostile:?} escaped to {path:?}");
            }
            if let Ok(path) = layout.member_workspace(hostile, hostile, hostile) {
                assert!(path.starts_with(&base), "{hostile:?} escaped to {path:?}");
            }
        }
        assert!(layout.team_dir("A/../..").is_err());
    }
}
