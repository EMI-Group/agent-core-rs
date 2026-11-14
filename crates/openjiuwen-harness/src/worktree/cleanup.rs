//! Ephemeral-worktree cleanup: age-based reaping with fail-closed safety checks.
//!
//! Port of `openjiuwen/harness/tools/worktree/cleanup.py`. Python walks the
//! managed `.worktrees` directory, keeps only slugs that look ephemeral
//! (`teammate-<8hex>`, `agent-<7hex>`, `agent-<team>-<member>-<8hex>`), skips the
//! current worktree and anything it cannot prove safe, and prunes git metadata
//! once after at least one removal.
//!
//! # Rules, in order
//!
//! 1. Only worktrees git lists under
//!    [`WorktreeManager::worktrees_dir`] are considered — never the main
//!    worktree.
//! 2. The directory modification time must be older than
//!    [`CleanupPolicy::max_age_days`]; an unreadable timestamp is a skip
//!    (`cannot read worktree metadata`).
//! 3. With [`CleanupPolicy::ephemeral_only`] (the default), a slug that matches
//!    no [`EPHEMERAL_PATTERNS`] entry is a skip (`not an ephemeral worktree`).
//! 4. The active worktree is never touched (`current worktree`).
//! 5. A dirty worktree is a skip (`uncommitted changes`).
//! 6. Unpushed state is fail-closed: `None` (probe failed) and `Some(true)` are
//!    skips (`cannot verify unpushed commits` / `unpushed commits`).
//!
//! Everything is driven through [`WorktreeManager`]'s public accessors and the
//! [`super::git`] helpers, so a scripted runner makes the whole pass offline and
//! deterministic; [`CleanupPolicy::now`] freezes the clock.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::SystemTime;

use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Result;

use crate::permission::paths::normalize_lexically;

use super::git;
use super::manager::WorktreeManager;
use super::store::WorktreeIndex;

/// Default maximum age of an ephemeral worktree before cleanup reaps it.
pub const DEFAULT_CLEANUP_AFTER_DAYS: u64 = 30;

/// Slug patterns that identify worktrees created for a transient owner.
///
/// `teammate-<member 8 hex>`, the legacy `agent-<7 hex>`, and
/// `agent-<team>-<member>-<8 hex>` (each label at most 24 characters).
pub const EPHEMERAL_PATTERNS: [&str; 3] = [
    r"^teammate-[0-9a-f]{8}$",
    r"^agent-[0-9a-f]{7}$",
    r"^agent-[a-z0-9._-]{1,24}-[a-z0-9._-]{1,24}-[0-9a-f]{8}$",
];

/// The compiled form of [`EPHEMERAL_PATTERNS`].
static COMPILED_EPHEMERAL_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    EPHEMERAL_PATTERNS
        .iter()
        .filter_map(|pattern| Regex::new(pattern).ok())
        .collect()
});

/// Whether `slug` names an ephemeral worktree.
///
/// The match is anchored (`^…$`): a slug that merely contains an ephemeral
/// prefix, such as `teammate-12345678-wip`, is not ephemeral.
#[must_use]
pub fn is_ephemeral_slug(slug: &str) -> bool {
    COMPILED_EPHEMERAL_PATTERNS
        .iter()
        .any(|pattern| pattern.is_match(slug))
}

/// Policy of one [`WorktreeManager::cleanup`] pass.
///
/// All fields have serde defaults and a [`CleanupPolicy::default`] that matches
/// the Python configuration: thirty days, ephemeral slugs only, a real removal
/// pass, no active worktree, and the wall clock as the age reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CleanupPolicy {
    /// Worktrees at least this old are candidates.
    #[serde(default = "default_max_age_days")]
    pub max_age_days: u64,
    /// Restrict reaping to [`is_ephemeral_slug`] matches.
    #[serde(default = "default_ephemeral_only")]
    pub ephemeral_only: bool,
    /// Only report candidates; perform no git mutation and no registry write.
    #[serde(default)]
    pub dry_run: bool,
    /// The worktree the caller is running in; never reaped.
    #[serde(default)]
    pub current_worktree: Option<PathBuf>,
    /// Age reference; `None` means the wall clock.
    #[serde(default)]
    pub now: Option<DateTime<Utc>>,
}

/// Default of [`CleanupPolicy::max_age_days`].
const fn default_max_age_days() -> u64 {
    DEFAULT_CLEANUP_AFTER_DAYS
}

/// Default of [`CleanupPolicy::ephemeral_only`].
const fn default_ephemeral_only() -> bool {
    true
}

impl Default for CleanupPolicy {
    fn default() -> Self {
        Self {
            max_age_days: DEFAULT_CLEANUP_AFTER_DAYS,
            ephemeral_only: true,
            dry_run: false,
            current_worktree: None,
            now: None,
        }
    }
}

impl CleanupPolicy {
    /// Overrides the age threshold in days.
    #[must_use]
    pub fn with_max_age_days(mut self, max_age_days: u64) -> Self {
        self.max_age_days = max_age_days;
        self
    }

    /// Chooses whether only ephemeral slugs are reaped.
    #[must_use]
    pub fn with_ephemeral_only(mut self, ephemeral_only: bool) -> Self {
        self.ephemeral_only = ephemeral_only;
        self
    }

    /// Turns the pass into a dry run (or back).
    #[must_use]
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Protects `current_worktree` from reaping.
    #[must_use]
    pub fn with_current_worktree(mut self, current_worktree: impl Into<PathBuf>) -> Self {
        self.current_worktree = Some(current_worktree.into());
        self
    }

    /// Freezes the age reference instead of reading the wall clock.
    #[must_use]
    pub fn with_now(mut self, now: DateTime<Utc>) -> Self {
        self.now = Some(now);
        self
    }

    /// The age reference: `now` when set, the wall clock otherwise.
    #[must_use]
    pub fn reference_time(&self) -> DateTime<Utc> {
        self.now.unwrap_or_else(Utc::now)
    }

    /// The age threshold in seconds (`max_age_days * 86_400`).
    #[must_use]
    pub const fn max_age_secs(&self) -> u64 {
        self.max_age_days.saturating_mul(86_400)
    }
}

/// One worktree the pass would reap.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CleanupCandidate {
    /// Slug (directory component) of the worktree.
    pub name: String,
    /// Absolute path of the worktree directory.
    pub path: PathBuf,
    /// Age of the directory's modification time, in whole seconds.
    pub age_secs: u64,
}

/// One worktree the pass kept, and why.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CleanupSkip {
    /// Slug (directory component) of the worktree.
    pub name: String,
    /// Absolute path of the worktree directory.
    pub path: PathBuf,
    /// Stable reason string (`uncommitted changes`, `current worktree`, …).
    pub reason: String,
}

/// Outcome of one cleanup pass.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CleanupReport {
    /// Whether the pass only reported candidates.
    pub dry_run: bool,
    /// Worktrees eligible for removal, oldest git entry first.
    pub candidates: Vec<CleanupCandidate>,
    /// Paths actually removed (always empty on a dry run).
    pub removed: Vec<PathBuf>,
    /// Worktrees kept, with the reason.
    pub skipped: Vec<CleanupSkip>,
}

impl CleanupReport {
    /// Number of worktrees removed.
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.removed.len()
    }
}

/// Runs one cleanup pass for `manager` under `policy`.
///
/// The caller holds the repository lock ([`WorktreeManager::cleanup`] does); this
/// function only inspects and mutates git state. A candidate whose `git worktree
/// remove` fails becomes a skip carrying the git message, so one stubborn
/// worktree never aborts the pass.
///
/// # Errors
///
/// Returns the typed error of a failing `git worktree list`, `git worktree
/// remove`-adjacent listing, `git status`, `git rev-list` or `git worktree
/// prune`, and of a corrupt registry. A worktree whose `git worktree remove`
/// fails is *not* an error: it becomes a skip.
pub(crate) async fn run(
    manager: &WorktreeManager,
    policy: &CleanupPolicy,
) -> Result<CleanupReport> {
    let repo = manager.ensure_repository().await?;
    let infos = git::worktree_list(manager.runner().as_ref(), &repo).await?;
    let dir = normalize_lexically(&manager.worktrees_dir());
    let now = policy.reference_time();
    let mut report = CleanupReport {
        dry_run: policy.dry_run,
        ..CleanupReport::default()
    };

    for info in infos {
        let path = normalize_lexically(&info.path);
        if path == dir || !path.starts_with(&dir) {
            continue;
        }
        let name = info.name();
        let Some(modified) = directory_mtime(&info.path) else {
            report
                .skipped
                .push(skip(&name, &info.path, "cannot read worktree metadata"));
            continue;
        };
        let age_secs = age_secs(now, modified);
        if age_secs < policy.max_age_secs() {
            report
                .skipped
                .push(skip(&name, &info.path, "not old enough"));
            continue;
        }
        if policy.ephemeral_only && !is_ephemeral_slug(&name) {
            report
                .skipped
                .push(skip(&name, &info.path, "not an ephemeral worktree"));
            continue;
        }
        if policy
            .current_worktree
            .as_deref()
            .is_some_and(|current| normalize_lexically(current) == path)
        {
            report
                .skipped
                .push(skip(&name, &info.path, "current worktree"));
            continue;
        }
        let changes = git::status_porcelain(manager.runner().as_ref(), &info.path).await?;
        if !changes.is_empty() {
            report
                .skipped
                .push(skip(&name, &info.path, "uncommitted changes"));
            continue;
        }
        match git::has_unpushed_commits(manager.runner().as_ref(), &info.path).await {
            Ok(Some(false)) => {}
            Ok(Some(true)) => {
                report
                    .skipped
                    .push(skip(&name, &info.path, "unpushed commits"));
                continue;
            }
            _ => {
                report
                    .skipped
                    .push(skip(&name, &info.path, "cannot verify unpushed commits"));
                continue;
            }
        }
        report.candidates.push(CleanupCandidate {
            name,
            path: info.path.clone(),
            age_secs,
        });
    }

    if policy.dry_run {
        return Ok(report);
    }

    for candidate in &report.candidates {
        match git::worktree_remove(manager.runner().as_ref(), &repo, &candidate.path, false).await {
            Ok(()) => report.removed.push(candidate.path.clone()),
            Err(error) => {
                report
                    .skipped
                    .push(skip(&candidate.name, &candidate.path, &error.to_string()));
            }
        }
    }

    // Git is the source of truth: drop the registry records of worktrees this
    // pass (or an earlier one) removed.
    let existing: Vec<PathBuf> = git::worktree_list(manager.runner().as_ref(), &repo)
        .await?
        .into_iter()
        .map(|info| info.path)
        .collect();
    let index_path = manager.index_path();
    let mut index = WorktreeIndex::load(&index_path)?;
    if index.retain_existing(&existing) > 0 {
        index.save(&index_path)?;
    }

    if !report.removed.is_empty() {
        git::worktree_prune(manager.runner().as_ref(), &repo).await?;
    }

    Ok(report)
}

/// Builds a skip entry.
fn skip(name: &str, path: &Path, reason: &str) -> CleanupSkip {
    CleanupSkip {
        name: name.to_owned(),
        path: path.to_path_buf(),
        reason: reason.to_owned(),
    }
}

/// The directory modification time, or `None` when it cannot be read.
fn directory_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
}

/// Age of `modified` relative to `now`, in whole non-negative seconds.
fn age_secs(now: DateTime<Utc>, modified: SystemTime) -> u64 {
    let age = now.signed_duration_since(DateTime::<Utc>::from(modified));
    u64::try_from(age.num_seconds()).unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn ephemeral_patterns_match_the_three_shapes_only() {
        for slug in [
            "teammate-0123abcd",
            "agent-0123abc",
            "agent-team-x-0123abcd",
            "agent-a-b-0123abcd",
        ] {
            assert!(is_ephemeral_slug(slug), "{slug} should be ephemeral");
        }
        for slug in [
            "feature",
            "teammate-0123ABCD",
            "teammate-0123abc",
            "teammate-0123abcd-wip",
            "agent-0123abcd",
            "agent-team-x",
            "agent--x-0123abcd",
            "agent-team-a-very-long-label-over-limit-0123ABCD",
            "worktree-agent-0123abc",
        ] {
            assert!(!is_ephemeral_slug(slug), "{slug} should not be ephemeral");
        }
        assert_eq!(EPHEMERAL_PATTERNS.len(), 3);
        assert_eq!(COMPILED_EPHEMERAL_PATTERNS.len(), 3);
    }

    #[test]
    fn policy_defaults_match_the_python_configuration() {
        let policy = CleanupPolicy::default();
        assert_eq!(policy.max_age_days, DEFAULT_CLEANUP_AFTER_DAYS);
        assert!(policy.ephemeral_only);
        assert!(!policy.dry_run);
        assert!(policy.current_worktree.is_none());
        assert!(policy.now.is_none());
        assert_eq!(policy.max_age_secs(), 30 * 86_400);
        assert!(policy.reference_time().timestamp() > 0);
    }

    #[test]
    fn policy_builders_override_every_field() {
        let now = DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let policy = CleanupPolicy::default()
            .with_max_age_days(7)
            .with_ephemeral_only(false)
            .with_dry_run(true)
            .with_current_worktree("/repo/.worktrees/current")
            .with_now(now);
        assert_eq!(policy.max_age_days, 7);
        assert_eq!(policy.max_age_secs(), 7 * 86_400);
        assert!(!policy.ephemeral_only);
        assert!(policy.dry_run);
        assert_eq!(
            policy.current_worktree.as_deref(),
            Some(Path::new("/repo/.worktrees/current"))
        );
        assert_eq!(policy.reference_time(), now);
    }

    #[test]
    fn policy_serde_defaults_keep_the_documented_values() {
        let decoded: CleanupPolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(decoded, CleanupPolicy::default());

        let decoded: CleanupPolicy =
            serde_json::from_str(r#"{"max_age_days": 1, "dry_run": true, "future": 1}"#).unwrap();
        assert_eq!(decoded.max_age_days, 1);
        assert!(decoded.dry_run);
        assert!(decoded.ephemeral_only, "ephemeral-only is the default");

        let json = serde_json::to_value(CleanupPolicy::default()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "max_age_days": 30,
                "ephemeral_only": true,
                "dry_run": false,
                "current_worktree": null,
                "now": null,
            })
        );
    }

    #[test]
    fn age_is_whole_non_negative_seconds() {
        let now = DateTime::parse_from_rfc3339("2026-02-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let older = SystemTime::from(now - chrono::Duration::seconds(3600));
        assert_eq!(age_secs(now, older), 3600);

        let future = SystemTime::from(now + chrono::Duration::seconds(60));
        assert_eq!(age_secs(now, future), 0, "a future mtime never is an age");
    }
}
