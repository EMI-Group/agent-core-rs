//! Cleanup-pass scenarios: ephemeral matching, dry runs, the age threshold, and
//! every fail-closed skip rule.
//!
//! Everything here runs offline against a scripted [`GitRunner`]; directory ages
//! are produced with `File::set_modified` and the age reference is frozen through
//! [`CleanupPolicy::with_now`], so no test depends on the wall clock.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use chrono::{DateTime, Duration, Utc};

use super::{temp_root, ScriptedGitRunner};
use crate::worktree::{
    is_ephemeral_slug, CleanupPolicy, RecordingWorktreeEvents, WorktreeEvent, WorktreeIndex,
    WorktreeManager, WorktreeRecord,
};

/// The frozen age reference of every test.
fn now() -> DateTime<Utc> {
    DateTime::parse_from_rfc3339("2026-06-01T00:00:00Z")
        .expect("frozen clock")
        .with_timezone(&Utc)
}

/// The day count used by the default policy.
const OLD_DAYS: i64 = 40;
/// A directory this young is below the thirty-day threshold.
const YOUNG_DAYS: i64 = 1;

/// Builds a porcelain listing for the main worktree plus `extras`.
fn listing(repo: &Path, extras: &[&Path]) -> String {
    let head = "1".repeat(40);
    let mut text = format!(
        "worktree {}\nHEAD {head}\nbranch refs/heads/main\n",
        repo.display()
    );
    for path in extras {
        write!(
            text,
            "\nworktree {}\nHEAD {head}\nbranch refs/heads/worktree-x\n",
            path.display()
        )
        .expect("write listing");
    }
    text
}

/// Creates `<repo>/.worktrees/<slug>` and ages it by `days` relative to [`now`].
///
/// Returns `false` when the platform refuses to set the modification time; the
/// caller then skips its age assertions with an explanatory comment.
fn aged_worktree(repo: &Path, slug: &str, days: i64) -> (PathBuf, bool) {
    let path = repo.join(".worktrees").join(slug);
    std::fs::create_dir_all(&path).expect("worktree directory");
    let modified: SystemTime = (now() - Duration::days(days)).into();
    let aged = File::open(&path).and_then(|dir| dir.set_modified(modified));
    match aged {
        Ok(()) => (path, true),
        Err(error) => {
            eprintln!("skipping age assertions: cannot set the directory mtime: {error}");
            (path, false)
        }
    }
}

/// Saves one registry record for `repo`.
fn store_record(repo: &Path, record: WorktreeRecord) {
    let path = crate::worktree::index_path(repo);
    let mut index = WorktreeIndex::load(&path).expect("load registry");
    index.upsert(record);
    index.save(&path).expect("save registry");
}

/// Loads the registry of `repo`.
fn registry(repo: &Path) -> WorktreeIndex {
    WorktreeIndex::load(&crate::worktree::index_path(repo)).expect("load registry")
}

/// Builds a manager over `repo` with a recording sink.
fn build_manager(
    repo: &Path,
    runner: Arc<ScriptedGitRunner>,
) -> (WorktreeManager, Arc<RecordingWorktreeEvents>) {
    let events = Arc::new(RecordingWorktreeEvents::new());
    let manager = WorktreeManager::with_runner(repo, runner).with_events(events.clone());
    (manager, events)
}

/// The skip reasons of a report, paired with the worktree names.
fn skip_reasons(skipped: &[crate::worktree::CleanupSkip]) -> Vec<(String, String)> {
    skipped
        .iter()
        .map(|skip| (skip.name.clone(), skip.reason.clone()))
        .collect()
}

#[test]
fn ephemeral_slugs_match_teammates_agents_and_teams() {
    assert!(is_ephemeral_slug("teammate-0123abcd"));
    assert!(is_ephemeral_slug("agent-0123abc"));
    assert!(is_ephemeral_slug("agent-team-x-0123abcd"));
    assert!(!is_ephemeral_slug("feature-login"));
    assert!(!is_ephemeral_slug("agent-team-x"));
}

#[tokio::test]
async fn a_dry_run_lists_candidates_without_touching_git_or_the_registry() {
    let (_dir, repo) = temp_root();
    let (path, aged) = aged_worktree(&repo, "agent-0123abc", OLD_DAYS);
    store_record(&repo, WorktreeRecord::new("agent-0123abc", path.clone()));

    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    runner.push_success(); // status --porcelain
    runner.push_success(); // rev-list unpushed probe
    let (manager, events) = build_manager(&repo, runner.clone());

    let policy = CleanupPolicy::default().with_now(now()).with_dry_run(true);
    let report = manager.cleanup(&policy).await.expect("dry run");

    assert!(report.dry_run);
    assert_eq!(report.candidates.len(), 1);
    assert_eq!(report.candidates[0].name, "agent-0123abc");
    assert_eq!(report.candidates[0].path, path);
    if aged {
        assert!(
            report.candidates[0].age_secs >= 30 * 86_400,
            "age {} is below the threshold",
            report.candidates[0].age_secs
        );
    }
    assert_eq!(report.removed_count(), 0);
    assert_eq!(report.skipped.len(), 0);
    assert_eq!(
        runner.commands(),
        [
            "rev-parse --show-toplevel".to_owned(),
            "worktree list --porcelain".to_owned(),
            "status --porcelain".to_owned(),
            "rev-list --max-count=1 HEAD --not --remotes".to_owned(),
        ],
        "a dry run may inspect but never mutate git state"
    );
    assert!(
        registry(&repo).get("agent-0123abc").is_some(),
        "a dry run must not write the registry"
    );
    assert_eq!(
        events.events(),
        [WorktreeEvent::Cleaned {
            dry_run: true,
            removed: Vec::new(),
            skipped: 0,
        }]
    );
}

#[tokio::test]
async fn the_age_threshold_removes_old_worktrees_and_keeps_young_ones() {
    let (_dir, repo) = temp_root();
    let (old, old_aged) = aged_worktree(&repo, "agent-0123abc", OLD_DAYS);
    let (young, young_aged) = aged_worktree(&repo, "agent-0123abd", YOUNG_DAYS);
    if !(old_aged && young_aged) {
        return;
    }
    store_record(&repo, WorktreeRecord::new("agent-0123abc", old.clone()));
    store_record(&repo, WorktreeRecord::new("agent-0123abd", young.clone()));

    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&old, &young]));
    runner.push_success(); // status --porcelain (old)
    runner.push_success(); // rev-list (old)
    runner.push_success(); // worktree remove (old)
    runner.push_ok(&listing(&repo, &[&young]));
    runner.push_success(); // worktree prune
    let (manager, events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(&CleanupPolicy::default().with_now(now()))
        .await
        .expect("cleanup");

    assert!(!report.dry_run);
    assert_eq!(report.candidates.len(), 1);
    assert_eq!(report.candidates[0].name, "agent-0123abc");
    assert_eq!(report.removed.as_slice(), std::slice::from_ref(&old));
    assert_eq!(
        skip_reasons(&report.skipped),
        [("agent-0123abd".to_owned(), "not old enough".to_owned())]
    );
    assert_eq!(
        runner.commands(),
        [
            "rev-parse --show-toplevel".to_owned(),
            "worktree list --porcelain".to_owned(),
            "status --porcelain".to_owned(),
            "rev-list --max-count=1 HEAD --not --remotes".to_owned(),
            format!("worktree remove {}", old.display()),
            "worktree list --porcelain".to_owned(),
            "worktree prune".to_owned(),
        ]
    );
    let index = registry(&repo);
    assert!(
        index.get("agent-0123abc").is_none(),
        "the removed worktree's record must be dropped"
    );
    assert!(
        index.get("agent-0123abd").is_some(),
        "the young worktree stays registered"
    );
    assert_eq!(
        events.events(),
        [WorktreeEvent::Cleaned {
            dry_run: false,
            removed: vec![old],
            skipped: 1,
        }]
    );
}

#[tokio::test]
async fn non_ephemeral_worktrees_are_kept_unless_ephemeral_only_is_off() {
    let (_dir, repo) = temp_root();
    let (path, aged) = aged_worktree(&repo, "feature-login", OLD_DAYS);
    if !aged {
        return;
    }
    store_record(&repo, WorktreeRecord::new("feature-login", path.clone()));

    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    runner.push_ok(&listing(&repo, &[&path])); // registry reconciliation
    let (manager, _events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(&CleanupPolicy::default().with_now(now()))
        .await
        .expect("cleanup");
    assert_eq!(report.candidates.len(), 0);
    assert_eq!(
        skip_reasons(&report.skipped),
        [(
            "feature-login".to_owned(),
            "not an ephemeral worktree".to_owned()
        )]
    );
    assert_eq!(runner.commands().len(), 3, "no per-worktree probe may run");
    assert!(registry(&repo).get("feature-login").is_some());

    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    runner.push_success(); // status
    runner.push_success(); // rev-list
    runner.push_success(); // remove
    runner.push_ok(&listing(&repo, &[]));
    runner.push_success(); // prune
    let (manager, _events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(
            &CleanupPolicy::default()
                .with_now(now())
                .with_ephemeral_only(false),
        )
        .await
        .expect("cleanup");
    assert_eq!(report.removed, [path]);
    assert!(registry(&repo).is_empty());
}

#[tokio::test]
async fn a_dirty_worktree_is_skipped() {
    let (_dir, repo) = temp_root();
    let (path, aged) = aged_worktree(&repo, "agent-0123abc", OLD_DAYS);
    if !aged {
        return;
    }
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    runner.push_ok(" M src/lib.rs\n");
    runner.push_ok(&listing(&repo, &[&path])); // registry reconciliation
    let (manager, _events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(&CleanupPolicy::default().with_now(now()))
        .await
        .expect("cleanup");
    assert_eq!(report.removed.len(), 0);
    assert_eq!(
        skip_reasons(&report.skipped),
        [("agent-0123abc".to_owned(), "uncommitted changes".to_owned())]
    );
    assert_eq!(runner.commands().len(), 4, "no removal may run");
}

#[tokio::test]
async fn unpushed_state_is_fail_closed_both_when_true_and_when_unknown() {
    // `Some(true)`: the worktree has commits no remote contains.
    let (_dir, repo) = temp_root();
    let (path, aged) = aged_worktree(&repo, "agent-0123abc", OLD_DAYS);
    if !aged {
        return;
    }
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    runner.push_success(); // status: clean
    runner.push_ok("ffffffff"); // rev-list: an unpushed commit
    runner.push_ok(&listing(&repo, &[&path])); // registry reconciliation
    let (manager, _events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(&CleanupPolicy::default().with_now(now()))
        .await
        .expect("cleanup");
    assert_eq!(report.removed.len(), 0);
    assert_eq!(
        skip_reasons(&report.skipped),
        [("agent-0123abc".to_owned(), "unpushed commits".to_owned())]
    );

    // `None`: the probe failed, so safety cannot be proven.
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    runner.push_success(); // status: clean
    runner.push_fail("fatal: no such ref"); // rev-list probe fails
    runner.push_ok(&listing(&repo, &[&path])); // registry reconciliation
    let (manager, _events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(&CleanupPolicy::default().with_now(now()))
        .await
        .expect("cleanup");
    assert_eq!(report.removed.len(), 0);
    assert_eq!(
        skip_reasons(&report.skipped),
        [(
            "agent-0123abc".to_owned(),
            "cannot verify unpushed commits".to_owned()
        )]
    );
    assert_eq!(runner.commands().len(), 5, "no removal may run");
}

#[tokio::test]
async fn the_current_worktree_is_never_reaped() {
    let (_dir, repo) = temp_root();
    let (path, aged) = aged_worktree(&repo, "agent-0123abc", OLD_DAYS);
    if !aged {
        return;
    }
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    runner.push_ok(&listing(&repo, &[&path])); // registry reconciliation
    let (manager, _events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(
            &CleanupPolicy::default()
                .with_now(now())
                .with_current_worktree(path.clone()),
        )
        .await
        .expect("cleanup");
    assert_eq!(report.removed.len(), 0);
    assert_eq!(
        skip_reasons(&report.skipped),
        [("agent-0123abc".to_owned(), "current worktree".to_owned())]
    );
    assert_eq!(runner.commands().len(), 3, "no probe may run");

    // A lexically different spelling of the same directory is still the current
    // worktree, because the comparison normalises both sides first.
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    let (manager, _events) = build_manager(&repo, runner.clone());
    runner.push_ok(&listing(&repo, &[&path])); // registry reconciliation
    let spelled = repo
        .join(".worktrees")
        .join("..")
        .join(".worktrees")
        .join("agent-0123abc");
    let report = manager
        .cleanup(
            &CleanupPolicy::default()
                .with_now(now())
                .with_current_worktree(spelled),
        )
        .await
        .expect("cleanup");
    assert_eq!(report.removed.len(), 0);
    assert_eq!(
        skip_reasons(&report.skipped),
        [("agent-0123abc".to_owned(), "current worktree".to_owned())]
    );
}

#[tokio::test]
async fn a_failed_removal_becomes_a_skip_and_the_registry_keeps_the_record() {
    let (_dir, repo) = temp_root();
    let (path, aged) = aged_worktree(&repo, "agent-0123abc", OLD_DAYS);
    if !aged {
        return;
    }
    store_record(&repo, WorktreeRecord::new("agent-0123abc", path.clone()));
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&path]));
    runner.push_success(); // status
    runner.push_success(); // rev-list
    runner.push_fail("fatal: worktree is locked"); // remove fails
    runner.push_ok(&listing(&repo, &[&path])); // still listed
    let (manager, events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(&CleanupPolicy::default().with_now(now()))
        .await
        .expect("cleanup");
    assert_eq!(report.removed.len(), 0);
    assert_eq!(
        skip_reasons(&report.skipped),
        [(
            "agent-0123abc".to_owned(),
            "tool: fatal: worktree is locked".to_owned()
        )]
    );
    assert!(
        !runner
            .commands()
            .iter()
            .any(|command| command == "worktree prune"),
        "nothing was removed, so git must not be pruned"
    );
    assert!(registry(&repo).get("agent-0123abc").is_some());
    assert_eq!(
        events.events(),
        [WorktreeEvent::Cleaned {
            dry_run: false,
            removed: Vec::new(),
            skipped: 1,
        }]
    );
}

#[tokio::test]
async fn the_main_worktree_and_foreign_directories_are_never_candidates() {
    let (_dir, repo) = temp_root();
    let (inside, aged) = aged_worktree(&repo, "agent-0123abc", OLD_DAYS);
    if !aged {
        return;
    }
    let outside = repo.join("somewhere-else");
    std::fs::create_dir_all(&outside).expect("foreign directory");
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&repo.to_string_lossy());
    runner.push_ok(&listing(&repo, &[&inside, &outside]));
    runner.push_success(); // status --porcelain (inside)
    runner.push_success(); // rev-list unpushed probe (inside)
    let (manager, _events) = build_manager(&repo, runner.clone());

    let report = manager
        .cleanup(&CleanupPolicy::default().with_now(now()).with_dry_run(true))
        .await
        .expect("dry run");
    assert_eq!(report.candidates.len(), 1);
    assert_eq!(report.candidates[0].path, inside);
    assert_eq!(
        runner.commands().len(),
        4,
        "the main worktree is not probed"
    );
}
