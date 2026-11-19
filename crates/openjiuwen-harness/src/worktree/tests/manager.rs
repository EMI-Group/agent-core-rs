//! `WorktreeManager` scenarios: creation, recovery, refusals, listing, removal,
//! locking, status, pruning, events, and one real-git round trip.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;

use super::{listing, real_git_available, temp_root, ScriptedGitRunner};
use crate::worktree::{
    RecordingWorktreeEvents, SparseConfig, WorktreeCreateOptions, WorktreeEvent, WorktreeIndex,
    WorktreeLockConfig, WorktreeManager, WorktreeRecord,
};

/// Builds a manager over a scripted runner and a recording event sink.
fn build_manager(
    repo: &Path,
    runner: Arc<ScriptedGitRunner>,
) -> (WorktreeManager, Arc<RecordingWorktreeEvents>) {
    let events = Arc::new(RecordingWorktreeEvents::new());
    let manager = WorktreeManager::with_runner(repo, runner).with_events(events.clone());
    (manager, events)
}

/// The stdout of `rev-parse --show-toplevel` for `repo`.
fn root(repo: &Path) -> String {
    repo.to_string_lossy().into_owned()
}

/// Saves one registry record for `repo`.
fn store_record(repo: &Path, record: WorktreeRecord) {
    let path = crate::worktree::index_path(repo);
    let mut index = WorktreeIndex::load(&path).unwrap();
    index.upsert(record);
    index.save(&path).unwrap();
}

#[tokio::test]
async fn create_checks_out_a_new_branch_and_records_it() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("fatal: Needed a single revision");
    runner.push_success();
    runner.push_ok("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    let (manager, events) = build_manager(&repo, runner.clone());

    let created = manager.create("feature", None, None).await.unwrap();
    let target = manager.target_path("feature").unwrap();
    assert_eq!(created.name, "feature");
    assert_eq!(created.slug, "feature");
    assert_eq!(created.path, target);
    assert_eq!(created.branch, "worktree-feature");
    assert_eq!(created.base, None);
    assert_eq!(
        created.head.as_deref(),
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    );
    assert!(!created.existed);

    assert_eq!(
        runner.commands(),
        [
            "rev-parse --show-toplevel".to_owned(),
            "worktree list --porcelain".to_owned(),
            "rev-parse --verify --quiet refs/heads/worktree-feature".to_owned(),
            format!("worktree add -b worktree-feature {} HEAD", target.display()),
            "rev-parse HEAD".to_owned(),
        ]
    );
    assert_eq!(runner.remaining(), 0);

    let index = WorktreeIndex::load(&manager.index_path()).unwrap();
    let record = index.get("feature").unwrap();
    assert_eq!(record.path, target);
    assert_eq!(record.branch.as_deref(), Some("worktree-feature"));
    assert_ne!(record.created_at, "");
    assert_eq!(record.status, crate::worktree::WorktreeState::Active);

    assert_eq!(
        events.events(),
        [WorktreeEvent::Created {
            name: "feature".to_owned(),
            path: target,
            branch: "worktree-feature".to_owned(),
            existed: false,
            owner: None,
        }]
    );
}

#[tokio::test]
async fn create_normalises_the_name_into_a_slug() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("no such branch");
    runner.push_success();
    runner.push_ok("bbbb");
    let (manager, _events) = build_manager(&repo, runner);

    let created = manager.create("Feature Login!", None, None).await.unwrap();
    assert_eq!(created.slug, "Feature-Login");
    assert_eq!(created.branch, "worktree-Feature-Login");
    assert!(created.path.ends_with(".worktrees/Feature-Login"));
}

#[tokio::test]
async fn create_recovers_an_existing_worktree_without_running_add() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    let target = repo.join(".worktrees/feature");
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    let (manager, events) = build_manager(&repo, runner.clone());

    let created = manager.create("feature", None, None).await.unwrap();
    assert!(created.existed);
    assert_eq!(created.path, target);
    assert_eq!(runner.commands().len(), 2);
    assert!(manager.index_path().exists());

    match &events.events()[0] {
        WorktreeEvent::Created { existed, .. } => assert!(existed),
        other => panic!("unexpected event {other:?}"),
    }
}

#[tokio::test]
async fn create_refuses_an_occupied_target_directory() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    std::fs::create_dir_all(&target).unwrap();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    let (manager, events) = build_manager(&repo, runner.clone());

    let error = manager.create("feature", None, None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("already exists"));
    assert_eq!(runner.commands().len(), 2, "no add must run");
    assert!(!manager.index_path().exists());

    assert!(matches!(
        events.events().last(),
        Some(WorktreeEvent::Error { action, .. }) if action == "create"
    ));
}

#[tokio::test]
async fn create_checks_out_an_existing_branch_instead_of_recreating_it() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_ok("cccc");
    runner.push_success();
    runner.push_ok("dddd");
    let (manager, _events) = build_manager(&repo, runner.clone());

    manager
        .create("feature", Some("existing-branch"), Some("main"))
        .await
        .unwrap();
    assert_eq!(
        runner.commands()[3],
        format!(
            "worktree add {} existing-branch",
            manager.target_path("feature").unwrap().display()
        )
    );
    assert_eq!(
        runner.commands()[2],
        "rev-parse --verify --quiet refs/heads/existing-branch"
    );
}

#[tokio::test]
async fn create_refuses_names_that_escape_the_repository() {
    let (_dir, repo) = temp_root();
    let (manager, _events) = build_manager(&repo, Arc::new(ScriptedGitRunner::new()));

    for name in ["../escape", "/etc/passwd", "a/../../b", "C:\\repo", "a\0b"] {
        let error = manager.create(name, None, None).await.unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::Security,
            "unexpected kind for {name:?}: {error}"
        );
    }
    assert!(manager.target_path("..").is_err());
    assert!(manager.target_path("").is_err());
}

#[tokio::test]
async fn create_reports_a_missing_repository_with_a_typed_error() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_fail("fatal: not a git repository (or any of the parent directories): .git");
    let (manager, _events) = build_manager(&repo, runner);

    let error = manager.create("feature", None, None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error
        .to_string()
        .contains(crate::worktree::NOT_A_REPOSITORY_MARKER));
}

#[tokio::test]
async fn list_attaches_registry_records_of_earlier_runs() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    store_record(
        &repo,
        WorktreeRecord::new("feature", target.clone()).with_branch("worktree-feature"),
    );
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    let (manager, _events) = build_manager(&repo, runner);

    let infos = manager.list().await.unwrap();
    assert_eq!(infos.len(), 2);
    assert_eq!(infos[0].recorded, None, "the main worktree is not managed");
    let linked = &infos[1];
    assert_eq!(linked.path, target);
    assert_eq!(linked.branch.as_deref(), Some("worktree-feature"));
    assert_eq!(
        linked.recorded.as_ref().map(|record| record.name.as_str()),
        Some("feature")
    );
}

#[tokio::test]
async fn remove_refuses_a_dirty_worktree_without_force() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    store_record(&repo, WorktreeRecord::new("feature", target.clone()));
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_ok(" M src/lib.rs\n?? notes.txt\n");
    let (manager, events) = build_manager(&repo, runner.clone());

    let error = manager.remove("feature", false).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("2 uncommitted change(s)"));
    assert_eq!(runner.commands().len(), 3, "no remove must run");
    assert!(matches!(
        events.events().last(),
        Some(WorktreeEvent::Error { action, .. }) if action == "remove"
    ));
    assert!(WorktreeIndex::load(&manager.index_path())
        .unwrap()
        .get("feature")
        .is_some());
}

#[tokio::test]
async fn remove_force_removes_the_worktree_and_its_record() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    store_record(&repo, WorktreeRecord::new("feature", target.clone()));
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success();
    let (manager, events) = build_manager(&repo, runner.clone());

    let removed = manager.remove("feature", true).await.unwrap();
    assert_eq!(removed.path, target);
    assert_eq!(
        runner.commands()[2],
        format!("worktree remove --force {}", target.display())
    );
    assert!(WorktreeIndex::load(&manager.index_path())
        .unwrap()
        .is_empty());
    assert_eq!(
        events.events(),
        [WorktreeEvent::Removed {
            name: "feature".to_owned(),
            path: target,
            forced: true,
        }]
    );
}

#[tokio::test]
async fn remove_reports_an_unknown_worktree_as_not_found() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    let (manager, _events) = build_manager(&repo, runner);

    let error = manager.remove("feature", false).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[tokio::test]
async fn lock_sends_the_reason_and_unlock_clears_it() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success();
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success();
    let (manager, events) = build_manager(&repo, runner.clone());

    manager.lock("feature", Some("maintenance")).await.unwrap();
    manager.unlock("feature").await.unwrap();

    assert_eq!(
        runner.commands()[2],
        format!("worktree lock --reason maintenance {}", target.display())
    );
    assert_eq!(
        runner.commands()[5],
        format!("worktree unlock {}", target.display())
    );
    assert_eq!(
        events.events(),
        [
            WorktreeEvent::Locked {
                name: "feature".to_owned(),
                path: target.clone(),
                reason: Some("maintenance".to_owned()),
            },
            WorktreeEvent::Unlocked {
                name: "feature".to_owned(),
                path: target,
            },
        ]
    );
}

#[tokio::test]
async fn locking_an_unknown_worktree_is_not_found() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    let (manager, _events) = build_manager(&repo, runner);

    let error = manager.lock("feature", None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[tokio::test]
async fn status_reports_dirty_state_and_registry_tracking() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    store_record(&repo, WorktreeRecord::new("feature", target.clone()));
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_ok(" M src/lib.rs\n");
    let (manager, _events) = build_manager(&repo, runner.clone());

    let status = manager.status("feature").await.unwrap();
    assert!(status.dirty);
    assert!(!status.is_clean());
    assert_eq!(status.changed_files, 1);
    assert!(status.tracked);
    assert!(!status.locked);
    assert_eq!(status.branch.as_deref(), Some("worktree-feature"));
    assert_eq!(runner.last_call().unwrap().cwd, target);
}

#[tokio::test]
async fn status_of_an_untracked_worktree_is_not_tracked() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_ok("");
    let (manager, _events) = build_manager(&repo, runner);

    let status = manager.status("feature").await.unwrap();
    assert!(status.is_clean());
    assert!(!status.tracked);
}

#[tokio::test]
async fn prune_drops_stale_records_and_reports_the_count() {
    let (_dir, repo) = temp_root();
    store_record(
        &repo,
        WorktreeRecord::new("stale", repo.join(".worktrees/stale")),
    );
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_success();
    runner.push_ok(&listing(&repo, None));
    let (manager, events) = build_manager(&repo, runner.clone());

    let removed = manager.prune().await.unwrap();
    assert_eq!(removed, 1);
    assert_eq!(
        runner.commands(),
        [
            "rev-parse --show-toplevel".to_owned(),
            "worktree prune".to_owned(),
            "worktree list --porcelain".to_owned(),
        ]
    );
    assert!(WorktreeIndex::load(&manager.index_path())
        .unwrap()
        .is_empty());
    assert_eq!(events.events(), [WorktreeEvent::Pruned { removed: 1 }]);
}

#[tokio::test]
async fn cleanup_stale_keeps_records_of_listed_worktrees() {
    let (_dir, repo) = temp_root();
    store_record(&repo, WorktreeRecord::new("main", repo.clone()));
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    let (manager, _events) = build_manager(&repo, runner);

    assert_eq!(manager.cleanup_stale().await.unwrap(), 0);
    assert!(WorktreeIndex::load(&manager.index_path())
        .unwrap()
        .get("main")
        .is_some());
}

#[tokio::test]
async fn index_records_the_owner_when_one_is_configured() {
    let (_dir, repo) = temp_root();
    let events = Arc::new(RecordingWorktreeEvents::new());
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("no branch");
    runner.push_success();
    runner.push_ok("eeee");
    let manager = WorktreeManager::with_runner(&repo, runner)
        .with_owner("agent-7")
        .with_events(events.clone());

    manager.create("feature", None, None).await.unwrap();
    let index = WorktreeIndex::load(&manager.index_path()).unwrap();
    assert_eq!(
        index.get("feature").unwrap().owner.as_deref(),
        Some("agent-7")
    );
    assert_eq!(manager.owner(), Some("agent-7"));
    assert!(matches!(
        events.events().as_slice(),
        [WorktreeEvent::Created { owner: Some(owner), .. }] if owner == "agent-7"
    ));
}

#[tokio::test]
async fn a_git_runner_failure_is_surfaced_and_evented() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("no branch");
    runner.push_fail("fatal: '<target>' is a missing but already registered worktree");
    let (manager, events) = build_manager(&repo, runner);

    let error = manager.create("feature", None, None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(error.to_string().contains("missing but already registered"));
    assert!(matches!(
        events.events().last(),
        Some(WorktreeEvent::Error { action, message })
            if action == "create" && message.contains("missing but already registered")
    ));
}

/// Runs a real git command in `cwd` and returns its stdout.
fn git(cwd: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(["-c", "user.name=test", "-c", "user.email=test@example.com"])
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[tokio::test]
async fn real_git_round_trip_creates_lists_and_removes_a_worktree() {
    if !real_git_available() {
        eprintln!("skipping: git is not installed");
        return;
    }
    let (_dir, repo) = temp_root();
    git(&repo, &["init", "-q"]);
    git(&repo, &["commit", "--allow-empty", "-q", "-m", "init"]);

    let manager = WorktreeManager::new(&repo);
    let reported = manager.ensure_repository().await.unwrap();
    assert_eq!(
        reported.canonicalize().unwrap(),
        repo.canonicalize().unwrap()
    );

    let created = manager.create("feature", None, None).await.unwrap();
    let target: PathBuf = repo.join(".worktrees/feature");
    assert!(!created.existed);
    assert_eq!(created.path, target);
    assert_eq!(created.branch, "worktree-feature");
    assert!(target.join(".git").exists());
    assert!(created.head.is_some());

    let listed = manager.list().await.unwrap();
    assert_eq!(listed.len(), 2);
    assert!(listed
        .iter()
        .any(|info| info.path == target && info.recorded.is_some()));

    let status = manager.status("feature").await.unwrap();
    assert!(status.is_clean());
    assert!(status.tracked);

    manager.lock("feature", Some("test")).await.unwrap();
    assert!(manager.status("feature").await.unwrap().locked);
    manager.unlock("feature").await.unwrap();

    manager.remove("feature", false).await.unwrap();
    assert!(!target.exists());
    assert!(WorktreeIndex::load(&manager.index_path())
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn create_with_options_applies_a_sparse_selection_and_records_it() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("no branch");
    runner.push_success(); // worktree add --no-checkout
    runner.push_success(); // sparse-checkout set
    runner.push_success(); // checkout HEAD
    runner.push_ok("abcdef0123456789");
    let (manager, events) = build_manager(&repo, runner.clone());
    let sparse = SparseConfig::cone(["src", "lib"]).unwrap();
    let target = manager.target_path("feature").unwrap();

    let created = manager
        .create_with_options(
            "feature",
            None,
            None,
            &WorktreeCreateOptions::default().with_sparse(sparse.clone()),
        )
        .await
        .unwrap();

    assert_eq!(
        runner.commands(),
        [
            "rev-parse --show-toplevel".to_owned(),
            "worktree list --porcelain".to_owned(),
            "rev-parse --verify --quiet refs/heads/worktree-feature".to_owned(),
            format!(
                "worktree add --no-checkout -b worktree-feature {} HEAD",
                target.display()
            ),
            "sparse-checkout set --cone -- src lib".to_owned(),
            "checkout HEAD".to_owned(),
            "rev-parse HEAD".to_owned(),
        ]
    );
    assert_eq!(runner.remaining(), 0);
    assert_eq!(created.sparse, Some(sparse.clone()));
    assert_eq!(created.head.as_deref(), Some("abcdef0123456789"));
    assert!(!created.existed);

    let record = WorktreeIndex::load(&manager.index_path())
        .unwrap()
        .get("feature")
        .cloned()
        .unwrap();
    assert_eq!(record.sparse, Some(sparse.clone()));
    assert_eq!(record.head.as_deref(), Some("abcdef0123456789"));

    assert_eq!(
        events.events(),
        [
            WorktreeEvent::Created {
                name: "feature".to_owned(),
                path: target.clone(),
                branch: "worktree-feature".to_owned(),
                existed: false,
                owner: None,
            },
            WorktreeEvent::SparseConfigured {
                name: "feature".to_owned(),
                path: target,
                mode: crate::worktree::SparseMode::Cone,
                paths: vec!["src".to_owned(), "lib".to_owned()],
            },
        ]
    );
}

#[tokio::test]
async fn a_failed_sparse_setup_removes_the_half_created_worktree() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("no branch");
    runner.push_success(); // worktree add --no-checkout
    runner.push_fail("fatal: sparse boom");
    runner.push_success(); // worktree remove --force
    let (manager, events) = build_manager(&repo, runner.clone());
    let target = manager.target_path("feature").unwrap();

    let error = manager
        .create_with_options(
            "feature",
            None,
            None,
            &WorktreeCreateOptions::default().with_sparse(SparseConfig::cone(["src"]).unwrap()),
        )
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(crate::worktree::is_sparse_failure(&error), "{error}");
    assert!(error.to_string().contains("sparse boom"));
    let expected = format!("worktree remove --force {}", target.display());
    assert_eq!(
        runner.commands().last().map(String::as_str),
        Some(expected.as_str()),
        "the half-created worktree is removed again"
    );
    assert!(WorktreeIndex::load(&manager.index_path())
        .unwrap()
        .get("feature")
        .is_none());
    assert!(matches!(
        events.events().last(),
        Some(WorktreeEvent::Error { action, .. }) if action == "create"
    ));
}

#[tokio::test]
async fn create_recovers_an_existing_sparse_worktree_and_reapplies_it() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    let sparse = SparseConfig::cone(["src"]).unwrap();
    store_record(
        &repo,
        WorktreeRecord::new("feature", target.clone()).with_sparse(sparse.clone()),
    );
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success(); // sparse-checkout set
    let (manager, events) = build_manager(&repo, runner.clone());

    let created = manager.create("feature", None, None).await.unwrap();

    assert!(created.existed);
    assert_eq!(created.sparse, Some(sparse.clone()));
    assert_eq!(
        runner.commands(),
        [
            "rev-parse --show-toplevel".to_owned(),
            "worktree list --porcelain".to_owned(),
            "sparse-checkout set --cone -- src".to_owned(),
        ],
        "recovery re-applies the selection but never runs checkout HEAD"
    );
    let record = WorktreeIndex::load(&manager.index_path())
        .unwrap()
        .get("feature")
        .cloned()
        .unwrap();
    assert_eq!(record.sparse, Some(sparse));
    assert_eq!(
        record.head.as_deref(),
        Some("2222222222222222222222222222222222222222")
    );
    assert!(matches!(
        events.events().as_slice(),
        [
            WorktreeEvent::Created { existed: true, .. },
            WorktreeEvent::SparseConfigured { .. }
        ]
    ));
}

#[tokio::test]
async fn apply_sparse_returns_the_recorded_selection_or_none() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    let sparse = SparseConfig::patterns(["*.rs"]).unwrap();
    store_record(
        &repo,
        WorktreeRecord::new("feature", target.clone()).with_sparse(sparse.clone()),
    );
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success(); // sparse-checkout set
    let (manager, events) = build_manager(&repo, runner.clone());

    assert_eq!(manager.apply_sparse("feature").await.unwrap(), Some(sparse));
    assert_eq!(
        runner.commands()[2],
        "sparse-checkout set --no-cone -- *.rs"
    );
    assert_eq!(
        events.events(),
        [WorktreeEvent::SparseConfigured {
            name: "feature".to_owned(),
            path: target.clone(),
            mode: crate::worktree::SparseMode::Patterns,
            paths: vec!["*.rs".to_owned()],
        }]
    );

    // A worktree without a registry record has nothing to re-apply.
    let plain = repo.join(".worktrees/plain");
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&plain, "worktree-plain"))));
    let (manager, events) = build_manager(&repo, runner.clone());
    assert_eq!(manager.apply_sparse("plain").await.unwrap(), None);
    assert_eq!(runner.commands().len(), 2, "no sparse command is issued");
    assert_eq!(events.events().len(), 0);
}

#[tokio::test]
async fn apply_sparse_reports_an_unlisted_worktree_as_not_found() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    let (manager, events) = build_manager(&repo, runner.clone());

    let error = manager.apply_sparse("feature").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(matches!(
        events.events().last(),
        Some(WorktreeEvent::Error { action, .. }) if action == "apply_sparse"
    ));
}

#[tokio::test]
async fn status_reports_the_recorded_sparse_selection() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    let sparse = SparseConfig::cone(["src"]).unwrap();
    store_record(
        &repo,
        WorktreeRecord::new("feature", target.clone()).with_sparse(sparse.clone()),
    );
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_ok("");
    let (manager, _events) = build_manager(&repo, runner);

    let status = manager.status("feature").await.unwrap();
    assert_eq!(status.sparse, Some(sparse));
    assert!(status.tracked);
}

#[tokio::test]
async fn change_summary_uses_the_recorded_head_as_its_base() {
    let (_dir, repo) = temp_root();
    let target = repo.join(".worktrees/feature");
    store_record(
        &repo,
        WorktreeRecord::new("feature", target.clone()).with_head("abc123"),
    );
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_ok(" M src/lib.rs\n?? new.txt\n");
    runner.push_ok(" 2 files changed, 4 insertions(+)");
    runner.push_ok("3");
    runner.push_ok("ffffffff");
    let (manager, _events) = build_manager(&repo, runner.clone());

    let summary = manager.change_summary("feature").await.unwrap();
    assert_eq!(summary.changed_files, 2);
    assert_eq!(summary.files, ["src/lib.rs", "new.txt"]);
    assert_eq!(summary.commits, 3);
    assert_eq!(summary.unpushed, Some(true));
    assert!(!summary.truncated);
    assert_eq!(
        runner.commands()[2..],
        [
            "status --porcelain".to_owned(),
            "diff --stat abc123..HEAD".to_owned(),
            "rev-list --count abc123..HEAD".to_owned(),
            "rev-list --max-count=1 HEAD --not --remotes".to_owned(),
        ]
    );
    assert_eq!(runner.last_call().unwrap().cwd, target);
}

#[tokio::test]
async fn a_disabled_lock_config_still_runs_every_operation() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("no branch");
    runner.push_success();
    runner.push_ok("abcd");
    runner.push_ok(&root(&repo));
    let target = repo.join(".worktrees/feature");
    runner.push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    runner.push_success();
    let (manager, _events) = build_manager(&repo, runner.clone());
    let manager = manager.with_lock_config(WorktreeLockConfig::disabled());
    assert!(!manager.lock_config().enabled);

    manager.create("feature", None, None).await.unwrap();
    manager.remove("feature", true).await.unwrap();

    assert_eq!(
        runner.commands(),
        [
            "rev-parse --show-toplevel".to_owned(),
            "worktree list --porcelain".to_owned(),
            "rev-parse --verify --quiet refs/heads/worktree-feature".to_owned(),
            format!("worktree add -b worktree-feature {} HEAD", target.display()),
            "rev-parse HEAD".to_owned(),
            "rev-parse --show-toplevel".to_owned(),
            "worktree list --porcelain".to_owned(),
            format!("worktree remove --force {}", target.display()),
        ]
    );
}

#[tokio::test]
async fn a_contended_repository_lock_times_out_with_a_typed_error() {
    let (_dir, repo) = temp_root();
    let runner = Arc::new(ScriptedGitRunner::new());
    let (manager, _events) = build_manager(&repo, runner.clone());
    let holder = manager.repo_lock().acquire().await.unwrap();
    assert!(holder.is_active(), "the advisory lock must be held");
    assert_eq!(manager.lock_path(), holder.lock_path());

    let events = Arc::new(RecordingWorktreeEvents::new());
    let contender = WorktreeManager::with_runner(&repo, runner.clone())
        .with_events(events.clone())
        .with_lock_config(WorktreeLockConfig::default().with_timeout(Duration::from_millis(1)));
    let error = contender.create("feature", None, None).await.unwrap_err();
    assert!(
        crate::worktree::is_lock_timeout(&error),
        "expected a lock timeout, got {error}"
    );
    assert!(
        runner.commands().is_empty(),
        "no git command may run without the lock"
    );
    assert!(matches!(
        events.events().last(),
        Some(WorktreeEvent::Error { action, message })
            if action == "create" && message.contains("worktree lock timeout")
    ));

    // Releasing the lock lets the same manager proceed.
    drop(holder);
    runner.push_ok(&root(&repo));
    runner.push_ok(&listing(&repo, None));
    runner.push_fail("no branch");
    runner.push_success();
    runner.push_ok("abcd");
    contender.create("feature", None, None).await.unwrap();
    assert_eq!(runner.commands().len(), 5);
}

/// A hanging git command must not hold the repository lock past its deadline.
#[cfg(unix)]
#[tokio::test]
async fn a_timed_out_git_command_releases_the_repository_lock() {
    use std::os::unix::fs::PermissionsExt;
    let (_dir, repo) = temp_root();
    let script = repo.join("slow-git");
    std::fs::write(&script, "#!/bin/sh\nexec sleep 30\n").unwrap();
    let mut permissions = std::fs::metadata(&script).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&script, permissions).unwrap();
    let runner = Arc::new(
        crate::worktree::ProcessGitRunner::with_program(script.to_string_lossy().into_owned())
            .with_timeout(Duration::from_millis(250)),
    );
    let manager = WorktreeManager::with_runner(&repo, runner);
    let started = std::time::Instant::now();
    let error = manager.create("feature", None, None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.to_string().contains("timed out after"));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the create must fail at the git deadline, took {:?}",
        started.elapsed()
    );
    let guard = tokio::time::timeout(Duration::from_secs(2), manager.repo_lock().try_acquire())
        .await
        .expect("the lock must be free once the create failed")
        .expect("the lock must be free once the create failed");
    assert!(guard.is_active());
}
