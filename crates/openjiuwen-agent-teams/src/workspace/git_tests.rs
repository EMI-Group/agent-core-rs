//! Tests of the `git` CLI layer: porcelain parsing, path normalization, repository discovery,
//! branch deletion classification and the bounded command runner.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use tempfile::TempDir;

use super::git::{
    existing_ancestor, normalize_path, parse_worktree_list, run_git, BranchDeleteOutcome,
    GitRepository,
};
use super::test_repo::{git_ok, TestRepository};
use openjiuwen_common::error::Error;

#[test]
fn parse_worktree_list_reads_every_attribute() {
    let porcelain = "\
worktree /repo
HEAD 1111111111111111111111111111111111111111
branch refs/heads/main

worktree /repo/.agent_teams/worktrees/agent-my-team-dev-1
HEAD 2222222222222222222222222222222222222222
branch refs/heads/worktree-agent-my-team-dev-1
locked rebase in progress

worktree /repo/detached
HEAD 3333333333333333333333333333333333333333
detached

worktree /repo/bare
bare
";
    let entries = parse_worktree_list(porcelain);
    assert_eq!(entries.len(), 4);
    assert_eq!(entries[0].path, Path::new("/repo"));
    assert_eq!(
        entries[0].branch.as_deref(),
        Some("main"),
        "the refs/heads/ prefix is stripped"
    );
    assert!(entries[0].locked.is_none());
    assert_eq!(entries[1].locked.as_deref(), Some("rebase in progress"));
    assert_eq!(
        entries[1].branch.as_deref(),
        Some("worktree-agent-my-team-dev-1")
    );
    assert!(entries[2].detached);
    assert!(entries[2].branch.is_none());
    assert!(entries[3].bare);
}

#[test]
fn parse_worktree_list_tolerates_unknown_lines_and_trailing_entries() {
    let porcelain = "\
worktree /repo
HEAD 1111111111111111111111111111111111111111
branch refs/heads/main
prunable gitdir file points to non-existent location
some-future-attribute value";
    let entries = parse_worktree_list(porcelain);
    assert_eq!(
        entries.len(),
        1,
        "a trailing entry without a blank line counts"
    );
    assert_eq!(entries[0].branch.as_deref(), Some("main"));
    assert_eq!(parse_worktree_list(""), Vec::new());
}

#[test]
fn normalize_path_drops_dot_segments() {
    assert_eq!(normalize_path(Path::new("/a/./b/../c")), Path::new("/a/c"));
    assert_eq!(normalize_path(Path::new("a/b/../c")), Path::new("a/c"));
    assert_eq!(normalize_path(Path::new("/..")), Path::new("/"));
    assert_eq!(normalize_path(Path::new(".")), Path::new(""));
}

#[test]
fn existing_ancestor_walks_up_to_a_real_directory() {
    let temp = TempDir::new().unwrap();
    let deep = temp.path().join("does/not/exist/yet");
    assert_eq!(existing_ancestor(&deep), temp.path());
    assert_eq!(existing_ancestor(temp.path()), temp.path());
}

#[tokio::test]
async fn run_git_captures_a_failed_command_instead_of_failing() {
    let temp = TempDir::new().unwrap();
    let output = run_git(temp.path(), &["rev-parse", "--show-toplevel"])
        .await
        .expect("the git binary is available");
    assert!(!output.succeeded());
    assert!(
        output.stderr.contains("not a git repository"),
        "{}",
        output.stderr
    );
    assert!(output.stderr_summary().contains("not a git repository"));
}

#[tokio::test]
async fn discover_resolves_the_repository_root() {
    let repo = TestRepository::init().await;
    let discovered = GitRepository::discover(&repo.teams_home().join("missing/deeper"))
        .await
        .expect("repository is discovered from a not-yet-existing sub-directory");
    assert_eq!(discovered.root, repo.root());
    assert_eq!(discovered.common_dir, repo.root().join(".git"));
    assert_eq!(discovered.git_dir, repo.root().join(".git"));
    assert_eq!(
        discovered.lock_file(),
        repo.root().join(".git").join("openjiuwen-worktree.lock")
    );
}

#[tokio::test]
async fn discover_reports_a_missing_repository_as_config() {
    let temp = TempDir::new().unwrap();
    let err = GitRepository::discover(temp.path())
        .await
        .expect_err("a temp directory is not a repository");
    assert!(matches!(err, Error::Config(_)), "{err:?}");
    assert!(
        err.to_string().contains("requires a git repository"),
        "{err}"
    );
}

#[tokio::test]
async fn discover_from_inside_a_worktree_resolves_the_main_repository() {
    let repo = TestRepository::init().await;
    let path = repo.worktree_path("agent-main-root");
    git_ok(
        repo.root(),
        &[
            "worktree",
            "add",
            "-b",
            "worktree-agent-main-root",
            path.to_str().unwrap(),
            "HEAD",
        ],
    )
    .await;
    let discovered = GitRepository::discover(&path)
        .await
        .expect("a worktree is inside a repository");
    assert_eq!(
        discovered.root,
        repo.root(),
        "commands always target the main repository, never the worktree"
    );
    assert!(
        discovered.git_dir.ends_with("worktrees/agent-main-root"),
        "{:?}",
        discovered.git_dir
    );
}

#[tokio::test]
async fn delete_branch_classifies_every_outcome() {
    let repo = TestRepository::init().await;
    let discovered = GitRepository::discover(repo.root()).await.unwrap();
    git_ok(repo.root(), &["branch", "feature"]).await;

    assert_eq!(
        discovered.delete_branch("feature").await.unwrap(),
        BranchDeleteOutcome::Deleted
    );
    assert_eq!(
        discovered.delete_branch("feature").await.unwrap(),
        BranchDeleteOutcome::Missing
    );

    let path = repo.worktree_path("agent-occupied");
    git_ok(
        repo.root(),
        &[
            "worktree",
            "add",
            "-b",
            "occupied",
            path.to_str().unwrap(),
            "HEAD",
        ],
    )
    .await;
    match discovered.delete_branch("occupied").await.unwrap() {
        BranchDeleteOutcome::InUse(detail) => {
            assert!(
                detail.contains("used by worktree") || detail.contains("checked out"),
                "{detail}"
            );
        }
        other => panic!("expected InUse, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_branch_rejects_unsafe_names() {
    let repo = TestRepository::init().await;
    let discovered = GitRepository::discover(repo.root()).await.unwrap();
    for bad in ["", "   ", "-D", "feature branch"] {
        let err = discovered.delete_branch(bad).await.expect_err(bad);
        assert!(matches!(err, Error::Validation(_)), "{bad:?}: {err:?}");
    }
}

#[tokio::test]
async fn resolve_ref_reports_both_answers() {
    let repo = TestRepository::init().await;
    let discovered = GitRepository::discover(repo.root()).await.unwrap();
    let head = discovered.resolve_ref("HEAD").await.unwrap();
    assert_eq!(head.as_deref(), repo.head(repo.root()).await.as_deref());
    assert!(discovered
        .resolve_ref("does-not-exist")
        .await
        .unwrap()
        .is_none());
}
