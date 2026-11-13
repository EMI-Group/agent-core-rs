//! Tests of [`super::worktree_git::GitWorktreeBackend`] against real temporary repositories.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use super::git::run_git;
use super::layout::MemberWorkspaceLayout;
use super::local::LocalWorkspaceProvider;
use super::test_repo::{git_ok, TestRepository};
use super::worktree::{worktree_branch, WorktreeBackend, WorktreeCreateRequest};
use super::worktree_git::{teammate_worktrees, GitWorktreeBackend};
use super::{IsolationMode, WorkspaceProvider, WorkspaceRequest};
use openjiuwen_common::error::Error;

#[tokio::test]
async fn create_list_lock_unlock_remove_round_trip() {
    let repo = TestRepository::init().await;
    let backend = GitWorktreeBackend::new();
    let request = repo.create_request("agent-my-team-dev-1-abcdef1234");

    let info = backend.create(&request).await.expect("worktree is created");
    assert_eq!(info.name, request.name);
    assert_eq!(info.path, request.path);
    assert_eq!(info.branch, worktree_branch(&request.name));
    assert!(info.path.is_dir(), "git checked the worktree out");
    assert!(info.path.join("README.md").is_file());
    assert!(repo.has_branch(&info.branch).await);

    let listed = backend
        .list(repo.root())
        .await
        .expect("worktrees are listed");
    assert_eq!(listed, vec![info.clone()]);

    backend
        .lock(&info, "review in progress")
        .await
        .expect("locked");
    let entries = backend.list(repo.root()).await.unwrap();
    assert_eq!(entries.len(), 1);
    let remove_locked = backend.remove(&info).await;
    assert!(
        remove_locked.is_err(),
        "a locked worktree must not be removed silently"
    );

    backend.unlock(&info).await.expect("unlocked");
    backend.remove(&info).await.expect("worktree is removed");
    assert!(!info.path.exists(), "the worktree directory is gone");
    assert!(
        !repo.has_branch(&info.branch).await,
        "the worktree branch is gone"
    );
    assert!(
        backend.list(repo.root()).await.unwrap().is_empty(),
        "no teammate worktree is registered any more"
    );
}

#[tokio::test]
async fn create_refuses_to_reuse_an_existing_worktree() {
    let repo = TestRepository::init().await;
    let backend = GitWorktreeBackend::new();
    let request = repo.create_request("agent-my-team-dev-1-reuse");

    backend.create(&request).await.expect("first create wins");
    let err = backend
        .create(&request)
        .await
        .expect_err("the second create must not reuse the worktree");
    assert!(matches!(err, Error::Validation(_)), "{err:?}");
    assert!(err.to_string().contains("refusing to reuse it"), "{err}");
}

#[tokio::test]
async fn concurrent_creates_have_exactly_one_winner() {
    let repo = TestRepository::init().await;
    let backend = Arc::new(GitWorktreeBackend::new());
    let request = repo.create_request("agent-my-team-dev-1-race");

    let (first, second) = tokio::join!(backend.create(&request), backend.create(&request),);
    let successes = usize::from(first.is_ok()) + usize::from(second.is_ok());
    assert_eq!(
        successes, 1,
        "exactly one create may win: {first:?} / {second:?}"
    );
    let loser = if first.is_ok() { second } else { first };
    let err = loser.expect_err("the loser reports an error");
    assert!(matches!(err, Error::Validation(_)), "{err:?}");

    let entries = backend.list(repo.root()).await.unwrap();
    assert_eq!(entries.len(), 1, "only one worktree was registered");
    let branches = repo.branches().await;
    assert_eq!(
        branches
            .iter()
            .filter(|name| name.starts_with("worktree-"))
            .count(),
        1,
        "{branches:?}"
    );
}

#[tokio::test]
async fn create_outside_a_repository_fails_with_config() {
    let temp = tempfile::TempDir::new().unwrap();
    let backend = GitWorktreeBackend::new();
    let request = WorktreeCreateRequest {
        team_name: "my-team".to_owned(),
        session_id: "session-1".to_owned(),
        member_name: "dev-1".to_owned(),
        name: "agent-my-team-dev-1-abcdef1234".to_owned(),
        path: temp.path().join("worktrees/agent-my-team-dev-1-abcdef1234"),
        base_ref: None,
    };
    let err = backend.create(&request).await.expect_err("no repository");
    assert!(matches!(err, Error::Config(_)), "{err:?}");
    assert!(
        err.to_string().contains("requires a git repository"),
        "{err}"
    );
    assert!(err.to_string().contains("git"), "{err}");
}

#[tokio::test]
async fn create_validates_the_request() {
    let repo = TestRepository::init().await;
    let backend = GitWorktreeBackend::new();

    let mut request = repo.create_request("agent-my-team-dev-1-invalid");
    request.name = "   ".to_owned();
    let err = backend.create(&request).await.expect_err("empty name");
    assert!(matches!(err, Error::Validation(_)), "{err:?}");

    let mut request = repo.create_request("agent-my-team-dev-1-invalid");
    request.path = std::path::PathBuf::from("relative/worktree");
    let err = backend.create(&request).await.expect_err("relative path");
    assert!(matches!(err, Error::Validation(_)), "{err:?}");
    assert!(err.to_string().contains("must be absolute"), "{err}");
}

#[tokio::test]
async fn create_uses_the_requested_base_ref() {
    let repo = TestRepository::init().await;
    let first_commit = repo.head(repo.root()).await.expect("head");
    repo.commit("second.txt", "second\n").await;
    let backend = GitWorktreeBackend::new();

    let mut request = repo.create_request("agent-my-team-dev-1-based");
    request.base_ref = Some(first_commit.clone());
    let info = backend
        .create(&request)
        .await
        .expect("created on the base ref");
    assert_eq!(
        repo.head(&info.path).await.as_deref(),
        Some(first_commit.as_str()),
        "the worktree starts on the requested commit"
    );
}

#[tokio::test]
async fn remove_of_an_unknown_worktree_is_not_found() {
    let repo = TestRepository::init().await;
    let backend = GitWorktreeBackend::new();
    let info = super::worktree::WorktreeInfo {
        name: "agent-my-team-dev-1-ghost".to_owned(),
        path: repo.worktree_path("agent-my-team-dev-1-ghost"),
        branch: worktree_branch("agent-my-team-dev-1-ghost"),
    };
    let err = backend.remove(&info).await.expect_err("not registered");
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
    assert!(err.to_string().contains("already removed"), "{err}");
}

#[tokio::test]
async fn pinned_repository_wins_over_discovery() {
    let repo = TestRepository::init().await;
    let other = TestRepository::init().await;
    let backend = GitWorktreeBackend::new().with_repo_root(repo.root());
    assert_eq!(backend.repo_root(), Some(repo.root()));

    // The request path lives in the *other* repository tree, but the pinned root decides.
    let request = other.create_request("agent-my-team-dev-1-pinned");
    let info = backend
        .create(&request)
        .await
        .expect("created in the pinned repo");
    let listed = backend.list(repo.root()).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].path, info.path);
    assert!(other.branches().await.iter().all(|name| name == "main"));
}

#[test]
fn teammate_list_filter_keeps_only_this_crates_worktrees() {
    let entries = super::git::parse_worktree_list(
        "\
worktree /repo
HEAD 1111111111111111111111111111111111111111
branch refs/heads/main

worktree /repo/hand-made
HEAD 2222222222222222222222222222222222222222
branch refs/heads/feature

worktree /repo/.agent_teams/worktrees/agent-team-dev-1
HEAD 3333333333333333333333333333333333333333
branch refs/heads/worktree-agent-team-dev-1

worktree /repo/.agent_teams/worktrees/agent-detached
HEAD 4444444444444444444444444444444444444444
detached
",
    );
    let worktrees = teammate_worktrees(entries);
    assert_eq!(worktrees.len(), 1);
    assert_eq!(worktrees[0].name, "agent-team-dev-1");
    assert_eq!(worktrees[0].branch, "worktree-agent-team-dev-1");
}

#[tokio::test]
async fn list_ignores_manual_worktrees() {
    let repo = TestRepository::init().await;
    let manual = repo.root().join("manual");
    git_ok(
        repo.root(),
        &[
            "worktree",
            "add",
            "-b",
            "feature",
            manual.to_str().unwrap(),
            "HEAD",
        ],
    )
    .await;
    let backend = GitWorktreeBackend::new();
    assert!(
        backend.list(repo.root()).await.unwrap().is_empty(),
        "a hand-made worktree is not a teammate worktree"
    );
}

#[tokio::test]
async fn provider_allocates_real_git_worktrees() {
    let repo = TestRepository::init().await;
    let provider = LocalWorkspaceProvider::new(MemberWorkspaceLayout::new(repo.teams_home()));
    let mut request = WorkspaceRequest::new("my-team", "session-1", "dev-1");
    request.isolation = IsolationMode::Worktree;

    let workspace = provider
        .allocate(&request)
        .await
        .expect("worktree allocated");
    assert_eq!(workspace.isolation, IsolationMode::Worktree);
    assert!(!workspace.is_shared());
    let info = workspace.worktree.as_ref().expect("worktree info");
    assert_eq!(workspace.root, info.path);
    assert!(info.path.is_dir());
    assert!(info.path.starts_with(
        repo.teams_home()
            .join("my-team/sessions/session-1/worktrees")
    ));
    assert_eq!(info.branch, worktree_branch(&info.name));
    assert!(repo.has_branch(&info.branch).await);

    // A second member gets its own worktree, not the first one.
    let mut other = WorkspaceRequest::new("my-team", "session-1", "dev-2");
    other.isolation = IsolationMode::Worktree;
    let second = provider.allocate(&other).await.expect("second worktree");
    assert_ne!(second.root, workspace.root);

    // Releasing removes the worktree and its branch.
    provider.release(&workspace).await.expect("released");
    assert!(!info.path.exists());
    assert!(!repo.has_branch(&info.branch).await);
    assert!(second.root.is_dir(), "the other member keeps its worktree");
    provider.release(&second).await.expect("released");
}

#[tokio::test]
async fn repository_lock_file_is_shared_by_both_backends() {
    let repo = TestRepository::init().await;
    let backend = GitWorktreeBackend::new();
    let request = repo.create_request("agent-my-team-dev-1-lock-path");
    backend.create(&request).await.unwrap();
    let lock_file = repo.root().join(".git").join(super::lock::LOCK_FILE_NAME);
    assert!(
        !lock_file.exists(),
        "the lock file is removed once the critical section ends"
    );
    let output = run_git(repo.root(), &["worktree", "list", "--porcelain"])
        .await
        .unwrap();
    assert_eq!(
        output.stdout.matches("worktree ").count(),
        2,
        "the main checkout plus one teammate worktree: {}",
        output.stdout
    );
}
