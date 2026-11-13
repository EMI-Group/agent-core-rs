//! Shared fixture for the workspace backend tests: a throwaway git repository.
//!
//! Every test that needs a repository gets its own `TempDir`, so tests never share `.git` state and
//! run in parallel safely. The fixture commits one file with a local `user.name`/`user.email`, so
//! `git` never reads a developer's global configuration and never prompts.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::git::run_git;
use super::worktree::WorktreeCreateRequest;

/// A temporary git repository with one commit.
pub(super) struct TestRepository {
    /// Keeps the temporary directory alive for the lifetime of the fixture.
    _temp: TempDir,
    /// Repository root (`<temp>/repo`).
    root: PathBuf,
}

impl TestRepository {
    /// Initializes the repository, configures an identity and commits one file.
    pub(super) async fn init() -> Self {
        let temp = TempDir::new().expect("temp dir");
        let root = temp.path().join("repo");
        std::fs::create_dir_all(root.join(".agent_teams")).expect("repo dir");
        git_ok(&root, &["init", "--quiet", "--initial-branch=main"]).await;
        git_ok(&root, &["config", "user.email", "teammate@example.invalid"]).await;
        git_ok(&root, &["config", "user.name", "openJiuwen Tests"]).await;
        std::fs::write(root.join("README.md"), "# fixture\n").expect("write readme");
        git_ok(&root, &["add", "README.md"]).await;
        git_ok(&root, &["commit", "--quiet", "-m", "init"]).await;
        Self { _temp: temp, root }
    }

    /// Returns the repository root.
    pub(super) fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the agent-teams tree used by the workspace layout (`<repo>/.agent_teams`).
    pub(super) fn teams_home(&self) -> PathBuf {
        self.root.join(".agent_teams")
    }

    /// Builds a create request for `name` under `<repo>/.agent_teams/worktrees/<name>`.
    pub(super) fn create_request(&self, name: &str) -> WorktreeCreateRequest {
        WorktreeCreateRequest {
            team_name: "my-team".to_owned(),
            session_id: "session-1".to_owned(),
            member_name: "dev-1".to_owned(),
            name: name.to_owned(),
            path: self.worktree_path(name),
            base_ref: None,
        }
    }

    /// Returns the target directory a [`Self::create_request`] uses.
    pub(super) fn worktree_path(&self, name: &str) -> PathBuf {
        self.teams_home().join("worktrees").join(name)
    }

    /// Commits `content` to `file_name`, returning the new commit id.
    pub(super) async fn commit(&self, file_name: &str, content: &str) -> String {
        std::fs::write(self.root.join(file_name), content).expect("write file");
        git_ok(&self.root, &["add", file_name]).await;
        git_ok(&self.root, &["commit", "--quiet", "-m", "update"]).await;
        self.head(&self.root).await.expect("head commit")
    }

    /// Returns the `HEAD` commit id of `cwd`.
    pub(super) async fn head(&self, cwd: &Path) -> Option<String> {
        let output = run_git(cwd, &["rev-parse", "HEAD"])
            .await
            .expect("git runs");
        output.succeeded().then(|| output.stdout.clone())
    }

    /// Returns the local branches of the repository, sorted.
    pub(super) async fn branches(&self) -> Vec<String> {
        let output = run_git(&self.root, &["branch", "--format=%(refname:short)"])
            .await
            .expect("git runs");
        let mut branches: Vec<String> = output
            .stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect();
        branches.sort();
        branches
    }

    /// Returns whether `branch` is a local branch of the repository.
    pub(super) async fn has_branch(&self, branch: &str) -> bool {
        self.branches().await.iter().any(|name| name == branch)
    }
}

/// Runs `git <args>` in `cwd` and asserts success.
pub(super) async fn git_ok(cwd: &Path, args: &[&str]) {
    let output = run_git(cwd, args).await.expect("git is available");
    assert!(
        output.succeeded(),
        "git {args:?} failed in {}: {}",
        cwd.display(),
        output.stderr
    );
}
