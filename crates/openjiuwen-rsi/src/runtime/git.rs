//! Git operations over a [`CommandRunner`] seam.
//!
//! Port of `harness_rsi/auto_harness/infra/git_operations.py` (v0.1.20) and the
//! `_slugify` helper in `worktree_manager.py`, minus the `GitCode` HTTP API: this
//! layer only runs `git` and parses its output, so it is testable with
//! [`crate::runtime::command::FakeCommandRunner`] and no real repository.
//!
//! Unlike the Python helper, a failed `git` invocation is never an error: every
//! method returns a typed result whose `success` flag carries the exit status,
//! exactly like the scripts that call it. Only a failure to run git at all
//! (missing executable, cancellation) is an [`Err`].
//!
//! Interactive credential prompts are disabled in every child
//! (`GIT_TERMINAL_PROMPT=0`, an empty `GIT_ASKPASS`, `GCM_INTERACTIVE=never`),
//! and standard error is merged into standard output, matching Python's
//! `stderr=STDOUT`.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use regex::Regex;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Result;

use super::command::{CommandOutcome, CommandRunner, CommandSpec};

/// The default `git` program name.
const DEFAULT_PROGRAM: &str = "git";
/// The default branch compared against (`origin/<base>`), matching Python.
const DEFAULT_BASE_BRANCH: &str = "develop";
/// The default remote pushed to.
const DEFAULT_REMOTE: &str = "origin";

/// Outcome of one git step, mirroring Python's `{"success", "output"}` dict.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GitStepResult {
    /// Whether the command exited with code `0`.
    pub success: bool,
    /// Combined output of the command (standard error included).
    pub output: String,
}

/// Outcome of a branch-creating command.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GitBranchResult {
    /// Whether the branch was created and checked out.
    pub success: bool,
    /// The requested branch name, when one was known.
    pub branch: Option<String>,
    /// Combined output of the command.
    pub output: String,
}

/// Structured `git status --porcelain` view.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GitStatus {
    /// Union of tracked modifications and untracked files, sorted.
    pub dirty_files: Vec<String>,
    /// Added/modified/deleted/renamed tracked paths, sorted.
    pub tracked_modified_files: Vec<String>,
    /// Untracked paths (status `??`), sorted.
    pub untracked_files: Vec<String>,
    /// New paths of rename entries, sorted.
    pub renamed_files: Vec<String>,
}

/// Outcome of cherry-picking a reference's commits.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GitCherryPick {
    /// Whether every picked commit applied cleanly.
    pub success: bool,
    /// Combined output of the pick and, on failure, the abort.
    pub output: String,
    /// Commits considered for the pick (best effort).
    pub commits: Vec<String>,
}

/// Git command helpers bound to one workspace.
///
/// Every command runs through the injected [`CommandRunner`] in `workspace`
/// with the configured environment, so a test can script the replies and assert
/// the exact invocations.
#[derive(Clone)]
pub struct GitOperations {
    runner: Arc<dyn CommandRunner>,
    workspace: PathBuf,
    program: String,
    base_branch: String,
    env: BTreeMap<String, String>,
    remote: String,
    timeout: Option<Duration>,
}

impl GitOperations {
    /// Creates operations for `workspace` using `runner`.
    ///
    /// The default environment disables interactive credential prompts and the
    /// default base branch is `develop`.
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>, workspace: impl Into<PathBuf>) -> Self {
        let mut env = BTreeMap::new();
        env.insert("GIT_TERMINAL_PROMPT".to_owned(), "0".to_owned());
        env.insert("GIT_ASKPASS".to_owned(), String::new());
        env.insert("GCM_INTERACTIVE".to_owned(), "never".to_owned());
        Self {
            runner,
            workspace: workspace.into(),
            program: DEFAULT_PROGRAM.to_owned(),
            base_branch: DEFAULT_BASE_BRANCH.to_owned(),
            env,
            remote: DEFAULT_REMOTE.to_owned(),
            timeout: None,
        }
    }

    /// Sets the branch compared against as `origin/<base>`.
    #[must_use]
    pub fn with_base_branch(mut self, base_branch: impl Into<String>) -> Self {
        self.base_branch = base_branch.into();
        self
    }

    /// Sets the git executable (default `git`).
    #[must_use]
    pub fn with_program(mut self, program: impl Into<String>) -> Self {
        self.program = program.into();
        self
    }

    /// Sets the push remote (default `origin`).
    #[must_use]
    pub fn with_remote(mut self, remote: impl Into<String>) -> Self {
        self.remote = remote.into();
        self
    }

    /// Adds one environment variable to every git child.
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Sets a deadline for every git child.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// The injected command runner.
    #[must_use]
    pub fn runner(&self) -> &dyn CommandRunner {
        self.runner.as_ref()
    }

    /// The workspace directory git runs in.
    #[must_use]
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// The git executable name.
    #[must_use]
    pub fn program(&self) -> &str {
        &self.program
    }

    /// The base branch compared against.
    #[must_use]
    pub fn base_branch(&self) -> &str {
        &self.base_branch
    }

    /// The push remote.
    #[must_use]
    pub fn remote(&self) -> &str {
        &self.remote
    }

    /// The extra environment applied to every command.
    #[must_use]
    pub fn env(&self) -> &BTreeMap<String, String> {
        &self.env
    }

    /// The per-command deadline, if one was configured.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Runs `git <args>` in the workspace with the configured environment.
    ///
    /// Standard error is merged into [`CommandOutcome::stdout`].
    ///
    /// # Errors
    ///
    /// Propagates [`CommandRunner::run`] failures: a missing git executable is
    /// [`openjiuwen_common::error::Error::Config`], an I/O failure is
    /// [`openjiuwen_common::error::Error::Tool`].
    pub async fn raw(&self, args: &[&str]) -> Result<CommandOutcome> {
        let mut spec = CommandSpec::new(self.program.clone())
            .args(args.iter().copied())
            .cwd(self.workspace.clone())
            .merge_stderr();
        for (key, value) in &self.env {
            spec = spec.env(key.clone(), value.clone());
        }
        if let Some(timeout) = self.timeout {
            spec = spec.timeout(timeout);
        }
        self.runner.run(spec).await
    }

    /// Creates and checks out `name` (`git checkout -b`).
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn create_branch(&self, name: &str) -> Result<GitBranchResult> {
        let outcome = self.raw(&["checkout", "-b", name]).await?;
        Ok(GitBranchResult {
            success: outcome.success(),
            branch: Some(name.to_owned()),
            output: outcome.stdout,
        })
    }

    /// Returns the current branch name, or `None` when git fails or HEAD is
    /// detached.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn current_branch(&self) -> Result<Option<String>> {
        let outcome = self.raw(&["rev-parse", "--abbrev-ref", "HEAD"]).await?;
        if !outcome.success() {
            return Ok(None);
        }
        let branch = outcome.stdout.trim();
        Ok(if branch.is_empty() {
            None
        } else {
            Some(branch.to_owned())
        })
    }

    /// Returns the current HEAD commit sha, or `None` when git fails.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn current_head(&self) -> Result<Option<String>> {
        let outcome = self.raw(&["rev-parse", "HEAD"]).await?;
        if !outcome.success() {
            return Ok(None);
        }
        let head = outcome.stdout.trim();
        Ok(if head.is_empty() {
            None
        } else {
            Some(head.to_owned())
        })
    }

    /// Collects `git status --porcelain --untracked-files=all` into buckets.
    ///
    /// Rename entries (`old -> new`) record the new path, backslashes are
    /// normalised to `/`, and every bucket is de-duplicated and sorted. A git
    /// failure yields an empty status rather than an error, matching Python.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn collect_status(&self) -> Result<GitStatus> {
        let outcome = self
            .raw(&["status", "--porcelain", "--untracked-files=all"])
            .await?;
        if !outcome.success() || outcome.stdout.is_empty() {
            return Ok(GitStatus::default());
        }
        Ok(parse_status(&outcome.stdout))
    }

    /// Returns the currently dirty files (tracked modifications plus untracked).
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::collect_status`].
    pub async fn list_dirty_files(&self) -> Result<Vec<String>> {
        Ok(self.collect_status().await?.dirty_files)
    }

    /// Returns the trimmed `git diff --stat` summary.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn diff_stat(&self) -> Result<String> {
        let outcome = self.raw(&["diff", "--stat"]).await?;
        Ok(outcome.stdout.trim().to_owned())
    }

    /// Returns the trimmed diff stat of `origin/<base>...HEAD`.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn diff_stat_against_base(&self) -> Result<String> {
        let reference = self.against_base_revision();
        let outcome = self.raw(&["diff", "--stat", &reference]).await?;
        Ok(outcome.stdout.trim().to_owned())
    }

    /// Returns the files changed by `revision` (`git diff --name-only`).
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn diff_name_only(&self, revision: &str) -> Result<Vec<String>> {
        let outcome = self.raw(&["diff", "--name-only", revision]).await?;
        Ok(parse_name_only(&outcome.stdout))
    }

    /// Returns the files changed by `origin/<base>...HEAD`.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn diff_name_only_against_base(&self) -> Result<Vec<String>> {
        let reference = self.against_base_revision();
        let outcome = self.raw(&["diff", "--name-only", &reference]).await?;
        Ok(parse_name_only(&outcome.stdout))
    }

    /// Returns the full diff of `origin/<base>...HEAD`, untrimmed.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn diff_against_base(&self) -> Result<String> {
        let reference = self.against_base_revision();
        Ok(self.raw(&["diff", &reference]).await?.stdout)
    }

    /// Returns the raw, right-trimmed `git status --porcelain` output.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn status_porcelain(&self) -> Result<String> {
        let outcome = self
            .raw(&["status", "--porcelain", "--untracked-files=all"])
            .await?;
        Ok(outcome.stdout.trim_end().to_owned())
    }

    /// Returns the trimmed summary of the last commit.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn show_last_commit_stat(&self) -> Result<String> {
        let outcome = self
            .raw(&["show", "--stat", "--format=fuller", "-1"])
            .await?;
        Ok(outcome.stdout.trim().to_owned())
    }

    /// Whether HEAD has commits not reachable from `origin/<base>`.
    ///
    /// A git failure or an unparseable count is `false`, matching Python.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn has_commits_against_base(&self) -> Result<bool> {
        let range = format!("origin/{}..HEAD", self.base_branch);
        let outcome = self.raw(&["rev-list", "--count", &range]).await?;
        if !outcome.success() {
            return Ok(false);
        }
        Ok(outcome
            .stdout
            .trim()
            .parse::<u64>()
            .is_ok_and(|count| count > 0))
    }

    /// Stages `paths` (`git add --`).
    ///
    /// An empty slice is rejected without running git, mirroring Python.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn add_paths(&self, paths: &[String]) -> Result<GitStepResult> {
        if paths.is_empty() {
            return Ok(GitStepResult {
                success: false,
                output: "no paths provided".to_owned(),
            });
        }
        let mut args: Vec<&str> = vec!["add", "--"];
        args.extend(paths.iter().map(String::as_str));
        let outcome = self.raw(&args).await?;
        Ok(GitStepResult {
            success: outcome.success(),
            output: outcome.stdout,
        })
    }

    /// Creates a commit with `message`.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn commit(&self, message: &str) -> Result<GitStepResult> {
        let outcome = self.raw(&["commit", "-m", message]).await?;
        Ok(GitStepResult {
            success: outcome.success(),
            output: outcome.stdout,
        })
    }

    /// Discards worktree changes with `git checkout .`.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn discard_worktree_changes(&self) -> Result<bool> {
        Ok(self.raw(&["checkout", "."]).await?.success())
    }

    /// Adds a worktree at `path` on a new `branch` from `from_ref`.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn worktree_add(
        &self,
        path: &Path,
        branch: &str,
        from_ref: &str,
    ) -> Result<GitStepResult> {
        let path = path.to_string_lossy();
        let outcome = self
            .raw(&["worktree", "add", "-b", branch, &path, from_ref])
            .await?;
        Ok(GitStepResult {
            success: outcome.success(),
            output: outcome.stdout,
        })
    }

    /// Force-removes the worktree at `path`.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn worktree_remove(&self, path: &Path) -> Result<GitStepResult> {
        let path = path.to_string_lossy();
        let outcome = self.raw(&["worktree", "remove", "--force", &path]).await?;
        Ok(GitStepResult {
            success: outcome.success(),
            output: outcome.stdout,
        })
    }

    /// Prunes stale worktree metadata.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn worktree_prune(&self) -> Result<GitStepResult> {
        let outcome = self.raw(&["worktree", "prune"]).await?;
        Ok(GitStepResult {
            success: outcome.success(),
            output: outcome.stdout,
        })
    }

    /// Pushes `branch` to the configured remote with escalating strategies.
    ///
    /// Attempts, in order: `--force-with-lease`, a plain push, then an explicit
    /// `HEAD:refs/heads/<branch>` refspec. The first success stops the ladder.
    /// The returned output lists every attempt as
    /// `[<n>] cmd=<command> code=<exit>` followed by its output.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn push(&self, branch: &str) -> Result<GitStepResult> {
        let attempts = [
            vec![
                "push".to_owned(),
                "--force-with-lease".to_owned(),
                self.remote.clone(),
                branch.to_owned(),
            ],
            vec!["push".to_owned(), self.remote.clone(), branch.to_owned()],
            vec![
                "push".to_owned(),
                self.remote.clone(),
                format!("HEAD:refs/heads/{branch}"),
            ],
        ];
        let mut chunks = Vec::with_capacity(attempts.len());
        for (index, attempt) in attempts.iter().enumerate() {
            let args: Vec<&str> = attempt.iter().map(String::as_str).collect();
            let command_line = CommandSpec::new(self.program.clone())
                .args(args.iter().copied())
                .command_line();
            let outcome = self.raw(&args).await?;
            chunks.push(format!(
                "[{}] cmd={command_line} code={}\n{}",
                index + 1,
                exit_label(outcome.code),
                outcome.stdout
            ));
            if outcome.success() {
                return Ok(GitStepResult {
                    success: true,
                    output: chunks.join("\n\n"),
                });
            }
        }
        Ok(GitStepResult {
            success: false,
            output: chunks.join("\n\n"),
        })
    }

    /// Cherry-picks the commits in `origin/<base>..reference` onto HEAD.
    ///
    /// On failure the pick is aborted and the abort output is appended to the
    /// failure output. The commit list is parsed best effort beforehand, so it
    /// is present on both outcomes.
    ///
    /// # Errors
    ///
    /// Propagates command-runner failures from [`GitOperations::raw`].
    pub async fn cherry_pick_ref_commits(&self, reference: &str) -> Result<GitCherryPick> {
        let commits = self.changed_commits(reference).await;
        let range = format!("origin/{}..{reference}", self.base_branch);
        let outcome = self.raw(&["cherry-pick", &range]).await?;
        if outcome.success() {
            return Ok(GitCherryPick {
                success: true,
                output: outcome.stdout,
                commits,
            });
        }
        let abort = self.raw(&["cherry-pick", "--abort"]).await?;
        let output = if abort.stdout.is_empty() {
            outcome.stdout
        } else {
            format!("{}\n{}", outcome.stdout, abort.stdout)
        };
        Ok(GitCherryPick {
            success: false,
            output,
            commits,
        })
    }

    /// The `origin/<base>...HEAD` revision spec used by diff helpers.
    fn against_base_revision(&self) -> String {
        format!("origin/{}...HEAD", self.base_branch)
    }

    /// Reads the commits of `origin/<base>..reference`, best effort.
    async fn changed_commits(&self, reference: &str) -> Vec<String> {
        let range = format!("origin/{}..{reference}", self.base_branch);
        match self.raw(&["rev-list", "--reverse", &range]).await {
            Ok(outcome) if outcome.success() => parse_lines(&outcome.stdout),
            Ok(_) => Vec::new(),
            Err(error) => {
                tracing::debug!(%error, "could not list cherry-pick commits; continuing");
                Vec::new()
            }
        }
    }
}

/// Renders an exit code for the push attempt log.
fn exit_label(code: Option<i32>) -> String {
    code.map_or_else(|| "none".to_owned(), |code| code.to_string())
}

/// Normalises one porcelain path: `/` separators and no surrounding blanks.
fn normalize_path(path: &str) -> String {
    path.trim().replace('\\', "/")
}

/// Parses porcelain status lines into [`GitStatus`].
fn parse_status(output: &str) -> GitStatus {
    let mut tracked = Vec::new();
    let mut untracked = Vec::new();
    let mut renamed = Vec::new();
    for line in output.lines() {
        if line.chars().count() < 4 {
            continue;
        }
        let marker: String = line.chars().take(2).collect();
        let raw_path: String = line.chars().skip(3).collect();
        let path = raw_path.trim();
        if path.is_empty() {
            continue;
        }
        let (path, rename_target) = match path.split_once(" -> ") {
            Some((_, new_path)) => (new_path.trim(), Some(new_path.trim())),
            None => (path, None),
        };
        if path.is_empty() {
            continue;
        }
        if let Some(target) = rename_target {
            renamed.push(normalize_path(target));
        }
        let normalized = normalize_path(path);
        if marker == "??" {
            untracked.push(normalized);
        } else {
            tracked.push(normalized);
        }
    }
    let tracked_modified_files = sorted_unique(tracked);
    let untracked_files = sorted_unique(untracked);
    let renamed_files = sorted_unique(renamed);
    let mut dirty: Vec<String> = tracked_modified_files
        .iter()
        .chain(untracked_files.iter())
        .cloned()
        .collect();
    dirty.sort();
    dirty.dedup();
    GitStatus {
        dirty_files: dirty,
        tracked_modified_files,
        untracked_files,
        renamed_files,
    }
}

/// Sorts and de-duplicates one status bucket.
fn sorted_unique(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values.dedup();
    values
}

/// Parses `--name-only` output into normalised, de-duplicated paths.
fn parse_name_only(output: &str) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for line in output.lines() {
        let normalized = normalize_path(line);
        if !normalized.is_empty() && !files.contains(&normalized) {
            files.push(normalized);
        }
    }
    files
}

/// Parses non-empty trimmed lines.
fn parse_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Converts a task topic into a filesystem-safe branch/worktree slug.
///
/// A topic naming an issue becomes `fix-issue-<n>` (matching Python's
/// `_slugify`); otherwise non-alphanumeric runs collapse to `-`, the result is
/// lowercased, trimmed, and truncated to 40 characters, with an empty result
/// becoming `task`.
#[must_use]
pub fn slugify_topic(topic: &str) -> String {
    if let Some(regex) = issue_regex() {
        if let Some(captures) = regex.captures(topic) {
            if let Some(number) = captures.get(1) {
                return format!("fix-issue-{}", number.as_str());
            }
        }
    }
    let mut slug = String::new();
    let mut pending_dash = false;
    for ch in topic.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.push(ch);
        } else {
            pending_dash = true;
        }
    }
    if slug.is_empty() {
        return "task".to_owned();
    }
    let slug: String = slug.to_lowercase().chars().take(40).collect();
    if slug.is_empty() {
        "task".to_owned()
    } else {
        slug
    }
}

/// The issue-number pattern, compiled once.
fn issue_regex() -> Option<&'static Regex> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    PATTERN
        .get_or_init(|| Regex::new(r"(?i)(?:fix-issue-|issue-|\bissue\s*#?|#)\s*(\d+)").ok())
        .as_ref()
}

/// Builds the subprocess environment for non-interactive `GitCode` auth.
///
/// Always disables terminal/GCM prompts. When both `user` and `token` are
/// present it also injects git-config environment entries: no credential
/// helper, a basic `AUTHORIZATION` extraheader scoped to
/// `https://gitcode.com`, HTTP/1.1, and low-speed limits — the same six entries
/// as Python's `build_git_auth_env`.
#[must_use]
pub fn build_auth_env(user: &str, token: &str) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.insert("GIT_TERMINAL_PROMPT".to_owned(), "0".to_owned());
    env.insert("GCM_INTERACTIVE".to_owned(), "never".to_owned());
    if user.is_empty() || token.is_empty() {
        return env;
    }
    let basic = STANDARD.encode(format!("{user}:{token}"));
    env.insert("GIT_CONFIG_COUNT".to_owned(), "6".to_owned());
    env.insert(
        "GIT_CONFIG_KEY_0".to_owned(),
        "credential.helper".to_owned(),
    );
    env.insert("GIT_CONFIG_VALUE_0".to_owned(), String::new());
    env.insert(
        "GIT_CONFIG_KEY_1".to_owned(),
        "credential.interactive".to_owned(),
    );
    env.insert("GIT_CONFIG_VALUE_1".to_owned(), "never".to_owned());
    env.insert(
        "GIT_CONFIG_KEY_2".to_owned(),
        "http.https://gitcode.com/.extraheader".to_owned(),
    );
    env.insert(
        "GIT_CONFIG_VALUE_2".to_owned(),
        format!("AUTHORIZATION: basic {basic}"),
    );
    env.insert("GIT_CONFIG_KEY_3".to_owned(), "http.version".to_owned());
    env.insert("GIT_CONFIG_VALUE_3".to_owned(), "HTTP/1.1".to_owned());
    env.insert(
        "GIT_CONFIG_KEY_4".to_owned(),
        "http.lowSpeedLimit".to_owned(),
    );
    env.insert("GIT_CONFIG_VALUE_4".to_owned(), "1".to_owned());
    env.insert(
        "GIT_CONFIG_KEY_5".to_owned(),
        "http.lowSpeedTime".to_owned(),
    );
    env.insert("GIT_CONFIG_VALUE_5".to_owned(), "60".to_owned());
    env
}

/// Removes credentials from git output before it reaches logs or errors.
///
/// Replaces the raw token and its `user:token` basic-auth base64 form with
/// `***`, rewrites `user:anything@` in URLs to `***:***@`, and redacts values of
/// `Authorization`, `private-token`, and `access_token` headers/parameters
/// (case-insensitive). Mirrors Python's `_sanitize_git_output`.
#[must_use]
pub fn scrub_credentials(text: &str, user: &str, token: &str) -> String {
    let mut sanitized = text.to_owned();
    if !token.is_empty() {
        sanitized = sanitized.replace(token, "***");
        if !user.is_empty() {
            let basic = STANDARD.encode(format!("{user}:{token}"));
            sanitized = sanitized.replace(&basic, "***");
        }
    }
    if !user.is_empty() {
        let pattern = format!(r"{}:[^@\s]+@", regex::escape(user));
        if let Ok(regex) = Regex::new(&pattern) {
            sanitized = regex.replace_all(&sanitized, "***:***@").into_owned();
        }
        // After the token replacement a URL can read `user:***@`.
        sanitized = sanitized.replace(&format!("{user}:***@"), "***:***@");
    }
    if let Some(regex) = header_regex() {
        sanitized = regex.replace_all(&sanitized, "${1}${2}***").into_owned();
    }
    sanitized
}

/// The credential-header pattern, compiled once.
fn header_regex() -> Option<&'static Regex> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    PATTERN
        .get_or_init(|| {
            Regex::new(r"(?i)(authorization|private-token|access_token)([=:]\s*)[^\s&]+").ok()
        })
        .as_ref()
}
