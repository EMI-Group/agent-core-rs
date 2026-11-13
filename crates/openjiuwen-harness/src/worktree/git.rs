//! Git CLI wrapper: the [`GitRunner`] seam, the real process runner, and the
//! worktree/status command helpers.
//!
//! Port of `openjiuwen/harness/tools/worktree/git.py`. Pure async subprocess
//! plumbing with no business logic: every helper takes the runner and the
//! directory to run in, and fails fast. The runner is a seam, so tests drive the
//! manager with scripted output instead of a real repository.
//!
//! Two failures are typed from the caller's point of view:
//!
//! - git is not installed → [`Error::config`] with [`NOT_INSTALLED_MARKER`],
//!   checked through `which::which`;
//! - the directory is not inside a repository → [`Error::config`] with
//!   [`NOT_A_REPOSITORY_MARKER`].
//!
//! Security: interactive credential prompts are suppressed
//! (`GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS=""`) and stdin is `/dev/null`, so a git
//! command can never hang waiting for input.
use std::path::{Path, PathBuf};
use std::process::Stdio;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use openjiuwen_common::error::{Error, Result};

use super::models::WorktreeInfo;

/// Name of the git executable.
pub const GIT_BINARY: &str = "git";

/// Substring of the [`Error::config`] message used when git is not installed.
pub const NOT_INSTALLED_MARKER: &str = "git is not installed";

/// Substring of the [`Error::config`] message used outside a repository.
pub const NOT_A_REPOSITORY_MARKER: &str = "not a git repository";

/// Whether a `git` executable is on `PATH`.
#[must_use]
pub fn git_available() -> bool {
    which::which(GIT_BINARY).is_ok()
}

/// Builds the typed "git is not installed" error for `program`.
pub fn not_installed_error(program: &str) -> Error {
    Error::config(format!(
        "{NOT_INSTALLED_MARKER}: `{program}` was not found on PATH; install git to use worktrees"
    ))
}

/// Builds the typed "not a git repository" error for `cwd`.
pub fn not_a_repository_error(cwd: &Path) -> Error {
    Error::config(format!(
        "{NOT_A_REPOSITORY_MARKER}: `{}` is not inside a git repository",
        cwd.display()
    ))
}

/// Builds the typed error of a failed git invocation.
pub fn git_failure_error(args: &[&str], stderr: &str) -> Error {
    Error::tool(format!("git {} failed: {}", args.join(" "), stderr.trim()))
}

/// Whether `error` reports that git is not installed.
#[must_use]
pub fn is_not_installed(error: &Error) -> bool {
    error.to_string().contains(NOT_INSTALLED_MARKER)
}

/// Whether `error` reports that the directory is outside a repository.
#[must_use]
pub fn is_not_a_repository(error: &Error) -> bool {
    error.to_string().contains(NOT_A_REPOSITORY_MARKER)
}

/// Captured output of a successful git invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GitOutput {
    /// Captured standard output, with the trailing newline trimmed.
    pub stdout: String,
    /// Captured standard error, with the trailing newline trimmed.
    pub stderr: String,
}

impl GitOutput {
    /// Builds an output with `stdout` and no standard error.
    #[must_use]
    pub fn new(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: String::new(),
        }
    }
}

/// Runs git commands.
///
/// `run` resolves to the captured output on exit code `0` and to a typed error
/// (carrying the git standard error) otherwise; there is no "failed but
/// successful" value. Implementations must not prompt — see the module docs.
#[async_trait]
pub trait GitRunner: Send + Sync {
    /// Runs `git <args>` in `cwd`, capturing its output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when git is not installed and
    /// [`Error::tool`] when the command exits non-zero.
    async fn run(&self, args: &[&str], cwd: &Path) -> Result<GitOutput>;
}

/// The real [`GitRunner`], spawning `git` through `tokio::process`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessGitRunner {
    program: String,
}

impl Default for ProcessGitRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessGitRunner {
    /// Builds a runner for the `git` executable.
    #[must_use]
    pub fn new() -> Self {
        Self {
            program: GIT_BINARY.to_owned(),
        }
    }

    /// Builds a runner for another executable, mainly to test the
    /// "git is not installed" path with a program that cannot exist.
    #[must_use]
    pub fn with_program(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
        }
    }

    /// The executable this runner spawns.
    #[must_use]
    pub fn program(&self) -> &str {
        &self.program
    }
}

#[async_trait]
impl GitRunner for ProcessGitRunner {
    async fn run(&self, args: &[&str], cwd: &Path) -> Result<GitOutput> {
        if which::which(&self.program).is_err() {
            return Err(not_installed_error(&self.program));
        }
        let output = Command::new(&self.program)
            .args(args)
            .current_dir(cwd)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ASKPASS", "")
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    not_installed_error(&self.program)
                } else {
                    Error::tool(format!(
                        "cannot run `{}` in `{}`: {error}",
                        self.program,
                        cwd.display()
                    ))
                }
            })?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if !output.status.success() {
            return Err(git_failure_error(args, &stderr));
        }
        Ok(GitOutput { stdout, stderr })
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parses `git worktree list --porcelain` output.
///
/// Each blank-line separated block yields one [`WorktreeInfo`]: `worktree
/// <path>`, optional `HEAD <sha>`, optional `branch refs/heads/<name>`, and the
/// flags `bare`, `detached`, `locked [<reason>]`, `prunable <reason>`. Blocks
/// without a `worktree` line are skipped, and unknown lines are ignored so a
/// newer git never breaks parsing.
#[must_use]
pub fn parse_worktree_list(text: &str) -> Vec<WorktreeInfo> {
    let mut entries = Vec::new();
    let mut current: Option<WorktreeInfo> = None;
    let flush = |current: &mut Option<WorktreeInfo>, entries: &mut Vec<WorktreeInfo>| {
        if let Some(info) = current.take() {
            if !info.path.as_os_str().is_empty() {
                entries.push(info);
            }
        }
    };
    for line in text.lines() {
        if line.trim().is_empty() {
            flush(&mut current, &mut entries);
            continue;
        }
        if let Some(path) = line.strip_prefix("worktree ") {
            flush(&mut current, &mut entries);
            current = Some(WorktreeInfo {
                path: PathBuf::from(path.trim()),
                ..WorktreeInfo::default()
            });
            continue;
        }
        let Some(info) = current.as_mut() else {
            continue;
        };
        if let Some(head) = line.strip_prefix("HEAD ") {
            info.head = Some(head.trim().to_owned());
        } else if let Some(branch) = line.strip_prefix("branch ") {
            let branch = branch.trim();
            let short = branch.strip_prefix("refs/heads/").unwrap_or(branch);
            info.branch = Some(short.to_owned());
        } else if line == "bare" {
            info.bare = true;
        } else if line == "detached" {
            info.detached = true;
        } else if let Some(reason) = line.strip_prefix("locked") {
            info.locked = true;
            let reason = reason.trim();
            info.lock_reason = (!reason.is_empty()).then(|| reason.to_owned());
        } else if let Some(reason) = line.strip_prefix("prunable ") {
            info.prunable = Some(reason.trim().to_owned());
        }
    }
    flush(&mut current, &mut entries);
    entries
}

/// Parses `git status --porcelain` output into its changed paths.
///
/// Blank lines are skipped; every other line is one changed path, so the result
/// length is the change count.
#[must_use]
pub fn parse_status_porcelain(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

// ---------------------------------------------------------------------------
// Command helpers
// ---------------------------------------------------------------------------

/// Resolves the repository root that contains `cwd`.
///
/// # Errors
///
/// Returns [`Error::config`] when git is missing or `cwd` is outside a
/// repository.
pub async fn find_git_root(runner: &dyn GitRunner, cwd: &Path) -> Result<PathBuf> {
    let args = ["rev-parse", "--show-toplevel"];
    match runner.run(&args, cwd).await {
        Ok(output) if !output.stdout.trim().is_empty() => Ok(PathBuf::from(output.stdout.trim())),
        Ok(_) => Err(not_a_repository_error(cwd)),
        Err(error) if is_not_a_repository(&error) => Err(not_a_repository_error(cwd)),
        Err(error) => Err(error),
    }
}

/// The checked-out branch of `cwd`, or `None` when HEAD is detached.
///
/// # Errors
///
/// Returns the runner's typed error when git cannot be run.
pub async fn current_branch(runner: &dyn GitRunner, cwd: &Path) -> Result<Option<String>> {
    let output = runner
        .run(&["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        .await?;
    let branch = output.stdout.trim();
    if branch.is_empty() || branch == "HEAD" {
        return Ok(None);
    }
    Ok(Some(branch.to_owned()))
}

/// Resolves `reference` to a commit SHA, or `None` when it does not resolve.
///
/// # Errors
///
/// Returns the runner's typed error when git cannot be run.
pub async fn rev_parse(
    runner: &dyn GitRunner,
    cwd: &Path,
    reference: &str,
) -> Result<Option<String>> {
    match runner.run(&["rev-parse", reference], cwd).await {
        Ok(output) => Ok(Some(output.stdout.trim().to_owned()).filter(|sha| !sha.is_empty())),
        Err(_) => Ok(None),
    }
}

/// Whether the local branch `branch` exists.
///
/// # Errors
///
/// Returns the runner's typed error when git cannot be run.
pub async fn branch_exists(runner: &dyn GitRunner, repo: &Path, branch: &str) -> Result<bool> {
    let reference = format!("refs/heads/{branch}");
    let args = ["rev-parse", "--verify", "--quiet", reference.as_str()];
    match runner.run(&args, repo).await {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == openjiuwen_common::error::ErrorKind::Tool => Ok(false),
        Err(error) => Err(error),
    }
}

/// Lists the repository's worktrees.
///
/// # Errors
///
/// Returns [`Error::config`] outside a repository and the runner's typed error
/// otherwise.
pub async fn worktree_list(runner: &dyn GitRunner, repo: &Path) -> Result<Vec<WorktreeInfo>> {
    let args = ["worktree", "list", "--porcelain"];
    let output = match runner.run(&args, repo).await {
        Ok(output) => output,
        Err(error) if is_not_a_repository(&error) => return Err(not_a_repository_error(repo)),
        Err(error) => return Err(error),
    };
    Ok(parse_worktree_list(&output.stdout))
}

/// Creates a worktree at `path` on a new `branch` based on `base`.
///
/// # Errors
///
/// Returns [`Error::tool`] when git rejects the command (occupied path, existing
/// branch, unknown base).
pub async fn worktree_add(
    runner: &dyn GitRunner,
    repo: &Path,
    path: &Path,
    branch: &str,
    base: &str,
) -> Result<()> {
    let path = path.to_string_lossy().into_owned();
    let args = ["worktree", "add", "-b", branch, path.as_str(), base];
    runner.run(&args, repo).await.map(|_| ())
}

/// Checks out an existing `branch` in a new worktree at `path`.
///
/// Used when the branch already exists: `git worktree add -b` would fail, and
/// re-creating the branch would lose history.
///
/// # Errors
///
/// Returns [`Error::tool`] when git rejects the command.
pub async fn worktree_add_existing(
    runner: &dyn GitRunner,
    repo: &Path,
    path: &Path,
    branch: &str,
) -> Result<()> {
    let path = path.to_string_lossy().into_owned();
    runner
        .run(&["worktree", "add", path.as_str(), branch], repo)
        .await
        .map(|_| ())
}

/// Removes the worktree at `path`, optionally forcing a dirty worktree.
///
/// # Errors
///
/// Returns [`Error::tool`] when git rejects the command.
pub async fn worktree_remove(
    runner: &dyn GitRunner,
    repo: &Path,
    path: &Path,
    force: bool,
) -> Result<()> {
    let path = path.to_string_lossy().into_owned();
    let mut args = vec!["worktree", "remove"];
    if force {
        args.push("--force");
    }
    args.push(path.as_str());
    runner.run(&args, repo).await.map(|_| ())
}

/// Prunes worktree metadata whose directories are gone.
///
/// # Errors
///
/// Returns the runner's typed error.
pub async fn worktree_prune(runner: &dyn GitRunner, repo: &Path) -> Result<()> {
    runner.run(&["worktree", "prune"], repo).await.map(|_| ())
}

/// Locks the worktree at `path`, optionally recording a `reason`.
///
/// # Errors
///
/// Returns [`Error::tool`] when git rejects the command.
pub async fn worktree_lock(
    runner: &dyn GitRunner,
    repo: &Path,
    path: &Path,
    reason: Option<&str>,
) -> Result<()> {
    let path = path.to_string_lossy().into_owned();
    let mut args = vec!["worktree", "lock"];
    if let Some(reason) = reason {
        args.push("--reason");
        args.push(reason);
    }
    args.push(path.as_str());
    runner.run(&args, repo).await.map(|_| ())
}

/// Unlocks the worktree at `path`.
///
/// # Errors
///
/// Returns [`Error::tool`] when git rejects the command.
pub async fn worktree_unlock(runner: &dyn GitRunner, repo: &Path, path: &Path) -> Result<()> {
    let path = path.to_string_lossy().into_owned();
    runner
        .run(&["worktree", "unlock", path.as_str()], repo)
        .await
        .map(|_| ())
}

/// The changed paths of `cwd` as reported by `git status --porcelain`.
///
/// # Errors
///
/// Returns the runner's typed error when git cannot be run.
pub async fn status_porcelain(runner: &dyn GitRunner, cwd: &Path) -> Result<Vec<String>> {
    let output = runner.run(&["status", "--porcelain"], cwd).await?;
    Ok(parse_status_porcelain(&output.stdout))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// Real `git worktree list --porcelain` output: a main worktree, a linked
    /// branch worktree with a lock reason, and a detached prunable one.
    const SAMPLE: &str = "worktree /repo\n\
HEAD 1111111111111111111111111111111111111111\n\
branch refs/heads/main\n\
\n\
worktree /repo/.worktrees/feature\n\
HEAD 2222222222222222222222222222222222222222\n\
branch refs/heads/worktree-feature\n\
locked maintenance window\n\
\n\
worktree /repo/.worktrees/bare\n\
bare\n\
\n\
worktree /repo/.worktrees/detached\n\
HEAD 3333333333333333333333333333333333333333\n\
detached\n\
prunable gitdir file points to non-existent location\n";

    #[test]
    fn worktree_list_parses_every_porcelain_field() {
        let entries = parse_worktree_list(SAMPLE);
        assert_eq!(entries.len(), 4);

        assert_eq!(entries[0].path, PathBuf::from("/repo"));
        assert_eq!(entries[0].branch.as_deref(), Some("main"));
        assert_eq!(
            entries[0].head.as_deref(),
            Some("1111111111111111111111111111111111111111")
        );
        assert!(!entries[0].locked);
        assert!(!entries[0].is_detached());

        assert_eq!(entries[1].path, PathBuf::from("/repo/.worktrees/feature"));
        assert_eq!(entries[1].branch.as_deref(), Some("worktree-feature"));
        assert!(entries[1].locked);
        assert_eq!(
            entries[1].lock_reason.as_deref(),
            Some("maintenance window")
        );

        assert!(entries[2].bare);
        assert!(entries[2].branch.is_none());

        assert!(entries[3].detached);
        assert_eq!(
            entries[3].prunable.as_deref(),
            Some("gitdir file points to non-existent location")
        );
        assert!(entries[3].is_detached());
    }

    #[test]
    fn locked_without_a_reason_has_no_reason() {
        let entries = parse_worktree_list("worktree /repo/wt\nlocked\n");
        assert!(entries[0].locked);
        assert!(entries[0].lock_reason.is_none());
    }

    #[test]
    fn malformed_and_empty_input_yields_no_entries() {
        assert_eq!(parse_worktree_list("").len(), 0);
        assert_eq!(
            parse_worktree_list("HEAD abc\nbranch refs/heads/main\n").len(),
            0
        );
        assert_eq!(parse_worktree_list("worktree \n").len(), 0);
    }

    #[test]
    fn status_porcelain_counts_changed_paths() {
        let status = parse_status_porcelain(" M src/lib.rs\n?? new.txt\n\n");
        assert_eq!(status.len(), 2);
        assert_eq!(parse_status_porcelain("").len(), 0);
        assert_eq!(parse_status_porcelain("\n  \n").len(), 0);
    }

    #[test]
    fn error_constructors_are_typed_and_marked() {
        let missing = not_installed_error("git");
        assert_eq!(missing.kind(), ErrorKind::Config);
        assert!(is_not_installed(&missing));
        assert!(missing.to_string().contains("install git"));

        let outside = not_a_repository_error(Path::new("/tmp"));
        assert_eq!(outside.kind(), ErrorKind::Config);
        assert!(is_not_a_repository(&outside));

        let failed = git_failure_error(&["worktree", "add"], "fatal: already exists\n");
        assert_eq!(failed.kind(), ErrorKind::Tool);
        assert_eq!(
            failed.to_string(),
            "tool: git worktree add failed: fatal: already exists"
        );
        assert!(!is_not_a_repository(&failed));
    }

    #[tokio::test]
    async fn a_missing_program_yields_the_not_installed_error() {
        let runner = ProcessGitRunner::with_program("openjiuwen-no-such-git-binary");
        assert_eq!(runner.program(), "openjiuwen-no-such-git-binary");
        let error = runner
            .run(&["status"], Path::new("."))
            .await
            .expect_err("missing binary");
        assert!(is_not_installed(&error));
    }

    #[test]
    fn default_runner_spawns_git() {
        assert_eq!(ProcessGitRunner::default(), ProcessGitRunner::new());
        assert_eq!(ProcessGitRunner::new().program(), GIT_BINARY);
    }
}
