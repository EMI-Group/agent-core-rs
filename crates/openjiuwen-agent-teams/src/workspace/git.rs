//! Async wrapper over the `git` CLI, used by the git worktree backend.
//!
//! Python parity source: `openjiuwen/harness/tools/worktree/git.py`. Every function is stateless and
//! pure subprocess work — context is passed by argument, nothing is cached globally.
//!
//! Contract of [`run_git`]:
//!
//! - the binary is resolved through [`which`], so a missing `git` fails with
//!   [`Error::Config`] naming the required binary instead of a bare I/O error;
//! - every invocation is bounded by [`GIT_TIMEOUT`] and its exit status, stdout and stderr are
//!   captured (stderr never leaks to the process console);
//! - interactive credential prompts are suppressed (`GIT_TERMINAL_PROMPT=0`, empty `GIT_ASKPASS`)
//!   and repository-locating variables inherited from the environment (`GIT_DIR`, `GIT_WORK_TREE`,
//!   `GIT_INDEX_FILE`) are removed, so a command always targets the directory it was given;
//! - stdin is `/dev/null`, so `git` can never block on terminal input.
//!
//! [`GitRepository::discover`] resolves the repository a path lives in and normalizes it the way
//! Python's `find_canonical_git_root` does: a path inside a *worktree* resolves to the main
//! repository root, so worktrees are always registered in one place and commands are never run
//! from inside the worktree they are about to remove.

use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;
use tracing::debug;

use openjiuwen_common::error::{Error, Result};

/// Wall-clock budget of one `git` invocation.
pub const GIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Name of the `git` binary looked up on `PATH`.
pub const GIT_BINARY: &str = "git";

/// Base ref used when a [`WorktreeCreateRequest`](crate::workspace::WorktreeCreateRequest) names
/// none.
pub const DEFAULT_BASE_REF: &str = "HEAD";

/// Prefix of a local branch ref in `git worktree list --porcelain` output.
pub(crate) const HEADS_REF_PREFIX: &str = "refs/heads/";

/// Captured result of one `git` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitOutput {
    /// Process exit code; `None` when the process was terminated by a signal.
    pub code: Option<i32>,
    /// Trimmed standard output.
    pub stdout: String,
    /// Trimmed standard error.
    pub stderr: String,
}

impl GitOutput {
    /// Returns `true` when `git` exited successfully.
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.code == Some(0)
    }

    /// Returns the first non-empty stderr line, or `"no stderr"` when the command was silent.
    ///
    /// Diagnostics embed this instead of the whole stderr so a multi-line `git` hint block does not
    /// bloat the error message.
    #[must_use]
    pub fn stderr_summary(&self) -> String {
        self.stderr
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("no stderr")
            .to_owned()
    }

    /// Returns stdout's first line, or `None` when stdout is empty.
    #[must_use]
    pub fn first_line(&self) -> Option<&str> {
        self.stdout.lines().map(str::trim).find(|l| !l.is_empty())
    }
}

/// Resolves the `git` binary through `PATH`.
///
/// # Errors
///
/// Returns [`Error::Config`] when `git` is not installed (or not on `PATH`); the message names the
/// binary and what needs it.
pub fn resolve_git() -> Result<PathBuf> {
    which::which(GIT_BINARY).map_err(|_| {
        Error::config(format!(
            "`{GIT_BINARY}` was not found on PATH; worktree isolation requires the git CLI, \
             install git or inject a different worktree backend"
        ))
    })
}

/// Runs `git <args>` with `cwd` as working directory and captures its output.
///
/// # Errors
///
/// Returns [`Error::Config`] when the `git` binary cannot be resolved, [`Error::Timeout`] when the
/// command does not finish within [`GIT_TIMEOUT`], and [`Error::SysOp`] when it cannot be spawned.
/// A non-zero exit status is *not* an error here — inspect [`GitOutput::succeeded`].
pub async fn run_git(cwd: &Path, args: &[&str]) -> Result<GitOutput> {
    let program = resolve_git()?;
    let mut command = Command::new(program);
    command
        .current_dir(cwd)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", "")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let label = format!("git {}", args.join(" "));
    debug!(command = %label, cwd = %cwd.display(), "running git");
    let output = tokio::time::timeout(GIT_TIMEOUT, command.output())
        .await
        .map_err(|_| {
            Error::timeout(format!(
                "{label} timed out after {}s in {}",
                GIT_TIMEOUT.as_secs(),
                cwd.display()
            ))
        })?
        .map_err(|error| Error::sys_op(format!("running {label} in {}: {error}", cwd.display())))?;
    Ok(GitOutput {
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
    })
}

/// Builds the [`Error::SysOp`] reported for a failed `git` command.
fn command_error(action: &str, cwd: &Path, output: &GitOutput) -> Error {
    let code = output
        .code
        .map_or_else(|| "signal".to_owned(), |code| code.to_string());
    Error::sys_op(format!(
        "{action} failed (git exit code {code}) in {}: {}",
        cwd.display(),
        output.stderr_summary()
    ))
}

/// Normalizes a path lexically: `.` components are dropped and `..` pops one component.
///
/// No filesystem access happens, so this works for paths that do not exist yet (a worktree target
/// directory, for example). A `..` above the root of a relative path is preserved.
#[must_use]
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let at_root = matches!(
                    normalized.components().next_back(),
                    None | Some(Component::RootDir | Component::Prefix(_))
                );
                if !at_root {
                    normalized.pop();
                }
            }
            Component::Normal(_) | Component::RootDir | Component::Prefix(_) => {
                normalized.push(component);
            }
        }
    }
    normalized
}

/// Returns `path` itself when it exists, else its nearest existing ancestor.
///
/// `git` needs a real directory to start from: a session tree may not exist yet when the first
/// member is allocated.
#[must_use]
pub fn existing_ancestor(path: &Path) -> PathBuf {
    let mut candidate = Some(path);
    while let Some(current) = candidate {
        if current.is_dir() {
            return current.to_path_buf();
        }
        candidate = current.parent();
    }
    PathBuf::from(".")
}

/// Converts a path to `&str`, refusing non-UTF-8 paths.
fn path_str(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        Error::validation(format!(
            "worktree path {} is not valid UTF-8 and cannot be passed to git",
            path.display()
        ))
    })
}

/// The repository a worktree operation runs against.
///
/// `root` is the *main* repository root: discovery from inside a teammate worktree resolves the
/// worktree's `commondir` back to the repository that owns it (Python `find_canonical_git_root`).
/// `git` commands therefore never run from inside the worktree they are about to remove, and every
/// worktree of a repository is registered in the same `.git/worktrees` directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitRepository {
    /// Main repository root (the directory that contains `.git`).
    pub root: PathBuf,
    /// Absolute git directory of the path discovery started from (a worktree's own git dir when
    /// discovery started inside one).
    pub git_dir: PathBuf,
    /// Absolute common git directory shared by every worktree of the repository.
    pub common_dir: PathBuf,
}

impl GitRepository {
    /// Discovers the repository containing `start`.
    ///
    /// `start` need not exist: discovery walks up to its nearest existing ancestor first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `start` is not inside a git repository (the message names the
    /// directory that was searched), and whatever [`run_git`] reports when `git` is missing or the
    /// command fails unexpectedly.
    pub async fn discover(start: &Path) -> Result<Self> {
        let anchor = existing_ancestor(start);
        let output = run_git(
            &anchor,
            &[
                "rev-parse",
                "--show-toplevel",
                "--absolute-git-dir",
                "--git-common-dir",
            ],
        )
        .await?;
        if !output.succeeded() {
            return Err(Error::config(format!(
                "worktree isolation requires a git repository: {} is not inside one ({})",
                anchor.display(),
                output.stderr_summary()
            )));
        }
        let mut lines = output.stdout.lines().map(str::trim);
        let (Some(toplevel), Some(git_dir), Some(common_dir)) =
            (lines.next(), lines.next(), lines.next())
        else {
            return Err(Error::config(format!(
                "git rev-parse returned an unexpected repository description in {}: {:?}",
                anchor.display(),
                output.stdout
            )));
        };
        let git_dir = normalize_path(Path::new(git_dir));
        // `--git-common-dir` is relative to the directory the command ran in for the main
        // repository (`.git`) and absolute inside a worktree; join before normalizing.
        let common_dir = normalize_path(&anchor.join(common_dir));
        let root = if common_dir.file_name() == Some(OsStr::new(".git")) {
            common_dir
                .parent()
                .map_or_else(|| normalize_path(Path::new(toplevel)), Path::to_path_buf)
        } else {
            normalize_path(Path::new(toplevel))
        };
        Ok(Self {
            root,
            git_dir,
            common_dir,
        })
    }

    /// Returns the lock file guarding this repository's worktree critical sections.
    #[must_use]
    pub fn lock_file(&self) -> PathBuf {
        self.common_dir.join(super::lock::LOCK_FILE_NAME)
    }

    /// Resolves `refspec` to a full commit id.
    ///
    /// # Errors
    ///
    /// Returns whatever [`run_git`] reports; an unresolvable ref is `Ok(None)`, not an error.
    pub async fn resolve_ref(&self, refspec: &str) -> Result<Option<String>> {
        let output = run_git(&self.root, &["rev-parse", "--verify", refspec]).await?;
        if output.succeeded() {
            Ok(output.first_line().map(str::to_owned))
        } else {
            Ok(None)
        }
    }

    /// Lists the worktrees registered in this repository.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when `git worktree list` fails.
    pub async fn worktree_entries(&self) -> Result<Vec<WorktreeEntry>> {
        let output = run_git(&self.root, &["worktree", "list", "--porcelain"]).await?;
        if !output.succeeded() {
            return Err(command_error("listing worktrees", &self.root, &output));
        }
        Ok(parse_worktree_list(&output.stdout))
    }

    /// Looks up one worktree by path.
    ///
    /// # Errors
    ///
    /// Returns whatever [`Self::worktree_entries`] reports.
    pub async fn worktree_entry(&self, path: &Path) -> Result<Option<WorktreeEntry>> {
        Ok(self
            .worktree_entries()
            .await?
            .into_iter()
            .find(|entry| same_path(&entry.path, path)))
    }

    /// Runs `git worktree add -b <branch> <path> <base_ref>`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `git` reports a name collision (existing branch or
    /// existing directory) — the caller can act on it by choosing another name or cleaning up —
    /// and [`Error::SysOp`] for every other failure. A non-UTF-8 `path` or `branch` is rejected
    /// with [`Error::Validation`].
    pub async fn worktree_add(&self, path: &Path, branch: &str, base_ref: &str) -> Result<()> {
        let target = path_str(path)?;
        let output = run_git(
            &self.root,
            &["worktree", "add", "-b", branch, target, base_ref],
        )
        .await?;
        if output.succeeded() {
            return Ok(());
        }
        if output.stderr.contains("already exists") {
            return Err(Error::validation(format!(
                "cannot create worktree {} on branch {branch}: {}",
                path.display(),
                output.stderr_summary()
            )));
        }
        Err(command_error("creating worktree", &self.root, &output))
    }

    /// Runs `git worktree remove --force <path>`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when the worktree cannot be removed; a locked worktree keeps
    /// `git`'s own message ("cannot remove a locked working tree") so the caller learns it must
    /// unlock first.
    pub async fn worktree_remove(&self, path: &Path) -> Result<()> {
        let target = path_str(path)?;
        let output = run_git(&self.root, &["worktree", "remove", "--force", target]).await?;
        if output.succeeded() {
            return Ok(());
        }
        Err(command_error("removing worktree", &self.root, &output))
    }

    /// Runs `git worktree prune`, dropping admin data of worktrees whose directory disappeared.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when the prune fails.
    pub async fn worktree_prune(&self) -> Result<()> {
        let output = run_git(&self.root, &["worktree", "prune"]).await?;
        if output.succeeded() {
            return Ok(());
        }
        Err(command_error("pruning worktrees", &self.root, &output))
    }

    /// Runs `git worktree lock --reason <reason> <path>`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the path is not a registered worktree of this repository,
    /// and [`Error::SysOp`] for every other failure.
    pub async fn lock_worktree(&self, path: &Path, reason: &str) -> Result<()> {
        let target = path_str(path)?;
        let output = run_git(
            &self.root,
            &["worktree", "lock", "--reason", reason, target],
        )
        .await?;
        if output.succeeded() {
            return Ok(());
        }
        if output.stderr.contains("not a working tree") || output.stderr.contains("is not a") {
            return Err(Error::not_found(format!(
                "worktree {} is not registered in the repository at {}: {}",
                path.display(),
                self.root.display(),
                output.stderr_summary()
            )));
        }
        Err(command_error("locking worktree", &self.root, &output))
    }

    /// Runs `git worktree unlock <path>`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the path is not a registered, locked worktree, and
    /// [`Error::SysOp`] for every other failure.
    pub async fn unlock_worktree(&self, path: &Path) -> Result<()> {
        let target = path_str(path)?;
        let output = run_git(&self.root, &["worktree", "unlock", target]).await?;
        if output.succeeded() {
            return Ok(());
        }
        if output.stderr.contains("not locked") || output.stderr.contains("not a working tree") {
            return Err(Error::not_found(format!(
                "worktree {} is not a locked worktree of the repository at {}",
                path.display(),
                self.root.display()
            )));
        }
        Err(command_error("unlocking worktree", &self.root, &output))
    }

    /// Runs `git branch -D <branch>` and classifies the outcome.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unsafe branch name (`-`-prefixed, whitespace, non-UTF-8)
    /// and [`Error::SysOp`] for an unexpected failure. A missing branch and a branch checked out in
    /// another worktree are *not* errors here; they are reported by [`BranchDeleteOutcome`] so the
    /// caller decides whether they matter.
    pub async fn delete_branch(&self, branch: &str) -> Result<BranchDeleteOutcome> {
        let trimmed = branch.trim();
        if trimmed.is_empty() || trimmed.starts_with('-') || trimmed.contains(char::is_whitespace) {
            return Err(Error::validation(format!(
                "invalid branch name {branch:?}: must be non-empty and cannot start with `-`"
            )));
        }
        let output = run_git(&self.root, &["branch", "-D", trimmed]).await?;
        if output.succeeded() {
            return Ok(BranchDeleteOutcome::Deleted);
        }
        if output.stderr.contains("not found") {
            return Ok(BranchDeleteOutcome::Missing);
        }
        if output.stderr.contains("used by worktree") || output.stderr.contains("checked out at") {
            return Ok(BranchDeleteOutcome::InUse(output.stderr_summary()));
        }
        Err(command_error("deleting branch", &self.root, &output))
    }
}

/// Outcome of [`GitRepository::delete_branch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchDeleteOutcome {
    /// The branch existed and was deleted.
    Deleted,
    /// The branch did not exist; there was nothing to delete.
    Missing,
    /// The branch still exists because another worktree has it checked out; the payload is git's
    /// own explanation.
    InUse(String),
}

/// One entry of `git worktree list --porcelain`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeEntry {
    /// Absolute directory of the worktree.
    pub path: PathBuf,
    /// `HEAD` commit id of the worktree, when git reported one.
    pub head: Option<String>,
    /// Branch checked out in the worktree, as a local branch name (`refs/heads/` stripped).
    pub branch: Option<String>,
    /// Lock reason, present when the worktree is locked.
    pub locked: Option<String>,
    /// Whether this entry is the bare repository itself.
    pub bare: bool,
    /// Whether the worktree is on a detached `HEAD`.
    pub detached: bool,
}

/// Parses `git worktree list --porcelain` output.
///
/// Entries are separated by blank lines; each entry carries `worktree <path>` first, then zero or
/// more of `HEAD <sha>`, `branch <ref>`, `bare`, `detached`, `locked [<reason>]`. Unknown lines are
/// ignored, so a future git version that adds attributes does not break parsing.
#[must_use]
pub fn parse_worktree_list(porcelain: &str) -> Vec<WorktreeEntry> {
    let mut entries = Vec::new();
    let mut current: Option<WorktreeEntry> = None;
    for line in porcelain.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            if let Some(entry) = current.take() {
                entries.push(entry);
            }
            continue;
        }
        if let Some(path) = line.strip_prefix("worktree ") {
            if let Some(entry) = current.take() {
                entries.push(entry);
            }
            current = Some(WorktreeEntry {
                path: normalize_path(Path::new(path.trim())),
                head: None,
                branch: None,
                locked: None,
                bare: false,
                detached: false,
            });
            continue;
        }
        let Some(entry) = current.as_mut() else {
            continue;
        };
        if let Some(head) = line.strip_prefix("HEAD ") {
            entry.head = Some(head.trim().to_owned());
        } else if let Some(branch) = line.strip_prefix("branch ") {
            let branch = branch.trim();
            entry.branch = Some(
                branch
                    .strip_prefix(HEADS_REF_PREFIX)
                    .unwrap_or(branch)
                    .to_owned(),
            );
        } else if line == "bare" {
            entry.bare = true;
        } else if line == "detached" {
            entry.detached = true;
        } else if line == "locked" {
            entry.locked = Some(String::new());
        } else if let Some(reason) = line.strip_prefix("locked ") {
            entry.locked = Some(reason.trim().to_owned());
        }
    }
    if let Some(entry) = current {
        entries.push(entry);
    }
    entries
}

/// Returns `true` when both paths address the same directory.
///
/// Compares the paths verbatim, then lexically normalized, then (when both exist) after
/// `canonicalize`, so a worktree reached through a symlinked or `..`-containing path is still
/// recognized as the same worktree.
#[must_use]
pub(crate) fn same_path(left: &Path, right: &Path) -> bool {
    if left == right || normalize_path(left) == normalize_path(right) {
        return true;
    }
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}
