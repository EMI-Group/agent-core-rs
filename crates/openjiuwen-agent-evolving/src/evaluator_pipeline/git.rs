//! Preparing the workspace a task's test command runs against.
//!
//! Python's benchmark adapters cloned inside the pipeline's image builder. The
//! Rust port separates "get the source tree" from "build an image": a
//! [`WorkspacePreparer`] turns a [`GitWorkspaceSpec`] into a checked-out tree,
//! applying an optional patch and resolving the final commit.
//!
//! The invocations are deliberately small and deterministic:
//!
//! | situation | command |
//! | --- | --- |
//! | fresh clone | `git clone [--depth 1] [--recurse-submodules] <url> <workspace>` |
//! | existing checkout | `git fetch --all --tags` (in the workspace) |
//! | revision | `git checkout <revision>` (in the workspace) |
//! | submodules | `git submodule update --init --recursive` (in the workspace) |
//! | patch | `git apply <workspace>/.openjiuwen-eval.patch` (in the workspace) |
//! | resolution | `git rev-parse HEAD` (in the workspace) |
//!
//! The commands run with the workspace as their working directory instead of
//! `git -C <workspace>`, which keeps the recorded command line identical to what
//! a human would type after `cd`.
//!
//! Only clone/fetch/checkout/submodule/apply failures are errors
//! ([`Error::validation`], naming the command and its standard error). The
//! final `git rev-parse HEAD` is a probe: when the workspace is not a git
//! repository the requested revision (or `None`) is reported instead, so a
//! plain directory with a patch still prepares successfully.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use tokio_util::sync::CancellationToken;

use super::command::{check_cancel, CommandRunner};
use super::models::{CommandSpec, ExecResult, GitWorkspaceSpec};

/// The file name the patch content is written to inside the workspace.
///
/// It is written, applied, and removed in one step; the name is fixed so a run
/// that is killed mid-apply leaves an obvious artefact instead of an anonymous
/// temporary file.
pub const PATCH_FILE_NAME: &str = ".openjiuwen-eval.patch";

/// A prepared workspace: where the tree lives and what it resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedWorkspace {
    /// The absolute workspace path, when it could be resolved as one.
    pub workspace: PathBuf,
    /// The commit `HEAD` points at after preparation, falling back to the
    /// requested revision when the tree is not a git repository.
    pub revision: Option<String>,
    /// `true` when a patch was applied.
    pub applied_patch: bool,
}

/// Clones, checks out, and patches a workspace through the command seam.
#[derive(Debug, Clone)]
pub struct WorkspacePreparer {
    runner: Arc<dyn CommandRunner>,
}

impl WorkspacePreparer {
    /// Creates a preparer that runs git through `runner`.
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>) -> Self {
        Self { runner }
    }

    /// The runner git commands go through.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn CommandRunner> {
        &self.runner
    }

    /// Prepares `workspace` according to `spec`.
    ///
    /// The directory is created when missing. When `spec.url` is set and the
    /// directory is not already a checkout, it is cloned; when it is already a
    /// checkout, it is fetched instead. The requested revision is checked out,
    /// submodules are initialised when asked, and the patch is applied last so
    /// it lands on the requested revision.
    pub async fn prepare(
        &self,
        workspace: &Path,
        spec: &GitWorkspaceSpec,
        cancel: &CancellationToken,
    ) -> Result<PreparedWorkspace> {
        let workspace = absolute(workspace);
        tokio::fs::create_dir_all(&workspace)
            .await
            .map_err(|error| {
                Error::sys_op(format!(
                    "failed to create workspace {}: {error}",
                    workspace.display()
                ))
            })?;
        check_cancel(cancel)?;

        let existing_checkout = tokio::fs::metadata(workspace.join(".git")).await.is_ok();
        let mut cloned = false;
        if let Some(url) = spec.url.as_deref().filter(|url| !url.trim().is_empty()) {
            if existing_checkout {
                self.git(&workspace, ["fetch", "--all", "--tags"], cancel)
                    .await?;
            } else {
                let mut args = vec![String::from("clone")];
                if spec.shallow {
                    args.push(String::from("--depth"));
                    args.push(String::from("1"));
                }
                if spec.submodules {
                    args.push(String::from("--recurse-submodules"));
                }
                args.push(url.to_owned());
                args.push(workspace.display().to_string());
                self.git_clone(args, cancel).await?;
                cloned = true;
            }
        }

        let is_checkout = existing_checkout || cloned;
        if let Some(revision) = spec
            .revision
            .as_deref()
            .filter(|rev| !rev.trim().is_empty())
        {
            if !is_checkout {
                return Err(Error::validation(format!(
                    "cannot check out `{revision}`: {} is not a git repository",
                    workspace.display()
                )));
            }
            self.git(&workspace, ["checkout", revision], cancel).await?;
        }

        if spec.submodules && !cloned && is_checkout {
            self.git(
                &workspace,
                ["submodule", "update", "--init", "--recursive"],
                cancel,
            )
            .await?;
        }

        let applied_patch = match spec.patch.as_deref() {
            Some(patch) => {
                self.apply_patch(&workspace, patch, cancel).await?;
                true
            }
            None => false,
        };

        let revision = self
            .resolve_revision(&workspace, spec.revision.clone(), cancel)
            .await;
        Ok(PreparedWorkspace {
            workspace,
            revision,
            applied_patch,
        })
    }

    /// Writes `patch` into the workspace, applies it with `git apply`, and
    /// removes the file again.
    async fn apply_patch(
        &self,
        workspace: &Path,
        patch: &str,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let patch_path = workspace.join(PATCH_FILE_NAME);
        tokio::fs::write(&patch_path, patch.as_bytes())
            .await
            .map_err(|error| {
                Error::sys_op(format!(
                    "failed to write patch file {}: {error}",
                    patch_path.display()
                ))
            })?;
        check_cancel(cancel)?;
        let patch_arg = patch_path.display().to_string();
        let applied = self
            .git(workspace, ["apply", patch_arg.as_str()], cancel)
            .await;
        let removed = tokio::fs::remove_file(&patch_path).await;
        if let Err(error) = removed {
            tracing::debug!(
                patch = %patch_path.display(),
                error = %error,
                "failed to remove the applied patch file"
            );
        }
        applied.map(|_| ())
    }

    /// Runs `git` in `workspace` with `args`, converting a failure into
    /// [`Error::validation`].
    async fn git<const N: usize>(
        &self,
        workspace: &Path,
        args: [&str; N],
        cancel: &CancellationToken,
    ) -> Result<ExecResult> {
        let spec = CommandSpec::new("git")
            .args(args)
            .with_cwd(workspace.to_path_buf());
        self.run_git(&spec, cancel).await
    }

    /// Runs `git clone` with pre-built arguments, in the process working
    /// directory (the destination is absolute).
    async fn git_clone(&self, args: Vec<String>, cancel: &CancellationToken) -> Result<ExecResult> {
        self.run_git(&CommandSpec::new("git").args(args), cancel)
            .await
    }

    /// Runs one git command and rejects a timeout or non-zero exit.
    async fn run_git(&self, spec: &CommandSpec, cancel: &CancellationToken) -> Result<ExecResult> {
        check_cancel(cancel)?;
        let result = self.runner.run(spec, cancel).await?;
        if result.timed_out {
            return Err(Error::validation(format!(
                "`{}` timed out",
                spec.command_line()
            )));
        }
        if !result.success() {
            return Err(Error::validation(format!(
                "`{}` failed (rc={}): {}",
                spec.command_line(),
                result.returncode,
                result.stderr.trim()
            )));
        }
        Ok(result)
    }

    /// Reads `HEAD` with `git rev-parse HEAD`.
    ///
    /// The probe never fails the preparation: a non-repository workspace (or a
    /// cancelled probe) keeps the requested revision.
    async fn resolve_revision(
        &self,
        workspace: &Path,
        requested: Option<String>,
        cancel: &CancellationToken,
    ) -> Option<String> {
        let spec = CommandSpec::new("git")
            .arg("rev-parse")
            .arg("HEAD")
            .with_cwd(workspace.to_path_buf());
        match self.runner.run(&spec, cancel).await {
            Ok(result) if result.success() => {
                let head = result.stdout.trim();
                if head.is_empty() {
                    requested
                } else {
                    Some(head.to_owned())
                }
            }
            _ => requested,
        }
    }
}

/// Resolves `path` to an absolute path, keeping it unchanged when the platform
/// cannot resolve it.
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}
