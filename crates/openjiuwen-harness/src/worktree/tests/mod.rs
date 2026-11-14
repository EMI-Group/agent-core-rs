//! Offline tests for the worktree helper.
//!
//! Everything here is deterministic: the manager is driven through a scripted
//! [`GitRunner`] that returns queued output and records every invocation, and the
//! one test that spawns the real `git` binary skips itself when git is absent.
//! The `sparse` and `changes` suites assert sparse-checkout command construction
//! and change-summary call order through the same runner.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use parking_lot::Mutex;

use openjiuwen_common::error::{Error, Result};

use super::git::{GitOutput, GitRunner};

mod changes;
mod cleanup;
mod lock;
mod manager;
mod sparse;
mod tool;
mod tools;

/// One recorded git invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScriptedCall {
    /// Arguments after the executable name.
    pub(super) args: Vec<String>,
    /// Directory the command ran in.
    pub(super) cwd: PathBuf,
}

impl ScriptedCall {
    /// The arguments joined by spaces, for compact assertions.
    pub(super) fn command(&self) -> String {
        self.args.join(" ")
    }
}

/// A queued response of the scripted runner.
enum ScriptedResponse {
    /// Exit code 0 with the given stdout.
    Ok(String),
    /// A failure rendered as [`Error::tool`].
    Fail(String),
}

/// A [`GitRunner`] that replays queued responses and records every call.
pub(crate) struct ScriptedGitRunner {
    responses: Mutex<VecDeque<ScriptedResponse>>,
    calls: Mutex<Vec<ScriptedCall>>,
}

impl ScriptedGitRunner {
    /// Builds a runner with no queued response.
    pub(super) fn new() -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Queues a successful invocation printing `stdout`.
    pub(super) fn push_ok(&self, stdout: &str) {
        self.responses
            .lock()
            .push_back(ScriptedResponse::Ok(stdout.to_owned()));
    }

    /// Queues a successful invocation with no output.
    pub(super) fn push_success(&self) {
        self.push_ok("");
    }

    /// Queues a failing invocation with `message` as the git stderr.
    pub(super) fn push_fail(&self, message: &str) {
        self.responses
            .lock()
            .push_back(ScriptedResponse::Fail(message.to_owned()));
    }

    /// Every recorded invocation, oldest first.
    pub(super) fn calls(&self) -> Vec<ScriptedCall> {
        self.calls.lock().clone()
    }

    /// The commands of every recorded invocation, oldest first.
    pub(super) fn commands(&self) -> Vec<String> {
        self.calls
            .lock()
            .iter()
            .map(ScriptedCall::command)
            .collect()
    }

    /// The last recorded invocation, when any.
    pub(super) fn last_call(&self) -> Option<ScriptedCall> {
        self.calls.lock().last().cloned()
    }

    /// Number of queued but unused responses.
    pub(super) fn remaining(&self) -> usize {
        self.responses.lock().len()
    }
}

impl Default for ScriptedGitRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GitRunner for ScriptedGitRunner {
    async fn run(&self, args: &[&str], cwd: &Path) -> Result<GitOutput> {
        self.calls.lock().push(ScriptedCall {
            args: args.iter().map(|arg| (*arg).to_owned()).collect(),
            cwd: cwd.to_path_buf(),
        });
        let response = self.responses.lock().pop_front().ok_or_else(|| {
            Error::tool(format!(
                "scripted git runner exhausted while running `git {}`",
                args.join(" ")
            ))
        })?;
        match response {
            ScriptedResponse::Ok(stdout) => Ok(GitOutput::new(stdout)),
            ScriptedResponse::Fail(message) => Err(Error::tool(message)),
        }
    }
}

/// The porcelain listing of a repository with a `main` worktree and, when
/// `extra` is given, one linked worktree at `extra`.
pub(crate) fn listing(repo: &Path, extra: Option<(&Path, &str)>) -> String {
    let main = format!(
        "worktree {}\nHEAD {}\nbranch refs/heads/main\n",
        repo.display(),
        "1".repeat(40)
    );
    match extra {
        Some((path, branch)) => format!(
            "{main}\nworktree {}\nHEAD {}\nbranch refs/heads/{branch}\n",
            path.display(),
            "2".repeat(40)
        ),
        None => main,
    }
}

/// Creates a temp workspace and returns it with its normalised root path.
pub(crate) fn temp_root() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let root = dir.path().to_path_buf();
    (dir, root)
}

/// Whether the real git binary is available (guards the single live test).
pub(crate) fn real_git_available() -> bool {
    super::git_available()
}
