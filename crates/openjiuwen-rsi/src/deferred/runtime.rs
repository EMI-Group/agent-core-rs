//! Deferred subprocess, extension-loading, CI, and git surfaces.

use serde_json::Value;

use openjiuwen_common::error::Result;

use super::not_implemented;

/// The surface name used by [`SubprocessEvaluator`]'s errors.
const SURFACE_SUBPROCESS_EVALUATOR: &str = "subprocess program evaluator";
/// The surface name used by [`RuntimeExtensionLoader`]'s errors.
const SURFACE_RUNTIME_EXTENSION_LOADER: &str = "runtime extension loader";
/// The surface name used by [`CiGate`]'s errors.
const SURFACE_CI_GATE: &str = "CI gate runner";
/// The surface name used by [`GitBackend`]'s errors.
const SURFACE_GIT_BACKEND: &str = "git backend";

/// Program evaluator that shells out to a subprocess — **deferred**.
///
/// Mirrors Python's `artifact_rsi.program_opt.runtime` subprocess runner and
/// `harness_rsi.evaluator.case_runner`: it writes the candidate to a temporary
/// directory, runs it, and parses the reply. That needs subprocess and pip
/// execution, which is outside the offline port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubprocessEvaluator;

impl SubprocessEvaluator {
    /// Mirrors the runtime's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`subprocess program evaluator`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE_SUBPROCESS_EVALUATOR,
            "needs subprocess and pip execution of candidate programs",
        ))
    }

    /// Mirrors `SubprocessEvaluator.evaluate`; always reports that it is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`subprocess program evaluator`).
    pub fn evaluate(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE_SUBPROCESS_EVALUATOR,
            "needs subprocess and pip execution of candidate programs",
        ))
    }
}

/// Loader for Python runtime extensions — **deferred**.
///
/// Mirrors Python's `harness_rsi.auto_harness.infra.runtime_extension_loader`,
/// which hot-loads extension modules by synthesising `sys.modules` entries and
/// running static checks. Python import machinery has no Rust analogue here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeExtensionLoader;

impl RuntimeExtensionLoader {
    /// Mirrors the loader's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`runtime extension loader`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE_RUNTIME_EXTENSION_LOADER,
            "needs synthetic `sys.modules` hot-loading of Python runtime extensions",
        ))
    }

    /// Mirrors `RuntimeExtensionLoader.load`; always reports that it is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`runtime extension loader`).
    pub fn load(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE_RUNTIME_EXTENSION_LOADER,
            "needs synthetic `sys.modules` hot-loading of Python runtime extensions",
        ))
    }
}

/// Pretest/CI gate runner — **deferred**.
///
/// Mirrors Python's `harness_rsi.auto_harness.infra.ci_gate_runner`, which runs
/// lint, type-check, and test commands in a worktree and extracts the failure
/// report. That needs subprocess `pytest`/`ruff`/`pyright` execution and
/// `git`-host HTTP access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CiGate;

impl CiGate {
    /// Mirrors the gate's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`CI gate runner`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE_CI_GATE,
            "needs subprocess pytest/ruff/pyright execution and CI-host HTTP access",
        ))
    }

    /// Mirrors `CiGate.run`; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`CI gate runner`).
    pub fn run(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE_CI_GATE,
            "needs subprocess pytest/ruff/pyright execution and CI-host HTTP access",
        ))
    }
}

/// Git worktree and patch backend — **deferred**.
///
/// Mirrors Python's `harness_rsi.auto_harness.infra.{git_operations,
/// worktree_manager}`, which clone a repository, create per-session worktrees,
/// apply patches, and push branches through subprocess `git` and the `GitCode`
/// HTTP API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GitBackend;

impl GitBackend {
    /// Mirrors the backend's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`git backend`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE_GIT_BACKEND,
            "needs subprocess git worktree/patch operations and `GitCode` HTTP access",
        ))
    }

    /// Mirrors `GitBackend.apply`; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`git backend`).
    pub fn apply(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE_GIT_BACKEND,
            "needs subprocess git worktree/patch operations and `GitCode` HTTP access",
        ))
    }
}
