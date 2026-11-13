//! Static verification of an executed optimisation plan.
//!
//! Python's `verification.py` validates the role integration worktrees: the plan
//! schema, the execution results, the declared write paths, and the presence of
//! every harness file an action claims to have produced. The Rust port keeps the
//! portable part of that contract and drops the worktree/agent machinery:
//!
//! - [`StaticActionVerifier`] checks every action's outcome, its written paths
//!   against its declared prefixes, and that its `target_path` now exists.
//! - [`WorkspaceProbe`] is the seam for "does this path exist", implemented for
//!   the real filesystem ([`FsWorkspaceProbe`]) and for tests
//!   ([`InMemoryWorkspaceProbe`]).
//! - LLM-driven repair (`HarnessRepairAgent` in Python) stays behind
//!   [`ActionVerifier`]: the optimizer re-plans
//!   repairable failures, it never asks a model itself.
use super::plan::OptimizationPlan;
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
/// Outcome of one verification check.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    /// The check's condition held.
    Passed,
    /// The check's condition did not hold (also the default for an unset check).
    #[default]
    Failed,
}
impl CheckStatus {
    /// Returns the `snake_case` name used in artifacts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
        }
    }
    /// Whether the check failed.
    #[must_use]
    pub const fn is_failed(self) -> bool {
        matches!(self, Self::Failed)
    }
}
/// One named verification check.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct VerificationCheck {
    /// Check name; stable so callers can assert on it (see [`CheckStatus`] and
    /// the `CHECK_*` constants in this module).
    pub name: String,
    /// Whether the check passed.
    pub status: CheckStatus,
    /// Human-readable evidence for the status.
    pub detail: String,
}
impl VerificationCheck {
    /// Builds a passed check.
    #[must_use]
    pub fn passed(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Passed,
            detail: detail.into(),
        }
    }
    /// Builds a failed check.
    #[must_use]
    pub fn failed(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Failed,
            detail: detail.into(),
        }
    }
}
/// What one action's execution reported.
///
/// Python's `MemberActionExecutionResult` also carries worktree and merge
/// bookkeeping; the Rust executor seam only needs the outcome and the paths the
/// action actually wrote.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ActionOutcome {
    /// Id of the action this outcome belongs to.
    pub action_id: String,
    /// Whether the executor considered the action successful.
    pub success: bool,
    /// Executor log or summary, kept for diagnostics.
    pub output: String,
    /// Absolute paths the action wrote.
    pub written_paths: Vec<PathBuf>,
    /// Failure reason, when `success` is `false`.
    pub error: Option<String>,
}
/// Verification result for one role.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RoleVerification {
    /// The role these checks belong to.
    pub role: String,
    /// The role's checks, sorted by action id then check name.
    pub checks: Vec<VerificationCheck>,
    /// Whether every check passed.
    pub passed: bool,
}
/// Overall verification result for a plan.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberVerificationResult {
    /// `Passed` only when every check passed.
    pub status: VerificationStatus,
    /// Roles that were verified, ascending.
    pub checked_roles: Vec<String>,
    /// Every check, sorted by action id then check name.
    pub checks: Vec<VerificationCheck>,
    /// Per-role breakdown.
    pub role_results: BTreeMap<String, RoleVerification>,
    /// Whether a re-plan could plausibly fix the failure (see
    /// [`StaticActionVerifier`]).
    pub repairable: bool,
    /// Verifier annotations.
    pub metadata: BTreeMap<String, Value>,
}
/// Overall verification status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    /// Every check passed.
    Passed,
    /// At least one check failed (also the default).
    #[default]
    Failed,
}
impl VerificationStatus {
    /// Returns the `snake_case` name used in artifacts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
        }
    }
}
/// Name of the check asserting that an action's outcome exists and succeeded.
pub const CHECK_ACTION_SUCCEEDED: &str = "action_succeeded";
/// Name of the check asserting that every written path is declared.
pub const CHECK_DECLARED_WRITE_PATHS: &str = "declared_write_paths";
/// Name of the check asserting that an action declared a write prefix at all.
pub const CHECK_NO_DECLARED_WRITE_PATHS: &str = "no_declared_write_paths";
/// Name of the check asserting that the action's target path exists.
pub const CHECK_TARGET_EXISTS: &str = "target_exists";
/// Answers "does this path exist" for the verifier.
#[async_trait]
pub trait WorkspaceProbe: Send + Sync {
    /// Returns whether `path` exists in the probed workspace.
    async fn exists(&self, path: &Path) -> bool;
}
/// [`WorkspaceProbe`] over the real filesystem.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FsWorkspaceProbe;
impl FsWorkspaceProbe {
    /// Creates the probe.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}
#[async_trait]
impl WorkspaceProbe for FsWorkspaceProbe {
    async fn exists(&self, path: &Path) -> bool {
        tokio::fs::metadata(path).await.is_ok()
    }
}
/// Test double: a fixed set of existing paths.
///
/// The set is process-local and lock-protected; poison is recovered instead of
/// panicking so a probe shared with a failed test assertion stays usable.
#[derive(Debug, Default)]
pub struct InMemoryWorkspaceProbe {
    /// The paths reported as existing.
    paths: RwLock<HashSet<PathBuf>>,
}
impl InMemoryWorkspaceProbe {
    /// Creates an empty probe.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a probe pre-populated with `paths`.
    #[must_use]
    pub fn with_paths(paths: impl IntoIterator<Item = PathBuf>) -> Self {
        let probe = Self::new();
        for path in paths {
            probe.insert(path);
        }
        probe
    }
    /// Records `path` as existing.
    pub fn insert(&self, path: impl Into<PathBuf>) {
        self.write().insert(path.into());
    }
    /// Records every path in `paths` as existing.
    pub fn insert_all(&self, paths: impl IntoIterator<Item = PathBuf>) {
        self.write().extend(paths);
    }
    /// Returns how many paths the probe reports as existing.
    #[must_use]
    pub fn len(&self) -> usize {
        self.read().len()
    }
    /// Returns whether the probe is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }
    /// Returns a read guard over the path set.
    fn read(&self) -> RwLockReadGuard<'_, HashSet<PathBuf>> {
        self.paths.read().unwrap_or_else(PoisonError::into_inner)
    }
    /// Returns a write guard over the path set.
    fn write(&self) -> RwLockWriteGuard<'_, HashSet<PathBuf>> {
        self.paths.write().unwrap_or_else(PoisonError::into_inner)
    }
}
#[async_trait]
impl WorkspaceProbe for InMemoryWorkspaceProbe {
    async fn exists(&self, path: &Path) -> bool {
        self.read().contains(path)
    }
}
/// Everything one verification needs: the plan, what ran, and where.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct VerificationRequest {
    /// The plan that was executed (a merged plan when repair rounds ran).
    pub plan: OptimizationPlan,
    /// The outcomes reported by the executor, across every round.
    pub outcomes: Vec<ActionOutcome>,
    /// Root the plan's relative paths are resolved against.
    pub workspace_root: PathBuf,
}
/// Name reported by [`StaticActionVerifier`].
pub const VERIFIER_NAME: &str = "static_action_verifier";
/// Verifies an executed plan.
#[async_trait]
pub trait ActionVerifier: Send + Sync {
    /// Returns the verifier's human-readable name.
    fn name(&self) -> &str;
    /// Verifies `request`'s plan against its outcomes.
    ///
    /// # Errors
    ///
    /// Returns an [`openjiuwen_common::error::Error`] only for infrastructure
    /// failures (for example an unreadable workspace); a failed check is a
    /// successful call that returns [`VerificationStatus::Failed`].
    async fn verify(&self, request: &VerificationRequest) -> Result<MemberVerificationResult>;
}
/// Verifies execution outcomes, declared write paths, and target existence.
///
/// Per action it evaluates three checks:
///
/// 1. [`CHECK_ACTION_SUCCEEDED`] — an [`ActionOutcome`] exists for the action and
///    its `success` flag is `true`.
/// 2. [`CHECK_DECLARED_WRITE_PATHS`] / [`CHECK_NO_DECLARED_WRITE_PATHS`] — every
///    written path lies inside one of the action's `declared_write_paths`
///    prefixes, contains no `..`, and stays under the workspace root. An action
///    with no declared prefix fails (it may not write blindly).
/// 3. [`CHECK_TARGET_EXISTS`] — the action's `target_path` exists according to the
///    [`WorkspaceProbe`].
///
/// The result is `repairable` when at least one `action_succeeded`/`target_exists`
/// check failed and no write violation did: a write violation is a contract
/// breach a re-plan must not silently paper over.
#[derive(Clone)]
pub struct StaticActionVerifier {
    /// The probe used for the `target_exists` check.
    pub probe: Arc<dyn WorkspaceProbe>,
}
impl StaticActionVerifier {
    /// Creates a verifier over `probe`.
    #[must_use]
    pub fn new(probe: Arc<dyn WorkspaceProbe>) -> Self {
        Self { probe }
    }
    /// Builds the `target_exists` check for one action.
    async fn target_check(
        &self,
        action: &super::plan::OptimizationAction,
        root: &Path,
    ) -> VerificationCheck {
        let target = action.target_path.trim();
        if target.is_empty() {
            return VerificationCheck::failed(
                CHECK_TARGET_EXISTS,
                "action declares no target_path to verify",
            );
        }
        match normalize_path(Path::new(target), root) {
            Ok(relative) => {
                let absolute = root.join(&relative);
                if self.probe.exists(&absolute).await {
                    VerificationCheck::passed(
                        CHECK_TARGET_EXISTS,
                        format!("`{}` exists", absolute.display()),
                    )
                } else {
                    VerificationCheck::failed(
                        CHECK_TARGET_EXISTS,
                        format!("`{}` does not exist", absolute.display()),
                    )
                }
            }
            Err(reason) => VerificationCheck::failed(
                CHECK_TARGET_EXISTS,
                format!("target_path `{target}` is invalid: {reason}"),
            ),
        }
    }
    /// Collects every check for `request`, sorted by action id then check name.
    async fn collect_checks(
        &self,
        request: &VerificationRequest,
    ) -> Vec<(String, VerificationCheck)> {
        let root = request.workspace_root.as_path();
        let mut entries: Vec<(String, VerificationCheck)> = Vec::new();
        for action in &request.plan.actions {
            let outcome = request
                .outcomes
                .iter()
                .find(|outcome| outcome.action_id == action.action_id);
            entries.push((action.action_id.clone(), success_check(action, outcome)));
            if let Some(outcome) = outcome {
                for check in write_checks(action, outcome, root) {
                    entries.push((action.action_id.clone(), check));
                }
            }
            entries.push((
                action.action_id.clone(),
                self.target_check(action, root).await,
            ));
        }
        entries.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.name.cmp(&right.1.name))
        });
        entries
    }
}
/// Makes `path` relative to `root`, rejecting `..` and escapes.
///
/// # Errors
///
/// Returns the reason as a `String` when the path is absolute outside `root`,
/// contains a `..` component, or cannot be made relative.
fn normalize_path(path: &Path, root: &Path) -> std::result::Result<PathBuf, String> {
    let relative = if path.is_absolute() {
        path.strip_prefix(root)
            .map_err(|_| format!("`{}` is outside the workspace root", path.display()))?
            .to_path_buf()
    } else {
        path.to_path_buf()
    };
    let mut normalized = PathBuf::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!("`{}` escapes via `..`", path.display()));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("`{}` is not a relative path", path.display()));
            }
        }
    }
    Ok(normalized)
}
/// Whether `written` is `declared` or sits inside it.
fn is_within(written: &Path, declared: &Path) -> bool {
    written.starts_with(declared)
}
/// Builds the write-path checks for one action and its outcome.
fn write_checks(
    action: &super::plan::OptimizationAction,
    outcome: &ActionOutcome,
    root: &Path,
) -> Vec<VerificationCheck> {
    let declared_entries: Vec<&str> = action
        .declared_write_paths
        .iter()
        .map(|path| path.trim())
        .filter(|path| !path.is_empty())
        .collect();
    if declared_entries.is_empty() {
        return vec![VerificationCheck::failed(
            CHECK_NO_DECLARED_WRITE_PATHS,
            format!("action `{}` declares no writable path", action.action_id),
        )];
    }
    let mut declared: Vec<PathBuf> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for entry in declared_entries {
        match normalize_path(Path::new(entry), root) {
            Ok(relative) if relative.as_os_str().is_empty() => {
                failures.push(format!("declared write path `{entry}` is empty"));
            }
            Ok(relative) => declared.push(relative),
            Err(reason) => failures.push(reason),
        }
    }
    for written in &outcome.written_paths {
        match normalize_path(written, root) {
            Ok(relative) => {
                if !declared.iter().any(|prefix| is_within(&relative, prefix)) {
                    failures.push(format!(
                        "`{}` is not inside a declared write path",
                        written.display()
                    ));
                }
            }
            Err(reason) => failures.push(reason),
        }
    }
    if failures.is_empty() {
        vec![VerificationCheck::passed(
            CHECK_DECLARED_WRITE_PATHS,
            format!(
                "{} written path(s) within declared prefixes",
                outcome.written_paths.len()
            ),
        )]
    } else {
        vec![VerificationCheck::failed(
            CHECK_DECLARED_WRITE_PATHS,
            failures.join("; "),
        )]
    }
}
/// Builds the outcome check for one action.
fn success_check(
    action: &super::plan::OptimizationAction,
    outcome: Option<&ActionOutcome>,
) -> VerificationCheck {
    match outcome {
        None => VerificationCheck::failed(
            CHECK_ACTION_SUCCEEDED,
            format!("no execution outcome recorded for `{}`", action.action_id),
        ),
        Some(outcome) if outcome.success => VerificationCheck::passed(
            CHECK_ACTION_SUCCEEDED,
            format!("`{}` reported success", action.action_id),
        ),
        Some(outcome) => VerificationCheck::failed(
            CHECK_ACTION_SUCCEEDED,
            outcome
                .error
                .clone()
                .unwrap_or_else(|| format!("`{}` reported failure", action.action_id)),
        ),
    }
}
#[async_trait]
impl ActionVerifier for StaticActionVerifier {
    fn name(&self) -> &str {
        VERIFIER_NAME
    }
    async fn verify(&self, request: &VerificationRequest) -> Result<MemberVerificationResult> {
        let entries = self.collect_checks(request).await;
        let (checked_roles, role_results) = group_by_role(&request.plan, &entries);
        let checks: Vec<VerificationCheck> =
            entries.iter().map(|(_, check)| check.clone()).collect();
        let (status, repairable, failed_count, write_violation) = {
            let failed_names: BTreeSet<&str> = checks
                .iter()
                .filter(|check| check.status.is_failed())
                .map(|check| check.name.as_str())
                .collect();
            let write_violation = failed_names.contains(CHECK_DECLARED_WRITE_PATHS)
                || failed_names.contains(CHECK_NO_DECLARED_WRITE_PATHS);
            let repairable = !write_violation
                && (failed_names.contains(CHECK_ACTION_SUCCEEDED)
                    || failed_names.contains(CHECK_TARGET_EXISTS));
            let status = if failed_names.is_empty() {
                VerificationStatus::Passed
            } else {
                VerificationStatus::Failed
            };
            (status, repairable, failed_names.len(), write_violation)
        };
        let mut metadata = BTreeMap::new();
        metadata.insert("verifier".to_owned(), Value::from(VERIFIER_NAME));
        metadata.insert(
            "actions_checked".to_owned(),
            Value::from(request.plan.actions.len()),
        );
        metadata.insert("checks_failed".to_owned(), Value::from(failed_count));
        metadata.insert("write_violation".to_owned(), Value::from(write_violation));
        Ok(MemberVerificationResult {
            status,
            checked_roles,
            checks,
            role_results,
            repairable,
            metadata,
        })
    }
}

/// Groups sorted check entries by action role and reports the verified roles.
fn group_by_role(
    plan: &OptimizationPlan,
    entries: &[(String, VerificationCheck)],
) -> (Vec<String>, BTreeMap<String, RoleVerification>) {
    let role_by_action: BTreeMap<&str, &str> = plan
        .actions
        .iter()
        .map(|action| (action.action_id.as_str(), action.role.as_str()))
        .collect();
    let mut role_checks: BTreeMap<String, Vec<VerificationCheck>> = BTreeMap::new();
    for (action_id, check) in entries {
        let role = role_by_action
            .get(action_id.as_str())
            .copied()
            .unwrap_or_default();
        role_checks
            .entry(role.to_owned())
            .or_default()
            .push(check.clone());
    }
    let checked_roles: Vec<String> = if role_checks.is_empty() {
        plan.targets
            .iter()
            .map(|target| target.target.role.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    } else {
        role_checks.keys().cloned().collect()
    };
    let role_results: BTreeMap<String, RoleVerification> = role_checks
        .into_iter()
        .map(|(role, checks)| {
            let passed = checks.iter().all(|check| !check.status.is_failed());
            let result = RoleVerification {
                role: role.clone(),
                checks,
                passed,
            };
            (role, result)
        })
        .collect();
    (checked_roles, role_results)
}
