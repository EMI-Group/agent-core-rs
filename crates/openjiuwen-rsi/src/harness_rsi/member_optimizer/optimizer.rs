//! The one-pass member optimisation loop: select → plan → execute → verify.
//!
//! Python's `member_optimizer/optimizer.py` runs a full pipeline over LLM agents
//! and git worktrees: role attribution → mechanism attribution → selection →
//! LLM planning → one `DeepAgent` executor per action → worktree verification and
//! repair → publish. The Rust port keeps the pipeline *shape* and the parts that
//! are deterministic:
//!
//! ```text
//! targets --select--> plan --execute (wave by wave)--> outcomes --verify-->
//!     repairable? --re-plan once--> execute + verify --> status
//! ```
//!
//! Only the collaborators are seams: [`ActionPlanner`],
//! [`ActionExecutor`], and [`ActionVerifier`]. The
//! default implementations ([`DeterministicActionExecutor`],
//! [`DeterministicActionPlanner`],
//! [`StaticActionVerifier`]) are offline
//! stand-ins that make the loop testable end to end without a model, a
//! subprocess, or a worktree.
use super::config::MemberOptimizerConfig;
use super::plan::{
    plan_waves, ActionPlanner, DeterministicActionPlanner, OptimizationAction, OptimizationPlan,
    PlanRequest,
};
use super::selection::{select_targets, MemberTarget, SelectedTarget};
use super::verify::{
    ActionOutcome, ActionVerifier, FsWorkspaceProbe, MemberVerificationResult,
    StaticActionVerifier, VerificationRequest, VerificationStatus, WorkspaceProbe,
};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
/// Name reported by [`DeterministicActionExecutor`].
pub const EXECUTOR_NAME: &str = "deterministic_action_executor";
/// One member-optimisation request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberOptimizationRequest {
    /// Id of this optimisation run; action ids derive from it.
    pub optimization_id: String,
    /// Directory artifacts would be written to by a caller-side publisher
    /// (this crate never writes it).
    pub output_dir: PathBuf,
    /// Candidate targets, before selection.
    pub targets: Vec<MemberTarget>,
    /// Root the plan's relative paths resolve against.
    pub workspace_root: PathBuf,
    /// Action groups the run may use; empty means "all allowed".
    pub allowlist: Vec<String>,
    /// Caller annotations, copied into the artifact metadata.
    pub metadata: BTreeMap<String, Value>,
}
/// Terminal status of one member-optimisation run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberOptimizationStatus {
    /// Every verification check passed.
    Success,
    /// Verification failed for some roles but at least one role passed.
    PartialSuccess,
    /// Verification failed for every checked role.
    Failed,
    /// Python parity value: the run was skipped before execution.
    #[default]
    Skipped,
    /// Python parity value: the analysis found no issues at all.
    NoIssues,
    /// Selection produced no target (all filtered or the list was empty).
    NoTargets,
    /// The planner rejected the plan (a validation failure).
    PlanningRejected,
    /// The planner produced a plan with no actions.
    NoActions,
}
impl MemberOptimizationStatus {
    /// Returns the `snake_case` name used in artifacts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::PartialSuccess => "partial_success",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::NoIssues => "no_issues",
            Self::NoTargets => "no_targets",
            Self::PlanningRejected => "planning_rejected",
            Self::NoActions => "no_actions",
        }
    }
    /// Whether the run ended before a plan was executed.
    #[must_use]
    pub const fn is_terminal_without_execution(self) -> bool {
        matches!(
            self,
            Self::Skipped
                | Self::NoIssues
                | Self::NoTargets
                | Self::PlanningRejected
                | Self::NoActions
        )
    }
}
impl std::fmt::Display for MemberOptimizationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
/// Promotion state of a member optimisation run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionStatus {
    /// The run succeeded and awaits a caller-side promotion gate.
    PendingGate,
    /// The run did not succeed, so nothing is promotable.
    #[default]
    NotEvaluated,
    /// The run never produced changes to promote.
    NotApplicable,
}
impl PromotionStatus {
    /// Returns the `snake_case` name used in artifacts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PendingGate => "pending_gate",
            Self::NotEvaluated => "not_evaluated",
            Self::NotApplicable => "not_applicable",
        }
    }
}
/// The artifact one member-optimisation pass produces.
///
/// This is a data document, not a directory: a caller publishes it (Python
/// writes `member_optimization.yaml` plus per-role worktrees) and
/// `optimization_id`/`output_dir` are carried so it can do so deterministically.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberOptimizationArtifact {
    /// Id of the run.
    pub optimization_id: String,
    /// Directory the caller should publish into.
    pub output_dir: PathBuf,
    /// Terminal status.
    pub status: MemberOptimizationStatus,
    /// The plan that was executed last, when one was produced.
    pub plan: Option<OptimizationPlan>,
    /// The final verification, when execution ran.
    pub verification: Option<MemberVerificationResult>,
    /// Promotion state.
    pub promotion_status: PromotionStatus,
    /// Selected targets, best first.
    pub targets: Vec<SelectedTarget>,
    /// Number of execution rounds (initial plan plus repairs).
    pub rounds: u32,
    /// Run metadata (planner/executor/verifier names, failure details, …).
    pub metadata: BTreeMap<String, Value>,
}
/// One execution wave of one plan round.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExecutionRequest {
    /// The plan the wave belongs to.
    pub plan: OptimizationPlan,
    /// Root the plan's relative paths resolve against.
    pub workspace_root: PathBuf,
    /// The action ids to execute, in order.
    pub wave: Vec<String>,
    /// `0` for the initial plan, `n` for the n-th repair round.
    pub round: u32,
    /// Action groups the run may use; empty means "all allowed".
    pub allowlist: Vec<String>,
}
/// Executes one wave of a plan.
///
/// Python runs one `DeepAgent` per action inside an isolated git worktree
/// (`action_executor.py`). That machinery is the caller's: implement this trait
/// with a real agent run and return one [`ActionOutcome`] per action.
#[async_trait]
pub trait ActionExecutor: Send + Sync {
    /// Returns the executor's human-readable name.
    fn name(&self) -> &str;
    /// Executes `request`'s wave and reports one outcome per action.
    ///
    /// # Errors
    ///
    /// Returns an [`openjiuwen_common::error::Error`] for infrastructure failures
    /// (unwritable workspace, missing action). A failed *action* is reported as
    /// [`ActionOutcome::success`] `false`, not as an error.
    async fn execute(&self, request: &ExecutionRequest) -> Result<Vec<ActionOutcome>>;
}
/// Deterministic offline stand-in for Python's per-action `DeepAgent` executor.
///
/// For every action in the wave it creates the action's declared write prefixes as
/// directories under the workspace root and writes a small JSON execution record at
/// the action's `target_path`. No subprocess, no network, no model: the point is to
/// produce the files a static verification can observe (the target file exists, and
/// the write stays inside a declared prefix).
///
/// Paths that are absolute or contain `..` are rejected with
/// [`openjiuwen_common::error::Error::Validation`], so the stand-in can never
/// write outside the workspace root.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeterministicActionExecutor;
impl DeterministicActionExecutor {
    /// Creates the executor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}
/// Resolves one declared write path under `root`.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Validation`] when the declared path
/// is absolute or contains a `..` component.
fn resolve_write_path(root: &Path, declared: &str) -> Result<Option<PathBuf>> {
    let trimmed = declared.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let relative = Path::new(trimmed);
    if relative.is_absolute() {
        return Err(Error::validation(format!(
            "declared write path `{trimmed}` must be relative to the workspace root"
        )));
    }
    let mut path = root.to_path_buf();
    for component in relative.components() {
        match component {
            Component::Normal(part) => path.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(Error::validation(format!(
                    "declared write path `{trimmed}` escapes the workspace root"
                )));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(Error::validation(format!(
                    "declared write path `{trimmed}` is not a relative path"
                )));
            }
        }
    }
    Ok(Some(path))
}
#[async_trait]
impl ActionExecutor for DeterministicActionExecutor {
    fn name(&self) -> &str {
        EXECUTOR_NAME
    }
    async fn execute(&self, request: &ExecutionRequest) -> Result<Vec<ActionOutcome>> {
        let mut outcomes = Vec::with_capacity(request.wave.len());
        for action_id in &request.wave {
            let action = request
                .plan
                .actions
                .iter()
                .find(|action| &action.action_id == action_id)
                .ok_or_else(|| {
                    Error::validation(format!(
                        "execution wave references unknown action `{action_id}`"
                    ))
                })?;
            let record = serde_json::json!({
                "action_id": action.action_id,
                "role": action.role,
                "operation": action.operation,
                "target_path": action.target_path,
                "round": request.round,
                "executor": EXECUTOR_NAME,
            });
            let payload = format!(
                "{}\n",
                serde_json::to_string_pretty(&record).map_err(|error| Error::sys_op(format!(
                    "failed to encode execution record: {error}"
                )))?
            );
            let target = resolve_write_path(&request.workspace_root, &action.target_path)?;
            for entry in &action.declared_write_paths {
                let Some(prefix) = resolve_write_path(&request.workspace_root, entry)? else {
                    continue;
                };
                if target.as_ref() == Some(&prefix) {
                    continue;
                }
                tokio::fs::create_dir_all(&prefix).await.map_err(|error| {
                    Error::sys_op(format!("failed to create `{}`: {error}", prefix.display()))
                })?;
            }
            let mut written: Vec<PathBuf> = Vec::new();
            if let Some(path) = target {
                if let Some(parent) = path.parent() {
                    tokio::fs::create_dir_all(parent).await.map_err(|error| {
                        Error::sys_op(format!("failed to create `{}`: {error}", parent.display()))
                    })?;
                }
                tokio::fs::write(&path, payload.as_bytes())
                    .await
                    .map_err(|error| {
                        Error::sys_op(format!("failed to write `{}`: {error}", path.display()))
                    })?;
                written.push(path);
            }
            outcomes.push(ActionOutcome {
                action_id: action.action_id.clone(),
                success: true,
                output: format!(
                    "materialised {} path(s) for `{}`",
                    written.len(),
                    action.action_id
                ),
                written_paths: written,
                error: None,
            });
        }
        Ok(outcomes)
    }
}
impl std::fmt::Display for PromotionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
/// Runs the select → plan → execute → verify pass.
///
/// Defaults are offline: a [`DeterministicActionPlanner`] with an empty
/// definition library (so a run without a supplied planner ends in
/// [`MemberOptimizationStatus::NoActions`]), a [`DeterministicActionExecutor`],
/// and a [`StaticActionVerifier`] over [`FsWorkspaceProbe`].
#[derive(Clone)]
pub struct MemberOptimizer {
    /// Limits and thresholds.
    config: MemberOptimizerConfig,
    /// Optional planner override.
    planner: Option<Arc<dyn ActionPlanner>>,
    /// Optional executor override.
    executor: Option<Arc<dyn ActionExecutor>>,
    /// Optional verifier override.
    verifier: Option<Arc<dyn ActionVerifier>>,
    /// Optional probe override, used when no verifier is supplied.
    probe: Option<Arc<dyn WorkspaceProbe>>,
}
impl MemberOptimizer {
    /// Creates an optimiser with `config`.
    #[must_use]
    pub fn new(config: MemberOptimizerConfig) -> Self {
        Self {
            config,
            planner: None,
            executor: None,
            verifier: None,
            probe: None,
        }
    }
    /// Overrides the planner.
    #[must_use]
    pub fn with_planner(mut self, planner: Arc<dyn ActionPlanner>) -> Self {
        self.planner = Some(planner);
        self
    }
    /// Overrides the executor.
    #[must_use]
    pub fn with_executor(mut self, executor: Arc<dyn ActionExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }
    /// Overrides the verifier.
    #[must_use]
    pub fn with_verifier(mut self, verifier: Arc<dyn ActionVerifier>) -> Self {
        self.verifier = Some(verifier);
        self
    }
    /// Overrides the workspace probe used by the default verifier.
    #[must_use]
    pub fn with_probe(mut self, probe: Arc<dyn WorkspaceProbe>) -> Self {
        self.probe = Some(probe);
        self
    }
    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &MemberOptimizerConfig {
        &self.config
    }
    /// Resolves the planner, falling back to the deterministic stand-in.
    fn resolve_planner(&self) -> Arc<dyn ActionPlanner> {
        self.planner.clone().unwrap_or_else(|| {
            Arc::new(
                DeterministicActionPlanner::new(Vec::new())
                    .with_max_actions(self.config.max_actions_cap()),
            )
        })
    }
    /// Resolves the executor, falling back to the deterministic stand-in.
    fn resolve_executor(&self) -> Arc<dyn ActionExecutor> {
        self.executor
            .clone()
            .unwrap_or_else(|| Arc::new(DeterministicActionExecutor::new()))
    }
    /// Resolves the probe, falling back to the filesystem.
    fn resolve_probe(&self) -> Arc<dyn WorkspaceProbe> {
        self.probe
            .clone()
            .unwrap_or_else(|| Arc::new(FsWorkspaceProbe::new()))
    }
    /// Resolves the verifier, falling back to the static verifier.
    fn resolve_verifier(&self) -> Arc<dyn ActionVerifier> {
        self.verifier
            .clone()
            .unwrap_or_else(|| Arc::new(StaticActionVerifier::new(self.resolve_probe())))
    }
    /// Executes every wave of `plan`, appending the outcomes to `outcomes`.
    async fn execute_plan(
        &self,
        executor: &dyn ActionExecutor,
        plan: &OptimizationPlan,
        round: u32,
        request: &MemberOptimizationRequest,
        outcomes: &mut Vec<ActionOutcome>,
    ) -> Result<()> {
        for wave in &plan.action_waves {
            let executed = executor
                .execute(&ExecutionRequest {
                    plan: plan.clone(),
                    workspace_root: request.workspace_root.clone(),
                    wave: wave.clone(),
                    round,
                    allowlist: request.allowlist.clone(),
                })
                .await?;
            outcomes.extend(executed);
        }
        Ok(())
    }
    /// Builds the repair plan for the roles that failed verification.
    ///
    /// Returns `Ok(None)` when no role failed, when the planner rejects the
    /// repair plan, or when the repair plan has no actions: repair is best-effort
    /// and never turns a verification failure into an infrastructure error.
    async fn repair_plan(
        &self,
        planner: &dyn ActionPlanner,
        optimization_id: &str,
        current: &OptimizationPlan,
        verification: &MemberVerificationResult,
        next_round: u32,
        allowlist: &[String],
    ) -> Result<Option<OptimizationPlan>> {
        let failed: BTreeSet<&str> = verification
            .role_results
            .iter()
            .filter(|(_, result)| !result.passed)
            .map(|(role, _)| role.as_str())
            .collect();
        let targets: Vec<SelectedTarget> = current
            .targets
            .iter()
            .filter(|target| failed.is_empty() || failed.contains(target.target.role.as_str()))
            .cloned()
            .collect();
        if targets.is_empty() {
            return Ok(None);
        }
        let request = PlanRequest {
            optimization_id: optimization_id.to_owned(),
            targets,
            round: next_round,
            allowlist: allowlist.to_vec(),
        };
        match planner.plan(&request).await {
            Ok(plan) if !plan.actions.is_empty() => Ok(Some(plan)),
            Ok(_) => Ok(None),
            Err(error) if error.kind() == ErrorKind::Validation => {
                tracing::warn!(
                    target: "openjiuwen_rsi::member_optimizer",
                    optimization_id,
                    round = next_round,
                    error = %error,
                    "repair plan rejected; keeping the failed verification",
                );
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
    /// Runs one full member-optimisation pass.
    ///
    /// The pass validates the config, selects targets, plans once, executes wave
    /// by wave, verifies, and — when verification is repairable and
    /// [`MemberOptimizerConfig::max_repair_rounds`] allows it — re-plans the
    /// failed roles and executes and verifies that round once more.
    ///
    /// # Errors
    ///
    /// - [`openjiuwen_common::error::Error::Config`] when the configuration is
    ///   invalid.
    /// - [`openjiuwen_common::error::Error::Validation`] when
    ///   `fail_on_unverifiable` is set and a non-repairable verification failure
    ///   ends the run.
    /// - Any planner/executor/verifier infrastructure error, propagated unchanged.
    ///   A planner *validation* failure is not propagated: it ends the run with
    ///   [`MemberOptimizationStatus::PlanningRejected`] and the error text in the
    ///   artifact metadata.
    #[allow(clippy::too_many_lines)] // the pass is one narrative; helpers carry the detail
    pub async fn optimize(
        &self,
        request: MemberOptimizationRequest,
    ) -> Result<MemberOptimizationArtifact> {
        self.config.validate()?;
        let selected = select_targets(
            &request.targets,
            self.config.max_roles_per_run,
            self.config.min_confidence,
        );
        tracing::debug!(
            target: "openjiuwen_rsi::member_optimizer",
            optimization_id = %request.optimization_id,
            candidates = request.targets.len(),
            selected = selected.len(),
            "selected member optimisation targets",
        );
        if selected.is_empty() {
            return Ok(Self::build_artifact(
                &request,
                MemberOptimizationStatus::NoTargets,
                None,
                None,
                selected,
                0,
                request.metadata.clone(),
            ));
        }
        let planner = self.resolve_planner();
        let plan_request = PlanRequest {
            optimization_id: request.optimization_id.clone(),
            targets: selected.clone(),
            round: 0,
            allowlist: request.allowlist.clone(),
        };
        let plan = match planner.plan(&plan_request).await {
            Ok(plan) => plan,
            Err(error) if error.kind() == ErrorKind::Validation => {
                tracing::warn!(
                    target: "openjiuwen_rsi::member_optimizer",
                    optimization_id = %request.optimization_id,
                    error = %error,
                    "planning rejected",
                );
                let mut metadata = request.metadata.clone();
                metadata.insert("error".to_owned(), Value::from(error.to_string()));
                metadata.insert("planner".to_owned(), Value::from(planner.name()));
                metadata.insert("stage".to_owned(), Value::from("planning"));
                return Ok(Self::build_artifact(
                    &request,
                    MemberOptimizationStatus::PlanningRejected,
                    None,
                    None,
                    selected,
                    0,
                    metadata,
                ));
            }
            Err(error) => return Err(error),
        };
        if plan.actions.is_empty() {
            let mut metadata = request.metadata.clone();
            metadata.insert("planner".to_owned(), Value::from(planner.name()));
            return Ok(Self::build_artifact(
                &request,
                MemberOptimizationStatus::NoActions,
                Some(plan),
                None,
                selected,
                0,
                metadata,
            ));
        }
        let executor = self.resolve_executor();
        let verifier = self.resolve_verifier();
        let repair_budget = u32::try_from(self.config.max_repair_rounds).unwrap_or(u32::MAX);
        let mut outcomes: Vec<ActionOutcome> = Vec::new();
        let mut retained: Vec<OptimizationAction> = Vec::new();
        let mut current = plan;
        let mut repair_rounds_used: u32 = 0;
        let (verification, merged) = loop {
            self.execute_plan(
                executor.as_ref(),
                &current,
                repair_rounds_used,
                &request,
                &mut outcomes,
            )
            .await?;
            let merged = compose_plan(&current, &retained)?;
            let verification = verifier
                .verify(&VerificationRequest {
                    plan: merged.clone(),
                    outcomes: outcomes.clone(),
                    workspace_root: request.workspace_root.clone(),
                })
                .await?;
            if !(verification.repairable && repair_rounds_used < repair_budget) {
                break (verification, merged);
            }
            let next_round = repair_rounds_used + 1;
            let Some(repaired) = self
                .repair_plan(
                    planner.as_ref(),
                    &request.optimization_id,
                    &current,
                    &verification,
                    next_round,
                    &request.allowlist,
                )
                .await?
            else {
                break (verification, merged);
            };
            let repaired_roles: BTreeSet<&str> = repaired
                .actions
                .iter()
                .map(|action| action.role.as_str())
                .collect();
            retained.extend(
                current
                    .actions
                    .iter()
                    .filter(|action| !repaired_roles.contains(action.role.as_str()))
                    .cloned(),
            );
            repair_rounds_used = next_round;
            current = repaired;
        };
        let status = status_for(&verification);
        if self.config.fail_on_unverifiable
            && status != MemberOptimizationStatus::Success
            && !verification.repairable
        {
            let failed = verification
                .checks
                .iter()
                .filter(|check| check.status.is_failed())
                .count();
            return Err(Error::validation(format!(
                "member optimisation `{}` failed static verification with {failed} failed check(s)",
                request.optimization_id
            )));
        }
        let mut metadata = request.metadata.clone();
        metadata.insert("planner".to_owned(), Value::from(planner.name()));
        metadata.insert("executor".to_owned(), Value::from(executor.name()));
        metadata.insert("verifier".to_owned(), Value::from(verifier.name()));
        metadata.insert(
            "execution_rounds".to_owned(),
            Value::from(repair_rounds_used + 1),
        );
        metadata.insert("repair_rounds".to_owned(), Value::from(repair_rounds_used));
        metadata.insert("outcomes".to_owned(), Value::from(outcomes.len()));
        Ok(Self::build_artifact(
            &request,
            status,
            Some(merged),
            Some(verification),
            selected,
            repair_rounds_used + 1,
            metadata,
        ))
    }
    /// Assembles an artifact from the parts of a finished run.
    #[allow(clippy::too_many_arguments)] // one private assembler for every terminal path
    fn build_artifact(
        request: &MemberOptimizationRequest,
        status: MemberOptimizationStatus,
        plan: Option<OptimizationPlan>,
        verification: Option<MemberVerificationResult>,
        targets: Vec<SelectedTarget>,
        rounds: u32,
        metadata: BTreeMap<String, Value>,
    ) -> MemberOptimizationArtifact {
        MemberOptimizationArtifact {
            optimization_id: request.optimization_id.clone(),
            output_dir: request.output_dir.clone(),
            status,
            plan,
            verification,
            promotion_status: promotion_for(status),
            targets,
            rounds,
            metadata,
        }
    }
}
/// Maps a verification result to a terminal status.
fn status_for(verification: &MemberVerificationResult) -> MemberOptimizationStatus {
    if verification.status == VerificationStatus::Passed {
        return MemberOptimizationStatus::Success;
    }
    if verification
        .role_results
        .values()
        .any(|result| result.passed)
    {
        MemberOptimizationStatus::PartialSuccess
    } else {
        MemberOptimizationStatus::Failed
    }
}
/// Maps a terminal status to its promotion state.
fn promotion_for(status: MemberOptimizationStatus) -> PromotionStatus {
    if status == MemberOptimizationStatus::Success {
        return PromotionStatus::PendingGate;
    }
    if status.is_terminal_without_execution() {
        PromotionStatus::NotApplicable
    } else {
        PromotionStatus::NotEvaluated
    }
}
/// Composes the verified plan: retained actions from earlier rounds plus the
/// current round's actions.
///
/// A repair round re-plans the roles that failed, so those roles' earlier actions
/// are dropped and their repair actions take their place; the actions of roles
/// that already passed are kept. The composition makes the final verification
/// read as one plan with one wave schedule.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Validation`] when the composed
/// actions cannot be scheduled (only reachable through an id collision).
fn compose_plan(
    current: &OptimizationPlan,
    retained: &[OptimizationAction],
) -> Result<OptimizationPlan> {
    let actions: Vec<OptimizationAction> = retained
        .iter()
        .cloned()
        .chain(current.actions.iter().cloned())
        .collect();
    let action_waves = plan_waves(&actions)?;
    let mut metadata = current.metadata.clone();
    metadata.insert("retained_actions".to_owned(), Value::from(retained.len()));
    Ok(OptimizationPlan {
        plan_id: current.plan_id.clone(),
        targets: current.targets.clone(),
        actions,
        action_waves,
        metadata,
    })
}
