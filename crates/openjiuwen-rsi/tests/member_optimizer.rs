#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Offline integration suite for team-member optimisation
//! (`openjiuwen_rsi::harness_rsi::member_optimizer`).
//!
//! Python parity: `openjiuwen/rsi/harness_rsi/member_optimizer` implements the
//! same pass through LLM attributors, an LLM planner, one `DeepAgent` executor per
//! action, and a repair agent. What is portable — the deterministic selection
//! formula, action typing, dependency waves, and static verification — is pinned
//! here against hand-computed values; everything model-driven is replaced by the
//! documented offline stand-ins so the suite stays deterministic and offline.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use openjiuwen_rsi::harness_rsi::member_optimizer::{
    plan_waves, rank_targets, select_targets, selection_score, ActionDefinition, ActionExecutor,
    ActionOutcome, ActionPlanner, ActionVerifier, AttributionInput, CheckStatus,
    DeterministicActionExecutor, DeterministicActionPlanner, ExecutionRequest,
    InMemoryWorkspaceProbe, IssueObservation, MemberOptimizationRequest, MemberOptimizationStatus,
    MemberOptimizer, MemberOptimizerConfig, MemberTarget, OptimizationAction, OptimizationPlan,
    PlanRequest, PromotionStatus, RunIf, SelectedTarget, Severity, StaticActionVerifier,
    SynthesizedAttributor, VerificationCheck, VerificationRequest, VerificationStatus,
    CHECK_ACTION_SUCCEEDED, CHECK_DECLARED_WRITE_PATHS, CHECK_NO_DECLARED_WRITE_PATHS,
    CHECK_TARGET_EXISTS,
};
use openjiuwen_rsi::{Error, ErrorKind, Result};

/// Tolerance for the hand-computed `f64` selection scores.
const EPSILON: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Builds a target with the given role, severity, confidence, and eligible actions.
fn target(
    role: &str,
    severity: Severity,
    confidence: f64,
    eligible_actions: &[&str],
) -> MemberTarget {
    MemberTarget {
        role: role.to_owned(),
        severity,
        confidence,
        eligible_actions: eligible_actions
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        issue_ids: vec![format!("{role}-issue")],
        member_name: Some(format!("{role}-member")),
        metadata: BTreeMap::new(),
    }
}

/// Wraps `target` with its hand-computed score.
fn selected(target: MemberTarget) -> SelectedTarget {
    SelectedTarget {
        score: selection_score(&target),
        target,
    }
}

/// Builds an action with an id, a role, and explicit dependencies.
fn action(action_id: &str, role: &str, depends_on: &[&str]) -> OptimizationAction {
    OptimizationAction {
        action_id: action_id.to_owned(),
        role: role.to_owned(),
        action_group: "prompt".to_owned(),
        operation: "modify".to_owned(),
        action_type: "edit".to_owned(),
        target_path: format!("members/{role}/prompt.md"),
        depends_on: depends_on.iter().map(|id| (*id).to_owned()).collect(),
        run_if: RunIf::DependencySucceeded,
        declared_write_paths: vec![format!("members/{role}/")],
        ..OptimizationAction::default()
    }
}

/// Builds an action definition targeting `members/<role>/prompt.md`.
fn definition(name: &str, role: &str) -> ActionDefinition {
    ActionDefinition {
        name: name.to_owned(),
        action_group: "prompt".to_owned(),
        operation: "modify".to_owned(),
        action_type: "edit".to_owned(),
        target_path: format!("members/{role}/prompt.md"),
        description: format!("update {role} prompt"),
        expected_effect: "fewer failed runs".to_owned(),
        declared_write_paths: vec![format!("members/{role}/")],
        ..ActionDefinition::default()
    }
}

/// Builds a plan whose waves are derived from its actions.
fn plan_of(actions: Vec<OptimizationAction>) -> OptimizationPlan {
    let action_waves = plan_waves(&actions).expect("fixture actions must be schedulable");
    OptimizationPlan {
        plan_id: "fixture-plan".to_owned(),
        actions,
        action_waves,
        ..OptimizationPlan::default()
    }
}

/// Builds a verification request for `actions` and `outcomes` under `root`.
fn verification_request(
    actions: Vec<OptimizationAction>,
    outcomes: Vec<ActionOutcome>,
    root: &str,
) -> VerificationRequest {
    VerificationRequest {
        plan: plan_of(actions),
        outcomes,
        workspace_root: PathBuf::from(root),
    }
}

/// The name of every check in `checks`, in order.
fn check_names(checks: &[VerificationCheck]) -> Vec<&str> {
    checks.iter().map(|check| check.name.as_str()).collect()
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

#[test]
fn selection_score_matches_the_python_formula() {
    // critical = 4*10, confidence 0.5 = 2.5, three eligible actions = 3/5.
    let scored = target("implement", Severity::Critical, 0.5, &["a", "b", "c"]);
    assert!((selection_score(&scored) - 43.1).abs() < EPSILON);
    // The issue term caps at five eligible actions.
    let capped = target(
        "verify",
        Severity::Low,
        0.0,
        &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    );
    assert!((selection_score(&capped) - 11.0).abs() < EPSILON);
    // Severity weights are critical 4, high 3, medium 2, low 1.
    assert!((Severity::Critical.weight() - 4.0).abs() < EPSILON);
    assert!((Severity::High.weight() - 3.0).abs() < EPSILON);
    assert!((Severity::Medium.weight() - 2.0).abs() < EPSILON);
    assert!((Severity::Low.weight() - 1.0).abs() < EPSILON);
}

#[test]
fn selection_orders_by_score_then_severity_then_role() {
    // beta: 40 + 0 + 0.2 = 40.2, alpha/gamma tie at 30 + 3 + 0.2 = 33.2.
    let targets = vec![
        target("gamma", Severity::High, 0.6, &["a"]),
        target("beta", Severity::Critical, 0.0, &["a"]),
        target("alpha", Severity::High, 0.6, &["a"]),
    ];
    let ranked = rank_targets(&targets, 0.0);
    let roles: Vec<&str> = ranked
        .iter()
        .map(|selected| selected.target.role.as_str())
        .collect();
    assert_eq!(roles, vec!["beta", "alpha", "gamma"]);
    assert!((ranked[0].score - 40.2).abs() < EPSILON);
    assert!((ranked[1].score - 33.2).abs() < EPSILON);
}

#[test]
fn selection_respects_beam_width_and_filters() {
    let targets = vec![
        target("implement", Severity::Critical, 0.9, &["a"]),
        target("verify", Severity::High, 0.6, &["a"]),
        target("summarise", Severity::Low, 0.4, &["a"]),
        target("no-actions", Severity::Critical, 0.9, &[]),
    ];
    // Beam width 1 keeps only the best target.
    let beam = select_targets(&targets, 1, 0.0);
    assert_eq!(beam.len(), 1);
    assert_eq!(beam[0].target.role, "implement");
    // The confidence floor drops the low-confidence target.
    let filtered = select_targets(&targets, 10, 0.5);
    let roles: Vec<&str> = filtered
        .iter()
        .map(|selected| selected.target.role.as_str())
        .collect();
    assert_eq!(roles, vec!["implement", "verify"]);
    // A target without eligible actions is never selectable.
    assert!(rank_targets(&targets, 0.0)
        .iter()
        .all(|selected| selected.target.role != "no-actions"));
}

#[test]
fn rank_targets_keeps_the_tail_that_select_targets_truncates() {
    let targets = vec![
        target("implement", Severity::Critical, 0.9, &["a"]),
        target("verify", Severity::High, 0.6, &["a"]),
        target("summarise", Severity::Low, 0.4, &["a"]),
    ];
    assert_eq!(rank_targets(&targets, 0.0).len(), 3);
    assert_eq!(select_targets(&targets, 2, 0.0).len(), 2);
}

#[test]
fn synthesized_attributor_maps_metadata_roles() {
    let mut eligible = BTreeMap::new();
    eligible.insert("implement".to_owned(), vec!["modify_prompt".to_owned()]);
    let input = AttributionInput {
        issues: vec![
            IssueObservation {
                issue_id: "i1".to_owned(),
                severity: Severity::High,
                confidence: 0.8,
                metadata: BTreeMap::from([("role".to_owned(), json!("implement"))]),
            },
            IssueObservation {
                issue_id: "i2".to_owned(),
                severity: Severity::Critical,
                confidence: 0.4,
                metadata: BTreeMap::from([("role".to_owned(), json!("implement"))]),
            },
            IssueObservation {
                issue_id: "i3".to_owned(),
                severity: Severity::Low,
                confidence: 0.2,
                metadata: BTreeMap::new(),
            },
        ],
        eligible_actions: eligible,
    };
    let result = SynthesizedAttributor::new().attribute(&input);
    assert_eq!(result.targets.len(), 1);
    let attributed = &result.targets[0];
    assert_eq!(attributed.role, "implement");
    assert_eq!(attributed.severity, Severity::Critical);
    assert!((attributed.confidence - 0.6).abs() < EPSILON);
    assert_eq!(attributed.issue_ids, vec!["i1", "i2"]);
    assert_eq!(attributed.eligible_actions, vec!["modify_prompt"]);
    assert_eq!(result.unassigned.len(), 1);
    assert_eq!(result.unassigned[0].issue_id, "i3");
}

// ---------------------------------------------------------------------------
// Waves and planning
// ---------------------------------------------------------------------------

#[test]
fn waves_layer_linear_dependencies() {
    let actions = vec![
        action("a", "implement", &[]),
        action("b", "implement", &["a"]),
        action("c", "implement", &["b"]),
    ];
    assert_eq!(
        plan_waves(&actions).unwrap(),
        vec![
            vec!["a".to_owned()],
            vec!["b".to_owned()],
            vec!["c".to_owned()]
        ]
    );
}

#[test]
fn waves_put_independent_actions_in_one_wave() {
    let actions = vec![
        action("c", "summarise", &[]),
        action("a", "implement", &[]),
        action("b", "verify", &[]),
    ];
    assert_eq!(
        plan_waves(&actions).unwrap(),
        vec![vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]]
    );
}

#[test]
fn waves_reject_unknown_dependencies() {
    let actions = vec![action("a", "implement", &["missing"])];
    let error = plan_waves(&actions).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("unknown action `missing`"));
}

#[test]
fn waves_reject_cycles_and_name_the_remaining_actions() {
    let actions = vec![
        action("a", "implement", &["b"]),
        action("b", "verify", &["a"]),
        action("c", "summarise", &[]),
    ];
    let error = plan_waves(&actions).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("dependency cycle"));
    assert!(message.contains(": a, b"));
}

#[test]
fn waves_reject_duplicate_action_ids() {
    let actions = vec![action("a", "implement", &[]), action("a", "verify", &[])];
    let error = plan_waves(&actions).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("duplicate action id `a`"));
}

/// Builds a planning request for `targets` with no allowlist.
fn plan_request(targets: Vec<SelectedTarget>, round: u32) -> PlanRequest {
    PlanRequest {
        optimization_id: "opt".to_owned(),
        targets,
        round,
        allowlist: Vec::new(),
    }
}

#[tokio::test]
async fn deterministic_planner_chains_actions_per_role() {
    let planner = DeterministicActionPlanner::new(vec![
        definition("modify_prompt", "implement"),
        definition("add_skill", "implement"),
    ]);
    let request = plan_request(
        vec![selected(target(
            "implement",
            Severity::High,
            0.9,
            &["modify_prompt", "add_skill"],
        ))],
        0,
    );
    let plan = planner.plan(&request).await.unwrap();
    assert_eq!(plan.plan_id, "opt/plan/r0");
    assert_eq!(plan.actions.len(), 2);
    assert_eq!(plan.actions[0].action_id, "opt/r0/a0");
    assert_eq!(plan.actions[1].action_id, "opt/r0/a1");
    assert_eq!(plan.actions[0].depends_on.len(), 0);
    assert_eq!(plan.actions[1].depends_on, vec!["opt/r0/a0".to_owned()]);
    assert_eq!(
        plan.action_waves,
        vec![vec!["opt/r0/a0".to_owned()], vec!["opt/r0/a1".to_owned()]]
    );
    assert_eq!(plan.actions[0].run_if, RunIf::DependencySucceeded);
    assert_eq!(
        plan.actions[0].attributed_issue_ids,
        vec!["implement-issue"]
    );
    assert_eq!(
        plan.actions[0].member_name.as_deref(),
        Some("implement-member")
    );
}

#[tokio::test]
async fn deterministic_planner_honours_caps_and_allowlists() {
    let definitions = vec![
        definition("modify_prompt", "implement"),
        definition("add_skill", "implement"),
    ];
    let capped = DeterministicActionPlanner::new(definitions.clone()).with_max_actions(1);
    let request = plan_request(
        vec![selected(target(
            "implement",
            Severity::High,
            0.9,
            &["modify_prompt", "add_skill"],
        ))],
        0,
    );
    assert_eq!(capped.plan(&request).await.unwrap().actions.len(), 1);
    // An allowlist naming an unrelated group filters every definition out.
    let mut restricted = request;
    restricted.allowlist = vec!["tool".to_owned()];
    let planner = DeterministicActionPlanner::new(definitions);
    assert_eq!(planner.plan(&restricted).await.unwrap().actions.len(), 0);
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

#[tokio::test]
async fn verification_passes_a_complete_outcome() {
    let root = "/workspace";
    let action = action("a", "implement", &[]);
    let target_path = PathBuf::from(root).join("members/implement/prompt.md");
    let probe = Arc::new(InMemoryWorkspaceProbe::with_paths([target_path.clone()]));
    let verification = StaticActionVerifier::new(probe)
        .verify(&verification_request(
            vec![action],
            vec![ActionOutcome {
                action_id: "a".to_owned(),
                success: true,
                output: "ok".to_owned(),
                written_paths: vec![target_path],
                error: None,
            }],
            root,
        ))
        .await
        .unwrap();
    assert_eq!(verification.status, VerificationStatus::Passed);
    assert!(!verification.repairable);
    assert_eq!(verification.checked_roles, vec!["implement"]);
    assert_eq!(
        check_names(&verification.checks),
        vec![
            CHECK_ACTION_SUCCEEDED,
            CHECK_DECLARED_WRITE_PATHS,
            CHECK_TARGET_EXISTS
        ]
    );
    assert!(verification.role_results["implement"].passed);
}

#[tokio::test]
async fn verification_flags_a_missing_outcome_as_repairable() {
    let root = "/workspace";
    let target_path = PathBuf::from(root).join("members/implement/prompt.md");
    let probe = Arc::new(InMemoryWorkspaceProbe::with_paths([target_path]));
    let verification = StaticActionVerifier::new(probe)
        .verify(&verification_request(
            vec![action("a", "implement", &[])],
            Vec::new(),
            root,
        ))
        .await
        .unwrap();
    assert_eq!(verification.status, VerificationStatus::Failed);
    assert!(verification.repairable);
    let failed: Vec<&str> = verification
        .checks
        .iter()
        .filter(|check| check.status == CheckStatus::Failed)
        .map(|check| check.name.as_str())
        .collect();
    assert_eq!(failed, vec![CHECK_ACTION_SUCCEEDED]);
}

#[tokio::test]
async fn verification_rejects_undeclared_write_paths() {
    let root = "/workspace";
    let target_path = PathBuf::from(root).join("members/implement/prompt.md");
    let probe = Arc::new(InMemoryWorkspaceProbe::with_paths([target_path]));
    let verification = StaticActionVerifier::new(probe)
        .verify(&verification_request(
            vec![action("a", "implement", &[])],
            vec![ActionOutcome {
                action_id: "a".to_owned(),
                success: true,
                output: String::new(),
                written_paths: vec![PathBuf::from(root).join("elsewhere/secret.txt")],
                error: None,
            }],
            root,
        ))
        .await
        .unwrap();
    assert_eq!(verification.status, VerificationStatus::Failed);
    assert!(!verification.repairable);
    assert_eq!(
        verification.metadata["write_violation"],
        serde_json::Value::from(true)
    );
    let write_check = verification
        .checks
        .iter()
        .find(|check| check.name == CHECK_DECLARED_WRITE_PATHS)
        .unwrap();
    assert_eq!(write_check.status, CheckStatus::Failed);
    assert!(write_check
        .detail
        .contains("not inside a declared write path"));
}

#[tokio::test]
async fn verification_rejects_parent_directory_escapes() {
    let root = "/workspace";
    let target_path = PathBuf::from(root).join("members/implement/prompt.md");
    let probe = Arc::new(InMemoryWorkspaceProbe::with_paths([target_path]));
    let verification = StaticActionVerifier::new(probe)
        .verify(&verification_request(
            vec![action("a", "implement", &[])],
            vec![ActionOutcome {
                action_id: "a".to_owned(),
                success: true,
                output: String::new(),
                written_paths: vec![PathBuf::from("../escape.txt")],
                error: None,
            }],
            root,
        ))
        .await
        .unwrap();
    assert!(!verification.repairable);
    let write_check = verification
        .checks
        .iter()
        .find(|check| check.name == CHECK_DECLARED_WRITE_PATHS)
        .unwrap();
    assert!(write_check.detail.contains("escapes via `..`"));
}

#[tokio::test]
async fn verification_rejects_actions_without_declared_paths() {
    let root = "/workspace";
    let mut action = action("a", "implement", &[]);
    action.declared_write_paths.clear();
    let probe = Arc::new(InMemoryWorkspaceProbe::with_paths([
        PathBuf::from(root).join("members/implement/prompt.md")
    ]));
    let verification = StaticActionVerifier::new(probe)
        .verify(&verification_request(
            vec![action],
            vec![ActionOutcome {
                action_id: "a".to_owned(),
                success: true,
                output: String::new(),
                written_paths: Vec::new(),
                error: None,
            }],
            root,
        ))
        .await
        .unwrap();
    assert!(!verification.repairable);
    assert_eq!(
        check_names(&verification.checks),
        vec![
            CHECK_ACTION_SUCCEEDED,
            CHECK_NO_DECLARED_WRITE_PATHS,
            CHECK_TARGET_EXISTS
        ]
    );
}

#[tokio::test]
async fn verification_flags_a_missing_target_as_repairable() {
    let root = "/workspace";
    let probe = Arc::new(InMemoryWorkspaceProbe::new());
    let verification = StaticActionVerifier::new(probe)
        .verify(&verification_request(
            vec![action("a", "implement", &[])],
            vec![ActionOutcome {
                action_id: "a".to_owned(),
                success: true,
                output: String::new(),
                written_paths: vec![PathBuf::from(root).join("members/implement/prompt.md")],
                error: None,
            }],
            root,
        ))
        .await
        .unwrap();
    assert_eq!(verification.status, VerificationStatus::Failed);
    assert!(verification.repairable);
    let target_check = verification
        .checks
        .iter()
        .find(|check| check.name == CHECK_TARGET_EXISTS)
        .unwrap();
    assert_eq!(target_check.status, CheckStatus::Failed);
}

// ---------------------------------------------------------------------------
// Optimizer end to end
// ---------------------------------------------------------------------------

/// Name reported by [`RepairingExecutor`].
const REPAIRING_EXECUTOR: &str = "repairing_executor";
/// Name reported by [`RejectingPlanner`].
const REJECTING_PLANNER: &str = "rejecting_planner";
/// Executor that fails one role's actions in round `0` and writes every other
/// action's target file, delegating the writing to the deterministic executor.
struct RepairingExecutor {
    /// Role failed in the first round.
    failing_role: String,
}

#[async_trait]
impl ActionExecutor for RepairingExecutor {
    fn name(&self) -> &str {
        REPAIRING_EXECUTOR
    }
    async fn execute(&self, request: &ExecutionRequest) -> Result<Vec<ActionOutcome>> {
        let mut outcomes = Vec::new();
        for action_id in &request.wave {
            let action = request
                .plan
                .actions
                .iter()
                .find(|action| &action.action_id == action_id)
                .expect("wave action must exist");
            if action.role == self.failing_role && request.round == 0 {
                outcomes.push(ActionOutcome {
                    action_id: action_id.clone(),
                    success: false,
                    output: "scripted failure".to_owned(),
                    written_paths: Vec::new(),
                    error: Some("scripted failure".to_owned()),
                });
                continue;
            }
            let mut written = Vec::new();
            for declared in &action.declared_write_paths {
                let path = request.workspace_root.join(declared).join("prompt.md");
                tokio::fs::create_dir_all(path.parent().expect("path has a parent"))
                    .await
                    .unwrap();
                tokio::fs::write(&path, b"ok").await.unwrap();
                written.push(path);
            }
            outcomes.push(ActionOutcome {
                action_id: action_id.clone(),
                success: true,
                output: "written".to_owned(),
                written_paths: written,
                error: None,
            });
        }
        Ok(outcomes)
    }
}

/// Planner that always rejects its plan with a validation error.
struct RejectingPlanner;

#[async_trait]
impl ActionPlanner for RejectingPlanner {
    fn name(&self) -> &str {
        REJECTING_PLANNER
    }
    async fn plan(&self, _request: &PlanRequest) -> Result<OptimizationPlan> {
        Err(Error::validation("no plan possible for these targets"))
    }
}

/// Builds an optimisation request over `targets`.
fn optimization_request(
    root: &std::path::Path,
    targets: Vec<MemberTarget>,
) -> MemberOptimizationRequest {
    MemberOptimizationRequest {
        optimization_id: "opt-1".to_owned(),
        output_dir: root.join("out"),
        targets,
        workspace_root: root.to_path_buf(),
        allowlist: Vec::new(),
        metadata: BTreeMap::from([("suite".to_owned(), json!("member_optimizer"))]),
    }
}

#[tokio::test]
async fn optimizer_reports_no_targets() {
    let temp = tempfile::tempdir().unwrap();
    let optimizer = MemberOptimizer::new(MemberOptimizerConfig::default());
    let artifact = optimizer
        .optimize(optimization_request(temp.path(), Vec::new()))
        .await
        .unwrap();
    assert_eq!(artifact.status, MemberOptimizationStatus::NoTargets);
    assert_eq!(artifact.promotion_status, PromotionStatus::NotApplicable);
    assert_eq!(artifact.rounds, 0);
    assert!(artifact.plan.is_none());
    assert!(artifact.verification.is_none());
}

#[tokio::test]
async fn optimizer_reports_no_actions_when_the_planner_has_no_definitions() {
    let temp = tempfile::tempdir().unwrap();
    let optimizer = MemberOptimizer::new(MemberOptimizerConfig::default())
        .with_planner(Arc::new(DeterministicActionPlanner::new(Vec::new())));
    let artifact = optimizer
        .optimize(optimization_request(
            temp.path(),
            vec![target("implement", Severity::High, 0.9, &["modify_prompt"])],
        ))
        .await
        .unwrap();
    assert_eq!(artifact.status, MemberOptimizationStatus::NoActions);
    assert_eq!(artifact.promotion_status, PromotionStatus::NotApplicable);
    assert_eq!(artifact.rounds, 0);
    let plan = artifact.plan.expect("the planner produced an empty plan");
    assert_eq!(plan.actions.len(), 0);
}

#[tokio::test]
async fn optimizer_runs_the_happy_path_end_to_end() {
    let temp = tempfile::tempdir().unwrap();
    let planner = DeterministicActionPlanner::new(vec![definition("modify_prompt", "implement")]);
    let optimizer = MemberOptimizer::new(MemberOptimizerConfig::default())
        .with_planner(Arc::new(planner))
        .with_executor(Arc::new(DeterministicActionExecutor::new()));
    let artifact = optimizer
        .optimize(optimization_request(
            temp.path(),
            vec![target("implement", Severity::High, 0.9, &["modify_prompt"])],
        ))
        .await
        .unwrap();
    assert_eq!(artifact.status, MemberOptimizationStatus::Success);
    assert_eq!(artifact.promotion_status, PromotionStatus::PendingGate);
    assert_eq!(artifact.rounds, 1);
    assert_eq!(artifact.targets.len(), 1);
    let verification = artifact.verification.expect("verification ran");
    assert_eq!(verification.status, VerificationStatus::Passed);
    assert_eq!(verification.checked_roles, vec!["implement"]);
    let plan = artifact.plan.expect("plan recorded");
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(plan.actions[0].action_id, "opt-1/r0/a0");
    assert_eq!(plan.action_waves, vec![vec!["opt-1/r0/a0".to_owned()]]);
    // The deterministic executor materialised the declared write path.
    assert!(temp.path().join("members/implement/prompt.md").is_file());
    assert_eq!(
        artifact.metadata["executor"],
        json!("deterministic_action_executor")
    );
}

#[tokio::test]
async fn optimizer_reports_partial_success_when_one_role_fails() {
    let temp = tempfile::tempdir().unwrap();
    let config = MemberOptimizerConfig {
        max_repair_rounds: 0,
        ..MemberOptimizerConfig::default()
    };
    let planner = DeterministicActionPlanner::new(vec![
        definition("modify_implement_prompt", "implement"),
        definition("modify_verify_prompt", "verify"),
    ]);
    let optimizer = MemberOptimizer::new(config)
        .with_planner(Arc::new(planner))
        .with_executor(Arc::new(RepairingExecutor {
            failing_role: "verify".to_owned(),
        }));
    let artifact = optimizer
        .optimize(optimization_request(
            temp.path(),
            vec![
                target(
                    "implement",
                    Severity::Critical,
                    0.9,
                    &["modify_implement_prompt"],
                ),
                target("verify", Severity::High, 0.8, &["modify_verify_prompt"]),
            ],
        ))
        .await
        .unwrap();
    assert_eq!(artifact.status, MemberOptimizationStatus::PartialSuccess);
    assert_eq!(artifact.promotion_status, PromotionStatus::NotEvaluated);
    assert_eq!(artifact.rounds, 1);
    let verification = artifact.verification.expect("verification ran");
    assert_eq!(verification.status, VerificationStatus::Failed);
    assert!(verification.role_results["implement"].passed);
    assert!(!verification.role_results["verify"].passed);
}

#[tokio::test]
async fn optimizer_executes_one_repair_round_for_a_repairable_failure() {
    let temp = tempfile::tempdir().unwrap();
    let planner = DeterministicActionPlanner::new(vec![
        definition("modify_implement_prompt", "implement"),
        definition("modify_verify_prompt", "verify"),
    ]);
    let optimizer = MemberOptimizer::new(MemberOptimizerConfig::default())
        .with_planner(Arc::new(planner))
        .with_executor(Arc::new(RepairingExecutor {
            failing_role: "verify".to_owned(),
        }));
    let artifact = optimizer
        .optimize(optimization_request(
            temp.path(),
            vec![
                target(
                    "implement",
                    Severity::Critical,
                    0.9,
                    &["modify_implement_prompt"],
                ),
                target("verify", Severity::High, 0.8, &["modify_verify_prompt"]),
            ],
        ))
        .await
        .unwrap();
    assert_eq!(artifact.status, MemberOptimizationStatus::Success);
    assert_eq!(artifact.rounds, 2);
    assert_eq!(artifact.metadata["repair_rounds"], json!(1));
    let plan = artifact.plan.expect("plan recorded");
    // The failed round-0 `verify` action is replaced by the round-1 repair action.
    let ids: Vec<&str> = plan
        .actions
        .iter()
        .map(|action| action.action_id.as_str())
        .collect();
    assert_eq!(ids, vec!["opt-1/r0/a0", "opt-1/r1/a0"]);
    assert_eq!(plan.actions[1].role, "verify");
    assert_eq!(plan.actions[1].depends_on.len(), 0);
}

#[tokio::test]
async fn optimizer_reports_planning_rejection_without_failing_the_call() {
    let temp = tempfile::tempdir().unwrap();
    let optimizer = MemberOptimizer::new(MemberOptimizerConfig::default())
        .with_planner(Arc::new(RejectingPlanner));
    let artifact = optimizer
        .optimize(optimization_request(
            temp.path(),
            vec![target("implement", Severity::High, 0.9, &["modify_prompt"])],
        ))
        .await
        .unwrap();
    assert_eq!(artifact.status, MemberOptimizationStatus::PlanningRejected);
    assert_eq!(artifact.promotion_status, PromotionStatus::NotApplicable);
    assert!(artifact.plan.is_none());
    let error = artifact.metadata["error"]
        .as_str()
        .expect("planning error recorded");
    assert!(error.contains("no plan possible for these targets"));
    assert_eq!(artifact.metadata["planner"], json!("rejecting_planner"));
}

#[tokio::test]
async fn optimizer_artifact_round_trips_through_json() {
    let temp = tempfile::tempdir().unwrap();
    let planner = DeterministicActionPlanner::new(vec![definition("modify_prompt", "implement")]);
    let optimizer = MemberOptimizer::new(MemberOptimizerConfig::default())
        .with_planner(Arc::new(planner))
        .with_executor(Arc::new(DeterministicActionExecutor::new()));
    let artifact = optimizer
        .optimize(optimization_request(
            temp.path(),
            vec![target("implement", Severity::High, 0.9, &["modify_prompt"])],
        ))
        .await
        .unwrap();
    let encoded = serde_json::to_string(&artifact).unwrap();
    let decoded: openjiuwen_rsi::harness_rsi::member_optimizer::MemberOptimizationArtifact =
        serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, artifact);
    assert!(encoded.contains("\"status\":\"success\""));
    assert!(encoded.contains("\"promotion_status\":\"pending_gate\""));
}

#[test]
fn config_defaults_match_python_and_validate() {
    let config = MemberOptimizerConfig::default();
    assert_eq!(config.max_roles_per_run, 2);
    assert_eq!(config.max_actions_per_plan, 3);
    assert_eq!(config.max_repair_rounds, 1);
    assert!((config.min_confidence - 0.0).abs() < EPSILON);
    assert!(!config.fail_on_unverifiable);
    assert_eq!(config.max_actions_cap(), 3);
    assert!(config.validate().is_ok());
    // `0` means unlimited for the action cap.
    let unlimited = MemberOptimizerConfig {
        max_actions_per_plan: 0,
        ..MemberOptimizerConfig::default()
    };
    assert_eq!(unlimited.max_actions_cap(), usize::MAX);
    // Zero roles, out-of-range confidence, and a non-finite threshold are config errors.
    let no_roles = MemberOptimizerConfig {
        max_roles_per_run: 0,
        ..MemberOptimizerConfig::default()
    };
    assert_eq!(no_roles.validate().unwrap_err().kind(), ErrorKind::Config);
    let bad_confidence = MemberOptimizerConfig {
        min_confidence: 1.5,
        ..MemberOptimizerConfig::default()
    };
    assert_eq!(
        bad_confidence.validate().unwrap_err().kind(),
        ErrorKind::Config
    );
    let nan_confidence = MemberOptimizerConfig {
        min_confidence: f64::NAN,
        ..MemberOptimizerConfig::default()
    };
    assert_eq!(
        nan_confidence.validate().unwrap_err().kind(),
        ErrorKind::Config
    );
}

#[test]
fn config_tolerates_unknown_fields_and_partial_documents() {
    let config: MemberOptimizerConfig = serde_json::from_value(json!({
        "max_roles_per_run": 1,
        "future_knob": true,
    }))
    .unwrap();
    assert_eq!(config.max_roles_per_run, 1);
    assert_eq!(config.max_actions_per_plan, 3);
    let defaulted: MemberOptimizerConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(defaulted, MemberOptimizerConfig::default());
}
