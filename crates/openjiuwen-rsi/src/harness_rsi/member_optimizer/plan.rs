//! Constrained optimisation actions and dependency-aware planning.
//!
//! Python's member optimiser turns a selected target into a `MemberOptimizationPlan`
//! of `MemberOptimizationAction`s and then schedules those actions into
//! dependency waves before any executor runs. The data contracts and the wave
//! algorithm are portable, so they live here as real code:
//!
//! - [`OptimizationAction`] mirrors Python's `MemberOptimizationAction` fields.
//! - [`plan_waves`] is Python's `build_action_waves` topological layering with
//!   typed errors instead of `ValueError`.
//! - [`DeterministicActionPlanner`] is the offline stand-in for Python's LLM
//!   planner (`action_planner.py`): pick the action definitions a target is
//!   eligible for, chain them per role, and compute the waves.
//!
//! A production caller supplies its own [`ActionPlanner`] (an LLM planner over
//! the harness refs) and reuses [`plan_waves`] to schedule the result.
use super::selection::SelectedTarget;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
/// When an action may run relative to its dependencies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunIf {
    /// Run only when every dependency succeeded (Python's default).
    #[default]
    DependencySucceeded,
    /// Run only when at least one dependency failed (a repair or rollback).
    DependencyFailed,
    /// Run regardless of dependency outcomes.
    Always,
}
/// One constrained optimisation action for a selected role.
///
/// Field names and defaults mirror Python's `MemberOptimizationAction`; the JSON
/// shape is therefore stable across both implementations.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OptimizationAction {
    /// Stable action id; unique within a plan and used by dependency edges.
    pub action_id: String,
    /// The role this action optimises.
    pub role: String,
    /// Action group (Python's `config`/`prompt`/`rail`/`skill`/`tool`).
    pub action_group: String,
    /// Operation within the group (`add`/`modify`/`remove`).
    pub operation: String,
    /// Free-form action type used by the executor.
    pub action_type: String,
    /// Harness file the action touches.
    pub target_path: String,
    /// What the action does.
    pub description: String,
    /// Why this action was chosen.
    pub rationale: String,
    /// Issues this action is meant to fix.
    pub attributed_issue_ids: Vec<String>,
    /// Action ids that must run first.
    pub depends_on: Vec<String>,
    /// Condition under which the action runs.
    pub run_if: RunIf,
    /// Skills the executor may use.
    pub allowed_skills: Vec<String>,
    /// Tools the executor may use.
    pub allowed_tools: Vec<String>,
    /// Expected observable effect.
    pub expected_effect: String,
    /// Known risks of applying the action.
    pub risk_notes: Vec<String>,
    /// Path prefixes the action may write; empty means "must not write".
    pub declared_write_paths: Vec<String>,
    /// Harness member name, when known.
    pub member_name: Option<String>,
    /// Planner annotations, never interpreted here.
    pub metadata: BTreeMap<String, Value>,
}
/// A dependency-aware member optimisation plan.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OptimizationPlan {
    /// Stable plan id.
    pub plan_id: String,
    /// The targets the plan was built for.
    pub targets: Vec<SelectedTarget>,
    /// The planned actions.
    pub actions: Vec<OptimizationAction>,
    /// Actions grouped into dependency waves; see [`plan_waves`].
    pub action_waves: Vec<Vec<String>>,
    /// Planner annotations.
    pub metadata: BTreeMap<String, Value>,
}
/// The template library a planner picks actions from.
///
/// Python loads these from the action-group policy (`action_groups/definitions.py`
/// plus `policy.py`). The Rust port keeps the same fields as plain serde data so a
/// caller can load a policy document, and [`DeterministicActionPlanner`] can turn
/// one definition into one [`OptimizationAction`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ActionDefinition {
    /// Definition name; matched against a target's `eligible_actions`.
    pub name: String,
    /// Action group; matched against a plan's allowlist.
    pub action_group: String,
    /// Operation within the group.
    pub operation: String,
    /// Free-form action type forwarded to the action.
    pub action_type: String,
    /// Harness file this definition targets.
    pub target_path: String,
    /// What applying the definition does.
    pub description: String,
    /// Skills an action created from this definition may use.
    pub default_skills: Vec<String>,
    /// Tools an action created from this definition may use.
    pub default_tools: Vec<String>,
    /// Expected observable effect.
    pub expected_effect: String,
    /// Known risks.
    pub risk_notes: Vec<String>,
    /// Path prefixes an action created from this definition may write.
    pub declared_write_paths: Vec<String>,
}
/// Groups `actions` into dependency waves.
///
/// Wave *n* holds every action whose dependencies are all satisfied by waves
/// `0..n`; within a wave ids are sorted ascending. This is Python's
/// `build_action_waves`, including its failure modes:
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Validation`] when two actions share
/// an id, when an action depends on an id that is not in `actions`, or when the
/// dependency graph contains a cycle (the message names the remaining ids).
pub fn plan_waves(actions: &[OptimizationAction]) -> Result<Vec<Vec<String>>> {
    let mut remaining: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for action in actions {
        if remaining
            .insert(action.action_id.clone(), action.depends_on.clone())
            .is_some()
        {
            return Err(Error::validation(format!(
                "duplicate action id `{}` in plan",
                action.action_id
            )));
        }
    }
    let known: BTreeSet<&str> = remaining.keys().map(String::as_str).collect();
    for (action_id, dependencies) in &remaining {
        for dependency in dependencies {
            if !known.contains(dependency.as_str()) {
                return Err(Error::validation(format!(
                    "action `{action_id}` depends on unknown action `{dependency}`"
                )));
            }
        }
    }
    let mut completed: BTreeSet<String> = BTreeSet::new();
    let mut waves: Vec<Vec<String>> = Vec::new();
    while !remaining.is_empty() {
        let ready: Vec<String> = remaining
            .iter()
            .filter(|(_, dependencies)| {
                dependencies
                    .iter()
                    .all(|dependency| completed.contains(dependency))
            })
            .map(|(action_id, _)| action_id.clone())
            .collect();
        if ready.is_empty() {
            let blocked: Vec<&str> = remaining.keys().map(String::as_str).collect();
            return Err(Error::validation(format!(
                "dependency cycle detected among actions: {}",
                blocked.join(", ")
            )));
        }
        for action_id in &ready {
            remaining.remove(action_id);
        }
        completed.extend(ready.iter().cloned());
        waves.push(ready);
    }
    Ok(waves)
}
/// One planning request: the targets, the round, and the action-group allowlist.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PlanRequest {
    /// Id of the optimisation run; [`DeterministicActionPlanner`] derives action
    /// ids from it.
    pub optimization_id: String,
    /// Targets to plan for, already beam-limited by selection.
    pub targets: Vec<SelectedTarget>,
    /// `0` for the initial plan, `n` for the n-th repair round.
    pub round: u32,
    /// Action groups the run may use; empty means "all allowed".
    pub allowlist: Vec<String>,
}
impl PlanRequest {
    /// Whether `action_group` passes the request's allowlist.
    #[must_use]
    pub fn allows_group(&self, action_group: &str) -> bool {
        self.allowlist.is_empty() || self.allowlist.iter().any(|allowed| allowed == action_group)
    }
}
/// Turns a [`PlanRequest`] into an executable [`OptimizationPlan`].
///
/// Python implements this with a planner agent (LLM) that reads the harness refs
/// and the issue evidence. The trait is the seam: implement it with that agent,
/// or use [`DeterministicActionPlanner`] offline.
#[async_trait]
pub trait ActionPlanner: Send + Sync {
    /// Returns the planner's human-readable name.
    fn name(&self) -> &str;
    /// Builds the plan for `request`.
    ///
    /// # Errors
    ///
    /// Implementations return [`openjiuwen_common::error::Error::Validation`]
    /// for a plan that cannot be scheduled (unknown dependency, cycle, duplicate
    /// id) and any other kind for infrastructure failures.
    async fn plan(&self, request: &PlanRequest) -> Result<OptimizationPlan>;
}
/// Deterministic offline stand-in for Python's LLM action planner.
///
/// For each selected target it walks [`Self::definitions`] in order and keeps the
/// definitions whose `name` appears in the target's `eligible_actions` (all
/// definitions when the list is empty), skipping groups outside the request's
/// allowlist, up to `max_actions` (`0` = unlimited). Each kept definition becomes
/// one [`OptimizationAction`] whose id is
/// `<optimization_id>/r<round>/a<n>`, and each action depends on the previous
/// action of the *same role* so a role's actions run in definition order.
///
/// The result is deterministic: same definitions, targets, and round ⇒ same
/// actions, ids, and waves.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DeterministicActionPlanner {
    /// The definition library, in the order definitions are considered.
    pub definitions: Vec<ActionDefinition>,
    /// Cap on actions per target; `0` means unlimited.
    pub max_actions: usize,
}
impl DeterministicActionPlanner {
    /// Creates a planner over `definitions` with no per-target cap.
    #[must_use]
    pub fn new(definitions: Vec<ActionDefinition>) -> Self {
        Self {
            definitions,
            max_actions: 0,
        }
    }
    /// Sets the per-target action cap (`0` = unlimited).
    #[must_use]
    pub fn with_max_actions(mut self, max_actions: usize) -> Self {
        self.max_actions = max_actions;
        self
    }
    /// Whether `definition` is eligible for `target` under `request`.
    fn is_eligible(
        definition: &ActionDefinition,
        target: &SelectedTarget,
        request: &PlanRequest,
    ) -> bool {
        if !request.allows_group(&definition.action_group) {
            return false;
        }
        target.target.eligible_actions.is_empty()
            || target
                .target
                .eligible_actions
                .iter()
                .any(|name| name == &definition.name)
    }
    /// Materialises one action from `definition` for `target`.
    fn build_action(
        definition: &ActionDefinition,
        target: &SelectedTarget,
        request: &PlanRequest,
        index: usize,
        previous: Option<&str>,
    ) -> OptimizationAction {
        let mut metadata = BTreeMap::new();
        metadata.insert("planner".to_owned(), Value::from(ACTION_PLANNER_NAME));
        metadata.insert(
            "definition".to_owned(),
            Value::from(definition.name.clone()),
        );
        OptimizationAction {
            action_id: format!("{}/r{}/a{}", request.optimization_id, request.round, index),
            role: target.target.role.clone(),
            action_group: definition.action_group.clone(),
            operation: definition.operation.clone(),
            action_type: definition.action_type.clone(),
            target_path: definition.target_path.clone(),
            description: definition.description.clone(),
            rationale: format!(
                "deterministic selection of definition `{}` for issue(s) {}",
                definition.name,
                target.target.issue_ids.join(", ")
            ),
            attributed_issue_ids: target.target.issue_ids.clone(),
            depends_on: previous.map(ToOwned::to_owned).into_iter().collect(),
            run_if: RunIf::DependencySucceeded,
            allowed_skills: definition.default_skills.clone(),
            allowed_tools: definition.default_tools.clone(),
            expected_effect: definition.expected_effect.clone(),
            risk_notes: definition.risk_notes.clone(),
            declared_write_paths: definition.declared_write_paths.clone(),
            member_name: target.target.member_name.clone(),
            metadata,
        }
    }
    /// Builds the plan for `request`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when the generated
    /// actions cannot be scheduled (the generated graph is acyclic by
    /// construction, so this only surfaces duplicate ids from an id collision in
    /// `optimization_id`).
    fn build_plan(&self, request: &PlanRequest) -> Result<OptimizationPlan> {
        let mut actions: Vec<OptimizationAction> = Vec::new();
        for target in &request.targets {
            let mut previous: Option<String> = None;
            let mut planned = 0usize;
            for definition in &self.definitions {
                if !Self::is_eligible(definition, target, request) {
                    continue;
                }
                if self.max_actions != 0 && planned >= self.max_actions {
                    break;
                }
                let action = Self::build_action(
                    definition,
                    target,
                    request,
                    actions.len(),
                    previous.as_deref(),
                );
                previous = Some(action.action_id.clone());
                actions.push(action);
                planned += 1;
            }
        }
        let action_waves = plan_waves(&actions)?;
        let mut metadata = BTreeMap::new();
        metadata.insert("planner".to_owned(), Value::from(ACTION_PLANNER_NAME));
        metadata.insert("round".to_owned(), Value::from(request.round));
        metadata.insert(
            "definitions".to_owned(),
            Value::from(self.definitions.len()),
        );
        metadata.insert(
            "allowlist".to_owned(),
            Value::from(request.allowlist.clone()),
        );
        Ok(OptimizationPlan {
            plan_id: format!("{}/plan/r{}", request.optimization_id, request.round),
            targets: request.targets.clone(),
            actions,
            action_waves,
            metadata,
        })
    }
}
/// The name reported by [`DeterministicActionPlanner`].
pub const ACTION_PLANNER_NAME: &str = "deterministic_action_planner";
#[async_trait]
impl ActionPlanner for DeterministicActionPlanner {
    fn name(&self) -> &str {
        ACTION_PLANNER_NAME
    }
    async fn plan(&self, request: &PlanRequest) -> Result<OptimizationPlan> {
        self.build_plan(request)
    }
}
