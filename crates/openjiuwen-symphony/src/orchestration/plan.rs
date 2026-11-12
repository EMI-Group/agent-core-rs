//! The typed orchestration plan: steps, edges, JGF projection, and Kahn stages.
//!
//! A [`OrchestrationPlan`] is the planner's serializable answer to one request: the
//! ordered capability steps that should run, the `can_feed` edges wiring their outputs
//! into later inputs, and the bookkeeping (score, status, title) a host renders.
//!
//! The plan is the stable seam between planning and execution, so every projection is
//! deterministic and documented:
//!
//! - [`OrchestrationPlan::planned_graph`] renders a JGF v2-ish document with one node
//!   per step (sorted by step id) and one `can_feed` edge per plan edge (sorted by
//!   `(source, target)`).
//! - [`OrchestrationPlan::to_value`] is the plan's JSON plus that `planned_graph`
//!   member, and [`OrchestrationPlan::from_value`] ignores the extra member, so the two
//!   round-trip the plan fields.
//! - [`OrchestrationPlan::stages`] layers the steps with Kahn's algorithm for hosts
//!   that execute level by level; a cycle is an [`crate::error::Error::Validation`].
//!
//! No hash-map iteration order, clock, or allocation ordering leaks into any output:
//! nodes and edges are sorted explicitly and ties break on identifiers.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{validation, Result};
use crate::graph::CAN_FEED;
use crate::model::CapabilityIo;

/// The lifecycle status of a plan, matching Python's planner outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    /// A complete plan whose root inputs are all satisfied.
    #[default]
    Ready,
    /// A plan exists but at least one root input must be supplied by the caller.
    NeedsInput,
    /// No candidate plan could be built for the request.
    NoPlan,
}

/// One directed `can_feed` link between two plan steps.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PlanEdge {
    /// The step id producing the value.
    pub source: String,
    /// The step id consuming the value.
    pub target: String,
}

/// One capability invocation inside a plan.
///
/// `step_id` is the capability's qualified id (`"{capability_type}:{capability_id}"`),
/// which is also the node id in [`OrchestrationPlan::planned_graph`] and the state key
/// the executor writes per step.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PlanStep {
    /// The qualified capability id of this step.
    pub step_id: String,
    /// The capability type of the step's capability.
    pub capability_type: String,
    /// The display name of the step's capability.
    pub name: String,
    /// The typed inputs the step consumes.
    pub inputs: Vec<CapabilityIo>,
    /// The typed outputs the step produces.
    pub outputs: Vec<CapabilityIo>,
    /// Input names no earlier step or caller-supplied input fills.
    pub missing_inputs: Vec<String>,
    /// Input names filled by an earlier step or by the caller.
    pub filled_inputs: Vec<String>,
    /// The semantic relevance score assigned during planning.
    pub score: f64,
}

/// The complete, serializable result of planning one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct OrchestrationPlan {
    /// A stable identifier for the plan (deterministic for a given request).
    pub plan_id: String,
    /// The user request the plan answers.
    pub query: String,
    /// The response language: `"cn"` or `"en"`.
    pub language: String,
    /// Whether the plan is ready, needs caller input, or does not exist.
    pub status: PlanStatus,
    /// A human-readable title for the plan.
    pub title: String,
    /// Optional explanation, for example which root inputs are missing.
    pub message: Option<String>,
    /// The executable steps, in planning order.
    pub steps: Vec<PlanStep>,
    /// The data-flow edges between steps.
    pub edges: Vec<PlanEdge>,
    /// The overall plan score produced by the ranking formula.
    pub score: f64,
}

impl Default for OrchestrationPlan {
    /// Returns an empty `"cn"` plan with [`PlanStatus::NoPlan`] and score `0.0`.
    fn default() -> Self {
        Self {
            plan_id: String::new(),
            query: String::new(),
            language: "cn".to_string(),
            status: PlanStatus::NoPlan,
            title: String::new(),
            message: None,
            steps: Vec::new(),
            edges: Vec::new(),
            score: 0.0,
        }
    }
}

impl OrchestrationPlan {
    /// Builds an empty plan with the given identity and language.
    ///
    /// The status starts as [`PlanStatus::NoPlan`] because there are no steps yet; the
    /// planner promotes it with [`Self::with_status`]. `language` is expected to be
    /// `"cn"` or `"en"` (see `normalize_language` in the planner module).
    #[must_use]
    pub fn new(
        plan_id: impl Into<String>,
        query: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            plan_id: plan_id.into(),
            query: query.into(),
            language: language.into(),
            ..Self::default()
        }
    }

    /// Replaces the plan status.
    #[must_use]
    pub fn with_status(mut self, status: PlanStatus) -> Self {
        self.status = status;
        self
    }

    /// Replaces the display title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Sets the explanatory message.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Replaces the plan steps.
    #[must_use]
    pub fn with_steps(mut self, steps: Vec<PlanStep>) -> Self {
        self.steps = steps;
        self
    }

    /// Replaces the plan edges.
    #[must_use]
    pub fn with_edges(mut self, edges: Vec<PlanEdge>) -> Self {
        self.edges = edges;
        self
    }

    /// Replaces the overall plan score.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// Reports whether the plan has no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Projects the plan onto a JSON-graph document.
    ///
    /// The result is `{"graph": {"id", "type": "planned_graph", "directed": true,
    /// "nodes", "edges"}}`. Nodes are `{"id": step_id, "label": name, "node_type":
    /// capability_type}` sorted by step id; edges are `{"source", "target", "relation":
    /// "can_feed"}` sorted by `(source, target)`.
    #[must_use]
    pub fn planned_graph(&self) -> Value {
        let mut nodes: Vec<(String, Value)> = self
            .steps
            .iter()
            .map(|step| {
                (
                    step.step_id.clone(),
                    json!({
                        "id": step.step_id,
                        "label": step.name,
                        "node_type": step.capability_type,
                    }),
                )
            })
            .collect();
        nodes.sort_by(|(left, _), (right, _)| left.cmp(right));

        let mut edges: Vec<((String, String), Value)> = self
            .edges
            .iter()
            .map(|edge| {
                (
                    (edge.source.clone(), edge.target.clone()),
                    json!({
                        "source": edge.source,
                        "target": edge.target,
                        "relation": CAN_FEED,
                    }),
                )
            })
            .collect();
        edges.sort_by(|(left, _), (right, _)| left.cmp(right));

        json!({
            "graph": {
                "id": self.plan_id,
                "type": "planned_graph",
                "directed": true,
                "nodes": nodes.into_iter().map(|(_, node)| node).collect::<Vec<_>>(),
                "edges": edges.into_iter().map(|(_, edge)| edge).collect::<Vec<_>>(),
            }
        })
    }

    /// Serializes the plan and appends the [`Self::planned_graph`] projection.
    ///
    /// The result is the plan's regular JSON object plus a `"planned_graph"` member, so
    /// a host can render and store one document. Serialization of this type only fails
    /// for a non-finite `score`, in which case the function returns `Value::Null` (use
    /// [`Self::with_score`] with a finite score to avoid that).
    #[must_use]
    pub fn to_value(&self) -> Value {
        let planned_graph = self.planned_graph();
        match serde_json::to_value(self) {
            Ok(Value::Object(mut map)) => {
                map.insert("planned_graph".to_string(), planned_graph);
                Value::Object(map)
            }
            Ok(other) => other,
            Err(_) => Value::Null,
        }
    }

    /// Parses a plan, ignoring a `"planned_graph"` member if present.
    ///
    /// Unknown members are tolerated, so [`Self::to_value`] round-trips the plan fields.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] naming `plan` when `value` is not an
    /// object shaped like [`OrchestrationPlan`].
    pub fn from_value(value: &Value) -> Result<Self> {
        serde_json::from_value(value.clone()).map_err(|error| validation("plan", error))
    }

    /// Layers the steps with Kahn's algorithm, returning one sorted id list per stage.
    ///
    /// Every edge endpoint must name one of the plan's steps; an edge endpoint outside
    /// the step set is an error rather than a silently dropped link, and a cycle is
    /// reported instead of looping.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] naming `plan.edges` when an edge
    /// references an unknown step or when the edges contain a cycle.
    pub fn stages(&self) -> Result<Vec<Vec<String>>> {
        let step_ids: BTreeSet<&str> = self
            .steps
            .iter()
            .map(|step| step.step_id.as_str())
            .collect();
        let mut indegree: BTreeMap<String, usize> = step_ids
            .iter()
            .map(|step_id| ((*step_id).to_string(), 0_usize))
            .collect();
        let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for edge in &self.edges {
            if !step_ids.contains(edge.source.as_str()) {
                return Err(validation(
                    "plan.edges",
                    format!("edge source `{}` is not a plan step", edge.source),
                ));
            }
            if !step_ids.contains(edge.target.as_str()) {
                return Err(validation(
                    "plan.edges",
                    format!("edge target `{}` is not a plan step", edge.target),
                ));
            }
            let neighbours = adjacency.entry(edge.source.clone()).or_default();
            if neighbours.insert(edge.target.clone()) {
                if let Some(count) = indegree.get_mut(&edge.target) {
                    *count += 1;
                }
            }
        }

        let mut ready: BTreeSet<String> = indegree
            .iter()
            .filter(|(_, count)| **count == 0)
            .map(|(step_id, _)| step_id.clone())
            .collect();
        let mut stages: Vec<Vec<String>> = Vec::new();
        let mut visited = 0_usize;
        while !ready.is_empty() {
            let stage: Vec<String> = ready.iter().cloned().collect();
            let mut next: BTreeSet<String> = BTreeSet::new();
            for step_id in &stage {
                visited += 1;
                let Some(neighbours) = adjacency.get(step_id) else {
                    continue;
                };
                for neighbour in neighbours {
                    let Some(count) = indegree.get_mut(neighbour) else {
                        continue;
                    };
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        next.insert(neighbour.clone());
                    }
                }
            }
            stages.push(stage);
            ready = next;
        }

        if visited != indegree.len() {
            return Err(validation(
                "plan.edges",
                "contains a cycle: not every step can be layered",
            ));
        }
        Ok(stages)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn step(step_id: &str) -> PlanStep {
        PlanStep {
            step_id: step_id.to_string(),
            capability_type: "tool".to_string(),
            name: format!("cap {step_id}"),
            ..PlanStep::default()
        }
    }

    fn edge(source: &str, target: &str) -> PlanEdge {
        PlanEdge {
            source: source.to_string(),
            target: target.to_string(),
        }
    }

    #[test]
    fn status_serializes_snake_case_and_defaults_to_ready() {
        assert_eq!(PlanStatus::default(), PlanStatus::Ready);
        assert_eq!(
            serde_json::to_value(PlanStatus::NeedsInput).unwrap(),
            json!("needs_input")
        );
        assert_eq!(
            serde_json::from_value::<PlanStatus>(json!("no_plan")).unwrap(),
            PlanStatus::NoPlan
        );
    }

    #[test]
    fn default_plan_is_empty_and_no_plan() {
        let plan = OrchestrationPlan::default();
        assert!(plan.is_empty());
        assert_eq!(plan.status, PlanStatus::NoPlan);
        assert_eq!(plan.language, "cn");
        assert!((plan.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn builders_set_every_optional_field() {
        let plan = OrchestrationPlan::new("p1", "do it", "en")
            .with_status(PlanStatus::Ready)
            .with_title("Plan: do it")
            .with_message("hello")
            .with_steps(vec![step("tool:a")])
            .with_edges(vec![edge("tool:a", "tool:b")])
            .with_score(1.5);
        assert_eq!(plan.plan_id, "p1");
        assert_eq!(plan.query, "do it");
        assert_eq!(plan.language, "en");
        assert_eq!(plan.status, PlanStatus::Ready);
        assert_eq!(plan.title, "Plan: do it");
        assert_eq!(plan.message.as_deref(), Some("hello"));
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.edges.len(), 1);
        assert!((plan.score - 1.5).abs() < f64::EPSILON);
        assert!(!plan.is_empty());
    }

    #[test]
    fn planned_graph_sorts_nodes_and_edges() {
        let plan = OrchestrationPlan::new("p", "q", "en")
            .with_steps(vec![step("tool:b"), step("tool:a")])
            .with_edges(vec![edge("tool:b", "tool:c"), edge("tool:a", "tool:b")]);
        let graph = plan.planned_graph();
        assert_eq!(graph["graph"]["id"], "p");
        assert_eq!(graph["graph"]["type"], "planned_graph");
        assert_eq!(graph["graph"]["directed"], true);
        let nodes = graph["graph"]["nodes"].as_array().unwrap();
        let ids: Vec<&str> = nodes
            .iter()
            .map(|node| node["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["tool:a", "tool:b"]);
        assert_eq!(nodes[0]["label"], "cap tool:a");
        assert_eq!(nodes[0]["node_type"], "tool");
        let edges = graph["graph"]["edges"].as_array().unwrap();
        assert_eq!(edges[0]["source"], "tool:a");
        assert_eq!(edges[0]["target"], "tool:b");
        assert_eq!(edges[0]["relation"], "can_feed");
        assert_eq!(edges[1]["source"], "tool:b");
    }

    #[test]
    fn to_value_adds_planned_graph_and_round_trips() {
        let plan = OrchestrationPlan::new("p", "q", "cn")
            .with_status(PlanStatus::NeedsInput)
            .with_steps(vec![step("tool:a")]);
        let value = plan.to_value();
        assert!(value.get("planned_graph").is_some());
        assert_eq!(value["status"], "needs_input");
        let parsed = OrchestrationPlan::from_value(&value).unwrap();
        assert_eq!(parsed, plan);
    }

    #[test]
    fn from_value_rejects_malformed_input() {
        let error = OrchestrationPlan::from_value(&json!("not a plan")).unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().starts_with("plan:"));
    }

    #[test]
    fn stages_layer_a_chain_one_depth_at_a_time() {
        let plan = OrchestrationPlan::new("p", "q", "en").with_steps(vec![
            step("tool:a"),
            step("tool:b"),
            step("tool:c"),
            step("tool:d"),
        ]);
        let plan = plan.with_edges(vec![
            edge("tool:a", "tool:b"),
            edge("tool:a", "tool:c"),
            edge("tool:b", "tool:d"),
            edge("tool:c", "tool:d"),
            // A parallel duplicate must not double-count the indegree.
            edge("tool:a", "tool:c"),
        ]);
        let stages = plan.stages().unwrap();
        assert_eq!(
            stages,
            vec![
                vec!["tool:a".to_string()],
                vec!["tool:b".to_string(), "tool:c".to_string()],
                vec!["tool:d".to_string()],
            ]
        );
    }

    #[test]
    fn stages_order_each_layer_by_step_id() {
        let plan = OrchestrationPlan::new("p", "q", "en").with_steps(vec![
            step("tool:z"),
            step("tool:m"),
            step("tool:a"),
        ]);
        assert_eq!(
            plan.stages().unwrap(),
            vec![vec![
                "tool:a".to_string(),
                "tool:m".to_string(),
                "tool:z".to_string()
            ]]
        );
    }

    #[test]
    fn stages_reject_cycles() {
        let plan = OrchestrationPlan::new("p", "q", "en")
            .with_steps(vec![step("tool:a"), step("tool:b")])
            .with_edges(vec![edge("tool:a", "tool:b"), edge("tool:b", "tool:a")]);
        let error = plan.stages().unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().starts_with("plan.edges:"));
        assert!(error.message().contains("cycle"));
    }

    #[test]
    fn stages_reject_unknown_edge_endpoints() {
        let plan = OrchestrationPlan::new("p", "q", "en")
            .with_steps(vec![step("tool:a")])
            .with_edges(vec![edge("tool:a", "tool:ghost")]);
        let error = plan.stages().unwrap_err();
        assert!(error.message().contains("tool:ghost"));
    }

    #[test]
    fn empty_plan_stages_is_empty() {
        assert!(
            OrchestrationPlan::default().stages().unwrap().is_empty(),
            "an empty plan has no stages"
        );
    }
}
