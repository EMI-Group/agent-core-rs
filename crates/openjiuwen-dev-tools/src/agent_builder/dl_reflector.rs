//! Rule-based repair of a workflow DSL document, with an optional model pass.
//!
//! Ports the reflector stage of
//! `openjiuwen/dev_tools/agent_builder/builders/workflow/dl_reflector.py`:
//! Python validates a generated `SimpleIR` list and collects human-readable
//! errors which the builder feeds back into the designer. This port starts from
//! the typed [`WorkflowDsl`] instead, repairs it deterministically, and — when a
//! [`Model`] is configured — asks for one improved revision of the document.
//!
//! # Deterministic repair
//!
//! [`DlReflector::reflect`] clones its input and never mutates it. The rules
//! below run in this order, and every repair appends one [`ReflectionIssue`], so
//! `issues` is ordered by rule first and by document position second:
//!
//! 1. **Empty document** — a document without nodes becomes a `start -> end`
//!    chain, recorded as [`ReflectionIssueKind::MissingStart`] followed by
//!    [`ReflectionIssueKind::MissingEnd`].
//! 2. **Invalid node** ([`ReflectionIssueKind::InvalidNode`]) — a blank node id
//!    becomes `node_<index>` (1-based document position, uniquified against the
//!    ids already present) and a blank title takes the node id.
//! 3. **Duplicate node id** ([`ReflectionIssueKind::DuplicateNodeId`]) — the
//!    second and later occurrence of an id is renamed `<id>_2`, `<id>_3`, …
//!    Edges keep referencing the first occurrence.
//! 4. **Dangling edge** ([`ReflectionIssueKind::DanglingEdge`]) — edges naming an
//!    unknown endpoint are dropped.
//! 5. **Missing start** ([`ReflectionIssueKind::MissingStart`]) — a node of type
//!    `start` is inserted at the front and connected to the first node.
//! 6. **Missing end** ([`ReflectionIssueKind::MissingEnd`]) — a node of type
//!    `end` is appended and every sink is connected to it.
//! 7. **Unreachable node** ([`ReflectionIssueKind::UnreachableNode`]) — a node
//!    the start cannot reach is connected from the nearest earlier reachable
//!    node in document order.
//! 8. **Cycle** ([`ReflectionIssueKind::Cycle`]) — back edges reported by
//!    `find_cycle` are removed until the graph is acyclic.
//!
//! The repaired document is validated last, so a [`Reflection`] whose `passed`
//! flag is `false` still holds a structurally valid graph. `passed` is exactly
//! `issues.is_empty()`: an already-valid document is returned unchanged with no
//! issues, and any repair sets it to `false`.
//!
//! # The optional model pass
//!
//! [`DlReflector::reflect_with_model`] runs the deterministic repair first and
//! then offers the baseline document plus the caller's instructions to the
//! model, which answers with one fenced JSON document. A revision is accepted
//! only when it parses, validates, and keeps exactly the baseline node id set;
//! otherwise — including on a model error or an answer without choices — the
//! baseline is returned with one extra [`ReflectionIssueKind::InvalidNode`]
//! issue whose message starts with `model revision rejected: `. That kind is
//! reused as the generic "rejected revision" marker because a rejected document
//! is not representable as a graph-level defect. Model problems are never
//! errors, so the caller always receives a usable document.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::cycle_checker::find_cycle;
use super::workflow_dsl::{DslEdge, DslNode, WorkflowDsl};
use crate::util::extract_json_fence;

/// The system prompt used for a model-assisted revision.
///
/// It names exactly the keys a [`WorkflowDsl`] reads, so a compliant answer
/// parses without repair and keeps the node ids the baseline comparison needs.
const REFLECTOR_SYSTEM_PROMPT: &str = r#"You are the openJiuwen workflow DSL reflector.
Repair and improve the workflow document the user sends.
Answer with exactly one ```json fenced code block holding the whole document:
- `name`: the workflow name
- `description`: a human-readable description
- `nodes`: a list of `{ "id", "type", "title", "description", "inputs", "outputs", "data" }`
- `edges`: a list of `{ "source", "target", "condition" }`
Keep every node id of the document you were given and add no new one; keep the
graph acyclic and keep a `start` and an `end` node. Do not add other keys and do
not wrap the document in prose."#;

/// The category of one reflection issue.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReflectionIssueKind {
    /// An edge named a node that does not exist and was dropped.
    #[default]
    DanglingEdge,
    /// A node id occurred more than once and a later copy was renamed.
    DuplicateNodeId,
    /// The document had no node of type `start` and one was inserted.
    MissingStart,
    /// The document had no node of type `end` and one was appended.
    MissingEnd,
    /// A directed cycle was found and its back edge was removed.
    Cycle,
    /// A node field was missing or blank and was filled in.
    ///
    /// Also marks a model revision that was rejected.
    InvalidNode,
    /// A node the start could not reach was connected to the graph.
    UnreachableNode,
}

/// One repair or rejection recorded while reflecting a document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ReflectionIssue {
    /// What kind of problem was found.
    pub kind: ReflectionIssueKind,
    /// A human-readable description of the repair or rejection.
    pub message: String,
    /// The node the issue is about, when one id is meaningful.
    pub node_id: Option<String>,
}

/// The outcome of validating and repairing one workflow document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Reflection {
    /// Whether the input needed no repair (`issues` is empty).
    pub passed: bool,
    /// Every repair performed, in rule order then document order.
    pub issues: Vec<ReflectionIssue>,
    /// The repaired document; structurally valid even when `passed` is `false`.
    pub repaired: WorkflowDsl,
}

/// Repairs a workflow DSL document and optionally asks a model for a revision.
///
/// [`DlReflector::new`] is purely deterministic; [`DlReflector::with_model`] adds
/// one model round trip in [`DlReflector::reflect_with_model`].
pub struct DlReflector {
    /// The model used for revisions, when one was supplied.
    model: Option<Arc<dyn Model>>,
}

impl DlReflector {
    /// Creates a reflector that never calls a model.
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Creates a reflector that may ask `model` for a revision.
    #[must_use]
    pub fn with_model(model: Arc<dyn Model>) -> Self {
        Self { model: Some(model) }
    }

    /// Validates `dsl` and repairs what can be repaired deterministically.
    ///
    /// The input is never mutated. Every repair appends one [`ReflectionIssue`];
    /// `passed` is `true` exactly when no repair was needed. The repaired
    /// document always satisfies [`WorkflowDsl::validate`].
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when the repaired
    /// document still fails [`WorkflowDsl::validate`], which the rules above are
    /// designed to make impossible.
    pub fn reflect(&self, dsl: &WorkflowDsl) -> Result<Reflection> {
        let mut repaired = dsl.clone();
        let mut issues = Vec::new();

        if repaired.nodes.is_empty() {
            repaired.nodes.push(start_node("start".to_string()));
            repaired.nodes.push(end_node("end".to_string()));
            repaired.edges.push(edge("start", "end"));
            issues.push(issue(
                ReflectionIssueKind::MissingStart,
                Some("start"),
                "the document had no nodes; inserted the `start` node".to_string(),
            ));
            issues.push(issue(
                ReflectionIssueKind::MissingEnd,
                Some("end"),
                "the document had no nodes; appended the `end` node".to_string(),
            ));
        }

        repair_invalid_nodes(&mut repaired, &mut issues);
        repair_duplicate_ids(&mut repaired, &mut issues);
        drop_dangling_edges(&mut repaired, &mut issues);
        insert_start(&mut repaired, &mut issues);
        insert_end(&mut repaired, &mut issues);
        connect_unreachable(&mut repaired, &mut issues);
        break_cycles(&mut repaired, &mut issues);

        repaired.validate()?;

        Ok(Reflection {
            passed: issues.is_empty(),
            issues,
            repaired,
        })
    }

    /// Repairs `dsl`, then lets the configured model revise the result.
    ///
    /// With no model this is exactly [`Self::reflect`]. With one, the model
    /// receives the system contract, the baseline repaired document as JSON, and
    /// `instructions`. Its answer is accepted only when it parses as a
    /// [`WorkflowDsl`], validates, and keeps the baseline node id set; an
    /// accepted revision keeps the baseline `passed` flag and issues.
    ///
    /// A model failure, an answer without choices, an answer without a JSON
    /// fence, an invalid document, or a document with different node ids all
    /// keep the baseline and append one
    /// [`ReflectionIssueKind::InvalidNode`] issue named
    /// `model revision rejected: <reason>`, which also clears `passed`.
    ///
    /// # Errors
    ///
    /// Only the deterministic repair can fail, exactly as in [`Self::reflect`].
    pub async fn reflect_with_model(
        &self,
        dsl: &WorkflowDsl,
        instructions: &str,
    ) -> Result<Reflection> {
        let baseline = self.reflect(dsl)?;
        let Some(model) = self.model.as_ref() else {
            return Ok(baseline);
        };

        let messages = revision_messages(&baseline.repaired, instructions);
        let request = ModelRequestConfig::new(model.model_name());
        let answer = match model.invoke(&messages, &request).await {
            Ok(response) => response
                .choices
                .into_iter()
                .next()
                .map(|choice| choice.message.text().into_owned()),
            Err(error) => {
                return Ok(rejected(baseline, &format!("model call failed: {error}")));
            }
        };
        let Some(answer) = answer else {
            return Ok(rejected(baseline, "model returned no choices"));
        };

        let Some(value) = extract_json_fence(&answer) else {
            return Ok(rejected(baseline, "answer holds no ```json code fence"));
        };
        let improved = match serde_json::from_value::<WorkflowDsl>(value) {
            Ok(improved) => improved,
            Err(error) => {
                return Ok(rejected(
                    baseline,
                    &format!("answer is not a workflow document: {error}"),
                ));
            }
        };
        if let Err(error) = improved.validate() {
            return Ok(rejected(
                baseline,
                &format!("revised document is invalid: {error}"),
            ));
        }

        let baseline_ids = node_ids(&baseline.repaired);
        let improved_ids = node_ids(&improved);
        if improved_ids != baseline_ids {
            let expected: Vec<String> = baseline_ids.into_iter().collect();
            let actual: Vec<String> = improved_ids.into_iter().collect();
            return Ok(rejected(
                baseline,
                &format!(
                    "revised document changed the node ids: expected [{}], got [{}]",
                    expected.join(", "),
                    actual.join(", ")
                ),
            ));
        }

        Ok(Reflection {
            passed: baseline.passed,
            issues: baseline.issues,
            repaired: improved,
        })
    }
}

impl Default for DlReflector {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DlReflector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DlReflector")
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

/// Builds one issue for a repair or rejection.
fn issue(kind: ReflectionIssueKind, node_id: Option<&str>, message: String) -> ReflectionIssue {
    ReflectionIssue {
        kind,
        message,
        node_id: node_id.map(str::to_string),
    }
}

/// The ids of every node, in sorted order.
fn node_ids(dsl: &WorkflowDsl) -> BTreeSet<String> {
    dsl.nodes.iter().map(|node| node.id.clone()).collect()
}

/// A `start` node carrying the standard user-input declaration.
fn start_node(id: String) -> DslNode {
    DslNode {
        id,
        node_type: "start".to_string(),
        title: "Start".to_string(),
        description: "Workflow entry".to_string(),
        inputs: BTreeMap::from([(
            "query".to_string(),
            json!({"type": "string", "description": "user input"}),
        )]),
        ..DslNode::default()
    }
}

/// An `end` node.
fn end_node(id: String) -> DslNode {
    DslNode {
        id,
        node_type: "end".to_string(),
        title: "End".to_string(),
        description: "Workflow exit".to_string(),
        ..DslNode::default()
    }
}

/// An unconditional edge.
fn edge(source: &str, target: &str) -> DslEdge {
    DslEdge {
        source: source.to_string(),
        target: target.to_string(),
        condition: None,
    }
}

/// `base`, or `base_2`, `base_3`, … when `base` is already taken.
fn unique_id(base: &str, taken: &BTreeSet<String>) -> String {
    if !taken.contains(base) {
        return base.to_string();
    }
    let mut suffix = 2usize;
    loop {
        let candidate = format!("{base}_{suffix}");
        if !taken.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

/// Rule 2: fills blank node ids and titles.
fn repair_invalid_nodes(dsl: &mut WorkflowDsl, issues: &mut Vec<ReflectionIssue>) {
    let mut taken = node_ids(dsl);
    for (index, node) in dsl.nodes.iter_mut().enumerate() {
        let position = index + 1;
        let mut fields: Vec<&str> = Vec::new();
        if node.id.trim().is_empty() {
            let id = unique_id(&format!("node_{position}"), &taken);
            taken.insert(id.clone());
            node.id.clone_from(&id);
            fields.push("id");
        }
        if node.title.trim().is_empty() {
            node.title.clone_from(&node.id);
            fields.push("title");
        }
        if !fields.is_empty() {
            let id = node.id.clone();
            issues.push(issue(
                ReflectionIssueKind::InvalidNode,
                Some(id.as_str()),
                format!(
                    "node {position}: filled in a blank {}",
                    fields.join(" and ")
                ),
            ));
        }
    }
}

/// Rule 3: renames repeated node ids, keeping the first occurrence.
fn repair_duplicate_ids(dsl: &mut WorkflowDsl, issues: &mut Vec<ReflectionIssue>) {
    let mut taken = node_ids(dsl);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (index, node) in dsl.nodes.iter_mut().enumerate() {
        if seen.insert(node.id.clone()) {
            continue;
        }
        let original = node.id.clone();
        let renamed = unique_id(&original, &taken);
        taken.insert(renamed.clone());
        node.id.clone_from(&renamed);
        issues.push(issue(
            ReflectionIssueKind::DuplicateNodeId,
            Some(original.as_str()),
            format!(
                "node {}: duplicate id `{original}` renamed to `{renamed}`",
                index + 1
            ),
        ));
    }
}

/// Rule 4: drops edges with an unknown endpoint.
fn drop_dangling_edges(dsl: &mut WorkflowDsl, issues: &mut Vec<ReflectionIssue>) {
    let ids = node_ids(dsl);
    let mut kept = Vec::with_capacity(dsl.edges.len());
    for edge in std::mem::take(&mut dsl.edges) {
        if ids.contains(&edge.source) && ids.contains(&edge.target) {
            kept.push(edge);
            continue;
        }
        let unknown = if ids.contains(&edge.source) {
            edge.target.clone()
        } else {
            edge.source.clone()
        };
        issues.push(issue(
            ReflectionIssueKind::DanglingEdge,
            Some(unknown.as_str()),
            format!(
                "dropped edge `{}` -> `{}`: node `{unknown}` does not exist",
                edge.source, edge.target
            ),
        ));
    }
    dsl.edges = kept;
}

/// Rule 5: inserts a `start` node at the front and connects it.
fn insert_start(dsl: &mut WorkflowDsl, issues: &mut Vec<ReflectionIssue>) {
    if dsl.nodes.iter().any(|node| node.node_type == "start") {
        return;
    }
    let id = unique_id("start", &node_ids(dsl));
    dsl.nodes.insert(0, start_node(id.clone()));
    if let Some(target) = dsl.nodes.get(1).map(|node| node.id.clone()) {
        dsl.edges.push(edge(&id, &target));
    }
    issues.push(issue(
        ReflectionIssueKind::MissingStart,
        Some(id.as_str()),
        format!("inserted the missing `start` node `{id}`"),
    ));
}

/// Rule 6: appends an `end` node and connects every sink to it.
fn insert_end(dsl: &mut WorkflowDsl, issues: &mut Vec<ReflectionIssue>) {
    if dsl.nodes.iter().any(|node| node.node_type == "end") {
        return;
    }
    let id = unique_id("end", &node_ids(dsl));
    let sinks: Vec<String> = dsl
        .nodes
        .iter()
        .filter(|node| !dsl.edges.iter().any(|edge| edge.source == node.id))
        .map(|node| node.id.clone())
        .collect();
    dsl.nodes.push(end_node(id.clone()));
    for sink in sinks {
        dsl.edges.push(edge(&sink, &id));
    }
    issues.push(issue(
        ReflectionIssueKind::MissingEnd,
        Some(id.as_str()),
        format!("appended the missing `end` node `{id}`"),
    ));
}

/// Rule 7: connects nodes the start cannot reach.
fn connect_unreachable(dsl: &mut WorkflowDsl, issues: &mut Vec<ReflectionIssue>) {
    let Some(start) = dsl
        .nodes
        .iter()
        .find(|node| node.node_type == "start")
        .map(|node| node.id.clone())
    else {
        return;
    };
    let mut reachable = reachable_from(dsl, &start);
    let mut additions = Vec::new();
    for (index, node) in dsl.nodes.iter().enumerate() {
        if reachable.contains(&node.id) {
            continue;
        }
        let source = dsl.nodes[..index]
            .iter()
            .rev()
            .find(|earlier| reachable.contains(&earlier.id))
            .map(|earlier| earlier.id.clone());
        let Some(source) = source else {
            continue;
        };
        additions.push(edge(&source, &node.id));
        reachable.insert(node.id.clone());
        issues.push(issue(
            ReflectionIssueKind::UnreachableNode,
            Some(node.id.as_str()),
            format!(
                "node `{}` was unreachable; connected `{source}` -> `{}`",
                node.id, node.id
            ),
        ));
    }
    dsl.edges.extend(additions);
}

/// Rule 8: removes back edges until the graph is acyclic.
fn break_cycles(dsl: &mut WorkflowDsl, issues: &mut Vec<ReflectionIssue>) {
    let limit = dsl.edges.len() + 1;
    for _ in 0..limit {
        let Some(path) = find_cycle(dsl) else {
            return;
        };
        if path.len() < 2 {
            return;
        }
        let source = path.get(path.len() - 2).cloned().unwrap_or_default();
        let target = path.last().cloned().unwrap_or_default();
        let Some(position) = dsl
            .edges
            .iter()
            .position(|edge| edge.source == source && edge.target == target)
        else {
            return;
        };
        dsl.edges.remove(position);
        let root = path.first().cloned().unwrap_or_default();
        issues.push(issue(
            ReflectionIssueKind::Cycle,
            Some(root.as_str()),
            format!(
                "removed back edge `{source}` -> `{target}` to break the cycle {}",
                path.join(" -> ")
            ),
        ));
    }
}

/// The ids reachable from `start` over the document's edges.
fn reachable_from(dsl: &WorkflowDsl, start: &str) -> BTreeSet<String> {
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in &dsl.edges {
        adjacency
            .entry(edge.source.as_str())
            .or_default()
            .push(edge.target.as_str());
    }

    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        if !reachable.insert(node.to_string()) {
            continue;
        }
        if let Some(neighbours) = adjacency.get(node) {
            for neighbour in neighbours {
                queue.push_back(*neighbour);
            }
        }
    }
    reachable
}

/// The system contract plus the baseline document and the caller's request.
fn revision_messages(dsl: &WorkflowDsl, instructions: &str) -> Vec<BaseMessage> {
    let document = serde_json::to_string_pretty(dsl).unwrap_or_else(|error| {
        tracing::debug!(%error, "failed to serialize the workflow DSL for revision");
        String::new()
    });
    vec![
        BaseMessage::system(REFLECTOR_SYSTEM_PROMPT),
        BaseMessage::user(format!(
            "Current workflow document:\n```json\n{document}\n```\n\nRequested improvement:\n{instructions}"
        )),
    ]
}

/// Returns the baseline with one recorded rejection.
///
/// The appended issue also clears `passed`, keeping `passed == issues.is_empty()`.
fn rejected(mut baseline: Reflection, reason: &str) -> Reflection {
    tracing::debug!(%reason, "workflow model revision rejected");
    baseline.issues.push(ReflectionIssue {
        kind: ReflectionIssueKind::InvalidNode,
        message: format!("model revision rejected: {reason}"),
        node_id: None,
    });
    baseline.passed = false;
    baseline
}

#[cfg(test)]
mod tests;
