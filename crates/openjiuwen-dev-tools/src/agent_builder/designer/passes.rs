//! The three designer passes: basic design, branch design, and reflection.
//!
//! Ports the three stages of
//! `openjiuwen.dev_tools.agent_builder.builders.workflow.workflow_designer`:
//! Python's basic design plans the structure, its branch design adds the
//! decision points, and its reflection evaluation answers with the corrected
//! design parsed out of a `## New Workflow Design` section. The Rust port keeps
//! the three stages and replaces the Markdown sections with typed
//! [`DesignPass`] records over one [`WorkflowDesign`] per pass.
//!
//! Every pass has a deterministic offline implementation and asks the model only
//! when [`WorkflowDesigner::with_model`] wired one. A model error, a missing
//! fence, an unparseable document, or a document failing
//! [`WorkflowDesign::validate`] is never an error: that pass falls back to its
//! deterministic result and records `fallback` with the reason.

use std::collections::BTreeSet;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::super::cycle_checker::find_cycle;
use super::super::outcome::{GenerationProvenance, GenerationSource};
use super::super::workflow_dsl::{DslEdge, DslNode, WorkflowDsl};
use super::{
    DesignEdge, DesignNode, DesignNodeKind, WorkflowDesign, WorkflowDesigner, BRANCH_MARKERS,
    DESCRIPTION_LIMIT, DESIGN_SYSTEM_PROMPT,
};

mod clauses;

use self::clauses::{slugify, step_title, task_clauses, truncate_chars};
use crate::util::extract_json_fence;

/// Which of the three designer passes produced a record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DesignPassKind {
    /// The basic pass: the workflow's structure and steps.
    #[default]
    Basic,
    /// The branch pass: the decision points.
    Branch,
    /// The reflection pass: the repaired, final design.
    Reflection,
}

impl DesignPassKind {
    /// The pass's wire name: `"basic"`, `"branch"`, or `"reflection"`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Basic => "basic",
            Self::Branch => "branch",
            Self::Reflection => "reflection",
        }
    }
}

/// One designer pass's contribution to the final design.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DesignPass {
    /// Which pass produced this record.
    pub kind: DesignPassKind,
    /// Where the pass's output came from.
    pub source: GenerationSource,
    /// A short note about what the pass did; empty for an accepted model answer.
    pub detail: String,
    /// How many nodes the pass's design holds.
    pub node_count: usize,
    /// How many edges the pass's design holds.
    pub edge_count: usize,
}

impl DesignPass {
    /// Records one pass's outcome over the design it produced.
    fn new(
        kind: DesignPassKind,
        source: GenerationSource,
        detail: String,
        design: &WorkflowDesign,
    ) -> Self {
        Self {
            kind,
            source,
            detail,
            node_count: design.nodes.len(),
            edge_count: design.edges.len(),
        }
    }
}

/// The result of the three-pass designer pipeline.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ThreePassDesign {
    /// The final design: the reflection pass's output.
    pub design: WorkflowDesign,
    /// Exactly three records, in basic, branch, reflection order.
    pub passes: Vec<DesignPass>,
    /// One provenance record per pass, oldest first.
    pub provenance: GenerationProvenance,
    /// One note per repair the reflection pass applied, oldest first.
    pub notes: Vec<String>,
}

/// One pass's design and its record.
pub(super) struct PassOutcome {
    /// The design the pass produced.
    pub(super) design: WorkflowDesign,
    /// The pass's record, counted over `design`.
    pub(super) pass: DesignPass,
}

/// The reflection pass's design, record, and repair notes.
pub(super) struct ReflectionOutcome {
    /// The repaired design.
    pub(super) design: WorkflowDesign,
    /// The pass's record, counted over `design`.
    pub(super) pass: DesignPass,
    /// One note per applied repair.
    pub(super) notes: Vec<String>,
}

/// The id of the branch node the branch pass inserts.
const BRANCH_ID: &str = "branch";
/// The description every inserted branch node carries.
const BRANCH_DESCRIPTION: &str = "Choose the next step";
/// The maximum number of characters a rejection reason keeps.
const REASON_LIMIT: usize = 160;

/// The JSON contract every pass's model answer must follow.
const DESIGN_CONTRACT: &str = r#"Answer with exactly one ```json fenced code block and nothing else inside it,
using these keys:
- `name`: a lowercase, hyphen-separated workflow name
- `description`: one sentence describing the workflow
- `nodes`: a list of {"id", "type", "title", "description", "config"} objects,
  where `type` is one of "start", "llm", "branch", "tool", "end"; `config`
  holds the node instruction under "instruction", the declared data under
  "inputs" and "outputs", and branch predicates under "conditions"
- `edges`: a list of {"source", "target", "condition"} objects naming node ids

Every design needs exactly one "start" node, at least one "end" node, unique
node ids, and edges whose endpoints are node ids.
Do not add other keys and do not wrap the document in prose."#;

/// The role sentence prepended to the basic pass's prompt.
const BASIC_DESIGN_ROLE: &str = "You are the openJiuwen workflow designer running the basic design pass.
Plan the workflow's initial inputs, its functional modules, and the implementation steps of each module.";

/// The role sentence prepended to the branch pass's prompt.
const BRANCH_DESIGN_ROLE: &str =
    "You are the openJiuwen workflow designer running the branch design pass.
Add the decision points the request needs to the basic design you are given.";

/// The role sentence prepended to the reflection pass's prompt.
const REFLECTION_ROLE: &str =
    "You are the openJiuwen workflow designer running the reflection pass.
Review the workflow design you are given and answer with the corrected, complete design.";

/// Runs the basic pass: the deterministic structure, or a model's structure.
pub(super) async fn run_basic(designer: &WorkflowDesigner, query: &str) -> Result<PassOutcome> {
    let baseline = basic_design(query)?;
    let detail = steps_detail(&baseline);
    let Some(model) = designer.model.as_deref() else {
        return Ok(PassOutcome {
            pass: DesignPass::new(
                DesignPassKind::Basic,
                GenerationSource::Deterministic,
                detail,
                &baseline,
            ),
            design: baseline,
        });
    };

    let system = system_prompt(BASIC_DESIGN_ROLE);
    match ask(model, &system, basic_question(query))
        .await
        .and_then(|answer| parse_design(&answer))
    {
        Ok(design) => Ok(PassOutcome {
            pass: DesignPass::new(
                DesignPassKind::Basic,
                GenerationSource::Model,
                String::new(),
                &design,
            ),
            design,
        }),
        Err(error) => Ok(PassOutcome {
            pass: DesignPass::new(
                DesignPassKind::Basic,
                GenerationSource::Fallback,
                rejected(&error),
                &baseline,
            ),
            design: baseline,
        }),
    }
}

/// Runs the branch pass over `basic`.
pub(super) async fn run_branch(
    designer: &WorkflowDesigner,
    query: &str,
    basic: &WorkflowDesign,
) -> Result<PassOutcome> {
    let (baseline, detail) = branch_design(query, basic);
    let markers = needs_branch(query);
    let Some(model) = designer.model.as_deref() else {
        return Ok(PassOutcome {
            pass: DesignPass::new(
                DesignPassKind::Branch,
                GenerationSource::Deterministic,
                detail,
                &baseline,
            ),
            design: baseline,
        });
    };

    let system = system_prompt(BRANCH_DESIGN_ROLE);
    let answer = ask(model, &system, branch_question(query, basic)).await;
    let accepted = answer
        .and_then(|answer| parse_design(&answer))
        .and_then(|design| {
            if markers && !has_branch_node(&design) {
                return Err(Error::validation(
                    "model answer has no branch node while the request asks for a decision",
                ));
            }
            Ok(design)
        });
    match accepted {
        Ok(design) => Ok(PassOutcome {
            pass: DesignPass::new(
                DesignPassKind::Branch,
                GenerationSource::Model,
                String::new(),
                &design,
            ),
            design,
        }),
        Err(error) => Ok(PassOutcome {
            pass: DesignPass::new(
                DesignPassKind::Branch,
                GenerationSource::Fallback,
                rejected(&error),
                &baseline,
            ),
            design: baseline,
        }),
    }
}

/// Runs the reflection pass over `merged`.
pub(super) async fn run_reflection(
    designer: &WorkflowDesigner,
    query: &str,
    merged: &WorkflowDesign,
) -> Result<ReflectionOutcome> {
    let (baseline, notes) = reflection_design(merged)?;
    let detail = repair_detail(&notes);
    let Some(model) = designer.model.as_deref() else {
        return Ok(ReflectionOutcome {
            pass: DesignPass::new(
                DesignPassKind::Reflection,
                GenerationSource::Deterministic,
                detail,
                &baseline,
            ),
            design: baseline,
            notes,
        });
    };

    let system = system_prompt(REFLECTION_ROLE);
    match ask(model, &system, reflection_question(query, merged))
        .await
        .and_then(|answer| parse_design(&answer))
    {
        Ok(design) => Ok(ReflectionOutcome {
            pass: DesignPass::new(
                DesignPassKind::Reflection,
                GenerationSource::Model,
                String::new(),
                &design,
            ),
            design,
            notes: Vec::new(),
        }),
        Err(error) => Ok(ReflectionOutcome {
            pass: DesignPass::new(
                DesignPassKind::Reflection,
                GenerationSource::Fallback,
                rejected(&error),
                &baseline,
            ),
            design: baseline,
            notes,
        }),
    }
}

/// Whether the request asks for a decision, and so needs a branch node.
fn needs_branch(query: &str) -> bool {
    let lowered = query.to_lowercase();
    BRANCH_MARKERS.iter().any(|marker| {
        if marker.is_ascii() {
            contains_word(&lowered, marker)
        } else {
            lowered.contains(marker)
        }
    })
}

/// Whether `needle` occurs in `haystack` delimited by non-alphanumeric chars.
fn contains_word(haystack: &str, needle: &str) -> bool {
    let mut from = 0usize;
    while let Some(offset) = haystack[from..].find(needle) {
        let start = from + offset;
        let end = start + needle.len();
        let before = haystack[..start]
            .chars()
            .next_back()
            .is_none_or(|character| !character.is_alphanumeric());
        let after = haystack[end..]
            .chars()
            .next()
            .is_none_or(|character| !character.is_alphanumeric());
        if before && after {
            return true;
        }
        from = end;
    }
    false
}

/// The single-round-trip design used by [`WorkflowDesigner::design`].
pub(super) async fn design_with_model(model: &dyn Model, query: &str) -> Result<WorkflowDesign> {
    let answer = ask(model, DESIGN_SYSTEM_PROMPT, query.to_string()).await?;
    parse_design(&answer)
}

/// A short rejection reason, capped for a provenance detail.
pub(super) fn rejection_reason(error: &Error) -> String {
    truncate_chars(&error.to_string(), REASON_LIMIT)
}

/// The single-pass offline design: the basic design plus the legacy branch node.
///
/// The legacy branch keeps the entry point's historical shape — id `branch`,
/// title `Decide`, and `config.conditions = ["default"]` — so
/// [`WorkflowDesigner::design_offline`] answers exactly what it always did.
pub(super) fn offline_design(query: &str) -> Result<WorkflowDesign> {
    let design = basic_design(query)?;
    if !needs_branch(query) {
        return Ok(design);
    }
    Ok(with_branch(
        design,
        "Decide".to_string(),
        vec!["default".to_string()],
    ))
}

/// The deterministic basic pass: `start` -> `llm_1..N` -> `end`.
///
/// The request is split into at most [`MAX_STEPS`] task clauses; each clause
/// becomes one LLM node carrying its instruction. This pass never inserts a
/// branch node and never calls a model.
pub(super) fn basic_design(query: &str) -> Result<WorkflowDesign> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Err(Error::validation("query must not be blank"));
    }

    let mut steps = task_clauses(trimmed);
    if steps.is_empty() {
        steps.push(trimmed.to_string());
    }

    let mut nodes = vec![start_node()];
    let mut edges = Vec::new();
    let mut previous = "start".to_string();

    for (position, clause) in steps.iter().enumerate() {
        let id = format!("llm_{}", position + 1);
        nodes.push(DesignNode {
            id: id.clone(),
            kind: DesignNodeKind::Llm,
            title: step_title(clause, position + 1),
            description: clause.clone(),
            config: json!({
                "instruction": clause,
                "inputs": {"query": {"type": "string"}},
                "outputs": {"result": {"type": "string"}},
            }),
        });
        edges.push(link(&previous, &id));
        previous = id;
    }

    nodes.push(end_node());
    edges.push(link(&previous, "end"));

    let design = WorkflowDesign {
        name: slugify(&steps[0]),
        description: truncate_chars(trimmed, DESCRIPTION_LIMIT),
        nodes,
        edges,
    };
    design.validate()?;
    Ok(design)
}

/// The deterministic branch pass, as `(design, detail)`.
pub(super) fn branch_design(query: &str, basic: &WorkflowDesign) -> (WorkflowDesign, String) {
    if !needs_branch(query) {
        return (
            basic.clone(),
            "no decision markers in the request".to_string(),
        );
    }
    if has_branch_node(basic) || basic.nodes.iter().any(|node| node.id == BRANCH_ID) {
        return (
            basic.clone(),
            "the design already has a branch node".to_string(),
        );
    }
    let design = with_branch(
        basic.clone(),
        branch_title(basic),
        vec!["condition".to_string(), "default".to_string()],
    );
    (
        design,
        "decision markers found; inserted the `branch` node before the end node".to_string(),
    )
}

/// The deterministic reflection pass, as `(design, notes)`.
pub(super) fn reflection_design(merged: &WorkflowDesign) -> Result<(WorkflowDesign, Vec<String>)> {
    let mut repaired = merged.clone();
    let mut notes = Vec::new();

    if repaired.nodes.is_empty() {
        repaired.nodes.push(start_node());
        repaired.nodes.push(end_node());
        repaired.edges.push(link("start", "end"));
        notes.push("inserted the missing start node `start`".to_string());
        notes.push("appended the missing end node `end`".to_string());
    }

    rename_duplicate_ids(&mut repaired, &mut notes);
    drop_unknown_edges(&mut repaired, &mut notes);
    insert_missing_start(&mut repaired, &mut notes);
    insert_missing_end(&mut repaired, &mut notes);
    connect_unreachable(&mut repaired, &mut notes);
    break_cycles(&mut repaired, &mut notes);

    repaired.validate()?;
    Ok((repaired, notes))
}

/// Inserts the branch node before the end node and rewires the edges into it.
fn with_branch(
    mut design: WorkflowDesign,
    title: String,
    conditions: Vec<String>,
) -> WorkflowDesign {
    if design.nodes.iter().any(|node| node.id == BRANCH_ID) {
        return design;
    }
    let Some(end_index) = design
        .nodes
        .iter()
        .position(|node| node.kind == DesignNodeKind::End)
    else {
        return design;
    };
    let end_id = design.nodes[end_index].id.clone();

    design.nodes.insert(
        end_index,
        DesignNode {
            id: BRANCH_ID.to_string(),
            kind: DesignNodeKind::Branch,
            title,
            description: BRANCH_DESCRIPTION.to_string(),
            config: json!({"conditions": conditions}),
        },
    );

    let mut edges = Vec::with_capacity(design.edges.len() + 1);
    for mut edge in std::mem::take(&mut design.edges) {
        if edge.target == end_id {
            edge.target = BRANCH_ID.to_string();
        }
        edges.push(edge);
    }
    edges.push(link(BRANCH_ID, &end_id));
    design.edges = edges;
    design
}

/// A title for the inserted branch node, derived like the LLM step titles.
fn branch_title(basic: &WorkflowDesign) -> String {
    let candidate = basic
        .nodes
        .iter()
        .rev()
        .find(|node| node.kind == DesignNodeKind::Llm)
        .map_or(basic.description.as_str(), |node| node.description.as_str());
    let position = basic
        .nodes
        .iter()
        .position(|node| node.kind == DesignNodeKind::End)
        .unwrap_or(basic.nodes.len())
        + 1;
    step_title(candidate, position)
}

/// Whether the design already holds a branch node.
fn has_branch_node(design: &WorkflowDesign) -> bool {
    design
        .nodes
        .iter()
        .any(|node| node.kind == DesignNodeKind::Branch)
}

/// Renames the second and later occurrence of a node id.
fn rename_duplicate_ids(design: &mut WorkflowDesign, notes: &mut Vec<String>) {
    let mut taken: BTreeSet<String> = BTreeSet::new();
    for node in &mut design.nodes {
        if taken.insert(node.id.clone()) {
            continue;
        }
        let original = node.id.clone();
        let renamed = unique_id(&original, &taken);
        taken.insert(renamed.clone());
        node.id.clone_from(&renamed);
        notes.push(format!(
            "renamed the duplicate node id `{original}` to `{renamed}`"
        ));
    }
}

/// Drops every edge naming a node that is not in the design.
fn drop_unknown_edges(design: &mut WorkflowDesign, notes: &mut Vec<String>) {
    let known: BTreeSet<String> = design.nodes.iter().map(|node| node.id.clone()).collect();
    let mut kept = Vec::with_capacity(design.edges.len());
    for edge in std::mem::take(&mut design.edges) {
        if known.contains(&edge.source) && known.contains(&edge.target) {
            kept.push(edge);
        } else {
            notes.push(format!(
                "dropped the edge `{}` -> `{}` because it names an unknown node",
                edge.source, edge.target
            ));
        }
    }
    design.edges = kept;
}

/// Inserts a missing start node at the front and connects it to the first node.
fn insert_missing_start(design: &mut WorkflowDesign, notes: &mut Vec<String>) {
    if design
        .nodes
        .iter()
        .any(|node| node.kind == DesignNodeKind::Start)
    {
        return;
    }
    let Some(first) = design.nodes.first().map(|node| node.id.clone()) else {
        return;
    };
    let taken: BTreeSet<String> = design.nodes.iter().map(|node| node.id.clone()).collect();
    let id = unique_id("start", &taken);
    let mut start = start_node();
    start.id.clone_from(&id);
    design.nodes.insert(0, start);
    design.edges.push(link(&id, &first));
    notes.push(format!("inserted the missing start node `{id}`"));
}

/// Appends a missing end node and connects every sink to it.
fn insert_missing_end(design: &mut WorkflowDesign, notes: &mut Vec<String>) {
    if design
        .nodes
        .iter()
        .any(|node| node.kind == DesignNodeKind::End)
    {
        return;
    }
    let taken: BTreeSet<String> = design.nodes.iter().map(|node| node.id.clone()).collect();
    let id = unique_id("end", &taken);
    let sources = design
        .edges
        .iter()
        .map(|edge| edge.source.clone())
        .chain(std::iter::once(id.clone()))
        .collect::<BTreeSet<String>>();
    let mut sinks: Vec<String> = design
        .nodes
        .iter()
        .filter(|node| !sources.contains(&node.id))
        .map(|node| node.id.clone())
        .collect();
    if sinks.is_empty() {
        if let Some(last) = design.nodes.last() {
            sinks.push(last.id.clone());
        }
    }
    let mut end = end_node();
    end.id.clone_from(&id);
    design.nodes.push(end);
    for sink in sinks {
        design.edges.push(link(&sink, &id));
    }
    notes.push(format!("appended the missing end node `{id}`"));
}

/// Connects every node the start node cannot reach.
fn connect_unreachable(design: &mut WorkflowDesign, notes: &mut Vec<String>) {
    let Some(root) = design
        .nodes
        .iter()
        .find(|node| node.kind == DesignNodeKind::Start)
        .map(|node| node.id.clone())
    else {
        return;
    };
    let mut reachable = reachable_from(&root, design);
    for index in 0..design.nodes.len() {
        let id = design.nodes[index].id.clone();
        if reachable.contains(&id) {
            continue;
        }
        let earlier = design.nodes[..index]
            .iter()
            .rev()
            .find(|node| reachable.contains(&node.id))
            .map_or_else(|| root.clone(), |node| node.id.clone());
        reachable.insert(id.clone());
        reachable.extend(reachable_from(&id, design));
        design.edges.push(link(&earlier, &id));
        notes.push(format!(
            "connected the unreachable node `{id}` from `{earlier}`"
        ));
    }
}

/// Breaks every cycle `find_cycle` reports by dropping its back edge.
fn break_cycles(design: &mut WorkflowDesign, notes: &mut Vec<String>) {
    let bound = design.nodes.len() + 1;
    for _ in 0..bound {
        let Some(path) = find_cycle(&as_dsl(design)) else {
            return;
        };
        let (Some(target), Some(source)) = (path.last(), path.get(path.len().saturating_sub(2)))
        else {
            return;
        };
        let before = design.edges.len();
        design
            .edges
            .retain(|edge| edge.source != *source || edge.target != *target);
        if design.edges.len() == before {
            return;
        }
        notes.push(format!(
            "removed the back edge `{source}` -> `{target}` closing a cycle"
        ));
    }
}

/// The nodes reachable from `root` over the design's edges.
fn reachable_from(root: &str, design: &WorkflowDesign) -> BTreeSet<String> {
    let mut reachable = BTreeSet::new();
    if !design.nodes.iter().any(|node| node.id == root) {
        return reachable;
    }
    reachable.insert(root.to_string());
    let mut frontier = vec![root.to_string()];
    while let Some(current) = frontier.pop() {
        for edge in design.edges.iter().filter(|edge| edge.source == current) {
            if reachable.insert(edge.target.clone()) {
                frontier.push(edge.target.clone());
            }
        }
    }
    reachable
}

/// Projects the design onto the DSL graph so the cycle checker can read it.
fn as_dsl(design: &WorkflowDesign) -> WorkflowDsl {
    WorkflowDsl {
        name: design.name.clone(),
        description: design.description.clone(),
        nodes: design
            .nodes
            .iter()
            .map(|node| DslNode {
                id: node.id.clone(),
                node_type: node.kind.as_str().to_string(),
                title: node.title.clone(),
                description: node.description.clone(),
                data: node.config.clone(),
                ..DslNode::default()
            })
            .collect(),
        edges: design
            .edges
            .iter()
            .map(|edge| DslEdge {
                source: edge.source.clone(),
                target: edge.target.clone(),
                condition: edge.condition.clone(),
            })
            .collect(),
    }
}

/// `base`, or `base_2`, `base_3`, ... when `base` is already taken.
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

/// A `start` node carrying the standard user-input declaration.
fn start_node() -> DesignNode {
    DesignNode {
        id: "start".to_string(),
        kind: DesignNodeKind::Start,
        title: "Start".to_string(),
        description: "Workflow entry".to_string(),
        config: json!({"inputs": {"query": {"type": "string", "description": "user input"}}}),
    }
}

/// An `end` node.
fn end_node() -> DesignNode {
    DesignNode {
        id: "end".to_string(),
        kind: DesignNodeKind::End,
        title: "End".to_string(),
        description: "Workflow exit".to_string(),
        config: Value::Null,
    }
}

/// An unconditional edge.
fn link(source: &str, target: &str) -> DesignEdge {
    DesignEdge {
        source: source.to_string(),
        target: target.to_string(),
        condition: None,
    }
}

/// The detail recorded for a deterministic basic pass.
fn steps_detail(design: &WorkflowDesign) -> String {
    let steps = design
        .nodes
        .iter()
        .filter(|node| node.kind == DesignNodeKind::Llm)
        .count();
    if steps == 1 {
        "1 step synthesised".to_string()
    } else {
        format!("{steps} steps synthesised")
    }
}

/// The detail recorded for the deterministic reflection pass.
fn repair_detail(notes: &[String]) -> String {
    match notes.len() {
        0 => "the design validated without repair".to_string(),
        1 => "1 repair applied".to_string(),
        count => format!("{count} repairs applied"),
    }
}

/// The detail recorded when a model answer was rejected.
fn rejected(error: &Error) -> String {
    format!("model answer rejected: {}", rejection_reason(error))
}

/// Assembles a pass's system prompt from its role sentence and the contract.
fn system_prompt(role: &str) -> String {
    format!("{role}\n\n{DESIGN_CONTRACT}")
}

/// The basic pass's user message.
fn basic_question(query: &str) -> String {
    format!("## User request\n{query}")
}

/// The branch pass's user message: the request plus the basic design.
fn branch_question(query: &str, basic: &WorkflowDesign) -> String {
    format!(
        "## User request\n{query}\n\n## Basic design\n{}",
        design_json(basic)
    )
}

/// The reflection pass's user message: the request plus the merged design.
fn reflection_question(query: &str, merged: &WorkflowDesign) -> String {
    format!(
        "## User request\n{query}\n\n## Workflow design\n{}",
        design_json(merged)
    )
}

/// One design rendered as JSON for a pass's prompt.
fn design_json(design: &WorkflowDesign) -> String {
    serde_json::to_string_pretty(design).unwrap_or_else(|_| String::from("{}"))
}

/// Asks `model` one system-prompted question and returns the answer text.
async fn ask(model: &dyn Model, system: &str, question: String) -> Result<String> {
    let messages = vec![BaseMessage::system(system), BaseMessage::user(question)];
    let request = ModelRequestConfig::new(model.model_name());
    let response = model.invoke(&messages, &request).await?;
    let Some(choice) = response.choices.into_iter().next() else {
        return Err(Error::llm("model returned no choices"));
    };
    Ok(choice.message.text().into_owned())
}

/// Parses one fenced JSON design out of a model answer and validates it.
fn parse_design(answer: &str) -> Result<WorkflowDesign> {
    let value = extract_json_fence(answer)
        .ok_or_else(|| Error::validation("model answer contained no JSON code fence"))?;
    let design: WorkflowDesign = serde_json::from_value(value).map_err(|error| {
        Error::validation(format!("failed to parse workflow design JSON: {error}"))
    })?;
    design.validate()?;
    Ok(design)
}

#[cfg(test)]
mod tests;
