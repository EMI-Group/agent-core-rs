//! The workflow-designer stage: a typed workflow design and its synthesis.
//!
//! Ports `openjiuwen.dev_tools.agent_builder.builders.workflow.workflow_designer`.
//! Python runs a three-stage model pipeline (basic design, branch design,
//! reflection evaluation) and answers with free-form Markdown; this port answers
//! with a typed [`WorkflowDesign`] that converts into the platform
//! [`WorkflowDsl`].
//!
//! [`WorkflowDesigner::design`] is the single-round-trip entry point: one model
//! answer when a model is wired, else the deterministic synthesis.
//! [`WorkflowDesigner::design_three_pass`] runs Python's pipeline as three
//! recorded passes ([`DesignPass`]), [`WorkflowDesigner::stream_design`] streams
//! the same pipeline as [`DesignEvent`]s, and every entry point records
//! [`GenerationProvenance`], so a degraded design is never silent.
//!
//! A design is validated before it is used: at least one node, unique ids,
//! resolvable edges, exactly one start node, and at least one end node.
//! [`WorkflowDesign::to_workflow_dsl`] is the one design-to-DSL converter.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::outcome::GenerationProvenance;
use super::progress::ProgressStage;
use super::workflow_dsl::{DslEdge, DslNode, WorkflowDsl};

mod passes;
mod stream;

pub use self::passes::{DesignPass, DesignPassKind, ThreePassDesign};
pub use self::stream::DesignEvent;

/// The instruction sent to the model by [`WorkflowDesigner::design`].
///
/// It names every key the parser reads, so a compliant answer round-trips
/// through [`WorkflowDesign`] without repair.
const DESIGN_SYSTEM_PROMPT: &str = r#"You are the openJiuwen workflow designer.
Turn the user's request into one complete workflow design.

Answer with exactly one ```json fenced code block and nothing else inside it,
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

/// The provenance detail recorded for a deterministic design.
const DETERMINISTIC_DESIGN_DETAIL: &str = "no model configured; deterministic design synthesis";

/// The maximum number of task steps the offline designer turns into LLM nodes.
const MAX_STEPS: usize = 4;
/// The maximum number of words an English step title keeps.
const TITLE_WORDS: usize = 8;
/// The maximum number of characters a CJK step title keeps.
const CJK_TITLE_CHARS: usize = 20;
/// The maximum number of characters of a slugified workflow name.
const NAME_LIMIT: usize = 40;
/// The maximum number of characters of a synthesized workflow description.
const DESCRIPTION_LIMIT: usize = 160;

/// Sentence terminators that end a task clause.
const TERMINATORS: [char; 9] = ['.', ';', '。', '；', '\n', '!', '?', '！', '？'];

/// Sequencing phrases that introduce a new task clause; the English ones match
/// case-insensitively.
const SEQUENCE_MARKERS: [&str; 10] = [
    ", then ",
    " then ",
    "after that",
    "next,",
    "首先",
    "然后",
    "接着",
    "最后",
    "，然后",
    "，再",
];

/// Words and phrases that make the offline design branch.
const BRANCH_MARKERS: [&str; 14] = [
    "if",
    "when",
    "whether",
    "condition",
    "decision",
    "branch",
    "else",
    "otherwise",
    "如果",
    "否则",
    "当",
    "是否",
    "条件",
    "分支",
];

/// The kind of a [`DesignNode`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "String", into = "String")]
pub enum DesignNodeKind {
    /// The workflow entry point; a design has exactly one.
    #[default]
    Start,
    /// A model call that turns its inputs into text.
    Llm,
    /// A conditional split; its outgoing edges carry the conditions.
    Branch,
    /// A tool, plugin, API, or code step.
    Tool,
    /// The workflow exit; a design has at least one.
    End,
}

impl DesignNodeKind {
    /// The lowercase wire name of the kind.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Llm => "llm",
            Self::Branch => "branch",
            Self::Tool => "tool",
            Self::End => "end",
        }
    }

    /// Parses a node kind from a wire name or a Python DL node-type alias.
    ///
    /// Matching is case-insensitive and ignores `-`, `_`, and spaces, so
    /// `"llm_node"`, `"LLM"`, and `"LlmNode"` all parse as [`Self::Llm`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `value` when no alias matches.
    #[allow(clippy::should_implement_trait)] // the required API pairs it with `FromStr`.
    pub fn from_str(value: &str) -> Result<Self> {
        parse_kind(value)
    }
}

/// Parses a node kind, normalizing case and separators first.
fn parse_kind(value: &str) -> Result<DesignNodeKind> {
    let normalized: String = value
        .chars()
        .filter(|character| !matches!(character, '-' | '_' | ' '))
        .map(|character| character.to_ascii_lowercase())
        .collect();

    match normalized.as_str() {
        "start" | "begin" | "entry" => Ok(DesignNodeKind::Start),
        "llm" | "llmnode" | "llmnodes" | "model" | "generate" | "questioner" => {
            Ok(DesignNodeKind::Llm)
        }
        "branch" | "condition" | "conditions" | "router" | "route" | "switch" | "decision"
        | "intentdetection" => Ok(DesignNodeKind::Branch),
        "tool" | "plugin" | "api" | "code" | "function" | "toolnode" => Ok(DesignNodeKind::Tool),
        "end" | "output" | "finish" | "exit" | "endnode" => Ok(DesignNodeKind::End),
        _ => Err(Error::validation(format!(
            "unknown design node type `{value}`"
        ))),
    }
}

impl fmt::Display for DesignNodeKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for DesignNodeKind {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        parse_kind(value)
    }
}

impl TryFrom<String> for DesignNodeKind {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        parse_kind(&value)
    }
}

impl From<DesignNodeKind> for String {
    fn from(kind: DesignNodeKind) -> Self {
        kind.as_str().to_string()
    }
}

/// One node of a [`WorkflowDesign`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DesignNode {
    /// The node id, unique inside the design and referenced by [`DesignEdge`].
    pub id: String,
    /// The node kind; serialized as `"type"` to match the Python wire format.
    #[serde(rename = "type")]
    pub kind: DesignNodeKind,
    /// A human-readable node title.
    pub title: String,
    /// A human-readable description of what the node does.
    pub description: String,
    /// The node configuration: instruction, declared inputs/outputs, conditions.
    pub config: Value,
}

/// A directed connection between two [`DesignNode`]s.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DesignEdge {
    /// The id of the node the edge leaves.
    pub source: String,
    /// The id of the node the edge enters.
    pub target: String,
    /// The branch condition that must hold for the edge to be taken.
    pub condition: Option<String>,
}

/// A complete workflow design.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowDesign {
    /// The workflow name.
    pub name: String,
    /// A human-readable workflow description.
    pub description: String,
    /// The nodes of the graph, in document order.
    pub nodes: Vec<DesignNode>,
    /// The edges of the graph, in document order.
    pub edges: Vec<DesignEdge>,
}

impl WorkflowDesign {
    /// Checks the structural invariants of the design.
    ///
    /// A design needs at least one node, unique node ids, resolvable edge
    /// endpoints, exactly one [`DesignNodeKind::Start`] node, and at least one
    /// [`DesignNodeKind::End`] node. The first failing invariant is reported.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `workflow design has no nodes`,
    /// `duplicate node id: <id>`, `edge references unknown node: <id>`,
    /// `workflow design needs exactly one start node`, or
    /// `workflow design needs an end node`.
    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(Error::validation("workflow design has no nodes"));
        }

        let mut ids = BTreeSet::new();
        for node in &self.nodes {
            if !ids.insert(node.id.as_str()) {
                return Err(Error::validation(format!("duplicate node id: {}", node.id)));
            }
        }

        for edge in &self.edges {
            for endpoint in [edge.source.as_str(), edge.target.as_str()] {
                if !ids.contains(endpoint) {
                    return Err(Error::validation(format!(
                        "edge references unknown node: {endpoint}"
                    )));
                }
            }
        }

        if self
            .nodes
            .iter()
            .filter(|node| node.kind == DesignNodeKind::Start)
            .count()
            != 1
        {
            return Err(Error::validation(
                "workflow design needs exactly one start node",
            ));
        }
        if !self
            .nodes
            .iter()
            .any(|node| node.kind == DesignNodeKind::End)
        {
            return Err(Error::validation("workflow design needs an end node"));
        }

        Ok(())
    }

    /// Converts the design into the serialized workflow document.
    ///
    /// Each node keeps its kind as the DSL `type`, its title, description, and
    /// whole `config` (as the free-form `data`); the `inputs` and `outputs`
    /// objects of `config` are also lifted into the dedicated DSL maps. The
    /// produced document is validated before it is returned.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the design or the produced document
    /// fails [`Self::validate`] / [`WorkflowDsl::validate`].
    pub fn to_workflow_dsl(&self) -> Result<WorkflowDsl> {
        self.validate()?;

        let nodes = self
            .nodes
            .iter()
            .map(|node| DslNode {
                id: node.id.clone(),
                node_type: node.kind.as_str().to_string(),
                title: node.title.clone(),
                description: node.description.clone(),
                inputs: config_object(&node.config, "inputs"),
                outputs: config_object(&node.config, "outputs"),
                data: node.config.clone(),
            })
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|edge| DslEdge {
                source: edge.source.clone(),
                target: edge.target.clone(),
                condition: edge.condition.clone(),
            })
            .collect();

        let dsl = WorkflowDsl {
            name: self.name.clone(),
            description: self.description.clone(),
            nodes,
            edges,
        };
        dsl.validate()?;
        Ok(dsl)
    }
}

/// The `config[key]` object of a node, as an ordered map; empty when absent.
fn config_object(config: &Value, key: &str) -> BTreeMap<String, Value> {
    config
        .get(key)
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// A workflow design together with the provenance of the stage that produced it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DesignOutcome {
    /// The design the designer produced.
    pub design: WorkflowDesign,
    /// The stage records behind that design, oldest first.
    pub provenance: GenerationProvenance,
}

/// Designs workflows from a natural-language request.
///
/// Without a model the designer is deterministic and offline: it splits the
/// request into steps and synthesizes a start/LLM/end workflow. With a model it
/// performs one round trip and falls back to the offline design whenever the
/// answer cannot be parsed into a valid [`WorkflowDesign`].
#[derive(Clone)]
pub struct WorkflowDesigner {
    /// The optional model the model paths talk to.
    model: Option<Arc<dyn Model>>,
}

impl WorkflowDesigner {
    /// Creates an offline designer; every entry point runs its deterministic path.
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Creates a designer whose model paths ask `model` first.
    #[must_use]
    pub fn with_model(model: Arc<dyn Model>) -> Self {
        Self { model: Some(model) }
    }

    /// Designs a workflow, using the model when one is configured.
    ///
    /// This is [`Self::design_with_provenance`] without the provenance record.
    /// One system message describing the JSON contract is followed by one user
    /// message carrying `query`. A model error, a choice-less answer, a missing
    /// JSON fence, an unparseable document, or a document failing
    /// [`WorkflowDesign::validate`] is logged and replaced by
    /// [`Self::design_offline`], so model problems never fail the design.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query — the only error this
    /// entry point produces.
    pub async fn design(&self, query: &str) -> Result<WorkflowDesign> {
        Ok(self.design_with_provenance(query).await?.design)
    }

    /// Designs a workflow and records where the output came from.
    ///
    /// With no model the result is [`Self::design_offline`]'s, recorded as
    /// `deterministic` with `no model configured; deterministic design synthesis`
    /// as its detail. An
    /// accepted model answer is recorded as `model`; a rejected one is logged,
    /// replaced by the offline design, and recorded as `fallback` with the
    /// rejection reason. The record is warned about once per non-model stage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query.
    pub async fn design_with_provenance(&self, query: &str) -> Result<DesignOutcome> {
        if query.trim().is_empty() {
            return Err(Error::validation("query must not be blank"));
        }

        let mut provenance = GenerationProvenance::new();
        let design = match self.model.as_deref() {
            None => {
                let design = passes::offline_design(query)?;
                provenance.record_deterministic(
                    ProgressStage::GeneratingWorkflowDesign,
                    DETERMINISTIC_DESIGN_DETAIL,
                );
                design
            }
            Some(model) => match passes::design_with_model(model, query).await {
                Ok(design) => {
                    provenance.record_model(ProgressStage::GeneratingWorkflowDesign);
                    design
                }
                Err(error) => {
                    tracing::debug!(
                        %error,
                        "workflow design failed; falling back to the offline designer"
                    );
                    let design = passes::offline_design(query)?;
                    provenance.record_fallback(
                        ProgressStage::GeneratingWorkflowDesign,
                        passes::rejection_reason(&error),
                    );
                    design
                }
            },
        };

        provenance.warn_if_degraded();
        Ok(DesignOutcome { design, provenance })
    }

    /// Designs a workflow deterministically, without any model call.
    ///
    /// The request is split into at most `MAX_STEPS` task clauses at sentence
    /// terminators, sequencing phrases, and numbered list markers. Each clause
    /// becomes one LLM node carrying its instruction; the graph is closed by a
    /// start node and an end node, and a branch node is inserted when the
    /// request contains a decision word (`if`, `when`, `otherwise`, `如果`,
    /// `是否`, `分支`, ...). The result always passes
    /// [`WorkflowDesign::validate`].
    ///
    /// The stage is recorded as `deterministic` and warned about like any other
    /// non-model stage, because this entry point never consults a model.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query.
    pub fn design_offline(&self, query: &str) -> Result<WorkflowDesign> {
        let design = passes::offline_design(query)?;
        let mut provenance = GenerationProvenance::new();
        provenance.record_deterministic(
            ProgressStage::GeneratingWorkflowDesign,
            DETERMINISTIC_DESIGN_DETAIL,
        );
        provenance.warn_if_degraded();
        Ok(design)
    }

    /// Runs the basic design pass deterministically: `start` -> `llm_1..N` -> `end`.
    ///
    /// The request is split into at most `MAX_STEPS` task clauses (sentence
    /// terminators, sequencing phrases, numbered list markers, leading list
    /// markers stripped, empties and duplicates dropped, single-step fallback),
    /// and every clause becomes one LLM node carrying its instruction. The pass
    /// never inserts a branch node, never calls a model, and the result always
    /// passes [`WorkflowDesign::validate`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank `query`.
    pub fn design_basic_offline(&self, query: &str) -> Result<WorkflowDesign> {
        passes::basic_design(query)
    }

    /// Runs the branch design pass deterministically over `basic`.
    ///
    /// When the lowercased `query` holds a decision marker (`if`, `when`,
    /// `whether`, `condition`, `decision`, `branch`, `else`, `otherwise`,
    /// `如果`, `否则`, `当`, `是否`, `条件`, `分支`) and `basic` has no branch
    /// node, one branch node is inserted before the end node with
    /// `config.conditions = ["condition", "default"]`, and every edge into the
    /// end node is rewired through it. Without a marker, or when the design
    /// already has a branch node or an id `branch`, `basic` is returned
    /// unchanged, so the pass is idempotent and never duplicates the id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the produced design fails
    /// [`WorkflowDesign::validate`], which needs a malformed `basic` input.
    pub fn design_branch_offline(
        &self,
        query: &str,
        basic: &WorkflowDesign,
    ) -> Result<WorkflowDesign> {
        let (design, _) = passes::branch_design(query, basic);
        design.validate()?;
        Ok(design)
    }

    /// Runs the reflection pass deterministically over `with_branches`.
    ///
    /// The repairs run in this order and every one appends one note: duplicate
    /// node ids are renamed `<id>_2`, `<id>_3`, ...; edges naming an unknown
    /// node are dropped; a missing `start` node is inserted at the front and
    /// connected to the first node; a missing `end` node is appended and
    /// connected from every sink; an unreachable node is connected from the
    /// nearest earlier reachable node (the `start` node when none is earlier);
    /// and a cycle is broken by dropping the back edge `find_cycle` reports.
    /// An already-valid design yields no notes and is returned unchanged.
    ///
    /// `query` is the originating request: the deterministic repairs do not read
    /// it, and only the model-backed reflection pass uses it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the repaired design still fails
    /// [`WorkflowDesign::validate`] — an input no listed repair can fix.
    pub fn design_reflection_offline(
        &self,
        query: &str,
        with_branches: &WorkflowDesign,
    ) -> Result<(WorkflowDesign, Vec<String>)> {
        let _ = query;
        passes::reflection_design(with_branches)
    }

    /// Runs the three designer passes and reports each one.
    ///
    /// The passes run in Python's order — basic, branch, reflection — and each
    /// uses the model when one is wired, accepting an answer only when it
    /// parses, validates, and (for the branch pass, when the request holds a
    /// decision marker) actually contains a branch node. A rejected answer is
    /// never an error: that pass falls back to its deterministic result and
    /// records `fallback` with the reason. The result holds exactly three
    /// [`DesignPass`] records, one provenance record per pass, and one note per
    /// repair the reflection pass applied.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query, before any model call.
    pub async fn design_three_pass(&self, query: &str) -> Result<ThreePassDesign> {
        if query.trim().is_empty() {
            return Err(Error::validation("query must not be blank"));
        }

        let mut provenance = GenerationProvenance::new();
        let mut records = Vec::with_capacity(3);

        let basic = passes::run_basic(self, query).await?;
        provenance.record(
            ProgressStage::GeneratingWorkflowDesign,
            basic.pass.source,
            basic.pass.detail.clone(),
        );
        records.push(basic.pass);

        let branch = passes::run_branch(self, query, &basic.design).await?;
        provenance.record(
            ProgressStage::GeneratingWorkflowDesign,
            branch.pass.source,
            branch.pass.detail.clone(),
        );
        records.push(branch.pass);

        let reflection = passes::run_reflection(self, query, &branch.design).await?;
        provenance.record(
            ProgressStage::ValidatingDl,
            reflection.pass.source,
            reflection.pass.detail.clone(),
        );
        records.push(reflection.pass);

        provenance.warn_if_degraded();
        Ok(ThreePassDesign {
            design: reflection.design,
            passes: records,
            provenance,
            notes: reflection.notes,
        })
    }

    /// Streams the three designer passes as [`DesignEvent`]s.
    ///
    /// The stream owns a clone of the designer and of `query`, so it is
    /// `Send + 'static` and outlives both. It yields `pass_started` and
    /// `pass_completed` around each pass in Python's order, then one `completed`
    /// event holding the same [`ThreePassDesign`] [`Self::design_three_pass`]
    /// returns. A blank query yields one [`Error::Validation`] and ends the
    /// stream.
    pub fn stream_design(&self, query: &str) -> BoxStream<'static, Result<DesignEvent>> {
        stream::design_stream(self.clone(), query.to_string())
    }
}

impl Default for WorkflowDesigner {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for WorkflowDesigner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkflowDesigner")
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests;
