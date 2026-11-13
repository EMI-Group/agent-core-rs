//! The workflow-designer stage: a typed workflow design and its synthesis.
//!
//! Ports `openjiuwen.dev_tools.agent_builder.builders.workflow.workflow_designer`.
//! Python runs a three-stage model pipeline (basic design, branch design,
//! reflection evaluation) and answers with free-form Markdown; this port answers
//! with a typed [`WorkflowDesign`] that converts into the platform
//! [`WorkflowDsl`], and keeps a deterministic offline designer behind the same
//! entry point as the single model round trip.
//!
//! A design is validated before it is used: at least one node, unique ids,
//! resolvable edges, exactly one start node, and at least one end node.
//! [`WorkflowDesign::to_workflow_dsl`] is the one design-to-DSL converter.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::workflow_dsl::{DslEdge, DslNode, WorkflowDsl};
use crate::util::extract_json_fence;

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

/// Designs workflows from a natural-language request.
///
/// Without a model the designer is deterministic and offline: it splits the
/// request into steps and synthesizes a start/LLM/end workflow. With a model it
/// performs one round trip and falls back to the offline design whenever the
/// answer cannot be parsed into a valid [`WorkflowDesign`].
pub struct WorkflowDesigner {
    /// The optional model the model path talks to.
    model: Option<Arc<dyn Model>>,
}

impl WorkflowDesigner {
    /// Creates an offline designer; [`Self::design`] never calls a model.
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Creates a designer whose [`Self::design`] asks `model` first.
    #[must_use]
    pub fn with_model(model: Arc<dyn Model>) -> Self {
        Self { model: Some(model) }
    }

    /// Designs a workflow, using the model when one is configured.
    ///
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
        if query.trim().is_empty() {
            return Err(Error::validation("query must not be blank"));
        }
        let Some(model) = self.model.as_deref() else {
            return self.design_offline(query);
        };

        match design_with_model(model, query).await {
            Ok(design) => Ok(design),
            Err(error) => {
                tracing::debug!(
                    %error,
                    "workflow design failed; falling back to the offline designer"
                );
                self.design_offline(query)
            }
        }
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
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query.
    pub fn design_offline(&self, query: &str) -> Result<WorkflowDesign> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(Error::validation("query must not be blank"));
        }

        let mut steps = task_clauses(trimmed);
        if steps.is_empty() {
            steps.push(trimmed.to_string());
        }

        let mut nodes = vec![DesignNode {
            id: "start".to_string(),
            kind: DesignNodeKind::Start,
            title: "Start".to_string(),
            description: "Workflow entry".to_string(),
            config: json!({"inputs": {"query": {"type": "string", "description": "user input"}}}),
        }];
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
            edges.push(DesignEdge {
                source: previous,
                target: id.clone(),
                condition: None,
            });
            previous = id;
        }

        if needs_branch(trimmed) {
            nodes.push(DesignNode {
                id: "branch".to_string(),
                kind: DesignNodeKind::Branch,
                title: "Decide".to_string(),
                description: "Choose the next step".to_string(),
                config: json!({"conditions": ["default"]}),
            });
            edges.push(DesignEdge {
                source: previous,
                target: "branch".to_string(),
                condition: None,
            });
            previous = "branch".to_string();
        }

        nodes.push(DesignNode {
            id: "end".to_string(),
            kind: DesignNodeKind::End,
            title: "End".to_string(),
            description: "Workflow exit".to_string(),
            config: Value::Null,
        });
        edges.push(DesignEdge {
            source: previous,
            target: "end".to_string(),
            condition: None,
        });

        let design = WorkflowDesign {
            name: slugify(&steps[0]),
            description: truncate_chars(trimmed, DESCRIPTION_LIMIT),
            nodes,
            edges,
        };
        design.validate()?;
        Ok(design)
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

/// Asks the model for one fenced design document and validates it.
async fn design_with_model(model: &dyn Model, query: &str) -> Result<WorkflowDesign> {
    let messages = vec![
        BaseMessage::system(DESIGN_SYSTEM_PROMPT),
        BaseMessage::user(query),
    ];
    let request = ModelRequestConfig::new(model.model_name());
    let response = model.invoke(&messages, &request).await?;
    let Some(choice) = response.choices.into_iter().next() else {
        return Err(Error::llm("model returned no choices"));
    };

    let text = choice.message.text().into_owned();
    let value = extract_json_fence(&text)
        .ok_or_else(|| Error::validation("model answer contained no JSON code fence"))?;
    let design: WorkflowDesign = serde_json::from_value(value).map_err(|error| {
        Error::validation(format!("failed to parse workflow design JSON: {error}"))
    })?;
    design.validate()?;
    Ok(design)
}

/// Splits `query` into at most `MAX_STEPS` task clauses.
fn task_clauses(query: &str) -> Vec<String> {
    let mut clauses: Vec<String> = Vec::new();
    for numbered in split_numbered(query) {
        for sentence in split_terminators(&numbered) {
            for sequenced in split_sequences(sentence) {
                let clause = strip_list_marker(&sequenced).trim();
                if clause.chars().all(|character| !character.is_alphabetic()) {
                    continue;
                }
                if clauses.iter().any(|seen| seen == clause) {
                    continue;
                }
                clauses.push(clause.to_string());
                if clauses.len() == MAX_STEPS {
                    return clauses;
                }
            }
        }
    }
    clauses
}

/// Splits numbered list items (`1.`, `2)`, `3、`) out of `text`.
fn split_numbered(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;

    while index < text.len() {
        let bytes = text.as_bytes();
        let at_boundary = index == 0 || bytes[index - 1].is_ascii_whitespace();
        if at_boundary && bytes[index].is_ascii_digit() {
            let mut digits_end = index;
            while digits_end < text.len() && bytes[digits_end].is_ascii_digit() {
                digits_end += 1;
            }
            let rest = &text[digits_end..];
            let marker_len = if rest.starts_with('、') {
                '、'.len_utf8()
            } else if rest.starts_with('.') || rest.starts_with(')') {
                let after = &rest[1..];
                usize::from(after.is_empty() || after.starts_with(char::is_whitespace))
            } else {
                0
            };
            if marker_len > 0 {
                if index > start {
                    parts.push(text[start..index].to_string());
                    start = index;
                }
                index = digits_end + marker_len;
                continue;
            }
        }
        index += 1;
    }

    parts.push(text[start..].to_string());
    parts
}

/// Splits `text` at every sentence terminator, dropping the terminator.
fn split_terminators(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    for (index, character) in text.char_indices() {
        if TERMINATORS.contains(&character) {
            parts.push(&text[start..index]);
            start = index + character.len_utf8();
        }
    }
    parts.push(&text[start..]);
    parts
}

/// Splits `text` at every sequencing phrase, dropping the phrase itself.
fn split_sequences(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut rest = text;
    while let Some((start, length)) = find_sequence_marker(rest) {
        parts.push(rest[..start].to_string());
        rest = &rest[start + length..];
    }
    parts.push(rest.to_string());
    parts
}

/// The earliest sequencing phrase in `text`, as `(offset, length)`.
///
/// Among phrases starting at the same offset the longest one wins, so
/// `", then "` is preferred over `" then "`.
fn find_sequence_marker(text: &str) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (index, _) in text.char_indices() {
        if best.is_some_and(|(position, _)| position < index) {
            break;
        }
        for marker in SEQUENCE_MARKERS {
            let Some(length) = match_prefix_ci(&text[index..], marker) else {
                continue;
            };
            best = match best {
                Some((position, current)) if position == index && current >= length => {
                    Some((position, current))
                }
                _ => Some((index, length)),
            };
        }
    }
    best
}

/// The byte length of `prefix` when `text` starts with it, ignoring ASCII case.
fn match_prefix_ci(text: &str, prefix: &str) -> Option<usize> {
    let mut characters = text.chars();
    for wanted in prefix.chars() {
        if !characters.next()?.eq_ignore_ascii_case(&wanted) {
            return None;
        }
    }
    Some(prefix.len())
}

/// Strips one leading list marker (`-`, `*`, `1.`, `1)`, `1、`) from `text`.
fn strip_list_marker(text: &str) -> &str {
    let trimmed = text.trim_start();
    for marker in ['-', '*'] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return rest.trim_start();
        }
    }

    let mut digits_end = 0usize;
    for (index, character) in trimmed.char_indices() {
        if !character.is_ascii_digit() {
            break;
        }
        digits_end = index + character.len_utf8();
    }
    if digits_end > 0 {
        let rest = &trimmed[digits_end..];
        for marker in ['.', ')', '、'] {
            if let Some(rest) = rest.strip_prefix(marker) {
                return rest.trim_start();
            }
        }
    }
    trimmed
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

/// A deterministic title for an LLM step.
///
/// CJK clauses keep their first `CJK_TITLE_CHARS` characters, other clauses
/// their first `TITLE_WORDS` words; trailing punctuation is dropped. A clause
/// with no usable title falls back to `Step <position>`.
fn step_title(clause: &str, position: usize) -> String {
    let title = if clause.chars().any(is_cjk) {
        clause.chars().take(CJK_TITLE_CHARS).collect::<String>()
    } else {
        clause
            .split_whitespace()
            .take(TITLE_WORDS)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let title = title
        .trim()
        .trim_end_matches(is_trailing_punctuation)
        .trim();
    if title.is_empty() {
        format!("Step {position}")
    } else {
        title.to_string()
    }
}

/// Whether `character` ends a clause fragment and should not end a title.
fn is_trailing_punctuation(character: char) -> bool {
    matches!(
        character,
        '.' | ',' | ';' | ':' | '!' | '?' | '。' | '，' | '；' | '：' | '！' | '？' | '、'
    )
}

/// Whether `character` is a CJK ideograph or kana.
fn is_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{3040}'..='\u{30FF}'   // hiragana and katakana
        | '\u{3400}'..='\u{4DBF}' // CJK unified ideographs extension A
        | '\u{4E00}'..='\u{9FFF}' // CJK unified ideographs
        | '\u{F900}'..='\u{FAFF}' // CJK compatibility ideographs
    )
}

/// Slugifies `text` into a workflow name.
///
/// Alphanumerics (including CJK) are kept, every other run becomes one `-`, the
/// result is trimmed of `-` and capped at `NAME_LIMIT` characters. Text with
/// nothing to keep yields `"workflow"`.
fn slugify(text: &str) -> String {
    let mut slug = String::new();
    let mut separator = false;
    for character in text.chars() {
        if character.is_alphanumeric() {
            if separator && !slug.is_empty() {
                slug.push('-');
            }
            separator = false;
            slug.extend(character.to_lowercase());
        } else {
            separator = true;
        }
    }

    let capped: String = slug.chars().take(NAME_LIMIT).collect();
    let trimmed = capped.trim_matches('-');
    if trimmed.is_empty() {
        "workflow".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Truncates `text` to at most `limit` characters.
fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    text.chars().take(limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use serde_json::json;

    /// A model answer holding a valid design that uses Python node-type aliases.
    const MODEL_FENCE: &str = r#"Here is the design you asked for.
```json
{"name": "order-flow", "description": "handle an order",
 "nodes": [
   {"id": "start", "type": "Start", "title": "Start", "description": "entry"},
   {"id": "work", "type": "LLM", "title": "Do the work",
    "config": {"instruction": "handle the order"}},
   {"id": "done", "type": "Output", "title": "Done", "description": "exit"}],
 "edges": [{"source": "start", "target": "work"}, {"source": "work", "target": "done"}]}
```"#;

    /// A model answer whose design has no start node, so it fails validation.
    const INVALID_FENCE: &str = r#"```json
{"name": "broken", "description": "no start node",
 "nodes": [{"id": "work", "type": "llm"}, {"id": "done", "type": "end"}],
 "edges": [{"source": "work", "target": "done"}]}
```"#;

    /// Coerces a scripted model into the trait object the designer takes.
    fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
        let cloned: Arc<ScriptedModel> = Arc::clone(model);
        cloned
    }

    fn node(id: &str, kind: DesignNodeKind) -> DesignNode {
        DesignNode {
            id: id.to_string(),
            kind,
            ..DesignNode::default()
        }
    }

    fn edge(source: &str, target: &str) -> DesignEdge {
        DesignEdge {
            source: source.to_string(),
            target: target.to_string(),
            condition: None,
        }
    }

    fn sample_design() -> WorkflowDesign {
        WorkflowDesign {
            name: "demo".to_string(),
            description: "a demo design".to_string(),
            nodes: vec![
                node("start", DesignNodeKind::Start),
                node("work", DesignNodeKind::Llm),
                node("end", DesignNodeKind::End),
            ],
            edges: vec![edge("start", "work"), edge("work", "end")],
        }
    }

    fn llm_nodes(design: &WorkflowDesign) -> Vec<&DesignNode> {
        design
            .nodes
            .iter()
            .filter(|node| node.kind == DesignNodeKind::Llm)
            .collect()
    }

    fn descriptions(design: &WorkflowDesign) -> Vec<&str> {
        llm_nodes(design)
            .iter()
            .map(|node| node.description.as_str())
            .collect()
    }

    fn pairs(design: &WorkflowDesign) -> Vec<(&str, &str)> {
        design
            .edges
            .iter()
            .map(|edge| (edge.source.as_str(), edge.target.as_str()))
            .collect()
    }

    #[test]
    fn offline_design_synthesizes_a_single_step() {
        let design = WorkflowDesigner::new()
            .design_offline("Summarize the document.")
            .expect("offline design");

        let ids: Vec<&str> = design.nodes.iter().map(|node| node.id.as_str()).collect();
        assert_eq!(ids, ["start", "llm_1", "end"]);
        assert_eq!(design.nodes[0].kind, DesignNodeKind::Start);
        assert_eq!(design.nodes[1].kind, DesignNodeKind::Llm);
        assert_eq!(design.nodes[2].kind, DesignNodeKind::End);
        assert_eq!(design.nodes[1].title, "Summarize the document");
        assert_eq!(descriptions(&design), ["Summarize the document"]);
        assert_eq!(
            pairs(&design),
            [("start", "llm_1"), ("llm_1", "end")],
            "two sequential edges"
        );
        assert_eq!(design.name, "summarize-the-document");
        assert_eq!(design.description, "Summarize the document.");

        design.validate().expect("design is valid");
        let dsl = design.to_workflow_dsl().expect("design converts");
        dsl.validate().expect("dsl is valid");
        assert_eq!(dsl.nodes[0].node_type, "start");
        assert_eq!(
            dsl.nodes[0].inputs.get("query"),
            Some(&json!({"type": "string", "description": "user input"}))
        );
        assert_eq!(
            dsl.nodes[1].data["instruction"],
            json!("Summarize the document")
        );
    }

    #[test]
    fn offline_design_splits_sequenced_steps() {
        let design = WorkflowDesigner::new()
            .design_offline(
                "Collect the order. Then validate the address. Then generate the label.",
            )
            .expect("offline design");

        assert_eq!(
            descriptions(&design),
            [
                "Collect the order",
                "validate the address",
                "generate the label"
            ]
        );
        assert_eq!(
            pairs(&design),
            [
                ("start", "llm_1"),
                ("llm_1", "llm_2"),
                ("llm_2", "llm_3"),
                ("llm_3", "end")
            ]
        );
        design.validate().expect("design is valid");
    }

    #[test]
    fn offline_design_caps_the_number_of_steps() {
        let design = WorkflowDesigner::new()
            .design_offline("One. Two. Three. Four. Five. Six.")
            .expect("offline design");
        assert_eq!(llm_nodes(&design).len(), MAX_STEPS);
        assert_eq!(design.edges.len(), MAX_STEPS + 1);
        design.validate().expect("design is valid");
    }

    #[test]
    fn offline_design_splits_numbered_steps() {
        let design = WorkflowDesigner::new()
            .design_offline("1. Collect the order. 2) Validate the address. 3、Generate the label.")
            .expect("offline design");
        assert_eq!(
            descriptions(&design),
            [
                "Collect the order",
                "Validate the address",
                "Generate the label"
            ]
        );
        design.validate().expect("design is valid");
    }

    #[test]
    fn offline_design_dedupes_repeated_steps() {
        let design = WorkflowDesigner::new()
            .design_offline("Check the order. Check the order. Send it.")
            .expect("offline design");
        assert_eq!(descriptions(&design), ["Check the order", "Send it"]);
    }

    #[test]
    fn offline_design_adds_a_branch_node() {
        let design = WorkflowDesigner::new()
            .design_offline(
                "Check the request. If the order is valid, confirm it; otherwise reject it.",
            )
            .expect("offline design");

        let branches: Vec<&DesignNode> = design
            .nodes
            .iter()
            .filter(|node| node.kind == DesignNodeKind::Branch)
            .collect();
        assert_eq!(branches.len(), 1, "exactly one branch node");
        assert_eq!(branches[0].id, "branch");
        assert_eq!(branches[0].title, "Decide");
        assert!(
            design
                .edges
                .iter()
                .any(|edge| edge.source == "branch" && edge.target == "end"),
            "the branch leads to the end node"
        );
        assert_eq!(llm_nodes(&design).len(), 3);
        design.validate().expect("design is valid");
    }

    #[test]
    fn offline_design_does_not_branch_on_unrelated_words() {
        let design = WorkflowDesigner::new()
            .design_offline("Classify the request.")
            .expect("offline design");
        assert!(design
            .nodes
            .iter()
            .all(|node| node.kind != DesignNodeKind::Branch));
        assert_eq!(design.nodes.len(), 3);
    }

    #[test]
    fn offline_design_handles_cjk_requests() {
        let design = WorkflowDesigner::new()
            .design_offline("创建客服助手。然后回答问题。")
            .expect("offline design");

        let steps = llm_nodes(&design);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].description, "创建客服助手");
        assert_eq!(steps[1].description, "回答问题");
        assert_ne!(steps[0].title, "");
        assert_ne!(steps[1].title, "");
        assert_eq!(design.name, "创建客服助手");
        design.validate().expect("design is valid");
    }

    #[test]
    fn offline_design_falls_back_to_one_step_and_a_generic_name() {
        let design = WorkflowDesigner::new()
            .design_offline("...")
            .expect("offline design");
        assert_eq!(descriptions(&design), ["..."]);
        assert_eq!(llm_nodes(&design)[0].title, "Step 1");
        assert_eq!(design.name, "workflow");
        design.validate().expect("design is valid");
    }

    #[test]
    fn offline_design_is_deterministic() {
        let designer = WorkflowDesigner::new();
        let first = designer
            .design_offline("Collect the order. Then send it.")
            .expect("offline design");
        let second = designer
            .design_offline("Collect the order. Then send it.")
            .expect("offline design");
        assert_eq!(first, second);
    }

    #[test]
    fn offline_design_rejects_a_blank_query() {
        let error = WorkflowDesigner::new()
            .design_offline("   \n\t ")
            .expect_err("blank query");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("query must not be blank"));
    }

    #[tokio::test]
    async fn blank_query_never_reaches_the_model() {
        let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
        let designer = WorkflowDesigner::with_model(dynamic(&model));

        let error = designer.design("  ").await.expect_err("blank query");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("query must not be blank"));
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn design_returns_the_model_document() {
        let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
        let designer = WorkflowDesigner::with_model(dynamic(&model));

        let design = designer.design("handle an order").await.expect("design");
        assert_eq!(design.name, "order-flow");
        assert_eq!(design.description, "handle an order");
        assert_eq!(design.nodes.len(), 3);
        assert_eq!(design.nodes[1].kind, DesignNodeKind::Llm);
        assert_eq!(design.nodes[2].kind, DesignNodeKind::End);
        assert_eq!(
            design.nodes[1].config["instruction"],
            json!("handle the order")
        );
        design.validate().expect("design is valid");

        assert_eq!(model.call_count(), 1);
        let call = model.last_call().expect("one call");
        assert_eq!(call.messages.len(), 2);
        assert_eq!(call.messages[0].text(), DESIGN_SYSTEM_PROMPT);
        assert_eq!(call.messages[1].text(), "handle an order");
    }

    #[tokio::test]
    async fn model_problems_fall_back_to_the_offline_designer() {
        let query = "Collect the order.";
        let offline = WorkflowDesigner::new()
            .design_offline(query)
            .expect("offline design");
        let scenarios = [
            vec![ScriptedTurn::text(INVALID_FENCE)],
            vec![ScriptedTurn::text("which workflow do you want?")],
            vec![ScriptedTurn::text("```json\n{not: valid}\n```")],
            vec![ScriptedTurn::fail("boom")],
        ];

        for turns in scenarios {
            let model = Arc::new(ScriptedModel::new(turns));
            let designer = WorkflowDesigner::with_model(dynamic(&model));
            let design = designer.design(query).await.expect("fallback design");
            assert_eq!(design, offline, "the offline design replaces the answer");
            assert_eq!(model.call_count(), 1);
        }
    }

    #[test]
    fn node_kind_parses_wire_names_and_aliases() {
        let cases = [
            ("start", DesignNodeKind::Start),
            ("BEGIN", DesignNodeKind::Start),
            ("Entry", DesignNodeKind::Start),
            ("llm", DesignNodeKind::Llm),
            ("LLM", DesignNodeKind::Llm),
            ("llm_node", DesignNodeKind::Llm),
            ("llm-nodes", DesignNodeKind::Llm),
            ("LLMNode", DesignNodeKind::Llm),
            ("Questioner", DesignNodeKind::Llm),
            ("generate", DesignNodeKind::Llm),
            ("branch", DesignNodeKind::Branch),
            ("Condition", DesignNodeKind::Branch),
            ("conditions", DesignNodeKind::Branch),
            ("router", DesignNodeKind::Branch),
            ("switch", DesignNodeKind::Branch),
            ("decision", DesignNodeKind::Branch),
            ("IntentDetection", DesignNodeKind::Branch),
            ("tool", DesignNodeKind::Tool),
            ("Plugin", DesignNodeKind::Tool),
            ("Code", DesignNodeKind::Tool),
            ("api", DesignNodeKind::Tool),
            ("function", DesignNodeKind::Tool),
            ("end", DesignNodeKind::End),
            ("Output", DesignNodeKind::End),
            ("Finish", DesignNodeKind::End),
            ("Exit", DesignNodeKind::End),
            ("end_node", DesignNodeKind::End),
        ];
        for (value, expected) in cases {
            assert_eq!(
                DesignNodeKind::from_str(value).expect("known kind"),
                expected,
                "alias {value}"
            );
            assert_eq!(
                value.parse::<DesignNodeKind>().expect("known kind"),
                expected,
                "alias {value} through FromStr"
            );
        }
    }

    #[test]
    fn node_kind_rejects_unknown_names() {
        for value in ["", "widget", "start-node"] {
            let error = DesignNodeKind::from_str(value).expect_err("unknown kind");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("unknown design node type"));
            assert!(error.to_string().contains(value));
            assert!(value.parse::<DesignNodeKind>().is_err());
            assert!(DesignNodeKind::try_from(value.to_string()).is_err());
        }
    }

    #[test]
    fn node_kind_round_trips_through_display() {
        for kind in [
            DesignNodeKind::Start,
            DesignNodeKind::Llm,
            DesignNodeKind::Branch,
            DesignNodeKind::Tool,
            DesignNodeKind::End,
        ] {
            assert_eq!(kind.as_str(), kind.to_string());
            assert_eq!(
                kind.to_string()
                    .parse::<DesignNodeKind>()
                    .expect("round trip"),
                kind
            );
            assert_eq!(String::from(kind), kind.as_str());
        }
        assert_eq!(DesignNodeKind::default(), DesignNodeKind::Start);
    }

    #[test]
    fn node_serializes_the_lowercase_kind() {
        let node: DesignNode =
            serde_json::from_str(r#"{"id": "n1", "type": "Output", "title": "Out"}"#)
                .expect("parse");
        assert_eq!(node.kind, DesignNodeKind::End);

        let value = serde_json::to_value(&node).expect("serialize");
        assert_eq!(value["type"], json!("end"));
        assert_eq!(value["id"], json!("n1"));
        assert_eq!(value["config"], Value::Null);
        assert_eq!(
            serde_json::from_value::<DesignNode>(value).expect("reparse"),
            node
        );

        let defaulted: DesignNode = serde_json::from_str(r#"{"id": "n2"}"#).expect("parse");
        assert_eq!(defaulted.kind, DesignNodeKind::Start);
    }

    #[test]
    fn design_round_trips_through_json() {
        let design = sample_design();
        let text = serde_json::to_string(&design).expect("serialize");
        let value: Value = serde_json::from_str(&text).expect("reparse");
        assert_eq!(value["nodes"][1]["type"], json!("llm"));
        assert!(value["edges"][0]["condition"].is_null());
        let parsed: WorkflowDesign = serde_json::from_str(&text).expect("parse");
        assert_eq!(parsed, design);
    }

    #[test]
    fn to_workflow_dsl_maps_nodes_and_config() {
        let mut design = sample_design();
        design.nodes[0].config = json!({"inputs": {"query": {"type": "string"}}});
        design.nodes[1].description = "do the work".to_string();
        design.nodes[1].config = json!({
            "instruction": "do the work",
            "inputs": {"question": {"type": "string"}},
            "outputs": {"result": {"type": "string"}},
        });
        design.edges[1].condition = Some("done".to_string());

        let dsl = design.to_workflow_dsl().expect("converts");
        assert_eq!(dsl.name, "demo");
        assert_eq!(dsl.description, "a demo design");
        let types: Vec<&str> = dsl
            .nodes
            .iter()
            .map(|node| node.node_type.as_str())
            .collect();
        assert_eq!(types, ["start", "llm", "end"]);
        assert_eq!(dsl.nodes[1].description, "do the work");
        assert_eq!(
            dsl.nodes[1].inputs,
            BTreeMap::from([("question".to_string(), json!({"type": "string"}))])
        );
        assert_eq!(
            dsl.nodes[1].outputs,
            BTreeMap::from([("result".to_string(), json!({"type": "string"}))])
        );
        assert_eq!(dsl.nodes[1].data, design.nodes[1].config);
        assert!(dsl.nodes[2].inputs.is_empty());
        assert_eq!(dsl.edges[1].condition.as_deref(), Some("done"));
        dsl.validate().expect("dsl is valid");
    }

    #[test]
    fn validate_rejects_structural_errors() {
        let error = WorkflowDesign::default().validate().expect_err("no nodes");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("workflow design has no nodes"));

        let mut duplicate = sample_design();
        duplicate.nodes[1].id = "end".to_string();
        duplicate.edges.clear();
        let error = duplicate.validate().expect_err("duplicate ids");
        assert!(error.to_string().contains("duplicate node id: end"));

        let mut dangling = sample_design();
        dangling.edges[1].target = "ghost".to_string();
        let error = dangling.validate().expect_err("dangling edge");
        assert!(error
            .to_string()
            .contains("edge references unknown node: ghost"));

        let mut no_start = sample_design();
        no_start.nodes[0].kind = DesignNodeKind::Tool;
        let error = no_start.validate().expect_err("no start node");
        assert!(error
            .to_string()
            .contains("workflow design needs exactly one start node"));

        let mut two_starts = sample_design();
        two_starts.nodes[1].kind = DesignNodeKind::Start;
        let error = two_starts.validate().expect_err("two start nodes");
        assert!(error
            .to_string()
            .contains("workflow design needs exactly one start node"));

        let mut no_end = sample_design();
        no_end.nodes[2].kind = DesignNodeKind::Tool;
        let error = no_end.validate().expect_err("no end node");
        assert!(error
            .to_string()
            .contains("workflow design needs an end node"));

        sample_design().validate().expect("the sample is valid");
    }

    #[test]
    fn to_workflow_dsl_rejects_an_invalid_design() {
        let mut design = sample_design();
        design.nodes.pop();
        let error = design.to_workflow_dsl().expect_err("invalid design");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn debug_reports_whether_a_model_is_configured() {
        assert_eq!(
            format!("{:?}", WorkflowDesigner::new()),
            "WorkflowDesigner { has_model: false }"
        );
        assert_eq!(
            format!("{:?}", WorkflowDesigner::default()),
            "WorkflowDesigner { has_model: false }"
        );

        let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
        let wired = WorkflowDesigner::with_model(dynamic(&model));
        assert_eq!(format!("{wired:?}"), "WorkflowDesigner { has_model: true }");
    }
}
