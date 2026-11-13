//! Execution traces: normalized capability graphs and their outcome.
//!
//! A trace payload is normalized at the boundary: `skill:`/`capability:`
//! prefixes are stripped from node ids, blank nodes are dropped, and only
//! successful edges with known endpoints survive. The rest of the flow engine
//! only ever sees this normalized form.
use crate::error::{validation, Result};
use crate::flow::model::{normalize_capability_id, CAN_FEED};
use crate::model::CapabilityIo;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
/// The graph type used by an execution trace when the payload omits it.
const DEFAULT_GRAPH_TYPE: &str = "execution_graph";
/// The capability type assumed when a trace node declares none.
const DEFAULT_CAPABILITY_TYPE: &str = "skill";

/// The normalized outcome of one execution trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TraceOutcome {
    /// The traced task completed successfully.
    Success,
    /// The traced task failed.
    Failed,
    /// The traced task completed only partially.
    Partial,
    /// The payload carried an unrecognized outcome.
    #[default]
    Unknown,
}

impl TraceOutcome {
    /// Parses a raw outcome string using Python's alias sets.
    ///
    /// `success`/`succeeded`/`ok`/`done` map to [`TraceOutcome::Success`],
    /// `failed`/`failure`/`error` to [`TraceOutcome::Failed`],
    /// `partial`/`partially` to [`TraceOutcome::Partial`], and everything else
    /// (including a blank value) to [`TraceOutcome::Unknown`].
    #[must_use]
    pub fn from_raw(raw: &str) -> Self {
        let lowered = raw.trim().to_lowercase();
        match lowered.as_str() {
            "success" | "succeeded" | "ok" | "done" => Self::Success,
            "failed" | "failure" | "error" => Self::Failed,
            "partial" | "partially" => Self::Partial,
            _ => Self::Unknown,
        }
    }

    /// Returns the normalized wire spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failed => "failed",
            Self::Partial => "partial",
            Self::Unknown => "unknown",
        }
    }

    /// Reports whether the outcome is a fully successful execution.
    #[must_use]
    pub fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

impl fmt::Display for TraceOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for TraceOutcome {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for TraceOutcome {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::from_raw(&raw))
    }
}

/// One capability node inside an [`ExecutionGraph`].
///
/// IO ports are normalized to the fields a plan needs (`name`, `type`,
/// `required`, `description`); a port without a name or type is dropped, which
/// mirrors Python's `_normalize_capability_ports`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TraceNode {
    /// The capability id, with any `skill:`/`capability:` prefix removed.
    pub capability_id: String,
    /// The capability kind; defaults to `skill`.
    pub capability_type: String,
    /// The skill sub-kind recorded by the trace (`skill` for ordinary skills).
    pub skill_type: String,
    /// The provider-reported capability version.
    pub version: String,
    /// The capability description.
    pub description: String,
    /// The declared inputs.
    pub inputs: Vec<CapabilityIo>,
    /// The declared outputs.
    pub outputs: Vec<CapabilityIo>,
}

impl Default for TraceNode {
    fn default() -> Self {
        Self {
            capability_id: String::new(),
            capability_type: DEFAULT_CAPABILITY_TYPE.to_string(),
            skill_type: String::new(),
            version: String::new(),
            description: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

impl TraceNode {
    /// Renders the Python recipe `combination_structure.nodes` shape.
    ///
    /// The value is `{"label": "capability", "metadata": {...}}`, where the
    /// metadata carries `capability_type`, `skill_type` (when non-empty),
    /// `version`, `description`, `inputs`, and `outputs`.
    #[must_use]
    pub fn to_combination_value(&self) -> Value {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "capability_type".to_string(),
            Value::String(self.capability_type.clone()),
        );
        if !self.skill_type.is_empty() {
            metadata.insert(
                "skill_type".to_string(),
                Value::String(self.skill_type.clone()),
            );
        }
        metadata.insert("version".to_string(), Value::String(self.version.clone()));
        metadata.insert(
            "description".to_string(),
            Value::String(self.description.clone()),
        );
        metadata.insert("inputs".to_string(), ports_value(&self.inputs));
        metadata.insert("outputs".to_string(), ports_value(&self.outputs));
        serde_json::json!({
            "label": "capability",
            "metadata": metadata,
        })
    }

    /// Parses the Python recipe `combination_structure.nodes` shape.
    ///
    /// A value without a `metadata` object is treated as the metadata itself, so
    /// hand-written structures remain readable.
    #[must_use]
    pub fn from_combination_value(capability_id: &str, value: &Value) -> Self {
        let object = value.as_object();
        let metadata = object
            .and_then(|map| map.get("metadata"))
            .and_then(Value::as_object)
            .or(object);
        build_node(capability_id, object, metadata)
    }
}

/// Renders ports as a JSON array of `name`/`type`/`required`/`description`.
fn ports_value(ports: &[CapabilityIo]) -> Value {
    Value::Array(
        ports
            .iter()
            .map(|port| {
                serde_json::json!({
                    "name": port.name,
                    "type": port.io_type,
                    "required": port.required,
                    "description": port.description,
                })
            })
            .collect(),
    )
}

/// Builds a [`TraceNode`] from a raw node object and its metadata object.
fn build_node(
    capability_id: &str,
    object: Option<&serde_json::Map<String, Value>>,
    metadata: Option<&serde_json::Map<String, Value>>,
) -> TraceNode {
    let label = object
        .and_then(|map| map.get("label"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let declared_type = metadata
        .and_then(|map| map.get("capability_type"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let capability_type = if declared_type.is_empty() {
        if label.is_empty() {
            DEFAULT_CAPABILITY_TYPE.to_string()
        } else {
            label
        }
    } else {
        declared_type
    };
    let skill_type = metadata
        .and_then(|map| map.get("skill_type"))
        .map(|value| value.as_str().unwrap_or_default().trim().to_lowercase())
        .unwrap_or_default();
    let version = metadata
        .and_then(|map| map.get("version"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let description = metadata
        .and_then(|map| map.get("description"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let inputs = normalize_ports(metadata.and_then(|map| map.get("inputs")));
    let outputs = normalize_ports(metadata.and_then(|map| map.get("outputs")));
    TraceNode {
        capability_id: capability_id.to_string(),
        capability_type,
        skill_type,
        version,
        description,
        inputs,
        outputs,
    }
}

/// Normalizes a raw ports array, dropping entries without a name or type.
fn normalize_ports(value: Option<&Value>) -> Vec<CapabilityIo> {
    let Some(items) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let object = item.as_object()?;
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            let io_type = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            if name.is_empty() || io_type.is_empty() {
                return None;
            }
            let description = object
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            Some(CapabilityIo {
                name,
                io_type,
                description,
                required: object.get("required").and_then(Value::as_bool),
                default: None,
                metadata: BTreeMap::new(),
            })
        })
        .collect()
}

/// One directed collaboration edge inside an [`ExecutionGraph`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TraceEdge {
    /// The normalized source capability id.
    pub source: String,
    /// The normalized target capability id.
    pub target: String,
    /// The relation; defaults to `can_feed`.
    pub relation: String,
    /// Whether the traced hand-off succeeded.
    pub success: bool,
    /// An optional failure/observation reason preserved for traceability.
    pub reason: String,
}

impl Default for TraceEdge {
    fn default() -> Self {
        Self {
            source: String::new(),
            target: String::new(),
            relation: CAN_FEED.to_string(),
            success: false,
            reason: String::new(),
        }
    }
}

/// A normalized capability-level execution graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExecutionGraph {
    /// The trace-supplied graph id.
    pub id: String,
    /// The graph type; the JSON key is `"type"` for Python wire compatibility.
    #[serde(rename = "type", alias = "graph_type")]
    pub graph_type: String,
    /// A free-form graph label.
    pub label: String,
    /// Whether the edges are directed.
    pub directed: bool,
    /// Nodes keyed by normalized capability id.
    pub nodes: BTreeMap<String, TraceNode>,
    /// Successful edges, sorted by `(source, target, relation)`.
    pub edges: Vec<TraceEdge>,
}

impl Default for ExecutionGraph {
    fn default() -> Self {
        Self {
            id: String::new(),
            graph_type: DEFAULT_GRAPH_TYPE.to_string(),
            label: String::new(),
            directed: true,
            nodes: BTreeMap::new(),
            edges: Vec::new(),
        }
    }
}

/// One immutable execution evidence record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FlowTrace {
    /// The idempotency key of the trace.
    pub trace_id: String,
    /// The user query that produced the trace.
    pub query: String,
    /// The normalized execution outcome.
    pub outcome: TraceOutcome,
    /// The normalized execution graph.
    pub graph: ExecutionGraph,
}

impl Default for FlowTrace {
    fn default() -> Self {
        Self {
            trace_id: String::new(),
            query: String::new(),
            outcome: TraceOutcome::Unknown,
            graph: ExecutionGraph::default(),
        }
    }
}

impl FlowTrace {
    /// Normalizes one raw trace payload.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::validation)
    /// when the payload is not a JSON object, `trace_id` is blank, or `graph` is
    /// missing/not an object.
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_value(value: Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| validation("trace", "must be a JSON object"))?;
        let trace_id = object
            .get("trace_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if trace_id.is_empty() {
            return Err(validation("trace_id", "must not be blank"));
        }
        let graph_value = object
            .get("graph")
            .ok_or_else(|| validation("graph", "must be a JSON object"))?;
        let graph_object = graph_value
            .as_object()
            .ok_or_else(|| validation("graph", "must be a JSON object"))?;
        Ok(Self {
            trace_id,
            query: object
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string(),
            outcome: TraceOutcome::from_raw(
                object
                    .get("outcome")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ),
            graph: normalize_graph(graph_object),
        })
    }

    /// Returns the sorted successful edge triples of this trace.
    #[must_use]
    pub fn successful_edges(&self) -> Vec<(String, String, String)> {
        self.graph
            .edges
            .iter()
            .filter(|edge| edge.success)
            .map(|edge| {
                (
                    edge.source.clone(),
                    edge.target.clone(),
                    edge.relation.clone(),
                )
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Normalizes a raw graph object per the Python `normalize_execution_graph`
/// contract, keeping only truthy-success edges with known endpoints.
fn normalize_graph(object: &serde_json::Map<String, Value>) -> ExecutionGraph {
    let nodes = normalize_nodes(object);
    let edges = normalize_edges(object, &nodes);
    ExecutionGraph {
        id: object
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string(),
        graph_type: object
            .get("type")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(DEFAULT_GRAPH_TYPE)
            .trim()
            .to_string(),
        label: object
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string(),
        directed: object
            .get("directed")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        nodes,
        edges,
    }
}
/// Normalizes the `nodes` object of a raw graph, dropping blank ids.
fn normalize_nodes(object: &serde_json::Map<String, Value>) -> BTreeMap<String, TraceNode> {
    let mut nodes: BTreeMap<String, TraceNode> = BTreeMap::new();
    if let Some(raw_nodes) = object.get("nodes").and_then(Value::as_object) {
        for (raw_id, raw_node) in raw_nodes {
            let capability_id = normalize_capability_id(raw_id);
            if capability_id.is_empty() {
                continue;
            }
            let Some(node_object) = raw_node.as_object() else {
                continue;
            };
            let metadata = node_object.get("metadata").and_then(Value::as_object);
            let node = build_node(&capability_id, Some(node_object), metadata);
            nodes.insert(capability_id, node);
        }
    }
    nodes
}
/// Normalizes the successful `edges` of a raw graph, keeping known endpoints.
fn normalize_edges(
    object: &serde_json::Map<String, Value>,
    nodes: &BTreeMap<String, TraceNode>,
) -> Vec<TraceEdge> {
    let mut edges: Vec<TraceEdge> = Vec::new();
    let mut seen: BTreeSet<(String, String, String)> = BTreeSet::new();
    if let Some(raw_edges) = object.get("edges").and_then(Value::as_array) {
        for raw_edge in raw_edges {
            let Some(edge_object) = raw_edge.as_object() else {
                continue;
            };
            let metadata = edge_object.get("metadata").and_then(Value::as_object);
            let succeeded = metadata
                .and_then(|map| map.get("success"))
                .is_some_and(is_truthy);
            if !succeeded {
                continue;
            }
            let source = normalize_capability_id(
                edge_object
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            let target = normalize_capability_id(
                edge_object
                    .get("target")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            if source.is_empty() || target.is_empty() {
                continue;
            }
            if !nodes.contains_key(&source) || !nodes.contains_key(&target) {
                continue;
            }
            let relation = edge_object
                .get("relation")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            let relation = if relation.is_empty() {
                CAN_FEED.to_string()
            } else {
                relation
            };
            let reason = metadata
                .and_then(|map| map.get("reason"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            if !seen.insert((source.clone(), target.clone(), relation.clone())) {
                continue;
            }
            edges.push(TraceEdge {
                source,
                target,
                relation,
                success: true,
                reason,
            });
        }
    }
    edges.sort_by(|left, right| {
        (&left.source, &left.target, &left.relation).cmp(&(
            &right.source,
            &right.target,
            &right.relation,
        ))
    });
    edges
}
/// Python-style truthiness for an edge `success` marker.
///
/// `false`, `0`, `null`, empty strings/arrays/objects are false; everything else
/// is true. The strings `"false"` and `"0"` are also treated as false, which is
/// stricter than Python but never turns an explicit failure into a success.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => {
            let trimmed = text.trim();
            !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("false") && trimmed != "0"
        }
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}
