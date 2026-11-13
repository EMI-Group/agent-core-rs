//! The workflow definition the builder records.
//!
//! [`WorkflowDefinition`] is the builder's mutable record: components in declaration
//! order, the edges between them, conditional fan-out, the chosen start/end ids, and any
//! problems the builder already detected (reported later by
//! [`validate`](super::validate::validate)).
//!
//! Each component is stored with its [`NodeConfig`], which carries the input/output
//! templates, the declared abilities, and the per-node limits. A node runs exactly one
//! ability; [`WorkflowDefinition::selected_ability`] picks it from the declared
//! abilities when they are set, otherwise from the edge topology (the same inference
//! Python's `auto_complete_abilities` performs before compiling).
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_workflow::engine::{NodeConfig, WorkflowDefinition};
//! use openjiuwen_workflow::{StartComponent, WorkflowCard};
//!
//! let mut definition = WorkflowDefinition::new(WorkflowCard::new("wf"));
//! definition.insert_component("start", Arc::new(StartComponent::new()), NodeConfig::default());
//! assert_eq!(definition.component_ids(), vec!["start"]);
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_graph::Router;

use crate::ability::ComponentAbility;
use crate::card::{
    CompIoConfig, ExceptionConfig, NodeSpec, WorkflowCard, WorkflowConfig, WorkflowSpec,
};
use crate::component::Component;

/// The per-component configuration the builder records.
///
/// The two schema pairs are resolved against workflow state before the component runs;
/// `stream_*` variants are used when the node consumes or produces streams. `timeout_ms`
/// is stored in milliseconds and converted to Python's seconds in
/// [`WorkflowDefinition::spec`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct NodeConfig {
    /// The JSON template resolved against state for batch inputs.
    pub inputs_schema: Option<Value>,
    /// The JSON template describing batch outputs.
    pub outputs_schema: Option<Value>,
    /// The JSON template resolved against state for streamed inputs.
    pub stream_inputs_schema: Option<Value>,
    /// The JSON template describing streamed outputs.
    pub stream_outputs_schema: Option<Value>,
    /// Declared abilities; `None` (or empty) means "infer from topology".
    pub abilities: Option<Vec<ComponentAbility>>,
    /// Whether a fan-in waits for every predecessor (Python's `wait_for_all`).
    ///
    /// `Some(true)` or `None` keeps the all-predecessors CNF barrier every edge compiles
    /// to. `Some(false)` on a node with at least two distinct regular/stream/barrier
    /// predecessors compiles Python's first-arrival fan-in instead: the node fires per
    /// arrival and a key whose producer has not run yet resolves to `null`. A node with a
    /// single inbound source is unaffected (one source is one barrier). `Collect` and
    /// `Transform` nodes must keep it `true`; `false` is rejected with `[100011]`.
    pub wait_for_all: Option<bool>,
    /// How many times a failing invocation is retried.
    pub max_retries: u32,
    /// The per-node timeout in milliseconds; `None` means no timeout.
    pub timeout_ms: Option<u64>,
    /// What to do when the node fails after every retry; see [`crate::card::ExceptionConfig`].
    pub exception_config: Option<ExceptionConfig>,
    /// An optional display name.
    pub name: Option<String>,
}

/// The kind of edge that connects two components.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Batch data flows from source to target.
    #[default]
    Regular,
    /// The source streams frames to the target.
    Stream,
    /// Every source must finish before the target runs (barrier fan-in).
    Barrier,
}

impl EdgeKind {
    /// The lowercase wire name (`"regular"`, `"stream"`, `"barrier"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Stream => "stream",
            Self::Barrier => "barrier",
        }
    }
}

/// One edge: possibly several sources feeding one target.
///
/// A single-source edge is a regular edge; several sources express a barrier fan-in
/// (Python's `add_connections`).
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// The source component ids.
    pub sources: Vec<String>,
    /// The target component id.
    pub target: String,
    /// How the edge carries data.
    pub kind: EdgeKind,
}

/// The conditional fan-out recorded for one source component.
///
/// The routers run in registration order and their targets are unioned; the targets a
/// router may return are the ids in [`ConditionalSource::targets`].
#[derive(Clone)]
pub struct ConditionalSource {
    /// Every target any router may choose (deduplicated, insertion order).
    pub targets: Vec<String>,
    /// The routers deciding the targets, in registration order.
    pub routers: Vec<Arc<dyn Router>>,
}

impl fmt::Debug for ConditionalSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let routers: Vec<&str> = self.routers.iter().map(|router| router.name()).collect();
        f.debug_struct("ConditionalSource")
            .field("targets", &self.targets)
            .field("routers", &routers)
            .finish()
    }
}

/// One registered component and its node configuration.
#[derive(Clone)]
pub struct Entry {
    /// The component runnable.
    pub component: Arc<dyn Component>,
    /// The node's configuration.
    pub node: NodeConfig,
}

impl fmt::Debug for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry")
            .field("component_type", &self.component.component_type())
            .field("node", &self.node)
            .finish()
    }
}

/// Whether a node receives or emits batch or stream data.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Flow {
    regular: bool,
    stream: bool,
}

impl Flow {
    fn mark(&mut self, kind: EdgeKind) {
        if kind == EdgeKind::Stream {
            self.stream = true;
        } else {
            self.regular = true;
        }
    }

    const fn any(self) -> bool {
        self.regular || self.stream
    }
}

/// Everything the builder recorded about a workflow before it is validated and compiled.
///
/// Cloning is cheap: components are `Arc` handles and every other field is a plain
/// collection. The `Debug` implementation prints components by `component_type` because
/// runnables and routers are not `Debug`.
#[derive(Clone)]
pub struct WorkflowDefinition {
    /// The workflow's card.
    pub card: WorkflowCard,
    /// The workflow configuration (card + spec + nesting limit).
    pub config: WorkflowConfig,
    /// Components keyed by id.
    pub components: BTreeMap<String, Entry>,
    /// Component ids in declaration order.
    pub order: Vec<String>,
    /// Regular, stream, and barrier edges in insertion order.
    pub edges: Vec<Edge>,
    /// Conditional fan-out keyed by source component id.
    pub conditional: BTreeMap<String, ConditionalSource>,
    /// The start component id.
    pub start: Option<String>,
    /// The end component id.
    pub end: Option<String>,
    /// Problems the builder recorded; `validate` reports them verbatim.
    pub problems: Vec<String>,
}

impl fmt::Debug for WorkflowDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkflowDefinition")
            .field("card", &self.card)
            .field("config", &self.config)
            .field("components", &self.components)
            .field("order", &self.order)
            .field("edges", &self.edges)
            .field("conditional", &self.conditional)
            .field("start", &self.start)
            .field("end", &self.end)
            .field("problems", &self.problems)
            .finish()
    }
}

impl WorkflowDefinition {
    /// Creates an empty definition for `card` with Python's default configuration.
    #[must_use]
    pub fn new(card: WorkflowCard) -> Self {
        Self::with_config(WorkflowConfig {
            card,
            ..WorkflowConfig::default()
        })
    }

    /// Creates an empty definition from a full configuration, keeping its spec and
    /// nesting limit; the card is taken from `config.card`.
    #[must_use]
    pub fn with_config(config: WorkflowConfig) -> Self {
        Self {
            card: config.card.clone(),
            config,
            components: BTreeMap::new(),
            order: Vec::new(),
            edges: Vec::new(),
            conditional: BTreeMap::new(),
            start: None,
            end: None,
            problems: Vec::new(),
        }
    }

    /// Registers (or replaces) a component and records its id in declaration order.
    ///
    /// Re-inserting an existing id replaces the entry but keeps the original position in
    /// [`WorkflowDefinition::component_ids`].
    pub fn insert_component(
        &mut self,
        id: impl Into<String>,
        component: Arc<dyn Component>,
        node: NodeConfig,
    ) -> &mut Self {
        let id = id.into();
        if !self.order.iter().any(|existing| existing == &id) {
            self.order.push(id.clone());
        }
        self.components.insert(id, Entry { component, node });
        self
    }

    /// The registered entry for `id`, if any.
    #[must_use]
    pub fn component(&self, id: &str) -> Option<&Entry> {
        self.components.get(id)
    }

    /// The registered component ids in declaration order.
    #[must_use]
    pub fn component_ids(&self) -> Vec<&str> {
        self.order.iter().map(String::as_str).collect()
    }

    /// Records an edge from every one of `sources` to `target`.
    pub fn add_edge(&mut self, sources: Vec<String>, target: String, kind: EdgeKind) -> &mut Self {
        self.edges.push(Edge {
            sources,
            target,
            kind,
        });
        self
    }

    /// Adds a conditional edge from `from` to `to`, backed by `router`.
    ///
    /// `to` is recorded once even when several routers may choose it; routers accumulate
    /// in registration order.
    pub fn add_conditional(&mut self, from: &str, to: &str, router: Arc<dyn Router>) -> &mut Self {
        let conditional =
            self.conditional
                .entry(from.to_string())
                .or_insert_with(|| ConditionalSource {
                    targets: Vec::new(),
                    routers: Vec::new(),
                });
        if !conditional.targets.iter().any(|target| target == to) {
            conditional.targets.push(to.to_string());
        }
        conditional.routers.push(router);
        self
    }

    /// Records a problem for `validate` to report verbatim.
    pub fn record_problem(&mut self, problem: impl Into<String>) -> &mut Self {
        self.problems.push(problem.into());
        self
    }

    /// The problems the builder recorded, in insertion order.
    #[must_use]
    pub fn problems(&self) -> &[String] {
        &self.problems
    }

    /// The abilities a component supports.
    ///
    /// Declared abilities win when the node config carries a non-empty list; otherwise
    /// the component's own [`Component::abilities`] are used, falling back to
    /// `[Invoke]` (the same default the trait uses) when a component declares none. An
    /// unregistered id supports only `Invoke`.
    #[must_use]
    pub fn supported_abilities(&self, id: &str) -> Vec<ComponentAbility> {
        let Some(entry) = self.components.get(id) else {
            return vec![ComponentAbility::Invoke];
        };
        if let Some(declared) = entry
            .node
            .abilities
            .as_ref()
            .filter(|abilities| !abilities.is_empty())
        {
            return declared.clone();
        }
        let abilities = entry.component.abilities();
        if abilities.is_empty() {
            vec![ComponentAbility::Invoke]
        } else {
            abilities
        }
    }

    /// The single ability a node is driven with.
    ///
    /// Declared abilities win when they are set: the first of `Transform`, `Collect`,
    /// `Stream`, `Invoke` present in the declared list. Otherwise the ability is inferred
    /// from the edge topology:
    ///
    /// - a node without incoming edges (the start component) → `Invoke`
    /// - stream out + regular in → `Stream`
    /// - stream out + stream in → `Transform`
    /// - stream in + regular out → `Collect`
    /// - regular in + regular out → `Invoke`
    ///
    /// Anything else falls back to the first supported ability, then `Invoke`.
    #[must_use]
    pub fn selected_ability(&self, id: &str) -> ComponentAbility {
        let Some(entry) = self.components.get(id) else {
            return ComponentAbility::Invoke;
        };
        if let Some(declared) = entry
            .node
            .abilities
            .as_ref()
            .filter(|abilities| !abilities.is_empty())
        {
            return [
                ComponentAbility::Transform,
                ComponentAbility::Collect,
                ComponentAbility::Stream,
                ComponentAbility::Invoke,
            ]
            .into_iter()
            .find(|candidate| declared.contains(candidate))
            .unwrap_or(ComponentAbility::Invoke);
        }

        let incoming = self.incoming_flow(id);
        let outgoing = self.outgoing_flow(id);
        let inferred = if !incoming.any() {
            Some(ComponentAbility::Invoke)
        } else if outgoing.stream && incoming.regular {
            Some(ComponentAbility::Stream)
        } else if outgoing.stream && incoming.stream {
            Some(ComponentAbility::Transform)
        } else if incoming.stream && outgoing.regular {
            Some(ComponentAbility::Collect)
        } else if incoming.regular && outgoing.regular {
            Some(ComponentAbility::Invoke)
        } else {
            None
        };
        inferred.unwrap_or_else(|| {
            self.supported_abilities(id)
                .first()
                .copied()
                .unwrap_or(ComponentAbility::Invoke)
        })
    }

    /// The Python-compatible [`WorkflowSpec`] for this definition.
    ///
    /// Regular and barrier edges contribute to `edges` (one target entry per source, in
    /// insertion order), stream edges to both `stream_edges` and `stream_source_groups`,
    /// and every registered component to `comp_configs` with its ability list, retry
    /// count, and timeout in seconds (`-1.0` when unset).
    #[must_use]
    pub fn spec(&self) -> WorkflowSpec {
        let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut stream_edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut stream_source_groups: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
        for edge in &self.edges {
            for source in &edge.sources {
                if edge.kind == EdgeKind::Stream {
                    stream_edges
                        .entry(source.clone())
                        .or_default()
                        .push(edge.target.clone());
                    stream_source_groups
                        .entry(edge.target.clone())
                        .or_default()
                        .push(vec![source.clone()]);
                } else {
                    edges
                        .entry(source.clone())
                        .or_default()
                        .push(edge.target.clone());
                }
            }
        }

        let comp_configs: BTreeMap<String, NodeSpec> = self
            .components
            .iter()
            .map(|(id, entry)| (id.clone(), self.node_spec(id, &entry.node)))
            .collect();

        WorkflowSpec {
            edges,
            stream_edges,
            comp_configs,
            stream_source_groups,
            start_nodes: self.start.iter().cloned().collect(),
        }
    }

    /// Builds the [`NodeSpec`] for one node.
    fn node_spec(&self, id: &str, node: &NodeConfig) -> NodeSpec {
        let io_configs = if node.inputs_schema.is_some() || node.outputs_schema.is_some() {
            Some(CompIoConfig {
                inputs_schema: node.inputs_schema.clone(),
                outputs_schema: node.outputs_schema.clone(),
            })
        } else {
            None
        };
        let stream_io_configs =
            if node.stream_inputs_schema.is_some() || node.stream_outputs_schema.is_some() {
                Some(CompIoConfig {
                    inputs_schema: node.stream_inputs_schema.clone(),
                    outputs_schema: node.stream_outputs_schema.clone(),
                })
            } else {
                None
            };
        NodeSpec {
            io_configs,
            stream_io_configs,
            abilities: self.supported_abilities(id),
            max_retries: node.max_retries,
            // Python stores seconds; `Duration` keeps the conversion exact.
            timeout: node
                .timeout_ms
                .map_or(-1.0, |ms| Duration::from_millis(ms).as_secs_f64()),
            exception_config: node.exception_config.clone(),
            name: node.name.clone(),
        }
    }

    /// The batch and stream edges entering `id`.
    fn incoming_flow(&self, id: &str) -> Flow {
        let mut flow = Flow::default();
        for edge in &self.edges {
            if edge.target == id {
                flow.mark(edge.kind);
            }
        }
        if self
            .conditional
            .values()
            .any(|conditional| conditional.targets.iter().any(|target| target == id))
        {
            flow.regular = true;
        }
        flow
    }

    /// The batch and stream edges leaving `id`.
    fn outgoing_flow(&self, id: &str) -> Flow {
        let mut flow = Flow::default();
        for edge in &self.edges {
            if edge.sources.iter().any(|source| source == id) {
                flow.mark(edge.kind);
            }
        }
        if self
            .conditional
            .get(id)
            .is_some_and(|conditional| !conditional.targets.is_empty())
        {
            flow.regular = true;
        }
        flow
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::unnecessary_literal_bound
)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;

    use crate::components::{EndComponent, StartComponent};
    use crate::engine::test_support::{card, fake, router};
    fn chain(
        mid_abilities: Vec<ComponentAbility>,
        into_mid: EdgeKind,
        out_of_mid: EdgeKind,
    ) -> WorkflowDefinition {
        let mut definition = WorkflowDefinition::new(card());
        definition.insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        );
        definition.insert_component("mid", fake("mid", mid_abilities), NodeConfig::default());
        definition.insert_component(
            "end",
            Arc::new(EndComponent::default()),
            NodeConfig::default(),
        );
        definition
            .add_edge(vec!["start".to_string()], "mid".to_string(), into_mid)
            .add_edge(vec!["mid".to_string()], "end".to_string(), out_of_mid);
        definition
    }

    #[test]
    fn edge_kind_wire_names_are_snake_case() {
        for (kind, wire) in [
            (EdgeKind::Regular, "\"regular\""),
            (EdgeKind::Stream, "\"stream\""),
            (EdgeKind::Barrier, "\"barrier\""),
        ] {
            assert_eq!(serde_json::to_string(&kind).unwrap(), wire);
            assert_eq!(kind.as_str(), wire.trim_matches('"'));
        }
        assert_eq!(EdgeKind::default(), EdgeKind::Regular);
    }

    #[test]
    fn node_config_defaults_are_inert() {
        let node = NodeConfig::default();
        assert_eq!(node.abilities, None);
        assert_eq!(node.wait_for_all, None);
        assert_eq!(node.max_retries, 0);
        assert_eq!(node.timeout_ms, None);
        assert_eq!(node.name, None);
    }

    #[test]
    fn topology_inference_selects_the_expected_ability() {
        use ComponentAbility::{Collect, Invoke, Stream, Transform};

        for (abilities, incoming, outgoing, expected) in [
            (
                vec![Invoke, Stream],
                EdgeKind::Regular,
                EdgeKind::Stream,
                Stream,
            ),
            (
                vec![Invoke, Stream, Transform],
                EdgeKind::Stream,
                EdgeKind::Stream,
                Transform,
            ),
            (
                vec![Invoke, Collect],
                EdgeKind::Stream,
                EdgeKind::Regular,
                Collect,
            ),
            (
                vec![Invoke, Stream],
                EdgeKind::Regular,
                EdgeKind::Regular,
                Invoke,
            ),
            (
                vec![Invoke, Stream],
                EdgeKind::Barrier,
                EdgeKind::Stream,
                Stream,
            ),
        ] {
            let definition = chain(abilities, incoming, outgoing);
            assert_eq!(
                definition.selected_ability("mid"),
                expected,
                "incoming={incoming:?}, outgoing={outgoing:?}"
            );
        }
    }

    #[test]
    fn start_and_isolated_nodes_select_invoke() {
        let definition = chain(
            vec![ComponentAbility::Invoke],
            EdgeKind::Regular,
            EdgeKind::Regular,
        );
        assert_eq!(
            definition.selected_ability("start"),
            ComponentAbility::Invoke
        );

        let mut isolated = WorkflowDefinition::new(card());
        isolated.insert_component(
            "solo",
            fake("solo", vec![ComponentAbility::Stream]),
            NodeConfig::default(),
        );
        assert_eq!(isolated.selected_ability("solo"), ComponentAbility::Invoke);
    }

    #[test]
    fn sink_node_falls_back_to_supported_ability() {
        let mut definition = WorkflowDefinition::new(card());
        definition.insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        );
        definition.insert_component(
            "end",
            fake("end", vec![ComponentAbility::Collect]),
            NodeConfig::default(),
        );
        definition.add_edge(
            vec!["start".to_string()],
            "end".to_string(),
            EdgeKind::Regular,
        );
        assert_eq!(
            definition.selected_ability("end"),
            ComponentAbility::Collect
        );
    }

    #[test]
    fn conditional_edges_count_as_regular_inputs() {
        let mut definition = chain(
            vec![ComponentAbility::Invoke, ComponentAbility::Stream],
            EdgeKind::Regular,
            EdgeKind::Stream,
        );
        // Drop the regular start → mid edge so only the conditional one feeds `mid`.
        definition.edges.retain(|edge| edge.target != "mid");
        definition.add_conditional("start", "mid", router());
        assert_eq!(definition.selected_ability("mid"), ComponentAbility::Stream);
    }

    #[test]
    fn declared_abilities_win_in_priority_order() {
        let mut definition = chain(
            vec![ComponentAbility::Invoke],
            EdgeKind::Regular,
            EdgeKind::Regular,
        );
        definition.components.get_mut("mid").unwrap().node.abilities = Some(vec![
            ComponentAbility::Invoke,
            ComponentAbility::Stream,
            ComponentAbility::Collect,
            ComponentAbility::Transform,
        ]);
        assert_eq!(
            definition.selected_ability("mid"),
            ComponentAbility::Transform
        );

        definition.components.get_mut("mid").unwrap().node.abilities =
            Some(vec![ComponentAbility::Invoke, ComponentAbility::Stream]);
        assert_eq!(definition.selected_ability("mid"), ComponentAbility::Stream);

        definition.components.get_mut("mid").unwrap().node.abilities =
            Some(vec![ComponentAbility::Invoke]);
        assert_eq!(definition.selected_ability("mid"), ComponentAbility::Invoke);
    }

    #[test]
    fn empty_declared_abilities_fall_back_to_topology() {
        let mut definition = chain(
            vec![ComponentAbility::Invoke, ComponentAbility::Stream],
            EdgeKind::Regular,
            EdgeKind::Stream,
        );
        definition.components.get_mut("mid").unwrap().node.abilities = Some(Vec::new());
        assert_eq!(definition.selected_ability("mid"), ComponentAbility::Stream);
    }

    #[test]
    fn supported_abilities_prefer_declared_then_component_then_invoke() {
        let mut definition = chain(
            vec![ComponentAbility::Stream],
            EdgeKind::Regular,
            EdgeKind::Regular,
        );
        assert_eq!(
            definition.supported_abilities("mid"),
            vec![ComponentAbility::Stream]
        );

        definition.components.get_mut("mid").unwrap().node.abilities =
            Some(vec![ComponentAbility::Invoke, ComponentAbility::Collect]);
        assert_eq!(
            definition.supported_abilities("mid"),
            vec![ComponentAbility::Invoke, ComponentAbility::Collect]
        );

        definition.insert_component("bare", fake("bare", Vec::new()), NodeConfig::default());
        assert_eq!(
            definition.supported_abilities("bare"),
            vec![ComponentAbility::Invoke]
        );
        assert_eq!(
            definition.supported_abilities("missing"),
            vec![ComponentAbility::Invoke]
        );
    }

    #[test]
    fn accessors_follow_declaration_order() {
        let mut definition = WorkflowDefinition::new(card());
        for id in ["a", "b", "c"] {
            definition.insert_component(
                id,
                fake("fake", vec![ComponentAbility::Invoke]),
                NodeConfig::default(),
            );
        }
        definition.insert_component(
            "b",
            fake("replaced", vec![ComponentAbility::Stream]),
            NodeConfig::default(),
        );

        assert_eq!(definition.component_ids(), vec!["a", "b", "c"]);
        assert_eq!(definition.components.len(), 3);
        assert_eq!(
            definition
                .component("b")
                .unwrap()
                .component
                .component_type(),
            "replaced"
        );
        assert!(definition.component("missing").is_none());
    }

    #[test]
    fn conditional_targets_are_deduplicated_but_routers_accumulate() {
        let mut definition = WorkflowDefinition::new(card());
        definition.add_conditional("start", "a", router());
        definition.add_conditional("start", "a", router());
        definition.add_conditional("start", "b", router());

        let conditional = definition.conditional.get("start").unwrap();
        assert_eq!(conditional.targets, vec!["a", "b"]);
        assert_eq!(conditional.routers.len(), 3);
        assert_eq!(
            format!("{conditional:?}"),
            "ConditionalSource { targets: [\"a\", \"b\"], routers: [\"branch\", \"branch\", \"branch\"] }"
        );
    }

    #[test]
    fn new_and_with_config_keep_the_expected_card() {
        let definition = WorkflowDefinition::new(card());
        assert_eq!(definition.card, card());
        assert_eq!(definition.config.card, card());
        assert_eq!(definition.config.workflow_max_nesting_depth, 5);
        assert_eq!(definition.config.spec, WorkflowSpec::default());
        assert_eq!(definition.start, None);
        assert_eq!(definition.end, None);
        assert!(definition.problems().is_empty());

        let config = WorkflowConfig {
            card: WorkflowCard::new("configured"),
            workflow_max_nesting_depth: 3,
            spec: WorkflowSpec {
                start_nodes: vec!["start".to_string()],
                ..WorkflowSpec::default()
            },
        };
        let definition = WorkflowDefinition::with_config(config.clone());
        assert_eq!(definition.card, WorkflowCard::new("configured"));
        assert_eq!(definition.config, config);
        assert_eq!(definition.config.spec.start_nodes, vec!["start"]);
    }

    /// start → {llm, end}, plus a barrier fan-in from `tool` into `end`.
    fn spec_definition() -> WorkflowDefinition {
        let mut definition = WorkflowDefinition::new(card());
        definition.insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig {
                inputs_schema: Some(json!({"query": ""})),
                timeout_ms: Some(1500),
                max_retries: 2,
                name: Some("Start".to_string()),
                ..NodeConfig::default()
            },
        );
        definition.insert_component(
            "llm",
            fake(
                "llm",
                vec![ComponentAbility::Invoke, ComponentAbility::Stream],
            ),
            NodeConfig {
                outputs_schema: Some(json!({"result": ""})),
                stream_inputs_schema: Some(json!({"query": ""})),
                ..NodeConfig::default()
            },
        );
        definition.insert_component(
            "tool",
            fake("tool", vec![ComponentAbility::Invoke]),
            NodeConfig::default(),
        );
        definition.insert_component(
            "end",
            Arc::new(EndComponent::default()),
            NodeConfig::default(),
        );
        definition.start = Some("start".to_string());
        definition
            .add_edge(
                vec!["start".to_string()],
                "llm".to_string(),
                EdgeKind::Regular,
            )
            .add_edge(
                vec!["start".to_string()],
                "end".to_string(),
                EdgeKind::Regular,
            )
            .add_edge(vec!["llm".to_string()], "end".to_string(), EdgeKind::Stream)
            .add_edge(
                vec!["llm".to_string(), "tool".to_string()],
                "end".to_string(),
                EdgeKind::Barrier,
            );
        definition
    }

    #[test]
    fn spec_groups_edges_and_stream_source_groups() {
        let spec = spec_definition().spec();
        assert_eq!(spec.start_nodes, vec!["start"]);
        assert_eq!(
            spec.edges.get("start").unwrap(),
            &vec!["llm".to_string(), "end".to_string()]
        );
        assert_eq!(
            spec.edges.get("llm").unwrap(),
            &vec!["end".to_string()],
            "a barrier edge contributes every source to `edges`"
        );
        assert_eq!(spec.edges.get("tool").unwrap(), &vec!["end".to_string()]);
        assert_eq!(
            spec.stream_edges.get("llm").unwrap(),
            &vec!["end".to_string()]
        );
        assert_eq!(
            spec.stream_source_groups.get("end").unwrap(),
            &vec![vec!["llm".to_string()]]
        );
        assert_eq!(
            spec.stream_source_groups.len(),
            1,
            "only stream edges populate stream_source_groups"
        );
    }

    #[test]
    fn spec_converts_node_configs() {
        let spec = spec_definition().spec();

        let start = spec.comp_configs.get("start").unwrap();
        assert_eq!(
            start.io_configs,
            Some(CompIoConfig {
                inputs_schema: Some(json!({"query": ""})),
                outputs_schema: None,
            })
        );
        assert_eq!(start.stream_io_configs, None);
        assert_eq!(start.abilities, vec![ComponentAbility::Invoke]);
        assert!((start.timeout - 1.5).abs() < f64::EPSILON);
        assert_eq!(start.max_retries, 2);
        assert_eq!(start.name, Some("Start".to_string()));
        assert_eq!(start.exception_config, None);

        let llm = spec.comp_configs.get("llm").unwrap();
        assert_eq!(llm.io_configs.as_ref().unwrap().inputs_schema, None);
        assert_eq!(
            llm.stream_io_configs,
            Some(CompIoConfig {
                inputs_schema: Some(json!({"query": ""})),
                outputs_schema: None,
            })
        );
        assert!(
            (llm.timeout + 1.0).abs() < f64::EPSILON,
            "unset timeout is -1.0"
        );

        let end = spec.comp_configs.get("end").unwrap();
        assert_eq!(end.io_configs, None);
        assert_eq!(end.stream_io_configs, None);
        assert_eq!(end.abilities.len(), 4);
    }

    #[test]
    fn spec_emits_the_node_exception_config() {
        let mut definition = spec_definition();
        let config = ExceptionConfig {
            handle_type: "defaultOutputs".to_string(),
            default_outputs: Some(json!({"result": "recovered"})),
            metadata: BTreeMap::new(),
        };
        definition
            .components
            .get_mut("llm")
            .unwrap()
            .node
            .exception_config = Some(config.clone());

        let spec = definition.spec();
        assert_eq!(
            spec.comp_configs.get("llm").unwrap().exception_config,
            Some(config)
        );
        assert_eq!(
            spec.comp_configs.get("start").unwrap().exception_config,
            None
        );
    }

    #[test]
    fn spec_without_start_has_no_start_nodes() {
        let definition = WorkflowDefinition::new(card());
        assert!(definition.spec().start_nodes.is_empty());
        assert!(definition.spec().comp_configs.is_empty());
    }

    #[test]
    fn problem_recording_preserves_order() {
        let mut definition = WorkflowDefinition::new(card());
        definition
            .record_problem("first")
            .record_problem(String::from("second"));
        assert_eq!(definition.problems(), ["first", "second"]);
    }

    #[test]
    fn debug_prints_component_types_instead_of_runnables() {
        let definition = chain(
            vec![ComponentAbility::Invoke],
            EdgeKind::Regular,
            EdgeKind::Regular,
        );
        let rendered = format!("{definition:?}");
        assert!(rendered.starts_with("WorkflowDefinition"));
        assert!(rendered.contains("component_type: \"start\""));
        assert!(rendered.contains("component_type: \"mid\""));
        assert!(rendered.contains("Edge { sources: [\"start\"], target: \"mid\", kind: Regular }"));
    }

    #[test]
    fn node_config_round_trips_through_json() {
        let node = NodeConfig {
            inputs_schema: Some(json!({"query": "${start.query}"})),
            outputs_schema: Some(json!({"result": "${llm.result}"})),
            stream_inputs_schema: Some(json!({"query": "${start.query}"})),
            stream_outputs_schema: Some(json!({"delta": "${llm.delta}"})),
            abilities: Some(vec![ComponentAbility::Stream, ComponentAbility::Collect]),
            wait_for_all: Some(true),
            max_retries: 3,
            timeout_ms: Some(2500),
            exception_config: Some(ExceptionConfig {
                handle_type: "defaultOutputs".to_string(),
                default_outputs: Some(json!({"result": "recovered"})),
                metadata: BTreeMap::new(),
            }),
            name: Some("Node".to_string()),
        };
        let wire = serde_json::to_value(&node).unwrap();
        assert_eq!(wire["timeout_ms"], json!(2500));
        assert_eq!(wire["abilities"], json!(["stream", "collect"]));
        assert_eq!(wire["wait_for_all"], json!(true));
        assert_eq!(wire["exception_config"]["handle_type"], "defaultOutputs");
        let parsed: NodeConfig = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, node);

        let tolerant: NodeConfig =
            serde_json::from_str(r#"{"timeout_ms": 10, "unknown": {"a": 1}}"#).unwrap();
        assert_eq!(tolerant.timeout_ms, Some(10));
    }
}
