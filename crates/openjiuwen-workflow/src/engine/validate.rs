//! Compile-time validation of a recorded workflow definition.
//!
//! Ports the checks Python's `Workflow` performs while building and before compiling:
//! component ids, the start/end shape, edge endpoints, streaming capability, the
//! `wait_for_all` rule for collect/transform nodes, declared abilities, and overlapping
//! input/output schema keys.
//!
//! [`validate`] never stops at the first problem: every rule contributes a coded message
//! and all messages are joined with `"; "` into a single [`Error::Workflow`], so a caller
//! sees the whole list at once (the policy [`openjiuwen-graph`] uses for compile errors).
//!
//! [`openjiuwen-graph`]: openjiuwen_graph

use std::collections::BTreeSet;

use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::ability::ComponentAbility;
use crate::card::validate_component_id;
use crate::error::{
    compile_error, component_ability_invalid, component_schema_invalid, condition_edge_invalid,
    stream_edge_invalid,
};

use super::definition::{EdgeKind, WorkflowDefinition};

/// Validates a workflow definition and reports every problem it finds.
///
/// The checks are:
///
/// - problems the builder already recorded are included verbatim;
/// - exactly one start and one end component, both registered and distinct, and the end
///   has no outgoing edge (regular, stream, barrier, or conditional);
/// - the start component supports `Invoke`;
/// - every component id passes [`validate_component_id`];
/// - every edge source/target and conditional source/target is registered;
/// - every conditional source has at least one router;
/// - a stream edge's source supports `Stream` or `Transform` and its target supports
///   `Collect` or `Transform`;
/// - `wait_for_all: Some(false)` is rejected on a node selected as `Collect`/`Transform`;
/// - a declared ability the component does not implement is rejected;
/// - `inputs_schema`/`stream_inputs_schema` and `outputs_schema`/`stream_outputs_schema`
///   may not share a flattened key.
///
/// # Errors
///
/// Returns one [`Error::Workflow`] whose message is the `"; "`-joined list of coded
/// messages (each carrying Python's `"[<code>] "` prefix) when any rule fails.
pub fn validate(def: &WorkflowDefinition) -> Result<()> {
    let workflow = def.card.id.as_str();
    let mut problems = def.problems.clone();

    check_endpoints(def, workflow, &mut problems);
    check_component_ids(def, &mut problems);
    check_edges(def, workflow, &mut problems);
    check_conditionals(def, workflow, &mut problems);
    check_nodes(def, workflow, &mut problems);

    if problems.is_empty() {
        Ok(())
    } else {
        Err(Error::workflow(problems.join("; ")))
    }
}

/// Appends an error's coded message to the problem list.
#[allow(clippy::needless_pass_by_value)] // call sites build the coded error inline
fn record(problems: &mut Vec<String>, error: Error) {
    problems.push(error.message().into_owned());
}

/// Checks the start/end shape, the end's outgoing edges, and the start's ability.
fn check_endpoints(def: &WorkflowDefinition, workflow: &str, problems: &mut Vec<String>) {
    let start = def.start.as_deref();
    let end = def.end.as_deref();

    match start {
        None => record(
            problems,
            compile_error("the workflow has no start component", workflow),
        ),
        Some(id) if def.component(id).is_none() => record(
            problems,
            compile_error(
                &format!("the start component '{id}' is not registered"),
                workflow,
            ),
        ),
        Some(_) => {}
    }
    match end {
        None => record(
            problems,
            compile_error("the workflow has no end component", workflow),
        ),
        Some(id) if def.component(id).is_none() => record(
            problems,
            compile_error(
                &format!("the end component '{id}' is not registered"),
                workflow,
            ),
        ),
        Some(_) => {}
    }

    if let (Some(start), Some(end)) = (start, end) {
        if start == end {
            record(
                problems,
                compile_error(
                    &format!("the start component '{start}' is also the end component"),
                    workflow,
                ),
            );
        }
    }

    if let Some(end) = end.filter(|id| def.component(id).is_some()) {
        for edge in &def.edges {
            if edge.sources.iter().any(|source| source == end) {
                record(
                    problems,
                    compile_error(
                        &format!(
                            "the end component '{end}' has an outgoing {} edge to '{}'",
                            edge.kind.as_str(),
                            edge.target
                        ),
                        workflow,
                    ),
                );
            }
        }
        if let Some(conditional) = def.conditional.get(end) {
            if !conditional.targets.is_empty() {
                record(
                    problems,
                    compile_error(
                        &format!("the end component '{end}' has an outgoing conditional edge"),
                        workflow,
                    ),
                );
            }
        }
    }

    if let Some(start) = start.filter(|id| def.component(id).is_some()) {
        if !def
            .supported_abilities(start)
            .contains(&ComponentAbility::Invoke)
        {
            record(
                problems,
                component_ability_invalid(
                    start,
                    ComponentAbility::Invoke.as_str(),
                    "the start component must support 'invoke'",
                    workflow,
                ),
            );
        }
    }
}

/// Checks every registered component id.
fn check_component_ids(def: &WorkflowDefinition, problems: &mut Vec<String>) {
    for id in def.component_ids() {
        if let Err(error) = validate_component_id(id) {
            record(problems, error);
        }
    }
}

/// Checks that every edge endpoint exists and that stream edges are supported.
fn check_edges(def: &WorkflowDefinition, workflow: &str, problems: &mut Vec<String>) {
    for edge in &def.edges {
        for source in &edge.sources {
            if def.component(source).is_none() {
                record(
                    problems,
                    compile_error(
                        &format!("edge source '{source}' is not registered"),
                        workflow,
                    ),
                );
            }
        }
        if def.component(&edge.target).is_none() {
            record(
                problems,
                compile_error(
                    &format!("edge target '{}' is not registered", edge.target),
                    workflow,
                ),
            );
        }
        if edge.kind != EdgeKind::Stream {
            continue;
        }

        for source in &edge.sources {
            if def.component(source).is_none() {
                continue;
            }
            if !supports_stream_output(def, source) {
                record(
                    problems,
                    stream_edge_invalid(
                        "the source component does not support 'stream' or 'transform'",
                        source,
                        &edge.target,
                        workflow,
                    ),
                );
            }
        }
        if def.component(&edge.target).is_some() && !supports_stream_input(def, &edge.target) {
            record(
                problems,
                component_ability_invalid(
                    &edge.target,
                    ComponentAbility::Collect.as_str(),
                    "a stream target must support 'collect' or 'transform'",
                    workflow,
                ),
            );
        }
    }
}

/// Checks conditional sources: registered endpoints and at least one router.
fn check_conditionals(def: &WorkflowDefinition, workflow: &str, problems: &mut Vec<String>) {
    for (source, conditional) in &def.conditional {
        if def.component(source).is_none() {
            record(
                problems,
                compile_error(
                    &format!("conditional edge source '{source}' is not registered"),
                    workflow,
                ),
            );
        }
        for target in &conditional.targets {
            if def.component(target).is_none() {
                record(
                    problems,
                    compile_error(
                        &format!("conditional edge target '{target}' is not registered"),
                        workflow,
                    ),
                );
            }
        }
        if conditional.routers.is_empty() {
            record(
                problems,
                condition_edge_invalid("no router", source, workflow),
            );
        }
    }
}

/// Checks declared abilities, `wait_for_all`, and schema key overlap per component.
fn check_nodes(def: &WorkflowDefinition, workflow: &str, problems: &mut Vec<String>) {
    for id in def.component_ids() {
        let Some(entry) = def.component(id) else {
            continue;
        };
        let implemented = entry.component.abilities();
        if let Some(declared) = entry
            .node
            .abilities
            .as_ref()
            .filter(|abilities| !abilities.is_empty())
        {
            for ability in declared {
                if !implemented.contains(ability) {
                    record(
                        problems,
                        component_ability_invalid(
                            id,
                            ability.as_str(),
                            "the component does not implement the declared ability",
                            workflow,
                        ),
                    );
                }
            }
        }

        if entry.node.wait_for_all == Some(false) {
            let selected = def.selected_ability(id);
            if matches!(
                selected,
                ComponentAbility::Collect | ComponentAbility::Transform
            ) {
                record(
                    problems,
                    component_ability_invalid(
                        id,
                        selected.as_str(),
                        "stream components (TRANSFORM/COLLECT) must set 'wait_for_all' to True",
                        workflow,
                    ),
                );
            }
        }

        let node = &entry.node;
        check_schema_overlap(
            id,
            node.inputs_schema.as_ref(),
            "inputs_schema",
            node.stream_inputs_schema.as_ref(),
            "stream_inputs_schema",
            workflow,
            problems,
        );
        check_schema_overlap(
            id,
            node.outputs_schema.as_ref(),
            "outputs_schema",
            node.stream_outputs_schema.as_ref(),
            "stream_outputs_schema",
            workflow,
            problems,
        );
    }
}

/// Rejects a flattened key that appears in both the batch and the stream schema.
fn check_schema_overlap(
    id: &str,
    base: Option<&Value>,
    base_name: &str,
    stream: Option<&Value>,
    stream_name: &str,
    workflow: &str,
    problems: &mut Vec<String>,
) {
    let (Some(base), Some(stream)) = (base, stream) else {
        return;
    };
    let base_keys = flatten_keys(base);
    let stream_keys = flatten_keys(stream);
    let overlapping: Vec<&str> = base_keys
        .intersection(&stream_keys)
        .map(String::as_str)
        .collect();
    if overlapping.is_empty() {
        return;
    }
    record(
        problems,
        component_schema_invalid(
            id,
            &format!(
                "duplicate key both exist in {base_name} with {stream_name}, key={}",
                overlapping.join(", ")
            ),
            workflow,
        ),
    );
}

/// Collects every dotted path of the objects nested in `value`.
fn flatten_keys(value: &Value) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    flatten_into(value, "", &mut keys);
    keys
}

fn flatten_into(value: &Value, prefix: &str, keys: &mut BTreeSet<String>) {
    let Value::Object(map) = value else {
        return;
    };
    for (key, child) in map {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        keys.insert(path.clone());
        flatten_into(child, &path, keys);
    }
}

/// Whether `id` can emit a stream (`Stream` or `Transform`).
fn supports_stream_output(def: &WorkflowDefinition, id: &str) -> bool {
    def.supported_abilities(id).iter().any(|ability| {
        matches!(
            ability,
            ComponentAbility::Stream | ComponentAbility::Transform
        )
    })
}

/// Whether `id` can consume a stream (`Collect` or `Transform`).
fn supports_stream_input(def: &WorkflowDefinition, id: &str) -> bool {
    def.supported_abilities(id).iter().any(|ability| {
        matches!(
            ability,
            ComponentAbility::Collect | ComponentAbility::Transform
        )
    })
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

    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::card::WorkflowCard;
    use crate::engine::definition::{ConditionalSource, NodeConfig};

    use crate::components::{EndComponent, StartComponent};
    use crate::engine::test_support::{fake, router};
    /// start → end, both built-in components.
    fn definition() -> WorkflowDefinition {
        let mut definition = WorkflowDefinition::new(WorkflowCard::new("wf"));
        definition.insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        );
        definition.insert_component(
            "end",
            Arc::new(EndComponent::default()),
            NodeConfig::default(),
        );
        definition.start = Some("start".to_string());
        definition.end = Some("end".to_string());
        definition.add_edge(
            vec!["start".to_string()],
            "end".to_string(),
            EdgeKind::Regular,
        );
        definition
    }

    fn message_of(definition: &WorkflowDefinition) -> String {
        validate(definition).unwrap_err().message().into_owned()
    }

    #[test]
    fn valid_two_node_workflow_passes() {
        validate(&definition()).unwrap();
    }

    #[test]
    fn empty_workflow_reports_missing_start_and_end() {
        let mut definition = WorkflowDefinition::new(WorkflowCard::new("wf"));
        definition.insert_component(
            "node",
            fake("node", vec![ComponentAbility::Invoke]),
            NodeConfig::default(),
        );
        let message = message_of(&definition);
        assert!(message.contains("[100100]"), "{message}");
        assert!(message.contains("no start component"), "{message}");
        assert!(message.contains("no end component"), "{message}");
    }

    #[test]
    fn unregistered_start_and_end_are_reported() {
        let mut definition = definition();
        definition.start = Some("ghost".to_string());
        definition.end = Some("wraith".to_string());
        let message = message_of(&definition);
        assert!(
            message.contains("the start component 'ghost' is not registered"),
            "{message}"
        );
        assert!(
            message.contains("the end component 'wraith' is not registered"),
            "{message}"
        );
    }

    #[test]
    fn start_must_differ_from_end() {
        let mut definition = WorkflowDefinition::new(WorkflowCard::new("wf"));
        definition.insert_component(
            "node",
            fake("node", vec![ComponentAbility::Invoke]),
            NodeConfig::default(),
        );
        definition.start = Some("node".to_string());
        definition.end = Some("node".to_string());
        let message = message_of(&definition);
        assert!(
            message.contains("the start component 'node' is also the end component"),
            "{message}"
        );
        assert_eq!(message.matches(';').count(), 0, "{message}");
    }

    #[test]
    fn end_must_not_have_outgoing_edges() {
        let mut definition = definition();
        definition.add_edge(
            vec!["end".to_string()],
            "start".to_string(),
            EdgeKind::Regular,
        );
        definition.add_conditional("end", "start", router());
        let message = message_of(&definition);
        assert!(
            message.contains("the end component 'end' has an outgoing regular edge to 'start'"),
            "{message}"
        );
        assert!(
            message.contains("the end component 'end' has an outgoing conditional edge"),
            "{message}"
        );
    }

    #[test]
    fn component_ids_are_validated() {
        let mut definition = definition();
        definition.insert_component(
            "bad id",
            fake("bad", vec![ComponentAbility::Invoke]),
            NodeConfig::default(),
        );
        let message = message_of(&definition);
        assert!(message.contains("[100010]"), "{message}");
        assert!(
            message.contains("the component id is invalid for component 'bad id'"),
            "{message}"
        );
        assert_eq!(message.matches(';').count(), 0, "{message}");
    }

    #[test]
    fn unknown_edge_endpoints_are_reported() {
        let mut definition = definition();
        definition.add_edge(
            vec!["ghost".to_string()],
            "end".to_string(),
            EdgeKind::Regular,
        );
        definition.add_edge(
            vec!["start".to_string()],
            "phantom".to_string(),
            EdgeKind::Regular,
        );
        definition.add_conditional("start", "wraith", router());
        let message = message_of(&definition);
        assert!(
            message.contains("edge source 'ghost' is not registered"),
            "{message}"
        );
        assert!(
            message.contains("edge target 'phantom' is not registered"),
            "{message}"
        );
        assert!(
            message.contains("conditional edge target 'wraith' is not registered"),
            "{message}"
        );
    }

    #[test]
    fn start_must_support_invoke() {
        let mut definition = definition();
        definition.insert_component(
            "start",
            fake(
                "start",
                vec![ComponentAbility::Stream, ComponentAbility::Transform],
            ),
            NodeConfig::default(),
        );
        let message = message_of(&definition);
        assert!(message.contains("[100011]"), "{message}");
        assert!(
            message.contains("the ability is invalid for component 'start', ability=invoke"),
            "{message}"
        );
        assert!(
            message.contains("the start component must support 'invoke'"),
            "{message}"
        );
        assert_eq!(message.matches(';').count(), 0, "{message}");
    }

    #[test]
    fn conditional_source_needs_a_router() {
        let mut definition = definition();
        definition.conditional.insert(
            "start".to_string(),
            ConditionalSource {
                targets: vec!["end".to_string()],
                routers: Vec::new(),
            },
        );
        let message = message_of(&definition);
        assert!(message.contains("[100013]"), "{message}");
        assert!(message.contains("no router"), "{message}");
    }

    #[test]
    fn stream_edge_requires_a_streaming_source() {
        let mut definition = definition();
        definition.insert_component(
            "mid",
            fake("mid", vec![ComponentAbility::Invoke]),
            NodeConfig::default(),
        );
        definition.add_edge(
            vec!["start".to_string()],
            "mid".to_string(),
            EdgeKind::Regular,
        );
        definition.add_edge(vec!["mid".to_string()], "end".to_string(), EdgeKind::Stream);
        let message = message_of(&definition);
        assert!(message.contains("[100015]"), "{message}");
        assert!(
            message.contains("the source component does not support 'stream' or 'transform'"),
            "{message}"
        );
        assert!(message.contains("source='mid'"), "{message}");
        assert_eq!(message.matches(';').count(), 0, "{message}");
    }

    #[test]
    fn stream_edge_requires_a_collecting_target() {
        let mut definition = definition();
        definition.insert_component(
            "mid",
            fake(
                "mid",
                vec![ComponentAbility::Invoke, ComponentAbility::Stream],
            ),
            NodeConfig::default(),
        );
        definition.insert_component(
            "end",
            fake("end", vec![ComponentAbility::Invoke]),
            NodeConfig::default(),
        );
        definition.add_edge(
            vec!["start".to_string()],
            "mid".to_string(),
            EdgeKind::Regular,
        );
        definition.add_edge(vec!["mid".to_string()], "end".to_string(), EdgeKind::Stream);
        let message = message_of(&definition);
        assert!(message.contains("[100011]"), "{message}");
        assert!(
            message.contains("the ability is invalid for component 'end', ability=collect"),
            "{message}"
        );
        assert_eq!(message.matches(';').count(), 0, "{message}");
    }

    #[test]
    fn wait_for_all_false_is_rejected_for_a_collect_node() {
        let mut definition = definition();
        definition.insert_component(
            "mid",
            fake("mid", vec![ComponentAbility::Collect]),
            NodeConfig {
                abilities: Some(vec![ComponentAbility::Collect]),
                wait_for_all: Some(false),
                ..NodeConfig::default()
            },
        );
        definition.add_edge(
            vec!["start".to_string()],
            "mid".to_string(),
            EdgeKind::Regular,
        );
        definition.add_edge(
            vec!["mid".to_string()],
            "end".to_string(),
            EdgeKind::Regular,
        );
        let message = message_of(&definition);
        assert!(message.contains("[100011]"), "{message}");
        assert!(
            message.contains("the ability is invalid for component 'mid', ability=collect"),
            "{message}"
        );
        assert!(
            message.contains("must set 'wait_for_all' to True"),
            "{message}"
        );
        assert_eq!(message.matches(';').count(), 0, "{message}");
    }

    #[test]
    fn declared_ability_must_be_implemented() {
        let mut definition = definition();
        definition.insert_component(
            "mid",
            fake("mid", vec![ComponentAbility::Invoke]),
            NodeConfig {
                abilities: Some(vec![ComponentAbility::Stream]),
                ..NodeConfig::default()
            },
        );
        let message = message_of(&definition);
        assert!(message.contains("[100011]"), "{message}");
        assert!(
            message.contains("the ability is invalid for component 'mid', ability=stream"),
            "{message}"
        );
        assert!(
            message.contains("the component does not implement the declared ability"),
            "{message}"
        );
    }

    #[test]
    fn overlapping_input_schemas_are_reported() {
        let mut definition = definition();
        let node = &mut definition.components.get_mut("start").unwrap().node;
        node.inputs_schema = Some(json!({"query": "${inputs.query}", "nested": {"a": 1}}));
        node.stream_inputs_schema = Some(json!({"nested": {"a": 2}}));
        let message = message_of(&definition);
        assert!(message.contains("[100014]"), "{message}");
        assert!(
            message.contains(
                "duplicate key both exist in inputs_schema with stream_inputs_schema, \
                 key=nested, nested.a"
            ),
            "{message}"
        );
        assert_eq!(message.matches(';').count(), 0, "{message}");
    }

    #[test]
    fn overlapping_output_schemas_are_reported() {
        let mut definition = definition();
        let node = &mut definition.components.get_mut("end").unwrap().node;
        node.outputs_schema = Some(json!({"result": "${llm.result}"}));
        node.stream_outputs_schema = Some(json!({"result": "${llm.delta}"}));
        let message = message_of(&definition);
        assert!(message.contains("[100014]"), "{message}");
        assert!(
            message.contains(
                "duplicate key both exist in outputs_schema with stream_outputs_schema, key=result"
            ),
            "{message}"
        );
    }

    #[test]
    fn every_problem_is_joined_into_one_workflow_error() {
        let mut definition = WorkflowDefinition::new(WorkflowCard::new("wf"));
        definition.insert_component(
            "bad id",
            fake("bad", vec![ComponentAbility::Invoke]),
            NodeConfig::default(),
        );
        definition.record_problem("builder problem");

        let error = validate(&definition).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        let message = error.message().into_owned();
        let parts: Vec<&str> = message.split("; ").collect();
        assert_eq!(parts.len(), 4, "{message}");
        assert_eq!(parts[0], "builder problem");
        assert!(
            parts[1].starts_with("[100100]") && parts[1].contains("no start component"),
            "{message}"
        );
        assert!(
            parts[2].starts_with("[100100]") && parts[2].contains("no end component"),
            "{message}"
        );
        assert!(parts[3].starts_with("[100010]"), "{message}");
    }

    #[test]
    fn streaming_definition_with_branch_passes() {
        let mut definition = definition();
        definition.insert_component(
            "llm",
            fake(
                "llm",
                vec![ComponentAbility::Invoke, ComponentAbility::Stream],
            ),
            NodeConfig::default(),
        );
        definition.insert_component(
            "collect",
            fake("collect", vec![ComponentAbility::Collect]),
            NodeConfig::default(),
        );
        definition.start = Some("start".to_string());
        definition.end = Some("collect".to_string());
        definition.edges.clear();
        definition
            .add_edge(
                vec!["start".to_string()],
                "llm".to_string(),
                EdgeKind::Regular,
            )
            .add_edge(
                vec!["llm".to_string()],
                "collect".to_string(),
                EdgeKind::Stream,
            );
        definition.add_conditional("start", "llm", router());
        validate(&definition).unwrap();
    }
}
