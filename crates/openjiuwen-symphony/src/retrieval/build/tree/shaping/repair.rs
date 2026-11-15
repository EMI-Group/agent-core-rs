//! Minimal repair of a rejected node list: the `repair` tree-shaping stage.
//!
//! Python's `build/tree/repair.py` hands a subtree the builder refused to a model and uses
//! the answer when it validates. This stage ports that exchange for the flat node list
//! [`TreeSchema::validate`](crate::retrieval::build::schema::TreeSchema::validate) rejected.
//!
//! # Request and reply
//!
//! The user payload is
//! `{"stage":"repair","constraints":{…},"diagnostics":[{"code","cid","detail"}],"nodes":[…]}`.
//! Every diagnostic the caller supplied is rendered, and the nodes are rendered in
//! `(depth, cid)` order, at most
//! [`ShapingConfig::max_items_per_prompt`](crate::retrieval::build::tree::shaping::ShapingConfig::max_items_per_prompt)
//! of them (the first entries of that order), each with its `cid`, `type`, `worker_id`,
//! `description`, `select_when` and `dont_select_when`. The reply must be
//! `{"nodes":[{"cid":"Skills.documents","type":"branch"|"leaf","worker_id":"…","description":"…"}]}`.
//!
//! A reply node whose `cid` matches an input node keeps every field the reply does not
//! carry, so a minimal fix stays minimal; an entry with a blank `cid`, an unknown `type` or
//! a leaf with no `worker_id` is dropped. The reply is then rejected as a whole when the
//! multiset of its non-blank worker ids differs from the input's — a reply that loses,
//! duplicates or invents an item is not a repair — and the accepted nodes are sorted with
//! [`sort_nodes`](crate::retrieval::build::tree_text::sort_nodes) and re-validated.
//!
//! # The deterministic fallback
//!
//! The fallback preserves every leaf and prunes only branch nodes that carry no worker id
//! and have no descendant node, iterating to a fixed point so an emptied parent goes too.
//! Both paths finish with the same rule: a tree that still has diagnostics is
//! `Error::llm("tree shaping repair: the repaired tree is still invalid (<codes>)")`, naming
//! the surviving codes sorted, so a repair never quietly returns a broken tree.
//!
//! With no model configured, or with
//! [`ShapingConfig::max_calls`](crate::retrieval::build::tree::shaping::ShapingConfig::max_calls)
//! at `0`, the stage prunes without calling a model. When a model is called but fails or
//! answers unusably, [`FallbackMode::Deterministic`](crate::llm::FallbackMode) warns once and
//! prunes, while [`FallbackMode::Error`](crate::llm::FallbackMode) returns the failure.
//! An input whose diagnostics are empty is answered unchanged without a call.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::llm::{string_field, truncate_text, CompletionModel};
use crate::retrieval::build::schema::TreeDiagnostic;
use crate::retrieval::build::schema::TreeSchema;
use crate::retrieval::build::tree_text::sort_nodes;
use crate::retrieval::types::{IndexNode, NODE_TYPE_BRANCH, NODE_TYPE_LEAF};

use super::{call_model, no_usable_reply, on_model_failure, prompt_len, ShapingConfig};

/// The system instruction sent with every repair request.
const SYSTEM_PROMPT: &str = "Repair a capability routing tree. Return JSON only: \
{\"nodes\":[{\"cid\":\"Skills.documents\",\"type\":\"branch\",\"worker_id\":\"\",\"description\":\"…\"}]}. \
Keep every leaf's worker_id exactly once, keep every cid's dot-joined hierarchy, stay \
inside the given bounds, and change as little as the diagnostics require.";

/// Repairs a node list the schema guard rejected, minimally.
///
/// # Errors
///
/// `Error::Llm` when the repaired tree still has diagnostics (naming `repair` and the
/// surviving codes) and, under
/// [`FallbackMode::Error`](crate::llm::FallbackMode), the propagated model failure or
/// `Error::Llm` for an unusable reply.
pub async fn repair(
    nodes: &[IndexNode],
    diagnostics: &[TreeDiagnostic],
    schema: &TreeSchema,
    shaping: &ShapingConfig,
    model: Option<&dyn CompletionModel>,
) -> Result<Vec<IndexNode>> {
    let effective: Vec<TreeDiagnostic> = if diagnostics.is_empty() {
        schema.validate(nodes)
    } else {
        diagnostics.to_vec()
    };
    if effective.is_empty() {
        return Ok(nodes.to_vec());
    }
    let Some(model) = model else {
        return deterministic_repair(nodes, schema);
    };
    if shaping.max_calls == 0 {
        return deterministic_repair(nodes, schema);
    }
    let request = repair_request(nodes, &effective, schema, shaping);
    let mut warned = false;
    let repaired = match call_model(model, SYSTEM_PROMPT, &request).await {
        Ok(Some(value)) => read_nodes(&value, nodes),
        Ok(None) => None,
        Err(error) => {
            on_model_failure(shaping, &mut warned, "repair", error)?;
            return deterministic_repair(nodes, schema);
        }
    };
    let Some(mut repaired) = repaired else {
        on_model_failure(shaping, &mut warned, "repair", no_usable_reply("repair"))?;
        return deterministic_repair(nodes, schema);
    };
    sort_nodes(&mut repaired);
    ensure_valid(repaired, schema)
}

/// Renders the repair request.
fn repair_request(
    nodes: &[IndexNode],
    diagnostics: &[TreeDiagnostic],
    schema: &TreeSchema,
    shaping: &ShapingConfig,
) -> String {
    let mut ordered: Vec<&IndexNode> = nodes.iter().collect();
    ordered.sort_by(|left, right| {
        (left.depth(), left.cid.as_str()).cmp(&(right.depth(), right.cid.as_str()))
    });
    let rendered: Vec<Value> = ordered
        .iter()
        .take(prompt_len(ordered.len(), shaping))
        .map(|node| {
            json!({
                "cid": node.cid,
                "type": node.node_type,
                "worker_id": node.worker_id,
                "description": truncate_text(node.description.trim(), shaping.description_limit),
                "select_when": node.select_when,
                "dont_select_when": node.dont_select_when,
            })
        })
        .collect();
    let problems: Vec<Value> = diagnostics
        .iter()
        .map(|diagnostic| {
            json!({
                "code": diagnostic.code,
                "cid": diagnostic.cid,
                "detail": diagnostic.detail,
            })
        })
        .collect();
    json!({
        "stage": "repair",
        "constraints": {
            "max_depth": schema.max_depth,
            "max_children": schema.max_children,
            "max_nodes": schema.max_nodes,
        },
        "diagnostics": problems,
        "nodes": rendered,
    })
    .to_string()
}

/// Reads `{"nodes":[…]}` into an `IndexNode` list, rejecting an id-changing answer.
///
/// Returns `None` when the reply carries no `nodes` array, parses to no usable node, or
/// names a different multiset of non-blank worker ids than `input`.
fn read_nodes(value: &Value, input: &[IndexNode]) -> Option<Vec<IndexNode>> {
    let entries = value.get("nodes")?.as_array()?;
    let by_cid: BTreeMap<&str, &IndexNode> =
        input.iter().map(|node| (node.cid.trim(), node)).collect();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut nodes: Vec<IndexNode> = Vec::new();
    for entry in entries {
        let Some(cid) = string_field(entry, "cid") else {
            continue;
        };
        if !seen.insert(cid.clone()) {
            continue;
        }
        let node_type = match string_field(entry, "type").as_deref() {
            Some(NODE_TYPE_BRANCH) => NODE_TYPE_BRANCH,
            Some(NODE_TYPE_LEAF) => NODE_TYPE_LEAF,
            _ => continue,
        };
        let worker_id = string_field(entry, "worker_id").unwrap_or_default();
        if node_type == NODE_TYPE_LEAF && worker_id.is_empty() {
            continue;
        }
        let mut node = by_cid.get(cid.as_str()).map_or_else(
            || IndexNode::new(cid.clone(), node_type),
            |input| (*input).clone(),
        );
        node.node_type = node_type.to_string();
        node.worker_id = worker_id;
        if let Some(description) = string_field(entry, "description") {
            node.description = description;
        }
        nodes.push(node);
    }
    if nodes.is_empty() || worker_multiset(&nodes) != worker_multiset(input) {
        return None;
    }
    Some(nodes)
}

/// Counts the non-blank worker ids of a node list.
fn worker_multiset(nodes: &[IndexNode]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for node in nodes {
        let worker_id = node.worker_id.trim();
        if worker_id.is_empty() {
            continue;
        }
        *counts.entry(worker_id.to_string()).or_default() += 1;
    }
    counts
}

/// Prunes empty branches, sorts, and reports the surviving diagnostics as an error.
fn deterministic_repair(nodes: &[IndexNode], schema: &TreeSchema) -> Result<Vec<IndexNode>> {
    let mut pruned = prune_empty_branches(nodes);
    sort_nodes(&mut pruned);
    ensure_valid(pruned, schema)
}

/// Drops branch nodes that carry no worker id and no descendant, to a fixed point.
fn prune_empty_branches(nodes: &[IndexNode]) -> Vec<IndexNode> {
    let mut kept: Vec<IndexNode> = nodes.to_vec();
    loop {
        let parents: BTreeSet<String> = kept
            .iter()
            .filter_map(|node| node.parent_cid().map(str::trim).map(str::to_string))
            .collect();
        let before = kept.len();
        kept.retain(|node| {
            !(node.is_branch()
                && node.worker_id.trim().is_empty()
                && !parents.contains(node.cid.trim()))
        });
        if kept.len() == before {
            return kept;
        }
    }
}

/// Reports `nodes`, or the surviving diagnostics sorted by code.
fn ensure_valid(nodes: Vec<IndexNode>, schema: &TreeSchema) -> Result<Vec<IndexNode>> {
    let remaining = schema.validate(&nodes);
    if remaining.is_empty() {
        return Ok(nodes);
    }
    let mut codes: Vec<String> = remaining
        .into_iter()
        .map(|diagnostic| diagnostic.code)
        .collect();
    codes.sort();
    codes.dedup();
    Err(Error::llm(format!(
        "tree shaping repair: the repaired tree is still invalid ({})",
        codes.join(", ")
    )))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_retrieval::StaticCompletionModel;

    use super::super::fixtures::shaping;
    use super::*;
    use crate::error::ErrorKind;
    use crate::llm::FallbackMode;

    fn invalid_tree() -> Vec<IndexNode> {
        vec![
            IndexNode::branch("Skills").with_description("Capabilities for Skills."),
            IndexNode::branch("Skills.empty"),
            IndexNode::leaf("Skills.tool", "w1").with_description("Tool"),
        ]
    }

    fn dangling_tree() -> Vec<IndexNode> {
        vec![
            IndexNode::branch("Skills").with_description("Capabilities for Skills."),
            IndexNode::branch("Skills.ghost.b").with_description("Capabilities for b."),
            IndexNode::leaf("Skills.tool", "w1").with_description("Tool"),
        ]
    }

    #[tokio::test]
    async fn repair_accepts_a_minimal_model_fix() {
        let nodes = invalid_tree();
        let schema = TreeSchema::new();
        let diagnostics = schema.validate(&nodes);
        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        let model = StaticCompletionModel::single(
            r#"{"nodes":[
                {"cid":"Skills","type":"branch","description":"Capabilities for Skills."},
                {"cid":"Skills.tool","type":"leaf","worker_id":"w1"}
            ]}"#,
        );
        let repaired = repair(&nodes, &diagnostics, &schema, &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(
            repaired
                .iter()
                .map(|node| node.cid.as_str())
                .collect::<Vec<_>>(),
            vec!["Skills", "Skills.tool"]
        );
        // The reply left the leaf's description out, so the input's value stands.
        assert_eq!(repaired[1].description, "Tool");
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn repair_keeps_the_fields_the_reply_omits() {
        let mut nodes = invalid_tree();
        nodes[2].select_when = "when a tool is needed".to_string();
        let schema = TreeSchema::new();
        let diagnostics = schema.validate(&nodes);
        let model = StaticCompletionModel::single(
            r#"{"nodes":[
                {"cid":"Skills","type":"branch","description":"Capabilities for Skills."},
                {"cid":"Skills.tool","type":"leaf","worker_id":"w1"}
            ]}"#,
        );
        let repaired = repair(&nodes, &diagnostics, &schema, &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(repaired[1].select_when, "when a tool is needed");
    }

    #[tokio::test]
    async fn repair_rejects_a_reply_that_loses_or_invents_worker_ids() {
        let nodes = invalid_tree();
        let lost = json!({"nodes":[
            {"cid":"Skills","type":"branch","description":"d"},
            {"cid":"Skills.empty","type":"branch"}
        ]});
        assert!(read_nodes(&lost, &nodes).is_none());
        let invented = json!({"nodes":[
            {"cid":"Skills","type":"branch","description":"d"},
            {"cid":"Skills.tool","type":"leaf","worker_id":"w1"},
            {"cid":"Skills.other","type":"leaf","worker_id":"ghost"}
        ]});
        assert!(read_nodes(&invented, &nodes).is_none());
        let duplicated = json!({"nodes":[
            {"cid":"Skills","type":"branch","description":"d"},
            {"cid":"Skills.tool","type":"leaf","worker_id":"w1"},
            {"cid":"Skills.tool-2","type":"leaf","worker_id":"w1"}
        ]});
        assert!(read_nodes(&duplicated, &nodes).is_none());
        assert!(read_nodes(&json!({"nodes": []}), &nodes).is_none());
        assert!(read_nodes(&json!({"assignments": []}), &nodes).is_none());
    }

    #[tokio::test]
    async fn repair_reports_the_surviving_diagnostics() {
        let nodes = dangling_tree();
        let schema = TreeSchema::new();
        let diagnostics = schema.validate(&nodes);
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "missing_parent"));
        let model = StaticCompletionModel::single(
            r#"{"nodes":[
                {"cid":"Skills","type":"branch","description":"Capabilities for Skills."},
                {"cid":"Skills.ghost.b","type":"branch","description":"Capabilities for b."},
                {"cid":"Skills.tool","type":"leaf","worker_id":"w1"}
            ]}"#,
        );
        let error = repair(&nodes, &diagnostics, &schema, &shaping(), Some(&model))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("repair"), "{error}");
        assert!(error.message().contains("missing_parent"), "{error}");
    }

    #[tokio::test]
    async fn repair_falls_back_to_pruning_an_empty_branch() {
        let nodes = invalid_tree();
        let schema = TreeSchema::new();
        let diagnostics = schema.validate(&nodes);
        let model = StaticCompletionModel::single("not a JSON object");
        let repaired = repair(&nodes, &diagnostics, &schema, &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(
            repaired
                .iter()
                .map(|node| node.cid.as_str())
                .collect::<Vec<_>>(),
            vec!["Skills", "Skills.tool"]
        );
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn repair_prunes_without_a_model() {
        let nodes = invalid_tree();
        let schema = TreeSchema::new();
        let repaired = repair(&nodes, &[], &schema, &shaping(), None)
            .await
            .unwrap();
        assert_eq!(
            repaired
                .iter()
                .map(|node| node.cid.as_str())
                .collect::<Vec<_>>(),
            vec!["Skills", "Skills.tool"]
        );
    }

    #[tokio::test]
    async fn repair_error_mode_surfaces_the_failure() {
        let nodes = invalid_tree();
        let schema = TreeSchema::new();
        let diagnostics = schema.validate(&nodes);
        let model = StaticCompletionModel::single("not a JSON object");
        let shaping = shaping().with_fallback(FallbackMode::Error);
        let error = repair(&nodes, &diagnostics, &schema, &shaping, Some(&model))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("repair"), "{error}");
        assert!(error.message().contains("no usable JSON object"), "{error}");
    }

    #[tokio::test]
    async fn repair_error_mode_surfaces_an_invocation_failure() {
        let nodes = invalid_tree();
        let schema = TreeSchema::new();
        let diagnostics = schema.validate(&nodes);
        let model = StaticCompletionModel::new(Vec::new());
        let shaping = shaping().with_fallback(FallbackMode::Error);
        let error = repair(&nodes, &diagnostics, &schema, &shaping, Some(&model))
            .await
            .unwrap_err();
        assert!(error.message().contains("no scripted responses"), "{error}");
    }

    #[tokio::test]
    async fn repair_sorts_the_node_list() {
        let nodes = invalid_tree();
        let schema = TreeSchema::new();
        let diagnostics = schema.validate(&nodes);
        let model = StaticCompletionModel::single(
            r#"{"nodes":[
                {"cid":"Skills.tool","type":"leaf","worker_id":"w1"},
                {"cid":"Skills","type":"branch","description":"Capabilities for Skills."}
            ]}"#,
        );
        let repaired = repair(&nodes, &diagnostics, &schema, &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(
            repaired
                .iter()
                .map(|node| node.cid.as_str())
                .collect::<Vec<_>>(),
            vec!["Skills", "Skills.tool"]
        );
    }

    #[tokio::test]
    async fn repair_answers_a_valid_tree_unchanged() {
        let nodes = vec![
            IndexNode::branch("Skills").with_description("Capabilities for Skills."),
            IndexNode::leaf("Skills.tool", "w1").with_description("Tool"),
        ];
        let schema = TreeSchema::new();
        assert_eq!(schema.validate(&nodes), Vec::new());
        let model = StaticCompletionModel::single("unused");
        let repaired = repair(&nodes, &[], &schema, &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(repaired, nodes);
        assert_eq!(model.call_count(), 0);
    }

    #[test]
    fn prune_empty_branches_keeps_leaves_and_prunes_to_a_fixed_point() {
        let nodes = vec![
            IndexNode::branch("Skills").with_description("root"),
            IndexNode::branch("Skills.a").with_description("a"),
            IndexNode::branch("Skills.a.b").with_description("b"),
            IndexNode::leaf("Skills.tool", "w1"),
            IndexNode::leaf("Skills.blank", ""),
        ];
        let pruned = prune_empty_branches(&nodes);
        assert_eq!(
            pruned
                .iter()
                .map(|node| node.cid.as_str())
                .collect::<Vec<_>>(),
            vec!["Skills", "Skills.tool", "Skills.blank"]
        );
    }
}
