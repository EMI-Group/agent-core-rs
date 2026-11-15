//! Tree presets: the persisted-or-caller-supplied taxonomy that steers a build.
//!
//! Ports the preset path of Python `openjiuwen.symphony.retrieval.build`: the nested
//! `tree_index.yaml` document read by `build/io/tree.py`
//! (`load_tree_preset` / `parse_simple_nodes_yaml`) and flattened by
//! `build/tree/preset_writer.py`'s `flatten_capability_tree`. Where Python hands the model
//! a nested list of `dict`s, here the preset is a typed value object, `TreePreset`, that
//! can be written by hand, read back from a persisted `tree_index.yaml`, or rebuilt from a
//! flat `IndexNode` list.
//!
//! Python's static `SKILLSMP_ROOT_CATEGORIES` taxonomy of `build/tree/presets.py` (`tools`,
//! `business`, `development`, …) is **not** ported: it is a large example taxonomy that
//! nothing in the Python package imports, and a preset here always comes from a caller or
//! from a persisted tree document.
//!
//! # Override semantics
//!
//! A caller-supplied preset wins over every grouping strategy: the build stage assigns
//! items onto the preset's paths and derives only the leaf `cid`s, so a curated taxonomy
//! is never re-arranged by the rule, embedding or model grouping. That is why `TreePreset::flatten`
//! treats an explicit `cid` as authoritative even when it does not look like its parent's
//! `cid` plus one segment.
//!
//! # `cid` derivation
//!
//! A preset node may leave `cid` blank — a hand-written taxonomy usually does. Derivation
//! runs top-down, so a node only needs its own text: an explicit `cid` wins verbatim,
//! otherwise the node becomes `join_cid(parent, slug_term(worker_id or description))`
//! (seeded with `"node"` when both are blank) and gets a `-2`, `-3`, … suffix when a
//! sibling already claimed that `cid`. `node_type` defaults to `leaf` when the node names
//! a `worker_id` and to `branch` otherwise.
//!
//! # Document shapes
//!
//! `TreePreset::from_yaml` reads the same `nodes:` document as
//! [`parse_tree_yaml`](crate::retrieval::artifacts::parse_tree_yaml), with one extension: a
//! node may carry nested `children:`. A node that carries no `children:` key therefore
//! loads as a leaf entry of the flat document, which makes
//! `TreePreset::from_yaml(&render_tree_yaml(&nodes)).flatten() == nodes` hold for a
//! persisted index. Unknown keys are ignored, missing keys are defaulted, and a scalar of
//! any JSON type is stringified.
//!
//! Nothing here allocates a `HashMap` or depends on insertion order: every collection is a
//! `BTreeMap`/`BTreeSet`, so two runs over the same input produce the same paths.

use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind as IoErrorKind;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs;
use tracing::warn;

use crate::error::{validation, Error, Result};
use crate::retrieval::build::schema::{TreeDiagnostic, TreeSchema};
use crate::retrieval::build::tree_text::{slug_term, sort_nodes, unique_child_cid};
use crate::retrieval::types::{IndexNode, NODE_TYPE_BRANCH, NODE_TYPE_LEAF, TREE_INDEX_FILENAME};

/// A caller-supplied or persisted tree preset: nested nodes with top-down derivable cids.
///
/// Build it by hand, with [`TreePreset::from_nodes`] from a flat node list, or with
/// [`TreePreset::from_yaml`]/[`TreePreset::from_file`] from a persisted document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TreePreset {
    /// The top-level preset nodes.
    pub nodes: Vec<PresetNode>,
}

/// One nested node of a [`TreePreset`].
///
/// Every field mirrors [`IndexNode`] plus `children`;
/// `cid` and `node_type` may be blank and are derived by [`TreePreset::flatten`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PresetNode {
    /// The node's `cid`; blank when it should be derived from the parent and the text.
    pub cid: String,
    /// The node type; serialized as `type`. Blank means "derive from `worker_id`".
    #[serde(rename = "type")]
    pub node_type: String,
    /// The node's routing description.
    pub description: String,
    /// When a model should pick this node.
    pub select_when: String,
    /// When a model should not pick this node.
    pub dont_select_when: String,
    /// The description the node was derived from.
    pub source_description: String,
    /// The worker id of the item a leaf carries; empty for a branch.
    pub worker_id: String,
    /// The node's children, in document order.
    pub children: Vec<PresetNode>,
}

impl PresetNode {
    /// The text a derived `cid` segment comes from: the worker id, else the description.
    fn label_seed(&self) -> &str {
        let worker = self.worker_id.trim();
        if worker.is_empty() {
            self.description.trim()
        } else {
            worker
        }
    }

    /// Whether a `cid` can be derived: an explicit `cid`, a worker id or a description.
    fn derivable(&self) -> bool {
        !self.cid.trim().is_empty() || !self.label_seed().is_empty()
    }

    /// The node type to emit: the declared one, else `leaf` with a worker id and `branch`
    /// without.
    fn resolved_type(&self) -> &str {
        let declared = self.node_type.trim();
        if declared.is_empty() {
            if self.worker_id.trim().is_empty() {
                NODE_TYPE_BRANCH
            } else {
                NODE_TYPE_LEAF
            }
        } else {
            declared
        }
    }
}

/// One flattened preset node with the context the schema needs to check it.
///
/// [`TreeSchema::validate_preset`](crate::retrieval::build::schema::TreeSchema::validate_preset)
/// is the only consumer: `ancestors` carries the derived `cid`s above the node (a repeat is
/// a cycle) and `derivable` records whether the node itself supplied any `cid` text.
pub(crate) struct PresetEntry {
    /// The flattened node, with its `cid` and `node_type` resolved.
    pub(crate) node: IndexNode,
    /// The resolved `cid`s of the node's ancestors, outermost first.
    pub(crate) ancestors: Vec<String>,
    /// Whether the preset node itself supplied a `cid`, description or worker id.
    pub(crate) derivable: bool,
}

/// Walks `nodes` depth-first, resolving each `cid` and recording its ancestors.
fn collect_preset(
    nodes: &[PresetNode],
    parent: &str,
    ancestors: &[String],
    used: &mut BTreeSet<String>,
    out: &mut Vec<PresetEntry>,
) {
    for node in nodes {
        let cid = if node.cid.trim().is_empty() {
            let seed = node.label_seed();
            let seed = if seed.is_empty() { "node" } else { seed };
            unique_child_cid(parent, &slug_term(seed, "node"), used)
        } else {
            let explicit = node.cid.trim().to_string();
            used.insert(explicit.clone());
            explicit
        };
        out.push(PresetEntry {
            node: IndexNode {
                cid: cid.clone(),
                node_type: node.resolved_type().to_string(),
                description: node.description.clone(),
                select_when: node.select_when.clone(),
                dont_select_when: node.dont_select_when.clone(),
                source_description: node.source_description.clone(),
                worker_id: node.worker_id.clone(),
            },
            ancestors: ancestors.to_vec(),
            derivable: node.derivable(),
        });
        let mut nested = ancestors.to_vec();
        nested.push(cid.clone());
        collect_preset(&node.children, &cid, &nested, used, out);
    }
}

impl TreePreset {
    /// Rebuilds the nesting from a flat node list.
    ///
    /// Every `cid`'s ancestors are materialised, so a parent that has no explicit node in
    /// `nodes` becomes a derived branch node carrying the `cid` (and nothing else). Roots
    /// are the nodes nobody claims as a parent, siblings are sorted by `cid`, and a
    /// duplicated `cid` collapses to its last occurrence. A `cid` that is not a
    /// well-formed path (`.A`) still appears at the top level.
    #[must_use]
    pub fn from_nodes(nodes: &[IndexNode]) -> Self {
        let mut explicit: BTreeMap<String, IndexNode> = BTreeMap::new();
        let mut cids: BTreeSet<String> = BTreeSet::new();
        let mut child_sets: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for node in nodes {
            let cid = node.cid.trim().to_string();
            cids.insert(cid.clone());
            explicit.insert(cid.clone(), node.clone());
            let mut current = cid.as_str();
            while let Some(position) = current.rfind('.') {
                let parent = &current[..position];
                if parent.is_empty() {
                    break;
                }
                cids.insert(parent.to_string());
                child_sets
                    .entry(parent.to_string())
                    .or_default()
                    .insert(current.to_string());
                current = parent;
            }
        }
        let claimed: BTreeSet<&str> = child_sets
            .values()
            .flat_map(|children| children.iter().map(String::as_str))
            .collect();
        let roots: Vec<String> = cids
            .iter()
            .filter(|cid| !claimed.contains(cid.as_str()))
            .cloned()
            .collect();
        Self {
            nodes: roots
                .iter()
                .map(|cid| preset_from_cid(cid, &explicit, &child_sets))
                .collect(),
        }
    }

    /// Reads a preset from a `tree_index.yaml`-shaped document.
    ///
    /// The document is a mapping with a `nodes:` sequence, exactly like the persisted
    /// `tree_index.yaml`, except that a node may also carry nested `children:`. Unknown
    /// keys are ignored, missing keys become empty strings, a scalar of any JSON type is
    /// stringified, and `type` (or the legacy `node_type`) spells the node type. A non-
    /// mapping entry is skipped with a warning rather than failing the whole preset.
    ///
    /// # Errors
    ///
    /// `Error::Validation` naming `tree_index.yaml` when the YAML is unusable, the
    /// document is not a mapping, or it has no `nodes` sequence.
    pub fn from_yaml(text: &str) -> Result<Self> {
        let document: serde_yaml_ng::Value = serde_yaml_ng::from_str(text)
            .map_err(|error| validation(TREE_INDEX_FILENAME, format!("invalid YAML: {error}")))?;
        let document = serde_json::to_value(document).map_err(|error| {
            validation(
                TREE_INDEX_FILENAME,
                format!("unusable YAML document: {error}"),
            )
        })?;
        if !document.is_object() {
            return Err(validation(
                TREE_INDEX_FILENAME,
                "expected a mapping with a `nodes` sequence",
            ));
        }
        let Some(raw_nodes) = document.get("nodes") else {
            return Err(validation(TREE_INDEX_FILENAME, "missing `nodes` sequence"));
        };
        let Some(entries) = raw_nodes.as_array() else {
            return Err(validation(TREE_INDEX_FILENAME, "`nodes` is not a sequence"));
        };
        Ok(Self {
            nodes: preset_nodes_from_values(entries),
        })
    }

    /// Reads a preset from `path` with [`TreePreset::from_yaml`].
    ///
    /// # Errors
    ///
    /// `Error::NotFound` when the file does not exist, `Error::External` for another I/O
    /// failure, and the same validation errors as [`TreePreset::from_yaml`].
    pub async fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = match fs::read_to_string(path).await {
            Ok(text) => text,
            Err(error) if error.kind() == IoErrorKind::NotFound => {
                return Err(Error::not_found(format!("{} not found", path.display())));
            }
            Err(error) => {
                return Err(
                    Error::external(error).with_context(format!("reading `{}`", path.display()))
                );
            }
        };
        Self::from_yaml(&text)
    }

    /// Derives every `cid` top-down and returns the flat node list.
    ///
    /// An explicit `cid` wins verbatim; otherwise the node is
    /// `join_cid(parent, slug_term(worker_id or description or "node"))` with a numeric
    /// suffix when a sibling already used that `cid`. `node_type` is the declared type, or
    /// `leaf` for a node with a `worker_id` and `branch` otherwise. The result is sorted by
    /// `(depth, cid)`, the order every writer emits.
    #[must_use]
    pub fn flatten(&self) -> Vec<IndexNode> {
        let mut nodes: Vec<IndexNode> =
            self.entries().into_iter().map(|entry| entry.node).collect();
        sort_nodes(&mut nodes);
        nodes
    }

    /// The flattened nodes plus the context
    /// [`TreeSchema::validate_preset`](crate::retrieval::build::schema::TreeSchema::validate_preset)
    /// needs.
    pub(crate) fn entries(&self) -> Vec<PresetEntry> {
        let mut used = BTreeSet::new();
        let mut out = Vec::new();
        collect_preset(&self.nodes, "", &[], &mut used, &mut out);
        out
    }

    /// Every node's `cid` split into segments, sorted and deduplicated.
    ///
    /// This is the taxonomy hand-off to the grouping stage: each path is one place items
    /// may be assigned to.
    #[must_use]
    pub fn paths(&self) -> Vec<Vec<String>> {
        let mut paths: Vec<Vec<String>> = self
            .flatten()
            .iter()
            .map(|node| {
                node.segments()
                    .iter()
                    .map(|part| (*part).to_string())
                    .collect()
            })
            .collect();
        paths.sort();
        paths.dedup();
        paths
    }

    /// The paths of the nodes that are leaves: those carrying a `worker_id` and those with
    /// no children.
    ///
    /// Sorted and deduplicated like [`paths`](Self::paths).
    #[must_use]
    pub fn leaf_paths(&self) -> Vec<Vec<String>> {
        let nodes = self.flatten();
        let parents: BTreeSet<&str> = nodes.iter().filter_map(|node| node.parent_cid()).collect();
        let mut paths: Vec<Vec<String>> = nodes
            .iter()
            .filter(|node| {
                !node.worker_id.trim().is_empty() || !parents.contains(node.cid.as_str())
            })
            .map(|node| {
                node.segments()
                    .iter()
                    .map(|part| (*part).to_string())
                    .collect()
            })
            .collect();
        paths.sort();
        paths.dedup();
        paths
    }

    /// Checks the preset's flattened shape against `schema`.
    ///
    /// Delegates to
    /// [`TreeSchema::validate_preset`](crate::retrieval::build::schema::TreeSchema::validate_preset),
    /// so nesting-only problems (a `cycle`, an underivable `cid`) are reported as well.
    #[must_use]
    pub fn validate(&self, schema: &TreeSchema) -> Vec<TreeDiagnostic> {
        schema.validate_preset(self)
    }

    /// Whether the preset has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Rebuilds one nested preset node from a resolved `cid`.
fn preset_from_cid(
    cid: &str,
    explicit: &BTreeMap<String, IndexNode>,
    child_sets: &BTreeMap<String, BTreeSet<String>>,
) -> PresetNode {
    let mut node = explicit.get(cid).map_or_else(
        || PresetNode {
            cid: cid.to_string(),
            node_type: NODE_TYPE_BRANCH.to_string(),
            ..PresetNode::default()
        },
        |node| PresetNode {
            cid: node.cid.clone(),
            node_type: node.node_type.clone(),
            description: node.description.clone(),
            select_when: node.select_when.clone(),
            dont_select_when: node.dont_select_when.clone(),
            source_description: node.source_description.clone(),
            worker_id: node.worker_id.clone(),
            children: Vec::new(),
        },
    );
    if let Some(children) = child_sets.get(cid) {
        node.children = children
            .iter()
            .map(|child| preset_from_cid(child, explicit, child_sets))
            .collect();
    }
    node
}

/// Parses the entries of a preset's `nodes:` sequence.
fn preset_nodes_from_values(entries: &[Value]) -> Vec<PresetNode> {
    let mut nodes = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        if let Some(node) = preset_node_from_value(entry, index) {
            nodes.push(node);
        }
    }
    nodes
}

/// Parses one preset node, tolerating every missing or unknown key.
fn preset_node_from_value(value: &Value, index: usize) -> Option<PresetNode> {
    let Some(fields) = value.as_object() else {
        warn!(
            entry = index,
            "skipping a preset node that is not a mapping"
        );
        return None;
    };
    let node_type = field_string(fields, "type");
    let node_type = if node_type.trim().is_empty() {
        field_string(fields, "node_type")
    } else {
        node_type
    };
    let children = fields
        .get("children")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |entries| preset_nodes_from_values(entries));
    Some(PresetNode {
        cid: field_string(fields, "cid").trim().to_string(),
        node_type: node_type.trim().to_string(),
        description: field_string(fields, "description"),
        select_when: field_string(fields, "select_when"),
        dont_select_when: field_string(fields, "dont_select_when"),
        source_description: field_string(fields, "source_description"),
        worker_id: field_string(fields, "worker_id").trim().to_string(),
        children,
    })
}

/// Reads one field as a string, defaulting to `""`.
fn field_string(fields: &serde_json::Map<String, Value>, key: &str) -> String {
    fields.get(key).map_or(String::new(), scalar_string)
}

/// Stringifies a JSON scalar; `null` and containers become `""`.
fn scalar_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        _ => String::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use crate::retrieval::artifacts::render_tree_yaml;
    use crate::retrieval::build::schema::DIAGNOSTIC_CYCLE;

    fn branch(cid: &str, description: &str) -> IndexNode {
        IndexNode::branch(cid).with_description(description)
    }

    fn flat_tree() -> Vec<IndexNode> {
        vec![
            branch("Skills", "root"),
            branch("Skills.a", "alpha").with_select_when("pick alpha"),
            branch("Skills.b", "beta"),
            IndexNode::leaf("Skills.a.one", "worker-one").with_source_description("source"),
        ]
    }

    #[test]
    fn from_nodes_round_trips_through_flatten() {
        let nodes = flat_tree();
        let preset = TreePreset::from_nodes(&nodes);
        assert_eq!(preset.flatten(), nodes);
        assert!(!preset.is_empty());
        assert_eq!(preset.nodes.len(), 1);
        assert_eq!(preset.nodes[0].children.len(), 2);
        assert!(TreePreset::default().is_empty());
    }

    #[test]
    fn from_nodes_derives_missing_parents_as_branches() {
        let preset = TreePreset::from_nodes(&[IndexNode::leaf("A.B.C", "worker")]);
        let flat = preset.flatten();
        assert_eq!(
            flat,
            vec![
                IndexNode::branch("A"),
                IndexNode::branch("A.B"),
                IndexNode::leaf("A.B.C", "worker"),
            ]
        );
    }

    #[test]
    fn from_yaml_derives_cids_and_types() {
        let preset = TreePreset::from_yaml(
            "nodes:\n  - id: research\n    description: Research skills\n    children:\n      - \
             worker_id: deep research\n      - description: Web Search\n",
        )
        .unwrap();
        assert_eq!(preset.nodes.len(), 1);
        let flat = preset.flatten();
        let cids: Vec<&str> = flat.iter().map(|node| node.cid.as_str()).collect();
        assert_eq!(
            cids,
            vec![
                "research-skills",
                "research-skills.deep-research",
                "research-skills.web-search"
            ]
        );
        assert!(flat[0].is_branch());
        assert!(flat[1].is_leaf());
        assert_eq!(flat[1].worker_id, "deep research");
        assert!(flat[2].is_branch());
    }

    #[test]
    fn from_yaml_reads_a_flat_persisted_document() {
        let nodes = flat_tree();
        let preset = TreePreset::from_yaml(&render_tree_yaml(&nodes)).unwrap();
        assert_eq!(preset.flatten(), nodes);
    }

    #[test]
    fn from_yaml_derives_unique_sibling_cids() {
        let preset = TreePreset::from_yaml(
            "nodes:\n  - description: root\n    children:\n      - description: same\n      - \
             description: same\n",
        )
        .unwrap();
        let cids: Vec<String> = preset.flatten().iter().map(|n| n.cid.clone()).collect();
        assert_eq!(cids, vec!["root", "root.same", "root.same-2"]);
    }

    #[test]
    fn paths_and_leaf_paths_are_sorted_and_deduped() {
        let preset = TreePreset::from_nodes(&flat_tree());
        assert_eq!(
            preset.paths(),
            vec![
                vec!["Skills".to_string()],
                vec!["Skills".to_string(), "a".to_string()],
                vec!["Skills".to_string(), "a".to_string(), "one".to_string()],
                vec!["Skills".to_string(), "b".to_string()],
            ]
        );
        assert_eq!(
            preset.leaf_paths(),
            vec![
                vec!["Skills".to_string(), "a".to_string(), "one".to_string()],
                vec!["Skills".to_string(), "b".to_string()],
            ]
        );
    }

    #[test]
    fn validate_delegates_to_the_preset_schema() {
        let preset = TreePreset {
            nodes: vec![PresetNode {
                cid: "A".to_string(),
                node_type: "branch".to_string(),
                description: "a".to_string(),
                children: vec![PresetNode {
                    cid: "A".to_string(),
                    node_type: "branch".to_string(),
                    description: "b".to_string(),
                    ..PresetNode::default()
                }],
                ..PresetNode::default()
            }],
        };
        let diagnostics = preset.validate(&TreeSchema::new());
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_CYCLE));
    }

    #[test]
    fn presets_round_trip_through_json_with_the_python_type_key() {
        let preset = TreePreset::from_nodes(&flat_tree());
        let encoded = serde_json::to_value(&preset).unwrap();
        assert_eq!(encoded["nodes"][0]["type"], "branch");
        assert!(encoded["nodes"][0].get("node_type").is_none());
        let decoded: TreePreset = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, preset);
        let tolerant: TreePreset =
            serde_json::from_value(serde_json::json!({"nodes": [{"unknown": 1}]})).unwrap();
        assert_eq!(tolerant.nodes.len(), 1);
        assert_eq!(tolerant.flatten()[0].cid, "node");
    }

    #[test]
    fn malformed_yaml_is_a_validation_error() {
        for text in [
            "- 1\n- 2\n",
            "nodes: 3\n",
            "other: []\n",
            "nodes: [unclosed\n",
        ] {
            let error = TreePreset::from_yaml(text).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{text}");
            assert!(error.message().starts_with("tree_index.yaml: "), "{text}");
        }
    }

    #[tokio::test]
    async fn from_file_reads_a_preset() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tree_index.yaml");
        tokio::fs::write(&path, "nodes:\n  - description: root\n")
            .await
            .unwrap();
        let preset = TreePreset::from_file(&path).await.unwrap();
        assert_eq!(
            preset.flatten(),
            vec![IndexNode::branch("root").with_description("root")]
        );
        let missing = TreePreset::from_file(directory.path().join("absent.yaml")).await;
        assert_eq!(missing.unwrap_err().kind(), ErrorKind::NotFound);
    }
}
