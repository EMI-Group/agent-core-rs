//! Assembly and repair of the persisted node list.
//!
//! Ports the node-assembly half of Python `openjiuwen.symphony.retrieval.build.tree`
//! (`builder.py`'s tree assembly, `preset_writer.py`'s `flatten_capability_tree`,
//! `rename_leaf_nodes`, `cid_term`/`join_cid`/`unique_child_cid` handling, and
//! `repair.py`'s empty-branch pruning, openJiuwen SDK v0.1.20) as one flat, ordered
//! `Vec<IndexNode>`.
//!
//! # Shapes
//!
//! A built tree is a single flat list — the shape
//! [`IndexNode`](crate::retrieval::types::IndexNode) persists and
//! [`build_tree_root`](crate::retrieval::artifacts::build_tree_root) re-nests — whose
//! `cid`s encode the hierarchy:
//!
//! ```text
//! Skills                                  branch, depth 1 (the root)
//! Skills.documents                        branch, depth 2
//! Skills.documents.pdf-fill               branch, depth 3
//! Skills.documents.pdf-fill.pdffill       leaf,   depth 4, worker_id = "PdfFill"
//! ```
//!
//! Every leaf carries the item's `worker_id`, so a later incremental build can find and
//! replace or drop it ([`worker_node`](crate::retrieval::build::tree::worker_node),
//! [`prune_workers`](crate::retrieval::build::tree::prune_workers)).
//!
//! # Determinism
//!
//! Branch `cid`s come from
//! [`slug_term`](crate::retrieval::build::tree_text::slug_term) of the grouped path
//! segments, leaf `cid`s from `slug_term` of the worker id made unique with
//! [`unique_child_cid`](crate::retrieval::build::tree_text::unique_child_cid) (`-2`,
//! `-3`, …), and assignments are inserted in `(worker_id, path)` order, so two runs over
//! the same items produce byte-identical nodes. The list is always sorted by
//! `(depth, cid)` — parents before their descendants, siblings in `cid` order — after
//! every mutating helper. No clock, no hash-map iteration order.
//!
//! # LLM tree shaping
//!
//! The four opt-in model stages — `one_shot`, `expansion`, `repair` and `equivalence` —
//! live in [`shaping`](crate::retrieval::build::tree::shaping), each with a documented
//! deterministic fallback, so the default build never needs a model and the assembly here
//! stays model-free.
//!
//! # Branch descriptions
//!
//! Python asks the model to write one description per branch
//! (`build/tree/branch_descriptions.py`) and falls back to the branch name. The port
//! always uses the fallback:
//! [`branch_description`](crate::retrieval::build::tree::branch_description) derives a
//! stable sentence from the `cid` path, so a build never needs a model and never embeds
//! prose that could drift between runs.
//!
//! # Visualizer
//!
//! [`visualizer`](crate::retrieval::build::tree::visualizer) renders this node list and
//! its catalog as one `tree_index.html` document for humans. That file is an opt-in extra
//! artifact outside the three-artifact publish contract: a build writes it only when
//! [`BuildOptions::write_html`](crate::retrieval::build::workflows::BuildOptions::write_html)
//! is set, the search runtime never reads it back, and it is purely derived data.

/// Opt-in model-driven tree shaping, with deterministic fallbacks.
pub mod shaping;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::error::{validation, Result};
use crate::retrieval::build::grouping::{
    group_items, GroupAssignment, GroupingConfig, GENERAL_SEGMENT,
};
use crate::retrieval::build::scanner::SourceItem;
use crate::retrieval::build::tree_text::{join_cid, slug_term, sort_nodes, unique_child_cid};
use crate::retrieval::types::{IndexNode, META_DONT_SELECT_WHEN, META_SELECT_WHEN};

/// The opt-in `tree_index.html` visualizer of a built tree.
pub mod visualizer;

/// Builds the full flat node list (ordered `(depth, cid)`) for a set of placed items.
///
/// Every assignment must name a scanned item and every scanned item must have an
/// assignment — a tree that silently dropped items would be a build bug, so a mismatch
/// is `Error::Validation` naming the `assignments` argument. Duplicate worker ids are
/// allowed and matched in list order, so two items sharing a worker id get two leaves
/// (`…tool` and `…tool-2`).
///
/// A blank `root_segment` yields no nodes at all: without a root there is no `cid` space
/// to build in. An empty `items`/`assignments` pair yields the root branch node alone.
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error) naming the `assignments` argument when an
/// assignment has no scanned item, an item has no assignment, or one item is assigned
/// twice.
pub fn build_nodes(
    items: &[SourceItem],
    assignments: &[GroupAssignment],
    root_segment: &str,
) -> Result<Vec<IndexNode>> {
    let root = root_segment.trim();
    if root.is_empty() {
        return Ok(Vec::new());
    }
    // One queue per worker id, in item order, so duplicate ids pair up deterministically.
    let mut pending: BTreeMap<&str, VecDeque<usize>> = BTreeMap::new();
    for (index, item) in items.iter().enumerate() {
        pending
            .entry(item.worker_id.as_str())
            .or_default()
            .push_back(index);
    }
    let mut ordered: Vec<&GroupAssignment> = assignments.iter().collect();
    ordered.sort_by(|left, right| {
        (left.worker_id.as_str(), left.path.as_slice())
            .cmp(&(right.worker_id.as_str(), right.path.as_slice()))
    });

    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut nodes: Vec<IndexNode> = Vec::new();
    ensure_branch(&mut nodes, &mut used, root);
    for assignment in ordered {
        let Some(queue) = pending.get_mut(assignment.worker_id.as_str()) else {
            return Err(validation(
                "assignments",
                format!("worker id \"{}\" has no scanned item", assignment.worker_id),
            ));
        };
        let Some(index) = queue.pop_front() else {
            return Err(validation(
                "assignments",
                format!(
                    "worker id \"{}\" is assigned more than once",
                    assignment.worker_id
                ),
            ));
        };
        let Some(item) = items.get(index) else {
            continue;
        };
        let parent = ensure_chain(&mut nodes, &mut used, &assignment.path, root);
        let leaf = leaf_node(item, &parent, &mut used);
        nodes.push(leaf);
    }
    if let Some((worker_id, _)) = pending.iter().find(|(_, queue)| !queue.is_empty()) {
        return Err(validation(
            "assignments",
            format!("worker id \"{worker_id}\" has no group assignment"),
        ));
    }
    sort_nodes(&mut nodes);
    Ok(nodes)
}

/// Builds the nodes directly (grouping then construction) — the builder's entry point.
///
/// # Errors
///
/// The grouping stage's errors — a deferred
/// [`GroupingStrategy`](crate::retrieval::build::grouping::GroupingStrategy) is
/// [`Error::Unsupported`](crate::error::Error) — plus [`build_nodes`]'s validation.
pub fn build_tree(items: &[SourceItem], config: &GroupingConfig) -> Result<Vec<IndexNode>> {
    let assignments = group_items(items, config)?;
    build_nodes(items, &assignments, &config.root_segment)
}

/// The deterministic, model-free routing description of the branch at `path`.
///
/// Python enriches branch descriptions with a model (`branch_descriptions.py`) and falls
/// back to the branch name; the port derives them from the `cid` path instead, so the
/// sentence is stable across builds and needs no credentials:
///
/// ```
/// use openjiuwen_symphony::retrieval::build::tree::branch_description;
///
/// assert_eq!(branch_description(&["Skills".to_string()]), "Capabilities for Skills.");
/// assert_eq!(branch_description(&[]), "Capability tree root.");
/// ```
#[must_use]
pub fn branch_description(path: &[String]) -> String {
    let segments: Vec<&str> = path
        .iter()
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
        .collect();
    match segments.split_last() {
        None => "Capability tree root.".to_string(),
        Some((last, [])) => format!("Capabilities for {last}."),
        Some((last, prefix)) => format!("Capabilities for {last} (under {}).", prefix.join(" / ")),
    }
}

/// The leaf node for one item under `parent_cid`, with a unique cid from the worker id.
///
/// The leaf's segment is [`slug_term`] of the worker id (`item` when that is blank),
/// disambiguated with [`unique_child_cid`] and recorded in `used`. Its description is the
/// item's own description when non-blank and the item's name otherwise, `source_description`
/// mirrors the item's description, and the routing hints come from the item's metadata keys
/// [`META_SELECT_WHEN`] and [`META_DONT_SELECT_WHEN`] when present.
#[must_use]
pub fn leaf_node(item: &SourceItem, parent_cid: &str, used: &mut BTreeSet<String>) -> IndexNode {
    let cid = unique_child_cid(parent_cid, &slug_term(&item.worker_id, "item"), used);
    let description = if item.description.trim().is_empty() {
        item.name.clone()
    } else {
        item.description.clone()
    };
    let mut node = IndexNode::leaf(cid, item.worker_id.clone())
        .with_description(description)
        .with_source_description(item.description.clone());
    if let Some(value) = item.metadata.get(META_SELECT_WHEN) {
        node = node.with_select_when(value.clone());
    }
    if let Some(value) = item.metadata.get(META_DONT_SELECT_WHEN) {
        node = node.with_dont_select_when(value.clone());
    }
    node
}

/// Inserts one item under `parent_cid`, creating any missing branch, and returns its cid.
///
/// The whole branch chain of `parent_cid` is created when absent (`Skills.documents`
/// ensures `Skills` and `Skills.documents`), so a caller can add an item to a node it has
/// not built yet. A blank `parent_cid` puts the leaf at the top level.
pub fn insert_item(nodes: &mut Vec<IndexNode>, parent_cid: &str, item: &SourceItem) -> String {
    let mut used: BTreeSet<String> = nodes.iter().map(|node| node.cid.clone()).collect();
    let parent = ensure_branch_prefixes(nodes, &mut used, parent_cid.trim());
    let leaf = leaf_node(item, &parent, &mut used);
    let cid = leaf.cid.clone();
    nodes.push(leaf);
    sort_nodes(nodes);
    cid
}

/// Ensures one branch node per segment of `path`, returning the deepest `cid`.
///
/// Segments are slugged with [`slug_term`] (idempotent for an already-slugged path) and
/// blank segments are skipped; an empty `path` returns the root `cid` (the empty string
/// when `root_segment` is blank).
pub fn branch_chain(nodes: &mut Vec<IndexNode>, path: &[String], root_segment: &str) -> String {
    let mut used: BTreeSet<String> = nodes.iter().map(|node| node.cid.clone()).collect();
    let cid = ensure_chain(nodes, &mut used, path, root_segment.trim());
    sort_nodes(nodes);
    cid
}

/// Drops the leaves of `removed_worker_ids`, then the branches they emptied.
///
/// Branches are pruned bottom-up until the list is stable, so a branch left without
/// children is dropped in the same pass as its own emptied parent. Top-level branches
/// (depth 1) are never dropped: an incremental build may prune every item of a root and
/// still keep the root of its tree. The returned `cid`s are the branches that went away,
/// sorted; the removed leaves are not listed because the caller already named them.
pub fn prune_workers(
    nodes: &mut Vec<IndexNode>,
    removed_worker_ids: &BTreeSet<String>,
) -> Vec<String> {
    nodes.retain(|node| !(node.is_leaf() && removed_worker_ids.contains(&node.worker_id)));
    let mut dropped: BTreeSet<String> = BTreeSet::new();
    loop {
        let surviving: Vec<&IndexNode> = nodes
            .iter()
            .filter(|node| !dropped.contains(&node.cid))
            .collect();
        let parents: BTreeSet<&str> = surviving
            .iter()
            .filter_map(|node| node.parent_cid())
            .collect();
        let mut changed = false;
        for node in surviving {
            if node.is_branch() && node.depth() > 1 && !parents.contains(node.cid.as_str()) {
                dropped.insert(node.cid.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    nodes.retain(|node| !dropped.contains(&node.cid));
    sort_nodes(nodes);
    dropped.into_iter().collect()
}

/// `root_cid` itself plus every descendant (dot-prefix match), sorted.
///
/// `root_cid` is reported even when no node has that `cid`, so the answer always describes
/// the subtree the caller asked about. The order is byte order, which keeps a parent
/// ahead of its descendants for `cid`s built from [`join_cid`].
#[must_use]
pub fn subtree_cids(nodes: &[IndexNode], root_cid: &str) -> Vec<String> {
    let prefix = format!("{root_cid}.");
    let mut cids: BTreeSet<String> = BTreeSet::new();
    cids.insert(root_cid.to_string());
    for node in nodes {
        if node.cid.starts_with(&prefix) {
            cids.insert(node.cid.clone());
        }
    }
    cids.into_iter().collect()
}

/// Removes the subtree rooted at `root_cid` and returns how many nodes were dropped.
pub fn remove_subtree(nodes: &mut Vec<IndexNode>, root_cid: &str) -> usize {
    let doomed: BTreeSet<String> = subtree_cids(nodes, root_cid).into_iter().collect();
    let before = nodes.len();
    nodes.retain(|node| !doomed.contains(&node.cid));
    sort_nodes(nodes);
    before - nodes.len()
}

/// Maps every `cid` to its position in `nodes`; rebuild it after a mutation.
///
/// A duplicated `cid` keeps the last position, which is what a caller walking the list
/// forward would see.
#[must_use]
pub fn node_index(nodes: &[IndexNode]) -> BTreeMap<String, usize> {
    nodes
        .iter()
        .enumerate()
        .map(|(position, node)| (node.cid.clone(), position))
        .collect()
}

/// The leaf carrying `worker_id`, if the tree has one.
#[must_use]
pub fn worker_node<'a>(nodes: &'a [IndexNode], worker_id: &str) -> Option<&'a IndexNode> {
    nodes
        .iter()
        .find(|node| node.is_leaf() && node.worker_id == worker_id)
}

/// Creates the branch node for `cid` when it is neither present nor already recorded.
fn ensure_branch(nodes: &mut Vec<IndexNode>, used: &mut BTreeSet<String>, cid: &str) {
    if cid.trim().is_empty() || used.contains(cid) {
        return;
    }
    let path: Vec<String> = cid.split('.').map(str::to_string).collect();
    used.insert(cid.to_string());
    nodes.push(IndexNode::branch(cid).with_description(branch_description(&path)));
}

/// Creates every dot-prefix branch of `cid` and returns `cid`.
fn ensure_branch_prefixes(
    nodes: &mut Vec<IndexNode>,
    used: &mut BTreeSet<String>,
    cid: &str,
) -> String {
    if cid.is_empty() {
        return String::new();
    }
    let mut parent = String::new();
    for segment in cid.split('.') {
        parent = join_cid(&parent, segment);
        let branch = parent.clone();
        ensure_branch(nodes, used, &branch);
    }
    parent
}

/// Ensures the root branch plus one branch per slugged `path` segment.
fn ensure_chain(
    nodes: &mut Vec<IndexNode>,
    used: &mut BTreeSet<String>,
    path: &[String],
    root_segment: &str,
) -> String {
    let mut parent = String::new();
    if !root_segment.is_empty() {
        ensure_branch(nodes, used, root_segment);
        parent = root_segment.to_string();
    }
    for segment in path {
        if segment.trim().is_empty() {
            continue;
        }
        let cid = join_cid(&parent, &slug_term(segment, GENERAL_SEGMENT));
        ensure_branch(nodes, used, &cid);
        parent = cid;
    }
    parent
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::retrieval::build::grouping::DEFAULT_ROOT_SEGMENT;
    use crate::retrieval::types::{NODE_TYPE_BRANCH, NODE_TYPE_LEAF};

    fn item(worker_id: &str, name: &str, classification: &str) -> SourceItem {
        SourceItem {
            worker_id: worker_id.to_string(),
            name: name.to_string(),
            description: format!("{name} description"),
            classification: classification.to_string(),
            item_type: "skill".to_string(),
            ..SourceItem::default()
        }
    }

    fn cids(nodes: &[IndexNode]) -> Vec<&str> {
        nodes.iter().map(|node| node.cid.as_str()).collect()
    }

    fn assignment(worker_id: &str, path: &[&str]) -> GroupAssignment {
        GroupAssignment {
            worker_id: worker_id.to_string(),
            path: path.iter().map(|segment| (*segment).to_string()).collect(),
        }
    }

    #[test]
    fn build_nodes_builds_nested_cids_in_depth_order() {
        let items = vec![
            item("PdfFill", "PDF Fill", "Documents.PDF"),
            item("WebSearch", "Web Search", "Web"),
        ];
        let assignments = vec![
            assignment("PdfFill", &["documents", "pdf"]),
            assignment("WebSearch", &["web"]),
        ];
        let nodes = build_nodes(&items, &assignments, DEFAULT_ROOT_SEGMENT).unwrap();
        assert_eq!(
            cids(&nodes),
            vec![
                "Skills",
                "Skills.documents",
                "Skills.web",
                "Skills.documents.pdf",
                "Skills.web.websearch",
                "Skills.documents.pdf.pdffill",
            ]
        );
        let depths: Vec<usize> = nodes.iter().map(IndexNode::depth).collect();
        let mut sorted = depths.clone();
        sorted.sort_unstable();
        assert_eq!(depths, sorted);
        assert_eq!(nodes[0].node_type, NODE_TYPE_BRANCH);
        let leaf = nodes.last().unwrap();
        assert_eq!(leaf.node_type, NODE_TYPE_LEAF);
        assert_eq!(leaf.worker_id, "PdfFill");
        assert_eq!(leaf.description, "PDF Fill description");
        assert_eq!(leaf.source_description, "PDF Fill description");
        assert_eq!(leaf.parent_cid(), Some("Skills.documents.pdf"));
        assert_eq!(
            nodes[1].description,
            "Capabilities for documents (under Skills)."
        );
    }

    #[test]
    fn leaf_nodes_dedupe_duplicate_worker_ids() {
        let items = vec![item("tool", "One", "docs"), item("tool", "Two", "docs")];
        let assignments = vec![assignment("tool", &["docs"]), assignment("tool", &["docs"])];
        let nodes = build_nodes(&items, &assignments, "Skills").unwrap();
        let leaves: Vec<&IndexNode> = nodes.iter().filter(|node| node.is_leaf()).collect();
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].cid, "Skills.docs.tool");
        assert_eq!(leaves[1].cid, "Skills.docs.tool-2");
        assert_eq!(leaves[0].worker_id, "tool");
        assert_eq!(leaves[1].worker_id, "tool");

        // `leaf_node` suffixes a repeated worker id on the same parent.
        let mut used = BTreeSet::new();
        let first = leaf_node(&items[0], "Skills.docs", &mut used);
        let second = leaf_node(&items[1], "Skills.docs", &mut used);
        assert_eq!(first.cid, "Skills.docs.tool");
        assert_eq!(second.cid, "Skills.docs.tool-2");
        assert_eq!(second.description, "Two description");
    }

    #[test]
    fn leaf_node_carries_metadata_hints_and_description_fallbacks() {
        let mut named = item("w", "Only Name", "docs");
        named.description = String::new();
        named
            .metadata
            .insert(META_SELECT_WHEN.to_string(), "when fuzzy".to_string());
        named
            .metadata
            .insert(META_DONT_SELECT_WHEN.to_string(), "when exact".to_string());
        let mut used = BTreeSet::new();
        let leaf = leaf_node(&named, "Skills", &mut used);
        assert_eq!(leaf.description, "Only Name");
        assert_eq!(leaf.source_description, "");
        assert_eq!(leaf.select_when, "when fuzzy");
        assert_eq!(leaf.dont_select_when, "when exact");
        assert_eq!(leaf.cid, "Skills.w");
    }

    #[test]
    fn build_tree_with_no_items_yields_the_root_branch() {
        let nodes = build_tree(&[], &GroupingConfig::default()).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].cid, DEFAULT_ROOT_SEGMENT);
        assert!(nodes[0].is_branch());
        assert_eq!(nodes[0].description, "Capabilities for Skills.");
        assert!(worker_node(&nodes, "absent").is_none());

        let blank = GroupingConfig::default().with_root_segment("   ");
        assert_eq!(build_tree(&[], &blank).unwrap(), Vec::<IndexNode>::new());
        assert_eq!(
            build_tree(&[], &GroupingConfig::default().with_root_segment("")).unwrap(),
            Vec::<IndexNode>::new()
        );
    }

    #[test]
    fn build_tree_groups_then_builds() {
        let items = vec![
            item("a", "Alpha", "Documents.PDF"),
            item("b", "Beta", "Documents.PDF"),
            item("c", "Gamma", "Web"),
        ];
        let nodes = build_tree(&items, &GroupingConfig::default()).unwrap();
        assert_eq!(
            cids(&nodes),
            vec![
                "Skills",
                "Skills.documents",
                "Skills.web",
                "Skills.documents.pdf",
                "Skills.web.c",
                "Skills.documents.pdf.a",
                "Skills.documents.pdf.b",
            ]
        );
        let node = worker_node(&nodes, "b").unwrap();
        assert_eq!(node.cid, "Skills.documents.pdf.b");
        assert_eq!(node_index(&nodes)["Skills.documents.pdf.b"], 6);
    }

    #[test]
    fn build_nodes_rejects_mismatched_assignments() {
        let items = vec![item("a", "Alpha", "docs")];
        let error = build_nodes(&items, &[assignment("b", &["docs"])], "Skills").unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("has no scanned item"), "{error}");

        let error = build_nodes(&items, &[], "Skills").unwrap_err();
        assert!(
            error.message().contains("has no group assignment"),
            "{error}"
        );

        let duplicates = vec![assignment("a", &["docs"]), assignment("a", &["docs"])];
        let error = build_nodes(&items, &duplicates, "Skills").unwrap_err();
        assert!(
            error.message().contains("assigned more than once"),
            "{error}"
        );
    }

    #[test]
    fn insert_item_creates_the_missing_branch_chain() {
        let mut nodes = Vec::new();
        let cid = insert_item(&mut nodes, "Skills.documents.pdf", &item("w", "W", "docs"));
        assert_eq!(cid, "Skills.documents.pdf.w");
        assert_eq!(
            cids(&nodes),
            vec![
                "Skills",
                "Skills.documents",
                "Skills.documents.pdf",
                "Skills.documents.pdf.w"
            ]
        );
        assert_eq!(
            nodes[2].description,
            "Capabilities for pdf (under Skills / documents)."
        );

        // Inserting again reuses the existing chain and suffixes the cid.
        let again = insert_item(&mut nodes, "Skills.documents.pdf", &item("w", "W", "docs"));
        assert_eq!(again, "Skills.documents.pdf.w-2");
        assert_eq!(nodes.len(), 5);

        let top = insert_item(&mut nodes, "", &item("root-level", "R", "docs"));
        assert_eq!(top, "root-level");
    }

    #[test]
    fn branch_chain_slugs_skips_blanks_and_reuses_branches() {
        let mut nodes = Vec::new();
        let cid = branch_chain(
            &mut nodes,
            &[
                "Documents".to_string(),
                String::new(),
                "PDF Fill".to_string(),
            ],
            "Skills",
        );
        assert_eq!(cid, "Skills.documents.pdf-fill");
        assert_eq!(
            cids(&nodes),
            vec!["Skills", "Skills.documents", "Skills.documents.pdf-fill"]
        );
        assert_eq!(
            branch_chain(&mut nodes, &["documents".to_string()], "Skills"),
            "Skills.documents"
        );
        assert_eq!(nodes.len(), 3);
        assert_eq!(branch_chain(&mut nodes, &[], "Skills"), "Skills");
        let mut parents = Vec::new();
        assert_eq!(
            branch_chain(&mut parents, &["docs".to_string()], "  "),
            "docs"
        );
        assert_eq!(cids(&parents), vec!["docs"]);
    }

    #[test]
    fn prune_workers_drops_empty_branches_until_stable() {
        let items = vec![
            item("a", "Alpha", "Documents.PDF"),
            item("b", "Beta", "Web"),
        ];
        let assignments = vec![
            assignment("a", &["documents", "pdf"]),
            assignment("b", &["web"]),
        ];
        let mut nodes = build_nodes(&items, &assignments, "Skills").unwrap();
        let removed = BTreeSet::from(["a".to_string()]);
        let dropped = prune_workers(&mut nodes, &removed);
        assert_eq!(dropped, vec!["Skills.documents", "Skills.documents.pdf"]);
        assert_eq!(cids(&nodes), vec!["Skills", "Skills.web", "Skills.web.b"]);
        assert!(worker_node(&nodes, "a").is_none());

        // Removing every item keeps the root but drops the branch below it.
        let everything = BTreeSet::from(["a".to_string(), "b".to_string()]);
        let dropped = prune_workers(&mut nodes, &everything);
        assert_eq!(dropped, vec!["Skills.web"]);
        assert_eq!(cids(&nodes), vec!["Skills"]);
        assert!(nodes[0].is_branch());
    }

    #[test]
    fn subtree_cids_and_remove_subtree_cover_descendants() {
        let items = vec![
            item("a", "Alpha", "Documents.PDF"),
            item("b", "Beta", "Documents.Web"),
            item("c", "Gamma", "Data"),
        ];
        let assignments = vec![
            assignment("a", &["documents", "pdf"]),
            assignment("b", &["documents", "web"]),
            assignment("c", &["data"]),
        ];
        let mut nodes = build_nodes(&items, &assignments, "Skills").unwrap();
        assert_eq!(
            subtree_cids(&nodes, "Skills.documents"),
            vec![
                "Skills.documents",
                "Skills.documents.pdf",
                "Skills.documents.pdf.a",
                "Skills.documents.web",
                "Skills.documents.web.b",
            ]
        );
        assert_eq!(subtree_cids(&nodes, "Skills.absent"), vec!["Skills.absent"]);
        assert_eq!(remove_subtree(&mut nodes, "Skills.documents"), 5);
        assert_eq!(cids(&nodes), vec!["Skills", "Skills.data", "Skills.data.c"]);
        assert_eq!(remove_subtree(&mut nodes, "Skills.documents"), 0);
    }

    #[test]
    fn branch_description_derives_from_the_cid_path() {
        let path = vec![
            "Skills".to_string(),
            "Documents".to_string(),
            "PDF Fill".to_string(),
        ];
        assert_eq!(
            branch_description(&path),
            "Capabilities for PDF Fill (under Skills / Documents)."
        );
        assert_eq!(
            branch_description(&["Skills".to_string()]),
            "Capabilities for Skills."
        );
        assert_eq!(
            branch_description(&[String::new()]),
            "Capability tree root."
        );
        assert_eq!(branch_description(&[]), "Capability tree root.");
    }

    #[test]
    fn node_index_and_worker_node_lookups() {
        let nodes = build_tree(
            &[item("a", "Alpha", "docs"), item("b", "Beta", "docs")],
            &GroupingConfig::default(),
        )
        .unwrap();
        let index = node_index(&nodes);
        for (position, node) in nodes.iter().enumerate() {
            assert_eq!(index[&node.cid], position);
        }
        assert!(worker_node(&nodes, "a").unwrap().is_leaf());
        assert!(worker_node(&nodes, "Skills").is_none());
        assert!(worker_node(&nodes, "absent").is_none());
        assert_eq!(
            worker_node(&nodes, "b").unwrap().description,
            "Beta description"
        );
    }

    #[test]
    fn build_tree_respects_the_configured_root_and_fan_out() {
        let items: Vec<SourceItem> = (0..12)
            .map(|index| item(&format!("w{index:02}"), "Name", &format!("c{index:02}")))
            .collect();
        let config = GroupingConfig::default()
            .with_root_segment("Capabilities")
            .with_max_children(3)
            .with_max_depth(4);
        let nodes = build_tree(&items, &config).unwrap();
        assert_eq!(nodes[0].cid, "Capabilities");
        assert_eq!(nodes[0].description, "Capabilities for Capabilities.");
        let mut children: BTreeMap<String, usize> = BTreeMap::new();
        for node in nodes.iter().filter(|node| node.is_branch()) {
            let parent = node.parent_cid().unwrap_or_default();
            *children.entry(parent.to_string()).or_default() += 1;
        }
        for (parent, count) in children {
            assert!(count <= 3, "{parent} has {count} children");
        }
        assert_eq!(nodes.iter().filter(|node| node.is_leaf()).count(), 12);
    }

    #[test]
    fn tree_construction_is_deterministic() {
        let items: Vec<SourceItem> = (0..20)
            .map(|index| {
                item(
                    &format!("w{index:02}"),
                    &format!("Name {index}"),
                    &format!("group-{}", index % 4),
                )
            })
            .collect();
        let config = GroupingConfig::default().with_max_children(3);
        let first = build_tree(&items, &config).unwrap();
        let second = build_tree(&items, &config).unwrap();
        assert_eq!(first, second);
        let mut shuffled = items.clone();
        shuffled.reverse();
        assert_eq!(build_tree(&shuffled, &config).unwrap(), first);
    }
}
