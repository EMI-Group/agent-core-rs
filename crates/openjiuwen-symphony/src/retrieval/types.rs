//! Shared value objects for capability-tree index build and retrieval.
//!
//! Ports the data model of Python `openjiuwen.symphony.retrieval` (v0.1.20):
//! `common/models.py` ([`RetrieverItem`], [`RetrieverNode`], [`RetrieverChoice`],
//! [`RetrieverCandidate`], [`RetrieverTrace`]), `build/models.py` ([`CatalogRecord`] plus
//! the artifact file names) and the build stages' persisted node shape ([`IndexNode`]).
//!
//! Two Rust-only additions live here as well:
//!
//! - [`SearchCursor`] / [`CursorStep`] — a *resumable*, `serde`-serializable depth-first
//!   traversal cursor. Python keeps its search cursor as an opaque object; making the
//!   traversal state a plain value lets a caller stop mid-walk, persist the cursor, and
//!   resume it on another task.
//! - [`IndexManifest::content_hash`] — the SHA-256 of the rendered `catalog.jsonl`, so a
//!   reader can detect a catalog that was rewritten without its manifest.
//!
//! Every ordering here is deterministic: node children and node items are always read
//! sorted by id, map-valued fields are [`BTreeMap`]s, and no output depends on hash-map
//! iteration order or on the wall clock.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The persisted capability tree's file name inside an index directory.
pub const TREE_INDEX_FILENAME: &str = "tree_index.yaml";
/// The persisted catalog's file name inside an index directory (one JSON record per line).
pub const CATALOG_FILENAME: &str = "catalog.jsonl";
/// The index manifest's file name inside an index directory.
pub const MANIFEST_FILENAME: &str = "manifest.json";
/// The human-readable tree rendering's file name (written by the build stage).
pub const TREE_HTML_FILENAME: &str = "tree_index.html";

/// Manifest `mode` for a full index build.
pub const MODE_FULL: &str = "full";
/// Manifest `mode` for an incremental index build.
pub const MODE_INCREMENTAL: &str = "incremental";

/// Tree node `type` for a node that may have children.
pub const NODE_TYPE_BRANCH: &str = "branch";
/// Tree node `type` for a node that carries one catalog item.
pub const NODE_TYPE_LEAF: &str = "leaf";
/// Tree node `type` reserved for the synthetic root.
pub const NODE_TYPE_ROOT: &str = "root";

/// The node id of the synthetic root of an in-memory [`RetrieverNode`] tree.
pub const ROOT_NODE_ID: &str = "ROOT";

/// The manifest `item_type` used when a build does not name one.
pub const DEFAULT_ITEM_TYPE: &str = "skill";

/// Catalog metadata key holding the compacted item content.
pub const META_CONTENT: &str = "content";
/// Catalog metadata key holding the item's own source description.
pub const META_SOURCE_DESCRIPTION: &str = "source_description";
/// Catalog metadata key holding a branch's positive routing hint.
pub const META_SELECT_WHEN: &str = "select_when";
/// Catalog metadata key holding a branch's negative routing hint.
pub const META_DONT_SELECT_WHEN: &str = "dont_select_when";

/// The per-part character limits of a catalog record's `retrieval_text`.
///
/// The order is the order of [`build_retrieval_text`]'s parts: name, description,
/// content, worker id, cid.
pub const RETRIEVAL_TEXT_LIMITS: [usize; 5] = [200, 400, 1200, 120, 200];

/// One retrievable item attached to a [`RetrieverNode`].
///
/// Mirrors Python `RetrieverItem`: `payload` is the item's opaque identity (the persisted
/// `cid` for a skill index), while `item_id` and `label` are what the disclosure renders.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverItem {
    /// The stable item identity inside its node.
    pub item_id: String,
    /// The opaque payload a candidate resolves to (the item's `cid`).
    pub payload: String,
    /// The human-facing label; empty when the item has none.
    pub label: String,
    /// The item's routing description.
    pub description: String,
}

impl RetrieverItem {
    /// Builds an item with `item_id` and `payload`; `label` and `description` stay empty.
    #[must_use]
    pub fn new(item_id: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            payload: payload.into(),
            ..Self::default()
        }
    }

    /// Sets the display label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Sets the routing description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// One node of an in-memory capability tree.
///
/// Mirrors Python `RetrieverNode`: a node either branches into `children`, carries
/// `items`, or does both. Accessors always return sorted views, so a caller that builds
/// children in filesystem order still observes a deterministic tree.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverNode {
    /// The node's identity (its `cid` for a built index, [`ROOT_NODE_ID`] for the root).
    pub node_id: String,
    /// The human-facing label.
    pub label: String,
    /// The node's routing description.
    pub description: String,
    /// Child nodes, in insertion order; read them through [`RetrieverNode::children`].
    pub children: Vec<RetrieverNode>,
    /// Items attached directly to this node; read them through [`RetrieverNode::items`].
    pub items: Vec<RetrieverItem>,
}

impl RetrieverNode {
    /// Builds a childless, itemless node.
    #[must_use]
    pub fn new(node_id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            label: label.into(),
            ..Self::default()
        }
    }

    /// Sets the routing description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Replaces the children.
    #[must_use]
    pub fn with_children(mut self, children: Vec<Self>) -> Self {
        self.children = children;
        self
    }

    /// Replaces the items.
    #[must_use]
    pub fn with_items(mut self, items: Vec<RetrieverItem>) -> Self {
        self.items = items;
        self
    }

    /// Appends one child.
    pub fn push_child(&mut self, child: Self) {
        self.children.push(child);
    }

    /// Appends one item.
    pub fn push_item(&mut self, item: RetrieverItem) {
        self.items.push(item);
    }

    /// The children, always sorted by `node_id` (and then by label, as a tie-break).
    #[must_use]
    pub fn children(&self) -> Vec<&Self> {
        let mut children: Vec<&Self> = self.children.iter().collect();
        children.sort_by(|left, right| {
            left.node_id
                .cmp(&right.node_id)
                .then_with(|| left.label.cmp(&right.label))
        });
        children
    }

    /// The items, always sorted by `item_id` (and then by payload, as a tie-break).
    #[must_use]
    pub fn items(&self) -> Vec<&RetrieverItem> {
        let mut items: Vec<&RetrieverItem> = self.items.iter().collect();
        items.sort_by(|left, right| {
            left.item_id
                .cmp(&right.item_id)
                .then_with(|| left.payload.cmp(&right.payload))
        });
        items
    }

    /// Finds the node with `node_id`, depth-first in [`RetrieverNode::children`] order.
    ///
    /// The first match wins, so a duplicated id resolves to the shallowest-and-leftmost
    /// occurrence.
    #[must_use]
    pub fn find(&self, node_id: &str) -> Option<&Self> {
        if self.node_id == node_id {
            return Some(self);
        }
        self.children()
            .into_iter()
            .find_map(|child| child.find(node_id))
    }

    /// The whole subtree in pre-order: this node first, then each child's walk.
    #[must_use]
    pub fn walk(&self) -> Vec<&Self> {
        let mut nodes = vec![self];
        for child in self.children() {
            nodes.extend(child.walk());
        }
        nodes
    }

    /// The nodes of the subtree that have at least one child, in [`RetrieverNode::walk`] order.
    #[must_use]
    pub fn branches(&self) -> Vec<&Self> {
        self.walk()
            .into_iter()
            .filter(|node| !node.children.is_empty())
            .collect()
    }

    /// The nodes of the subtree with no children, in [`RetrieverNode::walk`] order.
    #[must_use]
    pub fn leaves(&self) -> Vec<&Self> {
        self.walk()
            .into_iter()
            .filter(|node| node.children.is_empty())
            .collect()
    }

    /// The longest path from this node to a childless node; a childless node has depth `0`.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.children
            .iter()
            .map(Self::depth)
            .max()
            .map_or(0, |deepest| deepest + 1)
    }

    /// How many items the subtree carries, this node's own items included.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.items.len() + self.children.iter().map(Self::item_count).sum::<usize>()
    }

    /// The ids of every item in the subtree, sorted.
    #[must_use]
    pub fn item_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .walk()
            .into_iter()
            .flat_map(|node| node.items().into_iter().map(|item| item.item_id.clone()))
            .collect();
        ids.sort();
        ids
    }

    /// Flattens every item of the subtree into one choice per item.
    ///
    /// Choices follow [`RetrieverNode::walk`] order, so they are ordered by node id and
    /// then by item id. `choice_id` is the item's `label` when non-blank, else its
    /// `item_id` — the same identity Python's disclosure renders to a model.
    #[must_use]
    pub fn flatten_choices(&self) -> Vec<RetrieverChoice> {
        self.walk()
            .into_iter()
            .flat_map(|node| {
                node.items()
                    .into_iter()
                    .map(|item| {
                        let choice_id = if item.label.trim().is_empty() {
                            item.item_id.clone()
                        } else {
                            item.label.clone()
                        };
                        RetrieverChoice {
                            choice_id,
                            payload: item.payload.clone(),
                            description: item.description.clone(),
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// One selectable item of a flattened tree, as a model sees it.
///
/// Mirrors Python `RetrieverChoice`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverChoice {
    /// The identity the model selects.
    pub choice_id: String,
    /// The opaque payload the choice resolves to.
    pub payload: String,
    /// The routing description shown next to the choice.
    pub description: String,
}

impl RetrieverChoice {
    /// Builds a choice with `choice_id` and `payload`.
    #[must_use]
    pub fn new(choice_id: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            choice_id: choice_id.into(),
            payload: payload.into(),
            description: String::new(),
        }
    }
}

/// One ranked retrieval result.
///
/// Mirrors Python `RetrieverCandidate`; `rank` is one-based and assigned by the search
/// runtime, and `branch_path` is the ancestor cid chain of the item's node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverCandidate {
    /// The one-based rank of this candidate.
    pub rank: usize,
    /// The item id of the candidate.
    pub item_id: String,
    /// The item payload (its `cid`) a caller resolves.
    pub payload: String,
    /// The ancestor cid chain of the item's node, root first.
    pub branch_path: Vec<String>,
    /// The item's display label.
    pub label: String,
    /// The item's routing description.
    pub description: String,
}

impl RetrieverCandidate {
    /// Builds a candidate with `rank`, `item_id` and `payload`.
    #[must_use]
    pub fn new(rank: usize, item_id: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            rank,
            item_id: item_id.into(),
            payload: payload.into(),
            ..Self::default()
        }
    }
}

/// One recorded traversal step of a retrieval run.
///
/// Mirrors Python `RetrieverTraceEvent`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverTraceEvent {
    /// The event kind (`enter`, `select`, `stop`, …).
    pub event_type: String,
    /// The node the event happened at.
    pub node_id: String,
    /// The traversal depth of the node.
    pub depth: usize,
    /// Event-specific detail; sorted by key so a trace renders deterministically.
    pub detail: BTreeMap<String, Value>,
}

/// An append-only record of a retrieval run.
///
/// Mirrors Python `RetrieverTrace`, without the lock: a Rust caller owns the trace it
/// records into, and shares it only behind its own synchronization.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverTrace {
    /// The events in the order they were recorded.
    pub events: Vec<RetrieverTraceEvent>,
}

impl RetrieverTrace {
    /// Appends one event with `detail`.
    pub fn record(
        &mut self,
        event_type: impl Into<String>,
        node_id: impl Into<String>,
        depth: usize,
        detail: BTreeMap<String, Value>,
    ) {
        self.events.push(RetrieverTraceEvent {
            event_type: event_type.into(),
            node_id: node_id.into(),
            depth,
            detail,
        });
    }

    /// Appends one event with an empty `detail`.
    pub fn record_simple(
        &mut self,
        event_type: impl Into<String>,
        node_id: impl Into<String>,
        depth: usize,
    ) {
        self.record(event_type, node_id, depth, BTreeMap::new());
    }

    /// The recorded events, in call order.
    #[must_use]
    pub fn events(&self) -> &[RetrieverTraceEvent] {
        &self.events
    }

    /// How many events were recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The event types in call order, duplicates kept.
    #[must_use]
    pub fn event_types(&self) -> Vec<&str> {
        self.events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect()
    }

    /// The most recently recorded event.
    #[must_use]
    pub fn last(&self) -> Option<&RetrieverTraceEvent> {
        self.events.last()
    }
}

/// One node to visit while walking a tree.
///
/// `path` is the ancestor node-id chain, root first, **excluding** the node itself.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CursorStep {
    /// The id of the node to visit.
    pub node_id: String,
    /// The traversal depth of the node (the root is depth `0`).
    pub depth: usize,
    /// The ancestor node ids, root first, excluding this node.
    pub path: Vec<String>,
}

/// A resumable depth-first traversal cursor over a [`RetrieverNode`] tree.
///
/// The cursor owns its own stack, so it is a plain value: a caller can stop mid-walk,
/// serialize it, and resume it later (or on another task) as long as it passes the same
/// tree to [`SearchCursor::advance`].
///
/// # Ordering guarantee
///
/// Visits are **pre-order**: a node is yielded before any of its descendants, and a
/// node's children are visited in `node_id` order (the order
/// [`RetrieverNode::children`] reports). Items are not steps — the walk yields nodes
/// only, and a caller reads a yielded node's items itself.
///
/// A step that no longer resolves in the tree (a stale cursor) is skipped; it still
/// counts towards [`SearchCursor::visited`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SearchCursor {
    /// The steps still to visit; the next step is popped from the end.
    pub pending: Vec<CursorStep>,
    /// How many steps have been popped, resolved or not.
    pub visited: usize,
    /// Whether the pending stack is empty and the walk is over.
    pub exhausted: bool,
}

impl SearchCursor {
    /// Seeds a cursor at `root`, which is visited first.
    #[must_use]
    pub fn new(root: &RetrieverNode) -> Self {
        Self {
            pending: vec![CursorStep {
                node_id: root.node_id.clone(),
                depth: 0,
                path: Vec::new(),
            }],
            visited: 0,
            exhausted: false,
        }
    }

    /// Pops the next step, queueing the resolved node's children for later visits.
    ///
    /// Returns `None` once the walk is exhausted. See [`SearchCursor`] for the ordering
    /// guarantee.
    pub fn advance(&mut self, tree: &RetrieverNode) -> Option<CursorStep> {
        while let Some(step) = self.pending.pop() {
            self.visited += 1;
            let Some(node) = tree.find(&step.node_id) else {
                continue;
            };
            let mut path = step.path.clone();
            path.push(step.node_id.clone());
            for child in node.children().into_iter().rev() {
                self.pending.push(CursorStep {
                    node_id: child.node_id.clone(),
                    depth: step.depth + 1,
                    path: path.clone(),
                });
            }
            return Some(step);
        }
        self.exhausted = true;
        None
    }

    /// How many steps were popped so far.
    #[must_use]
    pub fn visited(&self) -> usize {
        self.visited
    }

    /// How many steps are still queued.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the walk reached its end.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

/// One persisted node of `tree_index.yaml`.
///
/// Ports the Python build's node dictionaries: `cid` is the dot-joined path
/// (`Skills.foo.bar`), `type` is [`NODE_TYPE_BRANCH`] or [`NODE_TYPE_LEAF`], and a leaf
/// names the `worker_id` of the item it carries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IndexNode {
    /// The node's dot-joined path identity.
    pub cid: String,
    /// The node type; serialized as `type` to keep the Python artifact shape.
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
}

impl IndexNode {
    /// Builds a node with `cid` and `node_type`.
    #[must_use]
    pub fn new(cid: impl Into<String>, node_type: impl Into<String>) -> Self {
        Self {
            cid: cid.into(),
            node_type: node_type.into(),
            ..Self::default()
        }
    }

    /// Builds a branch node.
    #[must_use]
    pub fn branch(cid: impl Into<String>) -> Self {
        Self::new(cid, NODE_TYPE_BRANCH)
    }

    /// Builds a leaf node carrying `worker_id`.
    #[must_use]
    pub fn leaf(cid: impl Into<String>, worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            ..Self::new(cid, NODE_TYPE_LEAF)
        }
    }

    /// Sets the routing description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the positive routing hint.
    #[must_use]
    pub fn with_select_when(mut self, select_when: impl Into<String>) -> Self {
        self.select_when = select_when.into();
        self
    }

    /// Sets the negative routing hint.
    #[must_use]
    pub fn with_dont_select_when(mut self, dont_select_when: impl Into<String>) -> Self {
        self.dont_select_when = dont_select_when.into();
        self
    }

    /// Sets the source description the node was derived from.
    #[must_use]
    pub fn with_source_description(mut self, source_description: impl Into<String>) -> Self {
        self.source_description = source_description.into();
        self
    }

    /// Whether the node type is [`NODE_TYPE_BRANCH`] (trimmed, ASCII case-insensitive).
    #[must_use]
    pub fn is_branch(&self) -> bool {
        self.node_type.trim().eq_ignore_ascii_case(NODE_TYPE_BRANCH)
    }

    /// Whether the node type is [`NODE_TYPE_LEAF`] (trimmed, ASCII case-insensitive).
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.node_type.trim().eq_ignore_ascii_case(NODE_TYPE_LEAF)
    }

    /// The node's `cid` segments; empty for a blank `cid`.
    #[must_use]
    pub fn segments(&self) -> Vec<&str> {
        if self.cid.trim().is_empty() {
            return Vec::new();
        }
        self.cid.split('.').collect()
    }

    /// How many `.`-separated segments the `cid` has; `0` for a blank `cid`.
    ///
    /// Python's `sort_tree_nodes` measures depth the same way, so ordering by
    /// `(depth, cid)` puts parents before their descendants.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.segments().len()
    }

    /// The parent `cid` of this node, or `None` when the `cid` has no `.` separator.
    #[must_use]
    pub fn parent_cid(&self) -> Option<&str> {
        self.cid.rfind('.').map(|index| &self.cid[..index])
    }
}

/// The `manifest.json` document of one built index.
///
/// Ports Python `write_manifest`'s fields (`mode`, `count`, `item_paths`, `worker_ids`
/// and the optional `item_type`) and adds one Rust-only field: `content_hash`, the
/// SHA-256 hex digest of the rendered `catalog.jsonl`. A reader compares it with
/// [`catalog_content_hash`](crate::retrieval::artifacts::catalog_content_hash) to detect a
/// catalog rewritten without its manifest; an empty value means "not recorded", which is
/// what a manifest written by the Python build looks like.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct IndexManifest {
    /// The build mode: [`MODE_FULL`] or [`MODE_INCREMENTAL`].
    pub mode: String,
    /// How many catalog records the index holds.
    pub count: usize,
    /// The scanned item paths, as the build resolved them.
    pub item_paths: Vec<String>,
    /// The worker ids of the catalog records, in catalog order.
    pub worker_ids: Vec<String>,
    /// What one catalog record describes; [`DEFAULT_ITEM_TYPE`] unless a build overrides it.
    pub item_type: String,
    /// The SHA-256 hex digest of the rendered catalog; empty when not recorded.
    pub content_hash: String,
}

impl IndexManifest {
    /// Builds a manifest for `mode` and `item_type` with no items yet.
    #[must_use]
    pub fn new(mode: impl Into<String>, item_type: impl Into<String>) -> Self {
        Self {
            mode: mode.into(),
            item_type: item_type.into(),
            ..Self::default()
        }
    }
}

/// One catalog item of `catalog.jsonl`.
///
/// Ports Python `CatalogRecord`: the item's identity (`worker_id`, `cid`, `name`), where
/// it came from (`skill_path`), its position in the tree (`branch_path`, `category`), and
/// the text the search runtime scores (`retrieval_text` plus free-form `metadata`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CatalogRecord {
    /// The worker id of the item.
    pub worker_id: String,
    /// The leaf `cid` the item is attached to.
    pub cid: String,
    /// The item's display name.
    pub name: String,
    /// The item's description.
    pub description: String,
    /// Where the item was loaded from (a resolved path or a pass-through URI).
    pub skill_path: String,
    /// The ancestor `cid` segments, root first, excluding the leaf's own segment.
    pub branch_path: Vec<String>,
    /// The ancestor segments dot-joined; empty when the `cid` has no dot.
    pub category: String,
    /// The compacted text the search runtime scores.
    pub retrieval_text: String,
    /// Free-form metadata; sorted by key so the rendered line is deterministic.
    pub metadata: BTreeMap<String, String>,
}

impl CatalogRecord {
    /// Builds a record for `worker_id` at `cid`, deriving `branch_path` and `category`.
    ///
    /// Every `cid` segment except the last becomes a `branch_path` entry, and `category`
    /// is those segments dot-joined — empty when the `cid` has no `.` separator.
    #[must_use]
    pub fn new(worker_id: impl Into<String>, cid: impl Into<String>) -> Self {
        let cid = cid.into();
        let mut branch_path: Vec<String> = Vec::new();
        if let Some(index) = cid.rfind('.') {
            branch_path = cid[..index].split('.').map(str::to_string).collect();
        }
        let category = branch_path.join(".");
        Self {
            worker_id: worker_id.into(),
            cid,
            branch_path,
            category,
            ..Self::default()
        }
    }

    /// Sets the display name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets where the item was loaded from.
    #[must_use]
    pub fn with_skill_path(mut self, skill_path: impl Into<String>) -> Self {
        self.skill_path = skill_path.into();
        self
    }

    /// Sets the text the search runtime scores.
    #[must_use]
    pub fn with_retrieval_text(mut self, retrieval_text: impl Into<String>) -> Self {
        self.retrieval_text = retrieval_text.into();
        self
    }

    /// Replaces the metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    /// The metadata value for `key`, or `""` when the key is absent.
    #[must_use]
    pub fn metadata_of(&self, key: &str) -> &str {
        self.metadata.get(key).map_or("", String::as_str)
    }
}

/// A loaded index: the in-memory tree plus everything a retrieval run needs.
///
/// Produced by
/// [`load_index`](crate::retrieval::artifacts::load_index) and by a build's write path.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LoadedIndex {
    /// The root of the in-memory tree, labelled [`ROOT_NODE_ID`].
    pub root: RetrieverNode,
    /// Every item of the tree as a flat choice list, in tree order.
    pub choices: Vec<RetrieverChoice>,
    /// The catalog records, sorted by `cid`.
    pub records: Vec<CatalogRecord>,
    /// The index manifest.
    pub manifest: IndexManifest,
    /// Each item payload (its `cid`) mapped to its ancestor `cid` path, the synthetic
    /// [`ROOT_NODE_ID`] excluded.
    pub item_paths: BTreeMap<String, Vec<String>>,
}

/// Collapses whitespace and truncates `text` to `limit` characters.
///
/// Mirrors Python `compact_text`: every whitespace run becomes one space, the result is
/// trimmed, and a value longer than `limit` characters keeps its first `limit - 1`
/// characters (trailing whitespace trimmed) plus `"..."`. Truncation is character based,
/// so a multi-byte character is never split.
#[must_use]
pub fn compact_text(text: &str, limit: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= limit {
        return normalized;
    }
    let kept: String = normalized.chars().take(limit.saturating_sub(1)).collect();
    format!("{}...", kept.trim_end())
}

/// Builds a catalog record's `retrieval_text` from its five parts.
///
/// Mirrors Python `build_retrieval_text`: each part is compacted with its own
/// [`RETRIEVAL_TEXT_LIMITS`] entry (name, description, content, worker id, `cid`), empty
/// parts are dropped, and the rest are joined with `"\n"`.
#[must_use]
pub fn build_retrieval_text(
    name: &str,
    description: &str,
    content: &str,
    worker_id: &str,
    cid: &str,
) -> String {
    let parts = [
        compact_text(name, RETRIEVAL_TEXT_LIMITS[0]),
        compact_text(description, RETRIEVAL_TEXT_LIMITS[1]),
        compact_text(content, RETRIEVAL_TEXT_LIMITS[2]),
        compact_text(worker_id, RETRIEVAL_TEXT_LIMITS[3]),
        compact_text(cid, RETRIEVAL_TEXT_LIMITS[4]),
    ];
    parts
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn tree() -> RetrieverNode {
        RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![
            RetrieverNode::new("B", "B")
                .with_items(vec![
                    RetrieverItem::new("b2", "B.b2").with_label("Bee Two"),
                    RetrieverItem::new("b1", "B.b1"),
                ])
                .with_children(vec![
                    RetrieverNode::new("B.z", "z"),
                    RetrieverNode::new("B.a", "a").with_items(vec![RetrieverItem::new(
                        "deep", "B.a.deep",
                    )
                    .with_description("routed")
                    .with_label("Deep")]),
                ]),
            RetrieverNode::new("A", "A").with_items(vec![RetrieverItem::new("a1", "A.a1")]),
        ])
    }

    #[test]
    fn compact_text_collapses_whitespace_and_keeps_short_text() {
        assert_eq!(
            compact_text("  alpha \n beta \t gamma ", 40),
            "alpha beta gamma"
        );
        assert_eq!(compact_text("", 10), "");
    }

    #[test]
    fn compact_text_truncates_on_char_boundaries() {
        let text = "能力树索引构建";
        let compacted = compact_text(text, 4);
        assert_eq!(compacted, "能力树...");
        assert_eq!(compacted.chars().count(), 6);
        assert!(text.starts_with(&compacted[..compacted.len() - 3]));
    }

    #[test]
    fn compact_text_trims_the_kept_tail_and_handles_zero() {
        assert_eq!(compact_text("abc defgh", 5), "abc...");
        assert_eq!(compact_text("anything", 0), "...");
        assert_eq!(compact_text("abc", 3), "abc");
    }

    #[test]
    fn build_retrieval_text_uses_the_five_limits_and_drops_empty_parts() {
        let long_description = "d".repeat(500);
        let text = build_retrieval_text("name", &long_description, "", "worker", "Skills.leaf");
        let parts: Vec<&str> = text.split('\n').collect();
        assert_eq!(parts.len(), 4, "the empty content part is dropped");
        assert_eq!(parts[0], "name");
        assert_eq!(parts[1].chars().count(), 402);
        assert!(parts[1].ends_with("..."));
        assert_eq!(parts[2], "worker");
        assert_eq!(parts[3], "Skills.leaf");
        assert_eq!(build_retrieval_text("", "", "", "", ""), "");
    }

    #[test]
    fn children_and_items_are_read_sorted() {
        let tree = tree();
        let ids: Vec<&str> = tree
            .children()
            .iter()
            .map(|node| node.node_id.as_str())
            .collect();
        assert_eq!(ids, vec!["A", "B"]);
        let branch = tree.find("B").unwrap();
        let items: Vec<&str> = branch
            .items()
            .iter()
            .map(|item| item.item_id.as_str())
            .collect();
        assert_eq!(items, vec!["b1", "b2"]);
    }

    #[test]
    fn walk_is_pre_order_and_depth_counts_edges() {
        let tree = tree();
        let walked: Vec<&str> = tree
            .walk()
            .iter()
            .map(|node| node.node_id.as_str())
            .collect();
        assert_eq!(walked, vec!["ROOT", "A", "B", "B.a", "B.z"]);
        assert_eq!(tree.depth(), 2);
        assert_eq!(tree.find("B.a").unwrap().depth(), 0);
        assert_eq!(tree.item_count(), 4);
        assert_eq!(tree.item_ids(), vec!["a1", "b1", "b2", "deep"]);
    }

    #[test]
    fn branches_and_leaves_split_the_subtree() {
        let tree = tree();
        let branches: Vec<&str> = tree
            .branches()
            .iter()
            .map(|node| node.node_id.as_str())
            .collect();
        assert_eq!(branches, vec!["ROOT", "B"]);
        let leaves: Vec<&str> = tree
            .leaves()
            .iter()
            .map(|node| node.node_id.as_str())
            .collect();
        assert_eq!(leaves, vec!["A", "B.a", "B.z"]);
        assert!(tree.find("missing").is_none());
    }

    #[test]
    fn flatten_choices_prefers_the_label_and_keeps_tree_order() {
        let choices = tree().flatten_choices();
        let ids: Vec<&str> = choices
            .iter()
            .map(|choice| choice.choice_id.as_str())
            .collect();
        assert_eq!(ids, vec!["a1", "b1", "Bee Two", "Deep"]);
        let payloads: Vec<&str> = choices
            .iter()
            .map(|choice| choice.payload.as_str())
            .collect();
        assert_eq!(payloads, vec!["A.a1", "B.b1", "B.b2", "B.a.deep"]);
        assert_eq!(choices[3].description, "routed");
    }

    #[test]
    fn push_helpers_append_and_with_helpers_replace() {
        let mut node = RetrieverNode::new("n", "label")
            .with_description("d")
            .with_children(vec![RetrieverNode::new("x", "x")])
            .with_items(vec![RetrieverItem::new("i", "p")]);
        node.push_child(RetrieverNode::new("y", "y"));
        node.push_item(RetrieverItem::new("j", "q").with_description("j d"));
        assert_eq!(node.children().len(), 2);
        assert_eq!(node.items().len(), 2);
        assert_eq!(node.description, "d");
        assert_eq!(node.items()[1].description, "j d");
    }

    #[test]
    fn cursor_walks_the_tree_in_pre_order() {
        let tree = tree();
        let mut cursor = SearchCursor::new(&tree);
        let mut visited = Vec::new();
        while let Some(step) = cursor.advance(&tree) {
            visited.push((step.node_id, step.depth, step.path));
        }
        let ids: Vec<&str> = visited.iter().map(|(id, _, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["ROOT", "A", "B", "B.a", "B.z"]);
        assert_eq!(visited[0].1, 0);
        assert_eq!(visited[0].2, Vec::<String>::new());
        assert_eq!(visited[2].2, vec![ROOT_NODE_ID.to_string()]);
        assert_eq!(
            visited[3].2,
            vec![ROOT_NODE_ID.to_string(), "B".to_string()]
        );
        assert_eq!(cursor.visited(), 5);
        assert!(cursor.is_exhausted());
        assert_eq!(cursor.pending_len(), 0);
        assert!(cursor.advance(&tree).is_none());
    }

    #[test]
    fn cursor_resumes_from_its_serialized_state() {
        let tree = tree();
        let mut cursor = SearchCursor::new(&tree);
        assert!(cursor.advance(&tree).is_some());
        let encoded = serde_json::to_string(&cursor).unwrap();
        let mut resumed: SearchCursor = serde_json::from_str(&encoded).unwrap();
        let mut remaining = Vec::new();
        while let Some(step) = resumed.advance(&tree) {
            remaining.push(step.node_id);
        }
        assert_eq!(remaining, vec!["A", "B", "B.a", "B.z"]);
        assert_eq!(resumed.visited(), 5);
    }

    #[test]
    fn cursor_skips_a_stale_step_but_counts_it() {
        let tree = RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID);
        let mut cursor = SearchCursor::new(&tree);
        cursor.pending.push(CursorStep {
            node_id: "ghost".to_string(),
            depth: 1,
            path: vec![ROOT_NODE_ID.to_string()],
        });
        let steps: Vec<String> = std::iter::from_fn(|| cursor.advance(&tree))
            .map(|step| step.node_id)
            .collect();
        assert_eq!(steps, vec![ROOT_NODE_ID.to_string()]);
        assert_eq!(cursor.visited(), 2);
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn index_node_derives_depth_segments_and_parent() {
        let node = IndexNode::branch("Skills.foo.bar");
        assert_eq!(node.segments(), vec!["Skills", "foo", "bar"]);
        assert_eq!(node.depth(), 3);
        assert_eq!(node.parent_cid(), Some("Skills.foo"));
        assert!(node.is_branch());
        assert!(!node.is_leaf());
        assert_eq!(IndexNode::leaf("A", "w").worker_id, "w");
        let flat = IndexNode::new("Skills", " Branch ");
        assert!(flat.is_branch());
        assert_eq!(flat.parent_cid(), None);
        assert_eq!(IndexNode::default().depth(), 0);
    }

    #[test]
    fn index_node_serializes_type_as_the_python_key() {
        let node = IndexNode::leaf("Skills.x", "worker")
            .with_description("d")
            .with_select_when("s")
            .with_dont_select_when("n")
            .with_source_description("src");
        let value = serde_json::to_value(&node).unwrap();
        assert_eq!(value["type"], "leaf");
        assert!(value.get("node_type").is_none());
        let parsed: IndexNode = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, node);
    }

    #[test]
    fn catalog_record_derives_branch_path_and_category() {
        let record = CatalogRecord::new("worker", "Skills.foo.bar")
            .with_name("n")
            .with_description("d")
            .with_skill_path("/tmp/skill")
            .with_retrieval_text("text")
            .with_metadata(BTreeMap::from([("k".to_string(), "v".to_string())]));
        assert_eq!(record.branch_path, vec!["Skills", "foo"]);
        assert_eq!(record.category, "Skills.foo");
        assert_eq!(record.metadata_of("k"), "v");
        assert_eq!(record.metadata_of("absent"), "");
        let flat = CatalogRecord::new("worker", "Skills");
        assert_eq!(flat.branch_path, Vec::<String>::new());
        assert_eq!(flat.category, "");
    }

    #[test]
    fn trace_records_in_call_order() {
        let mut trace = RetrieverTrace::default();
        assert_eq!(trace, RetrieverTrace::default());
        trace.record_simple("enter", ROOT_NODE_ID, 0);
        trace.record(
            "select",
            "A",
            1,
            BTreeMap::from([("reason".to_string(), Value::from("lexical"))]),
        );
        assert_eq!(trace.len(), 2);
        assert_eq!(trace.event_types(), vec!["enter", "select"]);
        assert_eq!(trace.events()[0].detail.len(), 0);
        assert_eq!(trace.last().unwrap().node_id, "A");
        assert_eq!(trace.events()[1].detail["reason"], "lexical");
    }

    #[test]
    fn value_types_round_trip_through_json() {
        let loaded = LoadedIndex {
            root: tree(),
            choices: tree().flatten_choices(),
            records: vec![CatalogRecord::new("w", "A.a1")],
            manifest: IndexManifest::new(MODE_FULL, DEFAULT_ITEM_TYPE),
            item_paths: BTreeMap::from([("A.a1".to_string(), vec!["A".to_string()])]),
        };
        let encoded = serde_json::to_string(&loaded).unwrap();
        let decoded: LoadedIndex = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, loaded);
        let tolerant: LoadedIndex =
            serde_json::from_str(r#"{"root": {"node_id": "ROOT", "unknown": 1}}"#).unwrap();
        assert_eq!(tolerant.root.node_id, ROOT_NODE_ID);
        assert_eq!(tolerant.records, Vec::<CatalogRecord>::new());
        let candidate = RetrieverCandidate::new(2, "item", "payload");
        assert_eq!(candidate.rank, 2);
        assert_eq!(RetrieverChoice::new("c", "p").payload, "p");
    }
}
