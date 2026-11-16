//! Python flow value objects for the query-time search runtime.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.runtime.types` (v0.1.20) plus the two
//! exposure value objects it imports from `render/disclosure.py`
//! ([`ExposedNode`] and
//! [`SelectableResolution`]).
//!
//! Everything here is plain data: no model, no I/O, no traversal. The later waves
//! (rendering, traversal protocols, engine) build these values and hand them on.
//!
//! # Rust-only adjustments
//!
//! - [`SearchCursor`] names its node by id
//!   (`node_id`) instead of holding the [`RetrieverNode`] itself, so a cursor is
//!   serializable and cheap to copy; [`SearchCursor::resolve`] turns it back into a node
//!   reference against the tree the caller owns.
//! - [`SelectableResolution`] keeps
//!   `node_id` (Python keeps the whole `RetrieverNode`) and the [`RetrieverItem`] it
//!   resolves to.
//! - [`ExposedFragment`] adds
//!   `rendered_chars` / `truncated` — the bookkeeping of the Rust-only rendered-fragment
//!   char budget — and drops Python's `selectable_nodes` / `score_key` / `token_id`
//!   fields, which no ported stage reads. A code's score key is the code itself.
//!
//! Every collection is a `Vec` or a `BTreeMap`, so serialization order is deterministic.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::retrieval::types::{RetrieverCandidate, RetrieverItem, RetrieverNode};

/// The reserved abstain answer: never assigned as a candidate code.
pub const ABSTAIN_TOKEN: &str = "0";

/// One chat message of a disclosure prompt.
///
/// Mirrors Python's `{"role": ..., "content": ...}` message dictionaries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ChatMessage {
    /// The message role (`system`, `user`, …).
    pub role: String,
    /// The message text.
    pub content: String,
}

impl ChatMessage {
    /// Builds a message with `role` and `content`.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Builds a `system` message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Builds a `user` message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
}

/// The recursion cursor of a progressive search.
///
/// Mirrors Python `SearchCursor`: `branch_path` is the id chain that reached
/// `node_id` and always starts with the root's id, so `depth == branch_path.len() - 1`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SearchCursor {
    /// The id of the node this cursor points at.
    pub node_id: String,
    /// How deep the node sits below the root (the root is depth `0`).
    pub depth: usize,
    /// The id chain that reached the node, root id first and including the node's own id.
    pub branch_path: Vec<String>,
    /// How many results this subtree may contribute.
    pub top_k: usize,
}

impl SearchCursor {
    /// Seeds a cursor at `root`.
    ///
    /// The branch path is `[root.node_id]` and the depth is `0`, matching Python's
    /// `SearchCursor(node=root, depth=0, branch_path=(root.node_id,), top_k=…)`.
    #[must_use]
    pub fn new(root: &RetrieverNode, top_k: usize) -> Self {
        Self {
            node_id: root.node_id.clone(),
            depth: 0,
            branch_path: vec![root.node_id.clone()],
            top_k,
        }
    }

    /// Builds a cursor for an already-known node position.
    #[must_use]
    pub fn at(
        node_id: impl Into<String>,
        depth: usize,
        branch_path: Vec<String>,
        top_k: usize,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            depth,
            branch_path,
            top_k,
        }
    }

    /// Resolves the cursor's node inside `root`, or `None` when the id is not in the tree.
    #[must_use]
    pub fn resolve<'a>(&self, root: &'a RetrieverNode) -> Option<&'a RetrieverNode> {
        root.find(&self.node_id)
    }
}

/// One rendered tree node of an exposed fragment.
///
/// Mirrors Python `ExposedNode`: `canonical_id` is `node::<node_id>` for a branch and
/// `item::<payload>` for an item, and `is_selectable` marks the nodes a model may pick.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExposedNode {
    /// The canonical identity of the exposed node.
    pub canonical_id: String,
    /// The label rendered for the node.
    pub label: String,
    /// The description rendered for the node.
    pub description: String,
    /// Whether a model may select this node.
    pub is_selectable: bool,
    /// The canonical id of the selectable node this entry stands for, when it is not
    /// itself selectable.
    pub selectable_canonical_id: Option<String>,
    /// The exposed children, in rendering order.
    pub children: Vec<Self>,
}

/// One selectable option of an exposed fragment.
///
/// Mirrors Python `SelectableResolution`: `code` is the short identifier a model answers
/// with, `canonical_id` the exposure identity it resolves to, and `item` the
/// [`RetrieverItem`] a terminal resolution stands for.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SelectableResolution {
    /// The code the model selects (never [`ABSTAIN_TOKEN`]).
    pub code: String,
    /// The canonical exposure identity this code resolves to.
    pub canonical_id: String,
    /// The display name rendered next to the code.
    pub display_name: String,
    /// The node or item label.
    pub label: String,
    /// The node or item description.
    pub description: String,
    /// Whether selecting the code yields items instead of deeper branches.
    pub is_terminal: bool,
    /// The id chain that reached the resolution, root id first.
    pub branch_path: Vec<String>,
    /// The tree node the resolution stands for, when it is a branch.
    pub node_id: Option<String>,
    /// The item the resolution stands for, when it is terminal.
    pub item: Option<RetrieverItem>,
}

/// The rendered disclosure of one subtree.
///
/// Mirrors Python `ExposedFragment`, minus the prompt-building conveniences (a later wave
/// owns the prompt bank) and plus the Rust-only
/// `rendered_chars` / `truncated` char-budget bookkeeping.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExposedFragment {
    /// The exposed tree, rooted at the (possibly spliced) subtree root.
    pub root: ExposedNode,
    /// The rendered candidate block a prompt embeds.
    pub rendered_tree: String,
    /// Every code of this fragment mapped to the resolution it selects.
    pub code_to_resolution: BTreeMap<String, SelectableResolution>,
    /// The assigned codes in assignment order; the prompt's candidate order.
    pub candidate_codes: Vec<String>,
    /// A stable fingerprint of the exposure, used as a cache id.
    pub fragment_fingerprint: String,
    /// The width used to render codes (`0` when the codes have mixed widths).
    pub code_width: usize,
    /// Whether the compact code book (instead of numeric codes) was used.
    pub compact_codes_enabled: bool,
    /// Whether the exposure is a flat list instead of a nested tree.
    pub flat_list_mode: bool,
    /// How many characters [`ExposedFragment::rendered_tree`] holds.
    pub rendered_chars: usize,
    /// Whether the rendered fragment was cut to fit its char budget.
    pub truncated: bool,
}

impl ExposedFragment {
    /// The resolution `code` selects, or `None` when the code is not a candidate.
    #[must_use]
    pub fn resolution(&self, code: &str) -> Option<&SelectableResolution> {
        self.code_to_resolution.get(code)
    }

    /// The resolutions in [`ExposedFragment::candidate_codes`] order.
    ///
    /// A code without a resolution is skipped, so the result always describes selectable
    /// options.
    #[must_use]
    pub fn resolutions(&self) -> Vec<&SelectableResolution> {
        self.candidate_codes
            .iter()
            .filter_map(|code| self.code_to_resolution.get(code))
            .collect()
    }

    /// How many candidate codes the fragment carries.
    #[must_use]
    pub fn selectable_count(&self) -> usize {
        self.candidate_codes.len()
    }

    /// Whether the fragment exposes nothing selectable.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.candidate_codes.is_empty()
    }
}

/// A selected resolution, as the expansion stage consumes it.
///
/// Mirrors Python `SelectableTarget`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SelectableTarget {
    /// The resolution the model selected.
    pub resolution: SelectableResolution,
}

impl SelectableTarget {
    /// Wraps `resolution` as a target.
    #[must_use]
    pub fn new(resolution: SelectableResolution) -> Self {
        Self { resolution }
    }

    /// Whether the target yields items (rather than deeper branches).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.resolution.is_terminal
    }

    /// The id chain that reached the target, root id first.
    #[must_use]
    pub fn branch_path(&self) -> &[String] {
        &self.resolution.branch_path
    }

    /// The code the target was selected by.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.resolution.code
    }

    /// The canonical exposure identity of the target.
    #[must_use]
    pub fn canonical_id(&self) -> &str {
        &self.resolution.canonical_id
    }
}

/// The subtree currently exposed to a model, with its selectable targets.
///
/// Mirrors Python `CurrentSubtree`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CurrentSubtree {
    /// The cursor the subtree was exposed for.
    pub cursor: SearchCursor,
    /// The rendered disclosure.
    pub fragment: ExposedFragment,
    /// The selectable targets, in [`ExposedFragment::candidate_codes`] order.
    pub selectable_targets: Vec<SelectableTarget>,
}

/// The selection protocol a response is parsed with.
///
/// Mirrors Python `SelectionProtocol`; [`SelectionProtocol::default`] carries
/// [`ABSTAIN_TOKEN`] as its abstain token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SelectionProtocol {
    /// Whether the candidate codes come from the compact code book.
    pub compact_codes_enabled: bool,
    /// The codes a model may answer with, in candidate order.
    pub candidate_codes: Vec<String>,
    /// The code width, or `0` when the codes have mixed widths.
    pub code_width: usize,
    /// The token that means "nothing here is relevant".
    pub abstain_token: String,
}

impl Default for SelectionProtocol {
    fn default() -> Self {
        Self {
            compact_codes_enabled: false,
            candidate_codes: Vec::new(),
            code_width: 0,
            abstain_token: ABSTAIN_TOKEN.to_owned(),
        }
    }
}

/// Everything a prompt renderer needs: the fragment, its protocol and the messages.
///
/// Mirrors Python `PromptBundle`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptBundle {
    /// The rendered disclosure.
    pub fragment: ExposedFragment,
    /// How the model's answer is to be read.
    pub protocol: SelectionProtocol,
    /// The ready-to-send chat messages.
    pub messages: Vec<ChatMessage>,
}

/// How one selection (or one whole search) was answered.
///
/// Rust-only provenance marker: a caller that never reads a log line or a trace event can
/// read it off a [`SelectionResult`], a
/// [`ProgressiveSearchResult`](crate::retrieval::search::progressive::ProgressiveSearchResult)
/// or the facade's [`RetrievalResult`](crate::retrieval::search::service::RetrievalResult) to
/// learn whether the configured model actually produced the selection.
/// The degraded variants are the ones where it did not: [`SelectionProvenance::LexicalNoModel`],
/// [`SelectionProvenance::LexicalAfterModelError`] and [`SelectionProvenance::UnusableReply`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SelectionProvenance {
    /// The configured model answered.
    Model,
    /// The lexical scorer answered because
    /// [`SelectionMode::Lexical`](crate::retrieval::search::config::SelectionMode) asked for it
    /// (or because no selection was made at all).
    #[default]
    Lexical,
    /// [`SelectionMode::Generate`](crate::retrieval::search::config::SelectionMode) was
    /// configured but no model is wired, so the lexical scorer answered as a documented
    /// degradation (`MODE_LEXICAL_NO_MODEL`).
    LexicalNoModel,
    /// The model call or its reply failed and the fallback selector answered.
    LexicalAfterModelError,
    /// The embedding blend answered; no model is involved.
    EmbeddingBlend,
    /// A node with exactly one selectable target was taken without any selection.
    SingleSelectable,
    /// The model answered, but no code in its reply could be parsed, so the node selected
    /// nothing.
    UnusableReply,
}

impl SelectionProvenance {
    /// The `snake_case` name the trace event and the provenance fields carry.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Lexical => "lexical",
            Self::LexicalNoModel => "lexical_no_model",
            Self::LexicalAfterModelError => "lexical_after_model_error",
            Self::EmbeddingBlend => "embedding_blend",
            Self::SingleSelectable => "single_selectable",
            Self::UnusableReply => "unusable_reply",
        }
    }

    /// Parses [`SelectionProvenance::as_str`]; `None` for an unknown name.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "model" => Some(Self::Model),
            "lexical" => Some(Self::Lexical),
            "lexical_no_model" => Some(Self::LexicalNoModel),
            "lexical_after_model_error" => Some(Self::LexicalAfterModelError),
            "embedding_blend" => Some(Self::EmbeddingBlend),
            "single_selectable" => Some(Self::SingleSelectable),
            "unusable_reply" => Some(Self::UnusableReply),
            _ => None,
        }
    }

    /// Whether the selection is a degradation: the configured model did not produce it.
    ///
    /// [`SelectionProvenance::Model`], [`SelectionProvenance::Lexical`] chosen by
    /// configuration, [`SelectionProvenance::EmbeddingBlend`] and
    /// [`SelectionProvenance::SingleSelectable`] are not degradations.
    #[must_use]
    pub fn is_degraded(self) -> bool {
        matches!(
            self,
            Self::LexicalNoModel | Self::LexicalAfterModelError | Self::UnusableReply
        )
    }
}

/// One parsed selection answer.
///
/// Mirrors Python `SelectionResult`, plus the Rust-only [`SelectionResult::provenance`] /
/// [`SelectionResult::degraded`] markers so a selection that degraded away from the
/// configured model says so in the value itself.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SelectionResult {
    /// The model's raw answer.
    pub raw_output: String,
    /// The targets the answer selected, in answer order.
    pub selected_targets: Vec<SelectableTarget>,
    /// Whether the answer abstained.
    pub is_abstain: bool,
    /// How the selection was answered ([`SelectionProvenance`]).
    pub provenance: SelectionProvenance,
    /// Whether the configured model did not produce the answer
    /// ([`SelectionProvenance::is_degraded`]).
    pub degraded: bool,
}

impl SelectionResult {
    /// Returns this result with `provenance` set, and `degraded` derived from it.
    #[must_use]
    pub fn with_provenance(mut self, provenance: SelectionProvenance) -> Self {
        self.provenance = provenance;
        self.degraded = provenance.is_degraded();
        self
    }
}

/// One child subtree to recurse into, with the target that reached it.
///
/// Mirrors Python `ChildSearchCursor`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ChildSearchCursor {
    /// The cursor of the child subtree.
    pub cursor: SearchCursor,
    /// The target the child subtree was selected by.
    pub target: SelectableTarget,
}

/// What expanding one exposed subtree yields.
///
/// Mirrors Python `ExpansionPlan`: the items found directly at the subtree and the child
/// subtrees still to visit.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExpansionPlan {
    /// The candidates the subtree itself yielded.
    pub leaf_results: Vec<RetrieverCandidate>,
    /// The child subtrees to recurse into, in target order.
    pub child_cursors: Vec<ChildSearchCursor>,
}

/// The candidates one node contributed.
///
/// Mirrors Python `NodeSearchResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct NodeSearchResult {
    /// The candidates this node and its visited descendants produced.
    pub candidates: Vec<RetrieverCandidate>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::retrieval::types::ROOT_NODE_ID;

    fn tree() -> RetrieverNode {
        RetrieverNode::new(ROOT_NODE_ID, "Root").with_children(vec![RetrieverNode::new(
            "Skills.alpha",
            "Alpha",
        )
        .with_items(vec![
            RetrieverItem::new("alpha", "Skills.alpha.item").with_label("Alpha")
        ])])
    }

    #[test]
    fn chat_messages_carry_the_role() {
        assert_eq!(ChatMessage::system("s").role, "system");
        assert_eq!(ChatMessage::user("u").content, "u");
        assert_eq!(ChatMessage::new("tool", "t").role, "tool");
    }

    #[test]
    fn cursor_seeds_at_the_root_and_resolves_back() {
        let tree = tree();
        let cursor = SearchCursor::new(&tree, 5);
        assert_eq!(cursor.node_id, ROOT_NODE_ID);
        assert_eq!(cursor.depth, 0);
        assert_eq!(cursor.branch_path, vec![ROOT_NODE_ID.to_string()]);
        assert_eq!(cursor.top_k, 5);
        assert_eq!(
            cursor.resolve(&tree).map(|node| node.label.as_str()),
            Some("Root")
        );

        let child = SearchCursor::at(
            "Skills.alpha",
            1,
            vec![ROOT_NODE_ID.into(), "Skills.alpha".into()],
            3,
        );
        assert_eq!(
            child.resolve(&tree).map(|node| node.label.as_str()),
            Some("Alpha")
        );
        assert_eq!(child.depth, child.branch_path.len() - 1);
        let missing = SearchCursor::at("nope", 1, vec!["nope".into()], 1);
        assert!(missing.resolve(&tree).is_none());
    }

    fn resolution(code: &str, canonical_id: &str) -> SelectableResolution {
        SelectableResolution {
            code: code.to_owned(),
            canonical_id: canonical_id.to_owned(),
            display_name: "Display".to_owned(),
            label: "Label".to_owned(),
            branch_path: vec![ROOT_NODE_ID.to_owned(), "Skills".to_owned()],
            ..SelectableResolution::default()
        }
    }

    #[test]
    fn fragment_resolutions_follow_candidate_code_order() {
        let mut fragment = ExposedFragment {
            rendered_tree: "1 Alpha\n2 Beta".to_owned(),
            ..ExposedFragment::default()
        };
        fragment
            .code_to_resolution
            .insert("2".to_owned(), resolution("2", "item::beta"));
        fragment
            .code_to_resolution
            .insert("1".to_owned(), resolution("1", "node::alpha"));
        assert!(fragment.is_empty());
        assert_eq!(fragment.selectable_count(), 0);

        fragment.candidate_codes = vec!["2".to_owned(), "1".to_owned(), "9".to_owned()];
        let codes: Vec<&str> = fragment
            .resolutions()
            .into_iter()
            .map(|entry| entry.code.as_str())
            .collect();
        assert_eq!(codes, vec!["2", "1"], "an unknown code is skipped");
        assert_eq!(fragment.selectable_count(), 3);
        assert!(!fragment.is_empty());
        assert_eq!(
            fragment
                .resolution("1")
                .map(|entry| entry.canonical_id.as_str()),
            Some("node::alpha")
        );
        assert!(fragment.resolution("9").is_none());
    }

    #[test]
    fn selectable_target_exposes_the_resolution() {
        let mut resolution = resolution("AB", "item::alpha");
        resolution.is_terminal = true;
        let target = SelectableTarget::new(resolution);
        assert!(target.is_terminal());
        assert_eq!(target.code(), "AB");
        assert_eq!(target.canonical_id(), "item::alpha");
        assert_eq!(
            target.branch_path(),
            [ROOT_NODE_ID.to_owned(), "Skills".to_owned()]
        );
    }

    #[test]
    fn selection_protocol_defaults_to_the_abstain_token() {
        let protocol = SelectionProtocol::default();
        assert_eq!(protocol.abstain_token, ABSTAIN_TOKEN);
        assert_eq!(protocol.candidate_codes, Vec::<String>::new());
        assert_eq!(protocol.code_width, 0);
        assert!(!protocol.compact_codes_enabled);
    }

    #[test]
    fn flow_values_round_trip_through_json() {
        let bundle = PromptBundle {
            fragment: ExposedFragment {
                rendered_tree: "tree".to_owned(),
                candidate_codes: vec!["1".to_owned()],
                ..ExposedFragment::default()
            },
            protocol: SelectionProtocol {
                candidate_codes: vec!["1".to_owned()],
                code_width: 1,
                ..SelectionProtocol::default()
            },
            messages: vec![ChatMessage::system("sys"), ChatMessage::user("usr")],
        };
        let json = serde_json::to_value(&bundle).unwrap();
        assert_eq!(json["protocol"]["abstain_token"], ABSTAIN_TOKEN);
        let parsed: PromptBundle = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, bundle);

        let cursor = SearchCursor::at("n", 2, vec!["root".into(), "a".into(), "n".into()], 4);
        let parsed: SearchCursor =
            serde_json::from_str(&serde_json::to_string(&cursor).unwrap()).unwrap();
        assert_eq!(parsed, cursor);
        let defaulted: SearchCursor = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted, SearchCursor::default());
    }

    #[test]
    fn flow_containers_default_to_empty() {
        assert_eq!(CurrentSubtree::default().selectable_targets, vec![]);
        assert_eq!(ExpansionPlan::default().child_cursors, vec![]);
        assert_eq!(NodeSearchResult::default().candidates, vec![]);
        assert!(!SelectionResult::default().is_abstain);
        assert_eq!(
            ChildSearchCursor::default().cursor.branch_path,
            Vec::<String>::new()
        );
        assert_eq!(ExposedNode::default().children, vec![]);
    }
}
