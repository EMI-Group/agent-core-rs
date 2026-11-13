//! Progressive capability-tree retrieval with an LLM-guided walk.
//!
//! Python's `openjiuwen.symphony.retrieval` presents a capability tree to a model and
//! walks it level by level: the model receives the query plus the current node's
//! children and items and answers the ids worth descending into. This module ports that
//! walk onto the crate's [`CompletionModel`] seam.
//!
//! # Shapes
//!
//! A [`RetrieverNode`] is a labelled branch with child branches and leaf
//! [`RetrieverItem`]s; an item's [`payload`](RetrieverItem::payload) is the retrieved
//! text. [`TreeRetriever::search`] answers a [`TreeSearchResult`] holding the payloads in
//! rank order, the `method` that produced them, and an informational `elapsed_ms`.
//!
//! # The LLM path
//!
//! At each visited node the model receives the query, the node's label/description, and
//! one line per option (`branch <id>: <label> <description>` / `item <id>: ...`). It
//! answers option ids one per line, most relevant first, and `"0"` to abstain. Bullets
//! and numbering are stripped, unknown ids are dropped, and duplicate ids keep their
//! first position. A non-blank reply that selects nothing ends that branch (Python
//! behaviour); an empty reply or a model failure is a soft failure.
//!
//! # The deterministic fallback
//!
//! With no model configured — or when the root selection is a soft failure — the walk
//! ranks a node's children and items by query-term coverage of their
//! `label + " " + description` (the same coverage rule as
//! [`crate::orchestration::LexicalSemanticScorer`]; ties keep the declared order, so no
//! hash-map order leaks in) and descends greedily to the requested limit. The reported
//! method is [`METHOD_LEXICAL_FALLBACK`](crate::llm::tree::METHOD_LEXICAL_FALLBACK).
//! A soft failure deeper in the tree falls back
//! for that subtree only.
//!
//! # Bounds
//!
//! [`TreeRetrieverConfig::max_depth`] caps how many branch levels are entered (level `0`
//! is the root, so `0` reads the root's items only), and every walk stops once it holds
//! `top_k` payloads (`0` means unlimited, the crate's search sentinel). Payloads are
//! deduplicated, so two branches advertising the same payload return it once.
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::search::lexical::tokenize;

use super::{render_prompt, CompletionModel};

/// The `method` reported when the walk asked a model at every visited node.
pub const METHOD_LLM: &str = "llm";
/// The `method` reported when the walk used the lexical ranking.
pub const METHOD_LEXICAL_FALLBACK: &str = "lexical_fallback";
/// The reply token that abstains from the current node.
const ABSTAIN: &str = "0";
/// The default number of payloads a search returns.
const DEFAULT_TOP_K: usize = 10;
/// The default number of branch levels a walk enters.
const DEFAULT_MAX_DEPTH: usize = 16;
/// The system instruction sent with every selection prompt.
const SYSTEM_PROMPT: &str = "You select branches and items from a capability tree. Return option ids only, one per line, most relevant first. Answer 0 when none of the options is relevant.";

/// One leaf of the capability tree: a retrieved payload plus its description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverItem {
    /// Identifier unique among the items of one node.
    pub item_id: String,
    /// The payload returned when the item is selected.
    pub payload: String,
    /// Short human-readable label shown to the model.
    pub label: String,
    /// Longer description shown to the model and used by the lexical ranking.
    pub description: String,
}

impl RetrieverItem {
    /// Builds an item with `item_id` and its `payload`.
    #[must_use]
    pub fn new(item_id: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            payload: payload.into(),
            label: String::new(),
            description: String::new(),
        }
    }

    /// Sets the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// One branch of the capability tree.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieverNode {
    /// Identifier unique among the sibling branches of one node.
    pub node_id: String,
    /// Short human-readable label shown to the model.
    pub label: String,
    /// Longer description shown to the model and used by the lexical ranking.
    pub description: String,
    /// Child branches, in declared order.
    pub children: Vec<RetrieverNode>,
    /// Leaf items, in declared order.
    pub items: Vec<RetrieverItem>,
}

impl RetrieverNode {
    /// Builds an empty branch with `node_id`.
    #[must_use]
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            ..Self::default()
        }
    }

    /// Sets the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Replaces the child branches.
    #[must_use]
    pub fn with_children(mut self, children: Vec<RetrieverNode>) -> Self {
        self.children = children;
        self
    }

    /// Replaces the leaf items.
    #[must_use]
    pub fn with_items(mut self, items: Vec<RetrieverItem>) -> Self {
        self.items = items;
        self
    }

    /// Appends one child branch.
    pub fn push_child(&mut self, child: RetrieverNode) {
        self.children.push(child);
    }

    /// Appends one leaf item.
    pub fn push_item(&mut self, item: RetrieverItem) {
        self.items.push(item);
    }

    /// Looks up a child branch by id.
    #[must_use]
    pub fn child(&self, node_id: &str) -> Option<&Self> {
        self.children.iter().find(|child| child.node_id == node_id)
    }

    /// Looks up a leaf item by id.
    #[must_use]
    pub fn item(&self, item_id: &str) -> Option<&RetrieverItem> {
        self.items.iter().find(|item| item.item_id == item_id)
    }
}

/// Tuning knobs for a [`TreeRetriever`].
#[derive(Clone)]
pub struct TreeRetrieverConfig {
    /// Payloads returned per search; `0` means unlimited.
    pub top_k: usize,
    /// The completion model; `None` keeps the walk on the lexical ranking.
    pub model: Option<Arc<dyn CompletionModel>>,
    /// Branch levels the walk enters; `0` reads the root's items only.
    pub max_depth: usize,
}

impl Default for TreeRetrieverConfig {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            model: None,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }
}

impl TreeRetrieverConfig {
    /// Builds the default configuration: `top_k = 10`, no model, `max_depth = 16`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the per-search payload limit (`0` = unlimited).
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the completion model.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the branch-depth limit.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }
}

impl fmt::Debug for TreeRetrieverConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TreeRetrieverConfig")
            .field("top_k", &self.top_k)
            .field("model", &self.model.as_deref().map(CompletionModel::name))
            .field("max_depth", &self.max_depth)
            .finish()
    }
}

/// Per-request overrides for [`TreeRetriever::search`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TreeSearchRequest {
    /// The requested payload limit; `None` defers to
    /// [`TreeRetrieverConfig::top_k`], and `0` means unlimited.
    pub top_k: Option<usize>,
}

/// The payloads one tree search retrieved, in rank order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TreeSearchResult {
    /// The selected payloads, deduplicated and in rank order.
    pub payloads: Vec<String>,
    /// How the payloads were selected: [`METHOD_LLM`] or [`METHOD_LEXICAL_FALLBACK`].
    pub method: String,
    /// Wall-clock duration in milliseconds; informational only, never asserted on.
    pub elapsed_ms: u64,
}

/// A progressive capability-tree walker.
///
/// The retriever is cheap to clone (`Arc`-shared model behind the configuration) and
/// [`search`](Self::search) takes `&self`, so one tree serves concurrent queries.
pub type LlmTreeRetriever = TreeRetriever;

/// A progressive capability-tree walker.
#[derive(Debug, Clone)]
pub struct TreeRetriever {
    /// The limits and model used by every search.
    config: TreeRetrieverConfig,
    /// The tree root.
    root: RetrieverNode,
}

impl TreeRetriever {
    /// Builds a retriever with `config` and an empty root node.
    #[must_use]
    pub fn new(config: TreeRetrieverConfig) -> Self {
        Self {
            config,
            root: RetrieverNode::new("root"),
        }
    }

    /// Replaces the tree root.
    #[must_use]
    pub fn with_root(mut self, root: RetrieverNode) -> Self {
        self.root = root;
        self
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &TreeRetrieverConfig {
        &self.config
    }

    /// The tree root.
    #[must_use]
    pub fn root(&self) -> &RetrieverNode {
        &self.root
    }

    /// Retrieves the payloads most relevant to `query`.
    ///
    /// # Errors
    ///
    /// Model failures are not errors: they fall back to the lexical ranking. The
    /// signature stays fallible for parity with the other stages and future model
    /// adapters that may need to propagate a cancellation.
    pub async fn search(
        &self,
        query: &str,
        request: &TreeSearchRequest,
    ) -> Result<TreeSearchResult> {
        let started = Instant::now();
        let limit = request.top_k.unwrap_or(self.config.top_k);
        let mut payloads = Vec::new();
        let method = self.walk(query, limit, &mut payloads).await;
        Ok(TreeSearchResult {
            payloads,
            method: method.to_string(),
            elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    /// Walks the tree, collecting payloads, and reports which method produced them.
    async fn walk(&self, query: &str, limit: usize, payloads: &mut Vec<String>) -> &'static str {
        let mut actions: Vec<Walk<'_>> = Vec::new();
        let method = if let Some(model) = self.config.model.as_deref() {
            if let Some(selected) = self
                .request_selection(model, &self.root, query, limit)
                .await
            {
                let mut planned = selected_actions(&self.root, &selected, 0, self.config.max_depth);
                planned.reverse();
                actions = planned;
                METHOD_LLM
            } else {
                actions.push(Walk::Rank(&self.root, 0));
                METHOD_LEXICAL_FALLBACK
            }
        } else {
            actions.push(Walk::Rank(&self.root, 0));
            METHOD_LEXICAL_FALLBACK
        };
        while let Some(action) = actions.pop() {
            if reached(limit, payloads.len()) {
                break;
            }
            match action {
                Walk::Emit(payload) => push_payload(payloads, payload),
                Walk::Rank(node, depth) => {
                    let mut planned = lexical_actions(node, query, depth, self.config.max_depth);
                    planned.reverse();
                    actions.extend(planned);
                }
                Walk::Select(node, depth) => {
                    let mut selection = None;
                    if let Some(model) = self.config.model.as_deref() {
                        selection = self.request_selection(model, node, query, limit).await;
                    }
                    if let Some(selected) = selection {
                        let mut planned =
                            selected_actions(node, &selected, depth, self.config.max_depth);
                        planned.reverse();
                        actions.extend(planned);
                    } else {
                        actions.push(Walk::Rank(node, depth));
                    }
                }
            }
        }
        method
    }

    /// Asks the model which options of `node` to follow.
    ///
    /// `Some(ids)` is a usable reply (possibly empty, which ends the branch);
    /// `None` is a soft failure, so the caller falls back to the lexical ranking.
    async fn request_selection(
        &self,
        model: &dyn CompletionModel,
        node: &RetrieverNode,
        query: &str,
        limit: usize,
    ) -> Option<Vec<String>> {
        if node.children.is_empty() && node.items.is_empty() {
            return Some(Vec::new());
        }
        let prompt = render_prompt(SYSTEM_PROMPT, &selection_prompt(node, query, limit));
        match model.complete(&prompt).await {
            Ok(reply) if reply.trim().is_empty() => {
                tracing::warn!(
                    "tree retriever: model returned an empty selection; using the lexical ranking"
                );
                None
            }
            Ok(reply) => Some(parse_selection(&reply, node)),
            Err(error) => {
                tracing::warn!(%error, "tree retriever: model call failed; using the lexical ranking");
                None
            }
        }
    }
}

/// One pending step of a walk, resolved by the loop in [`TreeRetriever::walk`].
enum Walk<'a> {
    /// Push one item payload.
    Emit(&'a str),
    /// Ask the model which options of this node to follow.
    Select(&'a RetrieverNode, usize),
    /// Rank this node's options lexically and follow them.
    Rank(&'a RetrieverNode, usize),
}

/// Plans the actions for a model selection, in reply order.
fn selected_actions<'a>(
    node: &'a RetrieverNode,
    selected: &[String],
    depth: usize,
    max_depth: usize,
) -> Vec<Walk<'a>> {
    let mut actions = Vec::new();
    for id in selected {
        if let Some(child) = node.child(id) {
            if depth < max_depth {
                actions.push(Walk::Select(child, depth + 1));
            }
        } else if let Some(item) = node.item(id) {
            actions.push(Walk::Emit(&item.payload));
        }
    }
    actions
}

/// Plans the actions for the lexical ranking of one node, best candidate first.
fn lexical_actions<'a>(
    node: &'a RetrieverNode,
    query: &str,
    depth: usize,
    max_depth: usize,
) -> Vec<Walk<'a>> {
    let mut ranked: Vec<Ranked<'a>> = node
        .children
        .iter()
        .map(|child| Ranked {
            kind: RankedKind::Child(child),
            score: coverage(query, &child.label, &child.description),
        })
        .chain(node.items.iter().map(|item| Ranked {
            kind: RankedKind::Item(item),
            score: coverage(query, &item.label, &item.description),
        }))
        .collect();
    // A stable sort keeps the declared order for equal scores.
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut actions = Vec::new();
    for candidate in ranked {
        match candidate.kind {
            RankedKind::Child(child) => {
                if depth < max_depth {
                    actions.push(Walk::Rank(child, depth + 1));
                }
            }
            RankedKind::Item(item) => actions.push(Walk::Emit(&item.payload)),
        }
    }
    actions
}

/// One ranked child branch or leaf item.
struct Ranked<'a> {
    /// The candidate itself.
    kind: RankedKind<'a>,
    /// Query-term coverage in `[0, 1]`.
    score: f64,
}

/// The two kinds of tree option.
enum RankedKind<'a> {
    /// A child branch.
    Child(&'a RetrieverNode),
    /// A leaf item.
    Item(&'a RetrieverItem),
}

/// The fraction of distinct query terms found in `label` and `description`.
///
/// A blank or tokenless query scores `1.0`, matching
/// [`crate::orchestration::LexicalSemanticScorer`]: such a query does not discriminate
/// between options, so the declared order wins.
#[allow(clippy::cast_precision_loss)] // term counts are far below 2^53
fn coverage(query: &str, label: &str, description: &str) -> f64 {
    let query_terms: BTreeSet<String> = tokenize(query).into_iter().collect();
    if query_terms.is_empty() {
        return 1.0;
    }
    let document: BTreeSet<String> = tokenize(&format!("{label} {description}"))
        .into_iter()
        .collect();
    let matched = query_terms
        .iter()
        .filter(|term| document.contains(*term))
        .count();
    matched as f64 / query_terms.len() as f64
}

/// Renders the option list the model chooses from.
fn selection_prompt(node: &RetrieverNode, query: &str, limit: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("Query: {}", query.trim()));
    if limit == 0 {
        lines.push("Choose every relevant option.".to_string());
    } else {
        lines.push(format!("Choose up to {limit} options."));
    }
    let branch = option_line("branch", &node.node_id, &node.label, &node.description);
    if !node.label.trim().is_empty() || !node.description.trim().is_empty() {
        lines.push(format!("Current branch: {branch}"));
    }
    lines.push("Options:".to_string());
    for child in &node.children {
        lines.push(option_line(
            "branch",
            &child.node_id,
            &child.label,
            &child.description,
        ));
    }
    for item in &node.items {
        lines.push(option_line(
            "item",
            &item.item_id,
            &item.label,
            &item.description,
        ));
    }
    lines.join("\n")
}

/// Renders one `"<kind> <id>: <label> <description>"` option line.
fn option_line(kind: &str, id: &str, label: &str, description: &str) -> String {
    let text = format!("{kind} {id}: {} {}", label.trim(), description.trim());
    text.trim_end().to_string()
}

/// Parses a selection reply into known option ids, preserving reply order.
///
/// A `"0"` line abstains and ends the branch, unknown ids are dropped, and duplicate
/// ids keep their first position.
fn parse_selection(reply: &str, node: &RetrieverNode) -> Vec<String> {
    let mut selected: Vec<String> = Vec::new();
    for raw in reply.lines() {
        let line = clean_line(raw);
        if line.is_empty() {
            continue;
        }
        if line == ABSTAIN {
            return Vec::new();
        }
        let known = node.children.iter().any(|child| child.node_id == line)
            || node.items.iter().any(|item| item.item_id == line);
        if !known {
            tracing::debug!(option = line, "tree retriever: dropping unknown option id");
            continue;
        }
        if !selected.iter().any(|id| id == line) {
            selected.push(line.to_string());
        }
    }
    selected
}

/// Strips bullets, numbering, and code fences from one reply line.
fn clean_line(raw: &str) -> &str {
    let mut line = raw.trim();
    line = line.trim_start_matches(['-', '*', '+', '•']).trim_start();
    let digits_end = line
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(line.len());
    if digits_end > 0 {
        let rest = &line[digits_end..];
        if let Some(rest) = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')')) {
            line = rest.trim_start();
        }
    }
    line.trim_matches('`').trim()
}

/// Appends `payload` unless it is already present.
fn push_payload(payloads: &mut Vec<String>, payload: &str) {
    if payloads.iter().any(|existing| existing == payload) {
        return;
    }
    payloads.push(payload.to_string());
}

/// Whether the walk already holds `limit` payloads (`limit == 0` means unlimited).
fn reached(limit: usize, len: usize) -> bool {
    limit != 0 && len >= limit
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[allow(clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn node() -> RetrieverNode {
        RetrieverNode::new("root")
            .with_items(vec![RetrieverItem::new("i1", "p1")])
            .with_children(vec![
                RetrieverNode::new("c1").with_items(vec![RetrieverItem::new("i2", "p2")])
            ])
    }

    #[test]
    fn clean_line_strips_bullets_and_numbering() {
        assert_eq!(clean_line("  - item_a"), "item_a");
        assert_eq!(clean_line("1. item_a"), "item_a");
        assert_eq!(clean_line("2) item_a"), "item_a");
        assert_eq!(clean_line("`item_a`"), "item_a");
        assert_eq!(clean_line("0"), "0");
        assert_eq!(clean_line("   "), "");
    }

    #[test]
    fn parse_selection_drops_unknown_ids_and_dedupes() {
        let node = node();
        assert_eq!(
            parse_selection("i1\nunknown\nc1\ni1", &node),
            vec!["i1".to_string(), "c1".to_string()]
        );
        assert!(parse_selection("0\ni1", &node).is_empty());
        assert!(parse_selection("some prose only", &node).is_empty());
    }

    #[test]
    fn lexical_actions_rank_by_query_terms_and_keep_order_on_ties() {
        let node = RetrieverNode::new("root")
            .with_items(vec![
                RetrieverItem::new("i1", "p1").with_label("unrelated"),
                RetrieverItem::new("i2", "p2").with_label("translate text"),
            ])
            .with_children(vec![RetrieverNode::new("c1").with_label("translate")]);
        let actions = lexical_actions(&node, "translate", 0, 4);
        let rendered: Vec<String> = actions
            .iter()
            .map(|action| match action {
                Walk::Emit(payload) => format!("emit:{payload}"),
                Walk::Rank(node, _) => format!("rank:{}", node.node_id),
                Walk::Select(node, _) => format!("select:{}", node.node_id),
            })
            .collect();
        // The child and the translating item both cover the query exactly; the child is
        // declared before the items, so it stays first, and the unrelated item is last.
        assert_eq!(rendered, vec!["rank:c1", "emit:p2", "emit:p1"]);
    }

    #[test]
    fn selection_prompt_lists_every_option() {
        let prompt = selection_prompt(&node(), "translate", 3);
        assert!(prompt.contains("Query: translate"));
        assert!(prompt.contains("Choose up to 3 options."));
        assert!(prompt.contains("branch c1:"));
        assert!(prompt.contains("item i1:"));
    }

    #[test]
    fn reached_respects_the_unlimited_sentinel() {
        assert!(!reached(0, 100));
        assert!(!reached(3, 2));
        assert!(reached(3, 3));
    }
}
