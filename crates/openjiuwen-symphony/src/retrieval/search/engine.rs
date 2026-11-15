//! The bounded, resumable traversal engine of the query-time search runtime.
//!
//! Ports `RecursiveSearchEngine` of Python
//! `openjiuwen.symphony.retrieval.search.runtime.engine` (v0.1.20), adapted to an
//! async/deterministic design:
//!
//! - Python recurses (`_search_node` calls itself per branch); this port keeps an **explicit
//!   frame stack** in [`TraversalState`], so a bounded depth is a real bound (no stack
//!   growth) and a walk can be stopped, serialized and resumed;
//! - Python fans out over branches with a `ThreadPoolExecutor` and mutates one shared trace
//!   object from several threads; this port visits child cursors **sequentially in declared
//!   order**, so the trace order is deterministic (see [`RecursiveSearchEngine::step`]);
//! - [`TraversalState`] is `serde`-serializable: `begin` a walk, `step` it, serialize it,
//!   restore it and finish it, and the candidates and the trace are the ones an
//!   uninterrupted [`RecursiveSearchEngine::search`] produces.
//!
//! # The five seams
//!
//! The engine owns no search logic of its own. It calls the five protocols of
//! [`contracts`](crate::retrieval::search::contracts) —
//! [`CurrentSubtreeProvider`], [`SubtreeRenderer`], [`TopKSelector`], [`TargetExpander`] and
//! [`BranchReducer`] — and only decides *when* to call them. Selection semantics (the
//! lexical fallback, the fallback mode, the truncation) therefore live in the selector, and
//! the engine surfaces whatever it returns.
//!
//! # Per-node order
//!
//! For the node a cursor points at:
//!
//! 1. `provider.get_current_subtree(cursor)`, then a `fragment_built` event
//!    (`selectable_count`, `rendered_tree`);
//! 2. no selectable target ⇒ the node contributes nothing;
//! 3. exactly one selectable target ⇒ the **single-selectable shortcut**: no render, no
//!    model, one `fragment_selected` event with `mode = "single_selectable_shortcut"`;
//! 4. otherwise `selector.build_protocol` → `renderer.render_subtree` →
//!    `selector.select_topk` (the selector records `fragment_selected` itself — the engine
//!    never records it twice), and an abstention or an empty selection ends the node;
//! 5. `expander.expand_selected_targets`, then a `fragment_continue` event (`leaf_count`,
//!    `branch_count`, `selected_terminal_ids`, `selected_branch_ids`);
//! 6. the child cursors are pushed (in reverse, so the declared order is the visit order) and
//!    visited depth-first;
//! 7. once every child has finished, `reducer.reduce_branch_results` merges the node's own
//!    leaves with the children's results, and a `reduce_complete` event records
//!    `input_candidates` / `output_candidates`.
//!
//! The root's reduction becomes [`TraversalState::candidates`], bounded by the cursor's
//! `top_k` (`0` is the crate's "unlimited" sentinel). [`RecursiveSearchEngine::search`]
//! records a closing `search_complete` event (`steps`, `candidates`).
//!
//! # Bounds
//!
//! [`ProgressiveSearchConfig::max_steps`](crate::retrieval::search::config::ProgressiveSearchConfig::max_steps)
//! (`> 0`) caps node visits: the walk stops, keeps every leaf discovered so far (deduped by
//! payload and bounded by the root's `top_k`), records one `search_truncated` event carrying
//! `max_steps`, and reports the state as exhausted — never an error and never a panic.
//! [`ProgressiveSearchConfig::max_depth`](crate::retrieval::search::config::ProgressiveSearchConfig::max_depth)
//! is enforced by the expander; the engine additionally refuses to visit a cursor whose depth
//! exceeds it.
//!
//! # Deviations from Python
//!
//! - **Sequential branches.** Python's `ThreadPoolExecutor` makes trace order
//!   non-deterministic; the port visits children one at a time in declared order.
//! - **No recursion.** The walk is a loop over an explicit stack, which is what makes
//!   [`TraversalState`] serializable and resumable.
//! - **No legacy direct-recursion path.** Python's `progressive.py` still carries the
//!   pre-protocol recursion (`_search_node`, `_select_items`) that `FlatRetriever` uses; per
//!   that module's own known issues it is not ported — [`FlatRetriever`] instead ranks the
//!   flattened catalog in one pass (see [`crate::retrieval::search::flat`]).
//! - **`top_k == 0` is unlimited**, where Python's `[:top_k]` slicing would keep nothing.
//!
//! [`FlatRetriever`]: crate::retrieval::search::flat::FlatRetriever

use std::collections::BTreeMap;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::Result;
use crate::retrieval::search::config::ProgressiveSearchConfig;
use crate::retrieval::search::contracts::{
    BranchReducer, CurrentSubtreeProvider, SelectRequest, TargetExpander, TopKSelector,
};
use crate::retrieval::search::flow::{
    ChatMessage, NodeSearchResult, SearchCursor, SelectableTarget,
};
use crate::retrieval::search::render::SubtreeRenderer;
use crate::retrieval::search::selector::FRAGMENT_SELECTED_EVENT;
use crate::retrieval::types::{RetrieverCandidate, RetrieverTrace};

/// The trace event recorded when a node's subtree has been exposed.
pub const FRAGMENT_BUILT_EVENT: &str = "fragment_built";

/// The trace event recorded when a node's selection has been expanded.
pub const FRAGMENT_CONTINUE_EVENT: &str = "fragment_continue";

/// The trace event recorded when a node's branches have been merged.
pub const REDUCE_COMPLETE_EVENT: &str = "reduce_complete";

/// The trace event that closes one [`RecursiveSearchEngine::search`].
pub const SEARCH_COMPLETE_EVENT: &str = "search_complete";

/// The trace event recorded when a walk stopped at
/// [`ProgressiveSearchConfig::max_steps`](crate::retrieval::search::config::ProgressiveSearchConfig::max_steps).
pub const SEARCH_TRUNCATED_EVENT: &str = "search_truncated";

/// The `fragment_selected` mode of the single-selectable shortcut.
pub const MODE_SINGLE_SELECTABLE_SHORTCUT: &str = "single_selectable_shortcut";

/// The inputs of one walk: the model identity and the caller's query messages.
///
/// Python passes the same values as keyword arguments to `RecursiveSearchEngine.search`;
/// the port borrows them so a walk never clones the conversation per node visit.
#[derive(Debug, Clone, Copy)]
pub struct EngineRequest<'a> {
    /// The model identity a model-facing selector forwards, or `None` for its default.
    pub model: Option<&'a str>,
    /// The caller's query messages (the conversation a selection is made against).
    pub query_messages: &'a [ChatMessage],
}

/// One frame of the walk: a cursor, whether it has been visited, and what it collected.
///
/// A frame is pushed when a child cursor is expanded and popped once the node (and every
/// child pushed above it) has finished, so the stack is a depth-first traversal plan.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
struct Frame {
    /// The cursor this frame points at.
    cursor: SearchCursor,
    /// Whether the node has been exposed and expanded (and therefore awaits its reduction).
    visited: bool,
    /// The leaf candidates the node's own selection yielded.
    local_leaves: Vec<RetrieverCandidate>,
    /// The results its children have already reported, in visit order.
    child_results: Vec<NodeSearchResult>,
}

/// The resumable state of one progressive walk.
///
/// Everything the walk needs is in here — the frame stack, the counters, the query messages
/// and the trace — so the state round-trips through `serde_json` and a traversal can be
/// saved after any number of [`RecursiveSearchEngine::step`] calls and resumed with an
/// equivalent result.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TraversalState {
    /// The cursor the walk started at; the root's `top_k` bounds the final list.
    root_cursor: SearchCursor,
    /// The effective `top_k` of the root (`0` = unlimited).
    root_top_k: usize,
    /// The pending frames, the deepest node last.
    frames: Vec<Frame>,
    /// How many nodes have been visited.
    steps: usize,
    /// The root's reduced candidates, set when the walk finishes or is truncated.
    candidates: Vec<RetrieverCandidate>,
    /// Every leaf candidate discovered so far, in discovery (depth-first) order.
    discovered: Vec<RetrieverCandidate>,
    /// Whether the walk stopped at the step budget.
    truncated: bool,
    /// The query messages a selection is made against; part of the state so a resumed walk
    /// keeps talking about the same request.
    query_messages: Vec<ChatMessage>,
    /// The trace of the walk.
    trace: RetrieverTrace,
}

impl TraversalState {
    /// Starts a walk at `root_cursor`.
    ///
    /// The cursor's `top_k` is the effective bound of the final candidate list (`0` =
    /// unlimited), and the stack starts with a single unvisited root frame.
    #[must_use]
    pub fn new(root_cursor: SearchCursor) -> Self {
        let root_top_k = root_cursor.top_k;
        Self {
            frames: vec![Frame {
                cursor: root_cursor.clone(),
                ..Frame::default()
            }],
            root_cursor,
            root_top_k,
            ..Self::default()
        }
    }

    /// Attaches the query messages a selection is made against.
    #[must_use]
    pub fn with_query_messages(mut self, query_messages: Vec<ChatMessage>) -> Self {
        self.query_messages = query_messages;
        self
    }

    /// The query messages of this walk.
    #[must_use]
    pub fn query_messages(&self) -> &[ChatMessage] {
        &self.query_messages
    }

    /// The cursor the walk started at.
    #[must_use]
    pub fn root_cursor(&self) -> &SearchCursor {
        &self.root_cursor
    }

    /// Whether the frontier is exhausted (nothing left to visit and nothing left to reduce).
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.frames.is_empty()
    }

    /// How many nodes have been visited so far.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// How many frames are still on the stack.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.frames.len()
    }

    /// The candidates the root has produced (empty until the walk finishes or is truncated).
    #[must_use]
    pub fn candidates(&self) -> &[RetrieverCandidate] {
        &self.candidates
    }

    /// Whether the walk stopped at the step budget.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// The trace of the walk.
    #[must_use]
    pub fn trace(&self) -> &RetrieverTrace {
        &self.trace
    }

    /// The trace of the walk, for a caller that records its own events.
    pub fn trace_mut(&mut self) -> &mut RetrieverTrace {
        &mut self.trace
    }

    /// Takes the candidates out of the state, leaving an empty list behind.
    pub fn take_candidates(&mut self) -> Vec<RetrieverCandidate> {
        std::mem::take(&mut self.candidates)
    }
}

/// What one [`RecursiveSearchEngine::step`] did.
///
/// One step is one node visit. `candidates_after` counts the leaf candidates the walk has
/// discovered so far (this node's included): the root's reduced list is only known once the
/// walk completes, so a caller watching progress reads this counter.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TraversalStep {
    /// The node that was visited.
    pub node_id: String,
    /// Its traversal depth (the root is `0`).
    pub depth: usize,
    /// How many selectable targets the exposure carried.
    pub selectable_count: usize,
    /// The codes the node selected, in selection order.
    pub selected_codes: Vec<String>,
    /// How many leaf candidates the expansion produced.
    pub leaf_count: usize,
    /// How many child cursors the expansion planned.
    pub branch_count: usize,
    /// How many leaf candidates the walk has discovered so far.
    pub candidates_after: usize,
}

/// The bounded traversal state machine over [`ProgressiveSearchConfig`].
///
/// Holds the five seams plus the config and nothing else, so it is cheap to construct per
/// search and safe to share behind an `Arc`.
pub struct RecursiveSearchEngine {
    /// Exposes the subtree a cursor points at.
    provider: Arc<dyn CurrentSubtreeProvider>,
    /// Renders the prompt of one exposed subtree.
    renderer: Arc<dyn SubtreeRenderer>,
    /// Selects the targets of one exposed subtree.
    selector: Arc<dyn TopKSelector>,
    /// Turns selected targets into leaves and child cursors.
    expander: Arc<dyn TargetExpander>,
    /// Merges a node's leaves with its children's results.
    reducer: Arc<dyn BranchReducer>,
    /// The traversal config (the two hard bounds are the engine's business).
    config: ProgressiveSearchConfig,
}

impl RecursiveSearchEngine {
    /// Builds the engine over the five seams and `config`.
    #[must_use]
    pub fn new(
        provider: Arc<dyn CurrentSubtreeProvider>,
        renderer: Arc<dyn SubtreeRenderer>,
        selector: Arc<dyn TopKSelector>,
        expander: Arc<dyn TargetExpander>,
        reducer: Arc<dyn BranchReducer>,
        config: ProgressiveSearchConfig,
    ) -> Self {
        Self {
            provider,
            renderer,
            selector,
            expander,
            reducer,
            config,
        }
    }

    /// The traversal config the engine runs with.
    #[must_use]
    pub fn config(&self) -> &ProgressiveSearchConfig {
        &self.config
    }

    /// Drives the walk to completion and returns the root's candidates.
    ///
    /// A state that is already exhausted yields its candidates unchanged; each call closes
    /// with one `search_complete` event.
    ///
    /// # Errors
    ///
    /// Whatever the seams report: a provider, renderer or selector failure (including
    /// [`Error::Unsupported`](crate::error::Error) for a selection mode that needs a model).
    pub async fn search(
        &self,
        request: &EngineRequest<'_>,
        state: &mut TraversalState,
    ) -> Result<Vec<RetrieverCandidate>> {
        while self.step(request, state).await?.is_some() {}
        let candidates = state.candidates().to_vec();
        let detail = BTreeMap::from([
            ("steps".to_owned(), json!(state.steps())),
            ("candidates".to_owned(), json!(candidates.len())),
        ]);
        let root_node_id = state.root_cursor.node_id.clone();
        state
            .trace
            .record(SEARCH_COMPLETE_EVENT, root_node_id, 0, detail);
        Ok(candidates)
    }

    /// Advances the walk by one node visit; `Ok(None)` when the frontier is exhausted.
    ///
    /// The returned [`TraversalStep`] describes the node that was visited — including a node
    /// whose exposure had nothing selectable, whose selection abstained or was empty, and
    /// whose expansion planned no child. Reducing a node (and the parents it finishes) is not
    /// a step of its own: it happens while the following visit is looked for.
    ///
    /// # Errors
    ///
    /// Whatever a seam reports for the visited node.
    pub async fn step(
        &self,
        request: &EngineRequest<'_>,
        state: &mut TraversalState,
    ) -> Result<Option<TraversalStep>> {
        loop {
            let Some(top) = state.frames.last() else {
                return Ok(None);
            };
            if self.step_budget_reached(state) {
                self.truncate_walk(state);
                return Ok(None);
            }
            let cursor = top.cursor.clone();
            if top.visited {
                self.reduce_node(state);
                continue;
            }
            if self.depth_refused(&cursor) {
                Self::complete_node(state, NodeSearchResult::default());
                continue;
            }
            let step = self.visit_node(request, state, cursor).await?;
            return Ok(Some(step));
        }
    }

    /// Whether the visit budget is spent.
    fn step_budget_reached(&self, state: &TraversalState) -> bool {
        self.config.max_steps > 0 && state.steps >= self.config.max_steps
    }

    /// Whether `cursor` sits deeper than the configured bound.
    fn depth_refused(&self, cursor: &SearchCursor) -> bool {
        self.config.max_depth > 0 && cursor.depth > self.config.max_depth
    }

    /// Visits one node: expose, select, expand, and queue its children.
    async fn visit_node(
        &self,
        request: &EngineRequest<'_>,
        state: &mut TraversalState,
        cursor: SearchCursor,
    ) -> Result<TraversalStep> {
        if let Some(frame) = state.frames.last_mut() {
            frame.visited = true;
        }
        state.steps += 1;
        let subtree = self.provider.get_current_subtree(&cursor)?;
        let selectable_count = subtree.selectable_targets.len();
        let built_detail = BTreeMap::from([
            ("selectable_count".to_owned(), json!(selectable_count)),
            (
                "rendered_tree".to_owned(),
                json!(subtree.fragment.rendered_tree),
            ),
        ]);
        state.trace.record(
            FRAGMENT_BUILT_EVENT,
            &cursor.node_id,
            cursor.depth,
            built_detail,
        );

        let selected = self
            .select_targets(request, state, &cursor, &subtree, selectable_count)
            .await?;
        if selected.is_empty() {
            let step = TraversalStep {
                node_id: cursor.node_id.clone(),
                depth: cursor.depth,
                selectable_count,
                selected_codes: Vec::new(),
                leaf_count: 0,
                branch_count: 0,
                candidates_after: state.discovered.len(),
            };
            Self::complete_node(state, NodeSearchResult::default());
            return Ok(step);
        }

        let plan = self.expander.expand_selected_targets(&cursor, &selected);
        let leaf_count = plan.leaf_results.len();
        let selected_codes: Vec<String> = selected
            .iter()
            .map(|target| target.code().to_owned())
            .collect();
        let detail = BTreeMap::from([
            ("leaf_count".to_owned(), json!(leaf_count)),
            ("branch_count".to_owned(), json!(plan.child_cursors.len())),
            (
                "selected_terminal_ids".to_owned(),
                json!(plan
                    .leaf_results
                    .iter()
                    .map(|candidate| candidate.item_id.as_str())
                    .collect::<Vec<&str>>()),
            ),
            (
                "selected_branch_ids".to_owned(),
                json!(plan
                    .child_cursors
                    .iter()
                    .map(|child| child.cursor.node_id.as_str())
                    .collect::<Vec<&str>>()),
            ),
        ]);
        state.trace.record(
            FRAGMENT_CONTINUE_EVENT,
            &cursor.node_id,
            cursor.depth,
            detail,
        );

        state.discovered.extend(plan.leaf_results.iter().cloned());
        if let Some(frame) = state.frames.last_mut() {
            frame.local_leaves = plan.leaf_results;
        }
        let mut children: Vec<Frame> = Vec::with_capacity(plan.child_cursors.len());
        for child in &plan.child_cursors {
            if self.depth_refused(&child.cursor) {
                continue;
            }
            children.push(Frame {
                cursor: child.cursor.clone(),
                ..Frame::default()
            });
        }
        let branch_count = children.len();
        for frame in children.into_iter().rev() {
            state.frames.push(frame);
        }

        Ok(TraversalStep {
            node_id: cursor.node_id.clone(),
            depth: cursor.depth,
            selectable_count,
            selected_codes,
            leaf_count,
            branch_count,
            candidates_after: state.discovered.len(),
        })
    }

    /// Selects the targets of one exposure, applying the single-selectable shortcut.
    async fn select_targets(
        &self,
        request: &EngineRequest<'_>,
        state: &mut TraversalState,
        cursor: &SearchCursor,
        subtree: &crate::retrieval::search::flow::CurrentSubtree,
        selectable_count: usize,
    ) -> Result<Vec<SelectableTarget>> {
        if selectable_count == 0 {
            return Ok(Vec::new());
        }
        if selectable_count == 1 {
            let only = subtree.selectable_targets[0].clone();
            let detail = BTreeMap::from([
                ("mode".to_owned(), json!(MODE_SINGLE_SELECTABLE_SHORTCUT)),
                ("selected_codes".to_owned(), json!([only.code()])),
                (
                    "selected_canonical_ids".to_owned(),
                    json!([only.canonical_id()]),
                ),
            ]);
            state.trace.record(
                FRAGMENT_SELECTED_EVENT,
                &cursor.node_id,
                cursor.depth,
                detail,
            );
            return Ok(vec![only]);
        }
        let protocol = self.selector.build_protocol(subtree);
        let prompt = self
            .renderer
            .render_subtree(subtree, request.query_messages, &protocol)?;
        let selection = self
            .selector
            .select_topk(
                SelectRequest {
                    model: request.model,
                    cursor,
                    query_messages: request.query_messages,
                    subtree,
                    prompt: &prompt,
                },
                &mut state.trace,
            )
            .await?;
        if selection.is_abstain {
            return Ok(Vec::new());
        }
        Ok(selection.selected_targets)
    }

    /// Reduces the top frame (its children are done) and reports the result to its parent.
    fn reduce_node(&self, state: &mut TraversalState) {
        let Some(frame) = state.frames.last().cloned() else {
            return;
        };
        let input_candidates = frame.local_leaves.len()
            + frame
                .child_results
                .iter()
                .map(|result| result.candidates.len())
                .sum::<usize>();
        let reduced = self.reducer.reduce_branch_results(
            &frame.cursor,
            &frame.local_leaves,
            &frame.child_results,
        );
        let detail = BTreeMap::from([
            ("input_candidates".to_owned(), json!(input_candidates)),
            (
                "output_candidates".to_owned(),
                json!(reduced.candidates.len()),
            ),
        ]);
        state.trace.record(
            REDUCE_COMPLETE_EVENT,
            &frame.cursor.node_id,
            frame.cursor.depth,
            detail,
        );
        Self::complete_node(state, reduced);
    }

    /// Pops the top frame and reports `result` to the parent frame, or installs it as the
    /// root's candidates when no frame is left.
    fn complete_node(state: &mut TraversalState, result: NodeSearchResult) {
        state.frames.pop();
        if let Some(parent) = state.frames.last_mut() {
            parent.child_results.push(result);
        } else {
            let mut candidates = result.candidates;
            if state.root_top_k > 0 {
                candidates.truncate(state.root_top_k);
            }
            state.candidates = candidates;
        }
    }

    /// Stops the walk at the step budget, keeping what has been discovered.
    fn truncate_walk(&self, state: &mut TraversalState) {
        if state.truncated {
            state.frames.clear();
            return;
        }
        let root_cursor = state.root_cursor.clone();
        let reduced = self
            .reducer
            .reduce_branch_results(&root_cursor, &state.discovered, &[]);
        state.candidates = reduced.candidates;
        state.frames.clear();
        state.truncated = true;
        let detail = BTreeMap::from([("max_steps".to_owned(), json!(self.config.max_steps))]);
        let root_node_id = state.root_cursor.node_id.clone();
        state
            .trace
            .record(SEARCH_TRUNCATED_EVENT, root_node_id, 0, detail);
    }
}

impl std::fmt::Debug for RecursiveSearchEngine {
    /// Renders the engine's config and the mode it selects in.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RecursiveSearchEngine")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::retrieval::search::config::DisclosureConfig;
    use crate::retrieval::search::expand::DefaultTargetExpander;
    use crate::retrieval::search::reduce::DefaultBranchReducer;
    use crate::retrieval::search::render::build_exposed_fragment;
    use crate::retrieval::search::selector::DefaultTopKSelector;
    use crate::retrieval::search::subtree::DefaultCurrentSubtreeProvider;
    use crate::retrieval::types::{RetrieverItem, RetrieverNode, ROOT_NODE_ID};

    /// A fragment builder over the crate's renderer.
    fn builder() -> crate::retrieval::search::contracts::FragmentBuildFn {
        Arc::new(
            |node: &RetrieverNode,
             branch_path: &[String],
             disclosure: &DisclosureConfig,
             item_count: usize| {
                let counter = |_: &RetrieverNode| item_count;
                build_exposed_fragment(node, branch_path, disclosure, &counter)
            },
        )
    }

    /// A three-level tree: root → a single branch chain → one item.
    fn chain_tree() -> RetrieverNode {
        RetrieverNode::new(ROOT_NODE_ID, "Root").with_children(vec![RetrieverNode::new(
            "Skills", "Skills",
        )
        .with_children(vec![RetrieverNode::new("Skills.alpha", "Alpha")
            .with_items(vec![RetrieverItem::new("alpha.one", "alpha.one")
                .with_label("Alpha writer")
                .with_description("writes alpha documentation")])])])
    }

    /// A tree whose root exposes two branches and one of its own items.
    fn mixed_tree() -> RetrieverNode {
        RetrieverNode::new(ROOT_NODE_ID, "Root")
            .with_items(vec![
                RetrieverItem::new("flat.one", "flat.one").with_label("Flat")
            ])
            .with_children(vec![
                RetrieverNode::new("Skills.alpha", "Alpha").with_items(vec![
                    RetrieverItem::new("alpha.one", "alpha.one")
                        .with_label("Alpha writer")
                        .with_description("writes alpha documentation"),
                    RetrieverItem::new("alpha.two", "alpha.two")
                        .with_label("Alpha tester")
                        .with_description("tests alpha behaviour"),
                ]),
                RetrieverNode::new("Skills.beta", "Beta")
                    .with_items(vec![
                        RetrieverItem::new("beta.one", "beta.one").with_label("Beta compute")
                    ]),
            ])
    }

    /// A config that exposes one level of children as selectable branches.
    fn flat_branch_config() -> ProgressiveSearchConfig {
        flat_branch_config_with_steps(0)
    }

    /// [`flat_branch_config`] with a visit budget.
    fn flat_branch_config_with_steps(max_steps: usize) -> ProgressiveSearchConfig {
        ProgressiveSearchConfig {
            top_k: 3,
            max_steps,
            disclosure: DisclosureConfig {
                max_exposure_depth_per_call: 0,
                exposure_threshold: 100,
                compact_codes_enabled: true,
                ..DisclosureConfig::default()
            },
            ..ProgressiveSearchConfig::default()
        }
    }

    /// The engine over `tree` and `config`, selecting lexically.
    fn engine(tree: &RetrieverNode, config: ProgressiveSearchConfig) -> RecursiveSearchEngine {
        let provider = DefaultCurrentSubtreeProvider::with_builder(
            Arc::new(tree.clone()),
            config.clone(),
            builder(),
        )
        .unwrap();
        RecursiveSearchEngine::new(
            Arc::new(provider),
            Arc::new(crate::retrieval::search::render::DefaultSubtreeRenderer::new()),
            Arc::new(DefaultTopKSelector::new(config.clone())),
            Arc::new(DefaultTargetExpander::new(config.clone())),
            Arc::new(DefaultBranchReducer::new(config.clone())),
            config,
        )
    }

    /// A state seeded at `tree`'s root with `top_k` and the given query.
    fn state(tree: &RetrieverNode, top_k: usize, query: &str) -> TraversalState {
        TraversalState::new(SearchCursor::new(tree, top_k))
            .with_query_messages(vec![ChatMessage::user(query)])
    }

    /// Runs one search over `query` with no model identity.
    async fn search(
        engine: &RecursiveSearchEngine,
        walk: &mut TraversalState,
        query: &str,
    ) -> crate::error::Result<Vec<RetrieverCandidate>> {
        let messages = vec![ChatMessage::user(query)];
        let request = EngineRequest {
            model: None,
            query_messages: &messages,
        };
        engine.search(&request, walk).await
    }

    fn ids(candidates: &[RetrieverCandidate]) -> Vec<&str> {
        candidates
            .iter()
            .map(|candidate| candidate.item_id.as_str())
            .collect()
    }

    #[tokio::test]
    async fn a_single_selectable_node_descends_without_a_selection() {
        let tree = chain_tree();
        let engine = engine(&tree, flat_branch_config());
        let mut walk = state(&tree, 3, "alpha");
        let candidates = search(&engine, &mut walk, "alpha").await.unwrap();
        assert_eq!(ids(&candidates), vec!["alpha.one"]);
        assert_eq!(
            walk.steps(),
            3,
            "root, Skills and Skills.alpha were visited"
        );
        assert!(walk.is_exhausted());
        let modes: Vec<&serde_json::Value> = walk
            .trace()
            .events()
            .iter()
            .filter(|event| event.event_type == FRAGMENT_SELECTED_EVENT)
            .map(|event| &event.detail["mode"])
            .collect();
        assert!(modes
            .iter()
            .all(|mode| **mode == json!(MODE_SINGLE_SELECTABLE_SHORTCUT)));
        assert_eq!(modes.len(), 3);
    }

    #[tokio::test]
    async fn an_empty_exposure_contributes_nothing_and_records_no_reduce() {
        let tree = RetrieverNode::new(ROOT_NODE_ID, "Root")
            .with_children(vec![RetrieverNode::new("empty", "Empty")]);
        let engine = engine(&tree, flat_branch_config());
        let mut walk = state(&tree, 3, "anything");
        let candidates = search(&engine, &mut walk, "anything").await.unwrap();
        assert!(candidates.is_empty());
        // The root exposes its one child as a selectable branch, so the shortcut descends
        // into the empty node, which contributes nothing.
        assert_eq!(walk.steps(), 2, "the root and the empty child were visited");
        let event_types = walk.trace().event_types();
        assert!(event_types.contains(&FRAGMENT_BUILT_EVENT));
        assert_eq!(
            event_types
                .iter()
                .filter(|event| **event == REDUCE_COMPLETE_EVENT)
                .count(),
            1,
            "only the root reduces (the empty node has no branch list of its own)"
        );
        assert_eq!(event_types.last().copied(), Some(SEARCH_COMPLETE_EVENT));
    }

    #[tokio::test]
    async fn an_abstention_ends_the_node_without_a_continue_event() {
        let tree = mixed_tree();
        let engine = engine(&tree, flat_branch_config());
        let mut walk = state(&tree, 3, "0");
        let messages = vec![ChatMessage::user("0")];
        let request = EngineRequest {
            model: None,
            query_messages: &messages,
        };
        let step = engine.step(&request, &mut walk).await.unwrap().unwrap();
        assert_eq!(step.node_id, ROOT_NODE_ID);
        assert_eq!(step.selectable_count, 3);
        // The lexical selector has no opinion on "0" (`0` matches no document), so nothing
        // is selected and the node ends without a `fragment_continue`.
        assert!(step.selected_codes.is_empty());
        assert!(step.leaf_count == 0 && step.branch_count == 0);
        assert!(!walk
            .trace()
            .event_types()
            .contains(&FRAGMENT_CONTINUE_EVENT));
    }

    #[tokio::test]
    async fn the_step_budget_truncates_the_walk_and_keeps_partial_candidates() {
        let tree = mixed_tree();
        let engine = engine(&tree, flat_branch_config_with_steps(1));
        let mut walk = state(&tree, 3, "flat");
        let candidates = search(&engine, &mut walk, "flat").await.unwrap();
        assert_eq!(
            ids(&candidates),
            vec!["flat.one"],
            "the root's own leaf was discovered before the budget ran out"
        );
        assert!(walk.is_truncated());
        assert!(walk.is_exhausted());
        let event = walk.trace().last().unwrap();
        assert_eq!(event.event_type, SEARCH_COMPLETE_EVENT);
        let truncated = walk
            .trace()
            .events()
            .iter()
            .find(|event| event.event_type == SEARCH_TRUNCATED_EVENT)
            .unwrap();
        assert_eq!(truncated.detail["max_steps"], json!(1));
        assert_eq!(truncated.node_id, ROOT_NODE_ID);
    }

    #[tokio::test]
    async fn the_state_round_trips_and_resumes_to_the_same_result() {
        let tree = mixed_tree();
        let config = flat_branch_config();
        let uninterrupted = {
            let engine = engine(&tree, config.clone());
            let mut walk = state(&tree, 3, "alpha");
            let candidates = search(&engine, &mut walk, "alpha documentation")
                .await
                .unwrap();
            (candidates, walk.trace().clone())
        };
        let engine = engine(&tree, config);
        let mut walk = state(&tree, 3, "alpha");
        let messages = vec![ChatMessage::user("alpha")];
        let first = EngineRequest {
            model: None,
            query_messages: &messages,
        };
        engine.step(&first, &mut walk).await.unwrap().unwrap();
        let encoded = serde_json::to_string(&walk).unwrap();
        let mut restored: TraversalState = serde_json::from_str(&encoded).unwrap();
        assert_eq!(restored.steps(), 1);
        assert_eq!(restored.pending(), walk.pending());
        let mut rest = 0_usize;
        while engine
            .step(
                &EngineRequest {
                    model: None,
                    query_messages: &messages,
                },
                &mut restored,
            )
            .await
            .unwrap()
            .is_some()
        {
            rest += 1;
        }
        assert!(rest > 0);
        assert_eq!(restored.candidates(), uninterrupted.0.as_slice());
        assert!(restored.trace().events().len() <= uninterrupted.1.events().len());
        assert_eq!(
            restored.trace().event_types(),
            uninterrupted.1.event_types()[..restored.trace().events().len()].to_vec()
        );
    }

    #[tokio::test]
    async fn an_exhausted_state_reports_nothing_and_keeps_its_candidates() {
        let tree = chain_tree();
        let engine = engine(&tree, flat_branch_config());
        let mut walk = state(&tree, 3, "alpha");
        search(&engine, &mut walk, "alpha").await.unwrap();
        let messages = vec![ChatMessage::user("alpha")];
        let request = EngineRequest {
            model: None,
            query_messages: &messages,
        };
        assert!(engine.step(&request, &mut walk).await.unwrap().is_none());
        assert_eq!(walk.pending(), 0);
        let taken = walk.take_candidates();
        assert_eq!(ids(&taken), vec!["alpha.one"]);
        assert!(walk.candidates().is_empty());
    }

    #[tokio::test]
    async fn the_depth_bound_refuses_a_too_deep_root_cursor() {
        let tree = chain_tree();
        let config = ProgressiveSearchConfig {
            max_depth: 1,
            ..flat_branch_config()
        };
        let engine = engine(&tree, config);
        let deep = SearchCursor::at(
            "Skills.alpha",
            3,
            vec![
                ROOT_NODE_ID.to_owned(),
                "Skills".to_owned(),
                "Skills.alpha".to_owned(),
            ],
            3,
        );
        let mut walk = TraversalState::new(deep);
        assert!(search(&engine, &mut walk, "alpha")
            .await
            .unwrap()
            .is_empty());
        assert_eq!(walk.steps(), 0, "the refused cursor was never visited");
        assert!(walk.is_exhausted());
    }

    #[test]
    fn traversal_steps_and_state_serialize() {
        let cursor = SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], 4);
        let walk = TraversalState::new(cursor).with_query_messages(vec![ChatMessage::user("q")]);
        let json = serde_json::to_value(&walk).unwrap();
        assert_eq!(json["root_top_k"], 4);
        assert_eq!(json["query_messages"][0]["content"], "q");
        assert!(json.get("frames").is_some());
        let parsed: TraversalState = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.steps(), 0);
        assert_eq!(parsed.pending(), 1);
        let step = TraversalStep {
            node_id: "n".to_owned(),
            depth: 1,
            selectable_count: 2,
            selected_codes: vec!["1".to_owned()],
            leaf_count: 1,
            branch_count: 0,
            candidates_after: 1,
        };
        let parsed: TraversalStep =
            serde_json::from_str(&serde_json::to_string(&step).unwrap()).unwrap();
        assert_eq!(parsed, step);
        let defaulted: TraversalStep = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted, TraversalStep::default());
    }

    #[test]
    fn the_unlimited_sentinel_survives_into_the_state() {
        let tree = chain_tree();
        let walk = state(&tree, 0, "alpha");
        assert_eq!(walk.root_cursor().top_k, 0);
        let encoded = serde_json::to_string(&walk).unwrap();
        let restored: TraversalState = serde_json::from_str(&encoded).unwrap();
        assert_eq!(restored.root_cursor().top_k, 0);
    }

    #[tokio::test]
    async fn the_root_list_is_bounded_by_the_cursor_top_k() {
        let tree = mixed_tree();
        let engine = engine(&tree, flat_branch_config());
        let mut walk = state(&tree, 1, "alpha beta flat");
        let candidates = search(&engine, &mut walk, "alpha beta flat").await.unwrap();
        assert_eq!(candidates.len(), 1);
        let mut unlimited = state(&tree, 0, "alpha beta flat");
        let all = search(&engine, &mut unlimited, "alpha beta flat")
            .await
            .unwrap();
        assert!(all.len() > 1, "top_k 0 is unlimited");
    }
}
