//! The traversal-protocol seams of the query-time search runtime.
//!
//! Ports the four Python protocols of
//! `openjiuwen.symphony.retrieval.search.runtime.contracts` (v0.1.20) —
//! [`CurrentSubtreeProvider`], [`TopKSelector`], [`TargetExpander`] and
//! [`BranchReducer`] — plus the two callback surfaces the engine path injects:
//! [`FragmentBuildFn`] (subtree rendering) and [`GenerateFn`] (model-facing selection).
//!
//! Python's fifth protocol, `SubtreeRenderer`, has no trait here: rendering belongs to the
//! `render` module and reaches this layer only through [`FragmentBuildFn`], which is the
//! single injected callback [`DefaultCurrentSubtreeProvider`] needs. Keeping rendering
//! behind a callback is what lets this module compile and run with no renderer at all —
//! [subtree](crate::retrieval::search::subtree) returns
//! [`Error::Unsupported`](crate::error::Error) until a builder is supplied.
//!
//! # The request structs
//!
//! [`SelectRequest`] and [`GenerateRequest`] borrow the caller's cursor, subtree, prompt
//! and trace instead of cloning them, so they are deliberately **not** `serde`
//! serializable (they carry `&mut` references). The values they point at —
//! [`CurrentSubtree`], [`PromptBundle`],
//! [`SelectionProtocol`], [`SelectionResult`] — are the
//! serializable data types.
//!
//! [`DefaultCurrentSubtreeProvider`]: crate::retrieval::search::subtree::DefaultCurrentSubtreeProvider

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::retrieval::search::config::DisclosureConfig;
use crate::retrieval::search::flow::{
    ChatMessage, CurrentSubtree, ExpansionPlan, ExposedFragment, NodeSearchResult, PromptBundle,
    SearchCursor, SelectableResolution, SelectableTarget, SelectionProtocol, SelectionResult,
};
use crate::retrieval::types::{RetrieverCandidate, RetrieverNode, RetrieverTrace};

/// Renders one subtree into an [`ExposedFragment`].
///
/// The four arguments are the bound root node, the branch path of the node the cursor
/// points at, the disclosure configuration to render with, and how many items the
/// subtree holds (the exposure threshold's input). The rendering wave supplies the
/// implementation; this layer only calls it, so a caller that wants the traversal
/// without a renderer can still build, select, expand and reduce over hand-made
/// fragments.
pub type FragmentBuildFn = Arc<
    dyn Fn(&RetrieverNode, &[String], &DisclosureConfig, usize) -> Result<ExposedFragment>
        + Send
        + Sync,
>;

/// Exposes the subtree a cursor points at.
///
/// Python `CurrentSubtreeProvider`. The engine calls this once per visited node; the
/// default implementation
/// ([`DefaultCurrentSubtreeProvider`](crate::retrieval::search::subtree::DefaultCurrentSubtreeProvider))
/// caches the built fragment per `(tree generation, node, branch path, top_k)`.
pub trait CurrentSubtreeProvider: Send + Sync {
    /// Exposes [`SearchCursor::node_id`] of `cursor`, emitting a [`CurrentSubtree`] that
    /// carries `cursor` itself.
    ///
    /// # Errors
    ///
    /// The implementation's own failures — [`Error::NotFound`](crate::error::Error) when
    /// the node is not in the tree, [`Error::Unsupported`](crate::error::Error) when no
    /// fragment builder is configured.
    fn get_current_subtree(&self, cursor: &SearchCursor) -> Result<CurrentSubtree>;
}

/// Everything one selection needs.
///
/// Mirrors the keyword arguments of Python's `TopKSelector.select_topk`. `model` is the
/// model identity a selector forwards to the model seam (`None` lets a selector pick its
/// own default), and `prompt` is the rendered bundle a model-facing selector sends. The
/// struct borrows its inputs because a selection is a read-only view of the traversal
/// state plus the one mutable [`RetrieverTrace`] it records into.
pub struct SelectRequest<'a> {
    /// The model identity to select with, or `None` for the selector's default.
    pub model: Option<&'a str>,
    /// The cursor of the subtree being selected from.
    pub cursor: &'a SearchCursor,
    /// The caller's query messages.
    pub query_messages: &'a [ChatMessage],
    /// The exposed subtree.
    pub subtree: &'a CurrentSubtree,
    /// The rendered prompt bundle.
    pub prompt: &'a PromptBundle,
}

/// Picks the targets of the current subtree.
///
/// Python `TopKSelector`. [`TopKSelector::build_protocol`] describes how an answer is to
/// be read, [`TopKSelector::select_topk`] performs the selection.
#[async_trait]
pub trait TopKSelector: Send + Sync {
    /// Describes the answer format of `subtree`.
    fn build_protocol(&self, subtree: &CurrentSubtree) -> SelectionProtocol;

    /// Selects the targets of `request.subtree`, recording its decision into `trace`.
    ///
    /// # Errors
    ///
    /// The selector's own failures: an unseenable model error under
    /// [`FallbackMode::Error`](crate::llm::FallbackMode), or
    /// [`Error::Unsupported`](crate::error::Error) when the configured mode needs a model
    /// and none is wired.
    async fn select_topk(
        &self,
        request: SelectRequest<'_>,
        trace: &mut RetrieverTrace,
    ) -> Result<SelectionResult>;
}

/// Turns the selected targets into leaf candidates and child cursors.
///
/// Python `TargetExpander`.
pub trait TargetExpander: Send + Sync {
    /// Expands `selected` relative to `cursor`.
    fn expand_selected_targets(
        &self,
        cursor: &SearchCursor,
        selected: &[SelectableTarget],
    ) -> ExpansionPlan;
}

/// Merges one node's own leaves with its child results.
///
/// Python `BranchReducer`.
pub trait BranchReducer: Send + Sync {
    /// Reduces `local_leaves` and `child_results` into one candidate list for `cursor`.
    fn reduce_branch_results(
        &self,
        cursor: &SearchCursor,
        local_leaves: &[RetrieverCandidate],
        child_results: &[NodeSearchResult],
    ) -> NodeSearchResult;
}

/// Everything one model-facing selection needs.
///
/// Rust-only shape: Python passes the same values as keyword arguments to
/// `FragmentSelector.select`. `messages` is the rendered prompt (already carrying the
/// exposed fragment), while `fragment` is the exposure the answer's codes resolve
/// against; `node_id`/`depth`/`top_k` identify the cursor position for tracing and for a
/// callback's own bookkeeping.
pub struct GenerateRequest<'a> {
    /// The model identity to call, or `None` for the callback's default.
    pub model: Option<&'a str>,
    /// The id of the node being selected from.
    pub node_id: &'a str,
    /// The traversal depth of that node.
    pub depth: usize,
    /// How many candidates the selection may keep (`0` = unlimited).
    pub top_k: usize,
    /// The caller's query messages.
    pub query_messages: &'a [ChatMessage],
    /// The exposed fragment the answer's codes resolve against.
    pub fragment: &'a ExposedFragment,
    /// The rendered chat messages to send.
    pub messages: &'a [ChatMessage],
    /// The trace to record model-facing events into.
    pub trace: &'a mut RetrieverTrace,
}

/// Runs one model-facing selection.
///
/// Returns the model's raw answer together with the resolutions it selected, in answer
/// order. The callback owns prompt rendering, the model call and the code parsing; the
/// selector only truncates and interprets the answer.
pub type GenerateFn =
    Arc<dyn Fn(GenerateRequest<'_>) -> Result<(String, Vec<SelectableResolution>)> + Send + Sync>;

/// A [`TopKSelector`] fragment selector that defers to an injected [`GenerateFn`].
///
/// Mirrors Python `GenerateFragmentSelector`: the traversal layer stays model-agnostic,
/// and the engine path injects the callback that renders the disclosure, calls the model
/// and parses its answer.
#[derive(Clone)]
pub struct GenerateFragmentSelector {
    /// The injected model-facing selection callback.
    generate_fn: GenerateFn,
}

impl GenerateFragmentSelector {
    /// Wraps `generate_fn`.
    #[must_use]
    pub fn new(generate_fn: GenerateFn) -> Self {
        Self { generate_fn }
    }

    /// Runs the callback for `request`.
    ///
    /// # Errors
    ///
    /// Whatever the callback returns: the model or prompt failure it reports.
    pub fn generate(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<(String, Vec<SelectableResolution>)> {
        (self.generate_fn)(request)
    }
}

impl std::fmt::Debug for GenerateFragmentSelector {
    /// Renders the selector without its (unprintable) callback.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GenerateFragmentSelector")
            .finish_non_exhaustive()
    }
}
