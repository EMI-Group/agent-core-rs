//! The progressive retriever: the wired-up engine, its seams and the model path.
//!
//! Ports the ENGINE half of Python `openjiuwen.symphony.retrieval.search.runtime.progressive`
//! (v0.1.20) — `search`, `prepare_root`, `_build_recursive_search_engine`,
//! `_build_current_subtree_provider`, `_build_fragment_selector`, `_select_from_fragment_generate`
//! and `_complete` — on top of this crate's seams. Python's legacy direct-recursion path
//! (`_search_node`, `_select_items`) is deliberately not ported; the flat retriever
//! ([`FlatRetriever`](crate::retrieval::search::flat::FlatRetriever)) covers that shape.
//!
//! # Wiring
//!
//! [`ProgressiveRetriever::new`] binds a root and a config to the five seams of
//! [`contracts`](crate::retrieval::search::contracts):
//!
//! | seam | implementation |
//! | --- | --- |
//! | [`CurrentSubtreeProvider`] | [`DefaultCurrentSubtreeProvider`] with the render-backed fragment builder |
//! | [`SubtreeRenderer`] | [`DefaultSubtreeRenderer`] (renders the disclosure `PromptBundle`) |
//! | [`TopKSelector`] | [`LexicalTopKSelector`] by default, [`EmbeddingBlendedTopKSelector`] after [`ProgressiveRetriever::with_embedding`], [`ModelTopKSelector`] after [`ProgressiveRetriever::with_model`] |
//! | [`TargetExpander`] | [`DefaultTargetExpander`] |
//! | [`BranchReducer`] | [`DefaultBranchReducer`] |
//!
//! The provider caches one exposure per `(generation, node, branch path, top_k)`, so
//! [`ProgressiveRetriever::prepare_root`] warms exactly the fragments the walk will ask for and
//! the walk itself then costs no rendering.
//!
//! # The model path
//!
//! [`ModelTopKSelector`] is the async model-facing selector. Per call it
//!
//! 1. builds the prompt with
//!    [`build_disclosure_messages`](crate::retrieval::search::render::build_disclosure_messages)
//!    over the exposed fragment (preferring the messages the engine already rendered),
//! 2. records `llm_request` (`model`, `node_id`, `depth`, `prompt_chars`, `messages`),
//! 3. calls [`CompletionModel::complete`](crate::llm::CompletionModel) with the **single**
//!    prompt that joins the rendered system and user contents with `"\n\n"`, and
//! 4. records `llm_response` (`model`, `node_id`, `reply_chars`), then parses the reply with
//!    [`parse_selected_codes`](crate::retrieval::search::render::parse_selected_codes).
//!
//! Deviation from Python: Python sends the rendered messages as a chat request
//! (`_complete(system_prompt=…, query_messages=[…])`); [`CompletionModel::complete`] takes one
//! prompt, so the port flattens the two messages into one string. The prompt text is the same.
//!
//! The other deviation is the seam itself: this crate's
//! [`GenerateFn`](crate::retrieval::search::contracts::GenerateFn) is synchronous, so an async
//! model call cannot live inside it. [`ModelTopKSelector`] therefore performs the flow itself
//! and keeps the fallback selector for the deterministic path.
//!
//! A model failure propagates to the selector, which honours
//! [`FallbackMode`](crate::llm::FallbackMode): [`FallbackMode::Error`] surfaces it, while
//! [`FallbackMode::Deterministic`] warns and selects with the fallback selector instead.
//!
//! # Methods
//!
//! [`ProgressiveSearchResult::method`] is [`METHOD_PROGRESSIVE`] when the walk actually used the
//! model at least once (a `fragment_selected` event with `mode = "generate"`) and
//! [`METHOD_LEXICAL_FALLBACK`] when it never did (no model configured, or
//! [`SelectionMode::Lexical`]).
//!
//! # Resumability
//!
//! [`ProgressiveRetriever::begin`] returns a serializable
//! [`TraversalState`](crate::retrieval::search::engine::TraversalState);
//! [`ProgressiveRetriever::step`] advances it one node visit at a time. The state carries the
//! query messages, so a caller can save it, restore it later and finish the walk with the same
//! result an uninterrupted [`ProgressiveRetriever::search`] produces (that path additionally
//! closes the trace with `search_complete`).
//!
//! The retriever itself is `&self`-only and reusable: its only interior mutability is the
//! provider's cache (behind the provider's own `RwLock`, never held across an `.await`) and the
//! root binding that [`ProgressiveRetriever::rebind`] replaces.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{validation, Result};
use crate::llm::{CompletionModel, FallbackMode};
use crate::retrieval::search::blend_selector::EmbeddingBlendedTopKSelector;
use crate::retrieval::search::config::{ProgressiveSearchConfig, SelectionMode};
use crate::retrieval::search::contracts::{
    BranchReducer, CurrentSubtreeProvider, FragmentBuildFn, SelectRequest, TargetExpander,
    TopKSelector,
};
use crate::retrieval::search::engine::{
    EngineRequest, RecursiveSearchEngine, TraversalState, TraversalStep,
};
use crate::retrieval::search::expand::DefaultTargetExpander;
use crate::retrieval::search::flow::{
    ChatMessage, CurrentSubtree, SearchCursor, SelectableTarget, SelectionProtocol,
    SelectionResult, ABSTAIN_TOKEN,
};
use crate::retrieval::search::reduce::DefaultBranchReducer;
use crate::retrieval::search::render::{
    build_disclosure_messages, build_disclosure_prompt_parts, build_exposed_fragment,
    parse_selected_codes, DefaultSubtreeRenderer, SubtreeRenderer,
};
use crate::retrieval::search::selector::{
    is_abstain_output, record_selection, DefaultTopKSelector, MODE_GENERATE,
};
use crate::retrieval::search::subtree::DefaultCurrentSubtreeProvider;
use crate::retrieval::types::{RetrieverCandidate, RetrieverNode, RetrieverTrace};
use openjiuwen_retrieval::Embedding;

/// [`ProgressiveSearchResult::method`] when the walk used the model.
pub const METHOD_PROGRESSIVE: &str = "progressive";

/// [`ProgressiveSearchResult::method`] when the walk never used the model.
pub const METHOD_LEXICAL_FALLBACK: &str = "lexical_fallback";

/// The trace event recorded before a model call.
pub const LLM_REQUEST_EVENT: &str = "llm_request";

/// The trace event recorded after a model call.
pub const LLM_RESPONSE_EVENT: &str = "llm_response";

/// The result of one [`ProgressiveRetriever`] search.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ProgressiveSearchResult {
    /// The ranked candidates, best first.
    pub candidates: Vec<RetrieverCandidate>,
    /// The trace of the walk (the same trace the state carries).
    pub trace: RetrieverTrace,
    /// [`METHOD_PROGRESSIVE`] or [`METHOD_LEXICAL_FALLBACK`].
    pub method: String,
    /// How many nodes were visited.
    pub steps: usize,
    /// The final (exhausted) walk state, so a caller can inspect or resume it.
    pub state: TraversalState,
}

/// The model a [`ProgressiveRetriever`] selects with.
#[derive(Clone)]
struct ModelSelector {
    /// The model to call.
    model: Arc<dyn CompletionModel>,
    /// The model identity the trace and the request carry.
    name: String,
}

/// The progressive retriever: config, seams, root and the optional model.
pub struct ProgressiveRetriever {
    /// The bound root, replaceable by [`ProgressiveRetriever::rebind`].
    root: RwLock<Arc<RetrieverNode>>,
    /// The traversal config.
    config: ProgressiveSearchConfig,
    /// The exposure seam (caches one fragment per cursor).
    provider: Arc<DefaultCurrentSubtreeProvider>,
    /// The prompt rendering seam.
    renderer: Arc<dyn SubtreeRenderer>,
    /// The fallback selector (lexical, or embedding-blended when an embedder is configured).
    selector: Arc<dyn TopKSelector>,
    /// The expansion seam.
    expander: Arc<dyn TargetExpander>,
    /// The reduction seam.
    reducer: Arc<dyn BranchReducer>,
    /// The model-facing selector, when a model is configured.
    model: Option<ModelSelector>,
}

impl ProgressiveRetriever {
    /// Binds `root` and `config` to the five seams.
    ///
    /// Without a model the walk selects lexically (or with the embedding blend added by
    /// [`ProgressiveRetriever::with_embedding`]).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) when `config` is invalid (a `top_k` of `0`,
    /// an unusable depth or step bound) or when `root` has a blank node id.
    pub fn new(root: Arc<RetrieverNode>, config: ProgressiveSearchConfig) -> Result<Self> {
        config.validate()?;
        let builder: FragmentBuildFn = Arc::new(
            |node: &RetrieverNode,
             branch_path: &[String],
             disclosure: &crate::retrieval::search::config::DisclosureConfig,
             item_count: usize| {
                let counter = |_: &RetrieverNode| item_count;
                build_exposed_fragment(node, branch_path, disclosure, &counter)
            },
        );
        let provider = DefaultCurrentSubtreeProvider::with_builder(
            Arc::clone(&root),
            config.clone(),
            builder,
        )?;
        let selector: Arc<dyn TopKSelector> = Arc::new(DefaultTopKSelector::new(config.clone()));
        Ok(Self {
            root: RwLock::new(root),
            config: config.clone(),
            provider: Arc::new(provider),
            renderer: Arc::new(DefaultSubtreeRenderer::new()),
            selector,
            expander: Arc::new(DefaultTargetExpander::new(config.clone())),
            reducer: Arc::new(DefaultBranchReducer::new(config)),
            model: None,
        })
    }

    /// Wires a model, turning the walk into the generate path.
    #[must_use]
    pub fn with_model(
        mut self,
        model: Arc<dyn CompletionModel>,
        model_name: impl Into<String>,
    ) -> Self {
        self.model = Some(ModelSelector {
            model,
            name: model_name.into(),
        });
        self
    }

    /// Makes the fallback selector an embedding blend.
    ///
    /// The blend is used whenever the model path is not taken (no model, lexical mode, or a
    /// deterministic fallback) and its weight comes from
    /// [`EmbeddingScoringConfig::weight`](crate::retrieval::search::config::EmbeddingScoringConfig).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) when the embedding config is unusable (a
    /// weight outside `[0, 1]`).
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Result<Self> {
        self.config.embedding.validate()?;
        self.selector = Arc::new(EmbeddingBlendedTopKSelector::new(
            self.config.clone(),
            embedding,
        ));
        Ok(self)
    }

    /// Rebinds the retriever to `root` after a catalog rebuild.
    ///
    /// The provider's exposure cache is generation-keyed, so rebinding invalidates every cached
    /// fragment instead of serving stale ones.
    pub fn rebind(&self, root: Arc<RetrieverNode>) {
        {
            let mut binding = self
                .root
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *binding = Arc::clone(&root);
        }
        self.provider.rebind(root);
    }

    /// The config the retriever runs with.
    #[must_use]
    pub fn config(&self) -> &ProgressiveSearchConfig {
        &self.config
    }

    /// The bound root.
    #[must_use]
    pub fn root(&self) -> Arc<RetrieverNode> {
        Arc::clone(
            &self
                .root
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// The configured model's name, if any.
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model.as_ref().map(|model| model.name.as_str())
    }

    /// How many exposures the provider currently caches.
    #[must_use]
    pub fn warmed_fragments(&self) -> usize {
        self.provider.cached_len()
    }

    /// Warms one exposure per reachable node and returns how many nodes were warmed.
    ///
    /// Python's `prepare_root` walks every node pre-order, builds its subtree and (when it has
    /// selectable targets) renders the disclosure prompt so the first real search hits warm
    /// caches. The port does the same and reports the number of exposed nodes — every reachable
    /// node is exposed and cached, so this is the walking size of the tree.
    ///
    /// # Errors
    ///
    /// Whatever the provider or the renderer reports for a node.
    pub async fn prepare_root(&self, top_k: Option<usize>) -> Result<usize> {
        let root = self.root();
        let top_k = self.resolve_top_k(top_k);
        let mut warmed = 0_usize;
        for (node_id, chain) in node_chains(&root) {
            let depth = chain.len().saturating_sub(1);
            let cursor = SearchCursor::at(node_id, depth, chain, top_k);
            let subtree = self.provider.get_current_subtree(&cursor)?;
            if !subtree.selectable_targets.is_empty() {
                let _parts = build_disclosure_prompt_parts(&subtree.fragment, &[], Some(top_k));
            }
            warmed += 1;
        }
        Ok(warmed)
    }

    /// Searches for `query`, selecting with the configured seams.
    ///
    /// `top_k` falls back to [`ProgressiveSearchConfig::top_k`]; `0` means unlimited. The query
    /// text becomes one `user` message (Python prefixes a `From User:` sender marker; the port
    /// keeps the caller's words verbatim so the lexical fallback ranks exactly what was asked).
    ///
    /// # Errors
    ///
    /// The seams' failures, including
    /// [`Error::Unsupported`](crate::error::Error) when the selection mode needs a model that is
    /// not configured under [`FallbackMode::Error`](crate::llm::FallbackMode).
    pub async fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
    ) -> Result<ProgressiveSearchResult> {
        self.search_messages(&[ChatMessage::user(query)], top_k)
            .await
    }

    /// Searches for a conversation, selecting with the configured seams.
    ///
    /// # Errors
    ///
    /// See [`ProgressiveRetriever::search`].
    pub async fn search_messages(
        &self,
        query_messages: &[ChatMessage],
        top_k: Option<usize>,
    ) -> Result<ProgressiveSearchResult> {
        let mut state = self.begin(query_messages.to_vec(), top_k)?;
        let messages = state.query_messages().to_vec();
        let request = EngineRequest {
            model: self.model_name(),
            query_messages: &messages,
        };
        self.engine().search(&request, &mut state).await?;
        Ok(self.finish(state))
    }

    /// Starts a resumable walk for `query_messages`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) when the resolved `top_k` is unusable.
    pub fn begin(
        &self,
        query_messages: Vec<ChatMessage>,
        top_k: Option<usize>,
    ) -> Result<TraversalState> {
        let root = self.root();
        let cursor = SearchCursor::new(&root, self.resolve_top_k(top_k));
        Ok(TraversalState::new(cursor).with_query_messages(query_messages))
    }

    /// Advances a walk by one node visit; `Ok(None)` when it is exhausted.
    ///
    /// A walk driven by [`ProgressiveRetriever::begin`] and this method does not record
    /// `search_complete`: that event closes [`ProgressiveRetriever::search`]. Reduce a finished
    /// state with [`ProgressiveSearchResult`]-shaped accessors (`state.candidates()`,
    /// `state.trace()`) or call [`ProgressiveRetriever::finish`].
    ///
    /// # Errors
    ///
    /// The seams' failures for the visited node.
    pub async fn step(&self, state: &mut TraversalState) -> Result<Option<TraversalStep>> {
        let messages = state.query_messages().to_vec();
        let request = EngineRequest {
            model: self.model_name(),
            query_messages: &messages,
        };
        self.engine().step(&request, state).await
    }

    /// Assembles the result of a finished (or truncated) walk.
    #[must_use]
    pub fn finish(&self, state: TraversalState) -> ProgressiveSearchResult {
        let method = if used_the_model(&state) {
            METHOD_PROGRESSIVE
        } else {
            METHOD_LEXICAL_FALLBACK
        };
        ProgressiveSearchResult {
            candidates: state.candidates().to_vec(),
            trace: state.trace().clone(),
            method: method.to_owned(),
            steps: state.steps(),
            state,
        }
    }

    /// The engine over the current seams.
    fn engine(&self) -> RecursiveSearchEngine {
        let selector: Arc<dyn TopKSelector> = match self.model.clone() {
            Some(model) => Arc::new(ModelTopKSelector::new(
                model.model,
                model.name,
                self.config.clone(),
                Arc::clone(&self.selector),
            )),
            None => Arc::clone(&self.selector),
        };
        RecursiveSearchEngine::new(
            Arc::clone(&self.provider) as Arc<dyn CurrentSubtreeProvider>,
            Arc::clone(&self.renderer),
            selector,
            Arc::clone(&self.expander),
            Arc::clone(&self.reducer),
            self.config.clone(),
        )
    }

    /// Resolves a caller's `top_k` against the config (`0` stays the unlimited sentinel).
    fn resolve_top_k(&self, top_k: Option<usize>) -> usize {
        top_k.unwrap_or(self.config.top_k)
    }
}

impl std::fmt::Debug for ProgressiveRetriever {
    /// Renders the config, the model name and the warmed cache size.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProgressiveRetriever")
            .field("config", &self.config)
            .field("model", &self.model_name())
            .field("warmed_fragments", &self.warmed_fragments())
            .finish_non_exhaustive()
    }
}

/// The async model-facing [`TopKSelector`].
///
/// See the module docs for the per-call flow. Every call ends in exactly one
/// `fragment_selected` event: `mode = "generate"` for a model answer, and the fallback
/// selector's own mode when the model failed under
/// [`FallbackMode::Deterministic`](crate::llm::FallbackMode).
pub struct ModelTopKSelector {
    /// The model to call.
    model: Arc<dyn CompletionModel>,
    /// The model identity the trace and the request carry.
    model_name: String,
    /// The traversal config (selection mode, fallback mode, disclosure and truncation).
    config: ProgressiveSearchConfig,
    /// The selector used outside the model path.
    fallback: Arc<dyn TopKSelector>,
}

impl ModelTopKSelector {
    /// Builds the selector over `model`, its reported name, and `fallback`.
    #[must_use]
    pub fn new(
        model: Arc<dyn CompletionModel>,
        model_name: impl Into<String>,
        config: ProgressiveSearchConfig,
        fallback: Arc<dyn TopKSelector>,
    ) -> Self {
        Self {
            model,
            model_name: model_name.into(),
            config,
            fallback,
        }
    }

    /// The model identity the selector reports.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Asks the model which codes of `request`'s fragment to take.
    ///
    /// Records `llm_request` before the call and `llm_response` after it, then parses the reply
    /// against the fragment. The reply is returned verbatim, so the caller can report it.
    async fn complete(
        &self,
        request: &SelectRequest<'_>,
        trace: &mut RetrieverTrace,
    ) -> Result<(String, Vec<SelectableTarget>)> {
        let cursor = request.cursor;
        let messages = if request.prompt.messages.is_empty() {
            build_disclosure_messages(
                &request.subtree.fragment,
                request.query_messages,
                Some(cursor.top_k),
            )
        } else {
            request.prompt.messages.clone()
        };
        let prompt = prompt_text(&messages);
        let detail = BTreeMap::from([
            ("model".to_owned(), json!(self.model_name)),
            ("node_id".to_owned(), json!(cursor.node_id)),
            ("depth".to_owned(), json!(cursor.depth)),
            ("prompt_chars".to_owned(), json!(prompt.chars().count())),
            ("messages".to_owned(), json!(messages)),
        ]);
        trace.record(LLM_REQUEST_EVENT, &cursor.node_id, cursor.depth, detail);
        let reply = self.model.complete(&prompt).await?;
        trace.record(
            LLM_RESPONSE_EVENT,
            &cursor.node_id,
            cursor.depth,
            BTreeMap::from([
                ("model".to_owned(), json!(self.model_name)),
                ("node_id".to_owned(), json!(cursor.node_id)),
                ("reply_chars".to_owned(), json!(reply.chars().count())),
            ]),
        );
        let resolutions = parse_selected_codes(&request.subtree.fragment, &reply);
        let targets = resolutions.into_iter().map(SelectableTarget::new).collect();
        Ok((reply, targets))
    }
}

#[async_trait]
impl TopKSelector for ModelTopKSelector {
    /// Describes how to read the model's answer.
    fn build_protocol(&self, subtree: &CurrentSubtree) -> SelectionProtocol {
        SelectionProtocol {
            compact_codes_enabled: subtree.fragment.compact_codes_enabled,
            candidate_codes: subtree.fragment.candidate_codes.clone(),
            code_width: subtree.fragment.code_width,
            abstain_token: ABSTAIN_TOKEN.to_owned(),
        }
    }

    /// Selects the targets of `request.subtree` with the model, or with the fallback.
    ///
    /// # Errors
    ///
    /// The model's error under [`FallbackMode::Error`](crate::llm::FallbackMode); under
    /// [`FallbackMode::Deterministic`](crate::llm::FallbackMode) the error is warned and the
    /// fallback selector answers instead.
    async fn select_topk(
        &self,
        request: SelectRequest<'_>,
        trace: &mut RetrieverTrace,
    ) -> Result<SelectionResult> {
        if self.config.selection_mode == SelectionMode::Lexical {
            return self.fallback.select_topk(request, trace).await;
        }
        let cursor = request.cursor;
        let outcome = self.complete(&request, trace).await;
        let (raw_output, mut targets) = match outcome {
            Ok(answer) => answer,
            Err(error) => match self.config.fallback_mode {
                FallbackMode::Error => return Err(error),
                FallbackMode::Deterministic => {
                    tracing::warn!(
                        node_id = %cursor.node_id,
                        depth = cursor.depth,
                        model = %self.model_name,
                        error = %error,
                        "progressive selection failed; falling back to the lexical selector"
                    );
                    return self.fallback.select_topk(request, trace).await;
                }
            },
        };
        let limit = if cursor.top_k == 0 {
            usize::MAX
        } else {
            cursor.top_k
        };
        targets.truncate(limit);
        let is_abstain = is_abstain_output(&raw_output);
        if is_abstain {
            targets.clear();
        }
        let result = SelectionResult {
            raw_output,
            selected_targets: targets,
            is_abstain,
        };
        record_selection(trace, cursor, MODE_GENERATE, &result);
        Ok(result)
    }
}

impl std::fmt::Debug for ModelTopKSelector {
    /// Renders the model name and the selection policy.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelTopKSelector")
            .field("model", &self.model_name)
            .field("selection_mode", &self.config.selection_mode)
            .field("fallback_mode", &self.config.fallback_mode)
            .finish_non_exhaustive()
    }
}

/// Joins a rendered conversation into the single prompt a [`CompletionModel`] takes.
fn prompt_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|message| message.content.as_str())
        .filter(|content| !content.trim().is_empty())
        .collect::<Vec<&str>>()
        .join("\n\n")
}

/// Whether the walk used the model at least once.
fn used_the_model(state: &TraversalState) -> bool {
    state.trace().events().iter().any(|event| {
        event.event_type == crate::retrieval::search::selector::FRAGMENT_SELECTED_EVENT
            && event.detail.get("mode") == Some(&json!(MODE_GENERATE))
    })
}

/// The nodes of `root` with the id chain that reaches each, pre-order.
fn node_chains(root: &RetrieverNode) -> Vec<(String, Vec<String>)> {
    let mut chains = Vec::new();
    let mut stack: Vec<(&RetrieverNode, Vec<String>)> = vec![(root, vec![root.node_id.clone()])];
    while let Some((node, chain)) = stack.pop() {
        chains.push((node.node_id.clone(), chain.clone()));
        for child in node.children().into_iter().rev() {
            let mut child_chain = chain.clone();
            child_chain.push(child.node_id.clone());
            stack.push((child, child_chain));
        }
    }
    chains
}

/// Rejects a blank query, naming the field.
///
/// Shared with the facade wave, which validates the same input.
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error) naming `query`.
pub fn validate_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(validation("query", "must be non-empty"));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::llm::StaticCompletionModel;
    use crate::retrieval::search::config::{DisclosureConfig, EmbeddingScoringConfig};
    use crate::retrieval::types::{RetrieverItem, ROOT_NODE_ID};

    /// A three-level tree whose leaves are reachable only by descending.
    fn tree() -> RetrieverNode {
        RetrieverNode::new(ROOT_NODE_ID, "Root").with_children(vec![
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

    /// The config the tests share: branches are selectable, items are leaves.
    fn config() -> ProgressiveSearchConfig {
        ProgressiveSearchConfig {
            top_k: 3,
            disclosure: DisclosureConfig {
                max_exposure_depth_per_call: 0,
                exposure_threshold: 100,
                compact_codes_enabled: true,
                ..DisclosureConfig::default()
            },
            ..ProgressiveSearchConfig::default()
        }
    }

    fn fresh() -> ProgressiveRetriever {
        ProgressiveRetriever::new(Arc::new(tree()), config()).unwrap()
    }

    fn retriever() -> ProgressiveRetriever {
        fresh()
    }

    #[tokio::test]
    async fn a_lexical_walk_descends_and_reports_the_fallback_method() {
        let retriever = retriever();
        let result = retriever
            .search("alpha documentation", Some(3))
            .await
            .unwrap();
        assert_eq!(result.method, METHOD_LEXICAL_FALLBACK);
        let ids: Vec<&str> = result
            .candidates
            .iter()
            .map(|candidate| candidate.item_id.as_str())
            .collect();
        assert_eq!(ids, vec!["alpha.one", "alpha.two"]);
        assert_eq!(result.steps, 2);
        assert!(result.state.is_exhausted());
        assert_eq!(result.trace, *result.state.trace());
        assert_eq!(
            result.trace.event_types().last().copied(),
            Some(crate::retrieval::search::engine::SEARCH_COMPLETE_EVENT)
        );
    }

    #[tokio::test]
    async fn prepare_root_warms_every_node_and_keeps_the_result() {
        let retriever = retriever();
        let warmed = retriever.prepare_root(None).await.unwrap();
        assert_eq!(warmed, 3, "root plus two branches");
        assert_eq!(retriever.warmed_fragments(), 3);
        let warm = retriever.search("alpha", Some(3)).await.unwrap();
        let cold = fresh().search("alpha", Some(3)).await.unwrap();
        assert_eq!(warm.candidates, cold.candidates);
        assert_eq!(warm.trace.event_types(), cold.trace.event_types());
    }

    #[tokio::test]
    async fn a_scripted_model_drives_the_progressive_path() {
        // The root exposes two branches: code 1 = Skills.alpha, code 2 = Skills.beta. The model
        // takes the first, then both of Skills.alpha's items.
        let model = Arc::new(StaticCompletionModel::new(vec![
            "1".to_owned(),
            "1\n2".to_owned(),
        ]));
        let retriever = retriever().with_model(model.clone(), "scripted");
        let result = retriever.search("alpha", Some(3)).await.unwrap();
        assert_eq!(result.method, METHOD_PROGRESSIVE);
        let ids: Vec<&str> = result
            .candidates
            .iter()
            .map(|candidate| candidate.item_id.as_str())
            .collect();
        assert_eq!(ids, vec!["alpha.one", "alpha.two"]);
        assert_eq!(model.call_count(), 2);
        let events = result.trace.event_types();
        assert!(events.contains(&LLM_REQUEST_EVENT));
        assert!(events.contains(&LLM_RESPONSE_EVENT));
        assert_eq!(retriever.model_name(), Some("scripted"));
    }

    #[tokio::test]
    async fn an_error_mode_walk_without_a_model_is_unsupported() {
        let config = ProgressiveSearchConfig {
            fallback_mode: FallbackMode::Error,
            ..config()
        };
        let retriever = ProgressiveRetriever::new(Arc::new(tree()), config.clone()).unwrap();
        let error = retriever.search("alpha", None).await.unwrap_err();
        assert!(crate::error::is_unsupported(&error), "{error}");
        let model_retriever =
            retriever.with_model(Arc::new(StaticCompletionModel::single("")), "scripted");
        let lexical = ProgressiveSearchConfig {
            selection_mode: SelectionMode::Lexical,
            ..config
        };
        let lexical_retriever = ProgressiveRetriever::new(Arc::new(tree()), lexical).unwrap();
        assert!(lexical_retriever.search("alpha", None).await.is_ok());
        assert_eq!(
            model_retriever.search("alpha", None).await.unwrap().method,
            METHOD_PROGRESSIVE
        );
    }

    #[tokio::test]
    async fn a_deterministic_fallback_answers_when_the_model_fails() {
        struct Failing;
        #[async_trait]
        impl CompletionModel for Failing {
            async fn complete(&self, _prompt: &str) -> Result<String> {
                Err(crate::error::Error::llm("model unavailable"))
            }
            fn name(&self) -> &str {
                "failing"
            }
        }
        let retriever = retriever().with_model(Arc::new(Failing), "failing");
        let result = retriever
            .search("alpha documentation", Some(3))
            .await
            .unwrap();
        assert_eq!(result.method, METHOD_LEXICAL_FALLBACK);
        assert!(!result.candidates.is_empty());
        assert!(result.trace.event_types().contains(&LLM_REQUEST_EVENT));
    }

    #[tokio::test]
    async fn the_embedding_blend_replaces_the_lexical_fallback() {
        let mut config = config();
        config.embedding = EmbeddingScoringConfig::default().with_weight(0.5);
        let retriever = ProgressiveRetriever::new(Arc::new(tree()), config)
            .unwrap()
            .with_embedding(Arc::new(openjiuwen_retrieval::HashEmbedding::new(32)))
            .unwrap();
        let result = retriever.search("alpha", Some(3)).await.unwrap();
        assert_eq!(result.method, METHOD_LEXICAL_FALLBACK);
        assert!(!result.candidates.is_empty());
        assert_eq!(
            result.trace.events()[1].detail["mode"],
            json!(crate::retrieval::search::blend_selector::MODE_EMBEDDING_BLEND)
        );
    }

    #[tokio::test]
    async fn rebinding_clears_the_exposure_cache() {
        let retriever = retriever();
        retriever.prepare_root(None).await.unwrap();
        assert!(retriever.warmed_fragments() > 0);
        retriever.rebind(Arc::new(RetrieverNode::new(ROOT_NODE_ID, "Root")));
        assert_eq!(retriever.warmed_fragments(), 0);
        assert_eq!(retriever.root().node_id, ROOT_NODE_ID);
        assert!(retriever
            .search("alpha", None)
            .await
            .unwrap()
            .candidates
            .is_empty());
    }

    #[tokio::test]
    async fn a_blank_query_validates_through_the_shared_helper() {
        assert!(validate_query("  ").is_err());
        assert!(validate_query("alpha").is_ok());
    }

    #[test]
    fn the_prompt_is_the_single_flattened_conversation() {
        let messages = vec![ChatMessage::system("system"), ChatMessage::user("user")];
        assert_eq!(prompt_text(&messages), "system\n\nuser");
        assert_eq!(prompt_text(&[]), "");
        assert_eq!(prompt_text(&[ChatMessage::user("  ")]), "");
    }
}
