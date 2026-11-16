//! The Rust-only embedding-blended top-k selector.
//!
//! Python's progressive runtime has two selection paths: the model-facing generate selector and
//! the deterministic lexical selector. This module adds a third, Rust-only one that keeps the
//! lexical ranking as its backbone and nudges it with embedding similarity, so an offline
//! catalog can be browsed semantically without a model.
//!
//! # Formula
//!
//! Given the lexical hits `L` over the subtree's selectable targets (the same documents
//! [`LexicalTopKSelector`](crate::retrieval::search::selector::LexicalTopKSelector) builds) and
//! the embedding cosine `c` of each target:
//!
//! ```text
//! lex_norm = lex_score / max(lex_score)           // 0..1, the best lexical hit is 1.0
//! emb      = clamp((c + 1) / 2, 0.0, 1.0)         // -1..1 cosine mapped onto 0..1
//! blended  = (1 - w) * lex_norm + w * emb         // w = EmbeddingScoringConfig::weight
//! ```
//!
//! [`blend_scores`] implements exactly that
//! over the union of both key sets, sorted by score descending with ties broken by key. With
//! `w == 0.0` the formula degenerates to `lex_norm`, which is a monotone transform of the
//! lexical score — so `weight == 0.0` reproduces the lexical selector's order exactly (and
//! skips the embedder entirely).
//!
//! The blended list then passes through [`truncate_sorted_by_score`] with the config's
//! [`ScoreTruncationConfig`](crate::retrieval::search::truncation::ScoreTruncationConfig), and
//! is bounded by the cursor's `top_k` (`0` = unlimited).
//!
//! # Trace
//!
//! One `fragment_selected` event per call, with the same four detail keys the other selectors
//! record (`mode`, `raw_output`, `selected_canonical_ids`, `selected_codes`) and
//! `mode = "embedding_blend"`.
//!
//! # Determinism
//!
//! Documents are embedded in key order, the blend sorts on `(score, key)`, and the selection
//! preserves that order, so the same catalog and query always select the same targets. The
//! embedder is only touched when the weight is positive, and its store is shared across node
//! visits through the selector instance (a node's vectors are reused when it is revisited).
//!
//! No lock is held across an `.await`: [`EmbeddingScorer`] takes its own snapshot per call.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_retrieval::Embedding;

use crate::error::Result;
use crate::retrieval::search::config::ProgressiveSearchConfig;
use crate::retrieval::search::contracts::{SelectRequest, TopKSelector};
use crate::retrieval::search::embedding::{blend_scores, EmbeddingScorer};
use crate::retrieval::search::flow::{
    CurrentSubtree, SelectableTarget, SelectionProtocol, SelectionProvenance, SelectionResult,
    ABSTAIN_TOKEN,
};
use crate::retrieval::search::lexical::{LexicalDocument, LexicalIndex};
use crate::retrieval::search::selector::{
    document, query_text, ranking_expression, record_selection,
};
use crate::retrieval::search::truncation::truncate_sorted_by_score;
use crate::retrieval::types::RetrieverTrace;

/// The `fragment_selected` mode the blend selector records.
pub const MODE_EMBEDDING_BLEND: &str = "embedding_blend";

/// A [`TopKSelector`] that blends the lexical ranking with embedding similarity.
///
/// This is the selector [`ProgressiveRetriever`](crate::retrieval::search::progressive::ProgressiveRetriever)
/// installs as its fallback when an embedder is configured; it never calls a model, so it is
/// safe for a walk that must stay offline and deterministic.
#[derive(Debug)]
pub struct EmbeddingBlendedTopKSelector {
    /// The traversal config (the embedding weight, the document cap and the truncation).
    config: ProgressiveSearchConfig,
    /// The embedder-backed scorer over the targets seen so far.
    scorer: EmbeddingScorer,
}

impl EmbeddingBlendedTopKSelector {
    /// Builds the selector over `config` and `embedding`.
    #[must_use]
    pub fn new(config: ProgressiveSearchConfig, embedding: Arc<dyn Embedding>) -> Self {
        Self {
            config,
            scorer: EmbeddingScorer::new(embedding),
        }
    }

    /// The config the selector runs with.
    #[must_use]
    pub fn config(&self) -> &ProgressiveSearchConfig {
        &self.config
    }

    /// The embedding weight of the blend (`0.0` reproduces the lexical order).
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.config.embedding.weight
    }

    /// How many target vectors the scorer holds (diagnostics and tests).
    #[must_use]
    pub fn indexed_documents(&self) -> usize {
        self.scorer.len()
    }

    /// Selects at most `top_k` of `subtree`'s targets for `query`.
    ///
    /// `top_k == 0` means unlimited. The result never marks itself as an abstention: a blend
    /// that finds nothing has no opinion, whereas an abstention is a model's explicit "nothing
    /// relevant here".
    ///
    /// # Errors
    ///
    /// The embedder's failure, or [`Error::Retrieval`](crate::error::Error) when it returns an
    /// unusable vector. The lexical part alone cannot fail once the query is non-blank.
    pub async fn select(
        &self,
        subtree: &CurrentSubtree,
        query: &str,
        top_k: usize,
    ) -> Result<SelectionResult> {
        if subtree.selectable_targets.is_empty() || query.trim().is_empty() {
            return Ok(SelectionResult::default());
        }
        let by_canonical_id: BTreeMap<&str, &SelectableTarget> = subtree
            .selectable_targets
            .iter()
            .map(|target| (target.canonical_id(), target))
            .collect();
        let documents: Vec<LexicalDocument> =
            subtree.selectable_targets.iter().map(document).collect();
        let index = LexicalIndex::build(documents.iter().cloned());
        let hits = index.search(&ranking_expression(query))?;
        let scores = if self.config.embedding.weight > 0.0 {
            self.scorer
                .index_capped(&documents, self.config.embedding.max_documents)
                .await?;
            self.scorer.scores(query).await?
        } else {
            BTreeMap::new()
        };
        let blended = blend_scores(&hits, &scores, self.config.embedding.weight);
        let (kept, _decision) =
            truncate_sorted_by_score(&blended, |hit| hit.score, &self.config.truncation);
        let limit = if top_k == 0 { usize::MAX } else { top_k };
        let selected_targets = kept
            .iter()
            .filter_map(|hit| by_canonical_id.get(hit.key.as_str()).copied().cloned())
            .take(limit)
            .collect();
        Ok(SelectionResult {
            raw_output: String::new(),
            selected_targets,
            is_abstain: false,
            ..SelectionResult::default()
        }
        .with_provenance(SelectionProvenance::EmbeddingBlend))
    }
}

#[async_trait]
impl TopKSelector for EmbeddingBlendedTopKSelector {
    /// Describes how to read the model's answer.
    ///
    /// The blend selector never asks a model, but the engine renders the same disclosure for
    /// every selector, so it reports the fragment's own protocol.
    fn build_protocol(&self, subtree: &CurrentSubtree) -> SelectionProtocol {
        SelectionProtocol {
            compact_codes_enabled: subtree.fragment.compact_codes_enabled,
            candidate_codes: subtree.fragment.candidate_codes.clone(),
            code_width: subtree.fragment.code_width,
            abstain_token: ABSTAIN_TOKEN.to_owned(),
        }
    }

    /// Selects the targets of `request.subtree` and records one `fragment_selected` event.
    ///
    /// # Errors
    ///
    /// The embedder's failure (see [`Self::select`]).
    async fn select_topk(
        &self,
        request: SelectRequest<'_>,
        trace: &mut RetrieverTrace,
    ) -> Result<SelectionResult> {
        let cursor = request.cursor;
        let result = self
            .select(
                request.subtree,
                &query_text(request.query_messages),
                cursor.top_k,
            )
            .await?;
        record_selection(trace, cursor, MODE_EMBEDDING_BLEND, &result);
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::retrieval::search::config::DisclosureConfig;
    use crate::retrieval::search::contracts::CurrentSubtreeProvider;
    use crate::retrieval::search::subtree::DefaultCurrentSubtreeProvider;
    use crate::retrieval::types::{RetrieverItem, RetrieverNode, ROOT_NODE_ID};
    use openjiuwen_retrieval::HashEmbedding;
    use std::sync::Arc;

    /// A flat root with four labelled items.
    fn tree() -> RetrieverNode {
        RetrieverNode::new(ROOT_NODE_ID, "Root").with_items(vec![
            RetrieverItem::new("alpha.one", "alpha.one")
                .with_label("Alpha writer")
                .with_description("writes alpha documentation"),
            RetrieverItem::new("alpha.two", "alpha.two")
                .with_label("Alpha tester")
                .with_description("tests alpha behaviour"),
            RetrieverItem::new("beta.one", "beta.one")
                .with_label("Beta compute")
                .with_description("runs beta jobs"),
            RetrieverItem::new("gamma.one", "gamma.one")
                .with_label("Gamma viewer")
                .with_description("views gamma output"),
        ])
    }

    /// The config the tests share.
    fn config(weight: f64) -> ProgressiveSearchConfig {
        ProgressiveSearchConfig {
            top_k: 3,
            disclosure: DisclosureConfig {
                max_exposure_depth_per_call: 0,
                exposure_threshold: 100,
                compact_codes_enabled: true,
                ..DisclosureConfig::default()
            },
            embedding: crate::retrieval::search::config::EmbeddingScoringConfig::default()
                .with_weight(weight),
            ..ProgressiveSearchConfig::default()
        }
    }

    /// The render-backed fragment builder every provider in these tests uses.
    fn builder() -> crate::retrieval::search::contracts::FragmentBuildFn {
        Arc::new(
            |node: &RetrieverNode,
             branch_path: &[String],
             disclosure: &DisclosureConfig,
             item_count: usize| {
                let counter = |_: &RetrieverNode| item_count;
                crate::retrieval::search::render::build_exposed_fragment(
                    node,
                    branch_path,
                    disclosure,
                    &counter,
                )
            },
        )
    }

    /// The subtree of `tree`'s root under `config`.
    fn subtree(tree: &RetrieverNode, config: &ProgressiveSearchConfig) -> CurrentSubtree {
        let provider = DefaultCurrentSubtreeProvider::with_builder(
            Arc::new(tree.clone()),
            config.clone(),
            builder(),
        )
        .unwrap();
        provider
            .get_current_subtree(&crate::retrieval::search::flow::SearchCursor::new(tree, 3))
            .unwrap()
    }

    fn codes(result: &SelectionResult) -> Vec<&str> {
        result
            .selected_targets
            .iter()
            .map(SelectableTarget::code)
            .collect()
    }

    fn canonical_ids(result: &SelectionResult) -> Vec<&str> {
        result
            .selected_targets
            .iter()
            .map(SelectableTarget::canonical_id)
            .collect()
    }

    #[tokio::test]
    async fn a_zero_weight_reproduces_the_lexical_order() {
        let tree = tree();
        let config = config(0.0);
        let subtree = subtree(&tree, &config);
        let selector =
            EmbeddingBlendedTopKSelector::new(config.clone(), Arc::new(HashEmbedding::new(16)));
        let blended = selector
            .select(&subtree, "alpha documentation", 3)
            .await
            .unwrap();
        let lexical = crate::retrieval::search::selector::LexicalTopKSelector::new(config).select(
            &subtree,
            &[crate::retrieval::search::flow::ChatMessage::user(
                "alpha documentation",
            )],
            3,
        );
        assert_eq!(canonical_ids(&blended), canonical_ids(&lexical));
        assert_eq!(
            selector.indexed_documents(),
            0,
            "a zero weight never touches the embedder"
        );
        assert_eq!(codes(&blended), vec!["1", "2"]);
    }

    #[tokio::test]
    async fn the_blend_is_deterministic_and_weighted() {
        let tree = tree();
        let config = config(1.0);
        let subtree = subtree(&tree, &config);
        let selector =
            EmbeddingBlendedTopKSelector::new(config.clone(), Arc::new(HashEmbedding::new(32)));
        let first = selector.select(&subtree, "alpha", 0).await.unwrap();
        let second = selector.select(&subtree, "alpha", 0).await.unwrap();
        assert_eq!(first, second);
        assert_eq!(
            selector.indexed_documents(),
            subtree.selectable_targets.len(),
            "every target was embedded"
        );
        assert_ne!(first.selected_targets.len(), 0, "the blend selects targets");
        assert_eq!(
            EmbeddingBlendedTopKSelector::new(config, Arc::new(HashEmbedding::new(32))).weight(),
            1.0
        );
    }

    #[tokio::test]
    async fn the_selector_records_one_blend_event() {
        let tree = tree();
        let config = config(0.5);
        let subtree = subtree(&tree, &config);
        let selector = EmbeddingBlendedTopKSelector::new(config, Arc::new(HashEmbedding::new(16)));
        let cursor = subtree.cursor.clone();
        let messages = vec![crate::retrieval::search::flow::ChatMessage::user("alpha")];
        let prompt = crate::retrieval::search::flow::PromptBundle::default();
        let mut trace = RetrieverTrace::default();
        let result = selector
            .select_topk(
                SelectRequest {
                    model: None,
                    cursor: &cursor,
                    query_messages: &messages,
                    subtree: &subtree,
                    prompt: &prompt,
                },
                &mut trace,
            )
            .await
            .unwrap();
        assert_ne!(
            result.selected_targets.len(),
            0,
            "the blend selects targets"
        );
        assert_eq!(trace.len(), 1);
        let event = trace.last().unwrap();
        assert_eq!(event.event_type, "fragment_selected");
        assert_eq!(
            event.detail["mode"],
            serde_json::json!(MODE_EMBEDDING_BLEND)
        );
        assert_eq!(event.node_id, ROOT_NODE_ID);
        assert_eq!(
            event.detail["selected_codes"],
            serde_json::json!(codes(&result))
        );
        assert_eq!(
            event.detail["selected_canonical_ids"],
            serde_json::json!(canonical_ids(&result))
        );
    }

    #[tokio::test]
    async fn a_blank_query_or_an_empty_subtree_selects_nothing() {
        let tree = tree();
        let config = config(0.5);
        let subtree = subtree(&tree, &config);
        let selector = EmbeddingBlendedTopKSelector::new(config, Arc::new(HashEmbedding::new(16)));
        assert_eq!(
            selector
                .select(&subtree, "   ", 3)
                .await
                .unwrap()
                .selected_targets
                .len(),
            0,
            "a blank query selects nothing"
        );
        let empty = CurrentSubtree {
            selectable_targets: Vec::new(),
            ..subtree
        };
        assert_eq!(
            selector
                .select(&empty, "alpha", 3)
                .await
                .unwrap()
                .selected_targets
                .len(),
            0,
            "an empty subtree selects nothing"
        );
    }
}
