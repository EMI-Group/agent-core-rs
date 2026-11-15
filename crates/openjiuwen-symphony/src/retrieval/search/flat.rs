//! The one-shot (flat) retriever over a whole capability catalog.
//!
//! Python's `runtime/flat.py` ranks a flat choice list with the legacy
//! `_select_items` recursion that the progressive runtime replaced; that path is explicitly not
//! to be ported (see `runtime/CONTEXT.md`'s known issues). This module keeps the Python class's
//! *contract* — one call, one bounded ranked list, `search_complete` in the trace — and
//! implements it as a single deterministic pass over the flattened catalog.
//!
//! # What a call does
//!
//! 1. flattens the catalog (the root's [`RetrieverNode::flatten_choices`] for
//!    [`FlatRetriever::retrieve`], or the caller's [`RetrieverChoice`] list for
//!    [`FlatRetriever::retrieve_choices`]) into documents keyed by payload (`name` = choice id,
//!    `description` = the choice's description), so the payload is the ranking identity and a
//!    repeated payload collapses onto its first occurrence;
//! 2. ranks them with the crate's field-weighted BM25
//!    ([`LexicalIndex`](crate::retrieval::search::lexical::LexicalIndex)) using the same
//!    alternative expression the progressive selectors use;
//! 3. optionally blends embedding similarity with
//!    [`blend_scores`](crate::retrieval::search::embedding::blend_scores) — the same
//!    `(1 - w) * lex_norm + w * (cos + 1) / 2` formula and the same
//!    [`EmbeddingScoringConfig::weight`](crate::retrieval::search::config::EmbeddingScoringConfig)
//!    as the blend selector, so `weight == 0.0` reproduces the lexical order;
//! 4. applies the config's [`ScoreTruncationConfig`](crate::retrieval::search::truncation::ScoreTruncationConfig)
//!    and bounds the list by the resolved `top_k` (`0` = unlimited);
//! 5. emits [`RetrieverCandidate`]s with `rank` 1..n, `item_id` = the choice id, `payload`,
//!    `label` = the choice id, `description`, and the `branch_path` of the item's node.
//!
//! # Branch paths
//!
//! [`FlatRetriever::retrieve`] derives a candidate's path from the tree: the node id chain from
//! (and excluding) the root down to the item's node, so a tree whose root holds the item
//! directly yields an empty path. [`FlatRetriever::retrieve_choices`] has no tree, so it uses
//! the payload's dotted proper prefixes (`a.b.c` → `["a", "a.b"]`), the same shape
//! [`build_progressive_root`](crate::retrieval::search::subtree::build_progressive_root) gives
//! the items of a choice list.
//!
//! # Trace
//!
//! One `fragment_built` event (`selectable_count` = the flattened item count), one
//! `reduce_complete` event (`input_candidates` / `output_candidates`) and one `search_complete`
//! event (`steps` = 1, `candidates`), all on the retrieved root's node id (`flat_root` for a
//! choice list), so a flat trace reads like a one-node progressive trace.
//!
//! # Methods
//!
//! [`FlatSearchResult::method`] is [`METHOD_FLAT`] when an embedder is configured *and* its
//! weight is positive, and [`METHOD_FLAT_LEXICAL`] otherwise.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_retrieval::Embedding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{validation, Result};
use crate::retrieval::search::config::ProgressiveSearchConfig;
use crate::retrieval::search::embedding::{blend_scores, EmbeddingScorer};
use crate::retrieval::search::lexical::{LexicalDocument, LexicalIndex};
use crate::retrieval::search::selector::ranking_expression;
use crate::retrieval::search::truncation::truncate_sorted_by_score;
use crate::retrieval::types::{RetrieverCandidate, RetrieverChoice, RetrieverNode, RetrieverTrace};

/// The [`FlatSearchResult::method`] of a blended flat retrieval.
pub const METHOD_FLAT: &str = "flat";

/// The [`FlatSearchResult::method`] of a lexical-only flat retrieval.
pub const METHOD_FLAT_LEXICAL: &str = "flat_lexical";

/// The node id a choice-list retrieval reports in its trace (Python's `flat_root`).
pub const FLAT_ROOT_NODE_ID: &str = "flat_root";

/// The trace event that reports the flattened catalog size.
pub const FRAGMENT_BUILT_EVENT: &str = "fragment_built";

/// The trace event that reports the flat reduction.
pub const REDUCE_COMPLETE_EVENT: &str = "reduce_complete";

/// The trace event that closes one flat retrieval.
pub const SEARCH_COMPLETE_EVENT: &str = "search_complete";

/// The result of one [`FlatRetriever`] call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct FlatSearchResult {
    /// The ranked candidates, best first.
    pub candidates: Vec<RetrieverCandidate>,
    /// The trace of the call.
    pub trace: RetrieverTrace,
    /// [`METHOD_FLAT`] or [`METHOD_FLAT_LEXICAL`].
    pub method: String,
    /// The ranking keys (each candidate's payload) in rank order.
    pub ranked_keys: Vec<String>,
}

/// The one-shot retriever: rank a flattened catalog for a query in a single pass.
///
/// Holds only the config and the optional embedder, so it is reusable and `&self`-only.
#[derive(Debug, Default)]
pub struct FlatRetriever {
    /// The traversal config (the `top_k` default, the truncation and the embedding weight).
    config: ProgressiveSearchConfig,
    /// The embedder-backed scorer, if one is configured.
    scorer: Option<EmbeddingScorer>,
}

impl FlatRetriever {
    /// Builds the retriever over `config`, without an embedder.
    #[must_use]
    pub fn new(config: ProgressiveSearchConfig) -> Self {
        Self {
            config,
            scorer: None,
        }
    }

    /// Adds an embedder, turning the ranking into a lexical/embedding blend.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.scorer = Some(EmbeddingScorer::new(embedding));
        self
    }

    /// The config the retriever runs with.
    #[must_use]
    pub fn config(&self) -> &ProgressiveSearchConfig {
        &self.config
    }

    /// The method a call would report: [`METHOD_FLAT`] when the blend is active.
    #[must_use]
    pub fn method(&self) -> &'static str {
        if self.scorer.is_some() && self.config.embedding.weight > 0.0 {
            METHOD_FLAT
        } else {
            METHOD_FLAT_LEXICAL
        }
    }

    /// Ranks `root`'s leaf items for `query`.
    ///
    /// `top_k` falls back to [`ProgressiveSearchConfig::top_k`]; `0` means unlimited. An empty
    /// catalog is an empty result, not an error.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) for a blank `query`, and the embedder's
    /// failure when a blend is configured.
    pub async fn retrieve(
        &self,
        root: &RetrieverNode,
        query: &str,
        top_k: Option<usize>,
    ) -> Result<FlatSearchResult> {
        let paths = item_branch_paths(root);
        let choices = root.flatten_choices();
        self.rank(&choices, &paths, &root.node_id, query, top_k)
            .await
    }

    /// Ranks `choices` for `query`, deriving each branch path from the payload's dotted
    /// prefixes.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) for a blank `query`, and the embedder's
    /// failure when a blend is configured.
    pub async fn retrieve_choices(
        &self,
        choices: &[RetrieverChoice],
        query: &str,
        top_k: Option<usize>,
    ) -> Result<FlatSearchResult> {
        let paths = prefix_branch_paths(choices);
        self.rank(choices, &paths, FLAT_ROOT_NODE_ID, query, top_k)
            .await
    }

    /// The shared one-pass ranking.
    async fn rank(
        &self,
        choices: &[RetrieverChoice],
        paths: &BTreeMap<String, Vec<String>>,
        node_id: &str,
        query: &str,
        top_k: Option<usize>,
    ) -> Result<FlatSearchResult> {
        if query.trim().is_empty() {
            return Err(validation("query", "must be non-empty"));
        }
        let top_k = top_k.unwrap_or(self.config.top_k);
        let mut trace = RetrieverTrace::default();
        let mut unique: BTreeMap<String, &RetrieverChoice> = BTreeMap::new();
        for choice in choices {
            unique.entry(choice_key(choice)).or_insert_with(|| choice);
        }
        let documents: Vec<LexicalDocument> = unique
            .iter()
            .map(|(key, choice)| LexicalDocument {
                key: key.clone(),
                name: choice.choice_id.clone(),
                description: choice.description.clone(),
                body: String::new(),
                aliases: Vec::new(),
                category: paths
                    .get(key)
                    .map(|path| path.join(" > "))
                    .unwrap_or_default(),
            })
            .collect();
        trace.record(
            FRAGMENT_BUILT_EVENT,
            node_id,
            0,
            BTreeMap::from([("selectable_count".to_owned(), json!(documents.len()))]),
        );
        if documents.is_empty() {
            trace.record(
                REDUCE_COMPLETE_EVENT,
                node_id,
                0,
                BTreeMap::from([
                    ("input_candidates".to_owned(), json!(0)),
                    ("output_candidates".to_owned(), json!(0)),
                ]),
            );
            return Ok(FlatSearchResult {
                candidates: Vec::new(),
                trace: search_trace(trace, node_id, 0),
                method: self.method().to_owned(),
                ranked_keys: Vec::new(),
            });
        }

        let index = LexicalIndex::build(documents.iter().cloned());
        let hits = index.search(&ranking_expression(query))?;
        let scores = match self.scorer.as_ref() {
            Some(scorer) if self.config.embedding.weight > 0.0 => {
                scorer
                    .index_capped(&documents, self.config.embedding.max_documents)
                    .await?;
                scorer.scores(query).await?
            }
            _ => BTreeMap::new(),
        };
        let blended = blend_scores(&hits, &scores, self.config.embedding.weight);
        let (kept, _decision) =
            truncate_sorted_by_score(&blended, |hit| hit.score, &self.config.truncation);
        let limit = if top_k == 0 { usize::MAX } else { top_k };
        let candidates: Vec<RetrieverCandidate> = kept
            .iter()
            .take(limit)
            .enumerate()
            .filter_map(|(offset, hit)| {
                let choice = unique.get(hit.key.as_str())?;
                Some(RetrieverCandidate {
                    rank: offset + 1,
                    item_id: choice.choice_id.clone(),
                    payload: hit.key.clone(),
                    branch_path: paths.get(hit.key.as_str()).cloned().unwrap_or_default(),
                    label: choice.choice_id.clone(),
                    description: choice.description.clone(),
                })
            })
            .collect();
        trace.record(
            REDUCE_COMPLETE_EVENT,
            node_id,
            0,
            BTreeMap::from([
                ("input_candidates".to_owned(), json!(documents.len())),
                ("output_candidates".to_owned(), json!(candidates.len())),
            ]),
        );
        let ranked_keys: Vec<String> = candidates
            .iter()
            .map(|candidate| candidate.payload.clone())
            .collect();
        Ok(FlatSearchResult {
            candidates,
            trace: search_trace(trace, node_id, ranked_keys.len()),
            method: self.method().to_owned(),
            ranked_keys,
        })
    }
}

/// Closes a flat trace with its `search_complete` event.
fn search_trace(mut trace: RetrieverTrace, node_id: &str, candidates: usize) -> RetrieverTrace {
    trace.record(
        SEARCH_COMPLETE_EVENT,
        node_id,
        0,
        BTreeMap::from([
            ("steps".to_owned(), json!(1)),
            ("candidates".to_owned(), json!(candidates)),
        ]),
    );
    trace
}

/// The ranking identity of a choice: its payload, or its id when the payload is blank.
fn choice_key(choice: &RetrieverChoice) -> String {
    if choice.payload.is_empty() {
        choice.choice_id.clone()
    } else {
        choice.payload.clone()
    }
}

/// Maps each item key of `root` to the node id chain that reaches it (root excluded).
fn item_branch_paths(root: &RetrieverNode) -> BTreeMap<String, Vec<String>> {
    let mut paths = BTreeMap::new();
    let mut stack: Vec<(&RetrieverNode, Vec<String>)> = vec![(root, vec![root.node_id.clone()])];
    while let Some((node, chain)) = stack.pop() {
        let path = chain.get(1..).map(<[String]>::to_vec).unwrap_or_default();
        for item in node.items() {
            let key = if item.payload.is_empty() {
                item.item_id.clone()
            } else {
                item.payload.clone()
            };
            paths.entry(key).or_insert_with(|| path.clone());
        }
        for child in node.children().into_iter().rev() {
            let mut child_chain = chain.clone();
            child_chain.push(child.node_id.clone());
            stack.push((child, child_chain));
        }
    }
    paths
}

/// Maps each choice's key to its payload's dotted proper prefixes.
fn prefix_branch_paths(choices: &[RetrieverChoice]) -> BTreeMap<String, Vec<String>> {
    let mut paths = BTreeMap::new();
    for choice in choices {
        let key = choice_key(choice);
        let segments: Vec<&str> = choice
            .payload
            .split('.')
            .filter(|part| !part.is_empty())
            .collect();
        let mut prefixes: Vec<String> = Vec::new();
        for end in 1..segments.len() {
            prefixes.push(segments[..end].join("."));
        }
        paths.entry(key).or_insert(prefixes);
    }
    paths
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::error::is_unsupported;
    use crate::retrieval::types::RetrieverItem;
    use openjiuwen_retrieval::HashEmbedding;

    /// A tree with one nested branch and two root items.
    fn tree() -> RetrieverNode {
        RetrieverNode::new("ROOT", "Root")
            .with_items(vec![
                RetrieverItem::new("flat.one", "flat.one")
                    .with_label("Flat writer")
                    .with_description("writes flat documents"),
                RetrieverItem::new("flat.two", "flat.two")
                    .with_label("Flat tester")
                    .with_description("tests flat behaviour"),
            ])
            .with_children(vec![RetrieverNode::new("Skills.alpha", "Alpha")
                .with_items(vec![RetrieverItem::new("alpha.one", "alpha.one")
                    .with_label("Alpha writer")
                    .with_description("writes alpha documentation")])])
    }

    /// The config the tests share.
    fn config() -> ProgressiveSearchConfig {
        ProgressiveSearchConfig {
            top_k: 2,
            ..ProgressiveSearchConfig::default()
        }
    }

    fn keys(result: &FlatSearchResult) -> Vec<&str> {
        result
            .candidates
            .iter()
            .map(|candidate| candidate.payload.as_str())
            .collect()
    }

    #[tokio::test]
    async fn a_flat_retrieval_ranks_by_the_query_and_keeps_the_tree_paths() {
        let retriever = FlatRetriever::new(config());
        let result = retriever
            .retrieve(&tree(), "alpha documentation flat", Some(0))
            .await
            .unwrap();
        assert_eq!(result.method, METHOD_FLAT_LEXICAL);
        assert_eq!(keys(&result), vec!["alpha.one", "flat.two", "flat.one"]);
        assert_eq!(result.ranked_keys, keys(&result));
        assert_eq!(
            result.candidates[0].branch_path,
            vec!["Skills.alpha".to_owned()],
            "a nested item carries its node chain, root excluded"
        );
        assert!(result.candidates[1].branch_path.is_empty());
        assert!(result.candidates[2].branch_path.is_empty());
        assert_eq!(result.candidates[0].rank, 1);
        assert_eq!(result.candidates[0].item_id, "Alpha writer");
        assert_eq!(result.candidates[0].label, "Alpha writer");
        assert_eq!(
            result.candidates[0].description,
            "writes alpha documentation"
        );
    }

    #[tokio::test]
    async fn top_k_is_clamped_and_zero_is_unlimited() {
        let retriever = FlatRetriever::new(config());
        let clamped = retriever
            .retrieve(&tree(), "alpha flat", None)
            .await
            .unwrap();
        assert_eq!(clamped.candidates.len(), 2, "config.top_k applies");
        let unbounded = retriever
            .retrieve(&tree(), "alpha flat", Some(0))
            .await
            .unwrap();
        assert_eq!(unbounded.candidates.len(), 3);
    }

    #[tokio::test]
    async fn choices_derive_their_paths_from_the_payload_prefixes() {
        let retriever = FlatRetriever::new(config());
        let choices = vec![
            RetrieverChoice::new("Writer", "a.b.c"),
            RetrieverChoice::new("Viewer", "solo"),
        ];
        let result = retriever
            .retrieve_choices(&choices, "writer viewer", Some(0))
            .await
            .unwrap();
        assert_eq!(result.candidates.len(), 2);
        let by_payload: BTreeMap<&str, &RetrieverCandidate> = result
            .candidates
            .iter()
            .map(|candidate| (candidate.payload.as_str(), candidate))
            .collect();
        assert!(by_payload["solo"].branch_path.is_empty());
        assert_eq!(
            by_payload["a.b.c"].branch_path,
            vec!["a".to_owned(), "a.b".to_owned()]
        );
    }

    #[tokio::test]
    async fn a_blank_query_is_a_validation_error_and_an_empty_catalog_is_not() {
        let retriever = FlatRetriever::new(config());
        let error = retriever.retrieve(&tree(), "   ", None).await.unwrap_err();
        assert!(!is_unsupported(&error));
        assert!(matches!(error, crate::error::Error::Validation(_)));
        let empty = FlatRetriever::new(config())
            .retrieve(&RetrieverNode::new("ROOT", "Root"), "alpha", None)
            .await
            .unwrap();
        assert!(empty.candidates.is_empty());
        assert!(empty.ranked_keys.is_empty());
        assert_eq!(
            empty.trace.event_types(),
            vec![
                FRAGMENT_BUILT_EVENT,
                REDUCE_COMPLETE_EVENT,
                SEARCH_COMPLETE_EVENT
            ]
        );
    }

    #[tokio::test]
    async fn the_trace_carries_the_flat_shape() {
        let retriever = FlatRetriever::new(config());
        let result = retriever.retrieve(&tree(), "flat", Some(0)).await.unwrap();
        let types = result.trace.event_types();
        assert_eq!(
            types,
            vec![
                FRAGMENT_BUILT_EVENT,
                REDUCE_COMPLETE_EVENT,
                SEARCH_COMPLETE_EVENT
            ]
        );
        assert_eq!(
            result.trace.events()[0].detail["selectable_count"],
            json!(3)
        );
        assert_eq!(
            result.trace.events()[1].detail["input_candidates"],
            json!(3)
        );
        assert_eq!(
            result.trace.events()[1].detail["output_candidates"],
            json!(result.candidates.len())
        );
        assert_eq!(result.trace.events()[2].detail["steps"], json!(1));
        assert_eq!(result.trace.events()[0].node_id, "ROOT");
    }

    #[tokio::test]
    async fn a_zero_weight_blend_reproduces_the_lexical_order() {
        let lexical = FlatRetriever::new(config())
            .retrieve(&tree(), "alpha flat", Some(0))
            .await
            .unwrap();
        let mut blended_config = config();
        blended_config.embedding = blended_config.embedding.with_weight(0.0);
        let blended = FlatRetriever::new(blended_config)
            .with_embedding(Arc::new(HashEmbedding::new(16)))
            .retrieve(&tree(), "alpha flat", Some(0))
            .await
            .unwrap();
        assert_eq!(blended.method, METHOD_FLAT_LEXICAL);
        assert_eq!(keys(&blended), keys(&lexical));
    }

    #[tokio::test]
    async fn a_positive_weight_switches_to_the_blended_method() {
        let mut config = config();
        config.embedding = config.embedding.with_weight(0.5);
        let retriever = FlatRetriever::new(config).with_embedding(Arc::new(HashEmbedding::new(32)));
        assert_eq!(retriever.method(), METHOD_FLAT);
        let first = retriever.retrieve(&tree(), "alpha", Some(0)).await.unwrap();
        let second = retriever.retrieve(&tree(), "alpha", Some(0)).await.unwrap();
        assert_eq!(first, second);
        assert_eq!(first.method, METHOD_FLAT);
        assert_eq!(first.candidates.len(), 3);
    }

    #[tokio::test]
    async fn a_repeated_payload_collapses_onto_one_document() {
        let retriever = FlatRetriever::new(config());
        let choices = vec![
            RetrieverChoice::new("First", "same.payload"),
            RetrieverChoice::new("Second", "same.payload"),
        ];
        let result = retriever
            .retrieve_choices(&choices, "same", Some(0))
            .await
            .unwrap();
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].item_id, "First");
        assert_eq!(
            result.trace.events()[0].detail["selectable_count"],
            json!(1)
        );
    }
}
