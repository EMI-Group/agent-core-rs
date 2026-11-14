//! Query-time ranking: lexical, embedding and Reciprocal Rank Fusion retrieval.
//!
//! Ports the decision table of Python `ttse/index.py::rank_track` and the
//! `ttse/consult.py::_retrieve_one` selection step. Two independent rankings are
//! produced over the same pool:
//!
//! - **lexical** — the bank's effective score multiplied by the token overlap
//!   with the query ([`TipBank::top`](crate::ttse::TipBank::top) uses the same
//!   notion), which needs no model and is always available;
//! - **embedding** — cosine similarity through an [`EmbeddingIndex`], available
//!   only when the caller supplied one.
//!
//! [`RetrieveMode::Hybrid`] fuses both with Reciprocal Rank Fusion (`rrf_k = 60`,
//! equal weights, `score += 1 / (rrf_k + rank)` over one-based ranks), matching the
//! Python default. A single-branch mode falls back to the other branch, and a
//! short pool (< the requested limit) is returned as-is, so retrieval never
//! silently yields nothing while the bank holds tips.

use std::collections::BTreeMap;

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::ttse::bank::{jaccard, tokenize, TipBank};
use crate::ttse::config::TtseConfig;
use crate::ttse::consult::render_tips;
use crate::ttse::index::EmbeddingIndex;
use crate::ttse::models::{truncate_chars, ConsultResult, TipRecord};

/// The Reciprocal Rank Fusion constant Python uses (`DEFAULT_RRF_K`).
pub const RRF_K: u32 = 60;

/// How the two rankings are combined.
///
/// The variants mirror Python's `hybrid | embed | bm25` configuration plus the
/// `dump` mode its decision table returns when no ranking is available.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrieveMode {
    /// Fuse the lexical and embedding rankings with Reciprocal Rank Fusion.
    #[default]
    Hybrid,
    /// Prefer the embedding ranking, falling back to the lexical one.
    Embed,
    /// Prefer the lexical ranking (`bm25` in Python), falling back to embeddings.
    Lexical,
    /// Skip ranking and return the pool in bank order.
    Dump,
}

impl RetrieveMode {
    /// The stable `snake_case` label used by JSON and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Embed => "embed",
            Self::Lexical => "lexical",
            Self::Dump => "dump",
        }
    }
}

/// One ranked tip: the id, the score that ordered it and the branch ranks.
///
/// A branch rank is `None` when that branch did not return the tip, so a fused
/// result makes it obvious which rankings contributed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RankedTip {
    /// The content-addressed tip id.
    pub tip_id: String,
    /// The score the ranking used: RRF, cosine or the lexical score.
    pub score: f64,
    /// The one-based rank in the lexical branch, when it matched.
    pub lexical_rank: Option<usize>,
    /// The one-based rank in the embedding branch, when it matched.
    pub embed_rank: Option<usize>,
}

/// Ranks the bank against `query`.
///
/// The Python decision table is reproduced as follows:
///
/// 1. An empty bank, a zero limit, a blank query or [`RetrieveMode::Dump`]
///    returns the first `limit` tips in bank order; so does a pool no larger than
///    the limit (the `pool_le_topk` short-circuit).
/// 2. The lexical branch ranks tips with a positive query overlap, descending by
///    `score() * (1 + overlap)`, ties by ascending `tip_id`.
/// 3. The embedding branch asks the index for up to `max(4 * limit, 16)` hits and
///    keeps the ones the bank still holds.
/// 4. [`RetrieveMode::Hybrid`] fuses both when both are non-empty and otherwise
///    returns the branch that matched; [`RetrieveMode::Embed`] falls back to the
///    lexical branch (and [`RetrieveMode::Lexical`] to the embedding branch) when
///    its preferred branch yields nothing; an empty result from both means the
///    dump order.
///
/// # Errors
///
/// The embedder's errors from [`EmbeddingIndex::query`].
pub async fn rank(
    bank: &TipBank,
    index: Option<&EmbeddingIndex>,
    query: &str,
    limit: usize,
    mode: RetrieveMode,
) -> Result<Vec<RankedTip>> {
    if limit == 0 || bank.is_empty() {
        return Ok(Vec::new());
    }
    if mode == RetrieveMode::Dump || query.trim().is_empty() || bank.len() <= limit {
        return Ok(dump(bank, limit));
    }

    let lexical = lexical_ranking(bank, query);
    let mut embedded: Vec<(String, f64)> = Vec::new();
    if let Some(index) = index {
        let wanted = (limit.saturating_mul(4)).max(16);
        for hit in index.query(query, wanted).await? {
            if bank.get(&hit.tip_id).is_some() {
                embedded.push((hit.tip_id, hit.score));
            }
        }
    }

    let ranked = match mode {
        RetrieveMode::Hybrid => {
            if !lexical.is_empty() && !embedded.is_empty() {
                Some(rrf(&lexical, &embedded, limit))
            } else if !lexical.is_empty() {
                Some(branch(&lexical, true, limit))
            } else if !embedded.is_empty() {
                Some(branch(&embedded, false, limit))
            } else {
                None
            }
        }
        RetrieveMode::Embed => {
            if !embedded.is_empty() {
                Some(branch(&embedded, false, limit))
            } else if !lexical.is_empty() {
                Some(branch(&lexical, true, limit))
            } else {
                None
            }
        }
        RetrieveMode::Lexical => {
            if !lexical.is_empty() {
                Some(branch(&lexical, true, limit))
            } else if !embedded.is_empty() {
                Some(branch(&embedded, false, limit))
            } else {
                None
            }
        }
        RetrieveMode::Dump => None,
    };
    Ok(ranked.unwrap_or_else(|| dump(bank, limit)))
}

/// Selects and renders the tips relevant to `query`, ranked by `mode`.
///
/// The ranking is capped at [`TtseConfig::consult_limit`] tips and rendered with
/// [`render_tips`]; the rendering is produced once at unbounded width to decide
/// [`ConsultResult::truncated`], then cut to
/// [`TtseConfig::consult_max_chars`] characters. The bank is only read — hit
/// bookkeeping belongs to [`TtseEngine`](crate::ttse::TtseEngine).
///
/// # Errors
///
/// The embedder's errors from [`rank`].
pub async fn consult_async(
    bank: &TipBank,
    index: Option<&EmbeddingIndex>,
    query: &str,
    config: &TtseConfig,
    mode: RetrieveMode,
) -> Result<ConsultResult> {
    let ranked = rank(bank, index, query, config.consult_limit(), mode).await?;
    let tips: Vec<TipRecord> = ranked
        .iter()
        .filter_map(|hit| bank.get(&hit.tip_id).cloned())
        .collect();
    let full = render_tips(&tips, usize::MAX);
    let truncated = full.chars().count() > config.consult_max_chars;
    let rendered = if truncated {
        truncate_chars(&full, config.consult_max_chars)
    } else {
        full
    };
    Ok(ConsultResult {
        tips,
        rendered,
        truncated,
    })
}

/// The first `limit` tips in bank order, scored by their effective score.
fn dump(bank: &TipBank, limit: usize) -> Vec<RankedTip> {
    bank.tips
        .iter()
        .take(limit)
        .map(|tip| RankedTip {
            tip_id: tip.tip_id.clone(),
            score: tip.score(),
            lexical_rank: None,
            embed_rank: None,
        })
        .collect()
}

/// Ranks tips with a positive query overlap, descending, ties by `tip_id`.
fn lexical_ranking(bank: &TipBank, query: &str) -> Vec<(String, f64)> {
    let query_tokens = tokenize(query);
    let mut scored: Vec<(String, f64)> = bank
        .tips
        .iter()
        .filter_map(|tip| {
            let haystack = format!("{} {}", tip.rule, tip.keywords.join(" "));
            let overlap = jaccard(&query_tokens, &tokenize(&haystack));
            (overlap > 0.0).then(|| (tip.tip_id.clone(), tip.score() * (1.0 + overlap)))
        })
        .collect();
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored
}

/// Turns one branch's scored ids into ranked hits, cutting to `limit`.
fn branch(list: &[(String, f64)], lexical: bool, limit: usize) -> Vec<RankedTip> {
    list.iter()
        .take(limit)
        .enumerate()
        .map(|(index, (tip_id, score))| RankedTip {
            tip_id: tip_id.clone(),
            score: *score,
            lexical_rank: lexical.then_some(index + 1),
            embed_rank: (!lexical).then_some(index + 1),
        })
        .collect()
}

/// Fuses two ranked lists with Reciprocal Rank Fusion.
fn rrf(lexical: &[(String, f64)], embedded: &[(String, f64)], limit: usize) -> Vec<RankedTip> {
    let mut scores: BTreeMap<String, f64> = BTreeMap::new();
    let mut lexical_ranks: BTreeMap<String, usize> = BTreeMap::new();
    let mut embed_ranks: BTreeMap<String, usize> = BTreeMap::new();
    let rrf_k = f64::from(RRF_K);
    for (rank, (tip_id, _)) in lexical.iter().enumerate() {
        let rank = rank + 1;
        *scores.entry(tip_id.clone()).or_default() += 1.0 / (rrf_k + rank as f64);
        lexical_ranks.insert(tip_id.clone(), rank);
    }
    for (rank, (tip_id, _)) in embedded.iter().enumerate() {
        let rank = rank + 1;
        *scores.entry(tip_id.clone()).or_default() += 1.0 / (rrf_k + rank as f64);
        embed_ranks.insert(tip_id.clone(), rank);
    }
    let mut ranked: Vec<RankedTip> = scores
        .into_iter()
        .map(|(tip_id, score)| RankedTip {
            lexical_rank: lexical_ranks.get(&tip_id).copied(),
            embed_rank: embed_ranks.get(&tip_id).copied(),
            tip_id,
            score,
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.tip_id.cmp(&right.tip_id))
    });
    ranked.truncate(limit);
    ranked
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_common::error::Error;
    use openjiuwen_core::retrieval::embedding::Embedding;

    use super::*;
    use crate::ttse::models::{TipCategory, TipRecord};

    /// An embedder that answers from a fixed text-to-vector table.
    struct FixedEmbedding {
        vectors: BTreeMap<String, Vec<f32>>,
    }

    impl FixedEmbedding {
        fn new(entries: &[(&str, [f32; 2])]) -> Self {
            Self {
                vectors: entries
                    .iter()
                    .map(|(text, vector)| ((*text).to_owned(), vector.to_vec()))
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl Embedding for FixedEmbedding {
        fn model_name(&self) -> &str {
            "fixed"
        }

        fn dimension_hint(&self) -> Option<usize> {
            Some(2)
        }

        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            self.vectors.get(text).cloned().ok_or_else(|| {
                Error::validation(format!("fixed embedding has no vector for `{text}`"))
            })
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            let mut vectors = Vec::with_capacity(texts.len());
            for text in texts {
                vectors.push(self.embed_query(text).await?);
            }
            Ok(vectors)
        }
    }

    const QUERY: &str = "alpha beta gamma";
    const X: &str = "alpha beta gamma";
    const Y: &str = "alpha delta epsilon zeta";
    const Z: &str = "theta iota kappa";
    const W: &str = "lambda mu nu xi";

    fn bank() -> TipBank {
        let mut bank = TipBank::with_capacity(8);
        for (rule, category) in [
            (X, TipCategory::Tool),
            (Y, TipCategory::Workflow),
            (Z, TipCategory::Pitfall),
            (W, TipCategory::Other),
        ] {
            bank.add(TipRecord::new(rule, category).expect("valid rule"));
        }
        bank
    }

    fn id(rule: &str) -> String {
        crate::ttse::models::tip_id_for(TipCategory::Tool, rule)
    }

    fn id_of(bank: &TipBank, rule: &str) -> String {
        bank.tips()
            .iter()
            .find(|tip| tip.rule == rule)
            .expect("rule stored")
            .tip_id
            .clone()
    }

    async fn indexed(bank: &TipBank) -> EmbeddingIndex {
        let embedding: Arc<dyn Embedding> = Arc::new(FixedEmbedding::new(&[
            (QUERY, [1.0, 0.0]),
            (X, [1.0, 0.0]),
            (Z, [0.0, 1.0]),
        ]));
        let mut index = EmbeddingIndex::new(embedding);
        index.upsert(&id_of(bank, X), X).await.expect("upsert x");
        index.upsert(&id_of(bank, Z), Z).await.expect("upsert z");
        index
    }

    #[tokio::test]
    async fn a_short_pool_is_returned_in_bank_order() {
        let bank = bank();
        let ranked = rank(&bank, None, QUERY, 8, RetrieveMode::Hybrid)
            .await
            .expect("rank");
        assert_eq!(ranked.len(), 4);
        assert_eq!(ranked[0].tip_id, bank.tips()[0].tip_id);
        assert!(ranked.iter().all(|hit| hit.lexical_rank.is_none()));
        assert!(ranked.iter().all(|hit| hit.embed_rank.is_none()));
    }

    #[tokio::test]
    async fn dump_mode_keeps_bank_order_and_ignores_query() {
        let bank = bank();
        let index = indexed(&bank).await;
        let ranked = rank(&bank, Some(&index), QUERY, 2, RetrieveMode::Dump)
            .await
            .expect("rank");
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].tip_id, id_of(&bank, X));
        assert_eq!(ranked[1].tip_id, id_of(&bank, Y));
        assert!(ranked.iter().all(|hit| hit.lexical_rank.is_none()));
    }

    #[tokio::test]
    async fn the_mode_matrix_uses_the_expected_branch() {
        let bank = bank();
        let index = indexed(&bank).await;

        let embedded = rank(&bank, Some(&index), QUERY, 2, RetrieveMode::Embed)
            .await
            .expect("embed");
        assert_eq!(embedded[0].tip_id, id_of(&bank, X));
        assert_eq!(embedded[0].embed_rank, Some(1));
        assert!(embedded[0].lexical_rank.is_none());
        assert_eq!(embedded[1].tip_id, id_of(&bank, Z));

        let lexical = rank(&bank, Some(&index), QUERY, 2, RetrieveMode::Lexical)
            .await
            .expect("lexical");
        assert_eq!(lexical[0].tip_id, id_of(&bank, X));
        assert_eq!(lexical[0].lexical_rank, Some(1));
        assert!(lexical[0].embed_rank.is_none());
        assert_eq!(lexical[1].tip_id, id_of(&bank, Y));

        // Without an index the embedding branch is unavailable, so `Embed`
        // falls back to the lexical ranking instead of returning nothing.
        let fallback = rank(&bank, None, QUERY, 2, RetrieveMode::Embed)
            .await
            .expect("embed without index");
        assert_eq!(fallback[0].tip_id, id_of(&bank, X));
        assert_eq!(fallback[0].lexical_rank, Some(1));
    }

    #[tokio::test]
    async fn hybrid_fusion_prefers_a_tip_both_branches_return() {
        let bank = bank();
        let index = indexed(&bank).await;
        let ranked = rank(&bank, Some(&index), QUERY, 2, RetrieveMode::Hybrid)
            .await
            .expect("hybrid");

        assert_eq!(ranked[0].tip_id, id_of(&bank, X));
        assert_eq!(ranked[0].lexical_rank, Some(1));
        assert_eq!(ranked[0].embed_rank, Some(1));
        assert!(ranked[0].score > ranked[1].score);
        assert_eq!(ranked.len(), 2);

        let both = 2.0 / (f64::from(RRF_K) + 1.0);
        assert!((ranked[0].score - both).abs() < 1e-12);
        assert!((ranked[1].score - 1.0 / (f64::from(RRF_K) + 2.0)).abs() < 1e-12);
    }

    #[tokio::test]
    async fn empty_inputs_are_empty() {
        let empty = TipBank::with_capacity(4);
        assert!(rank(&empty, None, QUERY, 8, RetrieveMode::Hybrid)
            .await
            .expect("empty bank")
            .is_empty());
        let bank = bank();
        assert!(rank(&bank, None, QUERY, 0, RetrieveMode::Hybrid)
            .await
            .expect("zero limit")
            .is_empty());
        let dumped = rank(&bank, None, "   ", 2, RetrieveMode::Hybrid)
            .await
            .expect("blank query");
        assert_eq!(dumped.len(), 2);
        assert_eq!(dumped[0].tip_id, id_of(&bank, X));
    }

    #[tokio::test]
    async fn consult_async_limits_and_renders() {
        let bank = bank();
        let index = indexed(&bank).await;
        let config = TtseConfig {
            consult_top_k: 1,
            ..TtseConfig::default()
        };
        let result = consult_async(&bank, Some(&index), QUERY, &config, RetrieveMode::Hybrid)
            .await
            .expect("consult");
        assert_eq!(result.tips.len(), 1);
        assert_eq!(result.tips[0].tip_id, id_of(&bank, X));
        assert!(result.rendered.starts_with("- [tool] "));
        assert!(!result.truncated);

        let tight = TtseConfig {
            consult_max_chars: 12,
            ..config
        };
        let cut = consult_async(&bank, Some(&index), QUERY, &tight, RetrieveMode::Hybrid)
            .await
            .expect("consult");
        assert!(cut.truncated);
        assert_eq!(cut.rendered.chars().count(), 12);
        assert!(cut.rendered.ends_with('…'));
    }

    #[test]
    fn mode_labels_and_serde_round_trip() {
        for mode in [
            RetrieveMode::Hybrid,
            RetrieveMode::Embed,
            RetrieveMode::Lexical,
            RetrieveMode::Dump,
        ] {
            let json = serde_json::to_string(&mode).expect("serializable");
            assert_eq!(json, format!("\"{}\"", mode.as_str()));
            let restored: RetrieveMode = serde_json::from_str(&json).expect("deserializable");
            assert_eq!(restored, mode);
        }
        assert_eq!(RRF_K, 60);
        assert_ne!(id("alpha beta gamma"), "");
    }
}
