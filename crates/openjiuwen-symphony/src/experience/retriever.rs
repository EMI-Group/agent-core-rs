//! Query-time experience retrieval with explicit provenance.
//!
//! Python's `experience/retriever.py` is a one-method fast path: it asks the bank for the
//! single best embedding match above a threshold and flattens the matched skills. This
//! module keeps that fast path but makes the catalog list explicit, so a deployment can
//! retrieve over a store-backed [`ExperienceBank`], a
//! persisted [`ExperienceIndex`], or both.
//!
//! # Skill filters
//!
//! A request's `required_skills`/`excluded_skills` reach every configured catalog, but by
//! two different routes, because [`ExperienceIndex::search`] has no skill-filter seam:
//!
//! - the [`ExperienceBank`] applies them itself (its search options carry them), so bank
//!   hits arrive already filtered,
//! - the retriever applies them to the [`ExperienceIndex`] hits itself, with the same
//!   predicate the bank uses: an item listing any excluded skill is rejected, and the
//!   required skills must all be present — or, with `require_all_skills == false`, at
//!   least one of them.
//!
//! A filtered request therefore asks the index for `top_k = 0` (unlimited) and truncates
//! only after filtering, so a filtered-out index hit never consumes a caller's budget; an
//! unfiltered request keeps the single-shot `top_k` query.
//!
//! # Merge rules
//!
//! Each configured catalog is queried with *its own* ranking (the same `top_k` and a
//! `min_score`-free request), so a catalog behaves exactly as it does when called
//! directly. The returned hits are then merged on
//! [`ExperienceItem::dedupe_key`](crate::experience::ExperienceItem::dedupe_key):
//!
//! - the highest score wins; on an exact tie the smaller item id is kept,
//! - the `sources` of both hits are unioned, so a pattern found in both catalogs reports
//!   both,
//! - the winner's [`RankingProvenance`]
//!   travels with its score.
//!
//! The merged list is ordered by score descending then item id ascending, truncated to
//! `top_k` (`0` = unlimited), and only then filtered by `min_score` — the same order the
//! catalogs use internally, so a small `top_k` behaves identically whether the caller
//! queries a catalog or the retriever.
//!
//! # Provenance
//!
//! [`RankingProvenance`] is reported per
//! hit *and* for the report: `Embedding` only when every contributing catalog scored with
//! an embedder, `Lexical` as soon as one catalog fell back to the deterministic term
//! coverage score. A caller can therefore tell "these hits are semantic" from "these hits
//! are lexical" without inspecting the configuration.
//!
//! Skill filters change which hits survive, never how they were scored: the reporting is
//! the same whether a request filters or not, and a request whose every hit is filtered
//! out reports an empty `hits` list with the catalogs and provenance it queried.

use std::cmp::Ordering;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{validation, Result};

use super::{
    ExperienceBank, ExperienceHit, ExperienceIndex, ExperienceItem, ExperienceRanking,
    SearchOptions, DEFAULT_TOP_K,
};

/// One retrieval request: the query, the budget, the skill filters, and the threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrieveRequest {
    /// The natural-language query to match experience patterns against.
    pub query: String,
    /// The maximum number of hits; defaults to [`DEFAULT_TOP_K`], `0` means unlimited.
    pub top_k: usize,
    /// Skill ids every hit must list (all of them by default; see
    /// [`Self::require_all_skills`]).
    pub required_skills: Vec<String>,
    /// Skill ids no hit may list.
    pub excluded_skills: Vec<String>,
    /// Whether every required skill must be present (`true`) or just one (`false`).
    pub require_all_skills: bool,
    /// The minimum score a hit must reach, applied after `top_k` truncation.
    pub min_score: Option<f64>,
}

impl Default for RetrieveRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            top_k: DEFAULT_TOP_K,
            required_skills: Vec::new(),
            excluded_skills: Vec::new(),
            require_all_skills: true,
            min_score: None,
        }
    }
}

impl RetrieveRequest {
    /// A request for `query` with the default budget and no filters.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Self::default()
        }
    }

    /// A request that returns every hit (`top_k == 0`).
    #[must_use]
    pub fn unlimited(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            top_k: 0,
            ..Self::default()
        }
    }

    /// Sets the maximum number of hits (`0` means unlimited).
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Requires the listed skill ids.
    #[must_use]
    pub fn with_required_skills(
        mut self,
        required_skills: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.required_skills = required_skills.into_iter().map(Into::into).collect();
        self
    }

    /// Excludes items listing any of the listed skill ids.
    #[must_use]
    pub fn with_excluded_skills(
        mut self,
        excluded_skills: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.excluded_skills = excluded_skills.into_iter().map(Into::into).collect();
        self
    }

    /// Sets whether every required skill must be present, or just one.
    #[must_use]
    pub fn with_require_all_skills(mut self, require_all_skills: bool) -> Self {
        self.require_all_skills = require_all_skills;
        self
    }

    /// Sets the minimum score.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = Some(min_score);
        self
    }
}

/// Which catalog a retrieved hit came from, or which catalogs a report covered.
///
/// The declaration order is the deterministic report order (`Bank` before `Index`).
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum RetrieveSource {
    /// The store-backed [`ExperienceBank`].
    #[default]
    Bank,
    /// The persisted [`ExperienceIndex`].
    Index,
}

impl RetrieveSource {
    /// The `snake_case` wire name of the source.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bank => "bank",
            Self::Index => "index",
        }
    }
}

/// How a score was produced.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum RankingProvenance {
    /// The catalog scored with an embedding model (cosine similarity).
    Embedding,
    /// The catalog fell back to the deterministic lexical term-coverage score.
    #[default]
    Lexical,
}

impl RankingProvenance {
    /// The `snake_case` wire name of the provenance.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Lexical => "lexical",
        }
    }

    /// Whether this provenance is the lexical fallback.
    #[must_use]
    pub fn is_lexical(self) -> bool {
        matches!(self, Self::Lexical)
    }
}

/// One merged hit: the item, its winning score, the catalogs that produced it, and how
/// that score was ranked.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievedExperience {
    /// The matched experience item.
    pub item: ExperienceItem,
    /// The winning score; `[0, 1]` for lexical ranking, `[-1, 1]` for cosine.
    pub score: f64,
    /// Every catalog that produced this item, in deterministic source order.
    pub sources: Vec<RetrieveSource>,
    /// How the winning score was ranked.
    pub ranking: RankingProvenance,
}

impl RetrievedExperience {
    /// The id of the matched item.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.item.id
    }
}

/// The answer to one [`RetrieveRequest`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RetrievalReport {
    /// The query that was answered.
    pub query: String,
    /// The merged hits, in final order.
    pub hits: Vec<RetrievedExperience>,
    /// The catalogs the request queried, in deterministic source order.
    pub sources: Vec<RetrieveSource>,
    /// The report-wide provenance: `Embedding` only when every catalog scored with an
    /// embedder, `Lexical` as soon as one fell back.
    pub ranking: RankingProvenance,
    /// The hits the catalogs returned, counted before the retriever's client-side skill
    /// filter, dedupe, truncation, and `min_score`.
    ///
    /// The bank applies the request's skill filters itself, so its contribution is already
    /// filtered; the index has no skill-filter seam, so its contribution counts the hits it
    /// returned *before* the retriever dropped the ones the filter rejects. A request that
    /// filters every index hit away therefore still reports `total_candidates > 0`.
    pub total_candidates: usize,
}

impl RetrievalReport {
    /// The ids of the hits, in order.
    #[must_use]
    pub fn ids(&self) -> Vec<&str> {
        self.hits.iter().map(RetrievedExperience::id).collect()
    }

    /// Whether the request produced no hits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }
}

/// Query-time retrieval over an [`ExperienceBank`], an [`ExperienceIndex`], or both.
#[derive(Debug, Clone, Default)]
pub struct ExperienceRetriever {
    /// The store-backed catalog, when configured.
    bank: Option<Arc<ExperienceBank>>,
    /// The persisted embedding catalog, when configured.
    index: Option<Arc<ExperienceIndex>>,
}

impl ExperienceRetriever {
    /// A retriever with no catalog configured.
    ///
    /// Add one with [`Self::with_bank`] or [`Self::with_index`]; [`Self::retrieve`]
    /// rejects a request while neither is set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A retriever over `bank` only.
    #[must_use]
    pub fn for_bank(bank: Arc<ExperienceBank>) -> Self {
        Self {
            bank: Some(bank),
            index: None,
        }
    }

    /// A retriever over `index` only.
    #[must_use]
    pub fn for_index(index: Arc<ExperienceIndex>) -> Self {
        Self {
            bank: None,
            index: Some(index),
        }
    }

    /// Adds (or replaces) the bank catalog.
    #[must_use]
    pub fn with_bank(mut self, bank: Arc<ExperienceBank>) -> Self {
        self.bank = Some(bank);
        self
    }

    /// Adds (or replaces) the persisted index catalog.
    #[must_use]
    pub fn with_index(mut self, index: Arc<ExperienceIndex>) -> Self {
        self.index = Some(index);
        self
    }

    /// The configured bank catalog.
    #[must_use]
    pub fn bank(&self) -> Option<&Arc<ExperienceBank>> {
        self.bank.as_ref()
    }

    /// The configured index catalog.
    #[must_use]
    pub fn index(&self) -> Option<&Arc<ExperienceIndex>> {
        self.index.as_ref()
    }

    /// Whether at least one catalog is configured.
    #[must_use]
    pub fn has_source(&self) -> bool {
        self.bank.is_some() || self.index.is_some()
    }

    /// The catalogs a request would query, in deterministic source order.
    #[must_use]
    pub fn sources(&self) -> Vec<RetrieveSource> {
        let mut sources = Vec::with_capacity(2);
        if self.bank.is_some() {
            sources.push(RetrieveSource::Bank);
        }
        if self.index.is_some() {
            sources.push(RetrieveSource::Index);
        }
        sources
    }

    /// Retrieves the experience entries that answer `request`.
    ///
    /// The request's skill filters reach both catalogs: the bank applies them inside
    /// [`ExperienceBank::search_with_options`], and the retriever applies them to the
    /// [`ExperienceIndex`] hits with `passes_skill_filters`, the same rule the bank uses.
    /// A filtered request queries the index unlimited and truncates after filtering, so
    /// filtering never shortens the budget; `min_score` still applies last, and the
    /// reported provenance is unaffected by filtering.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] for a blank `query` or when no catalog
    /// is configured, and propagates the embedding model's error when a hit cannot be
    /// embedded.
    pub async fn retrieve(&self, request: &RetrieveRequest) -> Result<RetrievalReport> {
        if request.query.trim().is_empty() {
            return Err(validation("query", "must not be blank"));
        }
        if !self.has_source() {
            return Err(validation(
                "retriever",
                "no catalog is configured: use for_bank, for_index, or with_bank/with_index",
            ));
        }
        let options = SearchOptions {
            top_k: request.top_k,
            min_score: None,
            required_skills: request.required_skills.clone(),
            excluded_skills: request.excluded_skills.clone(),
            require_all_skills: request.require_all_skills,
        };
        // The index has no skill-filter seam, so the retriever filters its hits after the
        // fact. Asking it for `top_k` first would let filtered-out hits consume the budget,
        // so a filtered request asks for the whole catalog (`0` = unlimited) and the merge
        // stage truncates once every hit satisfies the filter. The unfiltered path keeps
        // the single-shot bounded request: same result, no extra ranking work.
        let index_top_k = if has_skill_filter(request) {
            0
        } else {
            request.top_k
        };
        let ranking = ExperienceRanking {
            top_k: index_top_k,
            min_score: None,
            require_all_skills: request.require_all_skills,
        };
        let mut merged: BTreeMap<String, RetrievedExperience> = BTreeMap::new();
        let mut sources = Vec::with_capacity(2);
        let mut total_candidates = 0_usize;
        let mut provenance = RankingProvenance::Embedding;
        if let Some(bank) = &self.bank {
            let bank_ranking = if bank.embedding().is_some() {
                RankingProvenance::Embedding
            } else {
                RankingProvenance::Lexical
            };
            for hit in bank.search_with_options(&request.query, &options).await? {
                total_candidates = total_candidates.saturating_add(1);
                absorb(&mut merged, hit, RetrieveSource::Bank, bank_ranking);
            }
            sources.push(RetrieveSource::Bank);
            provenance = combine(provenance, bank_ranking);
        }
        if let Some(index) = &self.index {
            // The index always ranks with its embedder: a blank query is rejected above,
            // so there is no lexical fallback to report.
            let index_ranking = RankingProvenance::Embedding;
            for hit in index.search(&request.query, &ranking).await? {
                // Counted before the client-side filter below: `total_candidates` reports
                // what the catalogs returned, not what the request kept.
                total_candidates = total_candidates.saturating_add(1);
                if !passes_skill_filters(
                    &hit.item,
                    &request.required_skills,
                    &request.excluded_skills,
                    request.require_all_skills,
                ) {
                    continue;
                }
                absorb(&mut merged, hit, RetrieveSource::Index, index_ranking);
            }
            sources.push(RetrieveSource::Index);
            provenance = combine(provenance, index_ranking);
        }
        sources.sort_unstable();
        let mut hits: Vec<RetrievedExperience> = merged.into_values().collect();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.item.id.cmp(&right.item.id))
        });
        if request.top_k > 0 {
            hits.truncate(request.top_k);
        }
        if let Some(min_score) = request.min_score {
            hits.retain(|hit| hit.score >= min_score);
        }
        Ok(RetrievalReport {
            query: request.query.clone(),
            hits,
            sources,
            ranking: provenance,
            total_candidates,
        })
    }
}

/// Merges one catalog hit into `merged`, keeping the strongest score per dedupe key.
fn absorb(
    merged: &mut BTreeMap<String, RetrievedExperience>,
    hit: ExperienceHit,
    source: RetrieveSource,
    ranking: RankingProvenance,
) {
    let incoming = RetrievedExperience {
        item: hit.item,
        score: hit.score,
        sources: vec![source],
        ranking,
    };
    let key = incoming.item.dedupe_key();
    match merged.entry(key) {
        Entry::Vacant(entry) => {
            entry.insert(incoming);
        }
        Entry::Occupied(mut entry) => {
            let existing = entry.get_mut();
            for value in &incoming.sources {
                if !existing.sources.contains(value) {
                    existing.sources.push(*value);
                }
            }
            existing.sources.sort_unstable();
            // The tie rule is exact on purpose: two catalogs scoring the same item with
            // the same model produce bit-identical cosines, and a tolerance would make the
            // winner depend on the merge order.
            let replaces = match incoming.score.total_cmp(&existing.score) {
                Ordering::Greater => true,
                Ordering::Equal => incoming.item.id < existing.item.id,
                Ordering::Less => false,
            };
            if replaces {
                let sources = std::mem::take(&mut existing.sources);
                *existing = RetrievedExperience {
                    sources,
                    ..incoming
                };
            }
        }
    }
}

/// Combines two provenances: `Lexical` as soon as either side is lexical.
fn combine(left: RankingProvenance, right: RankingProvenance) -> RankingProvenance {
    if left.is_lexical() || right.is_lexical() {
        RankingProvenance::Lexical
    } else {
        RankingProvenance::Embedding
    }
}

/// Reports whether `request` asks for any skill filtering at all.
///
/// Used only to pick the index request's budget: an empty filter needs no client-side pass.
fn has_skill_filter(request: &RetrieveRequest) -> bool {
    !request.required_skills.is_empty() || !request.excluded_skills.is_empty()
}

/// Reports whether `item` satisfies the skill filters of a request.
///
/// This is the retriever's copy of the rule the bank applies inside its search options, so
/// a hit is kept or dropped identically whichever catalog produced it: an item listing any
/// excluded skill is rejected first, and with no required skills that is the whole filter;
/// otherwise every required skill must be present, or — with `require_all_skills == false`
/// — at least one of them.
///
/// The edge cases match the bank exactly: skill matching is the case-sensitive exact
/// comparison of [`ExperienceItem::has_skill`], an empty filter keeps every item, repeated
/// required ids do not change the verdict, and a required id no item lists (`""` included)
/// rejects every item under `require_all_skills == false` as well.
fn passes_skill_filters(
    item: &ExperienceItem,
    required_skills: &[String],
    excluded_skills: &[String],
    require_all_skills: bool,
) -> bool {
    if excluded_skills
        .iter()
        .any(|excluded| item.has_skill(excluded))
    {
        return false;
    }
    if required_skills.is_empty() {
        return true;
    }
    if require_all_skills {
        required_skills
            .iter()
            .all(|required| item.has_skill(required))
    } else {
        required_skills
            .iter()
            .any(|required| item.has_skill(required))
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;
    use openjiuwen_retrieval::HashEmbedding;

    /// Builds an item with `pattern` and `skills`.
    fn item(pattern: &str, skills: &[&str]) -> ExperienceItem {
        ExperienceItem::new(pattern).with_skill_ids(skills.iter().copied())
    }

    /// A bank populated with `items`.
    async fn bank_with(items: Vec<ExperienceItem>) -> Arc<ExperienceBank> {
        let bank = ExperienceBank::new();
        bank.add_batch(items).await.expect("populate bank");
        Arc::new(bank)
    }

    /// An index populated with `items`.
    async fn index_with(items: Vec<ExperienceItem>) -> Arc<ExperienceIndex> {
        let index = ExperienceIndex::new(Arc::new(HashEmbedding::default()));
        index.add_batch(items).await.expect("populate index");
        Arc::new(index)
    }

    #[tokio::test]
    async fn a_bank_only_retriever_reports_lexical_provenance() {
        let bank = bank_with(vec![
            item("Summarize the report", &["summarizer"]),
            item("Translate text", &["translator"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_bank(bank);
        assert!(retriever.has_source());
        assert_eq!(retriever.sources(), vec![RetrieveSource::Bank]);
        let report = retriever
            .retrieve(&RetrieveRequest::new("summarize"))
            .await
            .expect("retrieve");
        assert_eq!(report.ranking, RankingProvenance::Lexical);
        assert_eq!(report.sources, vec![RetrieveSource::Bank]);
        // Lexical ranking keeps every item, scoring the unmatched pattern `0.0`.
        assert_eq!(report.hits.len(), 2);
        assert_eq!(report.hits[0].id(), "exp_0001");
        assert_eq!(report.hits[0].score, 1.0);
        assert_eq!(report.hits[0].sources, vec![RetrieveSource::Bank]);
        assert_eq!(report.hits[0].ranking, RankingProvenance::Lexical);
        assert_eq!(report.hits[1].score, 0.0);
        assert_eq!(report.total_candidates, 2);
        assert_eq!(report.query, "summarize");
    }

    #[tokio::test]
    async fn an_embedder_backed_bank_reports_embedding_provenance() {
        let bank = ExperienceBank::new().with_embedding(Arc::new(HashEmbedding::default()));
        bank.add(item("Summarize the report", &["summarizer"]))
            .await
            .expect("add");
        let retriever = ExperienceRetriever::for_bank(Arc::new(bank));
        let report = retriever
            .retrieve(&RetrieveRequest::new("summarize the report"))
            .await
            .expect("retrieve");
        assert_eq!(report.ranking, RankingProvenance::Embedding);
        assert_eq!(report.hits[0].ranking, RankingProvenance::Embedding);
    }

    #[tokio::test]
    async fn an_index_only_retriever_ranks_with_its_embedder() {
        let index = index_with(vec![
            item("Summarize the report", &["summarizer"]),
            item("Translate text to French", &["translator"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_index(index);
        let report = retriever
            .retrieve(&RetrieveRequest::new("translate text"))
            .await
            .expect("retrieve");
        assert_eq!(report.sources, vec![RetrieveSource::Index]);
        assert_eq!(report.ranking, RankingProvenance::Embedding);
        assert_eq!(report.hits.len(), 2);
        assert_eq!(report.hits[0].id(), "exp_0002");
        assert!(report.hits[0].score >= report.hits[1].score);
    }

    #[tokio::test]
    async fn both_catalogs_are_merged_and_sources_are_unioned() {
        let bank = bank_with(vec![
            item("Summarize the report", &["summarizer"]).with_id("exp_0002")
        ])
        .await;
        let index = index_with(vec![
            item("Summarize the report", &["summarizer"]).with_id("exp_0001")
        ])
        .await;
        let retriever = ExperienceRetriever::new().with_bank(bank).with_index(index);
        assert_eq!(
            retriever.sources(),
            vec![RetrieveSource::Bank, RetrieveSource::Index]
        );
        let report = retriever
            .retrieve(&RetrieveRequest::new("summarize the report"))
            .await
            .expect("retrieve");
        assert_eq!(report.total_candidates, 2);
        assert_eq!(
            report.sources,
            vec![RetrieveSource::Bank, RetrieveSource::Index]
        );
        // The lexical bank scores 1.0 while the cosine index scores below it, so the bank
        // hit wins and the sources still record both catalogs.
        assert_eq!(report.hits.len(), 1);
        assert_eq!(report.hits[0].id(), "exp_0002");
        assert_eq!(
            report.hits[0].sources,
            vec![RetrieveSource::Bank, RetrieveSource::Index]
        );
        // One catalog fell back to lexical scoring, so the report says so.
        assert_eq!(report.ranking, RankingProvenance::Lexical);
    }

    #[tokio::test]
    async fn an_equal_score_tie_keeps_the_smallest_id() {
        let bank = bank_with(vec![
            item("Summarize the report", &["summarizer"]).with_id("exp_0009")
        ])
        .await;
        let index = index_with(vec![
            item("Summarize the report", &["summarizer"]).with_id("exp_0001")
        ])
        .await;
        // Force an exact tie by using a bank whose embedder reproduces the index score.
        let embedding = Arc::new(HashEmbedding::default());
        let tied_bank = ExperienceBank::new().with_embedding(embedding.clone());
        tied_bank
            .add(item("Summarize the report", &["summarizer"]).with_id("exp_0009"))
            .await
            .expect("add");
        let tied_index = ExperienceIndex::new(embedding);
        tied_index
            .add(item("Summarize the report", &["summarizer"]).with_id("exp_0001"))
            .await
            .expect("add");
        let retriever =
            ExperienceRetriever::for_bank(Arc::new(tied_bank)).with_index(Arc::new(tied_index));
        let report = retriever
            .retrieve(&RetrieveRequest::new("summarize the report"))
            .await
            .expect("retrieve");
        assert_eq!(report.hits.len(), 1);
        assert_eq!(report.hits[0].id(), "exp_0001");
        assert_eq!(report.ranking, RankingProvenance::Embedding);
        // The unused fixtures above keep the plain-catalog merge path exercised.
        assert_eq!(ExperienceRetriever::for_bank(bank).sources().len(), 1);
        assert_eq!(ExperienceRetriever::for_index(index).sources().len(), 1);
    }

    #[tokio::test]
    async fn skill_filters_apply_inside_the_catalogs() {
        let bank = bank_with(vec![
            item("Summarize the report", &["summarizer", "report"]),
            item("Summarize the report", &["summarizer"]),
            item("Summarize the report", &["summarizer", "legacy"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_bank(bank);
        let all = RetrieveRequest::new("summarize").with_required_skills(["summarizer", "report"]);
        let report = retriever.retrieve(&all).await.expect("retrieve");
        assert_eq!(report.ids(), vec!["exp_0001"]);

        let any = RetrieveRequest::new("summarize")
            .with_required_skills(["report", "legacy"])
            .with_require_all_skills(false);
        let report = retriever.retrieve(&any).await.expect("retrieve");
        assert_eq!(report.hits.len(), 2);

        let excluded = RetrieveRequest::new("summarize").with_excluded_skills(["legacy"]);
        let report = retriever.retrieve(&excluded).await.expect("retrieve");
        assert_eq!(report.hits.len(), 2);
        assert!(report.ids().iter().all(|id| *id != "exp_0003"));
    }

    #[tokio::test]
    async fn index_hits_are_filtered_by_required_skills() {
        let index = index_with(vec![
            item("Summarize the report", &["summarizer", "report"]),
            item("Summarize the report", &["summarizer"]),
            item("Summarize the report", &["report"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_index(index);

        // Every required skill must be present.
        let all = RetrieveRequest::new("summarize the report")
            .with_required_skills(["summarizer", "report"]);
        let report = retriever.retrieve(&all).await.expect("retrieve");
        assert_eq!(report.ids(), vec!["exp_0001"]);
        // The index returned all three; the retriever dropped two of them.
        assert_eq!(report.total_candidates, 3);
        assert_eq!(report.ranking, RankingProvenance::Embedding);

        // Only one of them is enough with `require_all_skills == false`.
        let any = RetrieveRequest::new("summarize the report")
            .with_required_skills(["summarizer", "report"])
            .with_require_all_skills(false);
        let report = retriever.retrieve(&any).await.expect("retrieve");
        assert_eq!(report.hits.len(), 3);
        assert_eq!(report.total_candidates, 3);
    }

    #[tokio::test]
    async fn index_hits_are_filtered_by_excluded_skills() {
        let index = index_with(vec![
            item("Summarize the report", &["summarizer"]),
            item("Summarize the report", &["summarizer", "legacy"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_index(index);
        let report = retriever
            .retrieve(
                &RetrieveRequest::new("summarize the report").with_excluded_skills(["legacy"]),
            )
            .await
            .expect("retrieve");
        assert_eq!(report.ids(), vec!["exp_0001"]);
        assert_eq!(report.total_candidates, 2);
    }

    #[tokio::test]
    async fn an_index_filter_that_matches_nothing_leaves_an_empty_report() {
        let index = index_with(vec![
            item("Summarize the report", &["summarizer"]),
            item("Translate text", &["translator"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_index(index);

        let required = RetrieveRequest::new("summarize").with_required_skills(["missing"]);
        let report = retriever.retrieve(&required).await.expect("retrieve");
        assert!(report.hits.is_empty());
        assert_eq!(report.total_candidates, 2);
        assert_eq!(report.sources, vec![RetrieveSource::Index]);
        assert_eq!(report.ranking, RankingProvenance::Embedding);

        let excluded =
            RetrieveRequest::new("summarize").with_excluded_skills(["summarizer", "translator"]);
        let report = retriever.retrieve(&excluded).await.expect("retrieve");
        assert!(report.hits.is_empty());
        assert_eq!(report.total_candidates, 2);
        assert_eq!(report.query, "summarize");
    }

    #[tokio::test]
    async fn an_unfiltered_index_request_returns_every_hit() {
        let index = index_with(vec![
            item("Summarize the report", &["summarizer"]),
            item("Summarize the report", &["summarizer", "legacy"]),
            item("Translate text", &["translator"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_index(index);
        let unfiltered = retriever
            .retrieve(&RetrieveRequest::unlimited("summarize the report"))
            .await
            .expect("retrieve");
        assert_eq!(unfiltered.hits.len(), 3);
        assert_eq!(unfiltered.total_candidates, 3);
        assert!(unfiltered.ids().contains(&"exp_0002"));

        let filtered = retriever
            .retrieve(
                &RetrieveRequest::new("summarize the report").with_excluded_skills(["legacy"]),
            )
            .await
            .expect("retrieve");
        assert_eq!(filtered.total_candidates, 3);
        // The surviving order is the unfiltered order minus the dropped hit.
        let kept: Vec<&str> = unfiltered
            .ids()
            .into_iter()
            .filter(|id| *id != "exp_0002")
            .collect();
        assert_eq!(filtered.ids(), kept);
        assert_eq!(filtered.hits.len(), 2);
    }

    #[tokio::test]
    async fn index_filters_apply_before_top_k_truncation() {
        // One distinct skill per item, so excluding the top hit's skill drops exactly that
        // hit and nothing else.
        let index = index_with(vec![
            item("Summarize the report", &["summarizer"]),
            item("Summarize the report", &["reporting"]),
            item("Summarize the report", &["legacy"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_index(index);
        let unfiltered = retriever
            .retrieve(&RetrieveRequest::unlimited("summarize the report"))
            .await
            .expect("retrieve");
        assert_eq!(unfiltered.hits.len(), 3);
        let ranked = unfiltered.ids();
        let (top_id, runner_up_id) = (ranked[0].to_string(), ranked[1].to_string());
        let top_skill = unfiltered.hits[0]
            .item
            .skill_ids
            .first()
            .expect("single-skill fixture")
            .clone();

        // Unfiltered, the single slot goes to the strongest hit.
        let bounded = retriever
            .retrieve(&RetrieveRequest::new("summarize the report").with_top_k(1))
            .await
            .expect("retrieve");
        assert_eq!(bounded.ids(), vec![top_id.as_str()]);

        // Excluding the top hit's skill removes it *before* the truncation, so the budget
        // goes to the next surviving hit instead of being consumed by a rejected one.
        let filtered = retriever
            .retrieve(
                &RetrieveRequest::new("summarize the report")
                    .with_top_k(1)
                    .with_excluded_skills([top_skill]),
            )
            .await
            .expect("retrieve");
        assert_eq!(filtered.ids(), vec![runner_up_id.as_str()]);
        assert_eq!(filtered.total_candidates, 3);
    }

    #[tokio::test]
    async fn index_filters_compose_with_the_score_threshold() {
        let index = index_with(vec![
            item("Translate the document", &["translator"]),
            item("Translate the document", &["translator", "legacy"]),
            item("Bake a cake", &["baker"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_index(index);

        let unfiltered = retriever
            .retrieve(&RetrieveRequest::unlimited("translate the document"))
            .await
            .expect("retrieve");
        assert_eq!(unfiltered.hits.len(), 3);
        let top = unfiltered.hits[0].score;
        let lowest = unfiltered.hits.last().expect("three hits").score;
        assert!(top > lowest);
        let threshold = f64::midpoint(top, lowest);

        // Both survivors pass the filter; only the strong one clears the threshold.
        let filtered = retriever
            .retrieve(
                &RetrieveRequest::new("translate the document")
                    .with_excluded_skills(["legacy"])
                    .with_min_score(threshold),
            )
            .await
            .expect("retrieve");
        assert_eq!(filtered.ids(), vec!["exp_0001"]);
        assert_eq!(filtered.total_candidates, 3);

        // A filter that keeps only the weak hit leaves nothing above the threshold.
        let strict = retriever
            .retrieve(
                &RetrieveRequest::new("translate the document")
                    .with_required_skills(["baker"])
                    .with_min_score(threshold),
            )
            .await
            .expect("retrieve");
        assert!(strict.hits.is_empty());
        assert_eq!(strict.total_candidates, 3);
    }

    #[tokio::test]
    async fn both_catalogs_apply_the_skill_filter_to_their_own_hits() {
        let bank = bank_with(vec![
            item("Summarize the report", &["summarizer"]).with_id("exp_0002"),
            item("Summarize the report", &["summarizer", "legacy"]).with_id("exp_0004"),
        ])
        .await;
        let index = index_with(vec![
            item("Summarize the report quickly", &["summarizer"]).with_id("exp_0001"),
            item("Summarize the report quickly", &["summarizer", "legacy"]).with_id("exp_0003"),
        ])
        .await;
        let retriever = ExperienceRetriever::new().with_bank(bank).with_index(index);

        let request = RetrieveRequest::unlimited("summarize the report")
            .with_required_skills(["summarizer"])
            .with_excluded_skills(["legacy"]);
        let report = retriever.retrieve(&request).await.expect("retrieve");
        assert_eq!(
            report.sources,
            vec![RetrieveSource::Bank, RetrieveSource::Index]
        );
        // One bank hit (filtered by the bank) plus two index hits (filtered by the
        // retriever, which drops `exp_0003` afterwards).
        assert_eq!(report.total_candidates, 3);
        assert_eq!(report.hits.len(), 2);
        let bank_hit = report
            .hits
            .iter()
            .find(|hit| hit.id() == "exp_0002")
            .expect("bank hit survives");
        assert_eq!(bank_hit.sources, vec![RetrieveSource::Bank]);
        let index_hit = report
            .hits
            .iter()
            .find(|hit| hit.id() == "exp_0001")
            .expect("index hit survives");
        assert_eq!(index_hit.sources, vec![RetrieveSource::Index]);
        assert!(report
            .hits
            .iter()
            .all(|hit| hit.item.has_skill("summarizer")));
        assert!(report.hits.iter().all(|hit| !hit.item.has_skill("legacy")));
        assert_eq!(report.ranking, RankingProvenance::Lexical);

        // Without the filter the index's legacy hit is part of the merged answer.
        let unfiltered = retriever
            .retrieve(&RetrieveRequest::unlimited("summarize the report"))
            .await
            .expect("retrieve");
        assert!(unfiltered.ids().contains(&"exp_0003"));
    }

    #[tokio::test]
    async fn the_retriever_skill_filter_matches_the_bank_rules() {
        let items = vec![
            item("summarize", &["a", "b"]).with_id("exp_0001"),
            item("summarize", &["a"]).with_id("exp_0002"),
            item("summarize", &["b"]).with_id("exp_0003"),
            item("summarize", &[]).with_id("exp_0004"),
            item("summarize duplicate", &["a", "a"]).with_id("exp_0005"),
        ];
        let bank = bank_with(items.clone()).await;
        let filters: [(Vec<&str>, Vec<&str>, bool); 10] = [
            (vec![], vec![], true),
            (vec![], vec![], false),
            (vec!["a"], vec![], true),
            (vec!["a", "b"], vec![], true),
            (vec!["a", "b"], vec![], false),
            (vec!["a", "a"], vec![], true),
            (vec![], vec!["a"], true),
            (vec!["a"], vec!["b"], true),
            (vec![""], vec![], false),
            (vec!["missing"], vec![], false),
        ];
        for (required, excluded, require_all_skills) in filters {
            let required: Vec<String> = required.into_iter().map(Into::into).collect();
            let excluded: Vec<String> = excluded.into_iter().map(Into::into).collect();
            let options = SearchOptions {
                top_k: 0,
                min_score: None,
                required_skills: required.clone(),
                excluded_skills: excluded.clone(),
                require_all_skills,
            };
            let from_bank: Vec<String> = bank
                .search_with_options("summarize", &options)
                .await
                .expect("bank search")
                .iter()
                .map(|hit| hit.id().to_string())
                .collect();
            let expected: Vec<String> = items
                .iter()
                .filter(|item| passes_skill_filters(item, &required, &excluded, require_all_skills))
                .map(|item| item.id.clone())
                .collect();
            assert_eq!(
                from_bank, expected,
                "filters required={required:?} excluded={excluded:?} all={require_all_skills}"
            );
        }
    }

    #[tokio::test]
    async fn top_k_truncates_before_the_threshold_is_applied() {
        let bank = bank_with(vec![
            item("Summarize the report", &["summarizer"]),
            item("summarize", &["summarizer"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_bank(bank);
        let unlimited = retriever
            .retrieve(&RetrieveRequest::unlimited("summarize"))
            .await
            .expect("retrieve");
        assert_eq!(unlimited.hits.len(), 2);
        assert_eq!(unlimited.hits[0].score, 1.0);
        assert_eq!(unlimited.hits[1].score, 1.0);

        let bounded = retriever
            .retrieve(
                &RetrieveRequest::new("summarize")
                    .with_top_k(1)
                    .with_min_score(0.9),
            )
            .await
            .expect("retrieve");
        assert_eq!(bounded.hits.len(), 1);

        // A threshold above every score empties the result without widening `top_k`.
        let strict = retriever
            .retrieve(
                &RetrieveRequest::new("summarize")
                    .with_top_k(1)
                    .with_min_score(1.5),
            )
            .await
            .expect("retrieve");
        assert!(strict.is_empty());
        assert_eq!(strict.total_candidates, 1);
    }

    #[tokio::test]
    async fn repeated_runs_return_identical_reports() {
        let bank = bank_with(vec![
            item("Summarize the report", &["a"]),
            item("Summarize the report", &["b"]),
            item("Translate text", &["c"]),
        ])
        .await;
        let retriever = ExperienceRetriever::for_bank(bank);
        let request = RetrieveRequest::unlimited("summarize");
        let first = retriever.retrieve(&request).await.expect("first");
        let second = retriever.retrieve(&request).await.expect("second");
        assert_eq!(first, second);
        assert_eq!(first.ids(), vec!["exp_0001", "exp_0002", "exp_0003"]);
    }

    #[tokio::test]
    async fn invalid_requests_are_rejected() {
        let bank = bank_with(vec![item("Summarize the report", &["a"])]).await;
        let retriever = ExperienceRetriever::for_bank(bank);
        let error = retriever
            .retrieve(&RetrieveRequest::new("   "))
            .await
            .expect_err("blank query");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("query"));

        let empty = ExperienceRetriever::new();
        assert!(!empty.has_source());
        let error = empty
            .retrieve(&RetrieveRequest::new("summarize"))
            .await
            .expect_err("no source");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("no catalog"));
    }

    #[test]
    fn request_defaults_and_wire_names_round_trip() {
        let request: RetrieveRequest =
            serde_json::from_str(r#"{"query":"x","extra":1}"#).expect("deserialize");
        assert_eq!(request.top_k, DEFAULT_TOP_K);
        assert!(request.require_all_skills);
        assert!(request.min_score.is_none());
        assert!(request.required_skills.is_empty());
        assert_eq!(RetrieveSource::Bank.as_str(), "bank");
        assert_eq!(RetrieveSource::Index.as_str(), "index");
        assert!(RetrieveSource::Bank < RetrieveSource::Index);
        assert_eq!(RankingProvenance::Embedding.as_str(), "embedding");
        assert!(RankingProvenance::Lexical.is_lexical());
        assert!(!RankingProvenance::Embedding.is_lexical());
        let report = RetrievalReport::default();
        assert!(report.is_empty());
        assert!(report.ids().is_empty());
        assert_eq!(RankingProvenance::default(), RankingProvenance::Lexical);
    }
}
